#!/usr/bin/env python3
"""saturator.py — self-sustaining CPU-saturation driver for crack_live.

Work unit = (function x search-config) cell. Pattern generation in crack_live
is deterministic, so each cell is distinct, never-repeated work; the config
sequence is effectively unbounded (family singles -> pairs -> triples -> ...),
which is what keeps a 256-core box busy for days instead of minutes.

Runs ON a vast.ai box. As a co-tenant on training boxes it is launched by
eval_sidecar.sh's EVAL_MODE=saturate branch, which sources /workspace/eval/env.sh
first (venv + tree_sitter) and applies the nice-19/ionice-3/cgroup yield fence.

Env overrides (all optional — defaults target the baked eval-env layout):
  SAT_REPO      rb3-xenon repo root         (default /workspace/eval/rb3-xenon)
  SAT_UPREPO    monorepo root (pipeline)    (default /workspace/eval/upstream-monorepo)
  SAT_OUTBOX    finalized NDJSON shards +   (default /workspace/eval-out/corpus)
                SAT_STATUS.json; the sidecar's streamer ships this to B2
  SAT_HOST      shard/host label            (default $RUN_ID, else "sat")
  SAT_WORKERS   concurrent crack_live procs (default round(0.7 * cpu_count))
  SAT_SCRATCH   local scratch base          (default /workspace/sat)
Local scratch (never pushed): $SAT_SCRATCH/{ledger.db,out,tmp}.
Usage: python3 saturator.py [--host H] [--workers N]
"""
import argparse, hashlib, itertools, json, os, sqlite3, subprocess, sys, threading, time, uuid
from pathlib import Path

EVAL = "/workspace/eval"
REPO = os.environ.get("SAT_REPO", f"{EVAL}/rb3-xenon")
UPREPO = os.environ.get("SAT_UPREPO", f"{EVAL}/upstream-monorepo")
PIPELINE = f"{UPREPO}/tools/pipeline"
CRACK = f"{PIPELINE}/crack_live.py"
SAT = Path(os.environ.get("SAT_SCRATCH", "/workspace/sat"))     # local scratch
OUTBOX = Path(os.environ.get("SAT_OUTBOX", "/workspace/eval-out/corpus"))

FAMILIES = [
    "switch_case_reorder", "loop_body_assign_hoist", "signed_unsigned",
    "comparison_flip", "branch_polarity", "declaration_reorder",
    "statement_reorder", "variable_extraction", "early_return_merge",
]

# FULL family menu — the byte-match against retail is ground truth, so the SEARCH
# runs EVERY registered operator, INCLUDING semantics-changing ones (guard
# insert/elim, comparison_equivalence, demorgan, branch-invert, goto/loop/switch
# restructurers). A non-neutral edit that reaches byte-exact match is usually
# RECOVERING the true source (fixing our reconstruction's bug), not cheating.
# Neutrality is a LANDING-gate concern (verify a non-neutral patch harder before
# commit), not a reason to narrow the search. The farm never RUNS the code — it
# only compiles+diffs — so even "hazardous" (OOB/off-by-one) families are safe to
# explore. Family gating still means a non-applicable family emits ZERO candidates
# (costs gen time, not compile budget). Derived live from the registry so the menu
# is always the complete set; falls back to a curated subset if import fails.
def _all_families():
    try:
        sys.path.insert(0, UPREPO)
        from upstream_monorepo.patterns import list_patterns
        fams = sorted(list_patterns())
        # order the X360 regalloc/scheduling killers FIRST so they survive the
        # topk slice even when the full menu floods candidates.
        head = [f for f in REGALLOC_SOLO if f in fams]
        return head + [f for f in fams if f not in head]
    except Exception as e:
        print(f"WARN: full-family import failed ({e}); using default 9",
              file=sys.stderr)
        return list(FAMILIES)
# The register/scheduling families whose ordering search benefits from repeated
# passes (unseeded shuffle explores a fresh permutation each run).
REGALLOC_SOLO = [
    "mwcc_regorder_probe", "fpr_cascade_operand_hoist", "prologue_pressure",
    "float_literal_pressure", "parameter_live_range", "assignment_reorder",
    "member_init_reorder", "loop_var_hoist", "stack_array_hoist",
    "commutative_swap", "declaration_movement", "single_return",
    "temp_elimination",
]
WIDE_FAMILIES = _all_families()   # full registry, regalloc-first ordering

# Exact T1 token spellings from move_instantiator.T1_TOKEN_GROUPS —
# invalid tokens silently drop, so these must match verbatim.
_ID_KINDS = ("identifier", "field_identifier", "qualified_identifier",
             "type_identifier", "namespace_identifier", "statement_identifier")
MOVES_GROUPS = {
    "id_swap": ",".join(f"{op}:{k}" for op in ("UpdateLeaf", "Rebind", "ReplaceNode")
                        for k in _ID_KINDS),
    "bool_swap": ",".join(f"{op}:{k}" for op in ("UpdateLeaf", "ReplaceNode")
                          for k in ("true", "false")),
    "op_swap": "UpdateLeaf:operator",
    "delete": "DeleteNode:*",
    "move": "MoveNode:*",
}


def config_sequence():
    """Ordered (config_id, config) stream. Crackable work FIRST + deepest.

    Each config may carry `min_band` (skip targets below this live-DB percent)
    and `repeat` (run the config N times; only useful for unseeded-shuffle
    families, whose search explores a fresh permutation each pass). The ledger
    dedups by (symbol, config_id), and repeats get a `#k` suffix so each pass is
    a distinct never-repeated cell.
    """
    GATE = {"UPSTREAM_MONOREPO_GATE_ORDER": "1"}   # rank by priority before topk slice

    # ---- CRACK ZONE: band [92,100), where 100% is reachably close -----------
    # A0: FULL 116-family menu (incl. semantics-changing/bug-fix operators),
    # deep budget. config_id bumped to -full so the ledger re-runs it.
    yield "A0:full-crack", dict(gen="patterns", families=WIDE_FAMILIES,
                                budget=800, topk=96, env=GATE, min_band=92.0)
    # A1: moves generator, cranked caps, deep multi-move chains.
    yield "A1:moves-crack", dict(gen="moves", tokens=None, budget=800, topk=96,
                                 sites=64, fills=32, ptc=400, gcap=4000,
                                 min_band=92.0)
    # A2: register/scheduling families SOLO x3 (unseeded shuffle => new orders).
    for f in REGALLOC_SOLO:
        yield f"A2:regsolo-{f}", dict(gen="patterns", families=[f], budget=250,
                                      topk=48, min_band=92.0, repeat=3)

    # ---- MID BAND [80,92): still crackable with effort ----------------------
    yield "B0:full-mid", dict(gen="patterns", families=WIDE_FAMILIES,
                              budget=500, topk=80, env=GATE, min_band=80.0)
    yield "B1:moves-mid", dict(gen="moves", tokens=None, budget=500, topk=80,
                               sites=48, fills=24, ptc=300, gcap=2500,
                               min_band=80.0)

    # ---- CORPUS/HINTS: all bands, cheaper (training data + divergence hints) -
    yield "C0:full-all", dict(gen="patterns", families=WIDE_FAMILIES,
                              budget=300, topk=64, env=GATE)
    yield "C1:moves-all", dict(gen="moves", tokens=None, budget=300, topk=64,
                               sites=32, fills=16, ptc=200, gcap=1500)
    for g, toks in MOVES_GROUPS.items():
        yield f"C2:moves-{g}", dict(gen="moves", tokens=toks, budget=250,
                                    topk=48, sites=32, fills=16, ptc=200,
                                    gcap=1200)
    # env-slice variants over the wide menu (genuinely different compile slices)
    for name, env in (("nogate", {"UPSTREAM_MONOREPO_NO_GATE": "1"}),
                      ("slicediv", {"UPSTREAM_MONOREPO_SLICE_DIVERSITY": "1"})):
        yield f"C3:full-env-{name}", dict(gen="patterns", families=WIDE_FAMILIES,
                                          budget=250, topk=64, env=env)

    # ---- SATURATION TAIL: unbounded, keeps every core fed -------------------
    # deterministic k-subsets of the FULL wide menu — combinatorially vast.
    base = WIDE_FAMILIES
    for k in (2, 3):
        for combo in itertools.combinations(base[:16], k):
            yield f"Z{k}:" + "+".join(c[:10] for c in combo), dict(
                gen="patterns", families=list(combo), budget=120, topk=32)


def enumerate_targets():
    sys.path.insert(0, PIPELINE)
    sys.path.insert(0, UPREPO)
    os.chdir(REPO)
    from crack_live import select_near_misses
    tgts = select_near_misses(min_pct=10.0, max_pct=100.0, limit=200000,
                              repo_root=REPO)
    # value-band priority: [80,99.9) > [99.9,100) tie cluster > [50,80) > [10,50)
    def band(p):
        if 80.0 <= p < 99.9:
            return 0
        if p >= 99.9:
            return 1
        if p >= 50.0:
            return 2
        return 3
    tgts.sort(key=lambda t: (band(t.current_percent), -t.current_percent))
    return tgts


class Ledger:
    def __init__(self, path):
        self.lock = threading.Lock()
        self.con = sqlite3.connect(str(path), check_same_thread=False)
        self.con.execute("CREATE TABLE IF NOT EXISTS done("
                         "symbol TEXT, config TEXT, PRIMARY KEY(symbol,config))")
        self.con.execute("CREATE TABLE IF NOT EXISTS cracked(symbol TEXT PRIMARY KEY)")
        self.con.commit()

    def is_done(self, sym, cfg):
        with self.lock:
            r = self.con.execute("SELECT 1 FROM done WHERE symbol=? AND config=?",
                                 (sym, cfg)).fetchone()
            return r is not None

    def is_cracked(self, sym):
        with self.lock:
            return self.con.execute("SELECT 1 FROM cracked WHERE symbol=?",
                                    (sym,)).fetchone() is not None

    def mark_done(self, sym, cfg):
        with self.lock:
            self.con.execute("INSERT OR IGNORE INTO done VALUES(?,?)", (sym, cfg))
            self.con.commit()

    def mark_cracked(self, sym):
        with self.lock:
            self.con.execute("INSERT OR IGNORE INTO cracked VALUES(?)", (sym,))
            self.con.commit()

    def counts(self):
        with self.lock:
            d = self.con.execute("SELECT COUNT(*) FROM done").fetchone()[0]
            c = self.con.execute("SELECT COUNT(*) FROM cracked").fetchone()[0]
            return d, c


class ShardWriter:
    """Append NDJSON records; rotate to outbox as immutable shard files."""
    def __init__(self, outbox, host, max_records=300, max_age_s=600):
        self.outbox = Path(outbox)
        self.host = host
        self.lock = threading.Lock()
        self.max_records = max_records
        self.max_age_s = max_age_s
        self._open_new()

    def _open_new(self):
        self.part = self.outbox / "current.ndjson.part"
        self.fh = open(self.part, "a")
        self.n = 0
        self.opened = time.time()

    def _rotate_locked(self):
        self.fh.close()
        if self.part.stat().st_size > 0:
            ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
            nonce = uuid.uuid4().hex[:8]
            self.part.rename(self.outbox / f"shard-{self.host}-{ts}-{nonce}.ndjson")
        self._open_new()

    def write(self, rec):
        with self.lock:
            self.fh.write(json.dumps(rec, separators=(",", ":")) + "\n")
            self.fh.flush()
            self.n += 1
            if self.n >= self.max_records or time.time() - self.opened > self.max_age_s:
                self._rotate_locked()

    def flush_rotate(self):
        with self.lock:
            if self.n > 0:
                self._rotate_locked()


def build_cmd(nm, cfg, out_path):
    cmd = ["nice", "-n", "19", "ionice", "-c3",
           sys.executable, CRACK,
           "--symbol", nm.symbol,
           "--qualified-name", nm.qualified_name,
           "--source-path", nm.source_path,
           "--repo-root", REPO,
           "--isolate",
           "--budget", str(cfg["budget"]),
           "--topk", str(cfg["topk"]),
           "--out", str(out_path)]
    if nm.unit:
        cmd += ["--unit", nm.unit]
    if cfg["gen"] == "moves":
        cmd += ["--generator", "moves"]
        if cfg.get("tokens"):
            cmd += ["--moves-tokens", cfg["tokens"]]
        for flag, key in (("--moves-sites", "sites"), ("--moves-fills", "fills"),
                          ("--moves-per-token-cap", "ptc"),
                          ("--moves-global-cap", "gcap")):
            if cfg.get(key):
                cmd += [flag, str(cfg[key])]
    elif cfg.get("families"):
        cmd += ["--families", ",".join(cfg["families"])]
    return cmd


def _default_workers():
    try:
        return max(1, round(0.7 * (os.cpu_count() or 8)))
    except Exception:
        return 8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=os.environ.get("SAT_HOST")
                    or os.environ.get("RUN_ID") or "sat",
                    help="shard/host label (default $SAT_HOST|$RUN_ID|sat)")
    ap.add_argument("--workers", type=int,
                    default=int(os.environ.get("SAT_WORKERS") or _default_workers()),
                    help="concurrent crack_live procs (default 0.7*cpu_count)")
    ap.add_argument("--job-timeout", type=int, default=5400)
    args = ap.parse_args()

    for d in ("out", "tmp"):
        (SAT / d).mkdir(parents=True, exist_ok=True)
    OUTBOX.mkdir(parents=True, exist_ok=True)

    targets = enumerate_targets()
    print(f"host={args.host} workers={args.workers} repo={REPO} "
          f"outbox={OUTBOX}", flush=True)
    print(f"targets: {len(targets)}", flush=True)
    ledger = Ledger(SAT / "ledger.db")
    shards = ShardWriter(OUTBOX, args.host)

    # job queue: breadth-first over configs (all fns get config N before N+1).
    # min_band filters targets to the crackable band; repeat re-runs a config
    # (fresh unseeded-shuffle order each pass) under distinct #k config_ids.
    def jobs():
        for cfg_id, cfg in config_sequence():
            mb = cfg.get("min_band")
            reps = cfg.get("repeat", 1)
            for k in range(reps):
                cid = cfg_id if reps == 1 else f"{cfg_id}#{k}"
                for nm in targets:
                    if mb is not None and (nm.current_percent is None
                                           or nm.current_percent < mb):
                        continue
                    yield cid, cfg, nm

    job_iter = jobs()
    iter_lock = threading.Lock()
    stats = {"done": 0, "moved": 0, "cracked": 0, "errors": 0, "started": time.time()}

    def next_job():
        with iter_lock:
            while True:
                try:
                    cfg_id, cfg, nm = next(job_iter)
                except StopIteration:
                    return None
                if ledger.is_cracked(nm.symbol) or ledger.is_done(nm.symbol, cfg_id):
                    continue
                return cfg_id, cfg, nm

    def worker(wid):
        while True:
            job = next_job()
            if job is None:
                return
            cfg_id, cfg, nm = job
            env = dict(os.environ)
            env["TMPDIR"] = str(SAT / "tmp")
            for k, v in cfg.get("env", {}).items():
                env[k] = v
            h = hashlib.md5(f"{nm.symbol}|{cfg_id}".encode()).hexdigest()[:12]
            out_path = SAT / "out" / f"{h}.json"
            cmd = build_cmd(nm, cfg, out_path)
            t0 = time.time()
            rec = {"v": 1, "ts": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
                   "host": args.host, "nonce": uuid.uuid4().hex[:12],
                   "job_id": f"crackfarm-rb3xenon-sat-{args.host}",
                   "unit": nm.unit, "fn_symbol": nm.symbol,
                   "config_id": cfg_id,
                   "config": {"gen": cfg["gen"], "budget": cfg["budget"],
                              "topk": cfg["topk"],
                              "n_families": len(cfg.get("families", []) or []),
                              "min_band": cfg.get("min_band"),
                              "moves_caps": ([cfg.get("sites"), cfg.get("fills"),
                                              cfg.get("ptc"), cfg.get("gcap")]
                                             if cfg["gen"] == "moves" else None)},
                   "start_pct_db": nm.current_percent}
            try:
                p = subprocess.run(cmd, env=env, cwd=REPO,
                                   stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL,
                                   timeout=args.job_timeout)
                rec["exit"] = p.returncode
                if out_path.exists():
                    rep = json.loads(out_path.read_text())
                    rs = rep.get("results") or []
                    if rs:
                        r = rs[0]
                        rec.update(start_pct=r["start_pct"], final_pct=r["final_pct"],
                                   cracked=r["cracked"], compiles_used=r["compiles_used"],
                                   path=r["path"], plateaued=r["plateaued"])
                        if r["final_pct"] > r["start_pct"]:
                            stats["moved"] += 1
                        if r["cracked"]:
                            stats["cracked"] += 1
                            ledger.mark_cracked(nm.symbol)
                            print(f"CRACKED {nm.symbol} via {cfg_id} "
                                  f"path={r['path']}", flush=True)
            except subprocess.TimeoutExpired:
                rec["exit"] = "timeout"
            except Exception as e:
                rec["exit"] = f"error:{type(e).__name__}"
                stats["errors"] += 1
            rec["wall_s"] = round(time.time() - t0, 1)
            shards.write(rec)
            ledger.mark_done(nm.symbol, cfg_id)
            stats["done"] += 1

    threads = [threading.Thread(target=worker, args=(i,), daemon=True)
               for i in range(args.workers)]
    for t in threads:
        t.start()

    status_path = OUTBOX / "SAT_STATUS.json"
    while any(t.is_alive() for t in threads):
        time.sleep(60)
        d, c = ledger.counts()
        load = os.getloadavg()[0]
        st = {"ts": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
              "host": args.host, "workers": args.workers, "load1": load,
              "ledger_done": d, "ledger_cracked": c, **stats,
              "uptime_s": int(time.time() - stats["started"])}
        status_path.write_text(json.dumps(st))
        print(f"STATUS {json.dumps(st)}", flush=True)
        shards.flush_rotate() if time.time() - shards.opened > shards.max_age_s else None
    shards.flush_rotate()
    print("GRID EXHAUSTED", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""tokcost.py — what did those tokens actually COST? $/Mtok for a rented GPU box.

WHY THIS EXISTS
  We rent vast boxes by the HOUR and we consume them in TOKENS. Nothing in the
  fleet tooling ever converted one into the other, so "is this host cheap?" was
  answered with $/hr — a number that says nothing about throughput. A $2.13/hr
  box that decodes twice as fast as a $1.20/hr one is the cheaper box, and no
  existing view could say so.

  The gen side now writes a `gen_stats_v1` sidecar next to every generated cell
  (`gens.stats.json`): prompt/completion token counts and the wall clock the
  generation phase burned. Multiply that wall by the box's $/hr and you have a
  real unit cost.

TWO $ FIGURES, NEVER MIXED
  gen-attributed $/Mtok
      dph x (wall_s / 3600) / (completion_tokens / 1e6)
      What the GENERATION PHASE cost, and only that. It is a LOWER BOUND on the
      true cost of a token: the box was also rented while it was scoring, idle,
      booting and staging assets. Use it to compare hosts head-to-head — it is
      the term the host's silicon actually controls.
  box-amortized $/Mtok      (requires --box-wall-s, the total BILLED wall)
      dph x (box_wall_s / 3600) / (completion_tokens / 1e6)
      What the tokens cost us as a business: the whole rental divided by the
      tokens it produced. On the 2026-08-16 E3 run SCORING was 78-85% of arm
      wall clock, so this figure runs several times the gen-attributed one. That
      gap is not noise — it is the actual finding, and it is why the two are
      printed side by side and never averaged.
  It is a rollup-only figure by construction. Amortizing a single box wall over
  a per-cell token share yields the same number for every cell (the share
  cancels), so a per-file column would be the rollup value repeated.

COST SOURCE — OFFLINE BY DEFAULT
  In priority order:
    1. --dph FLOAT              explicit $/hr; always wins.
    2. --box IID                the box's own rate, from four LOCAL caches in
       order (`box_rate_cached`): `~/.cache/herdd/ls-snapshot.json`
       (`herdd ls --cached`), then `tools/vast/infra-metadata.db`
       (`instances.hourly`, then `runs.dph`), then the runmeta event mirror's
       `launched` event, then fleetd's journal. Four, because no single one
       covers a box's whole life — and by the time anyone asks what an eval's
       tokens cost, its box has been destroyed, so the last two are the normal
       path. Add --live to fall through to the vast API when all four miss.
    3. auto                     the run/job id inferred from the first path
       segment under the scan root, matched against `runs.run` in the same
       cache. This is what makes a plain `tokcost.py out/jobs` produce dollars
       with no flags, and it is where the per-host grouping comes from.
  No network is touched unless --live is passed. A run whose rate cannot be
  resolved still reports its tokens and tok/s, with the $ columns blank — an
  absent price is rendered as absent, never as zero.

USAGE
  tokcost.py out/jobs                              # scan, auto-resolve rates
  tokcost.py out/jobs/<JOB_ID> --dph 1.20
  tokcost.py <archive-run-dir> --box 47018759 --box-wall-s 7200
  tokcost.py out/jobs --json                       # machine-readable

Stdlib only, like everything else in tools/vast.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))

#: The only schema this tool understands. A sidecar carrying anything else is
#: skipped silently — the scan runs over whole job trees that hold plenty of
#: other json.
SCHEMA = "gen_stats_v1"

#: Basename suffix a candidate sidecar must end with. Deliberately broader than
#: the canonical `gens.stats.json` so a per-arm or per-shard variant
#: (`gens.arm-a.stats.json`) is picked up without a code change; SCHEMA is the
#: real filter.
STATS_SUFFIX = "stats.json"

#: Refuse to json.load anything larger than this. A gen_stats_v1 sidecar is a
#: few hundred bytes; a multi-MB `*stats.json` is something else entirely and
#: parsing it would only be a way to spend memory on a file we will discard.
MAX_STATS_BYTES = 1 << 20

#: Directories never worth walking into on a repo/archive tree.
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".next", ".venv"}


# --------------------------------------------------------------------------- #
# scan
# --------------------------------------------------------------------------- #
def find_stats_files(roots):
    """Walk `roots` recursively; return [(root, abspath)] for every file whose
    basename ends with STATS_SUFFIX, sorted and de-duplicated by real path (two
    roots may overlap). A root that does not exist contributes nothing — the
    caller reports it, this function does not raise."""
    out, seen = [], set()
    for root in roots:
        root = os.path.abspath(root)
        if os.path.isfile(root):
            if root not in seen:
                seen.add(root)
                out.append((os.path.dirname(root), root))
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
            for fn in sorted(filenames):
                if not fn.endswith(STATS_SUFFIX):
                    continue
                p = os.path.join(dirpath, fn)
                if p in seen:
                    continue
                seen.add(p)
                out.append((root, p))
    return out


def load_stats(path):
    """Parse one candidate sidecar. Returns the dict when it is a gen_stats_v1
    record, else None. Unreadable / unparseable / oversized / wrong-schema all
    degrade to None: a scan across a whole job tree must never die on one bad
    file."""
    try:
        if os.path.getsize(path) > MAX_STATS_BYTES:
            return None
        with open(path, "r", encoding="utf-8") as fh:
            d = json.load(fh)
    except (OSError, ValueError):
        return None
    if not isinstance(d, dict) or d.get("schema") != SCHEMA:
        return None
    return d


def _num(v):
    """Coerce to float, or None. Booleans are NOT numbers here (a stray
    `true` in a token field is a defect, not a 1)."""
    if isinstance(v, bool) or v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _int(v):
    f = _num(v)
    return None if f is None else int(f)


# --------------------------------------------------------------------------- #
# cost
# --------------------------------------------------------------------------- #
def usd_per_mtok(dph, wall_s, completion_tokens):
    """$/Mtok = dph x (wall_s/3600) / (completion_tokens/1e6). None whenever any
    input is missing or non-positive — an unpriced or token-less cell has no
    unit cost, and 0.0 would read as 'free'."""
    dph, wall_s = _num(dph), _num(wall_s)
    toks = _num(completion_tokens)
    if dph is None or wall_s is None or toks is None:
        return None
    if dph <= 0 or wall_s <= 0 or toks <= 0:
        return None
    return (dph * (wall_s / 3600.0)) / (toks / 1e6)


def cache_db_path(explicit=None):
    """Path to the local infra-metadata cache. --cache-db > $INFRA_METADATA_DB >
    tools/vast/infra-metadata.db. Mirrors herdd._infra_cache_db exactly so the
    two never drift onto different files."""
    if explicit:
        return explicit
    return os.environ.get("INFRA_METADATA_DB") or os.path.join(
        _HERE, "infra-metadata.db")


def ls_snapshot_path():
    """`herdd ls --cached`'s snapshot — mirrors herdd._LS_SNAPSHOT."""
    return os.path.join(
        os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")),
        "herdd", "ls-snapshot.json")


def runmeta_cache_dir():
    """The local mirror of `runs/<RUN_ID>/events/` that runmeta.read_run fills
    with rclone — mirrors runmeta's own default."""
    return os.path.join(
        os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")),
        "vast-runmeta")


def fleet_journal_path():
    """fleetd's append-only journal — mirrors fleet_report.journal_path()."""
    state_dir = os.environ.get("FLEETD_STATE_DIR")
    if state_dir:
        return os.path.join(os.path.expanduser(state_dir), "journal.ndjsonl")
    return os.path.expanduser("~/.local/state/vast-fleetd/journal.ndjsonl")


def _from_ls_snapshot(iid):
    """`instances[].dph_total` for this IID out of the `herdd ls --cached`
    snapshot. That field is WHAT WE ARE BILLED (on a bid box it is our standing
    bid, not the market floor), which is why it leads the chain.

    Caveat, and it is a real one: the tools/vast suite's own tests overwrite
    this file with fixture boxes (ids 1/2/3 at $0.10). Matching on an exact IID
    is what keeps that harmless — a real vast instance id is 8 digits and cannot
    collide — so never widen this to "first instance in the snapshot"."""
    try:
        with open(ls_snapshot_path(), "r", encoding="utf-8") as fh:
            snap = json.load(fh)
    except (OSError, ValueError):
        return None
    for inst in (snap or {}).get("instances") or []:
        if not isinstance(inst, dict) or str(inst.get("id")) != str(iid):
            continue
        dph = _num(inst.get("dph_total"))
        if not dph:
            return None
        label = inst.get("label") or ""
        run = None
        for tok in str(label).split():
            if tok.startswith("run:"):
                run = tok[len("run:"):]
        return {"dph": dph, "source": "cache:ls-snapshot.dph_total",
                "machine_id": inst.get("machine_id"),
                "gpu": inst.get("gpu_name"),
                "geo": (inst.get("geolocation") or "").lstrip(", ").strip()
                       or None,
                "run": run, "iid": str(iid)}
    return None


def _from_runmeta_cache(iid):
    """The `launched` event's `dph` for this IID, from the LOCAL runmeta event
    mirror. Events are immutable objects, so whatever rclone has already copied
    down stays valid forever — this reads it without a network call. Survives
    the box's destruction, which the instances-shaped sources do not."""
    import glob                                              # noqa: PLC0415
    root = runmeta_cache_dir()
    if not os.path.isdir(root):
        return None
    best = None
    for path in sorted(glob.glob(os.path.join(root, "*", "events", "*.json"))):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                ev = json.load(fh)
        except (OSError, ValueError):
            continue
        if not isinstance(ev, dict) or ev.get("event") != "launched":
            continue
        if str(ev.get("instance_id")) != str(iid):
            continue
        dph = _num(ev.get("dph"))
        if not dph:
            continue
        cand = {"dph": dph, "source": "cache:runmeta.launched.dph",
                "machine_id": ev.get("machine_id"),
                "gpu": ev.get("gpu"),
                "geo": ev.get("geolocation") or ev.get("geo"),
                "run": ev.get("run_id"), "iid": str(iid),
                "_ts": ev.get("ts") or ""}
        if best is None or cand["_ts"] >= best["_ts"]:
            best = cand
    if best:
        best.pop("_ts", None)
    return best


def _from_fleet_journal(iid):
    """The newest journalled `dph` for this IID. fleetd writes it on
    `unwatched`/`jobs_box_condemned`/`jobs_replaced` straight off the instance's
    `dph_total`, alongside a `dph_known` flag — an unreadable rate is journalled
    as 0.0 with `dph_known: false`, so the flag must be honoured or a box whose
    price fleetd could not read reads back as FREE."""
    path = fleet_journal_path()
    if not os.path.exists(path):
        return None
    hit = None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or '"dph"' not in line:
                    continue
                try:
                    ev = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(ev, dict) or str(ev.get("iid")) != str(iid):
                    continue
                if ev.get("dph_known") is False:
                    continue
                dph = _num(ev.get("dph"))
                if dph:
                    hit = dph                      # append-only: last wins
    except OSError:
        return None
    if hit is None:
        return None
    return {"dph": hit, "source": "cache:fleetd-journal.dph",
            "machine_id": None, "gpu": None, "geo": None, "run": None,
            "iid": str(iid)}


def _ro_conn(db):
    """Read-only sqlite connection, or None if the file is missing/unopenable.
    Read-only by URI on purpose: this tool has no business writing the cache
    that herdd owns."""
    if not db or not os.path.exists(db):
        return None
    try:
        return sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=3)
    except sqlite3.Error:
        return None


def _table_exists(conn, name):
    try:
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (name,)).fetchone()
    except sqlite3.Error:
        return False
    return bool(row)


def _from_infra_db(iid, db=None):
    """The dashboard's sqlite cache, two tables in order:

      instances.hourly  the LIVE box's own dph_total, i.e. what we are billed
                        (on a bid box that is our standing bid, not the market
                        floor — see herdd's `_dash_instance_rows`).
      runs.dph          the rate recorded for a run that ran on this IID. The
                        instances table only holds boxes that exist RIGHT NOW,
                        so this is the only rate here for a box we have already
                        destroyed — which is every box a finished eval ran on.
    """
    out = {"dph": None, "source": None, "machine_id": None, "gpu": None,
           "geo": None, "run": None, "iid": str(iid) if iid else None}
    conn = _ro_conn(cache_db_path(db))
    if conn is None or not iid:
        return None
    try:
        if _table_exists(conn, "instances"):
            try:
                r = conn.execute(
                    "SELECT hourly, machine_id, gpu, geo, run_id FROM instances "
                    "WHERE iid=?", (int(iid),)).fetchone()
            except (sqlite3.Error, ValueError):
                r = None
            if r:
                out["machine_id"] = r[1]
                out["gpu"] = r[2]
                out["geo"] = r[3]
                out["run"] = r[4]
                if _num(r[0]):
                    out["dph"] = _num(r[0])
                    out["source"] = "cache:instances.hourly"
                    return out
        if _table_exists(conn, "runs"):
            try:
                r = conn.execute(
                    "SELECT dph, gpu, run FROM runs WHERE instance_id=? AND "
                    "dph IS NOT NULL ORDER BY run", (str(iid),)).fetchone()
            except sqlite3.Error:
                r = None
            if r and _num(r[0]):
                out["dph"] = _num(r[0])
                out["source"] = "cache:runs.dph"
                out["gpu"] = out["gpu"] or r[1]
                out["run"] = out["run"] or r[2]
    finally:
        conn.close()
    # host facts with no price are still worth returning (they name the host);
    # a wholly empty hit is a miss, so the chain moves on.
    return out if any(out[k] for k in ("dph", "machine_id", "gpu", "run")) \
        else None


#: The offline resolution chain for `--box IID`, freshest-and-most-billed first.
#: Each entry is (name, callable); the first non-None hit with a `dph` wins.
#: Host facts (machine_id/gpu/geo) from an earlier priceless hit are folded
#: forward so a box can be NAMED even when its rate stays unknown.
_BOX_SOURCES = ("ls-snapshot", "infra-db", "runmeta-cache", "fleetd-journal")


def box_rate_cached(iid, db=None):
    """OFFLINE $/hr for a vast instance id. Returns
    {dph, source, machine_id, gpu, geo, run, iid} — every field optional, `dph`
    None when no local source knows the price. NEVER touches the network.

    Four sources are consulted in order (`_BOX_SOURCES`); the reason there are
    four is that no single one covers a box's whole life:

      ls-snapshot     `herdd ls --cached` — the freshest billed dph_total,
                      but only for boxes alive at the last `ls`.
      infra-db        the dashboard's sqlite cache: live `instances.hourly`,
                      then `runs.dph` for a run that ran on this IID.
      runmeta-cache   the `launched` event's dph in the local runmeta mirror.
                      Immutable objects — this one outlives the box.
      fleetd-journal  the dph fleetd observed when it stopped watching the box.

    A finished eval's box is DESTROYED by the time anyone asks what its tokens
    cost, so the last two are the normal path, not the exotic one."""
    facts = {"dph": None, "source": None, "machine_id": None, "gpu": None,
             "geo": None, "run": None, "iid": str(iid) if iid else None}
    if not iid:
        return facts
    lookup = {
        "ls-snapshot": lambda: _from_ls_snapshot(iid),
        "infra-db": lambda: _from_infra_db(iid, db),
        "runmeta-cache": lambda: _from_runmeta_cache(iid),
        "fleetd-journal": lambda: _from_fleet_journal(iid),
    }
    for name in _BOX_SOURCES:
        try:
            hit = lookup[name]()
        except Exception:
            hit = None                  # a broken cache is a miss, not a crash
        if not hit:
            continue
        for k in ("machine_id", "gpu", "geo", "run"):
            if facts[k] is None and hit.get(k) is not None:
                facts[k] = hit[k]
        if hit.get("dph"):
            facts["dph"] = hit["dph"]
            facts["source"] = hit["source"]
            return facts
    return facts


def run_rate_cached(run_id, db=None):
    """OFFLINE $/hr + host facts for a RUN id (the auto path). Same cache, the
    `runs` table, keyed by run name. Returns the same shape as
    `box_rate_cached`."""
    out = {"dph": None, "source": None, "machine_id": None, "gpu": None,
           "geo": None, "run": str(run_id) if run_id else None, "iid": None}
    conn = _ro_conn(cache_db_path(db))
    if conn is None or not run_id:
        return out
    try:
        if not _table_exists(conn, "runs"):
            return out
        try:
            r = conn.execute(
                "SELECT dph, gpu, instance_id FROM runs WHERE run=?",
                (str(run_id),)).fetchone()
        except sqlite3.Error:
            r = None
        if not r:
            return out
        out["gpu"] = r[1]
        out["iid"] = r[2]
        if _num(r[0]):
            out["dph"] = _num(r[0])
            out["source"] = "cache:runs.dph"
        # a live box for that run carries machine_id/geo the runs table lacks
        if r[2] and _table_exists(conn, "instances"):
            try:
                i = conn.execute(
                    "SELECT machine_id, geo FROM instances WHERE iid=?",
                    (int(r[2]),)).fetchone()
            except (sqlite3.Error, ValueError):
                i = None
            if i:
                out["machine_id"], out["geo"] = i[0], i[1]
    finally:
        conn.close()
    return out


def box_rate_live(iid):
    """LAST RESORT, network: ask vast what this instance costs. Only reached
    behind --live. Soft — any failure (no key, no such box, API down) yields a
    dph of None rather than an exit, because a missing price must degrade to a
    blank column and not to a dead report."""
    out = {"dph": None, "source": None, "machine_id": None, "gpu": None,
           "geo": None, "run": None, "iid": str(iid) if iid else None}
    try:
        if _HERE not in sys.path:
            sys.path.insert(0, _HERE)
        import herdd                                       # noqa: PLC0415
        herdd.load_env()
        ok, d, _err = herdd.request_soft("GET", f"v0/instances/{int(iid)}/")
        inst = (d or {}).get("instances") if isinstance(d, dict) else None
        if not ok or not isinstance(inst, dict):
            return out
        out["machine_id"] = inst.get("machine_id")
        out["gpu"] = inst.get("gpu_name")
        out["geo"] = (inst.get("geolocation") or "").lstrip(", ").strip() or None
        dph = _num(inst.get("dph_total"))
        if dph:
            out["dph"] = dph
            out["source"] = "live:dph_total"
    except Exception:
        return out
    return out


def _rel_run_id(root, path):
    """The run/job id a stats file belongs to: its FIRST path segment under the
    scan root (`out/jobs/<JOB_ID>/cells/.../gens.stats.json` -> `<JOB_ID>`).
    None when the file sits directly in the root (nothing to key on)."""
    try:
        rel = os.path.relpath(path, root)
    except ValueError:
        return None
    parts = rel.split(os.sep)
    return parts[0] if len(parts) > 1 else None


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def build_report(roots, *, dph=None, box=None, box_wall_s=None, cache_db=None,
                 live=False):
    """Scan `roots` and price everything. Returns the JSON document the CLI
    prints with --json and the text renderer formats; `rows` is per stats file,
    `rollup` totals them, `hosts` groups them for host comparison."""
    missing = [r for r in roots if not os.path.exists(r)]
    found = find_stats_files(roots)

    # --box / --dph resolve ONCE for the whole scan (they describe one box).
    box_facts = None
    if dph is None and box:
        box_facts = box_rate_cached(box, cache_db)
        if box_facts["dph"] is None and live:
            box_facts = box_rate_live(box)

    run_cache = {}

    def facts_for(run_id):
        if dph is not None:
            return {"dph": float(dph), "source": "flag:--dph",
                    "machine_id": None, "gpu": None, "geo": None,
                    "run": run_id, "iid": None}
        if box_facts is not None:
            f = dict(box_facts)
            f["run"] = f.get("run") or run_id
            return f
        if run_id not in run_cache:
            run_cache[run_id] = run_rate_cached(run_id, cache_db)
        return dict(run_cache[run_id])

    rows, skipped = [], 0
    for root, path in found:
        st = load_stats(path)
        if st is None:
            skipped += 1
            continue
        run_id = _rel_run_id(root, path)
        f = facts_for(run_id)
        comp = _int(st.get("completion_tokens")) or 0
        wall = _num(st.get("wall_s"))
        derived = (comp / wall) if (wall and wall > 0 and comp) else None
        host = ("m" + str(f["machine_id"])) if f.get("machine_id") else (
            f.get("gpu") or run_id or "unknown")
        rows.append({
            "path": path,
            "rel_path": os.path.relpath(path, root),
            "root": root,
            "run": run_id,
            "host": host,
            "machine_id": f.get("machine_id"),
            "iid": f.get("iid"),
            "gpu": f.get("gpu"),
            "geo": f.get("geo"),
            "model": st.get("model"),
            "prompts": _int(st.get("prompts")),
            "requests": _int(st.get("requests")),
            "prompt_tokens": _int(st.get("prompt_tokens")) or 0,
            "completion_tokens": comp,
            "total_tokens": _int(st.get("total_tokens")),
            "wall_s": wall,
            "gen_tok_per_s": _num(st.get("gen_tok_per_s")),
            "tok_per_s_derived": derived,
            "k": _int(st.get("k")),
            "concurrency": _int(st.get("concurrency")),
            # `concurrency` is RESOLVED, not configured: the box sizes it at
            # runtime off its own cpuset/quota. `concurrency_mode` says whether
            # that number was auto-derived or pinned by the operator, and is
            # absent on sidecars written before the field existed — which is
            # why every field here is read defensively and unknown keys are
            # simply ignored. A schema addition must never break a scan.
            "concurrency_mode": st.get("concurrency_mode"),
            "max_new": _int(st.get("max_new")),
            "resumed": bool(st.get("resumed")),
            "dph": f.get("dph"),
            "dph_source": f.get("source"),
            "gen_attributed_usd_per_mtok": usd_per_mtok(f.get("dph"), wall, comp),
        })

    rows.sort(key=lambda r: (r["run"] or "", r["rel_path"]))

    tot_comp = sum(r["completion_tokens"] for r in rows)
    tot_prompt = sum(r["prompt_tokens"] for r in rows)
    tot_wall = sum(r["wall_s"] or 0.0 for r in rows)
    # ONE rate for the rollup only when every priced row agrees on it; a scan
    # spanning two boxes at different prices has no single $/hr and must not
    # pretend otherwise.
    priced = {r["dph"] for r in rows if r["dph"] is not None}
    roll_dph = priced.pop() if len(priced) == 1 else None
    rollup = {
        "files": len(rows),
        "skipped_files": skipped,
        "prompt_tokens": tot_prompt,
        "completion_tokens": tot_comp,
        "wall_s": tot_wall,
        "tok_per_s_derived": (tot_comp / tot_wall) if tot_wall > 0 else None,
        "dph": roll_dph,
        "gen_attributed_usd_per_mtok": usd_per_mtok(roll_dph, tot_wall, tot_comp),
        "box_wall_s": _num(box_wall_s),
        "box_amortized_usd_per_mtok": usd_per_mtok(roll_dph, box_wall_s, tot_comp),
    }
    # gen wall as a share of the billed wall — the scoring/idle overhead made
    # visible in one number (measured 2026-08-16: scoring alone is 78-85%).
    bw = _num(box_wall_s)
    rollup["gen_wall_frac_of_box"] = (tot_wall / bw) if (bw and bw > 0) else None

    hosts = {}
    for r in rows:
        h = hosts.setdefault(r["host"], {
            "host": r["host"], "machine_id": r["machine_id"], "gpu": r["gpu"],
            "geo": r["geo"], "dph": r["dph"], "files": 0, "prompt_tokens": 0,
            "completion_tokens": 0, "wall_s": 0.0, "runs": [],
        })
        h["files"] += 1
        h["prompt_tokens"] += r["prompt_tokens"]
        h["completion_tokens"] += r["completion_tokens"]
        h["wall_s"] += r["wall_s"] or 0.0
        if r["run"] and r["run"] not in h["runs"]:
            h["runs"].append(r["run"])
        if h["dph"] is None:
            h["dph"] = r["dph"]
    for h in hosts.values():
        h["tok_per_s_derived"] = (h["completion_tokens"] / h["wall_s"]
                                  if h["wall_s"] > 0 else None)
        h["gen_attributed_usd_per_mtok"] = usd_per_mtok(
            h["dph"], h["wall_s"], h["completion_tokens"])

    return {
        "schema": "tokcost_v1",
        "roots": [os.path.abspath(r) for r in roots],
        "missing_roots": missing,
        "rows": rows,
        "rollup": rollup,
        "hosts": sorted(hosts.values(),
                        key=lambda h: (h["gen_attributed_usd_per_mtok"] is None,
                                       h["gen_attributed_usd_per_mtok"] or 0)),
    }


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #
def _money(v, places=2):
    return "—" if v is None else f"${v:,.{places}f}"


def _n(v, places=0):
    return "—" if v is None else f"{v:,.{places}f}"


def render(rep):
    rows, roll = rep["rows"], rep["rollup"]
    out = []
    for r in rep["missing_roots"]:
        out.append(f"warn: scan root does not exist: {r}")
    if not rows:
        out.append(f"no {SCHEMA} sidecars found under: "
                   + ", ".join(rep["roots"]))
        out.append(f"  (looked for files ending in {STATS_SUFFIX!r} whose "
                   f"\"schema\" is {SCHEMA!r}; "
                   f"{roll['skipped_files']} non-matching file(s) skipped)")
        return "\n".join(out)

    out.append("GEN TELEMETRY (gen_stats_v1) — per cell:")
    out.append(f"  {'CELL':<34} {'MODEL':<20} {'PROMPT':>10} {'COMPL':>11} "
               f"{'WALL s':>9} {'tok/s':>8} {'$/hr':>7} {'GEN $/Mtok':>11}")
    for r in rows:
        cell = r["rel_path"]
        if len(cell) > 34:
            cell = "…" + cell[-33:]
        tps = r["gen_tok_per_s"] if r["gen_tok_per_s"] is not None \
            else r["tok_per_s_derived"]
        out.append(
            f"  {cell:<34} {(r['model'] or '?'):<20.20} "
            f"{_n(r['prompt_tokens']):>10} {_n(r['completion_tokens']):>11} "
            f"{_n(r['wall_s'], 1):>9} {_n(tps, 1):>8} "
            f"{('—' if r['dph'] is None else format(r['dph'], '.3f')):>7} "
            f"{_money(r['gen_attributed_usd_per_mtok'], 3):>11}")

    out.append("")
    out.append("ROLLUP:")
    out.append(f"  files                    : {roll['files']}"
               + (f"  ({roll['skipped_files']} non-gen_stats file(s) skipped)"
                  if roll["skipped_files"] else ""))
    out.append(f"  prompt tokens            : {_n(roll['prompt_tokens'])}")
    out.append(f"  completion tokens        : {_n(roll['completion_tokens'])}")
    out.append(f"  gen wall (sum of cells)  : {_n(roll['wall_s'], 1)} s")
    out.append(f"  tok/s (completion/wall)  : {_n(roll['tok_per_s_derived'], 1)}")
    if roll["dph"] is None:
        srcs = sorted({r["dph_source"] for r in rows if r["dph_source"]})
        out.append("  $/hr                     : — (no single rate for this "
                   "scan" + (f"; sources seen: {', '.join(srcs)}" if srcs else
                             "; pass --dph or --box") + ")")
    else:
        src = next((r["dph_source"] for r in rows if r["dph"] == roll["dph"]),
                   None)
        out.append(f"  $/hr                     : {_money(roll['dph'], 4)}"
                   + (f"  [{src}]" if src else ""))
    out.append(f"  gen-attributed $/Mtok    : "
               f"{_money(roll['gen_attributed_usd_per_mtok'], 3)}"
               "   (generation wall only — a LOWER BOUND)")
    if roll["box_wall_s"]:
        out.append(f"  billed box wall          : {_n(roll['box_wall_s'], 1)} s"
                   + (f"  (gen = {roll['gen_wall_frac_of_box'] * 100:.1f}% of it)"
                      if roll["gen_wall_frac_of_box"] is not None else ""))
        out.append(f"  box-amortized $/Mtok     : "
                   f"{_money(roll['box_amortized_usd_per_mtok'], 3)}"
                   "   (whole rental / same tokens)")
    else:
        out.append("  box-amortized $/Mtok     : — (pass --box-wall-s SECONDS, "
                   "the total billed wall, to price the whole rental)")

    if len(rep["hosts"]) > 1:
        out.append("")
        out.append("PER HOST (cheapest gen-attributed first):")
        out.append(f"  {'HOST':<14} {'GPU':<14} {'FILES':>5} {'COMPL':>12} "
                   f"{'WALL s':>9} {'tok/s':>8} {'$/hr':>7} {'GEN $/Mtok':>11}")
        for h in rep["hosts"]:
            out.append(
                f"  {h['host']:<14.14} {(h['gpu'] or '?'):<14.14} "
                f"{h['files']:>5} {_n(h['completion_tokens']):>12} "
                f"{_n(h['wall_s'], 1):>9} {_n(h['tok_per_s_derived'], 1):>8} "
                f"{('—' if h['dph'] is None else format(h['dph'], '.3f')):>7} "
                f"{_money(h['gen_attributed_usd_per_mtok'], 3):>11}")
    return "\n".join(out)


# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="$/Mtok for rented GPU boxes, from gen_stats_v1 sidecars")
    ap.add_argument("roots", nargs="+", metavar="DIR",
                    help="directories (or files) to scan recursively for "
                         "*stats.json sidecars")
    ap.add_argument("--dph", type=float,
                    help="$/hr for the box; highest-priority cost source")
    ap.add_argument("--box", metavar="IID",
                    help="vast instance id — look its $/hr up in the LOCAL "
                         "infra-metadata cache (no network)")
    ap.add_argument("--box-wall-s", type=float, metavar="S",
                    help="total BILLED wall for the run, in seconds; enables "
                         "the box-amortized $/Mtok figure")
    ap.add_argument("--cache-db", metavar="PATH",
                    help="override the infra-metadata.db path (default: "
                         "tools/vast/infra-metadata.db, env INFRA_METADATA_DB)")
    ap.add_argument("--live", action="store_true",
                    help="allow ONE vast API call when --box misses the local "
                         "cache (off by default; this tool is offline)")
    ap.add_argument("--json", action="store_true", help="machine-readable dump")
    a = ap.parse_args(argv)

    rep = build_report(a.roots, dph=a.dph, box=a.box, box_wall_s=a.box_wall_s,
                       cache_db=a.cache_db, live=a.live)
    if a.json:
        print(json.dumps(rep, indent=2, default=str))
    else:
        print(render(rep))
    return 0


if __name__ == "__main__":
    sys.exit(main())

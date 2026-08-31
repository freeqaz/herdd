#!/usr/bin/env python3
"""onstart/jobd.py — box-side helper for onstart/jobd.sh.

jobd.sh (bash) owns the poll loop, downloads, entrypoint run, and heartbeats;
this helper does the three things bash should not hand-roll, all delegated to
`jobmeta.py` so there is ONE implementation of the event envelope + deterministic
bundle format shared with the laptop `herdd job` side:

  * `prepare` — parse a ticket's canonical JSON config into shell-sourceable
    vars (+ an env-exports file for the entrypoint, + a results-globs file).
    No YAML parser is needed box-side: submit already baked canonical JSON in.
  * `extract` — decompress + sha-verify + safely extract a bundle.
  * `emit`    — append one immutable lifecycle event (same key/envelope as the CLI).

Import seam: jobmeta.py + runmeta.py sit in tools/vast/ (the parent of this
onstart/ dir) in the repo, or FLAT beside this file when pushed by
`herdd job attach`. Both layouts are put on sys.path below.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE)):        # flat (attach) OR nested (repo)
    if _p not in sys.path:
        sys.path.insert(0, _p)

# CAPABILITY IMPORT (FAILCLOSED_DESIGN §3). This import used to be bare, so a
# missing transitive dependency in the shipped flat bundle killed EVERY jobd.py
# subcommand with a ModuleNotFoundError traceback on stderr — which every call
# site in jobd.sh discards. Box 47737955 (2026-08-13) burned $1.742/52min that
# way: `bidpolicy.py` was absent from ship_manifest.txt while jobmeta.py had
# grown an unguarded top-level `from bidpolicy import ...`.
#
# Catching it here does NOT make the failure survivable — nothing below works
# without jobmeta. It makes the failure *reportable*: `selftest` can name the
# missing module on stdout as JSON, and every other subcommand exits with the
# dedicated code EXIT_STRUCTURAL so jobd.sh can tell "this interpreter can never
# run my code" (never recovers -> fail closed) apart from "this one B2 PUT
# failed" (may recover -> fail open, with a counter). That distinction is the
# whole design; it cannot be made from a bare traceback.
EXIT_STRUCTURAL = 3

try:
    import jobmeta  # noqa: E402
    _IMPORT_ERR = None
except Exception as _exc:                          # noqa: BLE001 - report anything
    jobmeta = None
    _IMPORT_ERR = f"{type(_exc).__name__}: {_exc}"


def _tail(path, nbytes=2000):
    try:
        with open(path, "rb") as fh:
            try:
                fh.seek(-nbytes, os.SEEK_END)
            except OSError:
                fh.seek(0)
            return fh.read().decode("utf-8", "replace")
    except OSError:
        return None


def cmd_prepare(a):
    """Emit shell var assignments to stdout; write the entrypoint env + results
    globs to files. jobd.sh does: eval "$(jobd.py prepare ticket.json --env-out
    E --results-out R)"."""
    with open(a.ticket) as fh:
        ticket = json.load(fh)
    cfg = ticket.get("config") or {}
    needs = cfg.get("needs") or {}
    env = cfg.get("env") or {}
    results = cfg.get("results") or []
    # experiment-matrix association (jobmatrix.py): echoed back onto every
    # lifecycle event jobd emits, for audit. Absent on plain jobs -> empty vars.
    experiment = cfg.get("experiment") or {}
    # mid-run checkpoint sync (jobmeta schema): interval + globs (submit-side
    # validation already defaulted the globs to `results` when only checkpoint_s
    # was given). Absent/0 = off — old tickets keep working.
    checkpoint_s = cfg.get("checkpoint_s") or 0
    checkpoints = cfg.get("checkpoints") or []
    # declarative asset staging (N4): jobd.sh stages these onto the box BEFORE the
    # entrypoint. Emit a bash-parseable TSV (one asset per line) + a per-asset
    # require-globs sidecar dir, so jobd.sh never has to parse JSON. Fields are
    # validated slugs/paths (no tab, no newline), so a TSV split is unambiguous.
    assets = cfg.get("assets") or []

    with open(a.env_out, "w") as fh:
        for k, v in env.items():
            fh.write(f"export {k}={shlex.quote(str(v))}\n")
    with open(a.results_out, "w") as fh:
        for g in results:
            fh.write(f"{g}\n")
    if a.checkpoints_out:
        with open(a.checkpoints_out, "w") as fh:
            for g in checkpoints:
                fh.write(f"{g}\n")
    if a.assets_out:
        with open(a.assets_out, "w") as fh:
            for asset in assets:
                # `-` FOR AN ABSENT OPTIONAL FIELD, never an empty one. jobd.sh
                # splits these with `IFS=$'\t' read -r`, and a tab is IFS
                # WHITESPACE to bash: runs of them COLLAPSE, so `…\t\treceipt`
                # would land the receipt in `dest` and leave `receipt` empty.
                # `-` was already the reader's "no dest" spelling
                # (_link_asset_dest), so this only makes it the writer's too.
                dest = str(asset.get("dest") or "") or "-"
                receipt = str(asset.get("receipt") or "") or "-"
                opt = "1" if asset.get("optional") else "0"
                fh.write("\t".join([
                    str(asset.get("name", "")), str(asset.get("b2", "")),
                    str(asset.get("mode") or "copy"), opt, dest, receipt]) + "\n")
    if a.asset_require_dir and assets:
        os.makedirs(a.asset_require_dir, exist_ok=True)
        for asset in assets:
            req = asset.get("require") or []
            if req:
                with open(os.path.join(a.asset_require_dir, str(asset["name"])), "w") as fh:
                    for g in req:
                        fh.write(f"{g}\n")

    def q(v):
        return shlex.quote("" if v is None else str(v))
    out = [
        f"JOB_ID={q(ticket.get('job_id'))}",
        f"JOB_NAME={q(cfg.get('name'))}",
        f"JOB_BUNDLE_SHA={q(ticket.get('bundle_sha256'))}",
        f"JOB_ENTRYPOINT={q(cfg.get('entrypoint'))}",
        f"JOB_TIMEOUT_S={q(cfg.get('timeout_s') or jobmeta.DEFAULT_TIMEOUT_S)}",
        f"JOB_NEEDS_GPU={q(1 if needs.get('gpu') else 0)}",
        f"JOB_NEEDS_GPU_RAM_GB={q(needs.get('gpu_ram_gb') or 0)}",
        # gpus: card count for the scheduler — "all" rides through verbatim
        # (resolved against the live card count box-side); legacy gpu:true
        # tickets default to 1 card; CPU jobs to 0.
        f"JOB_NEEDS_GPUS={q(needs.get('gpus') if needs.get('gpus') is not None else (1 if needs.get('gpu') else 0))}",
        f"JOB_NEEDS_VENV={q(needs.get('venv') or 'none')}",
        f"JOB_MAX_RESTARTS={q(cfg.get('max_restarts', jobmeta.DEFAULT_MAX_RESTARTS))}",
        f"JOB_N_RESULTS={q(len(results))}",
        f"JOB_CHECKPOINT_S={q(checkpoint_s if isinstance(checkpoint_s, int) else 0)}",
        f"JOB_EXP_ID={q(experiment.get('exp_id'))}",
        f"JOB_ARM={q(experiment.get('arm'))}",
    ]
    print("\n".join(out))


def cmd_extract(a):
    sha = jobmeta.extract_bundle(a.zst, a.dest, expect_sha=a.sha)
    print(sha)


def cmd_emit(a):
    fields = {}
    for kv in a.field or []:
        if "=" in kv:
            k, v = kv.split("=", 1)
            # coerce integer-looking values (rc, counts) so the fold's numeric
            # helpers see ints, not strings (rc="7" would fold to None otherwise).
            fields[k] = int(v) if re.fullmatch(r"-?\d+", v or "") else v
    if a.tail_file:
        t = _tail(a.tail_file)
        if t is not None:
            fields["tail"] = t
    if a.results_json:
        try:
            with open(a.results_json) as fh:
                fields["results"] = json.load(fh)
        except (OSError, ValueError):
            pass
    if a.instance_id:
        fields["instance_id"] = a.instance_id
    actor = a.actor or (f"box:{a.instance_id}" if a.instance_id else None)
    ev = jobmeta.emit_event(a.job_id, a.event, actor=actor, **fields)
    # non-fatal: a dying box still exits cleanly even if the emit did not land.
    print(json.dumps({"event": a.event, "emitted": ev.get("_emitted", False),
                      "key": ev.get("_key")}))


def cmd_emit_box(a):
    """Append one immutable BOX-lifecycle event (jobs/nodes/<IID>/events/) — the
    per-box stream jobd uses for parked_self/drained. Separate from the per-job
    `emit` (different namespace + event set)."""
    fields = {}
    for kv in a.field or []:
        if "=" in kv:
            k, v = kv.split("=", 1)
            fields[k] = int(v) if re.fullmatch(r"-?\d+", v or "") else v
    ev = jobmeta.emit_box_event(a.instance_id, a.event, **fields)
    print(json.dumps({"event": a.event, "emitted": ev.get("_emitted", False),
                      "key": ev.get("_key")}))


def cmd_tail_snapshot(a):
    """Stage line-aligned snapshots of the checkpoint-glob matches the periodic
    pass's --min-age window would skip (task #110). Prints the number staged on
    stdout — jobd.sh reads that to decide whether a second push is worth doing —
    and the per-file refusal reasons on stderr, so a pass that stages nothing is
    never silent about why.

    Fail-soft by construction: any error here prints 0 and the ordinary
    age-filtered pass, which already ran, is unaffected."""
    try:
        with open(a.matchlist) as fh:
            rels = [l.strip() for l in fh if l.strip()]
    except OSError as exc:
        print("0")
        print(f"tail-snapshot: matchlist unreadable: {exc}", file=sys.stderr)
        return
    try:
        out = jobmeta.ckpt_tail_snapshot(
            a.run, rels, float(a.min_age), a.state, a.stage,
            max_bytes=int(float(a.max_mb) * 1024 * 1024))
    except Exception as exc:                       # never break the sync loop
        print("0")
        print(f"tail-snapshot: {type(exc).__name__}: {exc}", file=sys.stderr)
        return
    print(len(out["staged"]))
    for rel in out["staged"]:
        print(f"tail-snapshot: staged {rel}", file=sys.stderr)
    for rel, why in sorted(out["skipped"].items()):
        print(f"tail-snapshot: skip {rel} ({why})", file=sys.stderr)


def cmd_selftest(a):
    """Prove THIS interpreter can run the python half, before any money is at
    risk. Pure and offline by construction (FAILCLOSED_DESIGN §4): imports, then
    builds one job event and one box event through the real envelope builders.
    No B2, no network, no filesystem writes — so the ONLY thing it can report is
    a capability fault, which is exactly the class that never recovers and is
    therefore the only class licensed to fail closed. A network blip cannot make
    this red, which is what keeps a spot resume from tripping it (§7).

    Exit 0 => ok. Exit EXIT_STRUCTURAL => the python half is dead. stdout is one
    JSON object either way, so jobd.sh can put the reason on the bash/rclone
    beacon that is still working."""
    checks = []
    ok = True

    def chk(name, fn):
        nonlocal ok
        try:
            fn()
        except Exception as exc:                   # noqa: BLE001
            ok = False
            checks.append({"check": name, "ok": False,
                           "error": f"{type(exc).__name__}: {exc}"})
        else:
            checks.append({"check": name, "ok": True})

    if _IMPORT_ERR is not None:
        ok = False
        checks.append({"check": "import jobmeta", "ok": False,
                       "error": _IMPORT_ERR})
    else:
        checks.append({"check": "import jobmeta", "ok": True})
        chk("import runmeta", lambda: __import__("runmeta"))
        # the four entry points jobd.sh actually calls into
        for _n in ("emit_event", "emit_box_event", "extract_bundle",
                   "ckpt_tail_snapshot", "make_event", "make_box_event"):
            chk(f"jobmeta.{_n}", lambda n=_n: getattr(jobmeta, n))
        # exercise the envelope builders: importable-but-broken is still broken,
        # and these are pure so a green here means the emit path's CPU-side work
        # is sound even when B2 is unreachable.
        chk("make_event", lambda: jobmeta.make_event(
            "selftest-job", "heartbeat", actor="box:selftest"))
        chk("make_box_event", lambda: jobmeta.make_box_event(
            a.instance_id or "selftest", "boot"))

    bad = [c for c in checks if not c["ok"]]
    reason = bad[0]["error"] if bad else ""
    print(json.dumps({"ok": ok, "checks": checks, "reason": reason}))
    if not ok:
        sys.exit(EXIT_STRUCTURAL)


def main():
    ap = argparse.ArgumentParser(prog="jobd.py", description="box-side job helper")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("selftest")
    ps.add_argument("--instance-id", dest="instance_id", default=None)
    ps.set_defaults(func=cmd_selftest)

    pp = sub.add_parser("prepare")
    pp.add_argument("ticket")
    pp.add_argument("--env-out", dest="env_out", required=True)
    pp.add_argument("--results-out", dest="results_out", required=True)
    pp.add_argument("--checkpoints-out", dest="checkpoints_out", default=None)
    pp.add_argument("--assets-out", dest="assets_out", default=None)
    pp.add_argument("--asset-require-dir", dest="asset_require_dir", default=None)
    pp.set_defaults(func=cmd_prepare)

    pe = sub.add_parser("extract")
    pe.add_argument("zst")
    pe.add_argument("dest")
    pe.add_argument("--sha", default=None)
    pe.set_defaults(func=cmd_extract)

    pm = sub.add_parser("emit")
    pm.add_argument("job_id")
    pm.add_argument("event")
    pm.add_argument("--instance-id", dest="instance_id", default=None)
    pm.add_argument("--actor", default=None)
    pm.add_argument("--field", action="append", default=[], metavar="K=V")
    pm.add_argument("--tail-file", dest="tail_file", default=None)
    pm.add_argument("--results-json", dest="results_json", default=None)
    pm.set_defaults(func=cmd_emit)

    pt = sub.add_parser("tail-snapshot")
    pt.add_argument("--run", required=True)
    pt.add_argument("--matchlist", required=True)
    pt.add_argument("--min-age", dest="min_age", required=True)
    pt.add_argument("--state", required=True)
    pt.add_argument("--stage", required=True)
    pt.add_argument("--max-mb", dest="max_mb", default="128")
    pt.set_defaults(func=cmd_tail_snapshot)

    pb = sub.add_parser("emit-box")
    pb.add_argument("instance_id")
    pb.add_argument("event")
    pb.add_argument("--field", action="append", default=[], metavar="K=V")
    pb.set_defaults(func=cmd_emit_box)

    a = ap.parse_args()
    # Structural gate: `selftest` is the one subcommand that can run without a
    # working jobmeta (reporting the fault IS its job). Everything else needs it,
    # so exit with the dedicated code instead of raising NameError/AttributeError
    # deep inside a handler. Same failure, named — jobd.sh reads the code.
    if _IMPORT_ERR is not None and a.func is not cmd_selftest:
        print(f"jobd.py: python half is broken: {_IMPORT_ERR}", file=sys.stderr)
        sys.exit(EXIT_STRUCTURAL)
    a.func(a)


if __name__ == "__main__":
    main()

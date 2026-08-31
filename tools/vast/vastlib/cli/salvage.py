"""`herdd salvage` — the manual one-shot disk rescue off a DEAD box.

An argparse block plus a bounded local loop. Everything that decides anything
already landed lower: the survey/copy/verify state machine is
`supervise.retention._job_salvage_advance`, the record and its outcome vocabulary
are `boxes.salvage`. This module supplies what a CLI supplies — the flags, the
clock, and the JSON the operator reads.

Why a hand-rolled loop and not the supervision tick
---------------------------------------------------
fleetd runs the SAME advance function on its own cadence, but only for an
eviction it SAW. A box lost while no watch was armed, or one an operator finds
already parked, never enters that ladder — and the alternative the runbook is
trying to retire is hand-running `vastai` against a dead instance. So this
command drives the identical state machine from a `while` loop bounded by
`--wait`, and the timeout is a REPORTED OUTCOME (exit 3, "may still be in
flight"), never a hang.

The three exit codes are the contract
-------------------------------------
`0` terminal and quiet, `2` terminal but a `LOUD_OUTCOMES` verdict (partial,
unverifiable, refused — the ones where destroying the dead box would lose
bytes), `3` never reached a terminal phase inside the deadline. The record is
printed as JSON before any of them, so a `2` is still a readable result.

What is deliberately NOT here
-----------------------------
* The GPU-contract argument. Vast serves filesystem access to an `exited`
  instance, so neither side enters one; that is the finding
  `DISK_ACCESS_FINDINGS_2026-08-05.md` records and the reason the epilog says so.
* Any policy about WHICH checkpoints. `--salvage-keep-n` / `--salvage-max-gb`
  default to `None` here and resolve inside the record from
  `boxes.salvage.SALVAGE_KEEP_N` / `SALVAGE_MAX_GB` — the same constants the
  help text interpolates, so the printed default and the effective one are one
  read (cli-surface.json hazard H4).

Provenance: moved from `tools/vast/herdd.py` (`cmd_salvage`, parser block in
`main()`), plan §8 step 6, 2026-08-16, behavior-preserving.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any

from vastlib.boxes import lifecycle
from vastlib.boxes import salvage as salvage_mod
from vastlib.cli import _args
from vastlib.supervise import retention


# moved-from: herdd.cmd_salvage
def run(a: argparse.Namespace) -> None:
    """`herdd salvage <DEAD_IID> --to <BOX>` — the manual one-shot.

    Same code path fleetd runs, driven by a bounded local loop instead of the
    supervision tick. Exists because the automatic ladder only fires on an
    eviction it SAW: a box lost while no watch was armed, or one an operator
    finds parked, still needs the survey+copy+verify, and hand-running `vastai`
    is what the corrected runbook is trying to stop.
    """
    if a.dest is None:
        sys.exit("error: --to <INSTANCE_ID> is required (the box the disk is "
                 "copied INTO; any running box you own will do)")
    jc: dict[str, Any] = {"a": a, "iid": str(a.dest), "dry_run": bool(a.dry_run),
                          "instances": lifecycle._instances_soft()}
    now = time.time()
    rec = salvage_mod.new_record(a.id, now=now, dest_candidates=[str(a.dest)],
                                 job_id=a.job, keep_n=a.salvage_keep_n
                                 or salvage_mod.SALVAGE_KEEP_N,
                                 max_gb=a.salvage_max_gb or salvage_mod.SALVAGE_MAX_GB,
                                 deadline_s=float(a.wait),
                                 dest_wait_s=float(a.wait))
    deadline = now + float(a.wait) + 60.0
    while rec.get("phase") != "done" and time.time() < deadline:
        jc["instances"] = lifecycle._instances_soft()
        retention._job_salvage_advance(jc, rec, time.time())
        if rec.get("phase") == "done":
            break
        time.sleep(min(20.0, max(5.0, float(a.wait) / 30.0)))
    if rec.get("phase") != "done":
        print(f"!! salvage did not reach a terminal outcome inside {a.wait:g}s "
              f"(phase {rec.get('phase')}) — the copy may still be in flight; "
              f"re-run this command to re-verify before destroying {a.id}")
        sys.exit(3)
    print(json.dumps({"dead_iid": rec.get("dead_iid"),
                      "dest_iid": rec.get("dest_iid"),
                      "outcome": rec.get("outcome"),
                      "bytes": rec.get("bytes"),
                      "detail": rec.get("detail"),
                      "items": [{k: it.get(k) for k in
                                 ("job_id", "name", "bytes", "dest", "verify",
                                  "verify_reason", "b2")}
                                for it in rec.get("items") or []]}, indent=2))
    if rec.get("outcome") in salvage_mod.LOUD_OUTCOMES:
        sys.exit(2)


def add_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser],
               add_cmd: _args.AddCmd) -> argparse.ArgumentParser:
    psv = add_cmd(sub, "salvage",
                  "copy checkpoints off a STOPPED/EVICTED box's disk onto a "
                  "box you own, verify them byte-for-byte, then push to B2",
                  "tools/vast/RETENTION_SALVAGE.md (the runbook)",
                  "tools/vast/DISK_ACCESS_FINDINGS_2026-08-05.md (why this "
                  "works with no GPU contract)",
                  "NOTE: enters NO GPU contract on either side — vast serves "
                  "filesystem access to an `exited` instance, so this neither "
                  "resumes the dead box nor needs its GPUs back. The race is "
                  "HOST RECLAMATION (box 46859541 vanished ~30 min after its "
                  "eviction), so run this the moment you notice. fleetd does "
                  "it automatically for evictions it saw; this is the manual "
                  "one-shot for the ones it did not")
    psv.add_argument("id", type=int, help="the DEAD/stopped instance to read")
    psv.add_argument("--to", dest="dest", type=int, required=False,
                     metavar="IID",
                     help="destination instance (must be `running`); any box "
                          "you own works — it is only a landing zone")
    psv.add_argument("--job", default=None, metavar="JOB_ID",
                     help="restrict to one job id (default: every job on the disk)")
    psv.add_argument("--salvage-keep-n", dest="salvage_keep_n", type=int,
                     default=None, metavar="N",
                     help=f"newest N checkpoints per job (default "
                          f"{salvage_mod.SALVAGE_KEEP_N})")
    psv.add_argument("--salvage-max-gb", dest="salvage_max_gb", type=float,
                     default=None, metavar="GB",
                     help=f"refuse a transfer larger than this (default "
                          f"{salvage_mod.SALVAGE_MAX_GB:g})")
    psv.add_argument("--wait", type=float, default=salvage_mod.SALVAGE_DEADLINE_S,
                     metavar="S",
                     help=f"bounded wait for the copy to verify (default "
                          f"{salvage_mod.SALVAGE_DEADLINE_S:g}s). Timing out is a "
                          f"reported outcome, never a hang")
    psv.add_argument("--dry-run", action="store_true",
                     help="survey + plan only; initiate no copy")
    psv.set_defaults(func=run)
    return psv

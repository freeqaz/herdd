"""`herdd box <id>` — one box, one call: instance + jobs + watch/budget.

Why this module exists
----------------------
The question after a reschedule/handoff is always the same bundle — is the
instance up, what job is on it, which watch owns it and what is left of the
budget — and answering it took three commands (`ls` + `job ls --box` +
`fleet status`) glued with awk. This is the one-call answer, composed from the
producers those commands already render: `lifecycle._instances()`, the
`view._fold_fleet_jobs` fold, and the fleetd status rows. No daemon change:
the handoff mapping is already in the status payload (a row whose `target`
differs from its `iid`), so `box <old-id>` after a replacement answers with
the current box.

Two spellings, one dict: the human rendering is `key: value` lines over
exactly the dict `--json` prints, and the derived instance fields come from
`_ls_render._minimal_rows`, so every value spells like the `ls --minimal`
vocabulary agents already parse.

What is deliberately NOT here
-----------------------------
* The market probe (spot floors) and the fleet-wide image-digest sweep — both
  are network fan-outs `ls` amortizes over the whole fleet. Rates fall back to
  the instance's own stale fields, exactly as `ls --no-spot`.
* Raw payload fields (onstart, extra_env) and job `last_tail`. `show` is the
  raw view and its output carries live secrets; this view is safe to paste.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

from vastlib.boxes import health, lifecycle, reap
from vastlib.cli import _args, _docs, _ls_render
from vastlib.core import fmt, models
from vastlib.fleet import client as fleet_client
from vastlib.jobs import view

import bidpolicy

# The folded-view fields worth re-emitting per job. `last_tail` is excluded on
# the same rule as `_job_cell`: the raw container tail never leaves the fold.
_JOB_KEYS = ("job_id", "name", "entrypoint", "status", "display_status",
             "rc", "fail_reason", "instance_id", "ended_at", "n_checkpoints")


def _job_out(v: Any) -> dict[str, Any]:  # noqa: ANN401 — folded job view
    d = {k: v.get(k) for k in _JOB_KEYS if v.get(k) is not None}
    pg = view._job_progress(v)
    if pg:
        d["progress"] = pg
    d["cell"] = view._job_cell(v)
    return d


def _watch_row(iid: str) -> tuple[dict[str, Any] | None, str | None]:
    """The fleetd status row addressing this box — matched by current `iid` OR
    by the watch key (`target`), so an old id still answers after a handoff.
    (row, None) on a hit; (None, why) when fleetd cannot answer or has no row.
    Soft: a box view must render even with no daemon installed."""
    try:
        if not (os.path.exists(fleet_client.fleet_sock_path())
                or os.path.isdir(fleet_client.fleet_state_dir())):
            return None, "fleetd not installed"
        ok, data, err = fleet_client.fleet_request("status", _timeout=5,
                                                   _retries=0)
        if not ok:
            return None, f"fleetd unreachable ({err})"
    except Exception as e:
        return None, f"fleetd unreachable ({type(e).__name__}: {e})"
    for r in (data or {}).get("rows") or []:
        key, cur = str(r.get("target") or ""), str(r.get("iid") or "")
        if iid not in (key, cur):
            continue
        # SPENT prefers the ceiling's cumulative counter for the same reason
        # `fleet status` does: the watch's own counter reads $0.00 on a box
        # that inherited a ceiling with money already drawn.
        spent = r.get("ceiling_spend_usd")
        if spent is None:
            spent = r.get("spend_usd") or 0
        return {
            "watch": key,
            "current_box": cur,
            "handed_off": bool(key) and bool(cur) and key != cur,
            "profile": r.get("profile"),
            "state": r.get("state"),
            "spent_usd": spent,
            "budget_usd": r.get("budget_usd"),
            "remaining_usd": r.get("remaining_usd"),
            "paused": bool(r.get("paused")),
            "last_action": r.get("last_action"),
        }, None
    return None, "no watch addresses this box"


def _gather(iid: str) -> dict[str, Any]:
    """Everything `box` reports, as one JSON-safe dict."""
    ins = lifecycle._instances()
    inst = next((i for i in ins if str(i.get("id")) == iid), None)
    live = [i.get("id") for i in ins
            if (i.get("actual_status") or "").lower() in bidpolicy.LIVE_STATES]
    try:
        jobs_by_box = view._fold_fleet_jobs(set(live))
    except Exception:
        jobs_by_box = {}
    jobs = jobs_by_box.get(iid, [])
    out: dict[str, Any] = {"id": iid, "found": inst is not None}
    if inst is not None:
        # the idle ledger prunes boxes absent from its input, so it must see
        # the FULL fleet even though only one row is read back
        idle = reap._idle_secs_map(ins, live)
        try:
            fleet_health = health.gather_fleet_health([inst], jobs_by_box)
        except Exception:
            fleet_health = {}
        row = _ls_render._minimal_rows({
            "instances": [inst], "live_ids": live,
            "jobs_by_box": {iid: jobs}, "market": {},
            "idle_secs": {iid: idle[iid]} if iid in idle else {},
            "health": fleet_health})[0]
        row["ssh"] = f"{inst.get('ssh_host', '-')}:{inst.get('ssh_port', '-')}"
        row["image"] = models._instance_image(inst)
        out["box"] = row
        h = fleet_health.get(iid)
        if h and h.get("verdict") not in (None, "", health.GUARD_OK):
            out["health"] = {k: h.get(k) for k in ("verdict", "reason", "age_s")}
    out["jobs"] = [_job_out(v) for v in jobs]
    watch, note = _watch_row(iid)
    out["watch"] = watch
    if note:
        out["watch_note"] = note
    return out


def _print_human(d: dict[str, Any]) -> None:
    iid = d["id"]
    if not d.get("found"):
        print(f"box {iid}: NO instance with this id (destroyed, or an old "
              f"watch key)")
    else:
        b = d["box"]
        head = (f"box {iid}: {b['state']} ({b['status']}) — "
                f"{b['gpus']}x {b['gpu']}, {b['mode']}")
        if b["hourly"]:
            head += f" ${b['hourly']}/hr"
        if b["label"]:
            head += f", label {b['label']}"
        print(head)
        bits = []
        if b["gpu_util"]:
            bits.append(f"gpu_util {b['gpu_util']}%")
        if b["cpu_util"]:
            bits.append(f"cpu_util {b['cpu_util']}")
        if b["phase"]:
            bits.append(f"boot[{b['phase']}]")
        if b["idle"]:
            bits.append(f"idle {b['idle']}")
        if b["storage_day"]:
            bits.append(f"storage ${b['storage_day']}/day")
        if b["disk_gb"]:
            bits.append(f"disk {b['disk_used_gb'] or '?'}/{b['disk_gb']} GB "
                        f"used/alloc")
        bits.append(f"ssh {b['ssh']}")
        bits.append(f"image {b['image']}")
        print("  " + " · ".join(bits))
    h = d.get("health")
    if h:
        print(f"  !! {h.get('verdict')}: {h.get('reason')}")
    for j in d.get("jobs") or []:
        line = f"  job {j['cell']}"
        if j.get("rc") not in (None, 0):
            line += f" rc={j['rc']}"
        if j.get("fail_reason"):
            line += f" ({j['fail_reason']})"
        print(line)
    w = d.get("watch")
    if w:
        seg = f"  watch {w['watch']}: {w.get('state') or '-'}"
        if w.get("profile"):
            seg += f" (profile {w['profile']})"
        if w.get("budget_usd") is not None:
            seg += (f" — spent {fmt.dollars(w.get('spent_usd') or 0)} of "
                    f"{fmt.dollars(w['budget_usd'])}")
            if w.get("remaining_usd") is not None:
                seg += f", {fmt.dollars(w['remaining_usd'])} left"
        else:
            seg += (f" — spent {fmt.dollars(w.get('spent_usd') or 0)}, "
                    f"NO budget cap")
        if w.get("paused"):
            seg += " [PAUSED]"
        print(seg)
        if w.get("handed_off"):
            if w.get("current_box") == iid:
                print(f"    (this box replaced {w['watch']} — either id "
                      f"addresses the watch)")
            else:
                print(f"    REPLACED — the current box is "
                      f"{w.get('current_box')}; either id addresses the watch")
    elif d.get("watch_note"):
        print(f"  watch: none ({d['watch_note']})")


def run(a: argparse.Namespace) -> None:
    iid = str(a.id)
    d = _gather(iid)
    if getattr(a, "json", False):
        print(json.dumps(d, indent=2, sort_keys=True))
    else:
        _print_human(d)
    # gone AND unwatched is exit 2; a replaced box (watch still answers) is a
    # real answer, not an error
    if not d.get("found") and not d.get("watch"):
        sys.exit(2)


def add_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser],
               add_cmd: _args.AddCmd) -> argparse.ArgumentParser:
    p = add_cmd(sub, "box", "one box, one call: instance + jobs + watch/budget",
                _docs.DOC_OPERATIONS,
                _docs.DOC_README)
    p.add_argument("id", help="instance id — or an original watch key, which "
                              "answers with the box that replaced it")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=run)
    return p

"""vastlib.boxes.lifecycle — every call that starts, stops, renames or kills a box.

Why this module exists
----------------------
This is the money path. Each function below either starts billing, stops it, or
ends it permanently, and there is no undo for the last one. Collecting them
behind one boundary buys three things the flat file could not give:

* **One place where a mutation can be stubbed.** `_put_state_soft`,
  `_put_label_soft`, `_put_bid_soft`, `destroy_box` and `_destroy_soft` are the
  only functions in the package that issue a non-GET against a real instance id.
  86 `monkeypatch.setattr` sites across the suite exist to keep them away from
  the live fleet; they all keep working because every cross-module call here is
  in module-attribute form (`api.request_soft(...)`, never
  `from … import request_soft`).
* **The three-stop / two-destroy vocabulary, written down.** See below.
* **The revoke half of the teardown.** `_destroy_and_revoke` is the only caller
  of `_revoke_box_keys` in production: destroying a box without revoking its
  ephemeral B2 keys leaves live credentials pointing at a machine somebody else
  now rents.

Three stops, two destroys — they are NOT aliases
------------------------------------------------
Different callers bind different spellings, and the return shapes disagree.
Aliasing any pair of them is a silent arity bug at the binder:

  `stop_box(iid)`            -> `(ok, err)`   workflowctl's default stopper
  `_put_state_soft(iid, st)` -> `(ok, err)`   fleetd `Hooks.park` / `Hooks.resume`
  `_stop_instance_soft(iid)` -> `bool`        prints its own failure line

  `destroy_box(iid)`         -> `(ok, err)`   single shot, no retry; workflowctl
  `_destroy_soft(iid, ...)`  -> `(ok, err)`   4 tries, backoff, 404 == already
                                              gone == ok; fleetd `Hooks.destroy`

`fleetd.Hooks` returns the literal `(True, "dry-run")` from park/resume/destroy
when `self.dry_run` — slot 2 carries a HUMAN NOTE on the ok path. Nothing here
may assert `err is None` when `ok` is True (`core.result.OkErr` documents it).

The keep-token grammar is NOT re-implemented here
-------------------------------------------------
`_destroy_and_revoke` mints its revoke names through `models._label_value`, and
`boxes.reap` reads keep labels through `labels._reap_kept`. Neither parses the
token grammar itself. A second copy of those rules is what produced the
2026-08-02 un-revoked-key bug: `fleetd` appends `:keep` to a parked box's label,
so a fixed-width slice of `run:<RID>:keep` minted the revoke name
`run-<RID>:keep` and left the real `run-<RID>` key live on a destroyed box.

What is deliberately NOT here
-----------------------------
* **No keep/retention grammar** — `core.labels` owns it (above).
* **No guard verdict lattice, no health classification.** `_guard_fix_plan` and
  `cmd_guard` CALL into this module (`_destroy_and_revoke`, `stop_box`,
  `_emit_stopping_intent`); they live in `boxes.health` with the
  `GuardVerdict` lattice they are really about.
* **No reap policy.** The idle/zombie ledgers, their thresholds and `cmd_reap`
  are `boxes.reap` — a separate module because the systemd timer executes that
  one every 15 minutes and its two ledger schemas are byte-frozen.
* **No price arithmetic.** `_put_bid_soft` / `set_bid` perform the bid PUT and
  nothing else; what number to bid is `market.pricing`
  (`vastlib/market/__init__.py` states the same split from the other side).
* **No dead `_put_label_soft` twin.** `herdd.py` defines `_put_label_soft`
  TWICE — the first def (currently :4768, the plan text's ":4641" citation is
  stale) is shadowed at import by the second (:5319) and is dead code. Only the
  LIVE body is ported: `retries=2`, `return bool(ok), err`, and **no**
  `{"success": false}` folding. The dead twin folds `success: false` into `err`;
  adopting that stricter check is a parked behavior fix (plan §9), and deleting
  the dead def is plan §8 step 6 — not this port.
* **No `fleet_request`.** The daemon socket protocol is `fleet.client` (plan §5,
  step 5), a ring ABOVE this one. See the seam section below.

The cross-ring seam (new code, no `moved-from:` marker)
-------------------------------------------------------
Four names the ported bodies call live in rings this one may not import
(`fleet.client`, `jobs.*` — import-linter enforces the direction). They are
declared here as module attributes with their original names and signatures, so
that (a) the ported bodies stay verbatim, (b) the `monkeypatch.setattr` idiom
keeps working at the same attribute path, and (c) step 5/6 has one obvious place
to rebind. Until then they raise `NotImplementedError`, which is deliberate: a
silent no-op on `fleet_operator_intent` would let a human's stop read as OUTBID
and get the box RESCUED by the jobs ladder (SPOT_DESIGN §3.5), and that failure
is invisible. Nothing in `vastlib` is wired to a CLI before step 6, so the raise
costs nothing today and cannot be forgotten tomorrow. They carry no
`# moved-from:` marker on purpose (README §2 rule 7 — no marker means new code,
which is exactly the claim being made).

Provenance: moved from `tools/vast/herdd.py` (plan §8 step 3, 2026-08-16),
the lifecycle-mutation cluster plus `_destroy_soft` (:7239), `_revoke_box_keys`
(:7535, re-deferred here by `launch.json` and `ssh-remote.json`),
`_stop_instance_soft` (:18028) and `_get_instance` (:3330, re-deferred here by
`ssh-remote.json`). Behavior-preserving: bodies verbatim, annotations and
`core.result` types added. Step 3 is ADD-ONLY — `herdd.py` keeps its own
copies until step 6, so both are live and every existing `herdd.<name>` patch
still steers `herdd`'s own callers.

Plan §8 step 4 added the CREATE half — `_launch_preflight`, `launch_instance`,
`_emit_launched_soft` (all three ruled here by the orchestrator; see the section
banner for the two manifests that disagreed) plus `live_run_instances`, which
`_launch_preflight`'s dup guard cannot work without and which no step-3 manifest
claimed. Same add-only rules: `herdd.py` keeps its copies, and `launch/
launch.py`'s three raising seams for them are replaced by assignment rebinds off
this module.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import sys
import time
from typing import Any, Collection, Iterable, Sequence

import b2_mint_key

from vastlib.core import api, labels, machine_ledger, models, result
from vastlib.storage import b2

import bidpolicy
import jobmeta
import runmeta

# --------------------------------------------------------------------------- #
# CROSS-RING SEAM — new code, no `moved-from:` marker (README §2 rule 7).
# These four names are called by the verbatim bodies below but their real defs
# belong to rings this module may not import (import-linter: boxes -> fleet and
# boxes -> jobs are both upward). Declaring them here keeps the bodies verbatim
# and keeps the patch idiom at a stable attribute path; step 5 (fleet/, jobs/)
# rebinds them, step 6 wires the CLI. The raise is the reminder.
# --------------------------------------------------------------------------- #

_SEAM_HINT = ("not ported yet (plan §8 step 5) — rebind this module attribute, "
              "or stub it in your test with monkeypatch.setattr")


def fleet_operator_intent(iid: object, kind: str, reason: str | None = None) -> Any:  # noqa: ANN401 — daemon reply dict|None
    """SEAM for `herdd.fleet_operator_intent` -> `vastlib.fleet.client`.

    Tells a live daemon what a human is about to do to a box BEFORE the vast
    PUT/DELETE lands. Not a no-op stub on purpose: without the intent, the jobs
    ladder reads a human's stop as OUTBID and RESCUES the box (SPOT_DESIGN
    §3.5), which is exactly the kind of failure that is invisible until the
    invoice arrives.

    WIRED AT STEP 6, and this is the only one of the four here that is:
    `cli/_compose.py::bind()` assigns `fleet.client.fleet_operator_intent` onto
    this attribute when a COMMAND runs (never at import — read that module's
    docstring before patching this name in a test; patch the OWNER). The raise
    below is therefore what an unwired caller still gets, which is the point:
    every real caller (`cmd_stop`/`cmd_start`/`cmd_destroy`, `boxes.reap`)
    enters through the cli ring.
    """
    raise NotImplementedError(f"fleet_operator_intent: {_SEAM_HINT}")


def fleet_note_operator_stop(iid: object) -> Any:  # noqa: ANN401 — daemon reply dict|None
    """SEAM for `herdd.fleet_note_operator_stop` -> `vastlib.fleet.client`.

    Thin printer over `fleet_operator_intent(iid, "stop")`; its sole caller is
    `cmd_stop`, unguarded.
    """
    raise NotImplementedError(f"fleet_note_operator_stop: {_SEAM_HINT}")


def cmd_job_attach(a: argparse.Namespace) -> None:
    """SEAM for `herdd.cmd_job_attach` -> `vastlib.cli.job.attach`.

    `cmd_start` calls it to rotate a resumed jobs box's B2 key, already wrapped
    in `except SystemExit` / `except Exception`, so the raise degrades to the
    documented "auto-reattach failed — `herdd job attach <id>` to rotate"
    line rather than failing the resume.

    WIRED 2026-08-17 as `SEAM_BINDINGS` row five — the body lives ABOVE this
    ring (`cli/job/attach.py`; nothing below `cli` may import it), so the
    composition root is the only place that can close it. The raise below is
    what an UNWIRED process still gets, and that is not hypothetical: it is
    what the daemon got for a day. `job_lane._job_sup_reattach` and `cmd_start`
    both swallow it by design — an ssh refusal must not kill the babysitter or
    fail a resume — so the failure never reached a verdict and only ever
    surfaced as a `!!` line in the fleetd journal, three times in one window
    while every box came back holding its launch-baked B2 key
    (CREDENTIAL_LIFECYCLE.md: re-attach IS the rotation lane). A guarded caller
    makes an unbound seam quiet, not harmless.
    """
    raise NotImplementedError(f"cmd_job_attach: {_SEAM_HINT}")


# --------------------------------------------------------------------------- #
# Instance listing. Claimed here provisionally: no step-3 manifest owns these
# two, every driver below opens with one of them, and both are three-line
# wrappers over `core.api`. If a later step gives instance reads their own home
# the `moved-from:` markers make the second move mechanical.
# --------------------------------------------------------------------------- #

# moved-from: herdd._instances
def _instances() -> Any:  # noqa: ANN401 — vast returns a list, or a dict wrapping one
    d = api.request("GET", "v1/instances/")
    return d.get("instances", d) if isinstance(d, dict) else d


# moved-from: herdd._instances_soft
def _instances_soft() -> Any:  # noqa: ANN401 — mirrors _instances
    """Like _instances() but never sys.exits — returns [] on any API error, so
    read-only views (cmd_runs) and best-effort intent emits still proceed."""
    ok, d, _ = api.request_soft("GET", "v1/instances/")
    if not ok:
        return []
    return d.get("instances", d) if isinstance(d, dict) else d


# moved-from: herdd.find_matching_instance
def find_matching_instance(label: str) -> list[models.Payload]:
    """Argparse-free primitive: instances (BOTH running and stopped/parked)
    whose `label` matches exactly. Soft by contract (empty list on any API
    error, via `_instances_soft`) — the workflow controller reuses this to
    ADOPT an already-launched matching box instead of launching a duplicate
    (a parked twin still needs adopting, not just a live one).

    Ported here rather than into `workflows/ctl.py`, its only caller: it reads
    `_instances_soft` and belongs next to it, and duplicating a second
    instance-list filter one ring up is how two lookups drift apart.
    """
    return [i for i in _instances_soft() if i.get("label") == label]


# moved-from: herdd.live_run_instances
def live_run_instances(run_id: str | None = None,
                       instances: Sequence[models.Payload] | None = None,
                       ) -> list[models.Payload]:
    """Live vast instances labelled run:<run_id>. 'Live' = label == 'run:'+id
    AND actual_status in {running,loading,created}. run_id=None returns ALL live
    run:-labelled instances (cmd_runs builds a run_id->live-iids map in ONE API
    call). Pass instances= to reuse an already-fetched list."""
    ins = instances if instances is not None else _instances()
    out: list[models.Payload] = []
    for i in ins:
        rid = models._instance_run_label(i)
        if rid is None:
            continue
        if run_id is not None and rid != run_id:
            continue
        # `LIVE_STATES` is read off `bidpolicy` by module attribute, exactly as
        # `boxes.reap` reads it: the liveness vocabulary is Zone S's, shared
        # with the on-box ladder, and a second copy here is how the two drift.
        if (i.get("actual_status") or "").lower() in bidpolicy.LIVE_STATES:
            out.append(i)
    return out


# --------------------------------------------------------------------------- #
# Instance read + blocking wait
# --------------------------------------------------------------------------- #

# moved-from: herdd._get_instance
def _get_instance(iid: object) -> Any:  # noqa: ANN401 — one raw vast instance dict
    d = api.request("GET", f"v0/instances/{iid}/")
    return d.get("instances", d)


# moved-from: herdd._wait
def _wait(iid: object, target: str = "running", timeout: int = 600,
          interval: int = 8) -> bool:
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        st = _get_instance(iid).get("actual_status")
        if st != last:
            print(f"  [{int(time.time()%100000)}] status={st}")
            last = st
        if st == target:
            print(f"instance {iid} is {target}.")
            return True
        time.sleep(interval)
    sys.exit(f"error: timed out after {timeout}s waiting for {target} (last={last})")


# moved-from: herdd.cmd_wait
def cmd_wait(a: argparse.Namespace) -> None:
    # Function-local, and the two `boxes.ssh` call sites below are too:
    # `boxes.ssh` imports THIS module, so a module-level import would make the
    # pair a real cycle. Same ring, so import-linter is content either way;
    # keeping it local is the mechanical part of the port, and the call stays in
    # module-attribute form so the patch idiom survives.
    from vastlib.boxes import ssh
    _wait(a.id, target=a.state, timeout=a.timeout)
    if a.state == "running":
        ssh._print_ssh(a.id)


# --------------------------------------------------------------------------- #
# Intent emission — the B2 event schema is a frozen contract (`_last_stopping_
# actor` in supervise/ parses the `cli:` actor prefix these two write).
# --------------------------------------------------------------------------- #

# moved-from: herdd._cli_actor
def _cli_actor() -> str:
    return f"cli:{os.environ.get('HOSTNAME') or socket.gethostname()}"


# moved-from: herdd._run_id_for_instance
def _run_id_for_instance(iid: object,
                         instances: Sequence[models.Payload] | None = None) -> str | None:
    ins = instances if instances is not None else _instances_soft()
    for i in ins:
        if i.get("id") == iid:
            return models._instance_run_label(i)
    return None


# moved-from: herdd._emit_stopping_intent
def _emit_stopping_intent(iid: object, reason: str,
                          instances: Sequence[models.Payload] | None = None) -> None:
    """Best-effort append a `stopping` intent event (actor cli:<host>) BEFORE a
    vast stop/destroy. No-op for non-run:<ID> boxes or if B2/runmeta/rclone is
    unavailable — NEVER blocks the stop/destroy."""
    try:
        rid = _run_id_for_instance(iid, instances=instances)
        if not rid:
            return
        runmeta.emit_event(rid, "stopping", actor=_cli_actor(),
                           reason=reason, instance_id=iid)
    except Exception as e:
        print(f"note: could not emit stopping intent for {iid}: {e}",
              file=sys.stderr)


# moved-from: herdd._emit_resumed_intent
def _emit_resumed_intent(iid: object,
                         instances: Sequence[models.Payload] | None = None) -> None:
    """Best-effort `resumed` event after a successful start of a run:<ID> box.
    Clears the parked operator-stop intent so a later supervisor does not read
    the stale `stopping` event as operator intent (poll 2b /
    _last_stopping_actor). NEVER blocks the start."""
    try:
        rid = _run_id_for_instance(iid, instances=instances)
        if not rid:
            return
        runmeta.emit_event(rid, "resumed", actor=_cli_actor(), instance_id=iid)
    except Exception as e:
        print(f"note: could not emit resumed event for {iid}: {e}",
              file=sys.stderr)


# --------------------------------------------------------------------------- #
# The CREATE half — gate, PUT, record. Ported at plan §8 step 4 (the step-3
# lifecycle port landed the stop/start/destroy half without them).
#
# OWNERSHIP, recorded because two manifests disagreed. `market.json` deferred
# `launch_instance` / `_emit_launched_soft` to "launch/ and cli/";
# `launch.json` deferred `_launch_preflight` / `launch_instance` to "boxes/
# (lifecycle)" and `_emit_launched_soft` to "the supervise/journal
# neighbourhood, NOT launch". Orchestrator ruling at rev a1f2c8a5: ALL THREE
# land here. market.json's pointer is overruled; launch.json's boxes/ pointer
# is confirmed for two and extended to the third, which fits — this module
# already imports `runmeta` and owns `_cli_actor` / `_emit_stopping_intent` /
# `_emit_resumed_intent`, the same intent-emission neighbourhood, and the
# emitter's whole job is to record the box that `launch_instance` just rented.
#
# `launch/launch.py` no longer declares these three as raising seams: it binds
# them by module-level ASSIGNMENT off this module (see its rebind banner), so
# `monkeypatch.setattr(launch, "<name>", …)` keeps steering `_do_launch`.
# --------------------------------------------------------------------------- #

# moved-from: herdd._launch_preflight
def _launch_preflight(label: str | None, force: bool,
                      instances: Sequence[models.Payload] | None = None) -> None:
    """Refuse to launch a second box for a run that already has a vast instance.
    Gate ONLY on vast instances labelled run:<ID> (never on B2 history) —
    resume-by-RUN_ID with no twin passes clean. A LIVE twin is a double-writer;
    a STOPPED/parked twin still bills disk and vast may restart it later (also
    a double-writer) — resume or destroy it instead of launching over it.

    Handoff allowance (HANDOFF_DESIGN §2.1, T3): the understudy is deliberately a
    SECOND instance for the same run, labelled run:<ID>:handoff. That label's run
    id is '<ID>:handoff' — a DISTINCT id from the primary's '<ID>' — so the
    exact-match guard below (`live_run_instances`/`_instance_run_label` compare
    the whole suffix) refuses only ANOTHER understudy carrying the SAME :handoff
    label, and lets the live/parked primary run:<ID> stand (it must outlive the
    warmup and cutover). The allowance is narrowly scoped by that exact-match: a
    plain run:<ID> launch is UNCHANGED — it still refuses any live/parked
    run:<ID> twin — so the primary's dup guard is never weakened.

    Two port notes, both load-bearing:

    * The `sys.exit` calls STAY, and so does their message text. Callers assert
      on the substrings ('live instance [11]', 'STOPPED/parked', 'herdd start
      12', 'understudy'), and the run-lane understudy driver `_do_handoff_move`
      catches `SystemExit` specifically to abort a handoff with
      `understudy_unlaunchable`. A typed error here is a behavior change, not a
      cleanup.
    * `HANDOFF_LABEL_SUFFIX` is read from `core.labels`, never re-declared.
      A second copy of the suffix is the shape of the 2026-08-02 keep-token bug.
    """
    if force or not label or not label.startswith("run:"):
        return
    rid = label[len("run:"):]
    # noun for the refusal message: a run:<ID>:handoff launch that collides is a
    # duplicate UNDERSTUDY (not a duplicate primary) — clearer operator guidance.
    twin = "understudy" if rid.endswith(labels.HANDOFF_LABEL_SUFFIX) else "run"
    # reuse an already-fetched snapshot when the caller has one (the handoff
    # understudy launch passes st's per-tick instance list) — avoids a second hard
    # GET and lets the soft supervise loop drive this without a sys.exit-on-error API.
    ins = _instances() if instances is None else instances
    live = live_run_instances(rid, instances=ins)
    if live:
        ids = ", ".join(str(i.get("id")) for i in live)
        sys.exit(f"error: {twin} {rid!r} already has a live instance [{ids}]; "
                 f"refusing to launch a duplicate (double-writes the checkpoint). "
                 f"Destroy it first (herdd destroy {ids}) or pass --force.")
    parked = [i for i in ins if models._instance_run_label(i) == rid]
    if parked:
        ids = ", ".join(str(i.get("id")) for i in parked)
        sys.exit(f"error: {twin} {rid!r} has a STOPPED/parked instance [{ids}] "
                 f"(disk still billing; vast may restart it -> double-writer). "
                 f"Resume it (herdd start {ids} --wait 600 --retry 900) or "
                 f"destroy it (herdd destroy {ids} -y), or pass --force.")


# moved-from: herdd.launch_instance
def launch_instance(offer_id: object, body: dict[str, Any]) -> result.Soft:
    """Create one instance from an offer with a prepared body. Returns
    (ok, new_contract_id, err). Shared by cmd_launch and the supervisor's
    _relaunch so both go through the same PUT-and-parse. Soft: never sys.exit.

    THE MONEY MOVE — the only `PUT v0/asks/` in the tree, and the reason
    `api.request_soft` below is a MODULE ATTRIBUTE and not a `from … import`.
    conftest's `_block_mutating_api_calls` wraps the attribute
    `vastlib.core.api.request_soft`; a from-import here would bind past the
    guard and an unstubbed test would issue a real, billable PUT. Guarded, an
    unstubbed call returns the refusal triple and this function reports a
    launch failure — red, not a vacuous green. Keep it that way.

    Typing-forced change (the only one): the two bare failure tuples and the
    success tuple are constructed as `result.Soft`, matching the rest of this
    module. `NamedTuple` subclasses `tuple`, so every existing
    `== (True, 42, None)` and 3-way unpack is unaffected. `result.py`'s shape-A
    inventory misses this function for the same reason it missed
    `stop_box`/`destroy_box` in shape B — it was built from `_soft`-suffixed
    names."""
    ok, d, err = api.request_soft("PUT", f"v0/asks/{offer_id}/", body)
    if not ok:
        return result.Soft(False, None, err)
    if not isinstance(d, dict):
        return result.Soft(False, None, f"unexpected ask response: {d!r}")
    cid = d.get("new_contract")
    if cid:
        # vast allocates a real, BILLABLE contract for a bid launch while the
        # bid is still pending: it returns {"success": False, "new_contract": N,
        # "instance_api_key": ...} until the bid clears. Treat ANY response that
        # carries a new_contract as a launch and surface the id -- the caller
        # must know it to label/track/teardown the box, else success:False
        # orphans a billing instance silently (live-reproduced 2026-07-13).
        return result.Soft(True, cid, None)
    # no contract allocated -> a genuine failure (success:False or missing id)
    return result.Soft(False, None, f"launch failed: {d}")


#: Where a box's identity is filed. `jobs/nodes/<IID>/` is jobd's own per-box
#: segment, so the identity sits beside the hostfacts that need it.
#:
#: FULLY QUALIFIED, and that is the whole content of this constant. `rclone
#: rcat` takes a REMOTE-or-local path: handed a bare `jobs/nodes/...` it writes
#: a local file under the CWD, exits 0, and the `hard=False` caller reads that
#: as a successful B2 write. Shipped that way 2026-08-25 and measured the same
#: day: 0 identity objects in the bucket, one stray file in the repo root.
_IDENTITY_KEYFMT = "b2:{bucket}/jobs/nodes/{iid}/identity-{ts}.json"


def record_box_identity_soft(cid: object) -> str | None:
    """Write down `instance_id -> machine_id` at RENT TIME. Never raises.

    The same "last moment they exist" argument `_emit_launched_soft` makes for
    dlperf, applied to the mapping itself: vast drops the instance row when the
    box dies, and nothing else records it. Box-written hostfacts carry only
    `instance_id` (the box does not know its machine), and jobs-v2 events carry
    only `instance_id` too — verified 2026-08-24 — so a jobs-lane box has left
    no machine attribution anywhere at all.

    Deliberately NOT gated on a `run:` label. `_emit_launched_soft` returns
    early without one, which is exactly why the jobs lane has no record; every
    rented box needs this one regardless of what it is for.

    Written twice on purpose. The local ledger is what `ingest` reads; the B2
    object is what survives a laptop, and makes the mapping recoverable from a
    bare bucket copy with no API and no local state.
    """
    try:
        ok, resp, _ = api.request_soft("GET", f"v0/instances/{cid}/")
        if not ok:
            return None
        inst = (resp.get("instances", resp) if isinstance(resp, dict) else resp) or {}
        mid = inst.get("machine_id") if isinstance(inst, dict) else None
        if not mid:
            return None

        machine_ledger.record([(cid, mid)], now=time.time())

        bucket = os.environ.get("B2_BUCKET")
        if not bucket:
            return str(mid)             # ledger still has it; B2 is the spare
        ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        key = _IDENTITY_KEYFMT.format(bucket=bucket, iid=cid, ts=ts)
        blob = json.dumps({"instance_id": str(cid), "machine_id": str(mid),
                           "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                               time.gmtime()),
                           "source": "launch"}, sort_keys=True)
        b2._b2_rcat(key, blob, hard=False)
        return str(mid)
    except Exception:                   # noqa: BLE001
        # A metadata write must never fail a launch that already succeeded —
        # the same rule, and the same swallow, as `_emit_launched_soft`.
        return None


# moved-from: herdd._emit_launched_soft
def _emit_launched_soft(a: argparse.Namespace, body: dict[str, Any], cid: object,
                        offer_id: object, dph: object) -> None:
    """Best-effort runmeta `launched` for ANY run:-labelled box. Never raises.

    The RECORDING half of the dashboard's empty-column problem. Reading the
    fold back further tolerantly can only recover what an emitter wrote down;
    for a whole class of run nothing ever did. Until now the only emitter of a
    `launched` event was `cmd_train`, yet plain `herdd launch --label
    run:<id>` and the jobs/workflow arm launcher (which comes through this same
    function) emitted nothing at all — so those runs reach the dashboard with no
    gpu, no dph, no offer and no start time, and no amount of fold work fixes
    them, because their entire event log is the CLI's own later
    `stopping`/`resumed`. That is 19 of the 74 runs in the store today.

    `cmd_train` sets `_runmeta_launched` on its Namespace to suppress this one:
    it emits a richer event of its own (runset, config hash, host scorecard) a
    few lines after `_do_launch` returns, and two `launched` events in one epoch
    would make the newest-launch-like reads ambiguous for no gain.

    A metadata write must NEVER fail a launch that already succeeded, so every
    failure here is swallowed — the box is running either way. Do not narrow the
    bare `except Exception`, and do not let an annotation force an early raise:
    the instance GET sits inside the same guard on purpose."""
    if getattr(a, "_runmeta_launched", False):
        return
    run_id = models._label_value(body.get("label") or "", "run")
    if not run_id or not os.environ.get("B2_BUCKET"):
        return
    try:
        runmeta.validate_run_id(run_id)
    except runmeta.RunmetaError:
        return                          # a label we could not key an object on
    # ONE guard around the whole body, the instance GET included: the box is
    # already running by the time we get here, so nothing below may raise.
    try:
        fields: dict[str, Any] = {"instance_id": cid, "image": body.get("image"),
                                  "disk": body.get("disk"),
                                  "runtype": body.get("runtype")}
        if offer_id is not None:
            fields["offer_id"] = offer_id
        d = models._num_dph(dph)
        if d is not None:
            fields["dph"] = d           # the rate the cost estimate is built on
        ef = models._num_dph((body.get("env") or {}).get("ENTRY_FLOOR"))
        if ef is not None:
            fields["entry_floor"] = ef  # pre-rent market floor (defense controller)
        # the ACTUAL card + host scorecard, off the instance vast just created
        ok, resp, _ = api.request_soft("GET", f"v0/instances/{cid}/")
        if ok:
            inst = (resp.get("instances", resp)
                    if isinstance(resp, dict) else resp) or {}
            if isinstance(inst, dict):
                if inst.get("gpu_name"):
                    fields["gpu"] = inst["gpu_name"]
                # machine identity + the COMPUTE signals, captured here because
                # this is the last moment they exist: once a machine is fully
                # rented its offers leave the market, and dlperf/total_flops
                # become unrecoverable for that box forever. On 2026-08-06 that
                # made it unprovable whether box 47010337's offer had honestly
                # advertised the Max-Q part it turned out to be.
                #
                # These are the inputs the box-selection model runs on
                # (COMPUTE_OPTIMAL_BOX_SELECTION_2026-08-06.md): dlperf is the
                # only field that separates two hosts with byte-identical
                # total_flops and a 1.75x throughput gap, and it is re-sampled,
                # so the value AT RENT TIME is the one that explains the run.
                #
                # gpu_max_power is deliberately NOT here: the instance API does
                # not carry it (verified None on a live box). Its ground truth
                # is `gpu_plim` in the jobd heartbeat's host_metrics, which is
                # the on-box reading and strictly better than the advertised one.
                for k in ("machine_id", "inet_down", "geolocation",
                          "dlperf", "total_flops", "num_gpus", "gpu_ram",
                          "reliability2", "cuda_max_good"):
                    if inst.get(k) not in (None, ""):
                        fields[k] = inst[k]
        if not fields.get("gpu"):
            want = [g for g in (getattr(a, "gpu", None) or []) if g]
            if want:
                fields["gpu"] = ",".join(want)
        runmeta.emit_event(run_id, "launched", actor=_cli_actor(),
                           **{k: v for k, v in fields.items() if v is not None})
    except Exception as e:
        print(f">> note: runmeta launched-event emit failed ({e}) — non-fatal",
              file=sys.stderr)


# --------------------------------------------------------------------------- #
# The mutating PUTs. Everything below funnels through `api.request_soft`, which
# the conftest guard wraps for the whole suite — a non-GET from an unstubbed
# test is refused there, not sent.
# --------------------------------------------------------------------------- #

# moved-from: herdd._put_state_soft
def _put_state_soft(iid: object, state: str) -> result.OkErr:
    """PUT {"state": ...}. Vast can answer HTTP 200 with {"success": false,
    "msg": ...} (e.g. a start refused because the host's GPUs are re-rented) —
    surface that as an error too. Returns (ok, err)."""
    ok, d, err = api.request_soft("PUT", f"v0/instances/{iid}/", {"state": state})
    if not ok:
        return result.OkErr(False, err)
    if isinstance(d, dict) and d.get("success") is False:
        return result.OkErr(False, str(d.get("msg") or d))
    return result.OkErr(True, None)


# moved-from: herdd._put_label_soft
def _put_label_soft(iid: object, label: str) -> result.OkErr:
    """Set a box's label without exiting. Returns (ok, err). The ONE seam every
    automatic relabel goes through, so it is stubbable in the portable test lane
    (a label PUT is a real mutation of a real box — a test that reaches the vast
    API is a test that can rename someone's instance).

    Ported from the LIVE def (`herdd.py`:5319 at rev 2b188979), not the dead
    twin at :4768 that it shadows at import. The bodies differ and the
    difference is load-bearing: this one passes `retries=2` and returns
    `(bool(ok), err)` with NO `{"success": false}` folding, so a vast HTTP 200
    carrying `success: false` reads as SUCCESS here. Adopting the twin's
    stricter check is a parked behavior fix (plan §9); deleting the dead def is
    plan §8 step 6."""
    ok, _d, err = api.request_soft("PUT", f"v0/instances/{iid}/", {"label": label},
                                   retries=2)
    return result.OkErr(bool(ok), err)


# moved-from: herdd._put_bid_soft
def _put_bid_soft(iid: object, price: float) -> result.OkErr:
    """Change the standing bid on an EXISTING bid instance in place (running OR
    stopped): PUT v0/instances/bid_price/{id}/ (SPOT_DESIGN §3.2). Raising a
    stopped-because-outbid box above the machine's live min_bid makes vast
    auto-resume it. Returns (ok, err). A 429 (rate-limited) or transient error is
    soft — never fatal, never an eviction signal; the caller retries next poll.
    Vast can answer HTTP 200 with {"success": false, ...} — surface that too."""
    ok, d, err = api.request_soft("PUT", f"v0/instances/bid_price/{iid}/",
                                  {"client_id": "me", "price": float(price)})
    if not ok:
        return result.OkErr(False, err)
    if isinstance(d, dict) and d.get("success") is False:
        return result.OkErr(False, str(d.get("msg") or d))
    return result.OkErr(True, None)


# --------------------------------------------------------------------------- #
# The argparse-free primitives. `workflowctl` binds `stop_box` / `destroy_box`
# as its default stopper/destroyer and `bid_echo_probe` binds `set_bid`; their
# NAMES, ARITY and PARAMETER NAMES are a public contract
# (test_bid_echo_probe.py asserts `inspect.signature(set_bid).parameters ==
# ["iid", "price"]`).
# --------------------------------------------------------------------------- #

# moved-from: herdd.stop_box
def stop_box(iid: object) -> result.OkErr:
    """Argparse-free primitive: park one instance (GPU billing stops, disk
    persists on the same machine). Returns (ok, err). Soft by contract — never
    sys.exits; the workflow controller reuses this instead of shelling out to
    `herdd stop`."""
    return _put_state_soft(iid, "stopped")


# moved-from: herdd.destroy_box
def destroy_box(iid: object) -> result.OkErr:
    """Argparse-free primitive: soft DELETE of one instance. Returns (ok, err).
    Soft by contract — never sys.exits and does NOT revoke B2 keys (that stays
    in cmd_destroy); the workflow controller reuses this for teardown."""
    ok, _d, err = api.request_soft("DELETE", f"v0/instances/{iid}/")
    return result.OkErr(ok, err)


# A destroy that 404s reached the goal state: an instance vast does not have is
# the one box that certainly is NOT billing, so "retry, still billing" is the
# wrong answer. Substring match because `request_soft` flattens status and body
# into one error string. Incident: <bench>/archive/runs/2026-08-17-v13-chain-train.
_GONE_MARKERS = ("no_such_instance", "404")


def destroy_err_is_absent(err: object) -> bool:
    """True when a destroy failed because the instance is already gone.
    Idempotence, not leniency — every other failure leaves a box billing."""
    s = str(err or "").lower()
    return any(m in s for m in _GONE_MARKERS)


# moved-from: herdd.set_bid
def set_bid(iid: object, price: float) -> result.OkErr:
    """Argparse-free primitive: change the standing bid $/hr on an existing bid
    instance in place. Returns (ok, err). Soft by contract, no range check
    (cmd_bid keeps its own [0.001, 32] guard) — the workflow controller reuses
    this for defend/rescue."""
    return _put_bid_soft(iid, price)


# moved-from: herdd._stop_instance_soft
def _stop_instance_soft(iid: object) -> bool:
    """Best-effort park (GPU billing ends, disk kept).

    THIRD stop spelling, and the only one returning a BARE BOOL — see the module
    docstring. Not an alias of `stop_box`."""
    ok, err = _put_state_soft(iid, "stopped")
    if not ok:
        print(f"!! park failed for {iid}: {err} — park it by hand (herdd stop {iid})")
    return ok


# moved-from: herdd._destroy_soft
def _destroy_soft(iid: object, dry_run: bool = False, tries: int = 4) -> result.OkErr:
    """Destroy without sys.exit. 404 == already gone == ok. Retries transient
    failures so a husk never survives billing storage or becomes a double-writer.
    Returns (ok, err)."""
    if iid is None:
        return result.OkErr(True, None)
    if dry_run:
        print(f"[dry-run] would destroy husk {iid}")
        return result.OkErr(True, None)
    last = None
    for attempt in range(tries):
        ok, data, err = api.request_soft("DELETE", f"v0/instances/{iid}/", retries=2)
        if ok:
            return result.OkErr(True, None)
        if api._classify_http(err) == "fatal":
            if "HTTP 404" in (err or ""):
                return result.OkErr(True, None)       # already gone
            return result.OkErr(False, err)           # real fatal (401/403)
        last = err
        time.sleep(min(30.0, 2.0 * (2 ** attempt)))
    return result.OkErr(False, last)


# The destroy CONFIRMATION, homed with `_destroy_soft` because it is the second
# half of the same operation (`cli-surface.json` H3 / the raising seams
# `supervise/handoff.py` and `supervise/replacement.py` already carry, both of
# which name `boxes.lifecycle` as the target). It is shared by `supervise` and
# `job supervise` — two commands — so no `cli/<command>.py` can own it, and the
# supervise lanes may not import each other.
#
# 7 monkeypatch sites steer this name (`herdd-reexports.json`), which is why
# the two supervise modules bind it as their own module attribute rather than
# calling through `lifecycle.` — a patch of `lifecycle._confirm_gone` is NOT
# seen through those bindings. Closing those two seams is step 6d's job, not
# this file's.
# moved-from: herdd._confirm_gone
def _confirm_gone(iid: object, tries: int = 6) -> bool:
    """True once vast no longer reports the instance present (destroy confirmed).
    Treats {"instances": null}/None (HTTP 200 for a gone box) and HTTP 404 as
    gone. Enforces destroy-husk-before-relaunch (never launch a twin over a live
    husk)."""
    for _ in range(tries):
        ok, d, err = api.request_soft("GET", f"v0/instances/{iid}/", retries=2)
        if not ok:
            if "HTTP 404" in (err or ""):
                return True
            time.sleep(3)
            continue
        inst = d.get("instances", d) if isinstance(d, dict) else d
        if not inst:                                  # None / {"instances": null} / {}
            return True
        time.sleep(3)
    return False


# --------------------------------------------------------------------------- #
# Start-contention classification + the non-exiting state wait
# --------------------------------------------------------------------------- #

# A parked instance restarts on the SAME machine, so a start fails whenever
# someone else currently rents those GPUs. Vast phrases that refusal several
# ways; treat any of them as "busy: wait and retry", not a fatal.
# moved-from: herdd._START_BUSY_RE
_START_BUSY_RE = re.compile(
    r"insufficient|capacity|unavailable|not\s+available|no\s+(free|available)"
    r"|occupied|in\s+use|conflict|unable\s+to\s+(start|schedule)|try\s+again",
    re.I)


# moved-from: herdd._start_busy
def _start_busy(err: object) -> bool:
    """True when a start failure reads as GPU contention on the host
    (retryable by waiting) rather than a real fatal (auth, bad id, ...)."""
    return bool(err) and bool(_START_BUSY_RE.search(str(err)))


# moved-from: herdd._wait_states_soft
def _wait_states_soft(iid: object, targets: Collection[str], timeout: float,
                      interval: int = 8) -> result.OkData:
    """Poll one instance until actual_status lands in `targets`.
    Returns (ok, last_status). Never sys.exits — multi-id callers keep going.

    Shape C, not shape B: slot 2 is a STATUS STRING in both arms, never an
    error (`core.result.OkData`)."""
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        ok, d, _ = api.request_soft("GET", f"v0/instances/{iid}/", retries=2)
        if ok:
            inst = d.get("instances", d) if isinstance(d, dict) else d
            st = (inst or {}).get("actual_status")
            if st != last:
                print(f"  [{iid}] status={st}")
                last = st
            if st in targets:
                return result.OkData(True, st)
        time.sleep(interval)
    return result.OkData(False, last)


# --------------------------------------------------------------------------- #
# Operator drivers (`herdd stop|start|label|destroy|wait`)
# --------------------------------------------------------------------------- #

# moved-from: herdd.cmd_stop
def cmd_stop(a: argparse.Namespace) -> None:
    """Park: GPU billing ends, the container disk (image layers, weights,
    caches) persists on the SAME machine and keeps billing storage. A later
    `start` resumes in ~1-2 min instead of a full re-provision + re-pull."""
    ins = _instances_soft()
    failed = []
    for iid in a.id:
        _emit_stopping_intent(iid, "operator_stop", instances=ins)
        fleet_note_operator_stop(iid)      # B2: daemon must not rescue this park
        ok, err = stop_box(iid)
        if not ok:
            print(f"FAILED to stop {iid}: {err}", file=sys.stderr)
            failed.append(iid)
            continue
        print(f"stopping {iid}")
        if a.wait:
            okw, st = _wait_states_soft(iid, {"stopped", "exited"}, a.wait)
            if not okw:
                print(f"  [{iid}] not stopped after {a.wait}s (last={st}) — "
                      f"check `herdd show {iid}`", file=sys.stderr)
    okids = [i for i in a.id if i not in failed]
    if okids:
        print("parked: GPU billing stops; DISK keeps billing until destroy.\n"
              f"resume: herdd start {' '.join(map(str, okids))} --wait 600 --retry 900")
    if failed:
        sys.exit(f"error: could not stop {failed} — still billing GPU, retry!")


# moved-from: herdd._box_is_jobd
def _box_is_jobd(iid: object) -> bool:
    """True if this instance previously ran jobd (a jobs box), detected by its
    jobs/nodes/<IID>/ marker on B2 — ssh-free, best-effort. Any error => False
    (skip auto-reattach). Used by cmd_start to decide whether a resumed box
    needs its ephemeral B2 key rotated."""
    bucket = os.environ.get("B2_BUCKET")
    if not bucket:
        return False
    try:
        rc, out, _ = jobmeta._default_runner(          # type: ignore[no-untyped-call]
            ["lsf", f"b2:{bucket}/jobs/nodes/{iid}/"])
        return rc == 0 and bool((out or "").strip())
    except Exception:
        return False


# moved-from: herdd.cmd_start
def cmd_start(a: argparse.Namespace) -> None:
    """Resume a parked instance on its original machine (disk/caches intact).
    The start can be REFUSED while another renter holds the GPUs — --retry
    keeps re-asking; --wait then blocks until the container is running and
    prints the fresh ssh endpoint (host:port CHANGE across a park). For a jobs
    box (with --wait) the resume also auto-reattaches to rotate its B2 key, so a
    long park past the key TTL never strands the box (--no-reattach to skip)."""
    from vastlib.boxes import ssh  # local: boxes.ssh imports this module
    ins = _instances_soft()
    failed = []
    for iid in a.id:
        ok, err = _put_state_soft(iid, "running")
        deadline = time.time() + max(0, a.retry or 0)
        while not ok and _start_busy(err) and time.time() < deadline:
            print(f"  [{iid}] host GPUs busy ({err}); retrying for another "
                  f"{int(deadline - time.time())}s", file=sys.stderr)
            time.sleep(20)
            ok, err = _put_state_soft(iid, "running")
        if not ok:
            hint = ((" — the machine's GPUs are rented by someone else right "
                     "now. Retry later (--retry SECS keeps asking), or give up "
                     f"the warm disk: herdd destroy {iid} -y and relaunch.")
                    if _start_busy(err) else "")
            print(f"FAILED to start {iid}: {err}{hint}", file=sys.stderr)
            failed.append(iid)
            continue
        print(f"starting {iid}")
        _emit_resumed_intent(iid, instances=ins)
        fleet_operator_intent(iid, "start")   # B2: clears the dormant watch
        if a.wait:
            okw, st = _wait_states_soft(iid, {"running"}, a.wait)
            if not okw:
                print(f"  [{iid}] not running after {a.wait}s (last={st}) — if "
                      f"it sits 'stopped' the GPUs are likely re-rented; see "
                      f"`herdd show {iid}`", file=sys.stderr)
                failed.append(iid)
                continue
            i = _get_instance(iid)
            host, port, kind = ssh._pick_ssh_endpoint(i)
            print(f"  [{iid}] running.  ssh -p {port} root@{host}"
                  + (f"   ({kind} mapping)" if kind else ""))
            print(f"  [{iid}] NOTE: ssh endpoint + public port maps can CHANGE "
                  f"across a park (the api proxy endpoint may lag/stay stale — "
                  f"ssh/tunnel auto-probe the direct mapping); re-create "
                  f"tunnels (herdd tunnel {iid}); onstart re-ran on this boot.")
            print(f"  [{iid}] resume keeps the disk AS PARKED — refresh tooling "
                  f"with: herdd sync {iid}")
            # Autonomous key rotation: a resumed box reuses the launch-baked B2
            # key, which may have expired during a long park (the sole
            # suspend/outbid key-breakage vector). For a jobs box, re-attach
            # rotates a fresh scoped key onto it — best-effort, never fails the
            # resume. run:/serve: boxes rotate via supervise-relaunch /
            # launch_serve --on-box instead. (CREDENTIAL_LIFECYCLE.md)
            if not getattr(a, "no_reattach", False) and _box_is_jobd(iid):
                print(f"  [{iid}] jobs box — auto-reattach to rotate its B2 key "
                      f"(--no-reattach to skip)")
                try:
                    cmd_job_attach(argparse.Namespace(id=int(iid), dry_run=False))
                except SystemExit as e:
                    print(f"  [{iid}] auto-reattach skipped ({e}); onstart revives "
                          f"jobd with the baked key — `herdd job attach {iid}` "
                          f"to rotate by hand", file=sys.stderr)
                except Exception as e:
                    print(f"  [{iid}] auto-reattach failed ({type(e).__name__}: {e})"
                          f"; onstart revives jobd — `herdd job attach {iid}` to "
                          f"rotate", file=sys.stderr)
    if failed:
        sys.exit(f"error: could not start {failed}")


# moved-from: herdd.cmd_label
def cmd_label(a: argparse.Namespace) -> None:
    """The one label path that is NOT `_put_label_soft`: a HARD `api.request`,
    fatal on error. Deliberate — an operator typed this one."""
    api.request("PUT", f"v0/instances/{a.id}/", {"label": a.label})
    print(f"labeled {a.id} -> {a.label!r}")


# --------------------------------------------------------------------------- #
# Teardown: destroy + the G3 credential kill switch
# --------------------------------------------------------------------------- #

# moved-from: herdd._destroy_and_revoke
def _destroy_and_revoke(ids: Sequence[Any], label_ins: Sequence[models.Payload] | None,
                        intent: str, noun: str = "") -> list[Any]:
    """Destroy each instance in `ids` — `stopping` intent event first, then the
    soft DELETE, then the G3 kill switch (revoke the ephemeral B2 keys of every
    box that actually died: box-<IID> plus its run:/serve: alias). Keeps going
    on per-instance errors (a survivor still bills). Shared by
    destroy/guard/reap. Returns the ids that FAILED to destroy."""
    failed = []
    for iid in ids:
        _emit_stopping_intent(iid, intent, instances=label_ins)
        fleet_operator_intent(iid, "destroy", reason=intent)   # B2
        ok, err = destroy_box(iid)
        if ok:
            print(f"destroyed {noun}{iid}")
        elif destroy_err_is_absent(err):
            # Stays OUT of `failed` so the keys below still get revoked —
            # whoever won the destroy race may not have revoked them.
            print(f"already gone {noun}{iid} — vast has no such instance; "
                  f"nothing left to bill")
        else:
            print(f"FAILED to destroy {iid}: {err}", file=sys.stderr)
            failed.append(iid)
    revoke_names = set()
    for iid in ids:
        if iid in failed:
            continue
        revoke_names.add(f"box-{iid}")
        lab = next((i.get("label") or "" for i in (label_ins or [])
                    if i.get("id") == iid), "")
        # `_label_value`, NOT a fixed-width slice: fleetd appends `:keep` to the
        # label of every box it parks, so `run:<RID>:keep` used to mint the
        # revoke name `run-<RID>:keep` and leave the real `run-<RID>` key live.
        rid = models._label_value(lab, "run")
        sid = models._label_value(lab, "serve")
        if rid:
            revoke_names.add(f"run-{rid}")
        elif sid:
            revoke_names.add(f"serve-{sid}")
    _revoke_box_keys(revoke_names)
    return failed


# moved-from: herdd.cmd_destroy
def cmd_destroy(a: argparse.Namespace) -> None:
    ids = a.id
    ins = None
    if a.all:
        ins = _instances()
        ids = [i["id"] for i in ins]
    if not ids:
        print("nothing to destroy."); return   # noqa: E702 — verbatim body (plan §7.4)
    if not a.yes:
        ans = input(f"destroy {ids}? [y/N] ").strip().lower()
        if ans != "y":
            print("aborted."); return   # noqa: E702 — verbatim body (plan §7.4)
    label_ins = ins if ins is not None else _instances_soft()   # one lookup for the map
    failed = _destroy_and_revoke(ids, label_ins, "operator_destroy")
    if failed:
        sys.exit(f"error: could not destroy {failed} — still billing, retry!")


# moved-from: herdd._revoke_box_keys
def _revoke_box_keys(names: Iterable[str]) -> None:
    """Best-effort delete of ephemeral keys after teardown (G3 kill switch).
    Silent no-op without the minter pair; never blocks the caller.

    The `'<base>-'` PREFIX match is a contract, not an optimization: it is what
    tears down BOTH halves of a scoped `-ro`/`-rw` pair minted by
    `b2_mint_key.mint_pair` (documented from the other side at
    `b2_mint_key.py`:67,235 and `credbroker.py`:343). Narrowing it to an exact
    match silently orphans live keys on a destroyed box."""
    if not (os.environ.get("B2_MINTER_KEY_ID")
            and os.environ.get("B2_MINTER_APPLICATION_KEY")):
        return
    try:
        wanted = set()
        for nm in names:
            try:
                wanted.add(b2_mint_key.sanitize_name(nm))
            except b2_mint_key.MintError:
                continue
        auth = b2_mint_key._minter_auth()
        # Match the exact name (legacy single key) AND the '<base>-ro'/'-rw'
        # halves of a scoped pair (b2_mint_key.mint_pair) so teardown revokes
        # both keys of an Option-1b box. '<base>-' can't collide with a distinct
        # box: names carry full instance/run ids.
        def _match(kn: str) -> bool:
            return any(kn == w or kn.startswith(w + "-") for w in wanted)
        hit = [k for k in b2_mint_key.list_keys(auth) if _match(k["keyName"])]
        for k in hit:
            b2_mint_key.delete_key(auth, k["applicationKeyId"])
        if hit:
            print(f">> revoked {len(hit)} ephemeral B2 key(s) "
                  f"({', '.join(sorted(k['keyName'] for k in hit))})")
    except Exception as e:
        print(f">> note: ephemeral B2 key revoke skipped ({e})", file=sys.stderr)

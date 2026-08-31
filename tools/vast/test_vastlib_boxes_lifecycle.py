"""`vastlib.boxes.lifecycle` — the money path, characterized without touching a box.

Why this file exists
--------------------
Every function under test either starts billing, stops it, or ends a machine.
Nothing here may issue a real mutation, so the whole file drives the module over
a stubbed `vastlib.core.api.request_soft` (module attribute, replacing
conftest's live-fleet guard for the duration of the test) and asserts on the
RECORDED CALL rather than on any effect. No subprocess, no socket, no B2, no
rclone: `jobmeta._default_runner` and `b2_mint_key` are stubbed by module
attribute too.

What it pins, and why each one is worth a test
----------------------------------------------
1. **The three stop spellings and the two destroy spellings are distinct.**
   `stop_box` / `_put_state_soft` return `(ok, err)`; `_stop_instance_soft`
   returns a BARE BOOL; `destroy_box` is single-shot while `_destroy_soft`
   retries four times and treats 404 as success. fleetd binds the second and
   fifth, workflowctl the first and fourth — aliasing any pair is a silent arity
   bug at a binder, which is why the arities are asserted, not just the values.
2. **`_put_label_soft` is the LIVE def, not its dead twin.** The fat
   `herdd.py` defined the name twice; the surviving one passes `retries=2`
   and does NOT fold `{"success": false}` into an error. (The dead twin was
   deleted at plan §8 step 6 with the flat body — the launcher deliberately
   does NOT re-export this name, so `herdd._put_label_soft` no longer
   resolves at all and every patch site names `boxes.lifecycle`.) Both halves are asserted, because
   "port the live one" is otherwise unverifiable by reading the result.
3. **`_destroy_and_revoke` mints revoke names through `models._label_value`.**
   A `run:<RID>:keep` label must yield `run-<RID>`, not `run-<RID>:keep` — the
   2026-08-02 un-revoked-key bug. Also that a box which FAILED to destroy has
   its keys left alone, and that `fleet_operator_intent` is reached as a module
   attribute (the funnel `test_guard.py::_wire_guard` failed to patch in the
   2026-08-01 intent leak).
4. **The cross-ring seams raise.** `fleet_operator_intent`,
   `fleet_note_operator_stop` and `cmd_job_attach` are placeholders for symbols
   that land in `fleet/` and `jobs/` at plan §8 step 5. A silent no-op there
   would let a human's stop read as OUTBID and get the box rescued; the test
   fails the day someone "fixes" the raise into a `pass`.
5. **The CREATE half, added at plan §8 step 4** (section 10). `launch_instance`
   is the only `PUT v0/asks/` in the tree, so one test deliberately leaves it
   UNSTUBBED and asserts conftest's guard refuses it — that is what proves the
   body reaches `api.request_soft` by module attribute rather than through a
   `from … import` that would bind past the guard and rent a real box. The
   `success: False` + `new_contract` branch gets three tests of its own because
   that response is a box that is already billing. `_launch_preflight`'s exit
   MESSAGE TEXT is asserted (a caller catches `SystemExit` and operators read
   the text), and its `run:`-only scope is asserted as the reason the job lane's
   `job:<iid>:handoff` understudy has no dup guard — deliberate lane mirroring,
   plan §5, do not unify.

6. **`set_bid`'s signature.** `bid_echo_probe.py` calls
   `_herdd().set_bid(iid, float(price))` and `test_bid_echo_probe.py` asserts
   the parameter NAMES; the ported copy is pinned to the same pair here so the
   step-6 repoint cannot quietly reorder them.

Provenance: new in the vastlib package, plan §8 step 3 (`boxes/`). The expected
values are inherited from the then-live `herdd` copies, which
`test_lifecycle.py` asserted UNEDITED — the add-only port was proved by both
sides agreeing, not by a patch site moving. Step 6d deleted the flat copies;
`test_lifecycle.py` reaches this module now.
"""

from __future__ import annotations

import argparse
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vastlib.boxes import lifecycle  # noqa: E402
from vastlib.core import api, labels, models, result  # noqa: E402


class _Recorder:
    """A stub `request_soft` that records every call and replays a script."""

    def __init__(self, *replies):
        self.calls = []
        self._replies = list(replies)

    def __call__(self, method, path, body=None, **kw):
        self.calls.append({"method": method, "path": path, "body": body,
                           "kw": dict(kw)})
        if not self._replies:
            return result.Soft(True, {}, None)
        if len(self._replies) == 1:
            return self._replies[0]          # last reply repeats
        return self._replies.pop(0)


@pytest.fixture()
def rec(monkeypatch):
    """Replace `core.api.request_soft` (and with it conftest's guard) by a
    recorder. Module attribute, never `from … import` — that is the only form
    the ported bodies can be steered through."""
    def _mk(*replies):
        r = _Recorder(*replies)
        monkeypatch.setattr(api, "request_soft", r)
        return r
    return _mk


# --------------------------------------------------------------------------
# 1. The mutating PUTs
# --------------------------------------------------------------------------

def test_put_state_soft_ok(rec):
    r = rec(result.Soft(True, {}, None))
    assert lifecycle._put_state_soft(41, "stopped") == (True, None)
    assert r.calls[0]["method"] == "PUT"
    assert r.calls[0]["path"] == "v0/instances/41/"
    assert r.calls[0]["body"] == {"state": "stopped"}


def test_put_state_soft_folds_http_200_success_false(rec):
    """Vast answers 200 with `{"success": false, "msg": ...}` when a start is
    refused because the host's GPUs were re-rented. That is an ERROR here — and
    it is the difference from `_put_label_soft`, which does not fold it."""
    rec(result.Soft(True, {"success": False, "msg": "host GPUs unavailable"}, None))
    ok, err = lifecycle._put_state_soft(41, "running")
    assert (ok, err) == (False, "host GPUs unavailable")
    assert lifecycle._start_busy(err) is True     # and it reads as contention


def test_put_state_soft_transport_error_passes_err_through(rec):
    rec(result.Soft(False, None, "HTTP 403 on PUT v0/instances/41/: nope"))
    ok, err = lifecycle._put_state_soft(41, "running")
    assert ok is False
    assert "HTTP 403" in err


def test_put_bid_soft_shape_and_body(rec):
    r = rec(result.Soft(True, {}, None))
    assert lifecycle._put_bid_soft(41, 0.25) == (True, None)
    assert r.calls[0]["path"] == "v0/instances/bid_price/41/"
    assert r.calls[0]["body"] == {"client_id": "me", "price": 0.25}


def test_put_bid_soft_folds_success_false(rec):
    rec(result.Soft(True, {"success": False, "msg": "rate limited"}, None))
    ok, err = lifecycle._put_bid_soft(41, 0.25)
    assert (ok, err) == (False, "rate limited")


def test_put_label_soft_is_the_live_def_not_the_dead_twin(rec):
    """The fat `herdd.py` defined `_put_label_soft` twice; the LIVE one (the
    later def, which shadowed the earlier at import) passes `retries=2` and
    returns `(bool(ok), err)`. The dead twin passed no retries, and was deleted
    at plan §8 step 6 rather than ported — so this asserts the port took the
    right one of the two."""
    r = rec(result.Soft(True, {}, None))
    assert lifecycle._put_label_soft(41, "run:R1") == (True, None)
    assert r.calls[0]["body"] == {"label": "run:R1"}
    assert r.calls[0]["kw"].get("retries") == 2


def test_put_label_soft_does_not_fold_success_false(rec):
    """The other half of "which twin got ported": the LIVE body has no
    `{"success": false}` check, so a 200 carrying it reads as SUCCESS. Adopting
    the dead twin's stricter check is a parked behavior fix (plan §9) — if this
    test ever flips, that fix was applied as a drive-by."""
    rec(result.Soft(True, {"success": False, "msg": "nope"}, None))
    assert lifecycle._put_label_soft(41, "run:R1") == (True, None)


# --------------------------------------------------------------------------
# 2. Three stops, two destroys — distinct functions, distinct shapes
# --------------------------------------------------------------------------

def test_stop_box_is_put_state_stopped(rec):
    r = rec(result.Soft(True, {}, None))
    assert lifecycle.stop_box(41) == (True, None)
    assert r.calls[0]["body"] == {"state": "stopped"}


def test_stop_instance_soft_returns_a_bare_bool(rec, capsys):
    """THIRD stop spelling. Not an alias of `stop_box`: a bare bool, and it
    prints its own failure line. A binder that unpacks this as `(ok, err)`
    gets a TypeError, which is exactly why the arity is pinned."""
    rec(result.Soft(True, {}, None))
    got = lifecycle._stop_instance_soft(41)
    assert got is True and not isinstance(got, tuple)

    rec(result.Soft(False, None, "HTTP 500 on PUT"))
    got = lifecycle._stop_instance_soft(41)
    assert got is False and not isinstance(got, tuple)
    assert "park failed for 41" in capsys.readouterr().out


def test_destroy_box_is_single_shot(rec):
    r = rec(result.Soft(False, None, "HTTP 500 on DELETE"))
    ok, err = lifecycle.destroy_box(41)
    assert (ok, "HTTP 500" in err) == (False, True)
    assert len(r.calls) == 1                       # no retry loop of its own
    assert r.calls[0]["method"] == "DELETE"


def test_destroy_soft_treats_404_as_already_gone(rec):
    rec(result.Soft(False, None, "HTTP 404 on DELETE v0/instances/41/: gone"))
    assert lifecycle._destroy_soft(41) == (True, None)


def test_destroy_soft_stops_on_a_real_fatal(rec):
    r = rec(result.Soft(False, None, "HTTP 403 on DELETE v0/instances/41/: no"))
    ok, err = lifecycle._destroy_soft(41)
    assert ok is False and "HTTP 403" in err
    assert len(r.calls) == 1                       # fatal short-circuits


def test_destroy_soft_retries_transients_then_gives_up(rec, monkeypatch):
    monkeypatch.setattr(lifecycle.time, "sleep", lambda s: None)
    r = rec(*[result.Soft(False, None, "HTTP 503 on DELETE")] * 4)
    ok, err = lifecycle._destroy_soft(41, tries=4)
    assert ok is False and "HTTP 503" in err
    assert len(r.calls) == 4


def test_destroy_soft_dry_run_and_none_never_call_the_api(rec, capsys):
    r = rec()
    assert lifecycle._destroy_soft(None) == (True, None)
    assert lifecycle._destroy_soft(41, dry_run=True) == (True, None)
    assert r.calls == []
    assert "[dry-run] would destroy husk 41" in capsys.readouterr().out


def test_set_bid_signature_is_the_bid_echo_probe_contract():
    import inspect
    assert list(inspect.signature(lifecycle.set_bid).parameters) == ["iid", "price"]


# --------------------------------------------------------------------------
# 3. Start-contention classification
# --------------------------------------------------------------------------

@pytest.mark.parametrize("err", [
    "HTTP 400: insufficient capacity",
    "no free GPUs on this machine",
    "machine unavailable, try again later",
    "GPUs currently in use",
    "unable to start instance",
])
def test_start_busy_recognises_contention(err):
    assert lifecycle._start_busy(err) is True


@pytest.mark.parametrize("err", [None, "", "HTTP 401 unauthorized",
                                 "HTTP 404 no such instance"])
def test_start_busy_rejects_fatals(err):
    assert lifecycle._start_busy(err) is False


# --------------------------------------------------------------------------
# 4. _wait_states_soft — shape C, slot 2 is DATA
# --------------------------------------------------------------------------

def test_wait_states_soft_returns_status_in_both_arms(rec, monkeypatch):
    monkeypatch.setattr(lifecycle.time, "sleep", lambda s: None)
    rec(result.Soft(True, {"instances": {"actual_status": "stopped"}}, None))
    ok, st = lifecycle._wait_states_soft(41, {"stopped", "exited"}, 30)
    assert (ok, st) == (True, "stopped")

    rec(*[result.Soft(True, {"instances": {"actual_status": "running"}}, None)] * 50)
    ok, st = lifecycle._wait_states_soft(41, {"stopped"}, -1)
    assert ok is False and st is None              # deadline already past


# --------------------------------------------------------------------------
# 5. Intent emission — the B2 event schema and the `cli:` actor prefix
# --------------------------------------------------------------------------

def test_cli_actor_carries_the_frozen_prefix(monkeypatch):
    monkeypatch.setenv("HOSTNAME", "workstation")
    assert lifecycle._cli_actor() == "cli:workstation"


def test_emit_stopping_intent_is_a_noop_for_an_unlabelled_box(monkeypatch):
    seen = []
    monkeypatch.setattr(lifecycle.runmeta, "emit_event",
                        lambda *a, **k: seen.append((a, k)))
    lifecycle._emit_stopping_intent(41, "operator_stop",
                                    instances=[{"id": 41, "label": "scratch"}])
    assert seen == []


def test_emit_stopping_intent_writes_the_run_event(monkeypatch):
    seen = []
    monkeypatch.setenv("HOSTNAME", "workstation")
    monkeypatch.setattr(lifecycle.runmeta, "emit_event",
                        lambda *a, **k: seen.append((a, k)))
    lifecycle._emit_stopping_intent(41, "reap_idle_destroy",
                                    instances=[{"id": 41, "label": "run:R7"}])
    assert seen == [(("R7", "stopping"),
                     {"actor": "cli:workstation", "reason": "reap_idle_destroy",
                      "instance_id": 41})]


def test_emit_stopping_intent_never_blocks_the_mutation(monkeypatch, capsys):
    def _boom(*a, **k):
        raise RuntimeError("b2 down")
    monkeypatch.setattr(lifecycle.runmeta, "emit_event", _boom)
    lifecycle._emit_stopping_intent(41, "operator_stop",
                                    instances=[{"id": 41, "label": "run:R7"}])
    assert "could not emit stopping intent" in capsys.readouterr().err


def test_emit_resumed_intent_writes_the_resumed_event(monkeypatch):
    seen = []
    monkeypatch.setenv("HOSTNAME", "workstation")
    monkeypatch.setattr(lifecycle.runmeta, "emit_event",
                        lambda *a, **k: seen.append((a, k)))
    lifecycle._emit_resumed_intent(41, instances=[{"id": 41, "label": "run:R7"}])
    assert seen == [(("R7", "resumed"),
                     {"actor": "cli:workstation", "instance_id": 41})]


# --------------------------------------------------------------------------
# 6. _destroy_and_revoke — the money path and the revoke-name mint
# --------------------------------------------------------------------------

@pytest.fixture()
def wired(monkeypatch):
    """Every effect of `_destroy_and_revoke` replaced by a recorder. Patched by
    module attribute on `lifecycle` itself, which is the funnel the 2026-08-01
    intent leak proved must stay patchable: `test_guard.py::_wire_guard` stubbed
    destroy/emit/revoke but NOT `fleet_operator_intent`, and 20 real
    `operator_intent_destroy` events reached the live daemon."""
    log = {"destroyed": [], "intents": [], "stopping": [], "revoked": []}
    monkeypatch.setattr(lifecycle, "destroy_box",
                        lambda iid: log["destroyed"].append(iid) or result.OkErr(True, None))
    monkeypatch.setattr(lifecycle, "fleet_operator_intent",
                        lambda iid, kind, reason=None: log["intents"].append((iid, kind, reason)))
    monkeypatch.setattr(lifecycle, "_emit_stopping_intent",
                        lambda iid, reason, instances=None: log["stopping"].append((iid, reason)))
    monkeypatch.setattr(lifecycle, "_revoke_box_keys",
                        lambda names: log["revoked"].append(set(names)))
    return log


def test_destroy_and_revoke_mints_names_through_label_value(wired):
    """`run:<RID>:keep` -> `run-<RID>`, never `run-<RID>:keep`. fleetd appends
    `:keep` to every box it parks, so a fixed-width slice here left the real key
    live on a destroyed box (2026-08-02)."""
    ins = [{"id": 41, "label": "run:R7:keep"}, {"id": 42, "label": "serve:S3"}]
    assert lifecycle._destroy_and_revoke([41, 42], ins, "operator_destroy") == []
    assert wired["revoked"] == [{"box-41", "run-R7", "box-42", "serve-S3"}]
    assert wired["destroyed"] == [41, 42]
    assert wired["intents"] == [(41, "destroy", "operator_destroy"),
                                (42, "destroy", "operator_destroy")]
    assert wired["stopping"] == [(41, "operator_destroy"), (42, "operator_destroy")]


def test_destroy_and_revoke_leaves_a_survivors_keys_alone(monkeypatch, wired):
    """A box that failed to destroy still bills AND still needs its key."""
    monkeypatch.setattr(lifecycle, "destroy_box",
                        lambda iid: result.OkErr(False, "HTTP 500")
                        if iid == 42 else result.OkErr(True, None))
    failed = lifecycle._destroy_and_revoke([41, 42], [{"id": 41, "label": "run:R7"},
                                                      {"id": 42, "label": "run:R8"}],
                                           "reap_idle_destroy")
    assert failed == [42]
    assert wired["revoked"] == [{"box-41", "run-R7"}]


def test_destroy_err_is_absent_only_matches_a_gone_instance():
    """Pins that only an absence is forgiven; every other error still fails."""
    assert lifecycle.destroy_err_is_absent(
        "HTTP 404 on DELETE v0/instances/47935481/: {'success': False, "
        "'error': 'no_such_instance', 'msg': 'Instance 47935481 not found.'}")
    assert lifecycle.destroy_err_is_absent("no_such_instance")
    for still_billing in ("HTTP 500", "HTTP 429 rate limited", "HTTP 401 bad key",
                          "connection reset by peer", "timed out", None, ""):
        assert not lifecycle.destroy_err_is_absent(still_billing), still_billing


def test_destroy_and_revoke_treats_an_already_gone_box_as_done(monkeypatch, wired,
                                                               capsys):
    """Pins that a 404 destroy is not a failure to retry, and that its keys are
    still revoked."""
    monkeypatch.setattr(lifecycle, "destroy_box",
                        lambda iid: result.OkErr(
                            False, "HTTP 404 on DELETE v0/instances/42/: "
                                   "{'error': 'no_such_instance'}")
                        if iid == 42 else result.OkErr(True, None))
    failed = lifecycle._destroy_and_revoke([41, 42], [{"id": 41, "label": "run:R7"},
                                                      {"id": 42, "label": "run:R8"}],
                                           "operator_destroy")
    assert failed == []                       # nothing to retry
    assert wired["revoked"] == [{"box-41", "run-R7", "box-42", "run-R8"}]
    out = capsys.readouterr()
    assert "already gone 42" in out.out
    assert "nothing left to bill" in out.out
    assert "FAILED to destroy" not in out.err


def test_destroy_and_revoke_prefers_run_over_serve(wired):
    """The mint is an elif: a box labelled both ways revokes the run alias."""
    lifecycle._destroy_and_revoke([41], [{"id": 41, "label": "run:R7 serve:S3"}],
                                  "operator_destroy")
    assert wired["revoked"] == [{"box-41", "run-R7"}]


def test_destroy_and_revoke_label_lookup_survives_a_missing_map(wired):
    lifecycle._destroy_and_revoke([41], None, "operator_destroy")
    assert wired["revoked"] == [{"box-41"}]


def test_label_value_is_the_shared_grammar():
    """The port delegates, it does not re-derive. If this ever stops agreeing,
    a second copy of the keep-token rules has appeared somewhere."""
    assert models._label_value("run:R7:keep", "run") == "R7"
    assert models._label_value("run:R7 keep:evicted-until-20260805T183000Z",
                               "run") == "R7"


# --------------------------------------------------------------------------
# 7. _revoke_box_keys — the G3 kill switch and its PREFIX match
# --------------------------------------------------------------------------

class _FakeMint:
    MintError = ValueError

    def __init__(self, keys):
        self.keys = keys
        self.deleted = []

    def sanitize_name(self, n):
        return str(n)

    def _minter_auth(self):
        return {"auth": True}

    def list_keys(self, auth):
        return [{"keyName": k, "applicationKeyId": f"id-{k}"} for k in self.keys]

    def delete_key(self, auth, kid):
        self.deleted.append(kid)


def test_revoke_box_keys_is_a_silent_noop_without_the_minter_pair(monkeypatch):
    monkeypatch.delenv("B2_MINTER_KEY_ID", raising=False)
    monkeypatch.delenv("B2_MINTER_APPLICATION_KEY", raising=False)
    fake = _FakeMint(["box-41"])
    monkeypatch.setattr(lifecycle, "b2_mint_key", fake)
    lifecycle._revoke_box_keys({"box-41"})
    assert fake.deleted == []


def test_revoke_box_keys_matches_exact_and_scoped_pair(monkeypatch, capsys):
    """`'<base>-'` PREFIX, not equality: it is what tears down BOTH halves of a
    `-ro`/`-rw` scoped pair. `b2_mint_key.py`:67,235 and `credbroker.py`:343
    document the same contract from the other side; narrowing it orphans live
    keys on a destroyed box."""
    monkeypatch.setenv("B2_MINTER_KEY_ID", "k")
    monkeypatch.setenv("B2_MINTER_APPLICATION_KEY", "s")
    fake = _FakeMint(["box-41", "box-41-ro", "box-41-rw", "box-410", "run-R7"])
    monkeypatch.setattr(lifecycle, "b2_mint_key", fake)
    lifecycle._revoke_box_keys({"box-41"})
    assert sorted(fake.deleted) == ["id-box-41", "id-box-41-ro", "id-box-41-rw"]
    assert "revoked 3 ephemeral B2 key(s)" in capsys.readouterr().out


def test_revoke_box_keys_swallows_every_failure(monkeypatch, capsys):
    monkeypatch.setenv("B2_MINTER_KEY_ID", "k")
    monkeypatch.setenv("B2_MINTER_APPLICATION_KEY", "s")

    class _Boom(_FakeMint):
        def list_keys(self, auth):
            raise RuntimeError("b2 down")

    monkeypatch.setattr(lifecycle, "b2_mint_key", _Boom([]))
    lifecycle._revoke_box_keys({"box-41"})       # must not raise
    assert "ephemeral B2 key revoke skipped" in capsys.readouterr().err


# --------------------------------------------------------------------------
# 8. _box_is_jobd — B2 probe, never a subprocess in this lane
# --------------------------------------------------------------------------

def test_box_is_jobd_false_without_a_bucket(monkeypatch):
    monkeypatch.delenv("B2_BUCKET", raising=False)
    monkeypatch.setattr(lifecycle.jobmeta, "_default_runner",
                        lambda argv: pytest.fail("must not probe without a bucket"))
    assert lifecycle._box_is_jobd(41) is False


def test_box_is_jobd_reads_the_nodes_marker(monkeypatch):
    monkeypatch.setenv("B2_BUCKET", "bkt")
    seen = []
    monkeypatch.setattr(lifecycle.jobmeta, "_default_runner",
                        lambda argv: (seen.append(argv), (0, "jobd.json\n", ""))[1])
    assert lifecycle._box_is_jobd(41) is True
    assert seen == [["lsf", "b2:bkt/jobs/nodes/41/"]]


def test_box_is_jobd_any_failure_is_false(monkeypatch):
    monkeypatch.setenv("B2_BUCKET", "bkt")
    monkeypatch.setattr(lifecycle.jobmeta, "_default_runner",
                        lambda argv: (0, "   ", ""))
    assert lifecycle._box_is_jobd(41) is False

    def _boom(argv):
        raise RuntimeError("rclone missing")
    monkeypatch.setattr(lifecycle.jobmeta, "_default_runner", _boom)
    assert lifecycle._box_is_jobd(41) is False


# --------------------------------------------------------------------------
# 9. The cross-ring seams
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name, args", [
    ("fleet_operator_intent", (41, "stop")),
    ("fleet_note_operator_stop", (41,)),
    ("cmd_job_attach", (argparse.Namespace(id=41, dry_run=False),)),
])
def test_cross_ring_seams_raise_until_step_5_rebinds_them(name, args):
    """These three are NOT ports — they stand in for symbols landing in
    `fleet/` and `jobs/` at plan §8 step 5. The raise is load-bearing: a silent
    no-op on `fleet_operator_intent` means a human's stop reads as OUTBID and
    the jobs ladder rescues the box (SPOT_DESIGN §3.5), which nothing observes
    until the invoice. When step 5 rebinds them this test is what says so.

    All three are `SEAM_BINDINGS` rows now (`cmd_job_attach` joined 2026-08-17),
    so what this asserts is the UNBOUND default, which is exactly what a
    process that forgot to call `bind()` gets — the fleetd defect twice over.
    conftest's `_restore_cross_ring_seam_bindings` hands the census back after
    any test that binds, which is what keeps this order-independent."""
    with pytest.raises(NotImplementedError) as ei:
        getattr(lifecycle, name)(*args)
    assert "plan §8 step 5" in str(ei.value)


def test_cmd_stop_reaches_the_daemon_funnel(monkeypatch, rec):
    """`cmd_stop` calls `fleet_note_operator_stop` BEFORE the stop so the daemon
    marks the watch dormant. Asserted through the seam, which is also the proof
    the call is made by module attribute (a `from … import` would be
    unpatchable here)."""
    rec(result.Soft(True, {}, None))
    seen = []
    monkeypatch.setattr(lifecycle, "_instances_soft", lambda: [])
    monkeypatch.setattr(lifecycle, "_emit_stopping_intent",
                        lambda iid, reason, instances=None: None)
    monkeypatch.setattr(lifecycle, "fleet_note_operator_stop",
                        lambda iid: seen.append(iid))
    lifecycle.cmd_stop(argparse.Namespace(id=[41], wait=0))
    assert seen == [41]


# --------------------------------------------------------------------------
# 10. The CREATE half — live_run_instances, _launch_preflight, launch_instance,
#     _emit_launched_soft (plan §8 step 4).
#
# Port-time coverage for the four symbols the step-4 addition landed. The flat
# suite keeps its own 13 direct tests (`test_lifecycle.py` 80-154 / 768-812,
# `test_supervise.py` 1559-1616) UNEDITED under the §8 add-only amendment —
# they steer the still-live `herdd` copies. These are the same behaviours
# re-expressed against `vastlib`, so both copies are pinned independently and a
# drift in either direction shows up as a disagreement between two files.
# --------------------------------------------------------------------------

def _inst(iid, label, status):
    return {"id": iid, "label": label, "actual_status": status}


def test_live_run_instances_filters_by_label_and_live_state():
    ins = [_inst(11, "run:r1", "running"), _inst(12, "run:r1", "stopped"),
           _inst(13, "run:other", "running"), _inst(14, "scratch", "running"),
           _inst(15, None, "running")]
    assert [i["id"] for i in lifecycle.live_run_instances("r1", instances=ins)] == [11]


@pytest.mark.parametrize("status, live", [("running", True), ("loading", True),
                                          ("created", True), ("RUNNING", True),
                                          ("stopped", False), ("exited", False),
                                          ("", False), (None, False)])
def test_live_run_instances_reads_the_bidpolicy_vocabulary(status, live):
    """Liveness is `bidpolicy.LIVE_STATES` — Zone S's vocabulary, shared with
    the on-box ladder — and it is matched case-insensitively. A second copy of
    this set inside `vastlib` is how the workstation and the box drift apart."""
    ins = [_inst(11, "run:r1", status)]
    assert bool(lifecycle.live_run_instances("r1", instances=ins)) is live
    assert "running" in lifecycle.bidpolicy.LIVE_STATES


def test_live_run_instances_without_a_run_id_returns_every_live_run_box():
    """`run_id=None` is the `cmd_runs` shape: one API call, a map of every live
    run-labelled box."""
    ins = [_inst(11, "run:r1", "running"), _inst(12, "run:r2", "running"),
           _inst(13, "run:r3", "stopped"), _inst(14, "serve:s1", "running")]
    assert [i["id"] for i in lifecycle.live_run_instances(instances=ins)] == [11, 12]


def test_live_run_instances_fetches_when_no_snapshot_is_passed(monkeypatch):
    calls = []
    monkeypatch.setattr(lifecycle, "_instances",
                        lambda: calls.append(1) or [_inst(11, "run:r1", "running")])
    assert len(lifecycle.live_run_instances("r1")) == 1
    assert calls == [1]                      # the hard GET, exactly once
    lifecycle.live_run_instances("r1", instances=[])
    assert calls == [1]                      # a snapshot suppresses it


# --- _launch_preflight -----------------------------------------------------

def test_preflight_live_twin_blocks(monkeypatch):
    monkeypatch.setattr(lifecycle, "_instances",
                        lambda: [_inst(11, "run:r1", "running")])
    with pytest.raises(SystemExit) as e:
        lifecycle._launch_preflight("run:r1", force=False)
    assert "live instance [11]" in str(e.value)


def test_preflight_parked_twin_blocks_and_points_at_resume(monkeypatch):
    """A STOPPED twin still bills disk and vast may restart it — a second
    double-writer. The message names the resume command, and the text is
    asserted here for the same reason it is asserted flat: it is the operator's
    only instruction."""
    monkeypatch.setattr(lifecycle, "_instances",
                        lambda: [_inst(12, "run:r1", "stopped")])
    with pytest.raises(SystemExit) as e:
        lifecycle._launch_preflight("run:r1", force=False)
    msg = str(e.value)
    assert "STOPPED/parked" in msg and "herdd start 12" in msg


def test_preflight_force_passes(monkeypatch):
    monkeypatch.setattr(lifecycle, "_instances",
                        lambda: [_inst(12, "run:r1", "stopped")])
    lifecycle._launch_preflight("run:r1", force=True)      # no raise


def test_preflight_ignores_other_runs_and_unlabelled_boxes(monkeypatch):
    monkeypatch.setattr(lifecycle, "_instances",
                        lambda: [_inst(13, "run:other", "stopped"),
                                 _inst(14, "scratch", "running"),
                                 _inst(15, None, "stopped")])
    lifecycle._launch_preflight("run:r1", force=False)     # no raise


@pytest.mark.parametrize("label", [None, "", "upstream-monorepo", "job:41:handoff",
                                   "serve:s1"])
def test_preflight_fast_returns_on_a_non_run_label(monkeypatch, label):
    """The gate is `run:`-only, which is WHY the job lane's `job:<iid>:handoff`
    understudy has no twin-dup guard (plan §5 lane-mirroring NOTE — deliberate,
    do not unify). Asserted by the absence of any instance fetch."""
    monkeypatch.setattr(lifecycle, "_instances",
                        lambda: pytest.fail("a non-run: label must not fetch"))
    lifecycle._launch_preflight(label, force=False)


def test_preflight_uses_a_passed_snapshot_instead_of_a_second_get(monkeypatch):
    """`_do_handoff_move` is the only caller that passes `instances=` — it hands
    over the supervise tick's own list so the soft loop costs no extra hard GET.
    """
    monkeypatch.setattr(lifecycle, "_instances",
                        lambda: pytest.fail("instances= must suppress the GET"))
    with pytest.raises(SystemExit) as e:
        lifecycle._launch_preflight("run:r1", False,
                                    instances=[_inst(11, "run:r1", "running")])
    assert "live instance [11]" in str(e.value)


def test_preflight_handoff_twin_allowed_alongside_a_live_primary(monkeypatch):
    """HANDOFF_DESIGN §2.1 T3: the understudy is deliberately a SECOND box for
    the run, and its run-label `r1:handoff` is a DISTINCT id, so the exact-match
    guard lets the live primary stand (it must outlive the warmup)."""
    monkeypatch.setattr(lifecycle, "_instances",
                        lambda: [_inst(11, "run:r1", "running")])
    lifecycle._launch_preflight("run:r1:handoff", force=False)   # no raise


def test_preflight_plain_dup_still_refused_with_a_handoff_twin_present(monkeypatch):
    """The allowance must not weaken the primary's own dup guard."""
    monkeypatch.setattr(lifecycle, "_instances",
                        lambda: [_inst(11, "run:r1", "running"),
                                 _inst(12, "run:r1:handoff", "running")])
    with pytest.raises(SystemExit) as e:
        lifecycle._launch_preflight("run:r1", force=False)
    assert "live instance [11]" in str(e.value)


def test_preflight_second_understudy_refused_and_named_understudy(monkeypatch):
    monkeypatch.setattr(lifecycle, "_instances",
                        lambda: [_inst(11, "run:r1", "running"),
                                 _inst(12, "run:r1:handoff", "running")])
    with pytest.raises(SystemExit) as e:
        lifecycle._launch_preflight("run:r1:handoff", force=False)
    msg = str(e.value)
    assert "understudy" in msg and "live instance [12]" in msg


def test_preflight_reads_the_handoff_suffix_from_core_labels(monkeypatch):
    """The noun switch keys on `core.labels.HANDOFF_LABEL_SUFFIX`, never on a
    re-declared local copy — a second copy of a label token is the shape of the
    2026-08-02 un-revoked-key bug. Patching the shared constant must move the
    behaviour."""
    monkeypatch.setattr(lifecycle, "_instances",
                        lambda: [_inst(12, "run:r1:understudy", "running")])
    monkeypatch.setattr(labels, "HANDOFF_LABEL_SUFFIX", ":understudy")
    with pytest.raises(SystemExit) as e:
        lifecycle._launch_preflight("run:r1:understudy", force=False)
    assert "understudy " in str(e.value)     # the NOUN, not just the label text


# --- launch_instance — THE MONEY MOVE --------------------------------------

def test_launch_instance_puts_the_ask_with_the_prepared_body(rec):
    r = rec(result.Soft(True, {"success": True, "new_contract": 42}, None))
    assert lifecycle.launch_instance(36656763, {"image": "img:1"}) == (True, 42, None)
    assert r.calls == [{"method": "PUT", "path": "v0/asks/36656763/",
                        "body": {"image": "img:1"}, "kw": {}}]


def test_launch_instance_surfaces_the_contract_on_success_false(rec):
    """Vast's real BID response while the bid is pending: `success: False` WITH
    a live `new_contract` + `instance_api_key`. That contract is ALREADY
    BILLING. The pre-2026-07-13 code returned failure here and orphaned the box;
    any response carrying a `new_contract` is a launch."""
    rec(result.Soft(True, {"success": False, "new_contract": 44743274,
                           "instance_api_key": "FAKE-not-a-real-key"}, None))
    ok, cid, err = lifecycle.launch_instance(36656763, {"image": "alpine:latest"})
    assert ok and cid == 44743274 and err is None


def test_launch_instance_no_contract_is_a_genuine_failure(rec):
    rec(result.Soft(True, {"success": False, "msg": "no available gpus"}, None))
    ok, cid, err = lifecycle.launch_instance(1, {})
    assert not ok and cid is None and "no available gpus" in err


def test_launch_instance_http_error_passes_through(rec):
    rec(result.Soft(False, None, "HTTP 500 on PUT"))
    ok, cid, err = lifecycle.launch_instance(1, {})
    assert not ok and cid is None and "500" in err


def test_launch_instance_non_dict_response_is_a_failure(rec):
    rec(result.Soft(True, ["unexpected"], None))
    ok, cid, err = lifecycle.launch_instance(1, {})
    assert not ok and cid is None and "unexpected ask response" in err


def test_launch_instance_is_refused_by_the_conftest_guard_when_unstubbed():
    """THE GUARD TEST, and the reason `launch_instance` calls
    `api.request_soft` by MODULE ATTRIBUTE. This is the only `PUT v0/asks/` in
    the tree; conftest's `_block_mutating_api_calls` wraps that attribute, so an
    unstubbed call here is REFUSED and reported as a launch failure — red, not a
    vacuous green, and not a real rented box. A `from vastlib.core.api import
    request_soft` would bind past the guard and this test would spend money."""
    ok, cid, err = lifecycle.launch_instance(36656763, {"image": "img:1"})
    assert not ok and cid is None
    assert "test isolation" in err and "PUT v0/asks/36656763/" in err


def test_launch_instance_returns_a_shape_a_triple(rec):
    """The bare tuples became `result.Soft` at port time (the module's one
    documented typing change). `NamedTuple` subclasses `tuple`, so every
    existing 3-way unpack and tuple comparison is unaffected — asserted, because
    that equivalence is the whole licence for the change."""
    rec(result.Soft(True, {"new_contract": 42}, None))
    out = lifecycle.launch_instance(1, {})
    assert isinstance(out, tuple) and out == (True, 42, None)
    assert (out.ok, out.data, out.err) == (True, 42, None)


# --- _emit_launched_soft ---------------------------------------------------

@pytest.fixture()
def emitter(monkeypatch, rec):
    """B2 configured, `runmeta.emit_event` captured, the instance GET stubbed.
    Returns the list events land in. Nothing reaches B2, rclone or the API."""
    monkeypatch.setenv("B2_BUCKET", "bkt")
    monkeypatch.setenv("HOSTNAME", "workstation")
    emitted = []
    monkeypatch.setattr(lifecycle.runmeta, "emit_event",
                        lambda run_id, event, **f: emitted.append((run_id, event, f)))
    rec(result.Soft(True, {"gpu_name": "H100", "machine_id": 77,
                           "dlperf": 91.5, "num_gpus": 2}, None))
    return emitted


def test_emit_launched_soft_records_the_box_train_never_saw(emitter):
    """RECORDING fix: every run:-labelled launch writes a `launched` event, not
    just `cmd_train`'s. Asserts the price the cost estimate reads, the ACTUAL
    card rather than the selector, and the host scorecard captured at rent time
    (once a machine is fully rented its offer leaves the market and `dlperf` is
    unrecoverable for that box forever)."""
    a = argparse.Namespace(gpu=["h100"])
    lifecycle._emit_launched_soft(
        a, {"label": "run:myrun", "image": "img:1", "disk": 40,
            "runtype": "ssh_direct"}, 4242, offer_id=99, dph="1.25")
    assert len(emitter) == 1
    run_id, event, f = emitter[0]
    assert (run_id, event) == ("myrun", "launched")
    assert f["actor"] == "cli:workstation"
    assert f["instance_id"] == 4242 and f["offer_id"] == 99
    assert f["dph"] == 1.25                 # numeric: the cost estimate reads it
    assert f["gpu"] == "H100"               # the ACTUAL card, not the selector
    assert f["machine_id"] == 77 and f["dlperf"] == 91.5 and f["num_gpus"] == 2


def test_emit_launched_soft_stamps_the_entry_floor_from_the_body_env(emitter):
    """ENTRY_FLOOR is the PRE-RENT market read — the last uncontaminated one
    this box will ever give us — and it rides in on the launch body's env."""
    lifecycle._emit_launched_soft(
        argparse.Namespace(gpu=[]),
        {"label": "run:myrun", "env": {"ENTRY_FLOOR": "0.31"}}, 1, None, None)
    assert emitter[0][2]["entry_floor"] == 0.31


def test_emit_launched_soft_falls_back_to_the_gpu_selector(monkeypatch, emitter, rec):
    """No `gpu_name` on the instance -> the requested selector, joined. Only a
    fallback: the instance's own card is always preferred."""
    rec(result.Soft(True, {}, None))
    lifecycle._emit_launched_soft(argparse.Namespace(gpu=["H100", "", "A100"]),
                                  {"label": "run:myrun"}, 1, None, None)
    assert emitter[0][2]["gpu"] == "H100,A100"


def test_emit_launched_soft_is_a_noop_for_a_label_that_names_no_run(emitter):
    lifecycle._emit_launched_soft(argparse.Namespace(gpu=[]),
                                  {"label": "upstream-monorepo"}, 1, None, None)
    assert emitter == []


def test_emit_launched_soft_resolves_an_appended_keep_suffix(emitter):
    """`fleetd` appends `:keep` to a parked box's label; labels are appendable,
    so the run id is parsed through `models._label_value`, never sliced."""
    lifecycle._emit_launched_soft(argparse.Namespace(gpu=[]),
                                  {"label": "run:myrun:keep"}, 5, None, None)
    assert emitter[-1][0] == "myrun"


def test_emit_launched_soft_is_suppressed_by_cmd_trains_own_event(emitter):
    """`cmd_train` emits a richer `launched` a few lines later; two in one epoch
    make the newest-launch-wins reads ambiguous for no gain."""
    a = argparse.Namespace(gpu=[], _runmeta_launched=True)
    lifecycle._emit_launched_soft(a, {"label": "run:myrun"}, 7, 1, 1.0)
    assert emitter == []


def test_emit_launched_soft_drops_a_run_id_it_cannot_key_an_object_on(emitter):
    """`validate_run_id` refuses anything that would corrupt an object key; the
    emitter returns rather than writing a poisoned path."""
    lifecycle._emit_launched_soft(argparse.Namespace(gpu=[]),
                                  {"label": "run:../../etc"}, 1, None, None)
    assert emitter == []


def test_emit_launched_soft_never_fails_a_successful_launch(monkeypatch, capsys):
    """The box is ALREADY RUNNING by the time this is called. Every failure —
    the instance GET included, which is inside the same guard on purpose — is
    swallowed to stderr. Do not narrow the bare `except Exception`."""
    def boom(*a, **k):
        raise RuntimeError("B2 is down")
    monkeypatch.setenv("B2_BUCKET", "bkt")
    monkeypatch.setattr(lifecycle.runmeta, "emit_event", boom)
    monkeypatch.setattr(api, "request_soft", boom)
    lifecycle._emit_launched_soft(argparse.Namespace(gpu=[]),
                                  {"label": "run:r1"}, 1, 2, 0.5)
    assert "non-fatal" in capsys.readouterr().err


def test_emit_launched_soft_is_a_noop_without_a_bucket(monkeypatch):
    """No `B2_BUCKET` means no store to write to — and the check sits BEFORE the
    instance GET, so an unconfigured workstation issues no API call at all."""
    monkeypatch.delenv("B2_BUCKET", raising=False)
    monkeypatch.setattr(api, "request_soft",
                        lambda *a, **k: pytest.fail("must not reach the API"))
    monkeypatch.setattr(lifecycle.runmeta, "emit_event",
                        lambda *a, **k: pytest.fail("must not emit"))
    lifecycle._emit_launched_soft(argparse.Namespace(gpu=[]),
                                  {"label": "run:r1"}, 1, 2, 0.5)

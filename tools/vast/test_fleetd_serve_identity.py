"""fleetd knows WHAT a serve box is supposed to be serving, and acts when it isn't.

P2 taught the box to verify its own weights and stamp the grade-A sha12 into
the READY marker; nothing unattended read it. This file covers the half that
does: the watch carries the artifact it was registered for, the serve ladder
compares intent against the box's own claim every tick, and a disagreement
parks the box and withdraws it from the rescue/relaunch ladder.

Five layers, in the order a pin travels:

  1. THE GRAMMAR, from both ends. One table of marker lines is driven through
     `serve_ident` (python) and through `serve_ready.sh`'s own awk helpers
     (bash), so the two copies cannot drift into disagreeing about what a line
     means. The legacy shapes — no 4th field at all, and the `-` id
     placeholder — are in the table because they are the compatibility floor.
  2. THE CLASSIFIER, as a matrix. expected x observed, including every state
     that must NOT alarm (`off`, `unreadable`, `pending`).
  3. THE REGISTRATION, which is where a wrong pin is meant to die: at $0,
     against the committed registry, before the daemon carries it.
  4. THE LADDER TICK: what condemns, what only records, and the latch that
     stops a later rung from resurrecting a mismatched box.
  5. THE DAEMON: policy round-trip through the state file and across a
     restart, the derived alarms, the watch that is KEPT rather than ended,
     and the regression guard that a legacy serve watch produces exactly zero
     of any of it.

Toolchain-free: no vast API, no B2, no network. The one marker read is
`replacement._serve_status_line_soft`, monkeypatched at the module attribute
the way `test_boot_sla.py` steers it.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import serve_artifact  # noqa: E402
from vastlib.boxes import lifecycle  # noqa: E402
from vastlib.cli.fleet import watch as cli_watch  # noqa: E402
from vastlib.fleet import client, daemon  # noqa: E402
from vastlib.fleet import state as fleet_state  # noqa: E402
from vastlib.supervise import job_lane, replacement, serve_ident  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
READY_SH = os.path.join(_HERE, "serve_ready.sh")

bash_only = pytest.mark.skipif(not shutil.which("bash"), reason="needs bash")

NOW = 3_000_000.0
IDENT = "ad65f40a677e"          # mergeddemoa's grade-A sha12, from the committed registry
OTHER = "0123456789ab"

#: `(line, models-csv, ident)` — the grammar's whole surface, shared by the
#: python arm and the bash arm so neither can be "fixed" alone.
MARKER_TABLE = [
    # legacy: every marker written before the on-box gate existed
    ("READY 2026-08-24T00:00:00Z MERGEDDEMOA", "MERGEDDEMOA", ""),
    ("READY 2026-08-24T00:00:00Z a,b,c", "a,b,c", ""),
    ("READY 2026-08-24T00:00:00Z", "", ""),
    # gated
    ("READY 2026-08-24T00:00:00Z MERGEDDEMOA ident=%s" % IDENT, "MERGEDDEMOA", IDENT),
    ("READY 2026-08-24T00:00:00Z a,b ident=%s" % OTHER, "a,b", OTHER),
    # gated with no parseable id list: `-` is a placeholder, not a model
    ("READY 2026-08-24T00:00:00Z - ident=%s" % IDENT, "", IDENT),
]


# --------------------------------------------------------------------------- #
# 1. the grammar, pinned from BOTH ends
# --------------------------------------------------------------------------- #
def _bash_marker_field(line, fn):
    """Run one of serve_ready.sh's own parse helpers against a marker line."""
    src = open(READY_SH, encoding="utf-8").read()
    start = src.index("marker_models() {")
    end = src.index("# --- poll SERVE_STATUS")
    prog = ("#!/usr/bin/env bash\nset -euo pipefail\n" + src[start:end]
            + '\n%s "$1"\n' % fn)
    p = subprocess.run(["bash", "-c", prog, "_", line],
                       capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stderr
    return p.stdout.strip()


@pytest.mark.parametrize("line,models,ident", MARKER_TABLE)
def test_the_python_reader_parses_every_marker_shape(line, models, ident):
    assert serve_ident.marker_models(line) == models
    assert serve_ident.marker_ident(line) == ident


@bash_only
@pytest.mark.parametrize("line,models,ident", MARKER_TABLE)
def test_the_python_and_bash_readers_are_ONE_grammar(line, models, ident):
    """The point of the shared table. `serve_ready.sh` runs where our python is
    not guaranteed, so the grammar necessarily exists twice — this is what
    stops the second copy from being edited alone."""
    assert serve_ident.marker_models(line) == _bash_marker_field(
        line, "marker_models")
    assert serve_ident.marker_ident(line) == _bash_marker_field(
        line, "marker_ident")


@pytest.mark.parametrize("line,models,ident", MARKER_TABLE)
def test_the_detail_shape_agrees_with_the_whole_line_shape(line, models, ident):
    """`_serve_status_line_soft` hands its callers the line MINUS its token and
    timestamp. Both entry points must land on the same fields, or the daemon
    reads a different marker than serve_ready.sh does."""
    detail = " ".join(line.split()[2:])
    assert serve_ident.detail_fields(detail) == (models, ident)


def test_a_trailing_ident_never_reads_as_a_model_id():
    """Field 3 is the id CSV and `ident=` is field 4. A reader that scanned for
    the first token would serve an eval against a model named `ident=…`."""
    line = "READY 2026-08-24T00:00:00Z ident=%s" % IDENT
    # `ident=` in field 3 IS the id list positionally — the grammar is
    # positional on purpose, and this pins that reading rather than a
    # convenient one.
    assert serve_ident.marker_models(line) == "ident=%s" % IDENT
    assert serve_ident.marker_ident(line) == ""


def test_the_failed_reasons_are_the_four_the_box_can_write():
    """The set is a contract with onstart/serve_vllm.sh, and each has its own
    remedy because a mismatch (weights wrong) and a cannot-check (gate broken)
    send an operator to opposite halves of the system."""
    assert serve_ident.FAILED_REASONS == (
        "identity_mismatch", "identity_cannot_check",
        "identity_expect_missing", "identity_gate_missing")
    assert set(serve_ident.FAILED_REMEDY) == set(serve_ident.FAILED_REASONS)
    src = open(os.path.join(_HERE, "onstart", "serve_vllm.sh"),
               encoding="utf-8").read()
    for reason in serve_ident.FAILED_REASONS:
        assert f"status FAILED {reason}" in src, reason


# --------------------------------------------------------------------------- #
# 2. the classifier — expected x observed, every cell
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("expect,token,detail,state,observed,reason", [
    # no pin: `off` BEFORE the marker is even looked at
    (None, "READY", "m ident=%s" % OTHER, "off", None, None),
    ("", "READY", "m ident=%s" % OTHER, "off", None, None),
    # marker unreadable this tick — a B2 blip is not a poisoned serve
    (IDENT, None, None, "unreadable", None, None),
    # still booting: silence is not evidence
    (IDENT, "LAUNCHED", "", "pending", None, None),
    (IDENT, "PULLING", "base", "pending", None, None),
    (IDENT, "SELF_PARKED", "max_hours", "pending", None, None),
    # a FAILED nobody here owns stays somebody else's problem
    (IDENT, "FAILED", "oom", "pending", None, None),
    # the healthy shape
    (IDENT, "READY", "MERGEDDEMOA ident=%s" % IDENT, "verified", IDENT, None),
    (IDENT, "READY", "- ident=%s" % IDENT, "verified", IDENT, None),
    # case is not a difference of identity
    (IDENT, "READY", "m ident=%s" % IDENT.upper(), "verified", IDENT.upper(),
     None),
    # the poisoned-evals shape
    (IDENT, "READY", "MERGEDDEMOA ident=%s" % OTHER, "mismatch", OTHER, None),
    # legacy/unarmed: an absent claim is not a passing claim, and not a
    # mismatch either
    (IDENT, "READY", "MERGEDDEMOA", "unarmed", None, None),
    (IDENT, "READY", "", "unarmed", None, None),
    # the on-box gate's own refusals, reason preserved
    (IDENT, "FAILED", "identity_mismatch", "gate_failed", None,
     "identity_mismatch"),
    (IDENT, "FAILED", "identity_cannot_check", "gate_failed", None,
     "identity_cannot_check"),
    (IDENT, "FAILED", "identity_expect_missing", "gate_failed", None,
     "identity_expect_missing"),
    (IDENT, "FAILED", "identity_gate_missing", "gate_failed", None,
     "identity_gate_missing"),
])
def test_classify_matrix(expect, token, detail, state, observed, reason):
    rec = serve_ident.classify(expect, token, detail)
    assert rec["state"] == state
    assert rec["observed"] == observed
    assert rec["reason"] == reason
    assert rec["state"] in serve_ident.STATES


def test_only_three_states_are_worth_waking_someone_for():
    """`verified` and `pending` are healthy; `off`/`unreadable` mean this
    instrument has nothing to say. Alarming on `unreadable` would make every
    network wobble look like a poisoned serve."""
    assert serve_ident.ALARM_STATES == ("mismatch", "unarmed", "gate_failed")


# --------------------------------------------------------------------------- #
# 3. registration — where a wrong pin dies, at $0
# --------------------------------------------------------------------------- #
def _watch_ns(**kw):
    base = dict(target="47", profile="serve", budget=5.0, max_bid=None,
                keep=False, standing=False, reset_spend=False,
                no_handoff=False, strict_ceiling=False, rescue_wait=None,
                interval=45, wall_budget=48 * 3600, max_relaunch=3,
                artifact=None, expect_ident=None)
    base.update(kw)
    return argparse.Namespace(**base)


def test_the_registry_composes_the_same_sha_the_box_will_stamp():
    """The pin and the box's `identity_expect.json` come from one composer, so
    a registry edit moves both or neither."""
    want, art = cli_watch._registry_ident("mergeddemoa")
    doc = serve_artifact.compose_expectation(
        serve_artifact.registry.get("mergeddemoa", None))
    assert art == "mergeddemoa"
    assert want == doc["fingerprint_sha256"][:cli_watch.IDENT_SHA_LEN] == IDENT


def test_artifact_alone_derives_the_pin():
    assert cli_watch._resolve_identity_pin(
        _watch_ns(artifact="mergeddemoa")) == ("mergeddemoa", IDENT)


def test_a_matching_expect_ident_is_accepted_and_normalised():
    assert cli_watch._resolve_identity_pin(
        _watch_ns(artifact="mergeddemoa",
                  expect_ident=IDENT.upper())) == ("mergeddemoa", IDENT)


def test_an_expect_ident_that_disagrees_with_the_registry_is_REFUSED():
    """The $0 gate. A pin no correct box could satisfy would alarm forever on a
    box doing nothing wrong."""
    with pytest.raises(SystemExit) as e:
        cli_watch._resolve_identity_pin(
            _watch_ns(artifact="mergeddemoa", expect_ident=OTHER))
    msg = str(e.value.code)
    assert "does NOT match" in msg and IDENT in msg and OTHER in msg
    assert "Nothing has been registered" in msg


def test_expect_ident_without_an_artifact_is_REFUSED():
    """Nothing to verify it against: a typo and a genuinely wrong box look
    identical to the daemon."""
    with pytest.raises(SystemExit) as e:
        cli_watch._resolve_identity_pin(_watch_ns(expect_ident=IDENT))
    assert "needs --artifact" in str(e.value.code)


@pytest.mark.parametrize("profile", ["jobs", "run", "bare"])
def test_a_pin_on_a_profile_with_no_marker_is_REFUSED(profile):
    """Only a serve box writes a SERVE_STATUS marker, so the pin would be
    inert — and a gate that silently does nothing is worse than no gate,
    because the caller believes it ran."""
    with pytest.raises(SystemExit) as e:
        cli_watch._resolve_identity_pin(
            _watch_ns(profile=profile, artifact="mergeddemoa"))
    assert "--profile serve" in str(e.value.code)


def test_an_unknown_slug_is_REFUSED():
    with pytest.raises(SystemExit) as e:
        cli_watch._resolve_identity_pin(_watch_ns(artifact="no-such-artifact"))
    assert "no-such-artifact" in str(e.value.code)


#: A base entry with NULL pins, in a registry of its own. Committed seeds get
#: measured as merges publish them, so the unpinned case has no permanent live
#: example to point this at. `fingerprint_sha256` is not a key a `base` entry
#: may carry, so grade B is the only pin that can move it.
UNPINNED_BASE = {"schema_version": 1, "id": "unpinned-fixture", "kind": "base",
                 "b2_root": "base-models/unpinned-fixture",
                 "content_sha256": None, "n_files": None}


def _fixture_registry(tmp_path, monkeypatch, **over):
    """Point the registry at a one-entry fixture directory.

    The registry DIRECTORY is the seam, not `_registry_ident` — the refusal
    under test is composed inside that call, so stubbing the lookup would test
    the stub. Monkeypatched, so no operator invocation can reach it: the
    committed registry is the trust anchor and has no runtime override.
    """
    entry = dict(UNPINNED_BASE, **over)
    d = tmp_path / "registry"
    d.mkdir(exist_ok=True)
    (d / f"{entry['id']}.json").write_text(json.dumps(entry))
    monkeypatch.setattr(serve_artifact.registry, "REGISTRY_DIR", str(d))
    return entry["id"]


def test_an_artifact_with_no_identity_pin_is_REFUSED(tmp_path, monkeypatch):
    """An artifact with null pins: registering a watch against it would arm a
    check that can never fire."""
    slug = _fixture_registry(tmp_path, monkeypatch)
    with pytest.raises(SystemExit) as e:
        cli_watch._resolve_identity_pin(_watch_ns(artifact=slug))
    assert "NO identity pin" in str(e.value.code)


def test_a_content_rollup_ALONE_never_becomes_a_watch_pin(tmp_path, monkeypatch):
    """`ident=` is the grade-A fingerprint truncated, so an artifact with only a
    content rollup has nothing to pin to. Composing one anyway would register
    the literal `None` and alarm forever against a box doing nothing wrong."""
    slug = _fixture_registry(tmp_path, monkeypatch,
                             content_sha256="ab" * 32, n_files=26)
    with pytest.raises(SystemExit) as e:
        cli_watch._resolve_identity_pin(_watch_ns(artifact=slug))
    assert "NO identity pin" in str(e.value.code)


def test_no_flags_means_no_pin_and_no_registry_read(monkeypatch):
    monkeypatch.setattr(cli_watch, "_registry_ident",
                        lambda s: pytest.fail("unpinned watch read the registry"))
    assert cli_watch._resolve_identity_pin(_watch_ns()) == (None, None)


def test_the_client_sends_the_pin_in_the_policy(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(client, "_fleet_call_or_die",
                        lambda op, **kw: (calls.append((op, kw)),
                                          {"target": "47", "profile": "serve",
                                           "budget_usd": 5.0})[1])
    monkeypatch.setattr(client, "_fleet_requester", lambda: "tester")
    cli_watch.run(_watch_ns(artifact="mergeddemoa"))
    _op, kw = calls[0]
    assert kw["policy"]["artifact"] == "mergeddemoa"
    assert kw["policy"]["expect_ident"] == IDENT
    out = capsys.readouterr().out
    assert "IDENTITY: pinned to artifact mergeddemoa" in out and IDENT in out


# --------------------------------------------------------------------------- #
# 4. the ladder tick
# --------------------------------------------------------------------------- #
def _args(**kw):
    base = dict(id=41, dry_run=False, budget=5.0, max_bid=None, handoff=True,
                strict_ceiling=False, keep=False, serve_mode=True,
                artifact="mergeddemoa", expect_ident=IDENT)
    base.update(kw)
    return argparse.Namespace(**base)


def _jc(**kw):
    jc, _hf = job_lane.job_supervise_init(_args(**kw.pop("args", {})))
    jc.update(kw)
    return jc


def _inst(iid=41, status="running"):
    return {"id": iid, "actual_status": status, "machine_id": 7,
            "label": "serve:serve-260824-0001", "num_gpus": 1}


def _marker(monkeypatch, token, detail):
    monkeypatch.setattr(replacement, "_serve_status_line_soft",
                        lambda sid: (token, NOW - 5, detail))


def _no_park(monkeypatch):
    parked = []
    monkeypatch.setattr(lifecycle, "_stop_instance_soft",
                        lambda iid: (parked.append(str(iid)), True)[1])
    return parked


def test_the_pin_reaches_the_ladder_context():
    jc = _jc()
    assert jc["model_artifact"] == "mergeddemoa" and jc["expect_ident"] == IDENT


def test_a_verified_box_is_recorded_and_never_condemned(monkeypatch):
    _marker(monkeypatch, "READY", "MERGEDDEMOA ident=%s" % IDENT)
    jc = _jc()
    assert replacement._serve_identity_tick(jc, _inst(), NOW) is None
    assert jc["serve_identity"]["state"] == "verified"
    assert "serve_identity_condemned" not in jc


def test_a_mismatch_parks_the_box_and_returns_the_verdict(monkeypatch, capsys):
    _marker(monkeypatch, "READY", "MERGEDDEMOA ident=%s" % OTHER)
    parked = _no_park(monkeypatch)
    jc = _jc()
    assert replacement._serve_identity_tick(jc, _inst(), NOW) == "identity_mismatch"
    assert parked == ["41"]
    rec = jc["serve_identity"]
    assert (rec["state"], rec["observed"], rec["expected"]) == (
        "mismatch", OTHER, IDENT)
    assert rec["parked"] is True
    assert jc["serve_identity_condemned"] is True
    assert "SERVE IDENTITY MISMATCH" in capsys.readouterr().out
    ev = [e for e, _f in jc["ladder_journal"]]
    assert ev == ["serve_identity_mismatch"]


def test_the_condemn_latch_costs_no_further_marker_reads(monkeypatch):
    _marker(monkeypatch, "READY", "MERGEDDEMOA ident=%s" % OTHER)
    _no_park(monkeypatch)
    jc = _jc()
    assert replacement._serve_identity_tick(jc, _inst(), NOW) == "identity_mismatch"
    monkeypatch.setattr(replacement, "_serve_status_line_soft",
                        lambda sid: pytest.fail("a condemned box was re-read"))
    assert replacement._serve_identity_tick(
        jc, _inst(), NOW + 45) == "identity_mismatch"


def test_a_dry_run_condemns_without_touching_the_box(monkeypatch, capsys):
    _marker(monkeypatch, "READY", "MERGEDDEMOA ident=%s" % OTHER)
    monkeypatch.setattr(lifecycle, "_stop_instance_soft",
                        lambda iid: pytest.fail("dry run stopped a box"))
    jc = _jc(args={"dry_run": True})
    assert replacement._serve_identity_tick(jc, _inst(), NOW) == "identity_mismatch"
    assert jc["serve_identity"]["parked"] is False
    assert "[dry-run] would PARK" in capsys.readouterr().out


def test_a_failed_park_still_condemns_and_says_so(monkeypatch, capsys):
    _marker(monkeypatch, "READY", "MERGEDDEMOA ident=%s" % OTHER)
    monkeypatch.setattr(lifecycle, "_stop_instance_soft", lambda iid: False)
    jc = _jc()
    assert replacement._serve_identity_tick(jc, _inst(), NOW) == "identity_mismatch"
    assert jc["serve_identity"]["parked"] is False
    assert jc["serve_identity_condemned"] is True     # the withdrawal is not
    assert "PARK FAILED" in capsys.readouterr().out   # conditional on the park


@pytest.mark.parametrize("token,detail,state", [
    ("READY", "MERGEDDEMOA", "unarmed"),
    ("FAILED", "identity_mismatch", "gate_failed"),
    ("FAILED", "identity_cannot_check", "gate_failed"),
    ("LAUNCHED", "", "pending"),
])
def test_only_a_live_mismatch_condemns(monkeypatch, token, detail, state):
    """Loud but PASSIVE. An unarmed box never gated itself and a FAILED box
    already refused to serve — parking either out from under an operator
    mid-diagnosis helps nobody, and neither is poisoning an eval."""
    _marker(monkeypatch, token, detail)
    monkeypatch.setattr(lifecycle, "_stop_instance_soft",
                        lambda iid: pytest.fail("passive state parked a box"))
    jc = _jc()
    assert replacement._serve_identity_tick(jc, _inst(), NOW) is None
    assert jc["serve_identity"]["state"] == state


def test_an_unreadable_marker_gives_no_verdict(monkeypatch):
    _marker(monkeypatch, None, None)
    monkeypatch.setattr(lifecycle, "_stop_instance_soft",
                        lambda iid: pytest.fail("a B2 blip parked a box"))
    jc = _jc()
    assert replacement._serve_identity_tick(jc, _inst(), NOW) is None
    assert jc["serve_identity"]["state"] == "unreadable"


def test_the_state_clock_survives_a_tick_and_resets_on_a_change(monkeypatch):
    _marker(monkeypatch, "READY", "MERGEDDEMOA")
    jc = _jc()
    replacement._serve_identity_tick(jc, _inst(), NOW)
    replacement._serve_identity_tick(jc, _inst(), NOW + 45)
    assert jc["serve_identity"]["since"] == NOW      # same state, same clock
    _marker(monkeypatch, "READY", "MERGEDDEMOA ident=%s" % IDENT)
    replacement._serve_identity_tick(jc, _inst(), NOW + 90)
    assert jc["serve_identity"]["since"] == NOW + 90


def test_an_unpinned_watch_reads_nothing_and_leaves_nothing(monkeypatch):
    """The compatibility floor for every serve watch registered before P3."""
    monkeypatch.setattr(replacement, "_serve_status_line_soft",
                        lambda sid: pytest.fail("an unpinned watch read B2"))
    jc = _jc(args={"artifact": None, "expect_ident": None},
             serve_identity={"state": "mismatch"}, serve_identity_condemned=True)
    assert replacement._serve_identity_tick(jc, _inst(), NOW) is None
    # dropping the pin RELEASES the box: a latch that outlived the pin it was
    # made against is an alarm with no way to retract it
    assert "serve_identity" not in jc
    assert "serve_identity_condemned" not in jc


def test_no_instance_this_tick_is_no_verdict(monkeypatch):
    monkeypatch.setattr(replacement, "_serve_status_line_soft",
                        lambda sid: pytest.fail("no box, no marker read"))
    jc = _jc()
    assert replacement._serve_identity_tick(jc, None, NOW) is None


def test_the_verdict_is_in_the_control_contract():
    assert replacement.SERVE_IDENTITY_VERDICT in job_lane.JOB_SUP_VERDICTS


def test_the_tick_calls_the_check_before_every_other_rung():
    """Ordering is the whole design: stop-classify, bid rescue, the boot SLA
    and the replacement ladder all exist to put the endpoint back in service,
    and putting the WRONG weights back in service is worse than leaving them
    down. Read off the source so a later edit that moves it down shows up."""
    src = open(os.path.join(_HERE, "vastlib", "supervise", "job_lane.py"),
               encoding="utf-8").read()
    i = src.index("_serve_identity_tick(jc, inst, now)")
    for later in ("_serve_self_park_soft", "classify_job_box_stop",
                  "_serve_boot_sla_tick", "_job_resume_in_place("):
        assert src.index(later, i) > i, later


# --------------------------------------------------------------------------- #
# 5. the daemon: policy travel, alarms, and the watch that is KEPT
# --------------------------------------------------------------------------- #
def _rec(state="mismatch", **kw):
    base = {"state": state, "expected": IDENT, "observed": OTHER,
            "reason": None, "artifact": "mergeddemoa", "since": NOW - 300,
            "parked": True}
    base.update(kw)
    return base


def _alarms(w, target="47", now=NOW):
    return daemon._serve_identity_alarms(target, w, w.get("iid") or target, now)


def test_the_policy_rebuilds_into_the_ladder_namespace():
    """client dict -> watch record -> make_policy -> the ladder's argparse
    namespace, which is the same road every other policy field travels."""
    a = daemon.make_policy("serve", {"artifact": "mergeddemoa",
                                     "expect_ident": IDENT, "id": 41},
                           "41", budget_usd=5.0)
    assert a.serve_mode is True
    assert (a.artifact, a.expect_ident) == ("mergeddemoa", IDENT)
    jc, _hf = job_lane.job_supervise_init(a)
    assert (jc["model_artifact"], jc["expect_ident"]) == ("mergeddemoa", IDENT)


def test_an_old_policy_dict_rebuilds_with_no_pin():
    """A watch registered before P3 has neither key; `_Policy` answers None and
    the ladder reads that as "not checking"."""
    a = daemon.make_policy("serve", {"id": 41}, "41", budget_usd=5.0)
    assert a.artifact is None and a.expect_ident is None
    jc, _hf = job_lane.job_supervise_init(a)
    assert jc["expect_ident"] is None


def test_the_verdict_round_trips_through_the_watch_record():
    jc = {"serve_identity": _rec(), "serve_identity_condemned": True}
    w = {}
    fleet_state._serve_identity_persist(jc, w)
    assert w["serve_identity"]["observed"] == OTHER
    assert w["serve_identity_condemned"] is True
    fresh = {}
    fleet_state._serve_identity_restore(fresh, w)
    assert fresh["serve_identity_condemned"] is True
    assert fresh["serve_identity"] == w["serve_identity"]


def test_persist_POPS_a_verdict_whose_pin_was_dropped():
    """Re-registering without --artifact is an operator saying stop checking. A
    verdict that outlived its pin would alarm forever with nothing able to
    retract it — derived alarms have no clear path by construction."""
    w = {"serve_identity": _rec(), "serve_identity_condemned": True}
    fleet_state._serve_identity_persist({}, w)
    assert "serve_identity" not in w and "serve_identity_condemned" not in w


def test_the_verdict_survives_a_daemon_restart(tmp_path, monkeypatch):
    """state.json is the only thing that carries a mismatch across a restart,
    and the latch has to come with it — a restart that forgot it would hand
    the ladder a fresh licence to rescue the box it withdrew."""
    monkeypatch.setattr(daemon, "subprocess",
                        type("S", (), {"run": staticmethod(
                            lambda *a, **k: type("P", (), {
                                "returncode": 0, "stdout": "rev\n",
                                "stderr": ""})())})())
    d = str(tmp_path / "state")
    f1 = daemon.Fleet(d)
    f1.state["watches"]["47"] = {"target": "47", "iid": "47",
                                 "profile": "serve", "budget_usd": 5.0,
                                 "serve_identity": _rec(),
                                 "serve_identity_condemned": True}
    f1.save()
    reloaded = json.loads(open(os.path.join(d, "state.json")).read())
    assert reloaded["watches"]["47"]["serve_identity_condemned"] is True
    f2 = daemon.Fleet(d)
    w = f2.state["watches"]["47"]
    assert w["serve_identity"]["observed"] == OTHER
    jc = {}
    fleet_state._serve_identity_restore(jc, w)
    assert jc["serve_identity_condemned"] is True
    assert [k for k, _m in _alarms(w)] == ["watch:47:serve_identity_mismatch"]


@pytest.mark.parametrize("state,key", [
    ("mismatch", "watch:47:serve_identity_mismatch"),
    ("unarmed", "watch:47:serve_identity_unarmed"),
    ("gate_failed", "watch:47:serve_identity_failed"),
])
def test_each_alarming_state_gets_its_own_code(state, key):
    got = _alarms({"iid": "47",
                   "serve_identity": _rec(state, reason="identity_cannot_check")})
    assert [k for k, _m in got] == [key]


@pytest.mark.parametrize("state", ["off", "unreadable", "pending", "verified"])
def test_the_healthy_and_silent_states_never_alarm(state):
    assert _alarms({"iid": "47", "serve_identity": _rec(state)}) == []


def test_a_watch_with_no_verdict_at_all_never_alarms():
    assert _alarms({"iid": "47"}) == []
    assert _alarms({"iid": "47", "serve_identity": "not-a-dict"}) == []


def test_the_mismatch_alarm_says_what_to_run_next():
    """A refusal has to land where the operator can still act — including the
    2h reaper clock a parked box is now on."""
    _k, m = _alarms({"iid": "47", "serve_identity": _rec()})[0]
    assert "SERVE IDENTITY MISMATCH" in m
    assert OTHER in m and IDENT in m and "mergeddemoa" in m
    assert "PARKED" in m and "NOT destroyed" in m
    assert "launch_serve.sh --model-artifact" in m
    assert "fleet destroy 47" in m
    assert "2h" in m                                  # the reaper deadline
    assert "wrong label" in m                         # what it cost


def test_a_failed_park_changes_the_mismatch_remedy():
    _k, m = _alarms({"iid": "47", "serve_identity": _rec(parked=False)})[0]
    assert "PARK FAILED" in m and "herdd stop 47" in m


@pytest.mark.parametrize("reason", serve_ident.FAILED_REASONS)
def test_every_gate_failure_surfaces_its_own_reason_and_remedy(reason):
    _k, m = _alarms({"iid": "47",
                     "serve_identity": _rec("gate_failed", reason=reason)})[0]
    assert f"FAILED {reason}" in m
    assert serve_ident.FAILED_REMEDY[reason] in m
    assert "still billing" in m


def test_the_unarmed_alarm_names_both_ways_out():
    _k, m = _alarms({"iid": "47", "serve_identity": _rec("unarmed")})[0]
    assert "NO ident= field" in m
    assert "launch_serve.sh --model-artifact mergeddemoa" in m   # arm it
    assert "no --expect-ident" in m                          # or drop the pin


def test_the_mismatch_alarm_OUTRANKS_the_dormancy_silence():
    """S8 silences a dormant watch. A mismatched box is withdrawn RIGHT NOW and
    stays withdrawn, so this is a standing condition, not a leftover — the same
    reasoning that lets a budget park keep burning."""
    f = _fleet_stub()
    w = {"target": "47", "iid": "47", "profile": "serve", "budget_usd": 5.0,
         "state": "identity_mismatch", "dormant": True,
         "serve_identity": _rec()}
    keys = [k for k, _m in f._derive_watch_alarms("47", w, NOW)]
    assert keys == ["watch:47:serve_identity_mismatch"]


def test_a_budget_park_still_outranks_everything():
    """Ordering unchanged: a capped box is already parked and cannot poison an
    eval, so the money alarm stays first and this feature does not reorder it."""
    f = _fleet_stub()
    w = {"target": "47", "iid": "47", "profile": "serve", "budget_usd": 5.0,
         "state": "budget_parked", "spend_usd": 9.0, "serve_identity": _rec()}
    keys = [k for k, _m in f._derive_watch_alarms("47", w, NOW)]
    assert keys == ["watch:47:budget"]


def test_the_passive_states_ride_alongside_the_normal_alarms():
    """`unarmed`/`gate_failed` do not outrank anything — they are appended with
    the rest, so a box can report both a stalled rescue and an unarmed gate."""
    f = _fleet_stub()
    w = {"target": "47", "iid": "47", "profile": "serve", "budget_usd": 5.0,
         "state": "watched", "unrecoverable_since": NOW - 100,
         "serve_identity": _rec("unarmed")}
    keys = [k for k, _m in f._derive_watch_alarms("47", w, NOW)]
    assert set(keys) == {"watch:47:rescue_stalled",
                         "watch:47:serve_identity_unarmed"}


def test_a_legacy_serve_watch_produces_ZERO_new_alarms():
    """The regression guard. Every serve watch registered before P3 has no
    `serve_identity` key at all, and must derive exactly what it always did."""
    f = _fleet_stub()
    for w in ({"target": "47", "iid": "47", "profile": "serve",
               "budget_usd": 5.0, "state": "watched"},
              {"target": "47", "iid": "47", "profile": "serve",
               "budget_usd": None, "state": "watched"},
              {"target": "47", "iid": "47", "profile": "serve",
               "budget_usd": 5.0, "state": "watched",
               "unrecoverable_since": NOW - 100}):
        keys = [k for k, _m in f._derive_watch_alarms("47", w, NOW)]
        assert not [k for k in keys if "identity" in k], keys


def _fleet_stub():
    """A `Fleet` with no daemon around it: `_derive_watch_alarms` is pure over
    the watch record plus the health cache, which is what makes it callable
    from a status read (and from here) without ticking anything."""
    class _F(daemon.Fleet):
        def __init__(self):                            # no state dir, no git
            self._health = {}
            self.state = {"watches": {}, "ceilings": {}, "ceiling_by_box": {}}

        def _ceiling_spend(self, w):
            return float(w.get("spend_usd") or 0.0)

    return _F()


class _Hooks:
    """The `FleetHooks` protocol, scripted — same shape as
    `test_vastlib_fleet_daemon.py`'s, trimmed to what a serve watch touches.
    `jobs_tick` is the seam the real ladder sits behind, so returning its
    verdict here drives exactly the daemon path a mismatch takes."""

    def __init__(self):
        self.t = NOW
        self.boxes = {}
        self.verdict = None
        self.jc_writes = {}
        self.parked = []

    def now(self):
        return self.t

    def box(self, iid, **kw):
        self.boxes[str(iid)] = dict(id=int(iid), actual_status="running",
                                    intended_status="running", dph_total=0.5,
                                    label="serve:s1", **kw)

    def instances(self):
        return list(self.boxes.values())

    def instance(self, iid):
        return self.boxes.get(str(iid))

    def notifications(self):
        return {"notifications": []}, None

    def jobd_status_line(self, iid):
        return None

    def health(self, instances):
        return {}

    def park(self, iid):
        self.parked.append(str(iid))
        return True, None

    def resume(self, iid):
        return True, None

    def destroy(self, iid):
        self.boxes.pop(str(iid), None)
        return True, None

    def keep_label(self, iid, inst):
        return True, "keep"

    def drained(self, iid):
        return None

    def results_present(self, iid):
        return None

    def jobs_init(self, a):
        return ({"a": a, "iid": str(a.id), "spend_usd": 0.0,
                 "handoff_on": False,
                 "model_artifact": a.artifact, "expect_ident": a.expect_ident},
                {"phase": "IDLE"})

    def jobs_tick(self, jc, hf):
        jc.update(self.jc_writes)
        return self.verdict


def test_a_mismatch_verdict_drives_the_whole_daemon_path(tmp_path, monkeypatch):
    """The integration seam every unit above stops short of: the ladder's
    verdict comes back through `jobs_tick`, the daemon keeps the watch, mirrors
    the verdict onto the record, journals it once, and `fleet status` renders
    the alarm — all in one real tick."""
    monkeypatch.setenv("FLEETD_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setattr(daemon, "subprocess",
                        type("S", (), {"run": staticmethod(
                            lambda *a, **k: type("P", (), {
                                "returncode": 0, "stdout": "rev\n",
                                "stderr": ""})())})())
    h = _Hooks()
    h.box(47)
    f = daemon.Fleet(str(tmp_path / "state"), hooks=h)
    f.watch("47", "serve", budget_usd=5.0,
            policy={"id": 47, "budget": 5.0, "artifact": "mergeddemoa",
                    "expect_ident": IDENT},
            requester="tester")
    # the pin reached the rebuilt ladder
    f.tick()
    assert f.runtime["47"]["jc"]["expect_ident"] == IDENT

    h.verdict = "identity_mismatch"
    h.jc_writes = {"serve_identity": _rec(), "serve_identity_condemned": True}
    f.tick()

    w = f.state["watches"]["47"]                       # KEPT, not ended
    assert w["state"] == "identity_mismatch"
    assert w["serve_identity"]["observed"] == OTHER
    assert w["serve_identity_condemned"] is True
    keys = [r["key"] for r in f.alarm_records()]
    assert "watch:47:serve_identity_mismatch" in keys
    assert all(not r["sticky"] for r in f.alarm_records())   # DERIVED, not latched
    ev = [json.loads(ln)["event"]
          for ln in open(f.journal_path) if ln.strip()]
    assert ev.count("serve_identity_withdrawn") == 1

    f.tick()                                           # still burning, still one
    assert "watch:47:serve_identity_mismatch" in [r["key"]
                                                  for r in f.alarm_records()]
    ev = [json.loads(ln)["event"]
          for ln in open(f.journal_path) if ln.strip()]
    assert ev.count("serve_identity_withdrawn") == 1

    # ...and it RETRACTS when the pin is dropped, with no tick in between
    h.jc_writes = {}
    f.runtime["47"]["jc"].pop("serve_identity", None)
    f.runtime["47"]["jc"].pop("serve_identity_condemned", None)
    h.verdict = None
    f.tick()
    assert "serve_identity" not in f.state["watches"]["47"]
    assert not [k for k in [r["key"] for r in f.alarm_records()]
                if "identity" in k]


def test_the_daemon_keeps_the_watch_on_a_mismatch():
    """Ending it would pop the record the alarm derives from — an operator left
    with a stopped box, no alarm, and no way to find out why."""
    f = _fleet_stub()
    f.hooks = type("H", (), {"now": staticmethod(lambda: NOW)})()
    f.journal = lambda *a, **k: f.state.setdefault("_j", []).append((a, k))
    w = {"target": "47", "iid": "47", "profile": "serve", "budget_usd": 5.0,
         "serve_identity": _rec()}
    f.state["watches"]["47"] = w
    f._serve_watch_identity_mismatch("47", w)
    assert f.state["watches"]["47"] is w               # KEPT, not popped
    assert w["state"] == "identity_mismatch"
    assert w["identity_mismatch_since"] == NOW
    assert not w.get("dormant")                        # a defect report, not a
    ev = [a[0] for a, _k in f.state["_j"]]             # policy park
    assert ev == ["serve_identity_withdrawn"]
    f._serve_watch_identity_mismatch("47", w)          # idempotent: one journal
    assert len(f.state["_j"]) == 1

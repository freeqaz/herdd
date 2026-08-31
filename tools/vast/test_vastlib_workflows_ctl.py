"""`vastlib.workflows.ctl` — the ported controller, held to the traps that moved with it.

Why this file exists
--------------------
`test_workflow.py` is 4,011 lines and 33 patch sites, and every one of them
drives the FLAT `workflowctl.py`, which stays live through the add-only phase
(plan §8) and is deliberately left untouched. So none of it says anything about
the ported copy. This file covers the properties that a MOVE specifically
endangers — not the reconcile semantics that file already owns:

1. **The two `__file__` anchors.** The flat module resolved both the repo root
   and `rehearse.sh` with three `os.path.dirname` calls from `tools/vast/`. From
   `tools/vast/vastlib/workflows/ctl.py` that expression yields `tools/vast`, and
   NOTHING RAISES: `jobmeta.check_asset_staleness` would just compare a mutable
   B2 asset against sources that do not exist at that root, and the pre-spend
   staleness gate would pass vacuously. Only a comparison against the flat
   module's own resolution can catch it.
2. **The 27 cycle edges landing in `vastlib`, not in `cli`.** The whole point of
   this module's location. Asserted structurally (over the import graph) rather
   than by eye, because the failure mode — a name that resolves from `cli` —
   re-forms the cycle as a layer violation nobody notices until step 6.
3. **The module-attribute call form** (plan §8b). Every seam default is read
   through its module at CALL time, which is what keeps a `monkeypatch.setattr`
   steering it. A `from x import fn` anywhere in the chain makes 7 existing patch
   sites vacuous — green tests steering nothing.
4. **The two safety designs that look like inefficiency**: `build_box_observer`'s
   second-healthy-read before `gone` (a false `gone` is a DOUBLE SPEND) and
   `LiveCostObserver`'s keep-the-prior-snapshot on a read failure (zeroing it
   drops accrual). Neither is witnessed by any happy path.
5. **Restart durability = B2 OBJECT EXISTENCE, never memory.** `folded_spend`,
   `_teardown_attempts_seen` and `read_accepted_artifact` RE-READ on every call.
   A port that memoized any of them "for efficiency" passes every test that never
   restarts a controller — so these tests mutate the store BETWEEN calls.
6. **`_owned_instance_ids`' `str()` normalization** — the run-2ed9 crash: an
   adopted box records a str id, a fresh launch an int, and `sorted({str, int})`
   raises on every cost tick.
7. **The isinstance identity `load_workflow_module` depends on.** An authored
   spec file says `from workflow import Workflow`; if the controller resolved its
   class from `vastlib.workflows.spec` instead, every spec would fail to load
   with "WORKFLOW is a Workflow, not a Workflow".

What is deliberately NOT here
-----------------------------
* No re-testing of the 8-step reconcile order, the retry ladder, the credential
  gate or the canary store — `test_workflow.py` (52 `reconcile_tick` refs) and
  `test_workflow_preflight.py` own those and still pass unedited.
* No repoint of any existing test.
* No network, no box, no rclone. Every runner is `FakeB2`; every vast read is a
  stubbed module attribute. Two tests deliberately do NOT stub `request_soft`,
  to prove conftest's mutation guard is what answers.

Provenance: created 2026-08-16 alongside `vastlib/workflows/ctl.py`, plan §8
step 5.
"""

from __future__ import annotations

import ast
import json
import os
import sys
from pathlib import Path

import pytest

VAST_DIR = Path(__file__).resolve().parent
if str(VAST_DIR) not in sys.path:                      # entry-script sys.path shape
    sys.path.insert(0, str(VAST_DIR))

import workflowctl as flat                             # noqa: E402  twin, still live
from test_jobmeta import FakeB2                        # noqa: E402
from workflow import (                                 # noqa: E402
    ArtifactContract,
    InputRef,
    JobStage,
    ResourceProfile,
    RetryPolicy,
    Workflow,
)

from vastlib.workflows import ctl, meta as wm          # noqa: E402


@pytest.fixture
def b2(monkeypatch):
    """An in-memory rclone runner plus the bucket every helper resolves to."""
    monkeypatch.setenv("B2_BUCKET", "bkt")
    return FakeB2("bkt")


def _wf() -> Workflow:
    return Workflow(
        version=1, name="e2 paired", budget_usd=10.0, max_wall_s=14400,
        teardown="stop",
        profiles={"gen": ResourceProfile(image="reg/x:t211", image_digest="sha256:aaa",
                                         budget_usd=3.0, max_wall_s=7200)},
        stages=(
            JobStage(name="generate", bundle="b/gen", profile="gen",
                     outputs={"gens": ArtifactContract(
                         kind="generation",
                         manifest_path="results/artifact-manifest.json")},
                     retry=RetryPolicy(max_attempts=2, retry_on=("infrastructure",))),
            JobStage(name="score", bundle="b/score", profile="gen",
                     after=("generate",),
                     inputs={"input-generate": InputRef(
                         stage="generate", artifact="gens", dest="/workspace/in")}),
        ))


WF_ID = "20260713T120000-e2-paired-ab12"


# --------------------------------------------------------------------------- #
# 1 — the two __file__ anchors (the one part of the port that is not verbatim)
# --------------------------------------------------------------------------- #
def test_path_anchors_match_the_flat_module():
    """`_TOOLS_VAST_DIR` and `_REPO_ROOT` == what the flat module computes for
    ITSELF at runtime, from its own (shallower) location."""
    flat_py = os.path.abspath(flat.__file__)
    assert ctl._TOOLS_VAST_DIR == os.path.dirname(flat_py)
    assert ctl._REPO_ROOT == os.path.dirname(os.path.dirname(os.path.dirname(flat_py)))
    # (`ctl._repo_root() == flat._repo_root()` was dropped at step 7 — the flat
    # name re-exports THIS function, so it compared a value with itself. The
    # `flat.__file__` anchors above stay meaningful: the shim keeps the file at
    # tools/vast/workflowctl.py, which is exactly the anchor being asserted.)
    assert ctl._repo_root() == ctl._REPO_ROOT
    assert os.path.isdir(os.path.join(ctl._REPO_ROOT, "tools", "vast"))


def test_naive_file_arithmetic_here_would_be_wrong():
    """Copying the three-dirname expression verbatim lands two levels short.

    This is the whole hazard, and it is silent: `_default_asset_checker` would
    hand `jobmeta.check_asset_staleness` a root where no source file exists, and
    the pre-spend staleness gate would report nothing rather than fail.
    """
    naive = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(ctl.__file__))))
    assert naive.endswith(os.path.join("tools", "vast"))
    assert naive != ctl._REPO_ROOT
    assert naive == ctl._TOOLS_VAST_DIR


def test_rehearse_script_default_stays_in_tools_vast(monkeypatch, tmp_path):
    """`rehearse.sh` is NOT part of the package and does not move with it."""
    assert os.path.isfile(os.path.join(ctl._TOOLS_VAST_DIR, "rehearse.sh"))
    seen = {}

    def _fake_builder(script):
        seen["script"] = script
        return lambda *a, **k: (0, None)

    monkeypatch.setattr(ctl, "_build_default_stage_rehearser", _fake_builder)
    monkeypatch.setattr(ctl, "load_workflow_module", lambda p: _wf())
    ctl.rehearse_workflow("ignored.py", wf_id=WF_ID, workdir=str(tmp_path))
    assert seen["script"] == os.path.join(ctl._TOOLS_VAST_DIR, "rehearse.sh")


# --------------------------------------------------------------------------- #
# 2 — the cycle, broken by DIRECTION
# --------------------------------------------------------------------------- #
def _imported_modules(path: str) -> set[str]:
    tree = ast.parse(Path(path).read_text())
    mods: set[str] = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            mods.update(a.name for a in n.names)
        elif isinstance(n, ast.ImportFrom) and n.module:
            mods.add(n.module)
    return mods


def test_ctl_imports_no_cli_and_no_herdd():
    """The 27 edges resolve from `vastlib.*` (at or below this ring), Zone S, or
    the two absorbed flat siblings — never from `cli`, which would re-form the
    cycle as a layer violation."""
    mods = _imported_modules(ctl.__file__)
    assert not any(m == "herdd" or m.startswith("vastlib.cli") for m in mods), mods
    vastlib_mods = {m for m in mods if m.startswith("vastlib.")}
    assert vastlib_mods == {
        "vastlib.boxes", "vastlib.core", "vastlib.fleet", "vastlib.jobs",
        "vastlib.launch", "vastlib.market", "vastlib.supervise", "vastlib.workflows",
        # the DSL edge: bare `workflow` until step 7, repointed at its real home
        # once the flat file became a re-export shim (one class object either way)
        "vastlib.workflows.spec",
    }, vastlib_mods


def test_no_module_level_sys_path_mutation():
    """The flat file did `sys.path.insert(0, _HERE)`; plan §3 forbids it inside
    Zone P. Checked over the AST — the docstring names the call deliberately."""
    tree = ast.parse(Path(ctl.__file__).read_text())
    assert "sys" not in {n.names[0].name for n in ast.walk(tree)
                         if isinstance(n, ast.Import)}


def test_every_repointed_name_resolves_to_its_vastlib_home():
    """One assertion per cycle edge: the name exists where the port says it does,
    so a sibling that renames one fails HERE rather than at the first live tick."""
    for mod, attr in [
        (ctl.lifecycle, "stop_box"), (ctl.lifecycle, "destroy_box"),
        (ctl.lifecycle, "launch_instance"), (ctl.lifecycle, "find_matching_instance"),
        (ctl.health, "_get_instance_soft"), (ctl.health, "build_throughput_observer"),
        (ctl.ssh, "pub_key_text"), (ctl.ssh, "ssh_authorized_keys_snippet"),
        (ctl.api, "request_soft"), (ctl.models, "_instance_image"),
        (ctl.models, "_instance_env"), (ctl.models, "_num_dph"),
        (ctl.config, "DISK_DEFAULT_WORKFLOW_GB"), (ctl.offers, "pick_cheapest_offer"),
        (ctl.launch, "image_login_arg"), (ctl.launch, "hf_token_text"),
        (ctl.launch, "hf_login_snippet"), (ctl.bundle, "compose_jobs_launch_env"),
        (ctl.bundle, "_stage_jobd_bootstrap"), (ctl.jobs_view, "_job_progress"),
        (ctl.risk, "_ckpt_watchdog_alarm"), (ctl.run_lane, "_accrue_cost"),
        (ctl.fleet_client, "fleet_watch_best_effort"),
        (ctl.bidpolicy, "LIVE_STATES"), (ctl.bidpolicy, "BID_TARGET_MULT"),
        (ctl.imageref, "IMAGE_DIGEST_ENV"), (ctl.imageref, "image_tag_digest"),
        (ctl.imageref, "image_ref_digest"),
    ]:
        assert hasattr(mod, attr), f"{mod.__name__}.{attr}"


def test_exit_code_tables_agree():
    """TWO tables, ONE contract. `ctl.EXIT_*` are what `herdd`'s handlers
    `sys.exit()`; `wm._FAILURE_CLASS_EXIT_CODE` is what a terminal event's
    `failure_class` maps to. The port keeps both (they are separately frozen)
    and asserts the correspondence HERE instead of deriving one from the other."""
    assert (ctl.EXIT_OK, ctl.EXIT_INVALID, ctl.EXIT_FAILED, ctl.EXIT_CANCELLED,
            ctl.EXIT_ARTIFACT, ctl.EXIT_CREDENTIAL, ctl.EXIT_TIMEOUT) == \
        (0, 1, 2, 3, 4, 5, 124)
    assert set(wm._FAILURE_CLASS_EXIT_CODE.values()) <= {
        ctl.EXIT_INVALID, ctl.EXIT_FAILED, ctl.EXIT_ARTIFACT,
        ctl.EXIT_CREDENTIAL, ctl.EXIT_TIMEOUT}
    assert wm._FAILURE_CLASS_EXIT_CODE["CONFIG_INVALID"] == ctl.EXIT_INVALID
    assert wm._FAILURE_CLASS_EXIT_CODE["ARTIFACT_INVALID"] == ctl.EXIT_ARTIFACT
    assert wm._FAILURE_CLASS_EXIT_CODE["CREDENTIAL_EXPIRES"] == ctl.EXIT_CREDENTIAL
    assert wm._FAILURE_CLASS_EXIT_CODE["WALL_EXHAUSTED"] == ctl.EXIT_TIMEOUT
    assert wm._FAILURE_CLASS_EXIT_CODE["RETRY_EXHAUSTED"] == ctl.EXIT_FAILED
    # and the ported constants are the flat module's, value for value
    for name in ("EXIT_OK", "EXIT_INVALID", "EXIT_FAILED", "EXIT_CANCELLED",
                 "EXIT_ARTIFACT", "EXIT_CREDENTIAL", "EXIT_TIMEOUT"):
        assert getattr(ctl, name) == getattr(flat, name)


def test_terminal_exit_code_routes_through_the_failure_class_table():
    assert ctl._terminal_exit_code("succeeded") == ctl.EXIT_OK
    assert ctl._terminal_exit_code("cancelled") == ctl.EXIT_CANCELLED
    assert ctl._terminal_exit_code("failed") == ctl.EXIT_FAILED     # no class: prior
    assert ctl._terminal_exit_code("failed", "WALL_EXHAUSTED") == ctl.EXIT_TIMEOUT
    assert ctl._terminal_exit_code("failed", "CREDENTIAL_EXPIRES") == ctl.EXIT_CREDENTIAL
    assert ctl._terminal_exit_code("running") == ctl.EXIT_OK        # nonterminal read


def test_rehearsal_disclaimer_names_what_a_rehearsal_does_not_prove():
    """Was `== flat.REHEARSAL_DISCLAIMER`, a self-comparison since the step-7
    shim. Pinned to the operator-facing claim instead: this string is what
    stops a green rehearsal being read as a green run."""
    assert "REHEARSAL" in ctl.REHEARSAL_DISCLAIMER
    assert "not" in ctl.REHEARSAL_DISCLAIMER.lower()


# --------------------------------------------------------------------------- #
# 3 — the seven import-time env constants stay MODULE ATTRIBUTES
# --------------------------------------------------------------------------- #
def test_env_knobs_are_module_constants_read_at_import():
    """Routing these through `core.config` would change the test idiom
    (`test_workflow.py` monkeypatches the module attribute, not the env), so the
    port keeps them as they were and files the unification as later work."""
    assert isinstance(ctl.BOOT_DEADLINE_S, int) and ctl.BOOT_DEADLINE_S == 1500
    assert isinstance(ctl.JOB_HEARTBEAT_STALE_S, int)
    assert isinstance(ctl.BOOT_MIN_MBPS, float)
    assert isinstance(ctl.IMAGE_GATE_ENFORCE, int)
    # (the `== getattr(flat, name)` sweep was dropped at step 7: the flat module
    # re-exports these very objects, so it compared each knob with itself.)
    assert ctl.POLL_INTERVAL_S > 0 and ctl.HEARTBEAT_STALE_MULT >= 2
    assert ctl.LAUNCH_INET_DOWN_FLOOR_MBPS is None      # defer to herdd's knob
    for name in ("BOOT_DEADLINE_S", "JOB_HEARTBEAT_STALE_S", "BOOT_MIN_MBPS",
                 "IMAGE_GATE_ENFORCE", "TAKEOVER_WAIT_GRACE_S",
                 "TEARDOWN_MAX_ATTEMPTS", "CANARY_RECEIPT_TTL_S"):
        assert isinstance(getattr(ctl, name), (int, float)), name


def test_image_gate_enforce_is_patched_on_the_module_not_the_env(monkeypatch):
    assert ctl._image_gate_refuses(ctl.imageref.IMG_STALE) is True
    assert ctl._image_gate_refuses(ctl.imageref.IMG_UNRESOLVED) is True
    assert ctl._image_gate_refuses(ctl.imageref.IMG_FRESH) is False
    monkeypatch.setattr(ctl, "IMAGE_GATE_ENFORCE", 0)   # the escape hatch
    assert ctl._image_gate_refuses(ctl.imageref.IMG_STALE) is False


# --------------------------------------------------------------------------- #
# 4a — build_box_observer: a false `gone` is a double spend
# --------------------------------------------------------------------------- #
def _reader(*answers):
    seq = list(answers)
    calls = []

    def reader():
        calls.append(1)
        return seq.pop(0) if seq else (True, [])

    reader.calls = calls          # type: ignore[attr-defined]
    return reader


def test_gone_requires_a_second_independent_healthy_read():
    r = _reader((True, []), (True, []))
    assert ctl.build_box_observer(instances_reader=r)("77") == "gone"
    assert len(r.calls) == 2                       # confirmed, not guessed


def test_a_reappearance_on_the_confirm_read_is_not_gone():
    r = _reader((True, []), (True, [{"id": 77, "actual_status": "running"}]))
    assert ctl.build_box_observer(instances_reader=r)("77") == "live"


def test_a_transient_failure_is_unknown_on_either_read():
    assert ctl.build_box_observer(instances_reader=_reader((False, [])))("77") \
        == "unknown"
    r = _reader((True, []), (False, []))
    assert ctl.build_box_observer(instances_reader=r)("77") == "unknown"


def test_observer_maps_states_and_never_acts_on_an_unknown_one():
    def one(status):
        return _reader((True, [{"id": "77", "actual_status": status}]))
    assert ctl.build_box_observer(instances_reader=one("running"))("77") == "live"
    assert ctl.build_box_observer(instances_reader=one("exited"))("77") == "stopped"
    assert ctl.build_box_observer(instances_reader=one("stopped"))("77") == "stopped"
    assert ctl.build_box_observer(instances_reader=one("scheduling"))("77") == "unknown"
    assert ctl.STOPPED_STATES == {"exited", "stopped"}


def test_the_default_reader_goes_through_the_api_module_attribute(monkeypatch):
    """Not `lifecycle._instances_soft` — that returns [] on error and would read
    every owned box as `gone`. The default closes over `api.request_soft` so a
    failure is VISIBLE, and it must resolve the attribute at call time."""
    monkeypatch.setattr(ctl.api, "request_soft",
                        lambda *a, **k: (False, None, "500 boom"))
    assert ctl.build_box_observer()("77") == "unknown"
    monkeypatch.setattr(ctl.api, "request_soft",
                        lambda *a, **k: (True, {"instances": [
                            {"id": 77, "actual_status": "running"}]}, None))
    assert ctl.build_box_observer()("77") == "live"


# --------------------------------------------------------------------------- #
# 4b — LiveCostObserver: a transient read must not drop accrual
# --------------------------------------------------------------------------- #
def test_first_observe_has_dt_zero_so_there_is_no_phantom_spend():
    clock = iter([100.0, 100.0, 100.0])
    obs = ctl.LiveCostObserver(
        instances_reader=_reader((True, [{"id": "77", "actual_status": "running",
                                          "dph_total": 0.5}])),
        clock=lambda: next(clock))
    st = obs.get("77")
    assert st["dt"] == 0.0 and st["spend_usd"] == 0.0
    assert st["present"] is True and st["dph_total"] == 0.5


def test_a_transient_read_failure_keeps_the_prior_snapshot():
    """Zeroing `present` here would look tidier and would DROP accrual for the
    tick — the box is still running and still billing."""
    now = [100.0]
    live = (True, [{"id": "77", "actual_status": "running", "dph_total": 0.5}])
    r = _reader(live, (False, []))
    obs = ctl.LiveCostObserver(instances_reader=r, clock=lambda: now[0], ttl_s=0)
    assert obs.get("77")["present"] is True
    now[0] = 200.0
    st = obs.get("77")
    assert st["present"] is True                   # prior snapshot retained
    assert st["dt"] == 100.0 and st["dph_total"] == 0.5


def test_a_read_failure_with_no_prior_snapshot_is_absent_not_fabricated():
    obs = ctl.LiveCostObserver(instances_reader=_reader((False, [])),
                               clock=lambda: 100.0)
    assert obs.get("77")["present"] is False


def test_the_snapshot_is_memoized_for_one_ttl_window():
    now = [100.0]
    r = _reader((True, [{"id": "77", "actual_status": "running", "dph_total": 0.5}]),
                (True, [{"id": "77", "actual_status": "running", "dph_total": 0.5}]))
    obs = ctl.LiveCostObserver(instances_reader=r, clock=lambda: now[0], ttl_s=10)
    obs.get("77")
    obs.get("77")
    assert len(r.calls) == 1                       # N boxes cost ~1 GET/tick
    now[0] = 200.0
    obs.get("77")
    assert len(r.calls) == 2


def test_the_observer_is_the_dict_like_accrue_and_persist_cost_expects():
    obs = ctl.build_cost_observer(instances_reader=_reader((True, [])),
                                  clock=lambda: 1.0)
    assert isinstance(obs, ctl.LiveCostObserver)
    st = obs.get("77")
    st["spend_usd"] = 1.25
    obs["77"] = st
    assert "77" in obs and obs.get("77")["spend_usd"] == 1.25


# --------------------------------------------------------------------------- #
# 5 — restart durability: idempotency is B2 OBJECT EXISTENCE, never memory
# --------------------------------------------------------------------------- #
def test_folded_spend_re_reads_the_event_log_every_call(b2):
    assert ctl.folded_spend(WF_ID, runner=b2) == 0.0
    ctl.record_box_cost(WF_ID, 1.5, actor="me", runner=b2)
    assert ctl.folded_spend(WF_ID, runner=b2) == 1.5
    ctl.record_box_cost(WF_ID, 4.25, actor="me", runner=b2)
    assert ctl.folded_spend(WF_ID, runner=b2) == 4.25
    ctl.record_box_cost(WF_ID, 2.0, actor="me", runner=b2)
    assert ctl.folded_spend(WF_ID, runner=b2) == 4.25    # fold takes MAX, never lowers


def test_teardown_attempts_seen_re_reads_and_survives_a_fresh_process(b2):
    assert ctl._teardown_attempts_seen(WF_ID, runner=b2) == 0
    for _ in range(3):
        ctl.emit(WF_ID, "teardown_attempt", "me", runner=b2)
    assert ctl._teardown_attempts_seen(WF_ID, runner=b2) == 3
    assert ctl.TEARDOWN_MAX_ATTEMPTS == 5


def test_write_spec_is_write_once_and_noop_on_identical_bytes(b2):
    wf = _wf()
    assert ctl.write_spec(wf, WF_ID, runner=b2)["status"] == "written"
    assert ctl.write_spec(wf, WF_ID, runner=b2)["status"] == "noop"
    assert ctl.read_spec(WF_ID, runner=b2).name == wf.name
    other = Workflow(version=1, name="different", budget_usd=1.0, max_wall_s=1,
                     teardown="stop",
                     profiles={"gen": ResourceProfile(image="reg/x:t",
                                                      image_digest="sha256:a")},
                     stages=(JobStage(name="generate", bundle="b", profile="gen"),))
    with pytest.raises(ctl.WorkflowCtlError, match="different bytes"):
        ctl.write_spec(other, WF_ID, runner=b2)
    # and the stored bytes are exactly the canonical form
    assert b2.store[f"workflows/{WF_ID}/spec.json"].strip() == wm.canonical_spec_json(wf)


def test_read_accepted_artifact_is_a_probe_not_a_validator(b2):
    assert ctl.read_accepted_artifact(WF_ID, "generate", "gens", runner=b2) is None
    b2.store[ctl._artifact_key(WF_ID, "generate", "gens")] = "{not json"
    assert ctl.read_accepted_artifact(WF_ID, "generate", "gens", runner=b2) is None
    b2.store[ctl._artifact_key(WF_ID, "generate", "gens")] = json.dumps({"job_id": "j0"})
    assert ctl.read_accepted_artifact(WF_ID, "generate", "gens",
                                      runner=b2)["job_id"] == "j0"


def test_the_four_b2_key_templates_are_frozen():
    assert ctl._artifact_key(WF_ID, "generate", "gens") == \
        f"workflows/{WF_ID}/artifacts/generate/gens.json"
    assert ctl._verdict_key(WF_ID) == f"workflows/{WF_ID}/verdict.json"
    assert ctl._provenance_key(WF_ID) == f"workflows/{WF_ID}/provenance.json"
    assert ctl._report_key(WF_ID) == f"workflows/{WF_ID}/report.md"
    assert ctl._canary_receipt_key_ref("k") == "workflow-canary/receipts/k.json"


def test_a_canary_receipt_key_is_content_addressed_and_deterministic():
    """The `== flat.canary_receipt_key(**kw)` half went tautological at step 7
    (one function, two names). What still bites is content-addressing: a
    receipt key that ignored an input would validate the WRONG image."""
    kw = dict(image_digest="sha256:aaa", jobd_sha="deadbeef",
              model_manifest_sha=None, adapter_manifest_sha="", recipe_sha="r")
    assert ctl.canary_receipt_key(**kw) == ctl.canary_receipt_key(**kw)
    for field in ("image_digest", "jobd_sha", "recipe_sha"):
        other = dict(kw, **{field: "different"})
        assert ctl.canary_receipt_key(**other) != ctl.canary_receipt_key(**kw), field


def test_emit_is_best_effort_and_never_raises(b2):
    def dead_runner(args, input=None):
        return 1, "", "network is down"

    ev = ctl.emit(WF_ID, "controller_heartbeat", "me", runner=dead_runner)
    assert ev["_emitted"] is False and ev["_error"] == "network is down"
    assert ev["event"] == "controller_heartbeat"


def test_read_events_skips_one_unreadable_object_instead_of_failing(b2):
    ctl.emit(WF_ID, "submitted", "me", runner=b2)
    ctl.emit(WF_ID, "controller_started", "me", runner=b2)
    keys = sorted(k for k in b2.store if "/events/" in k)
    broken = keys[0]

    def flaky(args, input=None):
        if args[0] == "cat" and args[1].endswith(broken):
            return 1, "", "transient"
        return b2(args, input=input)

    assert len(ctl.read_events(WF_ID, runner=flaky)) == 1     # the other one survives
    assert len(ctl.read_events(WF_ID, runner=b2)) == 2        # immutable key, re-read


# --------------------------------------------------------------------------- #
# 6 — the run-2ed9 crash: one fold can hold BOTH a str and an int instance id
# --------------------------------------------------------------------------- #
def test_owned_instance_ids_normalizes_str_and_int():
    v = {"stages": {"generate": {"instance_id": "46216906"},   # adopted -> str
                    "score": {"instance_id": 46249864}}}        # launched -> int
    assert ctl._owned_instance_ids(v) == ["46216906", "46249864"]


def test_the_same_box_recorded_both_ways_counts_once():
    v = {"stages": {"a": {"instance_id": 4711}, "b": {"instance_id": "4711"}}}
    assert ctl._owned_instance_ids(v) == ["4711"]      # never two cost keys


def test_owned_boxes_remaining_excludes_a_released_box_whoever_released_it(b2):
    v = {"stages": {"a": {"instance_id": 4711}, "b": {"instance_id": "4712"}}}
    assert ctl.owned_boxes_remaining(v, WF_ID, runner=b2) == ["4711", "4712"]
    ctl.emit(WF_ID, "box_released", "jobd-self-park", runner=b2, instance_id=4711)
    assert ctl.owned_boxes_remaining(v, WF_ID, runner=b2) == ["4712"]


# --------------------------------------------------------------------------- #
# 7 — isinstance identity: an authored spec resolves `workflow` by BARE NAME
# --------------------------------------------------------------------------- #
_SPEC_SRC = '''
from workflow import ArtifactContract, JobStage, ResourceProfile, Workflow

WORKFLOW = Workflow(
    version=1, name="authored", budget_usd=1.0, max_wall_s=60, teardown="stop",
    profiles={"gen": ResourceProfile(image="reg/x:t", image_digest="sha256:a")},
    stages=(JobStage(name="generate", bundle="b", profile="gen",
                     outputs={"gens": ArtifactContract(
                         kind="generation",
                         manifest_path="results/artifact-manifest.json")}),),
)
'''


def test_load_workflow_module_accepts_a_bare_name_authored_spec(tmp_path):
    """THE identity requirement. The spec file below is byte-for-byte the shape
    every authored workflow on disk uses; if `ctl` resolved `Workflow` from
    `vastlib.workflows.spec` while the spec resolved the flat module, this would
    fail with 'WORKFLOW is a Workflow, not a Workflow'."""
    p = tmp_path / "workflow_spec.py"
    p.write_text(_SPEC_SRC)
    wf = ctl.load_workflow_module(str(p))
    assert wf.name == "authored"
    assert isinstance(wf, Workflow)
    assert type(wf) is type(flat.load_workflow_module(str(p)))   # ONE class object


def test_load_workflow_module_wraps_every_failure_in_one_type(tmp_path):
    missing = tmp_path / "no_workflow.py"
    missing.write_text("X = 1\n")
    with pytest.raises(ctl.WorkflowCtlError, match="module-level WORKFLOW"):
        ctl.load_workflow_module(str(missing))
    wrong = tmp_path / "wrong_type.py"
    wrong.write_text("WORKFLOW = 42\n")
    with pytest.raises(ctl.WorkflowCtlError, match="not a Workflow"):
        ctl.load_workflow_module(str(wrong))
    boom = tmp_path / "raises.py"
    boom.write_text("raise RuntimeError('boom')\n")
    with pytest.raises(ctl.WorkflowCtlError, match="error executing"):
        ctl.load_workflow_module(str(boom))
    unpinned = tmp_path / "unpinned.py"
    unpinned.write_text(_SPEC_SRC.replace('image_digest="sha256:a"', "").replace(
        'image="reg/x:t", ', 'image="reg/x:t"'))
    with pytest.raises(ctl.WorkflowCtlError, match="invalid workflow spec"):
        ctl.load_workflow_module(str(unpinned))


# --------------------------------------------------------------------------- #
# 3b — the module-attribute call form (plan §8b): patches must still steer
# --------------------------------------------------------------------------- #
def test_build_box_teardown_defaults_route_through_lifecycle(monkeypatch):
    calls = []
    monkeypatch.setattr(ctl.lifecycle, "stop_box",
                        lambda iid: (calls.append(("stop", iid)), (True, None))[1])
    monkeypatch.setattr(ctl.lifecycle, "destroy_box",
                        lambda iid: (calls.append(("destroy", iid)), (True, None))[1])
    td = ctl.build_box_teardown()
    assert td("77", "stop") is True
    assert td("77", "destroy") is True
    assert calls == [("stop", "77"), ("destroy", "77")]


def test_build_box_teardown_swallows_an_injected_raiser():
    def boom(iid):
        raise RuntimeError("api down")

    assert ctl.build_box_teardown(stopper=boom)("77", "stop") is False


def test_default_box_starter_reaches_api_request_soft(monkeypatch):
    seen = {}

    def fake(method, path, body=None, **kw):
        seen.update(method=method, path=path, body=body)
        return True, {"success": True}, None

    monkeypatch.setattr(ctl.api, "request_soft", fake)
    assert ctl._default_box_starter(77) == (True, None)
    assert seen == {"method": "PUT", "path": "v0/instances/77/",
                    "body": {"state": "running"}}


def test_default_box_starter_unwraps_a_success_false_body(monkeypatch):
    monkeypatch.setattr(ctl.api, "request_soft",
                        lambda *a, **k: (True, {"success": False, "msg": "outbid"},
                                         None))
    assert ctl._default_box_starter(77) == (False, "outbid")


def test_a_mutating_call_left_unstubbed_is_refused_by_conftest():
    """The other half of the seam contract: the guard fixture is what answers a
    PUT nobody stubbed, which is only true while the call goes through the
    `api` MODULE ATTRIBUTE."""
    ok, err = ctl._default_box_starter(77)
    assert ok is False and "conftest" in str(err)


def test_build_live_controller_deps_is_pure_and_keyed_exactly(monkeypatch):
    """Zero network at build time — which is what lets the CLI build deps before
    `claim_controller` decides whether the run happens at all."""
    monkeypatch.setattr(ctl.api, "request_soft",
                        lambda *a, **k: pytest.fail("deps construction did I/O"))
    monkeypatch.setattr(ctl.health, "build_throughput_observer", lambda **k: "obs")
    deps = ctl.build_live_controller_deps(_wf(), WF_ID, actor="me")
    assert set(deps) == set(ctl._LIVE_DEP_KEYS)
    assert deps["cred_provider"] is None
    assert deps["box_starter"] is ctl._default_box_starter


def test_resolve_controller_deps_filters_to_the_live_keys():
    """A key added to one side and not the other is silently dropped, so the
    filter and the key tuple are asserted together."""
    assert ctl._resolve_controller_deps(None, _wf(), WF_ID) == {}
    got = ctl._resolve_controller_deps(
        {"box_resolver": "r", "not_a_dep": "x"}, _wf(), WF_ID)
    assert got == {"box_resolver": "r"}
    seen = {}

    def factory(wf, wf_id):
        seen["wf_id"] = wf_id              # invoked AFTER wf_id is resolved
        return {"box_teardown": "t"}

    assert ctl._resolve_controller_deps(factory, _wf(), WF_ID) == {"box_teardown": "t"}
    assert seen["wf_id"] == WF_ID


# --------------------------------------------------------------------------- #
# The reconcile surface — one smoke per branch the port could have broken
# --------------------------------------------------------------------------- #
def test_default_box_resolver_never_spends(b2):
    assert ctl._default_box_resolver(_wf().stages[0], _wf(), 0) is None
    act = ctl.reconcile_tick(_wf(), WF_ID, runner=b2, actor="me",
                             now=wm.now_ts())
    assert act["action"] == "need_box" and act["stage"] == "generate"


def test_a_terminal_workflow_with_no_boxes_is_a_noop(b2):
    ctl.emit(WF_ID, "workflow_succeeded", "me", runner=b2)
    act = ctl.reconcile_tick(_wf(), WF_ID, runner=b2, actor="me", now=wm.now_ts())
    assert act == {"action": "noop_terminal", "status": "succeeded"}


def test_a_terminal_workflow_that_still_owns_a_box_asks_for_teardown(b2):
    ctl.emit(WF_ID, "box_acquired", "me", runner=b2, stage="generate", attempt=0,
             instance_id=4711)
    ctl.emit(WF_ID, "workflow_failed", "me", runner=b2, failure_class="RETRY_EXHAUSTED")
    act = ctl.reconcile_tick(_wf(), WF_ID, runner=b2, actor="me", now=wm.now_ts())
    assert act == {"action": "need_box_teardown", "boxes": ["4711"]}


def test_teardown_reconciles_then_stops_calling_a_released_box(b2):
    ctl.emit(WF_ID, "box_acquired", "me", runner=b2, stage="generate", attempt=0,
             instance_id=4711)
    ctl.emit(WF_ID, "workflow_failed", "me", runner=b2, failure_class="RETRY_EXHAUSTED")
    calls = []
    act = ctl.reconcile_tick(_wf(), WF_ID, runner=b2, actor="me", now=wm.now_ts(),
                             box_teardown=lambda iid, mode: (calls.append((iid, mode)),
                                                             True)[1])
    assert act["action"] == "teardown_reconciled" and act["released"] == ["4711"]
    assert calls == [("4711", "stop")]                  # Workflow.teardown
    act2 = ctl.reconcile_tick(_wf(), WF_ID, runner=b2, actor="me", now=wm.now_ts(),
                              box_teardown=lambda iid, mode: pytest.fail("re-torn"))
    assert act2["action"] == "noop_terminal"


def test_budget_exhausted_treats_a_falsy_cap_as_no_cap():
    """An unbudgeted spec deserializes `budget_usd` to 0.0; an unguarded
    `spent >= cap` would fire BUDGET_EXHAUSTED on tick one of every workflow."""
    unbudgeted = Workflow(version=1, name="x", budget_usd=0.0, max_wall_s=0,
                          teardown="stop",
                          profiles={"gen": ResourceProfile(image="r:t",
                                                           image_digest="sha256:a")},
                          stages=(JobStage(name="generate", bundle="b",
                                           profile="gen"),))
    assert ctl.budget_exhausted(unbudgeted, 0.0) is False
    assert ctl.budget_exhausted(unbudgeted, 999.0) is False
    assert ctl.budget_exhausted(_wf(), 9.99) is False
    assert ctl.budget_exhausted(_wf(), 10.0) is True
    assert ctl.budget_exhausted(_wf(), 3.0, profile=_wf().profiles["gen"]) is True


def test_accrue_and_persist_cost_is_a_strict_noop_without_an_observer(b2):
    v = {"stages": {"generate": {"instance_id": 4711}}}
    assert ctl.accrue_and_persist_cost(_wf(), WF_ID, v, actor="me", runner=b2) is None
    assert ctl.folded_spend(WF_ID, runner=b2) == 0.0


def test_accrue_adds_the_tick_DELTA_onto_the_durable_prior(b2, monkeypatch):
    """Never the observer's in-memory absolute — a restart reseeds it to 0 and a
    retarget drops the replaced box, so an absolute sum would stall the cap."""
    ctl.record_box_cost(WF_ID, 4.0, actor="me", runner=b2)      # durable prior

    def fake_accrue(st):
        st["spend_usd"] = (st.get("spend_usd") or 0.0) + 0.25
        return st

    monkeypatch.setattr(ctl.run_lane, "_accrue_cost", fake_accrue)
    obs = ctl.build_cost_observer(instances_reader=_reader((True, [])),
                                  clock=lambda: 1.0)
    v = {"stages": {"generate": {"instance_id": 4711}}}
    ev = ctl.accrue_and_persist_cost(_wf(), WF_ID, v, actor="me", runner=b2,
                                     cost_observer=obs)
    assert ev["cost_usd"] == 4.25
    assert ctl.folded_spend(WF_ID, runner=b2) == 4.25


def test_reconcile_active_box_returns_none_when_the_job_is_not_running(monkeypatch,
                                                                       b2):
    monkeypatch.setattr(ctl.jobmeta, "read_job", lambda *a, **k: {"status": "done"})
    out = ctl.reconcile_active_box(_wf(), WF_ID, _wf().stages[0],
                                   {"instance_id": 4711, "job_id": "j0"},
                                   actor="me", runner=b2, now=wm.now_ts())
    assert out is None


def test_the_ckpt_alarm_is_advisory_and_never_changes_the_action(monkeypatch, b2):
    """The one edge that moved to `jobs.risk`. It attaches `ckpt_alarm` and must
    not touch `action` — pinned in the flat suite at test_workflow.py:1623."""
    monkeypatch.setattr(ctl.jobmeta, "read_job",
                        lambda *a, **k: {"status": "started",
                                         "last_heartbeat_ts": wm.now_ts()})
    monkeypatch.setattr(ctl.risk, "_ckpt_watchdog_alarm",
                        lambda jv, now, **k: "job j0 — NO checkpoint")
    out = ctl.reconcile_active_box(
        _wf(), WF_ID, _wf().stages[0], {"instance_id": 4711, "job_id": "j0"},
        actor="me", runner=b2, now=wm.now_ts(),
        box_observer=lambda iid: "unknown")
    assert out["action"] == "noop_running"
    assert out["ckpt_alarm"] == "job j0 — NO checkpoint"


def test_seconds_between_never_fires_on_a_missing_or_bad_timestamp():
    assert ctl._seconds_between(None, wm.now_ts()) is None
    assert ctl._seconds_between("", wm.now_ts()) is None
    assert ctl._seconds_between("not-a-ts", wm.now_ts()) is None
    assert ctl._seconds_between("20260713T120000000Z", "20260713T120130000Z") == 90.0


def test_classify_job_failure_keyword_lists_are_load_bearing():
    for reason in ("spot preempted", "ssh timeout", "boot throughput floor",
                   "connection reset", "instance lost"):
        assert ctl._classify_job_failure({"status": "failed",
                                          "fail_reason": reason}) == "infrastructure"
    for reason in ("artifact manifest missing", "sha256 mismatch",
                   "postcondition failed"):
        assert ctl._classify_job_failure({"status": "failed",
                                          "fail_reason": reason}) == "postcondition"
    assert ctl._classify_job_failure({"status": "failed",
                                      "fail_reason": "exit 1"}) == "entrypoint"
    assert ctl._classify_job_failure({"status": "cancelled"}) == "infrastructure"


def test_stage_inflight_is_the_mirror_of_the_terminal_set():
    # (`== flat.STAGE_INFLIGHT` dropped at step 7 — same object, two names.)
    assert ctl.STAGE_INFLIGHT == frozenset({"stage_planned", "box_acquired",
                                            "stage_submitted", "stage_started"})
    assert not (ctl.STAGE_INFLIGHT & wm.STAGE_TERMINAL)
    assert ctl.STAGE_INFLIGHT <= wm.EVENTS


# --------------------------------------------------------------------------- #
# The local lock and the detach seam
# --------------------------------------------------------------------------- #
def test_the_local_lock_refuses_a_second_holder_and_steals_only_from_a_corpse(
        monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    fh = ctl.acquire_local_lock(WF_ID)
    try:
        with pytest.raises(ctl.WorkflowCtlError, match="holds the local lock"):
            ctl.acquire_local_lock(WF_ID)
        monkeypatch.setattr(ctl.os, "kill",
                            lambda pid, sig: (_ for _ in ()).throw(PermissionError()))
        with pytest.raises(ctl.WorkflowCtlError, match="--takeover refused"):
            ctl.acquire_local_lock(WF_ID, takeover=True)   # alive, other user
    finally:
        ctl.release_local_lock(fh)
    assert os.path.isdir(os.path.join(str(tmp_path), "vast-workflowctl"))


def test_spawn_detached_never_falls_back_to_nohup(monkeypatch):
    monkeypatch.setattr(ctl.shutil, "which", lambda name: None)
    with pytest.raises(ctl.DetachUnavailable) as e:
        ctl.spawn_detached(["python3", "herdd.py", "workflow", "run", "w.py"],
                           wf_id=WF_ID)
    assert str(e.value) == "python3 herdd.py workflow run w.py"


def test_spawn_detached_builds_a_same_dir_restart_on_failure_unit(monkeypatch):
    seen = {}

    class _P:
        returncode = 0
        stderr = ""

    monkeypatch.setattr(ctl.shutil, "which", lambda name: "/usr/bin/systemd-run")
    monkeypatch.setattr(ctl.subprocess, "run",
                        lambda cmd, **k: (seen.setdefault("cmds", []).append(cmd),
                                          _P())[1])
    out = ctl.spawn_detached(["python3", "herdd.py", "workflow", "run", "w.py"],
                             wf_id=WF_ID)
    cmd = seen["cmds"][-1]
    assert out["status"] == "detached" and out["unit"].startswith("wfctl-")
    assert "--same-dir" in cmd and "--property=Restart=on-failure" in cmd
    assert cmd[-4:] == ["workflow", "run", "w.py"][-3:] or cmd[-1] == "w.py"


def test_run_controller_detach_delegates_and_returns_ok(monkeypatch):
    seen = {}
    monkeypatch.setattr(ctl, "spawn_detached",
                        lambda argv, wf_id=None: seen.update(argv=argv, wf_id=wf_id))
    rc = ctl.run_controller(_wf(), WF_ID, actor="me", detach=True, argv=["x"])
    assert rc == ctl.EXIT_OK and seen == {"argv": ["x"], "wf_id": WF_ID}


def test_run_workflow_pins_the_detached_reexec_to_this_wf_id(monkeypatch, b2):
    """Without the pin the child re-plans and mints a SECOND, unadvertised id
    (found live 2026-07-15)."""
    seen = {}
    monkeypatch.setattr(ctl, "load_workflow_module", lambda p: _wf())
    monkeypatch.setattr(ctl, "run_controller",
                        lambda wf, wf_id, **kw: (seen.update(kw), ctl.EXIT_OK)[1])
    rc, out = ctl.run_workflow("w.py", wf_id=WF_ID, actor="me", detach=True,
                               argv=["python3", "herdd.py", "workflow", "run",
                                     "w.py"], runner=b2)
    assert rc == ctl.EXIT_OK and out["wf_id"] == WF_ID
    assert seen["argv"][-4:] == ["--detached-controller", "--wf-id", WF_ID][-3:] or \
        seen["argv"][-2:] == ["--wf-id", WF_ID]
    assert "--detached-controller" in seen["argv"]

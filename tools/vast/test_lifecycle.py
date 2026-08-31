"""Portable tests for herdd's park/resume (stop/start) lifecycle.

Runs in the toolchain-free lane (no real vast API, no B2/rclone, no network):
  * `_start_busy` — GPU-contention vs fatal classification of a start refusal.
  * `_put_state_soft` — HTTP-200-but-{"success": false} is an error.
  * `_launch_preflight` — a STOPPED/parked run:<ID> twin blocks a new launch
    (points at resume/destroy) exactly like a live twin blocks it today.
  * `_last_stopping_actor` — a later resumed/launched/relaunched event CLEARS
    the parked operator-stop intent (poll 2b must not read a stale park as
    operator_destroy after a resume).
  * `cmd_start` — the --retry loop re-PUTs on busy, gives up past the deadline,
    and emits `resumed` for run:-labelled boxes only.
"""
import argparse
import json
import os
import shutil
import socket
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import herdd  # noqa: E402
import imageref  # noqa: E402
import runmeta as rm  # noqa: E402
import jobmeta  # noqa: E402
from vastlib.boxes import lifecycle  # noqa: E402
from vastlib.boxes import ssh as ssh_mod  # noqa: E402
from vastlib.cli import _compose, _ls_render  # noqa: E402
from vastlib.cli import _runsets as runsets  # noqa: E402
from vastlib.cli import ls as cli_ls  # noqa: E402
from vastlib.cli import main as cli_main  # noqa: E402
from vastlib.core import api, config, fmt, models  # noqa: E402
from vastlib.jobs import bundle  # noqa: E402
from vastlib.jobs import view as jobs_view  # noqa: E402
from vastlib.launch import launch as launch_mod  # noqa: E402
from vastlib.launch import spec  # noqa: E402
from vastlib.market import offers, pricing  # noqa: E402
from vastlib.storage import b2  # noqa: E402
from vastlib.supervise import replacement  # noqa: E402


# =============================================================================
# _start_busy classification
# =============================================================================
@pytest.mark.parametrize("msg", [
    "Failed to start instance: insufficient GPU capacity on machine",
    "instance is unable to start: resources unavailable",
    "GPUs currently in use by another contract",
    "machine occupied, try again later",
    "cannot schedule: conflict with existing rental",
    "HTTP 400 on PUT v0/instances/1/: no available gpus",
])
def test_start_busy_contention_messages(msg):
    assert lifecycle._start_busy(msg)


@pytest.mark.parametrize("msg", [
    "HTTP 401 on PUT v0/instances/1/: unauthorized",
    "HTTP 404 on PUT v0/instances/1/: instance not found",
    "config: VASTAI_API_KEY not set (env or .env)",
    None,
    "",
])
def test_start_busy_fatal_messages_are_not_busy(msg):
    assert not lifecycle._start_busy(msg)


# =============================================================================
# _put_state_soft — success:false is an error
# =============================================================================
def test_put_state_soft_success_false_is_error(monkeypatch):
    monkeypatch.setattr(api, "request_soft",
                        lambda *a, **k: (True, {"success": False,
                                                "msg": "no available gpus"}, None))
    ok, err = lifecycle._put_state_soft(1, "running")
    assert not ok and "no available gpus" in err
    assert lifecycle._start_busy(err)


def test_put_state_soft_http_error_passthrough(monkeypatch):
    monkeypatch.setattr(api, "request_soft",
                        lambda *a, **k: (False, None, "HTTP 404 on PUT: gone"))
    ok, err = lifecycle._put_state_soft(1, "running")
    assert not ok and "404" in err


def test_put_state_soft_plain_200_is_ok(monkeypatch):
    monkeypatch.setattr(api, "request_soft",
                        lambda *a, **k: (True, {"success": True}, None))
    assert lifecycle._put_state_soft(1, "stopped") == (True, None)


# =============================================================================
# launch_instance — never orphan a billable contract on the success:False path
# =============================================================================
def test_launch_instance_surfaces_contract_on_success_false(monkeypatch):
    # vast's real bid response while the bid is pending: success:False WITH a
    # live new_contract + instance_api_key. The old guard discarded the id and
    # returned failure, orphaning a billing box (live-reproduced 2026-07-13).
    monkeypatch.setattr(api, "request_soft",
                        lambda *a, **k: (True, {"success": False,
                                                "new_contract": 44743274,
                                                "instance_api_key": "deadbeef"},
                                         None))
    ok, cid, err = lifecycle.launch_instance(36656763, {"image": "alpine:latest"})
    assert ok and cid == 44743274 and err is None


def test_launch_instance_success_true_returns_contract(monkeypatch):
    monkeypatch.setattr(api, "request_soft",
                        lambda *a, **k: (True, {"success": True,
                                                "new_contract": 42}, None))
    assert lifecycle.launch_instance(1, {}) == (True, 42, None)


def test_launch_instance_no_contract_is_failure(monkeypatch):
    # success:False AND no contract allocated -> a genuine failure, no id to leak
    monkeypatch.setattr(api, "request_soft",
                        lambda *a, **k: (True, {"success": False,
                                                "msg": "no available gpus"}, None))
    ok, cid, err = lifecycle.launch_instance(1, {})
    assert not ok and cid is None and "no available gpus" in err


def test_launch_instance_http_error_passthrough(monkeypatch):
    monkeypatch.setattr(api, "request_soft",
                        lambda *a, **k: (False, None, "HTTP 500 on PUT"))
    ok, cid, err = lifecycle.launch_instance(1, {})
    assert not ok and cid is None and "500" in err


# =============================================================================
# _launch_preflight — parked twin blocks like a live twin
# =============================================================================
def _inst(iid, label, status):
    return {"id": iid, "label": label, "actual_status": status}


def test_preflight_live_twin_still_blocks(monkeypatch):
    monkeypatch.setattr(lifecycle, "_instances",
                        lambda: [_inst(11, "run:r1", "running")])
    with pytest.raises(SystemExit) as e:
        lifecycle._launch_preflight("run:r1", force=False)
    assert "live instance [11]" in str(e.value)


def test_preflight_parked_twin_blocks_and_points_at_resume(monkeypatch):
    monkeypatch.setattr(lifecycle, "_instances",
                        lambda: [_inst(12, "run:r1", "stopped")])
    with pytest.raises(SystemExit) as e:
        lifecycle._launch_preflight("run:r1", force=False)
    msg = str(e.value)
    assert "STOPPED/parked" in msg and "herdd start 12" in msg


def test_preflight_parked_twin_force_passes(monkeypatch):
    monkeypatch.setattr(lifecycle, "_instances",
                        lambda: [_inst(12, "run:r1", "stopped")])
    lifecycle._launch_preflight("run:r1", force=True)   # no raise


def test_preflight_other_runs_and_unlabelled_ignored(monkeypatch):
    monkeypatch.setattr(lifecycle, "_instances",
                        lambda: [_inst(13, "run:other", "stopped"),
                                 _inst(14, "scratch", "running"),
                                 _inst(15, None, "stopped")])
    lifecycle._launch_preflight("run:r1", force=False)  # no raise


# =============================================================================
# _ssh_endpoints — direct mapping preferred over the (stale-able) api endpoint
# =============================================================================
def test_ssh_endpoints_direct_first_then_api():
    i = {"public_ipaddr": "1.2.3.4",
         "ports": {"22/tcp": [{"HostIp": "0.0.0.0", "HostPort": "17165"}]},
         "ssh_host": "ssh8.vast.ai", "ssh_port": 11244}
    assert ssh_mod._ssh_endpoints(i) == [
        ("1.2.3.4", 17165, "direct"), ("ssh8.vast.ai", 11244, "api")]


def test_ssh_endpoints_dedup_when_api_is_the_direct_ip():
    i = {"public_ipaddr": "1.2.3.4",
         "ports": {"22/tcp": [{"HostPort": "17165"}]},
         "ssh_host": "1.2.3.4", "ssh_port": 17165}
    assert ssh_mod._ssh_endpoints(i) == [("1.2.3.4", 17165, "direct")]


def test_ssh_endpoints_api_only_and_empty():
    assert ssh_mod._ssh_endpoints(
        {"ssh_host": "h", "ssh_port": 1}) == [("h", 1, "api")]
    assert ssh_mod._ssh_endpoints({}) == []
    assert ssh_mod._ssh_endpoints(
        {"public_ipaddr": "1.2.3.4", "ports": {"22/tcp": [{}]}}) == []


def test_pick_ssh_endpoint_falls_back_unprobed_when_nothing_answers(monkeypatch):
    def refuse(*a, **k):
        raise OSError("refused")
    monkeypatch.setattr(socket, "create_connection", refuse)
    i = {"ssh_host": "h", "ssh_port": 1}
    assert ssh_mod._pick_ssh_endpoint(i, probe_timeout=0.01) == ("h", 1, "api")
    assert ssh_mod._pick_ssh_endpoint({}, probe_timeout=0.01) == (None, None, None)


# =============================================================================
# _last_stopping_actor — resume clears parked operator intent
# =============================================================================
def _write_events(tmp_path, run_id, events):
    d = tmp_path / "vast-runmeta" / run_id / "events"
    d.mkdir(parents=True)
    for i, ev in enumerate(events):
        body = {"v": 1, "run_id": run_id, "nonce": f"{i:012x}", **ev}
        (d / f"{ev['ts']}-x-{i:012x}.json").write_text(json.dumps(body))


def test_stopping_intent_stands_when_last(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    _write_events(tmp_path, "r1", [
        {"ts": "20260710T010000000Z", "event": "launched", "actor": "cli:lap"},
        {"ts": "20260710T020000000Z", "event": "stopping", "actor": "cli:lap"},
    ])
    assert spec._last_stopping_actor("r1") == "cli:lap"


def test_resumed_clears_stopping_intent(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    _write_events(tmp_path, "r1", [
        {"ts": "20260710T010000000Z", "event": "stopping", "actor": "cli:lap"},
        {"ts": "20260710T020000000Z", "event": "resumed", "actor": "cli:lap"},
    ])
    assert spec._last_stopping_actor("r1") is None


def test_relaunched_clears_but_newer_stopping_stands(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    _write_events(tmp_path, "r1", [
        {"ts": "20260710T010000000Z", "event": "stopping", "actor": "cli:lap"},
        {"ts": "20260710T020000000Z", "event": "relaunched", "actor": "supervisor"},
        {"ts": "20260710T030000000Z", "event": "stopping", "actor": "cli:lap"},
    ])
    assert spec._last_stopping_actor("r1") == "cli:lap"


def test_resumed_event_type_is_in_schema():
    assert "resumed" in rm.EVENTS


def test_fold_tolerates_resumed_event():
    view = rm.fold_events([json.dumps(
        {"v": 1, "ts": "20260710T010000000Z", "actor": "cli:lap",
         "event": "resumed", "run_id": "r1", "nonce": "0" * 12})])
    assert view["parse_errors"] == 0


# =============================================================================
# cmd_start — retry loop + resumed emit
# =============================================================================
# MIGRATED (was MIGRATION-BLOCKED, plan §7 batch B2): the `fleet_operator_intent`
# seam at `boxes.lifecycle` is bound, not stubbed — `cli/_compose.py::bind()`
# assigns the real `fleet.client.fleet_operator_intent` onto it, which is what
# every command does before it runs. These tests call `lifecycle.cmd_start`
# DIRECTLY rather than through `cli/start.py`, so the harness calls `bind()`
# itself (same idiom as `test_broker_env._jobs_seams`); conftest's
# `_restore_cross_ring_seam_bindings` hands the census back afterwards so
# `test_vastlib_launch.py`'s raising-seam assertions stay order-independent.
#
# Binding rather than stubbing is what keeps the expectation unchanged: with
# conftest pointing `FLEETD_SOCK` at a path that cannot exist, the real intent
# call degrades to `nodaemon:FileNotFoundError` — exactly the harmless
# best-effort the flat body did. `cmd_job_attach` became `SEAM_BINDINGS` row
# five on 2026-08-17, so `bind()` below now points it at the real body too —
# inert here, and deliberately so: `cmd_start` only reaches it when
# `_box_is_jobd(iid)` is True, and that probe reads B2 through jobmeta, which
# conftest blocks, so it returns False and the reattach branch never runs. If a
# future harness stubs `_box_is_jobd` True, this is where an unexpected ssh
# would come from — stub `lifecycle.cmd_job_attach` in that test.
class _StartHarness:
    def __init__(self, monkeypatch, put_results, label="run:r1"):
        self.puts = []
        self.sleeps = []
        self.emitted = []
        results = iter(put_results)

        def fake_put(iid, state):
            self.puts.append((iid, state))
            return next(results)

        monkeypatch.setattr(lifecycle, "_put_state_soft", fake_put)
        monkeypatch.setattr(lifecycle, "_instances_soft",
                            lambda: [{"id": 21, "label": label,
                                      "actual_status": "stopped"}])
        _compose.bind()          # the cross-ring seam, wired the way a command does
        monkeypatch.setattr(time, "sleep", self.sleeps.append)
        monkeypatch.setattr(
            rm, "emit_event",
            lambda rid, event, **kw: self.emitted.append((rid, event)) or {})

    @staticmethod
    def args(**over):
        base = dict(id=[21], wait=0, retry=0)
        base.update(over)
        return argparse.Namespace(**base)


def test_start_retries_busy_then_succeeds_and_emits_resumed(monkeypatch):
    h = _StartHarness(monkeypatch, [
        (False, "no available gpus"),
        (False, "no available gpus"),
        (True, None),
    ])
    lifecycle.cmd_start(h.args(retry=600))
    assert len(h.puts) == 3
    assert h.emitted == [("r1", "resumed")]


def test_start_busy_past_deadline_fails_with_guidance(monkeypatch, capsys):
    h = _StartHarness(monkeypatch, [(False, "insufficient capacity")])
    with pytest.raises(SystemExit):
        lifecycle.cmd_start(h.args(retry=0))
    err = capsys.readouterr().err
    assert "rented by someone else" in err
    assert h.emitted == []                       # no resumed on failure


def test_start_fatal_does_not_retry(monkeypatch):
    h = _StartHarness(monkeypatch, [(False, "HTTP 404 on PUT: gone")])
    with pytest.raises(SystemExit):
        lifecycle.cmd_start(h.args(retry=600))
    assert len(h.puts) == 1                      # fatal -> no busy retry


def test_start_unlabelled_box_emits_nothing(monkeypatch):
    h = _StartHarness(monkeypatch, [(True, None)], label="scratch")
    lifecycle.cmd_start(h.args())
    assert h.emitted == []


# =============================================================================
# _load_runset_spot_config — runsets/<name>/config.yaml 'spot:' block
# (SPOT_DESIGN §3.4): optional, absent-is-{} by design.
# =============================================================================
def _write_runset_config(tmp_path, monkeypatch, runset, text):
    monkeypatch.setattr(runsets, "_HERE", str(tmp_path))
    d = tmp_path / "runsets" / runset
    d.mkdir(parents=True)
    (d / "config.yaml").write_text(text)


def test_spot_config_absent_is_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(runsets, "_HERE", str(tmp_path))    # no runsets/ dir at all
    assert runsets._load_runset_spot_config("nope") == {}


def test_spot_config_no_spot_block_is_empty(tmp_path, monkeypatch):
    _write_runset_config(tmp_path, monkeypatch, "r1", "name: r1\n")
    assert runsets._load_runset_spot_config("r1") == {}


def test_spot_config_reads_all_keys(tmp_path, monkeypatch):
    _write_runset_config(tmp_path, monkeypatch, "r1", """\
name: r1
spot:
  max_bid_mult: 1.25
  defend_at: 0.9
  rescue_wait_s: 900
  ckpt_interval_s: 180
  budget_usd: 40
""")
    cfg = runsets._load_runset_spot_config("r1")
    assert models._num_dph(cfg.get("max_bid_mult")) == 1.25
    assert models._num_dph(cfg.get("defend_at")) == 0.9
    assert models._num_dph(cfg.get("rescue_wait_s")) == 900
    assert models._num_dph(cfg.get("ckpt_interval_s")) == 180
    assert models._num_dph(cfg.get("budget_usd")) == 40


def test_spot_config_malformed_yaml_is_empty(tmp_path, monkeypatch):
    # unparseable under the stdlib fallback (PyYAML, if installed, is more
    # forgiving) -> advisory degrade to {}, never blocks a launch.
    monkeypatch.setattr(jobmeta, "_parse_job_yaml",
                        lambda text: (_ for _ in ()).throw(jobmeta.JobmetaError("x")))
    _write_runset_config(tmp_path, monkeypatch, "r1", "name: r1\n")
    assert runsets._load_runset_spot_config("r1") == {}


# =============================================================================
# image identity — digest stamping + ls staleness (2026-07-11)
# =============================================================================
def test_split_image_forms():
    assert imageref._split_image("registry.example.com/train:t215-latest") \
        == ("registry.example.com", "train", "t215-latest")
    assert imageref._split_image("registry.example.com/train") \
        == ("registry.example.com", "train", "latest")
    assert imageref._split_image("ubuntu") == (None, None, None)
    assert imageref._split_image("") == (None, None, None)


def test_image_tag_digest_skips_foreign_registries(monkeypatch):
    # an image off OUR registry must resolve to None without touching skopeo.
    # patch target is `imageref`, not `herdd`: image_tag_digest and its
    # _skopeo_digest callee both live in imageref since the I3 extraction, so
    # the inner call resolves through imageref's globals (plan §4 rule 1)
    monkeypatch.setattr(imageref, "_skopeo_digest",
                        lambda *a: pytest.fail("skopeo called for foreign registry"))
    imageref._digest_cache.clear()
    assert imageref.image_tag_digest("vllm/vllm-openai:latest") is None
    # and the RETIRED registry is foreign too — no credential path exists
    assert imageref.image_tag_digest(
        "registry.gitlab.com/example/project:train-t211-latest") is None


# --- image_ref_digest: by-digest / ours / docker.io / unresolvable ----------
# image_ref_digest and every callee it steers (_skopeo_digest, image_tag_digest,
# _split_image) moved to imageref.py together (I3), so the inner lookups resolve
# through imageref's globals — these tests patch `imageref`, not `herdd`
# (plan §4 rule 1). The CACHES are re-exported by identity, so clearing them via
# either module empties the one dict imageref reads.
DIG = "sha256:" + "0e737e59" + "0" * 56
OURS = "registry.example.com/train:t215-latest"


def _clear_ref_cache():
    imageref._ref_digest_cache.clear()
    imageref._digest_cache.clear()


def test_image_ref_digest_by_digest_self_certifies_without_skopeo(monkeypatch):
    # a by-digest ref is self-certifying: with skopeo ABSENT, return the in-ref
    # digest (a network blip must not desync plan vs box_resolver)
    _clear_ref_cache()
    monkeypatch.setattr(shutil, "which", lambda name: None)
    ref = f"axolotlai/axolotl@{DIG}"
    assert imageref.image_ref_digest(ref) == DIG


def test_image_ref_digest_by_digest_probe_matches(monkeypatch):
    # skopeo present + returns the SAME digest -> still the in-ref digest
    _clear_ref_cache()
    monkeypatch.setattr(imageref, "_skopeo_digest", lambda ref: DIG)
    assert imageref.image_ref_digest(f"axolotlai/axolotl@{DIG}") == DIG


def test_image_ref_digest_by_digest_probe_contradicts_fails_closed(monkeypatch):
    # a definitive skopeo contradiction (different digest) fails closed to None
    _clear_ref_cache()
    monkeypatch.setattr(imageref, "_skopeo_digest",
                         lambda ref: "sha256:" + "b" * 64)
    assert imageref.image_ref_digest(f"axolotlai/axolotl@{DIG}") is None


def test_image_ref_digest_our_registry_delegates_to_image_tag_digest(monkeypatch):
    # a tag on one of ours: the creds-ful resolver wins first
    _clear_ref_cache()
    monkeypatch.setattr(imageref, "image_tag_digest", lambda image: DIG)
    monkeypatch.setattr(imageref, "_skopeo_digest",
                         lambda ref: pytest.fail("bare skopeo used when the "
                                                 "creds-ful path resolved"))
    assert imageref.image_ref_digest(OURS) == DIG


def test_image_ref_digest_our_registry_falls_back_to_skopeo(monkeypatch):
    # our tag, creds-ful miss (no REGISTRY_AUTH_SECRET) -> bare skopeo fallback
    _clear_ref_cache()
    monkeypatch.setattr(imageref, "image_tag_digest", lambda image: None)
    monkeypatch.setattr(imageref, "_skopeo_digest", lambda ref: DIG)
    assert imageref.image_ref_digest(OURS) == DIG


def test_image_ref_digest_dockerio_tag_via_skopeo(monkeypatch):
    _clear_ref_cache()
    monkeypatch.setattr(imageref, "_skopeo_digest",
                         lambda ref: DIG if ref == "vllm/vllm-openai:latest" else None)
    assert imageref.image_ref_digest("vllm/vllm-openai:latest") == DIG


def test_image_ref_digest_unresolvable_fails_closed(monkeypatch):
    # unknown-host tag + skopeo unavailable -> None (fail-closed)
    _clear_ref_cache()
    monkeypatch.setattr(imageref, "_skopeo_digest", lambda ref: None)
    assert imageref.image_ref_digest("vllm/vllm-openai:latest") is None


def test_instance_env_wire_forms():
    assert models._instance_env(
        {"extra_env": [["A", "1"], ["-p 8000:8000", "1"]]}) == {
            "A": "1", "-p 8000:8000": "1"}
    assert models._instance_env({"extra_env": {"A": "1"}}) == {"A": "1"}
    assert models._instance_env({}) == {}
    assert models._instance_env({"extra_env": [["bad"], "junk"]}) == {}


def test_ls_flags_stale_image_and_has_no_storage_nag(monkeypatch, capsys):
    boxes = [
        {"id": 1, "actual_status": "running", "num_gpus": 1, "gpu_name": "X",
         "dph_total": 0.1, "label": "run:a",
         "image_uuid": "registry.example.com/train:t215-latest",
         "extra_env": [[imageref.IMAGE_DIGEST_ENV, "sha256:OLD"]]},
        {"id": 2, "actual_status": "exited", "num_gpus": 1, "gpu_name": "X",
         "dph_total": 0.1, "label": "run:b",
         "image_uuid": "registry.example.com/train:t215-latest",
         "extra_env": [[imageref.IMAGE_DIGEST_ENV, "sha256:NEW"]]},
        {"id": 3, "actual_status": "exited", "num_gpus": 1, "gpu_name": "X",
         "dph_total": 0.1, "label": "unstamped-old-box"},
    ]
    monkeypatch.setattr(lifecycle, "_instances", lambda: boxes)
    monkeypatch.setattr(imageref, "image_tag_digest", lambda img: "sha256:NEW")
    # market read is best-effort; force it empty so the test needs no network
    monkeypatch.setattr(_ls_render, "_market_map",
                        lambda ins, enabled=True, prog=None: {})
    monkeypatch.setattr(jobs_view, "_fold_fleet_jobs",
                        lambda live, prog=None: {})
    cli_ls.run(argparse.Namespace(json=False, minimal=False,
                                      cached=False, no_spot=True))
    out = capsys.readouterr().out
    assert "STALE-IMAGE" in out and "warn: 1 box(es)" in out
    assert "train:t215-latest" in out              # image column shown
    assert "bill STORAGE" not in out               # the parked nag is gone
    # box 2 carries a fresh stamp → its row must not be flagged STALE-IMAGE
    lines = [l for l in out.splitlines() if " 2  " in l and "exited" in l]
    assert lines and "STALE-IMAGE" not in lines[0]


def test_ls_json_rows_and_positional_id_filter(monkeypatch, capsys):
    boxes = [
        {"id": 1, "actual_status": "running", "num_gpus": 1, "gpu_name": "X",
         "dph_total": 0.1, "label": "run:a"},
        {"id": 2, "actual_status": "exited", "num_gpus": 1, "gpu_name": "X",
         "dph_total": 0.1, "label": "run:b"},
    ]
    monkeypatch.setattr(lifecycle, "_instances", lambda: boxes)
    monkeypatch.setattr(_ls_render, "_market_map",
                        lambda ins, enabled=True, prog=None: {})
    monkeypatch.setattr(jobs_view, "_fold_fleet_jobs",
                        lambda live, prog=None: {})
    # the fleetd-DOWN banner is host-state (a state dir on the dev box)
    monkeypatch.setattr(cli_ls, "fleet_daemon_banner", lambda: None)

    def ns(**over):
        base = dict(json=False, json_rows=False, minimal=False,
                    cached=False, no_spot=True, ids=[])
        base.update(over)
        return argparse.Namespace(**base)

    # --json-rows: the minimal table as data — same columns, same strings
    cli_ls.run(ns(json_rows=True))
    rows = json.loads(capsys.readouterr().out)
    assert [r["id"] for r in rows] == ["1", "2"]
    assert list(rows[0]) == list(_ls_render._MINIMAL_COLS)
    # positional ids narrow the raw --json view too (API shape preserved)
    cli_ls.run(ns(json=True, ids=["2"]))
    assert [i["id"] for i in json.loads(capsys.readouterr().out)] == [2]
    # ...and the minimal TSV
    cli_ls.run(ns(minimal=True, ids=["2"]))
    out = capsys.readouterr().out.splitlines()
    assert len(out) == 2 and out[1].split("\t")[1] == "2"
    # an id set matching nothing is exit 2, never an empty-but-green table
    with pytest.raises(SystemExit) as ei:
        cli_ls.run(ns(minimal=True, ids=["99"]))
    assert ei.value.code == 2


def test_sync_file_list_tracked_only():
    root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    files = bundle._sync_file_list(root, ["tools/vast/herdd.py"])
    assert files == ["tools/vast/herdd.py"]
    with pytest.raises(SystemExit):
        bundle._sync_file_list(root, ["no/such/path"])


# ---------------------------------------------------------------------------
# auto-bid pricing at launch (AUTOBID_DESIGN): agents never pass --price; the
# daemon derives it from the live floor, clamped strictly below on-demand.
# ---------------------------------------------------------------------------
def _launch_ns(**over):
    base = dict(offer=None, type="bid", price=None, env=None, port=None,
                jupyter=False, onstart=None, no_hf_token=True, hf_token=None,
                ssh=False, ssh_key_file=None, jobs=False, image="img:tag",
                disk=40, runtype="ssh_direct", label=None, template_id=None,
                no_registry_login=True, login=None, dry_run=True, wait=None,
                force=False)
    base.update(over)
    return argparse.Namespace(**base)


def _do_launch_dry(monkeypatch, ns):
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        launch_mod._do_launch(ns)
    return json.loads(buf.getvalue().splitlines()[-1] if "{" not in buf.getvalue()
                      else buf.getvalue()[buf.getvalue().index("{"):])


def test_auto_bid_price_is_1p2x_floor_clamped_below_on_demand():
    # The name is literal again: BID_TARGET_MULT went 1.2 -> 2.00 (2026-08-08
    # displacement audit, when this line briefly asserted 0.40) -> 1.20 (owner
    # ruling 2026-08-09: "pay near the market, not a fraction of on-demand").
    assert pricing._auto_bid_price(0.20) == 0.24               # 1.2 * 0.20
    # 1.2 x floor, under the 0.65 x on-demand cost cap ($0.65)
    assert pricing._auto_bid_price(0.20, on_demand=1.00) == 0.24
    # floor 0.60 / on-demand 1.00: the 0.65 cost cap lands just over the floor,
    # so the survival cushion (1.10 x 0.60 = $0.66) prices it, under the $0.75
    # hard ceiling.
    assert pricing._auto_bid_price(0.60, on_demand=1.00) == 0.66
    # THIN MARKET, 2026-08-09 (recalibration item A). floor 0.90 / on-demand 1.00
    # used to clamp onto `on_demand - 0.001` and later onto the $0.99 cushion —
    # both of them ~99% of list for a preemptible box. There is now a HARD ceiling
    # at 0.75 x on-demand and a bid that cannot fit under it is an escalation, not
    # a bigger number: the launch declines to auto-price this offer and the caller
    # picks another one (or is told to name a `--price`).
    assert pricing._auto_bid_price(0.90, on_demand=1.00) is None
    # the cheap-GPU epsilon case (floor 0.080, on-demand 0.082) is the same shape
    # — the floor is 98% of on-demand — and is likewise refused now. It used to
    # emit $0.081. Keeping the numbers so the change is legible.
    assert pricing._auto_bid_price(0.080, on_demand=0.082) is None
    assert pricing._auto_bid_price(None) is None


# --- the pinned-offer ladder, at the QUERY level ----------------------------
# MEASURED 2026-08-09 (task #72): vast's offer `id` filter returns HTTP 200 with
# ZERO rows in EVERY view for offers that are live and rentable in the same query
# without it; `v0/bundles/<id>`, `v0/asks/<id>` and `v0/offers/<id>` are 404; and
# chunk ids reshuffle between two identical unfiltered queries. The tests these
# replaced monkeypatched `_offer_pricing_soft` to hand back a row, i.e. they
# asserted a ladder rung the live API cannot produce, and the whole pinned-offer
# autobid path was 100% dead behind them. So: fake `request_soft` at the QUERY
# level instead, with the id filter answering nothing — the way it really does.
def _fake_api(monkeypatch, *, machine_rows=(), scan_rows=()):
    """`request_soft` faked by QUERY SHAPE. Any query carrying an `id` key
    answers zero rows (the dead filter); a `machine_id`-keyed query answers
    `machine_rows`; an unfiltered scan answers `scan_rows`. Returns the queries
    issued, so a test can assert which rungs actually ran."""
    seen = []

    def _fake(method, path, q=None, **k):
        seen.append(q)
        if not isinstance(q, dict):
            return (True, {"offers": []}, None)
        if "id" in q:
            return (True, {"offers": []}, None)          # THE DEAD FILTER
        if "machine_id" in q:
            return (True, {"offers": [dict(o) for o in machine_rows]}, None)
        return (True, {"offers": [dict(o) for o in scan_rows]}, None)

    monkeypatch.setattr(api, "request_soft", _fake)
    return seen


def test_offer_pricing_soft_is_dead_against_the_real_id_filter(monkeypatch):
    """Its documented contract is now "(None, None, None), almost always". Kept
    as a rung because it costs one soft POST and the filter may come back."""
    seen = _fake_api(monkeypatch)
    assert pricing._offer_pricing_soft(123) == (None, None, None)
    assert seen and seen[0].get("type") == "bid" and "id" in seen[0]
    monkeypatch.setattr(api, "request_soft", lambda *a, **k: (False, None, "HTTP 500"))
    assert pricing._offer_pricing_soft(123) == (None, None, None)


def test_offer_machine_scan_recovers_the_row_the_id_filter_cannot(monkeypatch):
    """The rung that actually works: an unfiltered query, matched on the id in
    Python. Never sends an `id` key (it would zero the result set)."""
    row = {"id": 555, "machine_id": 888, "num_gpus": 2, "min_bid": 0.20}
    seen = _fake_api(monkeypatch, scan_rows=[{"id": 111, "machine_id": 1}, row])
    ns = _launch_ns(offer=555, num_gpus=1, limit=20)
    got = offers._offer_machine_scan_soft(ns)
    assert got["machine_id"] == 888
    assert "id" not in seen[0]
    assert seen[0]["limit"] >= offers.OFFER_SCAN_LIMIT      # a scan, not a pick
    # a miss is expected and cheap — chunk ids reshuffle between listings
    _fake_api(monkeypatch, scan_rows=[{"id": 111, "machine_id": 1}])
    assert offers._offer_machine_scan_soft(_launch_ns(offer=555, limit=20)) is None


def test_do_launch_pinned_offer_is_priced_from_the_machine_scan(monkeypatch):
    """End to end: the id filter answers nothing, the scan recovers the row, and
    the floor comes off it — no --price, no dead end."""
    row = {"id": 555, "machine_id": 888, "num_gpus": 1, "min_bid": 0.20}
    _fake_api(monkeypatch, scan_rows=[row])
    monkeypatch.setattr(pricing, "_market_ondemand_soft",
                        lambda mid, g=None: 1.00 if mid == 888 else None)
    monkeypatch.setattr(imageref, "image_tag_digest", lambda img: None)
    ns = _launch_ns(offer=555, num_gpus=1, limit=20)
    body = _do_launch_dry(monkeypatch, ns)["body"]            # must NOT SystemExit
    assert ns.price == 0.24 and body["price"] == 0.24         # 1.2 * floor


def test_do_launch_offer_machine_prices_via_the_working_market_reads(monkeypatch):
    """`--offer-machine` keeps the pin and supplies the one thing the pin cannot
    resolve. The floor then comes from `_market_min_bid_soft`, which works."""
    _fake_api(monkeypatch)                       # every read is empty
    calls = []
    monkeypatch.setattr(pricing, "_market_min_bid_soft",
                        lambda mid, g=None: (calls.append((mid, g)),
                                             0.20 if mid == 888 else None)[1])
    monkeypatch.setattr(pricing, "_market_ondemand_soft",
                        lambda mid, g=None: 1.00 if mid == 888 else None)
    monkeypatch.setattr(imageref, "image_tag_digest", lambda img: None)
    ns = _launch_ns(offer=555, offer_machine=888, num_gpus=1, limit=20)
    body = _do_launch_dry(monkeypatch, ns)["body"]
    assert ns.price == 0.24 and body["price"] == 0.24         # 1.2 * floor
    assert calls == [(888, 1)]


def test_do_launch_takes_the_chunk_size_from_the_offer_row_not_num_gpus(monkeypatch):
    """Defect D5's shape (handoff-canary-3, 2026-07-15): a machine lists a floor
    PER CHUNK. Pricing a pinned 2-GPU chunk against the 1-GPU floor is the
    underbid vast parks on arrival, and `--num-gpus` defaults to 1."""
    row = {"id": 555, "machine_id": 888, "num_gpus": 2}       # row carries no floor
    _fake_api(monkeypatch, scan_rows=[row])
    calls = []
    monkeypatch.setattr(pricing, "_market_min_bid_soft",
                        lambda mid, g=None: (calls.append((mid, g)), 0.40)[1])
    monkeypatch.setattr(pricing, "_market_ondemand_soft", lambda mid, g=None: 2.00)
    monkeypatch.setattr(imageref, "image_tag_digest", lambda img: None)
    ns = _launch_ns(offer=555, num_gpus=1, limit=20)          # the WRONG count
    _do_launch_dry(monkeypatch, ns)
    assert calls == [(888, 2)], "the floor was read for the wrong chunk size"


def test_do_launch_no_price_source_fails_loud_naming_the_escape_hatches(monkeypatch):
    """The old message blamed the pin ("the offer may be gone, re-search and
    re-pin") — a misdiagnosis, since re-pinning cannot fix a filter that never
    answers. Name the paths that DO work."""
    _fake_api(monkeypatch)
    monkeypatch.setattr(pricing, "_market_min_bid_soft", lambda mid, g=None: None)
    monkeypatch.setattr(pricing, "_market_ondemand_soft", lambda mid, g=None: None)
    monkeypatch.setattr(imageref, "image_tag_digest", lambda img: None)
    with pytest.raises(SystemExit) as ei:
        _do_launch_dry(monkeypatch, _launch_ns(offer=555, num_gpus=1, limit=20))
    msg = str(ei.value)
    assert "tried" in msg and "id filter" in msg
    assert "--offer-machine" in msg and "--machine" in msg
    # the misdiagnosis is not just gone, it is contradicted in place
    assert "re-pinning changes nothing" in msg
    assert "may be gone" not in msg


def test_do_launch_pinned_offer_explicit_price_still_wins(monkeypatch):
    monkeypatch.setattr(
        pricing, "_offer_pricing_soft",
        lambda *a, **k: pytest.fail("explicit --price must skip pricing lookups"))
    monkeypatch.setattr(imageref, "image_tag_digest", lambda img: None)
    ns = _launch_ns(offer=555, price=0.50)
    body = _do_launch_dry(monkeypatch, ns)["body"]
    assert body["price"] == 0.50


def test_do_launch_autopick_bid_prices_without_explicit_price(monkeypatch):
    monkeypatch.setattr(offers, "search_offers",
        lambda a: [{"id": 123, "min_bid": 0.20, "dph_total": 1.00}])
    monkeypatch.setattr(fmt, "fmt_offer", lambda o: "offer-123")
    monkeypatch.setattr(imageref, "image_tag_digest", lambda img: None)
    ns = _launch_ns()
    body = _do_launch_dry(monkeypatch, ns)["body"]
    assert ns.price == 0.24 and body["price"] == 0.24         # 1.2 * 0.20 min_bid


def test_do_launch_offer_path_autoprices_no_requires_price(monkeypatch):
    # a pinned --offer + --type bid + no --price used to hard-error; now it derives
    monkeypatch.setattr(pricing, "_offer_pricing_soft", lambda oid: (0.20, 1.00, 999))
    monkeypatch.setattr(imageref, "image_tag_digest", lambda img: None)
    ns = _launch_ns(offer=555)
    body = _do_launch_dry(monkeypatch, ns)["body"]            # must NOT SystemExit
    assert ns.price == 0.24 and body["price"] == 0.24         # 1.2 * floor


def test_do_launch_explicit_price_is_escape_hatch_not_clamped(monkeypatch):
    monkeypatch.setattr(offers, "search_offers",
        lambda a: [{"id": 123, "min_bid": 0.20, "dph_total": 1.00}])
    monkeypatch.setattr(fmt, "fmt_offer", lambda o: "offer-123")
    monkeypatch.setattr(imageref, "image_tag_digest", lambda img: None)
    ns = _launch_ns(price=1.50)                               # explicit, above on-demand
    body = _do_launch_dry(monkeypatch, ns)["body"]
    assert body["price"] == 1.50                              # explicit NOT clamped


# =============================================================================
# cmd_train boot-pull wire — the full onstart/train.sh outgrew Vast's 16 KiB
# inline-onstart cap, so cmd_train stages it to b2:.../runs/<RUN>/train_main.sh
# and ships only the tiny stripped onstart/train_boot.sh on the wire.
# =============================================================================
def _drive_train(monkeypatch, argv, dry_run):
    """Drive cmd_train through the real argparse (so every default is faithful),
    with all B2/launch side effects stubbed. Returns
    (captured_launch_namespace, [(_b2_rcat path, body), ...], stdout)."""
    import io, contextlib
    monkeypatch.setattr(config, "load_env", lambda: None)
    monkeypatch.setattr(config, "load_herdd_config",
                        lambda: {"default_image": "testimg"})
    monkeypatch.setattr(b2, "_ensure_b2_remote", lambda: None)
    monkeypatch.setattr(spec, "_b2_eu_pairs", lambda: [])
    for k in ("B2_KEY_ID", "B2_APPLICATION_KEY", "B2_S3_ENDPOINT",
              "B2_MINTER_KEY_ID", "B2_MINTER_APPLICATION_KEY",
              "B2_BOX_KEY_ID", "B2_BOX_APPLICATION_KEY", "HF_TOKEN"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("B2_BUCKET", "testbucket")
    monkeypatch.setenv("B2_KEY_ID", "KID")
    monkeypatch.setenv("B2_APPLICATION_KEY", "APPKEY")
    monkeypatch.setenv("B2_S3_ENDPOINT", "https://s3.example.com")

    rcats = []
    monkeypatch.setattr(b2, "_b2_rcat",
                        lambda path, body, hard=True: rcats.append((path, body)) or True)
    monkeypatch.setattr(b2, "_rclone_soft", lambda *a, **k: (0, "", ""))
    captured = {}
    def _fake_do_launch(la):
        captured["la"] = la
        if la.dry_run:
            print(f">> onstart wire: {len(la.onstart.encode('utf-8'))} bytes")
        return None, None, None          # cid None -> cmd_train returns before babysit
    monkeypatch.setattr(launch_mod, "_do_launch", _fake_do_launch)

    monkeypatch.setattr(sys, "argv",
                        ["herdd", "train", "--run", "r1", "--runset", "rs"] + argv)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        cli_main.main()
    return captured.get("la"), rcats, buf.getvalue()


def test_cmd_train_dry_run_ships_stripped_bootstrap_and_prints_stage_line(monkeypatch):
    la, rcats, out = _drive_train(monkeypatch, ["--dry-run"], dry_run=True)
    # the wire is the STRIPPED train_boot.sh, comfortably under the 4 KiB budget
    assert la is not None
    wbytes = len(la.onstart.encode("utf-8"))
    assert wbytes < 4096, wbytes
    assert "runs/${RUN_ID}/train_main.sh" in la.onstart      # the boot-pull is present
    assert "rclone_bootstrap()" not in la.onstart            # NOT the full trainer
    # dry-run performs NO B2 writes and prints what it WOULD stage
    assert rcats == []
    assert "would stage onstart/train.sh" in out
    assert "b2:testbucket/runs/r1/train_main.sh" in out


def test_cmd_train_launch_stages_full_trainer_to_per_run_path(monkeypatch):
    la, rcats, out = _drive_train(monkeypatch, [], dry_run=False)
    # the wire is still the tiny bootstrap
    assert la is not None and len(la.onstart.encode("utf-8")) < 4096
    # the FULL, unstripped trainer was staged to the per-RUN B2 path (comments kept)
    staged = [(p, b) for (p, b) in rcats if p == "b2:testbucket/runs/r1/train_main.sh"]
    assert len(staged) == 1, rcats
    body = staged[0][1]
    assert body.startswith("#!/usr/bin/env bash")
    assert "onstart/train.sh" in body                        # unstripped (full-line comment kept)
    assert "rclone_bootstrap()" in body                      # the real trainer, not the boot wire


# =============================================================================
# handoff (Phase 2) launch plumbing — T3
#   * _launch_preflight lets a run:<ID>:handoff twin coexist with the live
#     primary run:<ID>, but still refuses a plain run:<ID> duplicate and a
#     SECOND understudy.
#   * _handoff_understudy_body builds the understudy launch off the captured
#     spec with the :handoff label + a nonce-suffixed B2 key name, honoring the
#     §2.3 candidate filter.
# =============================================================================
def test_preflight_handoff_twin_allowed_alongside_live_primary(monkeypatch):
    # the primary run:r1 is LIVE; launching its run:r1:handoff understudy must
    # pass the dup guard (the primary must outlive the warmup — HANDOFF_DESIGN §2.1)
    monkeypatch.setattr(lifecycle, "_instances",
                        lambda: [_inst(11, "run:r1", "running")])
    lifecycle._launch_preflight("run:r1:handoff", force=False)   # no raise


def test_preflight_plain_dup_still_refused_with_handoff_twin_present(monkeypatch):
    # allowing the :handoff twin must NOT weaken the plain-run dup refusal: a
    # live run:r1 still blocks a second plain run:r1 launch even when an
    # understudy is already up.
    monkeypatch.setattr(lifecycle, "_instances",
                        lambda: [_inst(11, "run:r1", "running"),
                                 _inst(12, "run:r1:handoff", "running")])
    with pytest.raises(SystemExit) as e:
        lifecycle._launch_preflight("run:r1", force=False)
    assert "live instance [11]" in str(e.value)


def test_preflight_second_understudy_refused(monkeypatch):
    # a run:r1:handoff twin already live -> refuse launching ANOTHER understudy
    # (exact-label match), and name it an "understudy" for clear guidance.
    monkeypatch.setattr(lifecycle, "_instances",
                        lambda: [_inst(11, "run:r1", "running"),
                                 _inst(12, "run:r1:handoff", "running")])
    with pytest.raises(SystemExit) as e:
        lifecycle._launch_preflight("run:r1:handoff", force=False)
    msg = str(e.value)
    assert "understudy" in msg and "live instance [12]" in msg


def _handoff_st(**over):
    # a minimal supervise state: primary paid 0.60/hr (on-demand 0.50 -> preferred
    # ceiling 0.75 x 0.50 = 0.375), 10h wall left, a captured launch spec that
    # names the B2 key pair as secret env (so the understudy's nonce key name
    # lands in the body).
    st = {
        "run_id": "r1",
        "dph_total": 0.60,
        "last_bid": 0.60,
        "on_demand": 0.50,
        "remaining_wall_h": 10.0,
        "launch_spec": {
            "image": "reg/img:tag", "disk": 100, "runtype": "ssh_direct",
            "env": {"RUN_ID": "r1"}, "runset": "rs1",
            "secret_env_keys": ["B2_KEY_ID", "B2_APPLICATION_KEY"],
        },
    }
    st.update(over)
    return st


# an offer that CLEARS §2.3: min_bid 0.10 -> target 1.2 x 0.10 = 0.12 (<= the
# PRIMARY's 0.375 pref ceiling), and (0.60-0.12)*10 = 4.8 savings >
# (0.60+0.12)*0.5 = 0.36 overhead.
_GOOD_OFFER = {"id": 999, "min_bid": 0.10, "dph_total": 0.50}
# an offer that FAILS §2.3(1): target 1.2 x 0.35 = 0.42 > the primary's 0.375
# preferred ceiling (cheaper than the primary, but not genuinely under the line
# we're escaping). min_bid raised 0.30 -> 0.35 at the 2026-08-09 return to a
# 1.20x multiple: 1.2 x 0.30 = 0.36 would slip UNDER the 0.375 line that
# 2.0 x 0.30 = 0.60 (and, originally, 0.36 vs the old 0.50-frac 0.25 line)
# used to breach — same reject branch, recomputed fixture.
_HOT_OFFER = {"id": 888, "min_bid": 0.35, "dph_total": 0.50}


# MIGRATED (was MIGRATION-BLOCKED, plan §7 batch B2) — the two BODY-BUILDING
# understudy tests. `_relaunch_body` landed in `vastlib.supervise.replacement`,
# the same module as `_handoff_understudy_body`, so the :2643 call reaches a real
# body and nothing has to be stubbed to make it run. The seams follow the
# subject: `_ship_b2_pair` at `launch.spec` (where `_resolve_secret` resolves
# it), and the nonce's `random`/`time` as `replacement.random`/`replacement.time`
# — the modules the ported body looks the two names up on.
def test_handoff_understudy_body_label_and_nonce_key(monkeypatch):
    captured = []
    monkeypatch.setattr(spec, "_ship_b2_pair",
                        lambda name, hours=None, dry_run=False:
                        (captured.append(name) or ("KID", "SEC")))
    a = argparse.Namespace(dry_run=True)
    body, bid, missing = replacement._handoff_understudy_body(
        _handoff_st(), a, _GOOD_OFFER)
    assert missing == []                                   # both B2 keys resolved
    assert bid == 0.12                                     # 1.2 * 0.10 min_bid
    # (0.20 during the 2026-08-08 2.00x era; 1.20 restored 2026-08-09)
    assert body["label"] == "run:r1:handoff"              # the twin marker
    assert body["price"] == 0.12
    assert body["env"]["B2_KEY_ID"] == "KID"
    # the mint MUST use a nonce-suffixed name, never the primary's plain run-r1
    # (revoke-then-mint by name would kill the primary's live key mid-run).
    assert captured and captured[0].startswith("run-r1-h")
    assert captured[0] != "run-r1"


def test_handoff_understudy_body_nonce_is_distinct_per_launch(monkeypatch):
    captured = []
    monkeypatch.setattr(spec, "_ship_b2_pair",
                        lambda name, hours=None, dry_run=False:
                        (captured.append(name) or ("KID", "SEC")))
    monkeypatch.setattr(replacement.random, "randint", lambda a, b: 0x1234)
    seq = iter([1000, 2000])
    monkeypatch.setattr(replacement.time, "time", lambda: next(seq))
    a = argparse.Namespace(dry_run=True)
    b1, _, _ = replacement._handoff_understudy_body(_handoff_st(), a, _GOOD_OFFER)
    b2, _, _ = replacement._handoff_understudy_body(_handoff_st(), a, _GOOD_OFFER)
    # two understudy launches for the same run mint DIFFERENT key names (the
    # jobs-launch nonce class): independent keys, no cross-revoke.
    names = {n for n in captured}
    assert "run-r1-h1000-1234" in names and "run-r1-h2000-1234" in names


def test_handoff_understudy_body_rejects_offer_failing_candidate_filter(monkeypatch):
    # a hot offer (its own bid above its own preferred ceiling) is rejected by
    # _handoff_candidate_ok BEFORE any body/mint work.
    monkeypatch.setattr(spec, "_ship_b2_pair",
                        lambda *a, **k: pytest.fail("must reject before minting"))
    a = argparse.Namespace(dry_run=True)
    body, bid, reason = replacement._handoff_understudy_body(
        _handoff_st(), a, _HOT_OFFER)
    assert body is None and bid is None and reason == "candidate_reject"


def test_handoff_understudy_body_rejects_short_remaining_run(monkeypatch):
    # a run with little wall left never amortizes the 2x-box window (§2.3(2)).
    monkeypatch.setattr(spec, "_ship_b2_pair",
                        lambda *a, **k: pytest.fail("must reject before minting"))
    a = argparse.Namespace(dry_run=True)
    body, bid, reason = replacement._handoff_understudy_body(
        _handoff_st(remaining_wall_h=0.3), a, _GOOD_OFFER)
    assert body is None and reason == "candidate_reject"


def test_handoff_pick_offer_returns_cheapest_qualifier(monkeypatch):
    # search returns a HOT offer first (rejected) then a GOOD one -> pick GOOD.
    monkeypatch.setattr(offers, "_search_offers_soft",
                        lambda a: [_HOT_OFFER, _GOOD_OFFER])
    a = argparse.Namespace()
    assert replacement._handoff_pick_offer(_handoff_st(), a) is _GOOD_OFFER


def test_handoff_pick_offer_none_when_no_qualifier(monkeypatch):
    monkeypatch.setattr(offers, "_search_offers_soft", lambda a: [_HOT_OFFER])
    a = argparse.Namespace()
    assert replacement._handoff_pick_offer(_handoff_st(), a) is None


# =============================================================================
# Fail-closed image resolution (owner ruling 2026-08-04). An unreadable
# herdd.yaml used to make `launch`/`supervise` silently default to stock
# pytorch/pytorch:2.4.0 -- no nvcc, no baked train env, not our vLLM fork -- so
# the box rented, pulled off Docker Hub, and only THEN failed (v7's run.sh dies
# at `[ -f /workspace/.train_env_activate ] || exit 3`, i.e. after the meter
# started). It also silently reintroduced the train/serve image seam the t211
# unification removes. cmd_train already refused; launch and supervise now match.
# =============================================================================
def test_require_image_passes_a_real_image_through():
    assert spec._require_image("some/img:tag", "launch") == "some/img:tag"


@pytest.mark.parametrize("missing", [None, ""])
def test_require_image_refuses_with_no_image(missing):
    with pytest.raises(SystemExit) as e:
        spec._require_image(missing, "launch")
    msg = str(e.value)
    # the message must name the expected image and say there is no fallback,
    # or the operator cannot tell a config problem from a network one
    assert "herdd.yaml" in msg
    assert spec._EXPECTED_DEFAULT_IMAGE in msg
    assert "NO fallback" in msg


def test_no_stock_pytorch_fallback_survives_in_code():
    """Grep-level proof that no CODE path can hand a container a stock pytorch
    image. Comments are exempt on purpose: the explanation of what the fallback
    was and why it was removed is exactly the history this repo keeps, and
    deleting it to satisfy a test would lose the reason.

    Class-C repoint (plan §7 hazard H1, batch B2): this is an ABSENCE-assert,
    so it goes VACUOUSLY GREEN — not red — the moment the code leaves the file
    it scans. It now scans the flat launcher AND the two vastlib modules that
    hold the image-resolution path (`launch/spec.py` owns `_require_image` and
    `_EXPECTED_DEFAULT_IMAGE`; `cli/launch.py` owns the parser default). The
    flat scan stays until the flat copy is deleted at 6d.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    scanned = [herdd.__file__,
               os.path.join(here, "vastlib", "launch", "spec.py"),
               os.path.join(here, "vastlib", "cli", "launch.py")]
    offenders = [(os.path.basename(f), n, ln)
                 for f in scanned
                 for n, ln in enumerate(open(f).read().splitlines(), 1)
                 if "pytorch/pytorch:2.4.0" in ln
                 and not ln.lstrip().startswith("#")]
    assert not offenders, f"stock pytorch image in code: {offenders}"


def test_launch_refuses_before_renting_when_no_image_resolves(monkeypatch):
    """The refusal must land BEFORE the offer search, not after the POST --
    refusing late is what costs a rented box."""
    searched = []
    monkeypatch.setattr(offers, "pick_cheapest_offer",
                        lambda *a, **k: searched.append(1) or (None, None, None))
    a = argparse.Namespace(image=None, offer=None, dry_run=True)
    with pytest.raises(SystemExit) as e:
        launch_mod._do_launch(a)
    assert spec._EXPECTED_DEFAULT_IMAGE in str(e.value)
    assert not searched, "offer search ran before the image gate"


def test_main_takes_no_fallback_when_default_image_is_absent(monkeypatch):
    """main() must resolve default_image lazily to None, NOT to a stock image --
    while still building a parser, so `ls`/`stop`/`destroy` keep working when
    the config is broken and a box is billing."""
    monkeypatch.setattr(config, "load_env", lambda: None)
    monkeypatch.setattr(config, "load_herdd_config", lambda: {})
    monkeypatch.setattr(sys, "argv", ["herdd", "launch", "--dry-run"])
    with pytest.raises(SystemExit) as e:
        cli_main.main()
    assert spec._EXPECTED_DEFAULT_IMAGE in str(e.value)

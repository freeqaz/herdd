"""The step-6 SHARED helpers: `cli/_ls_render.py`, `cli/_runsets.py`, and the
eleven names that landed BELOW `cli/` because more than one command reaches them.

What this file is for
---------------------
Plan §8 step 6 was add-only up to 6c, so for every symbol here the flat
definition in `herdd.py` was live in the same process. That made the
strongest possible assertion available and this file used it wherever the
function was pure enough to drive twice: **run both copies and compare**. A
port that drifted by one space in a help string, one column in a TSV or one
argv element failed here, at the site, instead of surfacing as a help-tree byte
diff nobody could localize.

**Step 6d ended it.** The launcher re-exports every one of these names by
identity, so each `== v.<fn>(…)` arm became a call to the same function twice.
They are deleted (each site says so), the pure-parity tests with them, and one
binding assertion at the bottom of the file replaces the lot. What survives is
what a differential comparison could never see — which was always the more
valuable half:

1. **Depth.** Four ported bodies computed a path from `__file__`, and every one
   of them sat in `tools/vast` before the move. `_runsets._HERE`,
   `spec._TOOLS_VAST_DIR` and `client._HERDD_SCRIPT` are compared against the
   same expression applied to `herdd.py`'s own path — the failure they guard
   is SILENT (an absent runsets dir returns `{}`, an absent `hf_login.sh` falls
   through to the inline copy, a wrong script path forks a child that exits
   without supervising anything).
2. **The `--minimal` TSV column order**, pinned literally (plan §4 frozen
   contract). Reordering it is invisible to every other test in the suite and
   silently re-labels every value downstream of the moved column.
3. **The `runner=_rclone_soft` def-time defaults** on the three `runs` folds:
   they bind at import, so patching `b2._rclone_soft` afterwards does NOT steer
   them. That is ported behavior, not a bug (the flat module had the identical
   trap), and it is asserted so nobody "fixes" it into a live network call in a
   test run.
4. **The monkeypatch idiom the six existing runset asserts use** —
   `monkeypatch.setattr(<module>, "_HERE", tmp)` — which only works while
   `_load_runset_config` reads the module global at CALL time.
"""

from __future__ import annotations

import os
import sys
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import herdd as v  # noqa: E402
from vastlib.boxes import lifecycle, reap  # noqa: E402
from vastlib.cli import _ls_render, _runsets  # noqa: E402
from vastlib.core import api, fmt  # noqa: E402
from vastlib.fleet import client  # noqa: E402
from vastlib.jobs import bundle, view  # noqa: E402
from vastlib.launch import spec  # noqa: E402
from vastlib.storage import b2  # noqa: E402

_HERE_DIR = os.path.dirname(os.path.abspath(__file__))


# =============================================================================
# cli/_ls_render.py — the two renderings, byte-compared against the flat ones
# =============================================================================
def _fleet_snapshot():
    """One `_gather_ls_data`-shaped dict exercising every branch the renderers
    have: a live bid box with jobs, a stopped on-demand box with an expiring
    keep-retention label, a zombie verdict, a BOOTING verdict, a stale image, an
    oversized disk and a box with no ssh key in its onstart."""
    return {
        "ts": 100000.0, "no_spot": False,
        "instances": [
            {"id": 41, "actual_status": "running", "num_gpus": 2,
             "gpu_name": "RTX 4090", "gpu_util": 95.0, "dph_total": 0.9,
             "dph_base": 0.8, "min_bid": 0.4, "machine_id": 77,
             "ssh_host": "h1", "ssh_port": 222, "label": "run:R1",
             "storage_total_cost": 0.08, "disk_space": 200.0,
             "disk_usage": 20.0, "is_bid": True, "start_date": 92800.0,
             "extra_env": [["HERDD_IMAGE_DIGEST", "sha256:abc"]],
             "image_uuid": "registry.example.com/train:t213-latest",
             "onstart": ""},
            {"id": 42, "actual_status": "exited", "num_gpus": 1,
             "gpu_name": "A100", "dph_total": 1.2, "dph_base": 1.1,
             "min_bid": 0.5, "machine_id": 78, "is_bid": False,
             "label": "keep:salvage-until-20260817T000000000Z",
             "storage_total_cost": 0.05, "disk_space": 100.0,
             "disk_usage": 5.0, "image_uuid": "pytorch/pytorch:2.4.0",
             "onstart": ""},
        ],
        "live_ids": [41],
        "jobs_by_box": {
            "41": [{"job_id": "j-abc123", "display_status": "running",
                    "status": "running", "name": "trainA", "gpu": True},
                   {"job_id": "j-cpu9", "display_status": "running",
                    "status": "running", "name": "compileB", "gpu": False}],
            "42": [{"job_id": "j-old", "display_status": "done",
                    "status": "done", "ended_at": "2026", "name": "old"}],
        },
        "fleet_watch": {"41": {"spend_usd": 3.21, "budget_usd": 16.0}},
        "fleet_spend_total": 12.5,
        "market": {"77": {"offers": [{"g": 2, "base": 0.7, "bid": 0.3}],
                          "max_gpus": 2},
                   "78": {"offers": [{"g": 1, "base": 1.0, "bid": 0.45}],
                          "max_gpus": 1}},
        "stale_ids": [41],
        "idle_secs": {"42": 90000.0},
        "health": {
            "41": {"iid": 41, "verdict": "ZOMBIE_JOBD_DEAD", "reason": "no hb",
                   "age_s": 3600,
                   "evidence": {"phase": "env-setup", "is_jobs_box": True,
                                "boot_age_s": 3600, "jobd_hb_age_s": 900}},
            "42": {"iid": 42, "verdict": "BOOTING", "reason": "boot",
                   "age_s": 60, "evidence": {"phase": "loading"}},
        },
    }


@pytest.mark.parametrize("cols", [None, 200, 120, 90, 60, 40])
def test_render_ls_reflows_at_every_width_without_raising(cols):
    """Was `…_is_byte_identical_to_the_flat_renderer`: the same snapshot through
    both renderers at every reflow width (the compact layout, the label
    ellipsis and the third resume-price line all switch on `cols`). One
    renderer since step 6d, so what is left is that every width still renders
    the fixture — the branch coverage the widths were chosen for — plus the
    column contract pinned literally below."""
    data = _fleet_snapshot()
    out = _ls_render._render_ls(data, fmt._Pal(True), banner="B", cols=cols)
    assert isinstance(out, list) and out[0] == "B"
    text = "\n".join(out)
    assert "41" in text and "42" in text


def test_render_ls_empty_and_uncolored_paths_do_not_raise():
    assert _ls_render._render_ls({}, fmt._Pal(False)) is not None
    data = _fleet_snapshot()
    assert _ls_render._render_ls(data, fmt._Pal(False))


def test_minimal_tsv_column_order_is_frozen():
    """Plan §4 frozen contract, pinned LITERALLY — a differential test against
    the flat renderer cannot catch a column both copies moved together, and
    agent consumers read this TSV positionally."""
    assert _ls_render._MINIMAL_COLS == (
        "state", "id", "status", "gpus", "gpu", "gpu_util", "mode",
        "hourly", "storage_day", "disk_gb", "disk_used_gb", "idle",
        "avail", "ondemand", "spot", "stale", "label", "jobs", "phase",
        # APPENDED 2026-08-21, never inserted: a dedicated CPU box reads
        # gpu_util 0 and looked idle to every consumer of this table.
        "cpu_util",
        # APPENDED 2026-08-27: uptime since the billing anchor, fleetd watch
        # economics, and the all-active-jobs-are-CPU tag.
        "uptime", "spend_usd", "budget_usd", "cpu_jobs")
    header = _ls_render._render_minimal(_fleet_snapshot()).splitlines()[0]
    assert header == "\t".join(_ls_render._MINIMAL_COLS)


def test_minimal_rows_is_the_tsv_as_data():
    """`ls --json-rows` and the frozen TSV are two spellings of `_minimal_rows`
    — same columns, same strings — so neither can drift from the other."""
    data = _fleet_snapshot()
    rows = _ls_render._minimal_rows(data)
    assert rows and all(tuple(r) == _ls_render._MINIMAL_COLS for r in rows)
    tsv = _ls_render._render_minimal(data).splitlines()
    assert tsv[1:] == ["\t".join(r[c] for c in _ls_render._MINIMAL_COLS)
                       for r in rows]


def test_uptime_budget_and_cpu_chip_render():
    """2026-08-27 additions. Live rows: `up <age> ≈$<upper-bound>` from
    start_date + dph, and the fleetd watch `$spend/$budget`; a job whose fold
    says gpu=False wears the `cpu` chip; the footer carries fleetd's watched
    total. The stale paint of an OLD snapshot (no fleet_watch/start_date keys)
    must render silence, not raise."""
    data = _fleet_snapshot()
    for cols in (None, 60):            # wide and compact layouts both carry it
        text = "\n".join(_ls_render._render_ls(data, fmt._Pal(False), cols=cols))
        assert "up 2h" in text and "≈$1.80" in text     # (100000-92800)s @ $0.9/hr
        assert "$3.21/$16.00" in text
        assert "cpu" in text
    assert "Σ$12.50" in "\n".join(
        _ls_render._render_ls(data, fmt._Pal(False)))
    old = {k: v for k, v in data.items()
           if k not in ("fleet_watch", "fleet_spend_total")}
    for i in old["instances"]:
        i.pop("start_date", None)
    assert _ls_render._render_ls(old, fmt._Pal(False))


def test_minimal_rows_uptime_budget_cpu_columns():
    """The appended columns: uptime only on live boxes, watch economics only on
    watched ones, and cpu_jobs=yes ONLY when every active job is explicitly
    gpu=False (the fixture's box 41 runs a mixed pair, so it must NOT tag)."""
    data = _fleet_snapshot()
    rows = {r["id"]: r for r in _ls_render._minimal_rows(data)}
    assert rows["41"]["uptime"] == "2h"
    assert rows["41"]["spend_usd"] == "3.2100"
    assert rows["41"]["budget_usd"] == "16.0000"
    assert rows["41"]["cpu_jobs"] == ""            # mixed gpu+cpu -> no tag
    assert rows["42"]["uptime"] == ""              # stopped: anchor means nothing
    assert rows["42"]["spend_usd"] == ""           # unwatched
    data["jobs_by_box"]["41"] = [
        {"job_id": "j-cpu9", "display_status": "running",
         "status": "running", "name": "compileB", "gpu": False}]
    rows = {r["id"]: r for r in _ls_render._minimal_rows(data)}
    assert rows["41"]["cpu_jobs"] == "yes"
    # tri-state: an old stream folds gpu=None -> unknown, never tagged
    data["jobs_by_box"]["41"][0]["gpu"] = None
    rows = {r["id"]: r for r in _ls_render._minimal_rows(data)}
    assert rows["41"]["cpu_jobs"] == ""


def test_filter_boxes_narrows_every_keyed_collection_without_mutating():
    """The positional-id filter must narrow the id-keyed maps too (a zombie
    scream for an unfiltered box would leak through `health`), and must leave
    the input dict intact — the snapshot cache saves the full-fleet gather."""
    data = _fleet_snapshot()
    out = _ls_render._filter_boxes(data, [41])
    assert [i["id"] for i in out["instances"]] == [41]
    assert out["live_ids"] == [41]
    assert set(out["jobs_by_box"]) == {"41"}
    assert set(out["health"]) == {"41"}
    assert out["stale_ids"] == [41]
    assert out["idle_secs"] == {}
    assert [i["id"] for i in data["instances"]] == [41, 42]
    assert set(data["jobs_by_box"]) == {"41", "42"}


def test_active_job_states_is_the_dashboard_injection():
    """`storage.dashcache` takes this tuple as a permanent injection
    (`DashDeps.active_job_states`), so a drift here re-groups the dashboard's
    box states without touching the dashboard. Was `…_matches_the_flat_tuple`;
    one tuple since step 6d, and the injection is the thing to pin."""
    from vastlib.storage import dashcache
    assert isinstance(_ls_render._ACTIVE_JOB_STATES, tuple)
    assert "running" in _ls_render._ACTIVE_JOB_STATES
    assert dashcache is not None


def test_market_map_disabled_is_empty_and_probes_nothing(monkeypatch):
    """`--no-spot` must cost zero network: the executor is never entered."""
    def _boom(mid):
        raise AssertionError("probed the market with spot reads disabled")

    monkeypatch.setattr(_ls_render.pricing, "_machine_offers_soft", _boom)
    assert _ls_render._market_map([{"machine_id": 1}], enabled=False) == {}
    assert _ls_render._market_map([], enabled=True) == {}


def test_market_map_folds_one_probe_per_unique_machine(monkeypatch):
    seen = []

    def _fake(mid):
        seen.append(mid)
        return None if mid == 9 else [{"g": 1, "base": 1.0, "bid": 0.5},
                                      {"g": 2, "base": 2.0, "bid": 0.9}]

    monkeypatch.setattr(_ls_render.pricing, "_machine_offers_soft", _fake)
    out = _ls_render._market_map([{"machine_id": 7}, {"machine_id": 7},
                                  {"machine_id": 9}])
    assert sorted(seen) == [7, 9]                  # unique machines only
    assert out["7"]["max_gpus"] == 2               # widest offer wins
    assert "9" not in out                          # a failed read is ABSENT


def test_stale_image_ids_flags_only_a_moved_digest(monkeypatch):
    ins = [
        {"id": 1, "image_uuid": "reg/x:tag",
         "extra_env": [["HERDD_IMAGE_DIGEST", "sha256:old"]]},
        {"id": 2, "image_uuid": "reg/y:tag",
         "extra_env": [["HERDD_IMAGE_DIGEST", "sha256:same"]]},
        {"id": 3, "image_uuid": "reg/z:tag"},          # never stamped
    ]
    monkeypatch.setattr(_ls_render.imageref, "image_tag_digest",
                        lambda img: {"reg/x:tag": "sha256:new",
                                     "reg/y:tag": "sha256:same"}.get(img))
    assert _ls_render._stale_image_ids(ins) == [1]


def test_an_unresolvable_digest_is_UNRESOLVED_never_folded_into_not_stale(
        monkeypatch):
    """A None from `image_tag_digest` used to land in the same bucket as "the
    tag has not moved", so an unset REGISTRY_AUTH_SECRET rendered exactly like
    a clean fleet. Split, so `ls` can say "could not check"."""
    ins = [
        {"id": 1, "image_uuid": "registry.example.com/train:t215-latest",
         "extra_env": [["HERDD_IMAGE_DIGEST", "sha256:old"]]},
        {"id": 2, "image_uuid": "reg/y:tag",
         "extra_env": [["HERDD_IMAGE_DIGEST", "sha256:same"]]},
        {"id": 3, "image_uuid": "reg/z:tag"},          # never stamped
    ]
    monkeypatch.setattr(_ls_render.imageref, "image_tag_digest",
                        lambda img: {"reg/y:tag": "sha256:same"}.get(img))
    out = _ls_render._image_check_ids(ins)
    assert out == {"stale": [], "unresolved": [1]}


def test_ls_banner_names_the_missing_secret_for_unchecked_boxes(monkeypatch):
    """The loud half of the fix: an unchecked box gets its own banner, and when
    the cause is the R2 credential the banner says so instead of leaving the
    operator to guess at a network blip."""
    monkeypatch.delenv(_ls_render.imageref.R2_SECRET_ENV, raising=False)
    data = {
        "ts": 0, "instances": [
            {"id": 9, "actual_status": "running", "num_gpus": 1,
             "gpu_name": "X", "dph_total": 0.1,
             "image_uuid": "registry.example.com/train:t215-latest"}],
        "live_ids": [9], "jobs_by_box": {}, "market": {}, "stale_ids": [],
        "unchecked_image_ids": [9], "idle_secs": {}, "health": {},
    }
    txt = "\n".join(_ls_render._render_ls(data, fmt._Pal(False)))
    assert "could NOT be checked" in txt and "UNKNOWN, not" in txt
    assert _ls_render.imageref.R2_SECRET_ENV in txt


def test_an_old_cached_snapshot_without_the_key_renders_no_unchecked_banner():
    """`unchecked_image_ids` post-dates the snapshot cache; a pre-2026-08-21
    snapshot knows nothing about it and must not be read as "all checked" OR
    crash the paint."""
    data = {"ts": 0, "instances": [], "live_ids": [], "jobs_by_box": {},
            "market": {}, "stale_ids": [], "idle_secs": {}, "health": {}}
    txt = "\n".join(_ls_render._render_ls(data, fmt._Pal(False)))
    assert "could NOT be checked" not in txt


def test_gather_ls_data_shape_and_health_degradation(monkeypatch):
    """The dict the renderer, the snapshot cache and `dash-cache` all share —
    and the `try/except` that keeps a health-read failure from breaking `ls`."""
    ins = [{"id": 5, "actual_status": "running", "machine_id": 1}]
    monkeypatch.setattr(_ls_render.lifecycle, "_instances", lambda: ins)
    monkeypatch.setattr(_ls_render.view, "_fold_fleet_jobs",
                        lambda live, prog=None: {"5": []})
    monkeypatch.setattr(_ls_render, "_market_map",
                        lambda i, enabled=True, prog=None: {})
    monkeypatch.setattr(_ls_render, "_image_check_ids",
                        lambda i, prog=None: {"stale": [], "unresolved": []})
    monkeypatch.setattr(_ls_render.reap, "_idle_secs_map", lambda i, live: {})
    monkeypatch.setattr(_ls_render, "_fleet_budget_map",
                        lambda prog=None: {"by_box": {"5": {"spend_usd": 1.0,
                                                            "budget_usd": 4.0}},
                                           "total_usd": 1.0})

    def _explode(*a, **k):
        raise RuntimeError("B2 down")

    monkeypatch.setattr(_ls_render.health, "gather_fleet_health", _explode)
    data = _ls_render._gather_ls_data(no_spot=True)
    assert set(data) == {"ts", "no_spot", "instances", "live_ids",
                         "jobs_by_box", "market", "stale_ids",
                         "unchecked_image_ids", "idle_secs", "health",
                         "fleet_watch", "fleet_spend_total"}
    assert data["live_ids"] == [5] and data["health"] == {}
    assert data["fleet_watch"] == {"5": {"spend_usd": 1.0, "budget_usd": 4.0}}
    assert data["fleet_spend_total"] == 1.0


# =============================================================================
# cli/_runsets.py — the runsets/ config readers (cli-surface.json H8)
# =============================================================================
def test_runsets_here_is_tools_vast():
    """`herdd._HERE` is `tools/vast`; this module sits three directories
    deeper. Get it wrong and both readers return `{}` forever — every runset
    silently loses its `env:` defaults and its `spot:` policy."""
    # `v._HERE` survives the thinning as a real second computation: the thin
    # launcher still does `_HERE = dirname(abspath(__file__))` for its
    # `sys.path` bootstrap, from a file at a frozen path (plan §4).
    assert _runsets._HERE == _HERE_DIR == v._HERE
    assert os.path.isdir(os.path.join(_runsets._HERE, "runsets"))


def _write_runset(tmp_path, body):
    d = tmp_path / "runsets" / "demo"
    d.mkdir(parents=True)
    (d / "config.yaml").write_text(body)


def test_load_runset_config_reads_both_sections(tmp_path, monkeypatch):
    """Was `…_matches_the_flat_reader`. Its flat arm additionally steered the
    second reader with `monkeypatch.setattr(v, "_HERE", …)`, which post-6d
    rebinds a name in the launcher's namespace that no reader consults — a
    re-export is not a patch point. Deleted; the values are stated instead."""
    _write_runset(tmp_path, "spot:\n  ckpt_interval_s: 180\n"
                            "env:\n  FOO: bar\n")
    monkeypatch.setattr(_runsets, "_HERE", str(tmp_path))
    assert _runsets._load_runset_config("demo") == {
        "spot": {"ckpt_interval_s": 180}, "env": {"FOO": "bar"}}
    assert _runsets._load_runset_spot_config("demo") == {"ckpt_interval_s": 180}


def test_load_runset_config_is_advisory_on_every_failure(tmp_path, monkeypatch):
    """Absent file, absent runsets dir, unparseable body and a non-dict body all
    degrade to `{}` — a launch is never blocked by this reader."""
    monkeypatch.setattr(_runsets, "_HERE", str(tmp_path))
    assert _runsets._load_runset_config("nope") == {}
    assert _runsets._load_runset_spot_config("nope") == {}
    _write_runset(tmp_path, "spot: not-a-mapping\n")
    assert _runsets._load_runset_spot_config("demo") == {}


def test_here_is_read_at_call_time(tmp_path, monkeypatch):
    """The idiom the six existing asserts use
    (`test_runset_env_defaults.py:90,96`, `test_lifecycle.py:317,322,335,349`)
    is `monkeypatch.setattr(<module>, "_HERE", tmp)`. It only works while the
    function reads the module global instead of capturing it."""
    _write_runset(tmp_path, "spot:\n  budget_usd: 3\n")
    assert _runsets._load_runset_spot_config("demo") == {}      # real tools/vast
    monkeypatch.setattr(_runsets, "_HERE", str(tmp_path))
    assert _runsets._load_runset_spot_config("demo") == {"budget_usd": 3}


# =============================================================================
# launch/spec.py — the image gate and the four credential helpers (H3)
# =============================================================================
def test_train_fallback_image_is_the_expected_default():
    """cli-surface.json H1. The help/refusal text interpolates the constant, so
    a drift is a half-landed image flip — what `test_rehearse.py:138`'s regex
    guard catches from the outside. The two `== v.<name>` arms went at step 6d;
    the arm that was never parity — the two constants agreeing with EACH OTHER
    — is the one that catches a half-landed flip."""
    assert spec._EXPECTED_DEFAULT_IMAGE == spec._TRAIN_FALLBACK_IMAGE


def test_require_image_passes_through_and_names_the_default():
    """The `str(new.value) == str(flat.value)` arm went at step 6d (one body);
    the refusal text is pinned by the substring below, which is how every other
    consumer matches it."""
    assert spec._require_image("img:tag", "launch") == "img:tag"
    with pytest.raises(SystemExit) as new:
        spec._require_image(None, "launch")
    assert spec._EXPECTED_DEFAULT_IMAGE in str(new.value)


def test_hf_token_text_resolution_order(tmp_path, monkeypatch):
    """Was `…_matches_flat`; the explicit-argument and env branches are stated
    outright now that there is one implementation."""
    for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))            # no token files
    assert spec.hf_token_text("  explicit  ") == "explicit"
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", " envtok ")
    assert spec.hf_token_text() == "envtok"


def test_hf_login_snippet_reads_the_repo_file():
    """The depth guard with teeth: if `_TOOLS_VAST_DIR` were wrong the probe
    would miss `onstart/hf_login.sh` and fall through to the INLINE fallback —
    no error, just a second copy that stops tracking edits to the real file."""
    path = os.path.join(_HERE_DIR, "onstart", "hf_login.sh")
    assert os.path.isfile(path), "the file the snippet must prefer is missing"
    assert spec._TOOLS_VAST_DIR == _HERE_DIR
    assert spec.hf_login_snippet() == open(path).read()


def test_image_login_arg_on_every_branch(monkeypatch):
    """Was `…_matches_flat_on_every_branch`. With no credentials in the env
    every image shape must yield nothing (the sweep's real content), and with
    the signing secret only OUR registry mints a login."""
    for k in ("GITLAB_REGISTRY", "GITLAB_DEPLOY_TOKEN", "GITLAB_DEPLOY_USER",
              "REGISTRY_AUTH_SECRET"):
        monkeypatch.delenv(k, raising=False)
    for image in (None, "", "pytorch/pytorch:2.4.0",
                  "registry.gitlab.com/x/y:tag", "registry.example.com/train:t"):
        assert not spec.image_login_arg(image), image
    assert spec.image_login_arg(None, "  -u a -p b h  ") == "-u a -p b h"
    monkeypatch.setenv("REGISTRY_AUTH_SECRET", "s" * 32)
    got = spec.image_login_arg("registry.example.com/train:t")
    assert got.startswith("-u vast -p ")
    assert got.endswith(" registry.example.com")
    # the retired registry stays credential-less even WITH a deploy token set
    monkeypatch.setenv("GITLAB_DEPLOY_TOKEN", "deploy-FAKE")
    monkeypatch.setenv("GITLAB_DEPLOY_USER", "gitlab+deploy-token-1")
    assert spec.image_login_arg("registry.gitlab.com/x/y:tag") is None


def test_mask_image_login_redacts_the_secret():
    """Was `…_redacts_like_flat` over four shapes; the redaction itself is the
    property, and it is asserted on every shape that carries a secret."""
    assert spec._mask_image_login(None) is None
    for text in ("-u u -p supersecret host", "-p  supersecret host"):
        assert "supersecret" not in (spec._mask_image_login(text) or "")


def test_parse_base_gate_stdout_reads_the_two_field_line():
    """Was `…_matches_flat` over seven stdout shapes. One parser since step 6d,
    so the shapes are asserted against their values — including the two that
    have to degrade rather than raise."""
    assert spec.parse_base_gate_stdout(None) == ("", None)
    assert spec.parse_base_gate_stdout("") == ("", None)
    assert spec.parse_base_gate_stdout("base-models/x") == ("base-models/x", None)
    assert spec.parse_base_gate_stdout("base-models/x\t42") == ("base-models/x", 42)
    assert spec.parse_base_gate_stdout("  base-models/x  \t 42 ") == (
        "base-models/x", 42)
    assert spec.parse_base_gate_stdout("x\tnot-a-number") == ("x", None)
    assert spec.parse_base_gate_stdout("x\t0") == ("x", None)


# =============================================================================
# launch/spec.py — the run-metadata soft reads (`_observe`'s three)
# =============================================================================
def test_read_run_soft_never_fabricates_a_terminal(monkeypatch):
    """S2: a failed refresh with no cached events reads `unknown`, never the
    terminal a supervisor would stop babysitting on."""
    monkeypatch.setattr(spec.runmeta, "_default_runner",
                        lambda args, input=None: (1, "", "rclone down"))
    monkeypatch.setattr(spec.runmeta, "read_run",
                        lambda rid, runner=None, live_iids=(): (
                            runner(["copy", "x"]), {"status": "done",
                                                    "display_status": "done",
                                                    "n_events": 0})[1])
    view_ = spec._read_run_soft("r1")
    assert view_["_cache_stale"] is True
    assert view_["status"] == view_["display_status"] == "unknown"


def test_read_run_soft_survives_a_raising_reader(monkeypatch):
    def _boom(rid, runner=None, live_iids=()):
        raise RuntimeError("corrupt cache")

    monkeypatch.setattr(spec.runmeta, "read_run", _boom)
    out = spec._read_run_soft("r1")
    assert out["status"] == "unknown" and out["_cache_stale"] is True
    assert "corrupt cache" in out["_read_error"]


def test_last_stopping_actor_is_the_most_recent_one(monkeypatch):
    events = [{"event": "stopping", "actor": "cli:lap"},
              {"event": "resumed"},
              {"event": "stopping", "actor": "cli:desk"}]
    monkeypatch.setattr(spec, "_raw_events_soft", lambda rid: events)
    assert spec._last_stopping_actor("r1") == "cli:desk"
    cleared = events[:2]
    monkeypatch.setattr(spec, "_raw_events_soft", lambda rid: cleared)
    assert spec._last_stopping_actor("r1") is None


def test_status_marker_soft_needs_a_bucket(monkeypatch):
    monkeypatch.delenv("B2_BUCKET", raising=False)
    assert spec._status_marker_soft("r1") is None
    monkeypatch.setenv("B2_BUCKET", "bkt")
    monkeypatch.setattr(b2, "_rclone_soft", lambda args: (0, "DONE\n", ""))
    assert spec._status_marker_soft("r1") == "DONE\n"
    monkeypatch.setattr(b2, "_rclone_soft", lambda args: (1, "", "no such object"))
    assert spec._status_marker_soft("r1") is None


# =============================================================================
# core/fmt.py — _ts_age_s (boxstate.py reaches it as `herdd._ts_age_s`)
# =============================================================================
@pytest.mark.parametrize("ts,parses", [(None, False), ("", False), (17, False),
                                       ("garbage", False),
                                       ("20260816T101112000Z", True),
                                       ("20260816T101112000", True)])
def test_ts_age_s_parses_only_the_stamp_shapes(ts, parses):
    """Was `…_matches_flat` (both copies, `None`-ness plus a 5s tolerance).
    `boxstate.py` reaches this as `herdd._ts_age_s`, which is why the binding
    is asserted at the bottom of this file; the parse/degrade split is the
    behavior."""
    got = fmt._ts_age_s(ts)
    assert (got is not None) is parses, ts


# =============================================================================
# boxes/lifecycle.py — _confirm_gone (shared by supervise and job supervise)
# =============================================================================
@pytest.mark.parametrize("reply, expected", [
    ((False, None, "HTTP 404 not found"), True),      # already gone
    ((True, {"instances": None}, None), True),        # 200 for a gone box
    ((True, {}, None), True),
    ((True, {"instances": {"id": 1}}, None), False),  # still there
])
def test_confirm_gone_reads_a_destroy_the_way_flat_does(monkeypatch, reply, expected):
    monkeypatch.setattr(api, "request_soft", lambda *a, **k: reply)
    monkeypatch.setattr(lifecycle.time, "sleep", lambda s: None)
    assert lifecycle._confirm_gone(1, tries=2) is expected


# =============================================================================
# boxes/reap.py — guard's two shared helpers
# =============================================================================
# `test_guard_evidence_bits_matches_flat` swept five evidence dicts through both
# copies. One copy since step 6d; `test_guard.py` owns the rendered bits.


def test_guard_fix_plan_is_the_graded_reaper_policy():
    """Same rows in, same `(action, why)` out as the flat helper — which is the
    only thing keeping `guard --fix` and the automatic reaper from grading a
    zombie differently (the 46682313 destroy)."""
    rows = [{"iid": 9, "verdict": "ZOMBIE_LOADING_STALL", "evidence": {}},
            {"iid": 10, "verdict": "ZOMBIE_PYHALF",
             "evidence": {"is_jobs_box": False}}]
    by_iid = {"9": {"label": "run:R"}, "10": {"label": "keep:why"}}
    new = reap._guard_fix_plan(rows, by_iid)
    # The `flat = v._guard_fix_plan(...)` arm went at step 6d — one plan
    # function, so `guard --fix` and the automatic reaper cannot grade
    # differently by construction; `test_guard.py` grades the policy itself.
    assert len(new) == len(rows)
    assert all(isinstance(a, str) and isinstance(w, str) for _h, a, w in new)


# =============================================================================
# jobs/view.py — the `runs` status fold (storage.json's refusal)
# =============================================================================
# `test_parse_farm_status_matches_flat` swept six status lines through both
# copies. One copy since step 6d; the fold's own tests below drive it through
# `_farm_status_by_run`, which is how the CLI reaches it.


def _farm_runner(script):
    def run(args):
        return script.get(tuple(args[:2]) if len(args) > 1 else tuple(args),
                          (1, "", "unexpected"))
    return run


def test_farm_status_by_run_gates_on_one_listing():
    calls = []

    def runner(args):
        calls.append(list(args))
        if args[0] == "lsf":
            return (0, "r1/\nr2/\n", "")
        return (0, "RUNNING 2026\n", "")

    assert view._farm_status_by_run("b2:x", ["r1", "zz"], runner=runner) == {"r1": "RUNNING"}
    assert calls[0][0] == "lsf"                    # one gating listing
    assert len(calls) == 2                         # only the MATCHING run is cat'd

    def dead(args):
        return (1, "", "rclone down")

    assert view._farm_status_by_run("b2:x", ["r1"], runner=dead) == {}


def test_ckpt_steps_by_run_reads_depth_four_and_summaries():
    listing = ("r1/checkpoint-100/\n"
               "r1/adapter/checkpoint-250/\n"
               "r2/arms/a/checkpoint-7/\n"
               "r2/arms/a/train_summary.json\n"
               "r3/\n")

    def runner(args):
        return (0, listing, "")

    out = view._ckpt_steps_by_run("b2:x", runner=runner)
    assert out["r1"]["step"] == 250                 # max across layouts
    assert out["r2"]["step"] == 7
    assert out["r2"]["summaries"] == ["b2:x/checkpoints/r2/arms/a/train_summary.json"]
    assert "r3" not in out                          # never mint an empty slot


def test_train_summary_step_is_capped_and_best_effort():
    seen = []

    def runner(args):
        seen.append(args[1])
        return (0, '{"global_steps": %d}' % len(seen), "")

    assert view._train_summary_step([f"p{i}" for i in range(9)], runner=runner) == 4
    assert len(seen) == view._MAX_SUMMARY_READS
    assert view._train_summary_step(["p"], runner=lambda a: (0, "not json", "")) is None


def test_runs_fold_runner_defaults_bind_at_def_time(monkeypatch):
    """PORTED TRAP, asserted so it is not "fixed" into a live call: the three
    folds take `runner=b2._rclone_soft` as a DEFAULT ARG, so a later
    `monkeypatch.setattr(b2, "_rclone_soft", …)` does NOT steer them — exactly
    as patching `herdd._rclone_soft` did not steer the flat copies. A caller
    that means to redirect the transport passes `runner=` explicitly."""
    import inspect
    for fn in (view._farm_status_by_run, view._ckpt_steps_by_run,
               view._train_summary_step):
        assert inspect.signature(fn).parameters["runner"].default is b2._rclone_soft
    monkeypatch.setattr(b2, "_rclone_soft", lambda args: (0, "PATCHED\n", ""))
    assert inspect.signature(view._ckpt_steps_by_run).parameters[
        "runner"].default is not b2._rclone_soft


# =============================================================================
# jobs/bundle.py — the SYNC-side ship set + its import gate
# =============================================================================
def test_sync_file_list_resolves_a_pathspec_and_refuses_an_empty_match():
    root = os.path.dirname(os.path.dirname(_HERE_DIR))
    paths = ["tools/vast/herdd.py"]
    assert bundle._sync_file_list(root, paths) == ["tools/vast/herdd.py"]
    with pytest.raises(SystemExit):
        bundle._sync_file_list(root, ["tools/vast/no-such-file-xyz"])


def _fake_shipcheck(gaps):
    m = types.ModuleType("shipcheck")
    m.import_closure_gaps = lambda root: gaps
    m.format_import_gaps = lambda g, *a: ["closure ok"] if not g else ["gap: x"]
    return m


def test_sync_import_gate_is_fail_closed(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "shipcheck", _fake_shipcheck({"x": ["y"]}))
    with pytest.raises(SystemExit) as e:
        bundle._sync_import_gate("/repo")
    assert "not import-closed" in str(e.value)
    bundle._sync_import_gate("/repo", warn_only=True)      # escape hatch
    assert "syncing anyway" in capsys.readouterr().err


def test_sync_import_gate_degrades_to_a_note(monkeypatch, capsys):
    """A guard must never be the reason a sync cannot run."""
    broken = types.ModuleType("shipcheck")

    def _boom(root):
        raise RuntimeError("shipcheck internals moved")

    broken.import_closure_gaps = _boom
    monkeypatch.setitem(sys.modules, "shipcheck", broken)
    bundle._sync_import_gate("/repo")                      # no raise
    assert "import-closure check skipped" in capsys.readouterr().err


def test_sync_import_gate_prints_the_clean_line(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "shipcheck", _fake_shipcheck({}))
    bundle._sync_import_gate("/repo")
    assert capsys.readouterr().out.strip() == "closure ok"


# =============================================================================
# fleet/client.py — the child `herdd supervise` argv
# =============================================================================
def test_supervise_argv_reexecutes_herdd():
    """The depth guard that matters most in this file: a wrong `__file__` walk
    forks `python3 .../client.py supervise …`, which has no `main()` — it exits
    immediately, the parent only checked that Popen succeeded, and the run is
    unsupervised with nothing said about it."""
    assert client._HERDD_SCRIPT == os.path.join(_HERE_DIR, "herdd.py")
    assert os.path.isfile(client._HERDD_SCRIPT)


@pytest.mark.parametrize("over", [
    {},
    {"strict_ceiling": True},
    {"handoff": False},
    {"wall_budget": 2.0, "boot_health": True, "boot_sla": False},
])
def test_supervise_argv_carries_the_flags_and_the_frozen_script_path(over):
    import argparse
    base = dict(strict_ceiling=False, handoff=True, wall_budget=None,
                boot_health=False, boot_sla=True)
    base.update(over)
    a = argparse.Namespace(**base)
    got = client._supervise_argv(a, "R1", 12.0, 0.4, 0.9, 300)
    assert got[1] == os.path.join(_HERE_DIR, "herdd.py")
    if over.get("wall_budget"):
        assert "--wall-budget" in got and str(2.0 * 3600.0) in got


# =============================================================================
# cli/{debug,metrics,shipcheck}.py — three more `__file__` anchors, same class
# of silent failure as the four above (a wrong depth is "script not found" /
# "probe not found" / an unresolvable bare-name import, always against a box
# the operator is already trying to diagnose). Imported function-locally so
# this block is a pure append to the shared module's import list.
# =============================================================================
def test_the_three_command_anchors_are_tools_vast():
    from vastlib.cli import debug as cli_debug
    from vastlib.cli import metrics as cli_metrics
    from vastlib.cli import shipcheck as cli_shipcheck

    flat_here = os.path.dirname(os.path.abspath(v.__file__))
    assert flat_here == _HERE_DIR
    for mod in (cli_debug, cli_metrics, cli_shipcheck):
        assert mod._TOOLS_VAST_DIR == flat_here, mod.__name__
    assert os.path.isfile(os.path.join(cli_debug._TOOLS_VAST_DIR, "debug_box.sh"))
    assert os.path.isfile(cli_metrics._metrics_probe_path())


# =============================================================================
# cli/sync.py — the ship-manifest pathspec parser (its two ship-set siblings
# live in jobs/bundle.py and are covered above; this one stayed in `cli/`).
# =============================================================================
def test_load_ship_manifest_parses_the_pathspecs():
    from vastlib.cli import sync as cli_sync

    root = os.path.dirname(os.path.dirname(_HERE_DIR))
    specs = cli_sync._load_ship_manifest(root)
    assert specs and all(isinstance(x, str) for x in specs)
    assert any(not s.startswith(":(exclude)") for s in specs)


def test_load_ship_manifest_refuses_a_manifest_with_no_includes(tmp_path):
    """Hard-error, not an empty rsync: a manifest of pure exclusions would make
    `git ls-files` match the WHOLE repo, which is the opposite of an allowlist."""
    from vastlib.cli import sync as cli_sync

    mdir = tmp_path / "tools" / "vast"
    mdir.mkdir(parents=True)
    (mdir / "ship_manifest.txt").write_text("# only comments\n\n!tools/vast/secret\n")
    with pytest.raises(SystemExit) as e:
        cli_sync._load_ship_manifest(str(tmp_path))
    assert "no include pathspecs" in str(e.value)

    with pytest.raises(SystemExit) as missing:
        cli_sync._load_ship_manifest(str(tmp_path / "nowhere"))
    assert "can't read ship manifest" in str(missing.value)


# =============================================================================
# the launcher's bindings — what the deleted differential half leaves behind
# =============================================================================
def test_the_launcher_re_exports_rather_than_redefines():
    """Every name this file used to drive twice, asserted as ONE object.

    These are not internal: `boxstate.py` reads `herdd._ts_age_s`,
    `hosts.py` / `hostfacts.py` / `bid_echo_probe.py` / `parked_lifecycle.py`
    address the flat module (workflowctl.py stopped at its step-7 shim, and
    shipcheck.py now path-loads vastlib/jobs/bundle.py, falling back here only
    on a pre-package checkout), and `launch_serve.sh` imports it in a heredoc.
    A second body under
    any of these names is a second answer reaching a real consumer — which is
    what the differential tests deleted above were watching for while there
    were legitimately two."""
    for name, home in (
            ("_ACTIVE_JOB_STATES", _ls_render), ("_render_ls", _ls_render),
            ("_render_minimal", _ls_render),
            ("_load_runset_config", _runsets),
            ("_load_runset_spot_config", _runsets),
            ("_EXPECTED_DEFAULT_IMAGE", spec), ("_TRAIN_FALLBACK_IMAGE", spec),
            ("_last_stopping_actor", spec), ("_mask_image_login", spec),
            ("_require_image", spec), ("_raw_events_soft", spec),
            ("hf_login_snippet", spec), ("hf_token_text", spec),
            ("image_login_arg", spec), ("parse_base_gate_stdout", spec),
            ("_ts_age_s", fmt),
            ("_guard_evidence_bits", reap), ("_guard_fix_plan", reap),
            ("_parse_farm_status", view), ("_sync_file_list", bundle),
            ("_supervise_argv", client)):
        assert getattr(v, name) is getattr(home, name), (
            f"herdd.{name} is a second body again — the launcher must "
            f"re-export {home.__name__}'s object, never redefine it")

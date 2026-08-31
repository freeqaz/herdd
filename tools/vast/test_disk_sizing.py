"""Portable tests for disk allocation/usage reading and the oversizing signal.

Runs in the toolchain-free lane (no vast API, no B2, no network). Covers the
inputs a config-derived `--disk` estimator will be calibrated against:
  * `_disk_gb` — allocated vs actually-used GB, and the NEGATIVE SENTINEL a
    still-booting box reports (`disk_usage: -1`), which must read as unknown.
  * `_disk_frac` — the used/allocated oversizing fraction.
  * the `ls --minimal` TSV carrying both columns for agent consumers.

Why this exists: storage bills on the ALLOCATED disk, `ls` reported only the
dollar cost, and `disk_usage` — present in every instances payload — was
referenced nowhere in the repo. A 2026-07-21 audit found a 160 GB allocation
billing $4.62/day while using 18 GB (8.9x oversized) and it was invisible.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import herdd  # noqa: E402


def test_disk_gb_reads_both_fields():
    assert herdd._disk_gb({"disk_space": 160.0, "disk_usage": 17.0}) == (160.0, 17.0)


def test_disk_gb_negative_usage_is_unknown_not_zero():
    """A box still `loading` reports `disk_usage: -1` until its container is
    provisioned (observed live, box 46246859). Passing that through reads as
    "uses -1 of 120 GB" — a 100%-wasted allocation — so every booting box would
    trip an oversizing warning and any average over the field would be poisoned.
    """
    alloc, used = herdd._disk_gb({"disk_space": 120.0, "disk_usage": -1})
    assert alloc == 120.0
    assert used is None
    assert herdd._disk_frac({"disk_space": 120.0, "disk_usage": -1}) is None


def test_disk_gb_negative_allocation_is_unknown():
    assert herdd._disk_gb({"disk_space": -1, "disk_usage": -1}) == (None, None)


def test_disk_gb_missing_fields():
    assert herdd._disk_gb({}) == (None, None)
    assert herdd._disk_gb({"disk_space": 120.0}) == (120.0, None)
    assert herdd._disk_gb({"disk_usage": 17.0}) == (None, 17.0)


def test_disk_gb_tolerates_string_values():
    # every other numeric field off this API is defensively coerced; match it.
    assert herdd._disk_gb({"disk_space": "160", "disk_usage": "17"}) == (160.0, 17.0)


def test_disk_frac_is_used_over_allocated():
    assert herdd._disk_frac({"disk_space": 160.0, "disk_usage": 16.0}) == 0.10
    assert herdd._disk_frac({"disk_space": 100.0, "disk_usage": 100.0}) == 1.0


def test_disk_frac_zero_allocation_does_not_divide_by_zero():
    assert herdd._disk_frac({"disk_space": 0, "disk_usage": 0}) is None


def test_minimal_tsv_exposes_disk_columns():
    """`--minimal` is the agent-facing contract: a stable, color-free TSV whose
    header documents the columns. Both disk figures must be bare parseable GB."""
    data = {
        "instances": [{
            "id": 4242, "actual_status": "running", "num_gpus": 1,
            "gpu_name": "RTX 5090", "is_bid": True,
            "disk_space": 160.0, "disk_usage": 17.0,
            "label": "run:probe",
        }],
        "live_ids": [4242],
    }
    out = herdd._render_minimal(data)
    header, row = out.splitlines()[0].split("\t"), out.splitlines()[1].split("\t")
    cells = dict(zip(header, row))
    assert cells["disk_gb"] == "160.0"
    assert cells["disk_used_gb"] == "17.0"


def test_minimal_tsv_disk_cells_empty_when_unknown():
    """An omitted or sentinel field must render as an empty cell (the documented
    N/A), never as `-1.0` or `0.0` — a consumer must not read a booting box as
    fully-wasted."""
    data = {
        "instances": [{
            "id": 4243, "actual_status": "loading", "num_gpus": 2,
            "gpu_name": "RTX 5090", "disk_space": 120.0, "disk_usage": -1,
        }],
        "live_ids": [],
    }
    out = herdd._render_minimal(data)
    cells = dict(zip(out.splitlines()[0].split("\t"),
                     out.splitlines()[1].split("\t")))
    assert cells["disk_gb"] == "120.0"
    assert cells["disk_used_gb"] == ""


def test_parse_base_gate_stdout_new_format():
    assert herdd.parse_base_gate_stdout(
        "base-models/qwen25-coder-7b\t15300000000") == (
            "base-models/qwen25-coder-7b", 15300000000)


def test_parse_base_gate_stdout_tolerates_old_subpath_only_script():
    """`--print-bytes` ships from the launcher, but the script can lag on a
    stale checkout and print only the subpath. That must still parse."""
    assert herdd.parse_base_gate_stdout("base-models/x") == ("base-models/x", None)


def test_parse_base_gate_stdout_unmeasurable_size_is_none_not_zero():
    """A caller sizing a disk must distinguish "could not measure" from "empty";
    0 would silently under-size every box."""
    for out in ("base-models/x\t", "base-models/x\tgarbage", "base-models/x\t0"):
        sub, n = herdd.parse_base_gate_stdout(out)
        assert sub == "base-models/x"
        assert n is None


def test_parse_base_gate_stdout_empty_and_whitespace():
    assert herdd.parse_base_gate_stdout("") == ("", None)
    assert herdd.parse_base_gate_stdout(None) == ("", None)
    assert herdd.parse_base_gate_stdout("  base-models/x  \t 42 ") == (
        "base-models/x", 42)


def test_minimal_tsv_header_and_row_widths_match():
    """Guards the column list against an edit that adds a header without a cell
    (or vice versa) — the failure mode that silently shifts every agent's
    column indices."""
    data = {"instances": [{"id": 1, "actual_status": "exited"}], "live_ids": []}
    lines = herdd._render_minimal(data).splitlines()
    assert len(lines[0].split("\t")) == len(lines[1].split("\t"))


# --- default_disk_gb: the per-user override (velvet P4b) --------------------- #
# The constants encode one workstation's habits and there was no way to change
# them short of editing the tree — unlike train_gpu_ram, which exists for
# exactly this purpose. These pin the precedence so a personal default cannot be
# silently ignored (or silently win over an explicit flag).

# Repointed to the ported module (plan §8 step 2): `default_disk_gb` and the
# DISK_DEFAULT_* constants now live in vastlib.core.config. Import/patch
# plumbing only — every expectation below is unchanged (plan §7.4).
from vastlib.cli import _runsets  # noqa: E402
from vastlib.core import config  # noqa: E402


def test_kinds_fall_back_to_their_own_constants(monkeypatch):
    monkeypatch.delenv("HERDD_DEFAULT_DISK", raising=False)
    monkeypatch.setenv("HERDD_CONFIG", "/nonexistent/herdd.yaml")
    assert config.default_disk_gb("launch") == config.DISK_DEFAULT_LAUNCH_GB
    assert config.default_disk_gb("train") == config.DISK_DEFAULT_TRAIN_GB
    assert config.default_disk_gb("serve") == config.DISK_DEFAULT_SERVE_GB


def test_an_explicit_cli_value_beats_every_other_source(monkeypatch):
    """The flag is the operator saying a number out loud. Nothing outranks it."""
    monkeypatch.setenv("HERDD_DEFAULT_DISK", "999")
    assert config.default_disk_gb("train", cli=25) == 25


def test_env_overrides_the_constant_and_per_kind_wins(monkeypatch):
    monkeypatch.setenv("HERDD_CONFIG", "/nonexistent/herdd.yaml")
    monkeypatch.setenv("HERDD_DEFAULT_DISK", "55")
    assert config.default_disk_gb("launch") == 55
    monkeypatch.setenv("HERDD_DEFAULT_DISK_TRAIN", "90")
    assert config.default_disk_gb("train") == 90
    assert config.default_disk_gb("launch") == 55, "per-kind must not leak"


def test_yaml_default_disk_is_read(monkeypatch, tmp_path):
    y = tmp_path / "herdd.yaml"
    y.write_text("default_disk: 35\ndefault_disk_train: 150\n")
    monkeypatch.delenv("HERDD_DEFAULT_DISK", raising=False)
    monkeypatch.delenv("HERDD_DEFAULT_DISK_TRAIN", raising=False)
    monkeypatch.setenv("HERDD_CONFIG", str(y))
    assert config.default_disk_gb("supervise") == 35
    assert config.default_disk_gb("train") == 150


def test_a_malformed_override_is_skipped_not_fatal(monkeypatch):
    """A typo in a personal config must not make every launch path unusable."""
    monkeypatch.setenv("HERDD_CONFIG", "/nonexistent/herdd.yaml")
    for bad in ("", "lots", "0", "-10"):
        monkeypatch.setenv("HERDD_DEFAULT_DISK", bad)
        assert config.default_disk_gb("launch") == config.DISK_DEFAULT_LAUNCH_GB


def test_an_unknown_kind_is_a_programming_error_not_a_silent_40(monkeypatch):
    import pytest as _pytest
    with _pytest.raises(KeyError):
        config.default_disk_gb("nosuchkind")


# --- the runset `disk:` key (velvet P4b) ------------------------------------ #
# Runsets could already declare budget_usd / max_bid_mult / ckpt_interval_s but
# NOT disk, so seven runset READMEs duplicated the number in prose
# (60/80/120/200) where nothing reads it and nothing keeps it true.

def _resolve_train_disk(tmp_path, monkeypatch, *, cli, yaml_text=None):
    """Exercise cmd_train's step-1.6 resolution without launching anything."""
    rs = tmp_path / "runsets" / "rs1"
    rs.mkdir(parents=True, exist_ok=True)
    if yaml_text is not None:
        (rs / "config.yaml").write_text(yaml_text)
    # `_HERE` is read from the runset reader's OWN globals (`cli._runsets`)
    # since plan §8 step 6d; patching the `herdd` re-export steers nothing and
    # the read below would fall through to the repo's real `runsets/` tree.
    monkeypatch.setattr(_runsets, "_HERE", str(tmp_path))
    monkeypatch.setenv("HERDD_CONFIG", "/nonexistent/herdd.yaml")
    monkeypatch.delenv("HERDD_DEFAULT_DISK", raising=False)
    monkeypatch.delenv("HERDD_DEFAULT_DISK_TRAIN", raising=False)
    cfg = herdd._load_runset_config("rs1")
    if cli is not None:
        return cli
    raw = cfg.get("disk")
    try:
        raw = int(float(raw)) if raw not in (None, "") else None
    except (TypeError, ValueError):
        raw = None
    return raw if (raw and raw > 0) else config.default_disk_gb("train")


def test_runset_disk_key_is_read(tmp_path, monkeypatch):
    assert _resolve_train_disk(tmp_path, monkeypatch, cli=None,
                               yaml_text="disk: 80\n") == 80


def test_an_explicit_disk_flag_beats_the_runset(tmp_path, monkeypatch):
    assert _resolve_train_disk(tmp_path, monkeypatch, cli=45,
                               yaml_text="disk: 80\n") == 45


def test_no_runset_key_falls_back_to_the_train_default(tmp_path, monkeypatch):
    assert _resolve_train_disk(tmp_path, monkeypatch, cli=None,
                               yaml_text="spot:\n  budget_usd: 5\n") == \
        config.DISK_DEFAULT_TRAIN_GB


def test_a_malformed_runset_disk_is_ignored_not_fatal(tmp_path, monkeypatch):
    """A typo in one runset must not make that runset unlaunchable."""
    for bad in ("disk: lots\n", "disk: 0\n", "disk: -5\n"):
        assert _resolve_train_disk(tmp_path, monkeypatch, cli=None,
                                   yaml_text=bad) == config.DISK_DEFAULT_TRAIN_GB


# --- forced-rehost sizing: `replacement_disk_gb` (task #69) ------------------ #
# The driftr3 H200 lane launched `--disk 110` (sized for a 56.8 GB two-stage
# merge transient) and the workload ended up on a 60 GB box, which then died on
# the bundle's own disk guard with rc 5. The sizing must be a property of the
# WORKLOAD, carried forward, not re-derived from whichever box last held it —
# so this helper takes the launch-time allocation as a first-class term.
import disksize  # noqa: E402


def test_replacement_disk_floors_at_the_launch_allocation():
    """The driftr3 shape: a hop already shrank the box to 60 GB and its usage
    snapshot says 25 GB. Both are evidence about the WRONG box — the job was
    sized at 110 GB and 110 GB is what the next one gets."""
    gb, why = disksize.replacement_disk_gb(launch_gb=110.0, allocated_gb=60.0,
                                           used_gb=25.0)
    assert gb == 110.0
    assert "110" in why and "launch" in why.lower()


def test_replacement_disk_keeps_the_current_allocation_when_it_is_larger():
    """A watch armed on a box bigger than its launch anchor (a hop that grew)
    must not be shrunk back to the anchor: max, never overwrite."""
    gb, _why = disksize.replacement_disk_gb(launch_gb=60.0, allocated_gb=160.0,
                                            used_gb=17.0)
    assert gb == 160.0


def test_replacement_disk_grows_for_a_nearly_full_box():
    """`max(allocated, used + margin)`: a box 92% full is evidence the
    allocation was too small, and the replacement is the only chance to fix it
    (a running instance's disk cannot be resized)."""
    gb, why = disksize.replacement_disk_gb(launch_gb=60.0, allocated_gb=60.0,
                                           used_gb=55.0)
    assert gb == 90.0                       # 55 x 1.4 + 12 = 89 -> round up 90
    assert "usage" in why


def test_replacement_disk_never_shrinks_on_a_usage_snapshot():
    """R7, restated as a property of the helper rather than of one call site:
    box 46914272 was evicted ~2 min into boot holding 5.7 of 50 GB, and
    `used x 1.4 + 12` sized its replacement at 20 GB. On a forced rehost the
    usage snapshot measures how far the restage got, not what the job needs."""
    gb, _why = disksize.replacement_disk_gb(launch_gb=None, allocated_gb=50.0,
                                            used_gb=5.7)
    assert gb == 50.0


def test_replacement_disk_is_unknown_when_nothing_is_readable():
    """The evicted-primary case: the box left the listing, so there is no
    allocation and no usage to read. That is a DECLARED unknown the caller has
    to warn about — never a silent fall back to a launch default."""
    gb, why = disksize.replacement_disk_gb(launch_gb=None, allocated_gb=None,
                                           used_gb=None)
    assert gb is None
    assert "no" in why.lower()


def test_replacement_disk_survives_a_vanished_primary_when_anchored():
    """...and with the anchor, the same vanished primary sizes correctly. This
    is the whole point of storing the sizing on the watch instead of the box."""
    gb, why = disksize.replacement_disk_gb(launch_gb=110.0, allocated_gb=None,
                                           used_gb=None)
    assert gb == 110.0 and "launch" in why.lower()


def test_replacement_disk_treats_sentinels_as_unknown_not_zero():
    """A booting box reports `disk_usage: -1`; `_disk_gb` already maps that to
    None, but a negative/garbage term arriving any other way must not win a
    max() against a real figure either."""
    for bad in (-1, 0, None, "", "lots"):
        gb, _why = disksize.replacement_disk_gb(launch_gb=110.0,
                                                allocated_gb=bad, used_gb=bad)
        assert gb == 110.0
    assert disksize.replacement_disk_gb(launch_gb=-1, allocated_gb=-1,
                                        used_gb=-1)[0] is None

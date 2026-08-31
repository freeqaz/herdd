"""`vastlib.core.config` — the ported knob/config resolver (plan §8 step 2).

Why this file exists
--------------------
`vastconf.py` moved into the package as `vastlib.core.config`, and the port's
whole claim is that it is BEHAVIOR-PRESERVING (plan §7.4). Two things have to be
true for that claim to hold, and neither was covered by an existing test:

1. **The precedence lattice still bites.** CLI > env > `herdd.yaml` >
   constant, per knob family, including the deliberate asymmetries (a malformed
   value is skipped, not fatal; the fail-closed adopt ceiling never returns a
   non-positive number). The pre-port coverage was scattered across
   `test_disk_sizing.py` (disk defaults), `test_boot_health.py` (`_boot_knob`,
   through the `herdd` facade), `test_fleetd_ceiling.py` (adopt ceiling) and a
   subprocess in `test_joblocal.py` (the local-GPU gate) — nothing exercised the
   module as a unit, so a port could have silently dropped a rung.
2. **`_HERE` still points at `tools/vast`.** It is the one line of the port that
   is not textually verbatim: the module sits two directories deeper, so the
   `dirname` chain grew. Get it wrong and `load_herdd_config()` silently reads
   NO repo defaults — every knob keeps working, on the built-in constants,
   which is the failure mode that looks green.

The section at the bottom used to be DIFFERENTIAL — the port compared against
the still-live flat module, symbol by symbol. Step 7 arrived and `vastconf.py`
became a re-export shim, so those comparisons became self-comparisons. The
prediction recorded here ("they degrade into a tautology, which is fine") was
half right and half wrong, and the wrong half is worth stating: two of them
derived their expectation by AST-parsing vastconf.py, which a shim empties, so
they degraded not into tautology but into asserting `[] == []` — green with the
port's every symbol and every marker deleted. The section was rewritten at step
7 against a FROZEN name list, plus `is` identity (the invariant a re-export
actually creates) and the literal constant values. See its header comment.

What is deliberately NOT here
-----------------------------
* No network, no B2, no vast API, no fleetd socket — this module touches none of
  them, and the suite's conftest guards would be doing the work if it did.
* No assertion about the CONTENT of the shipped `herdd.yaml`. `allow_local_gpu`
  is an owner ruling that may flip; a test that pins its value would make the
  ruling harder to exercise than editing the key. The tests below always pass an
  explicit `cfg` or redirect `HERDD_CONFIG` to a file they wrote.
* No coverage of the ~70 (measured: 105) stray `os.environ.get` sites that were
  in `herdd.py`. They are inventoried in `config.ENV_SITES_TODO` and route
  through config only when their owning function ports — which has now
  happened for the reads themselves (plan §8 step 6d emptied `herdd.py`;
  `grep -c os.environ.get herdd.py` is 0), while the ROUTING those rows
  describe is still open and still owned by plan §9. The inventory's
  "in herdd.py" wording is pre-6d; the rows are the authority, not the file
  they name.

Provenance: created 2026-08-16 alongside the port, step 2 of
docs/plans/vast-tooling-refactor-v2.md §8.
"""

from __future__ import annotations

import ast
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vastconf  # noqa: E402
from vastlib.core import config  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

# Every env var the module reads, so a stray one in the developer's shell cannot
# steer a test. `HERDD_CONFIG` is set per-test where it matters.
_ENV_KEYS = (
    "HERDD_CONFIG",
    config.LOCAL_GPU_ENV,
    config.JOBS_HANDOFF_UNSAFE_ENV,
    config.FLEETD_ADOPT_BUDGET_ENV,
    "HERDD_DEFAULT_DISK",
    "HERDD_DEFAULT_DISK_LAUNCH",
    "HERDD_DEFAULT_DISK_TRAIN",
    "HERDD_DEFAULT_DISK_SUPERVISE",
    "BOOT_MIN_MBPS",
    "BOOT_SLA_S",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in _ENV_KEYS:
        monkeypatch.delenv(k, raising=False)


@pytest.fixture
def no_yaml(monkeypatch):
    """Every yaml rung empty: the resolvers fall through to their constants."""
    monkeypatch.setattr(config, "load_herdd_config", lambda: {})
    return None


# --------------------------------------------------------------------------- #
# _HERE — the one line of the port that is not textually verbatim
# --------------------------------------------------------------------------- #
def test_repo_config_still_resolves_to_tools_vast_herdd_yaml():
    """The module moved two directories deeper; the config path must not.

    A wrong `_HERE` does not raise — `_load_yaml_file` returns {} for a missing
    path — so every knob would quietly resolve on its built-in constant and the
    shipped `herdd.yaml` (including `allow_local_gpu`) would stop being read.
    """
    assert config._HERE == HERE
    assert config._REPO_CONFIG == os.path.join(HERE, "herdd.yaml")
    assert os.path.isfile(config._REPO_CONFIG), "the committed defaults file"


def test_the_repo_defaults_are_actually_loaded():
    """Not "a dict came back" — the committed file's own keys came back."""
    on_disk = config._load_yaml_file(os.path.join(HERE, "herdd.yaml"))
    assert on_disk, "herdd.yaml parsed empty; the rest of this file is vacuous"
    merged = config.load_herdd_config()
    for k in on_disk:
        assert k in merged


# --------------------------------------------------------------------------- #
# load_env
# --------------------------------------------------------------------------- #
def test_load_env_reads_the_nearest_dotenv_and_never_clobbers(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text(
        "# a comment\n"
        "\n"
        "PLAIN=value\n"
        'QUOTED="quoted value"\n'
        "SQUOTED='sq'\n"
        "TRAILING=keepme # inline comment\n"
        "  SPACED  =  spaced  \n"
        "NOEQUALS\n"
        "ALREADY=from-dotenv\n"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ALREADY", "from-environ")
    for k in ("PLAIN", "QUOTED", "SQUOTED", "TRAILING", "SPACED", "NOEQUALS"):
        monkeypatch.delenv(k, raising=False)

    config.load_env()

    assert os.environ["PLAIN"] == "value"
    assert os.environ["QUOTED"] == "quoted value"
    assert os.environ["SQUOTED"] == "sq"
    assert os.environ["TRAILING"] == "keepme"
    assert os.environ["SPACED"] == "spaced"
    assert "NOEQUALS" not in os.environ
    assert os.environ["ALREADY"] == "from-environ", "an existing value wins"


def test_load_env_walks_up_and_stops_at_the_first_hit(tmp_path, monkeypatch):
    deep = tmp_path / "a" / "b"
    deep.mkdir(parents=True)
    (tmp_path / ".env").write_text("FROM_ROOT=1\n")
    monkeypatch.chdir(deep)
    monkeypatch.delenv("FROM_ROOT", raising=False)
    config.load_env()
    assert os.environ["FROM_ROOT"] == "1"

    (deep / ".env").write_text("FROM_ROOT=2\nFROM_DEEP=1\n")
    monkeypatch.delenv("FROM_ROOT", raising=False)
    monkeypatch.delenv("FROM_DEEP", raising=False)
    config.load_env()
    assert os.environ["FROM_DEEP"] == "1"
    assert os.environ["FROM_ROOT"] == "2", "the NEAREST .env is the one that is read"


# --------------------------------------------------------------------------- #
# yaml loading + merge order
# --------------------------------------------------------------------------- #
def test_parse_simple_yaml_flat_subset():
    out = config._parse_simple_yaml(
        "# leading comment\n"
        "a: 1\n"
        'b: "two"\n'
        "c: three # trailing\n"
        "d:\n"
        "e: null\n"
        "f: ~\n"
        "nocolon\n"
    )
    assert out == {"a": "1", "b": "two", "c": "three", "d": None, "e": None, "f": None}
    assert "nocolon" not in out


def test_load_yaml_file_missing_and_non_mapping(tmp_path):
    assert config._load_yaml_file(str(tmp_path / "nope.yaml")) == {}
    lst = tmp_path / "list.yaml"
    lst.write_text("- one\n- two\n")
    assert config._load_yaml_file(str(lst)) == {}, "a non-mapping document is not config"


def test_config_merge_order_repo_then_user_then_explicit(tmp_path, monkeypatch):
    repo = tmp_path / "repo.yaml"
    user = tmp_path / "user.yaml"
    extra = tmp_path / "extra.yaml"
    repo.write_text("shared: repo\nonly_repo: 1\n")
    user.write_text("shared: user\nonly_user: 2\n")
    extra.write_text("shared: extra\n")
    monkeypatch.setattr(config, "_REPO_CONFIG", str(repo))
    monkeypatch.setattr(config, "_USER_CONFIG", str(user))

    cfg = config.load_herdd_config()
    assert cfg["shared"] == "user", "the personal override wins over repo defaults"
    assert cfg["only_repo"] == 1 and cfg["only_user"] == 2, "merge is per-key"

    monkeypatch.setenv("HERDD_CONFIG", str(extra))
    assert config.load_herdd_config()["shared"] == "extra", "explicit path wins"
    assert config.load_herdd_config()["only_repo"] == 1, "...still per-key"


# --------------------------------------------------------------------------- #
# the local-GPU gate  (the only in-process coverage of it; test_joblocal.py
# drives the same gate end-to-end through a subprocess)
# --------------------------------------------------------------------------- #
def test_local_gpu_default_is_closed_when_the_config_says_nothing():
    assert config.local_gpu_allowed({}) is False


@pytest.mark.parametrize("raw,expect", [
    ("1", True), ("true", True), ("TRUE", True), (" yes ", True), ("on", True),
    ("0", False), ("false", False), ("", False), ("banana", False),
])
def test_local_gpu_env_wins_in_both_directions(monkeypatch, raw, expect):
    monkeypatch.setenv(config.LOCAL_GPU_ENV, raw)
    assert config.local_gpu_allowed({config.LOCAL_GPU_KEY: True}) is expect
    assert config.local_gpu_allowed({config.LOCAL_GPU_KEY: False}) is expect


def test_local_gpu_config_authorizes_when_the_env_is_absent():
    assert config.local_gpu_allowed({config.LOCAL_GPU_KEY: True}) is True
    assert config.local_gpu_allowed({config.LOCAL_GPU_KEY: "yes"}) is True
    assert config.local_gpu_allowed({config.LOCAL_GPU_KEY: "no"}) is False


def test_require_local_gpu_returns_quietly_when_authorized():
    assert config.require_local_gpu("lane", {config.LOCAL_GPU_KEY: True}) is None


def test_require_local_gpu_refuses_with_the_key_and_the_env_in_the_message():
    with pytest.raises(SystemExit) as e:
        config.require_local_gpu("herdd job run-local", {})
    msg = str(e.value)
    assert "herdd job run-local" in msg
    assert config.LOCAL_GPU_KEY in msg and config.LOCAL_GPU_ENV in msg
    assert "rehearse.sh" in msg, "the refusal names the ungated CPU-only path"


# --------------------------------------------------------------------------- #
# jobs-handoff safe-off switch
# --------------------------------------------------------------------------- #
def test_jobs_handoff_is_off_by_default_and_env_wins(monkeypatch):
    assert config.jobs_handoff_enabled({}) is False
    assert config.jobs_handoff_enabled({config.JOBS_HANDOFF_UNSAFE_KEY: True}) is True
    monkeypatch.setenv(config.JOBS_HANDOFF_UNSAFE_ENV, "0")
    assert config.jobs_handoff_enabled({config.JOBS_HANDOFF_UNSAFE_KEY: True}) is False
    monkeypatch.setenv(config.JOBS_HANDOFF_UNSAFE_ENV, "1")
    assert config.jobs_handoff_enabled({}) is True


def test_jobs_handoff_reads_the_live_config_when_no_cfg_is_passed(monkeypatch):
    monkeypatch.setattr(config, "load_herdd_config",
                        lambda: {config.JOBS_HANDOFF_UNSAFE_KEY: "true"})
    assert config.jobs_handoff_enabled() is True


# --------------------------------------------------------------------------- #
# fleetd auto-adopt ceiling — fail-closed
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad", ["", "   ", "banana", "0", "-1", "nan", "inf",
                                 "-inf", "None"])
def test_adopt_default_is_positive_for_every_garbage_env(monkeypatch, bad):
    monkeypatch.setenv(config.FLEETD_ADOPT_BUDGET_ENV, bad)
    v = config.fleetd_adopt_default_budget_usd({})
    assert v == config.ADOPT_DEFAULT_BUDGET_USD and v > 0


@pytest.mark.parametrize("bad", [None, "", "banana", 0, -1, float("nan"),
                                 float("inf")])
def test_adopt_default_is_positive_for_every_garbage_config_value(bad):
    v = config.fleetd_adopt_default_budget_usd({config.FLEETD_ADOPT_BUDGET_KEY: bad})
    assert v == config.ADOPT_DEFAULT_BUDGET_USD and v > 0
    assert not math.isnan(v) and not math.isinf(v)


def test_adopt_default_precedence_env_over_config(monkeypatch):
    cfg = {config.FLEETD_ADOPT_BUDGET_KEY: "3.25"}
    assert config.fleetd_adopt_default_budget_usd(cfg) == 3.25
    monkeypatch.setenv(config.FLEETD_ADOPT_BUDGET_ENV, "1.5")
    assert config.fleetd_adopt_default_budget_usd(cfg) == 1.5


def test_an_unreadable_config_is_the_default_not_an_exception(monkeypatch):
    monkeypatch.setattr(config, "load_herdd_config",
                        lambda: (_ for _ in ()).throw(OSError("disk on fire")))
    assert config.fleetd_adopt_default_budget_usd() == config.ADOPT_DEFAULT_BUDGET_USD
    assert config._adopt_cfg() == {}


def test_adopt_default_reads_the_live_config_when_no_cfg_is_passed(monkeypatch):
    monkeypatch.setattr(config, "load_herdd_config",
                        lambda: {config.FLEETD_ADOPT_BUDGET_KEY: 7.5})
    assert config.fleetd_adopt_default_budget_usd() == 7.5


# --------------------------------------------------------------------------- #
# _runset_env_defaults
# --------------------------------------------------------------------------- #
def test_runset_env_defaults_sorted_and_bool_coerced():
    out = config._runset_env_defaults(
        {"env": {"ZED": 1, "ALPHA": "a", "FLAG": True, "OFF": False, "F": 1.5}})
    assert out == ["ALPHA=a", "F=1.5", "FLAG=1", "OFF=0", "ZED=1"]


def test_runset_env_defaults_absent_block_is_empty_not_an_error():
    assert config._runset_env_defaults({}) == []
    assert config._runset_env_defaults({"env": None}) == []


def test_runset_env_block_must_be_a_mapping():
    with pytest.raises(ValueError, match="must be a mapping"):
        config._runset_env_defaults({"env": ["A=1"]})


@pytest.mark.parametrize("key", ["9BAD", "with-dash", "with space", "", "a.b"])
def test_runset_env_rejects_an_invalid_key(key):
    with pytest.raises(ValueError, match="invalid key"):
        config._runset_env_defaults({"env": {key: "x"}})


def test_runset_env_rejects_a_non_string_key():
    with pytest.raises(ValueError, match="invalid key"):
        config._runset_env_defaults({"env": {True: "x"}})


@pytest.mark.parametrize("key", sorted(config._RUNSET_ENV_RESERVED))
def test_runset_env_reserved_keys_fail_closed(key):
    with pytest.raises(ValueError, match="reserved"):
        config._runset_env_defaults({"env": {key: "x"}})


@pytest.mark.parametrize("key", ["B2_BUCKET", "LLM_BASE_URL", "OPENROUTER_API_KEY"])
def test_runset_env_reserved_prefixes_fail_closed(key):
    assert key.startswith(config._RUNSET_ENV_RESERVED_PREFIXES)
    with pytest.raises(ValueError, match="reserved"):
        config._runset_env_defaults({"env": {key: "x"}})


def test_runset_env_rejects_a_non_scalar_value():
    with pytest.raises(ValueError, match="must be a scalar"):
        config._runset_env_defaults({"env": {"OK": {"nested": 1}}})


# --------------------------------------------------------------------------- #
# _boot_knob — the canonical CLI > env > yaml > constant resolver
# --------------------------------------------------------------------------- #
def test_boot_knob_falls_back_to_the_constant(no_yaml):
    assert config._boot_knob("BOOT_MIN_MBPS") == config._BOOT_KNOB_DEFAULTS["BOOT_MIN_MBPS"]


def test_boot_knob_cli_outranks_everything(monkeypatch, no_yaml):
    monkeypatch.setenv("BOOT_MIN_MBPS", "9")
    assert config._boot_knob("BOOT_MIN_MBPS", cli=3) == 3.0


def test_boot_knob_env_outranks_the_constant(monkeypatch, no_yaml):
    monkeypatch.setenv("BOOT_MIN_MBPS", "9")
    assert config._boot_knob("BOOT_MIN_MBPS") == 9.0


def test_boot_knob_yaml_sits_between_env_and_the_constant(monkeypatch):
    monkeypatch.setattr(config, "load_herdd_config", lambda: {"BOOT_MIN_MBPS": "7"})
    assert config._boot_knob("BOOT_MIN_MBPS") == 7.0
    monkeypatch.setenv("BOOT_MIN_MBPS", "9")
    assert config._boot_knob("BOOT_MIN_MBPS") == 9.0


@pytest.mark.parametrize("bad", ["", "banana", "  "])
def test_boot_knob_malformed_sources_are_skipped_not_fatal(monkeypatch, bad):
    monkeypatch.setenv("BOOT_MIN_MBPS", bad)
    monkeypatch.setattr(config, "load_herdd_config", lambda: {"BOOT_MIN_MBPS": bad})
    assert config._boot_knob("BOOT_MIN_MBPS") == config._BOOT_KNOB_DEFAULTS["BOOT_MIN_MBPS"]


def test_boot_knob_an_unreadable_config_is_not_fatal(monkeypatch):
    monkeypatch.setattr(config, "load_herdd_config",
                        lambda: (_ for _ in ()).throw(OSError("disk on fire")))
    assert config._boot_knob("BOOT_SLA_S", cast=int) == 600


def test_boot_knob_casts_every_source_uniformly(monkeypatch, no_yaml):
    monkeypatch.setenv("BOOT_SLA_S", "900")
    assert config._boot_knob("BOOT_SLA_S", cast=int) == 900
    assert isinstance(config._boot_knob("BOOT_SLA_S", cast=int), int)
    assert isinstance(config._boot_knob("BOOT_SLA_S"), float)


def test_boot_knob_unknown_name_is_a_programming_error(no_yaml):
    with pytest.raises(KeyError):
        config._boot_knob("NO_SUCH_KNOB")


# --------------------------------------------------------------------------- #
# default_disk_gb
# --------------------------------------------------------------------------- #
def test_disk_kinds_fall_back_to_their_own_constants(no_yaml):
    assert config.default_disk_gb("launch") == config.DISK_DEFAULT_LAUNCH_GB
    assert config.default_disk_gb("supervise") == config.DISK_DEFAULT_SUPERVISE_GB
    assert config.default_disk_gb("train") == config.DISK_DEFAULT_TRAIN_GB
    assert config.default_disk_gb("workflow") == config.DISK_DEFAULT_WORKFLOW_GB
    assert config.default_disk_gb("fleetd") == config.DISK_DEFAULT_FLEETD_GB
    assert config.default_disk_gb("serve") == config.DISK_DEFAULT_SERVE_GB


def test_disk_cli_beats_every_other_source(monkeypatch, no_yaml):
    monkeypatch.setenv("HERDD_DEFAULT_DISK", "999")
    assert config.default_disk_gb("train", cli=25) == 25


def test_disk_env_per_kind_wins_over_the_global_and_does_not_leak(monkeypatch, no_yaml):
    monkeypatch.setenv("HERDD_DEFAULT_DISK", "55")
    assert config.default_disk_gb("launch") == 55
    monkeypatch.setenv("HERDD_DEFAULT_DISK_TRAIN", "90")
    assert config.default_disk_gb("train") == 90
    assert config.default_disk_gb("launch") == 55


def test_disk_yaml_rung(monkeypatch):
    monkeypatch.setattr(config, "load_herdd_config",
                        lambda: {"default_disk": 35, "default_disk_train": 150})
    assert config.default_disk_gb("supervise") == 35
    assert config.default_disk_gb("train") == 150


@pytest.mark.parametrize("bad", ["", "lots", "0", "-10"])
def test_disk_a_malformed_override_is_skipped_not_fatal(monkeypatch, no_yaml, bad):
    monkeypatch.setenv("HERDD_DEFAULT_DISK", bad)
    assert config.default_disk_gb("launch") == config.DISK_DEFAULT_LAUNCH_GB


def test_disk_an_unknown_kind_is_a_programming_error_not_a_silent_40():
    with pytest.raises(KeyError):
        config.default_disk_gb("nosuchkind")
    with pytest.raises(KeyError):
        config.default_disk_gb(None)


def test_disk_an_unreadable_config_is_not_fatal(monkeypatch):
    monkeypatch.setattr(config, "load_herdd_config",
                        lambda: (_ for _ in ()).throw(OSError("disk on fire")))
    assert config.default_disk_gb("train") == config.DISK_DEFAULT_TRAIN_GB


# --------------------------------------------------------------------------- #
# the patch idiom the whole test migration depends on (plan §8 porting
# mechanics (b)): cross-module calls are module-attribute lookups, so a
# monkeypatch on the module still steers the callee.
# --------------------------------------------------------------------------- #
def test_patching_load_herdd_config_steers_every_consumer(monkeypatch):
    calls = []

    def fake():
        calls.append(1)
        return {config.LOCAL_GPU_KEY: True,
                config.JOBS_HANDOFF_UNSAFE_KEY: True,
                config.FLEETD_ADOPT_BUDGET_KEY: 4.0,
                "default_disk": 77,
                "BOOT_MIN_MBPS": 42.0}

    monkeypatch.setattr(config, "load_herdd_config", fake)
    assert config.local_gpu_allowed() is True
    assert config.jobs_handoff_enabled() is True
    assert config.fleetd_adopt_default_budget_usd() == 4.0
    assert config.default_disk_gb("launch") == 77
    assert config._boot_knob("BOOT_MIN_MBPS") == 42.0
    assert len(calls) == 5, "every consumer went through the patched seam"


# --------------------------------------------------------------------------- #
# THE SHIM — what replaced the differential section at plan step 7
#
# Through step 6 this section compared the port against a still-live flat twin:
# `assert config.X == vastconf.X`, symbol by symbol. At step 7 `vastconf.py`
# became a re-export shim over `vastlib.core.config`, and every one of those
# asserts turned into `X == X`. Two of them were worse than tautological: they
# derived their expectation by AST-parsing vastconf.py for FunctionDef /
# ClassDef / Assign nodes, and a re-export shim has NONE of those — only an
# ImportFrom — so the parse returned an EMPTY list and both tests asserted
# `[] == []`. They would have stayed green with the port's every symbol and
# every `# moved-from:` marker deleted.
#
# So the expectation is frozen here instead of derived from the shim, and the
# section now pins the three things that are actually true post-shim and worth
# breaking on: the surface is exactly these 32 names, the port still carries a
# marker for each, and the two spellings are ONE object rather than two equal
# copies. The behavioral coverage that the value-for-value comparison used to
# stand in for lives in the precedence sections above, which drive `config`
# directly and were never differential.
# --------------------------------------------------------------------------- #
#: The 32 top-level names tools/vast/vastconf.py defined before it became a
#: shim. Frozen deliberately: derived-from-the-file was the vacuity bug.
_FLAT_TOP_LEVEL_NAMES = (
    "ADOPT_DEFAULT_BUDGET_USD", "DISK_DEFAULT_FLEETD_GB", "DISK_DEFAULT_LAUNCH_GB",
    "DISK_DEFAULT_SERVE_GB", "DISK_DEFAULT_SUPERVISE_GB", "DISK_DEFAULT_TRAIN_GB",
    "DISK_DEFAULT_WORKFLOW_GB", "FLEETD_ADOPT_BUDGET_ENV", "FLEETD_ADOPT_BUDGET_KEY",
    "JOBS_HANDOFF_UNSAFE_ENV", "JOBS_HANDOFF_UNSAFE_KEY", "LOCAL_GPU_ENV",
    "LOCAL_GPU_KEY", "_BOOT_KNOB_DEFAULTS", "_HERE", "_REPO_CONFIG",
    "_RUNSET_ENV_KEY_RE", "_RUNSET_ENV_RESERVED", "_RUNSET_ENV_RESERVED_PREFIXES",
    "_USER_CONFIG", "_adopt_cfg", "_boot_knob", "_load_yaml_file",
    "_parse_simple_yaml", "_runset_env_defaults", "default_disk_gb",
    "fleetd_adopt_default_budget_usd", "jobs_handoff_enabled", "load_env",
    "load_herdd_config", "local_gpu_allowed", "require_local_gpu",
)


def test_every_flat_symbol_exists_in_the_port():
    assert len(_FLAT_TOP_LEVEL_NAMES) == 32
    missing = [n for n in _FLAT_TOP_LEVEL_NAMES if not hasattr(config, n)]
    assert missing == [], f"the port dropped: {missing}"


def test_every_ported_symbol_carries_its_moved_from_marker():
    """Plan §7.1 generates the rename table from these markers; a missing one is
    a symbol the test migration cannot find."""
    src = open(config.__file__).read()
    markers = {ln.split("# moved-from:", 1)[1].strip()
               for ln in src.splitlines() if ln.strip().startswith("# moved-from:")}
    expected = {f"vastconf.{n}" for n in _FLAT_TOP_LEVEL_NAMES}
    assert expected - markers == set(), f"unmarked: {sorted(expected - markers)}"


def test_the_shim_exports_exactly_the_thirty_two_and_defines_nothing():
    """The shim must be a pure re-export: no body of its own (that would fork
    the resolver) and no widening of the surface (herdd.py's facade carries
    only 18 of these, and sizing the shim from it drops DISK_DEFAULT_SERVE_GB,
    which launch_serve.sh reads to size a real box's disk)."""
    assert tuple(sorted(vastconf.__all__)) == tuple(sorted(_FLAT_TOP_LEVEL_NAMES))
    tree = ast.parse(open(os.path.join(HERE, "vastconf.py")).read())
    defs = [n.name for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    assert defs == [], defs
    assigned = [t.id for n in tree.body if isinstance(n, ast.Assign)
                for t in n.targets if isinstance(t, ast.Name)]
    assert assigned == ["__all__"], assigned


def test_the_constants_are_identical_objects_not_merely_similar():
    """Post-shim this is an `is`, not an `==`: the whole point of a re-export is
    that `vastconf.X` and `config.X` are the same object, so a future edit that
    re-derives a value in the shim (rather than importing it) fails here."""
    for name in _FLAT_TOP_LEVEL_NAMES:
        assert getattr(vastconf, name) is getattr(config, name), name


def test_the_frozen_values_are_what_the_rest_of_the_repo_reads():
    """The literals the differential comparison used to stand in for. These are
    external contracts — env var names appear in herdd.yaml, docs and shell;
    the disk defaults size real rentals (DISK_DEFAULT_SERVE_GB is read by
    launch_serve.sh's heredoc through the shim) — so a change to one is a
    deliberate edit, not a refactor."""
    assert config.LOCAL_GPU_KEY == "allow_local_gpu"
    assert config.LOCAL_GPU_ENV == "HERDD_ALLOW_LOCAL_GPU"
    assert config.JOBS_HANDOFF_UNSAFE_KEY == "jobs_handoff_unsafe_enable"
    assert config.JOBS_HANDOFF_UNSAFE_ENV == "HERDD_JOBS_HANDOFF_UNSAFE"
    assert config.FLEETD_ADOPT_BUDGET_KEY == "fleetd_adopt_default_budget_usd"
    assert config.FLEETD_ADOPT_BUDGET_ENV == "FLEETD_ADOPT_DEFAULT_BUDGET_USD"
    assert config.ADOPT_DEFAULT_BUDGET_USD == 10.0
    assert (config.DISK_DEFAULT_LAUNCH_GB, config.DISK_DEFAULT_SUPERVISE_GB,
            config.DISK_DEFAULT_TRAIN_GB, config.DISK_DEFAULT_WORKFLOW_GB,
            config.DISK_DEFAULT_FLEETD_GB, config.DISK_DEFAULT_SERVE_GB) == (
        40, 40, 120, 40, 40, 60)
    assert config._RUNSET_ENV_KEY_RE.pattern == "^[A-Za-z_][A-Za-z0-9_]*$"
    assert config._RUNSET_ENV_RESERVED == frozenset({
        "RUN_ID", "RUNSET", "TRAIN_ENV_VER", "HF_TOKEN", "BASE_MODEL_B2",
        "SELFTEST_BASE_B2", "FAST_BOOT"})
    assert config._RUNSET_ENV_RESERVED_PREFIXES == ("B2_", "LLM_", "OPENROUTER_")


def test_the_resolvers_are_the_same_callables_through_both_spellings():
    """What the value-for-value loop became. Sampling `config.f(cfg) ==
    vastconf.f(cfg)` post-shim calls one function twice; asserting the binding
    is what still has content — and it is the invariant every remaining
    bare-name consumer (launch_serve.sh, local_smoke.py, parked_lifecycle.py,
    the herdd facade) actually depends on."""
    for name in ("local_gpu_allowed", "jobs_handoff_enabled", "require_local_gpu",
                 "fleetd_adopt_default_budget_usd", "default_disk_gb", "_boot_knob",
                 "_parse_simple_yaml", "_load_yaml_file", "load_herdd_config",
                 "load_env", "_runset_env_defaults", "_adopt_cfg"):
        assert getattr(vastconf, name) is getattr(config, name), name
    # ...and the launcher facade is the third leg of the same chain: a
    # `from vastconf import X` copy plus a later rebind would fork them silently.
    import herdd
    for name in ("load_env", "load_herdd_config", "jobs_handoff_enabled",
                 "_boot_knob", "_parse_simple_yaml", "_load_yaml_file",
                 "_runset_env_defaults", "_REPO_CONFIG", "_USER_CONFIG",
                 "_BOOT_KNOB_DEFAULTS", "_RUNSET_ENV_KEY_RE",
                 "_RUNSET_ENV_RESERVED", "_RUNSET_ENV_RESERVED_PREFIXES",
                 "JOBS_HANDOFF_UNSAFE_KEY", "JOBS_HANDOFF_UNSAFE_ENV",
                 "DISK_DEFAULT_LAUNCH_GB", "DISK_DEFAULT_SUPERVISE_GB",
                 "DISK_DEFAULT_TRAIN_GB"):
        assert getattr(herdd, name) is getattr(config, name), name


# ------------------------------------------------------- load_env walk anchors


def test_load_env_falls_back_to_the_install_tree(tmp_path, monkeypatch):
    """No .env anywhere up from CWD must not mean no .env at all: an
    absolute-path `python3 tools/vast/herdd.py …` run from outside the repo
    loses B2_BUCKET otherwise, and the failure surfaces far away as a
    RunmetaError instead of here."""
    cwd = tmp_path / "elsewhere"
    cwd.mkdir()
    anchor = tmp_path / "repo" / "tools" / "vast"
    anchor.mkdir(parents=True)
    (tmp_path / "repo" / ".env").write_text(
        "HERDD_TEST_ANCHOR_KEY=from-anchor\n")
    monkeypatch.chdir(cwd)
    monkeypatch.setattr(config, "_HERE", str(anchor))
    os.environ.pop("HERDD_TEST_ANCHOR_KEY", None)
    try:
        config.load_env()
        assert os.environ.get("HERDD_TEST_ANCHOR_KEY") == "from-anchor"
    finally:
        os.environ.pop("HERDD_TEST_ANCHOR_KEY", None)


def test_load_env_cwd_walk_wins_over_the_install_tree(tmp_path, monkeypatch):
    """Anchor order is precedence: a caller standing in another project keeps
    that project's .env; the install-tree walk is a fallback, never an
    override."""
    cwd = tmp_path / "other-project"
    cwd.mkdir()
    (cwd / ".env").write_text("HERDD_TEST_PRECEDENCE_KEY=from-cwd\n")
    anchor = tmp_path / "repo" / "tools" / "vast"
    anchor.mkdir(parents=True)
    (tmp_path / "repo" / ".env").write_text(
        "HERDD_TEST_PRECEDENCE_KEY=from-anchor\n")
    monkeypatch.chdir(cwd)
    monkeypatch.setattr(config, "_HERE", str(anchor))
    os.environ.pop("HERDD_TEST_PRECEDENCE_KEY", None)
    try:
        config.load_env()
        assert os.environ.get("HERDD_TEST_PRECEDENCE_KEY") == "from-cwd"
    finally:
        os.environ.pop("HERDD_TEST_PRECEDENCE_KEY", None)

"""Portable tests for the runset config.yaml env: block — _load_runset_config,
_load_runset_spot_config, and the pure _runset_env_defaults validator/coercer.
Runs in the toolchain-free lane (`pytest -m "not integration"`): no rclone, no
B2, no network, no creds.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import herdd as vc  # noqa: E402

# `_load_runset_config` / `_load_runset_spot_config` read `_HERE` from THEIR own
# module globals, which since plan §8 step 6d is `vastlib.cli._runsets` — the
# flat `herdd` name is now a re-export, and patching it steers nothing (the
# reads below would silently fall through to the repo's real `runsets/` tree
# and return {}). So the tmp-runset fixture patches the owner. Everything else
# in this file is a pure call and stays on the flat namespace.
from vastlib.cli import _runsets  # noqa: E402


# --------------------------------------------------------------------------- #
# _runset_env_defaults — pure helper (no filesystem, no launch)
# --------------------------------------------------------------------------- #
def test_env_defaults_sorted_and_coerced():
    # unsorted input -> sorted-by-key wire order; bool->1/0, int/float->str.
    out = vc._runset_env_defaults({"env": {
        "ZED": "z", "ALPHA": "a", "FLAG_ON": True, "FLAG_OFF": False,
        "COUNT": 3, "RATIO": 1.5,
    }})
    assert out == [
        "ALPHA=a", "COUNT=3", "FLAG_OFF=0", "FLAG_ON=1", "RATIO=1.5", "ZED=z",
    ]


def test_env_defaults_absent_block_is_empty():
    assert vc._runset_env_defaults({}) == []
    assert vc._runset_env_defaults({"spot": {"budget_usd": 40}}) == []


def test_env_defaults_bad_key_rejected():
    for bad in ("1LEADS_DIGIT", "HAS-DASH", "HAS SPACE", "has.dot"):
        with pytest.raises(ValueError):
            vc._runset_env_defaults({"env": {bad: "x"}})


def test_env_defaults_reserved_exact_rejected():
    for key in ("RUN_ID", "RUNSET", "HF_TOKEN", "BASE_MODEL_B2",
                "SELFTEST_BASE_B2", "FAST_BOOT", "TRAIN_ENV_VER"):
        with pytest.raises(ValueError):
            vc._runset_env_defaults({"env": {key: "x"}})


def test_env_defaults_reserved_prefix_rejected():
    # B2_ covers every var _b2_eu_pairs() emits, plus the shipped creds.
    for key in ("B2_BUCKET", "B2_REGION_EU", "LLM_API_KEY", "OPENROUTER_API_KEY"):
        with pytest.raises(ValueError):
            vc._runset_env_defaults({"env": {key: "x"}})


def test_env_defaults_non_scalar_rejected():
    with pytest.raises(ValueError):
        vc._runset_env_defaults({"env": {"K": {"nested": 1}}})
    with pytest.raises(ValueError):
        vc._runset_env_defaults({"env": {"K": ["a", "b"]}})


def test_env_defaults_non_mapping_env_rejected():
    with pytest.raises(ValueError):
        vc._runset_env_defaults({"env": "SDPA_BACKENDS=all"})


# --------------------------------------------------------------------------- #
# _load_runset_config / _load_runset_spot_config — via a tmp config.yaml
# --------------------------------------------------------------------------- #
def _write_runset(monkeypatch, tmp_path, name, body):
    monkeypatch.setattr(_runsets, "_HERE", str(tmp_path))
    d = tmp_path / "runsets" / name
    d.mkdir(parents=True)
    (d / "config.yaml").write_text(body)


def test_load_config_both_blocks(monkeypatch, tmp_path):
    _write_runset(monkeypatch, tmp_path, "demo", (
        "spot:\n"
        "  budget_usd: 40\n"
        "  ckpt_interval_s: 180\n"
        "env:\n"
        '  SDPA_BACKENDS: "all"\n'
        '  USE_LIGER: "1"\n'
    ))
    cfg = vc._load_runset_config("demo")
    assert cfg["spot"]["budget_usd"] == 40
    # env: block parses under the same one-level nesting the spot: block uses.
    assert vc._runset_env_defaults(cfg) == ["SDPA_BACKENDS=all", "USE_LIGER=1"]
    # thin extraction still returns just the spot sub-dict.
    assert vc._load_runset_spot_config("demo")["ckpt_interval_s"] == 180


def test_load_config_absent_file_is_empty(monkeypatch, tmp_path):
    monkeypatch.setattr(_runsets, "_HERE", str(tmp_path))
    assert vc._load_runset_config("nope") == {}
    assert vc._load_runset_spot_config("nope") == {}
    assert vc._runset_env_defaults(vc._load_runset_config("nope")) == []


def test_load_config_reserved_key_in_file_raises(monkeypatch, tmp_path):
    _write_runset(monkeypatch, tmp_path, "bad", (
        "env:\n"
        '  HF_TOKEN: "leak"\n'
    ))
    with pytest.raises(ValueError):
        vc._runset_env_defaults(vc._load_runset_config("bad"))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

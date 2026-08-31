"""Portable tests for jobmatrix.py — expansion determinism, merge semantics,
validation fail-fast, and the one-bundle/N-tickets submit over the fake
in-memory B2 runner. Runs in the toolchain-free lane (`pytest -m "not
integration"`): no rclone, no B2, no network, no creds.
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jobmatrix as jx  # noqa: E402
import jobmeta as jm  # noqa: E402
from test_jobmeta import FakeB2  # noqa: E402


class FakeB2Bin(FakeB2):
    """FakeB2 whose local->remote copyto tolerates binary files (zst bundles)."""
    def __call__(self, args, input=None):
        if args[0] == "copyto" and not args[1].startswith("b2:"):
            with open(args[1], "rb") as fh:
                self.store[self._key(args[2])] = fh.read()
            return 0, "", ""
        return super().__call__(args, input)


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
def _exp(**kw):
    base = dict(
        name="exp3",
        entrypoint="run.sh",
        timeout_s=600,
        env={"DATA_FILE": "train.jsonl", "LORA_R": "32"},
        results=["out/**"],
        needs={"gpu": True, "gpu_ram_gb": 48, "venv": "serve"},
        axes={
            "base": {
                "qwen": {"BASE_SLUG": "qwen3-8b"},
                "lfm": jx.Variant(env={"BASE_SLUG": "lfm25-1.2b-thinking"},
                                  needs={"gpu_ram_gb": 24}, timeout_s=300),
            },
            "rank": {
                "r16": {"LORA_R": "16"},
                "r64": {"LORA_R": "64"},
            },
        },
    )
    base.update(kw)
    return jx.Experiment(**base)


def _bundle(tmp_path, matrix_body=None):
    d = tmp_path / "job"
    d.mkdir(parents=True, exist_ok=True)
    (d / "run.sh").write_text("echo hi\n")
    if matrix_body:
        (d / "matrix.py").write_text(matrix_body)
    return str(d)


# --------------------------------------------------------------------------- #
# expansion
# --------------------------------------------------------------------------- #
def test_expand_cross_product_deterministic():
    arms = jx.expand(_exp())
    assert [a.name for a in arms] == ["qwen-r16", "qwen-r64", "lfm-r16", "lfm-r64"]
    assert arms[0].axes == {"base": "qwen", "rank": "r16"}
    assert [a.name for a in jx.expand(_exp())] == [a.name for a in arms]


def test_env_merge_later_axis_wins_and_arm_id_injected():
    a = {x.name: x for x in jx.expand(_exp())}["qwen-r64"]
    assert a.env["LORA_R"] == "64"                 # rank axis beats base env
    assert a.env["DATA_FILE"] == "train.jsonl"     # base env survives
    assert a.env["BASE_SLUG"] == "qwen3-8b"
    assert a.env["ARM_ID"] == "qwen-r64"


def test_variant_config_overrides_merge():
    arms = {x.name: x for x in jx.expand(_exp())}
    assert arms["lfm-r16"].needs == {"gpu": True, "gpu_ram_gb": 24, "venv": "serve"}
    assert arms["lfm-r16"].timeout_s == 300
    assert arms["qwen-r16"].needs["gpu_ram_gb"] == 48
    assert arms["qwen-r16"].timeout_s == 600


def test_exclude_drops_arm_and_all_dropped_raises():
    exp = _exp(exclude=lambda a: a.axes["base"] == "lfm")
    assert [a.name for a in jx.expand(exp)] == ["qwen-r16", "qwen-r64"]
    with pytest.raises(jx.MatrixError, match="every arm"):
        jx.expand(_exp(exclude=lambda a: True))


def test_reserved_env_and_bad_variant_key_raise():
    with pytest.raises(jx.MatrixError, match="reserved"):
        jx.expand(_exp(env={"ARM_ID": "nope"}))
    exp = _exp()
    exp.axes["rank"]["r16"] = {"EXP_ID": "nope"}
    with pytest.raises(jx.MatrixError, match="reserved"):
        jx.expand(exp)
    exp = _exp()
    exp.axes["rank"]["R_16"] = {}
    with pytest.raises(jx.MatrixError, match="slug"):
        jx.expand(exp)


def test_overlong_job_name_and_collision_raise():
    exp = _exp(name="a-very-long-experiment-name-indeed-yes")
    with pytest.raises(jx.MatrixError, match="40-char"):
        jx.expand(exp)
    exp = _exp()
    exp.axes = {"a": {"x-y": {}, "x": {}}, "b": {"z": {}, "y-z": {}}}
    with pytest.raises(jx.MatrixError, match="duplicate"):
        jx.expand(exp)          # "x-y"+"z" collides with "x"+"y-z"


def test_no_axes_is_an_error():
    with pytest.raises(jx.MatrixError, match="no axes"):
        jx.expand(_exp(axes={}))


# --------------------------------------------------------------------------- #
# loading + validation
# --------------------------------------------------------------------------- #
MATRIX_SRC = """\
from jobmatrix import Experiment, Variant
EXPERIMENT = Experiment(
    name="mx",
    entrypoint="run.sh",
    timeout_s=120,
    env={"K": "v"},
    results=["out/**"],
    needs={"gpu": True},
    axes={"m": {"a": {"M": "1"}, "b": {"M": "2"}}},
)
"""


def test_load_experiment_and_validate(tmp_path):
    src = _bundle(tmp_path, MATRIX_SRC)
    exp = jx.load_experiment(src)
    rows = jx.validate_experiment(exp, src)
    assert [arm.name for arm, _, _ in rows] == ["a", "b"]
    for arm, cfg, _ in rows:
        assert cfg["name"] == f"mx-{arm.name}"
        assert cfg["entrypoint"] == "run.sh"
        assert cfg["env"]["ARM_ID"] == arm.name
    with pytest.raises(jx.MatrixError, match="matrix.py"):
        jx.load_experiment(_bundle(tmp_path / "empty"))


def test_validate_catches_missing_entrypoint(tmp_path):
    d = tmp_path / "job2"
    d.mkdir()
    (d / "matrix.py").write_text(MATRIX_SRC)   # run.sh NOT created
    exp = jx.load_experiment(str(d))
    with pytest.raises(jm.JobmetaError, match="entrypoint"):
        jx.validate_experiment(exp, str(d))


# --------------------------------------------------------------------------- #
# submit: ONE bundle, N tickets, manifest
# --------------------------------------------------------------------------- #
def _submit(tmp_path, fake, **kw):
    src = _bundle(tmp_path, MATRIX_SRC)
    exp = jx.load_experiment(src)
    return jx.submit(exp, src, 44, runner=fake, bucket="bkt", actor="cli:t",
                     staging_dir=str(tmp_path / "stage"),
                     local_out=str(tmp_path / "expout"), log=lambda *a: None, **kw)


def test_submit_one_bundle_n_tickets_manifest(tmp_path):
    fake = FakeB2Bin(bucket="bkt")
    man = _submit(tmp_path, fake)
    bundles = [k for k in fake.store if k.startswith("jobs/bundles/")]
    tickets = [k for k in fake.store if k.startswith("jobs/queue/44/")]
    assert len(bundles) == 1 and len(tickets) == 2
    assert len(man["arms"]) == 2

    for a in man["arms"]:
        tk = json.loads(fake.store[f"jobs/queue/44/{a['job_id']}.json"])
        assert tk["bundle_sha256"] == man["bundle_sha256"]
        assert tk["config"]["env"]["EXP_ID"] == man["exp_id"]
        assert tk["config"]["env"]["ARM_ID"] == a["arm"]
        # first-class association block (audit seam — jobd echoes it on events)
        assert tk["config"]["experiment"] == {
            "exp_id": man["exp_id"], "arm": a["arm"], "axes": a["axes"]}
        evs = [k for k in fake.store if k.startswith(f"jobs/{a['job_id']}/events/")]
        assert len(evs) == 1
        sub = json.loads(fake.store[evs[0]])
        assert sub["exp_id"] == man["exp_id"] and sub["arm"] == a["arm"]

    stored = json.loads(fake.store[f"experiments/{man['exp_id']}/manifest.json"])
    assert stored["arms"] == man["arms"]


def test_submit_dry_run_mutates_nothing(tmp_path):
    fake = FakeB2Bin(bucket="bkt")
    man = _submit(tmp_path, fake, dry_run=True)
    assert fake.store == {} and len(man["arms"]) == 2


def test_submit_only_filter_and_no_match(tmp_path):
    fake = FakeB2Bin(bucket="bkt")
    man = _submit(tmp_path, fake, only="a")
    assert [a["arm"] for a in man["arms"]] == ["a"]
    with pytest.raises(jx.MatrixError, match="matched no arms"):
        _submit(tmp_path, FakeB2Bin(bucket="bkt"), only="zzz")


def test_submit_bundle_dedupe_hit_skips_upload(tmp_path):
    fake = FakeB2Bin(bucket="bkt")
    man1 = _submit(tmp_path / "one", fake)
    n_objects = len(fake.store)
    man2 = _submit(tmp_path / "two", fake)      # identical folder content
    assert man2["bundle_sha256"] == man1["bundle_sha256"]
    bundles = [k for k in fake.store if k.startswith("jobs/bundles/")]
    assert len(bundles) == 1
    # second submit adds tickets/events/manifest but no second bundle
    assert len(fake.store) == n_objects + 2 + 2 + 1


# --------------------------------------------------------------------------- #
# status
# --------------------------------------------------------------------------- #
def test_exp_status_folds_arms(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    fake = FakeB2Bin(bucket="bkt")
    man = _submit(tmp_path, fake)
    a0, a1 = man["arms"]
    jm.emit_event(a0["job_id"], "claimed", actor="box:44", runner=fake,
                  bucket="bkt", instance_id="44")
    jm.emit_event(a0["job_id"], "done", actor="box:44", runner=fake,
                  bucket="bkt", instance_id="44", rc=0)
    jm.emit_event(a1["job_id"], "failed", actor="box:44", runner=fake,
                  bucket="bkt", instance_id="44", rc=3, reason="boom")

    st = jx.exp_status(man["exp_id"], runner=fake, bucket="bkt")
    rows = {r["arm"]: r for r in st["arms"]}
    assert rows["a"]["status"] == "done" and rows["a"]["rc"] == 0
    assert rows["b"]["status"] == "failed" and rows["b"]["fail_reason"] == "boom"
    with pytest.raises(jx.MatrixError, match="no manifest"):
        jx.exp_status("20260710T000000-nope-0000", runner=fake, bucket="bkt")


# --------------------------------------------------------------------------- #
# CLI smoke ($0 paths only)
# --------------------------------------------------------------------------- #
def test_cli_expand_smoke(tmp_path, capsys):
    src = _bundle(tmp_path, MATRIX_SRC)
    jx.main(["expand", src])
    out = capsys.readouterr().out
    assert "2 arms" in out and "mx" in out and "a " in out

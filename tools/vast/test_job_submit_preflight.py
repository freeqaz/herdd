"""Portable tests for the `job submit` B2-staleness preflight (GAP 1).

Runs in the toolchain-free lane (`pytest -m "not integration"`): NO real vast
API, NO B2/rclone, NO network, NO creds. The B2 read is transport-injected via a
tiny in-memory fake runner (rclone contract: (args, input) -> (rc, out, err)),
modelled on test_jobmeta.py's FakeB2 `cat`/`lsf` subset.

The incident this closes (docs/plans/spot-resilient-eval-jobs.md, "Live-run
results", 2026-07-12): p2_reader_eval's `runset` asset pulls the LIVE
b2:runsets/base-reader-train; the local rehearsal uses LOCAL fixtures, so it was
blind to the fact that B2 still held a pre-EVAL_ONLY train.sh — the box ran the
wrong entrypoint. The preflight compares the single sentinel (train.sh) byte for
byte, so a never-re-staged code change is caught before the box runs it.
"""
import argparse
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jobmeta as jm  # noqa: E402
from vastlib.boxes import health  # noqa: E402
from vastlib.cli import _runsets  # noqa: E402
from vastlib.cli import train as cli_train  # noqa: E402
from vastlib.fleet import client as fleet_client  # noqa: E402
from vastlib.jobs import submit, view as jobs_view  # noqa: E402
from vastlib.storage import b2  # noqa: E402


# --------------------------------------------------------------------------- #
# in-memory fake rclone: only `cat b2:<bucket>/<key>` is exercised here.
# --------------------------------------------------------------------------- #
class FakeCat:
    """`store` maps a B2 key (no b2:bucket/ prefix) -> str body. `cat` returns it;
    anything unknown is a miss (rc=1). rc127=True simulates rclone-not-found /
    absent creds (every op fails soft)."""
    def __init__(self, store=None, bucket="bkt", rc127=False):
        self.store = dict(store or {})
        self.bucket = bucket
        self.rc127 = rc127
        self.calls = []

    def __call__(self, args, input=None):
        self.calls.append(list(args))
        if self.rc127:
            return 127, "", "rclone not found on PATH"
        op = args[0]
        if op == "cat":
            remote = args[1]
            prefix = f"b2:{self.bucket}/"
            assert remote.startswith(prefix), remote
            k = remote[len(prefix):]
            return (0, self.store[k], "") if k in self.store else (1, "", "not found")
        return 1, "", f"unexpected op {op}"


def _mkrepo(tmp_path, name="base-reader-train", sentinel="train.sh",
            body="echo hi\n", build=True):
    """Create a fake repo_root with tools/vast/runsets/<name>[/_build]/<sentinel>."""
    root = tmp_path / "repo"
    d = root / "tools" / "vast" / "runsets" / name
    if build:
        d = d / "_build"
    d.mkdir(parents=True)
    if sentinel is not None:
        (d / sentinel).write_text(body)
    return str(root), str(d)


def _asset(name="runset", b2="runsets/base-reader-train"):
    return {"name": name, "b2": b2, "mode": "sync", "optional": False, "require": []}


# --------------------------------------------------------------------------- #
# local_source_for_asset — the b2-prefix -> local-dir mapping heuristic
# --------------------------------------------------------------------------- #
def test_local_source_prefers_build(tmp_path):
    root, build = _mkrepo(tmp_path, build=True)
    assert build.endswith("_build")
    assert jm.local_source_for_asset("runsets/base-reader-train", root) == build


def test_local_source_falls_back_to_runset_dir(tmp_path):
    root, d = _mkrepo(tmp_path, build=False)
    assert jm.local_source_for_asset("runsets/base-reader-train", root) == d


def test_local_source_none_for_immutable_prefixes(tmp_path):
    root, _ = _mkrepo(tmp_path)
    assert jm.local_source_for_asset("base-models/qwen3-8b", root) is None
    assert jm.local_source_for_asset("checkpoints/base-reader/x", root) is None
    assert jm.local_source_for_asset("eval-env/foo", root) is None


def test_local_source_none_when_runset_absent(tmp_path):
    root, _ = _mkrepo(tmp_path, name="base-reader-train")
    assert jm.local_source_for_asset("runsets/does-not-exist", root) is None


# --------------------------------------------------------------------------- #
# check_asset_staleness — the single-sentinel byte-identity signal
# --------------------------------------------------------------------------- #
def test_stale_when_b2_differs(tmp_path):
    root, _ = _mkrepo(tmp_path, body="echo NEW EVAL_ONLY\n")
    fake = FakeCat({"runsets/base-reader-train/train.sh": "echo OLD train-block\n"})
    out = jm.check_asset_staleness([_asset()], repo_root=root, runner=fake, bucket="bkt")
    assert len(out) == 1
    f = out[0]
    assert f["status"] == "stale"
    assert f["sentinel"] == "train.sh"


def test_ok_when_byte_identical(tmp_path):
    body = "echo hi EVAL_ONLY\n"
    root, _ = _mkrepo(tmp_path, body=body)
    fake = FakeCat({"runsets/base-reader-train/train.sh": body})
    out = jm.check_asset_staleness([_asset()], repo_root=root, runner=fake, bucket="bkt")
    assert out[0]["status"] == "ok"


def test_unknown_when_creds_absent(tmp_path):
    root, _ = _mkrepo(tmp_path)
    fake = FakeCat(rc127=True)          # rclone/creds unavailable -> soft-fail
    out = jm.check_asset_staleness([_asset()], repo_root=root, runner=fake, bucket="bkt")
    assert out[0]["status"] == "unknown"


def test_unknown_when_b2_sentinel_missing(tmp_path):
    root, _ = _mkrepo(tmp_path)
    fake = FakeCat({})                  # object not present on B2
    out = jm.check_asset_staleness([_asset()], repo_root=root, runner=fake, bucket="bkt")
    assert out[0]["status"] == "unknown"


def test_skipped_for_immutable_prefix(tmp_path):
    root, _ = _mkrepo(tmp_path)
    fake = FakeCat({})
    a = _asset(name="base", b2="base-models/qwen3-8b")
    out = jm.check_asset_staleness([a], repo_root=root, runner=fake, bucket="bkt")
    assert out[0]["status"] == "skipped"
    assert fake.calls == []             # no B2 read for an immutable asset


def test_skipped_when_no_local_sentinel(tmp_path):
    # _build exists but holds no train.sh/run.sh/... -> nothing to compare.
    root, d = _mkrepo(tmp_path, sentinel=None)
    (__import__("pathlib").Path(d) / "data.bin").write_text("x")
    fake = FakeCat({"runsets/base-reader-train/train.sh": "whatever\n"})
    out = jm.check_asset_staleness([_asset()], repo_root=root, runner=fake, bucket="bkt")
    assert out[0]["status"] == "skipped"


def test_picks_run_sh_when_no_train_sh(tmp_path):
    root, _ = _mkrepo(tmp_path, sentinel="run.sh", body="echo NEW\n")
    fake = FakeCat({"runsets/base-reader-train/run.sh": "echo OLD\n"})
    out = jm.check_asset_staleness([_asset()], repo_root=root, runner=fake, bucket="bkt")
    assert out[0]["status"] == "stale" and out[0]["sentinel"] == "run.sh"


# --------------------------------------------------------------------------- #
# asset_preflight_report — findings -> (surfaced lines, refuse?)
# --------------------------------------------------------------------------- #
def _finding(status, name="runset", b2="runsets/base-reader-train",
             sentinel="train.sh", **extra):
    d = {"name": name, "b2": b2, "status": status, "sentinel": sentinel,
         "detail": "d"}
    d.update(extra)
    return d


def test_report_loud_warn_but_no_refuse_nonstrict():
    lines, refuse = jm.asset_preflight_report([_finding("stale")], strict=False)
    assert refuse is False
    assert any("ASSET STALE" in ln for ln in lines)
    assert any("train.sh" in ln for ln in lines)


def test_report_refuses_under_strict():
    lines, refuse = jm.asset_preflight_report([_finding("stale")], strict=True)
    assert refuse is True
    assert any("ASSET STALE" in ln for ln in lines)


def test_report_silent_on_ok_and_skipped():
    lines, refuse = jm.asset_preflight_report(
        [_finding("ok"), _finding("skipped", b2="base-models/x", sentinel=None)],
        strict=False)
    assert lines == [] and refuse is False


def test_report_confirms_a_tracked_ok_out_loud():
    """A declared `tracks:` contract reports that it was HONOURED, not just that
    nothing was wrong. The heuristic path stays quiet (above): an inference has
    nothing to confirm. This asymmetry is the point — the incident that motivated
    the preflight was a submit path that looked exactly as quiet as a clean one
    (jobmatrix had no asset check at all), so a silent pass must not be the
    operator's only evidence that the check ran."""
    lines, refuse = jm.asset_preflight_report(
        [dict(_finding("ok"), kind="tracks", local="tools/x.py")], strict=False)
    assert refuse is False
    assert any("provenance OK" in ln for ln in lines)


def test_report_note_on_unknown_never_refuses():
    for strict in (False, True):
        lines, refuse = jm.asset_preflight_report([_finding("unknown")], strict=strict)
        assert refuse is False                       # a blip never blocks a submit
        assert any(ln.lower().startswith("note") for ln in lines)


# --------------------------------------------------------------------------- #
# CLI wiring — cmd_job_submit honors the flags (B2 mutations never reached here)
# --------------------------------------------------------------------------- #
def _import_herdd():
    import herdd  # noqa
    return herdd


def _mkjobdir(tmp_path):
    d = tmp_path / "job"
    d.mkdir()
    (d / "run.sh").write_text("echo hi\n")
    (d / "job-config.yaml").write_text(
        "version: 1\nname: p2-probe\nentrypoint: run.sh\ntimeout_s: 60\n"
        "needs:\n  gpu: false\n  venv: none\n"
        "assets:\n  - name: runset\n    b2: runsets/base-reader-train\n"
        "    dest: runset\n    mode: sync\n")
    return str(d)


def _args(dir_, **over):
    ns = argparse.Namespace(dir=dir_, box="7", name=None, timeout=None, env=None,
                            dry_run=True, no_asset_check=False, strict_assets=False)
    for k, v in over.items():
        setattr(ns, k, v)
    return ns


def test_cli_strict_refuses_on_stale(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(jm, "check_asset_staleness",
                        lambda *a, **k: [_finding("stale")])
    monkeypatch.setattr(b2, "_ensure_b2_remote", lambda: None)
    with pytest.raises(SystemExit):
        submit.cmd_job_submit(_args(_mkjobdir(tmp_path), strict_assets=True))
    assert "STALE" in capsys.readouterr().err.upper()


def test_cli_default_warns_but_continues(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(jm, "check_asset_staleness",
                        lambda *a, **k: [_finding("stale")])
    monkeypatch.setattr(b2, "_ensure_b2_remote", lambda: None)
    # non-strict continues PAST the preflight; stop it at the next step (bundling)
    # so no B2 mutation is attempted.
    monkeypatch.setattr(jm, "write_bundle",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("REACHED_BUNDLE")))
    with pytest.raises(RuntimeError, match="REACHED_BUNDLE"):
        submit.cmd_job_submit(_args(_mkjobdir(tmp_path)))
    assert "ASSET STALE" in capsys.readouterr().err


def test_cli_no_asset_check_skips_preflight(tmp_path, monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("check_asset_staleness must not run under --no-asset-check")
    monkeypatch.setattr(jm, "check_asset_staleness", _boom)
    monkeypatch.setattr(b2, "_ensure_b2_remote", lambda: None)
    monkeypatch.setattr(jm, "write_bundle",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("REACHED_BUNDLE")))
    with pytest.raises(RuntimeError, match="REACHED_BUNDLE"):
        submit.cmd_job_submit(_args(_mkjobdir(tmp_path), no_asset_check=True))


# --------------------------------------------------------------------------- #
# `--env K=V` submit-time override (REMOTE_WAVE_PLAN §7: the frontier-wave
# README documented a flag that did not exist). It folds onto the yaml `env:`
# block PRE-validation, exactly like --name/--timeout, so the ticket wire format
# is unchanged and the bundle sha (folder-addressed) is untouched.
# --------------------------------------------------------------------------- #
def _envjob(tmp_path, env_yaml="env:\n  TARGET: dc3\n  K: '5'\n"):
    d = tmp_path / "envjob"
    d.mkdir()
    (d / "run.sh").write_text("echo hi\n")
    (d / "job-config.yaml").write_text(
        "version: 1\nname: p2-probe\nentrypoint: run.sh\ntimeout_s: 60\n"
        "needs:\n  gpu: false\n  venv: none\n" + env_yaml)
    return str(d)


def _submit_capture_cfg(monkeypatch, args):
    """Run cmd_job_submit far enough to see the VALIDATED config, then stop
    before any B2 work."""
    seen = {}
    real = jm.validate_job_config

    def _spy(raw, src):
        cfg, warns = real(raw, src)
        seen["cfg"] = cfg
        return cfg, warns
    monkeypatch.setattr(jm, "validate_job_config", _spy)
    monkeypatch.setattr(b2, "_ensure_b2_remote", lambda: None)
    monkeypatch.setattr(jm, "write_bundle",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("REACHED_BUNDLE")))
    with pytest.raises(RuntimeError, match="REACHED_BUNDLE"):
        submit.cmd_job_submit(args)
    return seen["cfg"]


def test_env_override_merges_onto_yaml_block(tmp_path, monkeypatch):
    cfg = _submit_capture_cfg(monkeypatch, _args(
        _envjob(tmp_path), env=["K=20", "ADAPTER_T_SHA256=abc123"]))
    assert cfg["env"]["TARGET"] == "dc3"           # untouched yaml key survives
    assert cfg["env"]["K"] == "20"                 # overridden
    assert cfg["env"]["ADAPTER_T_SHA256"] == "abc123"   # added


def test_env_override_absent_leaves_config_identical(tmp_path, monkeypatch):
    cfg = _submit_capture_cfg(monkeypatch, _args(_envjob(tmp_path)))
    assert cfg["env"] == {"TARGET": "dc3", "K": "5"}


def test_env_override_on_a_job_with_no_env_block(tmp_path, monkeypatch):
    cfg = _submit_capture_cfg(monkeypatch,
                              _args(_envjob(tmp_path, env_yaml=""), env=["TARGET=rb3"]))
    assert cfg["env"] == {"TARGET": "rb3"}


def test_env_override_value_may_contain_equals_and_be_empty(tmp_path, monkeypatch):
    cfg = _submit_capture_cfg(monkeypatch, _args(
        _envjob(tmp_path), env=["LORA_SPECS=t=checkpoints/v6", "K="]))
    assert cfg["env"]["LORA_SPECS"] == "t=checkpoints/v6"
    assert cfg["env"]["K"] == ""


def test_env_override_never_echoes_values(tmp_path, monkeypatch, capsys):
    _submit_capture_cfg(monkeypatch,
                        _args(_envjob(tmp_path), env=["VLLM_API_KEY=s3cr3t"]))
    out = capsys.readouterr()
    assert "VLLM_API_KEY" in out.out
    assert "s3cr3t" not in (out.out + out.err)


def test_env_override_rejects_bad_shapes():
    for bad in (["NOEQUALS"], ["BAD-KEY=1"], ["9LEADING=1"], ["with space=1"], ["=v"]):
        with pytest.raises(SystemExit):
            submit._apply_env_overrides({}, bad)


def test_env_override_helper_is_pure_when_no_pairs():
    raw = {"env": {"A": "1"}}
    assert submit._apply_env_overrides(raw, None) == []
    assert submit._apply_env_overrides(raw, []) == []
    assert raw == {"env": {"A": "1"}}               # not even normalized/copied


# =========================================================================== #
# DECLARED-PROVENANCE staleness (`tracks:`) — the 2026-07-31 gap.
#
# The sentinel check above compares ONE guessed file (train.sh) against a
# GITIGNORED local mirror (runsets/<name>/_build). On 2026-07-31 that combination
# green-lit a v7 training submit whose staged trainer was HALF the repo file:
# b2:runsets/witness-lifter/train_proposer_lora.py = 64,593 B vs
# tools/pipeline/ml_infra/train_proposer_lora.py at HEAD = 125,307 B (many
# commits back, including a `--quant` default change). train.sh matched on both
# sides, so nothing printed. `tracks:` replaces the guess with a declaration and
# fails CLOSED — before a box is rented.
# =========================================================================== #
import hashlib  # noqa: E402
import json as _json  # noqa: E402
import pathlib  # noqa: E402


class FakeB2Files:
    """Fake rclone over an in-memory {key: bytes} store. Implements the two ops
    the tracked check uses: `hashsum <algo> <remote>` (a METADATA read — the
    remote serves a stored digest, nothing is downloaded) and `size --json
    <remote>` (the fallback for an object with no usable hash).

    `algos` is which digests THIS remote speaks: our live `b2:` is rclone
    `type = s3` and answers md5 only, while a native-b2 remote answers sha1
    only — measured 2026-07-31, and the reason the check negotiates instead of
    assuming. `algos=()` simulates a multipart object (no usable hash)."""
    def __init__(self, store=None, bucket="bkt", algos=("md5", "sha1"),
                 no_hash=False, rc127=False):
        self.store = {k: (v.encode() if isinstance(v, str) else v)
                      for k, v in (store or {}).items()}
        self.bucket, self.rc127 = bucket, rc127
        self.algos = () if no_hash else tuple(algos)
        self.calls = []

    def __call__(self, args, input=None):
        self.calls.append(list(args))
        if self.rc127:
            return 127, "", "rclone not found on PATH"
        op = args[0]
        remote = args[-1]
        prefix = f"b2:{self.bucket}/"
        key = remote[len(prefix):] if remote.startswith(prefix) else remote
        if key not in self.store:
            return 1, "", "not found"
        body = self.store[key]
        if op == "hashsum":
            algo = args[1]
            if algo not in self.algos:
                return 1, "", "hash unsupported: hash type not supported"
            digest = hashlib.new(algo, body).hexdigest()
            return 0, f"{digest}  {key.rsplit('/', 1)[-1]}\n", ""
        if op == "size":
            return 0, _json.dumps({"count": 1, "bytes": len(body)}), ""
        return 1, "", f"unexpected op {op}"


TRACK_KEY = "runsets/witness-lifter/train_proposer_lora.py"
TRACK_LOCAL = "tools/pipeline/ml_infra/train_proposer_lora.py"


def _mkrepo_tracked(tmp_path, body="def main():\n    pass\n"):
    root = tmp_path / "repo"
    p = root / TRACK_LOCAL
    p.parent.mkdir(parents=True)
    p.write_text(body)
    return str(root)


# --------------------------------------------------------------------------- #
# schema
# --------------------------------------------------------------------------- #
def test_tracks_parses_in_both_yaml_parsers(monkeypatch):
    """`tracks:` inside an `assets:` item must use the FLOW form — a block map
    nested in a list item is the one shape the no-PyYAML fallback cannot
    represent. Both parsers must agree on it."""
    text = ("version: 1\nname: p\nentrypoint: run.sh\n"
            "assets:\n"
            "  - name: runset\n"
            "    b2: runsets/witness-lifter\n"
            f"    tracks: {{train_proposer_lora.py: {TRACK_LOCAL}}}\n")
    yaml_mod = pytest.importorskip("yaml")
    ref = yaml_mod.safe_load(text)
    monkeypatch.setitem(sys.modules, "yaml", None)
    assert jm._parse_job_yaml(text) == ref
    assert ref["assets"][0]["tracks"] == {"train_proposer_lora.py": TRACK_LOCAL}


def test_tracks_normalizes_and_rejects_escapes():
    assert jm._normalize_tracks(None, "x") == {}
    assert jm._normalize_tracks({"a.py": "b/c.py"}, "x") == {"a.py": "b/c.py"}
    for bad in ({"a.py": "/etc/passwd"}, {"a.py": "../../x"}, {"../a": "b"},
                {"/a.py": "b/c.py"}, {"a.py": ""}, {"": "b"}, {"a.py": "b2:bkt/x"}):
        with pytest.raises(jm.JobmetaError):
            jm._normalize_tracks(bad, "assets['r'].tracks")
    with pytest.raises(jm.JobmetaError):
        jm._normalize_tracks(["a"], "x")


def test_collect_tracked_folds_both_declaration_sites():
    cfg = {"tracks": {"runsets/x/a.py": "tools/a.py"},
           "assets": [{"name": "runset", "b2": "runsets/witness-lifter",
                       "tracks": {"train_proposer_lora.py": TRACK_LOCAL}},
                      {"name": "base", "b2": "base-models/m"}]}
    assert jm.collect_tracked(cfg) == {"runsets/x/a.py": "tools/a.py",
                                       TRACK_KEY: TRACK_LOCAL}
    assert jm.collect_tracked({}) == {}          # no declaration -> no-op


# --------------------------------------------------------------------------- #
# check_tracked_assets — fails CLOSED on a real mismatch
# --------------------------------------------------------------------------- #
def test_tracked_stale_when_b2_holds_older_bytes(tmp_path):
    """The incident, reproduced: B2's copy is a truncated older trainer."""
    root = _mkrepo_tracked(tmp_path, body="x" * 125307)
    fake = FakeB2Files({TRACK_KEY: "x" * 64593})
    out = jm.check_tracked_assets({TRACK_KEY: TRACK_LOCAL}, repo_root=root,
                                  runner=fake, bucket="bkt")
    assert len(out) == 1 and out[0]["status"] == "stale"
    assert out[0]["kind"] == "tracks" and out[0]["local"] == TRACK_LOCAL
    assert out[0]["detail"].split()[0] in ("md5", "sha1")   # a hash, not size


def test_tracked_ok_when_byte_identical(tmp_path):
    body = "def main():\n    pass\n"
    root = _mkrepo_tracked(tmp_path, body=body)
    out = jm.check_tracked_assets({TRACK_KEY: TRACK_LOCAL}, repo_root=root,
                                  runner=FakeB2Files({TRACK_KEY: body}), bucket="bkt")
    assert out[0]["status"] == "ok"


def test_tracked_falls_back_to_size_when_remote_has_no_hash(tmp_path):
    """A multipart B2 object stores no SHA-1; the check degrades to size rather
    than going blind (the incident's 64,593 vs 125,307 is caught by size alone)."""
    root = _mkrepo_tracked(tmp_path, body="x" * 125307)
    fake = FakeB2Files({TRACK_KEY: "x" * 64593}, no_hash=True)
    out = jm.check_tracked_assets({TRACK_KEY: TRACK_LOCAL}, repo_root=root,
                                  runner=fake, bucket="bkt")
    assert out[0]["status"] == "stale" and "size" in out[0]["detail"]
    # ...and identical sizes with no hash cannot be called stale
    fake2 = FakeB2Files({TRACK_KEY: "y" * 125307}, no_hash=True)
    out2 = jm.check_tracked_assets({TRACK_KEY: TRACK_LOCAL}, repo_root=root,
                                   runner=fake2, bucket="bkt")
    assert out2[0]["status"] == "ok"


@pytest.mark.parametrize("algos", [("md5", "sha1"), ("md5",), ("sha1",)])
def test_tracked_negotiates_the_hash_the_remote_speaks(tmp_path, algos):
    """Our live b2: remote is rclone `type = s3` (md5/ETag only); a native-b2
    remote is sha1 only. Hard-coding either silently degrades the check to a
    size compare, which is how a same-size edit would slip through."""
    root = _mkrepo_tracked(tmp_path, body="new\n")
    fake = FakeB2Files({TRACK_KEY: "old\n"}, algos=algos)
    out = jm.check_tracked_assets({TRACK_KEY: TRACK_LOCAL}, repo_root=root,
                                  runner=fake, bucket="bkt")
    assert out[0]["status"] == "stale"
    # ...and it used a real digest, not the size fallback (both bodies are 4 B)
    assert out[0]["detail"].split()[0] in algos
    ok = jm.check_tracked_assets({TRACK_KEY: TRACK_LOCAL}, repo_root=root,
                                 runner=FakeB2Files({TRACK_KEY: "new\n"}, algos=algos),
                                 bucket="bkt")
    assert ok[0]["status"] == "ok"


def test_tracked_same_size_different_bytes_is_caught(tmp_path):
    """The case a size-only check CANNOT see — and the reason the hash matters."""
    root = _mkrepo_tracked(tmp_path, body="AAAA\n")
    out = jm.check_tracked_assets({TRACK_KEY: TRACK_LOCAL}, repo_root=root,
                                  runner=FakeB2Files({TRACK_KEY: "BBBB\n"}),
                                  bucket="bkt")
    assert out[0]["status"] == "stale"


def test_tracked_unknown_without_creds_never_blocks(tmp_path):
    root = _mkrepo_tracked(tmp_path)
    out = jm.check_tracked_assets({TRACK_KEY: TRACK_LOCAL}, repo_root=root,
                                  runner=FakeB2Files(rc127=True), bucket="bkt")
    assert out[0]["status"] == "unknown"
    assert jm.asset_preflight_report(out)[1] is False


def test_tracked_broken_when_declared_local_file_is_gone(tmp_path):
    """A `tracks:` entry naming a path that no longer exists is a WRONG
    declaration — it must be loud, not silently unverifiable."""
    root = _mkrepo_tracked(tmp_path)
    out = jm.check_tracked_assets({TRACK_KEY: "tools/moved/away.py"},
                                  repo_root=root,
                                  runner=FakeB2Files({TRACK_KEY: "hi"}), bucket="bkt")
    assert out[0]["status"] == "broken"
    assert jm.asset_preflight_report(out)[1] is True


def test_tracked_check_is_read_only(tmp_path):
    root = _mkrepo_tracked(tmp_path)
    fake = FakeB2Files({TRACK_KEY: "different"})
    jm.check_tracked_assets({TRACK_KEY: TRACK_LOCAL}, repo_root=root,
                            runner=fake, bucket="bkt")
    assert all(c[0] in ("hashsum", "size") for c in fake.calls), fake.calls


def test_restage_hint_names_the_runset_build_script():
    assert jm.restage_hint(TRACK_KEY, TRACK_LOCAL) == \
        "bash tools/vast/runsets/witness-lifter/build.sh"
    assert "rclone copyto" in jm.restage_hint("corpora/x/f.jsonl", "tools/f.jsonl")


# --------------------------------------------------------------------------- #
# report policy: a DECLARATION refuses without --strict; a GUESS does not
# --------------------------------------------------------------------------- #
def _tracked_finding(status="stale"):
    return {"name": "runset", "b2": TRACK_KEY, "kind": "tracks", "status": status,
            "sentinel": "train_proposer_lora.py", "local": TRACK_LOCAL,
            "src": "/x", "detail": "sha1 differ", "restage": "bash build.sh"}


def test_tracked_stale_refuses_without_strict():
    lines, refuse = jm.asset_preflight_report([_tracked_finding()], strict=False)
    assert refuse is True
    assert any("ASSET STALE" in ln for ln in lines)
    assert any("bash build.sh" in ln for ln in lines)       # actionable
    assert any("--allow-stale-assets" in ln for ln in lines)


def test_allow_stale_downgrades_tracked_to_a_warning():
    lines, refuse = jm.asset_preflight_report([_tracked_finding()], allow_stale=True)
    assert refuse is False
    assert any("ASSET STALE" in ln for ln in lines)         # still loud
    assert jm.asset_preflight_report([_tracked_finding()], strict=True,
                                     allow_stale=True)[1] is False


def test_heuristic_stale_policy_is_unchanged():
    """The sentinel guess still only refuses under --strict-assets."""
    assert jm.asset_preflight_report([_finding("stale")], strict=False)[1] is False
    assert jm.asset_preflight_report([_finding("stale")], strict=True)[1] is True


def test_tracks_declaration_supersedes_the_sentinel_guess(tmp_path):
    """An asset that declares tracks: must not ALSO be sentinel-compared — the
    _build mirror is gitignored and was itself staler than B2 in the incident."""
    root, _ = _mkrepo(tmp_path, name="witness-lifter", body="local train.sh\n")
    a = dict(_asset(b2="runsets/witness-lifter"),
             tracks={"train_proposer_lora.py": TRACK_LOCAL})
    fake = FakeCat({"runsets/witness-lifter/train.sh": "B2 train.sh\n"})
    out = jm.check_asset_staleness([a], repo_root=root, runner=fake, bucket="bkt")
    assert out[0]["status"] == "skipped" and "tracks" in out[0]["detail"]
    assert fake.calls == []


def test_asset_preflight_is_the_one_shared_seam(tmp_path):
    """Both submit surfaces call this; it must cover BOTH declaration sites."""
    root = _mkrepo_tracked(tmp_path, body="new\n")
    cfg = {"assets": [{"name": "runset", "b2": "runsets/witness-lifter",
                       "tracks": {"train_proposer_lora.py": TRACK_LOCAL}}]}
    fake = FakeB2Files({TRACK_KEY: "old\n"})
    out = jm.asset_preflight(cfg, repo_root=root, runner=fake, bucket="bkt")
    assert [f for f in out if f.get("kind") == "tracks"][0]["status"] == "stale"
    assert jm.asset_preflight_report(out)[1] is True


# --------------------------------------------------------------------------- #
# CLI wiring — herdd job submit + jobmatrix submit
# --------------------------------------------------------------------------- #
def _trackjob(tmp_path):
    d = tmp_path / "trackjob"
    d.mkdir()
    (d / "run.sh").write_text("echo hi\n")
    (d / "job-config.yaml").write_text(
        "version: 1\nname: p2-probe\nentrypoint: run.sh\ntimeout_s: 60\n"
        "needs:\n  gpu: false\n  venv: none\n"
        f"tracks:\n  {TRACK_KEY}: {TRACK_LOCAL}\n")
    return str(d)


def test_cli_refuses_tracked_stale_without_strict(tmp_path, monkeypatch, capsys):
    """The whole point: no --strict-assets, and the submit still dies."""
    monkeypatch.setattr(jm, "asset_preflight",
                        lambda *a, **k: [_tracked_finding()])
    monkeypatch.setattr(b2, "_ensure_b2_remote", lambda: None)
    monkeypatch.setattr(jm, "write_bundle",
                        lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError("must refuse BEFORE bundling")))
    args = _args(_trackjob(tmp_path))
    args.allow_stale_assets = False
    with pytest.raises(SystemExit):
        submit.cmd_job_submit(args)
    assert "ASSET STALE" in capsys.readouterr().err


def test_cli_allow_stale_assets_proceeds(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(jm, "asset_preflight",
                        lambda *a, **k: [_tracked_finding()])
    monkeypatch.setattr(b2, "_ensure_b2_remote", lambda: None)
    monkeypatch.setattr(jm, "write_bundle",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("REACHED_BUNDLE")))
    args = _args(_trackjob(tmp_path))
    args.allow_stale_assets = True
    with pytest.raises(RuntimeError, match="REACHED_BUNDLE"):
        submit.cmd_job_submit(args)
    assert "ASSET STALE" in capsys.readouterr().err        # loud, but not fatal


def test_cli_checks_a_job_with_tracks_but_no_assets(tmp_path, monkeypatch):
    """The v7-matrix shape: the entrypoint pulls from B2 itself, so there is no
    `assets:` block to hang the old check on."""
    seen = {}
    def _spy(cfg, **k):
        seen["cfg"] = cfg
        return []
    monkeypatch.setattr(jm, "asset_preflight", _spy)
    monkeypatch.setattr(b2, "_ensure_b2_remote", lambda: None)
    monkeypatch.setattr(jm, "write_bundle",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("REACHED_BUNDLE")))
    args = _args(_trackjob(tmp_path))
    args.allow_stale_assets = False
    with pytest.raises(RuntimeError, match="REACHED_BUNDLE"):
        submit.cmd_job_submit(args)
    assert seen["cfg"]["tracks"] == {TRACK_KEY: TRACK_LOCAL}


def _matrixjob(tmp_path):
    d = tmp_path / "mx"
    d.mkdir(exist_ok=True)
    (d / "run.sh").write_text("echo hi\n")
    (d / "matrix.py").write_text(
        "from jobmatrix import Experiment\n"
        "EXPERIMENT = Experiment(name='mx', entrypoint='run.sh', timeout_s=60,\n"
        "    needs={'gpu': False, 'venv': 'none'},\n"
        f"    tracks={{{TRACK_KEY!r}: {TRACK_LOCAL!r}}},\n"
        "    axes={'a': {'one': {'K': '1'}, 'two': {'K': '2'}}})\n")
    return str(d)


def test_jobmatrix_submit_refuses_on_tracked_stale(tmp_path, monkeypatch):
    """jobmatrix was the surface with NO staleness check at all — and the one the
    v7 pair actually submits through."""
    import jobmatrix
    root = _mkrepo_tracked(tmp_path, body="new\n")
    fake = FakeB2Files({TRACK_KEY: "old\n"})
    monkeypatch.setattr(jobmatrix.jobmeta, "write_bundle",
                        lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError("must refuse BEFORE bundling")))
    exp = jobmatrix.load_experiment(_matrixjob(tmp_path))
    lines = []
    with pytest.raises(jobmatrix.MatrixError, match="stale asset"):
        jobmatrix.submit(exp, _matrixjob(tmp_path), "90001", runner=fake,
                         bucket="bkt", repo_root=root, dry_run=True,
                         log=lines.append)
    assert any("ASSET STALE" in ln for ln in lines)


def test_jobmatrix_submit_passes_when_staged_matches(tmp_path, monkeypatch):
    import jobmatrix
    root = _mkrepo_tracked(tmp_path, body="same\n")
    fake = FakeB2Files({TRACK_KEY: "same\n"})
    monkeypatch.setattr(jobmatrix.jobmeta, "write_bundle",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("REACHED_BUNDLE")))
    exp = jobmatrix.load_experiment(_matrixjob(tmp_path))
    with pytest.raises(RuntimeError, match="REACHED_BUNDLE"):
        jobmatrix.submit(exp, _matrixjob(tmp_path), "90001", runner=fake,
                         bucket="bkt", repo_root=root, dry_run=True, log=lambda *_: None)


def test_jobmatrix_no_asset_check_skips(tmp_path, monkeypatch):
    import jobmatrix
    root = _mkrepo_tracked(tmp_path, body="new\n")
    def _boom(*a, **k):
        raise AssertionError("preflight must not run under --no-asset-check")
    monkeypatch.setattr(jobmatrix.jobmeta, "asset_preflight", _boom)
    monkeypatch.setattr(jobmatrix.jobmeta, "write_bundle",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("REACHED_BUNDLE")))
    exp = jobmatrix.load_experiment(_matrixjob(tmp_path))
    with pytest.raises(RuntimeError, match="REACHED_BUNDLE"):
        jobmatrix.submit(exp, _matrixjob(tmp_path), "90001", runner=FakeB2Files(),
                         bucket="bkt", repo_root=root, dry_run=True,
                         check_assets=False, log=lambda *_: None)


def test_matrix_without_tracks_is_a_noop(tmp_path):
    """A matrix that declares nothing must not read B2 at all."""
    import jobmatrix
    d = pathlib.Path(_matrixjob(tmp_path))
    (d / "matrix.py").write_text(
        (d / "matrix.py").read_text().replace(
            f"    tracks={{{TRACK_KEY!r}: {TRACK_LOCAL!r}}},\n", ""))
    fake = FakeB2Files()
    exp = jobmatrix.load_experiment(str(d))
    assert exp.tracks is None
    try:
        jobmatrix.submit(exp, str(d), "90001", runner=fake, bucket="bkt",
                         repo_root=str(tmp_path), dry_run=True,
                         staging_dir=str(tmp_path / "stage"), log=lambda *_: None)
    except Exception:
        pass                                  # bundling/staging is not under test
    assert fake.calls == []


# --------------------------------------------------------------------------- #
# EVAL_ENV_VER pin gate (M4) — jobmeta.eval_env_pin_report + its CLI wiring.
#
# The incident: wave A submitted a `needs.venv: eval` bundle to a box launched
# with no EVAL_ENV_VER, jobd resolved eval-env/LATEST at boot, and LATEST was
# OLDER than the preflighted env (a pinned bake does not advance it). The FLOOR
# gate graded three revert-then-splice controls on code that predates
# floor_restore.py and PASSed on 2 bytes-moving controls instead of 5
# (docs/plans/witness/g2_push/FLOOR_DEGRADATION_2026-08-01.md). The remedy
# shipped as "an operator rule, not code"; these tests pin it as code.
# --------------------------------------------------------------------------- #
def _evalcfg(job_pin=None, venv="eval"):
    cfg = {"name": "wave", "entrypoint": "run.sh", "needs": {"gpu": True, "venv": venv},
           "env": {"TARGET": "rb3"}}
    if job_pin is not None:
        cfg["env"]["EVAL_ENV_VER"] = job_pin
    return cfg


def test_eval_pin_unpinned_everywhere_refuses():
    lines, refuse = jm.eval_env_pin_report(_evalcfg(), {}, box="7")
    assert refuse is True
    assert any("EVAL_ENV_VER UNPINNED" in ln for ln in lines)
    assert any("--env EVAL_ENV_VER=" in ln for ln in lines)   # says how to fix it


def test_eval_pin_non_eval_job_is_a_noop():
    """A job that never touches the baked env must not be gated (or read a box)."""
    for venv in ("none", "serve"):
        assert jm.eval_env_pin_report(_evalcfg(venv=venv), {}) == ([], False)


def test_eval_pin_box_launch_env_satisfies_the_gate():
    lines, refuse = jm.eval_env_pin_report(
        _evalcfg(), {"EVAL_ENV_VER": "20260731-0831-0ad5be27"}, box="7")
    assert refuse is False
    assert any("20260731-0831-0ad5be27" in ln for ln in lines)


def test_eval_pin_job_env_only_passes_but_says_it_does_not_steer_the_fetch():
    """jobd sources .job.env inside the ENTRYPOINT subshell, after check_venv has
    already provisioned — so a job-env pin documents rather than determines."""
    lines, refuse = jm.eval_env_pin_report(_evalcfg("20260731-2327-635b7d8c"), {}, box="7")
    assert refuse is False
    assert any(ln.startswith("~~ NOTE:") for ln in lines)
    assert any("--eval-env-ver" in ln for ln in lines)


def test_eval_pin_conflicting_pins_refuse():
    """The box env wins the fetch, the job env wins the artifact — a wave that
    grades on one and reports the other is exactly the confound M4 closes."""
    lines, refuse = jm.eval_env_pin_report(
        _evalcfg("aaa"), {"EVAL_ENV_VER": "bbb"}, box="7")
    assert refuse is True
    assert any("CONFLICT" in ln for ln in lines)


def test_eval_pin_blank_values_do_not_count_as_a_pin():
    for job_pin, box_env in ((" ", {}), (None, {"EVAL_ENV_VER": ""})):
        _, refuse = jm.eval_env_pin_report(_evalcfg(job_pin), box_env)
        assert refuse is True


def test_eval_pin_unreadable_box_record_still_refuses():
    """--local, or a soft vast-API failure: unknown is not permission."""
    lines, refuse = jm.eval_env_pin_report(_evalcfg(), {}, box_env_known=False)
    assert refuse is True
    assert any("could not be read" in ln for ln in lines)


# --- require_box_pin: the launcher's readback ------------------------------- #
# launch_jobs_box.sh rents the box, injects the pin as `launch --env
# EVAL_ENV_VER=<ver>`, then submits. The default job-pin-only NOTE is the wrong
# verdict for THAT caller: its box is cold (nothing has provisioned
# /workspace/eval yet), so a box env with no pin is not "already warm at the
# right version", it is proof the injection did not land — and the very next
# thing that happens is jobd fetching eval-env/LATEST. Opt-in, because the soft
# path is load-bearing for the other shape: a hand-resubmit onto a warm box.
def test_eval_pin_require_box_pin_refuses_a_job_only_pin():
    lines, refuse = jm.eval_env_pin_report(
        _evalcfg("20260806-2152-76cd109a"), {}, box="7", require_box_pin=True)
    assert refuse is True
    assert any("EVAL_ENV_VER NOT ON THE BOX" in ln for ln in lines)
    # names the version to relaunch with, so the fix is copy-pasteable
    assert any("--eval-env-ver 20260806-2152-76cd109a" in ln for ln in lines)


def test_eval_pin_require_box_pin_is_satisfied_by_a_box_pin():
    """The success path stays the ordinary one — strict mode adds a refusal,
    it does not add a second way to pass."""
    lines, refuse = jm.eval_env_pin_report(
        _evalcfg("20260806-2152-76cd109a"),
        {"EVAL_ENV_VER": "20260806-2152-76cd109a"}, box="7", require_box_pin=True)
    assert refuse is False
    assert any("launch env" in ln for ln in lines)


def test_eval_pin_require_box_pin_refuses_an_unreadable_box_record():
    lines, refuse = jm.eval_env_pin_report(
        _evalcfg("20260806-2152-76cd109a"), {}, box="7",
        box_env_known=False, require_box_pin=True)
    assert refuse is True
    assert any("could not be read" in ln for ln in lines)


def test_eval_pin_require_box_pin_does_not_gate_a_non_eval_job():
    """Strict mode must not turn `venv: none` into a gated job — the fetch it
    protects does not happen there at all."""
    for venv in ("none", "serve"):
        assert jm.eval_env_pin_report(
            _evalcfg(venv=venv), {}, require_box_pin=True) == ([], False)


def test_eval_pin_require_box_pin_still_refuses_a_conflict():
    lines, refuse = jm.eval_env_pin_report(
        _evalcfg("aaa"), {"EVAL_ENV_VER": "bbb"}, box="7", require_box_pin=True)
    assert refuse is True
    assert any("CONFLICT" in ln for ln in lines)


def test_eval_pin_default_is_unchanged_by_the_new_keyword():
    """The regression that would matter most: strict defaulting to ON would
    start refusing every resubmit onto a warm box launched without a pin."""
    lines, refuse = jm.eval_env_pin_report(_evalcfg("20260806-2152-76cd109a"),
                                           {}, box="7")
    assert refuse is False
    assert any(ln.startswith("~~ NOTE:") for ln in lines)


def _evaljobdir(tmp_path, job_pin=None):
    d = tmp_path / "evaljob"
    d.mkdir()
    (d / "run.sh").write_text("echo hi\n")
    envblock = f"env:\n  EVAL_ENV_VER: {job_pin}\n" if job_pin else ""
    (d / "job-config.yaml").write_text(
        "version: 1\nname: wave\nentrypoint: run.sh\ntimeout_s: 60\n"
        "needs:\n  gpu: false\n  venv: eval\n" + envblock)
    return str(d)


def test_cli_refuses_unpinned_eval_submit(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(health, "_get_instance_soft", lambda iid: {"extra_env": []})
    monkeypatch.setattr(b2, "_ensure_b2_remote", lambda: None)
    monkeypatch.setattr(jm, "write_bundle",
                        lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError("must refuse BEFORE bundling")))
    with pytest.raises(SystemExit):
        submit.cmd_job_submit(_args(_evaljobdir(tmp_path)))
    assert "EVAL_ENV_VER UNPINNED" in capsys.readouterr().err


def test_cli_eval_submit_proceeds_when_the_box_is_pinned(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(health, "_get_instance_soft",
                        lambda iid: {"extra_env": [["EVAL_ENV_VER", "20260731-0831"]]})
    monkeypatch.setattr(b2, "_ensure_b2_remote", lambda: None)
    monkeypatch.setattr(jm, "write_bundle",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("REACHED_BUNDLE")))
    with pytest.raises(RuntimeError, match="REACHED_BUNDLE"):
        submit.cmd_job_submit(_args(_evaljobdir(tmp_path)))
    assert "eval-env pin: 20260731-0831" in capsys.readouterr().err


def test_cli_submit_env_override_pins_it(tmp_path, monkeypatch, capsys):
    """`job submit --env EVAL_ENV_VER=<ver>` is the documented fix — it must
    actually satisfy the gate (the override lands before validation)."""
    monkeypatch.setattr(health, "_get_instance_soft", lambda iid: None)
    monkeypatch.setattr(b2, "_ensure_b2_remote", lambda: None)
    monkeypatch.setattr(jm, "write_bundle",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("REACHED_BUNDLE")))
    args = _args(_evaljobdir(tmp_path), env=["EVAL_ENV_VER=20260731-2327-635b7d8c"])
    with pytest.raises(RuntimeError, match="REACHED_BUNDLE"):
        submit.cmd_job_submit(args)
    assert "20260731-2327-635b7d8c" in capsys.readouterr().err


def test_cli_non_eval_submit_never_reads_the_box(tmp_path, monkeypatch):
    """A plain job's submit must stay API-free — the gate is the only new read."""
    def _boom(iid):
        raise AssertionError("no instance read for a needs.venv:none job")
    monkeypatch.setattr(health, "_get_instance_soft", _boom)
    monkeypatch.setattr(b2, "_ensure_b2_remote", lambda: None)
    monkeypatch.setattr(jm, "asset_preflight", lambda *a, **k: [])
    monkeypatch.setattr(jm, "write_bundle",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("REACHED_BUNDLE")))
    with pytest.raises(RuntimeError, match="REACHED_BUNDLE"):
        submit.cmd_job_submit(_args(_mkjobdir(tmp_path)))


# --------------------------------------------------------------------------- #
# GAP 3 — `herdd train --runset` had NO staleness gate at all
#
# `job submit` (GAP 1) and `jobmatrix submit` both route through
# jobmeta.asset_preflight. `herdd train --runset` — the OLDER, blunter
# launcher — did not. It rents a box and points it at b2:runsets/<NAME>/, whose
# contents were pushed by that runset's build.sh at some unrecorded time in the
# past, and ALL SEVEN runsets stage train_proposer_lora.py that way. That is the
# 2026-07-31 incident's exact shape (B2 64,593 B vs HEAD 125,307 B, including a
# --quant semantics change), except that here nothing read B2 at all — there was
# not even a sentinel heuristic to be fooled, so a stale staged trainer was not
# merely un-flagged, it was undetectable.
# --------------------------------------------------------------------------- #
RUNSET_CFG = {"tracks": {"train_proposer_lora.py": TRACK_LOCAL,
                         "train.sh": "tools/vast/runsets/witness-lifter/train.sh"}}


def test_runset_preflight_cfg_rebases_onto_the_runset_prefix():
    """A runset declares paths RELATIVE to its own B2 prefix; the probe must
    carry full keys or the check would read the wrong objects."""
    probe = jm.runset_preflight_cfg("witness-lifter", RUNSET_CFG)
    assert probe["tracks"] == {
        TRACK_KEY: TRACK_LOCAL,
        "runsets/witness-lifter/train.sh":
            "tools/vast/runsets/witness-lifter/train.sh"}
    assert probe["assets"] == []


def test_runset_preflight_cfg_rejects_a_malformed_declaration():
    """A bad `tracks:` must RAISE, not silently degrade to checking nothing —
    the failure mode this whole preflight exists to prevent."""
    for bad in ({"tracks": {"x.py": "/abs/path.py"}},
                {"tracks": {"x.py": "../escape.py"}},
                {"tracks": {"x.py": ""}},
                {"tracks": ["not", "a", "mapping"]}):
        with pytest.raises(jm.JobmetaError):
            jm.runset_preflight_cfg("witness-lifter", bad)


def test_runset_preflight_cfg_empty_when_undeclared():
    assert jm.runset_preflight_cfg("corpus-v4", {})["tracks"] == {}
    assert jm.runset_preflight_cfg("corpus-v4", None)["tracks"] == {}


def _train_args(**kw):
    base = dict(run="r1", runset="witness-lifter", disk=None, supervise=False,
                babysit=False, budget=None, no_asset_check=False,
                strict_assets=False, allow_stale_assets=False,
                with_eval=None, train_env_ver=None, fast_boot=True, image=None)
    base.update(kw)
    return argparse.Namespace(**base)


def _train_env(monkeypatch):
    # B2_BUCKET must be FakeB2Files' bucket: cmd_train threads the env bucket
    # into the preflight, and the fake asserts on the b2:<bucket>/ prefix.
    monkeypatch.setenv("B2_BUCKET", "bkt")
    for k in ("B2_KEY_ID", "B2_APPLICATION_KEY", "B2_S3_ENDPOINT"):
        monkeypatch.setenv(k, "x")


def _wire_train(monkeypatch, tmp_path, *, b2_body, local_body,
                cfg=RUNSET_CFG):
    """Point cmd_train's gate at a fake repo + fake B2. fast_boot=True with no
    --image makes step 3 (immediately AFTER the gate) sys.exit with a
    distinctive message, so 'passed the gate' is observable without letting the
    launcher touch the vast API."""
    root = tmp_path / "repo"
    for rel in set(cfg["tracks"].values()):
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(local_body)
    store = {f"runsets/witness-lifter/{k}": b2_body for k in cfg["tracks"]}
    fake = FakeB2Files(store)
    monkeypatch.setattr(_runsets, "_load_runset_config", lambda r: cfg)
    monkeypatch.setattr(submit, "_repo_root", lambda: str(root))
    monkeypatch.setattr(b2, "_ensure_b2_remote", lambda: None)
    monkeypatch.setattr(jm, "_default_runner", fake)
    return fake


def test_train_refuses_a_stale_staged_trainer(tmp_path, monkeypatch, capsys):
    """THE gate. A staged trainer that no longer matches the repo must FAIL the
    launch — loudly, and before a box is rented."""
    _train_env(monkeypatch)
    _wire_train(monkeypatch, tmp_path, b2_body="OLD\n", local_body="NEW\n")
    with pytest.raises(SystemExit) as e:
        cli_train.run(_train_args())
    err = capsys.readouterr().err
    assert "ASSET STALE" in err
    assert "refusing to launch" in str(e.value)
    assert "build.sh" in str(e.value)                  # actionable re-stage


def test_train_passes_when_the_staged_trainer_matches(tmp_path, monkeypatch, capsys):
    _train_env(monkeypatch)
    _wire_train(monkeypatch, tmp_path, b2_body="SAME\n", local_body="SAME\n")
    with pytest.raises(SystemExit) as e:
        cli_train.run(_train_args())
    assert "--fast-boot" in str(e.value)               # reached step 3 = gate passed
    assert "provenance OK" in capsys.readouterr().err  # and it SAID so


def test_train_allow_stale_assets_proceeds_loudly(tmp_path, monkeypatch, capsys):
    _train_env(monkeypatch)
    _wire_train(monkeypatch, tmp_path, b2_body="OLD\n", local_body="NEW\n")
    with pytest.raises(SystemExit) as e:
        cli_train.run(_train_args(allow_stale_assets=True))
    assert "--fast-boot" in str(e.value)
    assert "ASSET STALE" in capsys.readouterr().err    # loud, but not fatal


def test_train_no_asset_check_skips_the_read_entirely(tmp_path, monkeypatch):
    _train_env(monkeypatch)
    fake = _wire_train(monkeypatch, tmp_path, b2_body="OLD\n", local_body="NEW\n")
    with pytest.raises(SystemExit) as e:
        cli_train.run(_train_args(no_asset_check=True))
    assert "--fast-boot" in str(e.value)
    assert fake.calls == []


def test_train_unreadable_b2_never_blocks(tmp_path, monkeypatch, capsys):
    """An offline laptop / absent creds must degrade to a NOTE — the portable
    lane and a credentialless workstation cannot be blocked by this."""
    _train_env(monkeypatch)
    root = tmp_path / "repo"
    for rel in set(RUNSET_CFG["tracks"].values()):
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x\n")
    monkeypatch.setattr(_runsets, "_load_runset_config", lambda r: RUNSET_CFG)
    monkeypatch.setattr(submit, "_repo_root", lambda: str(root))
    monkeypatch.setattr(b2, "_ensure_b2_remote", lambda: None)
    monkeypatch.setattr(jm, "_default_runner", FakeB2Files(rc127=True))
    with pytest.raises(SystemExit) as e:
        cli_train.run(_train_args())
    assert "--fast-boot" in str(e.value)
    assert "UNVERIFIED" in capsys.readouterr().err


def test_train_undeclared_runset_says_so_out_loud(tmp_path, monkeypatch, capsys):
    """A runset with no `tracks:` must NOT pass in silence — a quiet pass is
    indistinguishable from a check that never ran, which is precisely how this
    gap survived three months."""
    _train_env(monkeypatch)
    monkeypatch.setattr(_runsets, "_load_runset_config", lambda r: {})
    monkeypatch.setattr(submit, "_repo_root", lambda: str(tmp_path))
    def _boom():
        raise AssertionError("must not touch B2 with nothing declared")
    monkeypatch.setattr(b2, "_ensure_b2_remote", _boom)
    with pytest.raises(SystemExit) as e:
        cli_train.run(_train_args())
    assert "--fast-boot" in str(e.value)
    assert "declares no `tracks:`" in capsys.readouterr().err


def test_train_malformed_tracks_refuses(tmp_path, monkeypatch):
    _train_env(monkeypatch)
    monkeypatch.setattr(_runsets, "_load_runset_config",
                        lambda r: {"tracks": {"x.py": "/abs/nope.py"}})
    with pytest.raises(SystemExit, match="config.yaml"):
        cli_train.run(_train_args())


# --------------------------------------------------------------------------- #
# Submit-time supervision advisory (box 47511739, 2026-08-12)
# --------------------------------------------------------------------------- #
# A `jobs` watch is scoped to the queue it was armed over. When every ticket
# goes terminal it emits `watch_finished {verdict: drained}` -- a NORMAL
# completion -- and the stray sweep re-adopts the box as `bare`, carrying the
# spend ceiling forward but not the ladder. Box 47511739 then took three more
# submits, on SPOT, with no outbid rescue and no eviction replacement. fleetd
# raised the right alarm and it went unread, so the notice belongs on the
# submit the operator is actually typing.
#
# ADVISORY, NEVER A REFUSAL: `launch_jobs_box.sh` submits BEFORE it watches, and
# fleetd keeps a jobs watch alive through `queue_empty` so that arming early
# does not park the box. "No watch yet" is the CORRECT state at submit time.

def _state_dir(tmp_path, payload):
    d = tmp_path / "fleetd-state"
    d.mkdir(exist_ok=True)
    if payload is not None:
        (d / "state.json").write_text(payload if isinstance(payload, str)
                                      else __import__("json").dumps(payload))
    os.environ["FLEETD_STATE_DIR"] = str(d)
    return d


@pytest.fixture(autouse=False)
def _clean_fleet_env():
    old = os.environ.get("FLEETD_STATE_DIR")
    yield
    if old is None:
        os.environ.pop("FLEETD_STATE_DIR", None)
    else:
        os.environ["FLEETD_STATE_DIR"] = old


@pytest.mark.parametrize("name,payload,iid,want", [
    ("policy jobs watch",
     {"watches": {"47511739": {"iid": 47511739, "profile": "jobs",
                               "budget_usd": 4.0, "spend_usd": 1.5388}},
      "ceiling_by_box": {"47511739": 1.5388}}, 47511739, "policy"),
    # the exact shape box 47511739 was in when the three probes were submitted
    ("lapsed: adopted bare holding an inherited ceiling",
     {"watches": {"47511739": {"iid": 47511739, "profile": "bare",
                               "budget_usd": 4.0, "spend_usd": 1.5388,
                               "adopted": True, "ceiling_source": "inherited"}},
      "ceiling_by_box": {"47511739": 1.5388}}, 47511739, "lapsed"),
    ("adopted bare on a provisional ceiling",
     {"watches": {"9": {"iid": 9, "profile": "bare", "budget_usd": 2.0,
                        "adopted": True, "ceiling_source": "default"}}}, 9, "bare"),
    ("watch exists but for a DIFFERENT box (fresh launch)",
     {"watches": {"1": {"iid": 1, "profile": "jobs"}}}, 47511739, "none"),
    ("empty watch set -- the null vector",
     {"watches": {}}, 47511739, "none"),
    ("no state file at all: fleetd never ran",
     None, 47511739, "unknown"),
    ("unparseable state file",
     "{not json", 47511739, "unknown"),
])
def test_fleet_watch_supervision_levels(tmp_path, _clean_fleet_env, name,
                                        payload, iid, want):
    _state_dir(tmp_path, payload)
    level, _detail = fleet_client.fleet_watch_supervision(iid)
    assert level == want, f"{name}: got {level!r}, want {want!r}"


def test_lapsed_advisory_names_the_ladder_and_the_remaining_headroom(
        tmp_path, _clean_fleet_env, capsys):
    """The warning has to say the two things that are NOT obvious: the ceiling
    survived (so this is not "unbudgeted"), and the ladder did not (so a spot
    box can lose the work silently). Numbers are box 47511739's real ones."""
    _state_dir(tmp_path, {
        "watches": {"47511739": {"iid": 47511739, "profile": "bare",
                                 "budget_usd": 4.0, "spend_usd": 1.5388,
                                 "adopted": True, "ceiling_source": "inherited"}},
        "ceiling_by_box": {"47511739": 1.5388}})
    submit._print_submit_supervision(47511739, "herdd")
    err = capsys.readouterr().err
    assert "$2.46 of $4.00 left" in err          # ceiling survived, and by how much
    assert "LADDER did not" in err
    assert "fleet watch 47511739 --profile jobs" in err


def test_no_watch_advises_but_does_not_warn(tmp_path, _clean_fleet_env, capsys):
    """rent -> submit -> arm is the DOCUMENTED order, so a fresh box with no
    watch must produce a note on stdout, never a warning on stderr -- otherwise
    every correct launch_jobs_box.sh run cries wolf and the real one gets lost."""
    _state_dir(tmp_path, {"watches": {}})
    submit._print_submit_supervision(47511739, "herdd")
    cap = capsys.readouterr()
    assert "no fleet watch yet" in cap.out
    assert "never before" in cap.out            # names the park hazard
    assert cap.err == ""


def test_unknown_state_is_silent_and_never_blocks(tmp_path, _clean_fleet_env,
                                                  capsys):
    """An unreadable state file is not evidence. It must print nothing at all:
    a line that fires on every submit is a line nobody reads."""
    _state_dir(tmp_path, None)
    submit._print_submit_supervision(47511739, "herdd")
    cap = capsys.readouterr()
    assert cap.out == "" and cap.err == ""


# --------------------------------------------------------------------------- #
# software-epoch stamp: `job submit` records the CLIENT checkout's HEAD in the
# ticket env, because the box receives a tar with no .git and nothing on the far
# side can otherwise name the software that produced the run.
# --------------------------------------------------------------------------- #
def _git_init(root):
    import subprocess
    env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@e",
               GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@e")
    (pathlib.Path(root) / "f.txt").write_text("one\n")
    # like the real checkout: the bundle staging dir lives under out/, which is
    # ignored — otherwise every submit would stamp itself `-dirty`.
    (pathlib.Path(root) / ".gitignore").write_text("out/\n")
    for c in (["init", "-q"], ["add", "f.txt", ".gitignore"], ["commit", "-qm", "c"]):
        subprocess.run(["git", "-C", str(root)] + c, check=True, env=env,
                       capture_output=True)
    return subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()


def _submit_to_the_stamp(monkeypatch, tmp_path, args):
    """Run cmd_job_submit past the bundle to the stamp, then stop. Returns
    (validated cfg, HEAD of the fake client checkout)."""
    fake_repo = tmp_path / "checkout"
    fake_repo.mkdir()
    head = _git_init(fake_repo)
    monkeypatch.setattr(submit, "_repo_root", lambda: str(fake_repo))
    monkeypatch.setattr(b2, "_ensure_b2_remote", lambda: None)

    seen = {}
    real = jm.validate_job_config

    def _spy(raw, src):
        cfg, warns = real(raw, src)
        seen["cfg"] = cfg
        return cfg, warns
    monkeypatch.setattr(jm, "validate_job_config", _spy)

    def _fake_bundle(src, out):
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "wb") as fh:
            fh.write(b"bundle")
        return {"sha256": "a" * 64, "zst_size": 6, "tar_size": 12}
    monkeypatch.setattr(jm, "write_bundle", _fake_bundle)
    monkeypatch.setattr(jm, "mint_job_id",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("REACHED_STAMP")))
    with pytest.raises(RuntimeError, match="REACHED_STAMP"):
        submit.cmd_job_submit(args)
    return seen["cfg"], head


def test_submit_stamps_the_client_head_into_the_ticket_env(tmp_path, monkeypatch):
    cfg, head = _submit_to_the_stamp(monkeypatch, tmp_path,
                                     _args(_envjob(tmp_path)))
    assert cfg["env"][jm.TRAINER_REV_ENV] == head
    assert cfg["env"]["TARGET"] == "dc3"        # the job's own env survives


def test_submit_stamp_lands_after_bundling(tmp_path, monkeypatch):
    """Ordering is the contract: the stamp rides the TICKET, never the tar. A
    bundle addressed over a HEAD-dependent env would miss the dedupe on every
    commit, and it would also break the `--env`-merge pins above."""
    at_bundle = {}
    fake_repo = tmp_path / "checkout"
    fake_repo.mkdir()
    _git_init(fake_repo)
    monkeypatch.setattr(submit, "_repo_root", lambda: str(fake_repo))
    monkeypatch.setattr(b2, "_ensure_b2_remote", lambda: None)
    seen = {}
    real = jm.validate_job_config

    def _spy(raw, src):
        cfg, warns = real(raw, src)
        seen["cfg"] = cfg
        return cfg, warns
    monkeypatch.setattr(jm, "validate_job_config", _spy)

    def _fake_bundle(src, out):
        at_bundle["env"] = dict(seen["cfg"]["env"])
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "wb") as fh:
            fh.write(b"bundle")
        return {"sha256": "a" * 64, "zst_size": 6, "tar_size": 12}
    monkeypatch.setattr(jm, "write_bundle", _fake_bundle)
    monkeypatch.setattr(jm, "mint_job_id",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("REACHED_STAMP")))
    with pytest.raises(RuntimeError, match="REACHED_STAMP"):
        submit.cmd_job_submit(_args(_envjob(tmp_path)))
    assert jm.TRAINER_REV_ENV not in at_bundle["env"]
    assert jm.TRAINER_REV_ENV in seen["cfg"]["env"]


def test_submit_off_a_git_checkout_leaves_the_env_unstamped(tmp_path, monkeypatch):
    """A submit from an exported tree degrades to UNMEASURED — absent, not ''."""
    nogit = tmp_path / "nogit"
    nogit.mkdir()
    monkeypatch.setattr(submit, "_repo_root", lambda: str(nogit))
    monkeypatch.setattr(b2, "_ensure_b2_remote", lambda: None)
    seen = {}
    real = jm.validate_job_config

    def _spy(raw, src):
        cfg, warns = real(raw, src)
        seen["cfg"] = cfg
        return cfg, warns
    monkeypatch.setattr(jm, "validate_job_config", _spy)

    def _fake_bundle(src, out):
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "wb") as fh:
            fh.write(b"bundle")
        return {"sha256": "a" * 64, "zst_size": 6, "tar_size": 12}
    monkeypatch.setattr(jm, "write_bundle", _fake_bundle)
    monkeypatch.setattr(jm, "mint_job_id",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("REACHED_STAMP")))
    with pytest.raises(RuntimeError, match="REACHED_STAMP"):
        submit.cmd_job_submit(_args(_envjob(tmp_path)))
    assert jm.TRAINER_REV_ENV not in seen["cfg"]["env"]

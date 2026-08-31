"""Portable tests for jobmeta.py — pure validation + fold + deterministic-hash +
a fake in-memory B2 runner. Runs in the toolchain-free lane (`pytest -m "not
integration"`): no rclone, no B2, no network, no creds.
"""
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jobmeta as jm  # noqa: E402


# --------------------------------------------------------------------------- #
# fixtures / helpers
# --------------------------------------------------------------------------- #
def _mkjob(tmp_path, name="probe-01", body="echo hi\n", extra=True):
    d = tmp_path / "job"
    (d / "sub").mkdir(parents=True)
    (d / "run.sh").write_text(body)
    (d / "sub" / "a.txt").write_text("data\n")
    if extra:
        (d / "job-config.yaml").write_text(
            "version: 1\n"
            f"name: {name}\n"
            "entrypoint: run.sh\n"
            "timeout_s: 60\n"
            "env:\n  FOO: \"bar\"\n  N: \"3\"\n"
            "results:\n  - \"out/**\"\n  - \"*.jsonl\"\n"
            "needs:\n  gpu: true\n  venv: serve\n")
    return d


def ev(event, ts, job_id="j", nonce=None, actor="box:7", **fields):
    d = {"v": 1, "ts": ts, "actor": actor, "event": event, "job_id": job_id,
         "nonce": nonce or (ts[-4:] + event[:2])}
    d.update(fields)
    return d


def T(n):
    return f"20260710T0000{n:02d}000Z"


# --------------------------------------------------------------------------- #
# id / slug validation
# --------------------------------------------------------------------------- #
def test_validate_job_id():
    for bad in ["", "a b", "a/b", "job:x", "x" * 65]:
        with pytest.raises(jm.JobmetaError):
            jm.validate_job_id(bad)
    for ok in ["20260710T074503-my-probe-1-0f91", "A" * 64]:
        assert jm.validate_job_id(ok) == ok


def test_slugify_and_mint():
    assert jm.slugify("My Probe-01!!") == "my-probe-01"
    with pytest.raises(jm.JobmetaError):
        jm.slugify("!!!")
    jid = jm.mint_job_id("Qwen4B Reason Probe")
    assert jm.JOB_ID_RE.match(jid)
    assert "-qwen4b-reason-probe-" in jid
    # deterministic prefix/nonce for the shape assertion
    jid2 = jm.mint_job_id("x", ts="20260710T074503", nonce4="abcd")
    assert jid2 == "20260710T074503-x-abcd"


# --------------------------------------------------------------------------- #
# yaml fallback parser (force the no-PyYAML path)
# --------------------------------------------------------------------------- #
def test_yaml_fallback_parser(monkeypatch):
    # simulate PyYAML absent so the targeted subset parser is exercised
    monkeypatch.setitem(sys.modules, "yaml", None)
    text = ("version: 1\nname: p\nentrypoint: run.sh\ntimeout_s: 42\n"
            "env:\n  A: \"1\"\n  B: two\n"
            "results:\n  - \"out/**\"\n  - x.jsonl\n"
            "needs:\n  gpu: true\n  venv: eval\n")
    d = jm._parse_job_yaml(text)
    assert d["version"] == 1 and d["name"] == "p" and d["timeout_s"] == 42
    assert d["env"] == {"A": "1", "B": "two"}
    assert d["results"] == ["out/**", "x.jsonl"]
    assert d["needs"] == {"gpu": True, "venv": "eval"}


def test_yaml_fallback_strips_inline_comments(monkeypatch):
    """YAML drops `<ws>#...` on unquoted scalars; the fallback must too — a
    trailing comment on needs.gpu_ram_gb crash-looped a live controller for 6h
    on a no-PyYAML interpreter (2026-07-30, run 2ed9). Quoted values keep '#'."""
    monkeypatch.setitem(sys.modules, "yaml", None)
    text = ("version: 1   # schema rev\n"
            "name: p\n"
            "entrypoint: run.sh\n"
            "needs:   # nested block header with comment\n"
            "  gpu: false  # CPU-only scoring\n"
            "  gpu_ram_gb: 24        # bf16 7B weights ~15 GB + KV cache\n"
            "env:\n"
            "  CHAN: \"#alerts\"   # quoted hash survives\n"
            "results:\n"
            "  - out/**   # per-arm outputs\n")
    d = jm._parse_job_yaml(text)
    assert d["version"] == 1
    assert d["needs"] == {"gpu": False, "gpu_ram_gb": 24}
    assert d["env"] == {"CHAN": "#alerts"}
    assert d["results"] == ["out/**"]


def test_yaml_fallback_list_of_mappings_and_flow_lists(monkeypatch):
    """The `assets` schema: a list of mappings with flow-list values. The
    fallback couldn't represent it and crash-looped run 2ed9's controller a
    SECOND time (2026-07-30) right after the comment fix. Differential-tested
    against PyYAML below when it's installed."""
    monkeypatch.setitem(sys.modules, "yaml", None)
    text = ("version: 1\nname: p\nentrypoint: run.sh\n"
            "assets:\n"
            "  - name: base\n"
            "    b2: base-models/m\n"
            "    require: [config.json, \"*.safetensors\", tokenizer.json]\n"
            "  - name: extra\n"
            "    optional: true\n"
            "results:\n  - out/**\n")
    d = jm._parse_job_yaml(text)
    assert d["assets"] == [
        {"name": "base", "b2": "base-models/m",
         "require": ["config.json", "*.safetensors", "tokenizer.json"]},
        {"name": "extra", "optional": True},
    ]
    assert d["results"] == ["out/**"]


def test_yaml_fallback_differential_on_real_bundles(monkeypatch):
    """Strongest guard: the fallback must parse the ACTUAL e2-paired bundle
    configs identically to PyYAML — the schema of record, not a toy."""
    yaml_mod = pytest.importorskip("yaml")
    here = os.path.dirname(os.path.abspath(__file__))
    for rel in ("../witness/jobs/e2-paired-gen/job-config.yaml",
                "../witness/jobs/e2-paired-score/job-config.yaml",
                # v7 carries a FLOW-mapping `tracks:` inside an assets item —
                # the one map shape the fallback represents (a block map nested
                # in a list item would be mis-attached), so pin the parity.
                "../witness/jobs/v7-longctx-train/job-config.yaml"):
        path = os.path.normpath(os.path.join(here, rel))
        if not os.path.isfile(path):
            pytest.skip(f"bundle config not present: {path}")
        text = open(path).read()
        ref = yaml_mod.safe_load(text)
        monkeypatch.setitem(sys.modules, "yaml", None)
        assert jm._parse_job_yaml(text) == ref
        monkeypatch.setitem(sys.modules, "yaml", yaml_mod)


# --------------------------------------------------------------------------- #
# config validation
# --------------------------------------------------------------------------- #
def test_validate_config_ok(tmp_path):
    d = _mkjob(tmp_path)
    cfg, warn = jm.validate_job_config(jm.load_job_config(str(d)), str(d))
    assert cfg["name"] == "probe-01" and cfg["entrypoint"] == "run.sh"
    assert cfg["timeout_s"] == 60 and cfg["env"] == {"FOO": "bar", "N": "3"}
    assert cfg["results"] == ["out/**", "*.jsonl"]
    # v2: a plain GPU job defaults to 1 scheduled card
    assert cfg["needs"] == {"gpu": True, "venv": "serve", "gpus": 1} and warn == []


def test_validate_config_json_alias(tmp_path):
    d = tmp_path / "j"
    d.mkdir()
    (d / "run.sh").write_text("echo hi\n")
    (d / "job-config.json").write_text(json.dumps(
        {"version": 1, "name": "p", "entrypoint": "run.sh"}))
    cfg, _ = jm.validate_job_config(jm.load_job_config(str(d)), str(d))
    assert cfg["timeout_s"] == jm.DEFAULT_TIMEOUT_S    # default filled


@pytest.mark.parametrize("mutate,msg", [
    (lambda c: c.pop("name"), "name"),
    (lambda c: c.pop("entrypoint"), "entrypoint"),
    (lambda c: c.__setitem__("entrypoint", "/abs/run.sh"), "escape"),
    (lambda c: c.__setitem__("entrypoint", "../run.sh"), "escape"),
    (lambda c: c.__setitem__("entrypoint", "missing.sh"), "not found"),
    (lambda c: c.__setitem__("timeout_s", 0), "positive"),
    (lambda c: c.__setitem__("timeout_s", 999999999), "exceeds"),
    (lambda c: c.__setitem__("results", ["/abs/x"]), "relative"),
    (lambda c: c.__setitem__("needs", {"venv": "bogus"}), "needs.venv"),
    (lambda c: c.__setitem__("needs", {"gpu_ram_gb": 0}), "needs.gpu_ram_gb"),
    (lambda c: c.__setitem__("needs", {"gpu_ram_gb": True}), "needs.gpu_ram_gb"),
    (lambda c: c.__setitem__("needs", {"gpu_ram_gb": "big"}), "needs.gpu_ram_gb"),
    (lambda c: c.__setitem__("needs", {"scratch_gb": 0}), "needs.scratch_gb"),
    (lambda c: c.__setitem__("needs", {"scratch_gb": -4}), "needs.scratch_gb"),
    (lambda c: c.__setitem__("needs", {"scratch_gb": "lots"}), "needs.scratch_gb"),
    (lambda c: c.__setitem__("needs", {"scratch_gb": True}), "needs.scratch_gb"),
    (lambda c: c.__setitem__("needs", {"disk_gb": 0}), "needs.disk_gb"),
    (lambda c: c.__setitem__("needs", {"disk_gb": "big"}), "needs.disk_gb"),
    (lambda c: c.__setitem__("needs", {"scratch_volatile": "yes"}),
     "needs.scratch_volatile"),
    (lambda c: c.__setitem__("needs", {"scratch_volatile": True}),
     "nothing to place"),
    (lambda c: c.__setitem__("needs", {"cpu_cores": 0}), "needs.cpu_cores"),
    (lambda c: c.__setitem__("needs", {"cpu_cores": -8}), "needs.cpu_cores"),
    (lambda c: c.__setitem__("needs", {"cpu_cores": True}), "needs.cpu_cores"),
    (lambda c: c.__setitem__("needs", {"cpu_cores": "many"}), "needs.cpu_cores"),
    (lambda c: c.__setitem__("needs", {"host_ram_gb": 0}), "needs.host_ram_gb"),
    (lambda c: c.__setitem__("needs", {"host_ram_gb": -96}), "needs.host_ram_gb"),
    (lambda c: c.__setitem__("needs", {"host_ram_gb": True}), "needs.host_ram_gb"),
    (lambda c: c.__setitem__("needs", {"host_ram_gb": "lots"}), "needs.host_ram_gb"),
    (lambda c: c.__setitem__("needs", {"cc_allow": "80,90"}), "needs.cc_allow"),
    (lambda c: c.__setitem__("needs", {"cc_allow": ["hopper"]}), "needs.cc_allow"),
    (lambda c: c.__setitem__("needs", {"cc_allow": [0]}), "needs.cc_allow"),
])
def test_validate_config_failures(tmp_path, mutate, msg):
    d = _mkjob(tmp_path)
    raw = jm.load_job_config(str(d))
    mutate(raw)
    with pytest.raises(jm.JobmetaError) as e:
        jm.validate_job_config(raw, str(d))
    assert msg in str(e.value)


def test_validate_config_gpu_ram_gb(tmp_path):
    # a VRAM floor is carried into the canonical needs and implies needs.gpu
    d = _mkjob(tmp_path)
    raw = jm.load_job_config(str(d))
    raw["needs"] = {"gpu_ram_gb": 48}          # no explicit gpu: should be implied
    cfg, _ = jm.validate_job_config(raw, str(d))
    assert cfg["needs"]["gpu_ram_gb"] == 48
    assert cfg["needs"]["gpu"] is True
    # absent -> key omitted from canonical needs
    raw2 = jm.load_job_config(str(d))
    raw2["needs"] = {"gpu": True, "venv": "serve"}
    cfg2, _ = jm.validate_job_config(raw2, str(d))
    assert "gpu_ram_gb" not in cfg2["needs"]


def test_validate_config_cpu_cores(tmp_path):
    """A core COUNT, and — unlike gpu_ram_gb — it must NOT imply needs.gpu.

    The case this exists for is a dedicated CPU bundle (compile/search work,
    no model endpoint). Implying a GPU would rent one for a job that never
    touches it, which is the shape `_job_shape` already calls "CPU"."""
    d = _mkjob(tmp_path)
    raw = jm.load_job_config(str(d))
    raw["needs"] = {"cpu_cores": 64}
    cfg, _ = jm.validate_job_config(raw, str(d))
    assert cfg["needs"]["cpu_cores"] == 64
    assert cfg["needs"]["gpu"] is False
    # absent -> key omitted from canonical needs
    raw2 = jm.load_job_config(str(d))
    raw2["needs"] = {"gpu": True}
    cfg2, _ = jm.validate_job_config(raw2, str(d))
    assert "cpu_cores" not in cfg2["needs"]


def test_validate_config_host_ram_gb(tmp_path):
    """The CPU-side twin of gpu_ram_gb: a host-RAM floor in GB that, like
    cpu_cores, must NOT imply needs.gpu — a bf16 CPU merge is the case."""
    d = _mkjob(tmp_path)
    raw = jm.load_job_config(str(d))
    raw["needs"] = {"host_ram_gb": 96}
    cfg, _ = jm.validate_job_config(raw, str(d))
    assert cfg["needs"]["host_ram_gb"] == 96.0
    assert cfg["needs"]["gpu"] is False
    # absent -> key omitted from canonical needs
    raw2 = jm.load_job_config(str(d))
    raw2["needs"] = {"gpu": True}
    cfg2, _ = jm.validate_job_config(raw2, str(d))
    assert "host_ram_gb" not in cfg2["needs"]


def test_validate_config_host_ram_gb_is_a_float(tmp_path):
    """RAM tiers are not integers — a "128 GB" host rents a 126 GB slice — so
    the declaration must be able to say 125.5 and reach the ticket as a float."""
    d = _mkjob(tmp_path)
    raw = jm.load_job_config(str(d))
    raw["needs"] = {"host_ram_gb": 125.5}
    cfg, _ = jm.validate_job_config(raw, str(d))
    assert cfg["needs"]["host_ram_gb"] == 125.5
    assert isinstance(cfg["needs"]["host_ram_gb"], float)


def test_validate_config_cc_allow(tmp_path):
    """needs.cc_allow is the bundle's own architecture allowlist: it reaches the
    canonical needs sorted and de-duped, implies needs.gpu, and accepts the
    `sm_90` / compute_cap (900) spellings people actually type."""
    d = _mkjob(tmp_path)
    raw = jm.load_job_config(str(d))
    raw["needs"] = {"cc_allow": [90, 80, "sm_86", 890, 80]}
    cfg, _ = jm.validate_job_config(raw, str(d))
    assert cfg["needs"]["cc_allow"] == [80, 86, 89, 90]
    assert cfg["needs"]["gpu"] is True


@pytest.mark.parametrize("needs", [{"gpu": True}, {"gpu": True, "cc_allow": []}])
def test_validate_config_cc_allow_absent_or_empty_is_unconstrained(tmp_path, needs):
    """Absent and empty must both mean NO CONSTRAINT — the pre-2026-08-19
    behaviour every arch-agnostic bundle still relies on. An empty list left in
    the ticket would invite a reader to treat it as an allowlist of nothing."""
    d = _mkjob(tmp_path)
    raw = jm.load_job_config(str(d))
    raw["needs"] = dict(needs)
    cfg, _ = jm.validate_job_config(raw, str(d))
    assert "cc_allow" not in cfg["needs"]


def test_validate_config_disk_knobs(tmp_path):
    """needs.scratch_gb (working state the entrypoint CREATES — build trees,
    per-worker worktrees) and needs.disk_gb (override the derivation outright)
    reach the canonical config so disksize can consume them. Neither implies a
    GPU, and both stay absent when undeclared."""
    d = _mkjob(tmp_path)
    raw = jm.load_job_config(str(d))
    raw["needs"] = {"scratch_gb": 60, "disk_gb": 200}
    cfg, _ = jm.validate_job_config(raw, str(d))
    assert cfg["needs"]["scratch_gb"] == 60.0
    assert cfg["needs"]["disk_gb"] == 200.0
    assert cfg["needs"]["gpu"] is False, "a disk figure must not imply a GPU"

    raw2 = jm.load_job_config(str(d))
    raw2["needs"] = {"venv": "none"}
    cfg2, _ = jm.validate_job_config(raw2, str(d))
    assert "scratch_gb" not in cfg2["needs"] and "disk_gb" not in cfg2["needs"]


def test_validate_config_symlink_escape(tmp_path):
    d = _mkjob(tmp_path)
    outside = tmp_path / "secret.txt"
    outside.write_text("s")
    os.symlink(str(outside), str(d / "leak"))
    with pytest.raises(jm.JobmetaError) as e:
        jm.validate_job_config(jm.load_job_config(str(d)), str(d))
    assert "escapes" in str(e.value)


# --------------------------------------------------------------------------- #
# deterministic content-addressed bundling  (the required verification)
# --------------------------------------------------------------------------- #
def test_deterministic_hash_identical_folders(tmp_path):
    """Two byte-identical folders built independently (different mtimes/owners)
    hash identically — the zeroed-mtime/uid/gid property."""
    a = tmp_path / "a"
    b = tmp_path / "b"
    for root in (a, b):
        (root / "sub").mkdir(parents=True)
        (root / "run.sh").write_text("echo hi\n")
        (root / "sub" / "a.txt").write_text("data\n")
    # skew b's mtimes hard — must NOT affect the hash
    os.utime(str(b / "run.sh"), (1, 1))
    os.utime(str(b / "sub" / "a.txt"), (123456, 123456))
    assert jm.bundle_sha256(str(a)) == jm.bundle_sha256(str(b))


def test_hash_changes_on_content(tmp_path):
    a = _mkjob(tmp_path, extra=False)
    s1 = jm.bundle_sha256(str(a))
    (a / "run.sh").write_text("echo DIFFERENT\n")
    assert jm.bundle_sha256(str(a)) != s1


def test_bundle_roundtrip_and_sha_verify(tmp_path):
    d = _mkjob(tmp_path, extra=False)
    sha = jm.bundle_sha256(str(d))
    out = tmp_path / "bundle.tar.zst"
    info = jm.write_bundle(str(d), str(out))
    assert info["sha256"] == sha
    dest = tmp_path / "extracted"
    got = jm.extract_bundle(str(out), str(dest), expect_sha=sha)
    assert got == sha
    assert (dest / "run.sh").read_text() == "echo hi\n"
    assert (dest / "sub" / "a.txt").read_text() == "data\n"
    # tamper -> sha mismatch raises
    with pytest.raises(jm.JobmetaError):
        jm.extract_bundle(str(out), str(tmp_path / "x"), expect_sha="0" * 64)


def test_extract_rejects_escaping_member(tmp_path):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        data = b"pwn"
        ti = tarfile.TarInfo("../evil.txt")
        ti.size = len(data)
        tf.addfile(ti, io.BytesIO(data))
    blob = jm.compress_tar(buf.getvalue())
    with pytest.raises(jm.JobmetaError) as e:
        jm.extract_bundle(blob, str(tmp_path / "dest"))
    assert "escapes" in str(e.value)


def test_store_fallback_roundtrip():
    # decompress auto-detects a raw (non-zstd) tar so a store-only blob works
    raw = b"not-a-zstd-frame-just-bytes"
    assert jm.decompress_zst(raw) == raw


def test_decompress_reads_a_cli_written_frame_without_content_size():
    """The two zstd implementations must interoperate in BOTH directions.

    compress_tar uses the `zstd` CLI on any host without the `zstandard`
    module; the CLI writes a STREAMING frame with no content size in the
    header. zstandard's one-shot .decompress() requires that field, so a
    bundle built on a workstation without the module could not be extracted
    on a box image that has it — which is the ordinary split, and it failed
    only AFTER the box was rented (box 48089639, 2026-08-19: "could not
    determine content size in frame header").
    """
    zstd_cli = shutil.which("zstd")
    if not zstd_cli:
        pytest.skip("zstd CLI not installed — cannot forge a streaming frame")
    payload = b"tar-ish payload " * 500
    p = subprocess.run([zstd_cli, "-q", "-10", "-c"], input=payload,
                       stdout=subprocess.PIPE, check=True)
    frame = p.stdout
    assert frame.startswith(jm._ZSTD_MAGIC)
    # Pin the precondition: a frame the one-shot API cannot read. If a future
    # CLI starts embedding the size this assert fires and the case is moot.
    try:
        import zstandard
    except ImportError:
        pass
    else:
        with pytest.raises(zstandard.ZstdError):
            zstandard.ZstdDecompressor().decompress(frame)
    assert jm.decompress_zst(frame) == payload


# --------------------------------------------------------------------------- #
# fold
# --------------------------------------------------------------------------- #
def test_fold_ordering_and_lifecycle():
    a = ev("submitted", T(1), actor="cli:h", bundle_sha256="deadbeef",
           name="p", entrypoint="run.sh", timeout_s=60, box="44")
    b = ev("claimed", T(2), instance_id="44")
    c = ev("started", T(3), instance_id="44")
    d = ev("done", T(9), rc=0, results=[{"path": "out/x", "size": 3}])
    v1 = jm.fold_events([a, b, c, d], live_iids={"44"})
    v2 = jm.fold_events([d, b, a, c], live_iids={"44"})   # shuffled
    assert v1 == v2
    assert v1["status"] == "done" and v1["display_status"] == "done"
    assert v1["bundle_sha256"] == "deadbeef" and v1["entrypoint"] == "run.sh"
    assert v1["instance_id"] == "44" and v1["rc"] == 0
    assert v1["results"] == [{"path": "out/x", "size": 3}]


def test_fold_gpu_shape_is_tristate():
    """`submitted.gpu` folds to view["gpu"]: True/False when stamped, None on a
    pre-2026-08-27 stream — readers render None as unknown, never as CPU."""
    old = [ev("submitted", T(1), actor="cli:h", box="44")]
    assert jm.fold_events(old)["gpu"] is None
    cpu = [ev("submitted", T(1), actor="cli:h", box="44", gpu=False)]
    assert jm.fold_events(cpu)["gpu"] is False
    gpu = [ev("submitted", T(1), actor="cli:h", box="44", gpu=True)]
    assert jm.fold_events(gpu)["gpu"] is True


def test_fold_terminal_precedence():
    started = ev("started", T(2), instance_id="1")
    done = ev("done", T(5), rc=0)
    stale = ev("started", T(9), instance_id="1")     # sorts AFTER done
    assert jm.fold_events([started, done, stale])["status"] == "done"
    failed = ev("failed", T(6), rc=2, reason="boom")
    v = jm.fold_events([started, done, failed], live_iids={"1"})
    assert v["status"] == "failed"    # failed NEWER than done -> failed stands
    assert v["rc"] == 2 and v["fail_reason"] == "boom"


def test_fold_interrupted_when_box_dead():
    # v2: a claimed/started job on a dead box is INTERRUPTED (jobd resumes it on
    # the next boot / `job retarget` moves it), not the v1 `lost` dead-end.
    evs = [ev("submitted", T(1), actor="cli:h", box="44"),
           ev("claimed", T(2), instance_id="44"),
           ev("started", T(3), instance_id="44")]
    dead = jm.fold_events(evs, live_iids=())          # box 44 gone
    assert dead["status"] == "started" and dead["display_status"] == "interrupted"
    alive = jm.fold_events(evs, live_iids={"44"})
    assert alive["display_status"] == "running" and alive["live"] is True


def test_fold_submitted_is_not_interrupted():
    # queued but not yet claimed: box may just be booting -> "submitted"
    v = jm.fold_events([ev("submitted", T(1), actor="cli:h", box="44")], live_iids=())
    assert v["status"] == "submitted" and v["display_status"] == "submitted"


def test_fold_target_box_follows_the_newest_retarget():
    """`retargeted` MOVES the ticket (the old one is deleted), so submitted.box is
    only the ORIGINAL target. Observed 2026-08-02: job status placed
    20260715T081939-68b3b57b-generate-a0 on 44960616 while its ticket sat under
    44967157, pointing a reader at a box that no longer held it. Load-bearing now
    that the boot-pull watchdog retargets every ticket it reschedules."""
    evs = [ev("submitted", T(1), actor="cli:h", box="44960616"),
           ev("retargeted", T(2), actor="cli:h", box="44967157",
              from_box="44960616")]
    v = jm.fold_events(evs, live_iids=())
    assert v["target_box"] == "44967157"
    assert v["retargeted_from"] == "44960616"
    assert v["status"] == "submitted"          # a moved pointer is not execution
    # chained moves take the LAST hop, not the first or the original
    evs.append(ev("retargeted", T(3), actor="cli:h", box="45000001",
                  from_box="44967157"))
    assert jm.fold_events(evs, live_iids=())["target_box"] == "45000001"


def test_fold_target_box_unchanged_without_a_retarget():
    """Inertness: the ordinary path must be byte-identical to before the fix."""
    v = jm.fold_events([ev("submitted", T(1), actor="cli:h", box="44")],
                       live_iids=())
    assert v["target_box"] == "44" and "retargeted_from" not in v


def test_fold_attempts_preempted_resumed():
    """A `resumed` with NO `kind` field (older/synthetic stream, predating the
    per-resume `kind` jobd always emits today) is inert for the attempts fold —
    conservative default, byte-identical to the pre-2026-08-09 behavior: attempts
    = count of `started`, and the orphan `preempted` (no matching
    `resumed{kind:preempt}` after it) still counts toward n_preempted."""
    evs = [ev("claimed", T(1), instance_id="44"),
           ev("started", T(2), instance_id="44"),
           ev("preempted", T(3), instance_id="44"),
           ev("resumed", T(4), instance_id="55"),
           ev("started", T(5), instance_id="55")]
    v = jm.fold_events(evs, live_iids={"55"})
    assert v["attempts"] == 2 and v["n_preempted"] == 1
    assert v["last_resumed_ts"] == T(4)
    assert v["instance_id"] == "55" and v["display_status"] == "running"
    assert v["status"] not in jm.TERMINAL
    # terminal stays sticky over the interruption chatter
    v2 = jm.fold_events(evs + [ev("done", T(6), instance_id="55", rc=0)],
                        live_iids=())
    assert v2["status"] == "done" and v2["display_status"] == "done"


# --------------------------------------------------------------------------- #
# 2026-08-09 drill: `attempts`/`n_preempted` must mirror the durable on-box
# counters (.attempts/.preempts), not double-count the relaunch `started` that
# follows every resume. jobd ALWAYS stamps `kind` on `resumed` (onstart/
# jobd.sh: "preempt"|"crash"|"retarget"|"requeue") — the fold keys off that,
# not off the (often-absent) trap `preempted` event, because vast delivers no
# signal on eviction and the trap almost never fires for a real preempt.
# --------------------------------------------------------------------------- #
def test_fold_attempts_signalless_preempt_drill_shape():
    """THE drill shape: a real vast eviction/park with NO signal reaching the
    daemon — no trap `preempted`, just jobd's own `resumed{kind:preempt}`
    (boot-nonce inference) ahead of the relaunch `started`. Durable evidence
    from the 2026-08-09 drill: attempts=1, n_preempted=1 on-box; the pre-fix
    fold read attempts=2, n_preempted=0 from this exact shape."""
    evs = [ev("claimed", T(1), instance_id="44"),
           ev("started", T(2), instance_id="44"),
           ev("resumed", T(3), instance_id="44", kind="preempt", detect="boot_change"),
           ev("started", T(4), instance_id="44"),
           ev("done", T(5), instance_id="44", rc=0)]
    v = jm.fold_events(evs, live_iids={"44"})
    assert v["attempts"] == 1
    assert v["n_preempted"] == 1


def test_fold_attempts_trap_preempt_shape_no_double_count():
    """THE trap shape: a signal DID land, so both the trap's `preempted` AND
    jobd's `resumed{kind:preempt}` exist for the SAME interruption — they must
    fold to n_preempted=1, not 2."""
    evs = [ev("claimed", T(1), instance_id="44"),
           ev("started", T(2), instance_id="44"),
           ev("preempted", T(3), instance_id="44"),
           ev("resumed", T(4), instance_id="44", kind="preempt", detect="trap"),
           ev("started", T(5), instance_id="44"),
           ev("done", T(6), instance_id="44", rc=0)]
    v = jm.fold_events(evs, live_iids={"44"})
    assert v["attempts"] == 1
    assert v["n_preempted"] == 1


def test_fold_attempts_genuine_crash_restart_shape():
    """A genuine crash-restart (same boot, e.g. OOM took the runner): jobd
    emits `resumed{kind:crash}`, which must NOT be subtracted out of attempts
    or counted as a preempt — this is exactly the budget the restart cap
    (max_restarts) is meant to drain."""
    evs = [ev("claimed", T(1), instance_id="44"),
           ev("started", T(2), instance_id="44"),
           ev("resumed", T(3), instance_id="44", kind="crash", detect="same_boot"),
           ev("started", T(4), instance_id="44"),
           ev("done", T(5), instance_id="44", rc=0)]
    v = jm.fold_events(evs, live_iids={"44"})
    assert v["attempts"] == 2
    assert v["n_preempted"] == 0


def test_fold_attempts_fresh_single_run_unaffected():
    """No interruption at all: attempts=1, n_preempted=0 — the fold must not
    regress the common case."""
    evs = [ev("claimed", T(1), instance_id="44"),
           ev("started", T(2), instance_id="44"),
           ev("done", T(3), instance_id="44", rc=0)]
    v = jm.fold_events(evs, live_iids={"44"})
    assert v["attempts"] == 1
    assert v["n_preempted"] == 0


def test_fold_attempts_two_preempt_resumes_in_a_row():
    """Two evictions on the same job (a genuinely contested spot market): both
    `resumed{kind:preempt}` subtract, and n_preempted counts both."""
    evs = [ev("claimed", T(1), instance_id="44"),
           ev("started", T(2), instance_id="44"),
           ev("resumed", T(3), instance_id="55", kind="preempt", detect="boot_change"),
           ev("started", T(4), instance_id="55"),
           ev("resumed", T(5), instance_id="66", kind="preempt", detect="boot_change"),
           ev("started", T(6), instance_id="66"),
           ev("done", T(7), instance_id="66", rc=0)]
    v = jm.fold_events(evs, live_iids={"66"})
    assert v["attempts"] == 1
    assert v["n_preempted"] == 2


# --------------------------------------------------------------------------- #
# requeue un-stick: a `resumed` NEWER than `failed` re-opens the job.
# ORDER is the whole rule (an ordinary crash/preempt `resumed` that preceded the
# failure must not resurrect it), and only `failed` un-sticks.
# --------------------------------------------------------------------------- #
def _failed_run(fail_ts=T(4)):
    """submitted -> claimed -> started -> failed on box 44."""
    return [ev("submitted", T(1), actor="cli:h", box="44", bundle_sha256="ab"),
            ev("claimed", T(2), instance_id="44"),
            ev("started", T(3), instance_id="44"),
            ev("failed", fail_ts, instance_id="44", rc=16, reason="rc=16")]


def test_fold_resumed_after_failed_reopens():
    evs = _failed_run()
    assert jm.fold_events(evs)["status"] == "failed"
    rq = ev("resumed", T(5), kind="requeue", instance_id="55",
            retargeted_from="44", from_box="44")
    v = jm.fold_events(evs + [rq], live_iids={"55"})
    assert v["reopened"] is True
    assert v["status"] not in jm.TERMINAL
    # a bare re-open = re-queued, awaiting a claim on the new box
    assert v["status"] == "submitted" and v["display_status"] == "submitted"
    assert v["instance_id"] == "55" and v["ended_at"] is None
    # the prior attempt's outcome is preserved, but nothing reports it as current
    assert v["rc"] is None and v["fail_reason"] is None
    assert v["prior_rc"] == 16 and v["prior_fail_reason"] == "rc=16"
    # and a re-opened job is no longer "unreachable" for `job wait`
    assert jm.wait_decision(v, "done") == "pending"


def test_fold_resumed_older_than_failed_does_not_reopen():
    # the ordinary crash/preempt resume: it PRECEDES the failure, so the job that
    # then failed stays failed. Shuffled input — the rule is (ts, nonce), not
    # arrival order.
    evs = [ev("submitted", T(1), actor="cli:h", box="44"),
           ev("claimed", T(2), instance_id="44"),
           ev("preempted", T(3), instance_id="44"),
           ev("resumed", T(4), instance_id="44", kind="preempt"),
           ev("started", T(5), instance_id="44"),
           ev("failed", T(6), instance_id="44", rc=2, reason="boom")]
    for order in (evs, list(reversed(evs))):
        v = jm.fold_events(order, live_iids={"44"})
        assert v["status"] == "failed" and v["reopened"] is False
        assert v["rc"] == 2 and v["prior_rc"] is None
        assert v["ended_at"] == T(6)


def test_fold_resumed_never_reopens_done_or_cancelled():
    """`done` (rc=0, results on B2) and `cancelled` (never-revive verdict) are
    sticky UNCONDITIONALLY — a later `resumed` is chatter, not a re-open."""
    base = [ev("submitted", T(1), actor="cli:h", box="44"),
            ev("started", T(2), instance_id="44")]
    late = ev("resumed", T(9), kind="requeue", instance_id="55")
    done = jm.fold_events(base + [ev("done", T(4), rc=0), late], live_iids={"55"})
    assert done["status"] == "done" and done["reopened"] is False
    canc = jm.fold_events(base + [ev("cancelled", T(4), reason="operator"), late],
                          live_iids={"55"})
    assert canc["status"] == "cancelled" and canc["reopened"] is False
    # a `failed` alongside a `done` is NOT the sole terminal -> still sticky
    both = jm.fold_events(base + [ev("done", T(3), rc=0),
                                  ev("failed", T(4), rc=9), late], live_iids={"55"})
    assert both["status"] == "failed" and both["reopened"] is False


def test_fold_stale_failed_never_outranks_newer_done():
    """THE 2026-08-06 status-fold defect: a requeued job that then SUCCEEDS holds
    an early attempt's `failed` AND a final `done` — and the set-based lattice
    ranked the stale `failed` above the newer `done` forever. Measured on
    20260806T082213-v11-qwen25c7b-chat-dec-train-aff8 (197 events, 5 boxes):
    `failed rc=1` at 09:34 on box 46962674, `done rc=0` at 21:05 on box 47011548
    with all gates passed and the adapter published, yet `job status` and
    `job wait --until terminal` both reported failed/rc=1 — the sanctioned
    monitoring primitive faking its rc=2 terminal-but-FAILED outcome on a
    completed, paid-for run. The newest terminal word must win."""
    evs = [ev("submitted", T(1), actor="cli:h", box="44", bundle_sha256="ab"),
           ev("claimed", T(2), instance_id="44"),
           ev("started", T(3), instance_id="44"),
           ev("failed", T(4), instance_id="44", rc=1, reason="rc=1",
              tail="ChildFailedError: stale traceback"),
           ev("resumed", T(5), kind="requeue", instance_id="55",
              retargeted_from="44", from_box="44"),
           ev("claimed", T(6), instance_id="55"),
           ev("started", T(7), instance_id="55"),
           ev("done", T(9), instance_id="55", rc=0,
              results=[{"path": "out/adapter", "size": 3}])]
    for order in (evs, list(reversed(evs))):   # rule is (ts, nonce), not arrival
        v = jm.fold_events(order, live_iids=())
        assert v["status"] == "done" and v["display_status"] == "done", v
        assert v["rc"] == 0 and v["fail_reason"] is None, v
        assert v["reopened"] is False and v["ended_at"] == T(9)
        assert v["results"] == [{"path": "out/adapter", "size": 3}]
        # the stale attempt survives as DIAGNOSIS, never as the outcome
        assert v["prior_rc"] == 1 and v["prior_fail_reason"] == "rc=1"
        # ...and its traceback must not surface as the job's tail
        assert v["last_tail"] != "ChildFailedError: stale traceback"
        # the sanctioned monitoring primitive sees success, not rc=2's fake
        assert jm.wait_decision(v, "terminal") == "match"
        assert jm.wait_decision(v, "done") == "match"


def test_fold_newer_failed_still_beats_older_done():
    """The order rule cuts both ways: when the `failed` is the NEWER terminal
    (the pre-existing test_fold_terminal_precedence shape) it stands — the fix
    is time-ordering, not a done-always-wins inversion."""
    v = jm.fold_events([ev("started", T(2), instance_id="1"),
                        ev("done", T(5), rc=0),
                        ev("failed", T(6), rc=2, reason="boom")], live_iids={"1"})
    assert v["status"] == "failed" and v["rc"] == 2
    assert v["prior_rc"] is None       # no demotion in this direction


def test_fold_reopened_status_tracks_the_new_attempt_only():
    """After the un-stick the status folds from the events AT-OR-AFTER the
    re-opening `resumed`: the OLD attempt's `started` must not make a freshly
    queued job look like it is running, and a second failure re-sticks it."""
    evs = _failed_run() + [ev("resumed", T(5), kind="requeue", instance_id="55")]
    queued = jm.fold_events(evs, live_iids={"55"})
    assert queued["display_status"] == "submitted"      # NOT "running" off T(3)

    claimed = jm.fold_events(evs + [ev("claimed", T(6), instance_id="55")],
                             live_iids={"55"})
    assert claimed["status"] == "claimed" and claimed["display_status"] == "running"

    running = jm.fold_events(evs + [ev("claimed", T(6), instance_id="55"),
                                    ev("started", T(7), instance_id="55")],
                             live_iids=())
    assert running["status"] == "started"
    assert running["display_status"] == "interrupted"   # box 55 gone
    assert running["attempts"] == 2

    refailed = jm.fold_events(evs + [ev("started", T(7), instance_id="55"),
                                     ev("failed", T(8), instance_id="55", rc=1,
                                        reason="again")], live_iids={"55"})
    assert refailed["status"] == "failed" and refailed["reopened"] is False
    assert refailed["rc"] == 1 and refailed["ended_at"] == T(8)

    # ...and a SECOND requeue re-opens it again.
    twice = jm.fold_events(evs + [ev("started", T(7), instance_id="55"),
                                  ev("failed", T(8), instance_id="55", rc=1),
                                  ev("resumed", T(9), kind="requeue",
                                     instance_id="66")], live_iids={"66"})
    assert twice["reopened"] is True and twice["status"] == "submitted"
    assert twice["instance_id"] == "66"


def test_fold_reopen_order_is_ts_then_nonce():
    """Same-second events: the tiebreak is the nonce, the same total order the
    display sort uses — so the rule never depends on dict/listing order."""
    evs = _failed_run(fail_ts=T(5))
    lo = ev("resumed", T(5), nonce="0000", kind="requeue")   # sorts BEFORE failed
    hi = ev("resumed", T(5), nonce="zzzz", kind="requeue")   # sorts AFTER failed
    failed_nonce = [e for e in evs if e["event"] == "failed"][0]["nonce"]
    assert "0000" < failed_nonce < "zzzz"
    assert jm.fold_events(evs + [lo])["reopened"] is False
    assert jm.fold_events(evs + [hi])["reopened"] is True


def test_validate_needs_gpus_and_max_restarts(tmp_path):
    src = tmp_path / "cfgsrc"
    src.mkdir()
    (src / "run.sh").write_text("true\n")
    base = {"version": 1, "name": "cfg-probe", "entrypoint": "run.sh",
            "results": ["out/**"], "needs": {"gpu": False, "venv": "none"}}

    # defaults: plain GPU job -> 1 card; CPU job -> no gpus key; max_restarts default
    cfg, _ = jm.validate_job_config(
        {**base, "needs": {"gpu": True, "venv": "none"}}, str(src))
    assert cfg["needs"]["gpus"] == 1 and cfg["max_restarts"] == jm.DEFAULT_MAX_RESTARTS
    cfg, _ = jm.validate_job_config(dict(base), str(src))
    assert "gpus" not in cfg["needs"]

    # explicit gpus implies gpu; "all" rides through verbatim
    cfg, _ = jm.validate_job_config(
        {**base, "needs": {"gpus": 2, "venv": "none"}}, str(src))
    assert cfg["needs"]["gpu"] is True and cfg["needs"]["gpus"] == 2
    cfg, _ = jm.validate_job_config(
        {**base, "needs": {"gpus": "all", "venv": "none"}}, str(src))
    assert cfg["needs"]["gpus"] == "all"

    # max_restarts: 0 = never resume; negatives/bools rejected
    cfg, _ = jm.validate_job_config({**base, "max_restarts": 0}, str(src))
    assert cfg["max_restarts"] == 0
    for bad in (-1, True, "two"):
        with pytest.raises(jm.JobmetaError):
            jm.validate_job_config({**base, "max_restarts": bad}, str(src))
    for bad in (0, -2, "some", False):
        with pytest.raises(jm.JobmetaError):
            jm.validate_job_config(
                {**base, "needs": {"gpus": bad, "venv": "none"}}, str(src))


def test_default_max_restarts_is_five_and_zero_is_honoured(tmp_path):
    """The default is 5 (spot evictions cluster; a checkpointing job resumes
    cheap), and an explicit 0 stays 0 — a deliberate one-shot is never raised."""
    src = tmp_path / "cfgsrc"
    src.mkdir()
    (src / "run.sh").write_text("true\n")
    base = {"version": 1, "name": "cfg-probe", "entrypoint": "run.sh",
            "results": ["out/**"], "needs": {"gpu": False, "venv": "none"}}

    assert jm.DEFAULT_MAX_RESTARTS == 5
    cfg, _ = jm.validate_job_config(dict(base), str(src))
    assert cfg["max_restarts"] == 5

    for pinned in (0, 1, 2, 4, 6):
        cfg, _ = jm.validate_job_config(
            {**base, "max_restarts": pinned}, str(src))
        assert cfg["max_restarts"] == pinned


def test_validate_config_assets_ok(tmp_path):
    """N4: a well-formed `assets:` list is validated laptop-side and baked into
    the canonical config verbatim (box side reads JSON only)."""
    d = _mkjob(tmp_path)
    raw = jm.load_job_config(str(d))
    raw["assets"] = [
        {"name": "base-model", "b2": "base-models/qwen25-coder-7b-instruct"},
        {"name": "reader-lora", "b2": "checkpoints/reader/adapter",
         "dest": "arms/reader", "mode": "sync",
         "require": ["adapter_config.json", "adapter_model.safetensors"],
         "optional": False},
        {"name": "extras", "b2": "aux/tables", "dest": "/workspace/extras",
         "optional": True},
    ]
    cfg, warn = jm.validate_job_config(raw, str(d))
    assets = cfg["assets"]
    assert [a["name"] for a in assets] == ["base-model", "reader-lora", "extras"]
    # defaults filled: mode=copy, optional=False, require=[]; dest omitted when absent
    a0 = assets[0]
    assert a0["b2"] == "base-models/qwen25-coder-7b-instruct"
    assert a0["mode"] == "copy" and a0["optional"] is False and a0["require"] == []
    assert "dest" not in a0
    a1 = assets[1]
    assert a1["mode"] == "sync" and a1["dest"] == "arms/reader"
    assert a1["require"] == ["adapter_config.json", "adapter_model.safetensors"]
    a2 = assets[2]
    assert a2["optional"] is True and a2["dest"] == "/workspace/extras"
    # absent -> no assets key (old tickets keep working)
    raw.pop("assets")
    cfg2, _ = jm.validate_job_config(raw, str(d))
    assert "assets" not in cfg2


def test_validate_config_asset_receipt_normalized_and_absent_stays_absent(tmp_path):
    """`receipt:` normalizes to a bare relative path, and an asset that does not
    declare one serializes with NO receipt key — the whole legacy-compat claim."""
    d = _mkjob(tmp_path)
    raw = jm.load_job_config(str(d))
    raw["assets"] = [
        {"name": "merged", "b2": "checkpoints/v10-merged", "receipt": "/PUSHED.json"},
        {"name": "nested", "b2": "checkpoints/x", "receipt": "meta/DONE.json"},
        {"name": "plain", "b2": "base-models/qwen"},
        {"name": "blank", "b2": "base-models/other", "receipt": "  "},
    ]
    cfg, _ = jm.validate_job_config(raw, str(d))
    by = {a["name"]: a for a in cfg["assets"]}
    assert by["merged"]["receipt"] == "PUSHED.json"     # leading '/' stripped
    assert by["nested"]["receipt"] == "meta/DONE.json"
    assert "receipt" not in by["plain"]
    assert "receipt" not in by["blank"]                  # empty == undeclared
    # and a legacy declaration is byte-identical to one validated without the
    # field ever having existed in the config
    raw2 = jm.load_job_config(str(d))
    raw2["assets"] = [{"name": "plain", "b2": "base-models/qwen"}]
    cfg2, _ = jm.validate_job_config(raw2, str(d))
    assert cfg2["assets"][0] == by["plain"]


# --------------------------------------------------------------------------- #
# `${VAR}` asset prefixes — submit-time resolution (ASSET_PARAMETERIZATION.md)
# --------------------------------------------------------------------------- #
def _param_job(tmp_path, b2, env=None):
    d = _mkjob(tmp_path)
    raw = jm.load_job_config(str(d))
    raw["env"] = {**(raw.get("env") or {}), **(env or {})}
    raw["assets"] = [{"name": "adapter", "b2": b2}]
    return d, raw


def test_asset_b2_placeholder_resolves_from_the_submit_env(tmp_path):
    """The happy path: the ticket carries the RESOLVED prefix and, as
    provenance, the template the bundle actually declared."""
    d, raw = _param_job(tmp_path, "${ADAPTER_B2}/model",
                        {"ADAPTER_B2": "checkpoints/v10-merged/abc"})
    a, = jm.validate_job_config(raw, str(d))[0]["assets"]
    assert a["b2"] == "checkpoints/v10-merged/abc/model"
    assert a["b2_template"] == "${ADAPTER_B2}/model"


def test_asset_b2_two_placeholders_and_a_literal_asset_keeps_no_template(tmp_path):
    d = _mkjob(tmp_path)
    raw = jm.load_job_config(str(d))
    raw["env"] = {"ROOT": "checkpoints/x", "SHA": "abc123"}
    raw["assets"] = [{"name": "a", "b2": "${ROOT}/${SHA}/model"},
                     {"name": "b", "b2": "base-models/qwen"}]
    by = {x["name"]: x for x in jm.validate_job_config(raw, str(d))[0]["assets"]}
    assert by["a"]["b2"] == "checkpoints/x/abc123/model"
    assert "b2_template" not in by["b"]      # a literal serializes as it always did


@pytest.mark.parametrize("env", [{}, {"ADAPTER_B2": ""}])
def test_asset_b2_unresolved_variable_refuses_at_submit(tmp_path, env):
    """Absent and EMPTY are the same answer. `KEY: ""` in `env:` is the house
    convention for 'required at submit', so treating it as a value would resolve
    a forgotten pin to a plausible-looking prefix."""
    d, raw = _param_job(tmp_path, "${ADAPTER_B2}/model", env)
    with pytest.raises(jm.JobmetaError) as e:
        jm.validate_job_config(raw, str(d))
    assert "${ADAPTER_B2}" in str(e.value) and "adapter" in str(e.value)


@pytest.mark.parametrize("bad", ["${lower}/model", "${A-B}/model", "${}/model",
                                 "${A:-x}/model"])
def test_asset_b2_malformed_placeholder_refuses(tmp_path, bad):
    """`${NAME}` only — no lowercase, no punctuation, no `${X:-default}`. A name
    that does not match would otherwise ship to the box as a literal B2 prefix."""
    d, raw = _param_job(tmp_path, bad, {"lower": "x", "A-B": "x", "A": "q"})
    with pytest.raises(jm.JobmetaError, match="malformed placeholder"):
        jm.validate_job_config(raw, str(d))


def test_asset_b2_bare_dollar_is_not_a_placeholder(tmp_path):
    """No shell-isms: a bare `$VAR` stays literal rather than silently expanding
    (rclone would then look for a key spelled exactly that way)."""
    d, raw = _param_job(tmp_path, "$ADAPTER_B2/model", {"ADAPTER_B2": "x/y"})
    a, = jm.validate_job_config(raw, str(d))[0]["assets"]
    assert a["b2"] == "$ADAPTER_B2/model" and "b2_template" not in a


@pytest.mark.parametrize("value,msg", [
    ("/absolute", "relative"),
    ("b2:bucket/x", "relative"),
    ("a/../b", "relative"),
])
def test_asset_b2_expansion_is_shape_checked_like_a_literal(tmp_path, value, msg):
    """An env value is operator input, not a trusted constant — resolution runs
    BEFORE the existing prefix rules, not instead of them."""
    d, raw = _param_job(tmp_path, "${ADAPTER_B2}", {"ADAPTER_B2": value})
    with pytest.raises(jm.JobmetaError, match=msg):
        jm.validate_job_config(raw, str(d))


def test_asset_placeholder_is_neutral_to_the_bundle_sha(tmp_path):
    """THE point of the design. The bundle is content-addressed over the FOLDER,
    which holds the template; two submits naming different artifacts reuse one
    bundle object, so `job requeue` identity and the dedupe survive."""
    d = tmp_path / "job"
    (d / "sub").mkdir(parents=True)
    (d / "run.sh").write_text("echo hi\n")
    (d / "sub" / "a.txt").write_text("data\n")
    (d / "job-config.yaml").write_text(
        "version: 1\nname: probe-01\nentrypoint: run.sh\ntimeout_s: 60\n"
        "env:\n  ADAPTER_B2: \"\"\n"
        "assets:\n  - name: adapter\n    b2: \"${ADAPTER_B2}/model\"\n")
    sha = jm.bundle_sha256(str(d))
    prefixes = []
    for value in ("checkpoints/one/aaa", "checkpoints/two/bbb"):
        raw = jm.load_job_config(str(d))
        raw["env"]["ADAPTER_B2"] = value           # what `--env` folds
        cfg, _ = jm.validate_job_config(raw, str(d))
        prefixes.append(cfg["assets"][0]["b2"])
        assert jm.bundle_sha256(str(d)) == sha     # the folder never moved
    assert prefixes == ["checkpoints/one/aaa/model", "checkpoints/two/bbb/model"]


def test_resolved_scope_and_receipt_read_the_resolved_prefix(tmp_path):
    """Everything downstream sees the RESOLVED key: the derived read scope, and
    the $0 receipt read that refuses before a box is rented."""
    d = _mkjob(tmp_path)
    raw = jm.load_job_config(str(d))
    raw["env"] = {"ADAPTER_B2": "checkpoints/v10/abc"}
    raw["assets"] = [{"name": "adapter", "b2": "${ADAPTER_B2}",
                      "receipt": "PUSHED.json"}]
    cfg, _ = jm.validate_job_config(raw, str(d))
    assert set(cfg["scope"]["read"]) == {"checkpoints/", "jobs/"}
    f, = jm.check_asset_receipts(
        cfg["assets"],
        runner=_receipt_runner({"checkpoints/v10/abc/PUSHED.json":
                                '{"complete": true, "files": 4}'}),
        bucket="bkt")
    assert (f["status"], f["b2"]) == ("ok", "checkpoints/v10/abc")


def test_make_ticket_refuses_an_unresolved_prefix(tmp_path):
    """The backstop at the one door every surface writes through. Reaching it
    means a resolution bug, not a config error — a rented box would otherwise
    pull a literal `${...}` key."""
    cfg = {"name": "probe-01", "assets": [{"name": "adapter",
                                           "b2": "${ADAPTER_B2}/model"}]}
    with pytest.raises(jm.JobmetaError, match="UNRESOLVED"):
        jm.make_ticket("probe-01-20260824T000000Z-abcd", "0" * 64, "me", cfg, "7")


def _receipt_runner(objects):
    """A jobmeta B2 runner stub answering `cat` from {key: body}; a key that is
    absent answers the way rclone does (rc 1, 'not found' on stderr)."""
    def run(argv):
        if argv[0] != "cat":
            return 1, "", "stub: unsupported op"
        key = argv[1].split("/", 1)[1]          # b2:<bucket>/<key>
        if key in objects:
            return 0, objects[key], ""
        return 1, "", f"directory not found: {key}"
    return run


@pytest.mark.parametrize("body,status,files", [
    ('{"complete": true, "files": 7, "ts_utc": "z"}', "ok", 7),
    ('{"complete": true}', "ok", None),
    ("not json at all", "ok", None),            # unreadable != incomplete
    ('{"complete": false, "files": 2}', "missing", 2),
])
def test_check_asset_receipts_reads_the_marker(body, status, files):
    assets = [{"name": "m", "b2": "checkpoints/v10", "receipt": "PUSHED.json"}]
    f, = jm.check_asset_receipts(
        assets, runner=_receipt_runner({"checkpoints/v10/PUSHED.json": body}),
        bucket="bkt")
    assert (f["status"], f["files"], f["kind"]) == (status, files, "receipt")


def test_check_asset_receipts_absent_marker_refuses_at_submit():
    """The $0 half of the gate: no marker on B2 -> refuse before a box is rented.
    Neither --strict-assets nor --allow-stale-assets is the escape."""
    assets = [{"name": "m", "b2": "checkpoints/v10", "receipt": "PUSHED.json"}]
    findings = jm.check_asset_receipts(assets, runner=_receipt_runner({}), bucket="b")
    assert [f["status"] for f in findings] == ["missing"]
    for kwargs in ({}, {"strict": True}, {"allow_stale": True}):
        lines, refuse = jm.asset_preflight_report(findings, **kwargs)
        assert refuse, kwargs
        assert any("ASSET INCOMPLETE" in ln for ln in lines)
        # the trailer naming flags that do NOT apply here is suppressed
        assert not any("drop --strict-assets" in ln for ln in lines)


def test_check_asset_receipts_transport_blip_never_blocks():
    """A read that fails for any reason OTHER than absence is 'unknown': a
    creds-less laptop or a B2 blip must not be able to refuse a submit."""
    def run(argv):
        return 1, "", "SignatureDoesNotMatch"
    assets = [{"name": "m", "b2": "checkpoints/v10", "receipt": "PUSHED.json"}]
    findings = jm.check_asset_receipts(assets, runner=run, bucket="b")
    assert [f["status"] for f in findings] == ["unknown"]
    lines, refuse = jm.asset_preflight_report(findings, strict=True)
    assert not refuse and any("receipt UNVERIFIED" in ln for ln in lines)


def test_check_asset_receipts_skips_undeclared_assets():
    """No `receipt:` -> no B2 read at all (an asset lane that never opted in
    must not gain a network dependency)."""
    calls = []

    def run(argv):
        calls.append(argv)
        return 0, "", ""
    assert jm.check_asset_receipts(
        [{"name": "m", "b2": "base-models/q"}], runner=run, bucket="b") == []
    assert calls == []


def test_scope_default_write_jobs_no_scope_key_for_plain_job(tmp_path):
    d = _mkjob(tmp_path)
    cfg, _ = jm.validate_job_config(jm.load_job_config(str(d)), str(d))
    assert "scope" not in cfg                       # plain job -> ticket unchanged


def test_scope_derived_from_assets_recorded(tmp_path):
    d = _mkjob(tmp_path)
    raw = jm.load_job_config(str(d))
    raw["assets"] = [{"name": "m", "b2": "base-models/qwen"},
                     {"name": "a", "b2": "checkpoints/x/adapter"}]
    cfg, _ = jm.validate_job_config(raw, str(d))
    assert cfg["scope"]["write"] == ["jobs/"]
    assert set(cfg["scope"]["read"]) == {"base-models/", "checkpoints/", "jobs/"}


def test_scope_explicit_write_outside_the_granted_prefixes_rejected(tmp_path):
    """`checkpoints/` became legal on 2026-08-05 — a jobs box now holds a second
    namePrefix-scoped write key for it (the publish grant). Everything else is
    still refused: the list is what a box has a KEY for, not a preference."""
    d = _mkjob(tmp_path)
    raw = jm.load_job_config(str(d))
    raw["scope"] = {"write": ["checkpoints/"]}
    cfg, _ = jm.validate_job_config(raw, str(d))
    assert cfg["scope"]["write"] == ["checkpoints/"]
    raw["scope"] = {"write": ["runsets/"]}
    with pytest.raises(jm.JobmetaError, match="scope.write"):
        jm.validate_job_config(raw, str(d))


def test_scope_asset_outside_explicit_read_rejected(tmp_path):
    d = _mkjob(tmp_path)
    raw = jm.load_job_config(str(d))
    raw["scope"] = {"read": ["base-models/"]}
    raw["assets"] = [{"name": "a", "b2": "checkpoints/x/adapter"}]
    with pytest.raises(jm.JobmetaError, match="outside the declared scope.read"):
        jm.validate_job_config(raw, str(d))


def test_scope_read_allowlist_permissive_by_default(tmp_path):
    d = _mkjob(tmp_path)
    raw = jm.load_job_config(str(d))
    raw["scope"] = {"read": ["aux/tables/"]}         # not in the default allowlist
    cfg, _ = jm.validate_job_config(raw, str(d))     # permissive: allowed
    assert cfg["scope"]["read"] == ["aux/tables/"]


def test_scope_read_allowlist_strict_rejects(tmp_path, monkeypatch):
    monkeypatch.setenv("B2_JOB_SCOPE_STRICT", "1")
    d = _mkjob(tmp_path)
    raw = jm.load_job_config(str(d))
    raw["scope"] = {"read": ["aux/tables/"]}
    with pytest.raises(jm.JobmetaError, match="not in the allowlist"):
        jm.validate_job_config(raw, str(d))
    # a whitelisted prefix passes even in strict mode
    raw["scope"] = {"read": ["base-models/"]}
    cfg, _ = jm.validate_job_config(raw, str(d))
    assert cfg["scope"]["read"] == ["base-models/"]


def test_scope_read_allowlist_strict_allows_checkpoints(tmp_path, monkeypatch):
    assert "checkpoints/" in jm.DEFAULT_JOB_READ_PREFIXES
    monkeypatch.setenv("B2_JOB_SCOPE_STRICT", "1")
    d = _mkjob(tmp_path)
    raw = jm.load_job_config(str(d))
    raw["scope"] = {"read": ["checkpoints/tuner-v2-repair/"]}
    cfg, _ = jm.validate_job_config(raw, str(d))
    assert cfg["scope"]["read"] == ["checkpoints/tuner-v2-repair/"]
    # the bare prefix also passes
    raw["scope"] = {"read": ["checkpoints/"]}
    cfg, _ = jm.validate_job_config(raw, str(d))
    assert cfg["scope"]["read"] == ["checkpoints/"]


@pytest.mark.parametrize("block,msg", [
    ("notalist", "must be a list"),
    ([{"b2": "x/y"}], "name"),
    ([{"name": "Bad_Name", "b2": "x/y"}], "name"),
    ([{"name": "with space", "b2": "x/y"}], "name"),
    ([{"name": "ok"}], "b2"),
    ([{"name": "ok", "b2": "/abs/path"}], "relative"),
    ([{"name": "ok", "b2": "a/../b"}], "relative"),
    ([{"name": "ok", "b2": "b2:bucket/x"}], "relative"),
    ([{"name": "ok", "b2": "x/y", "mode": "rsync"}], "mode"),
    ([{"name": "ok", "b2": "x/y", "optional": "yes"}], "optional"),
    ([{"name": "ok", "b2": "x/y", "require": "notalist_but_ok"}], None),  # str coerces
    ([{"name": "ok", "b2": "x/y", "require": ["/abs/glob"]}], "relative"),
    ([{"name": "ok", "b2": "x/y", "require": ["a/../b"]}], "relative"),
    ([{"name": "ok", "b2": "x/y", "dest": "/etc/passwd"}], "workspace"),
    ([{"name": "ok", "b2": "x/y", "dest": "../escape"}], ".."),
    ([{"name": "a", "b2": "x"}, {"name": "a", "b2": "y"}], "duplicate"),
    ([{"name": "ok", "b2": "x/y", "receipt": 3}], "string filename"),
    ([{"name": "ok", "b2": "x/y", "receipt": "../PUSHED.json"}], "relative"),
    ([{"name": "ok", "b2": "x/y", "receipt": "b2:bucket/PUSHED.json"}], "relative"),
    ([{"name": "ok", "b2": "x/y", "receipt": "P\tJ.json"}], "tab or newline"),
    ([{"name": "ok", "b2": "x/y", "receipt": "*.json"}], "not a glob"),
])
def test_validate_config_assets_failures(tmp_path, block, msg):
    d = _mkjob(tmp_path)
    raw = jm.load_job_config(str(d))
    raw["assets"] = block
    if msg is None:                         # the str-require case must SUCCEED
        cfg, _ = jm.validate_job_config(raw, str(d))
        assert cfg["assets"][0]["require"] == ["notalist_but_ok"]
        return
    with pytest.raises(jm.JobmetaError) as e:
        jm.validate_job_config(raw, str(d))
    assert msg in str(e.value)


def test_fold_live_iid_int_str_coercion():
    evs = [ev("started", T(2), instance_id="44")]
    assert jm.fold_events(evs, live_iids={44})["live"] is True   # int vs str


def test_fold_unparseable_skipped():
    good = ev("started", T(2), instance_id="1")
    bad_missing = {"v": 1, "ts": T(3), "actor": "x"}        # no event/job_id
    v = jm.fold_events([good, "", "{bad json", bad_missing], live_iids={"1"})
    assert v["parse_errors"] == 3 and v["n_events"] == 1
    assert v["status"] == "started"


def test_fold_heartbeat_tail():
    evs = [ev("started", T(2), instance_id="1"),
           ev("heartbeat", T(3), instance_id="1", tail="line1\n"),
           ev("heartbeat", T(5), instance_id="1", tail="line2\n")]
    v = jm.fold_events(evs, live_iids={"1"})
    assert v["last_tail"] == "line2\n" and v["last_heartbeat_ts"] == T(5)


def test_fold_heartbeat_host_metrics():
    # host_metrics rides on heartbeats; the fold keeps the latest one that has it,
    # and ignores heartbeats that carry only a tail (older probe / no-GPU tick).
    evs = [ev("started", T(2), instance_id="1"),
           ev("heartbeat", T(3), instance_id="1", host_metrics="gpu_util:90,cpu:12"),
           ev("heartbeat", T(5), instance_id="1", tail="no-metrics-here\n")]
    v = jm.fold_events(evs, live_iids={"1"})
    assert v["last_metrics"] == "gpu_util:90,cpu:12"
    # none present -> stays None
    v2 = jm.fold_events([ev("heartbeat", T(3), instance_id="1", tail="x")],
                        live_iids={"1"})
    assert v2["last_metrics"] is None


def test_fold_empty():
    v = jm.fold_events([])
    assert v["status"] == "unknown" and v["n_events"] == 0 and v["job_id"] is None


# --------------------------------------------------------------------------- #
# transport — fake in-memory B2 runner (runmeta contract, no rclone/net)
# --------------------------------------------------------------------------- #
class FakeB2:
    """Minimal rclone-shaped runner over an in-memory dict of key->bytes.
    Supports rcat/cat/lsf/copy/copyto — the subset jobmeta uses."""
    def __init__(self, bucket="bkt"):
        self.bucket = bucket
        self.store = {}          # key (without b2:bucket/) -> str body

    def _key(self, remote):
        prefix = f"b2:{self.bucket}/"
        assert remote.startswith(prefix), remote
        return remote[len(prefix):]

    def __call__(self, args, input=None):
        op = args[0]
        if op == "rcat":
            self.store[self._key(args[1])] = input
            return 0, "", ""
        if op == "cat":
            k = self._key(args[1])
            return (0, self.store[k], "") if k in self.store else (1, "", "not found")
        if op == "lsf":
            rec = "-R" in args
            target = self._key([x for x in args[1:] if x.startswith(f"b2:")][0])
            if not target.endswith("/"):     # file existence probe
                base = os.path.basename(target)
                return (0, base + "\n", "") if target in self.store else (1, "", "")
            names = []
            for k in self.store:
                if k.startswith(target):
                    rest = k[len(target):]
                    names.append(rest if rec else rest.split("/", 1)[0])
            return 0, "".join(f"{n}\n" for n in sorted(set(names))), ""
        if op == "copy":
            # positional src/dst with flags interleaved (--include NAME, --retries N)
            flags_with_val = {"--include", "--retries", "--min-age", "--transfers",
                              "--checkers"}
            pos, i = [], 1
            while i < len(args):
                if args[i] in flags_with_val:
                    i += 2
                elif args[i].startswith("--"):
                    i += 1
                else:
                    pos.append(args[i]); i += 1
            src, dst = pos
            inc = args[args.index("--include") + 1].lstrip("/") if "--include" in args else None
            if src.startswith("b2:"):
                # copy b2:.../PREFIX/  LOCALDIR  -> materialize files locally
                srck = self._key(src)
                for k, body in self.store.items():
                    if k.startswith(srck):
                        fp = os.path.join(dst, k[len(srck):])
                        os.makedirs(os.path.dirname(fp), exist_ok=True)
                        with open(fp, "w") as fh:
                            fh.write(body)
                return 0, "", ""
            # copy LOCALDIR b2:.../PREFIX/ [--include /NAME] -> list-based upload
            # (upload_bundle's copyto replacement; binary-tolerant like FakeB2Bin)
            dstk = self._key(dst if dst.endswith("/") else dst + "/")
            for fn in os.listdir(src):
                fp = os.path.join(src, fn)
                if not os.path.isfile(fp) or (inc and fn != inc):
                    continue
                with open(fp, "rb") as fh:
                    self.store[dstk + fn] = fh.read()
            return 0, "", ""
        if op == "copyto":
            src, dst = args[1], args[2]
            if src.startswith("b2:"):        # download
                k = self._key(src)
                if k not in self.store:
                    return 1, "", "not found"
                with open(dst, "w") as fh:
                    fh.write(self.store[k])
            else:                             # upload
                with open(src) as fh:
                    self.store[self._key(dst)] = fh.read()
            return 0, "", ""
        if op == "lsjson":
            # `self.mtimes` is OPT-IN: a key with no entry lists with no ModTime,
            # which is how a caller's undatable-object branch gets exercised.
            k = self._key(args[-1])
            if k not in self.store:
                return 1, "", "not found"
            row = {"Path": os.path.basename(k), "Name": os.path.basename(k),
                   "Size": len(self.store[k]), "IsDir": False}
            mt = getattr(self, "mtimes", {}).get(k)
            if mt:
                row["ModTime"] = mt
            return 0, json.dumps([row]), ""
        if op == "deletefile":
            k = self._key(args[1])
            if k in self.store:
                del self.store[k]
                return 0, "", ""
            return 1, "", "not found"
        return 1, "", f"unexpected op {op}"


def test_emit_roundtrip_and_read_job(tmp_path, monkeypatch):
    fake = FakeB2()
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    for e, kw in [("submitted", dict(actor="cli:h", bundle_sha256="ab", box="44",
                                     name="p", entrypoint="run.sh", timeout_s=60)),
                  ("claimed", dict(actor="box:44", instance_id="44")),
                  ("started", dict(actor="box:44", instance_id="44")),
                  ("done", dict(actor="box:44", instance_id="44", rc=0))]:
        r = jm.emit_event("j1", e, runner=fake, bucket="bkt", **kw)
        assert r["_emitted"] and r["_key"].startswith("jobs/j1/events/")
    v = jm.read_job("j1", runner=fake, bucket="bkt", live_iids={"44"})
    assert v["status"] == "done" and v["instance_id"] == "44"
    assert v["bundle_sha256"] == "ab"


def test_bundle_dedupe_check(tmp_path):
    fake = FakeB2()
    assert jm.bundle_exists("deadbeef", runner=fake, bucket="bkt") is False
    fake.store["jobs/bundles/deadbeef.tar.zst"] = "blob"
    assert jm.bundle_exists("deadbeef", runner=fake, bucket="bkt") is True


def test_ticket_write_read_and_queue(tmp_path):
    fake = FakeB2()
    cfg = {"version": 1, "name": "p", "entrypoint": "run.sh", "timeout_s": 60,
           "env": {}, "results": [], "needs": {"gpu": False, "venv": "none"}}
    tk = jm.make_ticket("20260710T0-p-ab", "sha1", "cli:h", cfg, "44")
    ok, key, _ = jm.write_ticket(tk, runner=fake, bucket="bkt")
    assert ok and key == "jobs/queue/44/20260710T0-p-ab.json"
    got = jm.read_ticket("44", "20260710T0-p-ab", runner=fake, bucket="bkt")
    assert got["config"]["name"] == "p" and got["bundle_sha256"] == "sha1"
    assert jm.list_queue("44", runner=fake, bucket="bkt") == ["20260710T0-p-ab"]
    assert jm.list_all_queued(runner=fake, bucket="bkt") == [("44", "20260710T0-p-ab")]


def test_has_events(tmp_path):
    fake = FakeB2()
    assert jm.has_events("j1", runner=fake, bucket="bkt") is False
    jm.emit_event("j1", "claimed", runner=fake, bucket="bkt", actor="box:1")
    assert jm.has_events("j1", runner=fake, bucket="bkt") is True


def test_pull_results_manifest(tmp_path):
    fake = FakeB2()
    fake.store["jobs/j1/results/out/a.txt"] = "A"
    fake.store["jobs/j1/results/log.jsonl"] = "{}"
    dest = tmp_path / "pulled"
    manifest = jm.pull_results("j1", str(dest), runner=fake, bucket="bkt")
    assert manifest == ["log.jsonl", "out/a.txt"]
    assert (dest / "out" / "a.txt").read_text() == "A"


def test_emit_soft_on_transport_failure():
    def broken(args, input=None):
        return 1, "", "b2 down"
    r = jm.emit_event("j1", "done", runner=broken, bucket="bkt", actor="box:1")
    assert r["_emitted"] is False and "b2 down" in r["_error"]


# --------------------------------------------------------------------------- #
# experiment-matrix association (jobmatrix seam)
# --------------------------------------------------------------------------- #
def test_validate_config_experiment_block(tmp_path):
    d = _mkjob(tmp_path)
    raw = jm.load_job_config(str(d))
    raw["experiment"] = {"exp_id": "20260710T000000-exp3-ab12",
                         "arm": "qwen3-8b-r16", "axes": {"base": "qwen3-8b", "rank": 16}}
    cfg, _ = jm.validate_job_config(raw, str(d))
    assert cfg["experiment"] == {"exp_id": "20260710T000000-exp3-ab12",
                                 "arm": "qwen3-8b-r16",
                                 "axes": {"base": "qwen3-8b", "rank": "16"}}
    raw.pop("experiment")                       # absent -> absent (plain job)
    cfg, _ = jm.validate_job_config(raw, str(d))
    assert "experiment" not in cfg


@pytest.mark.parametrize("block,msg", [
    ("notadict", "mapping"),
    ({"arm": "a"}, "exp_id"),
    ({"exp_id": "bad id!"}, "invalid JOB_ID"),
    ({"exp_id": "20260710T000000-e-ab12"}, "experiment.arm"),
    ({"exp_id": "20260710T000000-e-ab12", "arm": "sp ace"}, "experiment.arm"),
    ({"exp_id": "20260710T000000-e-ab12", "arm": "a", "axes": ["x"]}, "axes"),
])
def test_validate_config_experiment_failures(tmp_path, block, msg):
    d = _mkjob(tmp_path)
    raw = jm.load_job_config(str(d))
    raw["experiment"] = block
    with pytest.raises(jm.JobmetaError) as e:
        jm.validate_job_config(raw, str(d))
    assert msg in str(e.value)


def test_fold_surfaces_exp_id_and_arm():
    # submitted carries both; a later box event re-echoes them — last sighting wins
    v = jm.fold_events([
        ev("submitted", T(1), actor="cli:h", exp_id="20260710T000000-e-ab12", arm="a1"),
        ev("started", T(2), instance_id="7", exp_id="20260710T000000-e-ab12", arm="a1"),
        ev("done", T(3), instance_id="7", rc=0),
    ], live_iids={"7"})
    assert v["exp_id"] == "20260710T000000-e-ab12" and v["arm"] == "a1"
    # plain job: absent -> None
    v = jm.fold_events([ev("submitted", T(1), actor="cli:h")])
    assert v["exp_id"] is None and v["arm"] is None
    # audit survives losing the laptop side: box events alone still associate
    v = jm.fold_events([
        ev("started", T(2), instance_id="7", exp_id="20260710T000000-e-ab12", arm="a1"),
    ])
    assert v["exp_id"] == "20260710T000000-e-ab12" and v["arm"] == "a1"


# --------------------------------------------------------------------------- #
# box-lifecycle stream (jobs/nodes/<IID>/events/) — separate from the job fold
# --------------------------------------------------------------------------- #
def _bev(event, ts, iid="7", nonce=None, **fields):
    d = {"v": 1, "ts": ts, "actor": f"box:{iid}", "event": event,
         "instance_id": iid, "nonce": nonce or (ts[-4:] + event[:2])}
    d.update(fields)
    return json.dumps(d)


def test_fold_box_events_empty():
    v = jm.fold_box_events([])
    assert v["parked"] is False and v["drained_pending"] is False
    assert v["n_events"] == 0 and v["instance_id"] is None


def test_fold_box_events_parked_self():
    v = jm.fold_box_events([
        _bev("parked_self", T(3), reason="drained", idle_s=612, n_done=4, n_failed=1),
    ])
    assert v["parked"] is True and v["park_reason"] == "drained"
    assert v["n_done"] == 4 and v["n_failed"] == 1 and v["idle_s"] == 612
    assert v["parked_ts"] == T(3) and v["drained_pending"] is False


def test_fold_box_events_drained_pending_until_park():
    # a `drained` with no later parked_self = the box asked the laptop to park it
    v = jm.fold_box_events([_bev("drained", T(2), reason="no_job")])
    assert v["drained_pending"] is True and v["parked"] is False
    assert v["park_reason"] == "no_job"
    # a later parked_self supersedes it
    v = jm.fold_box_events([
        _bev("drained", T(2), reason="no_job"),
        _bev("parked_self", T(4), reason="no_job", n_done=0, n_failed=0),
    ])
    assert v["parked"] is True and v["drained_pending"] is False


def test_fold_box_events_tolerates_junk_and_unknown():
    v = jm.fold_box_events([
        "not json", "{}", json.dumps({"ts": T(1)}),          # 3 parse errors
        _bev("jobd_up", T(1)),                               # unknown-to-fold, tolerated
        _bev("parked_self", T(5), reason="drained", n_done=2, n_failed=0),
    ])
    assert v["parse_errors"] == 3 and v["parked"] is True
    assert v["last_event"] == "parked_self"


def test_emit_and_read_box_roundtrip():
    fake = FakeB2()
    jm.emit_box_event("42", "parked_self", runner=fake, bucket="bkt",
                      reason="drained", idle_s=600, n_done=3, n_failed=0)
    v = jm.read_box("42", runner=fake, bucket="bkt", cache_dir=None)
    # read_box caches to ~/.cache; pass an explicit dir to stay hermetic
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        v = jm.read_box("42", runner=fake, bucket="bkt", cache_dir=td)
    assert v["parked"] and v["park_reason"] == "drained" and v["n_done"] == 3
    assert v["instance_id"] == "42"


def test_box_events_do_not_touch_job_terminal_set():
    # the frozen per-job sets are unchanged by the box-lifecycle additions
    # (`cancelled` is a per-JOB terminal, added with the operator cancel path).
    assert jm.TERMINAL == frozenset({"done", "failed", "cancelled"})
    assert "parked_self" not in jm.EVENTS and "drained" not in jm.EVENTS
    assert "parked_self" in jm.BOX_TERMINAL


def test_fold_cancelled_is_terminal_and_non_resumable():
    """`herdd job cancel` writes a terminal `cancelled` event. It folds to a
    terminal, non-resumable status even for a job that WAS running on a still-live
    box — the distinguishing property from `interrupted` (which folds a dead-box
    claimed/started job and DOES resume)."""
    evs = [ev("submitted", T(1), actor="cli:h", box="44"),
           ev("claimed", T(2), instance_id="44"),
           ev("started", T(3), instance_id="44"),
           ev("cancelled", T(5), instance_id="44", reason="doomed run")]
    # live box: still terminal `cancelled`, NOT "running" (cancel overrides liveness)
    v = jm.fold_events(evs, live_iids={"44"})
    assert v["status"] == "cancelled" and v["display_status"] == "cancelled"
    assert v["status"] in jm.TERMINAL and v["fail_reason"] == "doomed run"
    # dead box: same terminal cancelled (never revived as interrupted)
    dead = jm.fold_events(evs, live_iids=())
    assert dead["display_status"] == "cancelled"


def test_fold_real_outcome_beats_late_cancel():
    """A genuine done/failed (the entrypoint reached an outcome, results on B2)
    outranks a `cancelled` written a beat later — so a race between a finishing
    job and a cancel reports the real result, not a spurious cancellation."""
    base = [ev("started", T(2), instance_id="9")]
    done = jm.fold_events(base + [ev("done", T(4), rc=0),
                                  ev("cancelled", T(5), reason="too late")],
                          live_iids={"9"})
    assert done["status"] == "done"
    failed = jm.fold_events(base + [ev("failed", T(4), rc=2, reason="boom"),
                                    ev("cancelled", T(5), reason="too late")],
                            live_iids={"9"})
    assert failed["status"] == "failed" and failed["fail_reason"] == "boom"


def test_cancel_marker_roundtrip():
    fake = FakeB2()
    assert jm.has_cancel_marker("j1", runner=fake, bucket="bkt") is False
    ok, _ = jm.write_cancel_marker("j1", actor="cli:h", reason="stop it",
                                   runner=fake, bucket="bkt")
    assert ok and jm.has_cancel_marker("j1", runner=fake, bucket="bkt") is True
    body = json.loads(fake.store["jobs/j1/CANCEL"])
    assert body["reason"] == "stop it" and body["actor"] == "cli:h"


def test_checkpoint_now_marker_roundtrip():
    fake = FakeB2()
    assert jm.has_checkpoint_now_marker("j1", runner=fake, bucket="bkt") is False
    ok, _ = jm.write_checkpoint_now_marker("j1", actor="cli:h", reason="pre-park",
                                           runner=fake, bucket="bkt")
    assert ok and jm.has_checkpoint_now_marker("j1", runner=fake, bucket="bkt") is True
    body = json.loads(fake.store["jobs/j1/CHECKPOINT_NOW"])
    assert body["reason"] == "pre-park" and body["actor"] == "cli:h"


def test_a_flush_marker_is_not_a_cancel_marker():
    """Separate keys, and neither probe may answer for the other — the two ride
    the same box-side poll and a crossed key would turn a flush into a kill."""
    fake = FakeB2()
    jm.write_checkpoint_now_marker("j1", runner=fake, bucket="bkt")
    assert jm.has_cancel_marker("j1", runner=fake, bucket="bkt") is False
    assert "jobs/j1/CANCEL" not in fake.store


def test_flush_marker_rejects_a_bad_job_id():
    fake = FakeB2()
    with pytest.raises(jm.JobmetaError):
        jm.write_checkpoint_now_marker("../evil", runner=fake, bucket="bkt")
    assert not fake.store


# --------------------------------------------------------------------------- #
# job wait --until decision (pure)
# --------------------------------------------------------------------------- #
def test_wait_decision_terminal_meta():
    assert jm.wait_decision({"status": "running", "display_status": "running"},
                            "terminal") == "pending"
    for s in ("done", "failed", "cancelled"):
        assert jm.wait_decision({"status": s, "display_status": s},
                                "terminal") == "match"


def test_wait_decision_specific_state_and_display():
    # match on the folded status
    assert jm.wait_decision({"status": "done", "display_status": "done"},
                            "done") == "match"
    # match on a display_status that is not a folded status (running/interrupted)
    assert jm.wait_decision({"status": "submitted", "display_status": "running"},
                            "running") == "match"
    assert jm.wait_decision({"status": "submitted", "display_status": "queued"},
                            "queued") == "match"


def test_wait_decision_unreachable_and_pending():
    # asked for done, but the job FAILED -> can never reach done
    assert jm.wait_decision({"status": "failed", "display_status": "failed"},
                            "done") == "unreachable"
    # asked for running, job already terminal -> unreachable
    assert jm.wait_decision({"status": "done", "display_status": "done"},
                            "running") == "unreachable"
    # not there yet, not terminal -> keep polling
    assert jm.wait_decision({"status": "submitted", "display_status": "queued"},
                            "running") == "pending"


# --- submit_with_id (M2-T2 s1-submit) ----------------------------------------
_JID = "20260713T120000-a1b2c3d4-score-a0"
_CFG = {"version": 1, "name": "p", "entrypoint": "run.sh", "timeout_s": 60,
        "env": {}, "results": [], "needs": {"gpu": False, "venv": "none"}}


def test_submit_with_id_fresh():
    fake = FakeB2()
    r = jm.submit_with_id(_JID, _CFG, "44", bundle_sha256="sha1",
                          actor="cli:h", runner=fake, bucket="bkt")
    assert r["status"] == "submitted"
    assert jm.read_ticket("44", _JID, runner=fake, bucket="bkt") is not None
    assert jm.has_events(_JID, runner=fake, bucket="bkt") is True


def test_submit_with_id_identical_noop():
    fake = FakeB2()
    r1 = jm.submit_with_id(_JID, _CFG, "44", bundle_sha256="sha1",
                           actor="cli:h", runner=fake, bucket="bkt")
    assert r1["status"] == "submitted"
    r2 = jm.submit_with_id(_JID, _CFG, "44", bundle_sha256="sha1",
                           actor="cli:h", runner=fake, bucket="bkt")
    assert r2["status"] == "noop"
    n_submitted = len([k for k in fake.store if k.startswith(f"jobs/{_JID}/events/")])
    assert n_submitted == 1


def test_submit_with_id_conflict():
    fake = FakeB2()
    jm.submit_with_id(_JID, _CFG, "44", bundle_sha256="sha1",
                      actor="cli:h", runner=fake, bucket="bkt")
    with pytest.raises(jm.JobmetaError):
        jm.submit_with_id(_JID, _CFG, "44", bundle_sha256="DIFFERENT",
                          actor="cli:h", runner=fake, bucket="bkt")


# --------------------------------------------------------------------------- #
# requeue_ticket — the re-open core (ticket reconstruction + `resumed`)
# --------------------------------------------------------------------------- #
def _seed_failed(fake, jid=_JID, box="44"):
    """A job that ran and FAILED on `box`, with its queue ticket still present
    (jobd never deletes tickets) and a prior checkpoint on B2."""
    jm.submit_with_id(jid, _CFG, box, bundle_sha256="sha1", actor="cli:h",
                      runner=fake, bucket="bkt")
    for e, kw in [("claimed", {}), ("started", {}),
                  ("failed", {"rc": 16, "reason": "rc=16"})]:
        jm.emit_event(jid, e, actor=f"box:{box}", instance_id=box,
                      runner=fake, bucket="bkt", **kw)
    fake.store[f"jobs/{jid}/checkpoints/gens_a.jsonl"] = "partial\n"
    fake.store[f"jobs/{jid}/results.DONE.json"] = json.dumps({"rc": 16})
    return jm.read_job(jid, runner=fake, bucket="bkt")


def test_requeue_ticket_reopens_the_fold(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    fake = FakeB2()
    assert _seed_failed(fake)["status"] == "failed"

    r = jm.requeue_ticket(_JID, "55", _CFG, "sha1", old_box="44", attempt=2,
                          actor="cli:h", runner=fake, bucket="bkt")
    assert r["status"] == "requeued"

    # 1. a ticket on the NEW box carrying both markers
    tk = jm.read_ticket("55", _JID, runner=fake, bucket="bkt")
    assert tk["box"] == "55" and tk["bundle_sha256"] == "sha1"
    assert tk["retargeted_from"] == "44"          # jobd checkpoint pull-back
    assert tk[jm.REQUEUE_TICKET_MARK] == r["requeued_ts"]
    assert tk["config"] == _CFG                   # ticket reconstructed verbatim

    # 2. the old queue pointer is gone (no double-run if box 44 comes back)
    assert jm.read_ticket("44", _JID, runner=fake, bucket="bkt") is None
    assert r["old_ticket_deleted"] is True

    # 3. the fold re-opens, off a `resumed` — no new event kind
    v = jm.read_job(_JID, runner=fake, bucket="bkt", live_iids={"55"})
    assert v["reopened"] is True and v["status"] == "submitted"
    assert v["prior_rc"] == 16
    bodies = [json.loads(b) for k, b in fake.store.items()
              if k.startswith(f"jobs/{_JID}/events/")]
    kinds = {e["event"] for e in bodies}
    assert kinds <= jm.EVENTS, f"emitted an event outside the frozen set: {kinds}"
    res = [e for e in bodies if e["event"] == "resumed"]
    assert len(res) == 1
    assert res[0]["kind"] == "requeue" and res[0]["retargeted_from"] == "44"
    assert res[0]["from_box"] == "44" and res[0]["instance_id"] == "55"
    assert res[0]["attempt"] == 2

    # 4. the checkpoints stay where the SAME JOB_ID already keeps them — the
    #    manual recipe's rclone-copy seed step is unnecessary by construction
    assert fake.store[f"jobs/{_JID}/checkpoints/gens_a.jsonl"] == "partial\n"


def test_requeue_reopens_even_when_the_failure_is_in_the_SAME_millisecond():
    """The un-stick must not be decided by a coin flip.

    `now_ts` is millisecond-resolution and `_ev_order` breaks a ts tie on the
    RANDOM nonce, so a requeue issued inside the same millisecond as the `failed`
    used to re-open the job only ~50% of the time — the ~8% flake this test
    pins shut (same-ms ~16% of runs x the 50% coin flip). Freezing the clock
    makes that window certain instead of rare: requeue_ticket must step its
    `resumed` past the newest event on record rather than trusting the clock.

    Sampled, not merely asserted once: the tie-break is random, so a single pass
    proves nothing about a bug whose failure rate is one half."""
    for _ in range(32):
        fake = FakeB2()
        # the `monkeypatch` FIXTURE is function-scoped and cannot be undone
        # per-iteration; MonkeyPatch.context() is the same tool, scoped to a block
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(jm, "now_ts", lambda: "20260710T120000000Z")
            _seed_failed(fake)                    # every event at the frozen ms
            r = jm.requeue_ticket(_JID, "55", _CFG, "sha1", old_box="44",
                                  actor="cli:h", runner=fake, bucket="bkt")
        v = jm.read_job_fresh(_JID, runner=fake, bucket="bkt", live_iids={"55"})
        assert v["reopened"] is True and v["status"] == "submitted", (
            f"requeue did not un-stick the fold: status={v['status']!r}")
        # strictly newer, so the order does not depend on the nonce at all
        assert r["requeued_ts"] > "20260710T120000000Z"


def test_requeue_ticket_raises_when_the_resumed_event_cannot_land(tmp_path):
    """Fail-closed: a ticket without its `resumed` would run on a box while every
    reader still called the job dead. Loud, never silent."""
    fake = FakeB2()
    _seed_failed(fake)
    real = fake.__call__

    def _no_events(args, input=None):
        if args[0] == "rcat" and "/events/" in args[1]:
            return 1, "", "b2 down"
        return real(args, input=input)

    with pytest.raises(jm.JobmetaError, match="FAILED to emit"):
        jm.requeue_ticket(_JID, "55", _CFG, "sha1", old_box="44", actor="cli:h",
                          runner=_no_events, bucket="bkt")


def test_requeue_ticket_raises_when_the_ticket_write_fails():
    fake = FakeB2()
    _seed_failed(fake)
    real = fake.__call__

    def _no_ticket(args, input=None):
        if args[0] == "rcat" and "/queue/" in args[1]:
            return 1, "", "b2 down"
        return real(args, input=input)

    with pytest.raises(jm.JobmetaError, match="ticket write failed"):
        jm.requeue_ticket(_JID, "55", _CFG, "sha1", old_box="44", actor="cli:h",
                          runner=_no_ticket, bucket="bkt")


# --- results.DONE + generation artifact-manifest validation (M2-T2 s3-manifest)
def _seed_manifest_job(fake, jid, *, arm_sha=None, kind="e2-generations"):
    body = "gen-line-1\ngen-line-2\n"
    arm_sha = hashlib.sha256(body.encode()).hexdigest() if arm_sha is None else arm_sha
    manifest = {"v": 1, "kind": kind,
                "arms": {"a": {"path": "results/gens_a.jsonl", "sha256": arm_sha,
                               "rows": 2}}}
    fake.store[f"jobs/{jid}/results.DONE.json"] = json.dumps({"rc": 0})
    fake.store[f"jobs/{jid}/results/results/artifact-manifest.json"] = json.dumps(manifest)
    fake.store[f"jobs/{jid}/results/results/gens_a.jsonl"] = body
    return manifest


def test_validate_generation_artifact_pass():
    fake = FakeB2()
    manifest = _seed_manifest_job(fake, "j1")
    r = jm.validate_generation_artifact("j1", expect_kind="e2-generations",
                                        runner=fake, bucket="bkt")
    assert r["kind"] == "e2-generations"
    assert r["manifest_sha256"] == hashlib.sha256(
        json.dumps(manifest).encode()).hexdigest()
    assert len(r["manifest_sha256"]) >= 12


def test_validate_generation_artifact_bad_hash():
    fake = FakeB2()
    _seed_manifest_job(fake, "j1", arm_sha="0" * 64)
    with pytest.raises(jm.JobmetaError):
        jm.validate_generation_artifact("j1", expect_kind="e2-generations",
                                        runner=fake, bucket="bkt",
                                        sleep_fn=lambda s: None)


def test_validate_generation_artifact_stale_overwrite_read_retries():
    """B2 overwrite eventual-consistency (live 2026-07-15 E2 incident): the
    arm file is checkpoint-synced empty DURING the run and overwritten with
    the full object at finalize; a read moments later can return the STALE
    empty version. The validator must re-read on mismatch — the first `cat`
    here serves empty bytes, the retry serves the settled full object, and
    validation SUCCEEDS with no real sleeping."""
    fake = FakeB2()
    _seed_manifest_job(fake, "j1")
    reads = {"n": 0}
    sleeps = []

    def racy_runner(args, input=None):
        if args[0] == "cat" and args[1].endswith("results/gens_a.jsonl"):
            reads["n"] += 1
            if reads["n"] == 1:
                return 0, "", ""    # stale pre-finalize checkpoint version
        return fake(args, input=input)

    r = jm.validate_generation_artifact("j1", expect_kind="e2-generations",
                                        runner=racy_runner, bucket="bkt",
                                        sleep_fn=sleeps.append)
    assert r["kind"] == "e2-generations"
    assert reads["n"] == 2                      # exactly one re-read
    # an EMPTY read routes to the stale schedule: one backoff, injected
    assert sleeps == [jm.ARM_SHA_STALE_BACKOFF_S]


def test_validate_generation_artifact_stale_empty_outlasts_old_budget():
    """Live recurrence 2026-07-16 (E2 run 0d9d): the stale EMPTY checkpoint
    version persisted for LONGER than the whole ~18s ARM_SHA_RETRIES budget,
    so a fully-valid generate raised a false ARTIFACT_INVALID. An empty read
    is never real corruption — it must be waited out on the LONG
    ARM_SHA_STALE_* exponential schedule: here the object stays empty for 6
    reads (past the old 3-retry budget) and validation still SUCCEEDS, with
    every backoff injected (no real sleeping)."""
    fake = FakeB2()
    _seed_manifest_job(fake, "j1")
    reads = {"n": 0}
    sleeps = []

    def racy_runner(args, input=None):
        if args[0] == "cat" and args[1].endswith("results/gens_a.jsonl"):
            reads["n"] += 1
            if reads["n"] <= 6:
                return 0, "", ""    # stale empty version, past the old budget
        return fake(args, input=input)

    r = jm.validate_generation_artifact("j1", expect_kind="e2-generations",
                                        runner=racy_runner, bucket="bkt",
                                        sleep_fn=sleeps.append)
    assert r["kind"] == "e2-generations"
    assert reads["n"] == 7
    assert len(sleeps) == 6 > jm.ARM_SHA_RETRIES   # retried PAST the old budget
    # exponential, capped: 2, 4, 8, 16, 32, 32 — all injected, none real
    assert sleeps == [2.0, 4.0, 8.0, 16.0, 32.0, 32.0]


def test_validate_generation_artifact_short_read_is_stale_not_corrupt():
    """A NON-empty read that is still SHORTER than the manifest arm's declared
    `rows` (a partial checkpoint copy) is classified stale — retried on the
    long budget, not fast-failed as corruption."""
    fake = FakeB2()
    _seed_manifest_job(fake, "j1")
    reads = {"n": 0}
    sleeps = []

    def racy_runner(args, input=None):
        if args[0] == "cat" and args[1].endswith("results/gens_a.jsonl"):
            reads["n"] += 1
            if reads["n"] <= 4:
                return 0, "gen-line-1\n", ""    # 1 of the declared 2 rows
        return fake(args, input=input)

    r = jm.validate_generation_artifact("j1", expect_kind="e2-generations",
                                        runner=racy_runner, bucket="bkt",
                                        sleep_fn=sleeps.append)
    assert r["kind"] == "e2-generations"
    assert len(sleeps) == 4 > jm.ARM_SHA_RETRIES   # survived past the fast budget
    assert sleeps == [2.0, 4.0, 8.0, 16.0]


def test_validate_generation_artifact_stale_empty_without_declared_rows():
    """Fallback for a manifest whose arm declares NO `rows` (older bundle): an
    EMPTY read is still classified stale — empty is never real corruption
    regardless of whether a row count is available — so it rides the long
    ARM_SHA_STALE_* schedule and settles, rather than fast-failing."""
    fake = FakeB2()
    body = "gen-line-1\n"
    arm_sha = hashlib.sha256(body.encode()).hexdigest()
    manifest = {"v": 1, "kind": "e2-generations",
                "arms": {"a": {"path": "results/gens_a.jsonl",
                               "sha256": arm_sha}}}   # no `rows`
    fake.store["jobs/j1/results.DONE.json"] = json.dumps({"rc": 0})
    fake.store["jobs/j1/results/results/artifact-manifest.json"] = json.dumps(manifest)
    fake.store["jobs/j1/results/results/gens_a.jsonl"] = body
    reads = {"n": 0}
    sleeps = []

    def racy_runner(args, input=None):
        if args[0] == "cat" and args[1].endswith("results/gens_a.jsonl"):
            reads["n"] += 1
            if reads["n"] <= 2:
                return 0, "", ""    # stale empty version, no rows to compare
        return fake(args, input=input)

    r = jm.validate_generation_artifact("j1", expect_kind="e2-generations",
                                        runner=racy_runner, bucket="bkt",
                                        sleep_fn=sleeps.append)
    assert r["kind"] == "e2-generations"
    assert sleeps == [2.0, 4.0]     # long stale schedule, all injected


def test_validate_generation_artifact_persistent_mismatch_still_raises():
    """Guard against the retry masking real corruption: an arm file whose
    bytes are FULL-LENGTH (at the declared rows) but NEVER match the declared
    sha is genuine corruption — it exhausts only the FAST linear budget and
    raises ARTIFACT_INVALID's JobmetaError without ever touching the long
    stale schedule (a real bad artifact must not be masked for minutes)."""
    fake = FakeB2()
    _seed_manifest_job(fake, "j1", arm_sha="0" * 64)
    sleeps = []
    with pytest.raises(jm.JobmetaError, match="sha256 mismatch"):
        jm.validate_generation_artifact("j1", expect_kind="e2-generations",
                                        runner=fake, bucket="bkt",
                                        sleep_fn=sleeps.append)
    assert len(sleeps) == jm.ARM_SHA_RETRIES    # fast budget only: 3, 6, 9
    assert sum(sleeps) == 18.0                  # nowhere near the ~158s stale budget


def test_validate_generation_artifact_wrong_kind():
    fake = FakeB2()
    _seed_manifest_job(fake, "j1", kind="e2-generations")
    with pytest.raises(jm.JobmetaError):
        jm.validate_generation_artifact("j1", expect_kind="e2-scores",
                                        runner=fake, bucket="bkt")


def test_validate_generation_artifact_missing_done():
    fake = FakeB2()
    _seed_manifest_job(fake, "j1")
    del fake.store["jobs/j1/results.DONE.json"]
    with pytest.raises(jm.JobmetaError):
        jm.validate_generation_artifact("j1", expect_kind="e2-generations",
                                        runner=fake, bucket="bkt")


def test_validate_generation_artifact_custom_manifest_path():
    """`manifest_path` is WORKDIR-RELATIVE and resolved under
    jobs/<job_id>/results/ — a bundle whose manifest is NOT at the e2
    default double-'results' key validates with its own path, and the
    default call (which looks at the e2 key) correctly fails."""
    fake = FakeB2()
    body = "gen-line-1\n"
    arm_sha = hashlib.sha256(body.encode()).hexdigest()
    manifest = {"v": 1, "kind": "e2-generations",
                "arms": {"a": {"path": "out/gens_a.jsonl", "sha256": arm_sha}}}
    fake.store["jobs/j1/results.DONE.json"] = json.dumps({"rc": 0})
    fake.store["jobs/j1/results/out/custom-manifest.json"] = json.dumps(manifest)
    fake.store["jobs/j1/results/out/gens_a.jsonl"] = body

    r = jm.validate_generation_artifact(
        "j1", expect_kind="e2-generations", runner=fake, bucket="bkt",
        manifest_path="out/custom-manifest.json")
    assert r["kind"] == "e2-generations"
    assert r["manifest_sha256"] == hashlib.sha256(
        json.dumps(manifest).encode()).hexdigest()

    # the DEFAULT manifest_path reads the e2 key, which this bundle never wrote
    with pytest.raises(jm.JobmetaError, match="manifest missing"):
        jm.validate_generation_artifact("j1", expect_kind="e2-generations",
                                        runner=fake, bucket="bkt")


# --- box publish-verify trust gate (the 5x cross-client false-fail fix) --------
def _add_publish_verified(fake, jid, files):
    """Seed a box-side `publish_verified` event covering `files` (results/-relative
    paths), the positive signal jobd stamps once every uploaded result read back
    from B2 at its final sha. Comma-joined `files`, matching jobd.sh's emit."""
    jm.emit_event(jid, "publish_verified", actor="box:44", runner=fake, bucket="bkt",
                  instance_id="44", files=",".join(files))


def test_validate_generation_artifact_box_verified_skips_body_reread():
    """THE 5x-failure bug (2026-07-15/16/19/20 + run 20260720T004614): the box
    published every arm and its OWN publish-verify PASSED (arm durable+correct on
    B2), but the CONTROLLER — a different B2 client/edge — read that same arm as
    EMPTY for longer than its whole retry budget and false-failed the stage. With
    a `publish_verified` event covering the arm, the controller must TRUST the box
    and never re-read the body, so a stale-empty cross-client read cannot fail it."""
    fake = FakeB2()
    _seed_manifest_job(fake, "j1")
    _add_publish_verified(fake, "j1", ["results/gens_a.jsonl"])
    body_reads = {"n": 0}

    def racy_runner(args, input=None):
        # simulate the controller's stale cross-client read of the arm BODY:
        # always empty. If the trust gate works this is never called.
        if args[0] == "cat" and args[1].endswith("results/results/gens_a.jsonl"):
            body_reads["n"] += 1
            return 0, "", ""
        return fake(args, input=input)

    r = jm.validate_generation_artifact("j1", expect_kind="e2-generations",
                                        runner=racy_runner, bucket="bkt",
                                        sleep_fn=lambda s: None)
    assert r["kind"] == "e2-generations"
    assert body_reads["n"] == 0                 # arm body NEVER re-downloaded
    assert r["manifest_sha256"]


def test_validate_generation_artifact_box_verified_absolute_arm_path():
    """The manifest may record an ABSOLUTE box arm path; the trust match
    normalizes both sides to the results/-relative suffix the box's `files` list
    carries, so coverage is still recognized."""
    fake = FakeB2()
    body = "gen-line-1\ngen-line-2\n"
    arm_sha = hashlib.sha256(body.encode()).hexdigest()
    manifest = {"v": 1, "kind": "e2-generations",
                "arms": {"a": {"path": "/workspace/jobs/j1/work/results/gens_a.jsonl",
                               "sha256": arm_sha, "rows": 2}}}
    fake.store["jobs/j1/results.DONE.json"] = json.dumps({"rc": 0})
    fake.store["jobs/j1/results/results/artifact-manifest.json"] = json.dumps(manifest)
    # deliberately do NOT seed the arm body — trust must skip the read entirely.
    _add_publish_verified(fake, "j1", ["results/gens_a.jsonl"])
    r = jm.validate_generation_artifact("j1", expect_kind="e2-generations",
                                        runner=fake, bucket="bkt",
                                        sleep_fn=lambda s: None)
    assert r["kind"] == "e2-generations"


def test_validate_generation_artifact_no_publish_verified_falls_back():
    """Fallback regression: with NO publish_verified event (older bundle) the
    existing per-arm re-read+retry path runs unchanged — a matching body passes,
    and an empty body still exhausts the stale budget and raises."""
    fake = FakeB2()
    _seed_manifest_job(fake, "j1")
    # no publish_verified event -> backstop; matching body passes as before.
    r = jm.validate_generation_artifact("j1", expect_kind="e2-generations",
                                        runner=fake, bucket="bkt")
    assert r["kind"] == "e2-generations"

    # and an arm whose body never settles still raises via the backstop.
    fake2 = FakeB2()
    _seed_manifest_job(fake2, "j1")

    def always_empty(args, input=None):
        if args[0] == "cat" and args[1].endswith("results/results/gens_a.jsonl"):
            return 0, "", ""
        return fake2(args, input=input)

    with pytest.raises(jm.JobmetaError, match="empty/short"):
        jm.validate_generation_artifact("j1", expect_kind="e2-generations",
                                        runner=always_empty, bucket="bkt",
                                        sleep_fn=lambda s: None)


def test_validate_generation_artifact_publish_verify_failed_not_trusted():
    """A `publish_verify_failed` event is NOT a positive signal — the controller
    must NOT trust it and must run the backstop (which here catches a real
    empty/stale body and raises)."""
    fake = FakeB2()
    _seed_manifest_job(fake, "j1")
    jm.emit_event("j1", "publish_verify_failed", actor="box:44", runner=fake,
                  bucket="bkt", instance_id="44", file="results/gens_a.jsonl")

    def always_empty(args, input=None):
        if args[0] == "cat" and args[1].endswith("results/results/gens_a.jsonl"):
            return 0, "", ""
        return fake(args, input=input)

    with pytest.raises(jm.JobmetaError, match="empty/short"):
        jm.validate_generation_artifact("j1", expect_kind="e2-generations",
                                        runner=always_empty, bucket="bkt",
                                        sleep_fn=lambda s: None)


def test_validate_generation_artifact_coverage_gap_not_trusted():
    """A publish_verified event that omits one declared arm is NOT full coverage
    — the controller must fall through to the backstop for safety rather than
    blindly pass (here the uncovered arm's body is stale-empty and raises)."""
    fake = FakeB2()
    body = "gen-line-1\ngen-line-2\n"
    arm_sha = hashlib.sha256(body.encode()).hexdigest()
    manifest = {"v": 1, "kind": "e2-generations",
                "arms": {"a": {"path": "results/gens_a.jsonl", "sha256": arm_sha,
                               "rows": 2},
                         "b": {"path": "results/gens_b.jsonl", "sha256": arm_sha,
                               "rows": 2}}}
    fake.store["jobs/j1/results.DONE.json"] = json.dumps({"rc": 0})
    fake.store["jobs/j1/results/results/artifact-manifest.json"] = json.dumps(manifest)
    fake.store["jobs/j1/results/results/gens_a.jsonl"] = body
    # arm 'b' body never lands; the event covers only arm 'a' -> not full coverage.
    _add_publish_verified(fake, "j1", ["results/gens_a.jsonl"])

    with pytest.raises(jm.JobmetaError, match="empty/short|missing"):
        jm.validate_generation_artifact("j1", expect_kind="e2-generations",
                                        runner=fake, bucket="bkt",
                                        sleep_fn=lambda s: None)


def test_validate_generation_artifact_structure_enforced_under_trust():
    """Even fully box-verified, manifest STRUCTURE is still enforced: a wrong
    kind, a bad schema version, or an arm missing path/sha256 all still RAISE."""
    # wrong kind (raises before the gate, but must still raise with the event set)
    fake = FakeB2()
    _seed_manifest_job(fake, "j1", kind="e2-generations")
    _add_publish_verified(fake, "j1", ["results/gens_a.jsonl"])
    with pytest.raises(jm.JobmetaError, match="kind"):
        jm.validate_generation_artifact("j1", expect_kind="e2-scores",
                                        runner=fake, bucket="bkt")

    # arm missing sha256, under full trust -> raises (the fallback would SKIP it)
    fake2 = FakeB2()
    manifest = {"v": 1, "kind": "e2-generations",
                "arms": {"a": {"path": "results/gens_a.jsonl"}}}  # no sha256
    fake2.store["jobs/j1/results.DONE.json"] = json.dumps({"rc": 0})
    fake2.store["jobs/j1/results/results/artifact-manifest.json"] = json.dumps(manifest)
    _add_publish_verified(fake2, "j1", ["results/gens_a.jsonl"])
    with pytest.raises(jm.JobmetaError, match="sha256"):
        jm.validate_generation_artifact("j1", expect_kind="e2-generations",
                                        runner=fake2, bucket="bkt")

    # bad schema version still raises even with a covering event
    fake3 = FakeB2()
    manifest3 = {"v": 2, "kind": "e2-generations",
                 "arms": {"a": {"path": "results/gens_a.jsonl", "sha256": "0" * 64}}}
    fake3.store["jobs/j1/results.DONE.json"] = json.dumps({"rc": 0})
    fake3.store["jobs/j1/results/results/artifact-manifest.json"] = json.dumps(manifest3)
    _add_publish_verified(fake3, "j1", ["results/gens_a.jsonl"])
    with pytest.raises(jm.JobmetaError, match="v!="):
        jm.validate_generation_artifact("j1", expect_kind="e2-generations",
                                        runner=fake3, bucket="bkt")


def test_read_job_events_resilient_to_bad_listing():
    """read_job_events never raises: an lsf failure or an unparseable body just
    yields fewer events (safe negative -> backstop), never a crash."""
    def broken(args, input=None):
        if args[0] == "lsf":
            return 1, "", "boom"
        return 1, "", ""
    assert jm.read_job_events("j1", runner=broken, bucket="bkt") == []

    fake = FakeB2()
    fake.store["jobs/j1/events/bad.json"] = "{not json"
    fake.store["jobs/j1/events/ok.json"] = json.dumps(
        {"v": 1, "ts": "t", "event": "publish_verified", "job_id": "j1",
         "nonce": "n", "files": "results/gens_a.jsonl"})
    evs = jm.read_job_events("j1", runner=fake, bucket="bkt")
    assert [e["event"] for e in evs] == ["publish_verified"]


def test_publish_verified_files_parsing():
    """_publish_verified_files: None when no event; union of results/-relative
    suffixes across events; tolerates comma-joined and JSON-list `files`."""
    assert jm._publish_verified_files([]) is None
    assert jm._publish_verified_files(
        [{"event": "done"}, {"event": "publish_verify_failed",
                             "file": "results/x"}]) is None
    got = jm._publish_verified_files([
        {"event": "publish_verified", "files": "results/a.jsonl,results/b.jsonl"},
        {"event": "publish_verified", "files": ["/ws/results/c.jsonl"]},
    ])
    assert got == {"results/a.jsonl", "results/b.jsonl", "results/c.jsonl"}


# =============================================================================
# read_job_fresh — the monitoring path that cannot be masked locally
# (2026-07-30 launch postmortem §7a: `submitted live=False` minutes after the
#  box had already moved)
# =============================================================================
def _ev(fake, job_id, name, event, **fields):
    body = {"v": 1, "ts": fields.pop("ts", "2026-07-30T00:00:00Z"), "actor": "a",
            "event": event, "job_id": job_id, "nonce": name}
    body.update(fields)
    fake.store[f"jobs/{job_id}/events/{name}.json"] = json.dumps(body)


def test_read_job_fresh_reads_per_key_and_never_caches(tmp_path):
    fake = FakeB2()
    _ev(fake, "j1", "e1", "submitted", box="42")
    _ev(fake, "j1", "e2", "claimed", instance_id="42")
    v = jm.read_job_fresh("j1", runner=fake, live_iids={"42"}, bucket="bkt")
    assert v["fresh"] is True and v["status"] == "claimed"
    assert v["unclaimed"] is False and v["live"] is True
    # per-key reads only: no `copy` (the cached path's transport) was issued
    assert not any(c[0] == "copy" for c in getattr(fake, "calls", []) or [])


def test_read_job_fresh_marks_unclaimed_so_live_false_is_not_reported_as_dead():
    """The exact misleading snapshot: only the CLI's own `submitted` event is
    visible, so instance_id is None and `live` is structurally False. `unclaimed`
    is what lets a caller render n/a instead of "the box is dead"."""
    fake = FakeB2()
    _ev(fake, "j2", "e1", "submitted", box="42")
    v = jm.read_job_fresh("j2", runner=fake, live_iids={"42"}, bucket="bkt")
    assert v["status"] == "submitted" and v["live"] is False
    assert v["unclaimed"] is True          # <- "no news", not "not running"
    assert v["done_marker"] is False


def test_read_job_fresh_done_marker_outranks_a_lagging_fold():
    """results.DONE.json is written once, as a NEW key, LAST — a `cat` hit is
    strong evidence the job finished even when no `done` event has surfaced."""
    fake = FakeB2()
    _ev(fake, "j3", "e1", "submitted", box="42")
    fake.store["jobs/j3/results.DONE.json"] = json.dumps({"rc": 0, "n_results": 2})
    v = jm.read_job_fresh("j3", runner=fake, live_iids=set(), bucket="bkt")
    assert v["status"] == "submitted"      # the fold is behind ...
    assert v["done_marker"] is True        # ... and this says so


def test_read_job_fresh_tolerates_an_empty_or_unreadable_log():
    fake = FakeB2()
    v = jm.read_job_fresh("j4", runner=fake, live_iids=set(), bucket="bkt")
    assert v["status"] == "unknown" and v["unclaimed"] is True
    assert v["done_marker"] is False


# =============================================================================
# WHOSE ATTEMPT IS THE DONE MARKER?  (false terminal, measured 2026-08-28 on
# 20260828T064840-v16-r64-8c87 — rc=3, requeued, retargeted, 42% into training,
# and `job status --fresh` said it FINISHED)
# =============================================================================
def _requeued_job(fake, jid, *, marker_rc=3, marker=True):
    """failed -> requeued (`resumed`) -> claimed elsewhere: a job that is RUNNING
    with the dead attempt's DONE marker still sitting at its key."""
    _ev(fake, jid, "e1", "submitted", ts="20260828T064840000Z", box="41")
    _ev(fake, jid, "e2", "started", ts="20260828T070000000Z", instance_id="41")
    _ev(fake, jid, "e3", "failed", ts="20260828T100000000Z", instance_id="41",
        rc=marker_rc, reason=f"rc={marker_rc}")
    _ev(fake, jid, "e4", "resumed", ts="20260828T104316128Z", kind="requeue")
    _ev(fake, jid, "e5", "claimed", ts="20260828T104500000Z", instance_id="42")
    if marker:
        fake.store[f"jobs/{jid}/results.DONE.json"] = json.dumps(
            {"rc": marker_rc, "n_results": 7})
    return jid


def test_fold_publishes_the_attempt_boundary_a_requeue_creates():
    fake = FakeB2()
    _requeued_job(fake, "rq1")
    v = jm.read_job_fresh("rq1", runner=fake, live_iids={"42"}, bucket="bkt")
    assert v["reopened"] is True
    assert v["reopened_at"] == "20260828T104316128Z"     # the `resumed`, not the fail


def test_fold_leaves_the_boundary_none_for_a_job_that_was_never_reopened():
    fake = FakeB2()
    _ev(fake, "nr1", "e1", "submitted", ts="20260828T064840000Z", box="41")
    _ev(fake, "nr1", "e2", "started", ts="20260828T070000000Z", instance_id="41")
    assert jm.fold_events([json.dumps({"v": 1, "ts": "20260828T064840000Z",
                                       "actor": "a", "event": "submitted",
                                       "job_id": "nr1", "nonce": "e1"})]
                          )["reopened_at"] is None


def test_a_prior_attempts_marker_is_graded_stale_not_finished():
    """THE INCIDENT. The marker is real, the fold says running, and the two are
    describing DIFFERENT attempts. `done_marker` stays True (the object IS
    there); the verdict is what stops a reader calling the job finished."""
    fake = FakeB2()
    _requeued_job(fake, "rq2")
    fake.mtimes = {"jobs/rq2/results.DONE.json": "2026-08-28T10:00:12.000000000Z"}
    v = jm.read_job_fresh("rq2", runner=fake, live_iids={"42"}, bucket="bkt")
    assert v["status"] == "claimed" and v["done_marker"] is True
    assert v["done_marker_verdict"] == jm.DONE_MARKER_STALE
    assert v["done_marker_ts"] == "20260828T100012000Z"   # before the re-open
    assert v["done_marker_rc"] == 3


def test_a_marker_written_after_the_reopen_is_still_a_genuine_b2_list_lag():
    """The condition the note was RIGHT about must not regress: a marker newer
    than the re-open belongs to the current attempt, which finished ahead of its
    own `done` event surfacing in the LIST."""
    fake = FakeB2()
    _requeued_job(fake, "rq3", marker_rc=0)
    fake.mtimes = {"jobs/rq3/results.DONE.json": "2026-08-28T11:59:00Z"}
    v = jm.read_job_fresh("rq3", runner=fake, live_iids={"42"}, bucket="bkt")
    assert v["status"] not in jm.TERMINAL
    assert v["done_marker_verdict"] == jm.DONE_MARKER_CURRENT


def test_an_undatable_marker_on_a_reopened_job_is_unknown_never_current():
    """Fail-closed: 'we could not tell' and 'it finished' are the two answers
    this seam exists to keep apart, so an unlistable mtime does NOT read as
    current."""
    fake = FakeB2()
    _requeued_job(fake, "rq4")
    fake.mtimes = {}                       # listed, but with no ModTime field
    v = jm.read_job_fresh("rq4", runner=fake, live_iids={"42"}, bucket="bkt")
    assert v["done_marker"] is True
    assert v["done_marker_ts"] is None
    assert v["done_marker_verdict"] == jm.DONE_MARKER_UNKNOWN


def test_the_markers_own_written_ts_outranks_the_b2_mtime_and_costs_no_listing():
    """A marker that carries jobd's stamp dates itself — no `lsjson` is issued
    at all, and the stamp wins over a contradicting mtime."""
    fake = FakeB2()
    _requeued_job(fake, "rq5")
    fake.store["jobs/rq5/results.DONE.json"] = json.dumps(
        {"rc": 3, "written_ts": "20260828T100012000Z", "instance_id": "41"})
    fake.mtimes = {"jobs/rq5/results.DONE.json": "2026-08-28T23:59:59Z"}
    seen = []

    def _rec(args, **kw):
        seen.append(list(args))
        return fake(args, **kw)

    v = jm.read_job_fresh("rq5", runner=_rec, live_iids={"42"}, bucket="bkt")
    assert v["done_marker_verdict"] == jm.DONE_MARKER_STALE
    assert v["done_marker_ts"] == "20260828T100012000Z"
    assert v["done_marker_box"] == "41"
    assert not [c for c in seen if c[0] == "lsjson"]


def test_a_job_that_was_never_reopened_costs_exactly_the_one_cat_it_always_did():
    """Cost contract: the mtime lookup is issued ONLY where the answer can
    change. A plain LIST-lag job must not grow a second B2 read."""
    fake = FakeB2()
    _ev(fake, "nc1", "e1", "submitted", ts="20260828T064840000Z", box="41")
    fake.store["jobs/nc1/results.DONE.json"] = json.dumps({"rc": 0, "n_results": 2})
    seen = []

    def _rec(args, **kw):
        seen.append(list(args))
        return fake(args, **kw)

    v = jm.read_job_fresh("nc1", runner=_rec, live_iids=set(), bucket="bkt")
    assert v["done_marker"] is True
    assert v["done_marker_verdict"] == jm.DONE_MARKER_CURRENT
    assert not [c for c in seen if c[0] == "lsjson"]


def test_a_legacy_marker_missing_every_new_field_never_raises():
    """Markers written before 2026-08-28 carry no `written_ts` and no
    `instance_id`; a marker can also be a bare JSON scalar. Neither may crash a
    read — the whole probe is best-effort by contract."""
    fake = FakeB2()
    for jid, body in (("lg1", "{}"), ("lg2", "12"), ("lg3", '"done"'),
                      ("lg4", '{"rc": null, "written_ts": 17}')):
        _requeued_job(fake, jid, marker=False)
        fake.store[f"jobs/{jid}/results.DONE.json"] = body
        v = jm.read_job_fresh(jid, runner=fake, live_iids={"42"}, bucket="bkt")
        assert v["done_marker"] is True
        assert v["done_marker_verdict"] == jm.DONE_MARKER_UNKNOWN


@pytest.mark.parametrize("mod,want", [
    ("2026-08-28T10:43:16.128000000Z", "20260828T104316128Z"),
    ("2026-08-28T10:43:16Z", "20260828T104316000Z"),
    ("2026-08-28T12:43:16.128+02:00", "20260828T104316128Z"),   # offset -> UTC
    ("2026-08-28T05:43:16.128-05:00", "20260828T104316128Z"),
    ("", None), ("not a time", None), (None, None),
])
def test_rclone_modtime_normalizes_to_a_lexically_orderable_utc_stamp(mod, want):
    assert jm._rclone_modtime_ts(mod) == want


def test_classify_done_marker_is_pure_and_tri_state():
    m = {"rc": 0}
    assert jm.classify_done_marker(None, "20260828T110000000Z", None) is None
    assert jm.classify_done_marker(m, None, None) == jm.DONE_MARKER_CURRENT
    assert jm.classify_done_marker(m, None, "20260828T104316128Z") == \
        jm.DONE_MARKER_UNKNOWN
    assert jm.classify_done_marker(m, "20260828T100000000Z",
                                   "20260828T104316128Z") == jm.DONE_MARKER_STALE
    assert jm.classify_done_marker(m, "20260828T110000000Z",
                                   "20260828T104316128Z") == jm.DONE_MARKER_CURRENT


# --------------------------------------------------------------------------- #
# vllm_sampler_findings — submit-time guard against the flashinfer startup JIT
#
# THE INCIDENT (three box starts and counting): with the flashinfer sampler on,
# vLLM JIT-compiles its sampling kernels at ENGINE STARTUP; the JIT #includes
# curand.h, which the baked images do not ship. nvcc fails, the engine core
# never comes up, and the job exits rc=1 BEFORE A SINGLE TOKEN — on a rented
# box, after the assets staged. Every canonical launcher pins the env var; a
# bundle that builds its own engine, or ships a STALE vendored copy of one of
# those launchers, silently opts out. This lint moves that discovery to submit.
# --------------------------------------------------------------------------- #
def _bundle(tmp_path, name, files):
    d = tmp_path / name
    for rel, body in files.items():
        p = d / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body)
    return str(d)


def test_sampler_lint_flags_a_shell_launcher_without_the_pin(tmp_path):
    b = _bundle(tmp_path, "b", {"run.sh": "#!/bin/sh\nvllm serve $MODEL --port 8000\n"})
    out = jm.vllm_sampler_findings(b, {})
    assert len(out) == 1 and out[0].startswith("run.sh starts a vLLM engine")
    assert "curand.h" in out[0]


def test_sampler_lint_flags_an_in_process_engine_without_the_pin(tmp_path):
    b = _bundle(tmp_path, "b", {
        "gen.py": "from vllm import LLM\nllm = LLM(model=base, dtype='bfloat16')\n"})
    assert len(jm.vllm_sampler_findings(b, {})) == 1


def test_sampler_lint_flags_a_stale_vendored_launcher(tmp_path):
    # The real 2026-08-03 shape: the bundle vendors a copy of the canonical
    # generator taken BEFORE the pin landed. The run.sh looks innocent.
    b = _bundle(tmp_path, "b", {
        "run.sh": "python3 witness/gen_probe_resumable.py --backend vllm\n",
        "witness/gen_probe_resumable.py": "from vllm import LLM\nllm = LLM(model=b)\n"})
    out = jm.vllm_sampler_findings(b, {})
    assert [o.split(" starts")[0] for o in out] == ["witness/gen_probe_resumable.py"]
    assert "STALE" in out[0]


def test_sampler_lint_passes_when_the_launcher_pins_it(tmp_path):
    b = _bundle(tmp_path, "b", {
        "run.sh": 'export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"\n'
                  "vllm serve $MODEL\n"})
    assert jm.vllm_sampler_findings(b, {}) == []


def test_sampler_lint_passes_when_job_config_env_sets_it(tmp_path):
    # The env: block covers the whole entrypoint, so it satisfies every file.
    b = _bundle(tmp_path, "b", {"run.sh": "vllm serve $MODEL\n"})
    assert jm.vllm_sampler_findings(b, {"VLLM_USE_FLASHINFER_SAMPLER": "0"}) == []


def test_sampler_lint_ignores_a_bundle_that_only_shells_out(tmp_path):
    # A run.sh that calls a generator which pins it ITSELF must not be nagged —
    # a lint that cries wolf gets ignored, and then the real one is missed.
    b = _bundle(tmp_path, "b", {
        "run.sh": "python3 $SRC/gen_probe_resumable.py --backend vllm --base $B\n"})
    assert jm.vllm_sampler_findings(b, {}) == []


def test_sampler_lint_ignores_docs_and_argv_builders(tmp_path):
    # `vllm serve` named in a python docstring (jobs/v3-gate-e/gen_v3.py) or
    # assembled as argv by a library (runsets/base-bakeoff/bakeoff_lib.py)
    # launches nothing in that file; nor does a shell COMMENT mentioning it.
    b = _bundle(tmp_path, "b", {
        "lib.py": '"""builds the `vllm serve` argv for the caller."""\nARGV = ["vllm", "serve"]\n',
        "notes.sh": "# we used to `vllm serve` here; now the runset does it\ntrue\n"})
    assert jm.vllm_sampler_findings(b, {}) == []


def test_sampler_lint_ignores_an_engine_named_only_inside_a_python_string(tmp_path):
    # The 2026-08-26 false positive: jobs/v1415-p0-chat27b-gen/check_saturation.py
    # parses a log plus a JSON artifact and builds no engine, but its docstring
    # and a `notes` string both say the harness drives vLLM via LLM().generate().
    # A `#`-only strip cannot see inside a string, so the gate charged it.
    b = _bundle(tmp_path, "b", {
        "check_saturation.py":
            '"""gen_probe_resumable drives vLLM IN-PROCESS via `LLM().generate()`,\n'
            'which never emits server engine lines."""\n'
            'import json\n'
            'def main(log, artifact):\n'
            '    o = {"notes": ["expected for the in-process LLM().generate() harness"]}\n'
            '    return json.dumps(o)\n'})
    assert jm.vllm_sampler_findings(b, {}) == []


def test_sampler_lint_still_flags_a_real_engine_in_a_file_that_also_has_docstrings(tmp_path):
    # The other direction, and the one that must never regress: stripping
    # strings must not blind the check to an actual construction next to them.
    b = _bundle(tmp_path, "b", {
        "gen.py":
            '"""a harness that talks about LLM().generate() at length."""\n'
            'from vllm import LLM\n'
            'llm = LLM(model=base)\n'})
    out = jm.vllm_sampler_findings(b, {})
    assert len(out) == 1 and out[0].startswith("gen.py starts a vLLM engine")


def test_sampler_lint_falls_back_when_a_py_file_will_not_tokenize(tmp_path):
    # Fail OPEN toward reporting: an unparseable .py must still be scanned by
    # the old line strip. Under-reporting this check costs a rented box.
    b = _bundle(tmp_path, "b", {"broken.py": "def f(:\nllm = LLM(model=b)\n"})
    assert len(jm.vllm_sampler_findings(b, {})) == 1


def test_sampler_lint_skips_staged_data_and_result_dirs(tmp_path):
    b = _bundle(tmp_path, "b", {
        "data/x.py": "llm = LLM(model=b)\n",
        "data-rb3/y.py": "llm = LLM(model=b)\n",
        "results/z.py": "llm = LLM(model=b)\n",
        "__pycache__/w.py": "llm = LLM(model=b)\n"})
    assert jm.vllm_sampler_findings(b, {}) == []


def test_validate_job_config_surfaces_the_sampler_finding(tmp_path):
    # The wiring that matters: `herdd job submit` prints these warnings before
    # anything is uploaded or any box is rented.
    b = _bundle(tmp_path, "b", {"run.sh": "vllm serve $MODEL\n"})
    _cfg, warns = jm.validate_job_config(
        {"name": "smoke", "entrypoint": "run.sh"}, b)
    assert any("VLLM_USE_FLASHINFER_SAMPLER" in w for w in warns)


# --------------------------------------------------------------------------- #
# jobs_watch_advice — the `fleet watch --profile jobs` ordering guard
#
# THE INCIDENT (box 46648873, 2026-08-03): a box was RESUMED to run more work,
# a jobs watch was armed before the new job was submitted, and the box was
# stopped 4 seconds later. Not a budget trip ($0.0001 of a $1.00 cap) — the
# queue still held the PREVIOUS session's DONE tickets, and the jobs ladder
# reads an all-terminal queue as "the work is finished, park it".
# --------------------------------------------------------------------------- #
def test_jobs_watch_advice_silent_when_a_ticket_is_pending():
    # Real work queued: the drain exit cannot fire, so say nothing. A guard
    # that fires on the NORMAL case gets ignored on the dangerous one.
    assert jm.jobs_watch_advice(["j1", "j2"],
                                [{"status": "done"}, {"status": "running"}]) is None


def test_jobs_watch_advice_warns_loudly_on_a_stale_all_terminal_queue():
    msg = jm.jobs_watch_advice(
        ["j-old-1", "j-old-2"], [{"status": "done"}, {"status": "failed"}])
    assert msg and "PARK" in msg
    assert "SUBMIT THE JOB FIRST" in msg and "--keep" in msg
    assert "j-old-1" in msg


def test_jobs_watch_advice_notes_but_does_not_alarm_on_an_empty_queue():
    # Arming minutes before a wave submits is the NORMAL launch order and
    # fleetd keeps the watch (queue_empty is transient) — note, do not shout.
    msg = jm.jobs_watch_advice([], [])
    assert msg and "PARK" not in msg and "no queued tickets" in msg


def test_jobs_watch_advice_counts_every_terminal_status():
    for st in sorted(jm.TERMINAL):
        assert jm.jobs_watch_advice(["j"], [{"status": st}]) is not None


def test_fold_instance_id_does_not_survive_a_retarget():
    """A retarget moves the ticket to a DIFFERENT box, so every box event before it
    describes a machine this job has left. Measured 2026-08-06 on
    20260806T071847-fit-ladder-ea7c: `retargeted` -> `claimed` was a 5 s gap, and in
    that window `target_box` had flipped to the new box while `instance_id` still
    named the old one. The old box is destroyed by then, so it is also absent from
    `live` -- and the job renders `interrupted` AGAINST A BOX IT IS NO LONGER ON.

    That disagreement is also why "wait for job status to report the new box id" is
    not by itself a usable resume gate: target_box and instance_id answer different
    questions and flip at different times."""
    evs = [ev("submitted", T(1), actor="cli:h", box="4695"),
           ev("claimed", T(2), instance_id="4695"),
           ev("started", T(3), instance_id="4695"),
           ev("retargeted", T(4), actor="cli:h", box="4697", from_box="4695")]
    # the old box is GONE -- destroyed before the retarget, which is the real shape
    v = jm.fold_events(evs, live_iids=())
    assert v["target_box"] == "4697"
    assert v["instance_id"] is None, "must not name the box the job has left"

    # once the NEW box claims, instance_id is that box and liveness is real again
    evs.append(ev("claimed", T(5), instance_id="4697"))
    v2 = jm.fold_events(evs, live_iids={"4697"})
    assert v2["instance_id"] == "4697"
    assert v2["display_status"] == "running"


def test_fold_host_metrics_are_attributed_to_the_reporting_box():
    """`last_metrics` fold independently of `instance_id`, so without an attribution
    check a DEAD box's GPU numbers render under the live box's status line -- a
    healthy gpu_util:100 for a machine that has already died. Same family as defect
    (7) (a resume line read off the prior box's log tail) and NOT closed by its
    mitigation, because gating on the box id does not touch this surface."""
    evs = [ev("claimed", T(1), instance_id="4695"),
           ev("heartbeat", T(2), instance_id="4695",
              host_metrics="gpu_util:100,gpu:OLD"),
           ev("retargeted", T(3), actor="cli:h", box="4697", from_box="4695"),
           ev("claimed", T(4), instance_id="4697")]
    v = jm.fold_events(evs, live_iids={"4697"})
    assert v["instance_id"] == "4697"
    # the only metrics on record belong to the OLD box -> report none, not those
    assert v.get("last_metrics") is None, "must not show a foreign box's metrics"

    # once the new box reports, we show ITS metrics, with provenance
    evs.append(ev("heartbeat", T(5), instance_id="4697",
                  host_metrics="gpu_util:42,gpu:NEW"))
    v2 = jm.fold_events(evs, live_iids={"4697"})
    assert v2["last_metrics"] == "gpu_util:42,gpu:NEW"
    assert v2["last_metrics_box"] == "4697"


def test_fold_host_metrics_unchanged_on_the_single_box_path():
    """Inertness: with no retarget, attribution must not drop anything."""
    evs = [ev("claimed", T(1), instance_id="44"),
           ev("heartbeat", T(2), instance_id="44", host_metrics="gpu_util:90"),
           ev("heartbeat", T(3), instance_id="44")]      # newer, but no metrics
    v = jm.fold_events(evs, live_iids={"44"})
    assert v["last_metrics"] == "gpu_util:90"
    assert v["last_metrics_box"] == "44"


# --------------------------------------------------------------------------- #
# the v9 requeue chain — REAL events, not a model of the writer
# --------------------------------------------------------------------------- #
# testfixtures/jobmeta/v9-gemma4-requeue-chain.jsonl is the verbatim B2 event
# log of 20260806T212132-v9-gemma4-dec-train-8818 (jobs/<id>/events/), reduced
# to every lifecycle event plus the first/last heartbeat and newest checkpoint
# per box, with heartbeat `tail` truncated to its last 100 chars. Nothing else
# is edited: the timestamps, nonces, actors, box ids and field sets are as
# emitted. It is here because a fixture invented by reading requeue_ticket would
# have had the shape the reader BELIEVED — and the belief (that a ticket only
# ever moves via `retargeted`) is exactly the bug.
V9_JOB_ID = "20260806T212132-v9-gemma4-dec-train-8818"


def _v9_events():
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "testfixtures", "jobmeta", "v9-gemma4-requeue-chain.jsonl")
    with open(p) as fh:
        return [json.loads(l) for l in fh if l.strip()]


def test_v9_fixture_is_the_real_log_shape():
    """Guard the fixture itself: if a future edit flattens it into something
    tidier than the night produced, the tests below stop meaning anything."""
    evs = _v9_events()
    assert all(e["job_id"] == V9_JOB_ID for e in evs)
    kinds = [(e["event"], e.get("kind")) for e in evs]
    # three operator retargets + two operator REQUEUES, and the requeues are
    # `resumed` events carrying `box` — the frozen-vocabulary move.
    assert sum(1 for e in evs if e["event"] == "retargeted") == 4
    requeues = [e for e in evs
                if e["event"] == "resumed" and e.get("kind") == "requeue"]
    assert [e["box"] for e in requeues] == ["47042386", "47045282"]
    assert all(e.get("box") for e in requeues)
    # jobd's own `resumed` (the continuations) carry NO box — that asymmetry is
    # what the fold keys on.
    assert all(not e.get("box") for e in evs
               if e["event"] == "resumed" and e.get("kind") != "requeue")
    assert ("failed", None) in kinds


def test_v9_target_box_follows_a_requeue_not_just_a_retarget():
    """THE DEFECT. Two operator requeues walked the ticket
    47041615 -> 47042386 -> 47045282, but `target_box` folded only `retargeted`,
    so `herdd job status --json` reported target_box=47041615 — destroyed hours
    earlier — for the whole of a live 156-step training run, while the job was
    heartbeating from 47045282. A ticket move is a ticket move whichever event
    carries it."""
    v = jm.fold_events(_v9_events(), live_iids={"47045282"})
    assert v["target_box"] == "47045282", \
        "target_box must name the box whose QUEUE holds the ticket"
    assert v["instance_id"] == "47045282"
    assert v["retargeted_from"] == "47042386"
    assert v["display_status"] == "running" and v["live"] is True
    assert v["reopened"] is True


def test_v9_requeue_chain_stays_reopened_not_failed():
    """Task #43's fold (stale `failed` outranking a newer outcome) must not
    reopen: the log holds TWO `failed` events, both older than the last requeue,
    and the job is running."""
    v = jm.fold_events(_v9_events(), live_iids={"47045282"})
    assert v["status"] == "started"
    assert v["rc"] is None and v["fail_reason"] is None
    # the dead attempts survive as diagnosis, never as the outcome
    assert v["prior_fail_reason"] == "asset_stage_slow:base"


def test_v9_fold_at_the_failure_is_still_terminal_failed():
    """Inertness at the other end of the window: truncated at 03:51:33 the fold
    is genuinely `failed` on 47041615. This is the state the ls disk cache
    froze — correct when written, wrong forever after."""
    evs = [e for e in _v9_events() if e["ts"] <= "20260807T035133081Z"]
    v = jm.fold_events(evs, live_iids={"47041615"})
    assert v["status"] == "failed" and v["target_box"] == "47041615"
    assert v["last_heartbeat_ts"] == "20260807T034002901Z"


def test_fold_target_box_ignores_a_plain_box_side_resumed():
    """Inertness: jobd's own `resumed` (crash/preempt/retarget continuation)
    carries no `box` and must not move the ticket pointer."""
    evs = [ev("submitted", T(1), actor="cli:h", box="44"),
           ev("claimed", T(2), instance_id="44"),
           ev("resumed", T(3), instance_id="44", kind="crash"),
           ev("started", T(4), instance_id="44")]
    v = jm.fold_events(evs, live_iids={"44"})
    assert v["target_box"] == "44" and "retargeted_from" not in v


def test_fold_requeue_from_box_placeholder_is_not_a_box_id():
    """requeue_ticket writes from_box='-' when the predecessor is unknown; that
    placeholder must not be laundered into `retargeted_from`."""
    evs = [ev("submitted", T(1), actor="cli:h", box="44"),
           ev("failed", T(2), instance_id="44", rc=1),
           ev("resumed", T(3), actor="cli:h", kind="requeue", box="55",
              instance_id="55", from_box="-")]
    v = jm.fold_events(evs, live_iids={"55"})
    assert v["target_box"] == "55"
    assert "retargeted_from" not in v


# --------------------------------------------------------------------------- #
# software-epoch stamp (`DS_TRAINER_REV`)
# --------------------------------------------------------------------------- #
def _git_repo(tmp_path, dirty=False):
    r = tmp_path / "repo"
    r.mkdir(parents=True)
    env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@e",
               GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@e")
    (r / "f.txt").write_text("one\n")
    for cmd in (["init", "-q"], ["add", "f.txt"], ["commit", "-qm", "c"]):
        subprocess.run(["git", "-C", str(r)] + cmd, check=True, env=env,
                       capture_output=True)
    if dirty:
        (r / "f.txt").write_text("two\n")
    return str(r)


def test_repo_head_rev_reads_head_and_marks_a_dirty_tree(tmp_path):
    clean = _git_repo(tmp_path / "a")
    rev = jm.repo_head_rev(clean)
    assert rev and len(rev) == 40 and not rev.endswith("-dirty")
    dirty = _git_repo(tmp_path / "b", dirty=True)
    assert jm.repo_head_rev(dirty).endswith("-dirty")


def test_repo_head_rev_outside_a_repo_is_none_not_empty(tmp_path):
    """A submit from a tarball checkout must degrade to UNMEASURED, and the
    trainer must see nothing rather than a blank epoch key."""
    d = tmp_path / "nogit"
    d.mkdir()
    assert jm.repo_head_rev(str(d)) is None


def test_stamp_trainer_rev_writes_into_the_ticket_env(tmp_path):
    repo = _git_repo(tmp_path / "r")
    cfg = {"env": {"FOO": "bar"}}
    rev = jm.stamp_trainer_rev(cfg, repo_root=repo)
    assert cfg["env"][jm.TRAINER_REV_ENV] == rev
    assert cfg["env"]["FOO"] == "bar"          # neighbours untouched


def test_stamp_trainer_rev_never_overwrites_an_explicit_pin(tmp_path):
    repo = _git_repo(tmp_path / "r")
    cfg = {"env": {jm.TRAINER_REV_ENV: "pinned-epoch"}}
    assert jm.stamp_trainer_rev(cfg, repo_root=repo) is None
    assert cfg["env"][jm.TRAINER_REV_ENV] == "pinned-epoch"


def test_stamp_trainer_rev_is_inert_off_a_repo_and_on_junk(tmp_path):
    d = tmp_path / "nogit"
    d.mkdir()
    cfg = {"env": {}}
    assert jm.stamp_trainer_rev(cfg, repo_root=str(d)) is None
    assert cfg["env"] == {}                    # absent, not ""
    assert jm.stamp_trainer_rev(None) is None
    assert jm.stamp_trainer_rev({}) is None


def test_validate_job_config_does_not_stamp(tmp_path):
    """The stamp is SUBMIT-side on purpose: config validation is pure, its
    output is pinned by tests, and a HEAD-dependent config would make
    submit_with_id's idempotent resubmit conflict after any commit."""
    d = _mkjob(tmp_path)
    cfg, _ = jm.validate_job_config(jm.load_job_config(str(d)), str(d))
    assert jm.TRAINER_REV_ENV not in cfg["env"]
    tk = jm.make_ticket("20260828T000000Z-probe-01-ab12", "s" * 64, "cli:t",
                        cfg, "44")
    assert jm.TRAINER_REV_ENV not in tk["config"]["env"]

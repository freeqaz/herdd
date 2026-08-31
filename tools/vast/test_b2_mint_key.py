"""Unit tests for b2_mint_key.py — transport stubbed, no network, no creds.

Also carries the C5 serve-script checks (launch_serve.sh / serve_vllm.sh):
bash-lint plus sandboxed runs of the cred-sensitive paths (mint shape, [b2]
re-key) with every external binary PATH-shimmed — no network, no vast."""
import json
import os
import shutil
import subprocess
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b2_mint_key as bmk


class FakeB2:
    """Stub for bmk._http: canned auth + in-memory key store."""

    def __init__(self, keys=None):
        self.keys = list(keys or [])
        self.created = []
        self.deleted = []

    def __call__(self, url, body=None, headers=None):
        call = url.rsplit("/", 1)[-1]
        if call == "b2_authorize_account":
            return {"accountId": "acct1", "authorizationToken": "tok",
                    "apiInfo": {"storageApi": {
                        "apiUrl": "https://api.fake",
                        "allowed": {"capabilities": [
                            "listKeys", "writeKeys",
                            "deleteKeys", "listBuckets"]}}}}
        if call == "b2_list_buckets":
            return {"buckets": [{"bucketName": body["bucketName"],
                                 "bucketId": "bkt1"}]}
        if call == "b2_list_keys":
            return {"keys": self.keys, "nextApplicationKeyId": None}
        if call == "b2_delete_key":
            self.deleted.append(body["applicationKeyId"])
            self.keys = [k for k in self.keys
                         if k["applicationKeyId"] != body["applicationKeyId"]]
            return {}
        if call == "b2_create_key":
            self.created.append(body)
            return {"applicationKeyId": "newkid", "applicationKey": "newsecret"}
        raise AssertionError(f"unexpected call {call}")


@pytest.fixture
def fake(monkeypatch):
    f = FakeB2()
    monkeypatch.setattr(bmk, "_http", f)
    monkeypatch.setenv("B2_MINTER_KEY_ID", "mk")
    monkeypatch.setenv("B2_MINTER_APPLICATION_KEY", "ms")
    monkeypatch.setenv("B2_BUCKET", "example-runs-bucket")
    return f


def test_sanitize_name():
    assert bmk.sanitize_name("run-abc_1.2/x") == "run-abc-1-2-x"
    assert bmk.sanitize_name("--a---b--") == "a-b"
    assert len(bmk.sanitize_name("x" * 300)) == 100
    with pytest.raises(bmk.MintError):
        bmk.sanitize_name("__..__")


def test_mint_basic(fake):
    kid, key = bmk.mint("run-r1", hours=6)
    assert (kid, key) == ("newkid", "newsecret")
    body = fake.created[0]
    assert body["keyName"] == "run-r1"
    assert body["bucketIds"] == ["bkt1"]
    assert body["validDurationInSeconds"] == 6 * 3600
    assert body["capabilities"] == ["listFiles", "readFiles", "writeFiles"]
    assert "namePrefix" not in body


def test_mint_refuses_deletefiles(fake):
    with pytest.raises(bmk.MintError, match="deleteFiles"):
        bmk.mint("run-r1", caps="listFiles,readFiles,writeFiles,deleteFiles")
    assert not fake.created


def test_mint_clamps_ttl(fake):
    bmk.mint("run-tiny", hours=0.01)
    assert fake.created[-1]["validDurationInSeconds"] == 3600
    bmk.mint("run-huge", hours=10 ** 9)
    assert fake.created[-1]["validDurationInSeconds"] == bmk.MAX_HOURS * 3600


def test_mint_revokes_name_collision(fake):
    fake.keys = [
        {"applicationKeyId": "old1", "keyName": "run-r1", "capabilities": []},
        {"applicationKeyId": "oth", "keyName": "run-other", "capabilities": []},
    ]
    bmk.mint("run-r1")
    assert fake.deleted == ["old1"]          # collision revoked, other kept
    assert fake.created[0]["keyName"] == "run-r1"


def test_mint_sanitizes_run_ids(fake):
    bmk.mint("run-lora_e2s8k.v2")
    assert fake.created[0]["keyName"] == "run-lora-e2s8k-v2"


def test_mint_prefix_passthrough(fake):
    bmk.mint("box-1", prefix="jobs/")
    assert fake.created[0]["namePrefix"] == "jobs/"


def test_mint_pair_read_bucketwide_write_scoped(fake):
    read, write = bmk.mint_pair("box-42", hours=3, write_prefix="jobs/")
    assert read == ("newkid", "newsecret") and write == ("newkid", "newsecret")
    ro, rw = fake.created
    # read key: bucket-wide (no namePrefix), no writeFiles
    assert ro["keyName"] == "box-42-ro"
    assert "namePrefix" not in ro
    assert ro["capabilities"] == ["listFiles", "readFiles"]
    # write key: scoped to the prefix, carries writeFiles
    assert rw["keyName"] == "box-42-rw"
    assert rw["namePrefix"] == "jobs/"
    assert "writeFiles" in rw["capabilities"]
    assert rw["validDurationInSeconds"] == 3 * 3600


# --- the PUBLISH grant (checkpoints/) — B2_PUBLISH_KEY_SCOPE_FIX_2026-08-05 --- #
def test_mint_publish_is_a_third_scoped_key(fake):
    assert bmk.mint_publish("box-42", hours=3) == ("newkid", "newsecret")
    body = fake.created[-1]
    assert body["keyName"] == "box-42-pub"          # '<base>-' => revoked on destroy
    assert body["namePrefix"] == "checkpoints/"     # ONE prefix, not a widening
    assert body["capabilities"] == ["listFiles", "readFiles", "writeFiles"]
    assert "deleteFiles" not in body["capabilities"]
    assert body["validDurationInSeconds"] == 3 * 3600   # self-expiring, same TTL


def test_mint_publish_cannot_widen_the_jobs_key(fake):
    """The fix is a SECOND key, never a broader one: the jobs/ write key must
    come back namePrefix-scoped exactly as before."""
    bmk.mint_pair("box-42", write_prefix="jobs/")
    bmk.mint_publish("box-42")
    names = {b["keyName"]: b for b in fake.created}
    assert names["box-42-rw"]["namePrefix"] == "jobs/"
    assert names["box-42-pub"]["namePrefix"] == "checkpoints/"
    assert "namePrefix" not in names["box-42-ro"]


def test_publish_prefix_env_override_and_disable():
    assert bmk.publish_prefix({}) == "checkpoints/"
    assert bmk.publish_prefix({"B2_PUBLISH_PREFIX": "adapters"}) == "adapters/"
    for off in ("", "  ", "0", "none", "OFF", "false"):
        assert bmk.publish_prefix({"B2_PUBLISH_PREFIX": off}) is None
    with pytest.raises(bmk.MintError):
        bmk.publish_prefix({"B2_PUBLISH_PREFIX": "/abs/"})


def test_mint_publish_disabled_mints_nothing(fake, monkeypatch):
    monkeypatch.setenv("B2_PUBLISH_PREFIX", "")
    assert bmk.mint_publish("box-42") is None
    assert not fake.created


def test_mint_pair_requires_write_prefix(fake):
    with pytest.raises(bmk.MintError, match="write_prefix"):
        bmk.mint_pair("box-42")
    assert not fake.created


def test_mint_pair_never_grants_delete(fake):
    bmk.mint_pair("box-42", write_prefix="jobs/")
    for body in fake.created:
        assert "deleteFiles" not in body["capabilities"]


def test_gc_only_expired_ephemerals(fake, capsys):
    now_ms = time.time() * 1000
    fake.keys = [
        {"applicationKeyId": "e1", "keyName": "run-dead",
         "expirationTimestamp": now_ms - 1000, "capabilities": []},
        {"applicationKeyId": "e2", "keyName": "run-live",
         "expirationTimestamp": now_ms + 10 ** 7, "capabilities": []},
        {"applicationKeyId": "s1", "keyName": "box-rw",
         "expirationTimestamp": None, "capabilities": []},
    ]
    bmk.main(["gc"])
    assert fake.deleted == ["e1"]


def test_mint_pair_cli_export_format(fake, capsys):
    """Default form: the four §2.1 wire-name export lines (eval-able)."""
    bmk.main(["mint-pair", "--name", "serve-s1", "--write-prefix", "serve/",
              "--hours", "3"])
    out = capsys.readouterr().out
    assert "export B2_KEY_ID=newkid\n" in out
    assert "export B2_APPLICATION_KEY=newsecret\n" in out
    assert "export B2_WRITE_KEY_ID=newkid\n" in out
    assert "export B2_WRITE_APPLICATION_KEY=newsecret\n" in out
    ro, rw = fake.created
    assert ro["keyName"] == "serve-s1-ro" and "namePrefix" not in ro
    assert rw["keyName"] == "serve-s1-rw" and rw["namePrefix"] == "serve/"
    assert rw["validDurationInSeconds"] == 3 * 3600


def test_mint_pair_cli_json(fake, capsys):
    bmk.main(["mint-pair", "--name", "serve-s1", "--write-prefix", "serve/",
              "--json"])
    out = json.loads(capsys.readouterr().out)
    assert out["read"] == {"keyId": "newkid", "applicationKey": "newsecret"}
    assert out["write"] == {"keyId": "newkid", "applicationKey": "newsecret"}
    assert out["name"] == "serve-s1" and out["writePrefix"] == "serve/"


def test_mint_pair_cli_requires_args(fake, capsys):
    with pytest.raises(SystemExit):          # argparse: --write-prefix required
        bmk.main(["mint-pair", "--name", "serve-s1"])
    with pytest.raises(SystemExit):          # argparse: --name required
        bmk.main(["mint-pair", "--write-prefix", "serve/"])
    assert not fake.created


def test_mint_cli_export_format(fake, capsys):
    bmk.main(["mint", "--run", "r1", "--hours", "2",
              "--var-prefix", "SHIP_B2_"])
    out = capsys.readouterr().out
    assert "export SHIP_B2_KEY_ID=newkid" in out
    assert "export SHIP_B2_APPLICATION_KEY=newsecret" in out


# ------------------------------------------------------ C5 serve scripts ---
HERE = os.path.dirname(os.path.abspath(__file__))
SERVE_VLLM = os.path.join(HERE, "onstart", "serve_vllm.sh")
LAUNCH_SERVE = os.path.join(HERE, "launch_serve.sh")

# The only MACHINE-shaped variables the shelled scripts legitimately need:
# binaries to exec (PATH) and a locale. Everything else is supplied per call.
_ENV_PASSTHROUGH = ("PATH", "LANG", "LC_ALL", "LC_CTYPE", "TERM")


def _hermetic_env(**overrides):
    """A subprocess env built from an ALLOWLIST — never `dict(os.environ, ...)`.

    Every assertion in this section reads what launch_serve.sh / serve_vllm.sh
    MINT and ship, so the child must not inherit the developer's environment:
    the repo `.env` puts real `B2_*` / `HF_TOKEN` / `TS_AUTHKEY` values into
    `os.environ`, and with a wholesale copy a *new* variable in `.env` could
    change what a test observes with no code change (it also kept a real minter
    key on the env of a real script). Callers pass exactly what the scenario
    needs; anything they do not pass is absent by construction, which is what
    the "not shipped" assertions below actually mean.
    """
    env = {k: os.environ[k] for k in _ENV_PASSTHROUGH if k in os.environ}
    env.setdefault("PATH", os.defpath)
    env.update(overrides)
    return env


def test_serve_scripts_bash_lint():
    for p in (SERVE_VLLM, LAUNCH_SERVE):
        subprocess.run(["bash", "-n", p], check=True, env=_hermetic_env())


def _shim(bindir, name, body="#!/bin/sh\nexit 0\n"):
    os.makedirs(bindir, exist_ok=True)
    p = os.path.join(bindir, name)
    with open(p, "w") as f:
        f.write(body)
    os.chmod(p, 0o755)


def _ensure_b2(home, **env):
    """Run serve_vllm.sh's ensure_b2 in isolation (rclone PATH-shimmed)."""
    bindir = os.path.join(home, "bin")
    _shim(bindir, "rclone")
    # B2_WRITE_* are absent unless a caller sets them (the single-key box
    # shape); the hermetic base guarantees that with nothing to pop.
    e = _hermetic_env(HOME=home,
                      PATH=bindir + os.pathsep + os.environ.get("PATH", ""),
                      B2_BUCKET="bkt", B2_KEY_ID="K", B2_APPLICATION_KEY="S",
                      B2_S3_ENDPOINT="https://s3.fake")
    e.update(env)
    subprocess.run(
        ["bash", "-c",
         f'source <(sed -n "/^ensure_b2()/,/^}}/p" "{SERVE_VLLM}"); ensure_b2'],
        check=True, env=e)
    with open(os.path.join(home, ".config/rclone/rclone.conf")) as f:
        return f.read()


def test_ensure_b2_rekeys_on_rerun(tmp_path):
    """A --on-box re-run revoke-then-mints the serve keys, so ensure_b2 must
    REWRITE [b2]/[b2w] from env every time — keep-if-present strands the box
    on revoked creds (the documented rotation claim)."""
    home = str(tmp_path)
    conf = _ensure_b2(home)
    assert "access_key_id = K\n" in conf and "[b2w]" not in conf
    # an unrelated remote must survive the rewrite
    with open(os.path.join(home, ".config/rclone/rclone.conf"), "a") as f:
        f.write("\n[other]\ntype = local\n")
    conf = _ensure_b2(home, B2_KEY_ID="K2", B2_APPLICATION_KEY="S2",
                      B2_WRITE_KEY_ID="WK", B2_WRITE_APPLICATION_KEY="WS")
    assert "access_key_id = K2\n" in conf
    assert "access_key_id = K\n" not in conf          # revoked key gone
    assert "[b2w]" in conf and "access_key_id = WK\n" in conf
    assert "[other]" in conf
    # rotating back to a single-key serve drops the stale scoped remote
    conf = _ensure_b2(home, B2_KEY_ID="K3", B2_APPLICATION_KEY="S3")
    assert "access_key_id = K3\n" in conf and "[b2w]" not in conf
    assert "[other]" in conf


def _serve_sandbox(tmp_path):
    """launch_serve.sh copy with stubbed b2_mint_key/herdd/runmeta/b2_sync
    siblings; every stub records its argv as JSONL under <calls>/."""
    tools = tmp_path / "repo" / "x" / "tools"          # REPO_ROOT = repo/
    (tools / "onstart").mkdir(parents=True)
    shutil.copy(LAUNCH_SERVE, tools / "launch_serve.sh")
    calls = tmp_path / "calls"
    calls.mkdir()
    rec = ("#!/usr/bin/env python3\n"
           "import json, sys\n"
           f"open({str(calls)!r} + '/' + sys.argv[0].rsplit('/', 1)[-1]"
           " + '.jsonl', 'a').write(json.dumps(sys.argv[1:]) + '\\n')\n")
    (tools / "b2_mint_key.py").write_text(
        rec + "print('export B2_KEY_ID=RKID')\n"
              "print('export B2_APPLICATION_KEY=RSEC')\n"
              "if sys.argv[1] == 'mint-pair':\n"
              "    print('export B2_WRITE_KEY_ID=WKID')\n"
              "    print('export B2_WRITE_APPLICATION_KEY=WSEC')\n")
    (tools / "herdd.py").write_text(rec + "print('launched instance 999')\n")
    (tools / "runmeta.py").write_text(rec)
    (tools / "b2_sync.sh").write_text("#!/bin/sh\nexit 0\n")
    # The auto-disk-size step imports these from $HERE (sys.path.insert); the
    # sandbox must supply them itself — under the hermetic env there is no
    # PYTHONPATH leak of the real repo modules to paper over their absence.
    (tools / "disksize.py").write_text(
        "def serve_disk_gb(model_bytes, extra_gb=0.0):\n"
        "    return 60, {'complete': False}\n")
    (tools / "vastconf.py").write_text("DISK_DEFAULT_SERVE_GB = 60\n")
    (tools / "onstart" / "serve_vllm.sh").write_text(
        "#!/usr/bin/env bash\necho serve\n")
    bindir = tmp_path / "bin"
    _shim(str(bindir), "rclone")
    return tools, calls, bindir


def _launch(tools, calls, bindir, tmp_path, *extra):
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    # B2_BOX_*, B2_EPHEMERAL_HOURS, B2_WRITE_*, HF_TOKEN and every other
    # `.env` variable are absent by construction — the assertions below are
    # about what launch_serve.sh mints and puts on the wire, so an inherited
    # value would forge them (no popping needed under the allowlist).
    env = _hermetic_env(
        HOME=str(home),
        PATH=f"{bindir}{os.pathsep}{os.environ.get('PATH', '')}",
        TMPDIR=str(tmp_path),
        B2_MINTER_KEY_ID="mk", B2_MINTER_APPLICATION_KEY="ms",
        B2_BUCKET="bkt", B2_KEY_ID="opsK", B2_APPLICATION_KEY="opsS",
        B2_S3_ENDPOINT="https://s3.fake")
    r = subprocess.run(["bash", str(tools / "launch_serve.sh"),
                        "--model", "org/model", *extra],
                       env=env, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    mint = json.loads(
        (calls / "b2_mint_key.py.jsonl").read_text().splitlines()[-1])
    # launch_serve.sh calls herdd more than once (launch, then the fleetd
    # `fleet watch` registration) — pick the launch call, not the last one.
    argv = next(a for line in reversed(
        (calls / "herdd.py.jsonl").read_text().splitlines())
        if (a := json.loads(line))[0] == "launch")
    envs = dict(a.split("=", 1) for i, a in enumerate(argv)
                if i and argv[i - 1] == "--env")
    return mint, envs


def test_launch_serve_farm_on_ships_bucketwide_single_key(tmp_path):
    """--cpu-farm only: the eval sidecar writes evals/ via [b2], so the minted
    key must be the single bucket-wide read+write shape — a serve/-scoped pair
    silently strands all farm output (finding 2). Widening the key is part of
    the price of opting in, never a default."""
    tools, calls, bindir = _serve_sandbox(tmp_path)
    mint, envs = _launch(tools, calls, bindir, tmp_path, "--cpu-farm")
    assert mint[0] == "mint" and "--write-prefix" not in mint
    assert envs["B2_KEY_ID"] == "RKID"
    assert envs["B2_APPLICATION_KEY"] == "RSEC"
    assert "B2_WRITE_KEY_ID" not in envs
    assert envs["CRED_ROLE"] == "serve" and "B2_KEY_EXPIRES_AT" in envs
    assert envs["CPU_FARM"] == "1"


def test_launch_serve_farm_off_is_the_default_and_ships_scoped_pair(tmp_path):
    """Owner ruling 2026-08-21: the farm is dead and off by default, so the
    DEFAULT mint shape is the tighter serve/-scoped pair."""
    tools, calls, bindir = _serve_sandbox(tmp_path)
    mint, envs = _launch(tools, calls, bindir, tmp_path)
    assert mint[0] == "mint-pair"
    assert mint[mint.index("--write-prefix") + 1] == "serve/"
    assert envs["B2_KEY_ID"] == "RKID"                 # bucket-wide RO half
    assert envs["B2_WRITE_KEY_ID"] == "WKID"
    assert envs["B2_WRITE_APPLICATION_KEY"] == "WSEC"
    assert envs["CPU_FARM"] == "0"
    assert envs["CRED_ROLE"] == "serve" and "B2_KEY_EXPIRES_AT" in envs


def test_launch_serve_no_cpu_farm_still_accepted(tmp_path):
    """Back-compat: --no-cpu-farm is now redundant but must not error."""
    tools, calls, bindir = _serve_sandbox(tmp_path)
    mint, envs = _launch(tools, calls, bindir, tmp_path, "--no-cpu-farm")
    assert mint[0] == "mint-pair"
    assert envs["CPU_FARM"] == "0"

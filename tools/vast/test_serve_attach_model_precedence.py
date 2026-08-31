"""The two serve-attach silent no-ops found live on 2026-08-21.

Both were of the class where every surface a caller checks stays green: a
process running, `/v1/models` naming the model you asked for, `serve_ready.sh
--expect-models` PASS — and the `vllm serve` argv naming something else.

1. A `MODEL_B2` persisted into `/etc/environment` by an earlier launch outlived
   that run, was inherited by a later `--on-box` attach through pam_env, and
   overrode the `--model` that attach passed. The gates here are behavioural:
   run the real `serve_vllm.sh` under `DRY_RUN=1` against a seeded env file and
   read the argv it would exec.
2. `--on-box --restart` ran `pkill -f 'vllm serve'` over ssh, and the remote
   shell's own argv contains that string. That gate is a text pin — the kill
   only happens against a rented box — plus a live control proving the bracket
   form is what makes the difference.

Skipped when bash is unavailable (the portable lane runs it).
"""
import os
import re
import shutil
import subprocess

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
SERVE_SH = os.path.join(_HERE, "onstart", "serve_vllm.sh")
LAUNCH_SH = os.path.join(_HERE, "launch_serve.sh")

pytestmark = pytest.mark.skipif(not shutil.which("bash"), reason="needs bash")


def _launch_serve(tmp_path, *args, **env):
    """Run `launch_serve.sh` with BOTH ambient-credential channels closed.

    Filtering `os.environ` is not enough: the script sources `$REPO_ROOT/.env`
    itself, so on a checkout that has one it starts with real B2 creds and takes
    a different branch than on one that does not. `_LAUNCH_SERVE_ENV` points
    that source at an empty file so the verdict is the caller's env alone.
    """
    e = {k: v for k, v in os.environ.items()
         if k in ("PATH", "LANG", "TMPDIR")}
    e["HOME"] = _scratch_home(tmp_path)
    empty = tmp_path / "empty.env"
    empty.write_text("")
    e["_LAUNCH_SERVE_ENV"] = str(empty)
    # HOME survives the filter, so the disk-autosize step's `b2_sync.sh config`
    # used to rewrite the LIVE [b2] remote with this file's fake endpoint.
    e["RCLONE_CONFIG"] = str(tmp_path / "rclone.conf")
    e.update(env)
    return subprocess.run(["bash", LAUNCH_SH, *args], capture_output=True,
                          text=True, timeout=120, env=e, cwd=_HERE)


_FAKE_B2 = {"B2_BUCKET": "fake", "B2_KEY_ID": "fake",
            "B2_APPLICATION_KEY": "fake",
            "B2_S3_ENDPOINT": "https://example.invalid"}


def _scratch_home(tmp_path):
    """A throwaway $HOME for the shipped payloads run below.

    `serve_vllm.sh` writes `$HOME/.config/rclone/rclone.conf` from `B2_*`, and it
    hardcodes `$HOME` rather than honouring `RCLONE_CONFIG`. Forwarding the real
    HOME therefore overwrites the operator's live `[b2]` remote with `_FAKE_B2`'s
    `example.invalid` — observed twice on 2026-08-22, breaking `launch --jobs`
    workstation-wide while `rclone listremotes` still looked healthy.
    """
    h = tmp_path / "home"
    (h / ".config" / "rclone").mkdir(parents=True, exist_ok=True)
    return str(h)


def _serve_dry_run(env_file, tmp_path, **env):
    """Run the shipped serve payload with its /etc/environment redirected.

    Returns (argv list of the `vllm serve` line, combined output).
    """
    e = {k: v for k, v in os.environ.items()
         if k in ("PATH", "LANG", "TMPDIR")}
    e["HOME"] = _scratch_home(tmp_path)
    e.update({"DRY_RUN": "1", "MAX_HOURS": "0", "SERVE_DP": "1",
              "SERVE_ENV_FILE": env_file})
    e.update(env)
    p = subprocess.run(["bash", SERVE_SH], capture_output=True, text=True,
                       timeout=120, env=e, cwd=_HERE)
    assert p.returncode == 0, p.stderr
    line = [ln for ln in p.stdout.splitlines() if ln.startswith("vllm serve")]
    assert len(line) == 1, p.stdout
    return line[0].split(), p.stdout + p.stderr


# --------------------------------------------------------------------------- #
# 1. the persisted MODEL_B2
# --------------------------------------------------------------------------- #

def test_the_attach_shape_serves_the_model_it_was_given(tmp_path):
    """The exact failure: an earlier launch's MODEL_B2 sits in the env file, the
    attach passes a local merged dir. With the counterpart clear the launcher
    now ships, the merged dir is what gets served."""
    envf = tmp_path / "environment"
    envf.write_text("PATH=/usr/bin\nMODEL_B2=base-models/qwen35-9b\n")
    argv, _ = _serve_dry_run(str(envf), tmp_path, MODEL_ID="/workspace/merged/v13",
                             MODEL_B2="", SERVED_NAME="v13-r64")
    assert argv[2] == "/workspace/merged/v13"
    assert argv[argv.index("--served-model-name") + 1] == "v13-r64"


def test_both_set_still_resolves_to_MODEL_B2_but_says_so(tmp_path):
    """Precedence is deliberately UNCHANGED — q6-round1-evals and the
    eval-template bundles set MODEL_B2 alongside a staged MODEL_ID on purpose.
    What changed is that the override is no longer silent."""
    envf = tmp_path / "environment"
    envf.write_text("PATH=/usr/bin\n")
    argv, out = _serve_dry_run(str(envf), tmp_path, MODEL_ID="/workspace/merged/v13",
                               MODEL_B2="base-models/qwen35-9b",
                               SERVED_NAME="v13-r64")
    assert argv[2] == "/workspace/base-model"
    assert "MODEL_B2 WINS" in out


def test_the_env_file_is_replaced_key_by_key_not_appended(tmp_path):
    """The persistence vector itself. An append stacked a stale MODEL_B2 that
    outlived its run; unrelated lines must survive the rewrite untouched."""
    envf = tmp_path / "environment"
    envf.write_text("PATH=/usr/bin\nLANG=C\n"
                    "MODEL_B2=base-models/qwen35-9b\nSERVED_NAME=old\n")
    _serve_dry_run(str(envf), tmp_path, MODEL_ID="/workspace/merged/v13", MODEL_B2="",
                   SERVED_NAME="v13-r64")
    lines = envf.read_text().splitlines()
    assert "PATH=/usr/bin" in lines and "LANG=C" in lines
    assert "MODEL_B2=base-models/qwen35-9b" not in lines
    assert lines.count("MODEL_B2=") == 1
    assert [ln for ln in lines if ln.startswith("SERVED_NAME=")] \
        == ["SERVED_NAME=v13-r64"]


def test_a_resume_with_no_explicit_model_still_serves_the_launch_model(tmp_path):
    """The legitimate path the fix must not break: no MODEL_ID anywhere, the
    box's own MODEL_B2 resolves to the local pull as it always did."""
    envf = tmp_path / "environment"
    envf.write_text("PATH=/usr/bin\n")
    argv, _ = _serve_dry_run(str(envf), tmp_path, MODEL_B2="base-models/qwen35-9b",
                             SERVED_NAME="qwen35-9b")
    assert argv[2] == "/workspace/base-model"


@pytest.mark.parametrize("model,cleared", [
    ("/workspace/merged/v13", "MODEL_B2"),
    ("b2:base-models/qwen35-9b", "MODEL_ID"),
])
def test_the_attach_dry_run_shows_the_counterpart_clear(model, cleared, tmp_path):
    """--dry-run is the only way to check this without renting a box, so the
    clear has to be visible there and not only in the pushed env file."""
    p = _launch_serve(tmp_path, "--on-box", "12345678", "--model", model,
                      "--api-key-file", str(tmp_path / "key.txt"), "--dry-run",
                      API_KEY_FILE=str(tmp_path / "key.txt"), **_FAKE_B2)
    assert p.returncode == 0, p.stderr
    assert "%s=   (cleared" % cleared in p.stdout, p.stdout


def test_a_model_artifact_attach_ships_the_counterpart_clear_too(tmp_path):
    """`--model-artifact` resolves to a MODEL_B2, so the attach must clear
    MODEL_ID exactly as `--model b2:...` does.

    This is the path where the 2026-08-21 bug is most tempting to reintroduce:
    the model no longer comes from the command line, so it is easy to assume the
    box's own environment is authoritative. It is not — an attach's resolution
    is authoritative, and the counterpart var must die with it.
    """
    p = _launch_serve(tmp_path, "--on-box", "12345678",
                      "--model-artifact", "mergeddemoa",
                      "--api-key-file", str(tmp_path / "key.txt"), "--dry-run",
                      PATH=_rclone_stub(tmp_path) + os.pathsep
                      + os.environ.get("PATH", ""), **_FAKE_B2)
    assert p.returncode == 0, p.stdout + p.stderr
    assert "MODEL_ID=   (cleared" in p.stdout, p.stdout
    assert "model=checkpoints/mergeddemoa-merged/fdfa492a959d/model" in p.stdout


def test_a_model_artifact_attach_stages_the_gate_to_the_box(tmp_path):
    """The attach lane has no B2 boot-pull, so the identity payload goes over
    ssh into /workspace — the same three files, the same names, so
    serve_vllm.sh's resolution ladder finds them either way."""
    p = _launch_serve(tmp_path, "--on-box", "12345678",
                      "--model-artifact", "mergeddemoa",
                      "--api-key-file", str(tmp_path / "key.txt"), "--dry-run",
                      PATH=_rclone_stub(tmp_path) + os.pathsep
                      + os.environ.get("PATH", ""), **_FAKE_B2)
    for name in ("identity_expect.json", "serve_identity_gate.py",
                 "merged_fingerprint.py"):
        assert re.search(r"would stage \d+B -> /workspace/%s" % name, p.stdout), \
            (name, p.stdout)


def _rclone_stub(tmp_path):
    """A fixture `rclone` answering the artifact gate's four verbs.

    Local to this file rather than imported from `test_serve_model_artifact.py`:
    a test helper shared across modules is a second thing to keep in sync, and
    what is under test here is the ATTACH path, not the gate.
    """
    bindir = tmp_path / "arbin"
    bindir.mkdir(exist_ok=True)
    # 27 PAYLOAD objects — mergeddemoa's registry n_files. The marker is transport,
    # not payload, and is excluded from the count on both sides.
    names = [f"f{i}.safetensors" for i in range(1, 28)] + ["PUSHED.json"]
    stub = bindir / "rclone"
    stub.write_text(
        "#!/usr/bin/env bash\ncase \"$1\" in\n"
        "  lsf) cat <<'L'\n" + "\n".join(names) + "\nL\n    ;;\n"
        "  cat) echo '{\"complete\": true, \"files\": 27}';;\n"
        "  size) echo '{\"bytes\": 55834574848}';;\n"
        "  rcat) cat > /dev/null;;\n  *) exit 0;;\nesac\n")
    stub.chmod(0o755)
    return str(bindir)


# --------------------------------------------------------------------------- #
# 2. the self-killing --restart
# --------------------------------------------------------------------------- #

def test_restart_uses_a_pattern_that_cannot_match_its_own_shell():
    src = open(LAUNCH_SH, encoding="utf-8").read()
    assert "pkill -f '[v]llm serve'" in src
    assert "pkill -f 'vllm serve'" not in src, (
        "a bare pattern is present in the remote shell's own argv — it kills "
        "the ssh shell, ssh returns 143 and set -e ends the attach")


def test_the_restart_ssh_exit_code_is_checked_and_reraised():
    src = open(LAUNCH_SH, encoding="utf-8").read()
    assert "remote kill step FAILED" in src
    assert 'exit "$_krc"' in src


def test_a_failed_run_ends_with_a_banner_a_tail_cannot_miss(tmp_path):
    """Callers pipe this to `tail`, which returns its OWN 0. The banner is the
    last line on both streams so the failure is at least visible.

    The forced failure is the B2-creds gate (`--model b2:` with none), NOT the
    onstart cap — re-slimming serve_vllm.sh must not turn this green.
    """
    p = _launch_serve(tmp_path, "--model", "b2:x", "--dry-run",
                      "--api-key-file", str(tmp_path / "key.txt"))
    assert p.returncode != 0, p.stdout + p.stderr
    assert "B2_BUCKET" in p.stderr, p.stderr
    assert p.stdout.strip().splitlines()[-1].startswith("!! launch_serve.sh ABORTED")
    assert p.stderr.strip().splitlines()[-1].startswith("!! launch_serve.sh ABORTED")


def test_control_a_bare_pkill_pattern_really_does_self_match(tmp_path):
    """The control for the pin above: without it, the pin is a style rule. A
    neutral pattern is used so this cannot touch a real serve on the box."""
    bare = tmp_path / "bare.sh"
    bare.write_text("#!/usr/bin/env bash\n"
                    "pkill -f 'zzz_pkill_selfmatch_control' >/dev/null 2>&1\n"
                    "sleep 1\necho SURVIVED\n")
    bracket = tmp_path / "bracket.sh"
    bracket.write_text("#!/usr/bin/env bash\n"
                       "pkill -f '[z]zz_pkill_selfmatch_control' >/dev/null 2>&1 || true\n"
                       "sleep 1\necho SURVIVED\n")
    # the bare form's own argv carries the pattern only when the pattern is on
    # the command line, so run it as `bash -c` the way the ssh step does.
    b = subprocess.run(["bash", "-c", open(bare).read().split("\n", 1)[1]],
                       capture_output=True, text=True, timeout=60)
    assert "SURVIVED" not in b.stdout, "bare pattern did not self-match — control void"
    g = subprocess.run(["bash", str(bracket)], capture_output=True, text=True,
                       timeout=60)
    assert g.returncode == 0 and "SURVIVED" in g.stdout


# --------------------------------------------------------------------------- #
# 3. the oversize-onstart B2 staging fallback (2026-08-22)
#
# The inline wire is ~21kB against a 15872B check (vast's own cap is 16384), so
# EVERY serve launch goes through the B2 boot-pull fallback. Nothing pinned it.
# --------------------------------------------------------------------------- #

def _cap():
    src = open(LAUNCH_SH, encoding="utf-8").read()
    return int(re.search(r"^ONSTART_CAP=(\d+)$", src, re.M).group(1))


def test_the_oversize_onstart_stages_serve_vllm_to_b2_and_ships_the_boot_wire(tmp_path):
    """The path every real launch takes. Asserts the over-cap branch fired for
    the measured reason and that what replaces it fits."""
    p = _launch_serve(tmp_path, "--model", "b2:x", "--dry-run",
                      "--api-key-file", str(tmp_path / "key.txt"), **_FAKE_B2)
    assert p.returncode == 0, p.stdout + p.stderr
    over = re.search(r"onstart wire (\d+)B > (\d+) cap — staging serve_vllm\.sh to B2",
                     p.stderr)
    assert over, p.stderr
    assert int(over.group(1)) > int(over.group(2)) == _cap()
    boot = re.search(r"onstart wire: B2-BOOTSTRAP serve_boot\.sh \d+B "
                     r"\(\+pubkey \d+B \+128 = (\d+)B < (\d+) OK\)", p.stdout)
    assert boot, p.stdout
    assert int(boot.group(1)) < int(boot.group(2)) == _cap()
    assert re.search(r"would stage serve_vllm\.sh \(\d+B\) -> "
                     r"b2:fake/serve/\S+/serve_main\.sh", p.stdout), p.stdout


def test_the_oversize_onstart_with_no_b2_creds_refuses_instead_of_truncating(tmp_path):
    """The guard on that fallback: staging needs creds, and a wire vast would
    truncate must never reach the wire. `--type ondemand` clears the earlier
    spot-needs-a-marker gate so this reaches F-1a."""
    p = _launch_serve(tmp_path, "--model", "org/model", "--type", "ondemand",
                      "--dry-run", "--api-key-file", str(tmp_path / "key.txt"))
    assert p.returncode != 0, p.stdout + p.stderr
    m = re.search(r"F-1a: onstart wire \d+B \+ pubkey \d+B \+ 128 = (\d+)B > (\d+) cap",
                  p.stderr)
    assert m, p.stderr           # also keeps this from going vacuous if re-slimmed
    assert int(m.group(1)) > int(m.group(2)) == _cap()


def test_control_the_env_file_redirect_is_what_decides_the_verdict(tmp_path):
    """Why the redirect exists. Same argv, creds supplied only through the file
    `launch_serve.sh` sources on its own: the abort above becomes a success.
    This is the ambient dependence that made the banner test read the checkout
    it ran in rather than the code."""
    seeded = tmp_path / "seeded.env"
    seeded.write_text("".join("%s=%s\n" % kv for kv in _FAKE_B2.items()))
    p = _launch_serve(tmp_path, "--model", "b2:x", "--dry-run",
                      "--api-key-file", str(tmp_path / "key.txt"),
                      _LAUNCH_SERVE_ENV=str(seeded))
    assert p.returncode == 0, p.stdout + p.stderr
    assert "B2-BOOTSTRAP serve_boot.sh" in p.stdout

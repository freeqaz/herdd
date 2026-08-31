"""Portable tests for onstart/job_serve.sh's `--build-venv` gates — chiefly the
CUDA-init probe added 2026-07-30.

Hermetic: no pip, no CUDA, no GPU, no network. A fake `$SERVE_VENV/bin/python`
shim answers the handful of `-c '<snippet>'` probes the script runs (so the warm
fast path is taken and nothing is installed), and a fake `nvidia-smi` on PATH
decides whether the box "has a GPU".

WHY THIS TEST EXISTS: `import vllm` does not initialize CUDA, so the venv build's
import probe passed on a box whose driver (cuda_max_good 12.9) could not run the
torch cu130 wheels the then-default stock vllm==0.24.0 pulled. Both live frontier
waves therefore failed at ENGINE INIT — after a full S0 stage and a multi-GB
install. That is dated provenance, not the current stack: the shipped image bakes
vLLM and torch cu129, so the pip path does not run. The probe must still fail
loudly at provisioning time on a driver too old for whatever torch resolved, and
must never fail a CPU/rehearsal lane that has no driver at all.
"""
import os
import shutil
import stat
import subprocess
from pathlib import Path

BASH = shutil.which("bash") or "/bin/bash"

JOB_SERVE = (Path(__file__).resolve().parent / "onstart" / "job_serve.sh")


def _exec(p: Path, body: str):
    p.write_text(body)
    p.chmod(p.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return p


#: The shipped image's torch CUDA build (tools/vast/train-env/VLLM_PIN).
TORCH_CUDA = "12.9"


def _fake_include(tmp_path):
    inc = tmp_path / "include"
    inc.mkdir(exist_ok=True)
    (inc / "Python.h").write_text("/* fake */\n")
    return inc


def _shim_body(inc, cuda_ok, cuda_err):
    """A fake interpreter answering every probe job_serve.sh --build-venv runs:
    `import vllm`, `import distutils.core`, the sysconfig include dir (holding a
    Python.h so ensure_py_headers is a no-op), the CUDA-init snippet (succeeds or
    fails per `cuda_ok`), and the `torch.version.cuda` read the failure path uses
    to name the torch that broke. The zeros().cuda() case MUST stay ahead of
    torch.version.cuda — the init snippet holds both strings and `case` takes the
    first match."""
    cuda_branch = (
        f'  echo "cuda_ok {TORCH_CUDA}"; exit 0\n' if cuda_ok else
        f'  echo {cuda_err!r} >&2; exit 1\n')
    return f"""#!/usr/bin/env bash
snippet="${{2:-}}"
case "$snippet" in
  *"import vllm"*) exit 0 ;;
  *"import distutils.core"*) exit 0 ;;
  *"sysconfig"*"include"*) echo "{inc}"; exit 0 ;;
  *"torch.zeros(1).cuda()"*)
{cuda_branch}    ;;
  *"torch.version.cuda"*) echo "{TORCH_CUDA}"; exit 0 ;;
  *"sys.version_info"*) echo "3.12"; exit 0 ;;
esac
exit 0
"""


def _fake_serve_venv(tmp_path, *, cuda_ok=True, cuda_err="boom"):
    """A warm `$SERVE_VENV` whose python takes job_serve.sh's warm fast path."""
    venv = tmp_path / "serve"
    (venv / "bin").mkdir(parents=True)
    _exec(venv / "bin" / "python",
          _shim_body(_fake_include(tmp_path), cuda_ok, cuda_err))
    return venv


def _run(tmp_path, venv, *, with_gpu=True, extra_env=None):
    """`job_serve.sh --build-venv` in a hermetic PATH. `with_gpu` decides whether a
    (fake) nvidia-smi exists — i.e. whether the probe should run at all."""
    binp = tmp_path / "bin"
    binp.mkdir(exist_ok=True)
    env = dict(os.environ)
    if with_gpu:
        _exec(binp / "nvidia-smi", "#!/usr/bin/env bash\nexit 0\n")
        env["PATH"] = f"{binp}:{env['PATH']}"
    else:
        # A box with NO driver: nvidia-smi must be genuinely ABSENT, and the dev
        # machine running this test probably has a real one in /usr/bin. So PATH
        # becomes a curated symlink dir — the utilities job_serve.sh actually
        # shells out to, and nothing else.
        for tool in ("bash", "env", "dirname", "rm", "tail", "cat", "mkdir"):
            src = shutil.which(tool)
            if src and not (binp / tool).exists():
                (binp / tool).symlink_to(src)
        env["PATH"] = str(binp)
    env["SERVE_VENV"] = str(venv)
    env.pop("JOB_SERVE_SKIP_CUDA_PROBE", None)
    if extra_env:
        env.update(extra_env)
    return subprocess.run([BASH, str(JOB_SERVE), "--build-venv"],
                          env=env, capture_output=True, text=True, timeout=120)


def test_build_venv_passes_when_cuda_initializes(tmp_path):
    r = _run(tmp_path, _fake_serve_venv(tmp_path, cuda_ok=True))
    assert r.returncode == 0, r.stderr
    assert "CUDA init OK" in r.stdout
    assert "--build-venv complete" in r.stdout


def test_build_venv_fails_loudly_on_a_too_old_driver(tmp_path):
    """The measured failure text. It must abort --build-venv (rc 3) and name the
    remedy — not sail through and die later at engine init."""
    err = ("RuntimeError: The NVIDIA driver on your system is too old "
           "(found version 12090).")
    r = _run(tmp_path, _fake_serve_venv(tmp_path, cuda_ok=False, cuda_err=err))
    assert r.returncode == 3, (r.returncode, r.stdout, r.stderr)
    assert "CUDA INIT FAILED" in r.stderr
    assert "12090" in r.stderr                      # the driver's own words echoed
    assert "cuda_max_good" in r.stderr              # the rent-time remedy
    assert TORCH_CUDA in r.stderr                   # the torch that actually broke
    # The remedy names the config knob, not a number: the old hardcoded "13.0"
    # was the retired stock lane's floor and contradicted the CLI default.
    assert "LAUNCH_CUDA_MAX_GOOD" in r.stderr
    assert "13.0" not in r.stderr
    assert "--build-venv complete" not in r.stdout


def test_build_venv_fails_on_any_cuda_init_error_without_misdiagnosing(tmp_path):
    """A non-driver CUDA failure still aborts, but must NOT claim the driver is
    too old (that would send the operator renting boxes for no reason)."""
    r = _run(tmp_path, _fake_serve_venv(
        tmp_path, cuda_ok=False, cuda_err="CUDA error: out of memory"))
    assert r.returncode == 3
    assert "CUDA INIT FAILED" in r.stderr
    assert "cuda_max_good" not in r.stderr


def test_cpu_or_rehearsal_lane_skips_the_probe(tmp_path):
    """No nvidia-smi => no driver to prove anything against. Skip with a note,
    never fail — this is the $0 rehearsal / CPU-box path."""
    r = _run(tmp_path, _fake_serve_venv(tmp_path, cuda_ok=False,
                                        cuda_err="would have failed"),
             with_gpu=False)
    assert r.returncode == 0, r.stderr
    assert "skipping CUDA-init probe" in r.stdout
    assert "--build-venv complete" in r.stdout


def test_baked_image_vllm_short_circuits_before_any_pip_install(tmp_path):
    """The shipped path, and the reason the 12.8 rent floor does not leave this
    lane exposed: with vLLM importable from the image's python (t214 bakes it into
    system dist-packages), the global probe returns before $VLLM_SPEC is read — so
    the retired stock spec cannot pull a cu130 torch onto a CUDA-12 box here."""
    binp = tmp_path / "bin"
    binp.mkdir()
    _exec(binp / "python3", _shim_body(_fake_include(tmp_path), True, ""))
    r = _run(tmp_path, tmp_path / "no-such-venv")
    assert r.returncode == 0, r.stderr
    assert "vllm already importable in global python3" in r.stdout
    assert "building serve env" not in r.stdout     # i.e. pip was never reached
    assert "--build-venv complete" in r.stdout


def test_probe_is_explicitly_skippable(tmp_path):
    r = _run(tmp_path, _fake_serve_venv(tmp_path, cuda_ok=False,
                                        cuda_err="would have failed"),
             extra_env={"JOB_SERVE_SKIP_CUDA_PROBE": "1"})
    assert r.returncode == 0, r.stderr
    assert "CUDA-init probe skipped" in r.stdout

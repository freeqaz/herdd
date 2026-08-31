"""Damage ALREADY on disk must be loud, and a fixture must not be able to write it.

`test_b2_conf_guard.py` pins the write guard: a fixture endpoint cannot be
written to the live config. This file pins the two things that guard cannot do.

1. DETECTION. The guard refuses new damage and is blind to damage already
   written, so the 2026-08-22 clobber outlived it by two days: a poisoned
   `[b2]` still matches `grep '^\\[b2\\]'`, every presence probe read it as
   "already configured", and the only symptom was a DNS error pointing at the
   network rather than at the file.

2. PREVENTION AT SOURCE. The escape route was a test forwarding the real `$HOME`
   into a shipped writer. `8d8cb8c37` made every writer honour `RCLONE_CONFIG`,
   so `conftest._rclone_config_scratch` can now redirect the whole suite — and
   the manufactured-escape test below is what proves that redirect load-bearing
   rather than decorative.
"""
import os
import re
import shutil
import subprocess
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
B2_SYNC = os.path.join(_HERE, "b2_sync.sh")
ENSURE_BASE_MODEL = os.path.join(_HERE, "ensure_base_model.sh")

pytestmark = pytest.mark.skipif(not shutil.which("bash"), reason="needs bash")

REAL_ENDPOINT = "https://s3.us-west-004.backblazeb2.com"
POISON_ENDPOINT = "https://example.invalid"

# A secret-shaped value planted in the fixture config. `doctor` reports on a
# file full of live B2 keys, so "prints no key material" is a gate, not a nicety.
PLANTED_SECRET = "K004PLANTEDSECRETMUSTNOTBEPRINTED"


def _conf(path, endpoint, remote="b2", extra=""):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"[{remote}]\n"
        "type = s3\n"
        "provider = Other\n"
        "access_key_id = 004plantedkeyid\n"
        f"secret_access_key = {PLANTED_SECRET}\n"
        f"endpoint = {endpoint}\n"
        "region = us-west-004\n"
        + extra
    )
    return path


def _run(args, conf, **extra):
    """Run b2_sync.sh against CONF, with no ambient B2 creds."""
    e = {k: v for k, v in os.environ.items() if k in ("PATH", "LANG", "TMPDIR")}
    e.update(HOME=str(conf.parent.parent), RCLONE_CONFIG=str(conf), B2_BUCKET="fake")
    e.update({k: v for k, v in extra.items() if v is not None})
    return subprocess.run(["bash", B2_SYNC, *args], capture_output=True,
                          text=True, timeout=120, env=e, cwd=_HERE)


# --------------------------------------------------------------------------- #
# 1. doctor — the three verdicts
# --------------------------------------------------------------------------- #
def test_doctor_calls_a_poisoned_config_poisoned(tmp_path):
    conf = _conf(tmp_path / "h" / "rclone.conf", POISON_ENDPOINT)
    r = _run(["doctor"], conf)
    assert r.returncode == 4, r.stdout + r.stderr
    assert "POISONED" in r.stdout
    assert "example.invalid" in r.stdout


def test_doctor_passes_a_healthy_config(tmp_path):
    conf = _conf(tmp_path / "h" / "rclone.conf", REAL_ENDPOINT)
    r = _run(["doctor"], conf)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "POISONED" not in r.stdout


def test_doctor_reports_an_absent_config_distinctly(tmp_path):
    conf = tmp_path / "h" / "rclone.conf"          # never created
    r = _run(["doctor"], conf)
    assert r.returncode == 2, r.stdout + r.stderr  # 2 absent != 4 poisoned


def test_doctor_finds_a_poisoned_scoped_remote_too(tmp_path):
    """[b2w]/[b2p] carry their own keys and their own endpoint."""
    conf = _conf(tmp_path / "h" / "rclone.conf", REAL_ENDPOINT,
                 extra=f"\n[b2w]\ntype = s3\nendpoint = {POISON_ENDPOINT}\n")
    r = _run(["doctor"], conf)
    assert "[b2w] POISONED" in r.stdout, r.stdout
    # [b2] itself is fine, so the process verdict stays 0 — b2w is reported,
    # not fatal, because reads still work.
    assert r.returncode == 0, r.stdout


def test_doctor_prints_no_key_material(tmp_path):
    conf = _conf(tmp_path / "h" / "rclone.conf", REAL_ENDPOINT)
    r = _run(["doctor"], conf)
    assert PLANTED_SECRET not in (r.stdout + r.stderr)
    assert "004plantedkeyid" not in (r.stdout + r.stderr)


def test_doctor_does_not_install_rclone(tmp_path):
    """The answer is about a FILE, so it must work on the broken box it describes.

    PATH is narrowed to the text tools `doctor` legitimately needs, with rclone
    deliberately absent — an install attempt would hang or fail here rather than
    silently succeed because this workstation happens to have rclone.
    """
    conf = _conf(tmp_path / "h" / "rclone.conf", POISON_ENDPOINT)
    bindir = tmp_path / "bin"
    bindir.mkdir()
    for tool in ("awk", "grep", "sed", "cat", "mktemp", "dirname", "rm", "mv", "chmod"):
        found = shutil.which(tool)
        if found:
            os.symlink(found, bindir / tool)
    assert shutil.which("rclone", path=str(bindir)) is None

    e = {"PATH": str(bindir), "HOME": str(tmp_path / "h"),
         "RCLONE_CONFIG": str(conf), "B2_BUCKET": "fake"}
    r = subprocess.run([shutil.which("bash"), B2_SYNC, "doctor"],  # PATH is narrowed
                       capture_output=True, text=True, timeout=120, env=e, cwd=_HERE)
    assert r.returncode == 4, r.stdout + r.stderr
    assert "installing rclone" not in r.stderr


# --------------------------------------------------------------------------- #
# 2. the refusal, and its one exemption
# --------------------------------------------------------------------------- #
def test_a_poisoned_config_refuses_the_operation_loudly(tmp_path):
    conf = _conf(tmp_path / "h" / "rclone.conf", POISON_ENDPOINT)
    r = _run(["ls"], conf)
    assert r.returncode == 4, r.stdout + r.stderr
    assert "POISONED" in r.stderr


def test_the_refusal_names_the_repair_command(tmp_path):
    """Actionable, or it is just a nicer DNS error."""
    conf = _conf(tmp_path / "h" / "rclone.conf", POISON_ENDPOINT)
    r = _run(["ls"], conf)
    assert "b2_sync.sh config" in r.stderr
    assert ". .env" in r.stderr
    assert "b2_sync.sh doctor" in r.stderr


def test_config_is_never_blocked_by_the_damage_it_repairs(tmp_path):
    """`config` IS the restore path — refusing it would make the box unfixable."""
    conf = _conf(tmp_path / "h" / "rclone.conf", POISON_ENDPOINT)
    r = _run(["config"], conf, B2_KEY_ID="fake", B2_APPLICATION_KEY="fake",
             B2_S3_ENDPOINT=REAL_ENDPOINT)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "example.invalid" not in conf.read_text()   # healed
    assert _run(["doctor"], conf).returncode == 0


def test_a_presence_probe_alone_would_have_passed_this_file(tmp_path):
    """Pins the defect's shape: `[b2]` IS present, and that was the bug."""
    conf = _conf(tmp_path / "h" / "rclone.conf", POISON_ENDPOINT)
    assert "[b2]" in conf.read_text()                  # a presence grep says ok
    assert _run(["doctor"], conf).returncode == 4      # usability says no


def test_ensure_base_model_probes_usability_not_presence():
    """Source pin: it self-configures, and a poisoned stanza must not satisfy it."""
    src = open(ENSURE_BASE_MODEL).read()
    assert "b2_sync.sh\" doctor" in src, "must ask doctor, not grep for a stanza"
    assert "grep -qs '^\\[b2\\]' \"$RCONF\"" not in src


# --------------------------------------------------------------------------- #
# 3. the manufactured escape
# --------------------------------------------------------------------------- #
def test_the_suite_redirects_rclone_config_away_from_the_operator(tmp_path):
    live = os.path.expanduser("~/.config/rclone/rclone.conf")
    assert "RCLONE_CONFIG" in os.environ, "conftest._rclone_config_scratch not active"
    assert os.environ["RCLONE_CONFIG"] != live
    assert not os.environ["RCLONE_CONFIG"].startswith(os.path.dirname(live))


def test_a_writer_inheriting_os_environ_cannot_reach_the_live_path(tmp_path):
    """THE 2026-08-22 ESCAPE, reproduced — and refused by the redirect alone.

    The leak shape verbatim: a test builds its child env as `dict(os.environ)`
    and forwards a `$HOME` whose `.config/rclone/rclone.conf` is the file it
    must not touch. Both of the write guard's nets are deliberately disarmed —
    a REAL endpoint (so the reserved-name check cannot fire) and no
    `PYTEST_CURRENT_TEST` (so the pytest net cannot fire) — because a guard
    that fires here would prove the guard, not the redirect. The only thing
    standing between this call and the operator's file is `RCLONE_CONFIG`.

    Without `conftest._rclone_config_scratch` this test fails on both asserts:
    the writer resolves `$HOME/.config/rclone/rclone.conf` and creates it.
    """
    sandbox_home = tmp_path / "operator-home"          # stands in for the real $HOME
    live = sandbox_home / ".config" / "rclone" / "rclone.conf"

    env = dict(os.environ)                             # ← the leak
    env.pop("PYTEST_CURRENT_TEST", None)               # guard net 2 disarmed
    env.update(HOME=str(sandbox_home), B2_KEY_ID="fake",
               B2_APPLICATION_KEY="fake", B2_BUCKET="fake",
               B2_S3_ENDPOINT=REAL_ENDPOINT)           # guard net 1 disarmed
    scratch = env.get("RCLONE_CONFIG")                 # absent ⇒ writer uses $HOME

    r = subprocess.run(["bash", B2_SYNC, "config"], capture_output=True,
                       text=True, timeout=120, env=env, cwd=_HERE)

    assert r.returncode == 0, r.stdout + r.stderr
    assert not live.exists(), (
        f"a shipped writer reached the live path {live} — RCLONE_CONFIG was "
        f"{'unset' if scratch is None else scratch}"
    )
    assert scratch and "[b2]" in open(scratch).read(), "the write went somewhere unexpected"


def test_every_rclone_config_writer_honours_the_redirect():
    """The redirect is only as good as the writers that read it.

    A writer regressing to a hardcoded `$HOME` would leave the fixture above
    pointing at a file nobody writes — green, and no longer protecting anything.
    """
    import subprocess as sp
    tracked = sp.run(["git", "ls-files", "-z", "tools/"], cwd=_HERE + "/../..",
                     capture_output=True, text=True, timeout=120).stdout.split("\0")
    root = os.path.abspath(os.path.join(_HERE, "..", ".."))
    offenders = []
    for rel in tracked:
        if not rel.endswith(".sh"):
            continue
        try:
            body = open(os.path.join(root, rel), encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        for i, line in enumerate(body.splitlines(), 1):
            if ".config/rclone/rclone.conf" not in line or line.lstrip().startswith("#"):
                continue
            # Only RESOLUTIONS (`FOO=…`/`echo …`) bind a writer to a path. The
            # guard's own `[ "$rc" = "$HOME/…" ]` COMPARES against the live path
            # by design — that is how it recognises it — and must not be flagged.
            if not re.search(r"(^|\s|\()([A-Za-z_][A-Za-z0-9_]*=|echo )", line):
                continue
            if "RCLONE_CONFIG:-" not in line:
                offenders.append(f"{rel}:{i}")
    assert not offenders, (
        "these resolve the rclone config without honouring RCLONE_CONFIG, so the "
        "suite-wide redirect cannot protect the operator from them: " + ", ".join(offenders)
    )


# --------------------------------------------------------------------------- #
# 4. the detector discriminates DAMAGE from somebody else's healthy write
# --------------------------------------------------------------------------- #
# `conftest._protect_operator_rclone_conf` guards a MACHINE-GLOBAL path, so it
# sees writes it did not cause (a peer session's suite, a real `b2_sync.sh
# config`, fleetd). Charging every change to the running test blamed whichever
# test was slowest, and restoring every change reverted legitimate work.
def _conftest_mod():
    """THIS directory's conftest, as pytest already loaded it.

    Not `import conftest`: the repo root ships one too, so the bare name is
    decided by sys.path order and resolves differently in a single-file run
    than in a whole-root one. Not importlib-by-path either — re-executing this
    conftest would re-run its module-level `XDG_CACHE_HOME`/`VAST_HOSTREP_PATH`
    assignments and repoint them mid-session. Finding the live module by
    __file__ also asserts the fixture under test is the one actually in play.
    """
    want = os.path.realpath(os.path.join(_HERE, "conftest.py"))
    for mod in list(sys.modules.values()):
        if os.path.realpath(getattr(mod, "__file__", "") or "") == want:
            return mod
    raise AssertionError(f"{want} is not loaded — cannot test its fixture")


_rclone_conf_damage = _conftest_mod()._rclone_conf_damage

_HEALTHY = f"[b2]\ntype = s3\nendpoint = {REAL_ENDPOINT}\n"


@pytest.mark.parametrize("endpoint", [
    "https://example.invalid", "https://s3.example.com", "http://foo.test",
    "https://box.localhost", "https://example.org:443/bucket",
])
def test_a_reserved_endpoint_is_damage(endpoint):
    """The fixture placeholders that caused the 2026-08-22 clobber. Scheme,
    port and path are stripped before the host is judged, as in b2_sync.sh."""
    assert _rclone_conf_damage(f"[b2]\nendpoint = {endpoint}\n")


@pytest.mark.parametrize("body,why", [
    (None, "gone"), (b"", "empty"), (b"   \n\n", "empty"),
    (b"[r2tc]\nendpoint = https://real.example-host.net\n", "[b2] gone"),
])
def test_a_missing_or_emptied_config_is_damage(body, why):
    """Shapes no legitimate writer produces — text alone is enough to see them."""
    assert _rclone_conf_damage(body), why


def test_a_healthy_rewrite_is_not_damage():
    """THE FALSE ALARM THIS FIXES. Different bytes, still a usable `[b2]` — so
    it is another process's legitimate work, and neither reverting it nor
    reddening an unrelated test is this suite's business."""
    rotated = _HEALTHY.replace("type = s3", "type = s3\naccess_key_id = 004rotated")
    assert rotated != _HEALTHY
    assert _rclone_conf_damage(rotated) is None
    assert _rclone_conf_damage(_HEALTHY.encode()) is None


def test_a_poisoned_sibling_remote_is_damage_even_when_b2_is_healthy():
    """`r2tc`/`hp-b2`/`b2eu` share the file. The 2026-08-22 report noted the
    siblings survived and `rclone listremotes` still looked healthy — a fixture
    that lands on one of them instead is the same defect one remote over."""
    assert _rclone_conf_damage(_HEALTHY + "\n[r2tc]\nendpoint = https://a.invalid\n")


def test_the_detector_and_the_shell_guard_agree_on_reserved_names():
    """One roster, two layers. A name b2_sync.sh refuses to WRITE must be a name
    the conftest detector recognises once it is ON DISK, or damage the shell
    guard would have stopped passes silently through the backstop."""
    src = open(B2_SYNC).read()
    hosts = re.findall(r"\*\.([a-z]+)\|", src) + re.findall(r"\|([a-z]+)\)", src)
    for h in {"invalid", "test", "example", "localhost"} & set(hosts):
        assert _rclone_conf_damage(f"[b2]\nendpoint = https://h.{h}\n"), h


# --------------------------------------------------------------------------- #
# 5. ...and the FIXTURE is wired to that predicate — end to end
# --------------------------------------------------------------------------- #
# The tests above pin a pure function; these run a real pytest whose $HOME is a
# sandbox, so `_protect_operator_rclone_conf` guards a file we control and its
# two branches are observable. Without this, `_rclone_conf_damage` could be
# perfect and the fixture still compare bytes.
_INNER = """
import os
RC = os.path.expanduser("~/.config/rclone/rclone.conf")
HEALTHY_2 = {healthy!r}
POISON    = {poison!r}

def test_1_healthy_external_rewrite():
    open(RC, "w").write(HEALTHY_2)          # must survive, and must not fail

def test_2_poisoned_write():
    open(RC, "w").write(POISON)             # must fail, and must be reverted
"""


def test_the_fixture_discriminates_damage_from_a_healthy_rewrite(tmp_path):
    """The fixture end to end, in ONE inner pytest under a sandbox $HOME.

    Ordered so the second test's restore target is the FIRST test's bytes, not
    the session's: that is the re-baselining, and it is the difference between
    "leave a healthy rewrite alone" and "leave it alone until something else
    goes wrong, then resurrect the config from before it".

    Runs a real pytest because the predicate could be perfect and the fixture
    still compare bytes — these are the only tests here that exercise the
    wiring.
    """
    home = tmp_path / "home"
    (home / ".config" / "rclone").mkdir(parents=True)
    rc = home / ".config" / "rclone" / "rclone.conf"
    rc.write_text(_HEALTHY)                                  # session baseline
    healthy_2 = _HEALTHY.replace("type = s3", "type = s3\naccess_key_id = 004rotated")
    shutil.copy(os.path.join(_HERE, "conftest.py"), tmp_path / "conftest.py")
    (tmp_path / "test_inner.py").write_text(_INNER.format(
        healthy=healthy_2, poison=f"[b2]\nendpoint = {POISON_ENDPOINT}\n"))

    e = {k: v for k, v in os.environ.items() if k in ("PATH", "LANG")}
    e["HOME"] = str(home)
    e.pop("PYTEST_CURRENT_TEST", None)
    # sys.executable, never a bare `python3`: the repo venv is not the system
    # interpreter, and the inner run needs the same pytest this one is using.
    r = subprocess.run([sys.executable, "-m", "pytest", str(tmp_path),
                        "-p", "no:cacheprovider", "-q"],
                       capture_output=True, text=True, timeout=300, env=e)
    out = r.stdout + r.stderr

    # Both bodies pass; the guard fires in TEARDOWN, which pytest counts as a
    # separate error against the poisoning test — so "2 passed, 1 error", and
    # the summary must name the poisoning test and only it.
    assert "2 passed, 1 error" in out, out
    summary = out.split("short test summary")[-1]
    assert "test_2_poisoned_write" in summary, out
    assert "test_1_healthy_external_rewrite" not in summary, out
    assert "DAMAGED during this test" in out
    assert rc.read_text() == healthy_2, (
        "the poisoned write was not reverted to the healthy rewrite that "
        "preceded it — restore target was wrong, or the healthy write was "
        "itself reverted")

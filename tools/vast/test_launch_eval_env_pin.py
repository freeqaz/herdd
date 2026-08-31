"""Portable tests for `herdd launch --eval-env-ver` — the box-side eval pin.

WHY THE FLAG EXISTS ON `launch` AT ALL. jobd's `check_venv eval` provisions the
baked env via onstart/fetch_eval_env.sh, which resolves `${EVAL_ENV_VER:-}` from
the CONTAINER env and otherwise falls back to `rclone cat eval-env/LATEST`. That
happens before the entrypoint subshell where the job's own `.job.env` is
sourced, so a bundle-level or `job submit --env` pin documents the choice but
cannot steer the fetch. Only the box launch env can.

AMENDED 2026-08-25: `check_venv` now reads `EVAL_ENV_VER` straight out of
`.job.env` and runs the fetcher under it, so a ticket pin DOES steer a cold
box's fetch (`test_jobd.test_check_venv_eval_provisions_at_the_ticket_pin`).
The flag is still the only pin that works on a daemon predating that change,
and it stays the one an already-booted box can be given, so nothing here moves.

`--eval-env-ver` shipped on `train` only. Meanwhile docs 97 and 98,
eval-env/bake.sh's own header, jobmeta.eval_env_pin_report's refusal text, and
q6-round1-evals/README §4.1 all instruct the operator to run
`herdd launch --eval-env-ver <v>` — which exited 2, `unrecognized arguments`.
The single documented way to pin a jobs box was a command that does not parse.

It is SUGAR over the `--env` that was already there — one mechanism, one alias —
and it is validated in _do_launch's fail-closed prologue, beside the image
check, so a bad pin refuses before any offer search, key mint or B2 read.

No network, no vast API, no B2: the market and the launch PUT are both faked.
"""
import argparse
import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import herdd  # noqa: E402

# Every seam below is patched at the module `_do_launch` RESOLVES it through —
# `offers`/`fmt`/`imageref` by module attribute, and the two rebound
# `boxes.lifecycle` names at `launch.launch`, which is where `_do_launch` reads
# them (see that module's REBOUND banner). Since plan §8 step 6d the `herdd`
# spellings are re-exports: patching them would leave the real offer search and
# the real `PUT v0/asks/` live.
import imageref  # noqa: E402
from vastlib.core import fmt  # noqa: E402
from vastlib.launch import launch as launch_mod  # noqa: E402
from vastlib.market import offers  # noqa: E402

VER = "20260806-2152-76cd109a"
OTHER = "20260807-0503-84d35a08"          # what eval-env/LATEST actually was


def _ns(**over):
    base = dict(offer=None, type="ondemand", price=None, env=None, port=None,
                jupyter=False, onstart=None, no_hf_token=True, hf_token=None,
                ssh=False, ssh_key_file=None, jobs=False, image="img:tag",
                disk=40, runtype="ssh_direct", label=None, template_id=None,
                no_registry_login=True, login=None, dry_run=False, wait=None,
                force=False, eval_env_ver=None)
    base.update(over)
    return argparse.Namespace(**base)


@pytest.fixture
def body(monkeypatch):
    """Returns fn(ns) -> the raw launch request body."""
    def go(ns):
        bodies = []
        monkeypatch.setattr(offers, "search_offers",
                            lambda a: [{"id": 123, "min_bid": 0.20, "dph_total": 1.00}])
        monkeypatch.setattr(fmt, "fmt_offer", lambda o: "offer-123")
        monkeypatch.setattr(imageref, "image_tag_digest", lambda img: None)
        monkeypatch.setattr(launch_mod, "_launch_preflight", lambda label, force: None)
        monkeypatch.setattr(launch_mod, "launch_instance",
                            lambda oid, b: (bodies.append(b) or (True, 42, None)))
        herdd._do_launch(ns)
        return bodies[0]
    return go


def test_flag_is_accepted_by_the_launch_parser_on_the_real_argv():
    """THE regression: the documented command exited 2 `unrecognized arguments:
    --eval-env-ver`. Asserted through a real argv, because that is where it
    broke — the parser is built inside main(), so an in-process check would be
    testing something the operator never runs.

    Driven with an EMPTY pin so the run stops in the prologue: reaching that
    refusal proves argv parsed the flag, and nothing touches the market."""
    r = subprocess.run(
        [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      "herdd.py"),
         "launch", "--eval-env-ver", "", "--dry-run", "--gpu", "5090"],
        capture_output=True, text=True, timeout=120)
    assert "unrecognized arguments" not in (r.stderr + r.stdout)
    assert "empty value" in (r.stderr + r.stdout)


def test_flag_is_listed_in_launch_help():
    """Docs 97/98 and bake.sh tell operators to run it; `--help` must agree."""
    r = subprocess.run(
        [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      "herdd.py"), "launch", "--help"],
        capture_output=True, text=True, timeout=120)
    assert "--eval-env-ver" in r.stdout


def test_pin_reaches_the_box_env(body):
    assert body(_ns(eval_env_ver=VER))["env"]["EVAL_ENV_VER"] == VER


def test_absent_flag_ships_no_pin(body):
    """A box that names no version must not be given an invented one — an
    unpinned launch resolving LATEST is at least legible in the record."""
    assert "EVAL_ENV_VER" not in body(_ns())["env"]


def test_env_spelling_is_equivalent(body):
    """Sugar, not a second mechanism: both spellings produce the same wire."""
    a = body(_ns(eval_env_ver=VER))["env"]["EVAL_ENV_VER"]
    b = body(_ns(env=[f"EVAL_ENV_VER={VER}"]))["env"]["EVAL_ENV_VER"]
    assert a == b == VER


def test_agreeing_flag_and_env_are_not_a_conflict(body):
    assert body(_ns(eval_env_ver=VER,
                    env=[f"EVAL_ENV_VER={VER}"]))["env"]["EVAL_ENV_VER"] == VER


def test_contradicting_flag_and_env_refuse(body):
    """One box, one baked env. Silently picking a winner here decides which
    game tree the wave grades against."""
    with pytest.raises(SystemExit) as e:
        body(_ns(eval_env_ver=VER, env=[f"EVAL_ENV_VER={OTHER}"]))
    assert "contradicts" in str(e.value)


def test_empty_pin_refuses_rather_than_falling_back_to_latest(body):
    """`--eval-env-ver "$EEV"` with an unset EEV is the exact shape README §4.1
    warns about. An empty string must not read as "no opinion"."""
    for empty in ("", "   "):
        with pytest.raises(SystemExit) as e:
            body(_ns(eval_env_ver=empty))
        assert "empty value" in str(e.value)


def test_refusal_happens_before_any_market_call(monkeypatch):
    """Validated in the prologue beside the image check: refusing costs $0 only
    if it happens before the offer search and the ephemeral key mint."""
    called = []
    monkeypatch.setattr(offers, "search_offers",
                        lambda a: called.append("search") or [])
    with pytest.raises(SystemExit):
        herdd._do_launch(_ns(eval_env_ver=""))
    assert called == []


def test_pin_is_whitespace_normalised(body):
    """`rclone cat eval-env/LATEST` output is routinely newline-terminated and
    gets shell-interpolated straight into the flag."""
    assert body(_ns(eval_env_ver=f"  {VER}\n"))["env"]["EVAL_ENV_VER"] == VER

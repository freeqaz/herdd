"""The stale-pin launch gate (`vastlib.launch.spec`).

`default_image` resolves from whatever checkout herdd was invoked in, and the
primary checkout is shared — a session that leaves it on a pre-roll branch makes
every launch from it rent the old image. A stale checkout is internally
consistent (a roll moves the yaml and the source constant in one commit), so the
only local truth is `origin/main`.

What these pin: the gate fires ONLY on a positive, default-sourced disagreement;
an explicit `--image`, an unreadable canonical pin, and an up-to-date checkout
are all silent; and the refusal names the override.
"""
from __future__ import annotations

import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from vastlib.launch import spec  # noqa: E402

OLD = "registry.example.com/train:t214-latest"
NEW = "registry.example.com/train:t215-latest"


def test_fires_when_the_checkout_default_is_behind():
    why = spec.image_pin_verdict(OLD, OLD, NEW)
    assert why
    assert OLD in why and NEW in why
    assert "--image" in why          # the refusal must name its own override


def test_silent_when_image_was_named_explicitly():
    """A deliberate rollback is intent, not an accident: `--image` makes the
    resolved value differ from the checkout default, and that is never
    second-guessed."""
    assert spec.image_pin_verdict(OLD, NEW, NEW) is None


def test_silent_when_up_to_date():
    assert spec.image_pin_verdict(NEW, NEW, NEW) is None


def test_silent_when_canonical_is_unreadable():
    """No git, no remote ref, or a checkout outside a repo: a gate that cannot
    read the truth must not invent a disagreement."""
    assert spec.image_pin_verdict(OLD, OLD, None) is None
    assert spec.image_pin_verdict(OLD, OLD, "") is None


def test_silent_when_there_is_no_image():
    assert spec.image_pin_verdict(None, OLD, NEW) is None
    assert spec.image_pin_verdict("", OLD, NEW) is None


# --- canonical read -------------------------------------------------------

class _CP:
    def __init__(self, returncode=0, stdout=""):
        self.returncode = returncode
        self.stdout = stdout


def test_canonical_reads_default_image_from_the_ref():
    got = spec.canonical_default_image(
        runner=lambda args: _CP(0, f"# comment\ndefault_image: {NEW}\nx: 1\n"))
    assert got == NEW


def test_canonical_is_soft_on_every_failure():
    assert spec.canonical_default_image(runner=lambda a: _CP(128, "")) is None
    assert spec.canonical_default_image(runner=lambda a: _CP(0, "no pin\n")) is None

    def _boom(args):
        raise OSError("git not found")

    assert spec.canonical_default_image(runner=_boom) is None


def test_canonical_asks_git_for_the_pinned_ref_and_path():
    seen = {}

    def _runner(args):
        seen["args"] = args
        return _CP(0, f"default_image: {NEW}\n")

    spec.canonical_default_image(runner=_runner, repo_root="/repo")
    assert seen["args"][:3] == ["git", "-C", "/repo"]
    assert f"{spec._CANONICAL_PIN_REF}:{spec._CANONICAL_PIN_PATH}" in seen["args"]


@pytest.mark.parametrize("stdout", ["default_image:   spaced\n",
                                    "default_image:tight\n"])
def test_canonical_tolerates_yaml_spacing(stdout):
    assert spec.canonical_default_image(runner=lambda a: _CP(0, stdout))

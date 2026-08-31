"""The `cli/` composition root's WIRING — the three cross-ring seams, bound.

Why this file exists
--------------------
`vastlib/launch/launch.py` reaches three names it may not import:
`compose_jobs_launch_env` (defined in `jobs/bundle.py`),
`fleet_watch_best_effort` and `fleet_operator_intent` (both `fleet/client.py`).
All three live in rings ABOVE their callers, so the call sites are raising
seams and only `cli` may close them (`vastlib/cli/_compose.py`).

That left a real claim standing in the step-6 notes: **"the `--jobs` launch path
is dead."** It was true — `herdd launch --jobs` on the vastlib arm would have
walked its whole prologue, resolved an image, searched the market, assembled an
env, and then raised `NotImplementedError` on the one call that turns a plain
box into a jobs box. This file is the proof that it is no longer true, and it is
written to fail if the binding is removed: the composer recorder can only fire
if `cli/launch.py::run` bound it, because the test drives `a.func(a)` off
`build_parser()` and NEVER calls `cli.main.main()` (which does its own bind).

The census hazard, and the fixture that closes it
------------------------------------------------
`test_vastlib_launch.py::test_every_seam_exists_and_raises` asserts that two of
these three still raise. Binding is a module-attribute assignment and therefore
PROCESS-GLOBAL, and this file sorts before that one, so without care a green run
here would make that assertion pass or fail on collection order — a test whose
verdict depends on import order is not a test. `_restore_seam_bindings` is
autouse and snapshots all three attributes around every test in this file, so
the census this file perturbs is the census it hands back. That is also why
`_compose.bind()` is called from `run()` and not at `cli` import: importing a
module must not change what another module's attributes do.

Isolation: no network, no B2, no mint, no vast API — and no `.env`, which is a
separate and sharper hazard here than in most files: driving the REAL parser
runs `config.load_env()`, and a leaked B2 master key mints a real key in
whatever test runs next (`_no_dotenv_and_no_env_leak` below has the measured
case). Every transport is stubbed
at its OWNING module — `api.request_soft`, `offers.search_offers`,
`launch.launch_instance` / `_launch_preflight` / `_emit_launched_soft`,
`imageref.image_tag_digest`, `b2._rclone_soft` — and the two upper-ring
definitions are replaced by recorders, so nothing this file runs can compose a
real jobs bundle or register a real watch. $0.
"""
from __future__ import annotations

import argparse
import ast
import importlib
import os
import pathlib
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fleetd  # noqa: E402  the Zone E launcher — a composition root too
import imageref  # noqa: E402
import runmeta  # noqa: E402

from vastlib.boxes import lifecycle  # noqa: E402
from vastlib.cli import _compose  # noqa: E402
from vastlib.cli import launch as cli_launch  # noqa: E402
from vastlib.cli import main as cli_main  # noqa: E402
from vastlib.cli import supervise as cli_supervise  # noqa: E402
from vastlib.cli import train as cli_train  # noqa: E402
from vastlib.core import api  # noqa: E402
from vastlib.core import config as cli_config  # noqa: E402
from vastlib.fleet import client as fleet_client  # noqa: E402
from vastlib.fleet import daemon as fleet_daemon  # noqa: E402
from vastlib.jobs import bundle  # noqa: E402
from vastlib.launch import launch as launch_mod  # noqa: E402
from vastlib.market import offers as offers_mod  # noqa: E402
from vastlib.storage import b2  # noqa: E402


@pytest.fixture(autouse=True)
def _no_dotenv_and_no_env_leak(monkeypatch):
    """`build_parser()` calls `config.load_env()` — do not let it run, and do not
    let anything this file runs leave a key in `os.environ`.

    MEASURED, not theoretical (2026-08-16): `cli.main.build_parser()` opens with
    `config.load_env()`, which `setdefault`s the developer's real `.env` — B2
    master key, bucket, endpoint — into the process environment. `_run_argv`
    below drives the real parser, so before this fixture existed this file
    exported live B2 credentials to every test that ran after it, and
    `test_supervise.py::test_spec_roundtrip_write_capture_relaunch_body` (whose
    `_ship_b2_env` is NOT stubbed, because with no creds it takes the local
    box-scoped branch) reached the real API and MINTED AND REVOKED A REAL B2
    KEY. The stub is the fix; the snapshot is the backstop for whatever else a
    launch path `setdefault`s. Same hazard `test_vastlib_cli_surface.py`'s
    `_pinned_environment` exists for, and the same reason
    `test_vastlib_cli_main.py` stubs `load_env` around its `main()` calls.

    `conftest.py::_pin_process_environment` now restores `os.environ` for the
    whole suite (a second leaker, `test_vastlib_cli_workflow.py::flat_group`,
    drives flat `herdd.main()` and predates this file), and
    `_block_real_b2_mint` refuses the key API outright. Both are backstops; the
    `load_env` stub here is the fix, because not reading the developer's
    credentials at all beats restoring them afterwards."""
    monkeypatch.setattr(cli_config, "load_env", lambda: None)
    before = dict(os.environ)
    yield
    if dict(os.environ) != before:
        os.environ.clear()
        os.environ.update(before)


@pytest.fixture(autouse=True)
def _restore_seam_bindings():
    """Hand back exactly the seam census we were given (see the header).

    Snapshot-and-restore rather than `monkeypatch.setattr`, because the thing
    that mutates these attributes is the code under test, not the test.

    `conftest.py::_restore_cross_ring_seam_bindings` now does the same job for
    the WHOLE suite — it had to, because `cli.main.main()` and `cli/train.py`'s
    `run()` bind too and three other test files drive them. This one stays: it
    is local to the file whose entire subject is the binding, it documents the
    contract at the point of use, and two restores of the same three attributes
    cost nothing."""
    saved = [(launch_mod, "compose_jobs_launch_env"),
             (launch_mod, "fleet_watch_best_effort"),
             (lifecycle, "fleet_operator_intent")]
    before = [(m, n, getattr(m, n)) for m, n in saved]
    yield
    for mod, name, fn in before:
        setattr(mod, name, fn)


# =============================================================================
# the census: importing `cli` must not bind anything
# =============================================================================
def test_importing_cli_leaves_the_seams_raising():
    """The whole reason `bind()` is called from `run()` and not from an import.

    `vastlib.cli.launch`, `vastlib.cli.main` and `vastlib.cli._compose` are all
    imported at the top of this file. If any of them bound on import, the two
    `launch` seams would already be live here — and
    `test_vastlib_launch.py`'s census would then pass or fail depending on
    which test file pytest collected first."""
    for name in ("compose_jobs_launch_env", "fleet_watch_best_effort"):
        with pytest.raises(NotImplementedError):
            getattr(launch_mod, name)(*([None] * 2))
    with pytest.raises(NotImplementedError):
        lifecycle.fleet_operator_intent(None, "stop")


# =============================================================================
# bind() itself
# =============================================================================
def test_bind_points_every_row_of_the_table_at_its_owner():
    """`SEAM_BINDINGS` is the wiring as data. Asserting identity (not just
    callability) is what proves the assignment happened rather than some
    same-named local: `is` fails on a delegating wrapper, which is the shape
    that would quietly reintroduce a second implementation."""
    _compose.bind()
    for caller_mod, name, owner_mod in _compose.SEAM_BINDINGS:
        caller = importlib.import_module(caller_mod)
        owner = importlib.import_module(owner_mod)
        assert getattr(caller, name) is getattr(owner, name), name


def test_the_conftest_seam_roster_matches_compose():
    """The suite-wide restore fixture is only a guard if it names every seam.

    `conftest.py::_CROSS_RING_SEAMS` is what keeps a bind performed by ANY test
    (`cli.main.main()` in `test_vastlib_cli_main.py`, `cli_train.run()` in
    `test_job_submit_preflight.py`, ...) from leaking into
    `test_vastlib_launch.py`'s census. A fourth row added to `SEAM_BINDINGS`
    without a line there would reopen exactly that order-dependence, silently
    and only in some collection orders — so the two rosters are pinned to each
    other here rather than trusted to stay in sync."""
    import conftest
    assert (tuple((caller, name) for caller, name, _owner in _compose.SEAM_BINDINGS)
            == conftest._CROSS_RING_SEAMS)


def test_bind_is_idempotent():
    _compose.bind()
    first = (launch_mod.compose_jobs_launch_env,
             launch_mod.fleet_watch_best_effort,
             lifecycle.fleet_operator_intent)
    _compose.bind()
    assert (launch_mod.compose_jobs_launch_env,
            launch_mod.fleet_watch_best_effort,
            lifecycle.fleet_operator_intent) == first


def test_bind_reads_the_owner_at_call_time(monkeypatch):
    """The steering contract, stated in `_compose`'s docstring and asserted
    here so it cannot rot: a test patches the OWNER and the command sees it. A
    late bind that had snapshotted `bundle.compose_jobs_launch_env` at import
    would silently ignore the patch and the test would go vacuously green
    against the real composer."""
    sentinel = object()
    monkeypatch.setattr(bundle, "compose_jobs_launch_env", sentinel)
    _compose.bind()
    assert launch_mod.compose_jobs_launch_env is sentinel


def test_a_patch_of_the_CALL_SITE_is_overwritten_by_the_next_bind(monkeypatch):
    """The other half of the same contract, and the trap it sets. These three
    are the OPPOSITE of `launch.py`'s `_REBOUND_SEAMS` (bound once, at import,
    where the call-site attribute IS the patch point). Patching `launch_mod`
    here does nothing, and a test that did it would report on a stub the
    command never called."""
    monkeypatch.setattr(launch_mod, "compose_jobs_launch_env", object())
    _compose.bind()
    assert launch_mod.compose_jobs_launch_env is bundle.compose_jobs_launch_env


# =============================================================================
# `herdd launch --jobs`, end to end through the parser
# =============================================================================
_ARGV_BASE = ["launch", "--image", "img:tag", "--no-ssh", "--no-hf-token",
              "--no-registry-login", "--type", "ondemand"]


def _wire(monkeypatch, *, composer=None, watcher=None):
    """Stub every transport `_do_launch` can reach, plus the two upper-ring
    definitions. Returns the list captured launch bodies land in.

    Each stub is installed at the module that OWNS the name, without
    `raising=False`, so a seam that moves fails loudly here instead of going
    vacuous."""
    bodies: list[dict] = []
    monkeypatch.setattr(api, "request_soft",
                        lambda *a, **k: (False, None, "no network in tests"))
    monkeypatch.setattr(b2, "_rclone_soft", lambda args: (1, "", "no b2 in tests"))
    # `fmt.fmt_offer` reads id/num_gpus/gpu_name WITHOUT a default — ported
    # behavior, so the fake row carries them rather than the printer being
    # stubbed out (stubbing it would hide a real KeyError on the launch path).
    monkeypatch.setattr(offers_mod, "search_offers",
                        lambda a: [{"id": 123, "min_bid": 0.20, "dph_total": 1.00,
                                    "num_gpus": 1, "gpu_name": "RTX_4090",
                                    "machine_id": 555}])
    monkeypatch.setattr(imageref, "image_tag_digest", lambda img: None)
    monkeypatch.setattr(launch_mod, "_require_image", lambda image, what: image)
    monkeypatch.setattr(launch_mod, "_launch_preflight", lambda label, force: None)
    monkeypatch.setattr(launch_mod, "_emit_launched_soft",
                        lambda a, body, cid, oid, dph: None)
    monkeypatch.setattr(launch_mod, "launch_instance",
                        lambda oid, body: (bodies.append(body) or (True, 42, None)))
    if composer is not None:
        monkeypatch.setattr(bundle, "compose_jobs_launch_env", composer)
    if watcher is not None:
        monkeypatch.setattr(fleet_client, "fleet_watch_best_effort", watcher)
    return bodies


def _run_argv(argv):
    """Drive the REAL dispatch shape — `build_parser().parse_args()` then
    `a.func(a)` — deliberately WITHOUT `cli.main.main()`. `main()` binds too, so
    going through it would prove nothing about `cli/launch.py::run`; this way
    the only thing that can wire the composer is the command itself."""
    a = cli_main.build_parser().parse_args(argv)
    # The dispatch target is asserted, not assumed: everything below concludes
    # "`cli/launch.py::run` bound the seam", which is only true while `launch` is
    # what the parser dispatches to. A registry edit that pointed the subcommand
    # elsewhere would otherwise leave these tests green about the wrong module.
    assert a.func is cli_launch.run
    a.func(a)
    return a


def test_cli_launch_jobs_reaches_the_composer_through_the_binding(monkeypatch):
    """THE claim this file exists to kill: "the --jobs launch path is dead".

    Before the binding, this argv raised `NotImplementedError` out of
    `_do_launch` — after resolving an image and searching the market, which is
    the expensive half. The recorder below can only fire if `run()` bound
    `bundle.compose_jobs_launch_env` onto `launch_mod`."""
    seen = {}

    def _composer(env, onstart, **kw):
        seen["env"] = env
        seen["onstart"] = onstart
        seen["kw"] = kw
        env["JOBD_BOOT"] = "1"                      # the composer MUTATES env
        return "ONSTART-FROM-COMPOSER", "sha-deadbeef"

    bodies = _wire(monkeypatch, composer=_composer)
    _run_argv(_ARGV_BASE + ["--jobs"])

    assert seen, "the composer was never reached — the seam is still raising"
    # ...and its return value is what the create body actually carries. A bind
    # that reached the composer but dropped its output would be the same bug
    # with a longer stack trace: a box that boots without jobd.
    assert bodies[0]["onstart"] == "ONSTART-FROM-COMPOSER"
    assert bodies[0]["env"]["JOBD_BOOT"] == "1"
    assert seen["env"] is bodies[0]["env"], "env is mutated in place, not copied"


def test_the_composer_gets_the_jobs_flags_the_parser_produced(monkeypatch):
    """The keywords are the `--jobs` sub-flags, and they travel by NAME. A bind
    behind a narrower stub signature would have type-checked and then dropped
    one of these on the floor — which is the failure `launch.py`'s
    `compose_jobs_launch_env` docstring records for `key_base` / `timeout_s` /
    `bootstrap_stager`."""
    seen = {}

    def _composer(env, onstart, **kw):
        seen.update(kw)
        return None, "sha"

    _wire(monkeypatch, composer=_composer)
    _run_argv(_ARGV_BASE + ["--jobs", "--no-idle-park",
                            "--idle-park-grace", "900",
                            "--no-job-deadline", "1800"])
    assert seen["no_idle_park"] is True
    assert seen["idle_park_grace"] == 900
    assert seen["no_job_deadline"] == 1800
    assert seen["dry_run"] is False


def test_a_plain_launch_never_calls_the_composer(monkeypatch):
    """The binding must not change what a NON-jobs launch does. `--jobs` is the
    gate; wiring the seam is not the same as firing it."""
    calls = []
    bodies = _wire(monkeypatch,
                   composer=lambda env, onstart, **kw: calls.append(kw) or (None, ""))
    _run_argv(_ARGV_BASE)
    assert calls == []
    assert bodies and "JOBD_BOOT" not in bodies[0].get("env", {})


def test_driving_the_parser_leaves_no_environment_behind(monkeypatch):
    """The isolation the autouse fixture buys, asserted where it can rot.

    A launch driven through the REAL parser is the one shape in this file that
    can fold `.env` into `os.environ` — and a leaked B2 master key is not inert
    downstream: it flips `_ship_b2_env` off its local-creds branch and mints a
    real key in whatever test runs next (that is how this was found). Comparing
    against a snapshot rather than checking specific names, because the leak's
    contents are whatever the developer's `.env` happens to hold."""
    before = dict(os.environ)
    _wire(monkeypatch)
    _run_argv(_ARGV_BASE)
    changed = {k: v for k, v in os.environ.items() if before.get(k) != v}
    assert changed == {}, f"launch leaked env: {sorted(changed)}"
    assert not (set(before) - set(os.environ)), "launch deleted env keys"


def test_cli_launch_fleet_watch_reaches_the_daemon_client(monkeypatch):
    """The second seam, on the same command. `--fleet-watch` closes the
    launch->watch gap (FLEETD_DESIGN §3 B1); unbound, it raised AFTER the PUT —
    i.e. after a box was already rented and billing. Explicit `--fleet-watch`
    is now a no-op affirmation of the default (below) — pinned so existing
    scripts that still pass it keep working."""
    watched = []
    _wire(monkeypatch,
          watcher=lambda target, profile="bare", budget_usd=None, policy=None:
              watched.append((target, profile, policy)) or True)
    _run_argv(_ARGV_BASE + ["--fleet-watch"])
    assert watched, "fleet_watch_best_effort was never reached"
    target, profile, policy = watched[0]
    assert (target, profile) == (42, "bare")
    assert policy == {"launched_label": "herdd"}


def test_launch_defaults_to_fleet_watch_on(monkeypatch):
    """FLEET_REVIEW_2026-08-20 item 3: 272 journaled bare auto-adoptions came
    from `launch`/`train` registering only under an opt-in flag. `--fleet-watch`
    is default ON since 2026-08-20 — no flag at all must still register."""
    watched = []
    _wire(monkeypatch,
          watcher=lambda target, profile="bare", budget_usd=None, policy=None:
              watched.append((target, profile, policy)) or True)
    _run_argv(_ARGV_BASE)
    assert watched, "fleet_watch_best_effort was never reached with no flag passed"


def test_a_jobs_launch_says_the_watch_it_just_made_is_bare(monkeypatch, capsys):
    """The registration above is `bare` — no ladder — and the flag that made it
    reads like supervision. A jobs box is the case that owes the follow-up, so
    the hint rides the launch line where the operator still has the id."""
    _wire(monkeypatch,
          composer=lambda env, onstart, **kw: (onstart, "deadbeef"),
          watcher=lambda *a, **k: True)
    _run_argv(_ARGV_BASE + ["--jobs"])
    out = capsys.readouterr().out
    assert "no bid defense" in out
    assert "fleet watch 42 --profile jobs --budget <USD> --standing" in out


def test_a_non_jobs_launch_gets_no_ladder_hint(monkeypatch, capsys):
    """`--profile jobs` is the wrong command for a serve or manual box, and
    wrong advice is worse than silence. Also keeps the plain launch banner from
    growing three lines every operator learns to skip."""
    _wire(monkeypatch, watcher=lambda *a, **k: True)
    _run_argv(_ARGV_BASE)
    assert "no bid defense" not in capsys.readouterr().out


def test_no_fleet_watch_also_silences_the_ladder_hint(monkeypatch, capsys):
    """Opting out of registration opts out of the nudge about it."""
    _wire(monkeypatch, composer=lambda env, onstart, **kw: (onstart, "deadbeef"),
          watcher=lambda *a, **k: True)
    _run_argv(_ARGV_BASE + ["--jobs", "--no-fleet-watch"])
    assert "no bid defense" not in capsys.readouterr().out


def test_launch_with_no_fleet_watch_registers_nothing(monkeypatch):
    watched = []
    _wire(monkeypatch,
          watcher=lambda *a, **k: watched.append(a) or True)
    _run_argv(_ARGV_BASE + ["--no-fleet-watch"])
    assert watched == []


# =============================================================================
# `herdd train`'s OWN --fleet-watch (train.py:709-715, registers `run:<RUN>`,
# distinct from the `la.fleet_watch=False` synthetic Namespace that reaches
# `_do_launch` two lines above it — see that line's comment)
# =============================================================================
_TRAIN_ARGV_BASE = ["train", "--run", "r1", "--runset", "__test_no_such_runset__",
                    "--no-asset-check"]


def _run_train_argv(argv):
    """`_run_argv`'s twin for `train`, which dispatches to `cli_train.run` —
    `_run_argv` hardcodes the `launch` dispatch-target assertion, so it cannot
    be reused as-is here."""
    a = cli_main.build_parser().parse_args(argv)
    assert a.func is cli_train.run
    a.func(a)
    return a


def _wire_train(monkeypatch, *, watcher=None):
    """Stub every transport `cli_train.run` touches on the way to the
    fleet-watch gate: the runset name has no `runsets/<name>/` directory, so
    `_load_runset_config`/the base-model gate no-op for free; everything past
    that (B2 marker writes, image digest, the launched-event subprocess, and
    `_do_launch` itself) is stubbed at its owning module, same idiom as
    `_wire()` above. `os.execv` is stubbed too — a real `--supervise` run
    replaces THIS process, so a test driving that branch must never let it
    reach the real call."""
    monkeypatch.setenv("B2_BUCKET", "bkt")
    for k in ("B2_KEY_ID", "B2_APPLICATION_KEY", "B2_S3_ENDPOINT"):
        monkeypatch.setenv(k, "x")
    monkeypatch.setattr(b2, "_ensure_b2_remote", lambda: None)
    monkeypatch.setattr(b2, "_b2_rcat", lambda path, body, hard=True: True)
    monkeypatch.setattr(b2, "_rclone_soft", lambda args: (0, "", ""))
    monkeypatch.setattr(imageref, "image_tag_digest", lambda img: None)
    monkeypatch.setattr(api, "request_soft",
                        lambda *a, **k: (False, None, "no network in tests"))
    monkeypatch.setattr(subprocess, "run",
                        lambda *a, **k: subprocess.CompletedProcess(a, 0, "", ""))
    monkeypatch.setattr(os, "execv", lambda *a, **k: None)
    monkeypatch.setattr(launch_mod, "_do_launch", lambda la: (42, 99, 1.0))
    if watcher is not None:
        monkeypatch.setattr(fleet_client, "fleet_watch_best_effort", watcher)


def test_train_defaults_to_fleet_watch_on(monkeypatch):
    """Same gap as `launch`, second launcher (FLEET_REVIEW_2026-08-20 item 3):
    `train`'s `--fleet-watch` is now default ON too — no flag at all must
    still register the run:<RUN>-labelled watch."""
    watched = []
    _wire_train(monkeypatch,
               watcher=lambda target, profile="bare", budget_usd=None, policy=None:
                   watched.append((target, profile, policy)) or True)
    _run_train_argv(_TRAIN_ARGV_BASE)
    assert watched, "fleet_watch_best_effort was never reached with no flag passed"
    target, profile, policy = watched[0]
    assert (target, profile) == ("run:r1", "bare")
    assert policy == {"instance_id": 42}


def test_train_no_fleet_watch_registers_nothing(monkeypatch):
    watched = []
    _wire_train(monkeypatch, watcher=lambda *a, **k: watched.append(a) or True)
    _run_train_argv(_TRAIN_ARGV_BASE + ["--no-fleet-watch"])
    assert watched == []


def test_train_supervise_skips_fleet_watch_even_with_default_on(monkeypatch):
    """The carve-out this task must preserve, unchanged (train.py:712's `and
    not supervise`): `--supervise` arms its own handoff, so it must NOT also
    register a bare fleetd watch — even now that `--fleet-watch` defaults to
    True. Only the upstream default moved; this gate did not."""
    watched = []
    _wire_train(monkeypatch, watcher=lambda *a, **k: watched.append(a) or True)
    monkeypatch.setattr(fleet_client, "_supervise_argv",
                        lambda *a, **k: ["herdd", "supervise", "r1"])
    _run_train_argv(_TRAIN_ARGV_BASE + ["--supervise", "--budget", "5"])
    assert watched == [], "train --supervise must not ALSO register a bare fleetd watch"


# =============================================================================
# the other two commands that reach `_do_launch` without `main()`
# =============================================================================
# Both are driven to their FIRST validation failure with a recorder in place of
# `bind`: that proves the ordering (wiring happens before anything that can
# exit), which is the property that matters. `cli/supervise.py` is the one where
# a late bind would be worst — it reaches `_do_launch` indirectly, hours in,
# through the run lane's eviction relaunch.

def test_train_binds_before_its_first_validation(monkeypatch):
    called = []
    monkeypatch.setattr(_compose, "bind", lambda: called.append("bind"))
    with pytest.raises(SystemExit):
        cli_train.run(argparse.Namespace(run="not a valid run id!", runset="rs"))
    assert called == ["bind"]


def test_supervise_binds_before_its_first_validation(monkeypatch):
    called = []
    monkeypatch.setattr(_compose, "bind", lambda: called.append("bind"))
    # `runmeta.validate_run_id` raises RunmetaError (not SystemExit) — the point
    # is only that the bind already happened when it did.
    with pytest.raises(runmeta.RunmetaError):
        cli_supervise.run(argparse.Namespace(run_id="not a valid run id!",
                                             budget=None, dry_run=False))
    assert called == ["bind"]


# =============================================================================
# the third seam: `fleet_operator_intent` on the boxes ring
# =============================================================================
def test_the_operator_intent_seam_is_wired_by_the_same_composition(monkeypatch):
    """`boxes.lifecycle` calls it from `cmd_stop` / `cmd_start` / `cmd_destroy`
    and `boxes.reap`, all of which are `cli` commands. It rides the same
    `bind()` for the same reason — `fleet` sits above `boxes` — and it is the
    one whose silent no-op costs the most: a human's `herdd stop` read as
    OUTBID gets the box rescue-resumed and billing all night (SPOT_DESIGN
    §3.5)."""
    called = []
    monkeypatch.setattr(fleet_client, "fleet_operator_intent",
                        lambda iid, kind, reason=None:
                            called.append((iid, kind, reason)) or {"ok": True})
    _compose.bind()
    assert lifecycle.fleet_operator_intent(7, "stop", reason="test") == {"ok": True}
    assert called == [(7, "stop", "test")]


# =============================================================================
# the FIFTH entry point: `tools/vast/fleetd.py`, which is not in this ring
# =============================================================================
# THE DEFECT THESE PIN (live, 2026-08-17). `bind()` had four callers and all
# four were `cli` modules. The fleet daemon does not go through `cli` — its own
# composition root is `fleet.daemon`, which MAY NOT import `cli` (top layer of
# `importlinter.ini`) — so inside the fleetd process every seam stayed raising:
#
#   !! pull watchdog: replacement launch failed (unlaunchable: replacement
#      launch error: compose_jobs_launch_env: not ported yet ...)
#
# reached as `daemon.tick -> supervise.{run_lane,job_lane} ->
# replacement._relaunch -> launch._do_launch`, every tick, with the condemned
# box never replaced. Two seams further down the same call are
# `fleet_watch_best_effort` (successor launched UNWATCHED) and, via
# `lifecycle._destroy_and_revoke`, `fleet_operator_intent`.
#
# `tools/vast/fleetd.py` is Zone E, outside `root_package = vastlib`, so it is
# the one place allowed to hold both halves. Both tests below are driven off
# `_compose.SEAM_BINDINGS` / the launcher's own AST, so a fifth seam row or a
# deleted call is a failure here rather than a silent hole in the daemon.

def test_the_fleetd_launcher_binds_every_seam(monkeypatch):
    """RUN-time behavior, not source inspection: after the launcher's entry
    path, every `SEAM_BINDINGS` row points at its owner BY IDENTITY.

    The census is forced UNBOUND first. Without that this test would be
    vacuous the moment any earlier test in the session ran a `cli` command —
    the exact process-global-bind hazard this file's header describes, turned
    against the assertion instead of against the census.

    `main` is replaced so nothing dispatches: the claim is about composition,
    and `run()` resolves `main` as a module global, so this patch steers it."""
    dispatched: list[object] = []
    monkeypatch.setattr(fleetd, "main",
                        lambda argv=None: (dispatched.append(argv), 0)[1])
    for caller_mod, name, _owner_mod in _compose.SEAM_BINDINGS:
        monkeypatch.setattr(importlib.import_module(caller_mod), name, object())

    assert fleetd.run(["serve", "--once"]) == 0
    assert dispatched == [["serve", "--once"]], "run() must still dispatch"

    for caller_mod, name, owner_mod in _compose.SEAM_BINDINGS:
        caller = importlib.import_module(caller_mod)
        owner = importlib.import_module(owner_mod)
        assert getattr(caller, name) is getattr(owner, name), (
            f"fleetd never bound {caller_mod}.{name} — the daemon will raise "
            f"on it the first time a tick reaches that lane")


def test_the_fleetd_entry_path_still_performs_the_binding():
    """The other direction: deleting the call must not go quiet.

    The test above proves `run()` binds; this one proves `run()` is what the
    systemd unit's `ExecStart={python} {script} serve` actually reaches, and
    that the bind is on the RUN path rather than at module scope. AST, not
    grep — the comment block above this section names every seam by hand and
    would otherwise satisfy a string search on its own."""
    tree = ast.parse(pathlib.Path(fleetd.__file__).read_text())

    guards = [n for n in tree.body
              if isinstance(n, ast.If) and "__main__" in ast.unparse(n.test)]
    assert len(guards) == 1, "the launcher has exactly one __main__ guard"
    assert "run(" in ast.unparse(guards[0]), (
        "the __main__ guard no longer routes through `run()` — the daemon "
        "would start with every cross-ring seam raising")

    fns = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}

    def calls(fn: str) -> set[str]:
        return {ast.unparse(c.func)
                for c in ast.walk(fns[fn]) if isinstance(c, ast.Call)}

    assert "_bind_cross_ring_seams" in calls("run")
    assert "_compose.bind" in calls("_bind_cross_ring_seams")

    # ...and NOT at import. A module-level `bind()` here would rebind the seams
    # for every one of the ~20 test modules that `import fleetd`, which is what
    # made `test_vastlib_launch.py`'s census order-dependent (four measured
    # failures behind `test_vastlib_cli_main.py`).
    top_level_calls = {ast.unparse(c.func)
                       for n in tree.body
                       if not isinstance(n, (ast.FunctionDef, ast.ClassDef))
                       for c in ast.walk(n) if isinstance(c, ast.Call)}
    assert "_compose.bind" not in top_level_calls
    assert "_bind_cross_ring_seams" not in top_level_calls


def test_the_launcher_did_not_wrap_main_to_get_the_binding():
    """`fleetd.py`'s header rule: every re-export is a plain `from … import`
    binding, so `inspect.getsource` and `monkeypatch` land on the daemon's own
    object. Closing the seam gap by wrapping `main` would have bought the fix
    at the cost of that contract — hence a separate name (`run`)."""
    assert fleetd.main is fleet_daemon.main
    assert "main" in fleetd.__all__ and "run" not in fleetd.__all__

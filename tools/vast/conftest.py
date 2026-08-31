"""Pytest conftest for the vast tooling suite — isolate tests from the LIVE fleet.

Why this file exists
--------------------
`herdd`'s control-plane helpers (`fleet_operator_intent`,
`fleet_watch_best_effort`, ... — all of them funnel through `fleet_request`)
talk to the fleetd daemon over a unix socket whose path defaults to
`fleet_state_dir()/fleetd.sock`, i.e. the *real* daemon on whatever workstation
the suite happens to run on. They are deliberately best-effort ("no daemon, no
problem"), which is exactly what made the leak invisible: nothing fails, the
request simply lands.

Measured on this box 2026-08-01: `journalctl --user -u vast-fleetd` carried 20
`operator_intent_destroy` events for box `"9"` and 40 `operator_intent_start`
events for box `"21"` — fixture ids from `test_guard.py` / the supervise tests,
not machines. Fake box `9` had also persisted into the daemon's `intents` map in
`~/.local/state/vast-fleetd/state.json`. `test_guard.py::_wire_guard`
monkeypatches `destroy_box`, `_emit_stopping_intent` and `_revoke_box_keys` —
but `_destroy_and_revoke` *also* calls `fleet_operator_intent`, which was left
live.

The blast radius was nil only because fixture ids (`9`, `21`, `inst-A`) cannot
collide with a real 8-digit vast instance id. That is luck, not isolation: an
operator intent is a real control-plane decision — `operator_intent_stop` marks
a watch dormant so the bid ladder will NOT rescue that box — so a fixture that
ever borrowed a realistic id could silently disarm supervision on a live,
billing machine.

The fix is one env var rather than more per-test monkeypatching, so it covers
every current and future test without each one having to remember. Pointing
`FLEETD_SOCK` at a path that cannot exist makes `fleet_request` return
`nodaemon:FileNotFoundError`, which is the designed and already-exercised
fallback path.

Tests that genuinely want a daemon (`test_fleetd.py` stands up a real socket
server) override the variable themselves; because this fixture is function
scoped and applied first, their assignment still wins.
"""

from __future__ import annotations

import atexit
import os
import shutil
import tempfile

import pytest

# --------------------------------------------------------------------------- #
# No test writes into the operator's real ~/.cache — set AT CONFTEST IMPORT.
# --------------------------------------------------------------------------- #
# Same class of leak as the fleetd socket below, same fix, found the same way —
# by measuring. `jobmeta.read_job`/`read_box` materialize event bodies under
# `$XDG_CACHE_HOME/vast-jobmeta/<job_id>/events/`, and `test_jobmeta.py`'s
# `FakeB2.copy` op is a REAL local write: it writes its in-memory store to
# whatever destination directory it is handed. Any `read_job(...)` in a test
# that does not pin `XDG_CACHE_HOME` (e.g. `test_jobmeta.py::_seed_failed`,
# called from eight tests, one of them a 32-iteration sampling loop) therefore
# drops fake events into the developer's real cache, with a fresh random nonce
# per run, so they accumulate forever.
#
# Measured on this box 2026-08-17: `~/.cache/vast-jobmeta/` held
# `20260713T120000-a1b2c3d4-score-a0/events/` with 49,816 fixture bodies /
# 195 MB — a job id that exists only in `test_jobmeta.py` and has ZERO objects
# on B2. Not inert: rclone scans the destination tree on every incremental
# copy, so the junk made a bulk `rclone copy` of the whole `jobs/` prefix take
# 68.6s against a 14.6s listing of the same keys. A fixture was taxing a live
# operator command.
#
# WHY THIS IS A MODULE-LEVEL ASSIGNMENT AND NOT AN AUTOUSE FIXTURE, which is
# the mistake this comment exists to stop the next person repeating. Several
# modules resolve their cache paths AT IMPORT — `boxes/reap.py`'s `_IDLE_LEDGER`
# and `_ZOMBIE_LEDGER` are module constants, and `test_vastlib_boxes_reap.py`
# asserts they equal the same formula recomputed at CALL time. A function-scoped
# `monkeypatch.setenv` runs after collection, i.e. after those constants are
# frozen, so it forks the two reads apart and turns a correct frozen-formula
# test red (measured: exactly one failure across 6,509 tests). Conftest import
# happens BEFORE test modules are imported, so setting it here keeps import-time
# and call-time readers agreeing — the same reasoning, and the same mechanism,
# as the repo-root conftest's `UPSTREAM_MONOREPO_REPO` mkdtemp.
#
# Session-wide rather than per-test on purpose: an incremental cache behaves
# like one across a run, and a test that wants tighter isolation pins its own
# `XDG_CACHE_HOME` or passes `cache_dir=` (several already do, and they still
# win — this only moves the DEFAULT).
_XDG_CACHE_SANDBOX = tempfile.mkdtemp(prefix="vast-tests-xdg-cache-")
os.environ["XDG_CACHE_HOME"] = _XDG_CACHE_SANDBOX
atexit.register(shutil.rmtree, _XDG_CACHE_SANDBOX, True)

# --------------------------------------------------------------------------- #
# No test reads or writes the operator's real host-reputation store.
# --------------------------------------------------------------------------- #
# `vastlib.market.hostrep` reorders `pick_offers` results by a DURABLE per-machine
# score persisted under the fleetd state dir. Left pointing at the real file, the
# developer's own condemnation history would decide the order a market fixture
# comes back in — so a test asserting "cheapest first" would pass on a clean
# workstation and fail on one that had rented a bad host that week. That is the
# hermeticity hazard the suite must not have, and it is invisible on CI.
#
# Sandboxed via the store's OWN env override rather than `FLEETD_STATE_DIR`,
# which several tests assert the default of. Module level for the same reason as
# XDG_CACHE_HOME above (conftest import precedes test-module import), though
# hostrep resolves its path per call and would tolerate a fixture.
_HOSTREP_SANDBOX = tempfile.mkdtemp(prefix="vast-tests-hostrep-")
os.environ["VAST_HOSTREP_PATH"] = os.path.join(_HOSTREP_SANDBOX,
                                               "host_reputation.json")
atexit.register(shutil.rmtree, _HOSTREP_SANDBOX, True)


@pytest.fixture(autouse=True)
def _isolate_host_reputation(monkeypatch, tmp_path):
    """...and a FRESH one per test, because the store accumulates.

    The module-level assignment above keeps the operator's real file out of the
    suite; this keeps tests out of each other's. `hostrep.note_strike` appends,
    so with one shared sandbox a supervise test that condemns fixture machine
    `7` leaves `7` blocked for every later test in the process — and fixture
    machine ids are small integers that collide constantly. Measured: five
    `_job_excluded_machines` tests and three eviction tests passed alone and
    failed in a full run, which is the worst failure shape to debug.

    Function-scoped is safe here where it was not for `XDG_CACHE_HOME`: hostrep
    resolves its path per call, never freezing it into a module constant.
    """
    monkeypatch.setenv("VAST_HOSTREP_PATH", str(tmp_path / "host_reputation.json"))
    import sys
    mod = sys.modules.get("vastlib.market.hostrep")
    if mod is not None:                    # drop the read cache keyed on the old path
        mod._cache.update({"path": None, "t": 0.0, "data": None})


@pytest.fixture(autouse=True)
def _clear_account_fault_latch():
    """`core.acctfault` latches an account-level API refusal in PROCESS state,
    and a live latch makes `hostrep.note_strike` a no-op. One test that feeds an
    `insufficient_credit` error through the HTTP funnel would otherwise silence
    every strike written for the next 15 minutes of the run — the same
    cross-test leak the hostrep sandbox above exists to stop, in a variable
    instead of a file. Cleared both sides, so neither direction leaks."""
    import sys
    mod = sys.modules.get("vastlib.core.acctfault")
    if mod is not None:
        mod.clear()
    yield
    if mod is not None:
        mod.clear()


@pytest.fixture(autouse=True)
def _isolate_fleetd_socket(monkeypatch, tmp_path):
    """Point every test's fleet client at a socket that does not exist.

    Autouse and function scoped: a test that wants a live socket (or wants to
    assert the *default* path, via `delenv`) overrides this from its own body.
    """
    monkeypatch.setenv("FLEETD_SOCK", str(tmp_path / "no-such-fleetd.sock"))


# High enough that the on-demand clamp never binds a unit fixture's 1.2x-floor
# bid — legacy expected prices (0.12 from floor 0.10, etc.) stay valid.
_TEST_MARKET_ONDEMAND = 5.0

# Every module that owns a `_market_ondemand_soft` module attribute the suite
# could reach — the same dual-module roster, for the same reason, as
# `_GUARDED_REQUEST_SOFT_MODULES` below. `herdd` is the historical copy;
# `vastlib.market.pricing` is the same probe after the vastlib port (plan §8
# step 3), and during the add-only phase BOTH are live and independently
# callable, so a fixture that knows about one of them is not a guard.
# Looked up in `sys.modules`, so an unimported module costs nothing.
# `test_vastlib_market.py::test_every_market_ondemand_guard_target_exists`
# asserts each entry still resolves to a callable.
_GUARDED_MARKET_ONDEMAND_MODULES = ("herdd", "vastlib.market.pricing")


@pytest.fixture(autouse=True)
def _isolate_market_ondemand(monkeypatch):
    """Stub the live on-demand market probe for every test that imported herdd.

    Since the doc 50 R1 family fix (2026-08-06), the launch and handoff paths
    read the on-demand clamp/filter reference from `_market_ondemand_soft`
    (the machine's live on-demand offers) instead of the BID offer row's own
    `dph_total` — which is the interruptible price, not on-demand. Left live,
    that probe walks up from CWD to the repo `.env`, finds a real API key, and
    unit tests silently query the real market (the same leak shape the fleetd
    socket isolation above exists to stop).

    Returns a constant 5.0 regardless of machine_id — the real function
    returns None for a missing machine_id, but unit fixtures routinely omit
    machine_id and the tests that exercise the None/unreadable-market path
    patch this to None themselves (their function-scoped patch wins).

    Wraps EVERY module in `_GUARDED_MARKET_ONDEMAND_MODULES`, not just
    `herdd`: since the vastlib port there are two live copies of the probe,
    and the `hasattr` degradation below (correct for an unimported module) is
    exactly what would make a RENAME silent — hence the meta-test named on that
    roster."""
    import sys
    for _modname in _GUARDED_MARKET_ONDEMAND_MODULES:
        mod = sys.modules.get(_modname)
        if mod is None or not hasattr(mod, "_market_ondemand_soft"):
            continue
        monkeypatch.setattr(mod, "_market_ondemand_soft",
                            lambda mid, num_gpus=None: _TEST_MARKET_ONDEMAND)


@pytest.fixture(autouse=True)
def _pin_process_environment():
    """Hand every test back the `os.environ` it was given.

    MEASURED 2026-08-16, and the reason this is suite-wide rather than per-file:
    any test that builds the real CLI parser runs `load_env()` (flat
    `herdd.main()` or `vastlib.cli.main.build_parser()`), which `setdefault`s
    the developer's `.env` — B2 master key, bucket, endpoint — straight into the
    process environment, where it OUTLIVES the test. `monkeypatch` cannot undo
    it: the mutation is the code's, not the test's.

    That leak is not inert. `test_supervise.py::test_spec_roundtrip_write_capture_relaunch_body`
    asserts the local box-scoped credential branch, which is only taken when no
    master creds are visible — with a leaked `.env` it took the mint branch
    instead and MINTED AND REVOKED A REAL B2 KEY, twice observed, from a unit
    test run. `test_vastlib_cli_workflow.py::flat_group` (drives `herdd.main()`
    to capture its parser) and `test_vastlib_cli_launch.py` (drives
    `build_parser()`) are the two leakers found; a pin here covers the ones
    nobody has written yet, and `_block_real_b2_mint` below is the fail-closed
    backstop for when a leak slips through anyway.

    Restore is unconditional-but-cheap: compare first, rewrite only on drift, so
    the common case is one dict build per test.
    """
    import os
    before = dict(os.environ)
    yield
    if dict(os.environ) != before:
        os.environ.clear()
        os.environ.update(before)


@pytest.fixture(autouse=True)
def _rclone_config_scratch(tmp_path_factory, monkeypatch):
    """Every test in this suite writes rclone config to its OWN scratch path.

    This is the PREVENTION the detector below could not be. When that detector
    was written the shipped shell writers hardcoded `$HOME`, so `RCLONE_CONFIG`
    was not a lever for the class and only `HOME` was — and redirecting `HOME`
    suite-wide changes what a hundred unrelated tests resolve. `8d8cb8c37`
    (2026-08-24) made all fifteen writers resolve
    `${RCLONE_CONFIG:-$HOME/.config/rclone/rclone.conf}`, which turns the cheap
    lever on: one env var, read by rclone itself and by every writer, touching
    nothing else a test depends on.

    It covers the exact leak shape that caused the 2026-08-22 clobber — a test
    building its child env as `dict(os.environ)` and forwarding the real `$HOME`
    — because the scratch path rides along in that same dict. A test that builds
    a minimal env by hand (`test_b2_conf_guard.py::_env`) deliberately drops it
    and exercises the guard's own predicate instead; that is the point of that
    file, so this fixture must not be the only line of defence.
    """
    scratch = tmp_path_factory.mktemp("rclone-conf") / "rclone.conf"
    monkeypatch.setenv("RCLONE_CONFIG", str(scratch))


def _rclone_conf_damage(text):
    """Reason string if `text` is a BROKEN operator config, else None.

    The endpoint half mirrors `b2_sync.sh::b2_reserved_host_reason` — a
    fixture's placeholder host is an RFC 6761/2606 reserved name, so it can
    never resolve — plus the two degenerate shapes text alone can show: the
    file emptied, and the `[b2]` stanza gone. Pure text on purpose: rclone need
    not be installed and the endpoint need not resolve.
    """
    if text is None:
        return "the file is gone"
    body = text.decode("utf-8", "replace") if isinstance(text, bytes) else text
    if not body.strip():
        return "the file is empty"
    stanza, endpoints = None, {}
    for line in body.splitlines():
        s = line.strip()
        if s.startswith("[") and s.endswith("]"):
            stanza = s[1:-1]
        elif stanza and "=" in s and s.split("=", 1)[0].strip() == "endpoint":
            endpoints.setdefault(stanza, s.split("=", 1)[1].strip())
    if "b2" not in endpoints:
        return "the [b2] remote is gone"
    for remote, ep in sorted(endpoints.items()):
        host = ep.split("://", 1)[-1].split("/", 1)[0].split(":", 1)[0].lower()
        if (host in _RESERVED_HOSTS
                or host.endswith(_RESERVED_HOST_SUFFIXES)
                or host in _RESERVED_DOC_DOMAINS
                or host.endswith(tuple("." + d for d in _RESERVED_DOC_DOMAINS))):
            return f"[{remote}] endpoint host {host!r} is a reserved test name"
    return None


# RFC 6761 (.invalid/.test/.example/.localhost) and RFC 2606 (example.com/net/org).
_RESERVED_HOSTS = frozenset({"invalid", "test", "example", "localhost"})
_RESERVED_HOST_SUFFIXES = (".invalid", ".test", ".example", ".localhost")
_RESERVED_DOC_DOMAINS = ("example.com", "example.net", "example.org")


@pytest.fixture(autouse=True)
def _protect_operator_rclone_conf():
    """No test rewrites the OPERATOR's `~/.config/rclone/rclone.conf`.

    MEASURED twice, 2026-08-22 12:20 and 21:53: the live `[b2]` remote came back
    as `endpoint = https://example.invalid`, `access_key_id = fake` — the
    `_FAKE_B2` fixture values from `test_serve_attach_model_precedence.py`. The
    other three remotes (`r2tc`, `hp-b2`, `b2eu`) are untouched, so
    `rclone listremotes` looks perfectly healthy and nothing fails until a real
    transfer runs. Blast radius is the whole workstation: `b2:` is the remote the
    jobd bootstrap bundle stages through, so every `herdd launch --jobs` fails
    box-wide until someone repairs it by hand.

    The route is a shipped ON-BOX writer reached from a test. `serve_vllm.sh`
    (and `b2_sync.sh config`, `jobd_boot.sh`, `train.sh`, …) write
    `$HOME/.config/rclone/rclone.conf`.

    CORRECTION 2026-08-24: the paragraph here used to say the shell writers
    hardcode `$HOME` and do NOT honour `RCLONE_CONFIG`, so only `HOME` was a
    lever for the class and this had to be a DETECTOR rather than a redirect.
    That was true when written and is now false — `8d8cb8c37` made all fifteen
    writers honour `RCLONE_CONFIG`, and `_rclone_config_scratch` above is the
    redirect that premise ruled out.

    This stays as the backstop, because the redirect is not universal: a test
    that builds a minimal child env by hand drops `RCLONE_CONFIG` on the floor,
    and a writer that regresses to a hardcoded `$HOME` would silently leave the
    redirect pointing at a file nobody writes. What belongs here is the thing
    that turns a silent machine-breaking side effect into a red test — and,
    because a failed assertion after the fact still leaves the operator broken,
    it RESTORES the file before failing.

    IT FIRES ON DAMAGE, NOT ON CHANGE — and that distinction is the whole
    correctness of the fixture, because THIS PROCESS IS NOT THE ONLY WRITER.
    The path is machine-global: a peer session's suite, a real `b2_sync.sh
    config`, or fleetd can rewrite it at any instant. A bare before/after
    comparison cannot tell whose write it saw, so it charged the change to
    whichever test happened to be running. MEASURED 2026-08-27 07:28, with a
    peer's tools lane running concurrently: 323 teardown failures across 86
    files, none of which touch rclone —
    `test_notify_policy.py::test_no_row_can_emit_a_price_the_rails_would_not`
    (3.3 s of pure bid arithmetic, opens no file) among them. One external
    write produces exactly one such failure, so that run saw many; what wrote
    it that often is still unidentified, and this fixture is the wrong place
    to find out.

    Restoring on any change was the worse half: a healthy external rewrite is
    somebody's legitimate work, and reverting it under them is damage this
    fixture CAUSES. Two concurrent suites could each revert the other.

    So the predicate is `_rclone_conf_damage`, not `!=`: a config that still
    carries a `[b2]` remote pointing at a resolvable host is somebody else's
    business, and the next test re-baselines on it silently. Reserved-name
    poison, an emptied file or a vanished `[b2]` is damage no legitimate writer
    produces — restore it and fail, whoever wrote it.

    The gap this leaves, deliberately: a test that overwrites the operator's
    keys with plausible-looking ones is invisible here. That case belongs to
    `_rclone_config_scratch` (redirect) and `b2_guard_live_rclone_config`
    (refusal under `PYTEST_CURRENT_TEST`), which both act BEFORE the write. A
    detector that cannot attribute must not adjudicate a write it cannot prove
    is ours.
    """
    import os
    rc = os.path.expanduser("~/.config/rclone/rclone.conf")
    try:
        before = open(rc, "rb").read()
    except OSError:
        before = None            # absent (CI); nothing of the operator's to guard
    yield
    if before is None:
        return
    try:
        after = open(rc, "rb").read()
    except OSError:
        after = None
    if after == before:
        return
    damage = _rclone_conf_damage(after)
    if damage is None:
        return              # healthy rewrite by another process — not ours to undo
    with open(rc, "wb") as fh:          # repair FIRST — a red test must not
        fh.write(before)                # leave the workstation broken
    os.chmod(rc, 0o600)
    raise AssertionError(
        f"the operator's {rc} was DAMAGED during this test and has been "
        f"restored: {damage}. Most likely a shipped writer reached the real "
        "$HOME — give the subprocess a scratch HOME or set RCLONE_CONFIG. "
        "This fixture cannot prove the write was this test's (the path is "
        "machine-global), but the damage is real either way."
    )


@pytest.fixture(autouse=True)
def _block_real_b2_mint(monkeypatch):
    """No test mints, revokes or lists a REAL B2 application key.

    `b2_mint_key._http` is the single JSON funnel every mint path goes through,
    and its own docstring already calls it "seam for tests" — so refusing there
    covers `mint`, `mint_pair`, `mint_publish`, `revoke_by_name` and
    `delete_key` at once, without shadowing any unit under test (a test that
    exercises those functions stubs `_http` or a higher name itself, and its
    monkeypatch runs after this one).

    Fail-closed with `MintError`, the exception every caller in `launch/spec.py`
    is already written to survive, rather than a silent no-op: a mint that
    quietly returns fake creds would put unusable keys in a launch body and the
    test would assert on them happily.

    Added with the `.env` pin above, after a unit-test run was found minting and
    revoking real keys named `run-r1` on the developer's account.
    """
    import sys
    mod = sys.modules.get("b2_mint_key")
    if mod is None or not hasattr(mod, "_http"):
        return

    def _refuse(url, body=None, headers=None):
        raise mod.MintError(
            "test isolation: real B2 key API blocked (conftest "
            "_block_real_b2_mint) — stub `b2_mint_key._http` (or the mint "
            "helper you are testing) in your test")
    monkeypatch.setattr(mod, "_http", _refuse)


_CROSS_RING_SEAMS = (
    ("vastlib.launch.launch", "compose_jobs_launch_env"),
    ("vastlib.launch.launch", "fleet_watch_best_effort"),
    ("vastlib.boxes.lifecycle", "fleet_operator_intent"),
    ("vastlib.boxes.lifecycle", "fleet_note_operator_stop"),
    ("vastlib.boxes.lifecycle", "cmd_job_attach"),
)


@pytest.fixture(autouse=True)
def _restore_cross_ring_seam_bindings(monkeypatch):
    """Hand every test back the seam census it was given.

    `vastlib/cli/_compose.py::bind()` points three names that a LOWER ring calls
    at the HIGHER ring that defines them (`compose_jobs_launch_env` ->
    `jobs.bundle`, `fleet_watch_best_effort` / `fleet_operator_intent` ->
    `fleet.client`). Binding is a module-attribute assignment, so it is
    PROCESS-GLOBAL and outlives the test that triggered it — and it is triggered
    by simply running a command: `cli.main.main()` binds, and so do
    `cli/launch.py`, `cli/train.py` and `cli/supervise.py` at the top of their
    `run()`. `test_vastlib_cli_main.py`, `test_lifecycle.py` and
    `test_job_submit_preflight.py` all do exactly that today.

    Without this fixture, `test_vastlib_launch.py`'s seam census (those two
    names must still RAISE, because `launch` may not import `jobs`/`fleet`)
    passes or fails on pytest's collection ORDER — measured: green alone, four
    failures behind `test_vastlib_cli_main.py`. A test whose verdict depends on
    which file was collected first is not a test.

    Same shape as the three fixtures above and for the same reason: one
    sys.modules-keyed roster instead of per-file cleanup, so a command test
    added later cannot forget. `monkeypatch.setattr(mod, name, <current value>)`
    is the snapshot — it writes back what is already there and monkeypatch's
    teardown restores that same value, so a bind performed DURING the test is
    undone and a bind that somehow happened at IMPORT time is still visible to
    the census (it is what gets snapshotted).

    `test_vastlib_cli_launch.py::test_the_conftest_seam_roster_matches_compose`
    pins this roster to `_compose.SEAM_BINDINGS`, so a seam added there without
    a line here is a failure rather than a silent hole.

    One degradation, shared with the two fixtures above and harmless for the
    same reason: a module not yet in `sys.modules` is skipped, so a test that is
    itself the first to import `vastlib.launch.launch` AND binds inside the same
    test leaks. Test modules are imported at collection, before any test runs,
    so the census file's own imports have always happened by then.
    """
    import sys
    for _modname, _attr in _CROSS_RING_SEAMS:
        mod = sys.modules.get(_modname)
        if mod is None or not hasattr(mod, _attr):
            continue
        monkeypatch.setattr(mod, _attr, getattr(mod, _attr))


_READ_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})

# Every module that owns a `request_soft` module attribute the suite could reach.
# `herdd` is the historical one; `vastlib.core.api` is the same function after
# the vastlib port (plan §8 step 2) — during the add-only phase BOTH are live and
# independently callable, so both must be wrapped or the newer one is an
# unguarded hole straight to the live API. Kept as a tuple of NAMES, looked up in
# sys.modules, so an unimported module is simply skipped (a test that never
# imports vastlib pays nothing).
# `test_vastlib_core_api.py::test_every_guard_target_exists` asserts each entry
# still resolves to a callable — a rename that silently un-guards a target is
# exactly the vacuous-guard failure this whole file exists to prevent.
_GUARDED_REQUEST_SOFT_MODULES = ("herdd", "vastlib.core.api")


def _guarded_request_soft(real):
    """Wrap one module's `request_soft`: reads pass through, mutations refuse."""
    def guarded(method, path, body=None, *a, **k):
        if str(method).upper() in _READ_METHODS:
            return real(method, path, body, *a, **k)
        return (False, None,
                f"test isolation: {method} {path} blocked (conftest "
                f"_block_mutating_api_calls) — stub `request_soft` in your "
                f"test to assert on the call")
    return guarded


@pytest.fixture(autouse=True)
def _block_mutating_api_calls(monkeypatch):
    """Refuse every state-changing vast API call for the whole suite.

    Same leak shape, and the same fix, as the two fixtures above. `request_soft`
    walks up from CWD to the repo `.env`, finds a real API key, and issues the
    request against whatever id the fixture used. The blast radius has only ever
    been nil because fixture ids (`41`, `701`, `9000`) cannot collide with a
    real 8-digit instance — luck, not isolation.

    Added 2026-08-16 with the retention QUIESCE (`herdd._job_quiesce_box`),
    which put a `stop` + a bid pin on the eviction-replacement path that a dozen
    portable tests drive end to end. Before this fixture that path would have
    PUT `{"state": "stopped"}` and `{"price": 0.001}` at the live API on every
    one of those runs.

    THE LAYER IS DELIBERATE. The guard wraps `request_soft` — the single seam
    every mutating helper funnels through — and NOT the helpers themselves.
    Wrapping `_put_state_soft`/`_put_bid_soft` was tried first and broke twelve
    tests that exercise those helpers *correctly*, over their own `request_soft`
    stub: the guard was shadowing the very unit under test. At this layer a test
    that stubs `request_soft` replaces the guard outright (that is the isolation
    it already has), and a test that stubs nothing is the one that would have
    reached the network. Refusing reads would be wrong for the same reason — a
    GET cannot change the fleet, and several tests legitimately probe the
    unreachable-API path.

    Returns the same `(False, None, err)` shape a missing API key produces,
    which every `request_soft` caller is contractually required to survive.

    Wraps EVERY module in `_GUARDED_REQUEST_SOFT_MODULES`, not just `herdd`:
    since the vastlib port there are two live copies of the funnel, and a guard
    that knows about one of them is not a guard.
    """
    import sys
    for _modname in _GUARDED_REQUEST_SOFT_MODULES:
        mod = sys.modules.get(_modname)
        if mod is None or not hasattr(mod, "request_soft"):
            continue
        monkeypatch.setattr(mod, "request_soft",
                            _guarded_request_soft(mod.request_soft))

"""`vastlib.supervise.journal` — the four ported journal leaves, held to the wire.

Why this file exists
--------------------
`journal.py` is the first symbol move of the vastlib port (plan §8 step 2), and
what it moved is a **wire contract**: every record `_sup_emit` and
`_job_handoff_emit` produce becomes one immutable object in the B2 append-only
log (`runs/<RUN_ID>/events/`, `jobs/nodes/<IID>/events/`), read back by
`herdd runs`, the dashboard and the fleetd watch tables. Plan §4 lists "B2
event schemas" as a frozen contract; §7.4 forbids expectation changes. So the
port needs a test that looks at the actual bytes rather than at a mock's call
count.

The suite's existing coverage of these four is real but indirect: 53
`monkeypatch.setattr(herdd, ...)` sites REPLACE the emitters, and
`test_supervise.py` patches one level DOWN (`runmeta.emit_event`,
`jobmeta.emit_box_event`) and asserts on the captured `(event, fields)`. Both
styles are blind to the envelope, the object key and the serialization — and
`_iso_z` had **zero** coverage anywhere in `tools/vast` despite `fleetd.py`
calling it. This file closes exactly that gap.

Where the stub goes, and why there
----------------------------------
The transport seam is NOT re-implemented by the port: `journal._sup_emit` calls
`runmeta.emit_event`, which builds the envelope, derives the key and shells
`rclone rcat b2:<bucket>/<key>` through its module-level `_default_runner`. To
see the bytes, the stub has to sit BELOW that — so these tests rebind the
`subprocess` name inside the `runmeta` / `jobmeta` module namespaces. That is
the lowest point at which the whole real path (envelope, key, `json.dumps`
separators, trailing newline, rc handling) still executes.

`runner=` cannot be used instead: `emit_event`'s `runner` default is bound at
def time, so patching `runmeta._default_runner` steers nothing, and the ported
wrapper does not (and must not) grow a `runner` parameter it never had.

**No test here can reach the network or B2 under any circumstance.** The stub
is installed before every emitting call; `.env` is absent from this worktree by
design, but nothing in this file relies on that — an unstubbed path would
attempt `rclone`, so the stub is the guard, not the missing credential. The
FakeRclone also asserts its own argv starts with `rclone`, which is how a
future refactor that swapped transports would be caught rather than silently
tolerated.

What is deliberately NOT here
-----------------------------
* No re-testing of `runmeta`/`jobmeta` themselves (`test_runmeta.py`,
  `test_jobmeta.py` own those, and Zone S is untouched by this refactor). The
  assertions below are about what the WRAPPERS hand them and hand back.
* No repointing of any existing test. Per the manifest, all 53 patch sites are
  deferred-seam: they patch `herdd.<name>` to steer callers that still live
  in `herdd.py`, and they migrate with those callers in plan step 4.
* No assertion on `ts` / `nonce` values. They are non-deterministic by design
  (millisecond clock, `os.urandom`) and form the object key; only their SHAPE
  and their presence in the key are contract.

Provenance: created 2026-08-16 alongside `vastlib/supervise/journal.py`,
plan §8 step 2.
"""

from __future__ import annotations

import datetime
import json
import re
import sys
from pathlib import Path

import pytest

VAST_DIR = Path(__file__).resolve().parent
if str(VAST_DIR) not in sys.path:                      # entry-script sys.path shape
    sys.path.insert(0, str(VAST_DIR))

import jobmeta                                          # noqa: E402  Zone S
import runmeta                                          # noqa: E402  Zone S

from vastlib.supervise import journal                   # noqa: E402


# --------------------------------------------------------------------------- #
# the transport stub
# --------------------------------------------------------------------------- #
class _Completed:
    """Stand-in for `subprocess.CompletedProcess` — the three fields the
    runners in runmeta/jobmeta actually read."""

    def __init__(self, returncode: int, stdout: str = "", stderr: str = ""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class _FakeSubprocess:
    """A `subprocess` module stand-in that records the rclone invocation.

    Rebound onto the `runmeta` / `jobmeta` module globals, so the real
    `_default_runner` body executes and everything above it (envelope, key,
    body serialization, rc branch) is exercised for real."""

    def __init__(self, returncode: int = 0, stderr: str = ""):
        self.returncode = returncode
        self.stderr = stderr
        self.calls: list[tuple[list[str], str | None]] = []

    def run(self, argv, capture_output=False, text=False, input=None, **kw):  # noqa: ANN001,ANN003,ANN201,A002
        assert argv and argv[0] == "rclone", (
            f"the journal transport is no longer rclone: argv={argv!r}")
        self.calls.append((list(argv), input))
        return _Completed(self.returncode, "", self.stderr)

    # --- readers -----------------------------------------------------------
    @property
    def argv(self) -> list[str]:
        assert len(self.calls) == 1, f"expected exactly one emit, got {len(self.calls)}"
        return self.calls[0][0]

    @property
    def body(self) -> str:
        assert len(self.calls) == 1, f"expected exactly one emit, got {len(self.calls)}"
        payload = self.calls[0][1]
        assert payload is not None, "rclone rcat was called with no stdin body"
        return payload

    @property
    def remote(self) -> str:
        """The `b2:<bucket>/<key>` argument of `rclone rcat <remote>`."""
        argv = self.argv
        assert argv[1] == "rcat", f"expected `rclone rcat ...`, got {argv!r}"
        return argv[2]

    @property
    def key(self) -> str:
        remote = self.remote
        _, _, rest = remote.partition(":")
        bucket, _, key = rest.partition("/")
        assert bucket, f"no bucket in remote {remote!r}"
        return key


@pytest.fixture
def rclone_run(monkeypatch) -> _FakeSubprocess:                     # noqa: ANN001
    """Stub the run-lane (runmeta) rclone seam. B2_BUCKET set to a fixture value."""
    fake = _FakeSubprocess()
    monkeypatch.setattr(runmeta, "subprocess", fake)
    monkeypatch.setenv("B2_BUCKET", "test-bucket")
    return fake


@pytest.fixture
def rclone_box(monkeypatch) -> _FakeSubprocess:                    # noqa: ANN001
    """Stub the jobs-lane rclone seam — which is RUNMETA's, not jobmeta's.

    Measured while writing this file: `jobmeta` does
    `from runmeta import ... _default_runner, _bucket ...`, so a box event's
    transport and its bucket read both execute inside the `runmeta` module
    namespace. Patching `jobmeta.subprocess` catches nothing and the emit goes
    to the real rclone. Recorded here because it is the exact shape of an
    isolation stub that looks installed and is not."""
    fake = _FakeSubprocess()
    monkeypatch.setattr(runmeta, "subprocess", fake)
    monkeypatch.setenv("B2_BUCKET", "test-bucket")
    monkeypatch.delenv("B2_WRITE_KEY_ID", raising=False)
    return fake


# `20260816T142530123Z` — fixed width, no colons, millisecond precision.
_TS_RE = re.compile(r"^\d{8}T\d{6}\d{3}Z$")
_NONCE_RE = re.compile(r"^[0-9a-f]{12}$")


# --------------------------------------------------------------------------- #
# _sup_emit — the run-lane envelope, key and byte shape
# --------------------------------------------------------------------------- #
def test_sup_emit_envelope_key_set(rclone_run: _FakeSubprocess) -> None:
    """The v1 envelope: exactly six framing keys, actor pinned to `supervisor`,
    plus the caller's fields. `_key`/`_emitted` are the RETURN annotation and
    are not part of the persisted record."""
    rec = journal._sup_emit("run-abc", "heartbeat", box=41, dph=0.44)

    on_wire = json.loads(rclone_run.body)
    assert set(on_wire) == {"v", "ts", "actor", "event", "run_id", "nonce",
                            "box", "dph"}
    assert on_wire["v"] == runmeta.SCHEMA_VERSION == 1
    assert on_wire["actor"] == "supervisor"
    assert on_wire["event"] == "heartbeat"
    assert on_wire["run_id"] == "run-abc"
    assert on_wire["box"] == 41 and on_wire["dph"] == 0.44
    assert _TS_RE.match(on_wire["ts"]), on_wire["ts"]
    assert _NONCE_RE.match(on_wire["nonce"]), on_wire["nonce"]

    # what the caller gets back: the event plus the two transport fields
    assert rec["_emitted"] is True
    assert set(rec) == set(on_wire) | {"_key", "_emitted"}


def test_sup_emit_key_format(rclone_run: _FakeSubprocess) -> None:
    """`runs/<run_id>/events/<ts>-<actor_slug>-<nonce>.json`, written to
    `b2:<bucket>/<key>`. The key is the whole concurrency story — one immutable
    object per event, lexicographically sortable by ts."""
    rec = journal._sup_emit("run-abc", "supervised")
    on_wire = json.loads(rclone_run.body)

    assert rclone_run.remote == f"b2:test-bucket/{rec['_key']}"
    assert rclone_run.key == rec["_key"]
    assert rec["_key"] == (
        f"runs/run-abc/events/{on_wire['ts']}-supervisor-{on_wire['nonce']}.json")


def test_sup_emit_body_is_compact_json_plus_newline(rclone_run: _FakeSubprocess) -> None:
    """`json.dumps(ev, separators=(",", ":")) + "\\n"` — no spaces, one newline.

    Byte-level because the object is immutable once written: a whitespace change
    would silently fork the format for every reader of the historical log."""
    journal._sup_emit("run-abc", "cost", cost_usd=1.25, hours=2.5)
    body = rclone_run.body

    assert body.endswith("\n") and body.count("\n") == 1
    assert ", " not in body and ": " not in body
    assert body == json.dumps(json.loads(body), separators=(",", ":")) + "\n"


def test_sup_emit_drops_none_valued_fields(rclone_run: _FakeSubprocess) -> None:
    """Optional fields stay ABSENT, never null — the fold distinguishes the two."""
    journal._sup_emit("run-abc", "relaunched", new_box=99, reason=None,
                      old_price=None, price=0.31)

    on_wire = json.loads(rclone_run.body)
    assert "reason" not in on_wire and "old_price" not in on_wire
    assert on_wire["new_box"] == 99 and on_wire["price"] == 0.31
    assert None not in on_wire.values()


def test_sup_emit_swallow_record_has_no_key(monkeypatch) -> None:              # noqa: ANN001
    """The wrapper's OWN failure record is exactly `{_emitted: False, _error}`.

    Deliberately different from runmeta's rc!=0 record (next test), which
    carries the full event and `_key`. Callers only read `_emitted`, but both
    shapes are reproduced verbatim by the port. This is also the path that keeps
    an unpatched run-lane test off the network: with no B2_BUCKET,
    `runmeta._bucket` raises and the swallow absorbs it."""
    monkeypatch.delenv("B2_BUCKET", raising=False)
    # belt and braces: even if _bucket somehow resolved, no rclone can run
    exploding = _FakeSubprocess()
    monkeypatch.setattr(runmeta, "subprocess", exploding)

    rec = journal._sup_emit("run-abc", "heartbeat", box=41)

    assert rec == {"_emitted": False, "_error": rec["_error"]}
    assert set(rec) == {"_emitted", "_error"}
    assert "B2_BUCKET" in rec["_error"]
    assert exploding.calls == []


def test_sup_emit_transport_failure_keeps_the_event(monkeypatch) -> None:      # noqa: ANN001
    """rc!=0 comes back through runmeta: full event + `_key` + `_error`.

    The asymmetry with the previous test is the contract, not an accident."""
    monkeypatch.setenv("B2_BUCKET", "test-bucket")
    monkeypatch.setattr(runmeta, "subprocess",
                        _FakeSubprocess(returncode=1, stderr="  b2: 403 forbidden\n"))

    rec = journal._sup_emit("run-abc", "evicted", box=41)

    assert rec["_emitted"] is False
    assert rec["_error"] == "b2: 403 forbidden"          # stripped by runmeta
    assert rec["_key"].startswith("runs/run-abc/events/")
    assert rec["event"] == "evicted" and rec["actor"] == "supervisor"


# --------------------------------------------------------------------------- #
# _job_handoff_emit — the jobs lane keys on the BOX, not the run
# --------------------------------------------------------------------------- #
def test_job_handoff_emit_envelope_key_set(rclone_box: _FakeSubprocess) -> None:
    """Same v1 framing, `instance_id` in place of `run_id`, actor
    `job-supervise`. `iid` is stringified — box ids arrive as ints."""
    jctx = {"iid": 47219058}
    rec = journal._job_handoff_emit(jctx, "handoff_armed", to_box=47219872,
                                    reason=None)

    on_wire = json.loads(rclone_box.body)
    assert set(on_wire) == {"v", "ts", "actor", "event", "instance_id", "nonce",
                            "to_box"}
    assert on_wire["actor"] == "job-supervise"
    assert on_wire["instance_id"] == "47219058"          # str, not int
    assert on_wire["event"] == "handoff_armed"
    assert "reason" not in on_wire                       # None dropped here too
    assert rec["_emitted"] is True


def test_job_handoff_emit_key_format(rclone_box: _FakeSubprocess) -> None:
    """`jobs/nodes/<IID>/events/<ts>-<actor_slug>-<nonce>.json` on `b2:`.

    The slug is filename-safe, so the actor `job-supervise` appears in the KEY
    as `job_supervise` (runmeta._actor_slug: anything outside `[A-Za-z0-9._]`
    becomes `_`) while the record's own `actor` field keeps the hyphen. The run
    lane's `supervisor` has no such character and is unchanged — which is why
    this only shows up on the jobs lane."""
    rec = journal._job_handoff_emit({"iid": "47219058"}, "box_evicted")
    on_wire = json.loads(rclone_box.body)

    assert on_wire["actor"] == "job-supervise"
    assert rec["_key"] == (
        "jobs/nodes/47219058/events/"
        f"{on_wire['ts']}-job_supervise-{on_wire['nonce']}.json")
    assert rclone_box.remote == f"b2:test-bucket/{rec['_key']}"
    assert rclone_box.body.endswith("\n")
    assert ", " not in rclone_box.body and ": " not in rclone_box.body


def test_job_handoff_emit_uses_the_split_write_key_when_present(
        monkeypatch, rclone_box: _FakeSubprocess) -> None:                     # noqa: ANN001
    """`B2_WRITE_KEY_ID` set -> the scoped `b2w:` remote (jobmeta._wq).

    The env read lives inside jobmeta, not in the wrapper; asserted here so a
    future "tidy-up" that hoists it into vastlib is caught."""
    monkeypatch.setenv("B2_WRITE_KEY_ID", "0055deadbeef")
    journal._job_handoff_emit({"iid": 41}, "pull_stalled")
    assert rclone_box.remote.startswith("b2w:test-bucket/jobs/nodes/41/events/")


def test_job_handoff_emit_swallow_record_has_no_key(monkeypatch) -> None:      # noqa: ANN001
    """Same swallow shape as the run lane: `{_emitted: False, _error}`, no `_key`."""
    monkeypatch.delenv("B2_BUCKET", raising=False)
    fake = _FakeSubprocess()
    monkeypatch.setattr(runmeta, "subprocess", fake)     # see the rclone_box fixture

    rec = journal._job_handoff_emit({"iid": 41}, "handoff_cutover")

    assert set(rec) == {"_emitted", "_error"}
    assert rec["_emitted"] is False
    assert fake.calls == []


# --------------------------------------------------------------------------- #
# the two in-memory queues — no I/O, aliasing preserved
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "fn,slot",
    [(journal._job_ladder_journal, "ladder_journal"),
     (journal._job_handoff_journal, "handoff_journal")],
)
def test_journal_queues_append_a_name_fields_tuple(fn, slot: str) -> None:     # noqa: ANN001
    """`(name, fields)` — the pair fleetd's drain unpacks. Nothing is emitted."""
    jctx: dict[str, object] = {"iid": 41}
    fn(jctx, "jobs_box_launched", iid=47219872, price=0.31)

    assert jctx[slot] == [("jobs_box_launched", {"iid": 47219872, "price": 0.31})]
    # the OTHER queue is untouched — the two slots are independent by design
    other = "handoff_journal" if slot == "ladder_journal" else "ladder_journal"
    assert other not in jctx


@pytest.mark.parametrize(
    "fn,slot",
    [(journal._job_ladder_journal, "ladder_journal"),
     (journal._job_handoff_journal, "handoff_journal")],
)
def test_journal_queues_truncate_in_place(fn, slot: str) -> None:              # noqa: ANN001
    """`del q[:-MAX]` mutates the SAME list object.

    Load-bearing: the caller's jctx and fleetd's drain alias one list. A
    `q = q[-MAX:]` rebind would leave the drain reading an orphan."""
    jctx: dict[str, object] = {}
    fn(jctx, "seed")
    queue = jctx[slot]
    assert isinstance(queue, list)

    for i in range(journal.JOB_HANDOFF_JOURNAL_MAX + 50):
        fn(jctx, f"e{i}")

    assert jctx[slot] is queue                            # same object, still
    assert len(queue) == journal.JOB_HANDOFF_JOURNAL_MAX == 200
    # oldest fall off the front; the newest is last
    assert queue[-1][0] == f"e{journal.JOB_HANDOFF_JOURNAL_MAX + 49}"
    assert "seed" not in [name for name, _ in queue]


@pytest.mark.parametrize(
    "fn,slot",
    [(journal._job_ladder_journal, "ladder_journal"),
     (journal._job_handoff_journal, "handoff_journal")],
)
def test_journal_queues_below_the_cap_drop_nothing(fn, slot: str) -> None:     # noqa: ANN001
    """`del q[:-200]` on a shorter list deletes NOTHING — the negative slice is
    correct as written and is the reason the cap is invisible to fleetd."""
    jctx: dict[str, object] = {}
    for i in range(5):
        fn(jctx, f"e{i}", n=i)
    assert [name for name, _ in jctx[slot]] == ["e0", "e1", "e2", "e3", "e4"]


def test_journal_queues_never_touch_the_transport(monkeypatch) -> None:        # noqa: ANN001
    """Neither queue function emits: no rclone, no B2, no bucket read."""
    run_fake, box_fake = _FakeSubprocess(), _FakeSubprocess()
    monkeypatch.setattr(runmeta, "subprocess", run_fake)
    monkeypatch.setattr(jobmeta, "subprocess", box_fake)
    monkeypatch.delenv("B2_BUCKET", raising=False)

    jctx: dict[str, object] = {"iid": 41}
    journal._job_ladder_journal(jctx, "jobs_box_destroyed", iid=99)
    journal._job_handoff_journal(jctx, "armed", to_box=99)

    assert run_fake.calls == [] and box_fake.calls == []


# --------------------------------------------------------------------------- #
# _iso_z — zero prior coverage anywhere in tools/vast, and fleetd calls it
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("falsy", [None, 0, 0.0, ""])
def test_iso_z_falsy_passes_through(falsy) -> None:                            # noqa: ANN001
    """None/0/"" -> None. Journal fields are dropped when None (see the emit
    tests above), so a missing deadline stays absent rather than rendering as
    the epoch."""
    assert journal._iso_z(falsy) is None


def test_iso_z_format_is_extended_iso_utc() -> None:
    """`%Y-%m-%dT%H:%M:%SZ`, always UTC — a DIFFERENT format from the event
    envelope's `runmeta.now_ts()` compact basic-ISO. Both are contract."""
    ts = datetime.datetime(2026, 8, 5, 18, 30, 0,
                           tzinfo=datetime.timezone.utc).timestamp()
    assert journal._iso_z(ts) == "2026-08-05T18:30:00Z"
    # sub-second input truncates, it does not round
    assert journal._iso_z(ts + 0.999) == "2026-08-05T18:30:00Z"
    # and it is emphatically not the envelope's shape
    assert not _TS_RE.match(journal._iso_z(ts) or "")


def test_iso_z_ignores_the_local_timezone(monkeypatch) -> None:                # noqa: ANN001
    """`fromtimestamp(ts, timezone.utc)` — the tz argument is what makes this
    machine-independent. fleetd renders retention deadlines with it."""
    import time as _time
    monkeypatch.setenv("TZ", "America/Los_Angeles")
    if hasattr(_time, "tzset"):
        _time.tzset()
    try:
        ts = datetime.datetime(2026, 1, 2, 3, 4, 5,
                               tzinfo=datetime.timezone.utc).timestamp()
        assert journal._iso_z(ts) == "2026-01-02T03:04:05Z"
    finally:
        monkeypatch.undo()
        if hasattr(_time, "tzset"):
            _time.tzset()


# --------------------------------------------------------------------------- #
# parity with the herdd copies (step 2 is ADD-ONLY: both exist right now)
# --------------------------------------------------------------------------- #
def _herdd():                                                                # noqa: ANN202
    return pytest.importorskip("herdd")


@pytest.mark.parametrize(
    "value",
    [None, 0, "", 1754418600.0, 1754418600, 0.5, 1e9, 2**31 + 0.25],
)
def test_iso_z_matches_the_herdd_copy(value) -> None:                        # noqa: ANN001
    """Byte-for-byte parity on the pure function fleetd.py:3008 calls as
    `herdd._iso_z`. That external caller is why herdd keeps its copy
    (add-only, plan §8); this test is what proves the two agree."""
    v = _herdd()
    assert journal._iso_z(value) == v._iso_z(value)


def test_journal_queues_stay_capped_and_write_only_their_own_slot() -> None:
    """The two pure queue functions, driven past the cap.

    This ran side by side against `herdd`'s copies until step 6d; the
    launcher carries no bodies now (`JOB_HANDOFF_JOURNAL_MAX` is not even one
    of the names it re-exports), so what the second arm was proving — the cap
    is honoured and nothing but the queue's own slot is written — is asserted
    on this module directly.
    """
    for fn, slot in ((journal._job_ladder_journal, "ladder_journal"),
                     (journal._job_handoff_journal, "handoff_journal")):
        mine: dict[str, object] = {}
        n_pushed = journal.JOB_HANDOFF_JOURNAL_MAX + 7
        for i in range(n_pushed):
            fn(mine, f"e{i}", n=i, iid=None)
        assert list(mine) == [slot], "a queue writer touched a foreign key"
        q = mine[slot]
        assert len(q) == journal.JOB_HANDOFF_JOURNAL_MAX, "the cap stopped holding"
        # oldest dropped, newest kept: the drain reads the RECENT decisions.
        assert q[-1][0] == f"e{n_pushed - 1}" and q[0][0] == "e7"


# --------------------------------------------------------------------------- #
# provenance markers — plan §7.1 generates the rename table from these
# --------------------------------------------------------------------------- #
def test_every_ported_symbol_carries_a_moved_from_marker() -> None:
    """A missing marker is a symbol the test migration cannot find (README §2).

    Checked as text, directly above the definition, because that is exactly how
    the table generator will read it."""
    src = (VAST_DIR / "vastlib" / "supervise" / "journal.py").read_text(encoding="utf-8")
    lines = src.splitlines()
    expected = {
        "def _sup_emit(": "herdd._sup_emit",
        "def _job_handoff_emit(": "herdd._job_handoff_emit",
        "def _job_handoff_journal(": "herdd._job_handoff_journal",
        "def _job_ladder_journal(": "herdd._job_ladder_journal",
        "def _iso_z(": "herdd._iso_z",
        "JOB_HANDOFF_JOURNAL_MAX = 200": "herdd.JOB_HANDOFF_JOURNAL_MAX",
    }
    for i, line in enumerate(lines):
        for prefix, origin in list(expected.items()):
            if line.startswith(prefix):
                assert lines[i - 1] == f"# moved-from: {origin}", (
                    f"{prefix!r} is missing its marker; line above is "
                    f"{lines[i - 1]!r}")
                del expected[prefix]
                break
    assert not expected, f"definitions not found in journal.py: {sorted(expected)}"

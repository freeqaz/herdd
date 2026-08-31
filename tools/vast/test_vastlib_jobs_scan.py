"""`vastlib.jobs.scan` — the bulk fold, held to its two properties.

Why this file exists
--------------------
`scan.fold_many` replaced a `jobmeta.read_job` per queued ticket in
`job orphans` and `job ls`. That change trades a shape everyone understands
(one job, one listing, one fold) for one that is faster and easier to get
subtly wrong, so both halves of the trade are pinned here:

1. **THE SPEED PROPERTY, as an assertion and not a benchmark.** N jobs cost a
   FIXED number of rclone subprocesses — 1 warm, 2 cold — never O(N). The
   injected runner counts its own calls, so this holds with no network, no B2
   and no clock. A regression to per-job I/O fails the count, not a timer.

2. **THE FRESHNESS PROPERTY.** An event that appeared on B2 since the last call
   is never missed, because nothing folded is cached and the key set is
   re-listed every time; a listed body that could not be fetched is reported as
   `scan_error` rather than folded around (a missing event folds to a YOUNGER
   status, and a younger status is what mints a false orphan — the 2026-07-30
   incident `jobmeta.read_job_fresh` documents).

No network: every test drives an in-memory fake with rclone's runner contract.

Provenance: created 2026-08-17 with `vastlib/jobs/scan.py`, against a measured
`job orphans` of 139.7 s over 275 tickets (138.3 s of it 275 `rclone copy`
subprocesses that transferred zero bytes).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

VAST_DIR = Path(__file__).resolve().parent
if str(VAST_DIR) not in sys.path:                      # entry-script sys.path shape
    sys.path.insert(0, str(VAST_DIR))

import jobmeta                                         # noqa: E402  Zone S

from vastlib.jobs import scan                          # noqa: E402

BUCKET = "bkt"


class FakeRclone:
    """rclone's runner contract over an in-memory `key -> body` store.

    Implements exactly the two ops `scan` uses — `lsf -R --filter-from` and
    `copy --files-from` — and records every invocation so a test can assert on
    the CALL COUNT. Deliberately not shared with `test_jobmeta.py`'s `FakeB2`:
    that one models the per-job ops and materializes a whole prefix, which would
    hide the very fan-out this file is here to bound.
    """

    def __init__(self, store: dict[str, str] | None = None) -> None:
        self.store: dict[str, str] = dict(store or {})
        self.calls: list[list[str]] = []
        self.fail_list = False
        self.fail_copy = False
        self.drop_from_copy: set[str] = set()   # keys the copy silently omits

    # -- helpers ---------------------------------------------------------
    def _prefix(self, remote: str) -> str:
        pfx = f"b2:{BUCKET}/"
        assert remote.startswith(pfx), remote
        return remote[len(pfx):]

    @staticmethod
    def _match(rel: str, rules: list[str]) -> bool:
        import fnmatch
        for rule in rules:
            sign, pat = rule[0], rule[2:].strip()
            if pat == "*":
                if sign == "-":
                    return False
                return True
            if fnmatch.fnmatchcase("/" + rel, pat):
                return sign == "+"
        return True

    # -- the runner ------------------------------------------------------
    def __call__(self, args, input=None):               # noqa: ANN001, ANN204
        self.calls.append(list(args))
        op = args[0]
        if op == "lsf":
            if self.fail_list:
                return 3, "", "directory not found"
            root = self._prefix([a for a in args if a.startswith("b2:")][0])
            rules = Path(args[args.index("--filter-from") + 1]).read_text().splitlines()
            hits = [k[len(root):] for k in sorted(self.store)
                    if k.startswith(root) and self._match(k[len(root):], rules)]
            return 0, "".join(h + "\n" for h in hits), ""
        if op == "copy":
            if self.fail_copy:
                return 5, "", "b2 down"
            root = self._prefix([a for a in args if a.startswith("b2:")][0])
            dst = Path([a for a in args[1:]
                        if not a.startswith(("-", "b2:")) and args[args.index(a) - 1]
                        not in ("--files-from", "--transfers", "--checkers")][0])
            rels = Path(args[args.index("--files-from") + 1]).read_text().split()
            for rel in rels:
                if rel in self.drop_from_copy:
                    continue
                body = self.store.get(root + rel)
                if body is None:
                    continue
                out = dst / rel
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(body)
            return 0, "", ""
        raise AssertionError(f"unexpected rclone op {op}: {args}")

    # -- store authoring -------------------------------------------------
    def add_event(self, job_id: str, event: str, *, ts: str, actor: str = "box:44",
                  **fields) -> str:                     # noqa: ANN003
        ev = jobmeta.make_event(job_id, event, actor, ts=ts, **fields)
        key = f"jobs/{job_id}/events/{jobmeta.event_key(ev)}"
        self.store[key] = json.dumps(ev, separators=(",", ":")) + "\n"
        return key

    def n_calls(self, op: str) -> int:
        return len([c for c in self.calls if c[0] == op])


def _seed(fake: FakeRclone, job_id: str, *, box: str = "44",
          terminal: str | None = None) -> None:
    """submitted -> claimed -> started [-> terminal], one job."""
    fake.add_event(job_id, "submitted", ts="20260817T000000000Z", actor="cli:h",
                   bundle_sha256="ab", box=box, name="p", entrypoint="run.sh",
                   timeout_s=60)
    fake.add_event(job_id, "claimed", ts="20260817T000001000Z", instance_id=box)
    fake.add_event(job_id, "started", ts="20260817T000002000Z", instance_id=box)
    if terminal:
        fake.add_event(job_id, terminal, ts="20260817T000003000Z",
                       instance_id=box, rc=0 if terminal == "done" else 9)


@pytest.fixture()
def cache(tmp_path):                                    # noqa: ANN001, ANN201
    d = tmp_path / "cache"
    d.mkdir()
    return str(d)


@pytest.fixture(autouse=True)
def _force_bulk_path(monkeypatch):                      # noqa: ANN001, ANN201
    """Take the BULK branch regardless of job count.

    `fold_many` falls back to `jobmeta.read_job` per job below
    `BULK_MIN_JOBS` (10 — the conservative side of the measured 7-10 crossover
    band; the bulk listing has a ~4 s floor that a 3-ticket `--box` scan should
    not pay). Most tests here are about the bulk path's SEMANTICS, not its
    threshold, and padding each of them to ten jobs would obscure what is being
    asserted. The threshold has its own tests below, which restore the real
    value. `BULK_MIN_BOXES` is deliberately NOT patched — the `fold_boxes`
    tests exercise the real one from both sides.
    """
    monkeypatch.setattr(scan, "BULK_MIN_JOBS", 1)


# --------------------------------------------------------------------------- #
# 1. THE SPEED PROPERTY — O(1) subprocesses, not O(N)
# --------------------------------------------------------------------------- #
def test_cold_scan_of_many_jobs_is_two_rclone_calls(cache):        # noqa: ANN001
    """60 jobs, empty cache: ONE listing + ONE bulk fetch. The defect this
    module replaced would be 60 `copy` calls (measured: 275 tickets -> 275
    subprocesses -> 138.3 s of a 139.7 s command)."""
    fake = FakeRclone()
    jids = [f"20260817T00000{i % 10}-bulk-{i:04d}" for i in range(60)]
    for j in jids:
        _seed(fake, j, terminal="done")

    views = scan.fold_many(jids, runner=fake, bucket=BUCKET, cache_dir=cache)

    assert len(views) == 60
    assert all(v["status"] == "done" for v in views.values())
    assert fake.n_calls("lsf") == 1
    assert fake.n_calls("copy") == 1
    assert len(fake.calls) == 2


def test_warm_scan_makes_no_copy_call_at_all(cache):               # noqa: ANN001
    """Second pass over an unchanged queue: the bodies are immutable and already
    cached, so the only remote work left is re-learning the key set."""
    fake = FakeRclone()
    jids = [f"20260817T000000-warm-{i:04d}" for i in range(25)]
    for j in jids:
        _seed(fake, j, terminal="failed")
    scan.fold_many(jids, runner=fake, bucket=BUCKET, cache_dir=cache)
    fake.calls.clear()

    views = scan.fold_many(jids, runner=fake, bucket=BUCKET, cache_dir=cache)

    assert all(v["status"] == "failed" for v in views.values())
    assert fake.n_calls("copy") == 0
    assert len(fake.calls) == 1


def test_call_count_does_not_grow_with_job_count(cache, tmp_path):  # noqa: ANN001
    """The property stated as a property: 4 jobs and 120 jobs cost the same."""
    counts = []
    for n in (4, 120):
        fake = FakeRclone()
        jids = [f"20260817T000000-grow{n}-{i:04d}" for i in range(n)]
        for j in jids:
            _seed(fake, j)
        d = tmp_path / f"cache{n}"
        d.mkdir()
        scan.fold_many(jids, runner=fake, bucket=BUCKET, cache_dir=str(d))
        counts.append(len(fake.calls))
    assert counts[0] == counts[1] == 2


def test_empty_job_list_touches_the_network_not_at_all(cache):     # noqa: ANN001
    fake = FakeRclone()
    assert scan.fold_many([], runner=fake, bucket=BUCKET, cache_dir=cache) == {}
    assert fake.calls == []


def test_listing_is_scoped_to_the_jobs_asked_for(cache):           # noqa: ANN001
    """The filter names the queue, not the bucket. An unfiltered recursive
    listing of `jobs/` walks every checkpoint and result ever written (78 236
    objects, 15.4 s measured) and grows without bound; the scoped one costs
    5.1 s and grows with the QUEUE."""
    fake = FakeRclone()
    _seed(fake, "20260817T000000-wanted-0001")
    _seed(fake, "20260817T000000-other-0002")
    fake.store["jobs/20260817T000000-wanted-0001/checkpoints/big.bin"] = "x"
    fake.store["jobs/20260817T000000-wanted-0001/results/gens.jsonl"] = "y"

    views = scan.fold_many(["20260817T000000-wanted-0001"], runner=fake,
                           bucket=BUCKET, cache_dir=cache)

    assert set(views) == {"20260817T000000-wanted-0001"}
    assert views["20260817T000000-wanted-0001"]["n_events"] == 3
    assert "--fast-list" not in fake.calls[0]


def test_a_job_with_no_events_folds_rather_than_raising(cache):    # noqa: ANN001
    """A ticket written moments ago has no events yet. That is a legitimate
    state (`submitted`, unclaimed), not an error."""
    fake = FakeRclone()
    views = scan.fold_many(["20260817T000000-brandnew-0001"], runner=fake,
                           bucket=BUCKET, cache_dir=cache)
    assert views["20260817T000000-brandnew-0001"]["n_events"] == 0


# --------------------------------------------------------------------------- #
# 2. THE FRESHNESS PROPERTY — a new event is never missed
# --------------------------------------------------------------------------- #
def test_an_event_appearing_after_a_warm_scan_is_picked_up(cache):  # noqa: ANN001
    """THE test this module exists to pass. The 2026-07-30 incident was a fold
    running minutes behind reality — a job that had already FAILED still read
    `submitted`, which is indistinguishable from "queued on a dead box". A
    cached view with a TTL would reproduce it exactly."""
    fake = FakeRclone()
    jid = "20260817T000000-latefail-0001"
    _seed(fake, jid)
    assert scan.fold_many([jid], runner=fake, bucket=BUCKET,
                          cache_dir=cache)[jid]["status"] == "started"

    fake.add_event(jid, "failed", ts="20260817T000009000Z",
                   instance_id="44", rc=16, reason="rc=16")

    v = scan.fold_many([jid], runner=fake, bucket=BUCKET, cache_dir=cache)[jid]
    assert v["status"] == "failed", "a new event was folded from a cached view"
    assert v["n_events"] == 4


def test_nothing_folded_is_cached_between_calls(cache):            # noqa: ANN001
    """Sharper than the test above: even with the key set UNCHANGED, the second
    call re-folds from bodies rather than replaying a stored verdict — so a
    corrected body (or a differing `live_iids`) cannot be masked."""
    fake = FakeRclone()
    jid = "20260817T000000-refold-0001"
    _seed(fake, jid)
    first = scan.fold_many([jid], runner=fake, bucket=BUCKET, cache_dir=cache)[jid]
    assert first["live"] is False                    # box 44 not in live_iids

    second = scan.fold_many([jid], runner=fake, bucket=BUCKET, cache_dir=cache,
                            live_iids={"44"})[jid]
    assert second["live"] is True


def test_the_fold_set_is_listed_union_cached(cache):               # noqa: ANN001
    """A body another reader already cached is folded even if this call's
    listing did not name it. A key once observed is never un-observed, so the
    union can only be FRESHER — and it keeps parity with `read_job`, which
    folds whatever the cache directory holds."""
    fake = FakeRclone()
    jid = "20260817T000000-union-0001"
    _seed(fake, jid)
    scan.fold_many([jid], runner=fake, bucket=BUCKET, cache_dir=cache)

    # a terminal event that exists LOCALLY but which the listing will not report
    key = fake.add_event(jid, "done", ts="20260817T000010000Z",
                         instance_id="44", rc=0)
    name = key.rsplit("/", 1)[1]
    (Path(jobmeta.event_cache_dir(jid, cache)) / name).write_text(fake.store[key])
    del fake.store[key]

    assert scan.fold_many([jid], runner=fake, bucket=BUCKET,
                          cache_dir=cache)[jid]["status"] == "done"


def test_a_listed_body_that_never_arrives_is_an_error_not_a_gap(cache):  # noqa: ANN001
    """Fail LOUD. Folding around a missing event reports a younger status than
    the truth, and a younger status on a dead box is exactly what mints a false
    ORPHAN_UNCLAIMED — the failure mode this whole lane is about."""
    fake = FakeRclone()
    jid = "20260817T000000-holed-0001"
    _seed(fake, jid)
    lost = fake.add_event(jid, "done", ts="20260817T000010000Z",
                          instance_id="44", rc=0)
    fake.drop_from_copy = {lost[len("jobs/"):]}

    v = scan.fold_many([jid], runner=fake, bucket=BUCKET, cache_dir=cache)[jid]
    assert "scan_error" in v
    assert "unreadable locally" in v["scan_error"]


def test_a_failed_listing_raises_instead_of_reporting_an_empty_queue(cache):  # noqa: ANN001
    """An unreadable listing that degraded to "no events" would fold EVERY
    ticket to unclaimed and report the whole fleet as orphaned. Tri-state or
    raise; never a confident wrong answer."""
    fake = FakeRclone()
    _seed(fake, "20260817T000000-unread-0001")
    fake.fail_list = True
    with pytest.raises(scan.ScanError, match="bulk job listing failed"):
        scan.fold_many(["20260817T000000-unread-0001"], runner=fake,
                       bucket=BUCKET, cache_dir=cache)


def test_a_failed_bulk_fetch_raises(cache):                        # noqa: ANN001
    fake = FakeRclone()
    _seed(fake, "20260817T000000-nofetch-0001")
    fake.fail_copy = True
    with pytest.raises(scan.ScanError, match="bulk event fetch failed"):
        scan.fold_many(["20260817T000000-nofetch-0001"], runner=fake,
                       bucket=BUCKET, cache_dir=cache)


def test_done_marker_rides_along_on_the_same_listing(cache):       # noqa: ANN001
    """`results.DONE.json` is written exactly once, LAST, as a new key — the
    strongest cheap evidence a job finished. The scoped listing already walks
    `<job_id>/`, so reporting it costs no extra round trip. Reported only; the
    verdict lattice is not touched."""
    fake = FakeRclone()
    jid = "20260817T000000-marked-0001"
    _seed(fake, jid)
    fake.store[f"jobs/{jid}/{scan.DONE_MARKER}"] = json.dumps({"rc": 0})

    views = scan.fold_many([jid, "20260817T000000-unmarked-0002"], runner=fake,
                           bucket=BUCKET, cache_dir=cache)
    assert views[jid]["done_marker"] is True
    assert views["20260817T000000-unmarked-0002"]["done_marker"] is False


# --------------------------------------------------------------------------- #
# 3. The filter grammar — the one place a job id reaches a config file
# --------------------------------------------------------------------------- #
def test_job_ids_are_validated_before_they_reach_the_filter_file():
    """`JOB_ID_RE` is what keeps an id from forging a filter rule or widening
    the glob. Assert the validation is actually wired, not merely nearby."""
    fake = FakeRclone()
    with pytest.raises(jobmeta.JobmetaError):
        scan.fold_many(["ok-1\n+ /**"], runner=fake, bucket=BUCKET)
    assert fake.calls == []


def test_small_queues_take_the_per_job_path_because_it_is_faster(monkeypatch,
                                                                 cache):  # noqa: ANN001
    """The bulk listing has a ~4 s floor that does not shrink with the job count
    (rclone must enumerate `jobs/`, 78 236 objects, before the filter can prune).
    Measured 2026-08-17: n=1 bulk 4.05 s vs loop 1.07 s; n=8 bulk 3.91 s vs loop
    4.74 s. A blanket bulk path would have made `job orphans --box <IID>` 4x
    SLOWER while making the unfiltered scan 21x faster."""
    monkeypatch.setattr(scan, "BULK_MIN_JOBS", 10)      # the real value
    seen = []
    monkeypatch.setattr(jobmeta, "read_job",
                        lambda jid, **kw: seen.append(jid) or {"status": "done"})
    fake = FakeRclone()
    views = scan.fold_many([f"20260817T000000-small-{i:04d}" for i in range(9)],
                           runner=fake, bucket=BUCKET, cache_dir=cache)
    assert len(seen) == 9
    assert fake.calls == [], "the small path must not take the bulk listing"
    # tri-state: NOT PROBED, which is not the same as "no marker"
    assert all(v["done_marker"] is None for v in views.values())


def test_at_the_threshold_the_bulk_path_takes_over(monkeypatch, cache):  # noqa: ANN001
    monkeypatch.setattr(scan, "BULK_MIN_JOBS", 10)      # the real value
    monkeypatch.setattr(jobmeta, "read_job",
                        lambda jid, **kw: pytest.fail("must take the bulk path"))
    fake = FakeRclone()
    jids = [f"20260817T000000-thresh-{i:04d}" for i in range(10)]
    for j in jids:
        _seed(fake, j, terminal="done")
    views = scan.fold_many(jids, runner=fake, bucket=BUCKET, cache_dir=cache)
    assert len(views) == 10
    assert fake.n_calls("lsf") == 1


def test_the_small_path_still_isolates_one_broken_log(monkeypatch, cache):  # noqa: ANN001
    monkeypatch.setattr(scan, "BULK_MIN_JOBS", 10)      # the real value

    def _read(jid, **kw):
        if jid.endswith("0001"):
            raise RuntimeError("cannot fold")
        return {"status": "done"}

    monkeypatch.setattr(jobmeta, "read_job", _read)
    views = scan.fold_many(["20260817T000000-iso-0001", "20260817T000000-iso-0002"],
                           runner=FakeRclone(), bucket=BUCKET, cache_dir=cache)
    assert views["20260817T000000-iso-0001"]["scan_error"] == "cannot fold"
    assert views["20260817T000000-iso-0002"]["status"] == "done"


def test_filter_lines_admit_only_the_two_key_shapes():
    lines = scan._filter_lines(["j-1", "j-2"])
    assert lines == ["+ /j-1/events/*.json", "+ /j-1/results.DONE.json",
                     "+ /j-2/events/*.json", "+ /j-2/results.DONE.json",
                     "- *"]


def test_listing_parse_drops_keys_for_jobs_nobody_asked_about():
    """A filter is a request, not a guarantee. A stray key must not become a
    phantom event on a job the caller never named."""
    events, done = scan._parse_listing(
        "j-1/events/a.json\nj-9/events/b.json\nj-1/results.DONE.json\n"
        "j-1/checkpoints/c.bin\nj-1/events/nested/d.json\n",
        {"j-1"})
    assert events == {"j-1": {"a.json"}}
    assert done == {"j-1"}


# --------------------------------------------------------------------------- #
# 4. fold_boxes — the same defect on the box-lifecycle logs
# --------------------------------------------------------------------------- #
def _seed_box(fake, iid, *, parked=True):                # noqa: ANN001, ANN201
    for ev, kw in [("jobd_up", {}), ("drained", {}),
                   ("parked_self", {"reason": "queue drained"})][:3 if parked else 2]:
        e = jobmeta.make_box_event(iid, ev, f"box:{iid}",
                                   ts=f"20260817T00000{len(fake.store) % 9}000Z", **kw)
        fake.store[f"jobs/nodes/{iid}/events/{jobmeta.event_key(e)}"] = \
            json.dumps(e, separators=(",", ":")) + "\n"


def test_many_boxes_cost_two_rclone_calls_not_two_per_box(cache, monkeypatch):  # noqa: ANN001
    """`cmd_job_ls` called `jobmeta.read_box` once per DISTINCT BOX — measured
    2026-08-17 at 154 boxes / 69.6 s, the reason `job ls` still took 78.6 s
    after the job folds were fixed. The box roster grows with fleet HISTORY."""
    monkeypatch.setattr(jobmeta, "read_box",
                        lambda iid, **kw: pytest.fail("must take the bulk path"))
    fake = FakeRclone()
    iids = [str(46000000 + i) for i in range(30)]
    for i in iids:
        _seed_box(fake, i)

    views = scan.fold_boxes(iids, runner=fake, bucket=BUCKET, cache_dir=cache)

    assert len(views) == 30
    assert all(v["parked"] is True for v in views.values())
    assert fake.n_calls("lsf") == 1 and fake.n_calls("copy") == 1


def test_box_scan_is_warm_on_the_second_pass(cache):     # noqa: ANN001
    fake = FakeRclone()
    iids = [str(46000000 + i) for i in range(30)]
    for i in iids:
        _seed_box(fake, i)
    scan.fold_boxes(iids, runner=fake, bucket=BUCKET, cache_dir=cache)
    fake.calls.clear()
    scan.fold_boxes(iids, runner=fake, bucket=BUCKET, cache_dir=cache)
    assert fake.n_calls("copy") == 0 and len(fake.calls) == 1


def test_a_failed_box_listing_degrades_to_the_per_box_path(cache, monkeypatch):  # noqa: ANN001
    """Unlike a JOB fold, an unreadable lifecycle log cannot mint a verdict — it
    only renders "not parked". So this one degrades instead of exiting, and it
    must degrade to the OLD behaviour rather than to a blank fleet."""
    fake = FakeRclone()
    fake.fail_list = True
    seen = []
    monkeypatch.setattr(jobmeta, "read_box",
                        lambda iid, **kw: seen.append(iid) or {"parked": True})
    views = scan.fold_boxes([str(46000000 + i) for i in range(30)],
                            runner=fake, bucket=BUCKET, cache_dir=cache)
    assert len(seen) == 30
    assert all(v["parked"] is True for v in views.values())


def test_a_small_box_roster_still_uses_read_box(cache, monkeypatch):  # noqa: ANN001
    seen = []
    monkeypatch.setattr(jobmeta, "read_box",
                        lambda iid, **kw: seen.append(iid) or {"parked": False})
    fake = FakeRclone()
    scan.fold_boxes(["41", "42"], runner=fake, bucket=BUCKET, cache_dir=cache)
    assert seen == ["41", "42"] and fake.calls == []


def test_one_unreadable_box_log_never_hides_the_others(cache, monkeypatch):  # noqa: ANN001
    def _read(iid, **kw):
        if iid == "41":
            raise RuntimeError("b2 down")
        return {"parked": True}

    monkeypatch.setattr(jobmeta, "read_box", _read)
    views = scan.fold_boxes(["41", "42"], runner=FakeRclone(), bucket=BUCKET,
                            cache_dir=cache)
    assert views["41"] == {"parked": False, "drained_pending": False}
    assert views["42"]["parked"] is True

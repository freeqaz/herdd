"""The instance -> machine ledger, and the resolver chain that reads it.

The defect this exists for: vast's API is the only source of that mapping and
drops a row when the box is destroyed, so `hostfacts.py ingest` could resolve
only boxes that were alive at the moment somebody ran it — 3 of 202 on
2026-08-24. Everything below is about the mapping outliving the box.

Two duplications are load-bearing and pinned here rather than asked for in a
comment: the ledger's on-disk path (spelled in vastlib AND in hostfacts, which
cannot import vastlib) and the entry shape both sides agree on.
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hostfacts as hf  # noqa: E402
from vastlib.core import config, machine_ledger as ml  # noqa: E402


@pytest.fixture
def ledger(tmp_path):
    return str(tmp_path / "machine_ledger.json")


# --------------------------------------------------------------------------- #
# the duplication that cannot be imported away
# --------------------------------------------------------------------------- #
def test_both_sides_agree_on_where_the_ledger_lives(monkeypatch, tmp_path):
    """hostfacts.py is a Zone S flat leaf shipped in the jobd bundle and must
    import bare-name under `python3 -P`, so it cannot import vastlib for this
    path. The default is therefore spelled twice; if the two ever disagree,
    fleetd writes a ledger `ingest` will never find and the whole lane fails
    silently and looks merely empty."""
    monkeypatch.delenv(ml.PATH_ENV, raising=False)
    monkeypatch.setenv("FLEETD_STATE_DIR", str(tmp_path / "state"))
    assert ml.store_path() == hf.ledger_path()
    assert ml.LEDGER_FILENAME == hf.LEDGER_FILENAME


def test_both_sides_honour_the_same_override(monkeypatch, tmp_path):
    p = str(tmp_path / "elsewhere.json")
    monkeypatch.setenv(ml.PATH_ENV, p)
    assert ml.store_path() == p == hf.ledger_path()
    assert ml.PATH_ENV == hf.LEDGER_PATH_ENV


def test_the_default_dir_matches_core_config(monkeypatch, tmp_path):
    """hostfacts' literal fallback must be the same directory core.config
    resolves when FLEETD_STATE_DIR is unset."""
    monkeypatch.delenv(ml.PATH_ENV, raising=False)
    monkeypatch.delenv("FLEETD_STATE_DIR", raising=False)
    assert os.path.dirname(hf.ledger_path()) == config.fleet_state_dir()


# --------------------------------------------------------------------------- #
# append-only: an entry is a historical fact
# --------------------------------------------------------------------------- #
def test_a_new_pair_is_recorded(ledger):
    assert ml.record([("48603388", "140799")], now=100.0, path=ledger) == 1
    assert ml.resolve("48603388", ledger) == "140799"


def test_the_mapping_outlives_the_box(ledger):
    """The whole point: the instance is long gone from the API and the answer
    is still here."""
    ml.record([("48603388", "140799")], now=100.0, path=ledger)
    # ...a thousand ticks later, that instance never seen again
    ml.record([("48999999", "140800")], now=9999.0, path=ledger)
    assert ml.resolve("48603388", ledger) == "140799"


def test_re_seeing_a_box_refreshes_last_seen_not_first_seen(ledger):
    later = 100.0 + ml.LAST_SEEN_REFRESH_S
    ml.record([("1", "m1")], now=100.0, path=ledger)
    ml.record([("1", "m1")], now=later, path=ledger)
    e = ml.load(ledger)["1"]
    assert e["first_seen"] == 100.0 and e["last_seen"] == later


def test_an_unchanged_fleet_does_not_rewrite_the_file(ledger):
    """A steady fleet costs one read per tick and no write.

    The tick clock is what makes this load-bearing: every tick carries a new
    `now`, so counting a bare `last_seen` refresh as a change rewrites the whole
    file — and journals a change — every ~50 s on a fleet where the mapping has
    not moved."""
    ml.record([("1", "m1")], now=100.0, path=ledger)
    before = os.stat(ledger).st_mtime_ns
    assert ml.record([("1", "m1")], now=100.0, path=ledger) == 0
    assert ml.record([("1", "m1")], now=150.0, path=ledger) == 0, "a later tick"
    assert ml.record([("1", "m1")], now=3500.0, path=ledger) == 0, "still inside"
    assert os.stat(ledger).st_mtime_ns == before


def test_last_seen_is_persisted_once_it_goes_properly_stale(ledger):
    """Lazy, not abandoned: the field still answers "roughly when did we last
    see this box", which is all any reader asks of it."""
    ml.record([("1", "m1")], now=100.0, path=ledger)
    ml.record([("1", "m1")], now=200.0, path=ledger)
    assert ml.load(ledger)["1"]["last_seen"] == 100.0, "in-memory only"

    stale = 100.0 + ml.LAST_SEEN_REFRESH_S
    assert ml.record([("1", "m1")], now=stale, path=ledger) == 0, \
        "a write, but nothing an operator wants journalled as a change"
    assert ml.load(ledger)["1"]["last_seen"] == stale


def test_a_real_change_still_writes_immediately(ledger):
    """The refresh throttle may never delay a mapping or a conflict — those are
    the facts the ledger exists to outlive the box with."""
    ml.record([("1", "m1")], now=100.0, path=ledger)
    before = os.stat(ledger).st_mtime_ns
    assert ml.record([("1", "m1"), ("2", "m2")], now=110.0, path=ledger) == 1
    assert ml.load(ledger)["2"]["machine_id"] == "m2"
    assert os.stat(ledger).st_mtime_ns != before
    assert ml.record([("1", "m9")], now=120.0, path=ledger) == 1
    assert ml.load(ledger)["1"]["conflicts"] == ["m9"]


def test_an_entry_with_an_unreadable_clock_gets_one(ledger):
    """A record from a hand-edited or older file must not be stuck unstale
    forever — absent or unparseable reads as stale, never as fresh."""
    p = json.loads(open(ledger).read()) if os.path.exists(ledger) else {}
    p["1"] = {"machine_id": "m1", "first_seen": 1.0, "last_seen": "yesterday"}
    open(ledger, "w").write(json.dumps(p))
    assert ml.record([("1", "m1")], now=500.0, path=ledger) == 0
    assert ml.load(ledger)["1"]["last_seen"] == 500.0


def test_a_conflicting_machine_is_kept_not_overwritten(ledger):
    """One instance id naming two machines makes every record it wrote
    ambiguous. Overwriting would hide that rather than fix it."""
    ml.record([("1", "m1")], now=100.0, path=ledger)
    ml.record([("1", "m2")], now=200.0, path=ledger)
    e = ml.load(ledger)["1"]
    assert e["machine_id"] == "m1"
    assert e["conflicts"] == ["m2"]


def test_a_conflicted_entry_resolves_to_nothing(ledger):
    """An ambiguous attribution is worse than an absent one — the record would
    be filed under a machine that may not have produced it."""
    ml.record([("1", "m1")], now=100.0, path=ledger)
    ml.record([("1", "m2")], now=200.0, path=ledger)
    assert ml.resolve("1", ledger) is None
    assert hf.ledger_machine_resolver(ledger)("1") is None


@pytest.mark.parametrize("pair", [(None, "m1"), ("1", None), ("", ""), (0, 0)])
def test_a_half_pair_is_skipped(ledger, pair):
    assert ml.record([pair], now=1.0, path=ledger) == 0


def test_a_missing_or_corrupt_file_reads_empty(tmp_path):
    """A resolver that died on a truncated write would take an ingest with it,
    and this index can always be rebuilt going forward."""
    assert ml.load(str(tmp_path / "nope.json")) == {}
    bad = tmp_path / "bad.json"
    bad.write_text("{ not json")
    assert ml.load(str(bad)) == {}
    assert hf.ledger_machine_resolver(str(bad))("1") is None


def test_stats_counts_machines_not_rentals(ledger):
    ml.record([("1", "m1"), ("2", "m1"), ("3", "m2")], now=1.0, path=ledger)
    s = ml.stats(ledger)
    assert s == {"instances": 3, "machines": 2, "conflicted": 0}


# --------------------------------------------------------------------------- #
# the resolver chain
# --------------------------------------------------------------------------- #
def test_the_ledger_resolves_what_the_api_cannot(ledger):
    ml.record([("48603388", "140799")], now=1.0, path=ledger)
    dead_api = lambda _iid: None                          # noqa: E731
    chain = hf.chained_resolver(hf.ledger_machine_resolver(ledger), dead_api)
    assert chain("48603388") == "140799"


def test_the_api_still_answers_for_a_box_the_ledger_never_saw(ledger):
    chain = hf.chained_resolver(hf.ledger_machine_resolver(ledger),
                                lambda iid: "999" if iid == "new" else None)
    assert chain("new") == "999"


def test_first_non_none_wins_in_order(ledger):
    chain = hf.chained_resolver(lambda _i: None, lambda _i: "a",
                                lambda _i: "b")
    assert chain("x") == "a"


def test_a_source_that_raises_does_not_kill_the_chain():
    def _boom(_iid):
        raise RuntimeError("API on fire")
    assert hf.chained_resolver(_boom, lambda _i: "ok")("x") == "ok"


def test_a_chain_of_nothing_resolves_to_none():
    assert hf.chained_resolver(None, lambda _i: None)("x") is None


# --------------------------------------------------------------------------- #
# end to end through ingest — a promotion, never a move
# --------------------------------------------------------------------------- #
class _Store:
    """The minimum of hostfacts' store protocol that `ingest` touches."""

    def __init__(self, node_records):
        self.node = dict(node_records)
        self.machine = {}

    def dirs(self, prefix):
        if prefix == hf.NODES:
            return sorted({k.split("/")[2] for k in self.node})
        return []

    def keys(self, prefix):
        if prefix == hf.BY_MACHINE:
            return sorted(self.machine)
        return sorted(k for k in self.node if k.startswith(prefix))

    def get(self, key):
        return self.node.get(key) or self.machine.get(key)

    def put(self, key, rec):
        self.machine[key] = rec
        return True


def _node_key(iid, ts="2026-08-25T01:02:03Z"):
    return f"{hf.NODES}/{iid}/{hf.NODE_LEAF}/cpu-{ts}.json"


def test_ingest_pins_a_dead_box_via_the_ledger(ledger):
    """The measured failure, inverted: this record's instance is gone from the
    API, and it still lands under its machine."""
    key = _node_key("48603388")
    store = _Store({key: {"kind": "cpu", "instance_id": "48603388",
                          "ts": "2026-08-25T01:02:03Z", "units": "pyops"}})
    ml.record([("48603388", "140799")], now=1.0, path=ledger)
    res = hf.ingest(store, hf.chained_resolver(
        hf.ledger_machine_resolver(ledger), lambda _i: None))
    assert len(res["pinned"]) == 1 and res["unresolved"] == []
    dest = res["pinned"][0]
    assert dest.startswith(f"{hf.BY_MACHINE}/140799/")
    assert store.machine[dest]["machine_id"] == "140799"
    assert store.machine[dest]["pinned_from"] == key
    # a PROMOTION: the original is untouched and still readable by instance
    assert key in store.node


def test_ingest_leaves_a_record_alone_when_nothing_resolves(ledger):
    key = _node_key("48603388")
    store = _Store({key: {"kind": "cpu", "instance_id": "48603388",
                          "ts": "t", "units": "pyops"}})
    res = hf.ingest(store, hf.chained_resolver(
        hf.ledger_machine_resolver(ledger), lambda _i: None))
    assert res["pinned"] == [] and res["unresolved"] == [key]
    assert store.machine == {} and key in store.node


def test_the_kind_rides_across_so_a_cpu_record_pins_as_cpu(ledger):
    key = _node_key("1")
    store = _Store({key: {"kind": "cpu", "instance_id": "1", "ts": "t",
                          "units": "pyops"}})
    ml.record([("1", "m1")], now=1.0, path=ledger)
    res = hf.ingest(store, hf.ledger_machine_resolver(ledger))
    assert "/cpu-" in res["pinned"][0]


def test_a_machine_id_on_the_record_beats_every_resolver(ledger):
    """`ingest` prefers what the record carries. A box that was told its machine
    id must not be re-attributed by a stale ledger."""
    key = _node_key("1")
    store = _Store({key: {"kind": "cpu", "instance_id": "1", "ts": "t",
                          "machine_id": "from_record", "units": "pyops"}})
    ml.record([("1", "from_ledger")], now=1.0, path=ledger)
    res = hf.ingest(store, hf.ledger_machine_resolver(ledger))
    assert res["pinned"][0].startswith(f"{hf.BY_MACHINE}/from_record/")


def test_ingest_is_idempotent(ledger):
    key = _node_key("1")
    store = _Store({key: {"kind": "cpu", "instance_id": "1", "ts": "t",
                          "units": "pyops"}})
    ml.record([("1", "m1")], now=1.0, path=ledger)
    r = hf.ledger_machine_resolver(ledger)
    assert len(hf.ingest(store, r)["pinned"]) == 1
    second = hf.ingest(store, r)
    assert second["pinned"] == [] and second["already"] == 1


def test_a_dry_run_writes_nothing(ledger):
    key = _node_key("1")
    store = _Store({key: {"kind": "cpu", "instance_id": "1", "ts": "t",
                          "units": "pyops"}})
    ml.record([("1", "m1")], now=1.0, path=ledger)
    res = hf.ingest(store, hf.ledger_machine_resolver(ledger), dry_run=True)
    assert len(res["pinned"]) == 1
    assert store.machine == {}


# --------------------------------------------------------------------------- #
# the file fleetd actually writes
# --------------------------------------------------------------------------- #
def test_the_written_file_is_plain_readable_json(ledger):
    ml.record([("1", "m1")], now=1.0, path=ledger)
    blob = json.loads(open(ledger).read())
    assert blob["1"]["machine_id"] == "m1"


def test_an_unwritable_path_is_survivable(tmp_path):
    """fleetd calls this from its tick loop; a full disk must not stop a
    reconcile."""
    p = tmp_path / "nodir"
    p.write_text("")            # a FILE where the directory should be
    assert ml.record([("1", "m1")], now=1.0, path=str(p / "x.json")) == 1


# --------------------------------------------------------------------------- #
# the second, laptop-independent path: identity written at RENT TIME
# --------------------------------------------------------------------------- #
def test_the_identity_object_resolves_without_a_ledger_or_an_api():
    """The copy that survives the laptop. A bare bucket and nothing else."""
    iid = "48603388"
    store = _Store({})
    store.node[f"{hf.NODES}/{iid}/identity-20260825T010203Z.json"] = {
        "instance_id": iid, "machine_id": "140799", "source": "launch"}
    assert hf.identity_machine_resolver(store)(iid) == "140799"


def test_the_newest_identity_object_wins():
    iid = "1"
    store = _Store({})
    for ts, mid in (("20260101T000000Z", "old"), ("20260825T010203Z", "new")):
        store.node[f"{hf.NODES}/{iid}/identity-{ts}.json"] = {
            "instance_id": iid, "machine_id": mid}
    assert hf.identity_machine_resolver(store)(iid) == "new"


def test_the_identity_resolver_ignores_other_objects_in_the_prefix():
    """`jobs/nodes/<IID>/` also holds events and hostfacts."""
    iid = "1"
    store = _Store({})
    store.node[f"{hf.NODES}/{iid}/hostfacts/cpu-x.json"] = {"machine_id": "wrong"}
    store.node[f"{hf.NODES}/{iid}/events/box_x.json"] = {"machine_id": "wrong"}
    assert hf.identity_machine_resolver(store)(iid) is None


def test_a_store_that_raises_on_list_is_not_fatal():
    class _Angry:
        def keys(self, _p):
            raise OSError("B2 down")
    assert hf.identity_machine_resolver(_Angry())("1") is None


def test_the_full_chain_prefers_ledger_then_identity_then_api(ledger):
    iid = "1"
    store = _Store({})
    store.node[f"{hf.NODES}/{iid}/identity-20260825T010203Z.json"] = {
        "instance_id": iid, "machine_id": "from_identity"}
    api = lambda _i: "from_api"                           # noqa: E731

    chain = hf.chained_resolver(hf.ledger_machine_resolver(ledger),
                                hf.identity_machine_resolver(store), api)
    assert chain(iid) == "from_identity"                  # no ledger entry yet

    ml.record([(iid, "from_ledger")], now=1.0, path=ledger)
    chain = hf.chained_resolver(hf.ledger_machine_resolver(ledger),
                                hf.identity_machine_resolver(store), api)
    assert chain(iid) == "from_ledger"


def test_a_box_with_neither_still_falls_through_to_the_api(ledger):
    store = _Store({})
    chain = hf.chained_resolver(hf.ledger_machine_resolver(ledger),
                                hf.identity_machine_resolver(store),
                                lambda _i: "from_api")
    assert chain("unknown-box") == "from_api"


# --------------------------------------------------------------------------- #
# the B2 half: an rcat target must be a REMOTE, or the write lands on disk
# --------------------------------------------------------------------------- #

def test_the_identity_key_is_a_qualified_remote_not_a_bare_prefix():
    """`rclone rcat <path>` writes a LOCAL file when `path` names no remote,
    and exits 0 doing it — so the `hard=False` caller cannot tell the
    difference and the durable half of the mapping silently never happens.
    Shipped bare 2026-08-25 and measured the same day: 0 identity objects in
    the bucket, one stray `jobs/nodes/<IID>/…json` in the repo root, while the
    local ledger looked healthy. Grep-proof: the whole defect is the absence
    of the `b2:` scheme, so the assertion is on the scheme."""
    from vastlib.boxes import lifecycle

    key = lifecycle._IDENTITY_KEYFMT.format(bucket="bkt", iid="42",
                                            ts="20260825T010203Z")
    assert key.startswith("b2:"), (
        "identity key names no remote — rclone will write it to the CWD")
    assert key == "b2:bkt/jobs/nodes/42/identity-20260825T010203Z.json"
    # The prefix the READER scans must still be the one the WRITER produces;
    # hostfacts spells it separately because it cannot import vastlib.
    assert f"/{hf.NODES}/42/" in key


def test_recording_an_identity_sends_a_remote_path_to_rcat(monkeypatch, ledger):
    """End to end over the seam that broke: whatever `_IDENTITY_KEYFMT` says,
    the string handed to rcat has to carry the bucket from the environment.
    A test on the format alone would pass while the caller dropped it."""
    from vastlib.core import api as core_api
    from vastlib.boxes import lifecycle

    sent = []
    monkeypatch.setattr(core_api, "request_soft",
                        lambda *a, **k: (True, {"machine_id": "7"}, None))
    monkeypatch.setattr(lifecycle.b2, "_b2_rcat",
                        lambda path, body, hard=True: sent.append((path, body)))
    monkeypatch.setattr(lifecycle.machine_ledger, "record",
                        lambda *a, **k: None)
    monkeypatch.setenv("B2_BUCKET", "my-bucket")

    assert lifecycle.record_box_identity_soft("48666004") == "7"
    (path, body), = sent
    assert path.startswith("b2:my-bucket/jobs/nodes/48666004/identity-")
    assert json.loads(body)["machine_id"] == "7"


def test_no_bucket_writes_nothing_rather_than_writing_it_locally(monkeypatch):
    """The ledger is still authoritative without B2, so an unset bucket is a
    skip — never a bare relative path that rcat would resolve to disk."""
    from vastlib.core import api as core_api
    from vastlib.boxes import lifecycle

    sent = []
    monkeypatch.setattr(core_api, "request_soft",
                        lambda *a, **k: (True, {"machine_id": "7"}, None))
    monkeypatch.setattr(lifecycle.b2, "_b2_rcat",
                        lambda path, body, hard=True: sent.append(path))
    monkeypatch.setattr(lifecycle.machine_ledger, "record",
                        lambda *a, **k: None)
    monkeypatch.delenv("B2_BUCKET", raising=False)

    assert lifecycle.record_box_identity_soft("48666004") == "7"
    assert sent == []

"""Unit pins for `vastlib.boxes.health` — the guard lattice and its contracts.

Why this file exists
--------------------
`health.py` is a verbatim-with-types move out of `herdd.py` (plan §8 step 3,
ADD-ONLY: `herdd.py` keeps its live copies until step 6, and `test_guard.py` /
`test_boot_health.py` / `test_boot_sla.py` keep testing THOSE, unedited). So this
file is not a re-run of the 1,300-line `test_guard.py`. It pins the four things
that are new or newly at risk *because* of the port:

1. **The contracts the module docstring declares frozen.** The eight verdict
   STRINGS, `BoxHealth`'s field order and `._asdict()`, and the evidence dict's
   key set (`FROZEN_EVIDENCE_KEYS` below, which is the contract — a count in
   prose drifted unnoticed once already) are read by `guard --json`, fleetd's
   `state.json` and journal,
   `workload_evidence`, `_zombie_confirm_map` and the dashboard. Nothing in the
   suite asserted the *shape* of those; `test_guard.py` asserts individual
   values through them. A port is exactly when a shape breaks silently.

2. **The `GuardVerdict` unification** (plan §5, integrator ruling 2026-08-16).
   Three module-level names collapsed into one enum — `_GUARD_ZOMBIE_VERDICTS`
   -> `.is_zombie`, `_GUARD_ADVISORY_VERDICTS` -> `.is_advisory`,
   `_GUARD_VERDICT_SHORT` -> `.short`. The membership is re-asserted here
   against the historical sets written out longhand, so a future edit to the
   enum cannot quietly move a verdict between "alarm" and "destroy-relevant".
   The str-mixin equality/hash is pinned in both directions because a mixed
   frozenset (`"OK" in {GuardVerdict.OK}`) is the failure mode the manifest
   flagged and it fails silently — it returns False, it does not raise.

3. **The one sanctioned deviation.** `boot_health_watch`'s `get_instance`
   default was a DEF-TIME bind to `_get_instance`; it is now None-resolved at
   CALL time through `boxes.lifecycle`. Pinned by patching
   `lifecycle._get_instance` and proving the watcher sees the patch, which the
   old def-time bind could not have done.

4. **The cross-module edges the port introduced**, each of which is a place a
   later "cleanup" could reintroduce a copy: `_is_jobs_box` reads
   `core.models._instance_env` (it must not re-copy the `extra_env` list->dict
   walk), the three jobd probes read `storage.b2._rclone_soft`,
   `_get_instance_soft` reads `core.api.request_soft`, and the two epoch
   folders read `core.fmt._ts_to_epoch` — which is a DIFFERENT parser from this
   module's own `_iso_ftz_to_epoch`, a distinction the source warns about twice.

Everything patched here is patched in module-attribute form
(`monkeypatch.setattr(b2, "_rclone_soft", ...)`), per plan §8(b) — the same
reason the port itself never writes `from ... import fn`.

No network: `_get_instance_soft` is a GET, so conftest's mutation guard would
let it THROUGH to the real API. Its test stubs `api.request_soft` outright.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import imageref  # noqa: E402
import herdd as v  # noqa: E402
from vastlib.boxes import health  # noqa: E402
from vastlib.core import api, fmt, models  # noqa: E402
from vastlib.storage import b2  # noqa: E402

NOW = 1_000_000.0

# The eight verdict strings, written out longhand rather than referenced. If the
# port renames one, this file must fail — referencing the constant would make
# the assertion tautological.
FROZEN_VERDICTS = (
    "OK", "BOOTING", "ZOMBIE_LOADING_STALL", "ZOMBIE_NO_JOBD",
    "ZOMBIE_TICKET_UNCLAIMED", "ZOMBIE_PYHALF", "STALE_IMAGE", "LOADING_SLOW",
)

FROZEN_ZOMBIES = frozenset({
    "ZOMBIE_LOADING_STALL", "ZOMBIE_NO_JOBD",
    "ZOMBIE_TICKET_UNCLAIMED", "ZOMBIE_PYHALF"})

FROZEN_ADVISORY = frozenset({"STALE_IMAGE", "LOADING_SLOW"})

FROZEN_SHORT = {
    "ZOMBIE_LOADING_STALL": "loading-stall",
    "ZOMBIE_NO_JOBD": "jobd-dead",
    "ZOMBIE_TICKET_UNCLAIMED": "ticket-unclaimed",
    "ZOMBIE_PYHALF": "pyhalf-broken",
    "STALE_IMAGE": "stale-image",
    "LOADING_SLOW": "loading-slow",
}

# BoxHealth's field order IS the `guard --json` row order.
FROZEN_BOXHEALTH_FIELDS = ("iid", "verdict", "reason", "age_s", "machine_id",
                           "evidence")

# Read by fleetd.workload_evidence, _guard_evidence_bits, _zombie_confirm_map
# and the dashboard. Every classification path must return all of them — this
# set IS the contract (the module docstring used to carry a hand-written count,
# which read "twelve" against thirteen keys until 2026-08-21).
# `ticket_fifo_blocked` joined 2026-08-17 with the FIFO gate on rule (5): a
# consumer that sees a long `ticket_age_s` and no verdict needs the reason the
# verdict was withheld, or the evidence reads as a contradiction.
# `cpu_util`/`cpu_cores_effective` joined 2026-08-21: every other liveness
# signal is GPU- or lane-shaped, so a dedicated CPU box read as idle to all of
# them and the safety net would park it mid-run.
FROZEN_EVIDENCE_KEYS = frozenset({
    "status", "boot_age_s", "is_jobs_box", "jobd_hb_age_s", "jobd_hb_src",
    "pyhalf", "ticket_age_s", "ticket_fifo_blocked", "phase", "pull_active",
    "pull_bytes", "image_state", "image_reason",
    "cpu_util", "cpu_cores_effective"})


@pytest.fixture(autouse=True)
def _pin_guard_knobs(monkeypatch):
    """Every knob this module reads, pinned to its shipped default.

    They are read at CALL time from env > herdd.yaml > constant, so a
    developer's `.env` or a personal `~/.config/herdd/herdd.yaml` would
    otherwise silently move the band boundaries these tests assert on."""
    for name, value in (("GUARD_LOADING_DEADLINE_S", "1500"),
                        ("GUARD_LOADING_HARD_S", "3600"),
                        ("GUARD_ENVSETUP_DEADLINE_S", "900"),
                        ("GUARD_JOBD_STALE_S", "600"),
                        ("GUARD_TICKET_DEADLINE_S", "1500")):
        monkeypatch.setenv(name, value)


# --------------------------------------------------------------------------- #
# 1. Frozen contracts
# --------------------------------------------------------------------------- #

def test_the_eight_verdict_strings_are_frozen():
    assert [health.GUARD_OK, health.GUARD_BOOTING,
            health.GUARD_ZOMBIE_LOADING_STALL, health.GUARD_ZOMBIE_NO_JOBD,
            health.GUARD_ZOMBIE_TICKET_UNCLAIMED, health.GUARD_ZOMBIE_PYHALF,
            health.GUARD_STALE_IMAGE,
            health.GUARD_LOADING_SLOW] == list(FROZEN_VERDICTS)


def test_no_verdict_string_has_a_second_definition_in_the_launcher():
    """Was a byte-for-byte agreement check between two live copies (the
    add-only phase). Step 6d left one: `herdd.py` is a thin launcher that
    re-exports the subset its external consumers reach — the same objects — and
    does not define GUARD_OK or GUARD_ZOMBIE_TICKET_UNCLAIMED at all.

    The strings are a WIRE contract (`guard --json`, fleetd's `state.json` and
    its journal all carry them), so what has to stay impossible is a SECOND
    definition drifting from this module's. `test_the_eight_verdict_strings_are_frozen`
    above pins the values themselves against `FROZEN_VERDICTS`.
    """
    for name in ("GUARD_OK", "GUARD_BOOTING", "GUARD_ZOMBIE_LOADING_STALL",
                 "GUARD_ZOMBIE_NO_JOBD", "GUARD_ZOMBIE_TICKET_UNCLAIMED",
                 "GUARD_ZOMBIE_PYHALF", "GUARD_STALE_IMAGE",
                 "GUARD_LOADING_SLOW"):
        flat = getattr(v, name, None)
        assert flat is None or flat == getattr(health, name), (
            f"{name}: the launcher carries a DIFFERENT verdict string — the "
            f"wire contract has forked")


def test_boxhealth_field_order_and_asdict_are_frozen():
    h = health.BoxHealth(7, health.GUARD_OK, "why", 3, 99, {"k": 1})
    assert health.BoxHealth._fields == FROZEN_BOXHEALTH_FIELDS
    assert tuple(h._asdict()) == FROZEN_BOXHEALTH_FIELDS
    assert h._asdict() == {"iid": 7, "verdict": "OK", "reason": "why",
                           "age_s": 3, "machine_id": 99, "evidence": {"k": 1}}
    # tuple-compatible: positional construction, indexing and bare-tuple
    # equality all still work, which is what the collections.namedtuple ->
    # typing.NamedTuple change had to preserve.
    assert h[1] == "OK"
    assert tuple(h) == (7, "OK", "why", 3, 99, {"k": 1})


# `test_boxhealth_field_order_matches_the_live_herdd_copy` was here.
# `v.BoxHealth` is the launcher's identity re-export of this class since plan
# §8 step 6d, so it compared `_fields` with itself; the frozen field tuple is
# asserted against `FROZEN_BOXHEALTH_FIELDS` in the test above, which is the
# statement it stood in for.


@pytest.mark.parametrize("instance,jobs,kwargs", [
    ({"id": 1, "actual_status": "loading", "start_date": NOW - 10}, (), {}),
    ({"id": 2, "actual_status": "loading", "start_date": NOW - 9000}, (), {}),
    ({"id": 3, "actual_status": "exited", "start_date": NOW - 10}, (), {}),
    ({"id": 4, "actual_status": "running", "start_date": NOW - 10}, (), {}),
    ({"id": 5, "actual_status": "running", "start_date": NOW - 10_000},
     ({"job_id": "j"},), {}),
    ({"id": 6, "actual_status": "running", "start_date": NOW - 10},
     ({"job_id": "j"},), {"jobd_pyhalf": True}),
    ({"id": 7, "actual_status": "running", "start_date": NOW - 10},
     ({"job_id": "j"},), {"jobd_hb_epoch": NOW - 1}),
    ({}, (), {}),
])
def test_every_classification_path_returns_all_evidence_keys(
        instance, jobs, kwargs):
    h = health.classify_box_health(instance, jobs=jobs, now=NOW, **kwargs)
    assert set(h.evidence) == FROZEN_EVIDENCE_KEYS
    assert h.verdict in FROZEN_VERDICTS


# --------------------------------------------------------------------------- #
# 2. The GuardVerdict unification
# --------------------------------------------------------------------------- #

def test_enum_covers_exactly_the_eight_verdicts():
    assert [m.value for m in health.GuardVerdict] == list(FROZEN_VERDICTS)


def test_each_member_value_is_the_very_constant_object():
    """Not merely equal — the SAME object, so the two can never drift."""
    assert health.GuardVerdict.OK.value is health.GUARD_OK
    assert health.GuardVerdict.LOADING_SLOW.value is health.GUARD_LOADING_SLOW
    assert (health.GuardVerdict.ZOMBIE_PYHALF.value
            is health.GUARD_ZOMBIE_PYHALF)


def test_a_member_equals_and_hashes_like_its_bare_string_both_ways():
    """The mixed-frozenset failure mode is SILENT — it returns False.

    `h.get("verdict") in {GuardVerdict...}` is exactly the shape three modules
    write, so equality is not enough: the hash has to agree too."""
    for s in FROZEN_VERDICTS:
        m = health.GuardVerdict(s)
        assert m == s and s == m
        assert hash(m) == hash(s)
        assert s in {m}          # string probe, member-keyed set
        assert m in {s}          # member probe, string-keyed set
        assert {m: 1}[s] == 1    # member key, string lookup


def test_is_zombie_is_exactly_the_historical_zombie_set():
    got = {m.value for m in health.GuardVerdict if m.is_zombie}
    assert got == FROZEN_ZOMBIES


def test_is_advisory_is_exactly_the_historical_advisory_set():
    got = {m.value for m in health.GuardVerdict if m.is_advisory}
    assert got == FROZEN_ADVISORY


def test_zombie_and_advisory_are_disjoint():
    """An advisory alarms; it never licenses a destroy. `guard --fix` acts on
    the zombie set alone, and STALE_IMAGE/LOADING_SLOW landing in it would make
    a healthy box destroyable — the 46682177/46682313 shape."""
    zombies = {m for m in health.GuardVerdict if m.is_zombie}
    advisory = {m for m in health.GuardVerdict if m.is_advisory}
    assert not (zombies & advisory)


def test_pyhalf_is_a_zombie_by_membership():
    """It SCREAMS with the other zombies on purpose. The "alarm, don't destroy"
    half is `parked_lifecycle.zombie_action` answering by name, and lives in
    boxes/reap.py — not in this lattice."""
    assert health.GuardVerdict.ZOMBIE_PYHALF.is_zombie


def test_membership_matches_the_live_herdd_frozensets():
    for m in health.GuardVerdict:
        assert m.is_zombie == (m.value in v._GUARD_ZOMBIE_VERDICTS), m.value
        assert m.is_advisory == (m.value in v._GUARD_ADVISORY_VERDICTS), m.value


def test_short_tags_match_and_ok_booting_fall_back_to_the_verdict():
    """The original dict had no key for OK or BOOTING and every caller wrote
    `.get(v, v)`; `.short` reproduces that fallback rather than inventing tags."""
    for s, tag in FROZEN_SHORT.items():
        assert health.GuardVerdict(s).short == tag
    assert health.GuardVerdict.OK.short == "OK"
    assert health.GuardVerdict.BOOTING.short == "BOOTING"
    assert {m.value: m.short for m in health.GuardVerdict
            if m.short != m.value} == FROZEN_SHORT


def test_short_matches_the_live_herdd_table():
    for m in health.GuardVerdict:
        assert m.short == v._GUARD_VERDICT_SHORT.get(m.value, m.value), m.value


def test_of_round_trips_and_degrades_to_none_on_an_unknown_verdict():
    for s in FROZEN_VERDICTS:
        assert health.GuardVerdict.of(s) is health.GuardVerdict(s)
    # A peer on a newer/older build must not crash a health tick.
    assert health.GuardVerdict.of("ZOMBIE_FROM_THE_FUTURE") is None
    assert health.GuardVerdict.of(None) is None
    assert health.GuardVerdict.of(17) is None


def test_string_side_predicates_answer_for_persisted_verdicts():
    for s in FROZEN_VERDICTS:
        assert health.verdict_is_zombie(s) is (s in FROZEN_ZOMBIES)
        assert health.verdict_is_advisory(s) is (s in FROZEN_ADVISORY)
        assert health.verdict_short(s) == FROZEN_SHORT.get(s, s)
    for junk in (None, "", "NOPE", 17):
        assert health.verdict_is_zombie(junk) is False
        assert health.verdict_is_advisory(junk) is False


def test_a_classified_verdict_is_a_plain_str_not_a_member():
    """`BoxHealth.verdict` stays `str`: a str-mixin Enum's __str__/__format__
    differ between the 3.10 floor and fleetd's 3.13 venv, so an f-string of a
    member is not version-stable while the raw string is."""
    h = health.classify_box_health(
        {"id": 1, "actual_status": "exited"}, now=NOW)
    assert type(h.verdict) is str


# --------------------------------------------------------------------------- #
# 3. Pull-progress parsing and the throughput sampler
# --------------------------------------------------------------------------- #

_PULL = ("aaaaaa: Downloading [====>    ]  1.5GB/3.0GB\n"
         "bbbbbb: Already exists\n"
         "cccccc: Extracting [=>       ]  0.5GB/2.0GB\n")


def test_parse_pull_progress_accepts_none_and_a_bare_dict_for_prev():
    """`boxstate.py` calls this with `{}`, the sampler with its own last return.
    `prev = prev or {}` has always made the two identical; keep it that way."""
    assert (health.parse_pull_progress(_PULL, None)
            == health.parse_pull_progress(_PULL, {}))


def test_parse_pull_progress_decimal_si_and_phases():
    got = health.parse_pull_progress(_PULL, None)
    assert got["downloading"] is True
    assert got["extracting"] is True
    assert got["layers"]["bbbbbb"] == 0          # cached: free, not slow
    assert got["total_bytes"] == int(1.5e9) + int(2.0e9)


def test_parse_pull_progress_byte_totals_never_decrease():
    """vast truncates status_msg to a tail window, so completed layers scroll
    out of view; the fold carries prev forward."""
    first = health.parse_pull_progress(_PULL, None)
    second = health.parse_pull_progress("dddddd: Downloading [=>] 1.0GB/9.0GB",
                                        first)
    assert second["total_bytes"] >= first["total_bytes"]
    assert "aaaaaa" in second["layers"]


# `test_parse_pull_progress_matches_the_live_herdd_copy` folded the same pull
# line onto three `prev` shapes through both copies. One copy since step 6d;
# the fold's accumulate/merge behavior is asserted by value above.


def _loading(msg, iid=1):
    return {"id": iid, "actual_status": "loading", "status_msg": msg}


def test_sampler_condemns_a_starved_pull_but_not_an_extract_only_window():
    s = health.BootThroughputSampler(min_mbps=5, window_s=100, deadline_s=1e9,
                                     start_t=0.0)
    assert s.feed(_loading("aaaaaa: Downloading [=>] 1.0MB/900.0MB"), 0.0) is None
    verdict = None
    for t in range(10, 201, 10):
        verdict = s.feed(_loading("aaaaaa: Downloading [=>] 1.0MB/900.0MB"),
                         float(t))
    assert verdict == "slow"
    assert s.phase == "downloading"

    e = health.BootThroughputSampler(min_mbps=5, window_s=100, deadline_s=1e9,
                                     start_t=0.0)
    for t in range(0, 201, 10):
        assert e.feed(_loading("cccccc: Extracting [=>] 0.5GB/2.0GB"),
                      float(t)) is None
    assert e.phase == "extracting"


def test_sampler_running_gone_and_deadline():
    s = health.BootThroughputSampler(min_mbps=5, window_s=100, deadline_s=50,
                                     start_t=0.0)
    assert s.feed({"actual_status": "running"}, 1.0) == "running"
    assert s.feed({"actual_status": "exited"}, 1.0) == "gone"
    assert s.feed(None, 1.0) == "gone"
    assert s.feed(_loading(""), 100.0) == "deadline"


# --------------------------------------------------------------------------- #
# 4. boot_health_watch — the sanctioned deviation
# --------------------------------------------------------------------------- #

def test_boot_health_watch_resolves_its_default_reader_at_call_time(monkeypatch):
    """THE deviation. In `herdd.py` the default was `get_instance=_get_instance`,
    bound at def time; here it is None, resolved through
    `boxes.lifecycle._get_instance` when the call happens. Patching the
    lifecycle module must therefore steer the watcher — which a def-time bind
    could not do, and which is the whole reason plan §8(b) forbids them."""
    from vastlib.boxes import lifecycle

    seen = []

    def fake_get_instance(iid):
        seen.append(iid)
        return {"actual_status": "running"}

    monkeypatch.setattr(lifecycle, "_get_instance", fake_get_instance)
    got = health.boot_health_watch(4242, min_mbps=5, window_s=10, poll_s=0,
                                   deadline_s=100, now=lambda: 0.0,
                                   sleep=lambda _s: None)
    assert got == "running"
    assert seen == [4242]


def test_boot_health_watch_failed_poll_is_no_sample_but_honors_the_deadline():
    clock = {"t": 0.0}

    def now():
        clock["t"] += 10.0
        return clock["t"]

    def raiser(_iid):
        raise RuntimeError("api blip")

    assert health.boot_health_watch(1, min_mbps=5, window_s=10, poll_s=0,
                                    deadline_s=30, get_instance=raiser,
                                    now=now, sleep=lambda _s: None) == "deadline"


def test_build_throughput_observer_only_surfaces_slow(monkeypatch):
    monkeypatch.setenv("BOOT_MIN_MBPS", "5")
    monkeypatch.setenv("BOOT_MBPS_WINDOW_S", "100")
    clock = {"t": 0.0}

    def now():
        clock["t"] += 10.0
        return clock["t"]

    obs = health.build_throughput_observer(
        get_instance=lambda _i: _loading("aaaaaa: Downloading [=>] 1.0MB/9.0GB"),
        now=now)
    out = [obs(7) for _ in range(30)]
    assert out[0] is None                       # window not full yet
    hits = [o for o in out if o is not None]
    assert hits and all(o["verdict"] == "slow" for o in hits)
    assert hits[0]["window_s"] == 100 and hits[0]["phase"] == "downloading"


# --------------------------------------------------------------------------- #
# 5. The cross-module edges the port introduced
# --------------------------------------------------------------------------- #

def test_get_instance_soft_goes_through_core_api_request_soft(monkeypatch):
    """A GET, so conftest's mutation guard would let it reach the real API —
    stub the module attribute instead of relying on the guard."""
    calls = []

    def fake(method, path, **kw):
        calls.append((method, path, kw.get("retries")))
        return (True, {"instances": {"id": 9}}, None)

    monkeypatch.setattr(api, "request_soft", fake)
    assert health._get_instance_soft(9) == {"id": 9}
    assert calls == [("GET", "v0/instances/9/", 2)]

    monkeypatch.setattr(api, "request_soft",
                        lambda *a, **k: (False, None, "HTTP 404 gone"))
    assert health._get_instance_soft(9) is None


def test_is_jobs_box_reads_models_instance_env_and_does_not_recopy_it(
        monkeypatch):
    # The wire form is a list of [K, V] pairs; only models._instance_env knows
    # that, so a re-copied walk would have to reproduce it.
    assert health._is_jobs_box({"extra_env": [["CRED_ROLE", "jobs"]]}, [])
    assert not health._is_jobs_box({"extra_env": [["CRED_ROLE", "train"]]}, [])
    # a folded ticket makes it a jobs box with no marker at all (ssh job attach)
    assert health._is_jobs_box({}, [{"job_id": "j"}])
    assert not health._is_jobs_box({}, [])

    # ...and it is genuinely delegating: steer the shared accessor and the
    # answer moves with it.
    monkeypatch.setattr(models, "_instance_env",
                        lambda _i: {"CRED_ROLE": "jobs"})
    assert health._is_jobs_box({}, [])


def _rclone_stub(replies):
    """(argv-prefix -> (rc, stdout, stderr)) fake for storage.b2._rclone_soft."""
    def run(args, **_kw):
        for key, reply in replies.items():
            if args[0] == key[0] and key[1] in args[1]:
                return reply
        return (1, "", "no stub")
    return run


def test_jobd_probes_read_b2_through_storage_rclone_soft(monkeypatch):
    monkeypatch.setenv("B2_BUCKET", "bkt")
    line = "RUNNING 1 2026-08-14T04:51:46Z staging=0 pyhalf=broken"
    monkeypatch.setattr(b2, "_rclone_soft", _rclone_stub({
        ("cat", "JOBD_STATUS"): (0, line, ""),
        ("lsf", "/jobs/nodes/"): (0, "JOBD_STATUS\n", ""),
    }))
    assert health._jobd_status_line_soft(5) == line
    assert health._jobd_status_pyhalf_soft(5) is True
    assert health._jobd_ever_stamped(5) is True
    assert health._jobd_heartbeat_epoch_soft(5) == health._iso_ftz_to_epoch(
        "2026-08-14T04:51:46Z")


def test_jobd_probes_are_soft_on_an_unreadable_read(monkeypatch):
    monkeypatch.setenv("B2_BUCKET", "bkt")
    monkeypatch.setattr(b2, "_rclone_soft", lambda *a, **k: (1, "", "boom"))
    assert health._jobd_status_line_soft(5) is None
    assert health._jobd_heartbeat_epoch_soft(5) is None
    assert health._scratch_probe_soft(5) is None
    # tri-state: unreadable is None (NOT sweepable), not False (provably absent)
    assert health._jobd_ever_stamped(5) is None


def test_jobd_probes_return_none_without_a_bucket(monkeypatch):
    monkeypatch.delenv("B2_BUCKET", raising=False)

    def explode(*_a, **_k):
        raise AssertionError("must not touch B2 without a bucket")

    monkeypatch.setattr(b2, "_rclone_soft", explode)
    assert health._jobd_status_line_soft(5) is None
    assert health._jobd_ever_stamped(5) is None
    assert health._scratch_probe_soft(5) is None


def test_pyhalf_is_tri_state_and_only_true_acts():
    assert health.jobd_status_pyhalf("RUNNING 1 ... pyhalf=broken") is True
    assert health.jobd_status_pyhalf("RUNNING 1 ... pyhalf=ok") is False
    assert health.jobd_status_pyhalf("RUNNING 1 2026-08-14T04:51:46Z") is None
    assert health.jobd_status_pyhalf(None) is None
    # `fleetd.pyhalf_broken` delegated to the herdd copy and was repointed at
    # step 6; the four-line agreement check that stood in for that repoint went
    # with the thinning (one parser now). What is still worth pinning is that
    # the daemon-side predicate reads THIS tri-state and only acts on True.
    import fleetd
    for line in ("x pyhalf=broken", "x pyhalf=ok", "x", None):
        assert fleetd.pyhalf_broken(line) is health.jobd_status_pyhalf(line)


def test_iso_ftz_and_ts_to_epoch_are_different_parsers():
    """Colon-BEARING %FT%TZ (jobd's heartbeat) vs the colon-free runmeta form.
    Each returns None on the other's input; merging them is called out twice in
    the source and would silently zero one of the two clocks."""
    ftz, compact = "2026-08-14T04:51:46Z", "20260814T045146000Z"
    assert health._iso_ftz_to_epoch(ftz) == fmt._ts_to_epoch(compact)
    assert health._iso_ftz_to_epoch(compact) is None
    assert fmt._ts_to_epoch(ftz) is None


def test_jobd_status_hb_epoch_scans_for_the_stamp_not_a_fixed_index():
    """jobd's STATE field is sometimes two tokens and extras trail after."""
    want = health._iso_ftz_to_epoch("2026-08-14T04:51:46Z")
    for line in ("BOOT 2026-08-14T04:51:46Z",
                 "RUNNING 3 2026-08-14T04:51:46Z staging=0 pyhalf=ok",
                 "  RUNNING 3 2026-08-14T04:51:46Z  "):
        assert health._jobd_status_hb_epoch(line) == want
    assert health._jobd_status_hb_epoch("RUNNING 3") is None


def test_job_liveness_epoch_takes_the_max_of_three_proofs():
    view = {"last_heartbeat_ts": "20260814T045146000Z",
            "last_checkpoint_ts": "20260814T040000000Z",
            "last_event_ts": "20260814T050000000Z"}
    assert health._job_liveness_epoch(view) == fmt._ts_to_epoch(
        "20260814T050000000Z")
    assert health._job_liveness_epoch({}) is None
    assert health._job_liveness_epoch(None) is None


def test_guard_unclaimed_ticket_age_only_counts_submitted_tickets():
    ep = fmt._ts_to_epoch("20260814T045146000Z")
    jobs = [{"display_status": "running", "last_event_ts": "20260814T040000000Z"},
            {"display_status": "submitted", "last_event_ts": "20260814T045146000Z"}]
    assert health._guard_unclaimed_ticket_age(jobs, ep + 90) == pytest.approx(90)
    assert health._guard_unclaimed_ticket_age([], 0.0) is None
    assert health._guard_unclaimed_ticket_age(None, 0.0) is None


def test_unclaimed_ticket_age_is_the_OLDEST_ticket():
    """Two queued tickets -> the age reported is the one that has waited
    longest. Pinned because the helper was rewritten around
    `_guard_oldest_submitted_epoch` (min epoch = max age) and an argmax/argmin
    slip here silently under-reports every multi-ticket queue."""
    jobs = [{"display_status": "submitted", "last_event_ts": "20260814T040000000Z"},
            {"display_status": "submitted", "last_event_ts": "20260814T045146000Z"}]
    now = fmt._ts_to_epoch("20260814T050000000Z")
    assert health._guard_unclaimed_ticket_age(jobs, now) == pytest.approx(3600)


def test_newest_running_claim_epoch_dates_the_claim_not_the_last_event():
    """`started_at` (fold min of claimed/started), never `last_event_ts` — the
    latter moves with every heartbeat, so using it would make a long-running
    job look freshly claimed and suppress a genuine FIFO skip forever."""
    jobs = [{"display_status": "running", "started_at": "20260814T040000000Z",
             "last_event_ts": "20260814T050000000Z"},
            {"display_status": "running", "started_at": "20260814T041000000Z",
             "last_event_ts": "20260814T050000000Z"},
            {"display_status": "submitted", "last_event_ts": "20260814T045000000Z"}]
    assert health._guard_newest_running_claim_epoch(jobs) == fmt._ts_to_epoch(
        "20260814T041000000Z")
    assert health._guard_newest_running_claim_epoch([]) is None
    assert health._guard_newest_running_claim_epoch(None) is None
    # `interrupted` is a claimed job on a box that is NOT live — it is not
    # occupying this box's cards, so it cannot block anything here.
    assert health._guard_newest_running_claim_epoch(
        [{"display_status": "interrupted", "started_at": "20260814T040000000Z"}]) is None


@pytest.mark.parametrize("jobs,blocked,why", [
    ([], False, "empty fold: nothing queued and nothing running"),
    ([{"display_status": "submitted", "last_event_ts": "20260814T040000000Z"}],
     False, "queued with NOTHING running — the genuine idle-daemon shape"),
    ([{"job_id": "20260814T040000-a", "display_status": "running",
       "started_at": "20260814T040500000Z"},
      {"display_status": "submitted", "last_event_ts": "20260814T041000000Z"}],
     True, "queued after the running job was SUBMITTED — ordinary FIFO wait"),
    ([{"job_id": "20260814T041000-b", "display_status": "running",
       "started_at": "20260814T041500000Z"},
      {"display_status": "submitted", "last_event_ts": "20260814T040000000Z"}],
     False, "the RUNNING job was submitted after the ticket — queue jumped"),
    ([{"job_id": "20260814T040000-a", "display_status": "running",
       "started_at": "20260814T040000000Z"}],
     True, "running, nothing queued — vacuously blocked, and rule 5 has no "
           "ticket age to fire on anyway"),
    ([{"display_status": "running", "started_at": "20260814T040000000Z"},
      {"display_status": "submitted", "last_event_ts": "20260814T040000000Z"}],
     True, "running but UNDATABLE (no JOB_ID to order by): an unprovable skip "
           "must not raise an alarm nobody can act on"),
    # The live regression. Ten arms batch-submitted inside 60 s and executed in
    # perfect FIFO order: every claim after the first post-dates every submit,
    # so a started_at-vs-submit comparison flags forever from job #2 onward.
    ([{"job_id": "20260818T052810-a", "display_status": "running",
       "started_at": "20260818T065500000Z"},
      {"job_id": "20260818T052815-b", "display_status": "submitted",
       "last_event_ts": "20260818T052815000Z"},
      {"job_id": "20260818T052820-c", "display_status": "submitted",
       "last_event_ts": "20260818T052820000Z"}],
     True, "box 47999495, 2026-08-18: batch submit + serial FIFO execution is "
           "the recommended shape and must stay silent — the running job was "
           "SUBMITTED first even though it was CLAIMED an hour later"),
])
def test_ticket_fifo_blocked_truth_table(jobs, blocked, why):
    assert health._guard_ticket_fifo_blocked(jobs) is blocked, why


def test_fifo_order_reads_submit_time_not_claim_time():
    """FIFO orders by SUBMIT time, so both sides of the comparison must be
    submit times. Same claim ordering, opposite submit ordering => opposite
    verdicts; if the predicate read `started_at` these two would agree."""
    claimed_late = {"display_status": "running", "started_at": "20260818T060000000Z"}
    ticket = {"display_status": "submitted", "last_event_ts": "20260818T052815000Z"}
    assert health._guard_ticket_fifo_blocked(
        [{**claimed_late, "job_id": "20260818T052810-a"}, ticket]) is True
    assert health._guard_ticket_fifo_blocked(
        [{**claimed_late, "job_id": "20260818T052820-c"}, ticket]) is False


def test_newest_running_submit_epoch_sources_the_job_id():
    assert health._guard_newest_running_submit_epoch(
        [{"job_id": "20260814T040000-a", "display_status": "running"},
         {"job_id": "20260814T041000-b", "display_status": "running"},
         {"job_id": "20260814T045000-c", "display_status": "submitted"}]
    ) == fmt._ts_to_epoch("20260814T041000000Z")
    assert health._guard_newest_running_submit_epoch([]) is None
    assert health._guard_newest_running_submit_epoch(None) is None
    # JOB_ID_RE does not require the timestamp prefix; an undatable id drops out.
    assert health._guard_newest_running_submit_epoch(
        [{"job_id": "legacy-job", "display_status": "running"}]) is None


# --------------------------------------------------------------------------- #
# 6. classify_box_health / gather_fleet_health band parity with the live copy
# --------------------------------------------------------------------------- #

_PULLING = "aaaaaa: Downloading [=>] 1.0GB/9.0GB"

_CASES = [
    # loading, inside the nominal deadline
    ({"id": 1, "actual_status": "loading", "start_date": NOW - 60}, (), {},
     "BOOTING"),
    # loading, past the deadline, pull still advancing, inside the hard bound
    ({"id": 2, "actual_status": "loading", "start_date": NOW - 2000,
      "status_msg": _PULLING}, (), {}, "LOADING_SLOW"),
    # loading, past the deadline, no pull activity at all
    ({"id": 3, "actual_status": "loading", "start_date": NOW - 2000}, (), {},
     "ZOMBIE_LOADING_STALL"),
    # loading, past the HARD bound even while pulling
    ({"id": 4, "actual_status": "loading", "start_date": NOW - 9000,
      "status_msg": _PULLING}, (), {}, "ZOMBIE_LOADING_STALL"),
    # not live
    ({"id": 5, "actual_status": "exited", "start_date": NOW - 10}, (), {}, "OK"),
    # running, not a jobs box
    ({"id": 6, "actual_status": "running", "start_date": NOW - 10_000}, (), {},
     "OK"),
    # the box's own confession outranks every inference below it
    ({"id": 7, "actual_status": "running", "start_date": NOW - 10},
     ({"job_id": "j"},), {"jobd_pyhalf": True}, "ZOMBIE_PYHALF"),
    # jobs box, jobd never stamped, past the env-setup deadline
    ({"id": 8, "actual_status": "running", "start_date": NOW - 5000},
     ({"job_id": "j"},), {}, "ZOMBIE_NO_JOBD"),
    # jobs box, jobd never stamped, still inside it
    ({"id": 9, "actual_status": "running", "start_date": NOW - 60},
     ({"job_id": "j"},), {}, "BOOTING"),
    # jobd heartbeat stale
    ({"id": 10, "actual_status": "running", "start_date": NOW - 10_000},
     ({"job_id": "j"},), {"jobd_hb_epoch": NOW - 5000, "jobd_hb_src": "jobs"},
     "ZOMBIE_NO_JOBD"),
    # jobd fresh, a submitted ticket sat unclaimed past the deadline
    ({"id": 11, "actual_status": "running", "start_date": NOW - 10_000},
     ({"job_id": "j", "display_status": "submitted",
       "last_event_ts": "20260814T045146000Z"},),
     {"jobd_hb_epoch": "FRESH", "now": None}, "ZOMBIE_TICKET_UNCLAIMED"),
    # everything fine
    ({"id": 12, "actual_status": "running", "start_date": NOW - 10_000},
     ({"job_id": "j"},), {"jobd_hb_epoch": NOW - 1}, "OK"),
]


def _case_call(kwargs):
    """Resolve a case's two clock sentinels into concrete numbers.

    `now=None` means "date this case off the ticket stamp in `jobs`", and
    `jobd_hb_epoch="FRESH"` means "one second before that `now`" — the pair
    exists so the ticket case can sit on a B2-derived clock without hardcoding
    a heartbeat that the shifted clock would make stale."""
    kwargs = dict(kwargs)
    now = kwargs.pop("now", NOW)
    if now is None:
        now = fmt._ts_to_epoch("20260814T045146000Z") + 5000
    if kwargs.get("jobd_hb_epoch") == "FRESH":
        kwargs["jobd_hb_epoch"] = now - 1
    return now, kwargs


@pytest.mark.parametrize("instance,jobs,kwargs,want", _CASES)
def test_classify_box_health_bands(instance, jobs, kwargs, want):
    now, kwargs = _case_call(kwargs)
    got = health.classify_box_health(instance, jobs=jobs, now=now, **kwargs)
    assert got.verdict == want


# `test_classify_box_health_is_identical_to_the_live_herdd_copy` ran the whole
# `_CASES` table through both classifiers and compared `_asdict()` — the
# strongest form of plan §7.4 ("a port that needs an expectation change is a
# found drift") while there were two. Step 6d left one: `v.classify_box_health`
# is this function. The same table still runs against the verdict bands in
# `test_classify_box_health_bands` above, and the evidence fields have their own
# tests below.


def test_stale_image_advisory_only_ever_overlays_ok(monkeypatch):
    """A destroy-relevant zombie must never be masked by an advisory, and the
    state lands in evidence either way so a stale zombie stays legible."""
    ok_box = {"id": 1, "actual_status": "running", "start_date": NOW - 10}
    h = health.classify_box_health(ok_box, now=NOW,
                                   image_state=imageref.IMG_STALE,
                                   image_reason="tag moved")
    assert h.verdict == "STALE_IMAGE"
    assert h.evidence["image_state"] == imageref.IMG_STALE

    zombie = {"id": 2, "actual_status": "loading", "start_date": NOW - 9000}
    z = health.classify_box_health(zombie, now=NOW,
                                   image_state=imageref.IMG_STALE,
                                   image_reason="tag moved")
    assert z.verdict == "ZOMBIE_LOADING_STALL"
    assert z.evidence["image_state"] == imageref.IMG_STALE


def test_fleet_image_states_resolves_the_R2_DEFAULT_IMAGE(monkeypatch):
    """THE 2026-08-19 DEFECT, at fleetd's end. `_fleet_image_states` kept its
    own $GITLAB_REGISTRY copy of "is this ours", so after the R2 cutover it
    looked up nothing and classified every box `not_applicable` — the health
    tick's staleness signal was dead for the fleet default image."""
    monkeypatch.delenv("GITLAB_REGISTRY", raising=False)
    img = f"{imageref.R2_REGISTRY_HOST}/train:t215-latest"
    seen = []

    def fake(m, **_k):
        seen.append(m)
        return ("sha256:" + "b" * 64, "miss")

    monkeypatch.setattr(imageref, "resolve_tag_digest_ttl", fake)
    out = health._fleet_image_states([
        {"id": 31, "image_uuid": img,
         "extra_env": [[imageref.IMAGE_DIGEST_ENV, "sha256:" + "a" * 64]]}])
    assert seen == [img], "the R2 tag must actually be looked up"
    assert out["31"][0] == imageref.IMG_STALE, out["31"]


def test_fleet_image_states_still_skips_a_foreign_registry(monkeypatch):
    """The widening must stay a widening: a public-registry serve box costs
    zero registry calls and classifies not_applicable."""
    monkeypatch.delenv("GITLAB_REGISTRY", raising=False)

    def explode(*_a, **_k):
        raise AssertionError("a foreign registry must cost no lookup")

    monkeypatch.setattr(imageref, "resolve_tag_digest_ttl", explode)
    out = health._fleet_image_states([
        {"id": 32, "image_uuid": "vllm/vllm-openai:latest",
         "extra_env": [[imageref.IMAGE_DIGEST_ENV, "sha256:" + "a" * 64]]}])
    assert out["32"][0] == imageref.IMG_NOT_APPLICABLE


def test_fleet_jobd_hb_epoch_skips_the_b2_reads_on_a_fresh_fold(monkeypatch):
    """A jobd-written job event already proves the daemon is alive; the whole
    point of the fold is that a healthy fleet costs zero extra network."""
    def explode(*_a, **_k):
        raise AssertionError("fresh fold must not read B2")

    monkeypatch.setattr(health, "_jobd_heartbeat_epoch_soft", explode)
    monkeypatch.setattr(health, "_jobd_status_pyhalf_soft", explode)
    ep = fmt._ts_to_epoch("20260814T045146000Z")
    got = health._fleet_jobd_hb_epoch(
        1, [{"last_event_ts": "20260814T045146000Z"}], ep + 10)
    assert got == (ep, "jobs", None)


def test_fleet_jobd_hb_epoch_takes_the_max_never_the_min(monkeypatch):
    """A missed zombie costs dollars and is recoverable; a FALSE zombie parks a
    run that may not be. The weak marker can only move the epoch FORWARD."""
    ep = fmt._ts_to_epoch("20260814T045146000Z")
    monkeypatch.setattr(health, "_jobd_heartbeat_epoch_soft", lambda _i: ep + 999)
    monkeypatch.setattr(health, "_jobd_status_pyhalf_soft", lambda _i: None)
    best, src, pyh = health._fleet_jobd_hb_epoch(
        1, [{"last_event_ts": "20260814T045146000Z"}], ep + 5000)
    assert best == ep + 999 and src == "jobs" and pyh is None


def test_gather_fleet_health_returns_dicts_and_never_raises(monkeypatch):
    monkeypatch.setattr(health, "_fleet_image_states", lambda _i: {})

    def explode(*_a, **_k):
        raise RuntimeError("B2 down")

    monkeypatch.setattr(health, "_fleet_jobd_hb_epoch", explode)
    out = health.gather_fleet_health(
        [{"id": 21, "actual_status": "running", "start_date": NOW - 10}],
        {"21": [{"job_id": "j"}]}, now=NOW)
    row = out["21"]
    assert isinstance(row, dict)
    assert set(row) == set(FROZEN_BOXHEALTH_FIELDS)
    # the swallow degrades pyhalf to unknown, never to broken
    assert row["evidence"]["pyhalf"] is None
    assert row["verdict"] in FROZEN_VERDICTS


def test_gather_fleet_health_on_an_empty_fleet_touches_nothing():
    assert health.gather_fleet_health([], {}, now=NOW) == {}
    assert health.gather_fleet_health(None, None, now=NOW) == {}

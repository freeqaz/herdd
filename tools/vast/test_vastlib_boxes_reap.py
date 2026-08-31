"""`vastlib.boxes.reap` — the two ledgers, the six knobs, and the policy driver.

Why this file exists
--------------------
`cmd_reap` is the one function in this package a machine runs unattended against
a live fleet: `herdd-reaper.timer` executes `herdd.py reap -y` every 15
minutes and DESTROYS boxes. Nothing here may reach a box, so every test drives
the module with `lifecycle._instances` and every mutation primitive stubbed by
module attribute, and every ledger write redirected into `tmp_path` by patching
the module attribute (`reap._IDLE_LEDGER`), never the environment — the paths
are read from `os.environ` at IMPORT time, which is itself asserted below.

The three things worth failing a build over
-------------------------------------------
1. **The ledger PATH formula and both KEY SHAPES.** They are byte-frozen. A
   changed path does not error; it silently resets every live box's clock on the
   workstation that runs the timer — every parked box restarts its 2 h fuse at
   0, every zombie restarts its 900 s confirmation. `idle-ledger.json` is
   `{iid: float}`; `zombie-ledger.json` is `{iid: {first, verdict, pull, hb,
   inet, disk}}`, the shape `test_guard.py::_seed` builds by hand.
2. **The knob asymmetry.** Five reap knobs are read straight from `os.environ`
   at run time; only `REAP_ZOMBIE_CONFIRM_S` goes through
   `config._boot_knob`'s CLI > env > yaml > constant precedence. Porting the
   five onto `_boot_knob` would be a behavior change (plan v1 §S5), so the
   asymmetry is pinned rather than tidied. `HERDD_REAP=0` in particular is the
   campaign kill switch, and it reaches the timer ONLY through
   `config.load_env()`'s walk-up `.env` discovery — the unit carries no
   `EnvironmentFile`.
3. **The exit-code contract.** `sys.exit(2)` on a preview with candidates and on
   `--json` with candidates; callers script on it.

Provenance: new in the vastlib package, plan §8 step 3 (`boxes/`). The expected
values are inherited from the then-live `herdd` copies: `test_guard.py`
asserted UNEDITED against `herdd.cmd_reap` and the add-only port was proved
by both sides agreeing. At step 6d the flat copy went and `test_guard.py`
drives `vastlib.boxes.reap.cmd_reap` directly — the launcher does not
re-export the `cmd_*` handlers at all (its docstring's deliberately-absent
list), so there is no second entry point to disagree with.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vastlib.boxes import health as boxhealth  # noqa: E402
from vastlib.boxes import lifecycle, reap  # noqa: E402
from vastlib.core import result  # noqa: E402


# --------------------------------------------------------------------------
# 1. The ledger paths — formula and import-time read
# --------------------------------------------------------------------------

def _expected(name):
    base = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
    return os.path.join(base, "herdd", name)


def test_ledger_paths_match_the_frozen_formula():
    """`<XDG_CACHE_HOME|~/.cache>/herdd/<name>.json`, byte for byte. `ls` and
    `reap` must read the SAME idle ledger or the 2 h clock forks."""
    assert reap._IDLE_LEDGER == _expected("idle-ledger.json")
    assert reap._ZOMBIE_LEDGER == _expected("zombie-ledger.json")


def test_ledger_paths_are_read_at_import_not_per_call(monkeypatch, tmp_path):
    """Setting the env var after import changes nothing — which is why a test
    that wants a temp ledger patches the MODULE ATTRIBUTE. Documented here
    because getting it wrong makes a test write the developer's real ledger."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert not reap._IDLE_LEDGER.startswith(str(tmp_path))
    assert not reap._ZOMBIE_LEDGER.startswith(str(tmp_path))


# --------------------------------------------------------------------------
# 2. _idle_secs_map — the first-observed-stopped clock
# --------------------------------------------------------------------------

@pytest.fixture()
def idle_ledger(monkeypatch, tmp_path):
    p = tmp_path / "cache" / "herdd" / "idle-ledger.json"
    monkeypatch.setattr(reap, "_IDLE_LEDGER", str(p))
    return p


def test_idle_map_stamps_a_newly_stopped_box(idle_ledger):
    ins = [{"id": 41}, {"id": 42}]
    got = reap._idle_secs_map(ins, live_ids=[42])
    assert set(got) == {"41"} and got["41"] < 1.0
    led = json.loads(idle_ledger.read_text())
    assert list(led) == ["41"] and isinstance(led["41"], float)


def test_idle_ledger_key_shape_is_iid_to_float(idle_ledger):
    """`{"<iid>": <epoch float>}` — the whole schema. `cli/ls` writes this file
    too; a shape change there resets every parked box's 2 h fuse."""
    idle_ledger.parent.mkdir(parents=True)
    idle_ledger.write_text(json.dumps({"41": 1000.0}))
    got = reap._idle_secs_map([{"id": 41}], live_ids=[])
    assert got["41"] > 1_000_000            # measured from the stored epoch


def test_idle_map_forgets_boxes_that_came_back_or_vanished(idle_ledger):
    idle_ledger.parent.mkdir(parents=True)
    idle_ledger.write_text(json.dumps({"41": 1000.0, "42": 1000.0, "43": 1000.0}))
    reap._idle_secs_map([{"id": 41}, {"id": 42}], live_ids=[42])
    assert list(json.loads(idle_ledger.read_text())) == ["41"]


def test_an_unreadable_ledger_degrades_to_first_sighting(monkeypatch, tmp_path):
    """Both the read and the write are swallowed. The read failure is treated as
    an EMPTY ledger, so every stopped box reads as freshly sighted (age ~0) —
    the safe direction: an unwritable cache dir makes the reaper wait, never
    destroy sooner. The docstring's "{} if the ledger can't be read/written" is
    true only of the write; the returned map is computed in memory."""
    bad = tmp_path / "nope"
    bad.mkdir()
    monkeypatch.setattr(reap, "_IDLE_LEDGER", str(bad))   # a directory
    got = reap._idle_secs_map([{"id": 41}], live_ids=[])
    assert set(got) == {"41"} and got["41"] < 1.0
    assert bad.is_dir()                                   # nothing was written


# --------------------------------------------------------------------------
# 3. _zombie_confirm_map — condemn on NO PROGRESS, not on age
# --------------------------------------------------------------------------

Z = boxhealth.GUARD_ZOMBIE_LOADING_STALL


@pytest.fixture()
def zombie_ledger(monkeypatch, tmp_path):
    p = tmp_path / "cache" / "herdd" / "zombie-ledger.json"
    monkeypatch.setattr(reap, "_ZOMBIE_LEDGER", str(p))
    return p


def _health(verdict=Z, hb_age=None):
    return {"41": {"verdict": verdict,
                   "evidence": {"jobd_hb_age_s": hb_age, "is_jobs_box": True}}}


def test_first_sighting_is_never_confirmed(zombie_ledger):
    out = reap._zombie_confirm_map(_health(), {"41": {}}, now=1000.0)
    assert out["41"] == {"confirmed": False, "since_s": 0, "note": "first sighting"}


def test_zombie_ledger_key_shape(zombie_ledger):
    """`{iid: {first, verdict, pull, hb, inet, disk}}` — the shape
    `test_guard.py::_seed` hand-builds. Six keys, no more, no fewer."""
    reap._zombie_confirm_map(_health(), {"41": {}}, now=1000.0)
    led = json.loads(zombie_ledger.read_text())
    assert set(led["41"]) == {"first", "verdict", "pull", "hb", "inet", "disk"}
    assert led["41"]["verdict"] == Z


def test_confirms_only_after_the_confirm_window(zombie_ledger):
    zombie_ledger.parent.mkdir(parents=True)
    zombie_ledger.write_text(json.dumps(
        {"41": {"first": 1000.0, "verdict": Z, "pull": {}, "hb": None,
                "inet": None, "disk": None}}))
    early = reap._zombie_confirm_map(_health(), {"41": {}}, now=1000.0 + 899)
    assert early["41"]["confirmed"] is False
    zombie_ledger.write_text(json.dumps(
        {"41": {"first": 1000.0, "verdict": Z, "pull": {}, "hb": None,
                "inet": None, "disk": None}}))
    late = reap._zombie_confirm_map(_health(), {"41": {}}, now=1000.0 + 901)
    assert late["41"]["confirmed"] is True
    assert late["41"]["since_s"] == 901


@pytest.mark.parametrize("inst, note", [
    ({"inet_down_billed": 1_000_000 + 50_001},
     "box download traffic advancing (env-setup/pull alive)"),
    ({"disk_usage": 10.0 + 0.6},
     "disk usage advancing (unpack/install alive)"),
])
def test_env_setup_liveness_signals_reset_the_clock(zombie_ledger, inst, note):
    """The 2026-08-02 fix: mid-provision a jobs box has FLAT pull bytes and NO
    heartbeat by definition, so without `inet_down_billed` / `disk_usage` a
    healthy long install is indistinguishable from a stall and gets destroyed."""
    zombie_ledger.parent.mkdir(parents=True)
    zombie_ledger.write_text(json.dumps(
        {"41": {"first": 1000.0, "verdict": Z, "pull": {}, "hb": None,
                "inet": 1_000_000, "disk": 10.0}}))
    out = reap._zombie_confirm_map(_health(), {"41": inst}, now=1000.0 + 5000)
    assert out["41"]["note"] == note
    assert out["41"]["confirmed"] is False       # clock restarted


def test_measured_cpu_work_resets_the_clock(zombie_ledger):
    """A dedicated CPU box is FLAT ON ALL FIVE of the other progress signals
    while burning cores: nothing pulls, jobd never stamps, the download counter
    sits still and disk does not grow. Without this it is confirmable as a
    zombie while doing exactly the work it was rented for."""
    zombie_ledger.parent.mkdir(parents=True)
    zombie_ledger.write_text(json.dumps(
        {"41": {"first": 1000.0, "verdict": Z, "pull": {}, "hb": None,
                "inet": 1_000_000, "disk": 10.0}}))
    h = _health()
    h["41"]["evidence"]["cpu_util"] = 19.98
    out = reap._zombie_confirm_map(h, {"41": {}}, now=1000.0 + 5000)
    assert out["41"]["note"] == "cpu 19.98 busy (compute alive)"
    assert out["41"]["confirmed"] is False       # clock restarted


def test_an_idle_cpu_reading_does_not_rescue_a_zombie(zombie_ledger):
    """The mirror: this must not become a blanket exemption. A quiet box with a
    stale verdict still confirms."""
    zombie_ledger.parent.mkdir(parents=True)
    zombie_ledger.write_text(json.dumps(
        {"41": {"first": 1000.0, "verdict": Z, "pull": {}, "hb": None,
                "inet": 1_000_000, "disk": 10.0}}))
    h = _health()
    h["41"]["evidence"]["cpu_util"] = 0.05
    out = reap._zombie_confirm_map(h, {"41": {}}, now=1000.0 + 5000)
    assert out["41"]["confirmed"] is True


def test_a_verdict_change_resets_the_clock(zombie_ledger):
    zombie_ledger.parent.mkdir(parents=True)
    zombie_ledger.write_text(json.dumps(
        {"41": {"first": 1000.0, "verdict": boxhealth.GUARD_ZOMBIE_NO_JOBD,
                "pull": {}, "hb": None, "inet": None, "disk": None}}))
    out = reap._zombie_confirm_map(_health(), {"41": {}}, now=1000.0 + 5000)
    assert out["41"]["confirmed"] is False
    assert "verdict changed" in out["41"]["note"]


def test_non_zombie_verdicts_are_dropped_from_the_ledger(zombie_ledger):
    """Recovered/destroyed/gone boxes leave no entry — the ledger is rebuilt
    from the current verdicts on every pass."""
    out = reap._zombie_confirm_map(
        {"41": {"verdict": boxhealth.GUARD_OK, "evidence": {}}},
        {"41": {}}, now=1000.0)
    assert out == {}
    assert json.loads(zombie_ledger.read_text()) == {}


def test_unreadable_ledger_degrades_to_alarm_only(monkeypatch, tmp_path):
    """Never to a FASTER destroy: confirmed=False for everything."""
    bad = tmp_path / "dir"
    bad.mkdir()
    monkeypatch.setattr(reap, "_ZOMBIE_LEDGER", str(bad))
    out = reap._zombie_confirm_map(_health(), {"41": {}}, now=1000.0)
    assert out["41"]["confirmed"] is False


def test_confirm_window_comes_from_the_boot_knob(zombie_ledger, monkeypatch):
    """The SIXTH knob, and the only one with CLI > env > yaml > constant
    precedence — the other five are bare `os.environ` reads (see below)."""
    zombie_ledger.parent.mkdir(parents=True)
    zombie_ledger.write_text(json.dumps(
        {"41": {"first": 1000.0, "verdict": Z, "pull": {}, "hb": None,
                "inet": None, "disk": None}}))
    monkeypatch.setenv("REAP_ZOMBIE_CONFIRM_S", "10")
    out = reap._zombie_confirm_map(_health(), {"41": {}}, now=1000.0 + 11)
    assert out["41"]["confirmed"] is True


# --------------------------------------------------------------------------
# 4. cmd_reap — the policy driver
# --------------------------------------------------------------------------

def _args(idle_hours=None, yes=False, as_json=False):
    """Exactly the Namespace `test_guard.py`:679 builds: three attributes."""
    return argparse.Namespace(idle_hours=idle_hours, yes=yes, json=as_json)


@pytest.fixture()
def fleet(monkeypatch, idle_ledger, capsys):
    """A reap with the live (zombie) lane off and every mutation stubbed.

    The zombie lane is disabled by env exactly as an operator would disable it,
    which keeps this fixture free of `parked_lifecycle` and of the fleet-health
    gather — those belong to `boxes.health`'s own tests."""
    monkeypatch.setenv("HERDD_REAP", "1")
    monkeypatch.setenv("HERDD_REAP_ZOMBIE", "0")
    monkeypatch.setenv("HERDD_REAP_DURABILITY", "0")
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.delenv("HERDD_REAP_IDLE_H", raising=False)
    log = {"destroyed": [], "parked": []}

    def _fake_destroy(ids, label_ins, intent, noun=""):
        log["destroyed"].append((list(ids), intent, noun))
        return []

    monkeypatch.setattr(lifecycle, "_destroy_and_revoke", _fake_destroy)
    monkeypatch.setattr(lifecycle, "stop_box",
                        lambda iid: log["parked"].append(iid) or result.OkErr(True, None))
    monkeypatch.setattr(lifecycle, "_emit_stopping_intent",
                        lambda iid, reason, instances=None: None)

    def _set(instances):
        monkeypatch.setattr(lifecycle, "_instances", lambda: instances)
    log["set"] = _set
    return log


def _stopped(iid, label="", age_s=None):
    return {"id": iid, "label": label, "actual_status": "stopped"}


def test_kill_switch_returns_before_reading_anything(monkeypatch, capsys):
    """`HERDD_REAP=0` is the campaign kill switch. It must return BEFORE the
    instance listing — a reap that lists first would still pay the API call and,
    worse, would prove nothing about the switch's placement."""
    monkeypatch.setenv("HERDD_REAP", "0")
    monkeypatch.setattr(lifecycle, "_instances",
                        lambda: pytest.fail("kill switch must return first"))
    reap.cmd_reap(_args())
    assert "reap disabled (HERDD_REAP=0)" in capsys.readouterr().out


@pytest.mark.parametrize("val", ["0", "no", "off", "OFF", " no "])
def test_kill_switch_spellings(monkeypatch, capsys, val):
    monkeypatch.setenv("HERDD_REAP", val)
    monkeypatch.setattr(lifecycle, "_instances",
                        lambda: pytest.fail("kill switch must return first"))
    reap.cmd_reap(_args())
    assert "reap disabled" in capsys.readouterr().out


def test_idle_box_past_the_threshold_previews_and_exits_2(fleet, idle_ledger, capsys):
    idle_ledger.parent.mkdir(parents=True)
    idle_ledger.write_text(json.dumps({"41": 0.0}))       # stopped at the epoch
    fleet["set"]([_stopped(41)])
    with pytest.raises(SystemExit) as ei:
        reap.cmd_reap(_args())
    assert ei.value.code == 2
    assert fleet["destroyed"] == []                       # preview destroys nothing
    out = capsys.readouterr().out
    assert "-> REAP" in out and "[preview] reap WOULD DESTROY 1" in out


def test_yes_executes_the_destroy(fleet, idle_ledger, capsys):
    idle_ledger.parent.mkdir(parents=True)
    idle_ledger.write_text(json.dumps({"41": 0.0}))
    fleet["set"]([_stopped(41)])
    reap.cmd_reap(_args(yes=True))
    assert fleet["destroyed"] == [([41], "reap_idle_destroy", "idle ")]
    assert "reaped 1 box(es)" in capsys.readouterr().out


def test_a_keep_label_opts_a_box_out(fleet, idle_ledger, capsys):
    """Delegated to `labels._reap_kept`; this module never parses the token
    grammar itself (the 2026-08-02 un-revoked-key bug is what a second copy
    costs)."""
    idle_ledger.parent.mkdir(parents=True)
    idle_ledger.write_text(json.dumps({"41": 0.0}))
    fleet["set"]([_stopped(41, "run:R7 keep:owner-asked")])
    reap.cmd_reap(_args(yes=True))
    assert fleet["destroyed"] == []
    assert "-> KEEP" in capsys.readouterr().out


def test_a_young_box_waits(fleet, idle_ledger, capsys):
    fleet["set"]([_stopped(41)])                          # first sighting: age ~0
    reap.cmd_reap(_args(yes=True))
    assert fleet["destroyed"] == []
    assert "-> WAIT" in capsys.readouterr().out


def test_live_boxes_are_not_in_the_idle_lane(fleet, capsys):
    fleet["set"]([{"id": 41, "label": "", "actual_status": "running"}])
    reap.cmd_reap(_args(yes=True))
    assert "no stopped boxes — nothing to reap." in capsys.readouterr().out


def test_idle_hours_flag_beats_the_env_default(fleet, idle_ledger, monkeypatch, capsys):
    monkeypatch.setenv("HERDD_REAP_IDLE_H", "9999")
    idle_ledger.parent.mkdir(parents=True)
    idle_ledger.write_text(json.dumps({"41": 0.0}))
    fleet["set"]([_stopped(41)])
    reap.cmd_reap(_args(idle_hours=0.0, yes=True))        # explicit flag wins
    assert fleet["destroyed"] == [([41], "reap_idle_destroy", "idle ")]


def test_a_garbage_idle_h_falls_back_to_the_default(fleet, idle_ledger,
                                                    monkeypatch, capsys):
    """`float("banana")` raises ValueError, and the fallback is the 2 h owner
    policy — not an exception in the systemd timer."""
    monkeypatch.setenv("HERDD_REAP_IDLE_H", "banana")
    fleet["set"]([_stopped(41)])
    reap.cmd_reap(_args(yes=True))
    assert f"idle>{reap.REAP_IDLE_H_DEFAULT:g}h" in capsys.readouterr().out


def test_json_mode_prints_the_frozen_keys_and_exits_2(fleet, idle_ledger, capsys):
    idle_ledger.parent.mkdir(parents=True)
    idle_ledger.write_text(json.dumps({"41": 0.0}))
    fleet["set"]([_stopped(41)])
    with pytest.raises(SystemExit) as ei:
        reap.cmd_reap(_args(as_json=True))
    assert ei.value.code == 2
    doc = json.loads(capsys.readouterr().out)
    assert set(doc) == {"idle_hours", "rows", "reap", "zombie_rows",
                        "zombie_destroy", "zombie_park"}
    assert doc["reap"] == [41]
    assert set(doc["rows"][0]) == {"iid", "label", "idle_s", "storage_day",
                                   "verdict"}


def test_json_mode_exits_0_with_no_candidates(fleet, capsys):
    fleet["set"]([_stopped(41)])
    with pytest.raises(SystemExit) as ei:
        reap.cmd_reap(_args(as_json=True))
    assert ei.value.code == 0


def test_reap_idle_default_is_the_owner_policy():
    assert reap.REAP_IDLE_H_DEFAULT == 2.0


# --------------------------------------------------------------------------
# 5. The knob asymmetry, stated as a test
# --------------------------------------------------------------------------

def test_five_knobs_are_bare_env_reads_not_boot_knobs():
    """Pinned, not tidied: routing these five through `config._boot_knob` would
    add a yaml rung to a kill switch that is env-only today, which is a behavior
    change (plan v1 §S5). Read as source text because the alternative — driving
    each knob from a `herdd.yaml` and asserting it does NOT bite — needs a
    yaml file the portable lane does not have."""
    src = open(reap.__file__).read()
    body = src.split("def cmd_reap(", 1)[1]
    for knob in ("HERDD_REAP", "HERDD_REAP_IDLE_H", "HERDD_REAP_ZOMBIE",
                 "HERDD_REAP_STALL"):
        assert f'os.environ.get("{knob}"' in body, knob
    assert 'os.environ.get("HERDD_REAP_DURABILITY"' in src
    assert '_boot_knob("REAP_ZOMBIE_CONFIRM_S"' in src
    assert "_boot_knob" not in body        # ... and nowhere in cmd_reap


def test_durability_advisory_is_skipped_by_its_own_knob(monkeypatch):
    monkeypatch.setenv("HERDD_REAP_DURABILITY", "0")

    def _boom(*a, **k):
        pytest.fail("HERDD_REAP_DURABILITY=0 must skip the B2 reads")

    monkeypatch.setattr(lifecycle, "_box_is_jobd", _boom)
    reap._reap_durability_advisory({"id": 41, "label": "run:R7"}, pal=None)


# --------------------------------------------------------------------------
# 6. The cross-ring seam
# --------------------------------------------------------------------------

def test_fold_fleet_jobs_seam_raises_when_nothing_is_bound(monkeypatch):
    """Step-5 INVERSION of `..._raises_until_step_5` (plan §7.4 licenses the
    expectation change: what changed is "not ported yet", not behavior).

    UNBOUND is still a raise, not an empty dict: both call sites swallow this
    into "no jobs fold", so a silent `{}` would degrade reap to the idle lane
    invisibly.
    """
    monkeypatch.setattr(reap, "_FOLD_FLEET_JOBS", None)
    with pytest.raises(NotImplementedError) as ei:
        reap._fold_fleet_jobs(set())
    assert "vastlib.jobs.view" in str(ei.value)


def test_importing_jobs_view_binds_the_fold_seam(monkeypatch):
    """`boxes` may not import `jobs` (the §5 edge is upward), so `jobs.view`
    injects itself. The forwarder must resolve `view._fold_fleet_jobs` at CALL
    time — a captured function object would make every existing
    `monkeypatch.setattr(view, "_fold_fleet_jobs", ...)` vacuous."""
    from vastlib.jobs import view

    assert reap._FOLD_FLEET_JOBS is not None
    seen = []
    monkeypatch.setattr(view, "_fold_fleet_jobs",
                        lambda live, prog=None: seen.append((live, prog)) or {"7": []})
    assert reap._fold_fleet_jobs({"7"}) == {"7": []}
    assert seen == [({"7"}, None)]


def test_cmd_reap_survives_the_seam(fleet, idle_ledger, capsys):
    """`cmd_reap` memoizes the jobs fold behind a `try/except` that swallows to
    `{}` — so the unported seam degrades reap to the idle lane rather than
    breaking it. The same swallow hides a real `boxes.health` bug, which is why
    it is written down here and in the module docstring."""
    idle_ledger.parent.mkdir(parents=True)
    idle_ledger.write_text(json.dumps({"41": 0.0}))
    fleet["set"]([_stopped(41)])
    reap.cmd_reap(_args(yes=True))
    assert fleet["destroyed"] == [([41], "reap_idle_destroy", "idle ")]

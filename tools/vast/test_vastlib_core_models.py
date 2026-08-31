"""`vastlib.core.models` — the pins that are NOT parity, after the thinning.

Why this file exists
--------------------
It was a parity file. Steps 2-5 of the refactor
(docs/plans/vast-tooling-refactor-v2.md §8) were ADD-ONLY: the payload readers
existed twice — ported into `vastlib/core/models.py` and still live in
`herdd.py`, where every caller read them. This layer is the one the money bugs
came out of (dph_base vs dph_total, `verified` vs `verification`, the
`disk_usage: -1` sentinel, the MB-vs-GB VRAM unit), so while both copies existed
something had to assert they agreed. Twelve tests drove one ported symbol each
through a table of payload shapes and compared the two implementations directly,
deliberately NOT re-asserting expected values — the expectations have homes
(`test_disk_sizing.py`, `test_ssh_access.py`, `test_dash_cache.py`,
`test_eviction_blindspot.py`, `test_lifecycle.py`, `test_supervise.py`), and a
parity test that restated them would pass while both copies drifted together.

**Plan §8 step 6d deleted the second copy**, and this file's own exit plan said
what to do about that: "when `herdd.py` becomes a thin launcher and its copies
are deleted, the parity half of this file goes with them: it asserts a
duplication, not a contract. The three non-parity tests stay." So the twelve
parity sweeps are gone, along with the payload tables that fed only them
(`LABELS`, `PREFIXES`, `INSTANCES`, `OFFERS`, `MARKETS`, `JCTXS`, `HFS` — a
curated blood-drawn-shapes corpus, kept in git history at the commit that
removed it, and reachable through `.port_manifests/models.json`).

What is left is what was never parity, and one thing the thinning created:

* **the delegation probe** — models must CALL `core.labels`, not carry a second
  copy of the label token rules;
* **the `None`-instance asymmetry** — ported, deliberately not fixed;
* **the `MarketRead` respelling** — `collections.namedtuple` + a `__defaults__`
  poke became a `typing.NamedTuple`, so the runtime object model is pinned
  field-for-field;
* **the `_num_dph` alias identity** — `test_ladder_core.py` asserts the alias;
  the port must not turn it into a wrapper;
* **the TypedDict rows** add no runtime object;
* **the launcher's bindings** — the residue of the deleted parity half. One
  body per name, re-exported by identity. A second body in `herdd.py` would
  reinstate exactly the two-copies condition the deleted tests policed, and now
  nothing else would see it.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ladder_core  # noqa: E402
import herdd  # noqa: E402
from vastlib.core import labels, models  # noqa: E402

RID = "20260729T181618-0ad8ae0c-generate-a0"   # a real shape, from box 46234244


# --- the label accessor: delegation ------------------------------------------

def test_label_value_actually_delegates_to_core_labels(monkeypatch):
    """NOT a parity test. The binding design constraint of this port: models
    must not carry a second copy of the keep-token rules, because a drifted
    duplicate is exactly the 2026-08-02 bug. Patching the grammar out from
    under it proves the call is live and resolved at call time (the
    module-attribute form plan §8b requires), not import-bound."""
    seen = []

    def sentinel(label, prefix):
        seen.append((label, prefix))
        return "SENTINEL"

    monkeypatch.setattr(labels, "label_value", sentinel)
    assert models._label_value(f"run:{RID}", "run") == "SENTINEL"
    assert models._instance_run_label({"label": f"run:{RID}"}) == "SENTINEL"
    assert models._instance_serve_label({"label": "serve:s-77"}) == "SENTINEL"
    assert seen == [(f"run:{RID}", "run"), (f"run:{RID}", "run"),
                    ("serve:s-77", "serve")]


def test_the_None_instance_asymmetry_is_ported_not_fixed():
    """`_instance_run_label` reads `i.get(...)` unguarded while
    `_instance_serve_label` uses `(i or {})`, so one raises on None and the
    other returns None. Untested in `test_label_grammar.py` and deliberately
    NOT changed by the port — a widening would be a behavior change in a step
    contracted to preserve behavior. Pinned here so the asymmetry is a decision
    with a test on it rather than an accident, and so the `Instance` model that
    eventually makes it moot has to do so on purpose.

    The `herdd` half of each pair (it raised/returned the same) went at step
    6d with the flat copy."""
    with pytest.raises(AttributeError):
        models._instance_run_label(None)
    assert models._instance_serve_label(None) is None


# --- the supervise-side ARM snapshot key tuple -------------------------------

def test_primary_shape_keys_are_the_persistence_contract():
    """The ARM snapshot's key tuple is a persistence contract: a watch armed by
    one revision is read back by the next. Was `…_are_identical`, comparing the
    tuple to `herdd._JOB_PRIMARY_SHAPE_KEYS`, which post-6d is this tuple."""
    assert "gpu_ram" in models._JOB_PRIMARY_SHAPE_KEYS
    assert herdd._JOB_PRIMARY_SHAPE_KEYS is models._JOB_PRIMARY_SHAPE_KEYS


# --- the non-parity pins -----------------------------------------------------

def test_num_dph_is_the_same_object_not_a_wrapper():
    """NOT a parity test. `test_ladder_core.py` asserts `herdd._num_dph is
    ladder_core.num_dph`; the port must keep the alias an alias, or 'a fix in
    ladder_core is a fix everywhere' stops being true. Both spellings are
    asserted because both are live: the flat one is what `test_ladder_core.py`
    and the ladder's other callers reach."""
    assert models._num_dph is ladder_core.num_dph
    assert models._num_dph is herdd._num_dph


def test_market_read_respelling_is_runtime_equivalent():
    """NOT a parity test. The port respells `collections.namedtuple` +
    `MarketRead.__new__.__defaults__ = ((), False)` as a `typing.NamedTuple`
    with ordinary defaults. Everything the callers and the meta-test in
    `test_bid_echo_probe.py` bind to has to survive that: the field order, the
    3-positional call shape `_market_min_bid_read` uses, the two defaults,
    tuple-unpacking, and `._fields` / `._asdict` / `._replace`.

    The `old = herdd.MarketRead(...)` half of each pair was dropped at step
    6d — `herdd.MarketRead` is this class — leaving the properties stated
    outright, which is what they always were a proxy for."""
    assert models.MarketRead._fields == ("ok", "listed", "min_bid", "floors",
                                         "scaled")

    new = models.MarketRead(True, False, None)          # the 3-positional shape
    assert new.floors == () and new.scaled is False
    assert new._asdict() == {"ok": True, "listed": False, "min_bid": None,
                             "floors": (), "scaled": False}

    floors = [2.562, 2.7]
    new = models.MarketRead(True, True, min(floors), floors=floors, scaled=True)
    ok, listed, min_bid, fl, scaled = new                # tuple-unpack
    assert (ok, listed, min_bid, fl, scaled) == (True, True, 2.562, floors, True)
    assert new._replace(ok=False).ok is False
    assert isinstance(new, tuple)


def test_machine_row_types_are_plain_dicts_at_runtime():
    """The MachineRow/MachineMarket TypedDicts are new (they had no named shape
    in `herdd.py`), so pin that they add no runtime object: `_rates` must keep
    accepting the raw dicts `_market_map` builds."""
    row: models.MachineRow = {"g": 4, "base": 2.8, "bid": 2.5}
    assert row == {"g": 4, "base": 2.8, "bid": 2.5}
    assert isinstance(row, dict)
    mkt: models.MachineMarket = {"offers": [row], "max_gpus": 4}
    assert isinstance(mkt, dict)
    assert models._rates({"id": 1, "num_gpus": 4, "machine_id": "12345"},
                         {"12345": mkt}) is not None


# --- what the deleted parity half leaves behind ------------------------------

def test_the_launcher_re_exports_rather_than_redefines():
    """One body per name. This is the whole residue of twelve parity sweeps.

    `herdd.<name>` is still a live external spelling — `boxstate.py`,
    `hosts.py`, `hostfacts.py`, `bid_echo_probe.py`, `parked_lifecycle.py` and
    `workflowctl.py` address the flat module — so a second body under any of
    these names would put the dashboard, the host picker and the CLI back on
    two different readers of the same payload, which is the condition that
    produced the dph_base/dph_total and `verified`/`verification` bugs.
    """
    for name in ("MarketRead", "SSH_INJECT_MARKER", "_JOB_PRIMARY_SHAPE_KEYS",
                 "_dash_verified", "_disk_frac", "_disk_gb", "_gpu_ram_gb",
                 "_instance_env", "_instance_image", "_instance_run_label",
                 "_instance_serve_label", "_instance_standing_bid",
                 "_job_primary_inst", "_job_primary_shape", "_label_value",
                 "_num_dph", "_rates", "_storage_day", "instance_has_ssh_inject",
                 "instance_ssh_install"):
        assert getattr(herdd, name) is getattr(models, name), (
            f"herdd.{name} is a second body again — the launcher must "
            f"re-export vastlib.core.models' object, never redefine it")

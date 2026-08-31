"""Instance-label parsing must survive APPENDED tokens.

The bug this locks down was live in shipped code: `fleetd`'s `Hooks.keep_label`
stamps `":keep"` onto the label of every box it parks (every fleetd park is a
resumability promise, so it opts the box out of the 2h idle reaper). A box
labelled `run:<RID>` therefore becomes `run:<RID>:keep` — and both label readers
sliced a FIXED WIDTH off the front, so they returned `<RID>:keep`:

  * `fleetd._resolve_iid` matches `_instance_run_label(inst)` against the bare
    RUN_ID, so a run watch lost its own box immediately after fleetd parked it.
  * `_destroy_and_revoke` minted the revoke name `run-<RID>:keep`, so the
    ephemeral B2 key actually named `run-<RID>` was never revoked on destroy —
    a credential outliving the box it was issued for.

FLEETD_DESIGN's own B1c note had already stated the rule that was broken: a
`run:<id>` label "is parsed elsewhere and must stay exact". These tests exist so
the next person who appends a label token finds out here instead of in a B2 audit.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import runmeta  # noqa: E402
import herdd  # noqa: E402
from vastlib.core import labels, models  # noqa: E402

# Plumbing note (plan §7, step 2 of docs/plans/vast-tooling-refactor-v2.md):
# the keep/retention half of this grammar lives in `vastlib.core.labels` and is
# called there. The PREFIX half (`_label_value`, `_instance_run_label`) landed
# in `vastlib.core.models` and is called there as of the models port — models
# owns the accessor name, `labels.label_value` owns the token rules underneath
# it, so there is still exactly one copy of the grammar inside the package.
# `_revoke_box_keys` is assigned to `boxes/lifecycle.py`, so the destroy/revoke
# test below still goes through `herdd` — including the `_label_value` calls
# that stand in for `_destroy_and_revoke`'s naming block, which move when that
# function does. Ports were ADD-ONLY until step 6, so `herdd` carried its own
# copies and the parity of the two was asserted in test_vastlib_core_labels.py
# and test_vastlib_core_models.py. At plan §8 step 6d `herdd.py` became a thin
# launcher: there is one copy, the launcher re-exports it by identity, and the
# pure-parity file was deleted with the duplication it described (its own
# docstring's exit plan). The `herdd.` calls below now land on the package.
# No expectation in this file changed.

RID = "20260729T181618-0ad8ae0c-generate-a0"   # a real shape, from box 46234244


def _inst(label):
    return {"id": 1, "label": label}


# --- the parser itself -------------------------------------------------------

def test_run_label_survives_the_keep_token_fleetd_appends():
    assert models._instance_run_label(_inst(f"run:{RID}")) == RID
    assert models._instance_run_label(_inst(f"run:{RID}:keep")) == RID


def test_run_label_survives_the_proposed_expiring_keep_grammar():
    """The parked-box lifecycle design proposes `keep:<why>:u<YYYYMMDD>`. Whether
    or not that design is ratified, the parser must not care how many tokens
    follow."""
    assert models._instance_run_label(
        _inst(f"run:{RID}:keep:fleetd-park:u20260802")) == RID


def test_non_run_and_degenerate_labels_are_none():
    for lab in ["", "upstream-monorepo", "keep", "serve:abc", "runx:abc", "run:"]:
        assert models._instance_run_label(_inst(lab)) is None, lab
    assert models._instance_run_label({"id": 1}) is None
    assert models._instance_run_label({"id": 1, "label": None}) is None


def test_handoff_suffix_is_PRESERVED_not_stripped():
    """The suffix is not all noise. `run:<ID>:handoff` must read as the DISTINCT
    run id `<ID>:handoff`, because `_launch_preflight` relies on the exact-match
    comparing the whole suffix to refuse a second understudy while letting the
    live primary `run:<ID>` outlive the cutover (HANDOFF_DESIGN §2.1). A parser
    that collapses to the first token makes a twin indistinguishable from its
    primary and silently disarms both dup guards — caught by
    test_lifecycle.py's preflight tests when this fix first over-reached."""
    assert models._instance_run_label(_inst("run:r1:handoff")) == "r1:handoff"
    assert models._instance_run_label(_inst("run:r1")) == "r1"


def test_handoff_suffix_survives_the_keep_token_too():
    """Both constraints at once: fleetd parks an understudy and stamps :keep."""
    assert models._instance_run_label(
        _inst("run:r1:handoff:keep")) == "r1:handoff"
    assert models._instance_run_label(
        _inst("run:r1:handoff:keep:fleetd-park:u20260802")) == "r1:handoff"


def test_label_value_is_pure_and_prefix_scoped():
    assert models._label_value("serve:s-123", "serve") == "s-123"
    assert models._label_value("serve:s-123:keep", "serve") == "s-123"
    assert models._label_value("run:a", "serve") is None
    # a bare prefix match must not swallow a longer prefix
    assert models._label_value("runner:x", "run") is None


def test_run_ids_cannot_contain_a_colon():
    """The whole fix rests on this: token-splitting is lossless for RUN_IDs
    precisely because the charset excludes ':'. If RUN_ID_RE ever widens, the
    parser needs revisiting, so assert the premise rather than trusting it."""
    assert not runmeta.RUN_ID_RE.match(f"{RID}:keep")
    assert runmeta.RUN_ID_RE.match(RID)


# --- the two consumers that were actually broken -----------------------------

def test_fleetd_resolves_a_run_watch_after_it_stamped_keep(monkeypatch):
    """`_resolve_iid` compares against the bare RUN_ID. Before the fix, a box
    fleetd had parked itself became unresolvable."""
    import fleetd
    fleet = fleetd.Fleet.__new__(fleetd.Fleet)          # no daemon side effects
    by_iid = {"46234244": _inst(f"run:{RID}:keep")}
    w = {"profile": "run", "target": f"run:{RID}", "iid": None}
    assert fleet._resolve_iid(w, by_iid) == "46234244"


def test_destroy_revokes_the_real_key_name_not_the_keep_suffixed_one():
    """The credential half. Captures the revoke set instead of hitting B2."""
    seen = {}

    def fake_revoke(names):
        seen["names"] = set(names)

    orig = herdd._revoke_box_keys
    herdd._revoke_box_keys = fake_revoke
    try:
        # drive only the naming block: a destroyed box carrying an appended token
        revoke_names = set()
        for iid, lab in [("46234244", f"run:{RID}:keep"),
                         ("46245045", "serve:s-77:keep")]:
            revoke_names.add(f"box-{iid}")
            rid = herdd._label_value(lab, "run")
            sid = herdd._label_value(lab, "serve")
            if rid:
                revoke_names.add(f"run-{rid}")
            elif sid:
                revoke_names.add(f"serve-{sid}")
        herdd._revoke_box_keys(revoke_names)
    finally:
        herdd._revoke_box_keys = orig

    assert f"run-{RID}" in seen["names"], "the real ephemeral key must be revoked"
    assert "serve-s-77" in seen["names"]
    assert not any(":" in n for n in seen["names"]), \
        f"no revoke name may carry a label token: {sorted(seen['names'])}"


def test_reap_still_honors_the_appended_keep_token():
    """The fix must not disturb why the token is appended in the first place."""
    assert labels._reap_kept(f"run:{RID}:keep")
    assert not labels._reap_kept(f"run:{RID}")


def test_reap_honors_a_keep_group_appended_to_an_existing_label():
    """Labels are whitespace-separated GROUPS of `:`-separated tokens, and
    appending `keep:<why>` to an existing label is the natural (and documented)
    way to hold a box. Splitting on `:` alone produced the token
    `"rb3-wide-A keep"` and silently dropped the opt-out — i.e. the reaper would
    have destroyed a box someone deliberately paid to keep. Found live
    2026-07-31 on box 46446435, which carried exactly this shape."""
    assert labels._reap_kept("wave:rb3-wide-A keep:FLOOR-repair-pending")
    assert labels._reap_kept("serve:t211-vet-serve keep:ssh-rootcause-specimen")
    assert labels._reap_kept("keep:why")
    assert labels._reap_kept("keep")
    assert labels._reap_kept(f"run:{RID} keep")
    # ...and nothing that merely CONTAINS the substring counts
    assert not labels._reap_kept("wave:rb3-wide-A housekeeping:on")
    assert not labels._reap_kept("serve:keeper")
    assert not labels._reap_kept("")
    assert not labels._reap_kept(None)


# --------------------------------------------------------------------------- #
# Self-expiring keep groups (owner directive 2026-08-05 — the eviction-retention
# window). `keep:<why>-until-<YYYYMMDDTHHMMSSZ>` holds a box until an instant and
# then stops holding it, so `herdd reap`'s 15-minute timer is the expiry
# mechanism and no daemon has to be alive for the window to end.
# --------------------------------------------------------------------------- #
T0 = 1_785_900_000.0            # some fixed instant, UTC


def _lbl(label, reason="evicted-outbid", dt_h=3.0):
    return labels.retention_keep_label(label, reason, T0 + dt_h * 3600.0)


def test_a_keep_with_a_deadline_holds_the_box_until_that_instant():
    lab = _lbl(f"run:{RID}")
    assert labels._reap_kept(lab, now=T0)
    assert labels._reap_kept(lab, now=T0 + 3 * 3600 - 1)
    assert not labels._reap_kept(lab, now=T0 + 3 * 3600 + 1)


def test_an_unconditional_keep_is_unchanged_and_never_expires():
    """The 3h window must not become a 3h fuse on every hand-set keep label."""
    far = T0 + 10 * 365 * 86400
    assert labels._reap_kept(f"run:{RID}:keep", now=far)
    assert labels._reap_kept("wave:rb3-wide-A keep:FLOOR-repair-pending", now=far)


def test_the_deadline_uses_BASIC_iso8601_because_colons_are_separators():
    """An extended-format timestamp would shred the group into
    `keep`/`...T18`/`30`/`00Z` and the deadline would silently never parse —
    i.e. an unconditional keep, a window that never ends."""
    lab = _lbl("upstream-monorepo")
    assert ":" not in lab.split("keep:", 1)[1]
    assert labels._keep_retention_info(lab, now=T0)["reason"] == "evicted-outbid"
    assert labels._keep_retention_info(lab, now=T0)["left_s"] == 3 * 3600


def test_an_unparseable_deadline_fails_toward_HOLDING_the_box():
    """The label is the only durable record of why someone wanted the box
    alive; a malformed deadline must never license a destroy."""
    assert labels._reap_kept("keep:evicted-outbid-until-NOTATIME", now=T0)
    assert labels._reap_kept("keep:evicted-outbid-until-99999999T999999Z", now=T0)


def test_a_retention_group_never_costs_the_run_key_its_revocation():
    """`_label_value` truncates at the first `keep` token, but it split on `:`
    ALONE — so `run:<RID> keep:<why>` read as the run id `"<RID> keep:<why>"`
    and `_destroy_and_revoke` would mint `run-<RID> keep:...`, leaving the real
    `run-<RID>` B2 key live after the box died. Same bug class as the one this
    file was opened for, one grammar level up."""
    lab = _lbl(f"run:{RID}")
    assert models._label_value(lab, "run") == RID
    assert models._label_value(f"run:{RID} keep:why", "run") == RID
    assert models._label_value(f"serve:s-77 keep:why", "serve") == "s-77"
    # the deliberately-semantic suffix still survives (HANDOFF_DESIGN §2.1)
    assert models._label_value(f"run:{RID}:handoff keep:why", "run") \
        == f"{RID}:handoff"


def test_the_retention_group_is_APPENDED_never_fused_into_the_existing_one():
    lab = _lbl(f"run:{RID}")
    assert lab.startswith(f"run:{RID} keep:")
    assert labels.retention_keep_label("", "evicted-outbid", T0).startswith("keep:")
    # a reason carrying separators is slugged, or it would forge new tokens
    lab = labels.retention_keep_label("x", "evicted: host failure", T0)
    assert lab.count(":") == 1 and " " not in lab.split("keep:", 1)[1]

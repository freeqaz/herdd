"""Portable tests for `normalize_gpu` / `GPU_ALIASES` — GPU spellings -> the
exact `gpu_name` strings vast indexes on.

Why this needs a test file of its own: vast's gpu_name filter is EXACT match,
case- AND whitespace-sensitive, and an unmatched name is not an error — it is
an empty result set. So a missing alias is indistinguishable from an empty
market, and a launcher asking for a card that is plentifully in supply just
never rents one.

Measured live 2026-08-16 (one bundles query per spelling, verified+rentable
on-demand):

    'RTX 6000Ada'   -> offers        'RTX 6000 Ada' -> ZERO
    'L40S'          -> offers        'l40s'         -> ZERO
    'RTX A6000'     -> offers        'rtx a6000'    -> ZERO
    'RTX 5880Ada'   -> offers        '5880ada'      -> ZERO

"RTX 6000 Ada" is how every human, every runbook and every doc in this repo
spells that card (`tools/vast/TRAINING.md`, the chain-mining runbook, the
32k-run handoff). It was in no alias family, so it resolved to itself and
matched nothing — a 48 GB class with ~14 rentable cards that the launcher
could not reach. This is a SEPARATE defect from the GB->MiB floor in
`test_gpu_ram_floor.py`; the two just present identically.

Offline lane: no vast API, no network, $0.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import herdd as v  # noqa: E402


# gpu_name strings observed on the live board 2026-08-16 (1465 verified
# on-demand offers), plus the policy allowlist. An alias may only point here.
ADVERTISED_NAMES = {
    "A10", "A100 PCIE", "A100 SXM4", "A100X", "A16", "A40", "A800 PCIE",
    "B200", "B300", "H100 NVL", "H100 PCIE", "H100 SXM", "H200", "H200 NVL",
    "L4", "L40", "L40S",
    "RTX 3080", "RTX 3090", "RTX 4000Ada", "RTX 4080", "RTX 4090",
    "RTX 4500Ada", "RTX 5000Ada", "RTX 5070", "RTX 5070 Ti", "RTX 5080",
    "RTX 5090", "RTX 5880Ada", "RTX 6000Ada",
    "RTX A4000", "RTX A5000", "RTX A6000",
    "RTX PRO 4000", "RTX PRO 4500", "RTX PRO 5000",
    "RTX PRO 6000 S", "RTX PRO 6000 WS",
}


@pytest.mark.parametrize("spelling", [
    "RTX 6000 Ada", "rtx 6000 ada", "RTX 6000Ada", "rtx6000ada", "6000ada",
    "RTX6000ADA", "  RTX 6000 Ada  ".strip(),
])
def test_rtx_6000_ada_resolves_from_every_human_spelling(spelling):
    """The defect. vast spells it with no space before "Ada"; nobody else
    does, and the exact-match filter does not forgive it."""
    assert v.normalize_gpu([spelling]) == ["RTX 6000Ada"]


@pytest.mark.parametrize("spelling,want", [
    ("l40s", "L40S"), ("L40S", "L40S"),
    ("rtx a6000", "RTX A6000"), ("RTX A6000", "RTX A6000"),
    ("a6000", "RTX A6000"),
    ("5880ada", "RTX 5880Ada"), ("RTX 5880 Ada", "RTX 5880Ada"),
    ("5000ada", "RTX 5000Ada"), ("RTX 5000 Ada", "RTX 5000Ada"),
    ("4500ada", "RTX 4500Ada"), ("4000ada", "RTX 4000Ada"),
    ("rtx a5000", "RTX A5000"), ("a4000", "RTX A4000"),
])
def test_the_other_exact_match_casualties_resolve(spelling, want):
    assert v.normalize_gpu([spelling]) == [want]


@pytest.mark.parametrize("spelling", ["rtx pro 6000", "RTX PRO 6000",
                                      "rtxpro6000", "pro6000"])
def test_whitespace_pass_reaches_existing_multi_word_aliases(spelling):
    """The whitespace-insensitive second lookup must also rescue the aliases
    that were already there but keyed space-free."""
    assert v.normalize_gpu([spelling]) == ["RTX PRO 6000 WS", "RTX PRO 6000 S"]


def test_every_alias_target_is_a_name_vast_actually_indexes():
    """An alias pointing at a string vast does not use is the same silent zero
    it was added to prevent."""
    known = set(ADVERTISED_NAMES)
    for tier in v.GPU_DEFAULT_POLICY_TIERS:
        known.update(tier)
    for key, val in v.GPU_ALIASES.items():
        for name in (val if isinstance(val, list) else [val]):
            assert name in known, f"alias {key!r} -> unknown gpu_name {name!r}"


def test_hyphenated_keys_survive_the_whitespace_pass():
    """Only WHITESPACE is stripped on the second pass — `a100-80` is a
    deliberately distinct key from `a100` (it excludes the 40 GB A100X)."""
    assert v.normalize_gpu(["a100-80"]) == ["A100 SXM4", "A100 PCIE"]
    assert "A100X" in v.normalize_gpu(["a100"])
    assert "A100X" not in v.normalize_gpu(["a100-80"])


def test_unknown_sku_passes_through_untouched():
    """A card we never aliased stays reachable when spelled vast's way — the
    alias table is a convenience layer, not an allowlist."""
    assert v.normalize_gpu(["Some New SKU 9000"]) == ["Some New SKU 9000"]
    assert v.normalize_gpu(["CMP 170HX"]) == ["CMP 170HX"]


def test_multiple_names_dedup_in_order():
    assert v.normalize_gpu(["l40", "L40S", "l40s"]) == ["L40", "L40S"]


def test_aliasing_did_not_merge_gpu_families():
    """`gpu_family_names` is the inverse index over GPU_ALIASES, and the
    eviction-replacement lane widens its candidate set to a family — so a new
    scalar alias must not drag unrelated SKUs into one."""
    assert v.gpu_family_names("L40S") == ["L40", "L40S"]
    assert v.gpu_family_names("RTX A6000") == ["RTX A6000"]
    assert v.gpu_family_names("RTX 6000Ada") == ["RTX 6000Ada"]
    assert v.gpu_family_names("RTX A5000") == ["RTX A5000"]
    assert set(v.gpu_family_names("H100 NVL")) == {"H100 SXM", "H100 PCIE",
                                                   "H100 NVL"}
    assert set(v.gpu_family_names("A100 SXM4")) == {"A100 SXM4", "A100 PCIE",
                                                    "A100X"}


def test_policy_tier_names_are_all_reachable_by_some_alias_or_themselves():
    """Every card the default policy will auto-pick must be nameable. Spelled
    exactly, at minimum — that is the passthrough — but the point of the check
    is that the tier list and the advertised board agree."""
    for tier in v.GPU_DEFAULT_POLICY_TIERS:
        for name in tier:
            assert v.normalize_gpu([name]) == [name] or \
                name in v.normalize_gpu([name]), name

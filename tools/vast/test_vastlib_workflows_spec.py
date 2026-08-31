"""`vastlib.workflows.spec` — the ported workflow DSL, held to its two traps.

Why this file exists
--------------------
`test_workflow.py` drives the FLAT `workflow.py`, which stays live and
unedited through the add-only phase (plan §8), so it cannot see the ported
copy at all. Two properties of that copy survive a move only if something here
checks them:

1. **Class identity across the bare-name authoring contract** (manifest
   `workflows-core.json` H1). Every authored spec on disk says
   `from workflow import Workflow, ...` — resolved off `sys.path` — and the
   controller path-loads that file and `isinstance`-checks the result. Two
   class objects means every spec fails to load, with the maximally misleading
   "must define a module-level WORKFLOW". `test_bare_name_import_is_a_different_
   class_today` proves the split is REAL right now (which is why the flat module
   is still the runtime authority), and
   `test_sys_modules_alias_closes_the_identity_split` proves the step-7 shim
   recipe in the module docstring actually closes it. If someone "cleans up" by
   deleting the flat file, the first test goes green for the wrong reason — so
   it also asserts the flat module still exists.

2. **The frozen vocabularies** (H7). `RETRY_CLASSES`, `TEARDOWN_CHOICES` and
   `RENTAL_CHOICES` cross boundaries no type checker can see: workflowmeta's
   subset check, B2-round-tripped `spec.json`, and — for the hyphenated
   `on-demand` — the already-landed `vastlib.market.offers` normalizer, which
   is coupled BY VALUE with no import edge. They are asserted against both the
   literal values and the flat module's, in both directions.

The shape checks themselves are re-tested here (not merely trusted) because
each one is a security or wire boundary someone could "simplify": the
`manifest_path` traversal refusal, the NAMES-only `secrets` rule, and the
bare-str/bool rejections that a naive `isinstance` rewrite loses.

What is deliberately NOT here
-----------------------------
* No repoint of `test_workflow.py`, `test_workflowmeta.py` or
  `test_workflow_preflight.py`. They keep driving the flat module and migrate
  with their callers at plan steps 6-7.
* No cross-object spec validation — that is `workflowmeta`'s, and it stays
  there precisely so it also binds on JSON-reloaded specs.
* No network, no subprocess. This module has neither; the one market
  assertion stubs `vastlib.core.api.request_soft`.

Provenance: created 2026-08-16 alongside `vastlib/workflows/spec.py`, plan §8
step 5, manifest `workflows-core.json`.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

VAST_DIR = Path(__file__).resolve().parent
if str(VAST_DIR) not in sys.path:                      # entry-script sys.path shape
    sys.path.insert(0, str(VAST_DIR))

import workflow as flat_workflow                       # noqa: E402  twin, still live

from vastlib.core import api                           # noqa: E402
from vastlib.market import offers                      # noqa: E402
from vastlib.workflows import spec                     # noqa: E402


# --------------------------------------------------------------------------- #
# H1 — class identity under the bare-name authoring contract
# --------------------------------------------------------------------------- #
_SPEC_SRC = """\
from workflow import JobStage, ResourceProfile, Workflow

WORKFLOW = Workflow(
    version=1, name="wf", budget_usd=1.0, max_wall_s=60, teardown="stop",
    profiles={"p": ResourceProfile(image="img:tag")},
    stages=(JobStage(name="s1", bundle="b", profile="p"),),
)
"""


def _load_authored_spec(path: Path):
    """The controller's loader shape (workflowctl.load_workflow_module:234):
    `spec_from_file_location` + exec, then an isinstance check by the CALLER."""
    loader_spec = importlib.util.spec_from_file_location("authored_wf", str(path))
    assert loader_spec is not None and loader_spec.loader is not None
    mod = importlib.util.module_from_spec(loader_spec)
    loader_spec.loader.exec_module(mod)
    return mod.WORKFLOW


def test_flat_workflow_module_still_exists():
    """The whole add-only phase rests on it. If this fails, the identity test
    below is passing for the wrong reason and every authored spec on disk
    (in-repo: tools/witness/workflows/e2-paired/workflow.py) is broken."""
    assert (VAST_DIR / "workflow.py").is_file()
    assert sys.modules["workflow"] is flat_workflow


def test_importing_the_port_does_not_hijack_the_bare_name():
    """Zone P may not touch `sys.modules` (or `sys.path`). If `spec.py` ever
    grows the `sys.modules.setdefault` that `jobmatrix.py:72` carries, authored
    specs start resolving to the package class while `workflowctl` — still
    `from workflow import Workflow` — checks the flat one."""
    assert sys.modules["workflow"] is not spec
    src = (VAST_DIR / "vastlib" / "workflows" / "spec.py").read_text()
    body = src.split('"""', 2)[2]              # skip the docstring (it QUOTES the recipe)
    assert "sys.modules" not in body
    assert "sys.path" not in body


def test_the_bare_name_and_the_port_are_ONE_class_since_the_shim(tmp_path):
    """THE hazard, executable — now in its closed state (INVERTED at step 7).

    Before the shim this asserted the split was open: a path-loaded spec
    resolved `Workflow` from the flat module, so an isinstance check against
    the ported class failed and the caller's error ("must define a module-level
    WORKFLOW") named the wrong cause entirely. `tools/vast/workflow.py` is now a
    pure re-export of `vastlib.workflows.spec`, so BOTH resolutions yield one
    class object. If this ever goes red, someone gave the flat file a body
    again — that is the two-class break coming back, not a test to relax."""
    p = tmp_path / "workflow_spec.py"
    p.write_text(_SPEC_SRC)
    wf = _load_authored_spec(p)

    assert isinstance(wf, flat_workflow.Workflow)      # the authoring contract
    assert isinstance(wf, spec.Workflow)               # ... and the port
    assert flat_workflow.Workflow is spec.Workflow


def test_sys_modules_alias_closes_the_identity_split(tmp_path, monkeypatch):
    """The step-7 recipe, proven rather than asserted in prose: with the bare
    name aliased to the ported module (what the flat re-export shim will do),
    the authored file and the loader share ONE class object."""
    monkeypatch.setitem(sys.modules, "workflow", spec)
    p = tmp_path / "workflow_spec.py"
    p.write_text(_SPEC_SRC)
    wf = _load_authored_spec(p)

    assert isinstance(wf, spec.Workflow)
    assert wf.stages[0].profile == "p"


# --------------------------------------------------------------------------- #
# H7 — the frozen vocabularies, both directions
# --------------------------------------------------------------------------- #
def test_frozen_vocabularies_are_byte_identical_to_the_flat_module():
    assert spec.RETRY_CLASSES == frozenset(
        {"infrastructure", "entrypoint", "postcondition"})
    assert spec.TEARDOWN_CHOICES == ("stop", "destroy")
    assert spec.RENTAL_CHOICES == ("bid", "on-demand")
    assert spec.STAGE_NAME_RE.pattern == r"^[a-z0-9][a-z0-9-]{0,31}$"

    assert spec.RETRY_CLASSES == flat_workflow.RETRY_CLASSES
    assert spec.TEARDOWN_CHOICES == flat_workflow.TEARDOWN_CHOICES
    assert spec.RENTAL_CHOICES == flat_workflow.RENTAL_CHOICES
    assert spec.STAGE_NAME_RE.pattern == flat_workflow.STAGE_NAME_RE.pattern


def test_rental_spelling_is_the_hyphenated_one_and_market_accepts_it(monkeypatch):
    """Direction: `market` depends on THIS constant's spelling (offers.py
    normalizes both `ondemand` and `on-demand`), and there is no import edge —
    market sits BELOW workflows in the ring DAG, so there must not be one.
    test_vastlib_market.py:206 pins the literal strings; this pins the CONSTANT,
    so normalizing it to `on_demand`/`ondemand` here fails on this side too."""
    assert "on-demand" in spec.RENTAL_CHOICES
    assert "on_demand" not in spec.RENTAL_CHOICES
    assert "ondemand" not in spec.RENTAL_CHOICES

    seen = []

    def _fake(method, path, body=None, *a, **k):
        seen.append(body)
        return True, {"offers": []}, None

    monkeypatch.setattr(api, "request_soft", _fake)
    for rental in spec.RENTAL_CHOICES:
        offers.pick_offers(rental=rental, gpu=["h100"], any_inet=True)
    assert [b["type"] for b in seen] == ["bid", "ondemand"]


def test_no_vastlib_import_edge_out_of_spec():
    """`spec` is the bottom of the workflows ring: workflowmeta/workflowctl sit
    above it, and market is coupled by value only. An import from here would
    invert the DAG (import-linter would catch the market one; this catches the
    intent for all of them)."""
    src = (VAST_DIR / "vastlib" / "workflows" / "spec.py").read_text()
    body = src.split('"""', 2)[2]
    assert "import vastlib" not in body
    assert "from vastlib" not in body


# --------------------------------------------------------------------------- #
# the shape checks — the two that are boundaries, then the coercion footguns
# --------------------------------------------------------------------------- #
def test_manifest_path_traversal_is_refused():
    """`manifest_path` is interpolated RAW into jobs/<job_id>/results/, and the
    same constructor runs on B2-round-tripped foreign spec.json via
    workflowmeta.spec_from_dict — so this is the only thing standing between a
    foreign spec and a write outside the results frame."""
    spec.ArtifactContract(kind="k", manifest_path="sub/dir/manifest.json")
    for bad in ("/abs/manifest.json", "../escape.json", "a/../../b.json"):
        with pytest.raises(spec.WorkflowError) as ei:
            spec.ArtifactContract(kind="k", manifest_path=bad)
        assert "must stay inside the results frame" in str(ei.value)


def test_secrets_are_names_only():
    """Structural, not advisory: the field being a tuple of str is what keeps a
    credential VALUE out of the spec and out of its canonical JSON."""
    spec.JobStage(name="s", bundle="b", profile="p", secrets=("B2_WRITE_KEY",))
    for bad in ("B2_KEY=hunter2", "has space"):
        with pytest.raises(spec.WorkflowError) as ei:
            spec.JobStage(name="s", bundle="b", profile="p", secrets=(bad,))
        assert "NAMES only" in str(ei.value)
    # A mapping does not raise — it collapses to its KEYS, which is the
    # structural guarantee itself: the value cannot reach the spec or its
    # canonical JSON even when an author passes one. (Same in the flat copy;
    # asserted rather than assumed, because "it raises" is the intuitive-but-
    # wrong reading of the docstring.)
    st = spec.JobStage(name="s", bundle="b", profile="p",
                       secrets={"B2_WRITE_KEY": "hunter2"})
    assert st.secrets == ("B2_WRITE_KEY",)
    assert "hunter2" not in repr(st)


def test_bare_str_is_not_a_sequence_of_str():
    """The classic 'a string is iterable' footgun: `after="dep"` must NOT
    silently become ('d','e','p')."""
    with pytest.raises(spec.WorkflowError) as ei:
        spec.JobStage(name="s", bundle="b", profile="p", after="dep")
    assert "not a bare str/bytes" in str(ei.value)
    assert spec.JobStage(name="s", bundle="b", profile="p",
                         after=["dep"]).after == ("dep",)


def test_bool_is_not_a_number_and_negatives_are_refused():
    """`bool` is an `int` subclass — an `isinstance(x, int)` rewrite that drops
    the explicit bool rejection accepts `budget_usd=True` as $1."""
    with pytest.raises(spec.WorkflowError):
        spec.Workflow(version=1, name="w", budget_usd=True, max_wall_s=0,
                      teardown="stop", profiles={}, stages=())
    with pytest.raises(spec.WorkflowError):
        spec.Workflow(version=1, name="w", budget_usd=-1.0, max_wall_s=0,
                      teardown="stop", profiles={}, stages=())
    with pytest.raises(spec.WorkflowError):
        spec.ResourceProfile(image="i", num_gpus=True)


def test_slug_error_quotes_the_pattern():
    """The pattern text is in the message on purpose — the slug bounds a raw B2
    key and the deterministic stage-attempt JOB_ID, so the fix has to be
    obvious from the error alone."""
    with pytest.raises(spec.WorkflowError) as ei:
        spec.JobStage(name="Not A Slug", bundle="b", profile="p")
    assert spec.STAGE_NAME_RE.pattern in str(ei.value)


def test_vocabulary_membership_is_enforced_where_it_is_declared():
    """`rental`/`teardown` are checked HERE; `retry_on` deliberately is not —
    its subset check lives in workflowmeta so it binds on JSON-reloaded specs
    too. A port that "helpfully" added it here would double-enforce in one
    place and still miss the reload path."""
    with pytest.raises(spec.WorkflowError):
        spec.ResourceProfile(image="i", rental="ondemand")     # the OTHER spelling
    with pytest.raises(spec.WorkflowError):
        spec.Workflow(version=1, name="w", budget_usd=0, max_wall_s=0,
                      teardown="park", profiles={}, stages=())
    assert spec.RetryPolicy(max_attempts=2, retry_on=("not-a-class",)).retry_on \
        == ("not-a-class",)


def test_frozen_dataclasses_normalize_through_object_setattr():
    """The frozen-dataclass mutation idiom: `__post_init__` coerces via
    `object.__setattr__`, so iterables become tuples and mappings become plain
    dicts while the instance stays immutable to callers."""
    prof = spec.ResourceProfile(image="i", gpu=["h100", "a100"], geo=iter(("US",)))
    assert prof.gpu == ("h100", "a100") and prof.geo == ("US",)
    with pytest.raises(Exception):                 # FrozenInstanceError
        prof.image = "other"

    st = spec.JobStage(name="s", bundle="b", profile="p",
                       outputs={"o": spec.ArtifactContract(kind="k",
                                                           manifest_path="m.json")})
    assert isinstance(st.outputs, dict) and set(st.outputs) == {"o"}


def test_wrong_member_types_are_named_by_key():
    for kwargs, needle in (
        ({"inputs": {"i": "not-an-inputref"}}, "must be an InputRef"),
        ({"outputs": {"o": "not-a-contract"}}, "must be an ArtifactContract"),
        ({"retry": {"max_attempts": 2}}, "must be a RetryPolicy"),
    ):
        with pytest.raises(spec.WorkflowError) as ei:
            spec.JobStage(name="s", bundle="b", profile="p", **kwargs)
        assert needle in str(ei.value)

    with pytest.raises(spec.WorkflowError) as ei:
        spec.Workflow(version=1, name="w", budget_usd=0, max_wall_s=0,
                      teardown="stop", profiles={"p": "not-a-profile"}, stages=())
    assert "must be a ResourceProfile" in str(ei.value)

    with pytest.raises(spec.WorkflowError) as ei:
        spec.Workflow(version=1, name="w", budget_usd=0, max_wall_s=0,
                      teardown="stop", profiles={}, stages=("not-a-stage",))
    assert "must be JobStage" in str(ei.value)


def test_version_pin_and_required_fields():
    with pytest.raises(spec.WorkflowError) as ei:
        spec.Workflow(version=2, name="w", budget_usd=0, max_wall_s=0,
                      teardown="stop", profiles={}, stages=())
    assert "must be 1" in str(ei.value)
    with pytest.raises(spec.WorkflowError):
        spec.ResourceProfile(image="")
    with pytest.raises(spec.WorkflowError):
        spec.InputRef(stage="s", artifact="a", dest="")
    with pytest.raises(spec.WorkflowError):
        spec.RetryPolicy(max_attempts=0)


def test_ported_and_flat_agree_on_every_refusal():
    """Behavior-preserving, checked rather than claimed: the same 12 inputs
    raise (or don't) identically in both copies, with the same message. A drift
    here is a found port defect, not a test to update (plan §7.4)."""
    cases = [
        ("ResourceProfile", {"image": ""}),
        ("ResourceProfile", {"image": "i", "rental": "ondemand"}),
        ("ResourceProfile", {"image": "i", "num_gpus": 0}),
        ("ResourceProfile", {"image": "i", "geo": "US"}),
        ("ResourceProfile", {"image": "i", "max_bid": -1}),
        ("ArtifactContract", {"kind": "k", "manifest_path": "../x"}),
        ("ArtifactContract", {"kind": "", "manifest_path": "x"}),
        ("InputRef", {"stage": "", "artifact": "a", "dest": "d"}),
        ("RetryPolicy", {"max_attempts": True}),
        ("JobStage", {"name": "S", "bundle": "b", "profile": "p"}),
        ("JobStage", {"name": "s", "bundle": "", "profile": "p"}),
        ("JobStage", {"name": "s", "bundle": "b", "profile": "p",
                      "secrets": ("A=1",)}),
    ]
    for cls_name, kwargs in cases:
        ported, flat = getattr(spec, cls_name), getattr(flat_workflow, cls_name)
        with pytest.raises(spec.WorkflowError) as p_ei:
            ported(**kwargs)
        with pytest.raises(flat_workflow.WorkflowError) as f_ei:
            flat(**kwargs)
        assert str(p_ei.value) == str(f_ei.value), cls_name

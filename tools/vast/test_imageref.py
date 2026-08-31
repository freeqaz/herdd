"""Stale-image classification and the TTL'd resolver (velvet plan P1).

The incident these exist for: three frontier-wave jobs died within seconds on
box 46240842 because its baked eval-env predated a module they imported. The
staleness SIGNAL already existed; nothing that schedules work consulted it.

P1 is alarm-only by design, so nothing here asserts a refusal — the point of
this layer is that the four states are computed CORRECTLY before any of them is
allowed to block work in P3. The state that matters most is `unresolved`: it is
the one a two-valued gate would silently collapse into "stale" (refusing every
public-registry box) or into "fresh" (defeating the whole feature).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import imageref  # noqa: E402

REG = imageref.R2_REGISTRY_HOST
OURS = f"{REG}/train:t215-latest"
GONE = "registry.gitlab.com"          # RETIRED — cut as a registry 2026-08-22
DIG_A = "sha256:" + "a" * 64
DIG_B = "sha256:" + "b" * 64


def s(**kw):
    kw.setdefault("image", OURS)
    kw.setdefault("stamped_digest", DIG_A)
    kw.setdefault("current_digest", DIG_A)
    return imageref.classify_image_staleness(**kw)[0]


# --- the four states ---------------------------------------------------------

def test_matching_digest_is_fresh():
    assert s() == imageref.IMG_FRESH


def test_moved_tag_is_stale():
    assert s(current_digest=DIG_B) == imageref.IMG_STALE


def test_unresolvable_is_its_OWN_state_not_stale_and_not_fresh():
    """The load-bearing distinction. `image_tag_digest` returns one ambiguous
    None for 'not our registry' / 'no creds' / 'API failure', so collapsing this
    into stale refuses healthy boxes and collapsing it into fresh defeats the
    feature. It must be separable so P3 can HOLD on it specifically."""
    assert s(current_digest=None) == imageref.IMG_UNRESOLVED


def test_foreign_registry_is_not_applicable_without_any_lookup():
    """The documented serve lane runs vllm/vllm-openai:latest. A two-valued
    gate would refuse every such box."""
    for img in ["vllm/vllm-openai:latest", "docker.io/library/ubuntu:22.04"]:
        assert imageref.classify_image_staleness(
            image=img, stamped_digest=DIG_A,
            current_digest=None)[0] == imageref.IMG_NOT_APPLICABLE, img


def test_by_digest_ref_cannot_drift_even_if_the_tag_moved():
    """Content-addressed refs are not_applicable BY CONSTRUCTION — asserted
    with a contradicting current_digest so a resolution-order bug can't pass."""
    assert imageref.classify_image_staleness(
        image=f"{REG}/train@sha256:" + "c" * 64,
        stamped_digest=DIG_A,
        current_digest=DIG_B)[0] == imageref.IMG_NOT_APPLICABLE


def test_unstamped_box_is_silent_not_alarming():
    """Legacy/unstamped boxes must not alarm: an alarm that fires on every old
    box is one operators learn to ignore, which costs the real signal."""
    assert s(stamped_digest=None) == imageref.IMG_NOT_APPLICABLE
    assert s(stamped_digest="") == imageref.IMG_NOT_APPLICABLE


def test_unparseable_and_empty_refs_do_not_crash():
    for img in ["", None, "ubuntu", "weird::ref"]:
        st, why = imageref.classify_image_staleness(
            image=img, stamped_digest=DIG_A, current_digest=DIG_B)
        assert st == imageref.IMG_NOT_APPLICABLE and why, img


def test_no_env_var_can_add_a_registry_to_the_set(monkeypatch):
    """`$GITLAB_REGISTRY` used to widen the set at run time. It no longer does,
    and neither does anything else: the set is a literal. An env-widenable
    predicate is how a host nobody audited becomes "ours" by accident, and the
    rollback lane it existed for was cut on 2026-08-22."""
    monkeypatch.setenv("GITLAB_REGISTRY", "registry.gitlab.com")
    assert imageref.our_registry_hosts() == (imageref.R2_REGISTRY_HOST,)
    assert not imageref.is_our_registry("registry.gitlab.com")
    assert imageref.classify_image_staleness(
        image="registry.gitlab.com/x/y:tag", stamped_digest=DIG_A,
        current_digest=DIG_B)[0] == imageref.IMG_NOT_APPLICABLE


# --- the registry set: R2 is ours, and was silently not ---------------------- #
R2 = imageref.R2_REGISTRY_HOST
R2_IMG = f"{R2}/train:t215-latest"


def test_the_R2_DEFAULT_IMAGE_classifies_and_is_not_short_circuited():
    """THE 2026-08-19 DEFECT. This classifier keyed on $GITLAB_REGISTRY alone,
    so every registry.example.com ref — i.e. herdd.yaml's `default_image`,
    i.e. the whole fleet — returned `not_applicable` and the fleetd health
    signal and the workflow resume gate were inert for the images we run.
    Asserted on the STALE leg, where the old code was maximally wrong."""
    st, why = imageref.classify_image_staleness(
        image=R2_IMG, stamped_digest=DIG_A, current_digest=DIG_B)
    assert st == imageref.IMG_STALE, why


def test_the_retired_registry_is_refused_not_merely_unclassified(monkeypatch):
    """A cut registry is NOT the same as a foreign one, and collapsing the two
    is the expensive mistake. docker.io is `not_applicable` — nothing to
    compare, proceed. registry.gitlab.com can never pull at all, so the launch
    path refuses it outright (`spec._require_image`); classification is not the
    layer that should be answering for it, and `is_retired_registry` is how a
    caller tells the two apart."""
    monkeypatch.delenv("GITLAB_REGISTRY", raising=False)
    assert imageref.is_retired_registry(GONE)
    assert not imageref.is_our_registry(GONE)
    assert not imageref.is_retired_registry("docker.io")
    assert not imageref.is_retired_registry(imageref.R2_REGISTRY_HOST)
    assert imageref.classify_image_staleness(
        image=f"{GONE}/example/project:train-t211-latest", stamped_digest=DIG_A,
        current_digest=DIG_B)[0] == imageref.IMG_NOT_APPLICABLE


def test_a_foreign_registry_is_still_not_applicable(monkeypatch):
    """The widening must not swallow the public serve lane: `unresolved` on a
    docker.io box would make the P3 gate HOLD every one of them."""
    monkeypatch.delenv("GITLAB_REGISTRY", raising=False)
    for img in ["vllm/vllm-openai:latest", "docker.io/library/ubuntu:22.04"]:
        assert imageref.classify_image_staleness(
            image=img, stamped_digest=DIG_A,
            current_digest=None)[0] == imageref.IMG_NOT_APPLICABLE, img


# --- the missing R2 secret must be LOUD, not indistinguishable from fresh ---- #

def test_missing_R2_secret_names_itself_in_the_unresolved_reason(monkeypatch):
    """A None digest from an unset credential is not evidence about the image.
    The reason string has to say which, or an operator reads UNRESOLVED as a
    flaky network and shrugs."""
    monkeypatch.delenv(imageref.R2_SECRET_ENV, raising=False)
    st, why = imageref.classify_image_staleness(
        image=R2_IMG, stamped_digest=DIG_A, current_digest=None)
    assert st == imageref.IMG_UNRESOLVED
    assert imageref.R2_SECRET_ENV in why and "not" in why.lower()


def test_r2_secret_missing_predicate_is_scoped_to_the_R2_host(monkeypatch):
    monkeypatch.delenv(imageref.R2_SECRET_ENV, raising=False)
    assert imageref.r2_secret_missing(R2_IMG)
    assert imageref.r2_secret_missing(OURS)          # OURS *is* the R2 host
    assert not imageref.r2_secret_missing(f"{GONE}/example/project:t")
    assert not imageref.r2_secret_missing("vllm/vllm-openai:latest")
    assert not imageref.r2_secret_missing("")
    monkeypatch.setenv(imageref.R2_SECRET_ENV, "s3cr3t")
    assert not imageref.r2_secret_missing(R2_IMG)


def test_creds_degradation_warns_on_stderr_exactly_once(monkeypatch, capsys):
    """`_r2_skopeo_creds` returning [] means the next skopeo call gets refused
    and the digest comes back None. Silent is the bug; a warning per box would
    be its own bug, so it fires once per process."""
    monkeypatch.delenv(imageref.R2_SECRET_ENV, raising=False)
    imageref.reset_secret_warning()
    assert imageref._r2_skopeo_creds(R2_IMG) == []
    assert imageref._r2_skopeo_creds(R2_IMG) == []
    err = capsys.readouterr().err
    assert err.count("!! image staleness is UNKNOWN") == 1, err
    assert imageref.R2_SECRET_ENV in err


def test_no_warning_for_a_non_R2_ref(monkeypatch, capsys):
    monkeypatch.delenv(imageref.R2_SECRET_ENV, raising=False)
    imageref.reset_secret_warning()
    assert imageref._r2_skopeo_creds(f"{GONE}/example/project:t") == []
    assert capsys.readouterr().err == ""


def test_every_state_carries_a_reason():
    for kw in [dict(current_digest=DIG_A), dict(current_digest=DIG_B),
               dict(current_digest=None), dict(stamped_digest=None),
               dict(image="ubuntu")]:
        full = dict(image=OURS, stamped_digest=DIG_A, current_digest=DIG_A)
        full.update(kw)
        st, why = imageref.classify_image_staleness(**full)
        assert isinstance(why, str) and why.strip(), (st, kw)


# --- the TTL'd resolver ------------------------------------------------------

def test_ttl_cache_hits_then_expires():
    """Both directions matter in a days-long daemon: caching a SUCCESS forever
    means a real image push is never noticed (the exact footgun), and caching a
    FAILURE forever pins the box's verdict for the daemon's life."""
    imageref.clear_ttl_cache()
    calls = []

    def fake(img):
        calls.append(img)
        return DIG_A if len(calls) == 1 else DIG_B

    d, st = imageref.resolve_tag_digest_ttl(OURS, ttl_s=100, now=1000,
                                            resolver=fake)
    assert (d, st) == (DIG_A, "miss")
    d, st = imageref.resolve_tag_digest_ttl(OURS, ttl_s=100, now=1099,
                                            resolver=fake)
    assert (d, st) == (DIG_A, "hit") and len(calls) == 1
    d, st = imageref.resolve_tag_digest_ttl(OURS, ttl_s=100, now=1100,
                                            resolver=fake)
    assert (d, st) == (DIG_B, "expired"), "a moved tag must be seen at expiry"


def test_ttl_cache_expires_a_cached_FAILURE_too():
    """A None cached forever under a fail-closed consumer would pin the box
    unresolved permanently — the reason this is not the plain _digest_cache."""
    imageref.clear_ttl_cache()
    seq = [None, DIG_A]

    def fake(_img):
        return seq.pop(0)

    assert imageref.resolve_tag_digest_ttl(OURS, ttl_s=10, now=0,
                                           resolver=fake)[0] is None
    assert imageref.resolve_tag_digest_ttl(OURS, ttl_s=10, now=5,
                                           resolver=fake)[0] is None   # cached
    assert imageref.resolve_tag_digest_ttl(OURS, ttl_s=10, now=10,
                                           resolver=fake)[0] == DIG_A


def test_ttl_cache_is_separate_from_the_process_cache():
    """Adding expiry to `_digest_cache` would silently change ls/launch
    behavior; long-lived callers opt in instead."""
    imageref.clear_ttl_cache()
    imageref._digest_cache.clear()
    imageref.resolve_tag_digest_ttl(OURS, ttl_s=10, now=0,
                                    resolver=lambda _i: DIG_A)
    assert imageref._digest_cache == {}


def test_ttl_resolution_runs_outside_the_lock():
    """A registry call under the lock would serialize the whole fleet's
    resolution behind one slow host. Proven by re-entering from the resolver:
    a lock held across the call would deadlock here."""
    imageref.clear_ttl_cache()

    def reentrant(_img):
        # a different key, so this is a genuine second acquisition
        imageref.resolve_tag_digest_ttl("other:tag", ttl_s=10, now=0,
                                        resolver=lambda _x: DIG_B)
        return DIG_A

    assert imageref.resolve_tag_digest_ttl(
        OURS, ttl_s=10, now=0, resolver=reentrant)[0] == DIG_A


# --- pin_relaunch_image, post-rolling-ruling (2026-08-04) ------------------- #
# HISTORY, because the inversion here is easy to misread as a regression.
# velvet P4a made a relaunch REPLAY the recorded digest when the tag had moved,
# so an evicted run's second half reproduced its first. The owner ruling of
# 2026-08-04 ("always pinned to the latest image. no hash pins or anything")
# retires that: resurrecting an old digest is exactly the pinning the ruling
# forbids, and it happened invisibly on the eviction path.
#
# The split that these tests enforce: the DECISION rolls forward, the
# OBSERVATION survives. Nothing about digest RECORDING changed — the launch
# stamp, spec.json's image_digest and classify_image_staleness are untouched,
# and a moved tag must still be REPORTED, because that report is now the only
# notice anyone gets that a run straddled two images.

def test_a_moved_tag_rolls_FORWARD_and_never_resurrects_the_old_digest():
    """The behaviour change. Was: pin `repo@<recorded>`. Now: take the tag."""
    ref, state, why = imageref.pin_relaunch_image(
        image=OURS, spec_digest=DIG_A, current_digest=DIG_B)
    assert state == imageref.PIN_ROLLED
    assert ref == OURS, "a relaunch must not resurrect the launch-time digest"
    assert "@sha256:" not in ref


def test_the_roll_forward_still_REPORTS_the_split_because_nothing_else_will():
    """Rolling silently would be the actual regression. The reason string is the
    only surviving signal that a run's halves ran different bytes, and all three
    herdd call sites print it."""
    _ref, _state, why = imageref.pin_relaunch_image(
        image=OURS, spec_digest=DIG_A, current_digest=DIG_B)
    assert "ROLLED FORWARD" in why
    assert DIG_A[:19] in why and DIG_B[:19] in why
    assert "second half" in why


def test_an_unresolvable_tag_replays_the_tag_and_admits_it_cannot_tell():
    ref, state, why = imageref.pin_relaunch_image(
        image=OURS, spec_digest=DIG_A, current_digest=None)
    assert (ref, state) == (OURS, imageref.PIN_UNVERIFIED)
    assert "unable to say" in why


def test_an_unmoved_tag_is_left_exactly_as_it_was():
    assert imageref.pin_relaunch_image(
        image=OURS, spec_digest=DIG_A, current_digest=DIG_A) == (
            OURS, imageref.PIN_MATCH, "tag still resolves to the launch digest")


def test_a_spec_with_no_recorded_digest_is_todays_behaviour_untouched():
    """Every spec written before P4a. The feature must not change them."""
    ref, state, _ = imageref.pin_relaunch_image(
        image=OURS, spec_digest=None, current_digest=DIG_B)
    assert (ref, state) == (OURS, imageref.PIN_NONE)


def test_a_pre_ruling_by_digest_spec_is_still_replayed_verbatim():
    """Old specs written before the de-pin may carry a by-digest ref. Replaying
    it verbatim is still the faithful reproduction of THAT launch — we do not
    rewrite history, we just stop writing new pins."""
    pinned = f"registry.gitlab.com/f/d@{DIG_A}"
    ref, state, why = imageref.pin_relaunch_image(
        image=pinned, spec_digest=DIG_A, current_digest=DIG_B)
    assert (ref, state) == (pinned, imageref.PIN_NONE)
    assert "predates the rolling ruling" in why


def test_no_input_shape_can_make_this_emit_a_by_digest_ref():
    """The ratchet. Whatever goes in — registry port, tagless ref, moved tag —
    what comes out is never a new `@sha256:` pin."""
    for img in ["localhost:5000/f/d:v1", "registry.gitlab.com/f/d",
                OURS, "vllm/vllm-openai:latest"]:
        for cur in [DIG_B, DIG_A, None]:
            ref, _s, _w = imageref.pin_relaunch_image(
                image=img, spec_digest=DIG_A, current_digest=cur)
            assert ref == img, (img, cur)


# --- image_ref_digest: absence is now fatal, "could not ask" is not --------- #

def test_a_by_digest_ref_the_registry_does_NOT_have_fails_closed(monkeypatch):
    """The 2026-08-02 incident. `sha256:193f9c25…` was read off a local podman
    inspect and pinned into three consumers; skopeo said `manifest unknown` and
    this function returned the digest anyway, because a by-digest ref was
    treated as self-certifying. Self-certifying only ever meant such a ref
    cannot DRIFT — never that it cannot be ABSENT."""
    imageref._ref_digest_cache.clear()
    monkeypatch.setattr(imageref, "_skopeo_digest", lambda r: None)
    monkeypatch.setattr(imageref, "_skopeo_absent", lambda r: True)
    assert imageref.image_ref_digest(f"registry.gitlab.com/f/d@{DIG_A}") is None


def test_a_probe_that_could_not_ASK_still_returns_the_in_ref_digest(monkeypatch):
    """The other half. No skopeo, a timeout, a 5xx, an auth failure — none of
    those are evidence of absence, and treating them as such would desync
    _plan_online from box_resolver on every network blip."""
    imageref._ref_digest_cache.clear()
    monkeypatch.setattr(imageref, "_skopeo_digest", lambda r: None)
    monkeypatch.setattr(imageref, "_skopeo_absent", lambda r: False)
    assert imageref.image_ref_digest(f"registry.gitlab.com/f/d@{DIG_A}") == DIG_A


def test_skopeo_absent_separates_a_real_404_from_an_auth_failure(monkeypatch):
    class P:
        def __init__(self, rc, err): self.returncode, self.stderr, self.stdout = rc, err, ""
    monkeypatch.setattr(imageref.shutil, "which", lambda n: "/usr/bin/skopeo")
    cases = {
        'reading manifest: manifest unknown': True,
        'Error: ... 404 Not Found': True,
        'unauthorized: authentication required': False,
        'net/http: TLS handshake timeout': False,
    }
    for err, want in cases.items():
        monkeypatch.setattr(imageref.subprocess, "run",
                            lambda *a, _e=err, **k: P(1, _e))
        assert imageref._skopeo_absent("x/y@" + DIG_A) is want, err
    monkeypatch.setattr(imageref.subprocess, "run", lambda *a, **k: P(0, ""))
    assert imageref._skopeo_absent("x/y@" + DIG_A) is False

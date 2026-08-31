#!/usr/bin/env python3
"""imageref — image-reference parsing and container-registry digest resolution.

Owns the "which env content is a box actually running?" question, as the single
source of truth for every module in tools/vast/:

  * ref parsing (`_split_image`) — host / path / tag off a docker image ref;
  * `IMAGE_DIGEST_ENV`, the box-env key a launch stamps the resolved digest into
    (what `ls` later compares against to raise STALE-IMAGE);
  * `our_registry_hosts` / `is_our_registry` — the ONE definition of "a host
    whose moving tags we can resolve", so no caller has to re-decide it (they
    used to, all keyed on GitLab, which is how staleness went inert for R2);
  * `is_retired_registry` — the counterpart: a host we deliberately no longer
    publish to or pull from, so launch paths can refuse it instead of renting
    a box that cannot pull;
  * tag resolution (`image_tag_digest`): creds-ful `skopeo inspect` on the R2
    Worker registry; per-process cached;
  * the general fail-CLOSED resolver `image_ref_digest` (by-digest
    self-certifying / our registries / anonymous `skopeo inspect` via
    `_skopeo_digest`), injected as both `digest_verifier=` and `image_resolver=`
    into workflowctl so plan and box-resolver agree byte-for-byte.

Both digest caches (`_digest_cache`, `_ref_digest_cache`) are per-process and
live here; `herdd.py` re-exports the SAME dict objects, so the suite's
`herdd._digest_cache.clear()` still empties the dict these functions read.

Leaf module: stdlib only, and it imports NOTHING from tools/vast — so it is
importable and testable without the 10k-line CLI, and there is no import cycle
with `herdd.py`. It is not I/O-free (the registry lookups ARE the job), but
every path degrades to None rather than raising, so callers own the
fail-closed / self-certifying policy.

Provenance: extracted from herdd.py 2026-07-30, increment I3 of
docs/plans/vast-tooling-refactor.md; behavior-preserving (the whole block moved
verbatim, banner comment and docstrings included). `herdd.py` re-imports every
name below into its own namespace, so `herdd.<name>` stays a valid reference —
and, for the names whose CALLERS also stay in herdd, a valid
`monkeypatch.setattr(herdd, ...)` target. Names called from INSIDE this module
(`_skopeo_digest`, `image_tag_digest`, `_split_image`)
resolve through THIS module's globals, so a test that must steer those must
patch `imageref.<name>` (plan §4 rule 1 corollary; test_lifecycle.py updated in
the same commit).

Future home of the queued tri-state `classify_image_staleness` + TTL'd resolver
(velvet plan P1) — see the refactor plan §8.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
import time


# --------------------------------------------------------------------------- #
# image identity — which env content is a box actually running?
#
# Our train image ships under a MOVING tag (train:t215-latest), so the image
# NAME on an instance can't tell you whether the box predates the last env
# push. Fix: at launch, resolve the tag's current content digest against the
# registry and stamp it into the box env (IMAGE_DIGEST_ENV). `ls` re-resolves
# the tag and flags boxes whose stamp no longer matches — those run the OLD
# env, and a park/resume will NOT refresh them (resume keeps the disk).
# --------------------------------------------------------------------------- #
IMAGE_DIGEST_ENV = "HERDD_IMAGE_DIGEST"
_digest_cache: dict = {}          # image ref -> digest|None (per-process)

# The R2 Worker registry (registry/R2_WORKER_REGISTRY_PLAN.md) is the LIVE
# image path since the 2026-08-12 cutover — it is what herdd.yaml's
# `default_image` names. It rejects every anonymous /v2/ request, so digest
# resolution against it must carry a minted pull token or it degrades to None,
# which would silently disable the launch stamp and STALE-IMAGE detection for
# the fleet default image.
R2_REGISTRY_HOST = "registry.example.com"
R2_SECRET_ENV = "REGISTRY_AUTH_SECRET"

# RETIRED registries — hosts we no longer publish to, pull from, or hold
# credentials for. GitLab was the pre-2026-08-12 lane; the owner cut it as a
# registry on 2026-08-22, the project and its PAT are being deleted, and every
# ref that named it is dead. Kept as a NAMED SET rather than dropped silently:
# a stale ref in a workflow file, an old spec.json, or somebody's shell history
# must produce a refusal that says where the image moved to, not a 20-minute
# `loading` hang on a rented box that will never resolve `denied: access
# forbidden`.
RETIRED_REGISTRY_HOSTS = ("registry.gitlab.com",)


def is_retired_registry(host):
    """True when `host` is a registry we deliberately cut. Launch paths refuse
    these fail-closed; there is no credential path left for them."""
    return bool(host) and host in RETIRED_REGISTRY_HOSTS


def our_registry_hosts():
    """Every registry host whose MOVING tags we can resolve a digest for, so a
    box launched from one can be proven fresh or stale.

    One entry since the GitLab cut (2026-08-22). It stays a TUPLE, and callers
    still ask this function rather than comparing against `R2_REGISTRY_HOST`
    themselves: three call sites keeping their own copy of "is this ours" is
    exactly what made staleness classification inert for `registry.example.com`
    between the R2 cutover and 2026-08-21.

    Anything NOT in the set (docker.io, vllm/…) has no drift signal we can
    read, and callers must treat it as `not_applicable` rather than refusing
    it — that is a different question from `is_retired_registry`, which is a
    refusal.
    """
    return (R2_REGISTRY_HOST,)


def is_our_registry(host):
    """True when `host` is one of ours (see `our_registry_hosts`)."""
    return bool(host) and host in our_registry_hosts()


def _split_image(image):
    """'registry.example.com/train:t215-latest' ->
    ('registry.example.com', 'train', 't215-latest'). Tag defaults to
    'latest'. Returns (None, None, None) for refs without a registry host."""
    if not image or "/" not in image:
        return None, None, None
    host, _, rest = image.partition("/")
    path, _, tag = rest.partition(":")
    if not path:
        return None, None, None
    return host, path, (tag or "latest")


def image_tag_digest(image):
    """Current content digest (sha256:...) of one of OUR registries' image
    tags, or None (not our registry / no creds / lookup failure — callers
    degrade to no staleness signal, never an error).

    One backend since the GitLab cut: creds-ful `skopeo inspect` against the R2
    Worker registry, which rejects every anonymous /v2/ request, so a missing
    $REGISTRY_AUTH_SECRET costs the whole signal (`_r2_skopeo_creds` says so
    once per process). Cached per-process per image ref so `ls` costs at most
    one resolve per unique image."""
    if image in _digest_cache:
        return _digest_cache[image]
    dg = None
    host, _path, _tag = _split_image(image)
    if host == R2_REGISTRY_HOST:
        dg = _skopeo_digest(image)
    _digest_cache[image] = dg
    return dg


_ref_digest_cache: dict = {}       # image ref -> digest|None (per-process)


# --- loud degradation when the R2 signing secret is absent ------------------ #
# The module-wide policy is "degrade to None, never raise". For the R2 registry
# that policy has a sharp edge: a None from a missing $REGISTRY_AUTH_SECRET is
# indistinguishable, at every call site, from "the tag has not moved". `ls`
# then prints no STALE-IMAGE banner and the operator reads silence as fresh.
# So the None stays (callers keep their fail-closed/advisory policy) and the
# CAUSE is announced once per process on stderr, plus exposed as a predicate
# any renderer can ask about.
_warned_missing_r2_secret = False


def r2_secret_missing(ref):
    """True when `ref` is on the R2 registry and $REGISTRY_AUTH_SECRET is not
    set — i.e. every digest answer for it will be None for a reason that has
    nothing to do with the image."""
    host, path, _tag = _split_image(str(ref or ""))
    return bool(host == R2_REGISTRY_HOST and path
                and not os.environ.get(R2_SECRET_ENV))


def reset_secret_warning():
    """Re-arm the once-per-process notice (tests, and any long-lived daemon
    that wants to re-announce after a config reload)."""
    global _warned_missing_r2_secret
    _warned_missing_r2_secret = False


def _warn_r2_secret(ref, why):
    """Say it ONCE per process, on stderr. Never raises."""
    global _warned_missing_r2_secret
    if _warned_missing_r2_secret:
        return
    _warned_missing_r2_secret = True
    try:
        print(f"!! image staleness is UNKNOWN, not fresh: {why} — cannot "
              f"resolve digests on {R2_REGISTRY_HOST} (e.g. {ref}). "
              f"Export {R2_SECRET_ENV} (it lives in .env, which "
              f"load_env does not read) to restore the check.", file=sys.stderr)
    except Exception:
        pass


def _r2_skopeo_creds(ref):
    """['--creds', 'vast:<token>'] for a registry.example.com ref, else [].
    Missing secret or mint module degrades to anonymous (the module-wide
    None-not-raise policy) — but LOUDLY, because the R2 registry answers an
    anonymous request with a refusal, so the degradation costs the whole
    staleness signal rather than just this call. The lazy import bends the leaf
    rule deliberately: registry.mint_token is itself stdlib-only and cycle-free,
    and inlining the HMAC construction here would fork the token format across
    two files."""
    host, path, _tag = _split_image(ref)
    if host != R2_REGISTRY_HOST or not path:
        return []
    secret = os.environ.get(R2_SECRET_ENV)
    if not secret:
        _warn_r2_secret(ref, f"{R2_SECRET_ENV} is not set")
        return []
    try:
        from registry.mint_token import mint
    except ImportError:
        _warn_r2_secret(ref, "registry.mint_token is not importable")
        return []
    repo = path.split("@", 1)[0]
    return ["--creds", "vast:" + mint(secret, repo, ttl_hours=1,
                                      instance="imageref")]


def _skopeo_digest(ref):
    """Content digest (sha256:...) of a docker image REF via `skopeo inspect`
    (anonymous, except minted creds for the R2 Worker registry), or None.
    Never raises — a missing skopeo, a network blip, a nonzero rc, or
    malformed output all degrade to None so the caller decides the
    fail-closed/self-certifying policy. skopeo's manifest-list digest format
    already matches our pinned image_digest."""
    if not shutil.which("skopeo"):
        return None
    try:
        p = subprocess.run(
            ["skopeo", "inspect", "--no-tags", "--format", "{{.Digest}}",
             *_r2_skopeo_creds(ref), "docker://" + ref],
            capture_output=True, text=True, timeout=60)
    except Exception:
        return None
    if p.returncode != 0:
        return None
    out = (p.stdout or "").strip()
    return out if out.startswith("sha256:") else None


# Registry error strings that mean "this manifest is NOT here", as opposed to
# "I could not ask". The distinction is the entire point: a transient network
# failure must stay advisory, but a definitive absence must be fatal, and
# collapsing the two is how a dangling ref passed for two days.
_ABSENT_MARKERS = (
    "manifest unknown", "name unknown", "not found", "404",
    "was deleted or has expired", "repository name not known",
    "unknown: manifest", "no such host",
)


def _skopeo_absent(ref):
    """True only when the registry DEFINITIVELY reports the ref missing.

    Everything else — skopeo not installed, a timeout, an auth failure, a 5xx —
    is False, because "I could not ask" is not "it is not there". Callers use
    this to fail closed on a real absence without letting a network blip take a
    launch path down with it.
    """
    if not shutil.which("skopeo"):
        return False
    try:
        p = subprocess.run(
            ["skopeo", "inspect", "--no-tags", "--format", "{{.Digest}}",
             *_r2_skopeo_creds(ref), "docker://" + ref],
            capture_output=True, text=True, timeout=60)
    except Exception:
        return False
    if p.returncode == 0:
        return False
    err = ((p.stderr or "") + (p.stdout or "")).lower()
    if "unauthorized" in err or "authentication required" in err:
        return False        # cannot see it != is not there
    return any(m in err for m in _ABSENT_MARKERS)


def image_ref_digest(image):
    """Resolve an image ref's content digest (sha256:...) for the workflow
    launch/plan gate, fail-CLOSED (None when unresolvable), cached per-process.

    Injected as BOTH `digest_verifier=` (build_box_resolver) and
    `image_resolver=` (plan_workflow online) so the pre-launch plan and the
    box resolver agree byte-for-byte. Three cases:
      1. BY-DIGEST ref ('...@sha256:...'): the digest is self-certifying for
         its SHAPE, but not for its EXISTENCE. See the note below.
      2. a tag on one of OUR registries (`our_registry_hosts`): try
         `image_tag_digest` first (creds-ful skopeo) and fall back to bare
         skopeo on None.
      3. any other host tag (docker.io, ...): skopeo tag inspect, else None.

    CASE 1 CHANGED 2026-08-04. It used to return the in-ref digest whenever the
    probe was anything other than a contradiction — INCLUDING when skopeo said
    the manifest does not exist. So `workflow plan --online` went GREEN on
    `sha256:193f9c25…`, a ref no box could pull, for two days. "Self-certifying"
    only ever meant "content-addressed refs cannot DRIFT"; it never meant they
    cannot be absent. Now a DEFINITIVE absence (`manifest unknown`, 404, …) is
    fatal, while "could not ask" — no skopeo, a timeout, an auth failure — stays
    advisory exactly as before, so a network blip still cannot desync
    _plan_online from box_resolver. Under the rolling ruling this path should be
    rare: by-digest refs are no longer written into anything."""
    if image in _ref_digest_cache:
        return _ref_digest_cache[image]
    dg = None
    if image and "@sha256:" in image:
        in_ref = "sha256:" + image.split("@sha256:", 1)[1]
        probed = _skopeo_digest(image)
        if probed is not None:
            dg = in_ref if probed == in_ref else None
        elif _skopeo_absent(image):
            dg = None       # the registry says it is NOT there — fail closed
        else:
            dg = in_ref     # could not ask; advisory, as before
    else:
        host, _path, _tag = _split_image(image)
        if is_our_registry(host):
            dg = image_tag_digest(image)
            if dg is None:
                dg = _skopeo_digest(image)
        elif host:
            dg = _skopeo_digest(image)
    _ref_digest_cache[image] = dg
    return dg


# --------------------------------------------------------------------------- #
# staleness classification (velvet plan P1) — is a box running the CURRENT env?
#
# The failure this exists for, 2026-07-30: three frontier-wave jobs died within
# seconds on box 46240842 because its baked eval-env predated
# tools/witness/plscore_frontier.py. The signal already existed
# (`_stale_image_ids`) but no scheduling path consulted it — `ls` printed a
# banner and that was all.
#
# WHY FOUR STATES, NOT A BOOLEAN. `image_tag_digest` returns ONE ambiguous None
# for three different situations (not our registry / no creds / API failure), so
# a two-valued gate would refuse every box on a foreign registry — the
# documented `vllm/vllm-openai:latest` serve lane resolves to None with no
# network call at
# all. The states separate "nothing to compare" from "could not compare", which
# is what lets a later enforcement phase (P3) proceed on the first and HOLD on
# the second.
# --------------------------------------------------------------------------- #
IMG_NOT_APPLICABLE = "not_applicable"   # nothing to compare — proceed silently
IMG_FRESH = "fresh"                     # definitive match
IMG_STALE = "stale"                     # definitive mismatch — the registry moved
IMG_UNRESOLVED = "unresolved"           # should have compared, could not


def classify_image_staleness(*, image, stamped_digest, current_digest):
    """PURE. `(state, reason)` — is this box's image the current one?

    `image`            the ref the box launched with.
    `stamped_digest`   IMAGE_DIGEST_ENV off the box env (None if unstamped).
    `current_digest`   what the tag resolves to NOW, or None if resolution
                       failed/was not attempted. The caller resolves; this
                       function decides, so it stays I/O-free and testable.

    Only refs on one of our registries (`our_registry_hosts`) can drift in a
    way we can detect.

    A `@sha256:`-pinned ref is `not_applicable` BY CONSTRUCTION, not by
    resolution: content-addressed refs cannot drift, so a pinned box is never
    stale no matter what the tag now points at.

    REGISTRY-AGNOSTIC SINCE 2026-08-21. This used to compare `host` against
    $GITLAB_REGISTRY alone, so every `registry.example.com` ref — the fleet
    default image since the R2 cutover — short-circuited to `not_applicable`
    and the whole classifier was inert for the images we actually run. The
    resolution side (`image_tag_digest`) had gained an R2 branch; this side had
    not, and nothing compared the two.
    """
    img = str(image or "")
    ours = our_registry_hosts()

    if "@sha256:" in img:
        return IMG_NOT_APPLICABLE, ("by-digest ref is content-addressed and "
                                    "cannot drift")
    host, _path, _tag = _split_image(img)
    if not is_our_registry(host):
        return IMG_NOT_APPLICABLE, (
            f"image is not on one of our registries ({', '.join(ours)}) — no "
            "drift signal exists for it")
    if not stamped_digest:
        # Pre-stamp boxes and any launch path that skipped the stamp. Silent on
        # purpose: alarming on every legacy box is how an alarm gets ignored.
        return IMG_NOT_APPLICABLE, (f"no {IMAGE_DIGEST_ENV} stamp on the box — "
                                    "launched before/outside the stamping path")
    if not current_digest:
        if r2_secret_missing(img):
            return IMG_UNRESOLVED, (
                f"tag digest did not resolve because {R2_SECRET_ENV} is not "
                f"set — {R2_REGISTRY_HOST} refuses anonymous reads. This is a "
                "MISSING CREDENTIAL, not evidence the box is current")
        return IMG_UNRESOLVED, ("tag digest did not resolve (no creds, API "
                                "failure, or tag gone) — cannot prove fresh OR "
                                "stale")
    if current_digest != stamped_digest:
        return IMG_STALE, (f"registry tag moved since launch: box has "
                           f"{str(stamped_digest)[:19]}…, tag is now "
                           f"{str(current_digest)[:19]}… — a park/resume will "
                           "NOT refresh it")
    return IMG_FRESH, "stamped digest matches the tag's current digest"


# --- TTL'd resolution ------------------------------------------------------- #
# MANDATORY for the daemon, not an optimization. `_digest_cache` above is a
# plain per-process dict with no expiry that caches None too, and in a
# days-long fleetd BOTH directions are disqualifying: a success is cached
# forever so a real image push is never noticed (precisely the footgun this
# feature exists to catch), and a failure is cached forever so a transient API
# blip pins the box's verdict for the daemon's life. `ls` escapes this only by
# being a short-lived process.
IMAGE_DIGEST_TTL_S = 900.0

_ttl_cache: dict = {}                   # image -> (expires_at, digest|None)
_ttl_lock = threading.Lock()


def resolve_tag_digest_ttl(image, *, ttl_s=None, now=None, resolver=None):
    """Tag digest with a TIME-BOUNDED cache. `(digest|None, cache_state)`.

    cache_state is `hit` / `miss` / `expired`, for the caller to journal — a
    verdict computed off a 14-minute-old digest should be legible as such.

    Deliberately a SEPARATE cache from `_digest_cache`: that one is keyed to a
    short-lived CLI process where "resolve once" is correct, and quietly adding
    expiry to it would change `ls`/launch behavior for no reason. Long-lived
    callers opt into this one.
    """
    ttl = IMAGE_DIGEST_TTL_S if ttl_s is None else float(ttl_s)
    t = time.time() if now is None else float(now)
    fn = resolver or image_tag_digest
    with _ttl_lock:
        ent = _ttl_cache.get(image)
        if ent and t < ent[0]:
            return ent[1], "hit"
        state = "expired" if ent else "miss"
    dg = fn(image)                       # OUTSIDE the lock: this does network I/O
    with _ttl_lock:
        _ttl_cache[image] = (t + ttl, dg)
    return dg, state


def clear_ttl_cache():
    """Drop every TTL entry (tests, and any explicit 'recheck now' path)."""
    with _ttl_lock:
        _ttl_cache.clear()


# --- relaunch image choice (was velvet P4a "faithful relaunch") -------------- #
# WHAT THIS USED TO DO, AND WHY IT NO LONGER DOES.
#
# P4a's premise was that a run evicted mid-flight wants the image it STARTED
# with, so when the tag had MOVED this function returned `repo@<recorded
# digest>` and the relaunch reproduced the run's original environment.
#
# Owner ruling 2026-08-04 retires that premise: "we should always be pinned to
# the latest image. no hash pins or anything. we operate with a rolling release
# cadence." A relaunch that resurrects an older digest is precisely the
# behaviour the ruling forbids, and it does it invisibly, on the eviction path,
# where nobody is watching. It also has a failure mode of its own that the
# rolling model removes: a recorded digest can be garbage-collected from a
# 10 GB-capped registry, and pinning to a GC'd digest turns a moved tag into an
# unpullable image on a box we are already paying for.
#
# So the DECISION inverts — a moved tag now rolls FORWARD — while the
# OBSERVATION is kept and made louder. That split is deliberate:
#
#   * PINNING (choosing an old digest to run)  -> removed. This is what the
#     ruling is about.
#   * RECORDING (knowing which image a run got) -> kept, everywhere. The
#     launch-time stamp (IMAGE_DIGEST_ENV), spec.json's `image_digest`, and
#     `classify_image_staleness` are all UNCHANGED. Under a rolling cadence
#     they get MORE useful, not less: they are now the only way to learn, after
#     the fact, that a run's two halves ran different bytes.
#
# The reason string on PIN_ROLLED is therefore not decoration — it is the sole
# surviving notice that a measurement just lost its comparability, and callers
# print it.

PIN_NONE = "none"        # nothing recorded / nothing to compare — replay the tag
PIN_MATCH = "match"      # tag still resolves to the launch digest — replay the tag
PIN_ROLLED = "rolled"    # tag MOVED — replay the TAG anyway, and say so loudly
PIN_UNVERIFIED = "unverified"   # could not resolve — replay the tag, say so


def pin_relaunch_image(*, image, spec_digest, current_digest):
    """PURE. `(image_ref, state, reason)` — which image ref should a relaunch use?

    Under the rolling ruling the answer is always THE TAG. `spec_digest` and
    `current_digest` no longer steer the choice; they decide what the caller is
    TOLD, which is the whole remaining value of having recorded them.

    `spec_digest`     the digest recorded in spec.json at the ORIGINAL launch.
    `current_digest`  what the tag resolves to now, or None if unresolvable.

    The name is now a misnomer and is kept only because three call sites in
    herdd.py destructure its 3-tuple; it pins nothing.
    """
    ref = str(image or "")
    if not ref or not spec_digest:
        return ref, PIN_NONE, "no launch digest recorded — replaying the tag"
    if "@sha256:" in ref:
        # A by-digest ref should no longer exist anywhere (see the de-pin in
        # shiplib.CONSUMERS), but if one is recorded in an old spec, replaying
        # it verbatim is still the faithful reproduction of THAT launch.
        return ref, PIN_NONE, (
            "spec recorded a by-digest ref — replaying it verbatim; note this "
            "predates the rolling ruling, nothing writes such refs now")
    if not current_digest:
        return ref, PIN_UNVERIFIED, (
            "tag digest did not resolve — replaying the tag, and unable to say "
            "whether the image moved since this run launched")
    if current_digest == spec_digest:
        return ref, PIN_MATCH, "tag still resolves to the launch digest"
    return ref, PIN_ROLLED, (
        f"IMAGE ROLLED FORWARD: the tag moved since launch (run started on "
        f"{str(spec_digest)[:19]}…, tag is now {str(current_digest)[:19]}…). "
        "Rolling release — the relaunch takes the NEW image, so this run's "
        "second half executes different bytes than its first. Recorded, not "
        "prevented: treat any paired/comparative number from this run as split "
        "across two envs.")

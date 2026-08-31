"""vastlib.core.labels — the instance-label GRAMMAR, and nothing that acts on it.

Why this module exists
----------------------
A vast instance label is the only durable, daemon-free record of what a box is
for. Three separate production bugs came from reading it with the wrong mental
model, and each one cost real money or a real credential:

  * `run:<RID>` read with a fixed-width slice returned `<RID>:keep` after fleetd
    stamped its park token — a run watch lost its own box, and
    `_destroy_and_revoke` minted `run-<RID>:keep`, so the ephemeral B2 key named
    `run-<RID>` was never revoked (2026-08-02).
  * `keep` matched as a SUBSTRING made `serve:keeper` un-reapable and
    `housekeeping:on` a permanent keep.
  * splitting on `:` ALONE missed the appended-GROUP form
    (`wave:rb3-wide-A keep:FLOOR-repair-pending`, live 2026-07-31), so the
    reaper would have destroyed a box someone deliberately paid to hold.

So the grammar is stated once, here, in one place that the readers and the one
writer share: **a label is whitespace-separated GROUPS of `:`-separated TOKENS.**
`keep` is a token match, never a substring and never a prefix slice. A keep
group whose reason carries `until-<YYYYMMDDTHHMMSSZ>` is self-expiring; an
unparseable deadline fails toward HOLDING the box, because destroying is the
irreversible direction.

Everything here is pure and stdlib-only: no HTTP, no clock except the `now`
parameter's default, no fleet action. That is what makes it testable without a
box and what keeps `core` at the bottom of the DAG.

What is deliberately NOT here
-----------------------------
* **Reap POLICY.** `REAP_IDLE_H_DEFAULT` (the 2h owner rule), the idle/zombie
  ledgers and `cmd_reap`'s sweep live in `boxes/reap.py`. They sit textually
  adjacent to this grammar in `herdd.py`, and a naive contiguous cut would
  take them: "how long is too long idle" is a decision, "does this label say
  keep" is a parse. Only the parse belongs at the bottom of the DAG.
* **The label WRITE seam.** `_put_label_soft` (the single PUT every automatic
  relabel goes through) is an API mutation and belongs in `boxes/lifecycle.py`.
  `retention_keep_label` composes the string and hands it back; it never sets it.
* **The prefix ACCESSORS** `_label_value` / `_instance_run_label` /
  `_instance_serve_label`. Plan §5 assigns those to `core/models.py`, and the
  rename table points `herdd._label_value` there. What DID land here, when
  `models.py` was ported (2026-08-16), is the string-level predicate underneath
  them — `label_value` at the bottom of this file. They are the same grammar
  one level up: the value truncates at the first `keep` TOKEN and stops at the
  first whitespace, i.e. exactly the two rules `_reap_kept` learned the hard
  way, and `retention_keep_label`'s docstring below cites that truncation as
  its reason for appending a GROUP. If those two implementations ever drift,
  the 2026-08-02 `run-<RID>:keep` bug comes back — so there is one copy, here,
  and `models._label_value` is a delegation.
* **The second composer, in the daemon.** `fleetd.Hooks.keep_label` hand-builds
  `label + ":keep"` / `"keep:fleetd-park"` inline (guarded by `_reap_kept`).
  That is the write half of this grammar re-implemented outside it; when
  `fleet/` is decomposed it should call a compose helper here.

Provenance: moved verbatim from `tools/vast/herdd.py` (plan §8 step 2, `core/`),
behavior-preserving — same tokens, same case-folding, same fail-toward-holding.
Names are unchanged from the originals so the §7.1 rename table is an identity
mapping for this module. Plan of record:
`docs/plans/vast-tooling-refactor-v2.md`.
"""

from __future__ import annotations

import datetime
import re
import time
from collections.abc import Iterable
from typing import TypedDict


class KeepRetention(TypedDict):
    """The read-only view `_keep_retention_info` returns (a plain dict at runtime).

    Typed here rather than left as a bare `dict[str, object]` so the render
    surfaces that do arithmetic on `left_s` and print `reason` get checked; the
    runtime object is exactly the dict literal `herdd` returned.
    """

    reason: str
    deadline_ts: float
    left_s: float


# The understudy (handoff twin) label suffix — a SECOND box for the same run,
# labelled run:<ID>:handoff (HANDOFF_DESIGN §2.1). Its run-label is the distinct
# id '<ID>:handoff', so the preflight dup guard treats it as its own run and
# never collides with the live primary run:<ID> it is migrating off.
# moved-from: herdd.HANDOFF_LABEL_SUFFIX
HANDOFF_LABEL_SUFFIX = ":handoff"


# A keep group may carry its own EXPIRY (owner directive 2026-08-05, the
# eviction-retention window): `keep:evicted-outbid-until-20260805T183000Z`.
# BASIC ISO-8601, no colons and no dashes in the timestamp — `:` is the label's
# own token separator, so an extended-format timestamp would shred the group
# into `keep`/`evicted-outbid-until-20260805T18`/`30`/`00Z` and the deadline
# would silently never parse.
# Case-insensitive: every consumer lowercases label tokens before matching (the
# `_reap_kept` grammar), so a case-sensitive pattern would never fire on a real
# label and every retention would read as an UNCONDITIONAL keep — a window that
# silently never expires, which is the exact failure this feature must not have.
# moved-from: herdd._KEEP_UNTIL_RE
_KEEP_UNTIL_RE = re.compile(r"(?:^|-)until-(\d{8}t\d{6}z)$", re.IGNORECASE)
# moved-from: herdd._KEEP_UNTIL_FMT
_KEEP_UNTIL_FMT = "%Y%m%dT%H%M%SZ"


# moved-from: herdd._keep_until_ts
def _keep_until_ts(group_tokens: Iterable[str]) -> float | None:
    """Epoch deadline carried by a keep GROUP's tokens, or None for an
    unconditional (never-expiring) keep. Unparseable timestamps read as None —
    a malformed deadline must fail toward HOLDING the box, never toward
    destroying it, because the label is the only durable record of why someone
    wanted it alive."""
    for t in group_tokens:
        m = _KEEP_UNTIL_RE.search(t)
        if not m:
            continue
        try:
            return datetime.datetime.strptime(
                m.group(1).upper(), _KEEP_UNTIL_FMT).replace(
                    tzinfo=datetime.timezone.utc).timestamp()
        except ValueError:
            return None
    return None


# moved-from: herdd._reap_kept
def _reap_kept(label: str | None, now: float | None = None) -> bool:
    """True when a box's label opts it out of the idle reaper: a `keep` token
    anywhere in the label — `keep`, `keep:<why>`, an existing label with `:keep`
    appended (`run:<ID>:keep`), or a `keep:<why>` group appended to an existing
    label (`serve:<ID> keep:<why>`).

    Labels are whitespace-separated GROUPS of `:`-separated tokens, and the
    natural way to keep a box that already has a label is to append a group —
    exactly what `wave:rb3-wide-A keep:FLOOR-repair-pending` (live 2026-07-31)
    does. Splitting on `:` alone yielded the token `"rb3-wide-A keep"`, which
    matches nothing, so the opt-out was silently dropped and the reaper would
    have destroyed a box someone deliberately paid to hold. Split on BOTH.

    A keep group whose reason carries `until-<YYYYMMDDTHHMMSSZ>` is a SELF-
    EXPIRING keep (2026-08-05): it holds the box until that instant and then
    stops holding it, so the reaper's next 15-minute pass reclaims the disk with
    no daemon involved. That is the whole expiry mechanism for the eviction
    retention window — it lives on the box, so it survives a fleetd restart, a
    watch that ended, and a workstation that was asleep. An unconditional keep
    is unchanged: it holds forever, exactly as before."""
    now = time.time() if now is None else now
    for group in (label or "").split():
        toks = [t.strip().lower() for t in group.split(":")]
        if "keep" not in toks:
            continue
        until = _keep_until_ts(toks)
        if until is None or until > now:
            return True
    return False


# moved-from: herdd._keep_retention_info
def _keep_retention_info(label: str | None, now: float | None = None) -> KeepRetention | None:
    """`{"reason", "deadline_ts", "left_s"}` for the FIRST self-expiring keep
    group on a label, else None. Read-only view for the surfaces that have to
    explain a box nobody expected to still be paying for (`herdd ls`,
    `fleet status`) — derived from the label alone, so it works for a retained
    box whose watch has long since ended."""
    now = time.time() if now is None else now
    for group in (label or "").split():
        toks = [t.strip().lower() for t in group.split(":")]
        if "keep" not in toks:
            continue
        until = _keep_until_ts(toks)
        if until is None:
            continue
        why = next((t for t in toks if t != "keep" and _KEEP_UNTIL_RE.search(t)),
                   "keep")
        return {"reason": why.rsplit("-until-", 1)[0] or "keep",
                "deadline_ts": until, "left_s": round(until - now, 1)}
    return None


# moved-from: herdd.retention_keep_label
def retention_keep_label(label: str | None, reason: object, deadline_ts: float) -> str:
    """Append a SELF-EXPIRING keep group to `label`, matching `_reap_kept`'s
    grammar exactly: `<existing> keep:<reason>-until-<YYYYMMDDTHHMMSSZ>`.

    Appending a GROUP (space-separated), never `:keep` onto the existing group,
    because `_label_value` truncates at the first `keep` token — stamping the
    suffix inline would rewrite `run:<RID>` to `run:<RID>:keep:...` and cost the
    box its B2 key revocation at destroy time (the 2026-08-02 `run-<RID>:keep`
    bug). Reason is slugged: any `:` or whitespace inside it would forge new
    tokens.

    `reason` is annotated `object` rather than `str` because the body has always
    coerced (`str(reason or "kept")`); narrowing it would be a semantic change
    at the call sites, which pass values read out of untyped state dicts."""
    slug = re.sub(r"[^A-Za-z0-9_.]+", "-", str(reason or "kept")).strip("-").lower()
    stamp = datetime.datetime.fromtimestamp(
        deadline_ts, datetime.timezone.utc).strftime(_KEEP_UNTIL_FMT)
    group = f"keep:{slug or 'kept'}-until-{stamp}"
    return f"{label} {group}".strip() if label else group


# moved-from: herdd._job_handoff_label
def _job_handoff_label(primary_iid: object) -> str:
    """The understudy's launch label: job:<primary_iid>:handoff. The :handoff
    suffix is the twin marker reconcile-on-restart scans for; keying it on the
    primary's iid (the jobs lane has no run_id) keeps one twin per migrated box.

    `primary_iid` is `object` for the same reason `retention_keep_label`'s
    `reason` is: every caller passes `jctx.get("iid")`, which is a str in
    practice but is neither typed nor non-None-guaranteed at the seam, and
    f-string formatting is total."""
    return f"job:{primary_iid}{HANDOFF_LABEL_SUFFIX}"


# NEW CODE, deliberately marker-less (README §2 rule 7): this is the string
# half of `herdd._label_value`, extracted so `core/models.py` can own the
# ACCESSOR — `_label_value`, `_instance_run_label`, `_instance_serve_label`,
# which is where plan §5 puts them and where the rename table points — WITHOUT
# owning a second copy of the token rules. The `# moved-from:` marker for the
# original therefore sits on `models._label_value` (one marker per original),
# and this function is the shared implementation it calls.
def label_value(label: str | None, prefix: str) -> str | None:
    """PURE. Value of a `<prefix>:<value>` label group, or None.

    LABELS ARE APPENDABLE, so a fixed-width slice is wrong. `fleetd`'s
    `Hooks.keep_label` stamps `":keep"` onto the label of EVERY box it parks
    (every fleetd park is a resumability promise, so it opts the box out of the
    idle reaper). A box labelled `run:<RID>` therefore becomes `run:<RID>:keep`,
    and `lbl[4:]` then yields `<RID>:keep` — silently, with two consequences:

      * `fleetd._resolve_iid` matches `_instance_run_label(inst)` against the
        bare RUN_ID, so a run watch LOSES ITS OWN BOX right after fleetd parks
        it, and falls back to a possibly stale cached iid.
      * `_destroy_and_revoke` built `f"run-{lab[4:]}"`, i.e. `run-<RID>:keep`,
        so the ephemeral B2 key actually named `run-<RID>` WAS NEVER REVOKED at
        destroy — a credential outliving the box it was minted for.

    FLEETD_DESIGN's B1c note already stated the rule this violated: a
    `run:<id>` label "is parsed elsewhere and must stay exact".

    BUT THE SUFFIX IS NOT ALL NOISE, so this cannot just take the first token.
    The handoff understudy is deliberately labelled `run:<ID>:handoff` and the
    dup guard depends on that reading as the DISTINCT run id `<ID>:handoff` —
    see `_launch_preflight`'s docstring, which spells out that the exact-match
    guard "compare[s] the whole suffix" so a second understudy is refused while
    the live primary `run:<ID>` is allowed to outlive the cutover
    (HANDOFF_DESIGN §2.1). Collapsing to the first token makes a twin
    indistinguishable from its primary and silently breaks both guards.

    So: keep the semantic suffix, drop the reap opt-out. The value is everything
    after the prefix, truncated at the first `keep` TOKEN — matching
    `_reap_kept`'s model exactly (a `:`-separated token equal to "keep"), which
    is the only thing anything appends. `run:r1:handoff:keep` -> `r1:handoff`;
    `run:r1:keep:fleetd-park:u20260802` -> `r1`. An empty value reads as None.

    2026-08-05: the value is also GROUP-bounded. `_reap_kept` learned in July
    that labels are whitespace-separated groups of `:`-separated tokens, and the
    natural way to keep an already-labelled box is to append a whole group
    (`wave:rb3-wide-A keep:FLOOR-repair-pending`, live 2026-07-31). Splitting on
    `:` alone read `run:r1 keep:why` as the run id `"r1 keep:why"` — the SAME
    class of bug this function was written to fix, one grammar level up, and it
    would have cost the `run-<RID>` B2 key its revocation on every box the
    eviction-retention window labels. The value stops at the first whitespace."""
    pre = prefix + ":"
    if not label or not label.startswith(pre):
        return None
    rest = label[len(pre):].split()
    if not rest:
        return None
    kept = []
    for tok in rest[0].split(":"):
        if tok.strip().lower() == "keep":
            break
        kept.append(tok)
    return ":".join(kept) or None

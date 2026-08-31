"""`herdd fleet watch` — register/upsert a watch, and print the cap that LANDED.

Why this module exists
----------------------
Two incidents shaped everything printed here:

* **The ceiling is durable and CUMULATIVE.** Box 46916278 was armed at $10 six
  times through a preempt loop — $60 of real ceiling while every box looked
  compliant — because the confirmation echoed the figure that was TYPED. It now
  prints the cap that landed, the headroom left under it, and where the ceiling
  came from.
* **A jobs watch against an already-terminal queue.** Box 46648873 (2026-08-03)
  was armed on a stale all-terminal queue and parked 4 s later.
  `_fleet_watch_jobs_order_warning` is the pre-flight for that, and it is
  best-effort by construction: one B2 listing plus a fold per ticket, and it
  says nothing rather than blocking the watch when B2 is unreachable.

`--standing`, `--reset-spend` and the replacement/salvage knobs are stated by
every registration, never inherited: re-running `fleet watch` without a flag is
how you turn that flag OFF, so a watch's policy is always the last thing typed.

What is deliberately NOT here
-----------------------------
* The policy defaults. `MAX_REPLACEMENTS`, `REPLACE_CEILING_MULT` and
  `REPLACEMENT_RETENTION_H` are read from `bidpolicy` — the Zone S leaf the
  daemon reads too — so the help text cannot drift from the enforced value.
  The salvage knobs come from `cli._args._add_salvage_args` for the same
  reason `job supervise` takes them: one flag block, two callers.
* Any decision. Every flag is serialized into the `policy` dict and sent; what
  a replacement costs and when a rescue fires is the daemon's.

Provenance: moved from `tools/vast/herdd.py::cmd_fleet_watch` and
`::_fleet_watch_jobs_order_warning`, plan §8 step 6.
"""

from __future__ import annotations

import argparse
import sys

import serve_artifact

from vastlib.cli import _args
from vastlib.core import fmt
from vastlib.fleet import client

import bidpolicy
import jobmeta

#: `ident=` in the READY marker is the grade-A fingerprint truncated here.
#: One constant, because `serve_identity_gate.py` truncates the box's side to
#: the same width and a pin composed at a different one can never match.
IDENT_SHA_LEN = 12


def _registry_ident(slug: str) -> tuple[str, str]:
    """`(expected-sha12, artifact-id)` recomposed from the COMMITTED registry.

    Same composer `launch_serve.sh` runs to freeze `identity_expect.json`, so
    the watch pin and the box's expectation cannot come from two readings of
    the registry that disagree. Exits 2 on an unknown slug or an artifact with
    no identity pin: an unsatisfiable pin registered at $0 becomes a permanent
    alarm on a box that is doing nothing wrong.
    """
    try:
        entry = serve_artifact.registry.get(slug, None)
        doc = serve_artifact.compose_expectation(entry)
    except serve_artifact.registry.RegistryError as e:
        raise SystemExit(f"!! fleet watch --artifact {slug}: {e}") from e
    except serve_artifact.Refusal as e:
        raise SystemExit(f"!! fleet watch --artifact {slug}: {e}") from e
    return str(doc["fingerprint_sha256"])[:IDENT_SHA_LEN], str(doc["artifact"])


def _resolve_identity_pin(a: argparse.Namespace) -> tuple[str | None, str | None]:
    """`(artifact, expect_ident)` for the policy, verified before the daemon
    ever carries it. Every refusal here is free — nothing has been rented, and
    nothing about the running fleet changes.

    Three refusals, and each one is a way the pin could silently mean nothing:

    * an identity pin on a NON-SERVE profile. There is no SERVE_STATUS marker
      to check it against, so the flag would be inert — the same reason
      `serve_ready.sh` refuses `--expect-ident` in `--base-url` mode rather
      than ignoring it.
    * `--expect-ident` with no `--artifact`. Nothing to verify it against, and
      a typo'd sha12 is indistinguishable from a genuinely wrong box.
    * `--expect-ident` that DISAGREES with the registry. The registry is the
      claim somebody signed; a hand-typed sha that differs from it is either
      stale or a slip, and either way the daemon would alarm forever against a
      pin no correct box could satisfy.

    `--artifact` alone is the encouraged shape: the sha is derived, so there is
    no third place for it to be wrong.
    """
    artifact = getattr(a, "artifact", None)
    typed = (getattr(a, "expect_ident", None) or "").strip().lower() or None
    if not artifact and not typed:
        return None, None
    if a.profile != "serve":
        raise SystemExit(
            f"!! --artifact/--expect-ident need `--profile serve`, not "
            f"{a.profile!r}: the check reads the box's SERVE_STATUS marker, "
            f"and no other profile writes one. A pin that cannot be checked is "
            f"worse than no pin — the caller believes it ran.")
    if not artifact:
        raise SystemExit(
            "!! --expect-ident needs --artifact: the sha12 is verified against "
            "the committed registry before the daemon carries it, and with no "
            "slug there is nothing to verify it against. A typo'd pin and a "
            "genuinely wrong box look identical to the daemon.")
    want, art_id = _registry_ident(str(artifact))
    if typed and typed != want:
        raise SystemExit(
            f"!! --expect-ident {typed} does NOT match artifact {art_id!r}: the "
            f"committed registry composes {want}.\n"
            f"!!   Nothing has been registered. The registry is the signed "
            f"claim; a pin that disagrees with it is stale or mistyped, and "
            f"either way no correct box could ever satisfy it — the daemon "
            f"would alarm forever on a box doing nothing wrong.\n"
            f"!!   Recompose it: python3 tools/vast/serve_artifact.py expect "
            f"{art_id}\n"
            f"!!   Or drop --expect-ident and let --artifact derive it.")
    return art_id, want


# moved-from: herdd._fleet_watch_jobs_order_warning
def _fleet_watch_jobs_order_warning(a: argparse.Namespace) -> None:
    """Print the ordering warning for `fleet watch <IID> --profile jobs`.

    Best-effort and never fatal: one B2 listing plus a fold per ticket. If B2 is
    unreachable we say nothing rather than block the watch. See
    jobmeta.jobs_watch_advice for the incident (box 46648873, 2026-08-03: a jobs
    watch armed against a stale all-terminal queue parked the box 4s later)."""
    if a.profile != "jobs" or a.keep or str(a.target).startswith("run:"):
        return
    try:
        jids = jobmeta.list_queue(str(a.target))
        views = []
        for j in jids:
            try:
                views.append(jobmeta.read_job(j))
            except Exception:
                return                      # partial view -> no confident advice
        advice = jobmeta.jobs_watch_advice(jids, views)
    except Exception:
        return
    if advice:
        print(f"!! {advice}", file=sys.stderr)


# moved-from: herdd.cmd_fleet_watch -> run
def run(a: argparse.Namespace) -> None:
    _fleet_watch_jobs_order_warning(a)
    artifact, expect_ident = _resolve_identity_pin(a)
    policy = {"budget": a.budget, "max_bid": a.max_bid, "keep": a.keep,
              "dry_run": False, "handoff": not a.no_handoff,
              "strict_ceiling": a.strict_ceiling, "rescue_wait": a.rescue_wait,
              "interval": a.interval, "wall_budget": a.wall_budget,
              "max_relaunch": a.max_relaunch,
              "max_replacements": getattr(a, "max_replacements", None),
              "replace_ceiling_mult": getattr(a, "replace_ceiling_mult", None),
              "replacement_verified": getattr(a, "replacement_verified", None),
              "replacement_retention_hours": getattr(
                  a, "replacement_retention_hours", None),
              "salvage": getattr(a, "salvage", None),
              "salvage_keep_n": getattr(a, "salvage_keep_n", None),
              "salvage_max_gb": getattr(a, "salvage_max_gb", None),
              # Stated by every registration like the rest of this dict: on the
              # own-key path a re-`watch` without --artifact is how you turn
              # the identity check OFF, and the daemon pops the stale verdict
              # and the condemn latch with it. (Addressing a REPLACED box's id
              # merges instead, per `_redirect_policy` — clearing a pin needs
              # the owning key, like every other flag.)
              "artifact": artifact, "expect_ident": expect_ident}
    data = client._fleet_call_or_die("watch", target=a.target, profile=a.profile,
                                     budget_usd=a.budget, policy=policy,
                                     reset_spend=bool(getattr(a, "reset_spend", False)),
                                     # ALWAYS explicit from the CLI (never omitted):
                                     # a registration states the whole watch, so
                                     # dropping --standing is how you turn it off.
                                     standing=bool(getattr(a, "standing", False)),
                                     requester=client._fleet_requester())
    # Print the cap that LANDED and the headroom under it, not the figure that
    # was typed. A ceiling is durable and cumulative: re-arming at $5 on a box
    # that already spent $2 leaves $3, and printing "budget=$5.00" there is how
    # a preempt-loop spent N x cap while every box looked compliant
    # (box 46916278: $10 armed six times, $60 of real ceiling).
    cap, left = data.get("budget_usd"), data.get("remaining_usd")
    src = data.get("ceiling_source")
    print(f"watching {data.get('target')} (profile={data.get('profile')}, "
          f"budget={fmt.dollars(cap) if cap is not None else 'none'}"
          f"{f', remaining {fmt.dollars(left)}' if left is not None else ''}"
          f"{f', ceiling {src}' if src else ''})")
    if src == "inherited" and a.budget is not None and cap != a.budget:
        print(f"   note: this box carries a DURABLE ceiling of {fmt.dollars(cap)} "
              f"(id {data.get('ceiling_id')}); the {fmt.dollars(a.budget)} you asked "
              f"for was not applied because this call did not arm a new cap")
    if left is not None and cap and left < cap:
        print(f"   note: {fmt.dollars(round(cap - left, 4))} has already been spent "
              f"against this ceiling by earlier watch epochs / successor boxes — "
              f"what is enforced is the {fmt.dollars(left)} remaining. Raise it by "
              f"naming a bigger --budget; `--reset-spend` starts the ceiling over")
    if expect_ident:
        print(f"   IDENTITY: pinned to artifact {artifact} (ident "
              f"{expect_ident}) — fleetd compares this against what the BOX "
              f"verified on its own weights every tick. A mismatch parks the "
              f"box and withdraws it from the ladder; it never destroys it. "
              f"Re-run without --artifact to stop checking")
    if data.get("standing"):
        print(f"   STANDING: this watch survives a queue drain — the box still "
              f"parks{' (suppressed by --keep)' if a.keep else ''}, the ladder "
              f"and the cap stay armed, and the next `job submit` to it is "
              f"supervised with no re-arm. The cap is CUMULATIVE across cycles: "
              f"a drain does not hand back a fresh budget")
    if data.get("redirected_from"):
        # The id the operator typed is the box; the id printed above is the
        # watch it belongs to. Say so, or the confirmation looks like it capped
        # something else.
        print(f"   note: {data['redirected_from']} is that watch's CURRENT box "
              f"(its original was replaced/handed off) — applied to watch "
              f"{data.get('target')}, keeping its accrued spend "
              f"({fmt.dollars(data.get('spend_usd') or 0)}) and merging its policy "
              f"flags (address {data.get('target')} directly to replace them)")


def add_parser(fsub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = fsub.add_parser("watch", help="register/upsert a watch (IID or run:RUN_ID)")
    p.add_argument("target")
    p.add_argument("--profile", default="bare",
                   choices=["run", "jobs", "serve", "bare"])
    p.add_argument("--budget", type=float, default=None, metavar="USD",
                   help="hard spend cap; breach = PARK + alarm (never destroy). "
                        "CUMULATIVE over the box's durable ceiling and its "
                        "successors: re-arming the same figure after $2 has been "
                        "spent leaves $3 of headroom, it does not grant a fresh "
                        "cap. Name a bigger number to raise it")
    p.add_argument("--reset-spend", dest="reset_spend", action="store_true",
                   help="start this box's durable ceiling over at $0 spent — a "
                        "genuinely new campaign on the same box. Loud and "
                        "journaled; without it a re-arm carries spend-to-date")
    p.add_argument("--max-bid", dest="max_bid", type=float, default=None)
    p.add_argument("--keep", action="store_true",
                   help="jobs profile: do not park when the queue drains")
    p.add_argument("--standing", action="store_true",
                   help="jobs profile: keep this watch ARMED across a queue "
                        "drain instead of ending it. The box still parks "
                        "(--keep is what suppresses that); the watch goes "
                        "dormant-but-armed and the next `job submit` to this "
                        "box is supervised with no re-arm, no bare adoption "
                        "and no LAPSED alarm. The budget cap is NOT reset by a "
                        "drain — one cumulative ceiling spans every cycle. "
                        "Stated by each registration: re-running `fleet watch` "
                        "without --standing turns it off")
    p.add_argument("--artifact", default=None, metavar="SLUG",
                   help="serve profile: the modelkit registry artifact this "
                        "box is supposed to be serving. fleetd compares it "
                        "against the identity the BOX verified on its own "
                        "weights (the READY marker's ident= field) every tick "
                        "and alarms when they disagree — a mismatch parks the "
                        "box and withdraws it from the rescue/relaunch ladder "
                        "(never destroys it). The expected sha12 is derived "
                        "from the committed registry, so this flag alone is "
                        "the safest form. launch_serve.sh passes it for you")
    p.add_argument("--expect-ident", dest="expect_ident", default=None,
                   metavar="SHA12",
                   help="serve profile: state the grade-A sha12 explicitly "
                        "instead of deriving it. Needs --artifact, and is "
                        "VERIFIED against the registry before the daemon "
                        "carries it — a pin that disagrees is refused at $0, "
                        "because no correct box could ever satisfy it")
    p.add_argument("--no-handoff", dest="no_handoff", action="store_true")
    p.add_argument("--strict-ceiling", dest="strict_ceiling", action="store_true")
    p.add_argument("--rescue-wait", dest="rescue_wait", type=int, default=None)
    p.add_argument("--max-replacements", dest="max_replacements", type=int,
                   default=None, metavar="N",
                   help="jobs profile: cap on AUTOMATIC replacement rentals "
                        "after an eviction (default "
                        f"{bidpolicy.MAX_REPLACEMENTS}). 0 disables auto-replacement and "
                        "restores the pre-2026-08-05 hand-rescue behavior")
    p.add_argument("--replace-ceiling-mult", dest="replace_ceiling_mult",
                   type=float, default=None, metavar="X",
                   help="jobs profile: a replacement may cost at most this x "
                        f"the ORIGINAL launch price (default {bidpolicy.REPLACE_CEILING_MULT:g})")
    p.add_argument("--replacement-verified", dest="replacement_verified",
                   choices=("0", "1"), default=None,
                   help="jobs profile: restrict AUTOMATIC replacement rentals "
                        "to vast-verified hosts (default 1). 0 widens the "
                        "candidate class to unverified hosts — for a market "
                        "where the verified book is empty; also settable as "
                        "JOB_REPLACEMENT_VERIFIED in the env or herdd.yaml")
    p.add_argument("--replacement-retention-hours",
                   dest="replacement_retention_hours", type=float,
                   default=None, metavar="H",
                   help="jobs profile: hold the EVICTED box this long after "
                        "replacing it, so state that never reached B2 can be "
                        f"salvaged (default {bidpolicy.REPLACEMENT_RETENTION_H:g}h, "
                        "~$0.27-$0.58 of allocated-disk storage). 0 destroys it "
                        "immediately")
    _args._add_salvage_args(p)
    p.add_argument("--interval", type=int, default=45)
    p.add_argument("--wall-budget", dest="wall_budget", type=float, default=48 * 3600)
    p.add_argument("--max-relaunch", dest="max_relaunch", type=int, default=3)
    p.set_defaults(fleetfunc=run)

"""vastlib.jobs.submit — the pre-spend gate wall in front of `job submit`.

Why this module exists
----------------------
`cmd_job_submit` is 200 lines of which about 130 are refusals. Five preflight
gates run before a single byte reaches B2, and the ORDER is load-bearing:
the free, pure, network-less ones (EVAL_ENV_VER pin, B2 write-scope) run first,
the B2-reading one (asset staleness) later, and the local bundle is only staged
after all of them have passed. Each gate exists because a real run was lost
without it — a wave graded by an unnamed eval env, two fully-trained arms marked
`failed` because `run.sh` wrote `checkpoints/` with the read-only remote, a box
that ran a locally-edited-but-never-re-staged `train.sh`. Refusing costs $0;
discovering the same fact mid-`rclone` on a rented machine does not.

The four contracts a port must not soften
-----------------------------------------
* **Every `sys.exit` string is an assertion.** Seven of them here, matched by
  test substrings ("refusing to submit — ...", "mutually exclusive"). Plan §7.4
  forbids expectation changes, so they are byte-identical including the
  em-dashes and the implicit multi-line concatenation.
* **`--env` values are never echoed.** `_apply_env_overrides` returns KEYS ONLY
  and the submit prints only the key list, because an `--env` value can be a
  credential. Nothing added here may widen that — no `repr` of the raw config,
  no debug print of the folded env.
* **`getattr(a, X, False)` is deliberate, not sloppiness.** `test_job_submit_
  preflight.py` builds `argparse.Namespace` stubs that omit flags entirely;
  tightening the arg object into a required dataclass field breaks 25 call
  sites. The argparse block itself is frozen by §4's CLI-surface diff test and
  moves to `cli/job/submit.py` at step 6.
* **The return value is asymmetric.** `cmd_job_submit` returns the `job_id` on
  the success path and `None` on `--dry-run` (a bare `return`). Both are
  preserved; a caller that treats the return as a job id must not be handed a
  dry-run arg.

Two API-touch rules, easy to lose
---------------------------------
* A plain job's submit stays **API-free**. The one `_get_instance_soft` read is
  conditional on `needs.venv == "eval"`, a non-`--local` submit, and a digit box
  id — that conditional is the rule, not an optimization.
* `_submit_disk_advisory` **never raises and never refuses**. Its whole body is
  inside a blanket `try/except` that degrades to a stderr NOTE, and
  `HERDD_DISK_ADVISORY=0|no|off` silences it. An estimate is not worth losing
  a job over.

What is deliberately NOT here
-----------------------------
* **`jobmatrix.submit_experiment`.** It is a SECOND copy of this pipeline (same
  vram gate, `write_bundle`, sha, `os.replace`, `BUNDLE_WARN_BYTES`) with its
  own cwd-relative staging dir and `MatrixError` instead of `sys.exit`. It is
  absorbed into `vastlib.workflows` (plan §3). The divergence is recorded, not
  unified — the same rule the supervise lanes follow.
* **`rehearse.sh`'s parallel gates.** The $0 CPU lane reimplements the
  validation and B2 write-scope seams in shell. Not a call site, but the place
  to check if any gate's semantics ever move.
* **The staging path policy.** `out/jobs/_bundles` is resolved from
  `_repo_root()` below, never from the cwd — see the constant's comment.

Provenance: moved verbatim from `tools/vast/herdd.py` (plan §8 step 5,
2026-08-16), rev ea8360dc. Behavior-preserving: bodies and comments copied,
annotations added, cross-module calls rewritten to module-attribute form
(`health._get_instance_soft`, `b2._ensure_b2_remote`, `models._disk_gb`, ...)
so the patch idiom survives, plus `_REPO_ROOT` in place of the inline `dirname`
chain, which moved two directories deeper with the file.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from typing import Any, Mapping, MutableMapping

import disksize
import joblocal

from vastlib.boxes import health, lifecycle
from vastlib.core import models
from vastlib.fleet import client
from vastlib.storage import b2

import jobmeta
import runmeta

__all__ = [
    "_apply_artifact_env",
    "_apply_env_overrides",
    "_print_submit_supervision",
    "_repo_root",
    "_submit_disk_advisory",
    "cmd_job_submit",
]

# The repo root — `tools/vast/herdd.py` spelled this as three `dirname`s above
# its own `__file__`, correct only because it sat at `tools/vast/`. This file is
# two directories deeper, so the same answer needs five. The failure is SILENT
# and expensive: `cmd_job_submit` joins `out/jobs/_bundles` onto it and then
# `os.replace()`s a real bundle into the result, so a wrong count stages job
# bundles into a tree nobody cleans up while every gate above still passes.
# Pinned by `test_vastlib_jobs_submit.py::test_repo_root_matches_the_herdd_
# computation`.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))


# `_repo_root` stays a FUNCTION, not just the constant above: it has three
# `monkeypatch.setattr` sites steering `cmd_job_submit`'s staging directory and
# `jobmeta.asset_preflight`'s repo base, and a module attribute is what a patch
# can replace. Plan §5 files it under `core.config` eventually; `core.config`
# exposes no repo-root helper today (checked), so re-homing it is a step-6/7
# move, not a second copy invented here.
# moved-from: herdd._repo_root
def _repo_root() -> str:
    return _REPO_ROOT


# moved-from: herdd._apply_env_overrides
def _apply_env_overrides(raw: MutableMapping[str, Any],
                         pairs: list[str] | None) -> list[str]:
    """Fold `--env K=V` (repeatable) onto a job-config's `env:` mapping, in the
    same pre-validation slot as `--name`/`--timeout`, so the override rides the
    normal `validate_job_config` -> ticket path. WIRE FORMAT UNCHANGED: the
    ticket still carries one flat `config.env` map and jobd.py's `prepare` still
    exports exactly that — a box cannot tell an overridden value from a
    hand-edited one, and existing tickets are untouched.

    The bundle sha is content-addressed over the FOLDER, so an override does not
    invalidate dedupe: two jobs differing only in env reuse one bundle object.
    Returns the list of overridden keys (for a value-free audit line — env can
    carry a token, so submit never echoes values).

    Keys are restricted to shell identifiers: jobd writes `export K=<quoted>`
    into `.job.env`, so a key with a space/dash would be a parse error ON THE
    BOX rather than a failure here."""
    if not pairs:
        return []
    env = dict(raw.get("env") or {})
    keys = []
    for kv in pairs:
        k, sep, v = str(kv).partition("=")
        if not sep:
            sys.exit(f"error: --env expects K=V (got {kv!r})")
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", k):
            sys.exit(f"error: --env key must be a shell identifier (got {k!r})")
        env[k] = v
        keys.append(k)
    raw["env"] = env
    return keys


def _apply_artifact_env(raw: MutableMapping[str, Any],
                        pairs: list[str] | None) -> list[str]:
    """Fold `--artifact PREFIX=<slug>` onto the job-config `env:` map from the
    COMMITTED modelkit registry — the composition half of `${VAR}` asset
    prefixes (tools/vast/ASSET_PARAMETERIZATION.md).

    One slug becomes `<PREFIX>_B2` (the payload prefix) plus the identity and
    serve facts beside it, so a submit never carries a hand-typed B2 path and
    the registry stays the single source for artifact identity. Runs BEFORE
    `_apply_env_overrides`, which makes a raw `--env <PREFIX>_B2=...` the
    documented escape hatch — it wins, and it bypasses the registry.

    Refuses an unknown slug (the registry is the trust anchor; a typo must not
    degrade to "no export" and then to a confusing unresolved-variable error)
    and refuses two `--artifact` flags claiming the same prefix. Returns the
    KEYS it set, never values — same audit rule as `_apply_env_overrides`."""
    if not pairs:
        return []
    # Local import: `modelkit` is a tools/vast package and a plain submit that
    # never names an artifact must not pay for loading the registry.
    from modelkit import registry as _registry

    env = dict(raw.get("env") or {})
    keys: list[str] = []
    seen: dict[str, str] = {}
    for kv in pairs:
        prefix, sep, slug = str(kv).partition("=")
        if not sep or not slug.strip():
            sys.exit(f"error: --artifact expects PREFIX=<registry slug> (got {kv!r})")
        prefix, slug = prefix.strip(), slug.strip()
        if prefix in seen and seen[prefix] != slug:
            sys.exit(f"error: --artifact {prefix} named twice ({seen[prefix]!r} "
                     f"and {slug!r}) — one prefix is one artifact")
        seen[prefix] = slug
        try:
            exports = _registry.env_exports(_registry.get(slug), prefix)
        except _registry.RegistryError as e:
            sys.exit(f"error: --artifact {kv}: {e}")
        env.update(exports)
        keys.extend(exports)
    raw["env"] = env
    return keys


# moved-from: herdd._submit_disk_advisory
def _submit_disk_advisory(cfg: Mapping[str, Any], asset_bytes: Mapping[str, Any],
                          bundle_info: Mapping[str, Any] | None,
                          box: str) -> None:
    """ADVISORY (velvet plan P2): what disk does this job actually need, and
    does the target box have it? Prints; refuses nothing.

    Two readings, and the SECOND is the one that pays for itself: with an
    explicit `--box`, compare the estimate against that box's real allocation,
    so an undersized box fails HERE, for $0, instead of mid-`rclone` on a rented
    machine as a generic `asset_stage_failed:<name>` with no mention of disk.

    Never blocks a submit and never raises — an estimate is not worth losing a
    job over. `HERDD_DISK_ADVISORY=0` silences it.
    """
    if os.environ.get("HERDD_DISK_ADVISORY", "1").strip().lower() \
            in ("0", "no", "off"):
        return
    try:
        gb, bd = disksize.estimate_disk_gb(  # type: ignore[no-untyped-call]
            cfg, asset_bytes, bundle_bytes=(bundle_info or {}).get("zst_size"))
        # "FLOOR", not "recommended": every unresolved term in the breakdown
        # can only push the real need up, and an undeclared needs.scratch_gb
        # makes the number blind to everything the entrypoint writes.
        print(f">> disk estimate: {gb:g}G FLOOR (a lower bound, not a target)",
              file=sys.stderr)
        for ln in disksize.format_breakdown(bd):  # type: ignore[no-untyped-call]
            print(ln, file=sys.stderr)
        inst = None
        if box and str(box).isdigit():
            try:
                inst = health._get_instance_soft(int(box))
            except Exception:
                inst = None
        if not inst:
            return
        alloc, used = models._disk_gb(inst)
        sev, msg = disksize.oversize_finding(  # type: ignore[no-untyped-call]
            declared_gb=alloc, recommended_gb=gb,
            storage_day_usd=models._storage_day(inst), breakdown=bd)
        if sev == "undersized":
            print(f"!! box {box}: {msg}", file=sys.stderr)
            if used:
                print(f"   (it already holds {used:g}G of its {alloc:g}G)",
                      file=sys.stderr)
        elif sev == "oversized":
            print(f"~~ box {box}: {msg}", file=sys.stderr)

        # velvet P4d: if the job declared its scratch RECONSTRUCTIBLE, say how
        # much of it this specific box could hold in RAM instead of on the disk
        # it bills for. Facts come from jobd's boot `scratch_probe` and nowhere
        # else — no probe means no placement, because an unverified assumption
        # about a box's filesystems must never shrink its allocation.
        if (cfg.get("needs") or {}).get("scratch_volatile"):
            facts = disksize.scratch_facts_from_probe(  # type: ignore[no-untyped-call]
                health._scratch_probe_soft(box))
            ram, on_disk, why = disksize.plan_scratch_placement(  # type: ignore[no-untyped-call]
                scratch_gb=(cfg["needs"] or {}).get("scratch_gb"),
                volatile=True, **facts)
            print(f"   scratch: {why}", file=sys.stderr)
            if ram > 0:
                print(f"   -> {ram:g}G of scratch can live on /dev/shm; only "
                      f"{on_disk:g}G of it needs disk", file=sys.stderr)
    except Exception as e:
        print(f"note: disk advisory skipped ({type(e).__name__}: {e})",
              file=sys.stderr)


# moved-from: herdd.cmd_job_submit
def cmd_job_submit(a: argparse.Namespace) -> str | None:
    """validate -> deterministic tar+zstd -> sha256 -> dedupe-check -> upload ->
    ticket -> `submitted` event -> print JOB_ID + follow-up commands.
    --dry-run does everything EXCEPT the B2 mutations (upload/ticket/event)."""
    src = os.path.abspath(a.dir)
    if not os.path.isdir(src):
        sys.exit(f"error: not a directory: {a.dir}")
    local = getattr(a, "local", False)
    if local:
        if a.box and str(a.box) != joblocal.local_box_id():
            sys.exit("error: --box and --local are mutually exclusive (--local "
                     f"queues onto {joblocal.local_box_id()})")
        box = joblocal.local_box_id()
    elif not a.box:
        sys.exit("error: --box <IID> is required (or --local to run on this "
                 "machine's GPUs — tools/vast/LOCAL_GPU_LANE.md)")
    else:
        box = str(a.box)
    try:
        raw = jobmeta.load_job_config(src)
        if a.name:
            raw["name"] = a.name
        if a.timeout is not None:
            raw["timeout_s"] = a.timeout
        art_keys = _apply_artifact_env(raw, getattr(a, "artifact", None))
        env_keys = _apply_env_overrides(raw, getattr(a, "env", None))
        cfg, warnings = jobmeta.validate_job_config(raw, src)
    except jobmeta.JobmetaError as e:
        sys.exit(f"error: {e}")
    for w in warnings:
        print(f"warn: {w}", file=sys.stderr)
    if art_keys:                # registry-composed; same keys-only audit rule
        print(f">> artifact env (from the modelkit registry): "
              f"{', '.join(sorted(art_keys))}")
    if env_keys:                # keys only — an env value can be a credential
        print(f">> env override (submit-time): {', '.join(env_keys)}")

    # EVAL_ENV_VER gate (M4): a `needs.venv: eval` job submitted with no pin
    # ANYWHERE — job `env:`/`--env` or the target box's launch env — is REFUSED,
    # not reported. Rationale, the wave-A FLOOR incident, and why the box env is
    # the pin that actually steers the fetch: jobmeta.eval_env_pin_report. First
    # gate in the submit, before any bundling or B2 read, so refusing is free.
    box_env, box_env_known = {}, False
    needs_eval_venv = ((cfg.get("needs") or {}).get("venv") == "eval")
    if needs_eval_venv and not local and str(box).isdigit():
        # ONE soft instance read, and only for the jobs the gate applies to —
        # a plain job's submit must stay API-free.
        try:
            inst = health._get_instance_soft(int(box))
        except Exception:
            inst = None
        if inst:
            box_env, box_env_known = models._instance_env(inst), True
    lines, refuse = jobmeta.eval_env_pin_report(  # type: ignore[no-untyped-call]
        cfg, box_env, box=box, box_env_known=box_env_known,
        require_box_pin=bool(getattr(a, "require_box_eval_pin", False)))
    for ln in lines:
        print(ln, file=sys.stderr)
    if refuse:
        sys.exit("error: refusing to submit — this needs.venv:eval job has no "
                 "usable EVAL_ENV_VER pin (see above). There is no override "
                 "flag: name the version, so the artifacts can say which env "
                 "graded the wave.")

    # B2 WRITE-SCOPE preflight: every B2 destination the bundle writes must be
    # covered by a key the box will actually hold. $0, pure, no network — so it
    # runs before the B2-reading checks below. Closes the v7 publish 403 (two
    # fully-trained arms marked failed because run.sh wrote checkpoints/ with the
    # read remote): jobmeta.b2_write_preflight.
    _wf = []
    try:
        _wf = jobmeta.b2_write_preflight(cfg, src)
    except Exception as e:                     # never crash a submit on the check
        print(f"note: B2 write-scope preflight skipped ({e})", file=sys.stderr)
    # `--local` runs against the rclone shim's local-dir bucket with NO keys at
    # all, so the grant table cannot bind there: report, never refuse. (The
    # findings still print — a local smoke is where you WANT to read them.)
    lines, refuse = jobmeta.b2_write_scope_report(  # type: ignore[no-untyped-call]
        _wf, allow_unscoped=local or getattr(a, "allow_unscoped_writes", False))
    for ln in lines:
        print(ln, file=sys.stderr)
    if refuse:
        sys.exit("error: refusing to submit — this bundle writes a B2 prefix the "
                 "box has no key for (see above). Route the write through the "
                 "granted remote, or pass --allow-unscoped-writes if this box "
                 "carries a single bucket-wide key.")

    # B2-staleness preflight (GAP 1): refuse when a `tracks:`-declared staged
    # object drifted from the repo file it mirrors, and warn — or, under
    # --strict-assets, refuse — when the runset sentinel heuristic sees drift.
    # Closes two incidents: the 2026-07-12 stale-runset run (a box ran a
    # pre-EVAL_ONLY train.sh edited locally but never re-staged) and the
    # 2026-07-31 half-stale trainer (b2 64,593 B vs HEAD 125,307 B — the
    # sentinel matched and the preflight said nothing). The local rehearsal uses
    # LOCAL fixtures and is structurally blind to B2 drift; this is the only
    # point that reads B2 and sees it. Runs BEFORE any bundling/upload so a
    # refusal costs nothing. --no-asset-check opts out; --allow-stale-assets
    # runs the staged bytes on purpose; absent creds / a blip degrade to a NOTE.
    asset_bytes = {}
    if (cfg.get("assets") or cfg.get("tracks")) and not getattr(a, "no_asset_check", False):
        b2._ensure_b2_remote()
        try:
            findings = jobmeta.asset_preflight(  # type: ignore[no-untyped-call]
                cfg, repo_root=_repo_root())
        except Exception as e:                 # defense in depth — never crash submit
            print(f"note: asset staleness preflight skipped ({e})", file=sys.stderr)
            findings = []
        lines, refuse = jobmeta.asset_preflight_report(  # type: ignore[no-untyped-call]
            findings, strict=getattr(a, "strict_assets", False),
            allow_stale=getattr(a, "allow_stale_assets", False))
        for ln in lines:
            print(ln, file=sys.stderr)
        if refuse:
            sys.exit("error: refusing to submit — stale or INCOMPLETE asset(s) "
                     "on B2 (see above). Re-stage or re-publish, or override "
                     "with --allow-stale-assets / --no-asset-check.")
        # Sizes for the disk advisory below — same loop, same already-ensured
        # remote, one `rclone size` per distinct prefix.
        try:
            asset_bytes = jobmeta.measure_asset_bytes(  # type: ignore[no-untyped-call]
                cfg["assets"])
        except Exception:
            asset_bytes = {}

    # needs.gpu_ram_gb vs measured VRAM (tools/vast/VRAM_SIZING.md). $0, pure,
    # no network. Refuses ONLY when the declared per-card floor is below a peak
    # this exact shape has already been measured to reach — advisory otherwise,
    # because the estimate picks the right card class about two thirds of the
    # time, which is enough to advise with and not enough to block on.
    try:
        _vram = jobmeta.vram_gate_findings(cfg)
    except Exception as e:                     # never crash a submit on advice
        _vram, _ = None, print(f"note: VRAM sizing check skipped ({e})",
                               file=sys.stderr)
    lines, refuse = jobmeta.vram_gate_report(  # type: ignore[no-untyped-call]
        _vram, allow_drift=getattr(a, "allow_vram_drift", False))
    for ln in lines:
        print(ln, file=sys.stderr)
    if refuse:
        sys.exit("error: refusing to submit — needs.gpu_ram_gb is below a peak "
                 "this shape has already measured (see above). Raise it, or "
                 "pass --allow-vram-drift if the shape really has changed.")

    # deterministic bundle + content address
    staging = os.path.join(_repo_root(), "out", "jobs", "_bundles")
    tmp_out = os.path.join(staging, "pending.tar.zst")
    try:
        info = jobmeta.write_bundle(src, tmp_out)
    except jobmeta.JobmetaError as e:
        sys.exit(f"error: {e}")
    sha = info["sha256"]
    final_out = os.path.join(staging, f"{sha}.tar.zst")
    os.replace(tmp_out, final_out)
    if info["zst_size"] > jobmeta.BUNDLE_WARN_BYTES:
        print(f"warn: bundle is {info['zst_size'] / (1<<20):.0f} MiB — stage large "
              f"inputs separately and fetch them from the entrypoint (stage_run.sh "
              f"pattern)", file=sys.stderr)

    _submit_disk_advisory(cfg, asset_bytes, info, box)

    # Software-epoch stamp: the box gets a tar with no .git, so the checkout
    # that produced this run is knowable later only if the submit says so.
    # Rides the ticket env (costs no bundle byte, cannot move the content
    # address above); a config that already pins the key wins. AFTER the bundle
    # so the `--env` merge the tests pin is the only thing shaping cfg["env"]
    # up to that point. jobmeta.stamp_trainer_rev.
    _rev = jobmeta.stamp_trainer_rev(cfg, repo_root=_repo_root())
    if _rev:
        print(f">> trainer rev: {_rev} (stamped as {jobmeta.TRAINER_REV_ENV})")

    job_id = jobmeta.mint_job_id(cfg["name"])
    print(f">> job {job_id}: name={cfg['name']} entrypoint={cfg['entrypoint']} "
          f"box={box} bundle={sha[:12]}… ({info['zst_size']} B, {info['tar_size']} B tar)")

    b2._ensure_b2_remote()
    try:
        exists = jobmeta.bundle_exists(sha)
    except runmeta.RunmetaError as e:
        sys.exit(f"error: {e}")
    print(f">> bundle dedupe: {'HIT (reusing existing object)' if exists else 'MISS (new upload)'}")

    if a.dry_run:
        print(">> [dry-run] NO B2 mutations (no upload, no ticket, no event).")
        print(f">> [dry-run] would upload -> jobs/bundles/{sha}.tar.zst" if not exists
              else ">> [dry-run] bundle already present — upload would be skipped")
        print(f">> [dry-run] would write ticket -> jobs/queue/{box}/{job_id}.json")
        print(f">> [dry-run] local bundle staged: {final_out}")
        return None                # verbatim: the flat original's bare `return`

    if not exists:
        ok, err = jobmeta.upload_bundle(  # type: ignore[no-untyped-call]
            final_out, sha)
        if not ok:
            sys.exit(f"error: bundle upload failed: {err}")

    actor = lifecycle._cli_actor()
    ticket = jobmeta.make_ticket(job_id, sha, actor, cfg, box)
    ok, key, err = jobmeta.write_ticket(ticket)  # type: ignore[no-untyped-call]
    if not ok:
        sys.exit(f"error: ticket write failed: {err}")
    jobmeta.emit_event(job_id, "submitted", actor=actor, bundle_sha256=sha,
                       name=cfg["name"], entrypoint=cfg["entrypoint"],
                       timeout_s=cfg["timeout_s"], box=box,
                       # the bid ladder's lost-work hint (JOBS_CONFIG.md
                       # `defend:`) — resolved at parse time, so this is always
                       # a concrete "dear"/"cheap", never the absent key.
                       defend=cfg.get("defend"),
                       # launch shape, so `ls` can tag CPU-only jobs without
                       # reading the bundle. Absent pre-2026-08-27 -> unknown.
                       gpu=bool((cfg.get("needs") or {}).get("gpu")))
    print(f">> submitted. ticket: {key}")
    print(f">> JOB_ID={job_id}")
    prog = os.path.basename(sys.argv[0])
    print(f">>   status : {prog} job status {job_id}{' --local' if local else ' --watch'}")
    print(f">>   logs   : {prog} job logs {job_id}{' --local' if local else ''}")
    print(f">>   pull   : {prog} job pull {job_id}{' --local' if local else ''}")
    if local:
        print(f">> (nothing runs until you start the local daemon — `{prog} job run-local`)")
    else:
        print(f">> (the box must be running jobd — `{prog} job attach {box}` if it is not)")
        # "this submit re-arms it" below was advisory until 2026-08-27: nothing
        # told the daemon, and the standing watch's own queue poll is silent on a
        # parked box and `unknown` on a B2 blip. Now the submit says so.
        client.fleet_ticket_placed(box, job_id, source="job submit",
                                   announce=False)
        _print_submit_supervision(box, prog)
    return job_id


# moved-from: herdd._print_submit_supervision
def _print_submit_supervision(box: str, prog: str) -> None:
    """Say what is supervising the box we just queued work onto.

    ADVISORY, NEVER A REFUSAL. The documented order is rent -> submit -> arm
    (`launch_jobs_box.sh` submits at line ~173 and watches at ~384), and fleetd
    keeps a `jobs` watch alive through `queue_empty` precisely so that arming
    BEFORE submitting does not park the box on the spot. So "no watch yet" is
    the correct state for a fresh box and must not block anything.

    The case worth shouting about is different and is invisible without this:
    a watch that ALREADY RAN and finished `drained`. Then the box looks
    supervised (it has a watch, and its spend ceiling really did survive) while
    the ladder that rescues an outbid spot box, replaces an evicted one, and
    parks it when the queue empties is gone. See `fleet_watch_supervision`.
    """
    try:
        level, d = client.fleet_watch_supervision(box)
    except Exception:
        return                                  # never break a submit over this
    rearm = f"{prog} fleet watch {box} --profile jobs --budget <USD>"
    if level == "policy":
        spend, cap = d.get("spend_usd"), d.get("budget_usd")
        money = ""
        if cap is not None:
            money = (f", ${float(spend or 0):.2f} of ${float(cap):.2f} spent"
                     f" (${max(0.0, float(cap) - float(spend or 0)):.2f} left)")
        standing = ""
        if d.get("standing"):
            standing = (" [STANDING, dormant — this submit re-arms it]"
                        if d.get("standing_dormant") else " [STANDING]")
        print(f">> supervision: `{d.get('profile')}` watch{money} — "
              f"spend-capable.{standing}")
    elif level == "lapsed":
        spend, cap = float(d.get("spend_usd") or 0), d.get("budget_usd")
        left = (f" ${max(0.0, float(cap) - spend):.2f} of ${float(cap):.2f} left."
                if cap is not None else "")
        print(f"warn: supervision is `bare` on an INHERITED ceiling — this box's "
              f"previous watch already finished (drained).{left}", file=sys.stderr)
        # the three `f` prefixes below carry no placeholder and never did —
        # operator text kept byte-identical to herdd's (plan §7.4), so the
        # F541s are waived rather than "fixed".
        print(f"warn: the spend ceiling survived; the LADDER did not — no outbid "  # noqa: F541
              f"rescue, no eviction replacement, no drain-park. On a spot box "  # noqa: F541
              f"that means this work can be lost silently.", file=sys.stderr)  # noqa: F541
        print(f"warn: re-arm after this submit:  {rearm}", file=sys.stderr)
    elif level == "bare":
        print(f">> supervision: `bare` (observation only). Arm one after this "
              f"submit:  {rearm}")
    elif level == "none":
        print(f">> supervision: no fleet watch yet — arm one AFTER this submit "
              f"(never before, it would park the box):  {rearm}")
    # "unknown" prints nothing: an unreadable state file is not evidence, and a
    # line that cried wolf on every submit is a line nobody reads.

"""`herdd train` — launch a pre-staged training runset on a rented box.

The 55-flag parser and the 700-line launcher that is the herdd-native port of
the now-deleted `launch_train.sh`. It is by a wide margin the largest command in
the surface, and unlike `launch` it is NOT a thin wrapper: the body is a
fourteen-step launch pipeline (runset config -> staleness preflight -> image
resolution -> fail-closed base-model gate -> container env -> B2 marker hygiene
-> onstart staging -> launch contract -> `_do_launch` -> launched event ->
supervise/babysit) and every step is a refusal point that costs $0 before a box
is rented.

What the port had to preserve, beyond the flag bytes
----------------------------------------------------
1. **`a.price` is read back AFTER `_do_launch` mutates the Namespace.** The
   command hands `_do_launch` a freshly-built `argparse.Namespace` (`la`) and
   `_do_launch` writes the resolved bid onto it; `_build_launch_spec` above and
   the supervise handoff below both read the caller's own `a.price`/`dph`. That
   write-back is a real coupling, not an accident — it is pinned by
   `test_vastlib_launch.py::test_do_launch_returns_and_mutates_the_namespace`
   ("cmd_train reads this back"). Reordering the two would silently drop the
   auto-bid from the spec.
2. **`_HERE` is `herdd`'s, not `vastconf`'s.** Every path this command joins
   (`eval-env/IMAGE`, `ensure_base_model.sh`, `runsets/<RUNSET>/base_models.txt`,
   `onstart/train.sh`, `onstart/train_boot.sh`, `runmeta.py`) resolves against
   `tools/vast`, and the flat file used its own module global. Ported as
   `cli._runsets._HERE`, which is the rename-table home for `herdd._HERE` and
   the one the suite monkeypatches — see that module's `_HERE` contract.
3. **The `--strict-ceiling` / `--handoff` / `--no-handoff` mutually-exclusive
   group.** Identical to `supervise`'s, DEFAULT = handoff, and invisible in
   `--help`: argparse renders a group and three loose flags the same way, so a
   port that dropped the group would quietly start accepting
   `--handoff --no-handoff` (cli-surface.json hazard H6, pinned by
   `test_vastlib_cli_main.py::MUTEX_COMMANDS`).
4. **`--disk` defaults to `None`, on purpose.** The flag has to distinguish
   "passed" from "omitted" so the runset's own `disk:` key can win over the
   global default; the three-way precedence (`--disk` > runset `disk:` >
   `herdd.yaml default_disk_train` > the `vastconf` constant) is resolved in
   step 1.6 of the body, not by argparse.

The two helpers that land here
------------------------------
`_first_noncomment_line` and `_strip_onstart_wire` were flat-file functions with
exactly one caller each — this command — and no home in any lower ring
(cli-surface.json lists both as NOT-PORTED). They are file readers in service of
one command's argv, the same placement argument `_runsets.py` and `_ls_render.py`
make, so they move with their caller rather than inventing a ring for them.
`_strip_onstart_wire` is the 16 KiB inline-onstart budget: Vast caps onstart at
16384 bytes including auto-prepends, so the wire is stripped and the REAL
trainer is staged per-run to B2 for the boot wire to pull.

What is deliberately NOT here
-----------------------------
* The launch. `launch/launch.py::_do_launch` — same code path `herdd launch`
  takes, reached by module attribute so the suite can steer it.
* The spot/handoff policy. `fleet/client._supervise_argv` builds the handoff
  argv; `supervise/run_lane.py` runs it after the `os.execv`.
* The runset config parse. `cli/_runsets.py` (`config.yaml`'s `spot:`/`env:`
  blocks); the staleness gate itself is `jobmeta.asset_preflight`, the same seam
  `herdd job submit` uses.

Provenance: moved from `tools/vast/herdd.py` (`cmd_train`, `_first_noncomment_line`,
`_strip_onstart_wire`, parser block in `main()`), plan §8 step 6, 2026-08-16,
behavior-preserving.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import subprocess
import sys
import time
from typing import Any

import imageref

from vastlib.boxes import health, lifecycle
from vastlib.cli import _args, _compose, _docs, _runsets
from vastlib.core import api, config, fmt, models
from vastlib.fleet import client as fleet_client
from vastlib.jobs import submit as jobs_submit
from vastlib.launch import launch as launch_mod
from vastlib.launch import spec as launch_spec
from vastlib.storage import b2

import bidpolicy
import jobmeta
import runmeta


# moved-from: herdd._first_noncomment_line
def _first_noncomment_line(path: str) -> str:
    r"""First non-blank, non-comment line of a pinned-IMAGE file, whitespace
    stripped (mirrors `grep -vE '^\s*(#|$)' FILE | head -n1 | tr -d '[:space:]'`).
    '' if the file is absent/empty."""
    try:
        for ln in open(path):
            s = ln.strip()
            if not s or s.startswith("#"):
                continue
            return "".join(s.split())
    except OSError:
        return ""
    return ""


# moved-from: herdd._strip_onstart_wire
def _strip_onstart_wire(path: str) -> str:
    r"""Shrink onstart/train.sh to fit Vast's 16 KiB inline-onstart cap: drop
    blank lines and full-line comments (except the shebang), strip leading
    indentation, and drop CONSERVATIVELY-SAFE trailing inline comments — only
    when the candidate ` # ...` tail contains no quote/backslash/expansion
    char and the prefix has balanced quotes, so a `#` inside a string never
    qualifies. Heredoc bodies (e.g. the rclone.conf <<EOF block) pass through
    VERBATIM. Extends the stripper ported from the now-deleted launch_train.sh
    (which kept inline comments); emitted shell semantics are unchanged."""
    out: list[str] = []
    hd: str | None = None
    for ln in open(path):
        if hd is None:                                   # outside a heredoc body
            s = ln.strip()
            if s == "" or (s.startswith("#") and not s.startswith("#!")):
                continue                                  # drop blank / full-line comment
            m2 = re.search(r"\s+#\s", s)
            if m2:
                tail, pre = s[m2.start():], s[:m2.start()]
                if (not re.search(r"""['"\\$`]""", tail)
                        and pre.count('"') % 2 == 0 and pre.count("'") % 2 == 0):
                    s = pre
            out.append(s + "\n")                          # drop leading indent too
            m = re.search(r"""<<-?\s*['"]?([A-Za-z_]\w*)""", ln)
            if m:
                hd = m.group(1)                           # entering a heredoc: keep everything
        else:
            out.append(ln)
            if ln.strip() == hd:
                hd = None
    return "".join(out)


# moved-from: herdd.cmd_train
def run(a: argparse.Namespace) -> None:
    """Launch a pre-staged training runset on a cheap box (ports the now-deleted
    launch_train.sh).

    DEFAULT IMAGE = the baked train image on our R2 registry (herdd.yaml
    default_image):
    full env + nvcc baked in, one authenticated pull, no rehydrate — this IS
    the fast boot. --with-eval pins the glibc-coupled eval-env image instead.
    --fast-boot/--train-env-ver rehydrate the B2 train-env tarball onto an
    explicit --image (REQUIRED — the slim-base auto-resolution path was
    removed 2026-07-10); HARD error if the env is unstaged on B2."""
    # Same cross-ring wiring `cli/launch.py::run` does and for the same reason:
    # this command reaches `launch_mod._do_launch` directly (step 12 below), so
    # driving `run(ns)` without going through `cli.main.main()` must not leave
    # `--fleet-watch` pointed at a raising seam. See `cli/_compose.py`.
    _compose.bind()
    RUN = a.run
    RUNSET = a.runset

    # 1. RUN_ID is used raw in B2 object keys, event filenames, container env vars,
    # and the run:<RUN_ID> vast label — reject anything that would corrupt those
    # (same regex as runmeta.validate_run_id, so bash/Python/this all agree).
    if not runmeta.RUN_ID_RE.match(RUN):
        sys.exit(f"error: invalid --run {RUN!r}: must match "
                 f"{runmeta.RUN_ID_RE.pattern} (letters/digits/._- , 1-64 chars)")

    # 1.5. Declarative runset config (runsets/<RUNSET>/config.yaml): the 'spot:'
    # block feeds spot DEFAULTS (SPOT_DESIGN §3.4) and the 'env:' block feeds
    # launch-env DEFAULTS (step 7). Every CLI flag below overrides its spot key;
    # an absent/empty file leaves every spot_* var None and env defaults empty, so
    # every wiring site no-ops and behavior is unchanged from before it existed.
    runset_cfg = _runsets._load_runset_config(RUNSET)
    # narrowed through a local (not inline, as the flat file had it) purely so the
    # strict lane can see the `isinstance` — same value, same {} on an absent block.
    # It is the idiom `_runsets._load_runset_spot_config` already uses.
    _spot_raw = runset_cfg.get("spot")
    spot_cfg: dict[str, Any] = _spot_raw if isinstance(_spot_raw, dict) else {}
    spot_max_bid_mult = models._num_dph(spot_cfg.get("max_bid_mult"))
    spot_defend_at = models._num_dph(spot_cfg.get("defend_at"))
    spot_rescue_wait_s = models._num_dph(spot_cfg.get("rescue_wait_s"))
    spot_ckpt_interval_s = models._num_dph(spot_cfg.get("ckpt_interval_s"))
    spot_budget_usd = models._num_dph(spot_cfg.get("budget_usd"))

    # 1.6. Runset `disk:` (velvet P4b). Runsets already declare budget_usd,
    # max_bid_mult and ckpt_interval_s but could NOT declare disk — so seven
    # runset READMEs duplicated the number in prose (60/80/120/200) where nothing
    # reads it and nothing keeps it true. Precedence: --disk > runset disk: >
    # herdd.yaml default_disk_train > the core.config constant.
    if a.disk is None:
        rs_disk = runset_cfg.get("disk")
        try:
            rs_disk = int(float(rs_disk)) if rs_disk not in (None, "") else None
        except (TypeError, ValueError):
            rs_disk = None                 # a malformed key is ignored, not fatal
        if rs_disk and rs_disk > 0:
            a.disk = rs_disk
            print(f">> disk {rs_disk}G (runsets/{RUNSET}/config.yaml `disk:`)",
                  file=sys.stderr)
        else:
            a.disk = config.default_disk_gb("train")

    # 2. --supervise hands off to `herdd supervise` instead of the --babysit poll
    # loop; the two are mutually exclusive and supervise wins if both are given.
    supervise = a.supervise
    babysit = a.babysit
    budget = a.budget if a.budget is not None else spot_budget_usd
    if supervise:
        if budget is None:
            sys.exit("error: --supervise requires --budget (or a runset "
                     "spot.budget_usd default)")
        if babysit:
            print(">> note: --supervise supersedes --babysit (both passed) — "
                  "babysit poll loop disabled", file=sys.stderr)
            babysit = False

    # B2 creds must be present (hoisted from step 4: the --fast-boot verify in
    # step 3 and the marker hygiene in step 9 both touch B2). load_env already ran.
    for k in ("B2_BUCKET", "B2_KEY_ID", "B2_APPLICATION_KEY", "B2_S3_ENDPOINT"):
        if not os.environ.get(k):
            sys.exit(f"error: {k} not set (env or .env)")
    bucket = os.environ["B2_BUCKET"]

    # 2.5. STAGED-RUNSET FRESHNESS GATE (GAP 3, 2026-08-04). The SAME seam as
    # `herdd job submit` and `jobmatrix submit` — jobmeta.asset_preflight —
    # finally reaches this launcher, which had no staleness check of ANY kind.
    #
    # Why that mattered. `train --runset NAME` rents a box and points it at
    # b2:runsets/NAME/, whose contents were pushed by that runset's build.sh at
    # some unrecorded time in the past. All seven runsets stage
    # train_proposer_lora.py that way. Nothing here read B2, so a trainer many
    # commits stale was not merely un-flagged — it was undetectable. That is the
    # 2026-07-31 incident's shape (b2 64,593 B vs HEAD 125,307 B, including a
    # --quant semantics change; HEAD is 147,880 B today), on the one path that
    # did not even have a sentinel heuristic to be fooled.
    #
    # Runs BEFORE any box is rented, so a refusal costs $0. Fails CLOSED on a
    # `tracks:` mismatch (an explicit operator contract, not a guess); absent
    # creds or a transport blip degrade to a NOTE and never block. A runset that
    # declares no `tracks:` gets a LOUD note rather than silence — a quiet pass
    # is indistinguishable from a check that never ran, which is exactly how
    # this gap survived.
    if not getattr(a, "no_asset_check", False):
        try:
            probe = jobmeta.runset_preflight_cfg(RUNSET, runset_cfg)  # type: ignore[no-untyped-call]
        except jobmeta.JobmetaError as e:
            sys.exit(f"error: runsets/{RUNSET}/config.yaml: {e}")
        if not probe["tracks"]:
            print(f"note: runsets/{RUNSET}/config.yaml declares no `tracks:` — "
                  f"the staged payload under b2:runsets/{RUNSET}/ is UNVERIFIED "
                  f"against this checkout. Add a tracks: block (see any other "
                  f"runset) so a stale staged trainer cannot reach a box.",
                  file=sys.stderr)
        else:
            b2._ensure_b2_remote()
            try:
                findings = jobmeta.asset_preflight(  # type: ignore[no-untyped-call]
                    probe, repo_root=jobs_submit._repo_root(), bucket=bucket)
            except Exception as e:        # defense in depth — never crash a launch
                print(f"note: runset staleness preflight skipped ({e})",
                      file=sys.stderr)
                findings = []
            lines, refuse = jobmeta.asset_preflight_report(  # type: ignore[no-untyped-call]
                findings, strict=getattr(a, "strict_assets", False),
                allow_stale=getattr(a, "allow_stale_assets", False))
            for ln in lines:
                print(ln, file=sys.stderr)
            if refuse:
                sys.exit(
                    f"error: refusing to launch — b2:runsets/{RUNSET}/ is STALE "
                    f"against this checkout (details above). The box would train "
                    f"the B2 bytes, not yours. Re-stage with\n"
                    f"     bash tools/vast/runsets/{RUNSET}/build.sh\n"
                    f"   or override with --allow-stale-assets (run the staged "
                    f"bytes on purpose) / --no-asset-check (skip the gate).")

    # 3. Image resolution. Default = the baked R2-hosted train image (herdd.yaml
    # default_image): full env + nvcc baked in, one authenticated pull, no B2
    # rehydrate, no ABI seam — this IS the fast boot. --with-eval pins the
    # glibc-coupled eval-env image instead. --fast-boot/--train-env-ver layer
    # the B2 train-env tarball rehydrate onto an explicit --image and HARD
    # error on any miss. The slim-base auto-resolution default was removed
    # 2026-07-10 (see the section comment above launch_spec._TRAIN_FALLBACK_IMAGE).
    with_eval = a.with_eval
    train_env_ver = a.train_env_ver
    fast_boot = bool(a.fast_boot or train_env_ver)
    image_explicit = a.image is not None
    image = a.image
    if with_eval and fast_boot:
        # Was a HARD conflict: the eval env was glibc-pinned to the axolotl
        # image while training ran its own, so no single box could host both.
        # Since 2026-08-02 eval-env/IMAGE is the unified t211 image, which is
        # also what training bakes into — so the combination is legal, and the
        # only surviving constraint is that the box run the image the eval env
        # was BAKED against (the venv symlinks its interpreter and the binaries
        # are glibc-linked). --fast-boot already demands an explicit --image,
        # so check agreement rather than refusing outright. Advisory, matching
        # how the rehydrate path below already treats image<->env ABI: the
        # operator owns it, and the MANIFEST image_ref check reports mismatches.
        _eval_pin = _first_noncomment_line(
            os.path.join(_runsets._HERE, "eval-env", "IMAGE"))
        if _eval_pin and image and _eval_pin != image:
            print(f"!! --with-eval + --fast-boot: --image {image} differs from "
                  f"the eval-env pin {_eval_pin}.\n"
                  f"   The baked eval env symlinks that image's python and links "
                  f"its glibc; if they disagree the sidecar falls back to its "
                  f"self-heal path (slower) or the eval env fails outright.",
                  file=sys.stderr)
    if fast_boot and not image_explicit:
        sys.exit("error: --fast-boot/--train-env-ver require an explicit "
                 "--image to rehydrate onto — the slim-base auto-resolution "
                 "path was REMOVED (the default baked train image IS "
                 "the fast boot and needs no rehydrate; just drop the flag)")

    fb_env: dict[str, str] = {}                       # FAST_BOOT / TRAIN_ENV_VER container env
    if with_eval:
        # the baked eval env is glibc/python/path-coupled to ONE image — pin the
        # digest in eval-env/IMAGE (bake + box must match) unless --image explicit.
        if not image_explicit:
            pinned = _first_noncomment_line(os.path.join(_runsets._HERE, "eval-env", "IMAGE"))
            if pinned:
                image = pinned
                print(f">> --with-eval: using pinned image {image}")
    if fast_boot:
        # rehydrate the B2 train-env tarball onto the user's --image. The user
        # owns the image<->env ABI match; the MANIFEST image_ref advisory below
        # surfaces an obvious mismatch without blocking (Blackwell-style images
        # legitimately differ from the bake base).
        b2._ensure_b2_remote()
        want_ver = train_env_ver
        if not want_ver:
            rc, out, _ = b2._rclone_soft(["cat", f"b2:{bucket}/train-env/LATEST"])
            want_ver = (out or "").strip() if rc == 0 else ""
            if not want_ver:
                sys.exit("error: --fast-boot: no train-env/LATEST in B2 — bake "
                         "one (tools/vast/train-env/bake.sh all) or pin "
                         "--train-env-ver")
        if not b2._b2_lsf_present(f"b2:{bucket}/train-env/env-{want_ver}.tar.zst"):
            flag = "--train-env-ver" if train_env_ver else "--fast-boot"
            sys.exit(f"error: {flag}: train-env/env-{want_ver}.tar.zst not "
                     f"found in B2 — bake one (tools/vast/train-env/bake.sh all)")
        rc, out, _ = b2._rclone_soft(
            ["cat", f"b2:{bucket}/train-env/env-{want_ver}.MANIFEST.json"])
        try:
            baked_in = (json.loads(out).get("image_ref") or "").strip() if rc == 0 else ""
        except Exception:
            baked_in = ""
        if baked_in and baked_in.removeprefix("docker.io/") != image:
            print(f">> note: env {want_ver} was baked in {baked_in}; rehydrating "
                  f"onto {image} — the python/torch ABIs must match",
                  file=sys.stderr)
        fb_env["FAST_BOOT"] = "1"
        # pin the launch-time-resolved version even on the LATEST path, so a
        # LATEST flip between launch and box boot can't swap the env.
        fb_env["TRAIN_ENV_VER"] = want_ver
        print(f">> fast-boot: train env {want_ver} (rehydrate from B2) onto "
              f"--image {image}")
    if not with_eval and not image_explicit:
        # the default: the baked R2-hosted train image from herdd.yaml (env +
        # nvcc baked in, no rehydrate).
        cfg = config.load_herdd_config()
        image = cfg.get("default_image")
        if not image:
            # Falling back to a stock axolotl pull used to be survivable. It no
            # longer is: that image ships no nvcc and no baked env, so the run
            # boots slowly off Docker Hub and then fails at the first compile.
            # Losing default_image means herdd.yaml is missing or unreadable
            # — say that, rather than renting a box that cannot train.
            sys.exit(
                "error: no default_image in herdd.yaml and no explicit "
                "--image.\n"
                "       Expected default_image: "
                f"{launch_spec._TRAIN_FALLBACK_IMAGE}\n"
                "       (the unified t211 train+serve+eval image). Without it "
                "there is no\n"
                "       image that can actually train — pass --image "
                "explicitly or restore the config.")

    ensure_sh = os.path.join(_runsets._HERE, "ensure_base_model.sh")

    # 5. BASE-MODEL ENFORCEMENT GATE (fail-closed). The box must NEVER fetch a base
    # from HuggingFace on-box (the HF Xet client deadlocks there — wedged
    # modelzoo-reader-01 for 45 min). Here, on the launcher, we guarantee every
    # base this run touches is present-and-COMPLETE on B2 (seeding HF->B2 locally
    # if absent) and pass the B2 subpath to the box; if a required base can't be
    # made present we REFUSE to launch. Bases: runsets/<RUNSET>/base_models.txt
    # ("<role> <hf-id>") and/or --ensure-base ROLE:HFID. --allow-hf = advisory.
    b2._ensure_b2_remote()
    base_model = a.base_model
    selftest_base = None
    base_entries: list[tuple[str, str]] = []
    # role -> measured bytes on B2, for sizing the box's disk from what this run
    # actually stages rather than from a hand-typed --disk.
    base_model_bytes: dict[str, int] = {}
    bm_manifest = os.path.join(_runsets._HERE, "runsets", RUNSET, "base_models.txt")
    if os.path.isfile(bm_manifest):
        for ln in open(bm_manifest):
            parts = ln.split()
            if not parts or parts[0].startswith("#"):
                continue
            if len(parts) >= 2:
                base_entries.append((parts[0], parts[1]))
    for e in a.ensure_base or []:  # type: ignore[misc]  # reuses the except-block name, verbatim
        if not e:
            continue
        role, _, hfid = e.partition(":")
        base_entries.append((role, hfid))

    # advisory (never seed / never fail) whenever we are only inspecting.
    check_mode = bool(a.check_base or a.allow_hf or a.dry_run)
    if not base_entries:
        if base_model:
            if b2._b2_lsf_present(f"b2:{bucket}/{base_model}"):
                print(f">> base-gate: explicit --base-model '{base_model}' present on B2")
            elif a.allow_hf:
                print(f"!! base-gate: --base-model '{base_model}' NOT on B2 but "
                      f"--allow-hf set — proceeding (box may HF-fetch)", file=sys.stderr)
            else:
                print(f"!! base-gate: --base-model '{base_model}' NOT present on B2 "
                      f"— refusing to launch.", file=sys.stderr)
                print(f"!!   seed it first:  {ensure_sh} <hf-id>   "
                      f"(or pass --allow-hf to bypass)", file=sys.stderr)
                sys.exit(7)
        else:
            print(f">> base-gate: no base_models.txt / --ensure-base / --base-model "
                  f"for runset '{RUNSET}' —")
            print(">>   not enforced (legacy runset resolves its own base). "
                  "Declare bases to enforce.", file=sys.stderr)
    else:
        print(f">> base-gate: enforcing {len(base_entries)} base model(s) "
              f"present-on-B2 for '{RUNSET}'")
        for role, hfid in base_entries:
            # --print-bytes widens stdout to "<subpath>\t<bytes>". The gate
            # already resolves every base this run touches, so it is the one
            # place that can hand a caller the model's real size for free —
            # which is what makes `--disk` computable instead of hand-typed.
            cmd = [ensure_sh, hfid, "--print-bytes"]
            if check_mode:
                cmd.append("--check-only")
            # capture stdout (the B2 subpath); let stderr stream to the terminal.
            r = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
            if check_mode:
                out = r.stdout.strip() if r.returncode == 0 else ""
            else:
                if r.returncode != 0:
                    print(f"!! base-gate: could not make '{hfid}' "
                          f"present-and-complete on B2 — refusing to launch.",
                          file=sys.stderr)
                    print(f"!!   rerun to seed:  {ensure_sh} {hfid}   "
                          f"(or pass --allow-hf to bypass)", file=sys.stderr)
                    sys.exit(7)
                out = r.stdout.strip()
            sub, base_bytes = launch_spec.parse_base_gate_stdout(out)
            if base_bytes:
                base_model_bytes[role] = base_bytes
            if sub:
                gb = f" ({base_bytes / 1e9:.1f} GB)" if base_bytes else ""
                print(f">>   [{role}] {hfid} -> b2:{sub}{gb}")
            elif a.check_base:
                print(f">>   [{role}] {hfid} -> ABSENT/incomplete (would seed HF->B2)")
            else:
                print(f"!!   [{role}] {hfid} ABSENT on B2 + --allow-hf — box MAY "
                      f"HF-fetch (RISK)", file=sys.stderr)
            if role == "train" and not base_model and sub:
                base_model = sub
            elif role == "selftest" and sub:
                selftest_base = sub

    # 6. --check-base: print the plan and exit BEFORE any write or launch.
    if a.check_base:
        print(">> --check-base PLAN (no box launched, no upload performed):")
        print(f">>   train --base-model : {base_model or '<absent — would seed HF->B2>'}")
        print(f">>   selftest base (B2) : {selftest_base or '<absent — would seed HF->B2>'}")
        if base_model_bytes:
            # DISTINCT bases, summed: jobd's asset cache is keyed by name, so two
            # roles pointing at the same B2 subpath stage it once. Summing per
            # role would double-count and over-size the box.
            tot = sum(base_model_bytes.values())
            per = "  ".join(f"{r}={b / 1e9:.1f}GB"
                            for r, b in sorted(base_model_bytes.items()))
            print(f">>   base weights       : {tot / 1e9:.1f} GB total  ({per})")
            print(f">>   disk floor         : ~{tot / 1e9:.0f} GB of weights "
                  f"(--disk is {a.disk} GB)")
            print(f">>     + eval env (MANIFEST `sizes.peak_bytes`) + checkpoints "  # noqa: F541 — verbatim print block (plan §7.4)
                  f"(save_total_limit x per-ckpt) + headroom")  # noqa: F541 — verbatim print block (plan §7.4)
        return

    # 7. Container env: B2 creds + run params, base subpaths, HF token, LLM creds
    # (eval only — SPEC MUST 19), fast-boot envs, then extra_env in precedence
    # order: runset config.yaml env: defaults < flag-derived EVAL_*/farm/hold/
    # ckpt envs < --env passthroughs (last wins on the wire).
    # Boxes get an ephemeral no-delete key (or the standing B2_BOX_* pair —
    # docs/plans/keyless-b2-ingest.md) under the unchanged on-box names; the
    # ops pair never leaves the workstation. TTL covers MAX_HOURS + debug-hold.
    train_maxh = 24.0                                 # train.sh watchdog default
    for kv in (a.env or []):
        if kv.startswith("MAX_HOURS="):
            try:
                train_maxh = float(kv.split("=", 1)[1])
            except ValueError:
                pass
    train_key_hours = launch_spec._ephemeral_hours(train_maxh * 3600)
    box_kid, box_key = launch_spec._ship_b2_pair(f"run-{RUN}",
                                     hours=train_key_hours,
                                     dry_run=a.dry_run)
    env_list = [
        f"RUN_ID={RUN}", f"RUNSET={RUNSET}",
        f"B2_KEY_ID={box_kid}",
        f"B2_APPLICATION_KEY={box_key}",
        f"B2_BUCKET={bucket}",
        f"B2_S3_ENDPOINT={os.environ['B2_S3_ENDPOINT']}",
        f"B2_REGION={os.environ.get('B2_REGION', 'us-west-004')}",
    ]
    env_list += [f"{k}={v}" for k, v in launch_spec._b2_eu_pairs()]   # region-aware read replica
    env_list += [f"{k}={v}" for k, v in launch_spec._cdn_pairs()]     # CDN weights mirror
    if base_model:
        env_list.append(f"BASE_MODEL_B2={base_model}")
    if selftest_base:                                 # small base off the on-box HF path
        env_list.append(f"SELFTEST_BASE_B2={selftest_base}")
    if os.environ.get("HF_TOKEN"):
        env_list.append(f"HF_TOKEN={os.environ['HF_TOKEN']}")
    if with_eval:                                     # LLM creds: eval runs ONLY (SPEC MUST 19)
        for k in ("LLM_BASE_URL", "LLM_API_KEY", "OPENROUTER_API_KEY"):
            if os.environ.get(k):
                env_list.append(f"{k}={os.environ[k]}")
    for k, v in fb_env.items():
        env_list.append(f"{k}={v}")
    extra_env: list[str] = []
    # Runset config.yaml env: defaults go FIRST so flag-derived entries and
    # explicit --env passthroughs (both below) override them. ValueError from the
    # helper (bad key / reserved key / non-scalar) is a config bug -> fail closed.
    try:
        runset_env = config._runset_env_defaults(runset_cfg)
    except ValueError as e:
        sys.exit(f"error: {e}")
    if runset_env:
        print(">> runset env defaults (config.yaml env:): " + " ".join(runset_env))
    extra_env += runset_env
    if a.fail_hold is not None:
        extra_env.append(f"FAIL_HOLD_MINUTES={a.fail_hold}")
    if with_eval:
        extra_env.append(f"EVAL_TARGETS={with_eval}")
    if a.eval_cmd is not None:
        extra_env.append(f"EVAL_CMD={a.eval_cmd}")
    if a.eval_grace is not None:
        extra_env.append(f"EVAL_GRACE_MINUTES={a.eval_grace}")
    if a.eval_jobs is not None:
        extra_env.append(f"EVAL_JOBS={a.eval_jobs}")
    if a.eval_env_ver is not None:
        extra_env.append(f"EVAL_ENV_VER={a.eval_env_ver}")
    if a.cpu_farm and a.no_cpu_farm:
        sys.exit("error: --cpu-farm and --no-cpu-farm are mutually exclusive")
    if a.cpu_farm:
        extra_env.append("CPU_FARM=1")
    if a.no_cpu_farm:
        extra_env.append("CPU_FARM=0")
    if a.farm_run is not None:
        extra_env.append(f"FARM_RUN_ID={a.farm_run}")
    # spot: ckpt_interval_s (SPOT_DESIGN §3.4) -> CKPT_INTERVAL; --ckpt-interval wins
    # over the runset default, and an explicit --env CKPT_INTERVAL=... (below) wins
    # over both (generic passthroughs are always last-wins).
    ckpt_interval = a.ckpt_interval if a.ckpt_interval is not None else spot_ckpt_interval_s
    if ckpt_interval is not None:
        extra_env.append(f"CKPT_INTERVAL={int(ckpt_interval)}")
    extra_env += (a.env or [])                        # --env passthroughs last (last wins)
    env_list += extra_env

    # 8. Offer search / pricing. bid by default (auto-picks bidpolicy.BID_TARGET_MULT x min_bid);
    # --on-demand pays the listed rate (and ignores --price).
    if a.on_demand and a.price is not None:
        print(">> note: --price ignored with --on-demand (pays listed rate)",
              file=sys.stderr)
    itype = "ondemand" if a.on_demand else "bid"
    # min-CUDA guard (config.LAUNCH_CUDA_MAX_GOOD — the image's own CUDA runtime
    # line): applied as a search filter when auto-picking, and ENFORCED against a
    # pinned --offer inside launch_mod._do_launch (task #111 — a pinned host used
    # to dodge it, which killed waves A/C on a driver older than the image).
    # --cuda 0 disables both.

    # 9. B2 marker hygiene (SKIP under --dry-run — it performs no writes). Reset the
    # STATUS marker BEFORE launch: on a resume (same --run) it still holds the prior
    # run's terminal DONE/FAILED, and babysit would read that stale marker and
    # destroy the new box before it boots. This advisory rcat also proves the B2
    # creds work before any money is spent. NOT the launched EVENT (emitted
    # post-contract below, once the instance id is known).
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if not a.dry_run:
        b2._b2_rcat(f"b2:{bucket}/checkpoints/{RUN}/STATUS", f"LAUNCHED {ts}\n")
        # clear stale debug-hold markers from a prior run of this id, or the box
        # would tear down on the first poll of its own future fail-hold.
        b2._rclone_soft(["deletefile", f"b2:{bucket}/checkpoints/{RUN}/STOP"])
        b2._rclone_soft(["deletefile", f"b2:{bucket}/checkpoints/{RUN}/EXTEND"])
        # same stale-marker reasoning for the eval channel (distinct key from the
        # training STATUS, SPEC MUST 15) and an explicit --farm-run namespace.
        if with_eval:
            b2._b2_rcat(f"b2:{bucket}/evals/{RUN}/EVAL_STATUS", f"LAUNCHED {ts}\n")
        if a.farm_run:
            b2._b2_rcat(f"b2:{bucket}/farm/{a.farm_run}/FARM_STATUS", f"LAUNCHED {ts}\n")

    # 10. Onstart wire + per-RUN trainer staging. The full onstart/train.sh grew
    # past Vast's 16 KiB inline-onstart cap (box-side handoff/spot guards), so we
    # ship a tiny boot-pull wire (onstart/train_boot.sh, ~2 KiB) and stage the REAL
    # trainer to B2 for it to pull+exec. Staging is PER-RUN (runs/<RUN>/train_main.sh),
    # NOT a shared eval-env path: concurrent launches of different train.sh versions
    # must never race, and a relaunch/understudy — which reuses THIS captured wire
    # (spec.onstart) with the same RUN_ID — then pulls the SAME train_main.sh version
    # as the primary. Full, UNSTRIPPED text (comments kept for box-side debuggability).
    train_main = open(os.path.join(_runsets._HERE, "onstart", "train.sh")).read()
    if not a.dry_run:
        # hard=True: a failed stage must abort before money is spent (a booted box
        # would otherwise pull a missing/stale trainer). --dry-run is read-only on B2.
        b2._b2_rcat(f"b2:{bucket}/runs/{RUN}/train_main.sh", train_main, hard=True)
    else:
        print(f">> would stage onstart/train.sh ({len(train_main.encode('utf-8'))} B) "
              f"-> b2:{bucket}/runs/{RUN}/train_main.sh")
    wire = _strip_onstart_wire(os.path.join(_runsets._HERE, "onstart", "train_boot.sh"))
    wbytes = len(wire.encode("utf-8"))
    if wbytes >= 16000:
        print(f"!! WARN: stripped onstart is {wbytes}B (Vast cap 16384 incl. "
              f"prepends) — launch may 400", file=sys.stderr)

    # 11. Resolve the GPU selector: an explicit --gpu name wins; else a VRAM floor
    # (flag or herdd.yaml train_gpu_ram) drops the name filter and the default
    # preferred-GPU policy takes over in search_offers (GPU_DEFAULT_POLICY_TIERS
    # — bf16-capable only, never pre-Ampere, --any-gpu to widen); else default
    # to h100 (back-compat, and no longer in tension with the allowlist: Hopper
    # rejoined it 2026-08-07). This is what lets an agent ask "does a 5090 fit
    # or do we need a 96GB Blackwell?" by VRAM.
    _cfg = config.load_herdd_config()
    train_gpu_ram = a.gpu_ram if a.gpu_ram is not None else _cfg.get("train_gpu_ram")
    train_gpu_ram = float(train_gpu_ram) if train_gpu_ram else 0.0
    gpu_name = a.gpu or _cfg.get("train_gpu") or (None if train_gpu_ram else "h100")
    gpu_desc = gpu_name if gpu_name else f"≥{train_gpu_ram:g}GB (any card)"

    # 11.5 Declarative launch contract: write runs/<RUN>/spec.json BEFORE the launch
    # PUT so a supervisor eviction-relaunch reproduces THIS box (image/onstart/env/
    # disk/runtype) with zero SSH (SPOT_DESIGN §3.1, fixes G1/G5). Secrets NEVER
    # land in the spec — only secret_env_keys (names); supervise re-injects values
    # from the local env/.env at relaunch. A spec with no instance is harmless; an
    # instance with no spec is the eviction bug. Best-effort (hard=False): a spec
    # write blip degrades relaunch to the event-scrape fallback, never blocks a run.
    if not a.dry_run:
        raw_login = launch_spec.image_login_arg(image, None)   # registry secret from .env
        spec = launch_spec._build_launch_spec(
            run_id=RUN, runset=RUNSET, image=image,
            image_login_ref=(launch_spec._mask_image_login(raw_login) if raw_login else None),
            # cached per-process, and launch_mod._do_launch resolves the same ref moments
            # later to stamp the box env — so this costs no extra API call
            image_digest=imageref.image_tag_digest(image),
            disk=a.disk, runtype="ssh_direct",
            gpu=([gpu_name] if gpu_name else []), gpu_ram=train_gpu_ram,
            num_gpus=a.num_gpus, cuda=a.cuda, env_list=env_list, onstart=wire,
            orig_bid=models._num_dph(a.price), max_bid=None,
            defend_at=(a.defend_at if a.defend_at is not None else spot_defend_at),
            rescue_wait_s=(a.rescue_wait if a.rescue_wait is not None else
                          (int(spot_rescue_wait_s) if spot_rescue_wait_s is not None
                           else None)))
        b2._b2_rcat(f"b2:{bucket}/runs/{RUN}/spec.json",
                 json.dumps(spec, separators=(",", ":")) + "\n", hard=False)

    # 11.6 cred-broker vars (docs/plans/cred-broker-buildout.md §2.1), appended
    # AFTER the spec snapshot on purpose: their CRED/KEY name-families match
    # _SECRET_ENV_RE, so putting them in the spec'd env would strand them in
    # secret_env_keys and a supervise relaunch would refuse on "missing" local
    # values. Guarded appends (not blind) keep setdefault semantics — a runset
    # env: default or --env passthrough already in env_list wins on the wire
    # (later entries overwrite earlier in launch_mod._do_launch's env dict).
    if not any(kv.startswith("CRED_ROLE=") for kv in env_list):
        env_list.append("CRED_ROLE=train")
    _exp = launch_spec._minted_expiry(f"run-{RUN}", train_key_hours)  # None unless minted this call
    if _exp and not any(kv.startswith("B2_KEY_EXPIRES_AT=") for kv in env_list):
        env_list.append(f"B2_KEY_EXPIRES_AT={_exp}")

    # 12. Launch. Reuse cmd_launch's exact path via launch_mod._do_launch: label run:<RUN>,
    # ssh pubkey injection, image_login auto-attach for the private R2 image.
    # --no-hf-token: the box pulls its base from B2 ONLY (base-gate guarantee), so
    # the 1183B hf_login onstart-prepend is dead weight that would push the wire
    # past the 16384 cap; HF_TOKEN still travels as a container env above.
    print(f">> launching {a.num_gpus} x {gpu_desc} ({itype}) for run '{RUN}' "
          f"<- runset '{RUNSET}'")
    la = argparse.Namespace(
        offer=a.offer, offer_machine=getattr(a, "offer_machine", None), type=itype,
        price=(None if a.on_demand else a.price),
        gpu=([gpu_name] if gpu_name else []), num_gpus=a.num_gpus,
        any_gpu=getattr(a, "any_gpu", False),
        any_inet=getattr(a, "any_inet", False),
        gpu_ram=train_gpu_ram, max_dph=None,
        host_disk=0, reliability=0.98, cuda=a.cuda,
        inet_down=a.inet_down, machine=a.machine, host=a.host, geo=a.geo,
        limit=20, unverified=False,
        image=image, disk=a.disk, runtype="ssh_direct", label=f"run:{RUN}",
        onstart=wire, env=env_list, port=None, jupyter=False,
        no_hf_token=True, hf_token=None, ssh=True, ssh_key_file=None,
        template_id=None, no_registry_login=False, login=None,
        dry_run=a.dry_run, force=a.force, wait=0,
        fleet_watch=False,          # registered below as run:<RUN>, not by IID — the
                                    # --fleet-watch default flip (2026-08-20) is on `a`
                                    # above, not on this synthetic `la`; leave False or
                                    # _do_launch double-registers by cid too
        # step 13 below emits a RICHER `launched` (runset/config/scorecard) —
        # suppress launch_mod._do_launch's generic one so an epoch has exactly one.
        _runmeta_launched=True,
    )
    if a.dry_run:
        print(f">> onstart wire: {wbytes} bytes (stripped from onstart/train_boot.sh)")
    cid, offer_id, dph = launch_mod._do_launch(la)
    if a.dry_run or not cid:
        return                                        # dry-run: no box, no event, no watch
    print(f">> instance {cid} launched. checkpoints: "
          f"b2:{bucket}/checkpoints/{RUN}  status marker: .../STATUS")
    if getattr(a, "fleet_watch", False) and not supervise:
        # B1b: a train box that is NOT handing off to supervise still gets a
        # daemon watch, so it is never in the unwatched launch gap.
        fleet_client.fleet_watch_best_effort(f"run:{RUN}", "bare", policy={"instance_id": cid})

    # 13. Emit the authoritative launched EVENT (post-contract, carries the
    # instance id) to runs/<RUN>/events/. Best-effort — never fail the launch.
    # offer_id/dph come straight from launch_mod._do_launch (parity improvement — the bash
    # sed-scraped them from stdout). Host-scorecard fields (machine_id/inet_down/
    # geolocation) via one best-effort instance GET feed tools/vast/hosts.py.
    # gpu= records the requested selector (a name, or the VRAM-floor descriptor);
    # the actual card lands via the machine_id/gpu_name on the box side.
    meta = ["--field", f"instance_id={cid}", "--field", f"gpu={gpu_desc}",
            "--field", f"image={image}", "--field", f"disk={a.disk}"]
    if train_gpu_ram:
        meta += ["--field", f"gpu_ram_gb={train_gpu_ram:g}"]
    if offer_id is not None:
        meta += ["--field", f"offer_id={offer_id}"]
    if dph is not None:
        meta += ["--field", f"dph={dph}"]
    if RUNSET:
        meta += ["--field", f"runset={RUNSET}"]
    ok, d, _ = api.request_soft("GET", f"v0/instances/{cid}/")
    if ok:
        inst = d.get("instances", d) if isinstance(d, dict) else d
        for k in ("machine_id", "inet_down", "geolocation"):
            v = (inst or {}).get(k)  # type: ignore[assignment]  # `v` was str above
            if v not in (None, ""):
                meta += ["--field", f"{k}={v}"]
    r = subprocess.run(  # type: ignore[assignment]  # `r` was CompletedProcess[str] above
        ["python3", os.path.join(_runsets._HERE, "runmeta.py"),
         "emit", RUN, "launched", *meta],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if r.returncode != 0:
        print(">> note: runmeta launched-event emit failed (non-fatal)",
              file=sys.stderr)

    # 14. Post-launch mode: supervise | babysit | none.
    if supervise:
        # spot.max_bid_mult/defend_at/rescue_wait_s (SPOT_DESIGN §3.4) feed the
        # supervise handoff as DEFAULTS; an explicit CLI flag on THIS command wins.
        # max_bid_mult needs the actual resolved bid (dph) — unknown until now.
        sup_max_bid = a.max_bid if a.max_bid is not None else (
            round(spot_max_bid_mult * dph, 3)
            if spot_max_bid_mult is not None and dph is not None else None)
        sup_defend_at = a.defend_at if a.defend_at is not None else spot_defend_at
        sup_rescue_wait = a.rescue_wait if a.rescue_wait is not None else (
            int(spot_rescue_wait_s) if spot_rescue_wait_s is not None else None)
        sup_argv = fleet_client._supervise_argv(a, RUN, budget, sup_max_bid,
                                   sup_defend_at, sup_rescue_wait)
        print(f">> handing off to supervisor (budget {fmt.dollars(budget)}): "
              f"herdd supervise {RUN}")
        os.execv(sys.executable, sup_argv)
    elif babysit:
        print(">> babysitting (polling STATUS every 60s; Ctrl-C to stop watching "
              "— instance keeps running,")
        print(">>  but the instance also parks itself on DONE/FAILED "
              "(TEARDOWN=destroy for the old self-destruct); babysit is the second net)")
        start_ts = time.time()
        boot_samp = None                              # boot-throughput sampler (opt-in)
        while True:
            rc, out, _ = b2._rclone_soft(["cat", f"b2:{bucket}/checkpoints/{RUN}/STATUS"])
            st = (out or "").strip() if rc == 0 else "starting"
            now = datetime.datetime.now(datetime.timezone.utc).strftime("%H:%M:%S")
            print(f"   [{now}] status: {st}")
            if st.startswith("DONE"):
                if not a.keep:
                    # second net mirrors the box's own default: PARK, don't destroy
                    # (idempotent on a box that already self-parked). Storage still
                    # bills until a `herdd destroy`.
                    okd, derr = lifecycle._put_state_soft(cid, "stopped")
                    print(f">> parked {cid} (run DONE) — resume: herdd start {cid}; "
                          f"destroy when finished: herdd destroy {cid} -y" if okd
                          else f">> {cid} already gone or park failed ({derr}) — "
                               f"verify: herdd ls")
                break
            elif st.startswith("FAILED") or st.startswith("STAGED"):
                print(f">> run did not complete ({st}) — instance parks itself "
                      f"after the debug-hold (KEEP_ON_FAIL=1 keeps it RUNNING)")
                print(f">>   logs/checkpoints: b2:{bucket}/checkpoints/{RUN}/  "
                      f"(onstart.log is pushed there)")
                print(">>   verify teardown:  herdd ls")
                print(">>   the box holds SSH-able ~FAIL_HOLD_MINUTES for debugging "
                      "— respinning is slow (re-pulls image/base)")
                print(f">>   keep it up : tools/vast/debug_box.sh extend {RUN}     "
                      f"tear down now: tools/vast/debug_box.sh stop {RUN}")
                print(f">>   its status : tools/vast/debug_box.sh status {RUN}     "
                      f"open a shell : herdd ls  then  herdd ssh <id>")
                break
            elif st == "starting" or st.startswith("LAUNCHED"):
                # >30 min without the box writing RUNNING usually means it never
                # booted rclone (bad image, dead host, outbid during load).
                if time.time() - start_ts > 1800:
                    print(f"!! not RUNNING after 30 min — check: herdd show {cid}  "
                          f"(destroy if wedged: herdd destroy {cid} -y)")
                # Boot-throughput watchdog (opt-in --boot-health): sample the
                # docker image-pull rate while the box is still pre-RUNNING. On a
                # sustained-slow host, DESTROY (never park — nothing warm) + report
                # a condemnation. babysit is a passive net with no relaunch loop,
                # so — unlike supervise — it stops here and leaves the relaunch to
                # the operator (or a `--supervise` handoff, which DOES auto-relaunch).
                if getattr(a, "boot_health", False):
                    inst = health._get_instance_soft(cid)
                    if inst is not None and (inst.get("actual_status") or "").lower() \
                            in health._BOOT_LOADING_STATES:
                        if boot_samp is None:
                            boot_samp = health.BootThroughputSampler(
                                min_mbps=config._boot_knob("BOOT_MIN_MBPS"),
                                window_s=config._boot_knob("BOOT_MBPS_WINDOW_S", cast=int),
                                deadline_s=10 ** 9, start_t=time.time())
                        if boot_samp.feed(inst, time.time()) == "slow":
                            machine = inst.get("machine_id")
                            window_s = int(config._boot_knob("BOOT_MBPS_WINDOW_S", cast=int))
                            try:
                                runmeta.emit_event(
                                    RUN, "boot_killed_slow", actor=lifecycle._cli_actor(),
                                    instance_id=cid, machine_id=machine,
                                    mbps=round(boot_samp.last_mbps or 0.0, 3),
                                    window_s=window_s, phase=boot_samp.phase)
                            except Exception:
                                pass
                            lifecycle._destroy_soft(cid)
                            print(f"!! CONDEMNED {cid} (machine {machine}): image pull "
                                  f"{boot_samp.last_mbps or 0.0:.2f} MB/s < "
                                  f"{config._boot_knob('BOOT_MIN_MBPS'):g} MB/s over {window_s}s "
                                  f"({boot_samp.phase}) — slow host, destroyed. "
                                  f"Relaunch on a different machine "
                                  f"(exclude {machine}); or use --supervise for "
                                  f"auto-relaunch.")
                            break
            # if the instance vanished (TEARDOWN=destroy / manual destroy), stop polling.
            okg, dg, errg = api.request_soft("GET", f"v0/instances/{cid}/", retries=2)
            present = okg and bool(
                dg.get("instances", dg) if isinstance(dg, dict) else dg)
            if not present:
                print(f">> instance {cid} no longer exists — done (last status: {st})")
                break
            time.sleep(60)
    else:
        print(">> not babysitting. The instance parks itself on completion "
              "(and at MAX_HOURS=24 by default); parked disk bills until destroyed.")
        print(f">>   resume: herdd start {cid}    teardown: herdd destroy {cid} -y"
              f"    audit: herdd ls")
        print(f">>   if it crashes, it holds for debugging: "
              f"tools/vast/debug_box.sh status|extend|stop {RUN}")


# --------------------------------------------------------------------------- #
# job — B2-mediated job submission (JOBS_DESIGN.md); jobmeta.py is the pure core
# --------------------------------------------------------------------------- #


def add_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser],
               add_cmd: _args.AddCmd) -> argparse.ArgumentParser:
    ptr = add_cmd(sub, "train",
                  "launch a pre-staged training runset on a cheap box "
                  "(fail-closed base-model gate, babysit/supervise) — the "
                  "herdd-native port of the now-deleted tools/vast/launch_train.sh",
                  _docs.DOC_TRAINING, _docs.DOC_SKILL_RUNS, _docs.DOC_SKILL_IMAGE,
                  "DEFAULT image: the baked R2-hosted train image (herdd.yaml "
                  "default_image — env + nvcc baked in, no rehydrate); "
                  "--with-eval pins the eval-env image")
    ptr.add_argument("--run", required=True,
                     help="RUN_ID (checkpoints/artifacts keyed on this; reuse to resume)")
    ptr.add_argument("--runset", required=True,
                     help="pre-staged bundle name in b2:.../runsets/NAME")
    ptr.add_argument("--gpu", default=None,
                     help="GPU alias (default h100, unless --gpu-ram is given). "
                          "Aliases: 5090, h100, rtxpro6000 (96GB Blackwell), b200, …")
    ptr.add_argument("--gpu-ram", dest="gpu_ram", type=float, default=None, metavar="GB",
                     help="min GPU VRAM in GB — drop the name filter and pick the "
                          "cheapest PREFERRED card that fits (default GPU policy: "
                          "bf16-capable cards >=32 GB first — 5090 / RTX PRO / "
                          "H100 / H200 / A100 / B200 rank on price alone; add "
                          "--any-gpu for truly ANY family). "
                          "Config default: train_gpu_ram in herdd.yaml.")
    ptr.add_argument("--any-gpu", dest="any_gpu", action="store_true",
                     help="disable the preferred-GPU policy on auto-pick and "
                          "consider ANY card matching the filters — may land on "
                          "pre-Ampere silicon with no bf16 (the 2026-08-03 "
                          "Quadro RTX 8000 incident)")
    ptr.add_argument("--num-gpus", dest="num_gpus", type=int, default=1)
    ptr.add_argument("--price", type=float, default=None,
                     help=f"bid $/hr (default: auto = {bidpolicy.BID_TARGET_MULT:g}x min_bid, "
                          "clamped below on-demand)")
    ptr.add_argument("--on-demand", "--ondemand", dest="on_demand", action="store_true",
                     help="non-interruptible instance (pays listed dph; --price ignored)")
    # default None so cmd_train can tell "flag passed" from "flag omitted" and
    # let the runset's own `disk:` key win over the global default.
    ptr.add_argument("--disk", type=int, default=None,
                     help=f"container disk GB (default: the runset's `disk:`, "
                          f"else herdd.yaml default_disk_train, else "
                          f"{config.default_disk_gb('train')})")
    ptr.add_argument("--image", default=None,
                     help="docker image (default: the baked R2-hosted train image, "
                          "herdd.yaml default_image; combine with --fast-boot "
                          "to rehydrate the B2 train env onto a custom image)")
    ptr.add_argument("--fast-boot", dest="fast_boot", action="store_true",
                     help="rehydrate the B2 train-env tarball onto an explicit "
                          "--image (REQUIRED with this flag); HARD error if no "
                          "env is staged. Pointless without --image: the default "
                          "baked image already carries the env")
    ptr.add_argument("--train-env-ver", dest="train_env_ver", default=None, metavar="V",
                     help="pin the B2 train-env tarball version (implies --fast-boot, "
                          "so also needs --image; HARD error if not staged on B2)")
    ptr.add_argument("--base-model", dest="base_model", default=None, metavar="B2SUB",
                     help="pre-staged base model B2 subpath (usually auto-derived by the gate)")
    ptr.add_argument("--ensure-base", dest="ensure_base", action="append", default=None,
                     metavar="ROLE:HFID",
                     help="extra base to gate (role train|selftest|aux; repeatable)")
    ptr.add_argument("--allow-hf", dest="allow_hf", action="store_true",
                     help="escape hatch: downgrade the base gate to advisory (box MAY HF-fetch)")
    ptr.add_argument("--check-base", dest="check_base", action="store_true",
                     help="run the gate check-only, print the resolved plan, then exit (no launch)")
    ptr.add_argument("--inet-down", dest="inet_down", type=float, default=None, metavar="MBPS",
                     help="min host download Mbps (default: the "
                          "LAUNCH_INET_DOWN_MBPS knob, "
                          f"{int(config._BOOT_KNOB_DEFAULTS['LAUNCH_INET_DOWN_MBPS'])} — "
                          "image + base-model pulls are bandwidth-bound; 0 "
                          "disables, --any-inet is the escape hatch)")
    ptr.add_argument("--any-inet", dest="any_inet", action="store_true",
                     help="disable the default inet-down floor on auto-pick")
    ptr.add_argument("--geo", action="append", default=None, metavar="CC",
                     help="restrict to 2-letter country code(s), e.g. US (repeatable)")
    ptr.add_argument("--cuda", type=float, default=config.LAUNCH_CUDA_MAX_GOOD,
                     metavar="V",
                     help="min host cuda_max_good (default 12.8 — the CUDA-12 "
                          "driver floor the cu129 image is rented at, config."
                          "LAUNCH_CUDA_MAX_GOOD; 0 disables). BEST-EFFORT under --offer: "
                          "vast's offer `id` filter resolves nothing, so a pinned "
                          "offer the listing scan misses degrades to a warning + the "
                          "on-box probe. Use --machine to keep it server-enforced")
    ptr.add_argument("--offer", type=int, default=None, help="pin an explicit vast offer id")
    ptr.add_argument("--offer-machine", dest="offer_machine", type=int, default=None,
                     metavar="ID",
                     help="the machine_id behind --offer, so a pinned offer can be "
                          "auto-priced off the machine's live market reads (the "
                          "offer `id` filter cannot price it — see `launch "
                          "--offer-machine`)")
    ptr.add_argument("--machine", action="append", type=int, default=None, metavar="ID",
                     help="restrict auto-pick to vetted machine_id(s) (repeatable)")
    ptr.add_argument("--host", action="append", type=int, default=None, metavar="ID",
                     help="restrict auto-pick to vetted host_id(s) (repeatable)")
    ptr.add_argument("--babysit", action="store_true",
                     help="poll B2 STATUS; park on DONE (second net — the box also self-parks; TEARDOWN=destroy restores destruct)")  # noqa: E501 — verbatim parser block (plan §7.4)
    ptr.add_argument("--supervise", action="store_true",
                     help="hand off to `herdd supervise` after launch (supersedes --babysit; needs --budget)")  # noqa: E501 — verbatim parser block (plan §7.4)
    ptr.add_argument("--budget", type=float, default=None, metavar="USD",
                     help="spend cap passed to `herdd supervise` (required with --supervise; "
                          "default: the runset's spot.budget_usd)")
    ptr.add_argument("--max-bid", dest="max_bid", type=float, default=None, metavar="USD",
                     help="with --supervise: max resume/defend bid $/hr, passed through "
                          "(default: the runset's spot.max_bid_mult x this launch's bid)")
    # threaded through to the supervise handoff. Three mutually-exclusive answers,
    # DEFAULT = handoff (HANDOFF_DESIGN §1/§6/§8; promoted to default 2026-07-15).
    ptr_ceil = ptr.add_mutually_exclusive_group()
    ptr_ceil.add_argument("--strict-ceiling", dest="strict_ceiling", action="store_true",
                     help=f"with --supervise: hard-cap the default ceiling at "
                          f"{bidpolicy.BID_CEILING_ONDEMAND_FRAC:g}x on-demand (let the box terminate "  # noqa: E501 — verbatim parser block (plan §7.4)
                          "above it) instead of get-and-hold; passed through")
    ptr_ceil.add_argument("--handoff", dest="handoff", action="store_true",
                     help="with --supervise: over the 0.50x ceiling, get-and-hold AND migrate "
                          "the run to a cheaper box; passed through. DEFAULT as of 2026-07-15")
    ptr_ceil.add_argument("--no-handoff", dest="handoff", action="store_false",
                     help="with --supervise: get-and-hold only, no migration (escape hatch)")
    ptr.set_defaults(handoff=True)
    ptr.add_argument("--defend-at", dest="defend_at", type=float, default=None,
                     help="with --supervise: proactive raise threshold, x last_bid, passed "
                          "through (default: the runset's spot.defend_at)")
    ptr.add_argument("--rescue-wait", dest="rescue_wait", type=int, default=None, metavar="SECS",
                     help="with --supervise: auto-resume stall cap before relaunch, passed "
                          "through (default: the runset's spot.rescue_wait_s)")
    ptr.add_argument("--wall-budget", dest="wall_budget", type=float, default=None,
                     metavar="HOURS",
                     help="forwarded to the child supervise; bounds the run wall-clock "
                          "AND the handoff amortization horizon (default: the child's 48h)")
    ptr.add_argument("--ckpt-interval", dest="ckpt_interval", type=int, default=None, metavar="SECS",  # noqa: E501 — verbatim parser block (plan §7.4)
                     help="checkpoint push cadence -> CKPT_INTERVAL env (default: the runset's "
                          "spot.ckpt_interval_s, else the box's own 180s default)")
    ptr.add_argument("--keep", action="store_true",
                     help="with --babysit, don't park on DONE (just report)")
    ptr.add_argument("--fail-hold", dest="fail_hold", default=None, metavar="M",
                     help="on crash, keep the box SSH-able M minutes before teardown (default park) "  # noqa: E501 — verbatim parser block (plan §7.4)
                          "(-> FAIL_HOLD_MINUTES)")
    ptr.add_argument("--with-eval", dest="with_eval", default=None, metavar="TARGETS",
                     help="compile+score evals on idle CPU (comma list dc3|rb3|rb3-xenon) "
                          "-> EVAL_TARGETS; pins the eval-env image")
    ptr.add_argument("--eval-cmd", dest="eval_cmd", default=None,
                     help="override per-target eval command (-> EVAL_CMD)")
    ptr.add_argument("--eval-grace", dest="eval_grace", default=None, metavar="M",
                     help="minutes to let evals finish after DONE (-> EVAL_GRACE_MINUTES)")
    ptr.add_argument("--eval-jobs", dest="eval_jobs", default=None, metavar="N",
                     help="eval parallelism cap (-> EVAL_JOBS)")
    ptr.add_argument("--eval-env-ver", dest="eval_env_ver", default=None, metavar="V",
                     help="pin the baked eval-env tarball version (-> EVAL_ENV_VER)")
    ptr.add_argument("--cpu-farm", dest="cpu_farm", action="store_true",
                     help="opt IN to the co-tenant CPU compile-farm (-> CPU_FARM=1). "
                          "DEAD FEATURE, default-OFF everywhere (owner ruling "
                          "2026-08-21): it starved a CPU-sensitive train 16x and "
                          "filled a serve box's disk (69 GB objcache).")
    ptr.add_argument("--no-cpu-farm", dest="no_cpu_farm", action="store_true",
                     help="explicitly disable the CPU compile-farm (-> CPU_FARM=0); "
                          "redundant — OFF is the default on every lane")
    ptr.add_argument("--farm-run", dest="farm_run", default=None, metavar="ID",
                     help="point the CPU farm at a different b2 farm/<ID>/ namespace (-> FARM_RUN_ID)")  # noqa: E501 — verbatim parser block (plan §7.4)
    ptr.add_argument("--env", action="append", default=None, metavar="KEY=VAL",
                     help="extra container env forwarded verbatim (repeatable; e.g. "
                          "KEEP_ON_FAIL=1, MAX_HOURS=N)")
    ptr.add_argument("--dry-run", dest="dry_run", action="store_true",
                     help="fully read-only: base gate check-only, NO B2 writes, NO launch; "
                          "print the resolved launch body (image_login masked) + wire size")
    ptr.add_argument("--force", action="store_true",
                     help="skip the live-run preflight (launch even if a run:<ID> box is live)")
    # Staged-runset freshness gate (GAP 3) — same flag names/semantics as
    # `herdd job submit`, so the two launchers are overridden the same way.
    ptr.add_argument("--strict-assets", dest="strict_assets", action="store_true",
                     help="also REFUSE on a heuristic (sentinel) staleness hit, "
                          "not just a `tracks:`-declared mismatch")
    ptr.add_argument("--allow-stale-assets", dest="allow_stale_assets",
                     action="store_true",
                     help="launch anyway when b2:runsets/NAME/ differs from the "
                          "repo files it declares it mirrors — i.e. train the "
                          "bytes currently on B2 on purpose (still printed loudly)")
    ptr.add_argument("--no-asset-check", dest="no_asset_check", action="store_true",
                     help="skip the staged-runset freshness preflight entirely")
    ptr.add_argument("--boot-health", dest="boot_health", action="store_true",
                     help="opt-in boot-throughput watchdog during the image pull: on a "
                          f"sustained-slow host (< {int(config._BOOT_KNOB_DEFAULTS['BOOT_MIN_MBPS'])} "  # noqa: E501 — verbatim parser block (plan §7.4)
                          f"MB/s over {config._BOOT_KNOB_DEFAULTS['BOOT_MBPS_WINDOW_S']}s) emit "
                          "boot_killed_slow + destroy. With --supervise it auto-relaunches "
                          "on a different machine; with --babysit it condemns + reports "
                          "(no relaunch loop). Knobs: BOOT_MIN_MBPS / BOOT_MBPS_WINDOW_S")
    ptr.add_argument("--no-boot-sla", dest="boot_sla", action="store_false",
                     default=True,
                     help="disable the default come-online boot SLA on the "
                          "--supervise handoff (owner directive 2026-08-03; "
                          f"BOOT_SLA_S={int(config._BOOT_KNOB_DEFAULTS['BOOT_SLA_S'])}s "
                          "to reach `running`, then destroy + relaunch on a "
                          "different machine)")
    ptr.add_argument("--fleet-watch", dest="fleet_watch", action="store_true", default=True,
                         help="no-op: ON by default since 2026-08-20 (FLEET_REVIEW_2026-08-20 "
                              "item 3; kept for back-compat, --no-fleet-watch opts out). "
                              "Registers the run box as run:<RUN> — a BARE watch, which is "
                              "observation + alarms and NOT supervision (no bid defense, no "
                              "relaunch). Skipped under --supervise, which arms its own "
                              "handoff; otherwise the spend-capable ladder is a separate "
                              "`fleet watch run:<RUN> --profile run --budget N`.")  # noqa: E501
    ptr.add_argument("--no-fleet-watch", dest="fleet_watch", action="store_false",
                         help="do NOT register the run box with fleetd after launch")
    ptr.set_defaults(func=run)
    return ptr

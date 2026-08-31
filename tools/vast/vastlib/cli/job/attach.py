"""vastlib.cli.job.attach — `herdd job attach`: install + start jobd on an existing box.

Why this module exists
----------------------
`attach` is the ssh-push half of the jobs lane: it pushes the jobd bundle into
`/workspace/jobd/`, writes `jobd.env` (B2 creds, box identity nonce, idle-park
knobs), starts the daemon detached, and installs the `/root/onstart.sh` hook so
it comes back with the box. It is also the ROTATION lane — re-attaching a
resumed box is how an expired ephemeral B2 key gets replaced.

Three properties of the body are load-bearing and easy to "simplify" away:

1. **The start step is CHECKED.** B2 staging succeeds whether or not the box is
   reachable (a cached bundle re-stages as a no-op), so an unchecked ssh leaves
   the success banner printing over a box where no daemon ever started —
   observed 2026-08-01 on a stopped/outbid box.
2. **The boot-bundle repoint is best-effort but LOUD.** A `launch --jobs` box
   re-pulls its LAUNCH-pinned bundle over `/workspace/jobd` on every container
   start, so a silent failure here means the next park/preempt resume rolls
   this attach back (live incident 2026-07-31, box 46347213).
3. **`getattr(a, ...)` on the idle-park knobs.** `supervise`'s `_reattach`
   builds a MINIMAL `Namespace` without them; the defaults are not argparse's.

What is deliberately NOT here
-----------------------------
* The bundle roster and the import gate — `jobs.bundle` owns
  `_job_attach_files` / `_jobd_import_gate` / `_stage_jobd_bootstrap` as the
  single source of truth (the flat-bundle test and shipcheck both read it
  there).
* Credential minting and the env-pair builders — `launch.spec`
  (`_ship_b2_env`, `_ephemeral_hours`, `_minted_expiry`, `_b2_eu_pairs`,
  `_r2_tc_pairs`); the broker nonce registration is `boxes.remote`.
* `--local`. Installing a daemon on a rented machine is a box concept, so
  `attach` is deliberately absent from `_JOB_LOCAL_SUBCOMMANDS`.

SEAM NOTE (closed 2026-08-17): `supervise.job_lane` + `cmd_start` call this
body through the `boxes.lifecycle.cmd_job_attach` attribute. Nothing below
`cli` may import this module, so that attribute stayed a raising seam until
`cli/_compose.py::bind()` took it as row five — which is also why the raise was
invisible for a day: both call sites swallow exceptions, so an unbound daemon
just stopped rotating B2 keys quietly.

Provenance: verbatim move of `tools/vast/herdd.py::cmd_job_attach` plus its
`main()`-inline `pja` parser block, plan §8 step 6, 2026-08-16. Callees are
resolved to their vastlib homes by module attribute; the body is otherwise
character-identical.
"""

from __future__ import annotations

import argparse
import os
import secrets
import shlex
import subprocess
import sys

from vastlib.boxes import lifecycle, remote
from vastlib.boxes import ssh as boxes_ssh
from vastlib.fleet import client
from vastlib.jobs import bundle
from vastlib.launch import spec


# moved-from: herdd.cmd_job_attach
def cmd_job_attach(a: argparse.Namespace) -> None:
    """Install + start jobd on an EXISTING box (ssh-push, mirroring launch_serve
    --on-box). Pushes the jobd bundle (_job_attach_files: jobd.sh/jobd.py/
    jobmeta.py/runmeta.py/b2_sync.sh/metrics_probe.py + cred_client.py/
    tailnet_join.sh + the venv provisioners fetch_eval_env.sh/job_serve.sh and
    its serve_vllm.sh helper) into /workspace/jobd/, forwards B2 creds, starts
    jobd detached. --dry-run prints the plan without touching the box."""
    iid = a.id
    files = bundle._job_attach_files()
    for f in files:
        if not os.path.isfile(f):
            sys.exit(f"error: missing file to push: {f}")
    bundle._jobd_import_gate(files)            # never push an unimportable bundle
    remote_dir = "/workspace/jobd"

    # B2 creds: ephemeral per-box key (else the standing no-delete pair —
    # keyless-b2-ingest.md). Attach is idempotent, so re-attaching a resumed
    # box is also how you ROTATE an expired key onto it.
    bucket = os.environ.get("B2_BUCKET")
    endpoint = os.environ.get("B2_S3_ENDPOINT")
    region = os.environ.get("B2_REGION", "us-west-004")
    if not (bucket and endpoint):
        sys.exit("error: B2_BUCKET/B2_S3_ENDPOINT required in env/.env "
                 "to run jobd on a box")
    # jobs box writes only under jobs/ -> scoped read/write key pair (falls back
    # to one bucket-wide key without a minter). CREDENTIAL_LIFECYCLE.md
    _att_hours = spec._ephemeral_hours()
    _b2env = spec._ship_b2_env(f"box-{iid}",
                               hours=_att_hours,
                               write_prefix="jobs/", dry_run=a.dry_run)

    if a.dry_run:
        print(f">> [dry-run/attach] would push {len(files)} files -> {iid}:{remote_dir}/")
        for f in files:
            print(f"     {os.path.basename(f)}")
        print(f">> [dry-run/attach] would start: JOBD_IID={iid} bash "
              f"{remote_dir}/jobd.sh (detached)")
        print(">> [dry-run/attach] B2 creds forwarded (masked); NO ssh, NO spend")
        return

    i = lifecycle._get_instance(iid)
    host, port, _ = boxes_ssh._pick_ssh_endpoint(i)
    if not (host and port):
        sys.exit(f"error: no ssh endpoint for {iid} (status={i.get('actual_status')})")
    boxes_ssh._warn_ssh_access(i)
    ssh = ["ssh", "-p", str(port), f"root@{host}",
           "-o", "StrictHostKeyChecking=accept-new", "-o", "LogLevel=ERROR"]
    scp = ["scp", "-P", str(port), "-o", "StrictHostKeyChecking=accept-new",
           "-o", "LogLevel=ERROR"]

    subprocess.run(ssh + [f"mkdir -p {remote_dir}"], check=True)
    subprocess.run(scp + files + [f"root@{host}:{remote_dir}/"], check=True)

    env_lines = [f"export {k}={shlex.quote(v)}" for k, v in _b2env]  # read + scoped write
    env_lines += [
        f"export B2_BUCKET={shlex.quote(bucket)}",
        f"export B2_S3_ENDPOINT={shlex.quote(endpoint)}",
        f"export B2_REGION={shlex.quote(region)}",
        f"export JOBD_IID={shlex.quote(str(iid))}",
        f"export INSTANCE_ID={shlex.quote(str(iid))}",
    ]
    # What this box costs, so a collector can price its own output. Nothing on
    # the box knows it otherwise, which is why no trained arm carries any cost
    # data at all and "does a locally-served 27B beat a frontier model at
    # matched spend?" is unanswerable from the artifacts we have
    # (BUDGET_METRIC_MODEL_2026-08-11.md §2c). `dph_total` is the BILLED rate
    # (bid + storage), which is the one a $/GPU-hour figure wants. Omitted, not
    # zeroed, when vast does not report it: a 0.0 would read downstream as a
    # free box.
    _dph = i.get("dph_total")
    if isinstance(_dph, (int, float)) and _dph > 0:
        env_lines.append(f"export BOX_DPH_USD={float(_dph):.6f}")
    # cred-broker identity (docs/plans/cred-broker-buildout.md §2.1): attach is
    # also the ROTATION lane, so ship a fresh nonce every time; the broker
    # learns it via the /v1/register sha256 below (never via extra_env — an
    # attach can't rewrite launch env). Expiry only when a key was actually
    # minted this call (standing-key fallback has no known expiry).
    _att_nonce = secrets.token_hex(16)
    env_lines += [
        f"export BOX_IDENTITY_NONCE={_att_nonce}",
        "export CRED_ROLE=jobs",
    ]
    _att_exp = spec._minted_expiry(f"box-{iid}", _att_hours)
    if _att_exp:
        env_lines.append(f"export B2_KEY_EXPIRES_AT={_att_exp}")
    if os.environ.get("CRED_BROKER_URL"):
        env_lines.append(
            f"export CRED_BROKER_URL={shlex.quote(os.environ['CRED_BROKER_URL'])}")
    # region-aware read replica
    env_lines += [f"export {k}={shlex.quote(v)}" for k, v in spec._b2_eu_pairs()]
    # shared Triton JIT cache
    env_lines += [f"export {k}={shlex.quote(v)}" for k, v in spec._r2_tc_pairs()]
    # CDN weights mirror (base-model pulls; b2x_boot.sh's rung-0 CDN tier)
    env_lines += [f"export {k}={shlex.quote(v)}" for k, v in spec._cdn_pairs()]
    # idle self-park knobs (per-attach opt-out + tunables). getattr: supervise's
    # _reattach builds a minimal Namespace without these.
    if getattr(a, "no_idle_park", False):
        env_lines.append("export JOBD_IDLE_PARK=0")
    if getattr(a, "idle_park_grace", None) is not None:
        env_lines.append(f"export JOBD_IDLE_PARK_S={int(a.idle_park_grace)}")
    if getattr(a, "no_job_deadline", None) is not None:
        env_lines.append(f"export JOBD_NO_JOB_PARK_S={int(a.no_job_deadline)}")
    remote_env = f"{remote_dir}/jobd.env"
    push_env = subprocess.run(
        ssh + [f"cat > {remote_env} && chmod 600 {remote_env}"],
        input="\n".join(env_lines) + "\n", text=True)
    if push_env.returncode != 0:
        sys.exit("error: failed to write jobd.env on the box")
    remote._broker_register(iid, _att_nonce)   # best-effort; silent without broker env
    # start detached (survives the ssh session); JOBD_STATUS marker follows.
    # This is the ONE step that must actually reach the box: the B2 staging above
    # succeeds against the bucket whether or not the box is reachable (a cached
    # bundle re-stages as a no-op), so without checking here the success banner
    # below prints over a box where no daemon ever started — observed 2026-08-01
    # on a stopped/outbid box, which then looks attached and silently does
    # nothing. Best-effort stays best-effort; unconditional SUCCESS does not.
    _start = subprocess.run(ssh + [
        f"chmod +x {remote_dir}/jobd.sh; . {remote_env} && "
        f"nohup bash {remote_dir}/jobd.sh >{remote_dir}/jobd.log 2>&1 </dev/null & disown; "
        f"echo JOBD_STATUS started pid $!"], check=False)
    if _start.returncode != 0:
        sys.exit(
            f"error: could not start jobd on {iid} (ssh exited {_start.returncode}).\n"
            f"       B2 staging DID succeed, so nothing about that output means the\n"
            f"       box is attached — no daemon is running there.\n"
            f"       Check it is live (`herdd ls`); a stopped/outbid box must be\n"
            f"       resumed first (`herdd start {iid} --wait 600`), then re-run\n"
            f"       `herdd job attach {iid}`.")
    # persistence hook: onstart re-runs on every container start (park/resume,
    # outbid auto-resume), but an attach-started daemon does NOT — append a
    # guarded restart block to /root/onstart.sh so jobd comes back with the box.
    # jobd's own flock makes the hook + a manual attach coexist (the second
    # daemon exits). Best-effort: `job supervise` re-attaches as the backstop.
    hook = (
        "grep -q jobd-autostart /root/onstart.sh 2>/dev/null || "
        f"printf '%s\\n' '[ -f {remote_env} ] && (. {remote_env} && "
        f"nohup bash {remote_dir}/jobd.sh >>{remote_dir}/jobd.log 2>&1 &) "
        "# jobd-autostart (herdd job attach)' >> /root/onstart.sh")
    subprocess.run(ssh + [hook], check=False)
    # keep a RESUME honest: a `launch --jobs` box's boot stanza re-pulls its
    # LAUNCH-pinned content-addressed bundle over /workspace/jobd on every
    # container start, so this attach's upgrade would be silently rolled back
    # by the next preempt/park resume (live incident 2026-07-31, box 46347213:
    # the auto-resume reinstalled a pre-9934aa90 jobd whose truncated-VRAM
    # scheduler matched no ticket, and the box billed full GPU doing nothing).
    # Re-stage the CURRENT bundle (sha-deduped, no-op when unchanged) and
    # repoint the stanza's pinned sha on the box; boxes without the stanza
    # (attach-only) match nothing and are unaffected.
    # Best-effort (like the hook itself): a failed re-stage must not fail the
    # attach, but it MUST be loud — it means the next resume rolls jobd back.
    try:
        boot_sha = bundle._stage_jobd_bootstrap()
        repoint = (r"sed -E -i 's|jobs/jobd-boot/[0-9a-f]{64}\.tar|"
                   f"jobs/jobd-boot/{boot_sha}.tar|g' /root/onstart.sh 2>/dev/null; "
                   f"grep -q 'jobs/jobd-boot/{boot_sha}' /root/onstart.sh 2>/dev/null "
                   "&& echo BOOT_SHA_REPOINTED || echo BOOT_SHA_NO_STANZA")
        subprocess.run(ssh + [repoint], check=False)
    except (Exception, SystemExit) as e:
        print(f"!! boot-bundle repoint FAILED ({type(e).__name__}: {e}) — a "
              f"park/preempt resume will re-pull the LAUNCH-time jobd over "
              f"this attach; re-run `job attach {iid}` once B2 staging works")
    print(f">> jobd attached to {iid} ({remote_dir}); marker: "
          f"b2:{bucket}/jobs/nodes/{iid}/JOBD_STATUS")
    print(">> resume persistence: /root/onstart.sh hook installed (jobd restarts with the box)")
    print(f">> submit work to it: {os.path.basename(sys.argv[0])} job submit <dir> --box {iid}")
    if getattr(a, "fleet_watch", False):
        if client.fleet_watch_best_effort(iid, "bare", policy={"attached": True}):  # B1b
            client.print_bare_watch_hint(iid, "jobs")


# The §5 command-module contract: `add_parser(sub)` + `run(a)`. `run` is an
# alias, not a second function — the parser binds the handler by identity at
# build time, so the name the dispatcher stores stays `cmd_job_attach`.
run = cmd_job_attach


def add_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> argparse.ArgumentParser:
    """Verbatim from `main()`: same help strings, same flag order, same dests."""
    pja = sub.add_parser("attach", help="install + start jobd on an existing box "
                         "(+ /root/onstart.sh hook so it restarts on resume)")
    pja.add_argument("id", type=int)
    pja.add_argument("--no-idle-park", dest="no_idle_park", action="store_true",
                     help="do NOT self-park when the queue drains (default: park)")
    pja.add_argument("--idle-park-grace", dest="idle_park_grace", type=int, default=None,
                     metavar="SECS", help="idle grace before self-park (default 600)")
    pja.add_argument("--no-job-deadline", dest="no_job_deadline", type=int, default=None,
                     metavar="SECS", help="park deadline if no job ever arrives (default 3600)")
    pja.add_argument("--dry-run", dest="dry_run", action="store_true")
    # Re-WORDED 2026-08-25 (was a re-wrap of the flat source's one 140-column
    # line): `bare` is not supervision, and this is the one of the three
    # `--fleet-watch` flags that is still opt-in. Fixture amended, not
    # regenerated — `test_vastlib_cli_surface.py --amend`.
    pja.add_argument("--fleet-watch", dest="fleet_watch", action="store_true",
                     help="register the box with fleetd (best-effort; closes the "
                          "launch->watch gap — FLEETD_DESIGN §3 B1). OPT-IN here, "
                          "unlike launch/train where it defaults ON. It registers a "
                          "BARE watch, which is NOT supervision — the spend-capable "
                          "ladder is a separate `fleet watch <IID> --profile jobs "
                          "--budget N`, armed AFTER the tickets exist")
    pja.set_defaults(jobfunc=cmd_job_attach)
    return pja

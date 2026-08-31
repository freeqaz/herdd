"""`herdd sync <id> [paths…]` — push this checkout's TRACKED tooling onto a box.

The park/resume companion. A resumed box keeps its disk exactly as parked, which
means it keeps your tooling exactly as parked too — every `start` is followed by
a `sync` or the box runs last week's code with this week's assumptions.

Three properties that are the command, not decoration
-----------------------------------------------------
* **Tracked files only** (`git ls-files`). `.env`, secrets, `*.db`, build junk
  and worktree scratch cannot ship, by construction rather than by an ignore
  list that someone has to maintain.
* **Additive.** rsync without `--delete`: a file removed from the checkout stays
  on the box. Rebake (or fresh-launch) to remove files — stated in the success
  line because the alternative is an operator debugging a stale module.
* **The probed ssh endpoint**, not the API's. The API record can be stale right
  after a resume, which is exactly when this command runs.

The import-closure gate
-----------------------
`_sync_import_gate` refuses a manifest that ships a module without the modules
it top-imports. It is fail-closed on purpose: the 2026-07-30 frontier wave
shipped `witness_frontier.py` without `inplace_build.py` and burned a rented box
on an S0 ImportError after the meter had started. `--no-import-check` downgrades
it to a warning. Explicit `paths` bypass it entirely — they override the
manifest wholesale, so a manifest-derived gate has nothing to say about them.

What is deliberately NOT here
-----------------------------
* Both gate halves. `_sync_file_list` and `_sync_import_gate` live one ring down
  in `jobs/bundle.py`, next to `_jobd_import_gate` — their bundle-side twin. Two
  ship sets, one detector, in one file where "they use the same shipcheck entry
  points" is checkable by reading. `_load_ship_manifest` stays HERE: it parses a
  CLI-facing allowlist file and has no bundle twin.
* Any deletion, and any write outside `--dest`.

Provenance: moved from `tools/vast/herdd.py` (`cmd_sync` + `_load_ship_manifest`,
parser block in `main()`), plan §8 step 6, 2026-08-16, behavior-preserving.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys

from vastlib.boxes import lifecycle
from vastlib.boxes import ssh as ssh_mod
from vastlib.cli import _args, _docs
from vastlib.jobs import bundle


# moved-from: herdd._load_ship_manifest
def _load_ship_manifest(repo_root: str) -> list[str]:
    """Pathspecs from tools/vast/ship_manifest.txt — the box allowlist (engine +
    crack chain + box scripts; tests/docs/research never ship). One pathspec per
    line; leading '!' becomes a ':(exclude)' pathspec; '#' comments and blank
    lines are skipped. Hard-errors if the manifest is missing or has no includes."""
    mpath = os.path.join(repo_root, "tools/vast/ship_manifest.txt")
    try:
        with open(mpath) as f:
            lines = f.readlines()
    except OSError as e:
        sys.exit(f"error: can't read ship manifest {mpath}: {e}")
    specs = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        specs.append(f":(exclude){line[1:].strip()}" if line.startswith("!") else line)
    if not any(not s.startswith(":(exclude)") for s in specs):
        sys.exit(f"error: {mpath} has no include pathspecs")
    return specs


# moved-from: herdd.cmd_sync
def run(a: argparse.Namespace) -> None:
    """rsync the repo's TRACKED tooling onto a box — the park/resume companion.
    A resumed box keeps its disk exactly as parked, which means it keeps your
    tooling exactly as parked too; run `sync` after `start` to bring the ship
    manifest's allowlist (default; pass paths to override wholesale, e.g. for
    tools/pipeline) up to this checkout's state, overlaying the baked eval-env
    tree additively (no deletions — rebake to remove files). Uses the probed
    ssh endpoint (the api one can be stale after a resume)."""
    repo_root = subprocess.run(["git", "rev-parse", "--show-toplevel"],
                               capture_output=True, text=True).stdout.strip()
    if not repo_root:
        sys.exit("error: sync must run from inside the upstream-monorepo repo")
    # explicit `paths` overrides the manifest wholesale, so the manifest-derived
    # gate does not apply to that call.
    if not a.paths:
        bundle._sync_import_gate(repo_root, warn_only=getattr(a, "no_import_check", False))
    paths = a.paths or _load_ship_manifest(repo_root)
    files = bundle._sync_file_list(repo_root, paths)
    i = lifecycle._get_instance(a.id)
    host, port, _ = ssh_mod._pick_ssh_endpoint(i)
    if not (host and port):
        sys.exit(f"error: no ssh endpoint yet (status={i.get('actual_status')})")
    ssh_mod._warn_ssh_access(i)
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".sync") as tf:
        tf.write("\n".join(files) + "\n")
        tf.flush()
        ssh_cmd = (f"ssh -p {port} -o StrictHostKeyChecking=accept-new "
                   f"-o LogLevel=ERROR")
        cmd = ["rsync", "-az", "--info=stats1",
               f"--files-from={tf.name}",
               "--rsync-path", f"mkdir -p {shlex.quote(a.dest)} && rsync",
               "-e", ssh_cmd, repo_root + "/", f"root@{host}:{a.dest}/"]
        if a.dry_run:
            cmd.insert(1, "--dry-run")
        print(f"sync {len(files)} tracked file(s) [{' '.join(paths)}] -> "
              f"{a.id}:{a.dest}  (via {host}:{port})")
        r = subprocess.run(cmd)
    if r.returncode != 0:
        sys.exit(f"error: rsync exited {r.returncode} — if the box lacks rsync: "
                 f"herdd ssh {a.id} --exec 'apt-get update -qq && "
                 f"apt-get install -y -qq rsync'")
    print(f"synced. NOTE: tracked files only — .env/secrets never ship; "        # noqa: F541 — verbatim body (plan §7.4)
          f"deletions don't propagate (fresh-launch a box for a clean slate).")  # noqa: F541 — verbatim body (plan §7.4)


def add_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser],
               add_cmd: _args.AddCmd) -> argparse.ArgumentParser:
    psy = add_cmd(sub, "sync",
                  "rsync tracked repo tooling onto a box (run after a resume — "
                  "the parked disk kept the OLD tooling)",
                  _docs.DOC_README, _docs.DOC_EVALS,
                  "NOTE: tracked files only (git ls-files) — .env/secrets/*.db "
                  "never ship; deletions don't propagate")
    psy.add_argument("id", type=int)
    psy.add_argument("paths", nargs="*",
                     help="repo paths to sync (default: tools/vast/ship_manifest.txt "
                          "pathspecs; overrides the manifest wholesale if given)")
    psy.add_argument("--dest", default="/workspace/eval/upstream-monorepo",
                     help="destination dir on the box — overlays the baked eval-env "
                          "tree (default /workspace/eval/upstream-monorepo); tracked files "
                          "only, additive (no deletions — rebake to remove files); "
                          "avoid syncing under a live farm/eval run")
    psy.add_argument("--dry-run", action="store_true",
                     help="show what would transfer, don't copy")
    psy.add_argument("--no-import-check", action="store_true",
                     help="downgrade the ship-manifest import-closure gate to a "
                          "warning (default: refuse to sync a manifest that ships "
                          "a module without the modules it top-imports)")
    psy.set_defaults(func=run)
    return psy

#!/usr/bin/env python3
"""triton_cache.py — share the Triton JIT cache across boxes. FAIL-OPEN, always.

WHY THIS EXISTS (measured, 2026-08-07)
--------------------------------------
Triton compiles its kernels on first use, per box. The fit-ladder wave-3 runs
measured that cost directly by submitting the SAME bundle twice to the same box
(the second run reuses the on-box cache):

    job                        submit->done   claim   asset pull   run
    H100 cold (fresh box)          656 s      17 s       66 s      572 s
    H100 warm (same box)           332 s       7 s        1 s      324 s
    A100 cold (fresh box)          845 s      94 s      130 s      621 s
    A100 warm (same box)           418 s      17 s        2 s      399 s

The run-phase delta is **248 s (H100) / 222 s (A100)** and the two Triton-free
control cells in the same bundle replicated within 1.3 %, so the delta is JIT,
not the box. That is **26-38 % of a cold short job's wall clock** — comparable
to or larger than the asset pull it sits next to.

Two separate numbers, do not conflate them:
  * ~50 s   — one training-shaped cell's fwd+bwd kernel set (`s12288` alone).
  * ~230 s  — this bundle's full set (fla_sanity across two fla versions x
              three sequence lengths, plus three cells). Triton-heavy benches
              and probes pay this; a long training run pays the former, once,
              and amortises it to noise.

So this is NOT a training-throughput lever. It is a **short-job latency** lever
(probes, smokes, evals, benches) and, more importantly, a **measurement-validity**
one: the cold-JIT tax is exactly what inflated the wave-2 baseline and produced
two wrong claims (a "3.03x host variance" that was cold-vs-warm, and an FSDP2
"2.82x" measured against a cold baseline). See
`docs/plans/witness/perf/FITTING_9B_ON_A_5090_2026-08-06.md` §10-§11.

WHERE IT STORES — R2, owner-directed 2026-08-07
-----------------------------------------------
The cache lives in Cloudflare R2, bucket `shared-triton-cache` (its OWN bucket,
so its credential's blast radius never touches the registry bucket): egress $0,
and the registry's WP8/WP9 geo measurements (R2 2.6-5.9x faster than GitLab in
Asia/PL) apply to any R2-served bytes. Boxes reach it with a bucket-scoped R2
API token shipped as `R2_TC_KEY_ID` / `R2_TC_SECRET_ACCESS_KEY` /
`R2_TC_ENDPOINT` (+ optional `R2_TC_BUCKET`); rclone config is injected via
ENVIRONMENT overlay, never argv (`resolve_remote`). Fallbacks, in order:
`TRITON_CACHE_REMOTE` (verbatim rclone base, operator override — checked
FIRST), the R2_TC_* lane, then B2 via `B2_BUCKET` (what every box already
has). No creds at all -> every command is a clean cold/skip.

TRUST NOTE: unlike the split B2 keys, the R2 token can overwrite and delete
within its bucket, and cache entries are executable kernel binaries — any box
holding the token can poison an entry that later boxes will run. That is the
same trust class as the code we already run from B2 assets on every box, the
digest sidecar still catches transit/at-rest corruption, and the blast radius
is one advisory cache bucket (delete = cold start). Rotate the token at the CF
dashboard if a box is suspected compromised.

The OCI registry (`tools/vast/registry/`, zot -> the same R2 account) was
considered and stays out: it is read-only by construction through the CF edge,
so the push half of this tool cannot go through it.

THE KEY IS A PARTITION, NOT AN IDENTITY (re-keyed 2026-08-21, owner-authorized)
-------------------------------------------------------------------------------
The key is `torch<ver>-triton<ver>-<sm>`, one namespace. It used to be
`torch<ver>-fla<ver>-<sm>`; the swap deliberately invalidated the whole bucket
once, and the reasoning is worth keeping because it decides the next such call.

Triton keys its OWN entries on (kernel source hash, compile options, backend +
arch, Triton version) — `compiler.py` hashes `triton_key()`, `src.hash()`,
`backend.hash()`, `options.hash()` and the observed env vars together, and that
sha256 IS the entry's directory name. So a foreign entry in our tarball is
**inert**: its directory is never looked up, and it can never yield a wrong
kernel. It is dead weight, nothing more. Our key therefore decides exactly ONE
thing — which entries travel together. Too coarse costs bytes; too fine costs a
miss, which costs minutes of JIT.

That makes the fields worth keying on the ones that render an entry 100 % dead
to the puller: **arch** and **Triton version**. `torch` earns its place as the
source of the inductor kernels that also land in this dir. `fla` failed three
ways: it is one kernel source among several (so partitioning the whole bucket
on it discards the shared majority), its version is already inside Triton's own
hash (so it buys no protection), and — decisively — it is the one field a
**bake cannot compute ahead of a box**.

`fla` was also, measurably, a lie. Every bench-bundle key on the live bucket
read `flanone` through 2026-08-20 while every key jobd wrote for the same torch
read `fla0.5.2`, on an image that bakes fla 0.5.2. The cause was in the
bundles: each calls `set_fla_env off` at top level before its cache pull, which
arms `FLA_FORCE_OFF=1` and a `sitecustomize.py` that sets `sys.modules["fla"] =
None`, so `import fla` raises for the rest of the script — including inside
this tool. The field reported an unrelated A/B knob's parked position, not the
box. Dropping `fla` from the key makes that defect moot rather than fixing it:
nothing keyed moves with the caller's PYTHONPATH any more. `detect()` still
REPORTS fla for diagnosis, and now says `masked_FLA_FORCE_OFF` instead of
`none` when that knob is what silenced it, so a sidecar can no longer be read
as "this box has no fla".

Triton's version is now stated rather than inferred from torch, because a baked
key has to state its dependencies: at bake time there is no GPU to ask, so
every field must come from the image's own package set.

WHAT THE OLD KEYS COST, AND WHY THERE IS NO GC
----------------------------------------------
The re-key orphaned 15 `…-fla…-…` keys, 689.4 MB, ~$0.01/month of R2 (measured
on the live bucket 2026-08-21, not estimated). There is deliberately **no reaper
here and should not be one**: a scheduled deleter of executable kernel blobs is
a standing risk with a one-cent-a-month upside, and the obvious heuristic is
wrong — the seven `torch2.11` keys are 345.5 MB, half the bucket, and back the
t212 rollback lane, which is still published.

Retention rule: **delete by hand, never on a timer, and only a key whose
torch/triton pair no longer appears in any published image.** `ls` is the eye —
it reports each key's schema (`v1-fla` legacy vs `v2-triton`), so the orphans
are visible without squinting at name shapes.

And not yet, for a reason that is not sentiment: while the bucket holds no
`v2-triton` key for an arch, the `v1-fla` key for that arch is the **rollback**
— reverting the re-key would land on a warm cache instead of a cold one. So the
order is: let one job per rented arch mint its `v2-triton` key, and only then
consider the eight `torch2.13.0_cu129-*` orphans (343.8 MB), which the current
image supersedes exactly. The `torch2.11` seven wait on the t212 lane being
unpublished, whenever that is. At a cent a month the default is to leave them.

THE INTEGRITY RULE
------------------
Triton keys its own cache on (kernel source hash, compile options, arch,
Triton version), so a *mismatched* entry is a MISS, never a wrong kernel. What
that keying does NOT protect against is a corrupted blob in transit or at rest:
a bit-flipped `.cubin` would be silently loaded into a training run. This
project's doctrine is that the compiler is the sole judge; importing an
unverified binary cache into the compiler's own toolchain would undercut it.

So every tarball is **pulled by digest**: `<key>.sha256` is fetched first, the
tarball is verified against it before a single byte is unpacked, and a mismatch
is treated as a miss. Immutable-by-digest is also why this is a tarball per key
rather than a mutable accumulating remote cache.

FAIL-OPEN IS NOT OPTIONAL
-------------------------
Every failure path here — no bucket, no rclone, no network, bad digest, corrupt
tar, unwritable dest — returns MISS and exit code 0. A cache is an optimisation;
it must never be able to fail a training run. The only non-zero exits are usage
errors (bad arguments), which happen before any job work.

USAGE
-----
    # what key would this box use? (no network)
    python3 triton_cache.py plan --detect

    # before the workload: populate the cache dir (fail-open)
    python3 triton_cache.py pull --dest "$TRITON_CACHE_DIR" --detect

    # after the workload: publish it if the remote has no copy for this key
    python3 triton_cache.py push --src "$TRITON_CACHE_DIR" --detect

    # telemetry: how big is it, how many entries
    python3 triton_cache.py stat --src "$TRITON_CACHE_DIR"

    # what is in the bucket? (read-only; the tool never deletes)
    python3 triton_cache.py ls

DOES THE TAX RECUR ON NOVEL SHAPES? (open, and this tool measures it)
--------------------------------------------------------------------
The wave-3 warm replicate ran the identical rows in the identical order, so its
cache covered exactly what it needed. That proves "re-running the same work is
fast" — NOT that a cache generalises to sequence shapes it has not seen. If
real training re-tunes per novel length, the tax recurs all run long and this
tool is worth far more than the one-time numbers above.

`push` answers it for free: it reports `entries_added` (entries present at push
that were not in the pulled tarball). If that stays at 0 across runs on the same
key, the tax is one-time. If it keeps growing, it recurs. Read it out of the
`triton_cache_push.json` sidecar in the job's results.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

REMOTE_PREFIX = "triton-cache"
_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


# --------------------------------------------------------------------------
# key
# --------------------------------------------------------------------------
def sanitize(part: str) -> str:
    """One key component, filesystem- and URL-safe. Empty -> 'unknown'."""
    out = _SAFE.sub("_", (part or "").strip())
    return out or "unknown"


def cache_key(torch_ver: str, triton_ver: str, sm: str, extra: str = "") -> str:
    """The cache identity: `torch<ver>-triton<ver>-<sm>[-<extra>]`.

    (triton, arch) are what actually decide whether an entry is reachable at
    all; `torch` rides along because the inductor kernels in this same dir are
    its. `extra` is an operator escape hatch — no shipped caller passes it, and
    a test enforces that (one bucket, one namespace).

    A key that is too COARSE costs nothing but a partial hit — the missing
    kernels compile as usual. A key that is too FINE costs a full miss. So err
    coarse: this deliberately does not include the model, sequence length, or
    the kernel libraries (fla &c.) whose versions Triton already hashes itself.
    """
    parts = [f"torch{sanitize(torch_ver)}", f"triton{sanitize(triton_ver)}",
             sanitize(sm)]
    if extra:
        parts.append(sanitize(extra))
    return "-".join(parts)


def detect() -> dict:
    """Best-effort (torch, triton, fla, sm) from the live environment.

    Never raises. `torch`, `triton` and `sm` are the key; `fla` is REPORTED
    only, for diagnosis — see "THE KEY IS A PARTITION" above for why keying on
    it was wrong.
    """
    info = {"torch": "none", "fla": "none", "triton": "none", "sm": "none"}
    try:
        import torch  # noqa: PLC0415
        info["torch"] = getattr(torch, "__version__", "none")
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability(0)
            info["sm"] = f"sm_{major}{minor}"
    except Exception:
        pass
    try:
        import fla  # noqa: PLC0415
        info["fla"] = getattr(fla, "__version__", "present")
    except Exception:
        # Distinguish "this box has no fla" from "an A/B arm parked it off".
        # Every bench bundle calls `set_fla_env off` before its cache pull,
        # which arms FLA_FORCE_OFF=1 and a sitecustomize that nulls the module
        # — reporting that as `none` is what made the old key unreadable.
        if os.environ.get("FLA_FORCE_OFF") not in (None, "", "0"):
            info["fla"] = "masked_FLA_FORCE_OFF"
    try:
        import triton  # noqa: PLC0415
        info["triton"] = getattr(triton, "__version__", "present")
    except Exception:
        pass
    return info


# --------------------------------------------------------------------------
# remote seam — R2 preferred (scoped per-bucket token, egress $0, and the geo
# numbers that favored it for the registry: WP8/WP9, 2.6-5.9x in Asia/PL),
# B2 as the fallback every box already has. `TRITON_CACHE_REMOTE` overrides
# both with a verbatim rclone remote base.
#
# Secrets travel via the rclone ENVIRONMENT (`RCLONE_CONFIG_<NAME>_<OPT>`),
# never argv — argv is world-readable in `ps` on a shared box.
# --------------------------------------------------------------------------
R2_REMOTE_NAME = "TRITONR2"          # rclone remote name minted from env below
R2_DEFAULT_BUCKET = "shared-triton-cache"


def resolve_remote(bucket_arg: str | None = None) -> dict | None:
    """Pick the remote. Returns {"read","write","env","backend"} or None.

    Order:
      1. `--bucket` on the CLI          -> legacy B2 against that bucket
      2. $TRITON_CACHE_REMOTE           -> verbatim rclone base (operator escape)
      3. $R2_TC_KEY_ID/_SECRET_ACCESS_KEY/_ENDPOINT -> R2, bucket $R2_TC_BUCKET
         (default shared-triton-cache), config injected via env overlay.
         `no_check_bucket` is required: the scoped token cannot HeadBucket, and
         without it rclone falls back to CreateBucket and 403s (measured
         2026-08-07 on the first probe).
      4. $B2_BUCKET                     -> B2 (reads `b2:`, writes
         $B2_WRITE_REMOTE, the split-key arrangement boxes already run)
      5. nothing                        -> None; every command reports cold/skip
    """
    if bucket_arg:
        w = os.environ.get("B2_WRITE_REMOTE", "b2")
        return {"read": f"b2:{bucket_arg}", "write": f"{w}:{bucket_arg}",
                "env": {}, "backend": "b2"}
    explicit = os.environ.get("TRITON_CACHE_REMOTE")
    if explicit:
        return {"read": explicit, "write": explicit, "env": {},
                "backend": "explicit"}
    kid = os.environ.get("R2_TC_KEY_ID")
    sec = os.environ.get("R2_TC_SECRET_ACCESS_KEY")
    ep = os.environ.get("R2_TC_ENDPOINT")
    if kid and sec and ep:
        bucket = os.environ.get("R2_TC_BUCKET", R2_DEFAULT_BUCKET)
        pfx = f"RCLONE_CONFIG_{R2_REMOTE_NAME}_"
        env = {pfx + "TYPE": "s3", pfx + "PROVIDER": "Cloudflare",
               pfx + "ACCESS_KEY_ID": kid, pfx + "SECRET_ACCESS_KEY": sec,
               pfx + "ENDPOINT": ep, pfx + "NO_CHECK_BUCKET": "true"}
        base = f"{R2_REMOTE_NAME}:{bucket}"
        return {"read": base, "write": base, "env": env, "backend": "r2"}
    b = os.environ.get("B2_BUCKET")
    if b:
        w = os.environ.get("B2_WRITE_REMOTE", "b2")
        return {"read": f"b2:{b}", "write": f"{w}:{b}", "env": {}, "backend": "b2"}
    return None


def _rclone(args: list, rclone: str, timeout: int, env: dict | None = None) -> tuple:
    try:
        p = subprocess.run([rclone] + args, capture_output=True, timeout=timeout,
                           env={**os.environ, **(env or {})})
        return p.returncode, (p.stdout or b"").decode("utf-8", "replace")
    except Exception as e:  # missing binary, timeout, anything
        return 127, f"{type(e).__name__}: {e}"


def _remote_get(remote: dict, name: str, dest: Path, rclone: str, timeout: int) -> bool:
    rc, _ = _rclone(["copyto", f"{remote['read']}/{REMOTE_PREFIX}/{name}", str(dest),
                     "--retries", "2", "--low-level-retries", "3"],
                    rclone, timeout, remote["env"])
    return rc == 0 and dest.exists()


def _remote_put(remote: dict, name: str, src: Path, rclone: str, timeout: int) -> bool:
    rc, _ = _rclone(["copyto", str(src), f"{remote['write']}/{REMOTE_PREFIX}/{name}",
                     "--retries", "2", "--low-level-retries", "3"],
                    rclone, timeout, remote["env"])
    return rc == 0


def _remote_exists(remote: dict, name: str, rclone: str, timeout: int) -> bool:
    rc, out = _rclone(["lsf", f"{remote['read']}/{REMOTE_PREFIX}/{name}"],
                      rclone, timeout, remote["env"])
    return rc == 0 and out.strip() != ""


# --------------------------------------------------------------------------
# local cache dir
# --------------------------------------------------------------------------
def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stat_dir(d: Path) -> dict:
    """Entry count + byte size of a Triton cache dir. Entries are its immediate
    children (Triton stores one directory per compiled kernel)."""
    if not d.is_dir():
        return {"entries": 0, "bytes": 0, "exists": False}
    entries, total = 0, 0
    for child in d.iterdir():
        entries += 1
        if child.is_dir():
            for root, _, files in os.walk(child):
                for f in files:
                    try:
                        total += os.path.getsize(os.path.join(root, f))
                    except OSError:
                        pass
        else:
            try:
                total += child.stat().st_size
            except OSError:
                pass
    return {"entries": entries, "bytes": total, "exists": True}


def _safe_extract(tar_path: Path, dest: Path) -> int:
    """Extract to a staging dir, then move each top-level entry into place.

    Two reasons not to extract straight into `dest`:
      1. A torn write (killed mid-extract) would leave a half-written .cubin
         that Triton would happily load. Staging + per-entry rename makes each
         entry appear atomically.
      2. Path traversal: members are checked to stay inside the staging dir.
    Returns the number of entries installed.
    """
    dest.mkdir(parents=True, exist_ok=True)
    installed = 0
    with tempfile.TemporaryDirectory(dir=str(dest.parent)) as stage_s:
        stage = Path(stage_s)
        with tarfile.open(tar_path, "r:gz") as tf:
            for m in tf.getmembers():
                target = (stage / m.name).resolve()
                if not str(target).startswith(str(stage.resolve()) + os.sep):
                    continue  # traversal attempt — skip, do not fail the job
                if m.issym() or m.islnk():
                    continue  # a cache needs no links; refuse them
                try:
                    # filter="data" is the hardened extractor (3.12+, backported
                    # to 3.10.12/3.11.4). Our own traversal + link checks above
                    # stay as the floor for older interpreters on box images.
                    tf.extract(m, str(stage), filter="data")
                except TypeError:
                    tf.extract(m, str(stage))
        for child in stage.iterdir():
            final = dest / child.name
            if final.exists():
                continue  # never clobber a locally-compiled entry
            try:
                os.replace(str(child), str(final))
                installed += 1
            except OSError:
                pass
    return installed


def _make_tar(src: Path, tar_path: Path) -> int:
    n = 0
    with tarfile.open(tar_path, "w:gz") as tf:
        for child in sorted(src.iterdir()):
            tf.add(str(child), arcname=child.name)
            n += 1
    return n


# --------------------------------------------------------------------------
# commands — all of these return 0 no matter what goes wrong
# --------------------------------------------------------------------------
def key_inputs(a) -> dict:
    """Exactly what produced the key, and where each field came from.

    Recorded in every sidecar because the failure this cannot otherwise be
    diagnosed from is silent: a field that DETECTS differently on the next box
    changes the key, and a changed key is indistinguishable from a cold start.
    """
    if a.key:
        return {"source": "explicit", "key": a.key}
    d = detect() if a.detect else {}
    out = {"source": "derived", "detected": d or None}
    for f in ("torch", "triton", "sm"):
        given = getattr(a, f, None)
        out[f] = {"value": given or d.get(f, "none"),
                  "from": "flag" if given else ("detect" if d else "default")}
    out["extra"] = {"value": a.extra or "", "from": "flag" if a.extra else "unset"}
    # reported, not keyed — see "THE KEY IS A PARTITION" in the module doc
    out["fla_detected"] = getattr(a, "fla", None) or d.get("fla")
    return out


def _key_from_args(a) -> str:
    if a.key:
        return a.key
    d = detect() if a.detect else {}
    return cache_key(a.torch or d.get("torch", "none"),
                     a.triton or d.get("triton", "none"),
                     a.sm or d.get("sm", "none"),
                     a.extra or "")


def split_key(name: str) -> dict:
    """Best-effort fields back out of a key. For `ls` only — inventory, never
    used to decide a hit.

    Reads BOTH schemas on purpose: the bucket still holds the pre-2026-08-21
    `torch<v>-fla<v>-<sm>` objects, and `schema` is how an operator tells an
    orphan from a live key without pattern-matching the name by eye.
    """
    parts = name.split("-")
    got = {"torch": None, "triton": None, "fla": None, "sm": None,
           "extra": "", "schema": "unknown"}
    if len(parts) >= 1 and parts[0].startswith("torch"):
        got["torch"] = parts[0][len("torch"):]
    if len(parts) >= 2 and parts[1].startswith("triton"):
        got["triton"] = parts[1][len("triton"):]
        got["schema"] = "v2-triton"
    elif len(parts) >= 2 and parts[1].startswith("fla"):
        got["fla"] = parts[1][len("fla"):]
        got["schema"] = "v1-fla"      # orphaned by the 2026-08-21 re-key
    if len(parts) >= 3:
        got["sm"] = parts[2]
    if len(parts) >= 4:
        got["extra"] = "-".join(parts[3:])
    return got


def cmd_plan(a) -> int:
    key = _key_from_args(a)
    remote = resolve_remote(a.bucket)
    print(json.dumps({"key": key,
                      "tarball": f"{REMOTE_PREFIX}/{key}.tar.gz",
                      "digest_sidecar": f"{REMOTE_PREFIX}/{key}.sha256",
                      "backend": remote["backend"] if remote else None,
                      "remote": remote["read"] if remote else None,
                      "detected": detect() if a.detect else None,
                      "key_inputs": key_inputs(a)}))
    return 0


def cmd_ls(a) -> int:
    """Read-only inventory of the remote bucket. Never writes, never deletes.

    There is no GC in this tool on purpose (689 MB of R2 is ~$0.01/month, so a
    reaper would cost more attention than it saves) — but a bucket nobody can
    see is how stale namespaces accumulate unnoticed. This is the eye.
    """
    res = {"op": "ls", "keys": [], "namespaces": {}, "schemas": {},
           "totals": {"keys": 0, "objects": 0, "bytes": 0}}
    remote = resolve_remote(a.bucket)
    if not remote:
        res["reason"] = "no remote configured"
        print(json.dumps(res)); return 0
    res["backend"] = remote["backend"]
    rc, out = _rclone(["lsjson", f"{remote['read']}/{REMOTE_PREFIX}"],
                      a.rclone, a.timeout, remote["env"])
    if rc != 0:
        res["reason"] = f"listing failed (rc={rc})"
        print(json.dumps(res)); return 0
    try:
        objs = [o for o in json.loads(out or "[]") if not o.get("IsDir")]
    except Exception as e:
        res["reason"] = f"unparseable listing: {type(e).__name__}: {e}"
        print(json.dumps(res)); return 0
    keys: dict = {}
    for o in objs:
        name = o.get("Name", "")
        for suf, field in ((".tar.gz", "tarball"), (".sha256", "digest")):
            if name.endswith(suf):
                k = keys.setdefault(name[: -len(suf)], {"key": name[: -len(suf)]})
                k[field] = {"bytes": o.get("Size", 0), "modified": o.get("ModTime")}
                break
    for k, rec in sorted(keys.items()):
        rec.update(split_key(k))
        rec["complete"] = "tarball" in rec and "digest" in rec
        res["keys"].append(rec)
        tb = (rec.get("tarball") or {}).get("bytes", 0)
        ns = rec.get("extra") or ""
        n = res["namespaces"].setdefault(ns, {"keys": 0, "bytes": 0})
        n["keys"] += 1
        n["bytes"] += tb
        # `v1-fla` here is an orphan of the 2026-08-21 re-key: nothing can hit
        # it any more. Retention is by hand, never on a timer — module doc.
        s = res["schemas"].setdefault(rec["schema"], {"keys": 0, "bytes": 0})
        s["keys"] += 1
        s["bytes"] += tb
    res["totals"] = {"keys": len(keys), "objects": len(objs),
                     "bytes": sum(o.get("Size", 0) for o in objs)}
    print(json.dumps(res))
    return 0


def cmd_pull(a) -> int:
    key = _key_from_args(a)
    dest = Path(a.dest)
    res = {"op": "pull", "key": key, "hit": False, "reason": None,
           "entries_installed": 0, "key_inputs": key_inputs(a)}
    remote = resolve_remote(a.bucket)
    if not remote:
        res["reason"] = "no remote configured (R2_TC_*/B2_BUCKET/TRITON_CACHE_REMOTE all absent)"
        print(json.dumps(res)); return 0
    res["backend"] = remote["backend"]
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        digest_f, tar_f = tmp / "d.sha256", tmp / "c.tar.gz"
        # digest FIRST: it is what makes the tarball verifiable
        if not _remote_get(remote, f"{key}.sha256", digest_f, a.rclone, a.timeout):
            res["reason"] = "no digest sidecar for this key (cold)"
            print(json.dumps(res)); return 0
        want = digest_f.read_text(errors="replace").split()[0].strip() \
            if digest_f.stat().st_size else ""
        if not re.fullmatch(r"[0-9a-f]{64}", want or ""):
            res["reason"] = "malformed digest sidecar"
            print(json.dumps(res)); return 0
        if not _remote_get(remote, f"{key}.tar.gz", tar_f, a.rclone, a.timeout):
            res["reason"] = "digest present but tarball fetch failed"
            print(json.dumps(res)); return 0
        got = sha256_file(tar_f)
        if got != want:
            # Treat as a miss, loudly. Never unpack unverified bytes.
            res["reason"] = f"DIGEST MISMATCH want={want[:12]} got={got[:12]}"
            print(json.dumps(res)); return 0
        res["digest"] = got
        try:
            res["entries_installed"] = _safe_extract(tar_f, dest)
            res["hit"] = True
        except Exception as e:
            res["reason"] = f"extract failed: {type(e).__name__}: {e}"
    res["dir"] = stat_dir(dest)
    print(json.dumps(res))
    return 0


def cmd_push(a) -> int:
    key = _key_from_args(a)
    src = Path(a.src)
    res = {"op": "push", "key": key, "pushed": False, "reason": None,
           "key_inputs": key_inputs(a)}
    st = stat_dir(src)
    res["dir"] = st
    # entries_added answers the "does the tax recur?" question (see module doc)
    if a.baseline_entries is not None:
        res["entries_added"] = max(0, st["entries"] - a.baseline_entries)
    remote = resolve_remote(a.bucket)
    if not remote:
        res["reason"] = "no remote configured (R2_TC_*/B2_BUCKET/TRITON_CACHE_REMOTE all absent)"
        print(json.dumps(res)); return 0
    res["backend"] = remote["backend"]
    if not st["exists"] or st["entries"] == 0:
        res["reason"] = "cache dir empty or absent"; print(json.dumps(res)); return 0
    if not a.force and _remote_exists(remote, f"{key}.sha256", a.rclone, a.timeout):
        # --update: the jobd hook's mode. A remote copy exists, but this box
        # compiled kernels beyond it (entries grew past the recorded baseline)
        # — replace the remote so the tax stays one-time fleet-wide. No growth
        # -> keep the remote (and skip re-uploading identical bytes).
        grown = (a.update and a.baseline_entries is not None
                 and st["entries"] > a.baseline_entries)
        if not grown:
            # Say WHY it is not "grown". A caller that omits --update but DID
            # compile new kernels is the frozen-key defect: the remote stays at
            # whatever the first box pushed while every later box recompiles the
            # difference and throws it away. Measured on the live bucket
            # 2026-08-21 — one key sat at 817 entries for five days while the
            # pushes it refused carried 1,574 / 1,695 / 1,788.
            n = (st["entries"] - a.baseline_entries) \
                if a.baseline_entries is not None else None
            if n and n > 0 and not a.update:
                res["stale_remote"] = True
                res["reason"] = (f"remote already has this key and this box compiled "
                                 f"{n} entries beyond the pull baseline — pass "
                                 f"--update to publish them (or --force to replace)")
            else:
                res["reason"] = "remote already has this key (use --force to replace)"
            print(json.dumps(res)); return 0
        res["updating"] = True
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        tar_f, digest_f = tmp / "c.tar.gz", tmp / "d.sha256"
        try:
            res["entries_packed"] = _make_tar(src, tar_f)
        except Exception as e:
            res["reason"] = f"tar failed: {type(e).__name__}: {e}"
            print(json.dumps(res)); return 0
        digest = sha256_file(tar_f)
        res["digest"] = digest
        res["bytes"] = tar_f.stat().st_size
        digest_f.write_text(digest + "\n")
        # tarball FIRST, digest LAST: the sidecar is the commit point, so a
        # torn upload leaves a key that `pull` reports cold rather than one
        # whose digest points at bytes that are not there yet.
        if not _remote_put(remote, f"{key}.tar.gz", tar_f, a.rclone, a.timeout):
            res["reason"] = "tarball upload failed"; print(json.dumps(res)); return 0
        if not _remote_put(remote, f"{key}.sha256", digest_f, a.rclone, a.timeout):
            res["reason"] = "digest upload failed (key stays cold)"
            print(json.dumps(res)); return 0
        res["pushed"] = True
    print(json.dumps(res))
    return 0


def cmd_stat(a) -> int:
    print(json.dumps({"op": "stat", "src": a.src, **stat_dir(Path(a.src))}))
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, need_dir=None):
        p.add_argument("--key", help="use this exact key instead of deriving one")
        p.add_argument("--torch"); p.add_argument("--triton"); p.add_argument("--sm")
        # Not in the key since 2026-08-21. Still accepted, because rejecting it
        # would exit non-zero inside a job, and this tool's whole contract is
        # that it cannot fail one. Recorded in the sidecar as `fla_detected`.
        p.add_argument("--fla", help="REPORTED ONLY — no longer part of the key")
        p.add_argument("--extra", help="extra key discriminator")
        p.add_argument("--detect", action="store_true",
                       help="fill missing key parts from the live torch/triton/GPU")
        p.add_argument("--bucket",
                       help="legacy: force the B2 backend against this bucket "
                            "(default remote resolution: $TRITON_CACHE_REMOTE, "
                            "else R2 via $R2_TC_*, else B2 via $B2_BUCKET)")
        p.add_argument("--rclone", default=os.environ.get("RCLONE_BIN", "rclone"))
        p.add_argument("--timeout", type=int, default=180)
        if need_dir == "dest":
            p.add_argument("--dest", required=True, help="TRITON_CACHE_DIR")
        elif need_dir == "src":
            p.add_argument("--src", required=True, help="TRITON_CACHE_DIR")

    p = sub.add_parser("plan", help="print the key and remote paths, no network")
    common(p); p.set_defaults(func=cmd_plan)
    p = sub.add_parser("pull", help="populate the cache dir (fail-open)")
    common(p, "dest"); p.set_defaults(func=cmd_pull)
    p = sub.add_parser("push", help="publish the cache dir (fail-open)")
    common(p, "src")
    p.add_argument("--force", action="store_true", help="replace an existing key")
    p.add_argument("--update", action="store_true",
                   help="replace an existing key ONLY if entries grew past "
                        "--baseline-entries (the jobd hook's mode)")
    p.add_argument("--baseline-entries", type=int, default=None,
                   help="entry count from the pull, to report entries_added")
    p.set_defaults(func=cmd_push)
    p = sub.add_parser("stat", help="entries + bytes of a cache dir")
    p.add_argument("--src", required=True); p.set_defaults(func=cmd_stat)
    p = sub.add_parser("ls", help="inventory the remote bucket (read-only)")
    common(p); p.set_defaults(func=cmd_ls)
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    try:
        return a.func(a)
    except Exception as e:
        # Last-resort net: a bug in this file must not fail the job either.
        print(json.dumps({"op": getattr(a, "cmd", "?"), "hit": False,
                          "pushed": False,
                          "reason": f"unhandled {type(e).__name__}: {e}"}))
        return 0


if __name__ == "__main__":
    sys.exit(main())

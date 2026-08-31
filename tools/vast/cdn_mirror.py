#!/usr/bin/env python3
"""Mirror base models into the public CDN bucket as sub-ceiling chunks.

Cloudflare's Free/Pro/Business CDN silently refuses to cache a response over
~512 MB, and every useful weight shard we own is larger than that -- so a
straight copy would put 100% of the bytes on the uncacheable path and buy
nothing. This splits each oversized object into chunks below the measured
ceiling.

The split is done with ``b2_copy_file`` + a byte ``range``, which is a
server-side operation: no egress, no download, no local disk. That is the whole
reason this is cheap enough to run over hundreds of GB.

Reassembly is driven by CDN_MANIFEST.json, written beside each model.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request

# Serialize backoff sleeps: when B2 rate-limits the account, having every
# worker back off at once is the point -- staggering them just re-storms it.
_throttle_lock = threading.Lock()

# The Cloudflare Free/Pro/Business cacheable ceiling is exactly 512 MiB and is
# INCLUSIVE: measured 2026-08-26 on zone example.com, 536870912 B caches and
# 536870913 B never does (bench/results/20260826-cloudflare-cache-ceiling.txt).
# The docs say "512 MB" and mean MiB. This is a one-byte cliff whose failure
# mode is silent -- an oversized object is not rejected, it just never caches,
# visible only in cf-cache-status. We sit one MiB under it so that nothing
# incidental (a manifest tweak, a future header) can push a chunk over.
DEFAULT_CHUNK_BYTES = 511 * 1024 * 1024

B2_AUTH_URL = "https://api.backblazeb2.com/b2api/v3/b2_authorize_account"


class B2:
    def __init__(self, key_id: str, app_key: str) -> None:
        req = urllib.request.Request(B2_AUTH_URL)
        import base64

        tok = base64.b64encode(f"{key_id}:{app_key}".encode()).decode()
        req.add_header("Authorization", f"Basic {tok}")
        d = json.load(urllib.request.urlopen(req))
        self.token = d["authorizationToken"]
        self.account_id = d["accountId"]
        api = d["apiInfo"]["storageApi"]
        self.api_url = api["apiUrl"]
        self.download_url = api["downloadUrl"]

    def call(self, endpoint: str, body: dict, retries: int = 10) -> dict:
        data = json.dumps(body).encode()
        for attempt in range(retries):
            req = urllib.request.Request(
                f"{self.api_url}/b2api/v3/{endpoint}",
                data=data,
                headers={"Authorization": self.token,
                         "Content-Type": "application/json"},
            )
            try:
                return json.load(urllib.request.urlopen(req))
            except urllib.error.HTTPError as e:
                payload = e.read().decode(errors="replace")
                # 429/503 are the documented back-pressure codes; everything
                # else is a real error and retrying only hides it. B2
                # rate-limits b2_copy_file at the ACCOUNT level, so this fires
                # under concurrency even though no single call is expensive --
                # honour Retry-After when it sends one rather than guessing.
                if e.code in (429, 503) and attempt < retries - 1:
                    wait = e.headers.get("Retry-After")
                    try:
                        delay = float(wait) if wait else min(2 ** attempt, 60)
                    except ValueError:
                        delay = min(2 ** attempt, 60)
                    with _throttle_lock:
                        time.sleep(delay + attempt)
                    continue
                raise RuntimeError(f"{endpoint} -> {e.code} {payload}") from None
        raise RuntimeError(f"{endpoint}: retries exhausted")

    def bucket_id(self, name: str) -> str:
        r = self.call("b2_list_buckets",
                      {"accountId": self.account_id, "bucketName": name})
        if not r["buckets"]:
            raise SystemExit(f"no such bucket: {name}")
        return r["buckets"][0]["bucketId"]

    def list_slugs(self, bucket_id: str, prefix: str) -> list[str]:
        """Top-level directory names under prefix. Uses B2's `delimiter`, so it
        pages over folders instead of over every shard in the estate."""
        out, start = [], None
        while True:
            body = {"bucketId": bucket_id, "prefix": prefix, "delimiter": "/",
                    "maxFileCount": 1000}
            if start:
                body["startFileName"] = start
            r = self.call("b2_list_file_names", body)
            for f in r["files"]:
                name = f["fileName"]
                if name.endswith("/") and name != prefix:
                    out.append(name[len(prefix):].rstrip("/"))
            start = r.get("nextFileName")
            if not start:
                return sorted(s for s in out if s)

    def list_files(self, bucket_id: str, prefix: str) -> list[dict]:
        out, start = [], None
        while True:
            body = {"bucketId": bucket_id, "prefix": prefix, "maxFileCount": 1000}
            if start:
                body["startFileName"] = start
            r = self.call("b2_list_file_names", body)
            out.extend(r["files"])
            start = r.get("nextFileName")
            if not start:
                return out


def plan_chunks(size: int, chunk: int) -> list[tuple[int, int]]:
    """Byte ranges (inclusive start, inclusive end) covering [0, size)."""
    if size <= chunk:
        return [(0, size - 1)]
    return [(off, min(off + chunk, size) - 1) for off in range(0, size, chunk)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", required=True,
                    help="comma-separated model slugs under the source prefix, "
                         "or `all` to enumerate every slug there. Already-mirrored "
                         "chunks are skipped by size, so `all` is the refresh "
                         "command as well as the first-run one")
    ap.add_argument("--src-bucket", default=os.environ.get("B2_BUCKET"))
    ap.add_argument("--src-prefix", default="base-models")
    ap.add_argument("--dst-bucket", default=os.environ.get("B2_CDN_BUCKET"))
    ap.add_argument("--dst-prefix", default=os.environ.get("B2_CDN_PREFIX"),
                    help="entropy prefix; keeps the public host from being a "
                         "scannable open mirror")
    ap.add_argument("--chunk-bytes", type=int, default=DEFAULT_CHUNK_BYTES)
    ap.add_argument("--concurrency", type=int, default=8,
                    help="parallel server-side copies. Our network is not in "
                         "the path, so the only limit is B2's ACCOUNT-level "
                         "rate limit -- 32 reliably earns a 429")
    ap.add_argument("--apply", action="store_true",
                    help="without this, print the plan and copy nothing")
    a = ap.parse_args()

    for name, val in (("--src-bucket", a.src_bucket), ("--dst-bucket", a.dst_bucket),
                      ("--dst-prefix", a.dst_prefix)):
        if not val:
            raise SystemExit(f"{name} is required (or set its env var)")

    key_id = os.environ.get("MASTER_KEY_ID")
    app_key = os.environ.get("MASTER_APPLICATION_KEY")
    if not (key_id and app_key):
        raise SystemExit("MASTER_KEY_ID / MASTER_APPLICATION_KEY not in env "
                         "(source ~/.config/b2/.env)")

    b2 = B2(key_id, app_key)
    src_id, dst_id = b2.bucket_id(a.src_bucket), b2.bucket_id(a.dst_bucket)

    dst_listing = b2.list_files(dst_id, a.dst_prefix + "/")
    existing = {f["fileName"]: f["contentLength"] for f in dst_listing}
    existing_sha1 = {f["fileName"]:
                     (None if f.get("contentSha1") in (None, "none")
                      else f["contentSha1"])
                     for f in dst_listing}

    grand_objs = grand_chunks = grand_bytes = grand_skipped = 0

    if a.models.strip() == "all":
        models = b2.list_slugs(src_id, a.src_prefix.rstrip("/") + "/")
        if not models:
            raise SystemExit(f"no model slugs under {a.src_prefix}/")
        print(f"--models all -> {len(models)} slugs: {', '.join(models)}\n")
    else:
        models = [m.strip() for m in a.models.split(",") if m.strip()]

    for model in models:
        src_files = b2.list_files(src_id, f"{a.src_prefix}/{model}/")
        if not src_files:
            print(f"!! {model}: no source objects", file=sys.stderr)
            continue

        manifest = {"version": 1, "model": model,
                    "chunk_bytes": a.chunk_bytes, "files": []}
        n_chunks = n_bytes = n_skipped = 0
        # Copies are server-side, so the limit is B2's, not our network's.
        # Collect them all first and issue them concurrently.
        pending: list[tuple[dict, dict]] = []

        for f in sorted(src_files, key=lambda x: x["fileName"]):
            rel = f["fileName"][len(f"{a.src_prefix}/{model}/"):]
            if rel.endswith("/") or rel == "CDN_MANIFEST.json":
                continue
            size = f["contentLength"]
            ranges = plan_chunks(size, a.chunk_bytes)
            multi = len(ranges) > 1
            parts = []

            for i, (lo, hi) in enumerate(ranges):
                dst_rel = f"{rel}.part{i:04d}" if multi else rel
                dst_name = f"{a.dst_prefix}/{a.src_prefix}/{model}/{dst_rel}"
                want = hi - lo + 1
                part = {"name": dst_rel, "offset": lo, "length": want}
                parts.append(part)
                if existing.get(dst_name) == want:
                    # B2 already hashed it on the way in; reuse rather than
                    # re-copy. Per-chunk sha1 is the only integrity check
                    # available for a large file -- B2 reports contentSha1
                    # "none" for anything uploaded via the large-file API,
                    # which is every shard we actually care about.
                    part["sha1"] = existing_sha1.get(dst_name)
                    n_skipped += 1
                    continue
                if a.apply:
                    body = {"sourceFileId": f["fileId"], "fileName": dst_name,
                            "metadataDirective": "COPY",
                            "destinationBucketId": dst_id}
                    if multi:
                        body["range"] = f"bytes={lo}-{hi}"
                        # A ranged copy cannot inherit the source's metadata:
                        # the sha1 and cache-control describe the whole object.
                        body["metadataDirective"] = "REPLACE"
                        body["contentType"] = "application/octet-stream"
                        body["fileInfo"] = {
                            "b2-cache-control":
                                "public, max-age=31536000, immutable"}
                    pending.append((body, part))
                n_chunks += 1
                n_bytes += want

            # A large-file source reports contentSha1 "none"; its whole-file
            # digest, if any, lives in fileInfo. Record whichever exists and
            # let the per-part sha1s carry verification either way.
            whole = f.get("contentSha1")
            if whole in (None, "none"):
                whole = f.get("fileInfo", {}).get("large_file_sha1")
            manifest["files"].append(
                {"path": rel, "size": size, "sha1": whole, "parts": parts})

        if pending:
            t0 = time.time()

            def _copy(job):
                body, part = job
                r = b2.call("b2_copy_file", body)
                sha = r.get("contentSha1")
                part["sha1"] = None if sha in (None, "none") else sha

            with futures.ThreadPoolExecutor(max_workers=a.concurrency) as ex:
                # list() forces the iterator so a failed copy raises here
                # rather than silently leaving a hole in the manifest.
                list(ex.map(_copy, pending))
            rate = n_bytes / 2**30 / max(time.time() - t0, 1e-9)
            print(f"  {len(pending)} copies in {time.time() - t0:.0f}s "
                  f"({rate:.1f} GB/s server-side)")

        if a.apply:
            blob = json.dumps(manifest, indent=1).encode()
            up = b2.call("b2_get_upload_url", {"bucketId": dst_id})
            import hashlib
            req = urllib.request.Request(
                up["uploadUrl"], data=blob,
                headers={"Authorization": up["authorizationToken"],
                         "X-Bz-File-Name":
                             f"{a.dst_prefix}/{a.src_prefix}/{model}/CDN_MANIFEST.json",
                         "Content-Type": "application/json",
                         "X-Bz-Content-Sha1": hashlib.sha1(blob).hexdigest()})
            urllib.request.urlopen(req)

        print(f"{model:<26} {len(manifest['files']):>3} objs  "
              f"{n_chunks:>4} new chunks  {n_bytes / 2**30:>7.1f} GB  "
              f"skipped {n_skipped}")
        grand_objs += len(manifest["files"])
        grand_chunks += n_chunks
        grand_bytes += n_bytes
        grand_skipped += n_skipped

    verb = "copied" if a.apply else "WOULD copy (dry run; pass --apply)"
    print(f"\n{verb}: {grand_chunks} chunks / {grand_bytes / 2**30:.1f} GB "
          f"across {grand_objs} objects; {grand_skipped} already present")
    return 0


if __name__ == "__main__":
    sys.exit(main())

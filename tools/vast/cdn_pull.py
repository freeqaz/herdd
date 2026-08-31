#!/usr/bin/env python3
"""Reassemble a model from CDN-cached chunks described by CDN_MANIFEST.json.

The chunking that makes the objects cacheable also makes them parallel: each
chunk is a separate URL, so N concurrent chunk fetches are N TCP flows without
any range-request machinery. That matters on vast, which shapes per-flow, so
throughput is flows x per-flow-rate.

Verifies every chunk's sha1 against the manifest before it counts as done.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import os
import sys
import threading
import time
import urllib.error
import urllib.request

# When the origin rate-limits the account, every worker backing off together is
# the point; staggering them just re-storms it.
_throttle_lock = threading.Lock()

CHUNK_READ = 4 << 20

# Cloudflare's Managed Free Ruleset 403s the default "Python-urllib/x.y" agent
# (measured: same URL is 200 under curl/8.0 or any other token). Every request
# here must carry a real one or the whole lane fails closed at the edge.
USER_AGENT = "b2x-cdn/1"


def _get(url: str, timeout: int):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    return urllib.request.urlopen(req, timeout=timeout)


def fetch_one(url: str, dest: str, offset: int, length: int,
              want_sha1: str | None, retries: int = 8) -> tuple[int, str | None]:
    """Fetch one chunk into dest at offset. Returns (bytes, error-or-None)."""
    for attempt in range(retries):
        try:
            h = hashlib.sha1()
            got = 0
            with _get(url, 120) as r, \
                    open(dest, "r+b") as fh:
                fh.seek(offset)
                while True:
                    buf = r.read(CHUNK_READ)
                    if not buf:
                        break
                    fh.write(buf)
                    h.update(buf)
                    got += len(buf)
            if got != length:
                raise IOError(f"short read {got} != {length}")
            if want_sha1 and h.hexdigest() != want_sha1:
                raise IOError("sha1 mismatch")
            return got, None
        except urllib.error.HTTPError as e:
            # A COLD edge passes the miss through to B2, so this path inherits
            # B2's ACCOUNT-level rate limit -- measured: 7 chunks lost to 429
            # on a cold cache at concurrency 32, where the old ladder (4 tries,
            # ~4.75 s total) could not outlast it. Warm hits never see this.
            if attempt == retries - 1:
                return 0, f"{url.rsplit('/', 1)[-1]}: {e}"
            if e.code in (429, 503):
                wait = e.headers.get("Retry-After")
                try:
                    delay = float(wait) if wait else min(2 ** attempt, 30)
                except ValueError:
                    delay = min(2 ** attempt, 30)
                with _throttle_lock:
                    time.sleep(delay)
            else:
                time.sleep(1.5 ** attempt)
        except Exception as e:  # noqa: BLE001 - retry any transport failure
            if attempt == retries - 1:
                return 0, f"{url.rsplit('/', 1)[-1]}: {e}"
            time.sleep(1.5 ** attempt)
    return 0, "unreachable"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dest", required=True, help="local directory to fill")
    ap.add_argument("--host", default=os.environ.get("B2_CDN_HOST"))
    ap.add_argument("--bucket", default=os.environ.get("B2_CDN_BUCKET"))
    ap.add_argument("--prefix", default=os.environ.get("B2_CDN_PREFIX"))
    ap.add_argument("--src-prefix", default="base-models")
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--only", help="fetch just this file (substring match)")
    ap.add_argument("--stats-env",
                    help="write CDN_BYTES/CDN_SECS/CDN_MBPS/CDN_FAILED here. "
                         "The caller CANNOT derive bytes from du: we "
                         "preallocate every destination, so du reports full "
                         "size even when chunks never arrived")
    a = ap.parse_args()

    for name, val in (("--host", a.host), ("--bucket", a.bucket),
                      ("--prefix", a.prefix)):
        if not val:
            raise SystemExit(f"{name} is required (or set its env var)")

    import json
    base = f"https://{a.host}/file/{a.bucket}/{a.prefix}/{a.src_prefix}/{a.model}"
    with _get(f"{base}/CDN_MANIFEST.json", 60) as r:
        manifest = json.load(r)

    os.makedirs(a.dest, exist_ok=True)
    jobs, total = [], 0
    for f in manifest["files"]:
        if a.only and a.only not in f["path"]:
            continue
        dest = os.path.join(a.dest, f["path"])
        os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
        # Preallocate so every chunk can be written at its own offset.
        with open(dest, "wb") as fh:
            fh.truncate(f["size"])
        for p in f["parts"]:
            jobs.append((f"{base}/{p['name']}", dest, p["offset"],
                         p["length"], p.get("sha1")))
            total += p["length"]

    if not jobs:
        raise SystemExit("nothing to fetch (check --only)")

    print(f"{len(jobs)} chunks / {total / 2**30:.2f} GB "
          f"at concurrency {a.concurrency}")
    t0 = time.time()
    errs = []
    done = 0
    with cf.ThreadPoolExecutor(max_workers=a.concurrency) as ex:
        futs = [ex.submit(fetch_one, *j) for j in jobs]
        for fut in cf.as_completed(futs):
            got, err = fut.result()
            if err:
                errs.append(err)
            else:
                done += got
                pct = 100 * done / total
                el = time.time() - t0
                print(f"\r  {pct:5.1f}%  {done / 2**30:6.2f} GB  "
                      f"{done / 2**20 / max(el, 1e-9):7.1f} MB/s", end="")
    el = time.time() - t0
    print(f"\n{done / 2**30:.2f} GB in {el:.1f}s = "
          f"{done / 2**20 / max(el, 1e-9):.1f} MB/s")
    if a.stats_env:
        with open(a.stats_env, "w") as fh:
            fh.write(f"CDN_BYTES={done}\n")
            fh.write(f"CDN_SECS={el:.3f}\n")
            fh.write(f"CDN_MBPS={done / 1e6 / max(el, 1e-9):.2f}\n")
            fh.write(f"CDN_CHUNKS={len(jobs)}\n")
            fh.write(f"CDN_FAILED={len(errs)}\n")
    if errs:
        print(f"\n{len(errs)} chunk(s) FAILED:", file=sys.stderr)
        for e in errs[:10]:
            print(f"  {e}", file=sys.stderr)
        return 1
    print("all chunks verified against manifest sha1")
    return 0


if __name__ == "__main__":
    sys.exit(main())

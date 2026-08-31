#!/usr/bin/env python3
"""Mint a pull token for the R2 Worker registry (registry.example.com).

Token format — must match the Worker's verify exactly
(R2 Worker registry contract):

    payload   compact JSON, this key order, no spaces:
                  {"repo":"<repo>","exp":<unix int>,"instance":"<id>"}
    token     b64url(payload) "." b64url(HMAC-SHA256(secret, b64url(payload)))

b64url is UNPADDED; the HMAC is computed over the ASCII bytes of the b64url'd
payload, with the UTF-8 bytes of the secret as key. Docker Basic auth: username
is the literal "vast", the token is the password.

The signing secret is REGISTRY_AUTH_SECRET. It lives in `.env`
(never committed) and this tool deliberately does NOT parse that file — export
the variable before invoking:

    export $(grep '^REGISTRY_AUTH_SECRET=' .env)
    python3 tools/vast/registry/mint_token.py --repo train

Prints the token to stdout, nothing else.
"""
import argparse
import base64
import hashlib
import hmac
import json
import os
import sys
import time


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _b64url_decode(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def mint(secret, repo, ttl_hours=6, instance="publisher", now=None):
    """Return a registry token for `repo`, expiring `ttl_hours` from `now`
    (default: current time). `instance` is informational (vast id, "publisher",
    "herdd"); the Worker checks only repo and exp."""
    if now is None:
        now = time.time()
    exp = int(now + ttl_hours * 3600)
    payload = json.dumps({"repo": repo, "exp": exp, "instance": instance},
                         separators=(",", ":"))
    p64 = _b64url(payload.encode("utf-8"))
    sig = hmac.new(secret.encode("utf-8"), p64.encode("ascii"),
                   hashlib.sha256).digest()
    return f"{p64}.{_b64url(sig)}"


def verify(secret, token, repo, now=None):
    """The Worker's verify, mirrored in Python so both sides of the contract
    are testable here. Returns (ok, reason); reason is "ok" on success, else
    one of malformed / bad-hmac / bad-payload / expired / repo-mismatch.
    HMAC first (constant-time), then exp <= now, then repo (token repo "*"
    matches any requested repo)."""
    if now is None:
        now = time.time()
    parts = token.split(".")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        return False, "malformed"
    p64, s64 = parts
    want = _b64url(hmac.new(secret.encode("utf-8"), p64.encode("ascii"),
                            hashlib.sha256).digest())
    if not hmac.compare_digest(want, s64):
        return False, "bad-hmac"
    try:
        payload = json.loads(_b64url_decode(p64))
        exp = int(payload["exp"])
        tok_repo = payload["repo"]
    except (ValueError, KeyError, TypeError):
        return False, "bad-payload"
    if exp <= now:
        return False, "expired"
    if tok_repo != "*" and tok_repo != repo:
        return False, "repo-mismatch"
    return True, "ok"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", required=True,
                    help='repo the token grants ("*" matches any)')
    ap.add_argument("--ttl-hours", type=float, default=6)
    ap.add_argument("--instance", default="publisher")
    ap.add_argument("--secret-env", default="REGISTRY_AUTH_SECRET",
                    help="env var holding the signing secret")
    a = ap.parse_args(argv)
    secret = os.environ.get(a.secret_env)
    if not secret:
        sys.exit(f"error: {a.secret_env} not set. It lives in .env "
                 f"(never committed); export it first, e.g.:\n"
                 f"  export $(grep '^{a.secret_env}=' .env)")
    print(mint(secret, a.repo, ttl_hours=a.ttl_hours, instance=a.instance))


if __name__ == "__main__":
    main()

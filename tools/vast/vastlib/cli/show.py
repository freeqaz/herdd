"""`herdd show <id>` — dump one instance's raw API record as JSON.

The rawest read in the CLI: one GET, `json.dumps`, no formatting opinion. It
exists so a human or an agent can see fields the rendered `ls` deliberately
drops, which is also why it prints `d.get("instances", d)` — the endpoint's
envelope varies and the fallback keeps the unwrapped shape readable.

SECURITY — secret VALUES are redacted by default (2026-08-28)
-------------------------------------------------------------
The instance record carries whatever was passed at launch, and that includes
the container env: `B2_*` application keys, `HF_TOKEN`, `TS_AUTHKEY`, the
minted registry `image_login` string, and any `KEY=VALUE` a launch put in
`extra_env` or inlined into the `onstart` script.

Every secret VALUE is replaced with a redaction marker before printing; the
KEY survives, so the record still answers "was this env var set?". Pass
`--reveal-secrets` for the old raw behaviour when you genuinely need a value
(per-box serve-bearer extraction is the real case).

Why the default flipped. It used to print them, on the reasoning that a
redacting `show` would be "a third view, not a raw one" — but nothing consumes
this output programmatically, while THREE downstream systems exist purely to
clean up after it (`data_supply/claude_projects/scrub.py` and
`swap_trajectory/audit.py` both name it as a known-hostile command, and
`retention_report.sh` warns about it). That is a lot of compensating machinery
for one `json.dumps`, and it fails open: a near-miss on 2026-08-09 left a
`show` dump holding live B2 keys inside a git worktree for ~2 h. The escape
hatch is preserved exactly — it just has to be asked for.

Detection reuses `launch.spec._is_secret_env` — the same predicate that keeps
launch secrets out of the durable B2 spec and out of `launch --dry-run` stdout
— widened by `_is_secret` below for names that appear only in the API record
(`image_login`). Redaction is deliberately over-broad (any
`TOKEN|KEY|SECRET|PASS|LOGIN|…`-shaped name, any `scheme://user:pass@host`
value): over-redacting costs a flag, under-redacting costs a credential.

A TEST for this command must never run it against real credentials. The tests
feed a fixture record through a patched `core.api.request` and assert on that;
nothing in the suite may capture the real environment's output.

What is deliberately NOT here
-----------------------------
* Field selection or pretty-printing. `show` is the escape hatch you reach for
  precisely when the curated views have dropped what you need. `ls --json` and
  `box --json` are the curated machine-readable reads.

Provenance: moved from `tools/vast/herdd.py` (`cmd_show`, parser block in
`main()`), plan §8 step 6, 2026-08-16, behavior-preserving. Redaction added
2026-08-28 (a deliberate CLI-surface change; fixture amended).
"""

from __future__ import annotations

import argparse
import json
import re
import sys

from vastlib.cli import _args, _docs
from vastlib.core import api
from vastlib.launch import spec

#: What a withheld value is replaced with. Names the flag that returns it, so
#: a reader is never left guessing why a field looks empty.
REDACTED = "[REDACTED — herdd show --reveal-secrets]"

#: `NAME=VALUE` inside free text: `extra_env` entries and the `export FOO=bar`
#: lines of an onstart script both carry secrets this way, and neither is a
#: dict the key-walk can see.
_ASSIGN_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=(\S+)")

#: Names `spec._SECRET_ENV_RE` does not carry, because it governs env vars WE
#: pass at launch while this walks the API's own record. `image_login` is the
#: worked example: the minted `robot:<password>` registry string, which launch
#: never sees as an env var (it stores an `image_login_ref` instead), so the
#: shared predicate has no reason to know the name — but the API record hands
#: back the real value. Layered here rather than widened there: that predicate
#: decides what lands in the durable, never-deleted B2 spec, and this fix has
#: no business moving that boundary.
#: `NONCE` is here because `BOX_IDENTITY_NONCE` is what a box proves itself
#: with to the credential broker — an impersonation credential whose name says
#: nothing about that.
_EXTRA_SECRET_NAME_RE = re.compile(r"LOGIN|BEARER|NONCE", re.I)


def _is_secret(k: str | None, v: str | None) -> bool:
    """`spec._is_secret_env` widened by the record-only names above.

    `spec` is reached by MODULE ATTRIBUTE, never `from … import`: that module
    documents the reason, and it binds here for the same one — a stale binding
    must fail loudly rather than fail open and print the credential.
    """
    return bool(spec._is_secret_env(k, v)
                or _EXTRA_SECRET_NAME_RE.search(k or ""))


def _redact_text(text: str, hits: list[int]) -> str:
    """Redact `NAME=VALUE` assignments (and URL credentials) inside a string."""
    def sub(m: "re.Match[str]") -> str:
        if _is_secret(m.group(1), m.group(2)):
            hits.append(1)
            return f"{m.group(1)}={REDACTED}"
        return m.group(0)

    out = _ASSIGN_RE.sub(sub, text)
    # A bare credential-shaped value carrying no NAME= (scheme://user:pass@host).
    if _is_secret(None, out):
        hits.append(1)
        return REDACTED
    return out


def redact(obj: object, hits: list[int]) -> object:
    """Recursively withhold secret values from an API record.

    Fails CLOSED: a secret-shaped KEY withholds its whole value whatever the
    value's type, because a nested container under `image_auth`/`ssh_key` is
    exactly where an unmodelled credential would hide.
    """
    if isinstance(obj, dict):
        out: dict[object, object] = {}
        for k, v in obj.items():
            if isinstance(k, str) and _is_secret(
                    k, v if isinstance(v, str) else ""):
                hits.append(1)
                out[k] = REDACTED
            else:
                out[k] = redact(v, hits)
        return out
    if isinstance(obj, list):
        # `extra_env` comes back as [NAME, VALUE] PAIRS, not "NAME=VALUE"
        # strings: the name and its value are SEPARATE elements, so neither
        # the dict key-walk nor the text rule above can see the pairing.
        # Measured on a live record before this branch existed — 16 of 31
        # pairs were credential-bearing (TS_AUTHKEY, HF_TOKEN, six B2 key
        # pairs, R2_TC_SECRET_ACCESS_KEY, B2_CDN_PREFIX) and every one of
        # them printed in full.
        if (len(obj) == 2 and isinstance(obj[0], str)
                and isinstance(obj[1], str) and _is_secret(obj[0], obj[1])):
            hits.append(1)
            return [obj[0], REDACTED]
        return [redact(x, hits) for x in obj]
    if isinstance(obj, str):
        return _redact_text(obj, hits)
    return obj


# moved-from: herdd.cmd_show
def run(a: argparse.Namespace) -> None:
    d = api.request("GET", f"v0/instances/{a.id}/")
    rec = d.get("instances", d)
    if getattr(a, "reveal_secrets", False):
        print("!! herdd show --reveal-secrets: LIVE CREDENTIALS on stdout — "
              "do not redirect, paste or log this.", file=sys.stderr)
    else:
        hits: list[int] = []
        rec = redact(rec, hits)
        if hits:
            # stderr, so stdout stays valid JSON for `| jq`.
            print(f"** {len(hits)} secret value(s) redacted "
                  "(--reveal-secrets to print them)", file=sys.stderr)
    print(json.dumps(rec, indent=2))


def add_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser],
               add_cmd: _args.AddCmd) -> argparse.ArgumentParser:
    psh = add_cmd(sub, "show", "show one instance (raw json)", _docs.DOC_README)
    psh.add_argument("id", type=int)
    psh.add_argument("--reveal-secrets", action="store_true",
                     help="print secret env values instead of redacting them")
    psh.set_defaults(func=run)
    return psh

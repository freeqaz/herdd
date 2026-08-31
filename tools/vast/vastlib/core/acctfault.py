"""vastlib.core.acctfault — was this failure OUR ACCOUNT's fault, or a HOST's?

Why this exists
---------------
Two layers in this tree draw permanent conclusions from a failed rental:
`market.hostrep` writes a decaying, cross-session STRIKE against the machine,
and fleetd's derived alarms tell the operator which remedy to reach for. Both
assume the failure says something about the host. An account-level refusal says
nothing about any host — it is true of every offer in the market simultaneously
— so charging it to whichever machine happened to be under the cursor is a
durable, silent distortion of every later offer ranking, and telling the
operator to "raise the ceiling" is advice that cannot work at any price.

Measured 2026-08-25: the vast balance hit $0.000/−$1.504 and the replacement
ladder emitted 76 `insufficient_credit` refusals over 57 minutes. Every alarm it
raised named three remedies (`job retarget`, raise the budget/cap, `fleet
destroy`) and the actual remedy — top up the account — was not among them.

What counts, and what deliberately does not
-------------------------------------------
The predicate is "the host did nothing wrong", not the single string
`insufficient_credit`. So an expired key, a revoked key, a suspended account and
a missing key all classify here: none of them is evidence about a machine, and
none of them clears by retrying somewhere else.

What must NOT classify here is anything a DIFFERENT host could have satisfied.
A stalled image pull, `Required resources are currently unavailable`, an empty
market, a price over the ceiling — those are host or market facts and the
reputation ledger is entitled to them. Over-broadening this predicate does not
fail loudly; it just stops the ledger learning, so the bar is that the condition
be true of the whole market at once.

429 is excluded on purpose. It is rate limiting, `core.api` already classifies
it TRANSIENT and retries it, and a burst of our own polling is not a statement
about the account's ability to rent.

The latch
---------
`classify` is pure and is the whole rule. `note`/`recent` add the one piece of
state that makes it useful at a seam that never sees the API error: an eviction
condemns a machine from an instance listing, not from a failed PUT, so at the
moment the strike is written the credit outage is invisible. A process-local,
time-bounded latch closes that gap — fleetd is long-lived, so "we were refused
for credit 90 seconds ago" is exactly the context the strike seam lacks.

It is deliberately one-directional and lossy: a live latch can only SUPPRESS a
strike, never create one. During a real credit outage a genuinely bad host may
escape a strike it earned; it will earn another one the next time it is rented,
because recurrence is what that ledger scores. The reverse error — a permanent
mark against an innocent host, invisible until someone reads the file — has no
such self-correction.
"""

from __future__ import annotations

import re
import time
from typing import Any

#: The refusal-reason code the replacement ladder stores when a launch failed
#: because of the account. Lives here, not beside the ladder, so `fleet.daemon`
#: can branch its alarm text on it without importing the supervise ring.
REASON = "account_blocked"

#: How long an observed account fault keeps suppressing host strikes. A credit
#: outage is not an instant: vast stops boxes over minutes, and the evictions it
#: produces arrive after the refusal that named the cause. Wide enough to cover
#: that spread, short enough that a topped-up account is trusted again on the
#: next watch tick rather than the next day.
WINDOW_S = 900.0

#: `classify` codes -> what the operator should read. One phrase per code; the
#: REMEDIES table below is the actionable half and is kept separate because
#: alarm text and journal fields want different lengths.
LABELS: dict[str, str] = {
    "insufficient_credit": "insufficient credit",
    "auth": "the API key was rejected",
    "permission": "the account is not permitted to rent",
    "no_api_key": "no API key is configured",
    # The code-only fallback: a stored refusal reason proves the class but
    # carries no text to name the cause more precisely.
    REASON: "the vast.ai API refused this account",
}

REMEDIES: dict[str, str] = {
    "insufficient_credit":
        "top up the vast.ai balance (console.vast.ai/billing)",
    "auth":
        "mint a new key and set VASTAI_API_KEY (console.vast.ai/account)",
    "permission":
        "check the account standing at console.vast.ai/account",
    "no_api_key":
        "set VASTAI_API_KEY in the environment or .env",
    REASON:
        "check the balance and the key with `herdd whoami`",
}

#: Substring probes, lowercased, checked in order. Each entry is a token we have
#: actually seen or that vast documents; a bare HTTP status is handled below and
#: only for the two codes that mean "who you are", never for 400/404.
_TOKENS: tuple[tuple[str, str], ...] = (
    ("insufficient_credit", "insufficient_credit"),
    ("insufficient credit", "insufficient_credit"),
    ("lacks credit", "insufficient_credit"),
    ("lack credit", "insufficient_credit"),
    ("billing page", "insufficient_credit"),
    ("negative balance", "insufficient_credit"),
    ("out of credit", "insufficient_credit"),
    ("invalid_api_key", "auth"),
    ("invalid api key", "auth"),
    ("bad api key", "auth"),
    ("api key is invalid", "auth"),
    ("expired api key", "auth"),
    ("unauthorized", "auth"),
    ("authentication failed", "auth"),
    ("account suspended", "permission"),
    ("account is suspended", "permission"),
    ("account disabled", "permission"),
    ("account_disabled", "permission"),
    ("account is banned", "permission"),
    ("not_authorized", "permission"),
    ("permission denied", "permission"),
)

#: The missing-key shape `core.api._api_key_soft` emits. Its own prefix already
#: routes it FATAL; this names it as ours rather than the market's.
_NO_KEY = "vastai_api_key not set"

_latch: dict[str, Any] = {"code": None, "ts": 0.0, "detail": None}


def classify(err: object) -> str | None:
    """PURE. An account/operator-caused failure code, or None.

    Accepts anything an error channel in this tree carries: the `"HTTP 400 on
    PUT …"` string `core.api.request_soft` returns, the `SystemExit` its raising
    twin throws, a bare exception, or None. Unrecognised is None — the safe
    direction, because None leaves every existing behaviour exactly as it was.
    """
    if err is None:
        return None
    text = str(err).strip().lower()
    if not text:
        return None
    if _NO_KEY in text:
        return "no_api_key"
    for token, code in _TOKENS:
        if token in text:
            return code
    # Bare status, for the two codes that are statements about the CALLER. 400
    # and 404 are deliberately absent: vast returns 400 for a stale ask id and
    # 404 for a deleted instance, neither of which is an account condition.
    m = re.search(r"http (\d{3})", text)
    if m:
        status = int(m.group(1))
        if status == 401:
            return "auth"
        if status == 403:
            return "permission"
    return None


def describe(code: str | None) -> str:
    """One operator-facing sentence for `code`, remedy included."""
    if not code:
        return ""
    what = LABELS.get(code, code)
    how = REMEDIES.get(code)
    return (f"the ACCOUNT cannot rent at any price: {what}"
            + (f" — {how}" if how else "")
            + ". No retry, re-bid, ceiling raise or host change can clear this")


def note(err: object, *, now: float | None = None) -> str | None:
    """Classify `err` and, if it is ours, latch it. Returns the code or None.

    Called from the ONE HTTP funnel, so every lane in the tree contributes and
    none has to remember to. Never raises: a bookkeeping helper on an error path
    that can itself fail is a second failure at the worst moment.
    """
    try:
        code = classify(err)
    except Exception:
        return None
    if code is None:
        return None
    try:
        _latch.update({"code": code, "ts": time.time() if now is None
                       else float(now), "detail": str(err)[:300]})
    except Exception:
        return code
    return code


def recent(*, now: float | None = None,
           window_s: float | None = None) -> dict[str, Any] | None:
    """The latched account fault if it is still inside the window, else None.

    `{code, ts, detail, age_s}`. Read by every seam that would otherwise blame a
    host for a condition the account caused.
    """
    code = _latch.get("code")
    if not code:
        return None
    now = time.time() if now is None else float(now)
    win = WINDOW_S if window_s is None else float(window_s)
    try:
        age = now - float(_latch.get("ts") or 0.0)
    except (TypeError, ValueError):
        return None
    if age < 0 or age > win:
        return None
    return {"code": str(code), "ts": float(_latch["ts"]),
            "detail": _latch.get("detail"), "age_s": age}


def clear() -> None:
    """Forget the latch. The operator verb after a top-up, and what the test
    suite calls between tests so one credit fixture cannot silence another."""
    _latch.update({"code": None, "ts": 0.0, "detail": None})

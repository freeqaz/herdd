"""vastlib.storage.dashcache — the /admin snapshot writer, read-only by construction.

Why this module exists
----------------------
`infra-metadata.db` is how an INTERNET-FACING dashboard shows the fleet without
the browser ever touching the vast API, the fleetd socket, or a credential: this
code reads, projects onto a positive column allowlist, and writes a sqlite
snapshot the Next.js side opens `-readonly` through the `sqlite3` CLI. Three
rules bind every line below and they are the reason this is its own module
rather than a section of a command file:

1. **No mutation, ever.** No destroy/park/bid/launch/reap verb is reachable from
   here and nothing on this path takes a `-y`. The rule is enforced as a
   source-text test, not a review note (see "The purity test" below).
2. **A positive allowlist at the write.** Every INSERT names its columns and
   every value is named explicitly off the record — no dict is splatted. That is
   what keeps `extra_env` (which carries `HF_TOKEN` and the B2 keys verbatim),
   `image_login`, `onstart`, `ssh_host`, `ssh_port`, `public_ipaddr`,
   `hostname`, `host_id`, `requester` and `last_tail` out of sqlite: there is no
   code path that names them.
3. **Exit 0 or 1 ONLY.** The `reap --json` / `guard --json` "exit 2 when there
   are findings" convention must not leak here — the Node caller treats any
   nonzero as total failure. A section that raises is caught and SKIPPED with
   its old rows intact; exit 1 is reserved for "cannot open the DB".

stdout stays empty by design (summaries go to stderr), structurally, via
`contextlib.redirect_stdout(sys.stderr)` around the whole section loop — a
`print()` added later to any transitive callee would otherwise land on the exact
channel an `execFile` caller reads.

Frozen contracts this file carries (plan §4)
--------------------------------------------
* **The `dash-cache` argv.** `dashboard/lib/vast-admin.ts` freezes
  `['tools/vast/herdd.py', 'dash-cache', '--sections']` and spawns it with a
  comma list; two more literal copies live in the admin UI components. The flag
  names below and `DASH_SECTIONS` are half of that contract, and the entry-script
  path is the other half — which is why the CLI wiring stays in `cli/` (plan §8
  step 6) and `tools/vast/herdd.py` keeps its exact path forever.
* **`_INFRA_CACHE_SCHEMA`'s column names**, read by the dashboard's field
  mappers through the sqlite3 CLI. A cross-repo contract: renaming a column here
  blanks a panel there, silently.
* **The journal mode stays the rollback default.** The reader opens the file
  read-only and a read-only connection cannot open a WAL database, so every
  page — the runs view included — breaks silently if that is ever flipped.
* **`DASH_GPUS_DEFAULT`** must stay in sync with `GPU_SPECS` in
  `dashboard/lib/gpu-throughput.ts`. A class probed here with no spec row there
  renders no throughput, and vice versa; a wrong `gpu_name` string returns an
  EMPTY market rather than an error.

The purity test
---------------
`test_dash_cache.py` carries a source-text test that tokenizes the dash-cache
block and asserts it names no mutating helper. `test_vastlib_storage.py` carries
its twin against THIS file's source, with the same banned token list, and that
is why this module has two hard import rules:

* **it must not import `subprocess`**, and
* **it must not import `vastlib.storage.b2`** — the two storage modules are
  deliberately non-adjacent. Nothing on the dashboard path shells out; the only
  I/O here is sqlite plus `core.api.request_soft` (GET/POST reads).

The seams that point sideways or UP (`DashDeps`)
------------------------------------------------
Three of the four sections read things `storage/` does not own, and one of those
directions is permanently illegal:

* `_gather_ls_data`, `_job_cell` and `_ACTIVE_JOB_STATES` are `cli/ls`
  territory, and `cli` is ABOVE `storage` in the plan §5 DAG. `storage` may
  never import it, so those three arrive as injections **permanently**.
* the fleet section writer and the market offer-query builder are DEFERRED to
  plan step 5 by integrator ruling (`_dash_write_fleet` needs `fleet.client`,
  which also sits above `storage`; `_dash_offer_query` needs
  `market.offers.build_search_query`). Both are `None`-able hooks here.
* `_is_secret_env` / `_SECRET_VAL_RE` (`launch.spec`) and
  `REAP_IDLE_H_DEFAULT` (`boxes.reap`) are same-ring modules, so importing them
  would be legal — and is deliberately not done. This module's import closure is
  `core` ONLY, which is what makes rule 1 structural rather than textual:
  `boxes.reap` would drag `cmd_reap` and, through `boxes.lifecycle`,
  `destroy_box` / `_destroy_and_revoke` into the same process graph as the
  internet-facing snapshot writer, and `launch.spec` would drag the credential
  MINT. Two regexes and a float are not worth that. (`boxes.reap` says the same
  thing from its side: "storage.dashcache re-derives this policy … a SECOND
  reader of REAP_IDLE_H_DEFAULT".) If they are ever collapsed into imports, the
  argument to re-check is the closure, not the arity.

An uninjected `cmd_dash_cache(a)` therefore still runs the `account` section and
reports the other three as SKIPPED on stderr — which is the section-failure path
that already existed, with the exit-code contract intact. The composition root
(plan §8 steps 5-6) builds one `DashDeps` and passes it in.

What is deliberately NOT here
-----------------------------
* **`_dash_verified`.** It already lives in `core.models` (ported at step 2;
  `test_vastlib_core_models.py` pinned it against the `herdd` copy until step
  6d deleted that copy, and now pins the re-export binding). This module
  CONSUMES it — `models._dash_verified(o)` — and must never grow a second copy:
  the whole point of that function is that the bundles RESPONSE spells the field
  `verification: "verified"` while the SEARCH FILTER key is `verified`, and two
  copies of that trap is one copy too many.
* **`_ANSI_RE`.** Same story: `core.fmt` owns it (`fmt._ANSI_RE`), no local
  recompile.
* **`_dash_write_fleet` and `_dash_offer_query`** — deferred to plan step 5, as
  above. They are not stubbed here either; a missing hook raises and the section
  is skipped.
* **Any reaper invocation.** `_dash_reap_threshold_s` READS the reaper's
  deadline and `_dash_instance_rows` recomputes the idle verdict in-process from
  the same ledger `ls` and `reap` share. Shelling out to the reaper would put a
  destroy verb on the dashboard's path (rule 1).
* **`_infra_cache_write`'s caller.** The runs-snapshot writer lives with
  `cmd_runs` (the runs fold, `cli/`), but the DDL is shared with the dash-cache
  path, so the schema and the five `_infra_cache_*` symbols live here — in
  storage, where the file lives — rather than in the command that happens to
  fill one of its nine tables.

Provenance: moved verbatim from `tools/vast/herdd.py` (plan §8 step 3,
2026-08-16), rev 2b188979. Behavior-preserving: bodies copied, annotations
added, plus the documented mechanical changes — `TOOLS_VAST_DIR` in place of
`os.path.dirname(os.path.abspath(__file__))` (the file moved two directories
deeper, and `*.db` is gitignored, so a wrong default would fail SILENTLY as an
empty database the dashboard reads forever), the `DashDeps` injections above,
and `typing.cast` on the one `getattr` result whose type the checker cannot see.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import datetime
import math
import os
import re
import sqlite3
import sys
import threading
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Callable, cast

from vastlib.core import api, config, fmt, labels, models

__all__ = [
    "DASH_DISK_OVERSIZED_FRAC",
    "DASH_DISK_OVERSIZED_GB",
    "DASH_GPUS_DEFAULT",
    "DASH_MARKET_MAX_RPS",
    "DASH_NUM_GPUS_DEFAULT",
    "DASH_OFFERS_KEPT",
    "DASH_OFFER_LIMIT",
    "DASH_SECTIONS",
    "DASH_STATUS_MAX",
    "DASH_TICK_STALE_S",
    "DashDeps",
    "TOOLS_VAST_DIR",
    "cmd_dash_cache",
]

# `tools/vast/` — where `infra-metadata.db` has always lived and where the
# dashboard's `sqlite3 -readonly` reader looks for it. In the flat module this
# was `os.path.dirname(os.path.abspath(__file__))`; this file sits two
# directories deeper. Computed exactly the way `core.config._HERE` is, and
# pinned by `test_vastlib_storage.py` — a wrong default fails SILENTLY (`*.db`
# is gitignored, sqlite happily CREATES the missing file, and every /admin page
# then reads a permanently empty database).
TOOLS_VAST_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass(frozen=True, slots=True)
class DashDeps:
    """The reads this module cannot import — see the module docstring.

    PERMANENT (upward edges; `cli` and `fleet` both sit above `storage` in the
    plan §5 DAG, so these can never become imports):
      `gather_ls_data` / `job_cell` / `active_job_states`  — `cli/ls`
      `write_fleet`                                        — `fleet/client`

    DEFERRED to plan step 5 by integrator ruling:
      `write_fleet`   (`herdd._dash_write_fleet`, the only section returning a
                       2-tuple `(rows, err)`; the fleetd ops it may use are
                       frozen READ-ONLY: `status` and `spend`)
      `offer_query`   (`herdd._dash_offer_query` over
                       `market.offers.build_search_query`)

    SIDEWAYS, and injected ON PURPOSE (importing them is legal and would widen
    this module's import closure from `core`-only to one containing `cmd_reap`,
    `destroy_box` and the credential mint — see the module docstring):
      `is_secret_env` / `secret_val_re`   — `launch.spec`
      `reap_idle_h_default`               — `boxes.reap.REAP_IDLE_H_DEFAULT`
    """

    #: `launch.spec._is_secret_env(k, v)` — name-family OR credential-shaped value.
    is_secret_env: Callable[[str, str], bool]
    #: `launch.spec._SECRET_VAL_RE` — the `scheme://user:pass@host` matcher.
    secret_val_re: re.Pattern[str]
    #: `boxes.reap.REAP_IDLE_H_DEFAULT` — hours, the 2h owner policy.
    reap_idle_h_default: float
    #: `cli/ls._gather_ls_data(no_spot=...)` — one fleet read, already folded.
    gather_ls_data: Callable[..., Mapping[str, Any]]
    #: `cli/ls._job_cell(v)` — the per-job display cell.
    job_cell: Callable[[Mapping[str, Any]], str]
    #: `cli/ls._ACTIVE_JOB_STATES` — ("running", "claimed", "submitted").
    active_job_states: Sequence[str]
    #: DEFERRED (step 5): `fleet` section writer, `(rows, err)`.
    write_fleet: Callable[[sqlite3.Connection], tuple[int, str | None]] | None = None
    #: DEFERRED (step 5): one market probe's search-query body.
    offer_query: Callable[[str, int, str], Mapping[str, Any]] | None = None
    #: The class-census search body (all GPUs, one page) — `_dash_discover_gpus`.
    #: Optional, and its absence is a SKIP rather than an error: discovery is a
    #: widening of the built-in probe set, so an unwired census degrades to the
    #: old static behaviour instead of failing the market section.
    census_query: Callable[[], Mapping[str, Any]] | None = None


def _need(deps: DashDeps | None) -> DashDeps:
    """The injected reads, or a section-fatal error naming why they are missing.

    Raised INSIDE `cmd_dash_cache`'s per-section `try`, so an uninjected run
    reports the section as SKIPPED on stderr and still exits 0 — the same shape
    as any other failing section, and the exit-code contract is preserved.
    """
    if deps is None:
        raise RuntimeError(
            "dash-cache: no DashDeps injected (the ls/fleet/market reads live "
            "above or beside storage/ and are wired by the composition root)")
    return deps


def _need_hook(hook: Callable[..., Any] | None, name: str) -> Callable[..., Any]:
    """One DEFERRED hook, or a section-fatal error. Same skip semantics as above."""
    if hook is None:
        raise RuntimeError(f"dash-cache: DashDeps.{name} is not wired yet "
                           f"(deferred to plan §8 step 5)")
    return hook


# -- infra-metadata.db cache (dashboard SWR store) ----------------------------
# Written here (Python stdlib sqlite3), read by the Next.js dashboard via the
# sqlite3 CLI. Default rollback journal + busy_timeout (NOT WAL) so the dashboard
# read-only reader can open it -- a read-only connection cannot open a WAL db.
# moved-from: herdd._INFRA_CACHE_SCHEMA
_INFRA_CACHE_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs(
  run TEXT PRIMARY KEY, status TEXT, terminal INTEGER, gpu TEXT, dph REAL,
  latest_step INTEGER, cost_usd REAL, relaunch_count INTEGER, instance_id TEXT,
  live INTEGER, n_events INTEGER, parse_errors INTEGER, supervised TEXT,
  farm TEXT, started_at TEXT, ended_at TEXT, last_event_ts TEXT,
  cost_source TEXT);
CREATE TABLE IF NOT EXISTS meta(
  key TEXT PRIMARY KEY, fetched_at TEXT, fetched_ts REAL);

CREATE TABLE IF NOT EXISTS instances(
  iid INTEGER PRIMARY KEY, machine_id INTEGER,
  state TEXT, status TEXT, mode TEXT,
  num_gpus INTEGER, gpu TEXT, gpu_util REAL,
  hourly REAL, storage_day REAL, ondemand REAL, spot REAL, avail INTEGER,
  disk_gb REAL, disk_used_gb REAL, disk_frac REAL, disk_oversized INTEGER,
  idle_s REAL, start_ts REAL, age_s REAL,
  label TEXT, keep INTEGER, run_id TEXT, geo TEXT,
  image_short TEXT, stale_image INTEGER,
  health_verdict TEXT, health_reason TEXT,
  reap_verdict TEXT, reap_wait_s REAL,
  n_jobs INTEGER, jobs TEXT);

CREATE TABLE IF NOT EXISTS market(
  gpu_name TEXT NOT NULL, num_gpus INTEGER NOT NULL, kind TEXT NOT NULL,
  n_offers INTEGER, price_field TEXT,
  p0 REAL, p10 REAL, p50 REAL, p90 REAL,
  best_total REAL, best_per_gpu REAL, best_machine_id INTEGER, best_geo TEXT,
  best_cuda REAL, best_inet_down REAL, best_reliability REAL,
  n_ok INTEGER, p0_ok REAL, p10_ok REAL, p50_ok REAL, p90_ok REAL,
  best_ok_total REAL, best_ok_per_gpu REAL, best_ok_machine_id INTEGER,
  best_ok_geo TEXT,
  ok_cuda_min REAL, ok_inet_down_min REAL, ok_reliability_min REAL,
  PRIMARY KEY(gpu_name, num_gpus, kind));

CREATE TABLE IF NOT EXISTS market_offers(
  gpu_name TEXT NOT NULL, num_gpus INTEGER NOT NULL, kind TEXT NOT NULL,
  rank INTEGER NOT NULL, machine_id INTEGER, geo TEXT,
  price REAL, price_per_gpu REAL, dph_total REAL, min_bid REAL,
  storage_cost REAL, gpu_ram_gb REAL, cuda_max_good REAL, reliability REAL,
  inet_down REAL, inet_up REAL, disk_space REAL,
  verified INTEGER, rentable INTEGER,
  compute_cap INTEGER, cpu_cores_effective REAL, cpu_ram_gb REAL, dlperf REAL,
  launch_ok INTEGER,
  PRIMARY KEY(gpu_name, num_gpus, kind, rank));

CREATE TABLE IF NOT EXISTS fleet(
  key TEXT PRIMARY KEY, daemon_up INTEGER, api_ok INTEGER, dry_run INTEGER,
  tick_age_s REAL, tick_stale INTEGER, rev TEXT, version INTEGER,
  spend_total_usd REAL, n_watches INTEGER, n_strays INTEGER,
  n_alarms INTEGER, n_sticky_alarms INTEGER);

CREATE TABLE IF NOT EXISTS fleet_watches(
  iid INTEGER PRIMARY KEY, target TEXT, profile TEXT, state TEXT,
  spend_usd REAL, budget_usd REAL, budget_frac REAL,
  paused INTEGER, pause_left_s REAL, pause_reason TEXT,
  dormant INTEGER, adopted INTEGER, stray INTEGER, last_action TEXT);

CREATE TABLE IF NOT EXISTS fleet_alarms(
  key TEXT PRIMARY KEY, iid INTEGER, msg TEXT, sticky INTEGER,
  since_ts REAL, age_s REAL, count INTEGER);

CREATE TABLE IF NOT EXISTS fleet_spend(
  iid INTEGER PRIMARY KEY, spend_usd REAL);

CREATE TABLE IF NOT EXISTS account(
  key TEXT PRIMARY KEY, credit REAL, balance REAL);
"""

# Columns added after the first snapshot shipped, per table. CREATE TABLE IF NOT
# EXISTS is a no-op on an existing DB, so new columns need an explicit guarded
# ALTER or every deployed cache stays one column behind forever — and a SELECT
# naming a missing column throws on the dashboard side, which reads as a blank
# panel rather than an error.
# moved-from: herdd._INFRA_CACHE_ADDED_COLS
_INFRA_CACHE_ADDED_COLS: dict[str, tuple[tuple[str, str], ...]] = {
    "runs": (("cost_source", "TEXT"),),
    # DESIGN_V10_MARKET_SHAPE.md §1.3 — the launch-parity lens over the FULL
    # exact-N sample, plus the predicate values it was built with.
    "market": (
        ("n_ok", "INTEGER"), ("p0_ok", "REAL"), ("p10_ok", "REAL"),
        ("p50_ok", "REAL"), ("p90_ok", "REAL"),
        ("best_ok_total", "REAL"), ("best_ok_per_gpu", "REAL"),
        ("best_ok_machine_id", "INTEGER"), ("best_ok_geo", "TEXT"),
        ("ok_cuda_min", "REAL"), ("ok_inet_down_min", "REAL"),
        ("ok_reliability_min", "REAL"),
    ),
    # the shape fields the "rent by SHAPE" board ranks on, same API response.
    "market_offers": (
        ("compute_cap", "INTEGER"), ("cpu_cores_effective", "REAL"),
        ("cpu_ram_gb", "REAL"), ("dlperf", "REAL"), ("launch_ok", "INTEGER"),
    ),
}


# moved-from: herdd._infra_cache_migrate
def _infra_cache_migrate(conn: sqlite3.Connection) -> None:
    for table, cols in _INFRA_CACHE_ADDED_COLS.items():
        have = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
        if not have:
            continue          # table absent (a partial DB): the DDL creates it
        for col, decl in cols:
            if col not in have:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl}")


# moved-from: herdd._infra_cache_db
def _infra_cache_db(a: object | None = None) -> str:
    """Path to the dashboard's sqlite cache. --cache-db > $INFRA_METADATA_DB >
    tools/vast/infra-metadata.db (next to the entry script; *.db is gitignored)."""
    if a is not None and getattr(a, "cache_db", None):
        # cast: `a` is the argparse Namespace, typed `object` at this seam
        # (plan §5 gives each command a typed Args dataclass at step 6).
        return cast(str, getattr(a, "cache_db"))
    return os.environ.get("INFRA_METADATA_DB") or os.path.join(
        TOOLS_VAST_DIR, "infra-metadata.db")


# moved-from: herdd._infra_cache_write
def _infra_cache_write(rows: Sequence[Mapping[str, Any]], db: str) -> None:
    """Atomically replace the cached runs snapshot. The DELETE+INSERT+meta upsert
    runs in ONE transaction, so a concurrent reader sees either the whole old or
    whole new snapshot, never a partial one; busy_timeout waits out the
    millisecond write rather than erroring 'database is locked'."""
    now = time.time()
    iso = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ")
    conn = sqlite3.connect(db, timeout=5)
    try:
        conn.execute("PRAGMA busy_timeout=3000")
        conn.executescript(_INFRA_CACHE_SCHEMA)
        _infra_cache_migrate(conn)
        with conn:
            conn.execute("DELETE FROM runs")
            conn.executemany(
                "INSERT INTO runs(run,status,terminal,gpu,dph,latest_step,"
                "cost_usd,relaunch_count,instance_id,live,n_events,parse_errors,"
                "supervised,farm,started_at,ended_at,last_event_ts,cost_source) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                [(
                    r["run"], r["status"], 1 if r["terminal"] else 0,
                    r["gpu"], r["dph"], r["latest_step"], r["cost_usd"],
                    r["relaunch_count"],
                    None if r["instance_id"] is None else str(r["instance_id"]),
                    1 if r["live"] else 0, r["n_events"], r["parse_errors"],
                    r["supervised"], r["farm"],
                    r["started_at"], r["ended_at"], r["last_event_ts"],
                    r.get("cost_source"),
                ) for r in rows])
            conn.execute(
                "INSERT OR REPLACE INTO meta(key,fetched_at,fetched_ts) "
                "VALUES('runs',?,?)", (iso, now))
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# dash-cache — the /admin snapshot writer (dashboard/DESIGN_V5_ADMIN.md §3).
# The three rules that bind every line below (no mutation / positive allowlist
# at the write / exit 0 or 1 only) are in this module's docstring, along with
# the frozen contracts. Read them before editing anything in this section.
# --------------------------------------------------------------------------- #
# moved-from: herdd.DASH_SECTIONS
DASH_SECTIONS = ("instances", "market", "fleet", "account")
# Every Blackwell and Hopper class vast actually LISTS, plus the cheap
# prior-generation tail. Owner directive 2026-08-05: the admin market view
# covers the whole Blackwell + H100/H200 field, not the six-card subset.
#
# These strings are the vast `gpu_name` field VERBATIM, verified against the
# live bundles API on 2026-08-05 (counts are verified+rentable offers at
# gpu_ram >= 20 GB, both modes). vast's naming is finicky and unvalidated: a
# wrong string returns an EMPTY market rather than an error, so re-verify with
# a read-only probe before editing this tuple, and keep it in sync with
# `GPU_SPECS` in dashboard/lib/gpu-throughput.ts (a class probed here with no
# spec row there renders no throughput, and vice versa).
#
# Deliberately absent: GB200/GB300 (not offered on vast), and the sub-20 GB
# Blackwell parts (RTX 5080/5070) — too small for a 7B bf16 training footprint,
# and each entry costs len(num_gpus) x 2 API probes per refresh.
# moved-from: herdd.DASH_GPUS_DEFAULT
DASH_GPUS_DEFAULT = (
    # Blackwell — datacenter
    "B200", "B300",
    # Blackwell — RTX PRO (the workhorse: this is what we actually rent)
    "RTX PRO 6000 WS", "RTX PRO 6000 S", "RTX PRO 5000",
    "RTX PRO 4500", "RTX PRO 4000",
    # Blackwell — GeForce
    "RTX 5090",
    # Hopper
    "H100 SXM", "H100 PCIE", "H100 NVL", "H200", "H200 NVL",
    # Ada / Ampere 48 GB pro+datacenter parts (added 2026-08-07 with the policy
    # widening — these are where the cheap 48-80 GB supply actually is: at the
    # 2026-08-07 probe RTX A6000 and RTX 6000Ada both floored at $0.133/hr bid
    # for 48 GB, and A800 PCIE at $0.129/hr for 80 GB)
    "L40S", "L40", "RTX A6000", "RTX 6000Ada", "A800 PCIE",
    # This tuple is FREELY EDITABLE again as of 2026-08-18. It used to be
    # interpolated into `--gpus` help, which put the probe list inside the
    # frozen CLI-surface fixture and made every edit here a CLI-contract
    # failure; `cli/dash_cache.py` now names the symbol instead. Re-verify a
    # new string against the live API before adding it — vast's naming is
    # unvalidated and a wrong string yields an EMPTY market, not an error.
    # prior generations (still the cheap tail of the market)
    "A100 SXM4", "A100 PCIE", "RTX 4090", "RTX 3090",
)
# moved-from: herdd.DASH_NUM_GPUS_DEFAULT
DASH_NUM_GPUS_DEFAULT = (1, 2, 4, 8)
# moved-from: herdd.DASH_OFFERS_KEPT
DASH_OFFERS_KEPT = 40          # market_offers rows per (gpu, num_gpus, kind).
                               # 40 makes the shape explorer a CENSUS rather
                               # than a top-N: measured 2026-08-17, 714 exact-N
                               # offers across 176 probes with a max of 38 for
                               # any one configuration, so nothing is truncated.
# moved-from: herdd.DASH_OFFER_LIMIT
DASH_OFFER_LIMIT = 128         # offers pulled per probe (the percentile sample)
# The vast API rate-limits `v0/bundles/` at 5 req/s (it says so in the 429 body:
# `'limit': 5.0`). The market survey is a gpus x counts x kinds cross product —
# 136 probes at the 2026-08-05 default list, 168 after the 2026-08-07 Ada/Ampere
# additions (~42 s per refresh at the pacing below) — so once the list grew past the
# original six cards, 8 unpaced workers tripped the limiter and three probes
# came back empty (a probe that 429s writes NO row, which reads as "we did not
# sample this configuration" on a page whose whole job is prices). Pace
# SUBMISSION below the limit instead of relying on backoff to sort it out.
# moved-from: herdd.DASH_MARKET_MAX_RPS
DASH_MARKET_MAX_RPS = 4.0

# -- dynamic class discovery (owner ask 2026-08-18) ---------------------------
# "can we dynamically show new GPUs as they show up even if they dont have all
# of the metadata? still good to have a complete picture."
#
# `DASH_GPUS_DEFAULT` is a hand-maintained list, so the board could only ever
# show silicon somebody had already thought to type in. Discovery reads the
# live offer board ONCE per market refresh, keeps the classes we could actually
# rent, and appends them to the probe set. A class needs no `GPU_SPECS` row to
# appear — the page renders what is known and dashes the rest.
#
# THE COST BOUND, and why it is a hard cap rather than a filter that happens to
# be small: the market survey is `classes x DASH_NUM_GPUS_DEFAULT x 2 kinds`
# probes, paced at DASH_MARKET_MAX_RPS. 22 classes is 176 probes (~44 s); the
# whole 35-class board would be 280 (~70 s). Discovery is driven by a THIRD
# PARTY's inventory, so a filter alone bounds nothing — vast can list twenty
# new SKUs tomorrow and a "filter to the good ones" rule would happily probe
# all of them. The cap is what makes the refresh cost knowable in advance.
#
# 34 is sized from a MEASUREMENT, not a round number. Discovery over the live
# board on 2026-08-18 (1,220 offers) found 29 usable classes — the 22 built-in
# plus 7 — costing 232 probes and ~58 s. 34 leaves five slots of real headroom
# for new silicon before the cap starts deferring, at a ceiling of 272 probes
# and ~68 s: still under 8% of the 15-minute refresh interval, so the cadence
# is untouched. When the cap does bind it is not silent — the deferred classes
# are named in the section's log line.
DASH_GPUS_MAX = 34

# One census read per refresh: 1 request out of ~241, so it does not earn a
# slower cadence of its own. A separate cadence would also need somewhere to
# PERSIST the discovered set between runs, and a stale persisted set is exactly
# the hand-maintained list this replaces. The cap already bounds the cost, so
# there is nothing left for a slower cadence to buy.
DASH_DISCOVER_LIMIT = 2000     # offers pulled for the census (one page)
# 22, not 24, and the two GB are not slack. `gpu_ram` is the advertised figure
# in MiB, and a nominal-24 GB part advertises less than 24 GiB of it: measured
# 2026-08-18, L4 reports 23034 MiB (22.49), A10 and some RTX A5000 23028
# (22.49), RTX 3090 24576 (24.00). A 24.0 floor therefore excluded exactly the
# 24 GB classes it was written to admit. 22 clears every nominal-24 part and
# still rejects the nominal-16 ones (RTX 4080 advertises 16376 MiB = 15.99).
DASH_DISCOVER_VRAM_MIN_GB = 22.0
# `compute_cap` is sm x10; sm_80 (Ampere) is the first bf16 silicon. This is
# the SAME test the per-offer writer persists, and it is a property of the
# chip, unlike `cuda_max_good`, which measures the host driver.
DASH_DISCOVER_BF16_CAP = 800
# A class the board lists exactly once is still a real answer ("this exists,
# here is its price"), so the floor is 1. It exists to reject a class that
# appears only as a malformed record.
DASH_DISCOVER_MIN_OFFERS = 1
# Sanity floor on the CENSUS itself, not on any class: a healthy read returns
# ~500 offers. A truncated or degraded response must not be allowed to redefine
# the probe set — below this we keep the known-good tuple and say so.
DASH_DISCOVER_MIN_CENSUS = 25
# moved-from: herdd.DASH_STATUS_MAX
DASH_STATUS_MAX = 200          # status_msg truncation
# moved-from: herdd.DASH_TICK_STALE_S
DASH_TICK_STALE_S = 135        # fleet_daemon_banner()'s 3*45 staleness line
# moved-from: herdd.DASH_DISK_OVERSIZED_FRAC
DASH_DISK_OVERSIZED_FRAC = 0.40   # _render_ls' oversizing warning threshold
# moved-from: herdd.DASH_DISK_OVERSIZED_GB
DASH_DISK_OVERSIZED_GB = 60

# /home/<user>/..., /Users/<user>/..., ~/.local/state/... -- an absolute machine
# path must never be published (repo rule) nor served off a public dashboard.
# `~` only counts as a path when a `/` follows it, so a bare "~2h left" in a
# status line survives intact instead of collapsing to "<path>2h left".
# moved-from: herdd._DASH_PATH_RE
_DASH_PATH_RE = re.compile(r"(?:/(?:home|Users)/[^/\s]+|~(?=/))(?:/[\w.+-]+)*/?")
# moved-from: herdd._DASH_KV_RE
_DASH_KV_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)=(\S+)")

# Publish-side widening of the `_SECRET_ENV_RE` family. Deliberately NOT folded
# into that regex: it also drives `_split_env_secrets`, where a `PAT`-ish match
# on `PATH` would strand the launch PATH out of the durable spec. Names first
# (`GITHUB_PAT` is a real launch env in this repo), then a VALUE-shape pass so a
# bare token with no `NAME=` in front of it -- the shape a vast `status_msg`
# echoing a failed `docker run` produces -- is redacted too.
# moved-from: herdd._DASH_SECRET_NAME_RE
_DASH_SECRET_NAME_RE = re.compile(r"PAT\b|PAT_|BEARER|API_?KEY|WEBHOOK", re.I)
# moved-from: herdd._DASH_TOKEN_RE
_DASH_TOKEN_RE = re.compile(
    r"\b(?:gh[pousr]_|github_pat_|glpat-|hf_|tskey-|sk-|xox[abprs]-|AKIA)"
    r"[A-Za-z0-9_-]{8,}")


# moved-from: herdd._dash_scrub
def _dash_scrub(s: object, limit: int | None = None, *,
                deps: DashDeps | None = None) -> str | None:
    """PURE. A free-text field (status_msg / label / health reason / alarm msg)
    reduced to something safe to publish: ANSI stripped, absolute machine paths
    collapsed to their basename (or `<path>`), secret-shaped `KEY=VALUE` pairs
    redacted, optionally truncated. None for anything empty.

    The path collapse serves two rules at once -- the repo's "never commit or
    publish an absolute machine path" and the v5 posture that every admin field
    is world-readable. The secret pass is a BACKSTOP only: the real guarantee is
    that no secret-bearing field is ever selected (rule 2 in the module
    docstring). It matches the `_SECRET_ENV_RE` name family WIDENED by
    `_DASH_SECRET_NAME_RE` (`GITHUB_PAT`, bearer, api-key -- names the B2-spec
    regex has no reason to carry), then makes a VALUE-shape pass for bare
    `ghp_`/`glpat-`/`hf_`-style tokens that carry no `NAME=` in front of them at
    all.

    `deps` supplies the two `launch.spec` members (`_is_secret_env`,
    `_SECRET_VAL_RE`) this ring cannot import yet; the pass order is otherwise
    unchanged and load-bearing.
    """
    if not isinstance(s, str) or not s.strip():
        return None
    d = _need(deps)          # after the early return: an empty field needs no deps
    out = fmt._ANSI_RE.sub("", s)

    def _path(m: re.Match[str]) -> str:
        base = m.group(0).rstrip("/").rsplit("/", 1)[-1]
        return base if base and base != "~" else "<path>"

    out = _DASH_PATH_RE.sub(_path, out)

    def _kv(m: re.Match[str]) -> str:
        k, v = m.group(1), m.group(2)
        secret = d.is_secret_env(k, v) or _DASH_SECRET_NAME_RE.search(k or "")
        return f"{k}=<redacted>" if secret else m.group(0)

    out = _DASH_KV_RE.sub(_kv, out)
    out = _DASH_TOKEN_RE.sub("<redacted>", out)
    out = d.secret_val_re.sub("://<redacted>@", out).strip()
    if limit is not None and len(out) > limit:
        out = out[:limit]
    return out or None


# moved-from: herdd._dash_pct
def _dash_pct(vals: Iterable[object], q: float) -> float | None:
    """PURE. Linear-interpolated percentile (q in 0..1) of a numeric list, or
    None when empty. Written out rather than borrowed from `statistics` so a
    2-offer probe still yields a defined p10/p90 instead of raising."""
    s = sorted(v for v in vals if isinstance(v, (int, float)))
    if not s:
        return None
    if len(s) == 1:
        return float(s[0])
    pos = (len(s) - 1) * max(0.0, min(1.0, q))
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return float(s[int(pos)])
    return float(s[lo] + (s[hi] - s[lo]) * (pos - lo))


# moved-from: herdd._dash_meta
def _dash_meta(conn: sqlite3.Connection, key: str, now: float | None = None) -> None:
    """Stamp one section's freshness row. Called INSIDE the section's
    transaction so `fetched_at` can never advertise data that did not land."""
    now = time.time() if now is None else now
    iso = datetime.datetime.fromtimestamp(
        now, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    conn.execute("INSERT OR REPLACE INTO meta(key,fetched_at,fetched_ts) "
                 "VALUES(?,?,?)", (key, iso, now))


# moved-from: herdd._dash_reap_threshold_s
def _dash_reap_threshold_s(deps: DashDeps | None = None) -> float:
    """The idle-reaper deadline `reap` itself would use (env override honoured),
    read WITHOUT invoking the reaper."""
    d = _need(deps)
    try:
        return max(0.0, float(os.environ.get("HERDD_REAP_IDLE_H", "")
                              or d.reap_idle_h_default) * 3600.0)
    except ValueError:
        return d.reap_idle_h_default * 3600.0


# moved-from: herdd._dash_instance_rows
def _dash_instance_rows(no_spot: bool = False, *,
                        deps: DashDeps | None = None) -> list[tuple[Any, ...]]:
    """One `_gather_ls_data()` fleet read projected onto the `instances` columns.

    This is the allowlist in code form: every value below is named explicitly
    off the instance record or off a pure helper, so no unlisted vast field --
    and in particular nothing from `extra_env` -- can reach the row tuple.
    `_instance_env` is consulted by `_stale_image_ids`/`gather_fleet_health`
    inside the gather, but only its digest STAMP is used, never returned here.

    The reap verdict is the idle lane's PREVIEW, recomputed in-process from the
    same first-observed-stopped ledger `ls` and `reap` share -- never by shelling
    out to the reaper, which would put a destroy verb on the dashboard's path.
    Live boxes get NULL: the idle lane only ever considers STOPPED boxes, and the
    live shapes the reaper's zombie lane may sweep (graded destroy/park on guard
    zombie verdicts) are already legible through `health_verdict`."""
    d = _need(deps)
    data = d.gather_ls_data(no_spot=no_spot)
    ins = data.get("instances") or []
    live = set(data.get("live_ids") or [])
    market = data.get("market") or {}
    jobs_by_box = data.get("jobs_by_box") or {}
    idle_secs = data.get("idle_secs") or {}
    stale = set(data.get("stale_ids") or [])
    health = data.get("health") or {}
    now = data.get("ts") or time.time()
    thresh = _dash_reap_threshold_s(d)

    rows = []
    for i in ins:
        iid = i.get("id")
        is_live = iid in live
        jobs = jobs_by_box.get(str(iid), [])
        act = [v for v in jobs if v.get("display_status") in d.active_job_states]
        # DEAD in the original and kept verbatim: nothing reads `state` — the
        # `state` COLUMN is filled from `actual_status` two dozen lines below,
        # and the dashboard's `boxState()` maps exactly those vast values
        # (running/loading/created/stopped/exited), not this active/suspended
        # vocabulary. Deleting it is a behavior-preserving cleanup, but it is a
        # cleanup, so it belongs to a later step and not to a verbatim port.
        state = ("active" if (is_live and act)          # noqa: F841
                 else "running" if is_live else "suspended")
        reserved, spot, avail = models._rates(i, market)
        is_bid = bool(i.get("is_bid"))
        # WHAT WE ARE BILLED is the instance's own `dph_total` -- on a BID box
        # that is OUR standing bid, which `supervise` ratchets up and never
        # lowers (box 44566398 J1: paid $2.34/hr against a $0.373 floor). The
        # market rates from `_rates` are COUNTERFACTUALS ("what this config
        # costs today") and are published as the separate `ondemand`/`spot`
        # columns. Writing the floor here understated the burn rate on exactly
        # the boxes the stuck-high-bid check exists to catch, and made
        # `vs_floor` a comparison of the floor against itself.
        hourly = models._num_dph(i.get("dph_total")) if is_live else None
        alloc, used = models._disk_gb(i)          # negatives already -> None
        frac = models._disk_frac(i)
        lab = i.get("label") or ""
        sec = idle_secs.get(str(iid))
        if sec is None:                           # live: not a reap candidate
            reap_verdict, reap_wait = None, None
        elif labels._reap_kept(lab):
            reap_verdict, reap_wait = "KEEP", None
        elif sec >= thresh:
            reap_verdict, reap_wait = "REAP", 0.0
        else:
            reap_verdict, reap_wait = "WAIT", round(thresh - sec, 1)
        h = health.get(str(iid)) or {}
        try:
            start_ts = float(i.get("start_date"))
        except (TypeError, ValueError):
            start_ts = None
        geo = (i.get("geolocation") or "").lstrip(", ").strip() or None
        rows.append((
            iid, i.get("machine_id"),
            (i.get("actual_status") or i.get("cur_state") or "").lower() or None,
            _dash_scrub(i.get("status_msg"), DASH_STATUS_MAX, deps=d),
            "bid" if is_bid else "ondemand",
            i.get("num_gpus"), i.get("gpu_name"),
            i.get("gpu_util") if is_live else None,
            hourly, models._storage_day(i), reserved, spot,
            None if avail is None else (1 if avail else 0),
            alloc, used, frac,
            1 if (frac is not None and frac < DASH_DISK_OVERSIZED_FRAC
                  and (alloc or 0) >= DASH_DISK_OVERSIZED_GB) else 0,
            sec, start_ts,
            (now - start_ts) if start_ts else None,
            _dash_scrub(lab, deps=d), 1 if labels._reap_kept(lab) else 0,
            models._label_value(lab, "run"), geo,
            fmt._image_short(models._instance_image(i)) or None,
            1 if iid in stale else 0,
            h.get("verdict"), _dash_scrub(h.get("reason"), deps=d),
            reap_verdict, reap_wait,
            len(act), ",".join(sorted(d.job_cell(v) for v in act)) or None,
        ))
    return rows


# moved-from: herdd._DASH_INSTANCES_INSERT
_DASH_INSTANCES_INSERT = (
    "INSERT INTO instances(iid,machine_id,state,status,mode,num_gpus,gpu,"
    "gpu_util,hourly,storage_day,ondemand,spot,avail,disk_gb,disk_used_gb,"
    "disk_frac,disk_oversized,idle_s,start_ts,age_s,label,keep,run_id,geo,"
    "image_short,stale_image,health_verdict,health_reason,reap_verdict,"
    "reap_wait_s,n_jobs,jobs) "
    "VALUES(" + ",".join("?" * 32) + ")")


# moved-from: herdd._dash_write_instances
def _dash_write_instances(conn: sqlite3.Connection, no_spot: bool = False, *,
                          deps: DashDeps | None = None) -> int:
    rows = _dash_instance_rows(no_spot=no_spot, deps=deps)
    with conn:
        conn.execute("DELETE FROM instances")
        conn.executemany(_DASH_INSTANCES_INSERT, rows)
        _dash_meta(conn, "instances")
    return len(rows)


# `_dash_verified` is NOT here — `core.models` owns it (see the module
# docstring). Every use below is `models._dash_verified(...)`.

# moved-from: herdd._DASH_MARKET_PACE_LOCK
_DASH_MARKET_PACE_LOCK = threading.Lock()
# Module-level MUTABLE state, evaluated at import and carried across calls: two
# probes in one process see the previous stamp. That is what `_sleep` exists for
# in the test.
# moved-from: herdd._DASH_MARKET_LAST_SEND
_DASH_MARKET_LAST_SEND = [0.0]


# moved-from: herdd._dash_market_pace
def _dash_market_pace(_sleep: Callable[[float], Any] = time.sleep) -> None:
    """Block until this thread may send its bundles probe, holding the pool to
    DASH_MARKET_MAX_RPS in aggregate. Cheap and exact enough: one lock, one
    monotonic stamp, no token bucket. Called once per probe ATTEMPT is not
    needed -- `request_soft`'s own jittered backoff covers a retry -- so it
    guards the initial send only."""
    with _DASH_MARKET_PACE_LOCK:
        now = time.monotonic()
        gap = 1.0 / DASH_MARKET_MAX_RPS
        wait = _DASH_MARKET_LAST_SEND[0] + gap - now
        if wait > 0:
            _sleep(wait)
            now += wait
        _DASH_MARKET_LAST_SEND[0] = now


def _dash_launch_floors() -> tuple[float, float, float]:
    """(cuda, inet_down, reliability) an auto-picked offer must clear — the
    launch defaults themselves (`cli/search`'s two, plus the inet knob), read
    once per probe because `_boot_knob` consults env and herdd.yaml."""
    return (float(config.LAUNCH_CUDA_MAX_GOOD),
            float(config._boot_knob("LAUNCH_INET_DOWN_MBPS")),
            float(config.LAUNCH_RELIABILITY_MIN))


def _dash_launch_ok(o: Mapping[str, Any],
                    floors: tuple[float, float, float]) -> bool:
    """PURE. Would our launch defaults actually be able to rent this offer?

    A missing or unparseable field FAILS: an unmeasured host property is not
    evidence of reachability, and the whole complaint this predicate answers is
    confident floors printed on hosts a launch silently filters out (measured
    2026-08-17: 33/57 bid floors failed at least one of these). `reliability or
    reliability2` is spelled exactly as the row writer spells it, so the
    persisted `launch_ok` and the persisted columns can never disagree."""
    cuda, inet, rel = floors
    got = (models._num_dph(o.get("cuda_max_good")),
           models._num_dph(o.get("inet_down")),
           models._num_dph(o.get("reliability") or o.get("reliability2")))
    return all(v is not None and v >= floor
               for v, floor in zip(got, (cuda, inet, rel)))


def _dash_gpu_name_ok(name: object) -> bool:
    """Is this a `gpu_name` we are willing to put in a probe URL and a page?

    Paranoid on purpose: the value is a third party's free text and it becomes
    a probe key, a sqlite primary-key component and a rendered label. Vast's
    real names are short ASCII with spaces and digits (`RTX PRO 6000 WS`).
    """
    if not isinstance(name, str):
        return False
    name = name.strip()
    if not (1 <= len(name) <= 40):
        return False
    return all(c.isalnum() or c in " -_." for c in name)


def _dash_discover_gpus(*, deps: DashDeps | None = None) -> tuple[list[str], str]:
    """`(probe set, note)` — the known-good tuple, widened by what vast lists.

    ADDITIVE ONLY. `DASH_GPUS_DEFAULT` is always kept in full and always first:
    those classes are audited, most carry `GPU_SPECS` rows, and a class whose
    supply is momentarily dry (L40 had zero offers in the 2026-08-18 census)
    must not silently vanish from the board because one read missed it.
    Discovery can therefore only ADD, and only up to `DASH_GPUS_MAX`.

    Ranked by offer count descending, so when the cap binds it spends the
    remaining slots on the deepest supply rather than on alphabetical luck.

    FALLS BACK, never fails. Every failure path — a bad read, a short census, a
    nonsense name, an unexpected shape — returns the known-good tuple with the
    reason in the note. A discovery bug must not be able to blank the board,
    which is why the caller gets a list it can always probe and a string it can
    always print, and never an exception.
    """
    d = deps if deps is not None else None
    query = getattr(d, "census_query", None) if d is not None else None
    base = list(DASH_GPUS_DEFAULT)
    if query is None:
        return base, "discovery skipped (census_query not wired)"
    try:
        ok, data, err = api.request_soft("POST", "v0/bundles/", query(), retries=2)
        if not ok:
            return base, f"discovery failed, kept the built-in set ({err})"
        offers = (data or {}).get("offers") or []
        if len(offers) < DASH_DISCOVER_MIN_CENSUS:
            return base, (f"discovery ignored, kept the built-in set "
                          f"(census returned {len(offers)} offers, "
                          f"under the {DASH_DISCOVER_MIN_CENSUS} sanity floor)")
        seen: dict[str, int] = {}
        vram: dict[str, float] = {}
        for o in offers:
            if not isinstance(o, Mapping):
                continue
            name = o.get("gpu_name")
            if not _dash_gpu_name_ok(name):
                continue
            name = str(name).strip()
            cap = models._num_dph(o.get("compute_cap"))
            if cap is None or cap < DASH_DISCOVER_BF16_CAP:
                continue        # pre-bf16 silicon: we cannot train or serve on it
            ram = models._num_dph(o.get("gpu_ram"))
            if ram is None or ram <= 0:
                continue
            seen[name] = seen.get(name, 0) + 1
            # The MINIMUM across the class, not this offer's value. Some hosts
            # advertise the whole box rather than one card (measured 2026-08-18:
            # RTX 4080 appears as both 16376 and 32760 MiB), and taking any
            # single offer would let a 2x16 GB box vouch for a 16 GB class. The
            # smallest figure any offer advertises is the one card we can count
            # on. Dividing by `num_gpus` is NOT the fix — it would wreck the
            # honest per-card rows, which are constant across widths.
            gb = ram / 1024.0
            vram[name] = min(vram[name], gb) if name in vram else gb
    except Exception as e:   # noqa: BLE001 — a blank board is worse than a stale one
        return base, f"discovery errored, kept the built-in set ({type(e).__name__}: {e})"

    known = set(base)
    fresh = sorted(
        (n for n, c in seen.items()
         if n not in known
         and c >= DASH_DISCOVER_MIN_OFFERS
         and vram.get(n, 0.0) >= DASH_DISCOVER_VRAM_MIN_GB),
        key=lambda n: (-seen[n], n),
    )
    room = max(0, DASH_GPUS_MAX - len(base))
    added, dropped = fresh[:room], fresh[room:]
    usable = sum(1 for n, c in seen.items()
                 if c >= DASH_DISCOVER_MIN_OFFERS
                 and vram.get(n, 0.0) >= DASH_DISCOVER_VRAM_MIN_GB)
    note = (f"discovered {usable} usable class(es) in {len(offers)} offers, "
            f"added {len(added)}: {','.join(added)}" if added
            else f"discovered {usable} usable class(es) in {len(offers)} offers, none new")
    if dropped:
        note += (f" — {len(dropped)} over the {DASH_GPUS_MAX}-class cap, "
                 f"deferred: {','.join(dropped)}")
    return base + added, note


# moved-from: herdd._dash_market_probe
def _dash_market_probe(gpu_name: str, num_gpus: int, kind: str, *,
                       deps: DashDeps | None = None
                       ) -> tuple[tuple[Any, ...], list[tuple[Any, ...]]]:
    """((market row), [offer rows]) for one (gpu, num_gpus, kind) probe.

    In BID mode the price you pay is `min_bid`; `dph_total` is the on-demand
    list price you do not pay -- so `price_field` switches and every percentile
    is taken over the field actually billed. The API sorts ascending, so
    `offers[0]` IS the floor and `p0 == best_total`. A probe that returns no
    offers still writes a row (`n_offers=0`, NULL prices): "unobtainable at any
    price" is a signal, not a gap.

    EXACT-N ONLY: the shared `build_search_query` filters `num_gpus >= N` (a
    launch happily takes a bigger box), but every column this row feeds is a
    per-CONFIGURATION claim -- the UI prints `p50`/`p90` on a row labelled `xN`
    and the spread chart's axis says "whole box". Percentiles taken over a mix
    of 1/2/4/8-GPU offers described no configuration at all (measured: the
    `1x H100 SXM bid` sample was mostly multi-GPU boxes, p50 $4.27 against
    $2.07 for the real single-GPU offers), and a cheap 2-GPU box could become
    the `x1` floor that `OurBid.floor` is graded against. So the response is
    post-filtered to offers of exactly this GPU count, order preserved.

    Every column is written TWICE over that sample: once permissively (the true
    market floor) and once through the launch-parity predicate (`_ok`, the floor
    something can actually be rented at). n_ok=0 with n_offers>0 is a finding —
    "listed, none reachable" — not a gap (DESIGN_V10_MARKET_SHAPE.md §1)."""
    d = _need(deps)
    query = _need_hook(d.offer_query, "offer_query")
    price_field = "dph_total" if kind == "ondemand" else "min_bid"
    _dash_market_pace()
    ok, dd, err = api.request_soft("POST", "v0/bundles/",
                                   query(gpu_name, num_gpus, kind),
                                   retries=4)
    if not ok:
        raise RuntimeError(f"bundles {gpu_name} x{num_gpus} {kind}: {err}")
    offers = [o for o in ((dd or {}).get("offers") or [])
              if (o.get("num_gpus") or 0) == num_gpus]
    prices = [p for p in (models._num_dph(o.get(price_field)) for o in offers)
              if p is not None]
    key = (gpu_name, num_gpus, kind)
    floors = _dash_launch_floors()
    if not offers:
        # the `ok_*_min` provenance describes the LENS, not the sample, so it is
        # written even here — a snapshot of nothing but empty configurations
        # would otherwise leave the UI with no filter values to label.
        return (key + (0, price_field) + (None,) * 11 + (0,) + (None,) * 8
                + floors), []
    # ONE evaluation per offer, reused by the aggregates and by the row column:
    # `n_ok == sum(launch_ok)` is the anti-drift seam the UI's default filter
    # rides on, and it holds only if both come from this list.
    ok_flags = [_dash_launch_ok(o, floors) for o in offers]
    # Over the FULL exact-N sample, never `offers[:DASH_OFFERS_KEPT]`: a floor
    # taken over the kept window would move when the window size changes.
    ok_offers = [o for o, ok in zip(offers, ok_flags) if ok]
    ok_prices = [p for p in (models._num_dph(o.get(price_field))
                             for o in ok_offers) if p is not None]
    best_ok = ok_offers[0] if ok_offers else None   # API sorts ascending
    best_ok_total = (models._num_dph(best_ok.get(price_field))
                     if best_ok is not None else None)
    best_ok_n = (best_ok.get("num_gpus") or num_gpus) if best_ok else num_gpus
    best = offers[0]
    best_total = models._num_dph(best.get(price_field))
    best_n = best.get("num_gpus") or num_gpus
    row = key + (
        len(offers), price_field,
        _dash_pct(prices, 0.0), _dash_pct(prices, 0.10),
        _dash_pct(prices, 0.50), _dash_pct(prices, 0.90),
        best_total,
        (best_total / best_n) if (best_total is not None and best_n) else None,
        best.get("machine_id"),
        (best.get("geolocation") or "").lstrip(", ").strip() or None,
        models._num_dph(best.get("cuda_max_good")),
        models._num_dph(best.get("inet_down")),
        models._num_dph(best.get("reliability") or best.get("reliability2")),
        len(ok_offers),
        _dash_pct(ok_prices, 0.0), _dash_pct(ok_prices, 0.10),
        _dash_pct(ok_prices, 0.50), _dash_pct(ok_prices, 0.90),
        best_ok_total,
        (best_ok_total / best_ok_n)
        if (best_ok_total is not None and best_ok_n) else None,
        best_ok.get("machine_id") if best_ok is not None else None,
        ((best_ok.get("geolocation") or "").lstrip(", ").strip() or None)
        if best_ok is not None else None,
    ) + floors
    orows = []
    for rank, (o, ok) in enumerate(zip(offers[:DASH_OFFERS_KEPT], ok_flags)):
        price = models._num_dph(o.get(price_field))
        n = o.get("num_gpus") or num_gpus
        ram = models._num_dph(o.get("gpu_ram"))
        cram = models._num_dph(o.get("cpu_ram"))
        orows.append(key + (
            rank, o.get("machine_id"),
            (o.get("geolocation") or "").lstrip(", ").strip() or None,
            price, (price / n) if (price is not None and n) else None,
            models._num_dph(o.get("dph_total")), models._num_dph(o.get("min_bid")),
            models._num_dph(o.get("storage_cost")),
            (ram / 1024.0) if ram is not None else None,
            models._num_dph(o.get("cuda_max_good")),
            models._num_dph(o.get("reliability") or o.get("reliability2")),
            models._num_dph(o.get("inet_down")), models._num_dph(o.get("inet_up")),
            models._num_dph(o.get("disk_space")),
            models._dash_verified(o), 1 if o.get("rentable") else 0,
            # `compute_cap` (sm x10) is the bf16 test; `cuda_max_good` measures
            # the host DRIVER and says nothing about the silicon, which is why
            # both are persisted. Cores must be the per-OFFER slice, never the
            # host's advertised `cpu_cores`.
            o.get("compute_cap"), models.effective_cores(o),
            (cram / 1024.0) if cram is not None else None,
            models._num_dph(o.get("dlperf")), 1 if ok else 0,
        ))
    return row, orows


# moved-from: herdd._DASH_MARKET_INSERT
_DASH_MARKET_INSERT = (
    "INSERT OR REPLACE INTO market(gpu_name,num_gpus,kind,n_offers,price_field,"
    "p0,p10,p50,p90,best_total,best_per_gpu,best_machine_id,best_geo,best_cuda,"
    "best_inet_down,best_reliability,"
    "n_ok,p0_ok,p10_ok,p50_ok,p90_ok,best_ok_total,best_ok_per_gpu,"
    "best_ok_machine_id,best_ok_geo,ok_cuda_min,ok_inet_down_min,"
    "ok_reliability_min) VALUES(" + ",".join("?" * 28) + ")")
# moved-from: herdd._DASH_OFFERS_INSERT
_DASH_OFFERS_INSERT = (
    "INSERT OR REPLACE INTO market_offers(gpu_name,num_gpus,kind,rank,"
    "machine_id,geo,price,price_per_gpu,dph_total,min_bid,storage_cost,"
    "gpu_ram_gb,cuda_max_good,reliability,inet_down,inet_up,disk_space,"
    "verified,rentable,compute_cap,cpu_cores_effective,cpu_ram_gb,dlperf,"
    "launch_ok) VALUES(" + ",".join("?" * 24) + ")")


# moved-from: herdd._dash_write_market
def _dash_write_market(conn: sqlite3.Connection, gpus: Sequence[str],
                       num_gpus_list: Sequence[int],
                       kinds: Sequence[str] = ("ondemand", "bid"), *,
                       deps: DashDeps | None = None) -> int:
    """Probe the gpus x num_gpus x kind cross product in parallel and replace
    both market tables in ONE transaction. A single failed probe is skipped (its
    key simply has no row this cycle) rather than losing the whole section."""
    keys = [(g, n, k) for g in gpus for n in num_gpus_list for k in kinds]
    if not keys:
        return 0
    rows, orows, failed = [], [], 0
    with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(8, len(keys))) as ex:
        futs = [ex.submit(_dash_market_probe, *k, deps=deps) for k in keys]
        for f in futs:
            try:
                r, orr = f.result()
            except Exception as e:
                failed += 1
                print(f"dash-cache market: probe failed: {e}", file=sys.stderr)
                continue
            rows.append(r)
            orows.extend(orr)
    if not rows:
        raise RuntimeError(f"every market probe failed ({failed}/{len(keys)})")
    with conn:
        conn.execute("DELETE FROM market")
        conn.execute("DELETE FROM market_offers")
        conn.executemany(_DASH_MARKET_INSERT, rows)
        conn.executemany(_DASH_OFFERS_INSERT, orows)
        _dash_meta(conn, "market")
    return len(rows)


# moved-from: herdd._dash_int
def _dash_int(x: object) -> int | None:
    """Ported with the fleet section it serves: `DashDeps.write_fleet` (plan §8
    step 5) is its only caller, and it belongs to the same fold."""
    try:
        return int(x)   # type: ignore[call-overload, no-any-return]  # object in, per the original
    except (TypeError, ValueError):
        return None


# moved-from: herdd._dash_write_account
def _dash_write_account(conn: sqlite3.Connection) -> int:
    """`GET v0/users/current/` reduced to the two numbers an operator needs.
    `email` and the account `id` are never selected: this page is public."""
    ok, d, err = api.request_soft("GET", "v0/users/current/", retries=2)
    if not ok:
        raise RuntimeError(f"users/current: {err}")
    with conn:
        conn.execute("DELETE FROM account")
        conn.execute(
            "INSERT INTO account(key,credit,balance) VALUES('account',?,?)",
            (models._num_dph((d or {}).get("credit")),
             models._num_dph((d or {}).get("balance"))))
        _dash_meta(conn, "account")
    return 1


# moved-from: herdd._dash_parse_sections
def _dash_parse_sections(spec: object) -> tuple[str, ...]:
    """Comma list -> the canonical-ordered section tuple. An unknown name is a
    usage error (exit 1), never a silent no-op."""
    if not spec:
        return DASH_SECTIONS
    want = [s.strip().lower() for s in str(spec).split(",") if s.strip()]
    bad = [s for s in want if s not in DASH_SECTIONS]
    if bad:
        sys.exit(f"error: unknown --sections {bad} "
                 f"(choose from {', '.join(DASH_SECTIONS)})")
    return tuple(s for s in DASH_SECTIONS if s in want)


# moved-from: herdd.cmd_dash_cache
def cmd_dash_cache(a: object, *, deps: DashDeps | None = None) -> None:
    """Refresh the dashboard's /admin snapshot in infra-metadata.db (READ-ONLY).

    Writes the `instances` / `market` / `fleet` / `account` tables plus their
    `meta` freshness rows, each section in its own transaction so a concurrent
    `sqlite3 -readonly` reader always sees a whole-old or whole-new snapshot.
    Every section is a pure read of the vast API, the fleetd socket's read ops,
    and the local idle ledger -- nothing here can create, park, bid on, or
    destroy anything (dashboard/DESIGN_V5_ADMIN.md §3, §10).

    Exit is 0 or 1 ONLY: a section that fails is reported on stderr and SKIPPED
    with its previous rows and stamp intact (a stale panel beats an empty one),
    and 1 is reserved for "the cache DB itself could not be opened". stdout is
    left empty on purpose.

    `deps` carries the reads that live above or beside `storage/` (see
    `DashDeps`). Without it the `account` section still runs and the other three
    report themselves SKIPPED -- the pre-existing section-failure path, with the
    exit-code contract unchanged."""
    sections = _dash_parse_sections(getattr(a, "sections", None))
    db = _infra_cache_db(a)
    try:
        conn = sqlite3.connect(db, timeout=5)
        conn.execute("PRAGMA busy_timeout=3000")
        # NOTE: no journal-mode pragma. The dashboard opens this file with
        # `sqlite3 -readonly`, which CANNOT open a WAL database -- flipping it
        # would silently break every page, the runs view included.
        conn.executescript(_INFRA_CACHE_SCHEMA)
        # CREATE TABLE IF NOT EXISTS is a no-op on a DEPLOYED cache, so without
        # this call a column added later never lands on the only DB that matters.
        _infra_cache_migrate(conn)
    except Exception as e:
        sys.exit(f"error: cannot open dashboard cache {db}: "
                 f"{type(e).__name__}: {e}")

    # An explicit --gpus is an operator decision and is obeyed verbatim; only
    # the DEFAULT path discovers, so a narrow hand-run probe stays narrow.
    gpus = [g.strip() for g in (getattr(a, "gpus", None) or "").split(",")
            if g.strip()]
    gpu_note = "--gpus (explicit)"
    if not gpus:
        # Only when the market section will actually run — discovery is a live
        # read, and `--sections fleet` should not pay for one.
        if "market" in sections:
            gpus, gpu_note = _dash_discover_gpus(deps=deps)
        else:
            gpus, gpu_note = list(DASH_GPUS_DEFAULT), "built-in set"
    try:
        ngpus = [int(n) for n in (getattr(a, "num_gpus", None) or "").split(",")
                 if n.strip()] or list(DASH_NUM_GPUS_DEFAULT)
    except ValueError:
        conn.close()
        sys.exit("error: --num-gpus takes a comma list of integers, e.g. 1,2,4,8")

    failed = []
    try:
        # An EMPTY stdout is a structural guarantee here, not a convention: a
        # `print()` added later to any transitive callee (the jobs fold, a
        # market probe, the health gather) would otherwise land on the exact
        # channel an execFile caller reads. Reroute the whole section loop.
        with contextlib.redirect_stdout(sys.stderr):
            for name in sections:
                t0 = time.time()
                try:
                    if name == "instances":
                        n = _dash_write_instances(
                            conn, no_spot=bool(getattr(a, "no_spot", False)),
                            deps=deps)
                        note = ""
                    elif name == "market":
                        n = _dash_write_market(conn, gpus, ngpus, deps=deps)
                        note = (f" ({len(gpus)}gpu x {len(ngpus)}cfg x 2)"
                                f" [{gpu_note}]")
                    elif name == "fleet":
                        n, err = _need_hook(_need(deps).write_fleet,
                                            "write_fleet")(conn)
                        note = f" (daemon DOWN: {err})" if err else ""
                    else:
                        n = _dash_write_account(conn)
                        note = ""
                except Exception as e:
                    failed.append(name)
                    print(f"dash-cache {name}: SKIPPED -- {type(e).__name__}: "
                          f"{e} (previous rows kept)", file=sys.stderr)
                    continue
                print(f"dash-cache {name}: {n} row(s){note} in "
                      f"{time.time() - t0:.1f}s -> {db}", file=sys.stderr)
    finally:
        conn.close()
    if failed:
        print(f"dash-cache: {len(failed)}/{len(sections)} section(s) skipped: "
              f"{','.join(failed)}", file=sys.stderr)
    # deliberate: a skipped section is NOT a nonzero exit. The Node caller
    # treats nonzero as total failure, and a partially-fresh cache is a
    # success -- staleness is expressed through the per-section `meta` stamp.

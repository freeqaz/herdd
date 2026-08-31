"""vastlib.core.config — configuration, environment and knob resolution.

Why this exists
---------------
One home for every question of the form "what value should this run use, and who
gets to say so". The answer is always the same lattice — **CLI > env >
`herdd.yaml` > built-in constant** — and the whole point of a single module is
that the lattice is written once instead of re-derived per call site. It owns:

* `.env` discovery/loading (`load_env`) — walk up from CWD, then from the
  install tree, never clobber;
* `herdd.yaml` loading + merge order (`load_herdd_config`, `_load_yaml_file`,
  `_parse_simple_yaml`): committed repo defaults -> personal override ->
  `$HERDD_CONFIG`;
* the declarative per-runset launch-env block validator (`_runset_env_defaults`
  + its reserved-key lattice);
* the canonical knob resolver `_boot_knob` and its `_BOOT_KNOB_DEFAULTS` table;
* the local-GPU authorization gate (`local_gpu_allowed` / `require_local_gpu`),
  the jobs-handoff safe-off switch, fleetd's fail-closed auto-adopt ceiling;
* the shared launch disk defaults (`DISK_DEFAULT_*_GB`, `default_disk_gb`).

This is the bottom of the DAG: stdlib only (PyYAML is opportunistic, never a
hard dependency), no vastlib import, no Zone S import. It stays importable and
testable without the CLI.

What is deliberately NOT here
-----------------------------
* **No policy restatement.** `local_gpu_allowed` enforces whatever
  `allow_local_gpu` says; the rule itself lives in the key's comment block in
  `tools/vast/herdd.yaml` and nowhere else. Same for the jobs-handoff switch.
* **`_load_runset_config` / `_load_runset_spot_config`** — they parse nested YAML
  through `jobmeta._parse_job_yaml` and resolve the `runsets/` directory through
  `herdd`'s own module-global `_HERE` (a different global from the `_HERE`
  below, and one the suite monkeypatches). They stay with their caller.
* **Accessors for the ~70 stray `os.environ.get` sites** the fat `herdd.py`
  carried (they moved with their owning functions into `vastlib` at step 6d;
  `herdd.py` itself now reads no env at all). They
  are inventoried at the bottom of this file (`ENV_SITES_TODO`) and route
  through here only when their owning function is itself ported — building
  accessors ahead of the callers would either change a precedence or leave dead
  code. Unifying them is plan §9 work, not this refactor.
* **No behavior change of any kind.** Notably: `_job_replacement_knob` has no
  yaml rung where `_rebid_knob` has one, and `_job_replacement_verified`
  deliberately bypasses the coercing resolver (`bool("0")` is `True`). Those
  asymmetries are load-bearing; a future port must not "tidy" them into one
  helper.

Provenance
----------
Verbatim move of `tools/vast/vastconf.py` (itself extracted from `herdd.py`
2026-07-30, increment I1 of the v1 plan), performed as step 2 of
`docs/plans/vast-tooling-refactor-v2.md` §8. Bodies are unchanged; only type
annotations were added. Every symbol carries its `# moved-from:` marker
(grammar: `vastlib/README.md` §2). The flat `vastconf.py` stays in place until
step 7 turns it into a deprecation shim, so both modules are live during the
port and `herdd.<name>` keeps resolving.
"""

from __future__ import annotations

import os
import re
from typing import Any, Callable, TypeVar, overload

# The parsed `herdd.yaml` shape. Values come from PyYAML (any scalar, or the
# stdlib fallback's str/None), so the value type is genuinely dynamic; keys are
# always strings.
ConfigMap = dict[str, Any]

# `_boot_knob`'s `cast` is the caller's coercion (float by default, but `int`,
# `str` and predicate-shaped casts are all in use in herdd). Keeping it generic
# means the knob's return type follows the cast instead of collapsing to `Any`.
_KnobT = TypeVar("_KnobT")


# moved-from: vastconf.load_env
def load_env() -> None:
    """Populate os.environ from the nearest .env (walking up), without clobbering.

    Two walk anchors, first hit wins: CWD (a caller standing in another repo
    keeps that repo's .env), then this install's own tree (`_HERE`) — so an
    absolute-path `python3 tools/vast/herdd.py …` from outside the repo still
    finds the repo .env instead of silently losing B2_BUCKET/API keys."""
    for start in (os.getcwd(), _HERE):
        d = start
        for _ in range(6):
            p = os.path.join(d, ".env")
            if os.path.isfile(p):
                for line in open(p):
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    k = k.strip()
                    # strip inline comments + surrounding quotes on the value
                    v = v.split(" #", 1)[0].strip().strip('"').strip("'")
                    os.environ.setdefault(k, v)
                return
            nd = os.path.dirname(d)
            if nd == d:
                break
            d = nd


# `_HERE` is THIS file's directory. In the flat module it was `tools/vast`; here
# the module sits one level deeper, so the repo-config path is resolved from the
# package parent — same file, same precedence, and `herdd.yaml` is still read
# from `tools/vast/`. This is the one line of the port that is not textually
# verbatim, and it is what keeps it behaviorally verbatim.
# moved-from: vastconf._HERE
_HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# moved-from: vastconf._REPO_CONFIG
_REPO_CONFIG = os.path.join(_HERE, "herdd.yaml")          # committed defaults
# moved-from: vastconf._USER_CONFIG
_USER_CONFIG = os.path.expanduser("~/.config/herdd/herdd.yaml")  # personal override


# moved-from: vastconf._parse_simple_yaml
def _parse_simple_yaml(text: str) -> ConfigMap:
    """Stdlib-only fallback parser for the FLAT-MAPPING subset of YAML this
    config actually uses (key: value per line, '#' comments, quoted or bare
    scalars). Used only when PyYAML isn't installed — herdd is stdlib-only
    by design, so a real YAML parser is a nice-to-have, not a hard dependency.
    Nested maps/lists are NOT supported by this fallback."""
    out: ConfigMap = {}
    for raw_line in text.splitlines():
        line = raw_line.split(" #", 1)[0].rstrip() if not raw_line.strip().startswith("#") else ""
        line = line.strip()
        if not line or ":" not in line:
            continue
        k, v = line.split(":", 1)
        k = k.strip()
        value: str | None = v.strip().strip('"').strip("'")
        if value is not None and value.lower() in ("null", "~", ""):
            value = None
        out[k] = value
    return out


# moved-from: vastconf._load_yaml_file
def _load_yaml_file(path: str) -> ConfigMap:
    if not os.path.isfile(path):
        return {}
    text = open(path).read()
    try:
        import yaml  # type: ignore[import-untyped]  # optional: real parser if installed
        data = yaml.safe_load(text)
        return data if isinstance(data, dict) else {}
    except ImportError:
        return _parse_simple_yaml(text)


# moved-from: vastconf.load_herdd_config
def load_herdd_config() -> ConfigMap:
    """Merge tools/vast/herdd.yaml (committed defaults) with an optional
    personal override at ~/.config/herdd/herdd.yaml, then an explicit
    HERDD_CONFIG=path (highest precedence). Later sources win per-key."""
    cfg: ConfigMap = {}
    cfg.update(_load_yaml_file(_REPO_CONFIG))
    cfg.update(_load_yaml_file(_USER_CONFIG))
    extra = os.environ.get("HERDD_CONFIG")
    if extra:
        cfg.update(_load_yaml_file(extra))
    return cfg


def fleet_state_dir() -> str:
    """fleetd's state/journal/socket directory (`$FLEETD_STATE_DIR` overrides —
    the tests and a dry-run soak use their own).

    Lives HERE, at the bottom of the DAG, rather than with the daemon client
    that names it, because two rings need it and the ring order forbids the
    edge: `fleet.client` is above `market`, and `market.hostrep` stores the host
    reputation file in this directory. Re-spelling the default in the second
    file is how the two silently disagree the next time it changes.
    `fleet.client.fleet_state_dir` remains the name every caller binds and every
    test monkeypatches; it delegates here.

    Read on EVERY call, never cached — `conftest.py` redirects the env for the
    whole suite, and a snapshot at import time re-arms the live-daemon leak that
    fixture exists to stop.
    """
    return os.path.expanduser(os.environ.get("FLEETD_STATE_DIR",
                                             "~/.local/state/vast-fleetd"))


# ---------------------------------------------------------------- local GPU
# ONE switch, ONE place: `allow_local_gpu` in tools/vast/herdd.yaml, whose
# comment block IS the rule. Owner ruling 2026-08-11 reversed the 2026-08-06
# ban and the shipped key is now `true` — small tests run locally, big runs rent
# spot. This module deliberately does not restate that; it only enforces
# whatever the key says, which is why the built-in default below stays False (a
# checkout with no config is conservative, the config file authorizes).
#
# Every local-GPU entry point calls local_gpu_allowed()/require_local_gpu()
# rather than restating the rule: `herdd job run-local` and
# `tools/vast/local_smoke.py` today. rehearse.sh is CPU-only by construction and
# is NOT gated. Do not copy this policy into a doc or another module — link to
# the key.
# moved-from: vastconf.LOCAL_GPU_KEY
LOCAL_GPU_KEY = "allow_local_gpu"
# moved-from: vastconf.LOCAL_GPU_ENV
LOCAL_GPU_ENV = "HERDD_ALLOW_LOCAL_GPU"


# moved-from: vastconf.local_gpu_allowed
def local_gpu_allowed(cfg: ConfigMap | None = None) -> bool:
    """True if local-GPU lanes may run. Built-in default False; shipped key True.

    The built-in default is deliberately the conservative one — it is what a
    checkout with no config file gets. Authorization lives in the config, where
    it is reviewable, not in this function.

    Env `HERDD_ALLOW_LOCAL_GPU` wins over the config file, in BOTH directions:
    `=1` authorizes a one-off without an edit someone forgets to revert, and
    `=0` re-closes the lane for a single command without touching the shipped
    key (useful when proving a bundle does not silently fall back to local).
    """
    env = os.environ.get(LOCAL_GPU_ENV)
    if env is not None:
        return str(env).strip().lower() in ("1", "true", "yes", "on")
    if cfg is None:
        cfg = load_herdd_config()
    return str(cfg.get(LOCAL_GPU_KEY, "false")).strip().lower() in (
        "1", "true", "yes", "on")


# moved-from: vastconf.require_local_gpu
def require_local_gpu(lane: str, cfg: ConfigMap | None = None) -> None:
    """Refuse a local-GPU lane unless authorized. Exits; never returns False."""
    if local_gpu_allowed(cfg):
        return
    raise SystemExit(
        f"error: {lane} runs on the LOCAL GPU and this checkout has not "
        f"authorized it ({LOCAL_GPU_KEY} is not set true).\n"
        "  The shipped tools/vast/herdd.yaml sets it TRUE (owner ruling "
        "2026-08-11: small tests local, big runs rent spot), so seeing this "
        "means the key was overridden, the config was not found, or "
        f"{LOCAL_GPU_ENV}=0 is set for this command.\n"
        f"  To authorize: {LOCAL_GPU_ENV}=1 <command>   (one-off)\n"
        f"                {LOCAL_GPU_KEY}: true in tools/vast/herdd.yaml "
        "or ~/.config/herdd/herdd.yaml   (standing)\n"
        "  CPU-only plumbing check, never gated: "
        "tools/vast/rehearse.sh <folder>")


# ------------------------------------------------- jobs-lane economic handoff
# SAFE-OFF since 2026-08-08 (incident 22:17Z, box 47214941; tasks #61/#62/#67).
# The economic handoff — rent a cheaper understudy, migrate the tickets, retire
# the primary — is DISABLED on fleetd's `jobs`/`serve` profiles. It fenced a
# RUNNING job four minutes into setup, parked the primary with its bid pinned to
# $0.001, and left the understudy with no watch and no budget cap.
#
# The individual defects are fixed (the watch now follows a completing migration;
# the fence has resumability preconditions and unwinds; the horizon is measured
# work, not a hang-detector ceiling). The switch is OFF anyway, because the
# feature's whole remaining value is a case the same-day autobid work already
# covers: on a machine where the cost cap binds, the cap does handoff's job
# without a second box.
#
# The name says what it is. Turning this on re-enables a lane that has cost us a
# cell of work and an unwatched rental, and whoever turns it on should have to
# type that.
# moved-from: vastconf.JOBS_HANDOFF_UNSAFE_KEY
JOBS_HANDOFF_UNSAFE_KEY = "jobs_handoff_unsafe_enable"
# moved-from: vastconf.JOBS_HANDOFF_UNSAFE_ENV
JOBS_HANDOFF_UNSAFE_ENV = "HERDD_JOBS_HANDOFF_UNSAFE"


# moved-from: vastconf.jobs_handoff_enabled
def jobs_handoff_enabled(cfg: ConfigMap | None = None) -> bool:
    """True if fleetd's jobs/serve profiles may arm an economic handoff. Default
    False. Env `HERDD_JOBS_HANDOFF_UNSAFE` wins over the config file (a one-off
    that nobody has to remember to revert), same precedence as the local-GPU
    switch. The ARM preconditions in bidpolicy still bind when this is on; only
    the profile-level default is what this flips."""
    env = os.environ.get(JOBS_HANDOFF_UNSAFE_ENV)
    if env is not None:
        return str(env).strip().lower() in ("1", "true", "yes", "on")
    if cfg is None:
        cfg = load_herdd_config()
    return str(cfg.get(JOBS_HANDOFF_UNSAFE_KEY, "false")).strip().lower() in (
        "1", "true", "yes", "on")


# ------------------------------------------------ fleetd auto-adopt ceiling
# The PROVISIONAL cap fleetd's safety net applies to a box it auto-adopts and
# for which no durable ceiling can be inherited (FLEETD_DESIGN "The ceiling
# ledger", path 1 of the 2026-08-03 filing: a box the operator believed was
# capped at $5 was in fact adopted `bare` with NO ceiling at all and billed
# unbounded).
#
# FAIL-CLOSED by construction: this resolver never returns None and never
# returns a non-positive number. A missing key, an unreadable config file, a
# garbage value, a negative value and a NaN all resolve to
# ADOPT_DEFAULT_BUDGET_USD. "Unlimited" is not expressible here — the way to
# raise a cap is `herdd fleet watch <IID> --budget <USD>`, which is an
# explicit, journaled, attributable act.
#
# The figure is a RUNAWAY BACKSTOP, not a budget. At the expensive tier
# ($2/hr+, the fuse threshold in fleetd.EXPENSIVE_DPH_USD) $10 is ~5h of
# unattended running before fleetd PARKS the box and alarms; on a cheap box it
# is days. A breach parks — resumable, never a destroy — so the cost of the
# default being too low is one `fleet resume` plus a real `fleet watch`, and
# the cost of it being absent is what the journal already recorded: 121 boxes
# that ticked uncapped and $23.87 of spend nobody had authorized a ceiling for.
# moved-from: vastconf.ADOPT_DEFAULT_BUDGET_USD
ADOPT_DEFAULT_BUDGET_USD = 10.0
# moved-from: vastconf.FLEETD_ADOPT_BUDGET_KEY
FLEETD_ADOPT_BUDGET_KEY = "fleetd_adopt_default_budget_usd"
# moved-from: vastconf.FLEETD_ADOPT_BUDGET_ENV
FLEETD_ADOPT_BUDGET_ENV = "FLEETD_ADOPT_DEFAULT_BUDGET_USD"


# moved-from: vastconf.fleetd_adopt_default_budget_usd
def fleetd_adopt_default_budget_usd(cfg: ConfigMap | None = None) -> float:
    """The provisional cap for an auto-adopted box with no inheritable ceiling.

    Precedence: `FLEETD_ADOPT_DEFAULT_BUDGET_USD` env > `herdd.yaml`
    (`fleetd_adopt_default_budget_usd`) > ~/.config override > the constant.
    Returns a POSITIVE float, always — see the fail-closed note above."""
    for raw in (os.environ.get(FLEETD_ADOPT_BUDGET_ENV),
                (cfg if cfg is not None else _adopt_cfg()).get(
                    FLEETD_ADOPT_BUDGET_KEY)):
        if raw is None or str(raw).strip() == "":
            continue
        try:
            v = float(str(raw).strip())
        except (TypeError, ValueError):
            return ADOPT_DEFAULT_BUDGET_USD          # garbage -> conservative
        # `v != v` catches NaN, which compares false against every bound and
        # would otherwise sail through as a cap no spend can ever breach; an
        # infinity is "unlimited" spelled in a way that reads like a number.
        if v != v or v in (float("inf"), float("-inf")) or v <= 0:
            return ADOPT_DEFAULT_BUDGET_USD
        return v
    return ADOPT_DEFAULT_BUDGET_USD


# moved-from: vastconf._adopt_cfg
def _adopt_cfg() -> ConfigMap:
    """Config for the adopt-default resolver; an unreadable config is not an
    error here, it is the conservative default (fail-closed)."""
    try:
        return load_herdd_config()
    except Exception:
        return {}


# Keys a config.yaml env: block may NOT set: creds/identity/boot params that
# cmd_train derives itself. A config.yaml trying to set these is a bug, so the
# helper fails closed (see _runset_env_defaults). Exact keys + prefix families;
# the B2_ prefix also covers every var _b2_eu_pairs() emits (all B2_*).
# moved-from: vastconf._RUNSET_ENV_RESERVED
_RUNSET_ENV_RESERVED = frozenset((
    "RUN_ID", "RUNSET", "HF_TOKEN", "BASE_MODEL_B2", "SELFTEST_BASE_B2",
    "FAST_BOOT", "TRAIN_ENV_VER",
))
# moved-from: vastconf._RUNSET_ENV_RESERVED_PREFIXES
_RUNSET_ENV_RESERVED_PREFIXES = ("B2_", "LLM_", "OPENROUTER_")
# moved-from: vastconf._RUNSET_ENV_KEY_RE
_RUNSET_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


# moved-from: vastconf._runset_env_defaults
def _runset_env_defaults(cfg: ConfigMap) -> list[str]:
    """Turn a parsed config.yaml's 'env:' block into ["K=V", ...], sorted by key
    for deterministic wire order. Owner directive: per-runset launch-env defaults
    must be declarative so they can't be forgotten at a manual launch. These feed
    the START of cmd_train's extra_env, so precedence is config.yaml env: <
    flag-derived < explicit --env (last wins on the wire).

    Validation (raises ValueError — the call site turns it into sys.exit so the
    helper stays testable without launching):
      - keys must match ^[A-Za-z_][A-Za-z0-9_]*$
      - values must be scalars; coerced via str(), with bool True/False -> "1"/"0"
        (bash consumers test = "1"; PyYAML would otherwise stringify to "True")
      - RESERVED keys (creds/identity/boot params cmd_train owns) are a hard
        error — fail closed, never silently ignore."""
    env = cfg.get("env")
    if env is None:
        return []
    if not isinstance(env, dict):
        raise ValueError("runset config.yaml 'env:' must be a mapping of KEY: value")
    out: list[str] = []
    # key=str: a non-string YAML key (bare true:/1: under PyYAML) must reach the
    # ValueError below, not TypeError out of sorted()
    for k in sorted(env, key=str):
        if not isinstance(k, str) or not _RUNSET_ENV_KEY_RE.match(k):
            raise ValueError(
                f"runset config.yaml env: invalid key {k!r} "
                f"(must match {_RUNSET_ENV_KEY_RE.pattern})")
        if k in _RUNSET_ENV_RESERVED or k.startswith(_RUNSET_ENV_RESERVED_PREFIXES):
            raise ValueError(
                f"runset config.yaml env: key {k!r} is reserved — cmd_train "
                f"sets creds/identity/boot params itself; remove it from env:")
        v = env[k]
        if isinstance(v, bool):
            v = "1" if v else "0"
        elif isinstance(v, (str, int, float)):
            v = str(v)
        else:
            raise ValueError(
                f"runset config.yaml env: value for {k!r} must be a scalar, "
                f"got {type(v).__name__}")
        out.append(f"{k}={v}")
    return out


# Boot-health knob defaults (Config knobs table). Precedence CLI > env >
# herdd.yaml > constant, resolved by _boot_knob (matches RCLONE_*/JOBD_*).
# moved-from: vastconf._BOOT_KNOB_DEFAULTS
_BOOT_KNOB_DEFAULTS: dict[str, float] = {
    "BOOT_MIN_MBPS": 5.0,          # sustained-throughput floor, MB/s (decimal).
                                   # AGGREGATE across layers (per-layer HWM sum
                                   # in parse_pull_progress), NOT per-TCP-flow:
                                   # hosts shape per-flow to ~1-16 MB/s, so a
                                   # single slow flow with fine aggregate must
                                   # not condemn. Owner directive 2026-08-02
                                   # named ~3 MB/s as the bad-host cut; 5 is
                                   # the justified tightening kept from P0
                                   # (2026-07-20): healthy hosts do 40+ MB/s,
                                   # the shaped-host failure class measures
                                   # 1-16, and 5 splits them with margin.
    "BOOT_MBPS_WINDOW_S": 300,     # window that must be FULL and low to condemn
    "BOOT_HEALTH_POLL_S": 20,      # vast-API poll cadence during `loading`
    "BOOT_MAX_HOST_RETRIES": 3,    # distinct machines tried per stage/run —
                                   # also the jobs-lane pull-watchdog relaunch
                                   # cap (the relaunch-loop guard)
    "LAUNCH_INET_DOWN_MBPS": 1000, # default advertised-download floor (Mb/s) for
                                   # AUTO-PICKED offers (owner directive
                                   # 2026-08-03: serve box 46682177 spent 39 min
                                   # in its image pull on an 805 Mb/s host while
                                   # a 26689 Mb/s host pulled the same image in
                                   # minutes). Slow boots are dominated by the
                                   # image pull, and per-host `inet_down` is the
                                   # best pick-time predictor we have — but it
                                   # is a whole-machine Ookla snapshot and hosts
                                   # shape per-TCP-flow (vast-per-flow memory),
                                   # so the floor is PREVENTION, not a
                                   # guarantee; the boot SLA below is the
                                   # enforcement. Measured 2026-08-03 (live
                                   # slow-host probe): inet_down is unreliable
                                   # in BOTH directions — a 321 Mb/s-rated host
                                   # out-pulled its rating from B2 AND booted
                                   # in ~6 min because the image was CACHED
                                   # (cache state dominates, and vast exposes
                                   # no cached-image signal; hosts.py's
                                   # scorecard is the only warmth proxy). So
                                   # this stays at the top of the conservative
                                   # range, never higher, and the soft pickers
                                   # deliberately fall back unfloored rather
                                   # than over-filter. Bypassed by explicit
                                   # --inet-down (0 disables), --any-inet, and
                                   # every --machine/--host/--offer pin. Soft
                                   # relaunch pickers retry WITHOUT the floor
                                   # when it empties the market (a rescue must
                                   # not fail for want of a fast host).
    "BOOT_SLA_S": 600,             # owner directive 2026-08-03: "longer than 10
                                   # minutes to come online is unacceptable."
                                   # Deadline from box start (start_date) to the
                                   # owning lifecycle's came-online milestone —
                                   # jobd's first JOBD_STATUS stamp (jobs), the
                                   # box-written SERVE_STATUS progress marker
                                   # (serve), the loading->running flip (run/
                                   # train). Enforced ONLY by the lifecycle that
                                   # OWNS the box (fleetd watch / supervise /
                                   # job supervise): breach = replacement on a
                                   # different machine, workload re-attached,
                                   # laggard destroyed. The passive `guard`
                                   # sweep is untouched — it never destroys a
                                   # loading box (2026-08-03 ruling).
                                   # <= 0 disables SLA enforcement everywhere.
    "BOOT_SLA_MAX_KILLS": 2,       # escalating tolerance: after this many
                                   # consecutive boot-SLA kills on one watch the
                                   # deadline WIDENS (x BOOT_SLA_BACKOFF_MULT
                                   # per further kill) instead of flapping;
                                   # BOOT_MAX_HOST_RETRIES stays the hard
                                   # disarm.
    "BOOT_SLA_BACKOFF_MULT": 2.0,  # deadline multiplier per kill past
                                   # BOOT_SLA_MAX_KILLS (600s, 600s, 1200s, ...)
    "BOOT_PULL_TIMEOUT_S": 600,    # owner directive 2026-08-02: a box that has
                                   # not finished its pull (reached `running`)
                                   # within 10 min is cut loose and the work
                                   # rescheduled on another host — the problem
                                   # is HOST QUALITY, not box death, and the
                                   # pull phase is GPU-unbilled (invoice-
                                   # verified), so the kill costs only the
                                   # wasted pull. Deliberately aggressive: a
                                   # host that needs longer for our image is a
                                   # host we do not want (46636056 took 1.3 h).
                                   # The throughput floor above fires EARLIER
                                   # on measurably-starved hosts; this is the
                                   # blunt backstop.
    # --- durable zombie-sweep (`herdd ls` scream + `guard`) knobs ---------- #
    # These drive the single-snapshot, age-based classification (classify_box_
    # health), NOT the rate-based P0 watcher above — one `ls` fold cannot compute
    # a throughput rate, so the durable layer condemns on AGE past a deadline.
    "GUARD_LOADING_DEADLINE_S": 1500,   # actual_status loading/created past this
                                        # = SUSPECT (mirrors BOOT_DEADLINE_S).
                                        # The loading phase is GPU-UNBILLED
                                        # (invoice-verified; storage only), so
                                        # this deadline guards SCHEDULE, not $.
                                        # DELIBERATELY UNCHANGED on 2026-08-03
                                        # despite its proven false positive at
                                        # 27m (46682177 cleared to OK at 40m):
                                        # the defect was never the number, it
                                        # was that a wall-clock age was allowed
                                        # to license an IRREVERSIBLE action.
                                        # Retuning it to whatever survived that
                                        # morning would just move the false
                                        # positive. Past this the classifier
                                        # asks for EVIDENCE instead (is the
                                        # pull advancing? -> LOADING_SLOW,
                                        # advisory) and the strongest licensed
                                        # action in the phase is a PARK.
    "GUARD_LOADING_HARD_S": 3600,       # ...but a bound based on progress still
                                        # needs a stop, or a status_msg frozen
                                        # mid-`Downloading` holds a dead box
                                        # forever (the 45373337 ~10h shape).
                                        # Past this, loading is ZOMBIE_LOADING_
                                        # STALL even while "pulling" — which
                                        # licenses a PARK, never a destroy.
                                        # 1h is chosen off measurement, not off
                                        # this morning's survivors: the healthy
                                        # pulls that morning ran 27-40 min on
                                        # 3 hosts (and would be LOADING_SLOW
                                        # here anyway, since they were actively
                                        # pulling); a warm-cache host does the
                                        # same image in ~30s and the workstation
                                        # times the blobs at ~4 min. The jobs
                                        # lane's own BOOT_PULL_TIMEOUT_S cuts a
                                        # slow host loose at 10 min by owner
                                        # directive, so 1h is 6x more patient
                                        # than the lane that actually matters
                                        # for schedule — and its outcome is
                                        # recoverable, where that one's is not.
    "GUARD_ENVSETUP_DEADLINE_S": 900,   # env-setup phase (2026-08-02 split):
                                        # actual_status running — where GPU
                                        # billing STARTS — but jobd has never
                                        # stamped JOBD_STATUS. Tighter than the
                                        # loading deadline because this phase
                                        # bills full GPU rate. Age is measured
                                        # from LAUNCH (the API exposes no
                                        # loading→running timestamp), so a slow
                                        # pull can consume it; the reap confirm
                                        # lane's progress evidence (download/
                                        # disk movement) is what protects a
                                        # legitimately slow install from the
                                        # AUTOMATIC sweep.
    "GUARD_JOBD_STALE_S": 600,          # a running jobs-box whose JOBD_STATUS
                                        # heartbeat is absent/older than this =
                                        # jobd dead (cadence is 60s; skew-tolerant)
    "GUARD_TICKET_DEADLINE_S": 1500,    # a submitted ticket unclaimed past this
                                        # while the box is up = jobd not claiming
    "REAP_ZOMBIE_CONFIRM_S": 900,       # the AUTOMATIC sweep acts only after the
                                        # SAME zombie verdict has persisted this
                                        # long across reap passes with NO progress
                                        # (pull bytes / jobd heartbeat / box
                                        # download traffic / disk usage) — one
                                        # extra 15-min timer period by default,
                                        # so auto-action ≈ deadline + 15-30 min,
                                        # strictly later than a manual `guard`
    # --- durable host reputation (vastlib.market.hostrep) -------------------- #
    # Every knob above forgets a bad host the moment its watch ends. These are
    # the DURABLE half: a decaying per-machine score, persisted across sessions,
    # that turns a condemnation into a standing preference against that host.
    # Owner directive 2026-08-20: "a cheap host that doesn't work is not worth
    # using for us -- paying more to have a fast boot time is worth it."
    "HOSTREP_HALF_LIFE_D": 14.0,        # a strike's weight halves every N days,
                                        # so the store forgives on its own and
                                        # no operator has to prune it. One
                                        # strike is worth 1.0 today, 0.5 in a
                                        # fortnight, 0.125 in six weeks.
    "HOSTREP_RECURRENCE_BONUS": 0.75,   # per EXTRA distinct day carrying a
                                        # strike (score x (1 + bonus*(days-1))).
                                        # This is the knob the owner directive
                                        # is actually about: four condemns in
                                        # one bad hour can be one transient
                                        # network event, but a host that fails
                                        # again three days later is failing
                                        # because of what it IS. Recurrence
                                        # across days is the strong evidence,
                                        # and only this term can see it.
    "HOSTREP_PENALTY_PER_POINT": 0.35,  # each point of score reads as +35% on
                                        # the machine's price when RANKING
                                        # offers (never when paying — see
                                        # hostrep.penalty). So one strike makes
                                        # a host lose to any clean host up to
                                        # 35% dearer, which is the directive
                                        # expressed as arithmetic.
    "HOSTREP_PENALTY_MAX": 4.0,         # ...bounded, so a bad host is deeply
                                        # unattractive but a market containing
                                        # only bad hosts still returns one. A
                                        # penalty is a preference, not a veto;
                                        # the veto is the block below.
    "HOSTREP_BLOCK_SCORE": 3.0,         # at/above this the machine is EXCLUDED
                                        # from automatic picks outright. Reached
                                        # by two strikes on two different days
                                        # (1.0 + ~0.86, x1.75 recurrence = 3.26)
                                        # — deliberately NOT by two strikes in
                                        # one session (2.0, no recurrence term).
    "HOSTREP_BLOCK_COOLDOWN_D": 14.0,   # a block, once earned, is held this long
                                        # even as the score decays under the
                                        # threshold. Without it a block bought
                                        # by recurrence evaporates in ~1.7 days
                                        # of decay and the next launch re-rents
                                        # the host we just condemned.
                                        #
                                        # 14 d, not 7 (owner, 2026-08-20: "if a
                                        # host fixes the issues, we eventually
                                        # retry it"). The cooldown is the RETRY
                                        # CLOCK, and it is set by how long a
                                        # host plausibly takes to fix itself —
                                        # a bad NIC, a saturated uplink, a
                                        # docker daemon pinned to one concurrent
                                        # download are operator-side changes on
                                        # a week-to-fortnight scale, not a
                                        # day one. Retrying sooner buys a
                                        # near-certain wasted rental; never
                                        # retrying buys a permanently shrinking
                                        # supply, which is the failure mode a
                                        # blocklist has and this must not.
                                        # Matched to HOSTREP_HALF_LIFE_D so the
                                        # retry lands exactly when the evidence
                                        # against the host has halved: it comes
                                        # back on PROBATION, still penalised
                                        # (~x1.56 for a two-strike block), so it
                                        # is picked only when clearly cheapest
                                        # and one more failure re-blocks it on a
                                        # third distinct day.
    "HOSTREP_MAX_STRIKES_KEPT": 40,     # per machine, newest-first. The score
                                        # is decayed, so entry 41 is worth
                                        # nothing; this bounds the file.
}


# The overloads exist so a caller's `cast` decides the return type (the default
# `float` for the ~30 bare call sites, `int`/`str` for the rest). They are pure
# typing: mypy rejects a TypeVar-parameterized parameter carrying a concrete
# default, so the implementation signature keeps the `= float` default and the
# overloads carry the precision. The `Any` return on the implementation is the
# honest one — the value's type IS whatever the caller's cast produces — hence
# the noqa the ruff config asks to be justified in place.
@overload
def _boot_knob(name: str, *, cli: object | None = ...) -> float: ...
@overload
def _boot_knob(name: str, *, cli: object | None = ...,
               cast: Callable[[Any], _KnobT]) -> _KnobT: ...
# moved-from: vastconf._boot_knob
def _boot_knob(name: str, *, cli: object | None = None,
               cast: Callable[[Any], Any] = float) -> Any:  # noqa: ANN401
    """Resolve a boot-health knob with precedence CLI > env > herdd.yaml >
    constant (the same convention the RCLONE_*/JOBD_* knobs use). `cli` is the
    already-parsed CLI value (None = flag not passed); `cast` coerces every
    source uniformly. A malformed env/yaml value is skipped, not fatal."""
    if cli is not None:
        return cast(cli)
    ev = os.environ.get(name)
    if ev not in (None, ""):
        try:
            return cast(ev)
        except (ValueError, TypeError):
            pass
    try:
        cfg = load_herdd_config()
    except Exception:
        cfg = {}
    if cfg.get(name) not in (None, ""):
        try:
            return cast(cfg[name])
        except (ValueError, TypeError):
            pass
    return cast(_BOOT_KNOB_DEFAULTS[name])


# --------------------------------------------------------------------------- #
# shared launch defaults
#
# One home for the container-disk defaults. Review doc §2 S3 found six --disk
# defaults across three values with no shared constant (herdd launch/supervise/
# train, launch_serve.sh, workflowctl's box body, fleetd's run-policy seed); the
# values below are those exact numbers, unchanged. The *estimator* that replaces
# hand-typed sizes is queued work (velvet P2/P4, future disksize.py) — this block
# only gives the constants a single source of truth.
# --------------------------------------------------------------------------- #
# moved-from: vastconf.DISK_DEFAULT_LAUNCH_GB
DISK_DEFAULT_LAUNCH_GB = 40
# moved-from: vastconf.DISK_DEFAULT_SUPERVISE_GB
DISK_DEFAULT_SUPERVISE_GB = 40
# moved-from: vastconf.DISK_DEFAULT_TRAIN_GB
DISK_DEFAULT_TRAIN_GB = 120
# moved-from: vastconf.DISK_DEFAULT_WORKFLOW_GB
DISK_DEFAULT_WORKFLOW_GB = 40
# moved-from: vastconf.DISK_DEFAULT_FLEETD_GB
DISK_DEFAULT_FLEETD_GB = 40
# moved-from: vastconf.DISK_DEFAULT_SERVE_GB
DISK_DEFAULT_SERVE_GB = 60   # launch_serve.sh FALLBACK only — since 2026-08-02 it
                             # auto-sizes from measured model bytes (disksize.
                             # serve_disk_gb) and reads this via python only when
                             # the model is unmeasurable (HF id / no B2 read)

# The other two launch-pick floors, beside the inet one (_BOOT_KNOB_DEFAULTS'
# LAUNCH_INET_DOWN_MBPS). They were bare argparse defaults in cli/search.py; the
# /admin market snapshot has to apply the SAME numbers to label an offer
# launch-reachable, and a second hardcoded copy there would drift silently into
# a floor no launch can actually rent (DESIGN_V10_MARKET_SHAPE.md §1.2).
# THE host-driver floor for every lane: CLI `--cuda`, the fleetd replacement
# rental, and the /admin market lens. It must TRACK THE IMAGE'S CUDA RUNTIME —
# `train-env/VLLM_PIN`'s `VLLM_TORCH` (today 2.13.0+cu129, Dockerfile.base.t213
# `CUDA_PKG_VER=12-9`), which is the CUDA-12.x family, and 12.8 is the floor that
# family is rented at (`launch_serve.sh` CUDA_MIN, and the reason cu129 was kept
# over cu130 in Dockerfile.base.t211:31-36 — cu130 would cost a driver>=580 host
# floor for zero x86_64 arch coverage). It was 13.0 until 2026-08-19, inherited
# from a stock `pip install vllm==0.24.0` that pulled a cu130 wheel; that lane is
# not what we ship, and the floor was excluding every 12.8-12.9 host for nothing.
# Raise it only when the image's CUDA runtime moves — never independently.
LAUNCH_CUDA_MAX_GOOD = 12.8    # image is cu129 -> CUDA-12 family floor (VLLM_PIN)
LAUNCH_RELIABILITY_MIN = 0.98  # host-uptime floor; the cheap tail is mostly below it


# moved-from: vastconf.default_disk_gb
def default_disk_gb(kind: str | None, *, cli: int | float | str | None = None) -> int:
    """Container disk in GB for a launch of `kind`, resolving CLI > env
    (HERDD_DEFAULT_DISK) > herdd.yaml `default_disk` > the per-kind constant
    above (velvet P4b).

    Why a per-USER override earns its keep: the constants encode one workstation's
    habits, and there was no way to change them short of editing the tree — unlike
    `train_gpu_ram`, which exists for exactly this purpose. `default_disk` sets the
    floor for every kind at once; a per-kind key (`default_disk_train`) wins over it.

    This is still a DEFAULT, not a derivation. The derived number comes from
    `disksize.estimate_disk_gb`, which needs a job-config to read; these paths
    launch a bare box that has no job yet, so the honest thing they can offer is a
    configurable default plus the submit-time advisory that follows.
    """
    key = str(kind or "").lower()
    const = {
        "launch": DISK_DEFAULT_LAUNCH_GB, "supervise": DISK_DEFAULT_SUPERVISE_GB,
        "train": DISK_DEFAULT_TRAIN_GB, "workflow": DISK_DEFAULT_WORKFLOW_GB,
        "fleetd": DISK_DEFAULT_FLEETD_GB, "serve": DISK_DEFAULT_SERVE_GB,
    }.get(key)
    if const is None:
        raise KeyError(f"unknown launch kind {kind!r}")
    if cli is not None:
        return int(cli)
    for src in (os.environ.get(f"HERDD_DEFAULT_DISK_{key.upper()}"),
                os.environ.get("HERDD_DEFAULT_DISK")):
        if src not in (None, ""):
            try:
                v = int(float(str(src)))
                if v > 0:
                    return v
            except (ValueError, TypeError):
                pass                      # malformed override is skipped, not fatal
    try:
        cfg = load_herdd_config()
    except Exception:
        cfg = {}
    for k in (f"default_disk_{key}", "default_disk"):
        if cfg.get(k) not in (None, ""):
            try:
                v = int(float(cfg[k]))
                if v > 0:
                    return v
            except (ValueError, TypeError):
                pass
    return const


# --------------------------------------------------------------------------- #
# ENV_SITES_TODO — the ~70 (measured: 105) stray `os.environ` reads the fat
# herdd.py carried when this inventory was taken, listed so a later port step
# can route them mechanically.
#
# THEY NO LONGER LIVE IN herdd.py. Step 6d thinned it to a launcher: measured
# 2026-08-16, `herdd.py` contains ZERO `os.environ`/`os.getenv` reads. Every
# row travelled with its OWNING FUNCTION into whichever vastlib module now hosts
# it (spread across ~30 files; the heaviest are launch/spec.py, workflows/ctl.py,
# jobs/bundle.py, boxes/reap.py). Keying the inventory by function rather than by
# line is what let it survive that move: look the function up in
# `.port_manifests/rename_table.json` and the row applies unchanged at its new
# home. Only the LOCATION prose was wrong — the rows themselves are still
# unrouted, and plan §9 still owns routing them.
#
# NOTHING below is implemented yet, deliberately. Plan §5 says these route
# "through config, name-preserving, same precedence"; the routing happens when
# the OWNING function ports, because an accessor built ahead of its caller either
# duplicates a precedence or sits dead. Each row is `owning function` ->
# `env var(s) it reads`, in the FAT herdd.py's source order (provenance now,
# not a location — the file no longer has these lines); `xN` is a repeat count
# (several functions read B2_BUCKET more than once). Line numbers are NOT
# recorded here — they drift with every rebase (that is what killed the v1 plan's
# citations); the authority is tools/vast/.port_manifests/config.json at rev
# 270ef4f1, which carries the exact ranges.
#
# Three of these are MODULE-LEVEL (`<module>` below: _LS_SNAPSHOT, _IDLE_LEDGER,
# _ZOMBIE_LEDGER), i.e. evaluated at import. Moving them inside a function would
# change WHEN the env is read — a test that setenv's XDG_CACHE_HOME after import
# cannot affect them today. That is a behavior change, not a port.
#
# Five read a non-literal key and are named accordingly (`<dynamic:…>`).
#
# Exactly three of these env sites also consult herdd.yaml, and NONE of
# them goes through _boot_knob — the asymmetry is deliberate and documented at
# both call sites:
#   _job_replacement_verified  ns > env JOB_REPLACEMENT_VERIFIED > yaml > True
#                              (bypasses _rebid_knob: bool("0") is True)
#   _rebid_knob                ns > env JOB_<NAME> > yaml JOB_<NAME> > bidpolicy
#   _job_replacement_knob      ns > env JOB_<NAME> > default   (NO yaml rung)
# And `default_image` / `train_gpu_ram` / `train_gpu` are yaml-ONLY knobs with no
# env twin: routing them through a resolver must not grow one.
#
#   api_key                     VASTAI_API_KEY, VAST_API_KEY
#   _api_key_soft               VASTAI_API_KEY, VAST_API_KEY
#   hf_token_text               HF_TOKEN|HUGGING_FACE_HUB_TOKEN|HUGGINGFACE_TOKEN
#   image_login_arg             REGISTRY_AUTH_SECRET
#   _emit_launched_soft         B2_BUCKET
#   _do_launch                  CRED_BROKER_URL x2, TS_AUTHKEY x2
#   _fold_fleet_jobs            XDG_CACHE_HOME
#   <module>                    XDG_CACHE_HOME x3
#   _color_on                   NO_COLOR, TERM
#   _jobd_ever_stamped          B2_BUCKET
#   _jobd_status_line_soft      B2_BUCKET
#   _scratch_probe_soft         B2_BUCKET
#   _fleet_image_states         (none — the registry-host question moved to
#                                imageref.is_our_registry, 2026-08-21)
#   _cli_actor                  HOSTNAME
#   _box_is_jobd                B2_BUCKET
#   _reap_durability_advisory   HERDD_REAP_DURABILITY
#   cmd_reap                    HERDD_REAP, HERDD_REAP_IDLE_H, HERDD_REAP_ZOMBIE,
#                               HERDD_REAP_STALL
#   _infra_cache_db             INFRA_METADATA_DB
#   _dash_reap_threshold_s      HERDD_REAP_IDLE_H
#   cmd_runs                    B2_BUCKET
#   _raw_events_soft            XDG_CACHE_HOME
#   _status_marker_soft         B2_BUCKET
#   _b2_eu_pairs                B2_KEY_ID_EU, B2_APPLICATION_KEY_EU, B2_S3_ENDPOINT_EU,
#                               B2_BUCKET_EU, B2_REGION_EU, B2_REGION_MODE
#   _r2_tc_pairs                R2_TC_KEY_ID, R2_TC_SECRET_ACCESS_KEY, R2_TC_ENDPOINT, R2_TC_BUCKET
#   _cdn_pairs                  B2_CDN_HOST, B2_CDN_BUCKET, B2_CDN_PREFIX
#   _ship_b2_pair               B2_MINTER_KEY_ID, B2_MINTER_APPLICATION_KEY, B2_BOX_KEY_ID,
#                               B2_BOX_APPLICATION_KEY, B2_KEY_ID, B2_APPLICATION_KEY
#   _ephemeral_hours            B2_EPHEMERAL_HOURS
#   _ship_b2_env                B2_MINTER_KEY_ID, B2_MINTER_APPLICATION_KEY
#   _revoke_box_keys            B2_MINTER_KEY_ID, B2_MINTER_APPLICATION_KEY
#   _resolve_secret             <dynamic:name>
#   _read_spec_soft             B2_BUCKET
#   _reset_run_markers          B2_BUCKET
#   _handoff_b2_write           B2_BUCKET
#   _handoff_synced_epoch_soft  B2_BUCKET
#   cmd_train                   B2_BUCKET|B2_KEY_ID|B2_APPLICATION_KEY|B2_S3_ENDPOINT, B2_BUCKET,
#                               B2_S3_ENDPOINT, B2_REGION, HF_TOKEN x2,
#                               LLM_BASE_URL|LLM_API_KEY|OPENROUTER_API_KEY x2
#   _submit_disk_advisory       HERDD_DISK_ADVISORY
#   cmd_job_logs                B2_BUCKET
#   _jobd_import_gate           JOBD_NO_IMPORT_CHECK
#   _stage_jobd_bootstrap       B2_BUCKET
#   compose_jobs_launch_env     B2_BUCKET, B2_S3_ENDPOINT, B2_REGION, CRED_BROKER_URL x2,
#                               TS_AUTHKEY x2
#   _broker_register            CRED_BROKER_URL, CRED_BROKER_ADMIN_TOKEN
#   cmd_job_attach              B2_BUCKET, B2_S3_ENDPOINT, B2_REGION, CRED_BROKER_URL x2
#   _serve_self_park_soft       B2_BUCKET
#   _handoff_job_b2_write       B2_BUCKET
#   _raw_job_events_soft        XDG_CACHE_HOME
#   _job_replacement_verified   JOB_REPLACEMENT_VERIFIED
#   _serve_status_line_soft     B2_BUCKET
#   _serve_relaunch_dir         XDG_STATE_HOME
#   _job_replacement_knob       <dynamic:JOB_{name.upper()}>
#   _rebid_knob                 <dynamic:JOB_{name.upper()}>
#   _salvage_b2_bytes           B2_BUCKET
#   _salvage_push_to_b2         B2_BUCKET
#   _salvage_enabled            SALVAGE_ENABLED
#   _replacement_cuda_floor     REPLACEMENT_CUDA_FLOOR
#   fleet_state_dir             FLEETD_STATE_DIR
#   fleet_sock_path             FLEETD_SOCK
#   _fleet_delegation_disabled  FLEETD_DISABLE, PYTEST_CURRENT_TEST
#   _fleet_requester            USER
# --------------------------------------------------------------------------- #

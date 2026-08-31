#!/usr/bin/env python3
"""boxstate.py — read-only, SSH-free "what is this vast box doing?" report.

This is the tool to reach for INSTEAD OF `ssh in and poke around`. It answers
"where is this box, right now?" by merging two authorities and NEVER touching the
box itself:

  * the vast.ai API instance record — reality (actual_status, the live
    docker-build/pull `status_msg` during provisioning, gpu, $/hr, geo, age); and
  * the run's Backblaze-B2 artifacts — history (checkpoints/<RUN_ID>/STATUS, the
    new boot_phases.tsv timeline + incremental onstart.log, and the append-only
    runs/<RUN_ID>/events/ runmeta log), PLUS the jobs-v2 layout under the SAME
    id (jobs/<ID>/events/, results/, results.DONE.json, CANCEL, log.txt —
    JOBS_DESIGN.md) — both namespaces are always probed; whichever actually
    has signal drives the report/VERDICT.

It is strictly read-only: no SSH, no writes, no instance mutation. Non-zero exit
is reserved for "cannot even query" (both the vast API AND B2 are unreachable);
a pessimistic VERDICT never changes the exit code.

USAGE
  boxstate.py                     # table of ALL live instances + their B2 status
  boxstate.py <RUN_ID>            # full report for one run (e.g. modelzoo-reader-06-eval)
  boxstate.py <INSTANCE_ID>       # full report, resolved via the box's run:<ID> label
  boxstate.py <JOB_ID>            # jobs-v2: the box comes from the job's own
                                  #   events (a jobs box has no run:<ID> label)
  boxstate.py <target> --json     # machine-readable full dump (for agents)
  boxstate.py <target> --tail 40  # show the last 40 lines of onstart.log (default 20)

CREDENTIALS
  VASTAI_API_KEY + B2_* come from the nearest .env walking up from CWD (reuses
  herdd.load_env). B2 reads go through the user's existing rclone `[b2]` remote
  (the same one b2_sync.sh writes to ~/.config/rclone/rclone.conf); if that remote
  is missing we best-effort `b2_sync.sh config` once, else the B2 sections degrade
  to "(B2 unreachable)".

DEGRADATION
  Boxes launched with the OLDER onstart (or a box still mid-provision, before
  anything lands on B2) simply won't have boot_phases.tsv / onstart.log / an event
  log yet. Every section degrades independently and says what is absent and why
  that is expected during provisioning — it never errors on a missing artifact.

Stdlib only (urllib/json/subprocess/argparse). Reuses herdd.py's helpers by
import (no argparse `main` is invoked at import time; herdd guards it under
`if __name__`), with local fallbacks if the import fails from a foreign CWD.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import subprocess
import sys
import time

# --------------------------------------------------------------------------- #
# helper binding — prefer herdd's (avoid drift), fall back if import fails
# --------------------------------------------------------------------------- #
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

try:                                   # importing herdd has no side effects
    import herdd                     # (its argparse main is __main__-guarded)
    _HAVE_HERDD = True
except Exception as _e:                # foreign CWD / partial checkout
    herdd = None
    _HAVE_HERDD = False
    _HERDD_ERR = _e

API = "https://console.vast.ai/api"
LIVE_STATES = {"running", "loading", "created"}
STALE_S = 300                          # STATUS/events not refreshed within this -> flag


if _HAVE_HERDD:
    load_env = herdd.load_env
    request_soft = herdd.request_soft
    rclone_soft = herdd._rclone_soft
    instances_soft = herdd._instances_soft
    instance_run_label = herdd._instance_run_label
    dollars = herdd.dollars
else:
    # ---- minimal, self-contained fallbacks (only the bits boxstate needs) ---
    import urllib.error
    import urllib.request

    def load_env() -> None:
        d = os.getcwd()
        for _ in range(6):
            p = os.path.join(d, ".env")
            if os.path.isfile(p):
                for line in open(p):
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    v = v.split(" #", 1)[0].strip().strip('"').strip("'")
                    os.environ.setdefault(k.strip(), v)
                return
            nd = os.path.dirname(d)
            if nd == d:
                break
            d = nd

    def request_soft(method, path, body=None, timeout=60):
        key = os.environ.get("VASTAI_API_KEY") or os.environ.get("VAST_API_KEY")
        if not key:
            return False, None, "config: VASTAI_API_KEY not set (env or .env)"
        data = json.dumps(body).encode() if body is not None else None
        url = f"{API}/{path.lstrip('/')}"
        req = urllib.request.Request(
            url, data=data, method=method,
            headers={"Authorization": "Bearer " + key,
                     "Content-Type": "application/json",
                     "Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                raw = r.read().decode()
            return True, (json.loads(raw) if raw else {}), None
        except urllib.error.HTTPError as e:
            return False, None, f"HTTP {e.code} on {method} {path}"
        except Exception as e:  # noqa: BLE001
            return False, None, f"error {e} on {method} {path}"

    def rclone_soft(args):
        try:
            r = subprocess.run(["rclone", *args], capture_output=True, text=True)
            return r.returncode, r.stdout, r.stderr
        except FileNotFoundError:
            return 127, "", "rclone not found on PATH"

    def instances_soft():
        ok, d, _ = request_soft("GET", "v1/instances/")
        if not ok:
            return []
        return d.get("instances", d) if isinstance(d, dict) else d

    def instance_run_label(i):
        # Strip the reap `keep` token that fleetd appends to every box it parks,
        # but KEEP a semantic suffix like `:handoff` (the understudy's distinct
        # run id). Mirrors herdd._label_value, which carries the full
        # rationale; this copy exists only for the no-herdd fallback path.
        lbl = i.get("label") or ""
        if not lbl.startswith("run:"):
            return None
        kept = []
        for tok in lbl[4:].split(":"):
            if tok.strip().lower() == "keep":
                break
            kept.append(tok)
        return ":".join(kept) or None

    def dollars(x):
        try:
            return f"${float(x):.3f}"
        except (TypeError, ValueError):
            return "$?"


# --------------------------------------------------------------------------- #
# time helpers
# --------------------------------------------------------------------------- #
def _now() -> float:
    return time.time()


def _fmt_age(sec) -> str:
    """Compact human age: 45s / 12m / 1h05m / 2d03h."""
    if sec is None:
        return "?"
    sec = int(max(0, sec))
    if sec < 60:
        return f"{sec}s"
    m, s = divmod(sec, 60)
    if m < 60:
        return f"{m}m"
    h, m = divmod(m, 60)
    if h < 24:
        return f"{h}h{m:02d}m"
    d, h = divmod(h, 24)
    return f"{d}d{h:02d}h"


def _epoch_age_s(epoch):
    try:
        return _now() - float(epoch)
    except (TypeError, ValueError):
        return None


def _iso_age_s(ts):
    """Age (s) from a STATUS-marker ISO ts 'YYYY-MM-DDTHH:MM:SSZ' (has colons).
    Scans for the LAST such token so 'FAILED ... 2026-07-09T09:51:02Z' works."""
    if not isinstance(ts, str):
        return None
    m = re.findall(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", ts)
    if not m:
        return None
    try:
        dt = datetime.datetime.strptime(m[-1], "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=datetime.timezone.utc)
        return (datetime.datetime.now(datetime.timezone.utc) - dt).total_seconds()
    except Exception:  # noqa: BLE001
        return None


def _runmeta_ts_age_s(ts):
    """Age (s) from a runmeta event ts 'YYYYMMDDTHHMMSSmmmZ' (no colons)."""
    if _HAVE_HERDD:
        try:
            return herdd._ts_age_s(ts)
        except Exception:  # noqa: BLE001
            pass
    if not isinstance(ts, str) or not ts:
        return None
    try:
        dt = datetime.datetime.strptime(ts.rstrip("Z")[:15], "%Y%m%dT%H%M%S").replace(
            tzinfo=datetime.timezone.utc)
        return (datetime.datetime.now(datetime.timezone.utc) - dt).total_seconds()
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------------- #
# B2 access (read-only, via the user's rclone [b2] remote)
# --------------------------------------------------------------------------- #
class B2:
    """Thin read-only B2 accessor. `ok` is False when B2 is wholly unreachable
    (no rclone / no bucket / no [b2] remote) — distinct from a single missing
    object (rc != 0 but the remote is fine)."""

    def __init__(self):
        self.bucket = os.environ.get("B2_BUCKET")
        self.ok = False
        self.reason = None
        if not self.bucket:
            self.reason = "B2_BUCKET not set (env or .env)"
            return
        rc, out, err = rclone_soft(["listremotes"])
        if rc == 127:
            self.reason = "rclone not found on PATH"
            return
        if "b2:" not in (out or ""):
            # best-effort one-shot config via b2_sync.sh, then re-check
            script = os.path.join(_HERE, "b2_sync.sh")
            if os.path.isfile(script):
                subprocess.run(["bash", script, "config"],
                               capture_output=True, text=True)
            rc, out, _ = rclone_soft(["listremotes"])
            if "b2:" not in (out or ""):
                self.reason = "rclone [b2] remote not configured"
                return
        self.ok = True

    def _p(self, path):
        return f"b2:{self.bucket}/{path.lstrip('/')}"

    def cat(self, path):
        """(present, text). present=False means the object is absent/unreadable.

        NOTE: `rclone cat` on a MISSING B2/S3 object exits rc=0 with empty stdout
        (the 404 goes to stderr), so rc alone can't tell present from absent — we
        treat empty output as absent. A legitimately-empty marker has no content
        to show anyway, so conflating the two is honest here."""
        if not self.ok:
            return False, None
        rc, out, _ = rclone_soft(["cat", self._p(path)])
        if rc != 0 or not out:
            return False, None
        return True, out

    def lsf(self, path):
        """(present, [names]). Same rc=0-for-missing caveat as cat(): a missing
        dir lists nothing at rc=0, so presence is 'has at least one name'. Also
        works on a single-object path (not just a 'directory' prefix) — `rclone
        lsf b2:bucket/key` lists just that key if present, nothing if absent."""
        if not self.ok:
            return False, []
        rc, out, _ = rclone_soft(["lsf", self._p(path)])
        if rc != 0:
            return False, []
        names = [x.rstrip() for x in (out or "").splitlines() if x.strip()]
        return bool(names), names

    def stat(self, path):
        """(present, size_bytes). Single-object probe via `rclone lsf -F sp`
        (size;name) — a presence+size check without downloading content, for
        markers we only need to know are THERE (CANCEL) or how big (log.txt)."""
        if not self.ok:
            return False, None
        rc, out, _ = rclone_soft(["lsf", self._p(path), "-F", "sp"])
        if rc != 0 or not (out or "").strip():
            return False, None
        size_s, _, _name = out.strip().splitlines()[0].partition(";")
        try:
            size = int(size_s)
        except ValueError:
            size = None
        return True, size


# --------------------------------------------------------------------------- #
# STATUS / boot_phases / events parsing (pure)
# --------------------------------------------------------------------------- #
def parse_status(text):
    """Parse a checkpoints/<id>/STATUS marker into
    {word, phase, ts, age_s, raw}. word is the leading token upper-cased
    (RUNNING/LAUNCHED/DONE/FAILED/STAGED/...); phase from a `phase=<name>` token
    (new); ts/age from the last ISO token. None if empty/absent."""
    s = (text or "").strip()
    if not s:
        return None
    word = s.split(None, 1)[0].upper()
    mphase = re.search(r"phase=(\S+)", s)
    mts = re.findall(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", s)
    return {
        "word": word,
        "phase": mphase.group(1) if mphase else None,
        "ts": mts[-1] if mts else None,
        "age_s": _iso_age_s(s),
        "raw": s,
    }


def parse_boot_phases(text):
    """Parse boot_phases.tsv rows (epoch\\telapsed\\tphase\\tdetail) into a
    timeline with per-phase durations. The last row is the CURRENT phase; its
    duration is measured against 'now'. [] if absent/empty."""
    rows = []
    for line in (text or "").splitlines():
        line = line.rstrip("\n")
        if not line.strip():
            continue
        cols = line.split("\t")
        if len(cols) < 3:
            continue
        epoch = None
        try:
            epoch = float(cols[0])
        except (TypeError, ValueError):
            pass
        rows.append({
            "epoch": epoch,
            "elapsed": cols[1] if len(cols) > 1 else "",
            "phase": cols[2] if len(cols) > 2 else "",
            "detail": cols[3] if len(cols) > 3 else "",
        })
    # per-phase duration = delta to next row's epoch; last = now - epoch (current)
    for i, r in enumerate(rows):
        if r["epoch"] is None:
            r["dur_s"] = None
            r["current"] = (i == len(rows) - 1)
            continue
        if i < len(rows) - 1 and rows[i + 1]["epoch"] is not None:
            r["dur_s"] = rows[i + 1]["epoch"] - r["epoch"]
            r["current"] = False
        else:
            r["dur_s"] = _epoch_age_s(r["epoch"])
            r["current"] = True
    return rows


def _gather_events_at(b2, prefix, last_n=5):
    """(present, total_count, [last_n event dicts]) under a B2 `.../events/`
    prefix. Shared by the legacy runmeta (`runs/<id>/events/`) and jobs-v2
    (`jobs/<id>/events/`) namespaces — both use the same
    `<ts>-<actor>-<nonce>.json` convention (runmeta.event_key / jobmeta.event_key),
    so filenames sort lexically == chronologically and the tail is the newest."""
    present, names = b2.lsf(prefix)
    if not present:
        return False, 0, []
    files = sorted(n for n in names if n.endswith(".json"))
    tail = files[-last_n:]
    evs = []
    for f in tail:
        ok, body = b2.cat(f"{prefix}{f}")
        if ok and body:
            try:
                evs.append(json.loads(body))
            except (ValueError, TypeError):
                pass
    return True, len(files), evs


def gather_events(b2, run_id, last_n=5):
    """(present, total_count, [last_n event dicts]) from the legacy
    runs/<run_id>/events/ runmeta log."""
    return _gather_events_at(b2, f"runs/{run_id}/events/", last_n)


def gather_job_events(b2, job_id, last_n=5):
    """(present, total_count, [last_n event dicts]) from the jobs-v2
    jobs/<job_id>/events/ log (JOBS_DESIGN.md; envelope = jobmeta.make_event)."""
    return _gather_events_at(b2, f"jobs/{job_id}/events/", last_n)


def parse_results_done(text):
    """Parse jobs/<job_id>/results.DONE.json (jobd.sh's marker-last publish
    step; JOBS_DESIGN.md) into {rc, duration_s, n_results, raw}. None if
    absent/unparseable — this is a probe, not a validator."""
    if not text:
        return None
    try:
        d = json.loads(text)
    except (ValueError, TypeError):
        return None
    return {
        "rc": d.get("rc"),
        "duration_s": d.get("duration_s"),
        "n_results": d.get("n_results"),
        "raw": d,
    }


def parse_cancel(text):
    """Parse the jobs/<job_id>/CANCEL marker (jobmeta.write_cancel_marker) into
    {ts, age_s, actor, reason, raw}. Falls back to a bare presence dict if the
    body isn't valid JSON (still a positive signal — the marker's mere
    existence is what jobd's cancel-watch acts on). None if absent."""
    if not text:
        return None
    try:
        d = json.loads(text)
    except (ValueError, TypeError):
        return {"ts": None, "age_s": None, "actor": None, "reason": None, "raw": text}
    return {
        "ts": d.get("ts"),
        "age_s": _runmeta_ts_age_s(d.get("ts")),
        "actor": d.get("actor"),
        "reason": d.get("reason"),
        "raw": text,
    }


def gather_jobs2(b2, job_id):
    """All jobs-v2 B2-side signals for one job id — jobs/<job_id>/ (JOBS_DESIGN.md
    layout: events/, results/, results.DONE.json, CANCEL, log.txt). Every field
    degrades independently, same discipline as gather_b2(). Always probed
    alongside the legacy runs/<id>/ namespace (gather_b2) — a run_id/job_id is
    the same string on the wire (the vast instance label is `run:<id>` either
    way), so rather than pattern-match which layout an id 'looks like' we just
    check both and let VERDICT prefer whichever actually has signal."""
    out = {"reachable": b2.ok, "reason": b2.reason,
           "events_present": False, "events_total": 0, "events": [],
           "results_present": False, "results": [],
           "done": None, "cancel": None,
           "log_present": False, "log_size": None}
    if not b2.ok:
        return out
    ev_present, ev_total, evs = gather_job_events(b2, job_id, last_n=5)
    out["events_present"] = ev_present
    out["events_total"] = ev_total
    out["events"] = evs
    # NOTE: jobs/<id>/results/ holds FINALIZED results only (written once at job
    # end). Mid-run checkpoints live under jobs/<id>/checkpoints/, so an
    # in-progress checkpointing job shows results_present=False here until it
    # finalizes — that is expected (live progress is read from events/heartbeats,
    # not this listing). The terminal read path (post-DONE) is unchanged.
    res_present, res_names = b2.lsf(f"jobs/{job_id}/results/")
    out["results_present"] = res_present
    out["results"] = res_names
    _, done_txt = b2.cat(f"jobs/{job_id}/results.DONE.json")
    out["done"] = parse_results_done(done_txt)
    _, cancel_txt = b2.cat(f"jobs/{job_id}/CANCEL")
    out["cancel"] = parse_cancel(cancel_txt)
    out["log_present"], out["log_size"] = b2.stat(f"jobs/{job_id}/log.txt")
    return out


def jobs_box_instance_id(b2sec2):
    """The box carrying this job, folded from the job's OWN events, or None.

    WHY THIS EXISTS (BOX_SATURATION_AUDIT_2026-07-30 §6 red flag 5): a jobs-v2
    box is NOT labelled `run:<job-id>`. `herdd launch --jobs` labels it however
    the operator asked and marks it with `CRED_ROLE=jobs`; ONE jobd then serves
    many job ids off `jobs/queue/<IID>/`, so a label could not name them all.
    Label resolution therefore finds nothing and every running job on such a box
    reported `(no live vast instance carries label run:<job-id>)` with a verdict
    of `stopped:` — wrong headline for the designated no-ssh diagnostic.

    The honest mapping is the one jobmeta.fold_events uses: jobd stamps
    `instance_id` on the events it emits (claimed / started / heartbeat /
    resumed), so the newest event that carries one names the box. Failing that,
    ANY box-emitted event's `actor` is `box:<IID>` (jobmeta.default_actor), which
    covers a tail made entirely of event kinds that omit the field.

    Read-only, no heuristics, no extra B2 reads — this folds the newest-events
    tail gather_jobs2 already fetched. A tail of pre-claim `cli:`-actor
    `submitted` events yields None, correctly: nothing has claimed the job, so no
    box carries it.
    """
    evs = (b2sec2 or {}).get("events") or []
    for e in reversed(evs):
        iid = e.get("instance_id")
        if iid:
            return str(iid)
    for e in reversed(evs):
        actor = str(e.get("actor") or "")
        if actor.startswith("box:") and actor[4:].strip().isdigit():
            return actor[4:].strip()
    return None


def resolve_jobs_box(b2sec2, instances):
    """(instance_dict_or_None, iid_or_None) for a jobs-v2 job id — the box its
    event log names, looked up in the live instance list (falling back to a
    direct fetch, so a job on a box that has since exited still resolves and
    reports honestly rather than as 'no instance')."""
    iid = jobs_box_instance_id(b2sec2)
    if not iid:
        return None, None
    for i in instances or ():
        if str(i.get("id")) == iid:
            return i, iid
    inst, _ = get_instance_soft(iid)
    return (inst or None), iid


def jobs2_has_signal(b2sec2):
    """Is there ANY jobs-v2 evidence for this id? Gates whether the JOBS-V2
    report section (and its pull on VERDICT) fires at all — a plain non-jobs-v2
    run correctly shows nothing here."""
    b2sec2 = b2sec2 or {}
    return bool(b2sec2.get("events_present") or b2sec2.get("results_present")
                or b2sec2.get("done") or b2sec2.get("cancel"))


# --------------------------------------------------------------------------- #
# gather: vast + b2
# --------------------------------------------------------------------------- #
_VAST_FIELDS = ("id", "actual_status", "intended_status", "status_msg",
                "label", "gpu_name", "num_gpus", "dph_total", "start_date",
                "machine_id", "geolocation", "public_ipaddr", "inet_down",
                "inet_up", "ssh_host", "ssh_port",
                # The only liveness a box with no event emitter has. BOTH are
                # needed: a compile box pins gpu_util at 0 while saturating its
                # cores, a serve box does the reverse.
                "cpu_util", "gpu_util", "cpu_cores_effective")


def _slim_instance(i):
    d = {k: i.get(k) for k in _VAST_FIELDS}
    d["run_id"] = instance_run_label(i)
    d["age_s"] = _epoch_age_s(i.get("start_date"))
    d["live"] = (i.get("actual_status") or "").lower() in LIVE_STATES
    return d


def get_instance_soft(iid):
    ok, d, err = request_soft("GET", f"v0/instances/{iid}/")
    if not ok:
        return None, err
    inst = d.get("instances", d) if isinstance(d, dict) else d
    return (inst or None), (None if inst else "instance not found")


def gather_b2(b2, run_id, tail_n):
    """All B2-side signals for one run. Every field degrades independently."""
    out = {"reachable": b2.ok, "reason": b2.reason,
           "status": None, "boot_phases": None, "boot_phases_present": False,
           "onstart_tail": None, "onstart_present": False,
           "events_present": False, "events_total": 0, "events": []}
    if not b2.ok:
        return out
    _, status_txt = b2.cat(f"checkpoints/{run_id}/STATUS")
    out["status"] = parse_status(status_txt)
    bp_present, bp_txt = b2.cat(f"checkpoints/{run_id}/boot_phases.tsv")
    out["boot_phases_present"] = bp_present
    out["boot_phases"] = parse_boot_phases(bp_txt) if bp_present else []
    log_present, log_txt = b2.cat(f"checkpoints/{run_id}/onstart.log")
    out["onstart_present"] = log_present
    if log_present:
        lines = (log_txt or "").splitlines()
        out["onstart_tail"] = lines[-tail_n:]
    ev_present, ev_total, evs = gather_events(b2, run_id, last_n=5)
    out["events_present"] = ev_present
    out["events_total"] = ev_total
    out["events"] = evs
    return out


# --------------------------------------------------------------------------- #
# VERDICT — rule-based synthesis, honest about unknowns
# --------------------------------------------------------------------------- #

# Past this age, "no B2 events" stops meaning "still booting" and starts
# meaning "this box has no emitter". Generous on purpose: the cost of guessing
# EMITTER too early is telling someone a genuinely stalled boot is fine.
_EMITTER_GRACE_S = 1800.0


def _util_phrase(vast):
    """CPU/GPU busy-ness for a box with no event log, from the vast payload.

    Both fields are reported when present, because which one matters depends on
    the box: a compile/search box pins gpu_util at 0 while saturating cores, and
    a serve box does the reverse. Absent is printed as `?`, never as 0 — vast
    does not always populate these, and a missing reading is not an idle one.
    """
    v = vast or {}
    cpu, gpu = v.get("cpu_util"), v.get("gpu_util")
    c = f"{cpu:.2f}" if isinstance(cpu, (int, float)) else "?"
    g = f"{gpu:.0f}%" if isinstance(gpu, (int, float)) else "?"
    return f"cpu {c}, gpu {g}"


def _last_status_msg_lines(vast, n=3):
    msg = (vast or {}).get("status_msg") or ""
    lines = [ln for ln in msg.splitlines() if ln.strip()]
    return lines[-n:]


def _pull_bytes_total(vast):
    """Total image-pull bytes SEEN so far, folded from the docker-pull layer
    counters in `status_msg` via herdd.parse_pull_progress. 0 when herdd is
    unavailable, status_msg carries no layer lines (apt/provision phase), or the
    parse yields nothing. SINGLE-snapshot: position, not rate — a rate needs two
    polls (that is the boot_health_watch job, not this read-only surface)."""
    if not _HAVE_HERDD or not vast:
        return 0
    try:
        prog = herdd.parse_pull_progress(vast.get("status_msg") or "", {})
    except Exception:
        return 0
    return int(prog.get("total_bytes") or 0)


def _fmt_bytes(n):
    for unit, div in (("GB", 1e9), ("MB", 1e6), ("kB", 1e3)):
        if n >= div:
            return f"{n / div:.1f}{unit}"
    return f"{int(n)}B"


def _pull_bytes_line(vast):
    """The ` · pull: N.NGB seen (single snapshot)` suffix for the provisioning
    verdict — surface, not decision. Empty string when no pull bytes are
    visible yet (registry auth / manifest negotiation, apt phase)."""
    n = _pull_bytes_total(vast)
    if n <= 0:
        return ""
    return (f" · pull: {_fmt_bytes(n)} seen, rate since last poll unknown "
            f"(single snapshot)")


def _latest_event(b2sec):
    evs = b2sec.get("events") or []
    return evs[-1] if evs else None


def verdict(vast, b2sec, b2sec2=None):
    """One synthesized line: where is the box + confidence, from vast reality +
    B2 history. Never raises; says UNKNOWN rather than guessing.

    `b2sec2` is the jobs-v2 lane (gather_jobs2, jobs/<id>/ layout) — optional
    (defaults to no signal) so existing callers keep working. It only ever
    FILLS IN gaps the legacy runs/<id>/ signals (`b2sec`) leave open: a legacy
    STATUS/event hit always keeps deciding first, since that mechanism is more
    specific (a per-word STATUS marker) than the jobs-v2 event stream."""
    b2sec2 = b2sec2 or {}
    vs = (vast.get("actual_status") or "").lower() if vast else None
    age = _fmt_age(vast.get("age_s")) if vast else "?"
    status = b2sec.get("status")
    ev = _latest_event(b2sec)
    ev_age = _runmeta_ts_age_s(ev.get("ts")) if ev else None
    boot = b2sec.get("boot_phases") or []
    cur_phase = boot[-1] if boot else None
    cancel2 = b2sec2.get("cancel")
    done2 = b2sec2.get("done")
    j2ev = _latest_event(b2sec2)
    j2ev_age = _runmeta_ts_age_s(j2ev.get("ts")) if j2ev else None

    # 0. nothing knowable
    if not vast and not b2sec.get("reachable"):
        return "UNKNOWN: cannot query vast API and B2 is unreachable"
    if not vast and (status is None and not b2sec.get("events")
                     and not jobs2_has_signal(b2sec2)):
        return "UNKNOWN: no live vast instance and no B2 artifacts for this run"

    # 1. provisioning (vast is loading/created — image build/pull, pre-onstart).
    # This phase is GPU-UNBILLED (invoice-verified 2026-07-20 + 2026-08-02:
    # storage only); billing starts at the loading→running flip, where onstart
    # (the env-setup phase we own) begins.
    if vs == "loading":
        tail = _last_status_msg_lines(vast, 1)
        d = tail[0].strip() if tail else "docker image build/pull"
        pull = _pull_bytes_line(vast)
        return f"provisioning: {d} (loading, {age}, GPU unbilled){pull}"
    if vs == "created":
        return (f"provisioning: container created, starting "
                f"(created, {age}, GPU unbilled)")

    # 2. box gone (exited/offline/None) — infer terminal from STATUS, else
    #    (jobs-v2) CANCEL / results.DONE.json
    if vs in (None, "exited", "offline", "stopped", "unknown"):
        if status and status["word"] == "DONE":
            return f"finished: run DONE, box gone ({_fmt_age(status['age_s'])} ago)"
        if status and status["word"] in ("FAILED",):
            return f"failed: box gone; {status['raw']}"
        if status:
            return (f"stopped: box not live; last STATUS {status['word']} "
                    f"({_fmt_age(status['age_s'])} ago)")
        if cancel2:
            return (f"cancelled: job cancelled ({cancel2.get('reason') or 'operator'}), "
                    f"box gone ({_fmt_age(cancel2.get('age_s'))} ago)")
        if done2:
            ok2 = done2.get("rc") in (0, None)
            return (f"{'finished' if ok2 else 'failed'}: job "
                    f"{'DONE' if ok2 else 'FAILED'} rc={done2.get('rc')} "
                    f"in {_fmt_age(done2.get('duration_s'))}, box gone")
        if jobs2_has_signal(b2sec2):
            # A jobs-v2 job with no live box is either INTERRUPTED (a box claimed
            # it and went away — jobd picks it back up on the next boot) or still
            # QUEUED (nothing has claimed it). Its own events say which; "stopped"
            # said neither.
            jbox = jobs_box_instance_id(b2sec2)
            if jbox:
                return (f"interrupted: jobs-v2 job was claimed by box {jbox}, which "
                        f"is not live now ({b2sec2.get('events_total')} events, no "
                        f"terminal marker) — jobd resumes it on that box's next boot")
            return ("queued: jobs-v2 job submitted, no box has claimed it yet "
                    f"({b2sec2.get('events_total')} events)")
        return "stopped: box not live and no STATUS marker on B2"

    # 3. box is running
    if vs == "running":
        if status and status["word"] == "DONE":
            return "finished: STATUS DONE but box still up (teardown pending)"
        if status and status["word"] == "FAILED":
            return f"failed: box still up (debug-hold?); {status['raw']}"
        if cancel2:
            return (f"cancelled: job cancelled ({cancel2.get('reason') or 'operator'}), "
                    f"box still up ({_fmt_age(cancel2.get('age_s'))} ago)")
        if done2:
            ok2 = done2.get("rc") in (0, None)
            return (f"{'finished' if ok2 else 'failed'}: job "
                    f"{'DONE' if ok2 else 'FAILED'} rc={done2.get('rc')}, "
                    f"box still up (teardown pending)")

        # 3a. still booting: a current boot phase that isn't the train loop
        if cur_phase and cur_phase.get("phase") and \
                cur_phase["phase"].lower() not in ("train", "training", "done", "ready"):
            dur = _fmt_age(cur_phase.get("dur_s"))
            extra = ""
            if b2sec.get("onstart_tail"):
                extra = f"; last log: {b2sec['onstart_tail'][-1].strip()[:80]}"
            return f"boot: phase {cur_phase['phase']} ({dur}){extra}"

        # 3b. staleness — BOTH the STATUS marker and the event log gone quiet
        status_stale = status is not None and status.get("age_s") is not None \
            and status["age_s"] > STALE_S
        events_stale = (ev_age is None) or (ev_age > STALE_S)
        if status_stale and events_stale and not b2sec.get("events"):
            return (f"STALE: STATUS not refreshed in {_fmt_age(status['age_s'])} "
                    f"and no events — box may be wedged")
        if status_stale and events_stale:
            return (f"STALE?: box live but STATUS {_fmt_age(status['age_s'])} old "
                    f"and last event {_fmt_age(ev_age)} old — check progress")

        # 3c. actively training / serving
        if ev:
            what = ev.get("event", "?")
            step = None
            for e in reversed(b2sec.get("events") or []):
                if e.get("event") == "checkpoint" and e.get("step") is not None:
                    step = e.get("step")
                    break
            if step is not None:
                return (f"training: checkpoint step {step}, last event {what} "
                        f"({_fmt_age(ev_age)} ago)")
            return (f"running: box live, last event {what} "
                    f"({_fmt_age(ev_age)} ago)")
        if status and status["word"] in ("RUNNING", "LAUNCHED"):
            return (f"running: box live, STATUS {status['word']} "
                    f"({_fmt_age(status['age_s'])} ago); no event log")

        # 3d. jobs-v2 events — only reached when the legacy STATUS/event log
        #     (above) had nothing to say
        if j2ev:
            j2_stale = (j2ev_age is None) or (j2ev_age > STALE_S)
            box = f" on box {vast['id']}" if vast and vast.get("id") else ""
            if j2_stale:
                return (f"STALE?: jobs-v2 job live{box} but last event "
                        f"{_fmt_age(j2ev_age)} old — check progress")
            return (f"running: jobs-v2 job{box}, {b2sec2.get('events_total')} events, "
                    f"last {j2ev.get('event', '?')} ({_fmt_age(j2ev_age)} ago)")
        # An ABSENT EMITTER is not absent progress. The B2 event log is written
        # by the jobs-v2 bundle wrapper, so a box rented directly (`launch`,
        # a serve box, a hand-driven eval) never writes one however hard it is
        # working — and "early boot?" on a box hours past boot sent a reader
        # hunting for a stall that was not there. Say which case this is, and
        # hand over the utilization the vast payload already carries.
        boot_s = (vast or {}).get("age_s")
        if boot_s is not None and boot_s > _EMITTER_GRACE_S:
            return (f"running: box live ({age}); no event emitter — rented "
                    f"outside jobs-v2, so there is no B2 log to read. "
                    f"Liveness from the vast payload: {_util_phrase(vast)}")
        return f"running: box live ({age}); no B2 progress signal yet (early boot?)"

    return f"unknown: vast actual_status={vs!r}"


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #
def _geo_short(geo):
    if not geo:
        return "?"
    # vast gives "Thailand, TH" — prefer the 2-letter code
    if "," in geo:
        return geo.rsplit(",", 1)[1].strip()[:6]
    return geo[:6]


def print_table(instances, b2):
    if not instances:
        print("no live instances.")
        return
    print(f"  {'ID':>10}  {'RUN / LABEL':<26} {'STATE':<9} {'GEO':<4} "
          f"{'GPU':<15} {'$/HR':>7} {'AGE':>6}  B2-STATUS")
    for i in instances:
        s = _slim_instance(i)
        rid = s["run_id"]
        run_col = rid or (s["label"] or "-")
        gpu = f"{s['num_gpus'] or '?'}x {s['gpu_name'] or '?'}"
        b2col = "-"
        if rid and b2.ok:
            _, txt = b2.cat(f"checkpoints/{rid}/STATUS")
            st = parse_status(txt)
            if st:
                b2col = st["word"]
                if st["phase"]:
                    b2col += f"/{st['phase']}"
                if st["age_s"] is not None:
                    b2col += f" ({_fmt_age(st['age_s'])})"
            else:
                b2col = "(no STATUS)"
        elif rid and not b2.ok:
            b2col = "(B2 unreachable)"
        print(f"  {str(s['id']):>10}  {run_col[:26]:<26} "
              f"{(s['actual_status'] or '?')[:9]:<9} {_geo_short(s['geolocation']):<4} "
              f"{gpu[:15]:<15} {dollars(s['dph_total']):>7} "
              f"{_fmt_age(s['age_s']):>6}  {b2col}")


def print_report(run_id, vast, b2sec, b2sec2, tail_n, jobs_box_iid=None):
    # ---- VAST ----
    print(f"== VAST (run {run_id}) ==")
    if not vast:
        print("  (no live vast instance carries label run:%s, and no jobs-v2 event "
              "names a box — B2-only view below)" % run_id)
    else:
        v = vast
        print(f"  instance      : {v['id']}   label={v['label']!r}")
        if jobs_box_iid:
            print(f"  resolved via  : the job's own event log (jobd on box "
                  f"{jobs_box_iid}) — a jobs box carries no run:{run_id} label")
        print(f"  actual_status : {v['actual_status']}   "
              f"intended={v.get('intended_status')}")
        print(f"  gpu           : {v.get('num_gpus')}x {v.get('gpu_name')}   "
              f"machine={v.get('machine_id')}")
        print(f"  cost          : {dollars(v.get('dph_total'))}/hr   "
              f"age={_fmt_age(v.get('age_s'))}")
        geo = v.get("geolocation")
        print(f"  geo / net     : {geo}   down={v.get('inet_down')}Mb/s "
              f"up={v.get('inet_up')}Mb/s   ip={v.get('public_ipaddr')}")
        ssh_h, ssh_p = v.get("ssh_host"), v.get("ssh_port")
        print(f"  ssh           : {('root@%s:%s' % (ssh_h, ssh_p)) if ssh_h else '(none yet)'}")
        msg_lines = _last_status_msg_lines(v, 3)
        if msg_lines:
            label = ("status_msg (provision output, last 3)"
                     if (v.get("actual_status") or "").lower() == "loading"
                     else "status_msg")
            print(f"  {label}:")
            for ln in msg_lines:
                print(f"      {ln.rstrip()}")

    # ---- B2 ----
    print(f"== B2 (run {run_id}) ==")
    if not b2sec.get("reachable"):
        print(f"  (B2 unreachable: {b2sec.get('reason')})")
    else:
        st = b2sec.get("status")
        if st:
            extra = f"  phase={st['phase']}" if st["phase"] else ""
            print(f"  STATUS        : {st['raw']}   "
                  f"(age {_fmt_age(st['age_s'])}){extra}")
        else:
            print("  STATUS        : (absent — expected pre-first-write / early provision)")

        # boot phase timeline (NEW artifact)
        if b2sec.get("boot_phases_present"):
            print("  boot_phases   :")
            for r in b2sec["boot_phases"]:
                mark = " <- current" if r.get("current") else ""
                dur = _fmt_age(r.get("dur_s"))
                detail = (" " + r["detail"]) if r.get("detail") else ""
                print(f"      {r['phase']:<18} {dur:>7}{mark}{detail}")
        else:
            print("  boot_phases   : (absent — box predates the boot-timeline onstart, "
                  "or nothing booted yet)")

        # recent runmeta events
        evs = b2sec.get("events") or []
        if b2sec.get("events_present"):
            print(f"  events        : {b2sec['events_total']} total; last {len(evs)}:")
            for e in evs:
                age = _fmt_age(_runmeta_ts_age_s(e.get("ts")))
                ph = f" phase={e['phase']}" if e.get("phase") else ""
                print(f"      {e.get('ts'):<20} {e.get('event','?'):<18} "
                      f"{e.get('actor','?'):<16} ({age} ago){ph}")
        else:
            # Past the grace window this is an ABSENT EMITTER, not a box that
            # has yet to emit: the event log is written by the jobs-v2 bundle
            # wrapper, so a directly-rented box never writes one at all.
            aged = (vast or {}).get("age_s")
            if aged is not None and aged > _EMITTER_GRACE_S:
                print("  events        : (none — box rented outside jobs-v2, so "
                      "nothing emits this log; not a stall)")
            else:
                print("  events        : (no runs/<id>/events log yet — expected "
                      "during provision, before the box emits)")

        # onstart.log tail (NEW: pushed incrementally during boot)
        if b2sec.get("onstart_present"):
            tail = b2sec.get("onstart_tail") or []
            print(f"  onstart.log   : last {len(tail)} lines (--tail {tail_n}):")
            for ln in tail:
                print(f"      {ln.rstrip()}")
        else:
            print("  onstart.log   : (absent — box predates incremental log push, "
                  "or boot has not written yet)")

    # ---- JOBS-V2 (jobs/<id>/ layout — JOBS_DESIGN.md) ----------------------
    # Always PROBED alongside the legacy runs/<id>/ section above (same id,
    # different namespace); only PRINTED when it actually has something to
    # show, so a plain non-jobs-v2 run's report is unchanged.
    if b2sec2.get("reachable") and jobs2_has_signal(b2sec2):
        print(f"== JOBS-V2 (job {run_id}) ==")
        if b2sec2.get("events_present"):
            evs2 = b2sec2.get("events") or []
            print(f"  events        : {b2sec2['events_total']} total; last {len(evs2)}:")
            for e in evs2:
                age = _fmt_age(_runmeta_ts_age_s(e.get("ts")))
                print(f"      {e.get('ts'):<20} {e.get('event', '?'):<18} "
                      f"{e.get('actor', '?'):<16} ({age} ago)")
        else:
            print("  events        : (none under jobs/<id>/events/)")
        if b2sec2.get("results_present"):
            print(f"  results/      : {len(b2sec2['results'])} object(s)")
        else:
            print("  results/      : (none yet)")
        done2 = b2sec2.get("done")
        if done2:
            print(f"  results.DONE  : rc={done2['rc']}  "
                  f"duration={_fmt_age(done2['duration_s'])}  "
                  f"n_results={done2['n_results']}")
        else:
            print("  results.DONE  : (absent — not terminal yet)")
        cancel2 = b2sec2.get("cancel")
        if cancel2:
            print(f"  CANCEL        : present ({cancel2.get('reason') or '?'}) "
                  f"age={_fmt_age(cancel2.get('age_s'))}")
        if b2sec2.get("log_present"):
            print(f"  log.txt       : {b2sec2['log_size']} bytes")

    # ---- VERDICT ----
    print("== VERDICT ==")
    print(f"  {verdict(vast, b2sec, b2sec2)}")


# --------------------------------------------------------------------------- #
# target resolution
# --------------------------------------------------------------------------- #
def resolve_target(arg, instances):
    """(run_id, instance_dict_or_None). Numeric arg that matches a live instance
    id -> that box (label -> run_id); else treat arg as a RUN_ID and find any
    live instance labelled run:<arg>."""
    if arg.isdigit():
        iid = int(arg)
        for i in instances:
            if i.get("id") == iid:
                return instance_run_label(i) or arg, i
        inst, _ = get_instance_soft(iid)   # not in the live list; fetch directly
        if inst:
            return instance_run_label(inst) or arg, inst
    for i in instances:
        if instance_run_label(i) == arg:
            return arg, i
    return arg, None


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    load_env()
    ap = argparse.ArgumentParser(
        prog="boxstate",
        description="Read-only, SSH-free vast-box state (vast API + B2 artifacts).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="docs: tools/vast/README.md, .claude/skills/herdd/SKILL.md")
    ap.add_argument("target", nargs="?",
                    help="RUN_ID or instance id (omit for a table of all live boxes)")
    ap.add_argument("--json", action="store_true",
                    help="machine-readable full dump")
    ap.add_argument("--tail", type=int, default=20, metavar="N",
                    help="onstart.log lines to show (default 20)")
    a = ap.parse_args()

    b2 = B2()

    # ---- no target: table of all live instances --------------------------- #
    if not a.target:
        ok, data, err = request_soft("GET", "v1/instances/")
        if not ok:
            if not b2.ok:
                sys.exit(f"error: cannot query vast API ({err}) and B2 is "
                         f"unreachable ({b2.reason})")
            sys.exit(f"error: cannot query vast API: {err}")
        ins = data.get("instances", data) if isinstance(data, dict) else data
        ins = ins or []
        live = [i for i in ins if (i.get("actual_status") or "").lower() in LIVE_STATES]
        if a.json:
            out = []
            for i in live:
                s = _slim_instance(i)
                if s["run_id"] and b2.ok:
                    _, txt = b2.cat(f"checkpoints/{s['run_id']}/STATUS")
                    s["b2_status"] = parse_status(txt)
                out.append(s)
            print(json.dumps({"b2_reachable": b2.ok, "b2_reason": b2.reason,
                              "instances": out}, indent=2, default=str))
            return
        print_table(live, b2)
        return

    # ---- one target: full report ------------------------------------------ #
    instances = instances_soft()
    run_id, inst = resolve_target(a.target, instances)
    b2sec = gather_b2(b2, run_id, a.tail)
    # always probe jobs-v2 in parallel — see gather_jobs2's docstring for why
    # this beats trying to pattern-match which namespace an id "looks like".
    b2sec2 = gather_jobs2(b2, run_id)
    # a jobs-v2 box carries no run:<job-id> label, so label resolution above finds
    # nothing and a RUNNING job used to report `stopped:` (§6 red flag 5 of
    # BOX_SATURATION_AUDIT_2026-07-30). The job's own events name its box.
    jobs_box_iid = None
    if inst is None and jobs2_has_signal(b2sec2):
        inst, jobs_box_iid = resolve_jobs_box(b2sec2, instances)
    vast = _slim_instance(inst) if inst else None

    # 'cannot even query' -> non-zero; a pessimistic verdict does NOT
    if vast is None and not b2.ok:
        sys.exit(f"error: no live vast instance for {a.target!r} and B2 is "
                 f"unreachable ({b2.reason})")

    if a.json:
        print(json.dumps({
            "run_id": run_id,
            "target": a.target,
            "vast": vast,
            "b2": b2sec,
            "jobs_v2": b2sec2,
            # which box the job's OWN events name, when no run:<id> label exists
            # (a jobs-v2 box); None for a label-resolved run.
            "jobs_box_iid": jobs_box_iid,
            "verdict": verdict(vast, b2sec, b2sec2),
            # image-pull bytes seen so far (0 outside the provisioning pull), so
            # an external poller / a human running boxstate twice can compute a
            # rate without re-parsing docker output (reuses parse_pull_progress).
            "pull_bytes_total": _pull_bytes_total(vast),
        }, indent=2, default=str))
        return

    print_report(run_id, vast, b2sec, b2sec2, a.tail, jobs_box_iid)


if __name__ == "__main__":
    main()

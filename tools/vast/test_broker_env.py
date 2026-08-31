"""Portable tests for the cred-broker launch integration (C4,
docs/plans/cred-broker-buildout.md §2.1). No network, no B2, no vast API:

  * _do_launch ships a fresh BOX_IDENTITY_NONCE (32-hex, unique per launch);
    an explicit --env override wins (setdefault semantics).
  * CRED_BROKER_URL ships only when set workstation-side; TS_AUTHKEY only when
    BOTH it and CRED_BROKER_URL are set (an authkey without a broker is inert).
  * --jobs path: CRED_ROLE=jobs; B2_KEY_EXPIRES_AT only when a key was
    actually minted this call (standing-key fallback has no known expiry).
  * _minted_expiry: mint-cache membership is the witness; epoch plausibility.
  * cmd_job_attach: fresh nonce + CRED_ROLE=jobs in jobd.env; best-effort
    /v1/register POST (sha256 of the nonce, X-Broker-Admin header) that can
    never fail the attach.
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.request

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import herdd  # noqa: E402  (path anchor + the flat mint ledgers; see _clean_seams)
import b2_mint_key  # noqa: E402
import imageref  # noqa: E402
from vastlib.boxes import lifecycle  # noqa: E402
from vastlib.boxes import ssh as boxes_ssh  # noqa: E402
from vastlib.cli import _compose  # noqa: E402  (binds the --jobs composer seam)
from vastlib.cli.job import attach as cli_job_attach  # noqa: E402
from vastlib.core import fmt  # noqa: E402
from vastlib.jobs import bundle  # noqa: E402
from vastlib.launch import launch as launch_mod  # noqa: E402
from vastlib.launch import spec  # noqa: E402
from vastlib.market import offers  # noqa: E402

_HEX32 = re.compile(r"^[0-9a-f]{32}$")


@pytest.fixture(autouse=True)
def _clean_seams(monkeypatch):
    # the workstation shell (or a folded .env) may carry real broker/B2 vars;
    # every test starts from "unconfigured" and opts in explicitly.
    for k in ("CRED_BROKER_URL", "TS_AUTHKEY", "CRED_BROKER_ADMIN_TOKEN",
              "B2_EPHEMERAL_HOURS", "B2_KEY_ID_EU", "B2_APPLICATION_KEY_EU",
              "B2_S3_ENDPOINT_EU", "B2_BUCKET_EU"):
        monkeypatch.delenv(k, raising=False)
    # ONE ledger since plan §8 step 6d. This used to clear BOTH — the flat
    # dicts that backed `herdd._do_launch` and `launch.spec`'s — because a
    # cleared one left the other dirty. The thin launcher has no dicts of its
    # own (it does not even re-export `_MINTED_SCOPED`), so `spec` is the whole
    # ledger and reaching through `herdd` would only raise.
    spec._MINTED_PAIRS.clear()
    spec._MINTED_SCOPED.clear()
    yield
    spec._MINTED_PAIRS.clear()
    spec._MINTED_SCOPED.clear()


def _launch_ns(**over):
    base = dict(offer=None, type="ondemand", price=None, env=None, port=None,
                jupyter=False, onstart=None, no_hf_token=True, hf_token=None,
                ssh=False, ssh_key_file=None, jobs=False, image="img:tag",
                disk=40, runtype="ssh_direct", label=None, template_id=None,
                no_registry_login=True, login=None, dry_run=False, wait=None,
                force=False)
    base.update(over)
    return argparse.Namespace(**base)


def _launch_body(monkeypatch, ns):
    """_do_launch against a fake market + fake launch PUT; returns the RAW
    request body (dry-run output redacts the CRED/KEY name families, so the
    non-dry path with a captured launch_instance is the honest observer)."""
    bodies = []
    # Seams follow the subject: `launch.launch._do_launch` resolves `search_offers`
    # through `offers`, `fmt_offer` through `core.fmt` and `image_tag_digest`
    # through the Zone-S sibling `imageref`; `_launch_preflight`/`launch_instance`
    # are module-level REBINDS captured in `launch.launch` at import, so a patch on
    # `boxes.lifecycle` would not be seen (launch/launch.py's own banner says so).
    monkeypatch.setattr(offers, "search_offers",
                        lambda a: [{"id": 123, "min_bid": 0.20, "dph_total": 1.00}])
    monkeypatch.setattr(fmt, "fmt_offer", lambda o: "offer-123")
    monkeypatch.setattr(imageref, "image_tag_digest", lambda img: None)
    monkeypatch.setattr(launch_mod, "_launch_preflight", lambda label, force: None)
    monkeypatch.setattr(launch_mod, "launch_instance",
                        lambda oid, body: (bodies.append(body) or (True, 42, None)))
    launch_mod._do_launch(ns)
    return bodies[0]


# =============================================================================
# BOX_IDENTITY_NONCE — always present, 32-hex, unique, --env wins
# =============================================================================
def test_nonce_always_present_32hex_unique(monkeypatch):
    b1 = _launch_body(monkeypatch, _launch_ns())
    b2 = _launch_body(monkeypatch, _launch_ns())
    n1 = b1["env"]["BOX_IDENTITY_NONCE"]
    n2 = b2["env"]["BOX_IDENTITY_NONCE"]
    assert _HEX32.match(n1) and _HEX32.match(n2)
    assert n1 != n2                              # fresh secret per launch


def test_explicit_env_nonce_wins(monkeypatch):
    b = _launch_body(monkeypatch,
                     _launch_ns(env=["BOX_IDENTITY_NONCE=deadbeef"]))
    assert b["env"]["BOX_IDENTITY_NONCE"] == "deadbeef"


# =============================================================================
# CRED_BROKER_URL / TS_AUTHKEY gating
# =============================================================================
def test_no_broker_vars_when_workstation_unconfigured(monkeypatch):
    b = _launch_body(monkeypatch, _launch_ns())
    assert "CRED_BROKER_URL" not in b["env"]
    assert "TS_AUTHKEY" not in b["env"]


def test_ts_authkey_never_ships_without_broker_url(monkeypatch):
    monkeypatch.setenv("TS_AUTHKEY", "tskey-abc")
    b = _launch_body(monkeypatch, _launch_ns())
    assert "TS_AUTHKEY" not in b["env"]          # authkey alone is inert
    assert "CRED_BROKER_URL" not in b["env"]


def test_broker_url_ships_alone(monkeypatch):
    monkeypatch.setenv("CRED_BROKER_URL", "http://rig.ts.net:8651")
    b = _launch_body(monkeypatch, _launch_ns())
    assert b["env"]["CRED_BROKER_URL"] == "http://rig.ts.net:8651"
    assert "TS_AUTHKEY" not in b["env"]


def test_both_set_ships_both(monkeypatch):
    monkeypatch.setenv("CRED_BROKER_URL", "http://rig.ts.net:8651")
    monkeypatch.setenv("TS_AUTHKEY", "tskey-abc")
    b = _launch_body(monkeypatch, _launch_ns())
    assert b["env"]["CRED_BROKER_URL"] == "http://rig.ts.net:8651"
    assert b["env"]["TS_AUTHKEY"] == "tskey-abc"


def test_explicit_env_broker_url_wins(monkeypatch):
    monkeypatch.setenv("CRED_BROKER_URL", "http://rig.ts.net:8651")
    b = _launch_body(monkeypatch,
                     _launch_ns(env=["CRED_BROKER_URL=http://other:1"]))
    assert b["env"]["CRED_BROKER_URL"] == "http://other:1"


# =============================================================================
# _minted_expiry — mint-cache membership is the witness
# =============================================================================
def test_minted_expiry_none_when_nothing_minted():
    assert spec._minted_expiry("job-launch-1-2", 168) is None
    assert spec._minted_expiry("", 168) is None
    assert spec._minted_expiry(None, 168) is None


def test_minted_expiry_epoch_plausible_single_and_scoped():
    spec._MINTED_PAIRS["run-r1"] = ("K", "S")
    spec._MINTED_SCOPED["box-7"] = ((("RK", "RS")), ("WK", "WS"))
    for base, hours in (("run-r1", 10), ("box-7", 168)):
        exp = spec._minted_expiry(base, hours)
        assert isinstance(exp, int)
        assert abs(exp - (time.time() + hours * 3600)) < 60


def test_minted_expiry_sanitizes_like_the_mint(monkeypatch):
    # _ship_b2_* sanitize before caching; the expiry lookup must match a base
    # whose raw form differs from its sanitized cache key.
    key = b2_mint_key.sanitize_name("run-r.1")
    spec._MINTED_PAIRS[key] = ("K", "S")
    assert spec._minted_expiry("run-r.1", 1) is not None


# =============================================================================
# --jobs launch path — CRED_ROLE + gated B2_KEY_EXPIRES_AT
#
# UNBLOCKED AND REPOINTED (was MIGRATION-BLOCKED, plan §7.2 batch B5). The port
# gap that blocked these three is closed: `jobs/` is a ring ABOVE `launch/`, so
# `launch/launch.py` may not import it and leaves `compose_jobs_launch_env` a
# raising seam — and `cli/_compose.py::bind()` now points it at
# `jobs.bundle.compose_jobs_launch_env` from the one ring allowed to see both.
# The old note recorded the consequence as "the entire `--jobs` launch path is
# dead in vastlib"; it is not, and these tests are the flat-parity half of the
# proof (`test_vastlib_cli_launch.py` drives the same path through the parser).
#
# The repoint is exactly the one that note prescribed: subject to
# `launch.launch._do_launch`, seams to `jobs.bundle._stage_jobd_bootstrap` /
# `jobs.bundle._jobd_boot_snippet` / `launch.spec._ship_b2_env`. `_launch_body`
# (above) is now the single helper — the `_launch_body_flat` fork it needed
# while blocked is gone, which also means these three no longer silently test a
# different `_do_launch` than every other test in this file.
#
# `_compose.bind()` is called explicitly here rather than relied upon: these
# tests drive `_do_launch` directly, not through `cli/launch.py::run`, and
# conftest's `_restore_cross_ring_seam_bindings` hands the seam census back
# afterwards so the raising-seam assertions in `test_vastlib_launch.py` stay
# order-independent.
# =============================================================================
def _jobs_ns(**over):
    return _launch_ns(jobs=True, no_idle_park=False, idle_park_grace=None,
                      no_job_deadline=None, **over)


def _jobs_seams(monkeypatch, minted):
    """The `--jobs` composer's own seams, each at the module that owns it.

    `bundle.compose_jobs_launch_env` reaches `_stage_jobd_bootstrap` and
    `_jobd_boot_snippet` as bare module globals of `jobs.bundle`, and the B2
    shipper as `spec._ship_b2_env` — so the mint ledger the expiry is read from
    is `spec._MINTED_SCOPED`, the one `_clean_seams` already clears. Nothing
    here reaches B2, the mint API or a subprocess."""
    monkeypatch.setenv("B2_BUCKET", "bkt")
    monkeypatch.setenv("B2_S3_ENDPOINT", "https://s3.example")
    monkeypatch.setattr(bundle, "_stage_jobd_bootstrap", lambda dry_run=False: "sha0")
    monkeypatch.setattr(bundle, "_jobd_boot_snippet", lambda sha: "# jobd boot\n")

    def fake_ship(base, hours, write_prefix=None, dry_run=False):
        if minted:      # simulate a real mint: cache under the sanitized name
            spec._MINTED_SCOPED[b2_mint_key.sanitize_name(base)] = (
                ("RK", "RS"), ("WK", "WS"))
            return [("B2_KEY_ID", "RK"), ("B2_APPLICATION_KEY", "RS"),
                    ("B2_WRITE_KEY_ID", "WK"), ("B2_WRITE_APPLICATION_KEY", "WS")]
        return [("B2_KEY_ID", "STANDING"), ("B2_APPLICATION_KEY", "STANDING")]
    monkeypatch.setattr(spec, "_ship_b2_env", fake_ship)
    # The seam under test: without this the composer is a raising stub and the
    # launch dies AFTER the market read (see this section's banner).
    _compose.bind()


def test_jobs_path_role_and_expiry_when_minted(monkeypatch):
    _jobs_seams(monkeypatch, minted=True)
    b = _launch_body(monkeypatch, _jobs_ns())
    env = b["env"]
    assert env["CRED_ROLE"] == "jobs"
    assert _HEX32.match(env["BOX_IDENTITY_NONCE"])
    exp = int(env["B2_KEY_EXPIRES_AT"])
    assert abs(exp - (time.time() + 168 * 3600)) < 120   # default TTL floor


def test_jobs_path_no_expiry_on_standing_key_fallback(monkeypatch):
    _jobs_seams(monkeypatch, minted=False)
    b = _launch_body(monkeypatch, _jobs_ns())
    assert "B2_KEY_EXPIRES_AT" not in b["env"]           # unknown expiry: don't lie
    assert b["env"]["CRED_ROLE"] == "jobs"


def test_jobs_path_explicit_role_override_wins(monkeypatch):
    _jobs_seams(monkeypatch, minted=True)
    b = _launch_body(monkeypatch, _jobs_ns(env=["CRED_ROLE=custom"]))
    assert b["env"]["CRED_ROLE"] == "custom"


# =============================================================================
# cmd_job_attach — fresh nonce + role in jobd.env; best-effort /v1/register
# =============================================================================
class _AttachHarness:
    def __init__(self, monkeypatch, minted=False):
        self.env_pushed = []
        monkeypatch.setenv("B2_BUCKET", "bkt")
        monkeypatch.setenv("B2_S3_ENDPOINT", "https://s3.example")

        def fake_ship(base, hours, write_prefix=None, dry_run=False):
            if minted:
                spec._MINTED_SCOPED[b2_mint_key.sanitize_name(base)] = (
                    ("RK", "RS"), ("WK", "WS"))
            return [("B2_KEY_ID", "RK"), ("B2_APPLICATION_KEY", "RS")]
        # `cli.job.attach.cmd_job_attach` reaches these three by module attribute:
        # `spec._ship_b2_env`, `lifecycle._get_instance`, `boxes_ssh._pick_ssh_endpoint`.
        monkeypatch.setattr(spec, "_ship_b2_env", fake_ship)
        monkeypatch.setattr(lifecycle, "_get_instance",
                            lambda iid: {"id": iid, "actual_status": "running"})
        monkeypatch.setattr(boxes_ssh, "_pick_ssh_endpoint",
                            lambda i: ("h", 22, "direct"))

        def fake_run(cmd, **kw):
            if "input" in kw:
                self.env_pushed.append(kw["input"])
            return type("R", (), {"returncode": 0})()
        monkeypatch.setattr(subprocess, "run", fake_run)

    @staticmethod
    def args(**over):
        base = dict(id=777, dry_run=False, no_idle_park=False,
                    idle_park_grace=None, no_job_deadline=None)
        base.update(over)
        return argparse.Namespace(**base)

    def jobd_env(self):
        assert self.env_pushed, "jobd.env was never pushed"
        return self.env_pushed[0]

    def nonce(self):
        m = re.search(r"^export BOX_IDENTITY_NONCE=([0-9a-f]{32})$",
                      self.jobd_env(), re.M)
        assert m, "no 32-hex nonce export in jobd.env"
        return m.group(1)


def test_attach_ships_nonce_and_role_no_expiry_on_fallback(monkeypatch):
    h = _AttachHarness(monkeypatch, minted=False)
    cli_job_attach.cmd_job_attach(h.args())
    txt = h.jobd_env()
    assert h.nonce()                                    # fresh 32-hex nonce
    assert "export CRED_ROLE=jobs" in txt
    assert "B2_KEY_EXPIRES_AT" not in txt               # standing-key fallback
    assert "CRED_BROKER_URL" not in txt                 # workstation unconfigured


def test_attach_ships_expiry_and_broker_url_when_available(monkeypatch):
    monkeypatch.setenv("CRED_BROKER_URL", "http://rig.ts.net:8651")
    h = _AttachHarness(monkeypatch, minted=True)
    cli_job_attach.cmd_job_attach(h.args())
    txt = h.jobd_env()
    m = re.search(r"^export B2_KEY_EXPIRES_AT=(\d+)$", txt, re.M)
    assert m and abs(int(m.group(1)) - (time.time() + 168 * 3600)) < 120
    assert "export CRED_BROKER_URL=http://rig.ts.net:8651" in txt


def test_attach_nonce_unique_per_attach(monkeypatch):
    h = _AttachHarness(monkeypatch)
    cli_job_attach.cmd_job_attach(h.args())
    cli_job_attach.cmd_job_attach(h.args())
    n1 = re.findall(r"^export BOX_IDENTITY_NONCE=([0-9a-f]{32})$",
                    "\n".join(h.env_pushed), re.M)
    assert len(n1) == 2 and n1[0] != n1[1]


# =============================================================================
# /v1/register POST — best-effort, correct wire shape, never fails the attach
# =============================================================================
class _FakeResp:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_attach_registers_nonce_sha256_with_broker(monkeypatch):
    monkeypatch.setenv("CRED_BROKER_URL", "http://rig.ts.net:8651/")
    monkeypatch.setenv("CRED_BROKER_ADMIN_TOKEN", "admintok")
    reqs = []
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda req, timeout=None: (reqs.append((req, timeout))
                                                   or _FakeResp()))
    h = _AttachHarness(monkeypatch)
    cli_job_attach.cmd_job_attach(h.args())
    assert len(reqs) == 1
    req, timeout = reqs[0]
    assert req.full_url == "http://rig.ts.net:8651/v1/register"
    assert timeout == 3
    assert req.get_header("X-broker-admin") == "admintok"
    body = json.loads(req.data.decode("utf-8"))
    assert body["instance_id"] == 777
    # the RAW nonce never travels — only its sha256, matching jobd.env's nonce
    assert body["nonce_sha256"] == hashlib.sha256(
        h.nonce().encode("utf-8")).hexdigest()
    assert h.nonce() not in json.dumps(body)


def test_attach_skips_register_without_broker_env(monkeypatch):
    # URL alone, token alone, neither: no POST attempt at all
    for url, tok in ((None, "t"), ("http://x:1", None), (None, None)):
        monkeypatch.delenv("CRED_BROKER_URL", raising=False)
        monkeypatch.delenv("CRED_BROKER_ADMIN_TOKEN", raising=False)
        if url:
            monkeypatch.setenv("CRED_BROKER_URL", url)
        if tok:
            monkeypatch.setenv("CRED_BROKER_ADMIN_TOKEN", tok)
        monkeypatch.setattr(urllib.request, "urlopen",
                            lambda *a, **k: pytest.fail("register POST attempted"))
        h = _AttachHarness(monkeypatch)
        cli_job_attach.cmd_job_attach(h.args())


def test_attach_survives_broker_error(monkeypatch):
    monkeypatch.setenv("CRED_BROKER_URL", "http://rig.ts.net:8651")
    monkeypatch.setenv("CRED_BROKER_ADMIN_TOKEN", "admintok")

    def boom(*a, **k):
        raise OSError("connection refused")
    monkeypatch.setattr(urllib.request, "urlopen", boom)
    h = _AttachHarness(monkeypatch)
    cli_job_attach.cmd_job_attach(h.args())          # must not raise
    assert h.jobd_env()                       # attach completed its env push


# =============================================================================
# jobd bundle completeness — the refresh lane (C6) is inert unless the client
# actually ships. jobd.sh resolves $JOBD_DIR/cred_client.py and cred_client
# runs tailnet_join.sh from its own dir; both delivery paths (boot tar via
# _stage_jobd_bootstrap and job-attach scp) build from _job_attach_files(),
# so this list is the single point of truth.
# =============================================================================
def test_jobd_bundle_ships_cred_client_and_tailnet_join():
    names = [os.path.basename(f) for f in bundle._job_attach_files()]
    assert "cred_client.py" in names
    assert "tailnet_join.sh" in names
    assert "fetch_eval_env.sh" in names             # needs.venv:eval self-provision
    assert "job_serve.sh" in names                  # needs.venv:serve self-provision
    assert "jobd.sh" in names                       # sanity: still the daemon bundle


# The EXACT set, pinned. Adding a script that jobd resolves at $JOBD_DIR without
# adding it here is the 2026-07-29 spot-smoke defect (REMOTE_WAVE_PLAN §7 #2:
# `job_serve.sh` missing => `needs.venv: serve` unprovisionable on a fresh
# `launch --jobs` box, worked around bundle-side). This test exists to make the
# next such addition a CONSCIOUS edit: if you are here because it failed, add the
# file to _job_attach_files() and to this set — do not just delete the assert.
_PINNED_JOBD_BUNDLE = {
    "jobd.sh", "jobd.py", "jobmeta.py", "runmeta.py", "b2_sync.sh",
    # jobmeta's UNGUARDED module-scope `from bidpolicy import DEFEND_*`
    # (84d09ab1). Missing from the bundle for its first hours in the field: the
    # repo hides it (jobd.py adds the parent dir, which is tools/vast/), the
    # flat /workspace/jobd/ has no parent, so every jobd.py call on every box
    # launched in that window died silently. See
    # test_jobd_bundle_imports_flat.py for the delivery-shaped regression test.
    "bidpolicy.py",
    "metrics_probe.py", "cred_client.py", "tailnet_join.sh",
    # Boot GEMM ceiling (2026-08-07). jobd runs it at $JOBD_DIR/gemm_probe.py
    # and it imports metrics_probe as a flat sibling for the busy-GPU guard.
    "gemm_probe.py",
    "fetch_eval_env.sh", "job_serve.sh", "serve_vllm.sh",
    # Preempt-forced checkpoint, both halves (added 2026-08-06). The jobs lane
    # carried neither, so `preempt_save.py` had never once run on a box: the
    # trainer could not import it and jobd had nothing to signal. See
    # test_preempt_save.py's delivery tests for the full root cause.
    "preempt_save.py", "preempt_trap.sh",
    # Shared cross-box Triton JIT cache (added 2026-08-07, 9237a820). jobd's
    # triton_cache_boot_pull / triton_cache_push_bg resolve it as
    # $JOBD_DIR/triton_cache.py and silently SKIP when absent, so shipping it is
    # what arms the hook at all — precisely the fail-quiet shape this pin exists
    # to make conscious. Ratified here, not asserted away: `_job_attach_files()`
    # already carries it with its own rationale.
    "triton_cache.py",
    # Boot CPU probe and its closure (added 2026-08-25). jobd resolves it at
    # $JOBD_DIR/cpu_probe.py, and `cpu_probe drop` imports hostfacts as a flat
    # sibling to build the record. hostfacts.py had never ridden THIS bundle —
    # the harvested producers get it from their own job bundle's jobcommon
    # symlink — so shipping the probe alone would put it on every box only to
    # die at `import hostfacts` and drop nothing, silently, forever. The same
    # fail-quiet shape as triton_cache above, which is what this pin is for.
    "cpu_probe.py", "hostfacts.py",
    # b2x transport shim (added 2026-08-25). Absent since the jobs lane existed,
    # and this pin RATIFIED the absence: jobd.sh sources the shim from $JOBD_DIR
    # and, finding nothing, defines `b2x_pull() { return 1; }`, so every b2x
    # site in the lane fell through to rclone and no test could tell. The worst
    # shape this pin guards against — not a crash, not a skip, but a working
    # fallback that makes the fast path's absence unobservable. Flow counts on
    # one object: rclone-stock 4, rclone-tuned 9, b2x 68.
    "b2x_boot.sh",
    # The CDN tier's worker (added 2026-08-27). b2x_boot.sh's rung 0 resolves it
    # beside itself and, finding nothing, logs a miss and falls through — the
    # same fail-quiet shape as the shim above, one rung further along: a working
    # b2x fallback would again make the fast path's absence unobservable.
    "cdn_pull.py",
}


def test_jobd_bundle_file_set_is_pinned():
    names = {os.path.basename(f) for f in bundle._job_attach_files()}
    assert names == _PINNED_JOBD_BUNDLE, (
        f"jobd bundle changed: +{sorted(names - _PINNED_JOBD_BUNDLE)} "
        f"-{sorted(_PINNED_JOBD_BUNDLE - names)}")


def test_jobd_bundle_ships_every_venv_provisioner():
    """Derived, not hand-listed: whatever `jobd.sh check_venv` resolves through
    `_venv_provisioner` falls back to `$JOBD_DIR/<name>` on a real box, so every
    such name MUST ride the flat bundle. A new `needs.venv:` kind fails here the
    moment its provisioner is wired but not shipped."""
    jobd_sh = os.path.join(os.path.dirname(os.path.abspath(herdd.__file__)),
                           "onstart", "jobd.sh")
    with open(jobd_sh) as fh:
        body = fh.read()
    wanted = set(re.findall(r"_venv_provisioner\s+(\S+\.sh)\s", body))
    assert wanted, "no _venv_provisioner call sites found — did jobd.sh change?"
    names = {os.path.basename(f) for f in bundle._job_attach_files()}
    assert wanted <= names, f"provisioner(s) not in the jobd bundle: {sorted(wanted - names)}"


def test_jobd_bundle_ships_job_serve_sibling_helpers():
    """job_serve.sh resolves its helpers as FLAT siblings ($HERE/<x>), and the
    bundle is flattened into /workspace/jobd/ — so every `$HERE/...` default in
    it must be in the bundle too, or the serve lane exits 2 'not found'."""
    here = os.path.join(os.path.dirname(os.path.abspath(herdd.__file__)), "onstart")
    with open(os.path.join(here, "job_serve.sh")) as fh:
        body = fh.read()
    siblings = set(re.findall(r"\$HERE/([A-Za-z0-9_.-]+)", body))
    assert "serve_vllm.sh" in siblings, "job_serve.sh no longer sources serve_vllm.sh?"
    names = {os.path.basename(f) for f in bundle._job_attach_files()}
    assert siblings <= names, f"job_serve.sh helper(s) missing: {sorted(siblings - names)}"


def test_jobd_bundle_files_exist_on_disk():
    # _stage_jobd_bootstrap sys.exits on a missing file — catch it at test time
    for f in bundle._job_attach_files():
        assert os.path.isfile(f), f"bundle file missing: {os.path.basename(f)}"


def test_jobd_bundle_flat_names_unique():
    # boot tar + attach scp both flatten to basenames in /workspace/jobd/
    names = [os.path.basename(f) for f in bundle._job_attach_files()]
    assert len(names) == len(set(names))


def test_attach_dry_run_touches_nothing(monkeypatch):
    monkeypatch.setenv("CRED_BROKER_URL", "http://rig.ts.net:8651")
    monkeypatch.setenv("CRED_BROKER_ADMIN_TOKEN", "admintok")
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda *a, **k: pytest.fail("register POST on dry-run"))
    h = _AttachHarness(monkeypatch)
    cli_job_attach.cmd_job_attach(h.args(dry_run=True))
    assert h.env_pushed == []                 # no ssh, no env push, no register


# --- the PUBLISH grant a jobs box ships with (B2_PUBLISH_KEY_SCOPE_FIX) ------ #
def _ship_env(monkeypatch, *, publish_prefix=None, pub_raises=False):
    """_ship_b2_env against stubbed mints — no network, no minter needed."""
    monkeypatch.setenv("B2_MINTER_KEY_ID", "mk")
    monkeypatch.setenv("B2_MINTER_APPLICATION_KEY", "ms")
    monkeypatch.setenv("B2_BUCKET", "bkt")
    if publish_prefix is not None:
        monkeypatch.setenv("B2_PUBLISH_PREFIX", publish_prefix)
    else:
        monkeypatch.delenv("B2_PUBLISH_PREFIX", raising=False)
    monkeypatch.setattr(b2_mint_key, "mint_pair",
                        lambda base, hours, write_prefix: (("ro", "rs"), ("rw", "ws")))
    seen = {}

    def _pub(base, hours=48, prefix=None, bucket=None, env=None):
        seen["called"] = (base, hours)
        if pub_raises:
            raise b2_mint_key.MintError("boom")
        return None if b2_mint_key.publish_prefix() is None else ("pub", "ps")

    monkeypatch.setattr(b2_mint_key, "mint_publish", _pub)
    spec._MINTED_PUBLISH.clear()
    env = dict(spec._ship_b2_env("box-1", hours=5, write_prefix="jobs/"))
    spec._MINTED_SCOPED.clear(); spec._MINTED_PUBLISH.clear()
    return env, seen


def test_jobs_box_ships_read_write_and_publish_keys(monkeypatch):
    env, seen = _ship_env(monkeypatch)
    assert env["B2_KEY_ID"] == "ro" and env["B2_WRITE_KEY_ID"] == "rw"
    assert env["B2_PUBLISH_KEY_ID"] == "pub"
    assert env["B2_PUBLISH_APPLICATION_KEY"] == "ps"
    assert seen["called"] == ("box-1", 5)          # same TTL as the pair


def test_publish_grant_can_be_switched_off(monkeypatch):
    env, _ = _ship_env(monkeypatch, publish_prefix="")
    assert "B2_PUBLISH_KEY_ID" not in env
    assert env["B2_WRITE_KEY_ID"] == "rw"          # the rest is unchanged


def test_a_failed_publish_mint_never_fails_the_launch(monkeypatch):
    """A box without the grant runs every non-publishing job exactly as before;
    the submit-time write-scope gate is what keeps a publishing bundle off it."""
    env, _ = _ship_env(monkeypatch, pub_raises=True)
    assert "B2_PUBLISH_KEY_ID" not in env
    assert env["B2_KEY_ID"] == "ro" and env["B2_WRITE_KEY_ID"] == "rw"


# =============================================================================
# cmd_job_attach — the box's billed rate, so a collector can price its output
# =============================================================================
def test_attach_ships_the_box_rate(monkeypatch):
    """Nothing on a box knows what it costs, which is why no trained arm
    carries cost data (BUDGET_METRIC_MODEL_2026-08-11.md §2c). `dph_total` is
    the BILLED rate (bid + storage), not `dph_base`."""
    h = _AttachHarness(monkeypatch)
    monkeypatch.setattr(lifecycle, "_get_instance",
                        lambda iid: {"id": iid, "actual_status": "running",
                                     "dph_total": 2.4711, "dph_base": 2.4})
    cli_job_attach.cmd_job_attach(h.args())
    assert "export BOX_DPH_USD=2.471100" in h.jobd_env()


def test_attach_omits_an_unreported_rate_rather_than_zeroing_it(monkeypatch):
    """A 0.0 on the box reads downstream as a free box — the same misreading
    that inverts an arm ranking when imputed dollars are taken as cash."""
    h = _AttachHarness(monkeypatch)
    monkeypatch.setattr(lifecycle, "_get_instance",
                        lambda iid: {"id": iid, "actual_status": "running",
                                     "dph_total": None})
    cli_job_attach.cmd_job_attach(h.args())
    assert "BOX_DPH_USD" not in h.jobd_env()

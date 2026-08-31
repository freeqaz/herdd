"""Portable tests for `vastlib.jobs.submit` — the pre-spend gate wall, ported at
plan §8 step 5.

What this file is for
---------------------
`test_job_submit_preflight.py` is THE submit test and it was NOT edited by this
step (add-only: its 207 `herdd.*` references kept testing the flat original,
which was still there; at plan §8 step 6d that original is gone and they reach
this module). So this file does not re-litigate the five gates' logic. It pins
the things the MOVE could break and the flat test cannot see:

* **`_repo_root`'s depth.** `herdd.py` walked three `dirname`s; this module is
  two directories deeper and needs five. Nothing raises when it is wrong —
  `cmd_job_submit` joins `out/jobs/_bundles` onto it and `os.replace()`s a real
  bundle into the result, so a wrong count stages job bundles into a tree nobody
  cleans up while every gate above still passes.
* **The module-attribute calling convention** (plan §8(b)). `_get_instance_soft`,
  `_ensure_b2_remote`, `_cli_actor`, `_disk_gb`, `_storage_day`,
  `_scratch_probe_soft` and `fleet_watch_supervision` all live in other modules
  now, and 14+ `monkeypatch.setattr` sites steer `cmd_job_submit` THROUGH them.
  A `from … import _ensure_b2_remote` would bind past every one of those patches
  and reach real B2. Each is asserted here by patching the OWNING module and
  observing the effect — the same thing the migrated patch sites will do.
* **The two advisories that must never raise.** `_submit_disk_advisory` and
  `_print_submit_supervision` are both wrapped in blanket handlers, and both are
  provoked with an exploding dependency to prove the swallow is still there.
* **`--env` values are never echoed.** A value can be a credential.

Offline lane: no network, no B2, no rclone, no vast API, $0. `jobmeta` is stubbed
at the function level (it is Zone S and unchanged); nothing here writes outside
`tmp_path`.
"""
from __future__ import annotations

import argparse
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jobmeta  # noqa: E402
import herdd as v  # noqa: E402
from vastlib.boxes import health, lifecycle  # noqa: E402
from vastlib.core import models  # noqa: E402
from vastlib.fleet import client  # noqa: E402
from vastlib.jobs import submit  # noqa: E402
from vastlib.storage import b2  # noqa: E402

#: The real jobmeta callables, bound at import — `pipeline` stubs them out, and
#: `real_validation` puts these back to exercise the shipping validation.
_REAL = {n: getattr(jobmeta, n) for n in
         ("load_job_config", "validate_job_config", "make_ticket")}


# =============================================================================
# _repo_root — the staging anchor
# =============================================================================
def test_repo_root_matches_the_herdd_computation():
    """Three `dirname`s from `tools/vast/herdd.py`, five from
    `tools/vast/vastlib/jobs/submit.py`. This test IS the pin.

    The anchor is the launcher's PATH, which plan §4 freezes — not its code, so
    the thinning at step 6d does not weaken it. The `== v._repo_root()` arm went
    with the thinning: that name is now an identity re-export of the callee on
    this line's left-hand side.
    """
    flat = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(v.__file__))))
    assert submit._repo_root() == flat
    assert os.path.isdir(os.path.join(submit._repo_root(), "tools", "vast"))
    assert v._repo_root is submit._repo_root, (
        "the launcher redefined _repo_root — a second depth computation is "
        "exactly the silent-wrong-directory bug this test exists for")


def test_repo_root_is_a_patchable_module_attribute(monkeypatch):
    """Three `monkeypatch.setattr` sites steer the staging dir and
    `asset_preflight`'s repo base through this name; a constant folded into the
    call sites would make all three vacuous."""
    monkeypatch.setattr(submit, "_repo_root", lambda: "/somewhere/else")
    assert submit._repo_root() == "/somewhere/else"


# =============================================================================
# _apply_env_overrides — keys out, values never
# =============================================================================
def test_env_overrides_fold_onto_the_config_and_return_keys_only():
    raw = {"env": {"KEEP": "1"}}
    keys = submit._apply_env_overrides(raw, ["A=1", "B=tok-secret"])
    assert keys == ["A", "B"]
    assert raw["env"] == {"KEEP": "1", "A": "1", "B": "tok-secret"}
    assert "tok-secret" not in "".join(keys)


# `test_env_overrides_match_the_flat_original` lived here: it folded the same
# override list through `submit._apply_env_overrides` and `v._apply_env_
# overrides` and compared both the return and the mutated dict. Plan §8 step 6d
# made `herdd._apply_env_overrides` an identity re-export of this module's
# function, so it ran one implementation twice and compared it with itself.
# Deleted; the two characterization tests around it own the behavior, and the
# identity of the binding is asserted once, above.


def test_env_overrides_keep_an_empty_value_and_split_on_the_FIRST_equals():
    raw = {}
    submit._apply_env_overrides(raw, ["EMPTY=", "URL=https://x/?a=b"])
    assert raw["env"] == {"EMPTY": "", "URL": "https://x/?a=b"}


def test_env_overrides_are_a_no_op_without_pairs():
    raw = {"env": {"K": "v"}}
    assert submit._apply_env_overrides(raw, None) == []
    assert submit._apply_env_overrides(raw, []) == []
    assert raw == {"env": {"K": "v"}}, "must not normalize the config either"


@pytest.mark.parametrize("bad,msg", [
    ("NOEQUALS", "--env expects K=V"),
    ("has-dash=1", "must be a shell identifier"),
    ("has space=1", "must be a shell identifier"),
    ("9leading=1", "must be a shell identifier"),
])
def test_env_overrides_refuse_what_jobd_could_not_export(bad, msg):
    """jobd writes `export K=<quoted>` into `.job.env`, so a bad key is a parse
    error ON THE BOX rather than a failure here."""
    with pytest.raises(SystemExit) as e:
        submit._apply_env_overrides({}, [bad])
    assert msg in str(e.value)


def test_env_override_refusal_does_not_echo_the_value():
    """The refusal quotes the KEY; a credential in the value must not reach a
    terminal, a CI log or a scrollback."""
    with pytest.raises(SystemExit) as e:
        submit._apply_env_overrides({}, ["bad-key=sk-live-SECRET"])
    assert "sk-live-SECRET" not in str(e.value)


# =============================================================================
# _apply_artifact_env — registry composition for `${VAR}` asset prefixes
# =============================================================================
def test_artifact_env_composes_the_prefix_from_the_committed_registry():
    """The whole point: a submit never carries a hand-typed B2 path, and
    `<PREFIX>_B2` is exactly what a `${<PREFIX>_B2}` asset template consumes."""
    raw = {"env": {"KEEP": "1"}}
    keys = submit._apply_artifact_env(raw, ["ADAPTER=mergeddemoa"])
    assert raw["env"]["ADAPTER_B2"] == \
        "checkpoints/mergeddemoa-merged/fdfa492a959d/model"
    assert raw["env"]["ADAPTER_SERVED_NAME"] == "MERGEDDEMOA"
    assert raw["env"]["KEEP"] == "1"
    assert set(keys) == {k for k in raw["env"] if k.startswith("ADAPTER_")}


def test_artifact_env_is_a_no_op_without_pairs():
    raw = {"env": {"K": "v"}}
    assert submit._apply_artifact_env(raw, None) == []
    assert submit._apply_artifact_env(raw, []) == []
    assert raw == {"env": {"K": "v"}}


def test_raw_env_is_the_escape_hatch_and_wins_over_the_registry():
    """Documented precedence: `--artifact` folds first, `--env` second, so a raw
    `--env <PREFIX>_B2=...` bypasses the registry on purpose."""
    raw = {}
    submit._apply_artifact_env(raw, ["ADAPTER=mergeddemoa"])
    submit._apply_env_overrides(raw, ["ADAPTER_B2=checkpoints/hand/typed"])
    assert raw["env"]["ADAPTER_B2"] == "checkpoints/hand/typed"
    assert raw["env"]["ADAPTER_ID"] == "mergeddemoa"      # the rest still rode along


@pytest.mark.parametrize("bad,msg", [
    ("NOEQUALS", "--artifact expects PREFIX=<registry slug>"),
    ("ADAPTER=", "--artifact expects PREFIX=<registry slug>"),
    ("ADAPTER=no-such-artifact", "no artifact"),
    ("lower=mergeddemoa", "must match"),
])
def test_artifact_env_refuses_a_bad_pair(bad, msg):
    """An unknown slug must REFUSE, not degrade to 'no export' and then to a
    confusing unresolved-variable error two frames later."""
    with pytest.raises(SystemExit) as e:
        submit._apply_artifact_env({}, [bad])
    assert msg in str(e.value)


def test_artifact_env_refuses_one_prefix_claiming_two_artifacts():
    with pytest.raises(SystemExit) as e:
        submit._apply_artifact_env({}, ["A=mergeddemoa", "A=qwen36-27b"])
    assert "named twice" in str(e.value)
    # the same slug twice is idempotent, not an error
    raw = {}
    submit._apply_artifact_env(raw, ["A=mergeddemoa", "A=mergeddemoa"])
    assert raw["env"]["A_ID"] == "mergeddemoa"


# =============================================================================
# _submit_disk_advisory — advisory, silenceable, and it never raises
# =============================================================================
def _cfg(**needs):
    return {"name": "j", "needs": needs} if needs else {"name": "j"}


def test_disk_advisory_is_silenced_by_the_env_knob(monkeypatch, capsys):
    monkeypatch.setattr(health, "_get_instance_soft",
                        lambda iid: pytest.fail("must not read the API"))
    for val in ("0", "no", "OFF", " off "):
        monkeypatch.setenv("HERDD_DISK_ADVISORY", val)
        submit._submit_disk_advisory(_cfg(), {}, {"zst_size": 1}, "123")
    assert capsys.readouterr().err == ""


def test_disk_advisory_prints_an_estimate_and_reads_the_box(monkeypatch, capsys):
    """The second reading is the one that pays for itself: with an explicit
    `--box`, an undersized box fails HERE for $0."""
    monkeypatch.delenv("HERDD_DISK_ADVISORY", raising=False)
    seen = []
    monkeypatch.setattr(health, "_get_instance_soft",
                        lambda iid: seen.append(iid) or {"id": iid})
    monkeypatch.setattr(models, "_disk_gb", lambda inst: (40.0, 12.0))
    monkeypatch.setattr(models, "_storage_day", lambda inst: 2.13)
    submit._submit_disk_advisory(_cfg(), {}, {"zst_size": 1024}, "4242")
    err = capsys.readouterr().err
    assert "disk estimate" in err
    assert seen == [4242], "the box read must go through boxes.health"


def test_disk_advisory_skips_the_api_for_a_non_digit_box(monkeypatch, capsys):
    """`--local` queues onto a hostname-shaped id; there is no instance to read."""
    monkeypatch.delenv("HERDD_DISK_ADVISORY", raising=False)
    monkeypatch.setattr(health, "_get_instance_soft",
                        lambda iid: pytest.fail("must not read the API"))
    submit._submit_disk_advisory(_cfg(), {}, {"zst_size": 1}, "local-hostname")
    assert "disk estimate" in capsys.readouterr().err


def test_disk_advisory_swallows_everything_and_never_refuses(monkeypatch,
                                                             capsys):
    """An estimate is not worth losing a job over. The blanket handler is the
    contract; provoke it so its NOTE text is pinned and the tests above can
    trust its absence."""
    import disksize
    monkeypatch.delenv("HERDD_DISK_ADVISORY", raising=False)

    def boom(*a, **k):
        raise RuntimeError("estimator exploded")

    monkeypatch.setattr(disksize, "estimate_disk_gb", boom)
    submit._submit_disk_advisory(_cfg(), {}, None, "1")       # must not raise
    assert "disk advisory skipped (RuntimeError" in capsys.readouterr().err


def test_disk_advisory_reads_scratch_facts_from_the_probe_only(monkeypatch,
                                                               capsys):
    """velvet P4d: no probe means no placement — an unverified assumption about
    a box's filesystems must never shrink its allocation."""
    monkeypatch.delenv("HERDD_DISK_ADVISORY", raising=False)
    probed = []
    monkeypatch.setattr(health, "_get_instance_soft", lambda iid: {"id": iid})
    monkeypatch.setattr(models, "_disk_gb", lambda inst: (100.0, 1.0))
    monkeypatch.setattr(models, "_storage_day", lambda inst: 1.0)
    monkeypatch.setattr(health, "_scratch_probe_soft",
                        lambda box: probed.append(box) or None)
    submit._submit_disk_advisory(_cfg(scratch_volatile=True, scratch_gb=10),
                                 {}, {"zst_size": 1}, "77")
    assert probed == ["77"], "the probe read must go through boxes.health"
    assert "scratch:" in capsys.readouterr().err


# =============================================================================
# _print_submit_supervision — advisory, and `unknown` says NOTHING
# =============================================================================
def _sup(monkeypatch, level, detail=None):
    monkeypatch.setattr(client, "fleet_watch_supervision",
                        lambda iid: (level, detail or {}))


def test_supervision_policy_line_reports_the_remaining_budget(monkeypatch,
                                                              capsys):
    _sup(monkeypatch, "policy", {"profile": "jobs", "spend_usd": 3.0,
                                 "budget_usd": 10.0})
    submit._print_submit_supervision("1", "herdd.py")
    out = capsys.readouterr().out
    assert "`jobs` watch" in out and "$3.00 of $10.00 spent" in out
    assert "$7.00 left" in out


def test_supervision_lapsed_is_the_one_that_shouts(monkeypatch, capsys):
    """A watch that already ran and finished `drained`: the box LOOKS supervised
    (the ceiling survived) while the ladder that rescues an outbid spot box is
    gone. On a spot box that means work lost silently."""
    _sup(monkeypatch, "lapsed", {"spend_usd": 1.0, "budget_usd": 5.0})
    submit._print_submit_supervision("1", "herdd.py")
    err = capsys.readouterr().err
    assert "INHERITED ceiling" in err and "$4.00 of $5.00 left" in err
    assert "the LADDER did not" in err
    assert "fleet watch 1 --profile jobs" in err


@pytest.mark.parametrize("level,needle", [
    ("bare", "observation only"),
    ("none", "no fleet watch yet"),
])
def test_supervision_bare_and_none_are_informational(monkeypatch, capsys,
                                                     level, needle):
    """"No watch yet" is the CORRECT state for a fresh box — the documented
    order is rent -> submit -> arm — so neither may read as a warning."""
    _sup(monkeypatch, level)
    submit._print_submit_supervision("1", "herdd.py")
    cap = capsys.readouterr()
    assert needle in cap.out
    assert cap.err == ""


def test_supervision_unknown_prints_nothing(monkeypatch, capsys):
    """Deliberate. An unreadable state file is not evidence, and a line that
    cried wolf on every submit is a line nobody reads. Do not add one."""
    _sup(monkeypatch, "unknown", {"profile": "jobs"})
    submit._print_submit_supervision("1", "herdd.py")
    assert capsys.readouterr() == ("", "")


def test_supervision_never_breaks_a_submit(monkeypatch, capsys):
    def boom(iid):
        raise RuntimeError("state.json is a directory")

    monkeypatch.setattr(client, "fleet_watch_supervision", boom)
    submit._print_submit_supervision("1", "herdd.py")       # must not raise
    assert capsys.readouterr() == ("", "")


def test_supervision_reaches_fleet_client_by_module_attribute(monkeypatch):
    """`jobs` and `fleet` are `:`-joined ring siblings, so this import is legal —
    but it must stay a module ATTRIBUTE or the patch above steers nothing."""
    seen = []
    monkeypatch.setattr(client, "fleet_watch_supervision",
                        lambda iid: seen.append(iid) or ("unknown", {}))
    submit._print_submit_supervision("9001", "herdd.py")
    assert seen == ["9001"]


# =============================================================================
# cmd_job_submit — the pipeline, with jobmeta stubbed at the seam
# =============================================================================
def _ns(**kw):
    """`argparse.Namespace` with only the attributes the real parser supplies.
    Every optional flag is read with `getattr(a, X, False)` on purpose — the
    flat test builds stubs that omit them, and 25 call sites depend on it."""
    base = dict(dir=".", box="123", name=None, timeout=None, env=None,
                dry_run=False)
    base.update(kw)
    return argparse.Namespace(**base)


@pytest.fixture
def pipeline(monkeypatch, tmp_path):
    """A hermetic `cmd_job_submit`: every gate passes, no B2, no API. Returns a
    dict recording what the effectful seams were asked to do."""
    log: dict[str, object] = {"events": [], "uploaded": [], "ensured": 0}
    src = tmp_path / "job"
    src.mkdir()
    staging = tmp_path / "out" / "jobs" / "_bundles"
    staging.mkdir(parents=True)

    monkeypatch.setattr(submit, "_repo_root", lambda: str(tmp_path))
    monkeypatch.setattr(b2, "_ensure_b2_remote",
                        lambda: log.__setitem__("ensured", log["ensured"] + 1))
    monkeypatch.setattr(lifecycle, "_cli_actor", lambda: "tester")
    monkeypatch.setattr(submit, "_print_submit_supervision",
                        lambda box, prog: None)
    # The disk advisory ALSO reads the instance for a digit box — see
    # `test_the_disk_advisory_is_a_SECOND_api_read_on_a_plain_submit`. Silenced
    # here so the guard below means what it says.
    monkeypatch.setenv("HERDD_DISK_ADVISORY", "0")
    monkeypatch.setattr(health, "_get_instance_soft",
                        lambda iid: pytest.fail("a plain job's submit must "
                                                "stay API-free"))

    cfg = {"name": "j1", "entrypoint": "run.sh", "timeout_s": 60,
           "needs": {}, "defend": "cheap"}
    monkeypatch.setattr(jobmeta, "load_job_config", lambda s: {"name": "j1"})
    monkeypatch.setattr(jobmeta, "validate_job_config", lambda raw, s: (cfg, []))
    monkeypatch.setattr(jobmeta, "eval_env_pin_report",
                        lambda *a, **k: ([], False))
    monkeypatch.setattr(jobmeta, "b2_write_preflight", lambda c, s: [])
    monkeypatch.setattr(jobmeta, "b2_write_scope_report",
                        lambda *a, **k: ([], False))
    monkeypatch.setattr(jobmeta, "vram_gate_findings", lambda c: None)
    monkeypatch.setattr(jobmeta, "vram_gate_report", lambda *a, **k: ([], False))

    def write_bundle(s, out):
        with open(out, "w") as fh:
            fh.write("bundle")
        return {"sha256": "a" * 64, "zst_size": 10, "tar_size": 20}

    monkeypatch.setattr(jobmeta, "write_bundle", write_bundle)
    monkeypatch.setattr(jobmeta, "mint_job_id", lambda n: "j1-20260816")
    monkeypatch.setattr(jobmeta, "bundle_exists", lambda sha: False)
    monkeypatch.setattr(jobmeta, "upload_bundle",
                        lambda p, sha: (log["uploaded"].append(sha), (True, None))[1])
    monkeypatch.setattr(jobmeta, "make_ticket",
                        lambda *a, **k: {"job_id": "j1-20260816"})
    monkeypatch.setattr(jobmeta, "write_ticket",
                        lambda t: (True, "jobs/queue/123/j1-20260816.json", None))
    monkeypatch.setattr(jobmeta, "emit_event",
                        lambda jid, ev, **k: log["events"].append((jid, ev, k)))
    log["src"] = str(src)
    log["staging"] = staging
    log["cfg"] = cfg
    return log


def test_submit_happy_path_returns_the_job_id_and_emits_submitted(pipeline,
                                                                  capsys):
    jid = submit.cmd_job_submit(_ns(dir=pipeline["src"]))
    assert jid == "j1-20260816"
    assert pipeline["uploaded"] == ["a" * 64]
    (ev_jid, ev, kw), = pipeline["events"]
    assert (ev_jid, ev) == ("j1-20260816", "submitted")
    assert kw["actor"] == "tester" and kw["box"] == "123"
    assert kw["defend"] == "cheap", "the bid ladder's lost-work hint"
    out = capsys.readouterr().out
    assert "JOB_ID=j1-20260816" in out and "ticket: jobs/queue/123/" in out


def test_submit_tells_fleetd_the_box_now_holds_a_ticket(pipeline, monkeypatch,
                                                        capsys):
    """"[STANDING, dormant — this submit re-arms it]" was advisory: nothing told
    the daemon, and the standing watch's own queue poll reads nothing on a
    parked box and `unknown` on a B2 blip (it had fired 0 times against 84
    drains by 2026-08-27). The submit says so now — silently, since the
    supervision line right below it already carries the wording."""
    seen = []
    monkeypatch.setattr(client, "fleet_ticket_placed",
                        lambda box, jid=None, **kw: seen.append((box, jid, kw)))
    submit.cmd_job_submit(_ns(dir=pipeline["src"]))
    assert seen == [("123", "j1-20260816",
                     {"source": "job submit", "announce": False})]
    assert "re-arms its ladder" not in capsys.readouterr().out


def test_a_local_submit_wakes_nothing(pipeline, monkeypatch):
    """`--local` queues onto this workstation; there is no box and no watch."""
    seen = []
    monkeypatch.setattr(client, "fleet_ticket_placed",
                        lambda *a, **k: seen.append(a))
    submit.cmd_job_submit(_ns(dir=pipeline["src"], box=None, local=True))
    assert seen == []


def test_submit_stages_the_bundle_under_repo_root_not_the_cwd(pipeline):
    submit.cmd_job_submit(_ns(dir=pipeline["src"]))
    staged = os.listdir(pipeline["staging"])
    assert staged == [f"{'a' * 64}.tar.zst"]
    assert "pending.tar.zst" not in staged, "os.replace must have moved it"


def test_submit_dry_run_returns_None_and_makes_no_b2_mutation(pipeline, capsys,
                                                              monkeypatch):
    """The asymmetric return is a contract: a caller that treats the result as a
    job id must not be handed a dry-run arg."""
    monkeypatch.setattr(jobmeta, "write_ticket",
                        lambda t: pytest.fail("no ticket on a dry run"))
    monkeypatch.setattr(jobmeta, "emit_event",
                        lambda *a, **k: pytest.fail("no event on a dry run"))
    assert submit.cmd_job_submit(_ns(dir=pipeline["src"], dry_run=True)) is None
    assert pipeline["uploaded"] == []
    out = capsys.readouterr().out
    assert "NO B2 mutations" in out
    # the LOCAL bundle is still staged — a dry run is a real rehearsal
    assert os.listdir(pipeline["staging"]) == [f"{'a' * 64}.tar.zst"]


def test_submit_prints_env_override_keys_and_never_values(pipeline, capsys):
    submit.cmd_job_submit(_ns(dir=pipeline["src"], env=["TOKEN=sk-live-SECRET"]))
    cap = capsys.readouterr()
    assert "env override (submit-time): TOKEN" in cap.out
    assert "sk-live-SECRET" not in cap.out + cap.err


@pytest.fixture
def real_validation(pipeline, monkeypatch):
    """`pipeline` with the REAL jobmeta validation and ticket minting restored,
    so a `${VAR}` asset prefix is resolved by the code that ships. Returns the
    log, with `log["ticket"]` holding what `write_ticket` was handed."""
    src = pipeline["src"]
    with open(os.path.join(src, "run.sh"), "w") as fh:
        fh.write("echo hi\n")
    with open(os.path.join(src, "job-config.yaml"), "w") as fh:
        fh.write("version: 1\nname: j1\nentrypoint: run.sh\ntimeout_s: 60\n"
                 "env:\n  ADAPTER_B2: \"\"\n"
                 "assets:\n  - name: adapter\n    b2: \"${ADAPTER_B2}/model\"\n")
    for fn, real in _REAL.items():
        monkeypatch.setattr(jobmeta, fn, real)
    monkeypatch.setattr(jobmeta, "mint_job_id",
                        lambda n: "j1-20260824T000000Z-abcd")
    monkeypatch.setattr(jobmeta, "asset_preflight", lambda cfg, **k: [])
    monkeypatch.setattr(jobmeta, "asset_preflight_report",
                        lambda f, **k: ([], False))
    monkeypatch.setattr(jobmeta, "measure_asset_bytes", lambda a: {})
    monkeypatch.setattr(jobmeta, "write_ticket",
                        lambda t: (pipeline.__setitem__("ticket", t),
                                   (True, "jobs/queue/123/j1.json", None))[1])
    return pipeline


def test_the_staged_ticket_records_the_RESOLVED_prefix(real_validation):
    """Resolution is submit-side: on-box code and the receipt gate read the
    ticket, so what jobd sees must be a real B2 key, never a template."""
    submit.cmd_job_submit(_ns(dir=real_validation["src"],
                              env=["ADAPTER_B2=checkpoints/v10/abc"]))
    a, = real_validation["ticket"]["config"]["assets"]
    assert a["b2"] == "checkpoints/v10/abc/model"
    assert a["b2_template"] == "${ADAPTER_B2}/model"


def test_submit_refuses_an_unresolved_asset_prefix_before_any_spend(
        real_validation, capsys):
    """Fail-closed, and BEFORE the bundle is staged or uploaded — the same
    posture as the asset-receipt preflight."""
    with pytest.raises(SystemExit) as e:
        submit.cmd_job_submit(_ns(dir=real_validation["src"]))
    assert "${ADAPTER_B2}" in str(e.value)
    assert real_validation["uploaded"] == []
    assert os.listdir(real_validation["staging"]) == []


def test_artifact_flag_resolves_the_prefix_end_to_end(real_validation, capsys):
    """`--artifact` is the intended spelling: no B2 path is typed anywhere and
    the registry is what the ticket ends up quoting."""
    with open(os.path.join(real_validation["src"], "job-config.yaml"), "w") as fh:
        fh.write("version: 1\nname: j1\nentrypoint: run.sh\ntimeout_s: 60\n"
                 "env:\n  ADAPTER_B2: \"\"\n"
                 "assets:\n  - name: adapter\n    b2: \"${ADAPTER_B2}\"\n")
    submit.cmd_job_submit(_ns(dir=real_validation["src"],
                              artifact=["ADAPTER=mergeddemoa"]))
    a, = real_validation["ticket"]["config"]["assets"]
    assert a["b2"] == "checkpoints/mergeddemoa-merged/fdfa492a959d/model"
    assert "artifact env (from the modelkit registry): ADAPTER_ADAPTER_IDENT" \
        in capsys.readouterr().out


def test_submit_requires_a_box(pipeline):
    with pytest.raises(SystemExit) as e:
        submit.cmd_job_submit(_ns(dir=pipeline["src"], box=None))
    assert "--box <IID> is required" in str(e.value)


def test_submit_refuses_box_and_local_together(pipeline):
    with pytest.raises(SystemExit) as e:
        submit.cmd_job_submit(_ns(dir=pipeline["src"], box="123", local=True))
    assert "mutually exclusive" in str(e.value)


def test_submit_refuses_a_missing_directory():
    with pytest.raises(SystemExit) as e:
        submit.cmd_job_submit(_ns(dir="/nonexistent/job/dir"))
    assert "not a directory" in str(e.value)


@pytest.mark.parametrize("fn,needle", [
    ("eval_env_pin_report", "no usable EVAL_ENV_VER pin"),
    ("b2_write_scope_report", "writes a B2 prefix the box has no key for"),
    ("vram_gate_report", "needs.gpu_ram_gb is below a peak"),
])
def test_each_gate_refusal_is_its_own_frozen_string(pipeline, monkeypatch,
                                                    fn, needle):
    """Seven distinct `sys.exit` strings, each matched by a test substring
    somewhere. Plan §7.4 freezes them; a shared "refused" message would make
    every one of those matches ambiguous."""
    monkeypatch.setattr(jobmeta, fn, lambda *a, **k: ([], True))
    with pytest.raises(SystemExit) as e:
        submit.cmd_job_submit(_ns(dir=pipeline["src"]))
    assert needle in str(e.value)


def test_the_free_gates_run_before_any_b2_read(pipeline, monkeypatch):
    """Gate ORDER is load-bearing: the pure, network-less refusals cost $0, so
    they must fire before `_ensure_b2_remote` is ever called."""
    monkeypatch.setattr(jobmeta, "eval_env_pin_report",
                        lambda *a, **k: ([], True))
    with pytest.raises(SystemExit):
        submit.cmd_job_submit(_ns(dir=pipeline["src"]))
    assert pipeline["ensured"] == 0


def test_the_eval_venv_gate_is_the_only_api_read(pipeline, monkeypatch):
    """`needs.venv: eval` + a non-local submit + a digit box id. The `pipeline`
    fixture fails the test if `_get_instance_soft` fires for a plain job; here
    it must fire exactly once, and through `boxes.health`."""
    seen = []
    monkeypatch.setattr(health, "_get_instance_soft",
                        lambda iid: seen.append(iid) or {"id": iid})
    monkeypatch.setattr(models, "_instance_env", lambda inst: {"EVAL_ENV_VER": "7"})
    pipeline["cfg"]["needs"] = {"venv": "eval"}
    got = {}
    monkeypatch.setattr(jobmeta, "eval_env_pin_report",
                        lambda cfg, box_env, **k: got.update(
                            box_env=box_env, known=k["box_env_known"]) or ([], False))
    submit.cmd_job_submit(_ns(dir=pipeline["src"]))
    assert seen == [123]
    assert got == {"box_env": {"EVAL_ENV_VER": "7"}, "known": True}


def test_asset_preflight_reads_b2_and_uses_the_patched_repo_root(pipeline,
                                                                 monkeypatch):
    """`_repo_root` is threaded into `jobmeta.asset_preflight`; this is one of
    the three patch sites the migration has to keep steering."""
    got = {}
    pipeline["cfg"]["assets"] = [{"name": "runset"}]
    monkeypatch.setattr(jobmeta, "asset_preflight",
                        lambda cfg, repo_root=None: got.update(root=repo_root) or [])
    monkeypatch.setattr(jobmeta, "asset_preflight_report",
                        lambda *a, **k: ([], False))
    monkeypatch.setattr(jobmeta, "measure_asset_bytes", lambda assets: {"runset": 1})
    submit.cmd_job_submit(_ns(dir=pipeline["src"]))
    assert got["root"] == submit._repo_root()
    assert pipeline["ensured"] >= 1, "the asset gate must ensure the b2: remote"


def test_no_asset_check_skips_the_b2_read_entirely(pipeline, monkeypatch):
    pipeline["cfg"]["assets"] = [{"name": "runset"}]
    monkeypatch.setattr(jobmeta, "asset_preflight",
                        lambda *a, **k: pytest.fail("--no-asset-check opted out"))
    submit.cmd_job_submit(_ns(dir=pipeline["src"], no_asset_check=True))


def test_a_jobmeta_error_during_validation_exits_cleanly(pipeline, monkeypatch):
    def boom(raw, s):
        raise jobmeta.JobmetaError("entrypoint missing")

    monkeypatch.setattr(jobmeta, "validate_job_config", boom)
    with pytest.raises(SystemExit) as e:
        submit.cmd_job_submit(_ns(dir=pipeline["src"]))
    assert "entrypoint missing" in str(e.value)


def test_a_write_scope_preflight_crash_degrades_to_a_note(pipeline, monkeypatch,
                                                          capsys):
    """`never crash a submit on the check` — the swallow is the contract."""
    def boom(cfg, src):
        raise RuntimeError("grant table unreadable")

    monkeypatch.setattr(jobmeta, "b2_write_preflight", boom)
    assert submit.cmd_job_submit(_ns(dir=pipeline["src"])) == "j1-20260816"
    assert "B2 write-scope preflight skipped" in capsys.readouterr().err


def test_the_disk_advisory_is_a_SECOND_api_read_on_a_plain_submit(pipeline,
                                                                  monkeypatch):
    """RECORDED, NOT CHANGED (plan §7.4). The eval-pin gate carries the comment
    "a plain job's submit must stay API-free", and for that gate it is true. But
    `_submit_disk_advisory` runs unconditionally afterwards and reads the same
    instance whenever `--box` is digits and `HERDD_DISK_ADVISORY` is not
    silenced — so a plain submit does in fact make one soft API call, from a
    different place. It is soft (`_get_instance_soft` swallows), so it degrades
    rather than failing, which is why it went unnoticed. The port preserves it
    verbatim; this test exists so the next reader of that comment knows the
    scope of the claim."""
    monkeypatch.delenv("HERDD_DISK_ADVISORY", raising=False)
    seen = []
    monkeypatch.setattr(health, "_get_instance_soft",
                        lambda iid: seen.append(iid) or None)
    submit.cmd_job_submit(_ns(dir=pipeline["src"]))
    assert seen == [123]

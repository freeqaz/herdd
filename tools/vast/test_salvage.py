"""Portable tests for instance->instance disk salvage of an EVICTED vast box.

Covers the two halves separately, because they fail differently:

  * `salvage.py` — the PURE core. `ls` parsing, what is worth copying, and above
    all `verify_salvage`, whose job is to refuse to call a short copy a success.
  * `herdd` wiring — the eviction ladder arms salvage AT THE MOMENT OF
    EVICTION (the race is host reclamation, ~30 min observed, not the 3h
    retention window), the retention sweep advances it, and the retention
    BACKSTOP refuses to destroy a source box whose copy is still in flight.

The central invariant, asserted from several directions: **verification
unavailable => do nothing and say so.** A destination we cannot read back is
`unverifiable`, never `salvaged`; a copy that landed short is `partial`, never
folded into success. Trusting a torn checkpoint is worse than losing it, because
a resume loads it without complaint.

Toolchain-free lane: no vast API, no ssh, no B2, no rclone — every seam is a
callable passed in or monkeypatched.
"""
import os
import re
import subprocess
import sys
import urllib.request

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import salvage as S      # noqa: E402
from vastlib.boxes import lifecycle as vlifecycle  # noqa: E402
from vastlib.boxes import remote as vremote  # noqa: E402
from vastlib.boxes import salvage as vsalvage  # noqa: E402
from vastlib.boxes import ssh as vssh  # noqa: E402
from vastlib.core import api as vapi  # noqa: E402
from vastlib.storage import b2 as vb2  # noqa: E402
from vastlib.supervise import journal as vjournal  # noqa: E402
from vastlib.supervise import retention as vretention  # noqa: E402

NOW = 3_000_000.0

# A realistic checkpoint: the measured shape from box 46859541's lost
# checkpoint-50 (adapter 323 MB fp32, optimizer 646 MB, tokenizer 11 MB).
CKPT_FILES = {"adapter_model.safetensors": 323_000_000,
              "optimizer.pt": 646_000_000,
              "tokenizer.json": 11_000_000}
CKPT_BYTES = sum(CKPT_FILES.values())


def _ls_l(entries):
    """Render `ls -l`-shaped output for (name, size, is_dir) triples."""
    lines = [f"total {len(entries) * 4}"]
    for name, size, is_dir in entries:
        mode = "drwxr-xr-x" if is_dir else "-rw-r--r--"
        lines.append(f"{mode} 1 root root {size} Aug  5 07:00 {name}")
    return "\n".join(lines)


def _ls_lr(sections):
    """Render `ls -lR`-shaped output for {path: [(name, size, is_dir)]}."""
    out = []
    for path, entries in sections.items():
        out.append(f"{path}:")
        out.append(_ls_l(entries))
        out.append("")
    return "\n".join(out)


def _dead_box_exec(job="J1", ckpts=("out/checkpoint-50",), files=None):
    """An `execute` transport for a dead box carrying `ckpts` under one job.

    Default fixture uses the REAL nested layout — jobd's tree is
    `work/out/checkpoint-<N>/` (and `work/arms/<name>/checkpoint-<N>/` for a
    multi-arm bundle), not `work/checkpoint-<N>/`. Measured against the live
    bucket 2026-08-05: 22,577 of 22,898 checkpoint objects carry the `out/`
    level. A fixture at the flat depth passes while the parser finds NOTHING on
    a real disk — a silent `nothing_found` over a full checkpoint tree.
    """
    files = files if files is not None else CKPT_FILES
    root = f"{S.JOBS_ROOT}/{job}/work"
    sections = {root: [("out", 4096, True)]}
    for c in ckpts:
        sections[f"{root}/{c}"] = [(n, sz, False) for n, sz in files.items()]

    def _exec(iid, cmd):
        if cmd == f"ls -1 {S.JOBS_ROOT}":
            return True, f"{job}\n", None
        if cmd == f"ls -lR {root}":
            return True, _ls_lr(sections), None
        return True, "", None
    return _exec


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #
def test_parse_ls_l_reads_sizes_and_dir_flag():
    got = S.parse_ls_l(_ls_l([("checkpoint-50", 4096, True),
                              ("optimizer.pt", 646_000_000, False)]))
    assert got == [S.LsEntry("checkpoint-50", 4096, True),
                   S.LsEntry("optimizer.pt", 646_000_000, False)]


@pytest.mark.parametrize("line,name,size", [
    ("-rw-r--r-- 1 root root 12 Aug  5  2026 old.bin", "old.bin", 12),
    ("-rw-r--r-- 1 root root 7 2026-08-05 07:00:00.000000000 +0000 ft.bin",
     "ft.bin", 7),
    ("-rw-r--r--. 1 root root 9 Aug  5 07:00 selinux.bin", "selinux.bin", 9),
    ("-rw-r--r-- 1 root root 5 Aug  5 07:00 name with spaces.bin",
     "name with spaces.bin", 5),
])
def test_parse_ls_l_date_and_name_shapes(line, name, size):
    got = S.parse_ls_l(line)
    assert got == [S.LsEntry(name, size, False)]


def test_parse_ls_l_skips_junk_rather_than_guessing():
    """An unparseable line must be DROPPED, never guessed at. Dropping biases the
    survey byte total DOWN, which pushes every downstream decision toward
    'copy more / verify stricter' — the safe direction."""
    text = ("ls: cannot access '/nope': No such file or directory\n"
            "total 4\n"
            "\n"
            "-rw-r--r-- 1 root root 5 Aug  5 07:00 real.bin\n")
    assert S.parse_ls_l(text) == [S.LsEntry("real.bin", 5, False)]


def test_parse_ls_l_symlink_keeps_the_link_name_only():
    got = S.parse_ls_l("lrwxrwxrwx 1 root root 9 Aug  5 07:00 link -> target")
    assert got[0].name == "link"


def test_parse_ls_lr_splits_sections():
    sections = S.parse_ls_lr(_ls_lr({
        "/workspace/jobs/J1/work": [("checkpoint-50", 4096, True)],
        "/workspace/jobs/J1/work/checkpoint-50": [("optimizer.pt", 646, False)],
    }))
    assert set(sections) == {"/workspace/jobs/J1/work",
                             "/workspace/jobs/J1/work/checkpoint-50"}
    assert sections["/workspace/jobs/J1/work/checkpoint-50"][0].size == 646


def test_ckpt_dirs_includes_nested_files_in_the_verified_set():
    """A nested file dropped by a partial copy must not read as complete, so the
    file map is keyed by path RELATIVE to the checkpoint dir."""
    root = "/workspace/jobs/J1/work"
    sections = S.parse_ls_lr(_ls_lr({
        root: [("checkpoint-50", 4096, True)],
        f"{root}/checkpoint-50": [("optimizer.pt", 646, False),
                                  ("extra", 4096, True)],
        f"{root}/checkpoint-50/extra": [("shard.bin", 100, False)],
    }))
    dirs = S.ckpt_dirs_from_survey(sections, root)
    assert len(dirs) == 1
    assert dirs[0].files == {"optimizer.pt": 646, "extra/shard.bin": 100}
    assert dirs[0].bytes == 746


def test_ckpt_dirs_sorted_newest_first_and_ignores_non_checkpoints():
    root = "/workspace/jobs/J1/work"
    sections = S.parse_ls_lr(_ls_lr({
        f"{root}/checkpoint-50": [("a", 1, False)],
        f"{root}/checkpoint-200": [("a", 2, False)],
        f"{root}/logs": [("train.log", 999, False)],
    }))
    assert [c.step for c in S.ckpt_dirs_from_survey(sections, root)] == [200, 50]


# --------------------------------------------------------------------------- #
# plan_salvage — the "optimized" half
# --------------------------------------------------------------------------- #
def test_plan_copies_only_the_newest_keep_n():
    cks = [S.CkptDir(f"checkpoint-{n}", n, CKPT_BYTES, dict(CKPT_FILES))
           for n in (300, 200, 100)]
    plan = S.plan_salvage(cks, b2_bytes={}, keep_n=1)
    assert plan.action == "copy"
    assert [c.name for c in plan.items] == ["checkpoint-300"]
    assert plan.bytes == CKPT_BYTES


def test_plan_skips_what_b2_already_has_at_full_size():
    cks = [S.CkptDir("checkpoint-50", 50, CKPT_BYTES, dict(CKPT_FILES))]
    plan = S.plan_salvage(cks, b2_bytes={"checkpoint-50": CKPT_BYTES})
    assert plan.action == "nothing"
    assert "already holds" in plan.reason


def test_plan_treats_a_SHORT_b2_object_as_not_safe():
    """A torn sync on B2 is exactly the case salvage exists for — the B2 copy
    existing is not the same as the B2 copy being complete."""
    cks = [S.CkptDir("checkpoint-50", 50, CKPT_BYTES, dict(CKPT_FILES))]
    plan = S.plan_salvage(cks, b2_bytes={"checkpoint-50": CKPT_BYTES - 1})
    assert plan.action == "copy"


def test_plan_with_UNREADABLE_b2_copies_anyway():
    """`b2_bytes=None` means we could not read B2. That must never be mistaken
    for 'B2 is empty' in the OTHER direction: unknown => copy more, not less."""
    cks = [S.CkptDir("checkpoint-50", 50, CKPT_BYTES, dict(CKPT_FILES))]
    plan = S.plan_salvage(cks, b2_bytes=None)
    assert plan.action == "copy"
    assert "UNREADABLE" in plan.reason


def test_plan_nothing_when_the_disk_has_no_checkpoints():
    assert S.plan_salvage([]).action == "nothing"


def test_plan_refuses_an_absurdly_large_transfer():
    huge = [S.CkptDir("checkpoint-1", 1, 900_000_000_000, {"x": 900_000_000_000})]
    plan = S.plan_salvage(huge, b2_bytes={})
    assert plan.action == "refuse"
    assert "fuse" in plan.reason


# --------------------------------------------------------------------------- #
# verify_salvage — the "lossless" half. THE fail-safe.
# --------------------------------------------------------------------------- #
def test_verify_ok_only_when_names_and_every_byte_count_match():
    v = S.verify_salvage(CKPT_FILES, dict(CKPT_FILES))
    assert v.status == "ok"
    assert v.bytes_seen == v.bytes_expected == CKPT_BYTES


def test_verify_missing_file_is_partial_not_ok():
    got = dict(CKPT_FILES)
    got.pop("optimizer.pt")
    v = S.verify_salvage(CKPT_FILES, got)
    assert v.status == "partial"
    assert v.missing == ("optimizer.pt",)


def test_verify_truncated_file_is_partial_not_ok():
    got = dict(CKPT_FILES, **{"optimizer.pt": 1})
    v = S.verify_salvage(CKPT_FILES, got)
    assert v.status == "partial"
    assert v.short == ("optimizer.pt",)
    assert "TORN" in v.reason


def test_verify_a_LARGER_destination_file_is_also_partial():
    """Bigger is not better — a size mismatch in either direction means these are
    not the bytes we surveyed."""
    got = dict(CKPT_FILES, **{"optimizer.pt": 646_000_001})
    assert S.verify_salvage(CKPT_FILES, got).status == "partial"


def test_verify_unreadable_destination_is_UNVERIFIABLE_never_ok():
    """THE fail-safe: no verification => not salvaged. The only thing worse than
    losing a checkpoint is believing you still have it."""
    v = S.verify_salvage(CKPT_FILES, None)
    assert v.status == "unverifiable"
    assert v.bytes_seen == 0


def test_verify_empty_destination_is_partial_not_unverifiable():
    """'read fine, nothing there' and 'could not read' are different answers and
    are never conflated."""
    assert S.verify_salvage(CKPT_FILES, {}).status == "partial"


def test_verify_empty_source_is_unverifiable_not_ok():
    """A source survey that listed nothing gives us nothing to check against, so
    it cannot be evidence that anything was salvaged."""
    assert S.verify_salvage({}, {}).status == "unverifiable"


# --------------------------------------------------------------------------- #
# destination paths — never collide with a LIVE writer
# --------------------------------------------------------------------------- #
def test_dest_path_stays_out_of_the_live_jobs_tree():
    p = S.dest_path("46859541", "J1", "checkpoint-50")
    assert p.startswith(S.SALVAGE_ROOT + "/")
    assert not p.startswith(S.JOBS_ROOT + "/")
    assert "46859541" in p           # namespaced: two salvages cannot collide


def test_b2_salvage_prefix_never_writes_the_checkpoints_prefix():
    """The replacement job is a LIVE WRITER of jobs/<ID>/checkpoints/ and its
    resume pull-back READS it. Salvage is evidence to inspect, not state to
    silently re-inject."""
    p = S.b2_salvage_prefix("J1", "46859541", "checkpoint-50")
    assert p == "jobs/J1/salvage/46859541/checkpoint-50"
    assert "/checkpoints/" not in "/" + p


# --------------------------------------------------------------------------- #
# pick_dest
# --------------------------------------------------------------------------- #
def test_pick_dest_prefers_the_first_ready_candidate():
    assert S.pick_dest(["A", "B"], {"A": "loading", "B": "running"}) == "B"


def test_pick_dest_refuses_a_loading_box():
    """A copy into a half-materialised rootfs is the silent partial this module
    exists to refuse."""
    assert S.pick_dest(["A"], {"A": "loading"}) is None
    assert S.pick_dest(["A"], {}) is None


# --------------------------------------------------------------------------- #
# advance() — the state machine
# --------------------------------------------------------------------------- #
def _advance(rec, **kw):
    kw.setdefault("now", NOW)
    kw.setdefault("execute", lambda *a: (False, "", "no"))
    kw.setdefault("copy_direct", lambda *a: (True, "initiated", None))
    kw.setdefault("statuses", {"DEST": "running"})
    return S.advance(rec, **kw)


def _rec(**kw):
    kw.setdefault("dest_candidates", ["DEST"])
    return S.new_record("DEAD", now=NOW, **kw)


def test_advance_happy_path_copies_then_verifies():
    calls = []
    rec = _rec()
    _advance(rec, execute=_dead_box_exec(),
             copy_direct=lambda *a: (calls.append(a), (True, "ok", None))[1],
             b2_bytes=lambda jid: {})
    assert rec["phase"] == "copying"
    assert len(calls) == 1
    src_iid, src, dst_iid, dst = calls[0]
    assert src == f"{S.JOBS_ROOT}/J1/work/out/checkpoint-50"
    assert dst_iid == "DEST" and dst.startswith(S.SALVAGE_ROOT)

    dest_root = rec["items"][0]["landed"]

    surveyed = []

    def dest_exec(iid, cmd):
        assert iid == "DEST"
        surveyed.append(cmd)
        if cmd != f"ls -lR {dest_root}":
            return True, f"ls: cannot access '{cmd.split()[-1]}': No such file or directory", None
        return True, _ls_lr({dest_root: [(n, s, False)
                                         for n, s in CKPT_FILES.items()]}), None
    _advance(rec, now=NOW + 60, execute=dest_exec)
    assert surveyed == [f"ls -lR {dest_root}"]     # surveyed the item's OWN path
    assert rec["outcome"] == S.OUTCOME_SALVAGED
    assert rec["items"][0]["verify"] == "ok"


def test_advance_partial_copy_is_reported_as_partial_at_the_deadline():
    rec = _rec()
    _advance(rec, execute=_dead_box_exec(), b2_bytes=lambda jid: {})
    dest_root = rec["items"][0]["landed"]

    def half(iid, cmd):
        return True, _ls_lr({dest_root: [("adapter_model.safetensors",
                                          323_000_000, False)]}), None
    _advance(rec, now=NOW + 10, execute=half)
    assert rec["phase"] == "copying"            # still inside the deadline
    _advance(rec, now=rec["deadline_ts"] + 1, execute=half)
    assert rec["outcome"] == S.OUTCOME_PARTIAL
    assert rec["outcome"] in S.LOUD_OUTCOMES
    assert "DO NOT RESUME" in rec["detail"]


def test_advance_unreadable_destination_ends_UNVERIFIABLE_never_salvaged():
    rec = _rec()
    _advance(rec, execute=_dead_box_exec(), b2_bytes=lambda jid: {})
    blind = lambda *a: (False, "", "instance not found")   # noqa: E731
    _advance(rec, now=rec["deadline_ts"] + 1, execute=blind)
    assert rec["outcome"] == S.OUTCOME_UNVERIFIABLE
    assert rec["outcome"] != S.OUTCOME_SALVAGED


def test_advance_dead_box_already_reclaimed():
    """Only an answer that NAMES the instance as absent earns `dead_box_gone`."""
    rec = _rec()
    _advance(rec, execute=lambda *a: (False, "", "HTTP 404 not_found"))
    assert rec["outcome"] == S.OUTCOME_DEAD_GONE
    assert "does not exist" in rec["detail"]


@pytest.mark.parametrize("err", [
    "result_url never returned 200 (gave up after 25s)",
    "HTTP 503 on PUT /v0/instances/command/1/",
    "URLError: [Errno 111] Connection refused",
    "TimeoutError: timed out",
])
def test_a_TRANSIENT_survey_failure_is_retried_not_declared_fatal(err):
    """THE regression that matters. The survey transport polls an ASYNCHRONOUS
    endpoint on a bounded budget, and `ls -lR` over several ~1 GB checkpoints on
    a host that just had an instance evicted is exactly the slow case. Turning
    that into `dead_box_gone` / `nothing_found` publishes an AUTHORITATIVE
    'everything on that disk is lost' built on a poll timeout — and an operator
    who reads it does not re-run the command."""
    rec = _rec()
    _advance(rec, execute=lambda *a: (False, "", err))
    assert rec["phase"] == "pending" and rec["outcome"] is None
    assert rec["last_survey_error"] == err

    # ... and a later HEALTHY tick still salvages.
    _advance(rec, now=NOW + 60, execute=_dead_box_exec(), b2_bytes=lambda j: {})
    assert rec["phase"] == "copying"
    assert [it["name"] for it in rec["items"]] == ["out/checkpoint-50"]


def test_a_transient_survey_failure_ends_UNVERIFIABLE_at_the_deadline():
    """Never `nothing_found`: we never got a complete survey, so we have no
    evidence about what was on that disk."""
    rec = _rec()
    _advance(rec, now=rec["deadline_ts"] + 1,
             execute=lambda *a: (False, "", "TimeoutError"))
    assert rec["outcome"] == S.OUTCOME_UNVERIFIABLE
    assert "NOT evidence the disk was empty" in rec["detail"]


def test_a_PER_JOB_survey_failure_is_not_a_quiet_skip():
    """The dead box listed its jobs, then the recursive listing failed. Skipping
    that job reports `nothing_found` over a disk full of checkpoints."""
    def flaky(iid, cmd):
        if cmd.startswith("ls -1 "):
            return True, "J1\n", None
        return False, "", "TimeoutError: timed out"
    rec = _rec()
    _advance(rec, execute=flaky)
    assert rec["phase"] == "pending" and rec["outcome"] is None


def test_an_UNPARSED_line_in_the_source_survey_is_not_a_survey():
    """The source map is the ONLY oracle `verify_salvage` checks against. A map
    that silently lost a file cannot notice that the copy lost it too — that is
    how an incomplete copy becomes a reported `salvaged` and gets pushed to B2
    under that label."""
    def truncated(iid, cmd):
        if cmd.startswith("ls -1 "):
            return True, "J1\n", None
        return True, ("/workspace/jobs/J1/work/out/checkpoint-50:\n"
                      "-rw-r--r-- 1 root root 5 Aug  5 07:00 a.bin\n"
                      "-rw-r--r-- 1 root ro"), None      # body cut mid-line
    rec = _rec()
    _advance(rec, execute=truncated)
    assert rec["phase"] == "pending" and rec["outcome"] is None
    assert "unparsed" in rec["last_survey_error"]


def test_advance_dest_never_ready_reports_rather_than_hanging():
    rec = _rec()
    _advance(rec, execute=_dead_box_exec(), b2_bytes=lambda j: {},
             statuses={"DEST": "loading"})
    assert rec["phase"] == "pending"                       # still waiting
    _advance(rec, now=rec["dest_deadline_ts"] + 1, execute=_dead_box_exec(),
             b2_bytes=lambda j: {}, statuses={"DEST": "loading"})
    assert rec["outcome"] == S.OUTCOME_DEST_NOT_READY


def test_a_destination_without_ROOM_is_skipped():
    """Jobs boxes run near full BY DESIGN (that is why jobd prunes), so
    `running` is not evidence of headroom — and a salvage that fills the landing
    box's disk kills whatever is running on it."""
    rec = _rec(dest_candidates=["FULL", "ROOMY"])
    _advance(rec, execute=_dead_box_exec(), b2_bytes=lambda j: {},
             statuses={"FULL": "running", "ROOMY": "running"},
             free_gb={"FULL": 0.2, "ROOMY": 500.0})
    assert rec["dest_iid"] == "ROOMY"


def test_unknown_free_space_does_not_disqualify_a_destination():
    """Refusing every box on missing telemetry would disable salvage entirely."""
    rec = _rec(dest_candidates=["UNKNOWN"])
    _advance(rec, execute=_dead_box_exec(), b2_bytes=lambda j: {},
             statuses={"UNKNOWN": "running"}, free_gb={})
    assert rec["dest_iid"] == "UNKNOWN"


def test_advance_nothing_newer_when_b2_is_current():
    rec = _rec()
    _advance(rec, execute=_dead_box_exec(),
             b2_bytes=lambda jid: {"out/checkpoint-50": CKPT_BYTES})
    assert rec["outcome"] == S.OUTCOME_NOTHING_NEWER


def test_advance_nothing_found_on_an_empty_disk():
    rec = _rec()
    _advance(rec, execute=lambda iid, cmd: (True, "", None))
    assert rec["outcome"] == S.OUTCOME_NOTHING_FOUND


def test_advance_copy_refused_when_vast_says_no():
    rec = _rec()
    _advance(rec, execute=_dead_box_exec(), b2_bytes=lambda jid: {},
             copy_direct=lambda *a: (False, "", "src_path not supported"))
    assert rec["outcome"] == S.OUTCOME_COPY_REFUSED


def test_advance_dry_run_initiates_nothing():
    calls = []
    rec = _rec()
    _advance(rec, execute=_dead_box_exec(), b2_bytes=lambda jid: {},
             copy_direct=lambda *a: (calls.append(a), (True, "", None))[1],
             dry_run=True)
    assert calls == []
    assert rec["outcome"] == S.OUTCOME_DISABLED


def test_advance_is_idempotent_once_done():
    rec = _rec()
    _advance(rec, execute=lambda *a: (False, "", "HTTP 404 not_found"))
    before = dict(rec)
    _advance(rec, execute=lambda *a: (True, "ANYTHING", None))
    assert rec["outcome"] == before["outcome"] == S.OUTCOME_DEAD_GONE


def test_push_to_b2_only_runs_for_a_VERIFIED_item():
    """Pushing an unverified copy is how a torn checkpoint gets laundered into
    durable storage. It must be gated on `verify == ok`."""
    pushes = []
    rec = _rec()
    _advance(rec, execute=_dead_box_exec(), b2_bytes=lambda jid: {})
    dest_root = rec["items"][0]["landed"]
    short = lambda i, c: (True, _ls_lr({dest_root: [("a", 1, False)]}), None)  # noqa: E731
    _advance(rec, now=NOW + 10, execute=short,
             push_to_b2=lambda *a: (pushes.append(a), (True, "ok"))[1])
    assert pushes == []

    full = lambda i, c: (True, _ls_lr(  # noqa: E731
        {dest_root: [(n, s, False) for n, s in CKPT_FILES.items()]}), None)
    _advance(rec, now=NOW + 20, execute=full,
             push_to_b2=lambda *a: (pushes.append(a), (True, "ok"))[1])
    assert len(pushes) == 1
    assert pushes[0][2] == "jobs/J1/salvage/DEAD/out/checkpoint-50"


def test_failed_b2_push_still_counts_as_salvaged_but_is_recorded():
    """The bytes are already safe on a box we own; a failed push downgrades the
    report, not the outcome."""
    rec = _rec()
    _advance(rec, execute=_dead_box_exec(), b2_bytes=lambda jid: {})
    dest_root = rec["items"][0]["landed"]
    full = lambda i, c: (True, _ls_lr(  # noqa: E731
        {dest_root: [(n, s, False) for n, s in CKPT_FILES.items()]}), None)
    _advance(rec, now=NOW + 20, execute=full,
             push_to_b2=lambda *a: (False, "no bucket"))
    assert rec["outcome"] == S.OUTCOME_SALVAGED
    assert rec["items"][0]["b2"] is None
    assert "0/1 pushed to B2" in rec["detail"]


# --------------------------------------------------------------------------- #
# herdd wiring
# --------------------------------------------------------------------------- #
def _jc(**kw):
    import argparse
    jc = {"a": argparse.Namespace(), "iid": "NEW", "instances": [],
          "dry_run": False}
    jc.update(kw)
    return jc


def test_retain_arms_salvage_at_the_MOMENT_of_eviction(monkeypatch):
    """The race is HOST RECLAMATION (~30 min observed), not the 3h window — so
    the record has to exist before anyone reads a runbook."""
    monkeypatch.setattr(vlifecycle, "_put_label_soft", lambda *a: (True, None))
    monkeypatch.setattr(vjournal, "_job_handoff_emit", lambda *a, **k: {})
    jc = _jc()
    rec = vretention._job_retain_or_destroy(jc, "DEAD", {"id": "DEAD"}, "outbid", NOW,
                                            new_iid="NEW")
    assert rec["status"] == "retained"
    assert rec["salvage"]["phase"] == "pending"
    assert rec["salvage"]["dead_iid"] == "DEAD"
    assert "NEW" in rec["salvage"]["dest_candidates"]


def test_already_gone_records_dead_box_gone_rather_than_silence(monkeypatch):
    """`dead_box_gone` is the measured failure rate of the whole idea; folding it
    into 'we didn't try' hides the number that decides whether salvage is worth
    it."""
    monkeypatch.setattr(vjournal, "_job_handoff_emit", lambda *a, **k: {})
    jc = _jc()
    rec = vretention._job_retain_or_destroy(jc, "DEAD", None, "host_failure", NOW,
                                            new_iid="NEW")
    assert rec["status"] == "already_gone"
    assert rec["salvage"]["outcome"] == S.OUTCOME_DEAD_GONE


def test_no_salvage_flag_disarms_it(monkeypatch):
    import argparse
    monkeypatch.setattr(vlifecycle, "_put_label_soft", lambda *a: (True, None))
    monkeypatch.setattr(vjournal, "_job_handoff_emit", lambda *a, **k: {})
    jc = _jc(a=argparse.Namespace(salvage=False))
    rec = vretention._job_retain_or_destroy(jc, "DEAD", {"id": "DEAD"}, "outbid", NOW,
                                            new_iid="NEW")
    assert rec["salvage"] is None


def test_retention_sweep_advances_salvage(monkeypatch):
    seen = []
    monkeypatch.setattr(vretention, "_job_salvage_advance",
                        lambda jc, rec, now: seen.append(rec))
    jc = _jc(retained_boxes=[{"iid": "DEAD", "status": "retained",
                              "deadline_ts": NOW + 9999,
                              "salvage": {"phase": "pending",
                                          "dead_iid": "DEAD"}}])
    vretention._job_retention_sweep(jc, NOW)
    assert len(seen) == 1


def test_salvage_step_error_never_kills_the_supervision_loop(monkeypatch):
    """...and a record that can NEVER advance must still let the box go. The
    try/except is what makes a permanently-stuck record possible, so the two
    properties belong in one test."""
    def boom(*a, **k):
        raise RuntimeError("api exploded")
    monkeypatch.setattr(vretention, "_job_salvage_advance", boom)
    monkeypatch.setattr(vjournal, "_job_handoff_emit", lambda *a, **k: {})
    destroyed = []
    monkeypatch.setattr(vlifecycle, "_destroy_and_revoke",
                        lambda ids, inst, why: destroyed.extend(ids) or [])
    sal = {"phase": "pending", "dead_iid": "DEAD", "deadline_ts": NOW - 7200,
           "started_ts": NOW - 10800}
    jc = _jc(instances=[{"id": "DEAD", "actual_status": "exited"}],
             retained_boxes=[{"iid": "DEAD", "status": "expired",
                              "deadline_ts": NOW - 10 * 3600, "salvage": sal}])
    for i in range(500):                     # 500 ticks of a stuck record
        vretention._job_retention_sweep(jc, NOW + i * 3600)   # must not raise
    assert sal["phase"] == "pending"          # it really never advanced
    assert destroyed == ["DEAD"]              # and the box was still reclaimed


def test_retention_backstop_DEFERS_while_a_copy_is_in_flight(monkeypatch):
    """Destroying the source mid-transfer aborts the copy — the exact data loss
    the backstop sits downstream of. Salvage has its own deadline, so this can
    only defer by a bounded amount."""
    destroyed = []
    monkeypatch.setattr(vlifecycle, "_destroy_and_revoke",
                        lambda ids, inst, why: destroyed.extend(ids) or [])
    monkeypatch.setattr(vjournal, "_job_handoff_emit", lambda *a, **k: {})
    monkeypatch.setattr(vretention, "_job_salvage_sweep", lambda jc, now: None)
    dl = NOW - 10 * 3600
    jc = _jc(instances=[{"id": "DEAD", "actual_status": "exited"}],
             retained_boxes=[{"iid": "DEAD", "status": "expired",
                              "deadline_ts": dl,
                              "salvage": {"phase": "copying",
                                          "dead_iid": "DEAD",
                                          "deadline_ts": NOW + 600}}])
    vretention._job_retention_sweep(jc, NOW)
    assert destroyed == []

    jc["retained_boxes"][0]["salvage"]["phase"] = "done"
    vretention._job_retention_sweep(jc, NOW)
    assert destroyed == ["DEAD"]


def test_retention_deferral_is_BOUNDED_by_the_salvage_deadline(monkeypatch):
    """A record whose advance keeps throwing never reaches a terminal phase.
    Deferring on `phase != done` alone would hold a BILLING box open forever —
    the exact waste the backstop exists to stop."""
    destroyed = []
    monkeypatch.setattr(vlifecycle, "_destroy_and_revoke",
                        lambda ids, inst, why: destroyed.extend(ids) or [])
    monkeypatch.setattr(vjournal, "_job_handoff_emit", lambda *a, **k: {})
    monkeypatch.setattr(vretention, "_job_salvage_sweep", lambda jc, now: None)
    sal = {"phase": "copying", "dead_iid": "DEAD",
           "deadline_ts": NOW - 2 * 3600, "started_ts": NOW - 3 * 3600}
    jc = _jc(instances=[{"id": "DEAD", "actual_status": "exited"}],
             retained_boxes=[{"iid": "DEAD", "status": "expired",
                              "deadline_ts": NOW - 10 * 3600, "salvage": sal}])
    vretention._job_retention_sweep(jc, NOW)
    assert destroyed == ["DEAD"]


def test_salvage_defer_until_falls_back_when_the_deadline_is_missing():
    """A record persisted by an older daemon may lack `deadline_ts`; it must
    still expire rather than defer forever."""
    assert vretention._salvage_defer_until({"started_ts": 100.0}) == \
        100.0 + S.SALVAGE_DEADLINE_S + vretention.SALVAGE_DEFER_GRACE_S
    # a record with neither field bounds at epoch 0 + the windows, i.e. long
    # past for any real clock — an absent deadline can never mean "forever"
    fallback = S.SALVAGE_DEADLINE_S + vretention.SALVAGE_DEFER_GRACE_S
    assert vretention._salvage_defer_until({}) == fallback
    assert vretention._salvage_defer_until({"deadline_ts": "junk",
                                            "started_ts": None}) == fallback
    assert vretention._salvage_defer_until({}) < NOW


def test_salvage_b2_bytes_returns_None_when_rclone_fails(monkeypatch):
    """Unreadable B2 must be `None` ('unknown'), which plan_salvage turns into
    'copy anyway'. Returning {} here would read as 'B2 is empty'... which happens
    to be safe, but returning {} on a LISTING ERROR when B2 is actually full is
    the same code path, and that one is not."""
    monkeypatch.setenv("B2_BUCKET", "bkt")
    monkeypatch.setattr(vb2, "_rclone_soft", lambda args: (1, "", "boom"))
    assert vsalvage._salvage_b2_bytes("J1") is None
    monkeypatch.delenv("B2_BUCKET")
    assert vsalvage._salvage_b2_bytes("J1") is None


def test_salvage_b2_bytes_sums_per_checkpoint(monkeypatch):
    monkeypatch.setenv("B2_BUCKET", "bkt")
    monkeypatch.setattr(vb2, "_rclone_soft", lambda args: (0, (
        "10|checkpoint-50/optimizer.pt\n"
        "5|checkpoint-50/adapter.safetensors\n"
        "7|checkpoint-100/optimizer.pt\n"
        "3|log.txt\n"), ""))
    assert vsalvage._salvage_b2_bytes("J1") == {"checkpoint-50": 15,
                                                "checkpoint-100": 7}


def test_push_script_writes_ONLY_the_prefix_it_was_given():
    """Not a check that the string `checkpoints` is absent — none of the three
    arguments contains it, so that would assert a property of the test's own
    inputs and would still pass if the function hardcoded a write to
    `checkpoints/`. Assert instead that every b2: destination in the script is
    the prefix passed in."""
    pfx = S.b2_salvage_prefix("J1", "DEAD", "out/checkpoint-50")
    sh = vsalvage._salvage_push_script("/workspace/salvage/DEAD/J1/out/checkpoint-50",
                                       "bkt", pfx)
    dests = re.findall(r"b2:[^\"\'\s]+", sh)
    assert dests, "the script names no b2 destination at all"
    assert all(d.rstrip("/") == f"bkt/{pfx}" for d in
               [d.replace("b2:", "") for d in dests]), dests
    # and the prefix itself is the one that avoids the live writer
    assert "/checkpoints/" not in "/" + pfx


def test_push_to_b2_refuses_without_a_bucket(monkeypatch):
    monkeypatch.delenv("B2_BUCKET", raising=False)
    ok, detail = vsalvage._salvage_push_to_b2("DEST", "/p", "jobs/J1/salvage/x")
    assert ok is False and "B2_BUCKET" in detail


def test_execute_soft_poll_gives_up_bounded(monkeypatch):
    """A survey that never returns must END, not spin — every wait carries a
    deadline, including this one."""
    monkeypatch.setattr(vapi, "request_soft",
                        lambda *a, **k: (True, {"success": True,
                                                "result_url": "http://x/y",
                                                "writeable_path": ""}, None))

    def boom(*a, **k):
        raise OSError("refused")
    monkeypatch.setattr(urllib.request, "urlopen", boom)
    ok, text, err = vremote._vast_execute_soft("DEAD", "ls -1 /", tries=3,
                                               _sleep=lambda s: None)
    assert ok is False and "refused" in err


def test_execute_soft_reports_a_404_as_failure(monkeypatch):
    """`404 Instance not found` on the dead box is the meaningful answer: the
    host reclaimed it."""
    monkeypatch.setattr(vapi, "request_soft",
                        lambda *a, **k: (False, None, "HTTP 404 not_found"))
    ok, _, err = vremote._vast_execute_soft("DEAD", "ls -1 /", _sleep=lambda s: None)
    assert ok is False and "404" in err


def test_copy_direct_soft_maps_success_false_to_failure(monkeypatch):
    monkeypatch.setattr(vapi, "request_soft",
                        lambda *a, **k: (True, {"success": False,
                                                "msg": "src_path not supported"},
                                         None))
    ok, _, err = vremote._vast_copy_direct_soft("A", "/a", "B", "/b")
    assert ok is False and "src_path" in err


def test_copy_direct_sends_int_instance_ids(monkeypatch):
    sent = {}

    def cap(method, path, body=None, **k):
        sent.update(body or {})
        return True, {"success": True, "msg": "ok"}, None
    monkeypatch.setattr(vapi, "request_soft", cap)
    vremote._vast_copy_direct_soft("46859541", "/a", "46861999", "/b")
    assert sent["src_id"] == 46859541 and sent["dst_id"] == 46861999
    assert sent["client_id"] == "me"


# --------------------------------------------------------------------------- #
# the checkpoint is NOT at a fixed depth (regression: the live layout)
# --------------------------------------------------------------------------- #
def test_split_ckpt_rel_finds_the_checkpoint_at_any_depth():
    assert S.split_ckpt_rel("out/checkpoint-50/optimizer.pt") == \
        ("out/checkpoint-50", 50, "optimizer.pt")
    assert S.split_ckpt_rel("arms/hex/checkpoint-50") == \
        ("arms/hex/checkpoint-50", 50, "")
    assert S.split_ckpt_rel("checkpoint-50/a") == ("checkpoint-50", 50, "a")
    assert S.split_ckpt_rel("out/logs") == (None, None, None)
    assert S.split_ckpt_rel("") == (None, None, None)


def test_ckpt_dirs_finds_the_REAL_nested_layout():
    """jobd's tree is work/out/checkpoint-<N>/. Keying on the first path segment
    finds NOTHING there — a silent `nothing_found` over a full disk."""
    root = "/workspace/jobs/J1/work"
    sections = S.parse_ls_lr(_ls_lr({
        root: [("out", 4096, True)],
        f"{root}/out": [("checkpoint-50", 4096, True)],
        f"{root}/out/checkpoint-50": [("optimizer.pt", 646, False)],
    }))
    dirs = S.ckpt_dirs_from_survey(sections, root)
    assert [d.name for d in dirs] == ["out/checkpoint-50"]
    assert dirs[0].files == {"optimizer.pt": 646}


def test_ckpt_dirs_keeps_multi_arm_roots_apart():
    root = "/workspace/jobs/J1/work"
    sections = S.parse_ls_lr(_ls_lr({
        f"{root}/arms/a/checkpoint-50": [("x", 1, False)],
        f"{root}/arms/b/checkpoint-50": [("x", 2, False)],
    }))
    dirs = S.ckpt_dirs_from_survey(sections, root)
    assert sorted(d.name for d in dirs) == ["arms/a/checkpoint-50",
                                            "arms/b/checkpoint-50"]


def test_salvage_b2_bytes_keys_match_the_box_side_survey(monkeypatch):
    """The B2 keys and the box-side CkptDir.name must be the SAME string, or
    every checkpoint reads as already-synced and salvage does nothing."""
    monkeypatch.setenv("B2_BUCKET", "bkt")
    monkeypatch.setattr(vb2, "_rclone_soft", lambda args: (0, (
        "10|out/checkpoint-50/optimizer.pt\n"
        "5|out/checkpoint-50/adapter.safetensors\n"
        "7|out/checkpoint-100/optimizer.pt\n"
        "3|out/trainer_state.json\n"), ""))
    got = vsalvage._salvage_b2_bytes("J1")
    assert got == {"out/checkpoint-50": 15, "out/checkpoint-100": 7}

    root = "/workspace/jobs/J1/work"
    sections = S.parse_ls_lr(_ls_lr({
        f"{root}/out/checkpoint-50": [("optimizer.pt", 10, False),
                                      ("adapter.safetensors", 5, False)],
    }))
    dirs = S.ckpt_dirs_from_survey(sections, root)
    assert dirs[0].name in got
    assert S.plan_salvage(dirs, b2_bytes=got).action == "nothing"


# --------------------------------------------------------------------------- #
# the dest-survey trichotomy must survive the PRODUCTION transport
# --------------------------------------------------------------------------- #
def test_dest_survey_absent_path_is_EMPTY_not_unreadable():
    """vast's `execute` returns HTTP 200 with the command's stderr in the body —
    it never surfaces the command's exit code. So `ok` alone cannot tell "the
    path is not there" from "we could not read". Key on the ls error text."""
    out = "ls: cannot access '/workspace/salvage/x': No such file or directory"
    got = S.survey_dest_files("D", "/workspace/salvage/x",
                              execute=lambda i, c: (True, out, None))
    assert got == {}
    assert S.verify_salvage(CKPT_FILES, got).status == "partial"


def test_dest_survey_unparseable_output_is_UNREADABLE_not_empty():
    """A body we cannot fully account for must not read as 'nothing landed' —
    that is the difference between `unverifiable` and `partial`."""
    got = S.survey_dest_files("D", "/p",
                              execute=lambda i, c: (True, "?? garbage ??", None))
    assert got is None
    assert S.verify_salvage(CKPT_FILES, got).status == "unverifiable"


def test_dest_survey_transport_failure_is_UNREADABLE():
    got = S.survey_dest_files("D", "/p",
                              execute=lambda i, c: (False, "", "boom"))
    assert got is None


def test_a_total_blackout_destination_ends_UNVERIFIABLE_not_partial():
    rec = _rec()
    _advance(rec, execute=_dead_box_exec(), b2_bytes=lambda j: {})
    blind = lambda i, c: (True, "?? garbage ??", None)      # noqa: E731
    _advance(rec, now=rec["deadline_ts"] + 1, execute=blind)
    assert rec["outcome"] == S.OUTCOME_UNVERIFIABLE


def test_parse_ls_l_strict_counts_what_it_could_not_account_for():
    ent, residual, absent = S.parse_ls_l_strict(
        "total 8\n"
        "-rw-r--r-- 1 root root 5 Aug  5 07:00 real.bin\n"
        "ls: cannot access '/nope': No such file or directory\n"
        "-rw-r--r-- 1 root ro\n")                 # truncated mid-line
    assert [e.name for e in ent] == ["real.bin"]
    assert absent == 1
    assert residual == 1


def test_strip_writeable_path_is_anchored_to_line_starts():
    """Upstream does a bare whole-body replace. An unanchored strip of a short or
    common prefix corrupts FILENAMES in the survey the byte-for-byte
    verification is checked against, and mangles the section headers the parser
    splits on."""
    text = ("/data:\n"
            "-rw-r--r-- 1 root root 5 Aug  5 07:00 my/data/file.bin\n")
    out = vremote._strip_writeable_path(text, "/data")
    assert out.splitlines()[0] == ":"
    assert out.splitlines()[1].endswith("my/data/file.bin")
    assert vremote._strip_writeable_path(text, "") == text


def test_execute_soft_poll_is_bounded_by_WALL_CLOCK_not_just_tries(monkeypatch):
    monkeypatch.setattr(vapi, "request_soft",
                        lambda *a, **k: (True, {"success": True,
                                                "result_url": "http://x/y",
                                                "writeable_path": ""}, None))
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("refused")))
    clock = {"t": 0.0}
    ok, _, err = vremote._vast_execute_soft(
        "DEAD", "ls -lR /", tries=10_000, budget_s=5.0,
        _sleep=lambda s: clock.__setitem__("t", clock["t"] + s),
        _now=lambda: clock["t"])
    assert ok is False and "gave up after 5s" in err
    assert clock["t"] < 20.0            # backoff, not 10k tight retries


def test_salvage_b2_bytes_empty_listing_means_B2_IS_EMPTY_not_unknown():
    """rc=0 with no output is a real answer: B2 holds nothing under that prefix,
    so copy everything. Only a FAILED listing is `None` ("unknown"), which
    plan_salvage also turns into copy-everything — the two agree here, and the
    distinction only matters when B2 is full and the listing broke."""
    import os as _os
    _os.environ["B2_BUCKET"] = "bkt"
    try:
        old = vb2._rclone_soft
        vb2._rclone_soft = lambda args: (0, "", "")
        assert vsalvage._salvage_b2_bytes("J1") == {}
        vb2._rclone_soft = lambda args: (1, "", "b2 unreachable")
        assert vsalvage._salvage_b2_bytes("J1") is None
    finally:
        vb2._rclone_soft = old
        _os.environ.pop("B2_BUCKET", None)


# --------------------------------------------------------------------------- #
# MECHANICS LEARNED FROM THE LIVE copy_direct RUN (2026-08-05)
#
# All four of these were wrong in the first implementation and all four were
# invisible to a fixture-only test suite, because a fixture answers whatever the
# author expected. They are pinned here against the OBSERVED behaviour of a real
# ~1 GB transfer, 46861081 -> 46864611 (both `exited`, no GPU contract):
# 12/12 files, 980,767,613 B, byte-exact.
# --------------------------------------------------------------------------- #
def test_copy_direct_NESTS_the_source_basename_in_the_destination():
    """`copy_direct` is rsync with NO trailing slash on the source, so it copies
    the source directory INTO the destination. Verifying at the destination
    itself finds one directory and zero files, forever — the verifier correctly
    refuses to call that success, so this path bug would have presented as a
    permanent `salvaged_partial` and looked like a data problem."""
    t = S.dest_path("46861081", "J1", "out/checkpoint-100", flat=True)
    assert t == "/workspace/salvage-46861081-J1-out_checkpoint-100"
    assert S.landed_path(t, "out/checkpoint-100") == t + "/checkpoint-100"


def test_nested_target_mirrors_the_source_tree():
    t = S.dest_path("DEAD", "J1", "out/checkpoint-100")
    assert t == "/workspace/salvage/DEAD/J1/out"
    assert S.landed_path(t, "out/checkpoint-100") == \
        "/workspace/salvage/DEAD/J1/out/checkpoint-100"


def test_a_checkpoint_with_no_parent_dir_still_resolves():
    t = S.dest_path("DEAD", "J1", "checkpoint-50")
    assert S.landed_path(t, "checkpoint-50") == "/workspace/salvage/DEAD/J1/checkpoint-50"


def test_verification_and_b2_push_both_use_the_LANDED_path():
    pushes, surveyed = [], []
    rec = _rec()
    _advance(rec, execute=_dead_box_exec(), b2_bytes=lambda j: {},
             prepare_dest=lambda iid, parents: True)
    it = rec["items"][0]
    assert it["landed"] == it["dest"] + "/checkpoint-50"

    def exec_(iid, cmd):
        surveyed.append(cmd)
        return True, _ls_lr({it["landed"]: [(n, s, False)
                                            for n, s in CKPT_FILES.items()]}), None
    _advance(rec, now=NOW + 10, execute=exec_,
             push_to_b2=lambda *a: (pushes.append(a), (True, "ok"))[1])
    assert surveyed == [f"ls -lR {it['landed']}"]
    assert pushes[0][1] == it["landed"]


def test_copy_direct_does_NOT_mkdir_p_so_a_stopped_dest_gets_the_FLAT_layout():
    """`execute` offers `ls`/`rm`/`du` and nothing that makes a directory, so a
    stopped destination cannot be prepared. OBSERVED failure of the nested
    attempt, AFTER the API answered `success: true`:
        rsync: [Receiver] mkdir ".../out/checkpoint-100" failed:
        No such file or directory (2)
    """
    rec = _rec()
    _advance(rec, execute=_dead_box_exec(), b2_bytes=lambda j: {},
             prepare_dest=lambda iid, parents: False)
    assert rec["dest_layout"] == "flat"
    d = rec["items"][0]["dest"]
    assert d.startswith("/workspace/salvage-")     # ONE new component under /workspace
    assert d.count("/") == 2


def test_a_preparable_destination_gets_the_readable_NESTED_layout():
    asked = []
    rec = _rec()
    _advance(rec, execute=_dead_box_exec(), b2_bytes=lambda j: {},
             prepare_dest=lambda iid, parents: asked.append((iid, parents)) or True)
    assert rec["dest_layout"] == "nested"
    assert asked == [("DEST", ["/workspace/salvage/DEAD/J1/out"])]


def test_a_raising_prepare_dest_degrades_to_flat_rather_than_failing():
    rec = _rec()
    _advance(rec, execute=_dead_box_exec(), b2_bytes=lambda j: {},
             prepare_dest=lambda i, p: (_ for _ in ()).throw(RuntimeError("ssh")))
    assert rec["dest_layout"] == "flat"
    assert rec["phase"] == "copying"


def test_no_prepare_dest_transport_at_all_means_flat():
    rec = _rec()
    _advance(rec, execute=_dead_box_exec(), b2_bytes=lambda j: {})
    assert rec["dest_layout"] == "flat"


def test_dest_survey_uses_the_SEPARATE_destination_transport():
    """`execute` is refused on a RUNNING instance (`400 … "Execute command only
    avail on stopped instances"`), and the destination is running by
    construction. One transport for both ends would report
    `salvage_unverifiable` on every real salvage — the fail-safe holding while
    the happy path never once occurs."""
    src_calls, dst_calls = [], []

    def src(iid, cmd):
        src_calls.append(iid)
        return _dead_box_exec()(iid, cmd)

    def dst(iid, cmd):
        dst_calls.append(iid)
        return True, "", None
    rec = _rec()
    S.advance(rec, now=NOW, execute=src, dest_execute=dst,
              copy_direct=lambda *a: (True, "ok", None),
              statuses={"DEST": "running"}, b2_bytes=lambda j: {})
    assert src_calls and not dst_calls          # pending phase: source only
    S.advance(rec, now=NOW + 10, execute=src, dest_execute=dst,
              copy_direct=lambda *a: (True, "ok", None),
              statuses={"DEST": "running"})
    assert dst_calls == ["DEST"]                # copying phase: destination only


def test_the_rsync_error_is_surfaced_from_the_SOURCE_status_msg():
    """vast reports a failed host-to-host transfer in the SOURCE instance's
    `status_msg` and nowhere else — the API's `success: true` only means the
    request was accepted. Without this a partial salvage is opaque."""
    rec = _rec()
    _advance(rec, execute=_dead_box_exec(), b2_bytes=lambda j: {})
    msg = ('rsync: [Receiver] mkdir "/workspace/salvage/x" failed: '
           'No such file or directory (2)\nrsync error: code 11')
    _advance(rec, now=rec["deadline_ts"] + 1,
             execute=lambda i, c: (True, "", None),
             copy_status=lambda iid: msg)
    assert rec["outcome"] in (S.OUTCOME_PARTIAL, S.OUTCOME_UNVERIFIABLE)
    assert "mkdir" in rec["detail"]
    assert "rsync error: code 11" in rec["copy_status_msg"]


def test_a_broken_copy_status_probe_never_changes_the_verdict():
    rec = _rec()
    _advance(rec, execute=_dead_box_exec(), b2_bytes=lambda j: {})
    _advance(rec, now=rec["deadline_ts"] + 1,
             execute=lambda i, c: (True, "", None),
             copy_status=lambda iid: (_ for _ in ()).throw(OSError("api")))
    assert rec["outcome"] in (S.OUTCOME_PARTIAL, S.OUTCOME_UNVERIFIABLE)
    assert rec["copy_status_msg"] is None


# --------------------------------------------------------------------------- #
# the herdd transport pair
# --------------------------------------------------------------------------- #
def test_dest_exec_uses_ssh_for_a_RUNNING_box_and_execute_for_a_STOPPED_one(
        monkeypatch):
    seen = []
    monkeypatch.setattr(vremote, "_ssh_exec_soft",
                        lambda i, r, **k: (seen.append(("ssh", i)), (True, "", None))[1])
    monkeypatch.setattr(vremote, "_vast_execute_soft",
                        lambda i, c, **k: (seen.append(("exec", i)), (True, "", None))[1])
    ex = vsalvage._mk_salvage_dest_exec({"R": "running", "S": "exited"})
    ex("R", "ls -lR /x")
    ex("S", "ls -lR /x")
    assert seen == [("ssh", "R"), ("exec", "S")]


def test_dest_exec_falls_back_to_ssh_when_vast_refuses_a_running_box(monkeypatch):
    """A stale status snapshot must degrade, not fail."""
    seen = []
    monkeypatch.setattr(vremote, "_ssh_exec_soft",
                        lambda i, r, **k: (seen.append("ssh"), (True, "out", None))[1])
    monkeypatch.setattr(
        vremote, "_vast_execute_soft",
        lambda i, c, **k: (False, "", "HTTP 400 …: Execute command only avail on "
                                      "stopped instances."))
    ok, text, _ = vsalvage._mk_salvage_dest_exec({})("X", "ls -lR /x")
    assert seen == ["ssh"] and ok and text == "out"


def test_ssh_exec_mirrors_execute_semantics_for_a_failing_command(monkeypatch):
    """`execute` returns HTTP 200 with the command's stderr in the body and never
    its exit code. ssh must classify the same way or the three-way absent /
    unreadable / listing split changes meaning by transport."""
    monkeypatch.setattr(vlifecycle, "_get_instance", lambda i: {"actual_status": "running"})
    monkeypatch.setattr(vssh, "_pick_ssh_endpoint", lambda i, **k: ("h", 22, "d"))

    class R:
        returncode, stdout, stderr = 2, "", ("ls: cannot access '/x': "
                                             "No such file or directory\n")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: R())
    ok, text, err = vremote._ssh_exec_soft("X", "ls -lR /x")
    assert ok is True and err is None
    assert S.survey_dest_files("X", "/x", execute=lambda i, c: (ok, text, err)) == {}


def test_ssh_exec_reports_a_TRANSPORT_failure_as_not_ok(monkeypatch):
    monkeypatch.setattr(vlifecycle, "_get_instance", lambda i: {"actual_status": "running"})
    monkeypatch.setattr(vssh, "_pick_ssh_endpoint", lambda i, **k: ("h", 22, "d"))

    class R:
        returncode, stdout, stderr = 255, "", "Connection refused\n"
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: R())
    ok, _, err = vremote._ssh_exec_soft("X", "ls /x")
    assert ok is False and "Connection refused" in err


def test_prepare_dest_refuses_a_STOPPED_box_without_attempting_ssh(monkeypatch):
    calls = []
    monkeypatch.setattr(vremote, "_ssh_exec_soft",
                        lambda i, r, **k: calls.append(r) or (True, "", None))
    assert vsalvage._mk_salvage_prepare_dest({"S": "exited"})("S", ["/a"]) is False
    assert calls == []


def test_prepare_dest_mkdir_p_s_every_parent_quoted(monkeypatch):
    calls = []
    monkeypatch.setattr(vremote, "_ssh_exec_soft",
                        lambda i, r, **k: calls.append(r) or (True, "", None))
    assert vsalvage._mk_salvage_prepare_dest({"R": "running"})(
        "R", ["/workspace/salvage/a b", "/workspace/salvage/c"]) is True
    assert "mkdir -p '/workspace/salvage/a b' /workspace/salvage/c" in calls[0]


def test_copy_status_reads_the_instance_status_msg(monkeypatch):
    monkeypatch.setattr(vapi, "request_soft",
                        lambda *a, **k: (True, {"instances":
                                                {"status_msg": "Done copying"}}, None))
    assert vsalvage._salvage_copy_status(1) == "Done copying"
    monkeypatch.setattr(vapi, "request_soft", lambda *a, **k: (False, None, "boom"))
    assert vsalvage._salvage_copy_status(1) is None


# --------------------------------------------------------------------------- #
# `execute`'s result_url is a FIXED PER-INSTANCE path (defect, 2026-08-05)
#
# Two callers surveying one box poll the SAME url and can read each other's
# output; one caller can read its own previous output. Salvage's byte-for-byte
# verification treats that listing as its ORACLE, so a crossed read is a wrong
# answer published as truth. Observed crossed once under concurrency; a solo
# re-test was 0/6 — which is why these tests construct the crossing directly
# instead of hoping to reproduce it.
# --------------------------------------------------------------------------- #

class _FakeResp:
    def __init__(self, body, status=200):
        self.body, self.status = body, status

    def read(self):
        return self.body.encode()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _capture_put(monkeypatch, url="http://x/y", writeable_path=""):
    """Record the command herdd actually PUTs, and answer success."""
    sent = []

    def _req(method, path, body=None, **k):
        sent.append((path, (body or {}).get("command")))
        return True, {"success": True, "result_url": url,
                      "writeable_path": writeable_path}, None

    monkeypatch.setattr(vapi, "request_soft", _req)
    return sent


def _nonce_of(command):
    m = re.search(r"__herdd_exec_([0-9a-f]+)_BEGIN__", command)
    assert m, f"no nonce marker in the PUT command: {command!r}"
    return m.group(1)


def test_execute_brackets_the_command_with_a_per_call_nonce(monkeypatch):
    sent = _capture_put(monkeypatch)
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda *a, **k: _FakeResp(""))
    vremote._vast_execute_soft("DEAD", "ls -lR /w", tries=1, _sleep=lambda s: None)
    vremote._vast_execute_soft("DEAD", "ls -lR /w", tries=1, _sleep=lambda s: None)
    assert len(sent) == 2
    # the real command still travels, intact
    assert all("ls -lR /w" in c for _, c in sent)
    # ... and every call gets its OWN nonce, or two concurrent callers would be
    # indistinguishable and the guard would be decorative
    assert _nonce_of(sent[0][1]) != _nonce_of(sent[1][1])


def test_execute_returns_only_the_block_between_its_own_markers(monkeypatch):
    sent = _capture_put(monkeypatch)
    bodies = []

    def _open(*a, **k):
        return _FakeResp(bodies[0])

    monkeypatch.setattr(urllib.request, "urlopen", _open)

    # Build the body the host would write for THIS call, with another caller's
    # listing sitting in the same log around it.
    def _req(method, path, body=None, **k):
        cmd = (body or {}).get("command")
        n = _nonce_of(cmd)
        bodies.append("\n".join([
            "/other/callers/listing:",
            "-rw-r--r-- 1 root root 9 Jan 1 00:00 NOT_OURS",
            f"__herdd_exec_{n}_BEGIN__",
            "/w:",
            "-rw-r--r-- 1 root root 5 Jan 1 00:00 ours.bin",
            f"__herdd_exec_{n}_END__",
        ]))
        return True, {"success": True, "result_url": "http://x/y",
                      "writeable_path": ""}, None

    monkeypatch.setattr(vapi, "request_soft", _req)
    ok, text, err = vremote._vast_execute_soft("DEAD", "ls -lR /w", tries=2,
                                               _sleep=lambda s: None)
    assert ok is True and err is None
    assert "ours.bin" in text
    assert "NOT_OURS" not in text          # the crossed listing is excluded
    assert "herdd_exec" not in text      # markers do not leak into the oracle
    assert sent == []                      # (the _req above replaced the capture)


def test_execute_correlates_even_when_vast_prefixes_the_marker_lines(monkeypatch):
    """vast prepends `writeable_path` to lines in some responses. An
    exact-line-match correlation would then never fire — and a guard that never
    correlates looks exactly like a guard that is working, while disabling every
    survey. Match by containment."""
    pfx = "/var/lib/docker/vol/"
    bodies = []

    def _req(method, path, body=None, **k):
        n = _nonce_of((body or {}).get("command"))
        bodies.append("\n".join([
            f"{pfx}__herdd_exec_{n}_BEGIN__",
            f"{pfx}w:",
            "-rw-r--r-- 1 root root 5 Jan 1 00:00 ours.bin",
            f"{pfx}__herdd_exec_{n}_END__",
        ]))
        return True, {"success": True, "result_url": "http://x/y",
                      "writeable_path": pfx}, None

    monkeypatch.setattr(vapi, "request_soft", _req)
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda *a, **k: _FakeResp(bodies[0]))
    ok, text, err = vremote._vast_execute_soft("DEAD", "ls -lR /w", tries=2,
                                               _sleep=lambda s: None)
    assert ok is True and err is None
    assert text.splitlines()[0] == "w:"     # prefix stripped, as before
    assert "ours.bin" in text


def test_execute_refuses_a_crossed_listing_rather_than_returning_it(monkeypatch):
    """THE regression. A body that is entirely another caller's output must
    never come back as `ok=True` — that is the read that got published as
    truth. It is reported as a (non-fatal, retryable) failure instead."""
    _capture_put(monkeypatch)
    other = "\n".join(["/somebody/elses/tree:",
                       "-rw-r--r-- 1 root root 9 Jan 1 00:00 THEIRS"])
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda *a, **k: _FakeResp(other))
    ok, text, err = vremote._vast_execute_soft("DEAD", "ls -lR /w", tries=3,
                                               _sleep=lambda s: None)
    assert ok is False
    assert text == ""
    assert "nonce" in err
    # and it must stay RETRYABLE — a correlation failure is not "the disk is
    # gone", and classifying it as fatal would publish an authoritative false
    # negative, which is the exact failure this module exists to avoid
    assert S.survey_is_fatal(err) is False


def test_execute_refuses_a_truncated_block_still_being_written(monkeypatch):
    """BEGIN present, END not: the host is mid-write. A partial `ls -lR` cannot
    be the oracle a byte-for-byte verification is checked against."""
    bodies = []

    def _req(method, path, body=None, **k):
        n = _nonce_of((body or {}).get("command"))
        bodies.append(f"__herdd_exec_{n}_BEGIN__\n/w:\n-rw-r--r-- 1 r r 5 x a")
        return True, {"success": True, "result_url": "http://x/y",
                      "writeable_path": ""}, None

    monkeypatch.setattr(vapi, "request_soft", _req)
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda *a, **k: _FakeResp(bodies[0]))
    ok, _, err = vremote._vast_execute_soft("DEAD", "ls -lR /w", tries=3,
                                            _sleep=lambda s: None)
    assert ok is False and "nonce" in err


def test_execute_markers_survive_a_failing_command(monkeypatch):
    """`;` not `&&`: a legitimately-absent path is a REAL answer this module
    distinguishes from an unreadable disk, so the end marker must still be
    emitted when the command exits non-zero."""
    sent = _capture_put(monkeypatch)
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda *a, **k: _FakeResp(""))
    vremote._vast_execute_soft("DEAD", "ls -lR /gone", tries=1, _sleep=lambda s: None)
    cmd = sent[0][1]
    assert "&&" not in cmd
    assert cmd.count(";") >= 2


def test_execute_does_NOT_fall_back_on_a_404(monkeypatch):
    """A 404 means the host reclaimed the instance — the one authoritative
    answer here. Falling back would fire a second pointless request and blur
    the verdict."""
    n = {"calls": 0}

    def _req(*a, **k):
        n["calls"] += 1
        return False, None, "HTTP 404 not_found"

    monkeypatch.setattr(vapi, "request_soft", _req)
    ok, _, err = vremote._vast_execute_soft("DEAD", "ls -1 /", _sleep=lambda s: None)
    assert ok is False and "404" in err
    assert n["calls"] == 1
    assert S.survey_is_fatal(err) is True


def test_execute_degrades_to_read_twice_when_vast_refuses_the_wrapped_form(
        monkeypatch, capsys):
    """If the endpoint allowlists `ls`/`rm`/`du`, insisting on the nonce would
    take out the whole recovery path. Degrade instead — to a real, weaker check
    (the body must be stable across two reads) — and say so out loud."""
    seen = []

    def _req(method, path, body=None, **k):
        cmd = (body or {}).get("command")
        seen.append(cmd)
        if "herdd_exec" in cmd:
            return False, None, "HTTP 400 invalid_args: unsupported command"
        return True, {"success": True, "result_url": "http://x/y",
                      "writeable_path": ""}, None

    monkeypatch.setattr(vapi, "request_soft", _req)
    bodies = iter(["still writing...", "/w:\nfinal", "/w:\nfinal"])
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda *a, **k: _FakeResp(next(bodies)))
    ok, text, err = vremote._vast_execute_soft("DEAD", "ls -lR /w", tries=5,
                                               _sleep=lambda s: None)
    assert ok is True and err is None
    assert text == "/w:\nfinal"
    assert seen[-1] == "ls -lR /w"          # bare command on the retry
    assert "WEAKER guard" in capsys.readouterr().err


def test_execute_degraded_mode_refuses_an_unstable_body(monkeypatch):
    """Two different reads means something is still writing (or crossing). In
    degraded mode that is the only signal available, so it must gate."""
    def _req(method, path, body=None, **k):
        if "herdd_exec" in (body or {}).get("command", ""):
            return False, None, "HTTP 400 invalid_args"
        return True, {"success": True, "result_url": "http://x/y",
                      "writeable_path": ""}, None

    monkeypatch.setattr(vapi, "request_soft", _req)
    n = {"i": 0}

    def _open(*a, **k):
        n["i"] += 1
        return _FakeResp(f"body-{n['i']}")

    monkeypatch.setattr(urllib.request, "urlopen", _open)
    ok, _, err = vremote._vast_execute_soft("DEAD", "ls -lR /w", tries=4,
                                            _sleep=lambda s: None)
    assert ok is False and "stable" in err
    assert S.survey_is_fatal(err) is False

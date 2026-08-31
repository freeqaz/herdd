"""`vastlib.boxes.salvage` — the absorbed module, its wire schema, and its seam.

Why this file exists
--------------------
`salvage.py` is absorbed whole here, and the ten transport-glue functions that
used to live 8,000 lines away in `herdd.py` come with it. Three classes of
thing therefore need pinning that `test_salvage.py` cannot pin for this copy
(it drives the flat module and `herdd`'s glue, both still live, and is left
UNEDITED):

1. **The wire schema.** The nine `OUTCOME_*` strings land in the fleetd journal
   and in `retained_boxes[].salvage.outcome`; `new_record`'s dict is persisted
   into `state.json` (plan §4 load-compat); `b2_salvage_prefix` writes a live B2
   prefix. Each is pinned key for key.
2. **The five namedtuples, respelled as `typing.NamedTuple`.** Field names,
   order, positional construction and `==` against a bare tuple — the
   `core.models.MarketRead` precedent.
3. **The injection seam, from both sides.** The glue functions are the SUPPLY
   side of the callables `advance()` consumes, and they reach
   `boxes.remote` / `boxes.ssh` / `boxes.lifecycle` / `storage.b2` as MODULE
   ATTRIBUTES. Five existing patch sites ride those edges; a from-import would
   make them vacuous, so each edge is exercised by patching the sibling module
   and asserting the patch was taken.

The load-bearing behavior under all of it is `survey_dest_files`' three-way
split: `None` = could not read, `{}` = read fine and the path is NOT THERE,
`{...}` = a listing. It keys the `{}` case off the `ls` ERROR TEXT precisely
because the transport's `ok` cannot be trusted (see `boxes/remote.py`). Two
tests below hold that split open.

What is deliberately NOT here
-----------------------------
* No network, no ssh, no rclone, no B2. Every transport is either an injected
  fake or a stubbed module attribute. `_salvage_copy_status` is the only GET in
  the module and the one call that WOULD reach the live API unstubbed, so it is
  never exercised without a stub.
* No re-testing of the parsing corpus (`test_salvage.py`'s 97 tests own the
  `ls -l`/`ls -lR` shapes); this file tests the fold points the port touched.
* No repoint of any existing test, and no test of the tick drivers
  (`_job_salvage_*`), which stay in `herdd.py` until plan step 4.

Provenance: created 2026-08-16 alongside `vastlib/boxes/salvage.py`, plan §8
step 3. Revised at plan step 7, when `tools/vast/salvage.py` became a re-export
shim over this module: every `assert S.X == flat.X` in sections 1-2 turned into
a self-comparison the moment the two names became one object, so each was
converted to the literal it was standing in for (§5 explains the swap, and adds
the two gates the shim itself needs — exact surface, and same-object identity).
`flat` is deliberately still imported: post-shim it is the thing under test, not
the reference.
"""

from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest

VAST_DIR = Path(__file__).resolve().parent
if str(VAST_DIR) not in sys.path:                      # entry-script sys.path shape
    sys.path.insert(0, str(VAST_DIR))

import salvage as flat                                 # noqa: E402  the twin

from vastlib.boxes import lifecycle, remote            # noqa: E402
from vastlib.boxes import salvage as S                 # noqa: E402
from vastlib.boxes import ssh                          # noqa: E402
from vastlib.core import api                           # noqa: E402
from vastlib.storage import b2                         # noqa: E402


def _ok(text):
    return lambda iid, cmd: (True, text, None)


def _fail(err="boom"):
    return lambda iid, cmd: (False, "", err)


# --------------------------------------------------------------------------- #
# 1. the wire schema, pinned to LITERALS
#
# These were `assert S.X == flat.X` twin comparisons through plan step 6. At
# step 7 `salvage.py` became a re-export shim over this very module, so every
# such comparison became `X == X` — green, and evidence of nothing. The values
# below are the measured step-6 values spelled out, which is what the twin
# assert was standing in for: this is a WIRE SCHEMA (fleetd's journal,
# `retained_boxes[].salvage.outcome`, state.json, a live B2 prefix), so the
# thing worth pinning was never "the two copies agree", it was "the strings
# never change". `flat` is still imported and still exercised below, where the
# shim's module object — not its values — is the thing under test.
# --------------------------------------------------------------------------- #
_OUTCOME_LITERALS = {
    "OUTCOME_SALVAGED": "salvaged",
    "OUTCOME_PARTIAL": "salvaged_partial",
    "OUTCOME_UNVERIFIABLE": "salvage_unverifiable",
    "OUTCOME_NOTHING_NEWER": "nothing_newer",
    "OUTCOME_NOTHING_FOUND": "nothing_found",
    "OUTCOME_DEAD_GONE": "dead_box_gone",
    "OUTCOME_DEST_NOT_READY": "dest_not_ready",
    "OUTCOME_COPY_REFUSED": "copy_refused",
    "OUTCOME_DISABLED": "salvage_disabled",
}


@pytest.mark.parametrize("name,value", sorted(_OUTCOME_LITERALS.items()))
def test_every_outcome_string_is_unchanged(name, value):
    assert getattr(S, name) == value


def test_the_outcome_vocabulary_is_exactly_these_nine():
    """A tenth outcome is a schema addition, not a refactor. Say so out loud."""
    assert {n for n in dir(S) if n.startswith("OUTCOME_")} == set(_OUTCOME_LITERALS)


def test_outcome_sets_are_unchanged():
    assert S.TERMINAL_OUTCOMES == frozenset(_OUTCOME_LITERALS.values())
    assert S.LOUD_OUTCOMES == frozenset({
        "salvaged_partial", "salvage_unverifiable", "nothing_found",
        "dead_box_gone", "dest_not_ready", "copy_refused",
    })
    assert S.OUTCOME_SALVAGED not in S.LOUD_OUTCOMES        # the quiet two
    assert S.OUTCOME_NOTHING_NEWER not in S.LOUD_OUTCOMES


def test_knobs_are_unchanged():
    for k, v in (("SALVAGE_KEEP_N", 1), ("SALVAGE_MAX_GB", 12.0),
                 ("SALVAGE_DEADLINE_S", 1800.0), ("SALVAGE_DEST_WAIT_S", 900.0),
                 ("SALVAGE_ROOT", "/workspace/salvage"),
                 ("JOBS_ROOT", "/workspace/jobs"),
                 ("DEST_READY_STATES", frozenset({"running"})),
                 ("DEST_FREE_MARGIN", 1.2)):
        assert getattr(S, k) == v, k


def test_new_record_is_key_for_key_the_persisted_shape():
    a = S.new_record(41, now=1000.0, dest_candidates=[42], job_id="j")
    # The key SET, not a count: this dict is persisted into fleetd's state.json
    # and re-loaded by a daemon that may be running the other copy. (The port
    # manifest calls it "15-key"; measured here it is 16 — the count was an
    # inherited claim, the set below is the measurement.)
    assert set(a) == {"dead_iid", "job_id", "dest_candidates", "dest_iid",
                      "phase", "outcome", "items", "started_ts",
                      "dest_deadline_ts", "deadline_ts", "keep_n", "max_gb",
                      "attempts", "detail", "bytes", "b2"}


def test_b2_prefix_never_points_at_the_live_checkpoints_tree():
    p = S.b2_salvage_prefix("j1", 41, "out/checkpoint-50")
    assert p == "jobs/j1/salvage/41/out/checkpoint-50"
    assert "/checkpoints/" not in p


# --------------------------------------------------------------------------- #
# 2. namedtuple -> NamedTuple equivalence
#
# Field names are spelled out rather than compared against `flat` for the same
# step-7 reason as section 1 — but note the IDENTITY assert below, which is the
# one thing the shim made newly checkable and newly load-bearing: the flat suite
# (test_salvage.py, left unedited) constructs these tuples through the bare name
# and passes them into ported functions, and that only stays honest while the
# two spellings are ONE class.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name,fields", [
    ("LsEntry", ("name", "size", "is_dir")),
    ("CkptDir", ("name", "step", "bytes", "files")),
    ("SalvagePlan", ("action", "items", "bytes", "reason")),
    ("Verification", ("status", "missing", "short", "bytes_seen",
                      "bytes_expected", "reason")),
    ("Step", ("phase", "outcome", "plan", "verification", "detail", "commands")),
])
def test_tuple_fields_match_the_flat_module(name, fields):
    assert getattr(S, name)._fields == fields
    # The shim re-exports, so these must be the same object, not a lookalike.
    assert getattr(flat, name) is getattr(S, name)


def test_tuples_construct_positionally_and_compare_as_plain_tuples():
    e = S.LsEntry("a.bin", 12, False)
    assert e == ("a.bin", 12, False)
    assert e.size == 12 and e._replace(size=13).size == 13
    c = S.CkptDir("out/checkpoint-50", 50, 12, {"a.bin": 12})
    assert tuple(c) == ("out/checkpoint-50", 50, 12, {"a.bin": 12})


# --------------------------------------------------------------------------- #
# 3. the three-way destination split (the trap this module turns on)
# --------------------------------------------------------------------------- #
_ABSENT = ("ls: cannot access '/workspace/salvage/41/j/out/checkpoint-50': "
           "No such file or directory\n")


def test_dest_survey_none_means_the_transport_failed():
    assert S.survey_dest_files(42, "/w", execute=_fail()) is None


def test_dest_survey_empty_dict_means_read_fine_but_not_there():
    """`ok=True` with an ls ERROR in the body — the whole reason `ok` is not
    the discriminator here."""
    assert S.survey_dest_files(42, "/w", execute=_ok(_ABSENT)) == {}


def test_dest_survey_returns_the_listing_keyed_relative_to_the_path():
    body = ("/w:\n"
            "total 8\n"
            "-rw-r--r-- 1 root root 12 Aug  5 07:00 a.bin\n"
            "drwxr-xr-x 2 root root  4 Aug  5 07:00 sub\n"
            "\n"
            "/w/sub:\n"
            "-rw-r--r-- 1 root root 34 Aug  5 07:00 b.bin\n")
    assert S.survey_dest_files(42, "/w", execute=_ok(body)) == {"a.bin": 12,
                                                               "sub/b.bin": 34}


def test_an_unparsed_line_poisons_the_oracle_rather_than_shrinking_it():
    assert S.survey_dest_files(42, "/w", execute=_ok("mystery\n")) is None


def test_verify_is_fail_closed_on_an_unreadable_destination():
    assert S.verify_salvage({"a": 1}, None).status == "unverifiable"
    assert S.verify_salvage({"a": 1}, {}).status == "partial"
    assert S.verify_salvage({"a": 1}, {"a": 1}).status == "ok"
    assert S.verify_salvage({"a": 2}, {"a": 1}).status == "partial"    # short


# --------------------------------------------------------------------------- #
# policy
# --------------------------------------------------------------------------- #
_CK = [S.CkptDir("out/checkpoint-100", 100, 10, {"a": 10}),
       S.CkptDir("out/checkpoint-50", 50, 10, {"a": 10})]


def test_plan_copies_the_newest_and_says_why():
    plan = S.plan_salvage(_CK, b2_bytes={}, keep_n=1)
    assert plan.action == "copy" and plan.items == (_CK[0],)
    assert "checkpoint-100" in plan.reason


def test_an_unreadable_b2_makes_us_copy_more_never_less():
    plan = S.plan_salvage(_CK, b2_bytes=None, keep_n=1)
    assert plan.action == "copy"
    assert "UNREADABLE" in plan.reason


def test_b2_already_holding_everything_is_nothing_not_a_copy():
    assert S.plan_salvage(_CK, b2_bytes={"out/checkpoint-100": 10,
                                         "out/checkpoint-50": 10}).action == "nothing"


def test_the_size_fuse_refuses_rather_than_guesses():
    big = [S.CkptDir("out/checkpoint-1", 1, 20_000_000_000, {"a": 20_000_000_000})]
    assert S.plan_salvage(big, b2_bytes={}, max_gb=12.0).action == "refuse"


def test_pick_dest_skips_a_box_that_is_known_not_to_fit():
    st = {"1": "running", "2": "running"}
    assert S.pick_dest(["1", "2"], st, free_gb={"1": 0.5},
                       need_bytes=2_000_000_000) == "2"
    assert S.pick_dest(["1"], st, free_gb={}, need_bytes=2_000_000_000) == "1"
    assert S.pick_dest(["1"], {"1": "loading"}) is None


# --------------------------------------------------------------------------- #
# the state machine, one bounded tick at a time
# --------------------------------------------------------------------------- #
_SURVEY_JOBS = "j1\n"
_SURVEY_TREE = (
    "/workspace/jobs/j1/work/out/checkpoint-50:\n"
    "-rw-r--r-- 1 root root 12 Aug  5 07:00 a.bin\n")


def _dead_box_exec(iid, cmd):
    if cmd.startswith("ls -1 "):
        return True, _SURVEY_JOBS, None
    return True, _SURVEY_TREE, None


def test_a_transient_survey_failure_stays_pending_rather_than_declaring_loss():
    rec = S.new_record(41, now=0.0, dest_candidates=["42"])
    out = S.advance(rec, now=1.0, execute=_fail("network timeout"),
                    copy_direct=lambda *a: (True, "", None),
                    statuses={"42": "running"})
    assert out["phase"] == "pending" and out["outcome"] is None
    assert "network timeout" in out["last_survey_error"]


def test_a_404_is_the_one_authoritative_negative():
    rec = S.new_record(41, now=0.0, dest_candidates=["42"])
    out = S.advance(rec, now=1.0, execute=_fail("HTTP 404 not found"),
                    copy_direct=lambda *a: (True, "", None),
                    statuses={"42": "running"})
    assert out["outcome"] == S.OUTCOME_DEAD_GONE


def test_a_full_tick_pair_copies_then_verifies():
    copies = []
    rec = S.new_record(41, now=0.0, dest_candidates=["42"], job_id="j1")
    rec = S.advance(rec, now=1.0, execute=_dead_box_exec,
                    copy_direct=lambda *a: copies.append(a) or (True, "go", None),
                    statuses={"42": "running"}, b2_bytes=lambda j: {})
    assert rec["phase"] == "copying" and len(copies) == 1
    assert rec["items"][0]["name"] == "out/checkpoint-50"
    landed = rec["items"][0]["landed"]
    rec = S.advance(rec, now=2.0, execute=_dead_box_exec,
                    copy_direct=lambda *a: (True, "", None),
                    statuses={"42": "running"},
                    dest_execute=_ok(f"{landed}:\n"
                                     "-rw-r--r-- 1 root root 12 Aug 5 07:00 a.bin\n"))
    assert rec["outcome"] == S.OUTCOME_SALVAGED
    assert rec["items"][0]["verify"] == "ok"


def test_a_done_record_is_never_advanced_again():
    rec = S.new_record(41, now=0.0)
    S._finish(rec, S.OUTCOME_SALVAGED, "done")
    before = dict(rec)
    out = S.advance(rec, now=9.0, execute=_fail(), copy_direct=lambda *a: None,
                    statuses={})
    assert out == before


# --------------------------------------------------------------------------- #
# the glue — every cross-module edge, exercised as a module attribute
# --------------------------------------------------------------------------- #
def test_dest_exec_uses_ssh_for_a_running_box(monkeypatch):
    seen = []
    monkeypatch.setattr(remote, "_ssh_exec_soft",
                        lambda iid, cmd: seen.append("ssh") or (True, "S", None))
    monkeypatch.setattr(remote, "_vast_execute_soft",
                        lambda iid, cmd: pytest.fail("running boxes refuse execute"))
    assert S._mk_salvage_dest_exec({"42": "running"})(42, "ls")[1] == "S"
    assert seen == ["ssh"]


def test_dest_exec_falls_back_to_ssh_on_vasts_own_refusal(monkeypatch):
    monkeypatch.setattr(remote, "_vast_execute_soft",
                        lambda iid, cmd: (False, "", "400: only avail on stopped"))
    monkeypatch.setattr(remote, "_ssh_exec_soft",
                        lambda iid, cmd: (True, "via-ssh", None))
    assert S._mk_salvage_dest_exec({})(42, "ls") == (True, "via-ssh", None)


def test_dest_exec_uses_execute_for_a_stopped_box(monkeypatch):
    monkeypatch.setattr(remote, "_vast_execute_soft",
                        lambda iid, cmd: (True, "via-execute", None))
    monkeypatch.setattr(remote, "_ssh_exec_soft",
                        lambda iid, cmd: pytest.fail("stopped boxes have no ssh"))
    assert S._mk_salvage_dest_exec({"41": "exited"})(41, "ls")[1] == "via-execute"


def test_prepare_dest_refuses_a_stopped_destination(monkeypatch):
    monkeypatch.setattr(remote, "_ssh_exec_soft",
                        lambda *a, **k: pytest.fail("no ssh to a stopped box"))
    assert S._mk_salvage_prepare_dest({"41": "exited"})(41, ["/a"]) is False
    assert S._mk_salvage_prepare_dest({"41": "running"})(41, []) is False


def test_prepare_dest_says_so_loudly_when_the_mkdir_fails(monkeypatch):
    monkeypatch.setattr(remote, "_ssh_exec_soft",
                        lambda iid, cmd: (False, "", "rc 255"))
    out = io.StringIO()
    with redirect_stdout(out):
        assert S._mk_salvage_prepare_dest({"42": "running"})(42, ["/a b"]) is False
    assert "!! salvage: could not prepare" in out.getvalue()


def test_prepare_dest_quotes_the_parents(monkeypatch):
    sent = []
    monkeypatch.setattr(remote, "_ssh_exec_soft",
                        lambda iid, cmd: sent.append(cmd) or (True, "", None))
    assert S._mk_salvage_prepare_dest({"42": "running"})(42, ["/a b"]) is True
    assert "'/a b'" in sent[0] and sent[0].endswith("echo MKDIR_OK")


def test_copy_status_reads_status_msg_through_the_api_module(monkeypatch):
    monkeypatch.setattr(api, "request_soft",
                        lambda m, p, *a, **k: (True, {"instances": {
                            "status_msg": "rsync: mkdir failed"}}, None))
    assert S._salvage_copy_status(41) == "rsync: mkdir failed"
    monkeypatch.setattr(api, "request_soft", lambda *a, **k: (False, None, "HTTP 404"))
    assert S._salvage_copy_status(41) is None


def test_b2_bytes_groups_by_checkpoint_dir_not_by_first_segment(monkeypatch):
    monkeypatch.setenv("B2_BUCKET", "bkt")
    monkeypatch.setattr(b2, "_rclone_soft",
                        lambda args: (0, "10|out/checkpoint-50/a.bin\n"
                                         "5|out/checkpoint-50/sub/b.bin\n"
                                         "7|out/checkpoint-100/a.bin\n"
                                         "3|junk/x\n", ""))
    assert S._salvage_b2_bytes("j1") == {"out/checkpoint-50": 15,
                                         "out/checkpoint-100": 7}


def test_b2_bytes_is_none_when_the_listing_fails_or_the_bucket_is_unset(monkeypatch):
    monkeypatch.setenv("B2_BUCKET", "bkt")
    monkeypatch.setattr(b2, "_rclone_soft", lambda args: (1, "", "no rclone"))
    assert S._salvage_b2_bytes("j1") is None                # unknown, not empty
    monkeypatch.delenv("B2_BUCKET", raising=False)
    monkeypatch.setattr(b2, "_rclone_soft",
                        lambda args: pytest.fail("no bucket, no call"))
    assert S._salvage_b2_bytes("j1") is None


def test_push_to_b2_needs_a_bucket_and_an_endpoint(monkeypatch):
    monkeypatch.delenv("B2_BUCKET", raising=False)
    ok, detail = S._salvage_push_to_b2(42, "/w", "jobs/j/salvage/41/c")
    assert ok is False and "B2_BUCKET" in detail
    monkeypatch.setenv("B2_BUCKET", "bkt")
    monkeypatch.setattr(lifecycle, "_get_instance",
                        lambda iid: {"actual_status": "loading"})
    monkeypatch.setattr(ssh, "_pick_ssh_endpoint", lambda i, **k: (None, None, None))
    ok, detail = S._salvage_push_to_b2(42, "/w", "p")
    assert ok is False and "loading" in detail


def test_push_to_b2_ships_the_script_base64_and_names_the_salvage_prefix(monkeypatch):
    monkeypatch.setenv("B2_BUCKET", "bkt")
    monkeypatch.setattr(lifecycle, "_get_instance", lambda iid: {"id": iid})
    monkeypatch.setattr(ssh, "_pick_ssh_endpoint", lambda i, **k: ("h", 22, "direct"))
    sent = {}

    class _R:
        returncode = 0
        stdout = "salvage-push: b2x ok"
        stderr = ""

    def _run(argv, **k):
        sent["argv"] = argv
        return _R()

    monkeypatch.setattr(S.subprocess, "run", _run)
    ok, detail = S._salvage_push_to_b2(42, "/w/c", "jobs/j1/salvage/41/out/checkpoint-50")
    assert ok is True and "b2x ok" in detail
    assert "base64 -d > /tmp/salvage_push.sh" in sent["argv"][-1]
    assert "salvage_push.sh" not in sent["argv"][-1].split("|")[0]   # not in the argv head


def test_push_script_targets_the_salvage_prefix_and_never_checkpoints():
    s = S._salvage_push_script("/w/c", "bkt", "jobs/j1/salvage/41/out/checkpoint-50")
    assert "b2:bkt/jobs/j1/salvage/41/out/checkpoint-50" in s
    assert "/checkpoints/" not in s


# --------------------------------------------------------------------------- #
# the tick-snapshot folds
# --------------------------------------------------------------------------- #
def test_salvage_is_on_by_default_and_disarmed_only_explicitly(monkeypatch):
    monkeypatch.delenv("SALVAGE_ENABLED", raising=False)

    class _A:
        salvage = None                                  # fleetd seeds True, not None

    assert S._salvage_enabled({"a": _A()}) is True      # None does NOT disarm
    assert S._salvage_enabled({}) is True
    _A.salvage = False
    assert S._salvage_enabled({"a": _A()}) is False
    _A.salvage = True
    monkeypatch.setenv("SALVAGE_ENABLED", "0")
    assert S._salvage_enabled({"a": _A()}) is False


def test_statuses_and_free_gb_fold_the_instance_snapshot():
    jc = {"instances": [{"id": 41, "actual_status": "RUNNING",
                         "disk_space": 100, "disk_util": 60},
                        {"id": 42, "actual_status": "exited"}]}
    assert S._salvage_statuses(jc) == {"41": "running", "42": "exited"}
    assert S._salvage_free_gb(jc) == {"41": 40.0}       # 42 has no telemetry


def test_dest_candidates_put_the_replacement_first_and_drop_the_dead_box():
    jc = {"instances": [{"id": 41, "actual_status": "running"},
                        {"id": 42, "actual_status": "running"},
                        {"id": 43, "actual_status": "loading"}]}
    assert S._salvage_dest_candidates(jc, 42, 41) == ["42"]
    assert S._salvage_dest_candidates(jc, None, 43) == ["41", "42"]


# --------------------------------------------------------------------------- #
# 5. the step-7 shim surface
#
# tools/vast/salvage.py is now a re-export shim over this module. Two things
# about that are worth a gate rather than a comment:
#
#   * the shim must carry EXACTLY the 51 names the flat file used to define. It
#     is easy to widen it by accident, because `vastlib.boxes.salvage` is a
#     SUPERSET — the ten `_salvage_*` / `_mk_salvage_*` helpers above came from
#     herdd.py and were never on the flat module. Re-exporting one of those
#     would let a caller reach a herdd-provenance name through the salvage
#     name, which no consumer has ever been able to do.
#   * every re-exported name must be the same OBJECT, not an equal copy. The
#     flat owner suite (test_salvage.py, 97 tests, unedited) drives the bare
#     name; a forked copy would leave it green while testing nothing.
# --------------------------------------------------------------------------- #
_SHIM_SURFACE = frozenset({
    "CkptDir", "DEST_FREE_MARGIN", "DEST_READY_STATES", "JOBS_ROOT",
    "LOUD_OUTCOMES", "LsEntry", "OUTCOME_COPY_REFUSED", "OUTCOME_DEAD_GONE",
    "OUTCOME_DEST_NOT_READY", "OUTCOME_DISABLED", "OUTCOME_NOTHING_FOUND",
    "OUTCOME_NOTHING_NEWER", "OUTCOME_PARTIAL", "OUTCOME_SALVAGED",
    "OUTCOME_UNVERIFIABLE", "SALVAGE_DEADLINE_S", "SALVAGE_DEST_WAIT_S",
    "SALVAGE_KEEP_N", "SALVAGE_MAX_GB", "SALVAGE_ROOT", "SalvagePlan", "Step",
    "TERMINAL_OUTCOMES", "Verification", "_CKPT", "_FATAL_SURVEY", "_LS_ABSENT",
    "_LS_LINE", "_advance_copying", "_advance_pending", "_copy_status_soft",
    "_finish", "_strip_ls_date", "advance", "b2_salvage_prefix",
    "checkpoint_step", "ckpt_dirs_from_survey", "dest_path", "landed_path",
    "new_record", "parse_ls_l", "parse_ls_l_strict", "parse_ls_lr",
    "parse_ls_lr_strict", "pick_dest", "plan_salvage", "split_ckpt_rel",
    "survey_dead_box", "survey_dest_files", "survey_is_fatal", "verify_salvage",
})


def test_the_shim_exports_exactly_the_fifty_one_flat_names():
    assert len(_SHIM_SURFACE) == 51
    assert set(flat.__all__) == _SHIM_SURFACE
    # and it does not leak the herdd-provenance ten
    assert not _SHIM_SURFACE & {"_mk_salvage_dest_exec", "_mk_salvage_prepare_dest",
                                "_salvage_b2_bytes", "_salvage_copy_status",
                                "_salvage_dest_candidates", "_salvage_enabled",
                                "_salvage_free_gb", "_salvage_push_script",
                                "_salvage_push_to_b2", "_salvage_statuses"}


@pytest.mark.parametrize("name", sorted(_SHIM_SURFACE))
def test_every_shim_name_is_the_same_object_as_the_port(name):
    assert getattr(flat, name) is getattr(S, name)


def test_the_shim_defines_no_body_of_its_own():
    """A shim that redefines anything forks the implementation it exists to
    prevent forking. Assert it at the AST, not by reading it."""
    import ast
    src = (VAST_DIR / "salvage.py").read_text()
    tree = ast.parse(src)
    defs = [n for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    assert defs == [], [n.name for n in defs]
    assigned = [t.id for n in tree.body if isinstance(n, ast.Assign)
                for t in n.targets if isinstance(t, ast.Name)]
    assert assigned == ["__all__"], assigned

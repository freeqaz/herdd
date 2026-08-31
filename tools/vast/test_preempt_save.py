"""Tests for the preempt-forced checkpoint: onstart/preempt_save.py + the
`preempt-local-save` block in onstart/preempt_trap.sh.

The property under test throughout is NOT "a checkpoint appears" — it is that a
checkpoint can never be marked COMPLETE unless every rank finished writing it.
Salvage now faithfully rescues whatever is on an evicted disk, so a torn
checkpoint carrying a green flag would be copied, byte-verified against its own
torn self, pushed to B2, and resumed from. Every "does not mark" assertion below
is guarding that path.

No torch, no transformers, no network: the callback takes an injected base class
and an injected all-reduce, so the whole state machine runs in the portable lane.
"""
import json
import os
import re
import stat
import shutil
import subprocess

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
TRAP = os.path.join(_HERE, "onstart", "preempt_trap.sh")

import sys
sys.path.insert(0, os.path.join(_HERE, "onstart"))
import preempt_save as ps                                    # noqa: E402


class _Base:
    """Stand-in for transformers.TrainerCallback (it defines no behaviour we use)."""


class _Ctl:
    def __init__(self):
        self.should_save = False


class _State:
    def __init__(self, step):
        self.global_step = step


class _Args:
    def __init__(self, out):
        self.output_dir = out


# --- the collective agreement ----------------------------------------------- #

def test_agree_is_collective_not_local():
    """A signal delivered to ONE rank must make EVERY rank save.

    This is the whole reason `agree()` exists. If rank 1 acted on its local flag
    and rank 0 did not, `_save_checkpoint`'s collectives would deadlock or one
    rank would write a directory alone — the torn checkpoint, by construction.
    """
    signalled, quiet = ps.SaveRequest(), ps.SaveRequest()
    signalled.request()
    fleet = [signalled, quiet]

    def all_reduce_max(v):                 # the real MAX all-reduce, in-process
        return max([1 if r.requested else 0 for r in fleet] + [v])

    assert signalled.agree(all_reduce_max) is True
    assert quiet.agree(all_reduce_max) is True, \
        "an unsignalled rank must still agree to save — otherwise ranks diverge"


def test_agree_false_when_nobody_signalled():
    a, b = ps.SaveRequest(), ps.SaveRequest()
    fleet = [a, b]
    red = lambda v: max([1 if r.requested else 0 for r in fleet] + [v])  # noqa: E731
    assert a.agree(red) is False and b.agree(red) is False


def test_agree_is_one_shot():
    """One signal must not set should_save on every remaining step of the run."""
    r = ps.SaveRequest()
    r.request()
    assert r.agree(None) is True
    r.fired = True
    assert r.agree(None) is False


def test_agree_falls_back_to_local_when_the_collective_dies():
    """A preemption kills peers, so the all-reduce itself can fail. That must not
    raise out of on_step_end — the fallback saves alone and simply never earns
    the completion marker."""
    r = ps.SaveRequest()
    r.request()

    def exploding(_v):
        raise RuntimeError("NCCL peer gone")

    assert r.agree(exploding) is True


def test_agree_token_is_identical_across_ranks():
    """Rank markers are only comparable if every rank writes the SAME token, so
    the token is agreed by the same MAX-reduce as the save decision."""
    fleet = [ps.SaveRequest(nonce=n) for n in (7, 99, 12)]

    def red(v):
        return max([r.nonce for r in fleet] + [v])

    tokens = {r.agree_token(red) for r in fleet}
    assert tokens == {99}, "all ranks must land on one token"


def test_agree_token_falls_back_to_the_local_nonce():
    """No group, or a dead collective: ranks may disagree, which can only make
    `finalize` see too FEW matching markers — it withholds the flag."""
    r = ps.SaveRequest(nonce=5)
    assert r.agree_token(None) == 5

    def boom(_v):
        raise RuntimeError("peer gone")
    assert r.agree_token(boom) == 5


def test_signal_handler_only_sets_a_flag():
    """Async-signal-safety: `request` must do one store and nothing else."""
    r = ps.SaveRequest()
    assert r.requested is False
    r.request(ps.SAVE_SIGNAL, None)        # called with a handler's signature
    assert r.requested is True


# --- completion marking ------------------------------------------------------ #

def test_finalize_waits_for_every_rank(tmp_path):
    ck = tmp_path / "checkpoint-50"
    ck.mkdir()
    ps.mark_rank_done(str(ck), 0, "tok")
    # rank 1 never reports
    assert ps.finalize(str(ck), 2, step=50, token="tok", wait_s=0.05,
                       sleep=lambda _s: None, now=_clock()) is False
    assert not ps.is_complete(str(ck)), \
        "a checkpoint missing a rank must NEVER be marked complete"


def test_finalize_marks_when_all_ranks_report(tmp_path):
    ck = tmp_path / "checkpoint-50"
    ck.mkdir()
    (ck / "w.bin").write_bytes(b"x" * 9)
    ps.mark_rank_done(str(ck), 0, "tok")
    ps.mark_rank_done(str(ck), 1, "tok")
    assert ps.finalize(str(ck), 2, step=50, token="tok") is True
    assert ps.is_complete(str(ck))
    body = json.load(open(os.path.join(str(ck), ps.COMPLETE_MARKER)))
    assert body["step"] == 50 and body["world_size"] == 2


def test_finalize_leaves_no_partial_marker_behind(tmp_path):
    """The marker is written-then-renamed, so a reader can never see it half
    written — the one file whose partial read would be a completeness claim."""
    ck = tmp_path / "checkpoint-50"
    ck.mkdir()
    ps.mark_rank_done(str(ck), 0, "tok")
    ps.finalize(str(ck), 1, step=50, token="tok")
    assert ps.COMPLETE_MARKER + ".partial" not in os.listdir(str(ck))


def test_is_complete_false_on_a_missing_dir(tmp_path):
    assert ps.is_complete(str(tmp_path / "nope")) is False


def _clock():
    t = {"v": 0.0}

    def now():
        t["v"] += 0.03
        return t["v"]
    return now


# --- the callback end to end ------------------------------------------------- #

def _fleet(tmp_path, world_size, signalled_rank=0, wait_s=3.0):
    """Build `world_size` ranks sharing one output_dir, as DDP would.

    `wait_s` is injected so a test of the "a rank never reports" path costs its
    own timeout rather than the production default — that test burned 20 s of
    real wall clock before `make_callback` exposed this.
    """
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    # One shared nonce models a SUCCESSFUL token reduce (every rank ends up with
    # the same value). `test_agree_token_*` below exercises the reduce itself.
    reqs = [ps.SaveRequest(nonce=4242) for _ in range(world_size)]
    reqs[signalled_rank].request()

    def red(v):
        # A real MAX all-reduce over whatever each rank proposed at this call
        # site: the requested flags for `agree`, the (shared) nonce for
        # `agree_token`. Taking the max against `v` covers both.
        return max([1 if r.requested else 0 for r in reqs] + [v])

    cbs = [ps.make_callback(reqs[i], rank=i, world_size=world_size,
                            all_reduce_max=red, base=_Base, log=lambda *_a: None,
                            wait_s=wait_s)
           for i in range(world_size)]
    return out, reqs, cbs


def test_ddp_two_ranks_produce_one_complete_checkpoint(tmp_path):
    """Ranks run CONCURRENTLY in real DDP, so rank 0 polls while rank 1 writes.
    Threads model that; running them serially with rank 0 first would deadlock
    rank 0 against a marker that cannot appear until after it returns."""
    import threading
    out, reqs, cbs = _fleet(tmp_path, 2, signalled_rank=1)
    args, state = _Args(str(out)), _State(50)

    ctls = [_Ctl() for _ in cbs]
    for cb, ctl in zip(cbs, ctls):
        cb.on_step_end(args, state, ctl)
    assert all(c.should_save for c in ctls), "every rank must decide to save"

    ck = out / "checkpoint-50"
    ck.mkdir()
    ts = [threading.Thread(target=cb.on_save, args=(args, state, ctl))
          for cb, ctl in zip(cbs, ctls)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=60)
    assert ps.is_complete(str(ck))


def test_a_rank_that_dies_before_on_save_blocks_the_marker(tmp_path):
    """THE torn-checkpoint case: rank 1 is killed mid-write, so its bytes are
    incomplete. Rank 0 finished and would happily write trainer_state.json — the
    marker must still be withheld."""
    out, reqs, cbs = _fleet(tmp_path, 2)
    args, state = _Args(str(out)), _State(50)
    for cb in cbs:
        cb.on_step_end(args, state, _Ctl())
    ck = out / "checkpoint-50"
    ck.mkdir()
    (ck / "trainer_state.json").write_text("{}")   # rank 0 got this far
    cbs[0].on_save(args, state, _Ctl())            # rank 1 never runs
    assert not ps.is_complete(str(ck)), \
        "trainer_state.json alone must not be read as all-rank completeness"


def test_routine_save_steps_checkpoints_are_not_marked(tmp_path):
    """on_save fires for every ordinary SAVE_STEPS checkpoint too. Marking those
    would make the flag meaningless and hide the one thing it certifies."""
    out, reqs, cbs = _fleet(tmp_path, 1)
    reqs[0].requested = False                      # nobody signalled
    args = _Args(str(out))
    ck = out / "checkpoint-20"
    ck.mkdir()
    cbs[0].on_save(args, _State(20), _Ctl())
    assert not ps.is_complete(str(ck))


def test_marker_lands_on_the_agreed_step_only(tmp_path):
    """A save at a DIFFERENT step than the one agreed is a routine checkpoint
    that merely raced us; it must not inherit the completion claim."""
    out, reqs, cbs = _fleet(tmp_path, 1)
    args = _Args(str(out))
    cbs[0].on_step_end(args, _State(50), _Ctl())
    other = out / "checkpoint-60"
    other.mkdir()
    cbs[0].on_save(args, _State(60), _Ctl())
    assert not ps.is_complete(str(other))


def test_callback_fires_only_once(tmp_path):
    out, reqs, cbs = _fleet(tmp_path, 1)
    args, cb = _Args(str(out)), cbs[0]
    first = _Ctl()
    cb.on_step_end(args, _State(50), first)
    second = _Ctl()
    cb.on_step_end(args, _State(51), second)
    assert first.should_save is True and second.should_save is False


def test_on_save_survives_an_unwritable_checkpoint_dir(tmp_path):
    """A full disk on the death path must not raise out of the trainer."""
    out, reqs, cbs = _fleet(tmp_path, 1)
    args = _Args(str(out))
    cbs[0].on_step_end(args, _State(50), _Ctl())
    cbs[0].on_save(args, _State(50), _Ctl())        # checkpoint-50 never created
    assert not ps.is_complete(str(out / "checkpoint-50"))


# --- the two FALSE-COMPLETE paths adversarial review found (2026-08-05) ------ #

def test_a_previous_attempts_rank_marker_does_not_count(tmp_path):
    """CRITICAL 1. transformers never cleans a checkpoint dir, so an abandoned
    attempt's `.preempt_rank_1_ok` survives into a re-entered `checkpoint-<N>`.
    Counting markers by FILENAME let rank 0 see "2/2 reported" when the second
    report came from a run that died hours earlier — a green flag over a
    one-rank checkpoint. The token is what makes a marker belong to an attempt.
    """
    ck = tmp_path / "checkpoint-100"
    ck.mkdir()
    ps.mark_rank_done(str(ck), 1, "OLD-ATTEMPT")      # residue from run A
    ps.mark_rank_done(str(ck), 0, "THIS-ATTEMPT")
    assert ps.ranks_done(str(ck), 2, "THIS-ATTEMPT") == {0}
    assert ps.finalize(str(ck), 2, step=100, token="THIS-ATTEMPT", wait_s=0.05,
                       sleep=lambda _s: None, now=_clock()) is False
    assert not ps.is_complete(str(ck))


def test_a_stale_complete_marker_does_not_certify_retorn_bytes(tmp_path):
    """CRITICAL 1(B)/(C). A dir that was legitimately complete, then re-entered
    and torn — or had a file renamed out of it by
    `_disable_checkpoint_optimizer_state` — must stop reading as complete. The
    inventory is what makes the claim falsifiable."""
    ck = tmp_path / "checkpoint-100"
    ck.mkdir()
    (ck / "optimizer.pt").write_bytes(b"o" * 64)
    (ck / "adapter_model.safetensors").write_bytes(b"a" * 32)
    ps.mark_rank_done(str(ck), 0, "t")
    assert ps.finalize(str(ck), 1, step=100, token="t") is True
    assert ps.is_complete(str(ck))

    (ck / "optimizer.pt").unlink()                    # the bnb-resume rename
    assert not ps.is_complete(str(ck)), \
        "a marker must not outlive the bytes it certified"


def test_a_truncated_file_falsifies_the_marker(tmp_path):
    ck = tmp_path / "checkpoint-100"
    ck.mkdir()
    (ck / "w.bin").write_bytes(b"x" * 100)
    ps.mark_rank_done(str(ck), 0, "t")
    ps.finalize(str(ck), 1, step=100, token="t")
    (ck / "w.bin").write_bytes(b"x" * 10)             # re-torn
    assert not ps.is_complete(str(ck))


def test_a_marker_with_no_inventory_is_not_a_claim(tmp_path):
    """Forward-compat: a marker written by the pre-review version carried no
    inventory. It must read as UNPROVEN, never as complete."""
    ck = tmp_path / "checkpoint-100"
    ck.mkdir()
    (ck / ps.COMPLETE_MARKER).write_text(json.dumps({"step": 100,
                                                     "world_size": 1}))
    assert not ps.is_complete(str(ck))


def test_rank0_clears_an_earlier_attempts_markers(tmp_path):
    out, reqs, cbs = _fleet(tmp_path, 1)
    ck = out / "checkpoint-50"
    ck.mkdir()
    (ck / ps.COMPLETE_MARKER).write_text("{}")        # stale green flag
    (ck / ps.RANK_MARKER.format(7)).write_text("old")
    (ck / "w.bin").write_bytes(b"x")
    args = _Args(str(out))
    cbs[0].on_step_end(args, _State(50), _Ctl())
    cbs[0].on_save(args, _State(50), _Ctl())
    assert ps.RANK_MARKER.format(7) not in os.listdir(str(ck))
    assert ps.is_complete(str(ck))                    # re-certified honestly


def test_the_trap_flush_excludes_the_markers(tmp_path):
    """CRITICAL 2. b2x orders NEWEST FIRST and the marker is by construction the
    newest file, so an un-excluded deadline flush would publish the completeness
    FLAG to B2 and truncate before the 646 MB optimizer — a green flag over
    weights that are not there."""
    src = open(TRAP).read()
    flush = src[src.index('"$B2X" push'):]
    assert "--exclude '.preempt_*'" in flush.split("fi")[0]
    assert flush.split("fi")[0].count("--exclude '.preempt_*'") >= 2, \
        "both the b2x and the rclone fallback must exclude the markers"


def test_all_ranks_armed_is_true_for_a_single_process():
    assert ps.all_ranks_armed(1) is True


def test_all_ranks_armed_false_when_a_peer_never_armed():
    """Asymmetric arming would hang a HEALTHY run at step 1 on a collective the
    unarmed ranks never join. Unless everyone armed, nobody installs."""
    assert ps.all_ranks_armed(2, all_reduce_min=lambda _v: 0) is False
    assert ps.all_ranks_armed(2, all_reduce_min=lambda v: v) is True


def test_all_ranks_armed_false_when_the_probe_raises():
    def boom(_v):
        raise RuntimeError("no group")
    assert ps.all_ranks_armed(2, all_reduce_min=boom) is False


# --- pid advertisement ------------------------------------------------------- #

def test_write_pid_round_trips(tmp_path):
    p = ps.write_pid(3, pid=4242, directory=str(tmp_path / "pids"))
    assert open(p).read() == "4242"
    assert os.path.basename(p) == "3.pid"


def test_write_pid_never_raises_on_a_bad_dir(tmp_path):
    bad = tmp_path / "file"
    bad.write_text("not a dir")
    assert ps.write_pid(0, directory=str(bad / "under-a-file")) is None


# --- the shell block --------------------------------------------------------- #

def _save_block():
    src = open(TRAP).read()
    m = re.search(r"# BEGIN preempt-local-save\n(.*)# END preempt-local-save",
                  src, re.S)
    assert m, "preempt-local-save markers missing from preempt_trap.sh"
    return m.group(1)


def test_trap_calls_the_local_save_before_the_b2_flush():
    """Ordering IS the feature: a local save is seconds, a ~1 GB B2 flush is not."""
    src = open(TRAP).read()
    call = src.index("_preempt_local_save || true")
    flush = src.index('"$B2X" push')
    assert call < flush, "the local save must be asked for BEFORE the B2 flush"


def _run_block(tmp_path, script, env=None):
    driver = tmp_path / "driver.sh"
    driver.write_text("set -uo pipefail\n" + _save_block() + "\n" + script)
    e = dict(os.environ)
    e.update(env or {})
    r = subprocess.run(["bash", str(driver)], capture_output=True, text=True,
                       env=e, timeout=120)
    return r.returncode, r.stdout + r.stderr


# --- the guard must FIRE, not shrug --------------------------------------- #
#
# THE defect this file previously enshrined. `preempt_save.py` was missing from
# every shipping manifest for its whole life, so `_preempt_local_save` hit the
# no-piddir branch on every box, every time — printed one `..` line into an
# onstart log nobody reads, and returned 0. The old assertions here (`"no pid
# dir" in out`, and for the disabled case `"preempt-save:" not in out`) treated
# that silence as CORRECT, which is why the feature could be dead for months
# with a green test suite.
#
# A skip is now an OUTCOME that reports: loud on stdout, and — where the lane
# supplied `_preempt_save_emit` — a B2 event, so its ABSENCE is detectable
# off-box. These tests assert the reporting, not just the rc.

def _emit_probe(sink):
    """A lane emitter stub, shaped like train.sh's / jobd.sh's real ones."""
    return f'_preempt_save_emit() {{ echo "EMIT:$1" >> "{sink}"; }}\n'


def test_no_piddir_is_loud_and_emits(tmp_path):
    """The exact branch that hid the missing module. It must SHOUT and EMIT."""
    ck = tmp_path / "ck"
    ck.mkdir()
    sink = tmp_path / "emits"
    rc, out = _run_block(
        tmp_path,
        _emit_probe(sink) + "_preempt_local_save; echo rc=$?",
        {"CKPT_DIR": str(ck), "PREEMPT_SAVE_PIDDIR": str(tmp_path / "absent")})
    assert "rc=0" in out, "the death path must still never fail"
    assert "!! preempt-save: no_piddir" in out, (
        "a skipped safety net must be loud (!!), not a '..' aside")
    assert sink.read_text().strip() == "EMIT:no_piddir", (
        "the skip must emit an event — otherwise a dead guard is invisible on B2")
    # the message has to be a LEAD, naming what to check
    assert "ModuleNotFoundError" in out and "preempt_save.py" in out


def test_skips_when_disabled_still_report(tmp_path):
    """Deliberate opt-out, but still answers 'why did nothing happen?'."""
    sink = tmp_path / "emits"
    rc, out = _run_block(
        tmp_path, _emit_probe(sink) + "_preempt_local_save; echo rc=$?",
        {"CKPT_DIR": str(tmp_path), "PREEMPT_SAVE_ENABLED": "0"})
    assert "rc=0" in out and "preempt-save: disabled" in out
    assert sink.read_text().strip() == "EMIT:disabled"


def test_skips_when_ckpt_dir_unset(tmp_path):
    """`set -u` is on in the caller; an unset CKPT_DIR must not explode."""
    sink = tmp_path / "emits"
    rc, out = _run_block(tmp_path,
                         _emit_probe(sink) + "_preempt_local_save; echo rc=$?", {})
    assert "rc=0" in out and "!! preempt-save: no_ckptdir" in out
    assert sink.read_text().strip() == "EMIT:no_ckptdir"


def test_reporting_never_breaks_the_death_path_without_an_emitter(tmp_path):
    """No lane emitter defined (standalone source) => still loud, still rc 0."""
    ck = tmp_path / "ck"
    ck.mkdir()
    rc, out = _run_block(tmp_path, "_preempt_local_save; echo rc=$?",
                         {"CKPT_DIR": str(ck),
                          "PREEMPT_SAVE_PIDDIR": str(tmp_path / "absent")})
    assert "rc=0" in out and "!! preempt-save: no_piddir" in out


def test_every_exit_path_reports(tmp_path):
    """No silent return may be re-introduced: every `return 0` in the block is
    preceded by a report (or is the report's own)."""
    block = _save_block()
    body = block[block.index("_preempt_local_save() {"):]
    lines = [ln.strip() for ln in body.splitlines()]
    for i, ln in enumerate(lines):
        if ln != "return 0" and not ln.endswith("; return 0; }"):
            continue
        window = " ".join(lines[max(0, i - 6):i + 1])
        assert "_preempt_save_report" in window, (
            f"silent exit re-introduced at: {ln!r} — every exit from "
            f"_preempt_local_save must report its outcome")


def test_shell_signals_the_advertised_pid_and_sees_the_new_marker(tmp_path):
    """End to end: a fake 'trainer' that writes the marker on SIGUSR1."""
    ck = tmp_path / "ck"
    (ck / "checkpoint-50").mkdir(parents=True)
    piddir = tmp_path / "pids"
    piddir.mkdir()
    script = f"""
victim() {{
  trap 'touch "{ck}/checkpoint-50/{ps.COMPLETE_MARKER}"; exit 0' USR1
  for _ in $(seq 1 60); do sleep 0.2; done
}}
victim & echo $! > "{piddir}/0.pid"
_preempt_local_save; echo rc=$?
wait 2>/dev/null || true
"""
    rc, out = _run_block(tmp_path, script,
                         {"CKPT_DIR": str(ck), "PREEMPT_SAVE_PIDDIR": str(piddir),
                          "PREEMPT_SAVE_WAIT_S": "15"})
    assert "rc=0" in out, out
    assert "preempt-save: complete" in out, out


def test_shell_does_not_accept_a_STALE_marker_as_this_saves_completion(tmp_path):
    """A marker left by an EARLIER preemption in the same run must not satisfy
    the wait — otherwise the trap reports success and flushes stale bytes."""
    ck = tmp_path / "ck"
    (ck / "checkpoint-10").mkdir(parents=True)
    (ck / "checkpoint-10" / ps.COMPLETE_MARKER).write_text("{}")   # stale
    piddir = tmp_path / "pids"
    piddir.mkdir()
    script = f"""
victim() {{ trap 'exit 0' USR1; for _ in $(seq 1 60); do sleep 0.2; done; }}
victim & echo $! > "{piddir}/0.pid"
_preempt_local_save; echo rc=$?
"""
    rc, out = _run_block(tmp_path, script,
                         {"CKPT_DIR": str(ck), "PREEMPT_SAVE_PIDDIR": str(piddir),
                          "PREEMPT_SAVE_WAIT_S": "3"})
    assert "rc=0" in out
    assert "!! preempt-save: timeout" in out, out


def test_shell_does_not_read_a_DELETED_marker_as_success(tmp_path):
    """`save_total_limit` rotation can delete a checkpoint carrying a marker from
    a previous process's forced save. That changes the marker LIST without any
    new save happening — requiring an ADDED path is what keeps it from reading as
    success and flushing stale bytes under a green flag."""
    ck = tmp_path / "ck"
    old = ck / "checkpoint-10"
    old.mkdir(parents=True)
    (old / ps.COMPLETE_MARKER).write_text("{}")
    piddir = tmp_path / "pids"
    piddir.mkdir()
    script = f"""
victim() {{ trap 'rm -rf "{old}"; exit 0' USR1; for _ in $(seq 1 60); do sleep 0.2; done; }}
victim & echo $! > "{piddir}/0.pid"
_preempt_local_save; echo rc=$?
"""
    rc, out = _run_block(tmp_path, script,
                         {"CKPT_DIR": str(ck), "PREEMPT_SAVE_PIDDIR": str(piddir),
                          "PREEMPT_SAVE_WAIT_S": "4"})
    assert "rc=0" in out
    assert "!! preempt-save: timeout" in out, out


def test_shell_ignores_a_dead_pid(tmp_path):
    ck = tmp_path / "ck"
    ck.mkdir()
    piddir = tmp_path / "pids"
    piddir.mkdir()
    # a pid that has certainly exited: spawn true(), reap it, reuse the number
    script = f"""
(exit 0) & dead=$!; wait $dead 2>/dev/null || true
echo $dead > "{piddir}/0.pid"
_preempt_local_save; echo rc=$?
"""
    rc, out = _run_block(tmp_path, script,
                         {"CKPT_DIR": str(ck), "PREEMPT_SAVE_PIDDIR": str(piddir)})
    assert "rc=0" in out and "!! preempt-save: no_live_pid" in out


def test_shell_ignores_a_garbage_pid_file(tmp_path):
    ck = tmp_path / "ck"
    ck.mkdir()
    piddir = tmp_path / "pids"
    piddir.mkdir()
    (piddir / "0.pid").write_text("not-a-pid")
    rc, out = _run_block(tmp_path, "_preempt_local_save; echo rc=$?",
                         {"CKPT_DIR": str(ck), "PREEMPT_SAVE_PIDDIR": str(piddir)})
    assert "rc=0" in out and "!! preempt-save: no_live_pid" in out


def test_shell_never_pattern_matches_its_own_argv():
    """CLAUDE.md's standing rule: a `pgrep -f`/`pkill -f` here would match the
    trap's own command line. Signalling must be by explicit pid only."""
    code = "\n".join(ln for ln in _save_block().splitlines()
                     if not ln.lstrip().startswith("#"))
    assert "pgrep" not in code and "pkill" not in code
    assert "kill -USR1 \"$pid\"" in code


def test_shell_wait_is_bounded():
    block = _save_block()
    assert 'i" -lt "$wait_s' in block, "the wait loop must have an upper bound"


# --- DELIVERY: the module must actually reach a box -------------------------- #
#
# THE root cause, 2026-08-06. Every test above this point passed for the whole
# life of the feature while it had never once run on a box, because they all test
# the module in-tree and nothing tested that the module SHIPS. `preempt_save.py`
# was absent from all three delivery paths simultaneously:
#
#   * tools/vast/eval-env/bake.sh's companion upload list (run lane, B2)
#   * herdd's `_job_attach_files()` jobd bundle          (jobs lane, B2)
#   * tools/vast/onstart/train.sh's companion pull         (run lane, box-side)
#
# and it is in NO image either — tools/vast/train-env/Dockerfile COPYs exactly
# one file (b2x) — so the trainer's old "the module ships in the box image"
# comment was simply false. Result: `ModuleNotFoundError: No module named
# 'preempt_save'` on every box in every lane, then a silent shell skip.
#
# These tests are the tripwire. They read the REAL manifests, so dropping the
# module from any lane fails the portable suite instead of dying quietly in an
# onstart log.

_VAST = _HERE


def test_train_sh_pulls_preempt_save_before_sourcing_the_trap():
    """Run lane, box side: the module must be on disk before the trainer starts."""
    src = open(os.path.join(_VAST, "onstart", "train.sh")).read()
    pull = src.index("eval-env/preempt_save.py")
    trap = src.index("eval-env/preempt_trap.sh")
    assert pull < trap, "stage the module before sourcing the trap that needs it"


def test_preempt_save_ships_in_the_jobd_bundle():
    """Jobs lane (v7..v11 — where production training actually runs).

    `_job_attach_files` moved to `vastlib/jobs/bundle.py` (plan §7.2 ruling C4 —
    §4 names bundle.py the single source of truth for it). The vastlib copy is
    REQUIRED; the flat `herdd.py` copy was scanned too for as long as it
    defined the function, so the property held in both while both existed.
    Plan §8 step 6d emptied the launcher, so the `re.search` guard below now
    drops it from `homes` — the degrade this test was written for, taken. The
    guard stays as a tripwire: a `def _job_attach_files` reappearing in the
    launcher is scanned again rather than silently trusted.

    The definition-matching regex takes the return annotation the port added and
    still needs a FOLLOWING `def` to bound the body — which
    `test_vastlib_jobs_bundle.py::test_attach_files_is_not_the_last_def_in_the_
    module` pins from the other side.
    """
    homes = [os.path.join(_VAST, "vastlib", "jobs", "bundle.py")]
    flat = os.path.join(_VAST, "herdd.py")
    if re.search(r"^def _job_attach_files\(\)", open(flat).read(), re.M):
        homes.append(flat)
    for home in homes:
        src = open(home).read()
        m = re.search(r"def _job_attach_files\(\)[^:]*:(.*?)\ndef ", src, re.S)
        assert m, f"_job_attach_files moved out of {home} — re-point this test"
        body = m.group(1)
        for f in ("preempt_save.py", "preempt_trap.sh"):
            assert f'"{f}"' in body, (
                f"{f} dropped from the jobd bundle ({os.path.basename(home)}): "
                f"the jobs lane loses the preempt-forced checkpoint entirely")


def test_jobd_exports_the_dir_it_stages_the_module_into():
    """The other half of the contract above."""
    src = open(os.path.join(_VAST, "onstart", "jobd.sh")).read()
    assert "export PREEMPT_SAVE_DIR=" in src
    assert 'PREEMPT_TRAP_NO_INSTALL=1 . "$JOBD_DIR/preempt_trap.sh"' in src, (
        "jobd must source the trap for the primitive ONLY — installing it would "
        "replace jobd's own TERM/INT handler")


def test_sourcing_the_trap_with_no_install_arms_nothing(tmp_path):
    """The guard that lets jobd borrow `_preempt_local_save` safely.

    If sourcing armed train.sh's `_preempt_trap`, jobd's `_jobd_preempt` would be
    silently overwritten and every per-job `preempted` event, breadcrumb and
    bounded flush would be lost on the death path.
    """
    probe = tmp_path / "probe.sh"
    probe.write_text(
        "PREEMPT_TRAP_NO_INSTALL=1\n"
        f'. "{TRAP}"\n'
        'trap -p TERM INT\n'
        'declare -F _preempt_local_save >/dev/null && echo PRIMITIVE_OK\n')
    r = subprocess.run(["bash", str(probe)], capture_output=True, text=True,
                       timeout=60)
    assert "PRIMITIVE_OK" in r.stdout, "the primitive must still be defined"
    assert "_preempt_trap" not in r.stdout, (
        "PREEMPT_TRAP_NO_INSTALL=1 must arm NOTHING — it armed a TERM/INT trap")


def test_bare_source_still_arms_the_run_lane_trap(tmp_path):
    """...and the guard must not have disarmed the run lane by accident."""
    probe = tmp_path / "probe.sh"
    probe.write_text(f'. "{TRAP}"\ntrap -p TERM INT\n')
    r = subprocess.run(["bash", str(probe)], capture_output=True, text=True,
                       timeout=60)
    assert "_preempt_trap" in r.stdout, "train.sh's trap must still self-install"


# --- jobs lane: the trigger that never existed ------------------------------- #

def _jobd_src():
    return open(os.path.join(_VAST, "onstart", "jobd.sh")).read()


def test_jobd_asks_the_trainer_before_it_flushes():
    """Ordering IS the feature, in BOTH lanes.

    A flush can only push bytes that already exist; killed 19 min into a 20 min
    SAVE_STEPS window those bytes are 19 min stale. The jobs lane had no SIGUSR1
    path at all — so `preempt_save.py` could have been staged perfectly and still
    never have fired on v7..v11.
    """
    # CODE ONLY, for the same reason as the trainer-probe test: `declare -F
    # _preempt_local_save` and the block comment both mention the name, so a
    # substring search on the raw source passes even with the CALL deleted.
    src = "\n".join(ln for ln in _jobd_src().splitlines()
                     if not ln.lstrip().startswith("#"))
    trap = src.index("_jobd_preempt() {")
    ask = src.index('CKPT_DIR="$wdir/work" _preempt_local_save', trap)
    flush = src.index('b2x_push "$wdir/work"', trap)
    assert ask < flush, (
        "jobd must ask the trainer for a fresh checkpoint BEFORE flushing")


def test_jobd_emits_the_preempt_save_outcome_per_job():
    src = _jobd_src()
    trap = src[src.index("_jobd_preempt() {"):]
    assert "_preempt_save_emit()" in trap and "preempt_save --instance-id" in trap, (
        "every preempt-save outcome — including a SKIP — must reach B2")
    assert "preempt_save" in __import__("jobmeta").EVENTS, (
        "the event kind must be declared in jobmeta.EVENTS")


def test_jobd_preempt_flush_excludes_the_completeness_markers():
    """`.preempt_*` must never be uploaded ahead of the bytes it certifies.

    b2x orders NEWEST FIRST and the marker is by construction the newest file in
    its checkpoint, so without the exclusion a deadline-truncated flush publishes
    a GREEN FLAG over weights that are not on B2 — inverting the prefix-
    completeness property newest-first exists to provide.
    """
    src = _jobd_src()
    trap = src[src.index("_jobd_preempt() {"):]
    excl = trap.index("--exclude '.preempt_*'")
    inc = trap.index('cinc+=(--include "$cpat")')
    assert excl < inc, "first-match-wins: the exclude must precede the includes"


# --- the concurrent-rank marker race ----------------------------------------- #
#
# Found 2026-08-06 by the first REAL 2-rank DDP run this module ever had
# (`preempt_save_smoke.py`, inside the box image). Every unit test above drove
# the ranks in an order that hid it, and the result was that under DDP — which
# is every production training run we do — a preempt-forced save could NEVER be
# marked complete, even with the module perfectly staged and the trigger
# perfectly wired. Observed: "checkpoint-1658 UNPROVEN — not every rank
# reported", then the shell timing out after its full 90 s.
#
# Root cause: rank 0 called `clear_markers(ckpt)` with no token, and the ranks
# reach `on_save` CONCURRENTLY, so a peer that had already reported for THIS
# attempt had its marker deleted by rank 0's stale-marker sweep.

def test_rank0_clear_must_not_delete_a_peers_marker_for_this_attempt(tmp_path):
    """The exact live interleaving: rank 1 marks, THEN rank 0 sweeps."""
    ck = tmp_path / "checkpoint-1658"
    ck.mkdir()
    ps.mark_rank_done(str(ck), 1, 4242)          # peer got here first (real DDP)
    ps.clear_markers(str(ck), keep_token=4242)   # rank 0's stale-marker sweep
    ps.mark_rank_done(str(ck), 0, 4242)
    assert ps.ranks_done(str(ck), 2, 4242) == {0, 1}, (
        "rank 0's sweep deleted a peer's report for the attempt in flight — "
        "finalize can then never see every rank and the checkpoint is never "
        "certified, which is the DDP-only failure the smoke test caught")


def test_clear_still_removes_an_earlier_attempts_markers_when_keeping(tmp_path):
    """The fix must not cost the stale-marker defence it exists for."""
    ck = tmp_path / "checkpoint-50"
    ck.mkdir()
    ps.mark_rank_done(str(ck), 1, "OLD")             # abandoned attempt
    (ck / ps.COMPLETE_MARKER).write_text('{"files": {}}')
    ps.clear_markers(str(ck), keep_token="NEW")
    assert ps.ranks_done(str(ck), 2, "OLD") == set(), \
        "a previous attempt's rank marker must still be swept"
    assert not (ck / ps.COMPLETE_MARKER).exists(), (
        "a stale COMPLETE_MARKER must ALWAYS be removed — only rank 0 writes it "
        "and it has not written this attempt's yet, so any that exists is stale")


def test_the_callback_passes_the_agreed_token_to_clear(tmp_path):
    """Wiring guard: `clear_markers` is only safe when given the token."""
    src = open(os.path.join(_HERE, "onstart", "preempt_save.py")).read()
    assert "clear_markers(ckpt, keep_token=request.token)" in src, (
        "on_save must pass the agreed token to clear_markers, or it wipes "
        "concurrent peers' reports under DDP")


def test_jobd_prelude_runs_under_set_u_and_stages_for_the_old_probe(tmp_path):
    """The jobs-lane prelude, executed for real under `set -uo pipefail`.

    Two things it must get right, both of which bit during development:

    * `set -u`. The first cut referenced `$ROOT`, which jobd defines ~40 lines
      BELOW this block — an unbound variable under `set -u` aborts the shell, so
      it would have killed the daemon at boot on every box. A static read of the
      file does not catch that; running it does.
    * The COMPATIBILITY copy. The jobs lane stages the trainer from
      b2:.../runsets/<name>/ and that copy is pinned by a sha1 `tracks:` gate in
      every job-config, so it cannot be restaged to pick up the new probe without
      invalidating authorised bundles. It already probes /workspace, so jobd puts
      the module where the OLD probe looks.
    """
    src = open(os.path.join(_HERE, "onstart", "jobd.sh")).read()
    start = src.index("# --- preempt-forced checkpoint primitive")
    end = src.index("\nIID=", start) if "\nIID=" in src[start:] else start + 2500
    prelude = src[start:end]

    staged = tmp_path / "jobd"
    root = tmp_path / "workspace"
    staged.mkdir()
    root.mkdir()
    for f in ("preempt_save.py", "preempt_trap.sh"):
        shutil.copy(os.path.join(_HERE, "onstart", f), staged / f)

    script = tmp_path / "prelude.sh"
    script.write_text(
        "set -uo pipefail\n"
        # Disposition probe, BEFORE the prelude. A signal that was already
        # SIG_IGN when bash started cannot be trapped at all (POSIX), and bash
        # reports it as `trap -- '' SIGINT` — so under a launcher that ignores
        # INT/TERM (any `&` job of a non-job-control shell) this test can
        # neither observe an arm nor be fooled by one. Say so and skip, rather
        # than pass silently in a state where the assertion cannot fail.
        "if trap -p INT TERM | grep -q \"^trap -- '' SIG\"; then echo IGN_AT_ENTRY; fi\n"
        f'JOBD_DIR="{staged}"\nJOBD_ROOT="{root}"\n'
        + prelude +
        '\necho "DIR=${PREEMPT_SAVE_DIR:-UNSET}"\n'
        'declare -F _preempt_local_save >/dev/null && echo PRIMITIVE_DEFINED\n'
        # Match the HANDLER, not merely any output.
        'if trap -p TERM INT | grep -q _preempt_trap; then echo TRAP_ARMED; fi\n'
        'echo OK\n')
    r = subprocess.run(["bash", str(script)], capture_output=True, text=True,
                       timeout=60)
    assert "OK" in r.stdout, (
        f"the jobd prelude aborted under `set -u` — this runs at daemon boot on "
        f"every box:\n{r.stdout}\n{r.stderr}")
    assert "unbound variable" not in r.stderr, r.stderr
    assert f"DIR={staged}" in r.stdout, "PREEMPT_SAVE_DIR must point at the bundle"
    assert "PRIMITIVE_DEFINED" in r.stdout
    if "IGN_AT_ENTRY" in r.stdout:
        pytest.skip("INT/TERM were SIG_IGN at bash entry — bash cannot install "
                    "the trap here, so the arm assertion below cannot fail. "
                    "Re-run without a signal-ignoring launcher.")
    assert "TRAP_ARMED" not in r.stdout, (
        "sourcing the trap must NOT arm it — that would replace jobd's own "
        "_jobd_preempt and lose every per-job preempted event and flush")
    assert (root / "preempt_save.py").is_file(), (
        "the module must also land in $JOBD_ROOT: the trainer staged on B2 today "
        "probes /workspace, and its runset copy is sha1-pinned so it cannot be "
        "restaged without invalidating every authorised bundle")

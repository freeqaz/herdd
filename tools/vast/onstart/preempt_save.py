"""Force ONE immediate, COMPLETE checkpoint out of a live HF Trainer.

Why this exists
---------------
`preempt_trap.sh` today does the slow thing on a preemption signal: it flushes
checkpoints that ALREADY EXIST to B2. It never asks the trainer to *make* a
fresh one. That ordering is backwards for the case that actually loses work —
being killed 19 minutes into a 20-minute `SAVE_STEPS` window, where the newest
bytes on disk are 19 minutes stale and no amount of flushing invents them.

A local save is ~0.5-11 s at our shape (~0.98 GB: adapter 323 MB + optimizer
646 MB); pushing that same ~1 GB to B2 is far slower and per-flow-shaped. And
since instance->instance salvage landed (`tools/vast/salvage.py`), a checkpoint
that only ever reaches LOCAL DISK is recoverable — so writing one is worth far
more than it used to be. Hence: save locally FIRST, flush second.

THE HAZARD THIS MODULE IS SHAPED BY
-----------------------------------
`transformers 5.13.0` writes checkpoints **IN PLACE**: no `tmp-checkpoint-N`
staging directory, no `os.rename`, and under DDP a NON-ZERO rank can create the
checkpoint directory before rank 0 has written a byte into it. So a checkpoint
directory EXISTING proves nothing, and a checkpoint killed mid-write is a
directory full of plausible-looking files.

That matters more here than anywhere else in the system, because salvage now
FAITHFULLY RESCUES whatever is on that disk. A torn checkpoint that reads as
complete would be copied, byte-verified against its own torn self, pushed to
B2, and eventually resumed from — training from corrupt weights with no error
anywhere in the chain. **A torn checkpoint is worse than no checkpoint.**

Two independent guarantees, neither of which relies on HF's write ordering:

1. **All ranks agree to save, or none does.** `TrainerControl.should_save` is
   read by every rank, and `_save_checkpoint` contains collective operations. If
   rank 1 sets it and rank 0 does not, the run either deadlocks or writes a
   directory only one rank contributed to. A signal is delivered to whichever
   process the OS felt like, at whatever moment — so the local flag is NEVER
   trusted directly. `agree()` does a one-element MAX all-reduce at a step
   boundary, which is the only point where every rank is provably in lockstep,
   and every rank acts on the REDUCED value.

2. **Completion is proven by a marker no partial write can forge.** Each rank
   writes `.preempt_rank_<i>_ok` from `on_save` — i.e. after ITS OWN
   `_save_checkpoint` work returned. Rank 0 then waits (bounded) for one marker
   per rank and only then writes `COMPLETE_MARKER`. So the marker means "every
   rank finished writing", which is strictly stronger than `trainer_state.json`
   (which only proves RANK 0 finished).

   Deliberately NOT a `dist.barrier()`: a NCCL barrier against a rank that is
   already dead — the exact situation a preemption creates — hangs forever, and
   hanging inside `on_save` is a worse failure than a missing marker. Polling
   files has a timeout; a barrier does not.

3. **The claim is FALSIFIABLE, and scoped to one attempt.** Adversarial review
   (2026-08-05) found two false-complete paths in the first cut of this module,
   both from the same root: the marker asserted nothing about the bytes it
   certified, and nothing ever invalidated it. transformers never cleans a
   checkpoint directory, so an abandoned attempt's `.preempt_rank_1_ok` survives
   into a re-entered `checkpoint-<N>` and let rank 0 count "2/2 reported" when
   the second report came from a run that died hours earlier — and a plain
   `SAVE_STEPS` save torn into a directory that still carried an old
   `COMPLETE_MARKER` inherited the green flag outright. So:

   * every rank marker carries an **attempt token**, MAX-reduced across ranks
     exactly like the save decision, and `ranks_done()` counts only markers
     bearing THIS attempt's token;
   * `COMPLETE_MARKER` records a **file inventory** (name + size, markers
     excluded) and `is_complete()` re-verifies it against the directory rather
     than testing `os.path.exists`. That is what turns the flag from "a file
     that exists" into a claim a later writer can be caught contradicting —
     including `_disable_checkpoint_optimizer_state`, which deliberately renames
     `optimizer.pt` out of an existing checkpoint on a bnb-quantized resume;
   * rank 0 clears any earlier attempt's markers before writing its own.

`COMPLETE_MARKER` is an ADDITIVE claim. Its absence does not mean a checkpoint
is torn (every checkpoint written by the normal `SAVE_STEPS` path lacks it);
its presence means a preempt-forced save was fully written by all ranks AND the
directory still matches what was certified. Read it that way and it can only
ever narrow trust, never widen it.

Caveat worth holding: under plain DDP + LoRA the non-zero ranks' only
contribution to a checkpoint is `rng_state_<i>.pth` (`save_model` is gated on
`args.should_save`), so their markers prove liveness rather than weights. The
extra strength over `trainer_state.json` is fully real only under sharded saves
(FSDP/DeepSpeed), which `--fsdp` does enable.

Pure and injectable: no `torch`, no `transformers`, no signal handling at
import time, so the whole state machine is unit-testable in the portable lane.
"""

from __future__ import annotations

import json
import os
import random
import signal
import time

#: Written by rank 0 ONLY after every rank reported its own save finished.
#: Presence => complete. Absence => unproven, never "torn".
COMPLETE_MARKER = ".preempt_save_complete"

#: One per rank, written by that rank from `on_save`.
RANK_MARKER = ".preempt_rank_{}_ok"

#: The signal `preempt_trap.sh` sends to ask for a save. SIGUSR1 and not
#: SIGTERM: the trainer must not confuse "checkpoint now" with "shut down", and
#: SIGTERM is already spoken for by the shell trap that wraps the process.
SAVE_SIGNAL = signal.SIGUSR1

#: How long rank 0 waits for the other ranks' markers before giving up and
#: writing NOTHING. Generous against a slow optimizer write, bounded because the
#: box is dying. The shell's PREEMPT_SAVE_WAIT_S (30 s) has to cover the WHOLE
#: sequence, and that sequence is SEQUENTIAL, not overlapping: reach the next
#: step boundary + rank 0's own save (0.5-11 s at our shape) + up to this much
#: polling. At 20 s that sum exceeds the ceiling, so the shell would time out and
#: begin the B2 flush WHILE the save is still in flight — uploading a
#: half-written adapter_model.safetensors as a COMPLETE B2 object (the periodic
#: loop avoids exactly this with `--min-age 45s`; the trap deliberately drops
#: it). 12 s keeps the worst case inside the ceiling.
RANK_WAIT_S = 12.0

#: Where each rank advertises its pid so the shell trap can signal it. Under
#: torchrun every rank is a separate process and `kill -USR1 -<pgid>` would
#: also hit unrelated children (SIGUSR1's default action is TERMINATE), so we
#: enumerate pids explicitly instead.
PIDDIR_ENV = "PREEMPT_SAVE_PIDDIR"
DEFAULT_PIDDIR = "/workspace/.preempt_save_pids"


# --- the flag a signal handler is allowed to touch -------------------------- #

class SaveRequest:
    """A one-shot "please checkpoint" flag, plus the rank-collective agreement.

    `requested` is the ONLY thing a signal handler writes — setting a boolean is
    async-signal-safe, and anything more (logging, I/O, locks) risks deadlocking
    against whatever the main thread was holding when the signal landed.
    """

    def __init__(self, nonce=None):
        self.requested = False      # set by the signal handler, any rank
        self.fired = False          # a save has already been agreed + issued
        self.agreed_step = None
        # Per-process proposal for this attempt's identity. MAX-reduced across
        # ranks at agreement time so every rank writes the SAME token.
        self.nonce = int(nonce if nonce is not None
                         else random.getrandbits(48))
        self.token = self.nonce

    def request(self, *_a):
        """Signal handler. Async-signal-safe by construction: one store."""
        self.requested = True

    def agree(self, all_reduce_max=None):
        """Has ANY rank been signalled? Returns the COLLECTIVE answer.

        `all_reduce_max(int) -> int` is injected (in production a one-element
        MAX all-reduce). Its result — never `self.requested` — is what decides,
        so a signal delivered to exactly one rank still produces an identical
        decision on all of them. With no distributed group the local flag is
        already the collective answer.

        Already-fired requests never re-fire: without that, one signal would set
        `should_save` on every subsequent step for the rest of the run.
        """
        if self.fired:
            return False
        local = 1 if self.requested else 0
        if all_reduce_max is not None:
            try:
                local = int(all_reduce_max(local))
            except Exception:
                # A collective that fails mid-preemption (a peer already gone)
                # must not take the trainer down. Fall back to the local flag:
                # at worst one rank saves alone and the completion marker is
                # never written, which reads as "incomplete" — the safe verdict.
                local = 1 if self.requested else 0
        return bool(local)


    def agree_token(self, all_reduce_max=None):
        """One attempt id every rank computes identically. Falls back to the
        local nonce when there is no group (world_size 1) or the collective
        fails — in the failure case ranks may disagree, which can only make
        `finalize` see too FEW matching markers and withhold the flag."""
        if all_reduce_max is None:
            return self.nonce
        try:
            return int(all_reduce_max(self.nonce))
        except Exception:
            return self.nonce


def install(request=None, signum=SAVE_SIGNAL):
    """Arm the handler. Returns the `SaveRequest` the callback should read."""
    req = request or SaveRequest()
    try:
        signal.signal(signum, req.request)
    except (ValueError, OSError):
        # Not the main thread, or the signal is unavailable. The trainer still
        # runs; it simply cannot be asked to save early.
        pass
    return req


# --- pid advertisement ------------------------------------------------------ #

def piddir():
    return os.environ.get(PIDDIR_ENV) or DEFAULT_PIDDIR


def write_pid(rank, pid=None, directory=None):
    """Advertise this rank's pid so the shell trap can signal exactly it."""
    d = directory or piddir()
    try:
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, f"{int(rank)}.pid")
        with open(path, "w") as fh:
            fh.write(str(int(pid if pid is not None else os.getpid())))
        return path
    except OSError:
        return None            # advertising is best-effort, never fatal


# --- completion marking ----------------------------------------------------- #

def _is_marker(name):
    return name.startswith(".preempt_")


def inventory(ckpt_dir):
    """`{name: size}` for every NON-marker file in the checkpoint. Never raises.

    This is what turns the completion flag from "a file that exists" into a
    claim that can be FALSIFIED. Without it the marker asserts nothing about the
    bytes it certifies, and any later writer that tears the directory inherits
    the green flag (a re-entered `checkpoint-<N>` after a resume, or
    `_disable_checkpoint_optimizer_state` renaming `optimizer.pt` out of it).
    """
    out = {}
    try:
        for name in os.listdir(ckpt_dir):
            if _is_marker(name):
                continue
            p = os.path.join(ckpt_dir, name)
            try:
                if os.path.isfile(p):
                    out[name] = os.path.getsize(p)
            except OSError:
                continue
    except OSError:
        return {}
    return out


def mark_rank_done(ckpt_dir, rank, token):
    """Called by EVERY rank once its own save work has returned.

    The `token` is what makes a marker belong to THIS attempt. transformers
    writes checkpoints in place and never clears the target directory, so a
    `.preempt_rank_1_ok` left by an earlier, ABANDONED attempt at the same step
    survives into a re-entered `checkpoint-<N>` — and counting markers by
    filename alone would let rank 0 see "2/2 ranks reported" when the second
    report came from a run that died hours ago.
    """
    path = os.path.join(ckpt_dir, RANK_MARKER.format(int(rank)))
    with open(path, "w") as fh:
        fh.write(str(token))
        fh.flush()
        os.fsync(fh.fileno())
    return path


def ranks_done(ckpt_dir, world_size, token):
    """Which ranks reported FOR THIS ATTEMPT. Reads each marker; never raises."""
    done = set()
    for r in range(int(world_size)):
        p = os.path.join(ckpt_dir, RANK_MARKER.format(r))
        try:
            with open(p) as fh:
                if fh.read().strip() == str(token):
                    done.add(r)
        except OSError:
            continue
    return done


def clear_markers(ckpt_dir, keep_token=None):
    """Drop markers left in this directory by an EARLIER attempt.

    Called by rank 0 before it marks, so a stale `COMPLETE_MARKER` cannot
    outlive the bytes it described. Best-effort: a marker we fail to remove is
    still token-checked by `ranks_done`, and `is_complete` still re-verifies the
    inventory, so this is defence in depth rather than the load-bearing guard.

    `keep_token` — THIS attempt's agreed token — is not optional in practice, and
    omitting it is a live DDP bug (found 2026-08-06 by the first real 2-rank run
    this module ever had; every prior test drove the ranks in an order that hid
    it). The ranks reach `on_save` CONCURRENTLY, so a peer routinely writes its
    `.preempt_rank_<i>_ok` BEFORE rank 0 gets here. A blanket wipe then deletes a
    marker belonging to the attempt now in flight, `finalize` waits out its whole
    timeout for a report that will never be rewritten, and `COMPLETE_MARKER` is
    never written at all — i.e. under DDP, which is every production training run
    we do, the feature could not produce a complete checkpoint even with perfect
    delivery and a perfect trigger. Observed: rank 1 marked, rank 0 cleared,
    `ranks_done` saw {0}, result "UNPROVEN — not every rank reported".

    Keeping current-token markers costs the stale-marker defence nothing: an
    abandoned attempt's token is a different 48-bit value, so its markers are
    still removed here AND still rejected by `ranks_done`.
    """
    try:
        names = os.listdir(ckpt_dir)
    except OSError:
        return
    keep = None if keep_token is None else str(keep_token)
    for name in names:
        if not _is_marker(name):
            continue
        path = os.path.join(ckpt_dir, name)
        # Never preserve a COMPLETE_MARKER: only rank 0 writes it and it has not
        # written this attempt's yet, so any that exists is by definition stale.
        if keep is not None and name != COMPLETE_MARKER:
            try:
                with open(path) as fh:
                    if fh.read().strip() == keep:
                        continue          # a peer's report for THIS attempt
            except OSError:
                pass
        try:
            os.unlink(path)
        except OSError:
            pass


def finalize(ckpt_dir, world_size, step=None, token=None, *, wait_s=RANK_WAIT_S,
             sleep=time.sleep, now=time.monotonic):
    """Rank 0: wait for every rank, then write `COMPLETE_MARKER`. Returns bool.

    Writing the marker is the LAST thing that happens to the directory, and it
    happens only when every rank has reported THIS attempt's token. Timing out
    writes nothing at all — an unproven checkpoint is left unproven rather than
    labelled complete, because the entire point is that salvage must not rescue
    a torn checkpoint under a green flag.
    """
    deadline = now() + float(wait_s)
    want = int(world_size)
    while True:
        if len(ranks_done(ckpt_dir, want, token)) >= want:
            break
        if now() >= deadline:
            return False
        sleep(0.25)
    tmp = os.path.join(ckpt_dir, COMPLETE_MARKER + ".partial")
    final = os.path.join(ckpt_dir, COMPLETE_MARKER)
    payload = json.dumps({"step": step, "world_size": want, "token": str(token),
                          "ts": time.time(), "files": inventory(ckpt_dir)},
                         sort_keys=True)
    # Write-then-rename so the marker itself can never be observed half-written
    # (the one file in the tree where that is cheap to guarantee, and the one
    # file whose partial read would be interpreted as a completeness claim).
    with open(tmp, "w") as fh:
        fh.write(payload)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, final)
    return True


def is_complete(ckpt_dir):
    """Did a preempt-forced save finish here, AND does the directory still match
    what was certified? Never raises.

    The inventory re-check is the point: a marker alone only says "some attempt
    finished here once". Comparing the recorded name/size set against the
    directory today is what makes the claim falsifiable, so a checkpoint that
    was re-entered and re-torn — or had a file renamed out of it — reports
    INCOMPLETE instead of inheriting an old green flag.
    """
    try:
        with open(os.path.join(ckpt_dir, COMPLETE_MARKER)) as fh:
            body = json.load(fh)
    except (OSError, ValueError):
        return False
    want = body.get("files")
    if not isinstance(want, dict):
        return False                      # a marker with no inventory is not a claim
    return inventory(ckpt_dir) == {str(k): int(v) for k, v in want.items()}


# --- the HF TrainerCallback ------------------------------------------------- #

def make_callback(request, *, rank=0, world_size=1, all_reduce_max=None,
                  base=None, log=print, wait_s=RANK_WAIT_S):
    """Build the `TrainerCallback` subclass, importing transformers LAZILY.

    `base` lets the tests pass a stub so the whole state machine runs with no
    `transformers` installed — the portable-lane discipline the rest of
    `tools/vast/` follows.
    """
    if base is None:                                  # pragma: no cover - import
        from transformers import TrainerCallback as base   # noqa: N813

    class PreemptSaveCallback(base):
        """Turn a signal into exactly one complete, all-rank checkpoint."""

        def on_step_end(self, args=None, state=None, control=None, **kw):
            # A step boundary is the ONLY point where every rank is provably
            # together, which is what makes the all-reduce below safe to do.
            if control is None or request.fired:
                return control
            if not request.agree(all_reduce_max):
                return control
            request.fired = True
            request.agreed_step = getattr(state, "global_step", None)
            # Agree a TOKEN the same way we agreed to save at all: a MAX-reduce
            # of each rank's nonce yields one value every rank computes
            # identically, which is what lets `ranks_done` tell this attempt's
            # markers from those an abandoned earlier attempt left in a
            # re-entered checkpoint-<N>.
            request.token = request.agree_token(all_reduce_max)
            control.should_save = True
            log(f".. preempt-save: all ranks agreed at step "
                f"{request.agreed_step} — forcing one checkpoint")
            return control

        def on_save(self, args=None, state=None, control=None, **kw):
            # Fires on EVERY rank after that rank's `_save_checkpoint` returned.
            if not request.fired or request.agreed_step is None:
                return control                        # a routine SAVE_STEPS save
            step = getattr(state, "global_step", None)
            if step != request.agreed_step:
                return control                        # not the save we asked for
            out = getattr(args, "output_dir", None)
            if not out:
                return control
            ckpt = os.path.join(out, f"checkpoint-{step}")
            # Every write below is best-effort. A full disk on the death path
            # must never raise out of a callback and abort the run — the whole
            # feature is a bonus, and losing training to it would be perverse.
            try:
                if int(rank) == 0:
                    # Drop an earlier attempt's markers BEFORE this attempt's
                    # go down, so a stale COMPLETE_MARKER cannot outlive the
                    # bytes it described even for the moments before we
                    # overwrite it. `request.token` is load-bearing: the ranks
                    # are here CONCURRENTLY, so without it this wipes peers'
                    # reports for the attempt in flight and finalize can never
                    # succeed (see clear_markers' docstring).
                    clear_markers(ckpt, keep_token=request.token)
                mark_rank_done(ckpt, rank, request.token)
            except OSError as e:
                log(f"!! preempt-save: rank {rank} could not mark {ckpt}: {e}")
                return control
            if int(rank) == 0:
                try:
                    ok = finalize(ckpt, world_size, step=step,
                                  token=request.token, wait_s=wait_s)
                except OSError as e:
                    log(f"!! preempt-save: could not finalize {ckpt}: {e}")
                    return control
                log(f".. preempt-save: {ckpt} "
                    + ("COMPLETE (all ranks)" if ok else
                       "UNPROVEN — not every rank reported; left unmarked"))
            return control

    return PreemptSaveCallback()


def all_ranks_armed(world_size, all_reduce_min=None):
    """Did EVERY rank get this far? One MIN-reduce, evaluated before any
    per-step collective is ever posted.

    The wiring in the trainer is fail-open per rank, and `on_step_end` posts an
    all-reduce every optimizer step. Asymmetric arming would therefore hang a
    perfectly healthy run at step 1 — armed ranks blocking on a collective the
    unarmed ones never join, with only the NCCL watchdog to break it. Reducing
    a constant 1 across the group answers "are we all here?": a rank that never
    armed never joins, so the ranks that did arm hit the group's own timeout
    here — at startup, loudly, before any training time is spent — instead of
    mid-run. A `False` (or an exception) simply disables the feature.
    """
    if int(world_size) <= 1:
        return True
    red = all_reduce_min if all_reduce_min is not None else torch_all_reduce_min()
    if red is None:
        return True                   # no group: single process by definition
    try:
        return int(red(1)) == 1
    except Exception:
        return False


def _torch_reduce(op_name):
    try:
        import torch
        import torch.distributed as dist
    except Exception:                                 # pragma: no cover
        return None
    if not (dist.is_available() and dist.is_initialized()):
        return None

    def _reduce(v):
        dev = ("cuda" if torch.cuda.is_available() else "cpu")
        t = torch.tensor([int(v)], device=dev)
        dist.all_reduce(t, op=getattr(dist.ReduceOp, op_name))
        return int(t.item())

    return _reduce


def torch_all_reduce_min():
    """Production `all_reduce_min`, or None when there is no process group."""
    return _torch_reduce("MIN")


def torch_all_reduce_max():
    """Production `all_reduce_max`, or None when there is no process group."""
    return _torch_reduce("MAX")

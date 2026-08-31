"""vastlib.cli._compose — wire the upward seams, at the one ring allowed to.

Why this module exists
----------------------
Three names are CALLED by a lower ring and DEFINED in a higher one:

    launch.launch.compose_jobs_launch_env   -> jobs.bundle
    launch.launch.fleet_watch_best_effort   -> fleet.client
    boxes.lifecycle.fleet_operator_intent   -> fleet.client

Each is a raising seam at its call site, and each raise is deliberate (plan
§7.3): a silent no-op on `compose_jobs_launch_env` would launch a `--jobs` box
with no jobd, no B2 creds and no minted key — a box that bills and can never
claim a ticket — and a silent no-op on `fleet_operator_intent` would let a
human's `herdd stop` read as OUTBID and get the box rescue-resumed all night
(SPOT_DESIGN §3.5). The seams are not scaffolding; they are the failure mode
made loud.

They cannot be closed where they are called. `jobs` and `fleet` sit ABOVE
`launch` and `boxes` in the dependency DAG (vastlib/README §1), and
import-linter reads the AST — so a deferred, inside-the-function import does
not dodge the contract either. The direction is the problem, not the timing.
`cli` is the ring that may import everything, so `cli` is where the wiring
belongs; this is the same injectable-default idiom `workflowctl.build_box_resolver`
already uses for its `jobs_composer` argument.

Why the bind happens at CALL time, not at import time
-----------------------------------------------------
The obvious spelling is three assignments at `cli/__init__.py` import. It is
wrong here for one measured reason: **importing a module must not change what
another module's attributes do.** `test_vastlib_launch.py` asserts that the two
`launch` seams raise, and `test_vastlib_supervise_*.py` drives `_do_launch`
through a partly-stubbed tree. If merely importing `vastlib.cli` — which any
test collecting a `cli/` test file does, in the same process — silently
rebound those names, that assertion would pass or fail on pytest's collection
ORDER. A test whose verdict depends on which file was imported first is not a
test. The census stays the census until a COMMAND actually runs.

So `bind()` is called from the entry points that are about to do the work:
`cli.main.main()` (every subcommand, the real composition root) and the three
command modules that reach `_do_launch` on their own (`cli/launch.py`,
`cli/train.py`, `cli/supervise.py` — the last through the run lane's relaunch).
Calling it more than once is free; it is three assignments of the same objects.

THE FIFTH CALLER IS NOT IN THIS RING, AND HAD TO BE ADDED THE HARD WAY.
`tools/vast/fleetd.py::run()` — the launcher the systemd unit executes — calls
`bind()` too, at RUN time, for the same reason and with the same import-order
care. It was missed when this module was written, and the miss was live: the
daemon's own composition root is `fleet.daemon`, which may not import `cli` (it
is the top layer), so inside the fleetd process every seam above stayed raising
and the first pull-condemned replacement launch died on
`compose_jobs_launch_env` — every tick, with the box never replaced. Zone E is
outside `importlinter.ini`'s `root_package = vastlib`, which is why the launcher
can hold both halves and `fleet.daemon` cannot.
`test_vastlib_cli_launch.py::test_the_fleetd_launcher_binds_every_seam` (and its
source twin) is the guard; the rule is that EVERY process entry point that can
reach `_do_launch` or `lifecycle`'s operator drivers must call `bind()`.

How to STEER these three in a test
----------------------------------
Patch the OWNING module, not `launch`/`lifecycle`:

    monkeypatch.setattr(bundle, "compose_jobs_launch_env", fake)   # steers
    monkeypatch.setattr(launch_mod, "compose_jobs_launch_env", fake)  # does NOT

`bind()` reads the owner's CURRENT attribute every time it runs, so an
owner-side patch installed before the command wins; a patch of the call-site
attribute is overwritten by the next `bind()`. That is the opposite of the
`_REBOUND_SEAMS` rule one ring down (where the assignment happens once, at
import, and the call-site attribute IS the patch point) — the difference is
exactly that these three are bound late, and it is why they are documented
here rather than in a comment on the assignment.

A bind is still PROCESS-GLOBAL once a test runs a command, so the suite hands
the census back rather than hoping: `tools/vast/conftest.py::
_restore_cross_ring_seam_bindings` snapshots exactly these three attributes
around every test (its roster is pinned to `SEAM_BINDINGS` by a test, so a
fourth row here cannot silently escape it). Without that,
`test_vastlib_launch.py`'s "these seams still raise" census passed or failed on
collection order — measured, four failures behind `test_vastlib_cli_main.py`.

What is deliberately NOT here
-----------------------------
* (`cmd_job_attach` used to be named here as the one remaining raising seam,
  with the note that "a fifth row pointing at `cli/job/attach.py` is a
  defensible next step". It is now row five — see the block above the table.
  The predicted cost was MEASURED before the row landed: three re-attaches on
  three boxes in one 4h45m fleetd window, 2026-08-17.)
  (`_reset_run_markers` used to be named here as "genuinely unported"; that
  went stale at step 6d, when its body MOVED into
  `supervise/replacement.py` — see the home ruling on the def there. Nothing
  named `_reset_run_markers` raises any more.)
  (`fleet_note_operator_stop` graduated INTO the roster
  once the 6f census showed its body landed and its `cmd_stop` call site was
  unguarded — a raise on a real `herdd stop`; `_box_lifecycle_soft` closed
  in job_lane itself via the same forwarder handoff uses.)
* **Any policy.** No env reads, no config, no ordering beyond the assignments.
  If this function ever needs a branch, the branch belongs in the thing being
  bound.

Provenance: new code (README §2 rule 7 — no `moved-from:` marker). The flat
`herdd.py` needs none of this: one module, one namespace, no rings.
"""

from __future__ import annotations

from vastlib.boxes import lifecycle
from vastlib.cli.job import attach as job_attach
from vastlib.fleet import client as fleet_client
from vastlib.jobs import bundle
from vastlib.launch import launch as launch_mod

# ROW FIVE, and the only one whose owner is inside `cli` itself. `cmd_job_attach`
# is the B2-key ROTATION lane (CREDENTIAL_LIFECYCLE.md): `cmd_start` and
# `supervise.job_lane._job_sup_reattach` call it through
# `boxes.lifecycle.cmd_job_attach` after a box comes back, because an
# attach-started daemon does not survive a resume. Nothing below `cli` may
# import the body, so the seam had nowhere lower to land and stayed raising.
#
# THE COST, MEASURED (not predicted) in the 2026-08-17 fleetd window: three
# re-attaches, three boxes (47966341 resumed-in-place, 47967469 and 47974737 on
# the eviction ladder), each ending in
#
#   !! jobd re-attach failed (NotImplementedError: cmd_job_attach: not ported
#      yet (plan §8 step 5) ...) — box onstart revives jobd on resume
#
# The swallow is deliberate and stays (an ssh refusal must never kill the
# babysitter), so the failure was invisible in every verdict — the box came
# back and ran, holding its LAUNCH-BAKED B2 key instead of a freshly minted
# scoped one. That is the whole rotation lane silently not running.

#: The wiring, as data, so a test can assert the census without importing five
#: modules: (module that CALLS the name, attribute name, module that DEFINES it).
SEAM_BINDINGS: tuple[tuple[str, str, str], ...] = (
    ("vastlib.launch.launch", "compose_jobs_launch_env", "vastlib.jobs.bundle"),
    ("vastlib.launch.launch", "fleet_watch_best_effort", "vastlib.fleet.client"),
    ("vastlib.boxes.lifecycle", "fleet_operator_intent", "vastlib.fleet.client"),
    ("vastlib.boxes.lifecycle", "fleet_note_operator_stop", "vastlib.fleet.client"),
    ("vastlib.boxes.lifecycle", "cmd_job_attach", "vastlib.cli.job.attach"),
)


def bind() -> None:
    """Point the five cross-ring seams at their real implementations.

    Idempotent, and cheap enough to call from every entry point that might need
    it — five attribute assignments. Deliberately reads the owning module's
    attribute at CALL time so an owner-side `monkeypatch.setattr` steers the
    command (see the module docstring)."""
    launch_mod.compose_jobs_launch_env = bundle.compose_jobs_launch_env
    launch_mod.fleet_watch_best_effort = fleet_client.fleet_watch_best_effort
    lifecycle.fleet_operator_intent = fleet_client.fleet_operator_intent
    lifecycle.fleet_note_operator_stop = fleet_client.fleet_note_operator_stop
    lifecycle.cmd_job_attach = job_attach.cmd_job_attach

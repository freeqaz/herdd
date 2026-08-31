"""vastlib.boxes — a rented machine's whole life: mutate it, reach it, judge it, reap it.

Why this layer exists
---------------------
Lifecycle mutation was the fourth-largest fan-in cluster (44 inbound calls) and
it is the ring where the money is: every stop, start, destroy, label and bid
PUT bills or stops billing. Grouping it with the two ways we touch a box (ssh,
remote exec) and the two ways we judge one (health, reap) puts every
side-effecting box operation behind one boundary that `supervise/` and `cli/`
call *down* into.

Planned modules (plan §5)
-------------------------
  lifecycle.py  stop / start / destroy / wait / label soft ops,
                destroy+revoke, operator-intent emission.
  health.py     `BootThroughputSampler`, `parse_pull_progress`,
                `classify_box_health`, `gather_fleet_health`, the jobd probes,
                and the `GuardVerdict` enum — one lattice with `.short`,
                `.is_zombie`, `.is_advisory`, absorbing BOTH `herdd`'s
                `_GUARD_*` sets and fleetd's re-derived presentation of them.
  ssh.py        endpoints, auth preflight, tunnel, key injection.
  remote.py     `_vast_execute_soft`, `_ssh_exec_soft`, the exec wrap/nonce
                protocol, copy. One of only two direct socket users in the
                package (the other is `core.api`).
  salvage.py    `salvage.py` absorbed — disk-salvage orchestration.
  reap.py       idle/zombie ledgers, keep-label retention, the `cmd_reap`
                policy that the every-15-min systemd timer executes.

What is deliberately NOT here
-----------------------------
* **No supervision loop.** Deciding to replace an evicted box, when to rebid,
  or whether a run is at risk is `supervise/`. This ring answers "do it" and
  "what state is it in", never "should we".
* No offer search or price arithmetic — that is `market/`, a sibling in the
  same ring, reached only where the plan's module list says so.
* No fleetd daemon protocol. `fleet/client.py` owns the socket contract — a
  ring ABOVE this one, so `lifecycle.py` and `reap.py` declare the three names
  they call across that line (`fleet_operator_intent`,
  `fleet_note_operator_stop`, and `jobs`' `_fold_fleet_jobs`) as raising SEAM
  attributes carrying no `moved-from:` marker. Step 5 rebinds them; the raise
  is what stops a silent no-op from disarming an operator intent in the
  meantime.

Provenance: skeleton created 2026-08-16, plan §8 step 1. Contents arrive in
step 3 as verbatim moves from `herdd.py` and `salvage.py`.

Note for the porter, corrected 2026-08-16 at the `lifecycle.py` port:
`_put_label_soft` is defined TWICE in `herdd.py` and the citation this file
carried (":4641 dead, :5192 live", inherited from plan §1) was already stale.
Measured at rev `2b188979`: **:4768 dead, :5319 live** — the later def shadows
the earlier at import. The bodies differ, and only the LIVE one is ported: it
passes `retries=2` and returns `(bool(ok), err)` with **no** `{"success":
false}` folding, where the dead twin folds `success: false` into `err`.
Adopting the dead twin's stricter success check is a parked behavior fix (plan
§9), not a drive-by change; deleting the dead def is plan §8 step 6. Line
numbers drift with every rebase — re-locate by name (`grep -n`) before quoting
either of these again.
"""

from __future__ import annotations

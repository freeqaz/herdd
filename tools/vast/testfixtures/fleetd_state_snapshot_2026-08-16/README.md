# fleetd live-state snapshot — 2026-08-16

A real `state.json` + journal tail, captured from a running `fleetd` on
2026-08-16 05:54 UTC and used as the round-trip fixture for the `fleet/`
package: re-serialising the loaded document must reproduce it byte-for-byte.

Do not regenerate it from the current writer — its value is that it was written
by an *older* daemon, so a loader change that silently drops or reshapes a field
shows up as a diff.

## Files

### `state.json` (69,577 bytes, schema `version: 1`)

Verbatim apart from the scrubbing below. **Not reformatted**: key order,
indentation, float precision and every unknown/undocumented field are exactly as
the daemon emitted them.

Top-level keys and cardinality as captured:

| key | type | len |
|---|---|---|
| `alarms` | dict | 0 |
| `ceiling_by_box` | dict | 54 |
| `ceilings` | dict | 35 |
| `destroys` | dict | 0 |
| `intents` | dict | 172 |
| `meta` | dict | 2 (`last_ok_tick_ts`, `saved_ts`) |
| `spend_by_box` | dict | 231 |
| `strays` | dict | 0 |
| `version` | int | — |
| `watches` | dict | 1 |

237 distinct box ids appear across the box-keyed maps. `alarms`, `destroys` and
`strays` are empty here, so a round-trip test gets no coverage of their element
shape from this file — only of the empty-container case.

The single watch (`profile: jobs`, `state: watched`) is the only entry carrying
the full `policy` / `replacement` sub-documents, including
`replacement.bid_history`.

### `journal_tail_200.ndjsonl` (200 lines, 50,818 bytes)

The last 200 lines of a 31,008-line journal, covering
**2026-08-16T05:24:09Z → 2026-08-16T05:54:11Z**. Kept as a tail rather than the
whole 6 MB file. 29 distinct `event` values occur in the window, including
`fleetd_started` / `fleetd_stopped` (so the file spans a daemon restart),
`eviction_replacement_decision`, `jobs_box_launched` / `_evicted` /
`_condemned` / `_destroyed` / `_eviction_survived`, `watch_registered` /
`_adopted` / `_auto_adopted` / `_dormant` / `_finished`, `ceiling_armed` /
`ceiling_box_bound`, `alarm_raised` / `alarm_resolved`, `spend_backfilled`,
`operator_intent_destroy`, `auto_adopt_refused`, and 97 `tick`s.

The sibling `fleetd_journal_path{1,2,3}_*.ndjsonl` fixtures are hand-picked
single-box narratives; this one is a raw time-window tail instead, deliberately,
because its job is schema fidelity rather than a scenario.

## Scrubbing

Both files were passed through literal string substitutions only. No keys were
added, removed, renamed or reordered; no values were truncated.

| from | to | occurrences (`state.json` / journal tail) |
|---|---|---|
| operator user@host | `operator@workstation` | 247 / 22 |
| absolute home path | `~` | 0 / 2 |

`operator@workstation` is the same placeholder the sibling journal fixtures use,
so the whole fixture set stays consistent. The two path hits are the `"sock"`
field of the `fleetd_started` events.

**Secret scan (negative result).** Before copying, `state.json` was walked
leaf-by-leaf (2,226 leaves): zero field names matching
`key|token|secret|passw|cred|auth|bearer|api`, and exactly one string value of
length >= 24 without spaces — `"eviction_retention_expired_destroy"`, a reason
code. The journal was grepped for `api_key`, `token`, `secret`,
`BEGIN OPENSSH`, IPv4 literals, `ssh`, and provider hostnames: zero hits.
fleetd's state document carries no credential material.

## Identifiers deliberately KEPT

None are secrets; all are load-bearing for a realistic round-trip test.

- **Box ids** — 237 in `state.json`, 14 in the journal tail. Id shape and length
  are what the state schema is keyed on; short synthetic ids would not exercise
  the real map.
- **Machine ids and bid prices** (`bid_history`, `entry_floor`,
  `launch_dph_anchor`, `p_alt`, `spend_by_box`) — real market/spend figures.
- **Instance label** on journal rows.
- Wall-clock unix timestamps throughout.

No SSH hostnames, ports or IPs appear anywhere in either file.

## Known oddity preserved on purpose

`intents` contains a key **`"9"`** (`kind: destroy`, `reason:
guard_zombie_destroy`). That is not an instance id — it is a **test fixture id
that leaked into the live daemon's state**, the incident documented in
`tools/vast/conftest.py`'s module docstring (the reason the autouse
`FLEETD_SOCK` isolation fixture exists). It is left in place: it is genuine
production data, and a state loader that chokes on a non-numeric-width box id
would be a real regression worth catching.

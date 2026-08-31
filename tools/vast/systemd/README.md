# systemd user units for the vast lane

`./install.sh` installs every `*.in` template here into
`~/.config/systemd/user/`, substituting `@DS_ROOT@` for the repo root. No sudo —
these are `systemctl --user` units.

```sh
tools/vast/systemd/install.sh            # install + start
tools/vast/systemd/install.sh --status   # what is installed, and when it last ran
tools/vast/systemd/install.sh --uninstall
```

## Why the units are in git at all

They were not, and that is how the problem below happened.

`herdd-reaper.{service,timer}` and the fleetd unit were hand-written straight
into `~/.config/systemd/user/`. Nothing in the repo said they should exist, so
nothing could notice when one of them didn't. `hostfacts.py ingest` was never
scheduled by anybody: it last ran **2026-08-07**, and by 2026-08-24 the store
held 202 box-written records of which **3** were still resolvable to a machine.
A missing timer is invisible in a way a failing one is not.

A committed unit cannot hardcode `/home/<user>/…` (CLAUDE.md), hence templates
plus a substituting installer rather than the unit files themselves.

## What is here

| unit | cadence | what it does |
|---|---|---|
| `herdd-hostfacts-ingest` | hourly | promotes instance-keyed host facts to `hostfacts/by-machine/` |
| `herdd-ckpt-retention` | daily 04:17 | REPORTS what the B2 intermediate-checkpoint sweep would free. Never deletes. |

**Not** here, and still hand-installed: `herdd-reaper` and `fleetd`. Moving
those is a separate change — the reaper destroys boxes, so re-pointing its unit
file is not something to do in passing.

## The ingest timer

`hostfacts ingest` copies a record from `jobs/nodes/<IID>/hostfacts/` to
`hostfacts/by-machine/<MACHINE>/`, so per-machine questions ("have I rented this
host before, was it slow?") can be answered at all. It is a **promotion**: the
original is never moved or deleted, an unresolvable record is left exactly where
it is, and a partial run is picked up by the next one. That is what makes it
safe unattended.

Hourly rather than the reaper's 15 minutes, deliberately. It used to be racing
box destruction — the instance→machine mapping lives only in vast's API and
disappears with the box — but `vastlib/core/machine_ledger.py` now writes that
mapping down from fleetd's 45-second tick, so ingest is no longer the thing
standing between a measurement and its attribution. Nothing about it is
time-critical any more.

If the timer is disabled, nothing breaks and nothing is lost: records stay
readable by instance and simply do not aggregate per machine until it runs.

A sweep is **~433 s** measured 2026-08-25 and grows with the store, which is why
the service bounds itself at `TimeoutStartSec=45min` and the timer counts its
hour from **completion** (`OnUnitInactiveSec`). Both are load-bearing: a
`Type=oneshot` unit defaults to an infinite start timeout, and a unit stuck in
`activating` can never be re-triggered — so one wedged run would silently end
every later one. `test_systemd_units.py` pins both.

## The checkpoint-retention report

`ckpt_retention.py` sweeps INTERMEDIATE training checkpoints under
`jobs/<JOB_ID>/checkpoints/`. It never touches `checkpoints/<RUN_NAME>/`, the
published final adapters — that distinction is the whole safety story and
`_assert_sweepable` fails closed on it.

**The timer runs `plan`, a dry run, and never `--apply`.** The asymmetry with
the reaper is deliberate: `reap` destroys rented boxes, which we can re-rent;
this deletes B2 objects, which is irreversible. Turning it into an auto-apply
should be an edit to the template, not a flag someone sets by accident. To
actually reclaim, read the report and run the sweep by hand.

Two things make the report worth reading rather than automating away. The
bucket was **614.9 GiB** when this landed with nothing ever having deleted any
of it, and 77% of that sits behind gate 5 (`CHECKPOINTS_PRUNED.json`), which
now needs `--sweep-box-pruned` *plus* an explicit
`--keep-first/--keep-last/--keep-stride` skeleton. The skeleton is not
ceremony: the early checkpoints are the dose curve, which is what showed v4's
echo collapse was trained-in rather than intrinsic.

It is also **slow** — one rclone pass over ~100k objects, tens of minutes,
which is why the unit carries a 45-minute deadline rather than trusting
`Type=oneshot`'s infinite default.

# herdd

Herd your GPU boxes. **herdd** is a control CLI and an always-on
fleet-supervision daemon (`fleetd`) for renting and running GPU workloads on
[vast.ai](https://vast.ai) — spot-bid aware, budget-capped, and built for
unattended operation.

**Status: alpha.** Battle-used, not battle-polished. This code has months of
heavy daily use behind it — training runs, eval sweeps, and model-serving
sessions across many rented spot boxes — but it was just extracted from a
private research monorepo, so expect rough edges, generic placeholder names
where private infrastructure used to be, and docs that are still being
rewritten.

## Background

herdd grew inside an ML research project that needed a lot of GPU time on a
budget: LoRA finetunes on rented 80 GB+ cards, vLLM serving for large
sampling runs, and multi-day eval sweeps — all on vast.ai spot instances,
where boxes get outbid mid-run, hosts vanish, and storage bills on
*allocated* disk whether you write a byte or not. Running that unattended
(much of the time driven by AI coding agents rather than a human at a
terminal) is what forced everything here into existence:

- **Spot economics** — a bid ladder that re-bids evicted instances, budget
  ceilings enforced by the daemon (a breach parks the box and alarms, never
  silently destroys work), and a reaper timer that destroys idle parked
  boxes before storage costs eat the spot savings.
- **Unattended supervision** — `fleetd` runs as a systemd user unit and owns
  all box babysitting: watching, re-bidding, parking, destroying, and
  journaling every decision. Nobody (human or agent) hand-polls a rental.
- **Eviction-tolerant jobs** — a self-contained bundle format
  (`herdd job …`) that ships code, environment, and launch plan to the
  box, checkpoints durable state to B2 as it runs, and resumes rather than
  restarts when a spot box is lost.
- **Boot observability** — diagnosing a box without SSH (`boxstate.py`),
  host selection from measured GPU rates, and $0 local rehearsal of job
  bundles inside the box image before renting anything.

The extraction kept the code and its ~7,500 tests; the private project's
names, hosts, and buckets were replaced with placeholders.

## What's here

- `tools/vast/herdd.py` — the CLI: search/launch/train/serve/ls/ssh/tunnel/
  park/resume/destroy/reap, a self-contained jobs lane (`herdd job …`), and
  fleet control (`herdd fleet watch/pause/park/destroy`).
- `tools/vast/fleetd.py` — the supervision daemon (systemd user unit): watches
  every box you rent, enforces budget ceilings, re-bids spot instances, parks
  or destroys idle boxes, and journals every decision.
- `tools/vast/vastlib/` — the engine: market search, launch planning, box
  lifecycle, B2/rclone storage lanes, jobs-v2 bundles, supervision policy.
- `tools/vast/onstart/` — box-side boot scripts (serve/train/jobs lanes).
- `tools/vast/systemd/` — installable timer units (reaper, retention).

## Quick start

```sh
python3 -m venv .venv && .venv/bin/pip install -e ".[test]"
export VAST_API_KEY=...   # or put it in .env
python3 tools/vast/herdd.py search --gpu RTX_5090
python3 tools/vast/herdd.py ls
```

Tests (no network, no GPU needed):

```sh
.venv/bin/python -m pytest tools/vast -q
```

## Configuration

Defaults live in `tools/vast/herdd.yaml`; personal overrides go in
`~/.config/herdd/herdd.yaml`. Secrets (API keys, registry tokens, B2
credentials) are read from the environment / a git-ignored `.env` — nothing in
this tree embeds one.

## License

MIT — see `LICENSE`.

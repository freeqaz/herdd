"""vastlib.storage — durable state that outlives a box: B2 objects and the local cache.

Why this layer exists
---------------------
The rclone/B2 seam was the third-largest fan-in cluster (51 inbound calls) and
it is the only thing standing between a destroyed box and lost work — every
park, every teardown, every run-event write goes through it. The sqlite
dash-cache is the other durable store, and the mapping flagged it as cleanly
separable today with a frozen argv contract the dashboard TypeScript spawns
directly.

Planned modules (plan §5)
-------------------------
  b2.py         the rclone seam, ensure-remote, rcat.
  dashcache.py  the sqlite infra-metadata cache, verbatim (cluster C15). Its
                argv literal is a FROZEN contract — four dashboard spawn sites
                pass `dash-cache ...` and none of them are in this repo.

What is deliberately NOT here
-----------------------------
* **The run-event log's schema.** Event bodies are built by `supervise.journal`
  and by `runmeta` (Zone S); this module moves bytes, it does not decide what
  the bytes say. Every event schema is a B2 wire contract and stays
  byte-identical across the port.
* No credential minting — that is `launch.spec` (mint) and `core.config`
  (discovery). This module consumes keys, it never creates them.
* No checkpoint retention policy: `ckpt_retention.py` is absorbed into
  `jobs/`, where the thing being retained actually lives.

Provenance: skeleton created 2026-08-16, plan §8 step 1. Contents arrive in
step 3 as verbatim moves from `herdd.py`.
"""

from __future__ import annotations

"""vastlib.cli._docs — the runbook pointers every `--help` epilog prints.

Why this module exists
----------------------
Every command's help page ends with a `docs:` block listing the runbooks for
that command, built by `_add_cmd(sub, name, help, *docs)`. The strings are
therefore **printed output**, which makes them inputs to the CLI-surface byte
diff (plan §4/§8): a moved comma changes a help page. In `herdd.py` thirteen
of them are module-level constants and the fourteenth — `DOC_WORKFLOW` — is a
local inside `main()`, which is an accident of when the `workflow` group was
added, not a design. Collecting all fourteen here gives the command modules one
import instead of fourteen literals, and gives the diff test one place to look
(cli-surface.json hazard MED-H7).

What is deliberately NOT here
-----------------------------
* `_docs_epilog` / `_add_cmd` — the epilog *renderer* and the `add_parser`
  factory are parser plumbing and belong with the composition root
  (`cli/_args.py` + `cli/main.py`), not with the data they format.
* Any doc pointer that is not printed by argparse. This module is the help
  text, not a link index; `docs/README.md` is the index.
* Any interpolation. These are frozen literals — every one is copied verbatim,
  including the em dashes and the backticks.

Provenance: verbatim move from `tools/vast/herdd.py` (plan §8 step 6,
2026-08-16). `DOC_README`..`DOC_NOTIFY` were module-level; `DOC_WORKFLOW` was a
`main()` local. Step 6 is ADD-ONLY at this commit: `herdd.py` keeps its own
copies and the byte diff is what proves the two sets equal.
"""

from __future__ import annotations

# Repo-relative doc pointers surfaced in each subcommand's --help epilog, so an
# agent mid-run can jump straight to the runbook for the command that just
# misbehaved. Paths are relative to the upstream-monorepo repo root.
# moved-from: herdd.DOC_README
DOC_README = "tools/vast/README.md (commands, .env setup, teardown & cost model)"
# moved-from: herdd.DOC_TRAINING
DOC_TRAINING = "tools/vast/TRAINING.md (training runbook — use `herdd train`)"
# moved-from: herdd.DOC_EVALS
DOC_EVALS = "tools/vast/EVALS_RUNBOOK.md (vLLM serving / co-tenant eval boxes)"
# moved-from: herdd.DOC_DEBUG
DOC_DEBUG = "tools/vast/DEBUG_BOX.md (post-FAILED SSH debug-hold)"
# moved-from: herdd.DOC_SUPERVISE
DOC_SUPERVISE = "tools/vast/SUPERVISE_DESIGN.md (relaunch policy, budgets, event contract)"
# moved-from: herdd.DOC_JOBS
DOC_JOBS = "tools/vast/JOBS_DESIGN.md (B2-mediated job submission — submit/status/pull)"
# moved-from: herdd.DOC_SKILL
DOC_SKILL = ".claude/skills/herdd/SKILL.md (agent quick-start, workflows, footguns)"
# moved-from: herdd.DOC_SKILL_RUNS
DOC_SKILL_RUNS = ".claude/skills/vast-runs/SKILL.md (runs/ event-log schema + hazards)"
# moved-from: herdd.DOC_SKILL_IMAGE
DOC_SKILL_IMAGE = ".claude/skills/push-train-image/SKILL.md (R2 registry image bake/publish/login)"
# moved-from: herdd.DOC_DASH_V5
DOC_DASH_V5 = ("tools/vast/dashboard/DESIGN_V5_ADMIN.md (/admin snapshot cache "
               "— read-only contract, table shapes, security clauses)")
# moved-from: herdd.DOC_FLEETD
DOC_FLEETD = ("tools/vast/FLEETD_DESIGN.md (the daemon: tick, profiles, watch "
              "lifecycle, journal + client surface)")
# moved-from: herdd.DOC_AUTOBID
DOC_AUTOBID = ("tools/vast/AUTOBID_DESIGN.md (bid ladder, the self-floor guard "
               "and its echo window, refusal + replacement rungs)")
# moved-from: herdd.DOC_FLEET_REVIEW
DOC_FLEET_REVIEW = ("tools/vast/FLEET_REVIEW_2026-08-14.md (the review this "
                    "report automates — what each aggregate is for)")
# moved-from: herdd.DOC_NOTIFY
DOC_NOTIFY = ("tools/vast/NOTIFY_DESIGN.md (vast's notification channel: the "
              "hidden pollable inbox, its structured outbid rows incl. the "
              "displacing new_min_bid, and why the poll is evidence-only)")

# Added 2026-08-27 (not part of the step-6 move): the agent cookbook — common
# operations, one command each, contract-checked by test_docs_contract.py.
DOC_OPERATIONS = ("tools/vast/OPERATIONS.md (common operations, one command "
                  "each — the agent cookbook)")

# The fourteenth. In `herdd.py` this one is a `main()` LOCAL (defined right
# above the `workflow` parser block) rather than a module-level constant —
# hoisting it here is the only non-textual change in the move, and it changes
# no printed byte.
# moved-from: herdd.DOC_WORKFLOW
DOC_WORKFLOW = ("docs/plans/herdd-autonomous-training-eval-roadmap.md "
                "(M2-T3: workflow CLI + reconciler)")

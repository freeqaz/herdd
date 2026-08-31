"""vastlib.cli._runsets — `runsets/<name>/config.yaml`, read the way `herdd` reads it.

Why this exists
---------------
`cmd_train` resolves per-runset launch defaults (the `env:` block) and per-runset
spot policy (the `spot:` block) out of `tools/vast/runsets/<runset>/config.yaml`.
Two helpers do that, and neither had a home:

* `core/config.py` **declined them explicitly** (its "deliberately NOT here"
  section): they parse NESTED yaml through `jobmeta._parse_job_yaml`, not the
  one-level `_parse_simple_yaml` that `herdd.yaml` uses, and they resolve the
  `runsets/` directory through `herdd`'s OWN module-global `_HERE` — a
  different global from `core.config._HERE`, and one the suite monkeypatches.
* `_load_runset_spot_config` is `cli-surface.json` **H8**: it is reachable from
  no `cmd_*` at all (`cmd_train` calls `_load_runset_config` and digs the
  `spot:` block out itself), yet six test asserts call it directly
  (`test_runset_env_defaults.py:90,96`, `test_lifecycle.py:317,322,335,349`).
  Deleting it as dead would have taken those with it; leaving it in the flat
  file would have left it with no rename-table entry when `herdd.py` is
  thinned. It lands here, next to the function it is a thin extraction of.

They sit in `cli/` and not below it because their only caller is `cli/train.py`
and their subject is a CLI-facing config file — the same placement argument
`_ls_render.py` makes for the `ls` renderers.

The `_HERE` contract (the whole reason this module needs a docstring)
--------------------------------------------------------------------
`herdd._HERE` is `dirname(abspath(__file__))` = `tools/vast`, and the flat
helper joins `_HERE/runsets/<runset>/config.yaml`. This module sits three
levels deeper, so the identical expression would resolve to
`tools/vast/vastlib/cli/runsets/...` — a directory that does not exist, and the
failure mode is SILENT: both helpers return `{}` for an absent file, so every
runset would quietly lose its `env:` defaults and its `spot:` policy and the
launch would proceed with the CLI defaults. The depth is therefore hoisted into
the `_HERE` module constant below and pinned by
`test_vastlib_cli_helpers.py::test_runsets_here_is_tools_vast`, exactly the way
`core.config._HERE` and `jobs.bundle.TOOLS_VAST_DIR` are.

Keeping it a module-level constant (rather than computing it inside the
function) is also what preserves the test idiom: the suite does
`monkeypatch.setattr(vc, "_HERE", str(tmp_path))` and then asserts on the parse,
so `_load_runset_config` must read the module global at CALL time.

What is deliberately NOT here
-----------------------------
* **`_runset_env_defaults`** — the pure validator/coercer that turns the parsed
  `env:` block into `KEY=VALUE` strings. It is `cmd_train`'s, it raises for the
  operator, and it is not a file reader; it moves with `cli/train.py`.
* **No second yaml parser.** `jobmeta._parse_job_yaml` is used precisely so an
  `env:` block parses identically under PyYAML and under the one-level fallback
  that `job-config.yaml` already relies on. Do not "upgrade" this to PyYAML
  directly — the fallback is the contract on a box without PyYAML.
* **No policy.** Both helpers are advisory by design: an absent or unparseable
  file returns `{}` and the launch proceeds exactly as it does today. The
  refusals live in `cmd_train`.

Provenance: verbatim-with-types move from `tools/vast/herdd.py`, plan §8
step 6 (`cli/`) of `docs/plans/vast-tooling-refactor-v2.md`. Step 6 is still
ADD-ONLY for `herdd.py`: the flat copies stay live until 6d, and the six
direct test sites repoint here at the test-migration step.
"""

from __future__ import annotations

import os
from typing import Any

import jobmeta

# `herdd._HERE` is `tools/vast`; this module is three directories deeper.
# Hoisted to a module constant for the same reason `core.config._HERE` and
# `jobs.bundle.TOOLS_VAST_DIR` are — and additionally because the test suite
# monkeypatches this attribute to point the loader at a tmp tree.
# moved-from: herdd._HERE
_HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# moved-from: herdd._load_runset_config
def _load_runset_config(runset: str) -> dict[str, Any]:
    """Full parsed runsets/<runset>/config.yaml — the file carries both the
    'spot:' block (SPOT_DESIGN §3.4) and a generic 'env:' block of per-runset
    launch-env defaults (see _runset_env_defaults). Reuses jobmeta's
    nested-YAML fallback (_parse_job_yaml: PyYAML if installed, else the same
    one-level `key: value` nesting job-config.yaml's env:/needs: blocks use) so
    an env: block parses under BOTH and there's no second parser to maintain.
    {} when config.yaml is absent or fails to parse — advisory, never blocks a
    launch (absent file = today's behavior exactly)."""
    path = os.path.join(_HERE, "runsets", runset, "config.yaml")
    if not os.path.isfile(path):
        return {}
    try:
        data = jobmeta._parse_job_yaml(open(path).read())
    except (jobmeta.JobmetaError, OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


# moved-from: herdd._load_runset_spot_config
def _load_runset_spot_config(runset: str) -> dict[str, Any]:
    """The 'spot:' sub-block of runsets/<runset>/config.yaml (SPOT_DESIGN §3.4):
    max_bid_mult, defend_at, rescue_wait_s, ckpt_interval_s, budget_usd — all
    optional, and cmd_train's CLI flags override every key. Thin extraction from
    _load_runset_config; {} when the file/block is absent (advisory)."""
    spot = _load_runset_config(runset).get("spot")
    return spot if isinstance(spot, dict) else {}

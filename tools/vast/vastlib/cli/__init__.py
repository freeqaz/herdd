"""vastlib.cli — the composition root: argv in, exit code out, and nothing below it.

Why this layer exists
---------------------
`main()` was 1,137 lines of hand-rolled parser wiring for 29 top-level commands
(69 command nodes once the four nested groups are expanded) and 473 flags. Every
command's arguments were read off a bare `argparse.Namespace`, which is why the
file is ~1% typed: there was nothing to annotate. Giving each command its own
module turns `main()` into a registry loop and gives the type checker a surface.

THE CONVENTIONS (this docstring is the contract every `cli/` module follows)
---------------------------------------------------------------------------
1. **One module per top-level command**, named after the command with dashes
   folded to underscores (`dash-cache` -> `dash_cache.py`). The four groups —
   `job`, `notify`, `fleet`, `workflow` — are subpackages whose `__init__.py`
   owns the group parser and whose submodules own one subcommand each.

2. **Every top-level command module exports exactly two public callables**::

       def add_parser(sub: _SubParsersAction, add_cmd: AddCmd) -> Any: ...
       def run(a: argparse.Namespace) -> None: ...

   `add_cmd` is `_args._add_cmd`, handed down by the composition root — the
   injection shape the flat file already used for its `add_fleet_parser` /
   `add_notify_parser` group builders, generalized to every command so the
   registry loop has one call shape. `add_parser` builds this command's parser
   and ends with `p.set_defaults(func=<the run callable>)`: the flat
   `a.func(a)` dispatch is preserved verbatim at the seam (plan §5), so `func`
   is set by the command module, never by `main`. `run` is the ported `cmd_*`
   body; where the body already landed in a lower ring (20 of 69 did —
   `cmd_stop`, `cmd_reap`, `cmd_dash_cache`, the whole `job` view/control set)
   the module is an argparse-only shim and `run` is the lower ring's function
   reached by MODULE ATTRIBUTE.

   `add_parser` may return the parser when a caller needs a handle for a
   post-hoc pass (`cli/job` uses that internally to hang `--local` on seven of
   its subparsers in one loop); the registry ignores the value.

3. **Runtime-resolved defaults are resolved by the module that prints them.**
   `main()` calls `load_env()` and nothing else before building parsers. The
   one flag whose default comes from `herdd.yaml` at runtime — `--image` on
   `launch` and on `supervise` — reads it through `cli/launch.default_image()`,
   so the value and the help text that quotes it can never disagree, and the
   registry keeps a single two-argument call shape (cli-surface.json hazard
   H5: this default is environment-dependent, so the CLI-surface diff must pin
   `herdd.yaml` on both arms). A `cli/` module must otherwise stay
   import-cheap: importing it may not call the API, read `.env`, or touch the
   network — every such read happens inside `add_parser` or `run`.

4. **Cross-module calls go through the module attribute**, never
   `from x import fn`::

       from vastlib.core import api
       api.request("GET", ...)          # patchable — tests steer this attribute

   This is not style. `monkeypatch.setattr(vastlib.core.api, "request_soft", …)`
   steers a module attribute; a `from … import` binds a second name that the
   patch never reaches (plan §8, porting mechanic (b)).

5. **Every ported def/class/constant carries `# moved-from: herdd.<name>`**
   on the line above it. `.port_manifests/gen_rename_table.py` regenerates the
   old-attr -> new-home table from those markers and 72 seam sites read it.

6. **Help text is byte-frozen.** Prog names, flag strings, defaults, help
   strings, subcommand ORDER, aliases (`stop (park)`, `start (resume)` — the
   only two in the surface) and the mutually-exclusive groups all reproduce
   exactly; the wave's fixture test diffs this parser tree against the flat
   one while both are alive. Two consequences worth stating:
     * Help text that f-string-interpolates a constant must interpolate the
       VASTLIB copy of that constant (`config._BOOT_KNOB_DEFAULTS`,
       `dashcache.DASH_SECTIONS`, `reap.REAP_IDLE_H_DEFAULT`, …). Several of
       those exist in both the flat file and vastlib during the add-only wave;
       importing the wrong one renders identically today and drifts silently
       later (cli-surface.json hazard H4).
     * A mutually-exclusive group is invisible to `--help`. Carry the group
       object, not the three flags it holds, or argparse quietly stops
       rejecting the pair (hazard H6).

7. **Shared parser plumbing lives once, and here is where:**
     `_docs.py`   the 15 `DOC_*` runbook pointers rendered into every epilog.
     `_args.py`   `_docs_epilog`, the `_add_cmd` factory + its `AddCmd`
                  Protocol, and `_add_salvage_args`.
     `search.py`  `add_search_filters` — 16 flags shared by `search`, `launch`
                  and `supervise`, read back by argparse DEST name. It lives
                  with its primary command rather than in `_args.py` because
                  it is a command's flag block, not composition plumbing.
     `launch.py`  `default_image()` — see (3).
     `_ls_render.py`  `_gather_ls_data` / `_market_map` / `_stale_image_ids`,
                  shared by `ls` and `dash-cache`.

What is deliberately NOT here
-----------------------------
* **Anything importable from below.** Nothing in `vastlib` may import `cli` —
  it is the top of the DAG, and import-linter enforces exactly that. Where a
  lower ring genuinely needs a `cli` read it takes it by INJECTION instead:
  `storage.dashcache.DashDeps` is the worked example (`gather_ls_data`,
  `job_cell`, `active_job_states` are `cli/ls` values handed down by
  `cli/dash_cache.py`).
* No policy and no I/O of its own: a command module parses, calls one function
  in the ring below, and renders. The render atoms are `core.fmt`.
* No new commands, no changed flags, no changed help text. `tools/vast/herdd.py`
  stays a thin script at its exact path and the CLI surface is byte-compared
  old-vs-new (plan §8 step 6) — ~30 callers of the literal path, the reaper
  systemd unit, four dashboard spawn sites with a frozen `dash-cache` argv,
  and 550 markdown references all depend on that.

Provenance: skeleton created 2026-08-16 (plan §8 step 1); conventions + the
registry root + the plain top-level commands ported 2026-08-16 at herdd.py
rev 7a177e2a, behavior-preserving. Manifest: `.port_manifests/cli-surface.json`.
"""

from __future__ import annotations

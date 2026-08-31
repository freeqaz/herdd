"""vastlib.cli._args — the parser plumbing every command module shares.

Why this module exists
----------------------
Three pieces of `main()`'s 1,123 lines are not any one command's business:

* `_docs_epilog(*docs)` renders the `docs:` block that ends every help page.
  It is printed output, so it is an input to the CLI-surface byte diff
  (plan §4/§8) — one renderer, not one per command module.
* `_add_cmd(sub, name, help, *docs, aliases=())` is the `sub.add_parser`
  factory that pairs that epilog with `RawDescriptionHelpFormatter`. The four
  group builders (`job` / `fleet` / `workflow` / `notify`) already take it by
  INJECTION in the flat file, which is the shape plan §5's registry loop wants:
  the composition root owns the factory and hands the same one to every module.
  `AddCmd` is that injected callable's type.
* `_add_salvage_args(p)` is a flag BLOCK shared by `fleet watch` and
  `job supervise`, so a daemon-registered watch and an inline supervisor take
  the same knobs. Its help interpolates the salvage constants, which is why it
  reads them from `vastlib.boxes.salvage` — the byte diff against the flat
  parser is what proves the ported constants equal the flat ones.

What is deliberately NOT here
-----------------------------
* The DOC_* strings. They are data, and they live in `cli/_docs.py`.
* `add_search_filters` — the 16-flag block shared by `search` and `launch`.
  It belongs to those two command modules' owner; it is listed here only so
  the next reader knows this module is not where it went.
* Any per-command flag. A flag read by exactly one command belongs in that
  command's module, next to the code that reads it.
* An `Args` dataclass base. Plan §5 gives each command module its own
  `Args.from_ns`; there is no shared field set to factor out, and inventing a
  base class would be a second representation of `argparse.Namespace`.

Provenance: moved from `tools/vast/herdd.py` (`main()`'s parser helpers and
the salvage flag block), plan §8 step 6, 2026-08-16. Step 6 is ADD-ONLY at this
commit: `herdd.py` keeps its own copies and the CLI-surface byte diff is what
proves the two parser trees identical.
"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING, Any, Protocol

from vastlib.boxes import salvage

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence


class AddCmd(Protocol):
    """The `_add_cmd` factory as the group builders receive it.

    Spelled as a Protocol rather than a `Callable[..., ArgumentParser]` alias
    so the `aliases=` keyword survives type checking at the call sites that use
    it (`ls`, `job`), instead of being erased by an ellipsis.
    """

    def __call__(self, sub: Any, name: str, help_: str,  # noqa: ANN401 — argparse's _SubParsersAction is private and generic over the parser class
                 *docs: str,
                 aliases: Sequence[str] = ()) -> argparse.ArgumentParser: ...


# moved-from: herdd._docs_epilog
def _docs_epilog(*docs: str) -> str:
    return "docs:\n" + "\n".join("  " + d for d in docs)


# moved-from: herdd._add_cmd
def _add_cmd(sub: Any, name: str, help_: str,  # noqa: ANN401 — see AddCmd
             *docs: str,
             aliases: Sequence[str] = ()) -> argparse.ArgumentParser:
    """sub.add_parser() with a 'docs:' epilog listing the runbooks for this command."""
    parser = sub.add_parser(name, help=help_, epilog=_docs_epilog(*docs),
                            aliases=list(aliases),
                            formatter_class=argparse.RawDescriptionHelpFormatter)
    return parser  # type: ignore[no-any-return]  # argparse's private action is untyped here


# moved-from: herdd._add_salvage_args
def _add_salvage_args(p: argparse.ArgumentParser) -> None:
    """The salvage knobs, shared by `fleet watch` and `job supervise` so a
    daemon-registered watch and an inline supervisor take the same flags."""
    p.add_argument("--no-salvage", dest="salvage", action="store_false",
                   default=None,
                   help="do NOT attempt instance->instance disk salvage of an "
                        "evicted box. Salvage is ON by default: it enters no "
                        "GPU contract on either side (vast serves filesystem "
                        "access to an `exited` instance) and costs bandwidth on "
                        f"~{salvage.SALVAGE_KEEP_N} checkpoint (~0.98 GB each)")
    p.add_argument("--salvage-keep-n", dest="salvage_keep_n", type=int,
                   default=None, metavar="N",
                   help=f"salvage the newest N checkpoints per job from the "
                        f"dead disk (default {salvage.SALVAGE_KEEP_N})")
    p.add_argument("--salvage-max-gb", dest="salvage_max_gb", type=float,
                   default=None, metavar="GB",
                   help=f"refuse to initiate a salvage transfer larger than "
                        f"this (default {salvage.SALVAGE_MAX_GB:g}; a checkpoint "
                        f"is ~0.98 GB, so the fuse is really against a "
                        f"misparsed survey)")

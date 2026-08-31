"""`vastlib.cli.notify` — the FOURTH dispatcher, and the exit code that is a contract.

Why this file exists
--------------------
Plan §5 names three nested dispatchers (`job` / `fleet` / `workflow`);
`notify` landed after the plan was written and is a fourth
(`.port_manifests/cli-surface.json` hazard H2). It is entirely cli-only — three
handlers, one shared GET and one exit code, none of them ported before step 6 —
so nothing in the existing suite covers it on either side of the move.

What it pins:

1. **The help tree.** `notify`'s three subparsers each carry an explicit
   `epilog=_docs_epilog(...)` and `RawDescriptionHelpFormatter`, which a naive
   port drops silently — the flags still work and the help page loses its docs
   block. This started as a two-tree diff (build BOTH trees in one process from
   the same injected `_add_cmd`, compare `format_help()`); **the flat tree is
   gone at plan §8 step 6d** — `herdd.add_notify_parser` is now an identity
   re-export of `vastlib.cli.notify.add_parser`, so both builders produced the
   SAME parser and the diff compared a tree with itself. Deleted. The frozen
   reference is now `testfixtures/cli_surface_flat_*.json` via
   `test_vastlib_cli_surface.py`, which captures `notify inbox|types|webhooks`
   from the pre-thinning flat tree; what stays here is the characterization —
   the subcommand order, the epilog, the formatter class.
2. **`NOTIFY_GONE_RC = 3`.** The inbox endpoint is HIDDEN (commented out of
   vast's published OpenAPI spec), so its 404 is an EXPECTED end state a caller
   scripts against, not a bug. Pinned as the exit code the 404 path actually
   takes, and pinned as the launcher's re-exported object (post-6d the value
   comparison is `3 == 3`; the binding is what can still break).
3. **Read-only by construction.** `_notify_get` issues exactly one GET and
   never a write — in particular never a `seen_through_at` PUT (NOTIFY_DESIGN
   D3). Asserted on the METHOD argument, not on a comment.
4. **`--json` is the fixture-capture path.** It must emit the payload verbatim,
   rows unmodified and `--limit` NOT applied, or the evidence this command
   exists to collect is laundered on the way out.

What is deliberately NOT here
-----------------------------
* `tools/vast/notify.py`'s renderers and accessors. That flat leaf is shared
  with fleetd's poll tick and has its own coverage; here it is stubbed so the
  test speaks about the CLI.
* Any live HTTP. `core.api.request_soft` is patched BY MODULE ATTRIBUTE at
  every call site, which is also the proof that the port kept the
  module-attribute seam the test migration (plan §7.2) depends on.

Provenance: created 2026-08-16 alongside `vastlib/cli/notify/`, plan §8 step 6.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import notify  # noqa: E402
import herdd as v  # noqa: E402

from vastlib.cli import _args  # noqa: E402
from vastlib.cli import notify as cli_notify  # noqa: E402
from vastlib.cli.notify import _get  # noqa: E402
from vastlib.core import api  # noqa: E402


# --- parser tree ------------------------------------------------------------- #
def _ported_tree() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="herdd")
    sub = ap.add_subparsers(dest="cmd", required=True)
    cli_notify.add_parser(sub, _args._add_cmd)
    return ap


def _subparsers(p: argparse.ArgumentParser) -> dict[str, argparse.ArgumentParser]:
    for act in p._actions:
        if isinstance(act, argparse._SubParsersAction):
            return dict(act.choices)
    return {}


def test_the_launcher_re_exports_the_group_builder_rather_than_redefining_it() -> None:
    """Post-6d replacement for the four flat-vs-ported parser diffs.

    Those diffs built one tree through `herdd.add_notify_parser` and one
    through `vastlib.cli.notify.add_parser` and compared them. The launcher's
    binding is an identity re-export (its docstring, rule 1), so the two
    builders are one function and every diff was a tree against itself. What
    can still break is the binding: a second `add_notify_parser` body in the
    launcher would give `herdd.py notify --help` a different page from the
    one `test_vastlib_cli_surface.py`'s frozen fixture pins.
    """
    assert v.add_notify_parser is cli_notify.add_parser
    assert v._add_cmd is _args._add_cmd


def test_subcommand_order_is_the_flat_order() -> None:
    assert list(_subparsers(_subparsers(_ported_tree())["notify"])) == [
        "inbox", "types", "webhooks"]


def test_every_subcommand_keeps_its_docs_epilog() -> None:
    """The silent-loss failure: flags keep working, the docs block vanishes."""
    subs = _subparsers(_subparsers(_ported_tree())["notify"])
    for name, p in subs.items():
        assert (p.epilog or "").startswith("docs:\n  tools/vast/NOTIFY_DESIGN.md"), name
        assert p.formatter_class is argparse.RawDescriptionHelpFormatter, name


# --- dispatch ---------------------------------------------------------------- #
def test_group_dispatches_through_notifyfunc() -> None:
    a = _ported_tree().parse_args(["notify", "types"])
    assert a.func is cli_notify.run
    seen: list[argparse.Namespace] = []
    a.notifyfunc = lambda ns: seen.append(ns)
    a.func(a)
    assert seen == [a]


@pytest.mark.parametrize("name", ["inbox", "types", "webhooks"])
def test_each_subcommand_binds_its_own_modules_run(name: str) -> None:
    sub = _subparsers(_subparsers(_ported_tree())["notify"])[name]
    bound = sub.get_default("notifyfunc")
    assert bound is getattr(cli_notify, name).run
    assert bound.__module__ == f"vastlib.cli.notify.{name}"


# --- the shared GET and the exit code ---------------------------------------- #
def test_gone_rc_is_three_and_the_launcher_re_exports_it() -> None:
    assert _get.NOTIFY_GONE_RC == 3
    assert v.NOTIFY_GONE_RC is _get.NOTIFY_GONE_RC


def test_notify_get_is_one_read_only_request(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tuple[str, str]] = []

    def _fake(method: str, path: str, *a: Any, **k: Any) -> tuple[Any, Any, Any]:
        seen.append((method, path))
        return (True, {"rows": []}, None)

    monkeypatch.setattr(api, "request_soft", _fake)
    data, err = _get._notify_get(notify.INBOX_PATH)
    assert seen == [("GET", notify.INBOX_PATH)]     # never a PUT: D3
    assert data == {"rows": []} and err is None


def test_notify_get_returns_none_payload_on_failure(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api, "request_soft",
                        lambda *a, **k: (False, {"junk": 1}, "http 404"))
    data, err = _get._notify_get(notify.INBOX_PATH)
    assert data is None and err == "http 404"


def test_inbox_404_exits_with_the_gone_code(monkeypatch: pytest.MonkeyPatch,
                                            capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(api, "request_soft", lambda *a, **k: (False, None, "http 404"))
    monkeypatch.setattr(notify, "is_gone", lambda err: True)
    with pytest.raises(SystemExit) as e:
        cli_notify.inbox.run(argparse.Namespace(json=False, limit=0))
    assert e.value.code == _get.NOTIFY_GONE_RC
    assert "hidden endpoint gone" in capsys.readouterr().err


def test_inbox_other_error_is_the_generic_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api, "request_soft", lambda *a, **k: (False, None, "boom"))
    monkeypatch.setattr(notify, "is_gone", lambda err: False)
    with pytest.raises(SystemExit) as e:
        cli_notify.inbox.run(argparse.Namespace(json=False, limit=0))
    assert e.value.code == "error: boom"


def test_inbox_json_is_verbatim_and_ignores_limit(
        monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    payload = {"rows": [{"id": 2}, {"id": 1}], "extra": "kept"}
    monkeypatch.setattr(api, "request_soft", lambda *a, **k: (True, payload, None))
    monkeypatch.setattr(notify, "render_inbox",
                        lambda *a, **k: pytest.fail("--json must not render"))
    cli_notify.inbox.run(argparse.Namespace(json=True, limit=1))
    assert json.loads(capsys.readouterr().out) == payload


def test_inbox_table_passes_the_limit_through(
        monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    seen: dict[str, Any] = {}

    def _render(data: Any, now: float, limit: int = 0) -> str:
        seen.update(data=data, limit=limit)
        return "TABLE\n"

    monkeypatch.setattr(api, "request_soft", lambda *a, **k: (True, {"rows": []}, None))
    monkeypatch.setattr(notify, "render_inbox", _render)
    cli_notify.inbox.run(argparse.Namespace(json=False, limit=5))
    assert seen["limit"] == 5
    assert capsys.readouterr().out == "TABLE\n"


@pytest.mark.parametrize("name,path,renderer", [
    ("types", notify.TYPES_PATH, "render_types"),
    ("webhooks", notify.WEBHOOKS_PATH, "render_webhooks"),
])
def test_types_and_webhooks_read_their_own_path(
        monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
        name: str, path: str, renderer: str) -> None:
    seen: list[str] = []
    monkeypatch.setattr(api, "request_soft",
                        lambda m, p, *a, **k: (seen.append(p), (True, {}, None))[1])
    monkeypatch.setattr(notify, renderer, lambda data: f"{name}-table\n")
    getattr(cli_notify, name).run(argparse.Namespace(json=False))
    assert seen == [path]
    assert capsys.readouterr().out == f"{name}-table\n"


@pytest.mark.parametrize("name", ["types", "webhooks"])
def test_types_and_webhooks_exit_on_error(monkeypatch: pytest.MonkeyPatch,
                                          name: str) -> None:
    monkeypatch.setattr(api, "request_soft", lambda *a, **k: (False, None, "nope"))
    with pytest.raises(SystemExit) as e:
        getattr(cli_notify, name).run(argparse.Namespace(json=False))
    assert e.value.code == "error: nope"

"""`vastlib.cli.fleet` — the ported `fleet` group, held to the help tree and the seams.

Why this file exists
--------------------
Step 6's acceptance bar is HELP-TREE BYTE FIDELITY: every prog name, flag,
default, help string, subcommand order and formatter has to reproduce exactly,
because ~30 callers, a systemd unit, four dashboard spawn sites and ~550
markdown references bind this surface. While the flat `herdd.py` and the
package were both alive, that bar was checkable directly — build BOTH parser
trees in one process, from the same injected `_add_cmd`, and diff them. That
was `test_help_tree_is_byte_identical_to_the_flat_parser`, and it was the test
this file existed for. Plan §8 step 6d ended it: the launcher re-exports the
package's builder, so both trees came from one function. The bar did not move,
it changed instrument — `test_vastlib_cli_surface.py` diffs the live tree
against a CAPTURE of the flat one, frozen in
`testfixtures/cli_surface_flat_*.json` before the thinning, which is the only
form of that comparison still possible. What remains here is everything the
diff never covered:

1. **The dispatch chain.** `func` on the group and `fleetfunc` on each of the
   sixteen subcommands are what `a.func(a)` -> `a.fleetfunc(a)` walks. A parser
   tree can be byte-identical with every handler pointing at the wrong module.
2. **`_fmt_age`, ported to `core.fmt`.** It is a SECOND age formatter and it is
   NOT `_age_str`: it takes None, tops out at h+minutes, and does not clamp
   negatives. Was pinned against the flat copy across fifteen boundaries; since
   6d there is one copy, so what is pinned is the difference from `_age_str`.
3. **`_fleetd_script`, ported to `fleet.deploy`.** The one deliberately
   non-verbatim body in this port: `dirname(abspath(__file__))` names
   `tools/vast` in the flat file and `vastlib/fleet` here, and the failure mode
   is a `subprocess.call` on a path that does not exist — at the moment an
   operator is trying to fix the daemon. Was pinned equal to the flat
   resolution; now pinned to the property that resolution stood for — the path
   names the real Zone E entry script.
4. **`restart`'s refusal.** A restart mid-recovery duplicated a whole recovery
   chain on 2026-08-08 (~$0.9, two actors on one job). The refusal must exit 2
   WITHOUT running systemctl, and `--force` must not even read the state file.
5. **The module-attribute seam.** Every callee is reached as
   `client.<name>` / `fmt.<name>`, never `from … import <name>`, so
   `monkeypatch.setattr(vastlib.fleet.client, …)` is seen by the CLI. Each
   behavioral test below patches exactly that way, which is what proves it.

What is deliberately NOT here
-----------------------------
* No repoint of any existing test. `test_fleetd.py` and friends drive
  `fleetd`/`herdd` and still pass — since 6d those names reach this package.
* No daemon behavior, no socket. Every command here is a `_fleet_call_or_die`
  away from the daemon and that call is patched; what the daemon DOES with a
  watch is `fleet/daemon.py`'s tests.
* No assertion on the flat file's own behavior. A `herdd.` symbol appears
  here only as an oracle (pre-6d) or as a BINDING check (post-6d), never as the
  system under test — except `v.salvage`, which is still a distinct module.

Provenance: created 2026-08-16 alongside `vastlib/cli/fleet/`, plan §8 step 6.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import herdd as v  # noqa: E402

from vastlib.cli import _args  # noqa: E402
from vastlib.cli import fleet as cli_fleet  # noqa: E402
from vastlib.core import fmt  # noqa: E402
from vastlib.fleet import client  # noqa: E402
from vastlib.fleet import deploy as fleet_deploy  # noqa: E402


# --- parser tree ------------------------------------------------------------- #
# `_flat_tree()` built the same group through `v.add_fleet_parser(sub, v._add_cmd)`
# and lived here. At plan §8 step 6d both of those names became identity
# re-exports of `vastlib.cli.fleet.add_parser` / `vastlib.cli._args._add_cmd`,
# so it returned a tree built by the SAME builder as `_ported_tree()` and every
# diff against it compared a tree with itself. Deleted, with the four tests
# that consumed it; the frozen reference for the help tree is now
# `testfixtures/cli_surface_flat_*.json`, captured from the pre-thinning flat
# parser and diffed by `test_vastlib_cli_surface.py` (it carries all sixteen
# `fleet <sub>` cells).
def _ported_tree() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="herdd")
    sub = ap.add_subparsers(dest="cmd", required=True)
    cli_fleet.add_parser(sub, _args._add_cmd)
    return ap


def _subparsers(p: argparse.ArgumentParser) -> dict[str, argparse.ArgumentParser]:
    for act in p._actions:
        if isinstance(act, argparse._SubParsersAction):
            return dict(act.choices)
    return {}


# `_action_shape` / `_parser_shape` built the comparable dicts the deleted
# flat-vs-ported shape diffs consumed; they have no other caller.


def test_the_launcher_re_exports_the_group_builder_rather_than_redefining_it() -> None:
    """What is left of the four flat-vs-ported parser diffs (shape, order,
    per-subcommand shape, byte-identical help).

    All four resolved both sides through one builder after step 6d. The binding
    is the part that can still break: a second `add_fleet_parser` body in the
    launcher would give `herdd.py fleet --help` a page the frozen
    `cli_surface` fixture does not describe, and ~30 callers, a systemd unit and
    four dashboard spawn sites read that surface."""
    assert v.add_fleet_parser is cli_fleet.add_parser
    assert v._add_cmd is _args._add_cmd


def test_subcommand_order_is_the_frozen_order() -> None:
    """The order was asserted against the flat tree AND against this literal;
    the literal is the half that survives.

    `hosts` (2026-08-20) is the one entry that was not in the flat tree. It sits
    after `spend` because it is read like `spend` is — an operator asking the
    daemon what it has been doing — and before the journal/unit half."""
    assert list(_subparsers(_subparsers(_ported_tree())["fleet"])) == [
        "ping", "status", "ack", "watch", "unwatch", "pause", "park",
        "resume", "destroy", "spend", "hosts", "log", "report", "install",
        "deploy", "uninstall", "restart"]


def test_help_text_interpolates_the_same_constants() -> None:
    """H4: the salvage / replacement defaults are f-strung INTO help. The
    frozen `cli_surface` fixture proves the rendered strings; this names the
    objects, so a future edit that hardcodes a number fails here with a
    readable message instead of as an opaque fixture diff."""
    watch = _subparsers(_subparsers(_ported_tree())["fleet"])["watch"]
    helps = " ".join(a.help or "" for a in watch._actions)
    import bidpolicy

    from vastlib.boxes import salvage
    assert f"default {bidpolicy.MAX_REPLACEMENTS}" in helps
    assert f"default {bidpolicy.REPLACE_CEILING_MULT:g}" in helps
    assert f"default {bidpolicy.REPLACEMENT_RETENTION_H:g}h" in helps
    assert f"default {salvage.SALVAGE_KEEP_N}" in helps
    assert f"default {salvage.SALVAGE_MAX_GB:g}" in helps
    # Step 7 came: `tools/vast/salvage.py` is now a re-export shim over
    # `vastlib.boxes.salvage`, so `v.salvage.SALVAGE_KEEP_N` and the name above
    # are ONE object and comparing them proves nothing. What is still worth
    # asserting is that the help text was interpolated from the value the two
    # spellings now share — so pin the literals instead, and pin the collapse
    # itself with `is`, which is the invariant the shim exists to create.
    assert (salvage.SALVAGE_KEEP_N, salvage.SALVAGE_MAX_GB) == (1, 12.0)
    assert v.salvage.SALVAGE_KEEP_N is salvage.SALVAGE_KEEP_N
    assert v.salvage.SALVAGE_MAX_GB is salvage.SALVAGE_MAX_GB


# --- the dispatch chain ------------------------------------------------------ #
def test_group_dispatches_through_fleetfunc() -> None:
    ap = _ported_tree()
    a = ap.parse_args(["fleet", "ping"])
    assert a.func is cli_fleet.run
    seen: list[argparse.Namespace] = []
    a.fleetfunc = lambda ns: seen.append(ns)
    a.func(a)
    assert seen == [a]


@pytest.mark.parametrize("name,module", [
    ("ping", "ping"), ("status", "status"), ("ack", "ack"), ("watch", "watch"),
    ("unwatch", "unwatch"), ("pause", "pause"), ("park", "park"),
    ("resume", "resume"), ("destroy", "destroy"), ("spend", "spend"),
    ("log", "log"), ("report", "report"), ("install", "install"),
    ("deploy", "deploy"), ("uninstall", "uninstall"), ("restart", "restart"),
])
def test_each_subcommand_binds_its_own_modules_run(name: str, module: str) -> None:
    sub = _subparsers(_subparsers(_ported_tree())["fleet"])[name]
    bound = sub.get_default("fleetfunc")
    mod = getattr(cli_fleet, module)
    assert bound is mod.run
    assert bound.__module__ == f"vastlib.cli.fleet.{module}"


# --- the two helpers that moved BELOW cli ------------------------------------ #
# `test_fmt_age_matches_the_flat_copy` swept fifteen inputs through
# `fmt._fmt_age` and `v._fmt_age`. One body since step 6d; the test below keeps
# the part that was never parity — that `_fmt_age` is NOT `_age_str`.


def test_fmt_age_is_not_age_str() -> None:
    """They differ exactly where the port would be easiest to get wrong."""
    assert fmt._fmt_age(None) == "?"
    assert fmt._fmt_age(7265) == "2h01"
    assert fmt._age_str(7265) == "2h"


def test_fleetd_script_still_resolves_the_zone_e_path() -> None:
    """The re-anchor pin. `== v._fleetd_script()` was the oracle while the flat
    body existed; post-6d that name re-exports this function, so what remains is
    the property itself — the resolved path is the real Zone E entry script."""
    ported = fleet_deploy._fleetd_script()
    assert os.path.basename(ported) == "fleetd.py"
    assert os.path.isfile(ported)
    assert v._fleetd_script is fleet_deploy._fleetd_script


def test_naive_file_arithmetic_here_would_be_wrong() -> None:
    """The bug the re-anchor avoids: a verbatim body names a nonexistent file."""
    naive = os.path.join(os.path.dirname(os.path.abspath(fleet_deploy.__file__)),
                         "fleetd.py")
    assert naive != fleet_deploy._fleetd_script()
    assert not os.path.exists(naive)


# --- behavior at the seams --------------------------------------------------- #
def test_ping_reports_down_and_exits_1(monkeypatch: pytest.MonkeyPatch,
                                       capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(client, "fleet_request",
                        lambda *a, **k: (False, None, "nodaemon:x"))
    with pytest.raises(SystemExit) as e:
        cli_fleet.ping.run(argparse.Namespace())
    assert e.value.code == 1
    assert "fleetd: DOWN (nodaemon:x)" in capsys.readouterr().out


def test_ping_flags_version_skew(monkeypatch: pytest.MonkeyPatch,
                                 capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(client, "fleet_request",
                        lambda *a, **k: (True, {"rev": "deadbee"}, None))
    monkeypatch.setattr(client, "_git_rev_short", lambda: "cafef00")
    cli_fleet.ping.run(argparse.Namespace())
    out = capsys.readouterr().out
    assert "!! version skew" in out and "fleet deploy" in out


def test_status_renders_rows_footnotes_and_alarms(
        monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    payload = {
        "tick_age_s": 3, "dry_run": False, "spend_total_usd": 1.5,
        "adopt_default_budget_usd": 5.0,
        "rows": [{"iid": "47", "target": "47", "profile": "jobs",
                  "state": "running", "ceiling_spend_usd": 4.9,
                  "budget_usd": 10.0, "remaining_usd": 5.1,
                  "ceiling_source": "default", "last_action": "tick"}],
        "retained": [{"iid": "48", "left_s": 7265, "est_cost_usd": 0.3,
                      "eviction_class": "outbid", "replacement_iid": "49"}],
        "alarm_records": [{"msg": "budget breach", "age_s": 120, "sticky": True,
                           "key": "k1"}],
    }
    monkeypatch.setattr(client, "_fleet_call_or_die", lambda *a, **k: payload)
    cli_fleet.status.run(argparse.Namespace(json=False))
    out = capsys.readouterr().out
    assert "$4.900" in out                       # the CEILING's spend, not the watch's
    assert "PROVISIONAL auto-adopt cap" in out
    assert "RETAINED 48" in out and "2h01 left" in out
    assert "NO keep label" in out
    assert "[LATCHED 2m ago] budget breach" in out
    assert "herdd fleet ack k1" in out


def test_status_watch_column_carries_the_handed_off_key(
        monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """A watch whose ladder replaced its box is filed under the ORIGINAL id;
    the mapping used to be a `**` footnote below the table, which an agent
    grepping rows never saw. It is the WATCH column now."""
    payload = {
        "tick_age_s": 3, "dry_run": False, "spend_total_usd": 0.1,
        "rows": [
            {"iid": "48671690", "target": "48670932", "profile": "jobs",
             "state": "watched", "spend_usd": 0.067, "budget_usd": 4.0,
             "remaining_usd": 3.933, "last_action": "tick"},
            {"iid": "47", "target": "47", "profile": "jobs",
             "state": "watched", "spend_usd": 0.0, "budget_usd": 1.0,
             "remaining_usd": 1.0, "last_action": "tick"},
        ],
    }
    monkeypatch.setattr(client, "_fleet_call_or_die", lambda *a, **k: payload)
    cli_fleet.status.run(argparse.Namespace(json=False))
    out = capsys.readouterr().out.splitlines()
    assert "WATCH" in out[1]
    handed = next(line for line in out if line.startswith("48671690"))
    assert "48670932" in handed
    same = next(line for line in out if line.startswith("47"))
    assert "48670932" not in same
    assert not any(line.startswith("**") and "CURRENT box" in line
                   for line in out)


def test_status_falls_back_to_string_alarms_on_an_older_daemon(
        capsys: pytest.CaptureFixture[str]) -> None:
    cli_fleet.status._print_fleet_alarms({"alarms": ["old style"]})
    assert capsys.readouterr().out == "!! old style\n"


# --- item 4: aged-ceiling collapse -------------------------------------------- #
# `ceiling_rows()` (vastlib/fleet/rows.py) carries no timestamp, so the
# collapse predicate is `last_verdict == "instance_gone"` and `live_boxes`
# empty — not a measured age. These pin that every such row collapses to the
# count line by default and reappears under --all, while a live ceiling and an
# orphan with a different verdict (e.g. a normal `drained` end) are untouched
# either way.
def _status_payload(ceilings: list[dict[str, Any]]) -> dict[str, Any]:
    return {"tick_age_s": 1, "dry_run": False, "spend_total_usd": 0,
            "adopt_default_budget_usd": 5.0, "rows": [], "retained": [],
            "alarm_records": [], "ceilings": ceilings}


_LIVE = {"ceiling_id": "c-live", "cap_usd": 5.0, "spend_usd": 1.0,
          "remaining_usd": 4.0, "source": "explicit", "epochs": 1,
          "last_verdict": "drained", "live_boxes": ["47"]}
_GONE = {"ceiling_id": "c-gone", "cap_usd": 10.0, "spend_usd": 0.0,
          "remaining_usd": 10.0, "source": "default", "epochs": 1,
          "last_verdict": "instance_gone", "live_boxes": []}
_DRAINED_ORPHAN = {"ceiling_id": "c-drained", "cap_usd": 2.0, "spend_usd": 0.5,
                    "remaining_usd": 1.5, "source": "explicit", "epochs": 1,
                    "last_verdict": "drained", "live_boxes": []}


def test_status_collapses_aged_instance_gone_ceilings_by_default(
        monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(client, "_fleet_call_or_die",
                        lambda *a, **k: _status_payload([_GONE]))
    cli_fleet.status.run(argparse.Namespace(json=False, all=False))
    out = capsys.readouterr().out
    assert "c-gone" not in out
    assert "durable ceilings with no live watch" not in out
    assert "+ 1 aged ceilings for boxes gone" in out
    assert "--all to list" in out


def test_status_all_shows_the_aged_rows_individually(
        monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(client, "_fleet_call_or_die",
                        lambda *a, **k: _status_payload([_GONE]))
    cli_fleet.status.run(argparse.Namespace(json=False, all=True))
    out = capsys.readouterr().out
    assert "c-gone" in out
    assert "durable ceilings with no live watch" in out
    assert "aged ceilings for boxes gone" not in out


def test_status_leaves_live_and_non_gone_orphan_ceilings_visible(
        monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(
        client, "_fleet_call_or_die",
        lambda *a, **k: _status_payload([_LIVE, _DRAINED_ORPHAN, _GONE]))
    cli_fleet.status.run(argparse.Namespace(json=False, all=False))
    out = capsys.readouterr().out
    # _LIVE has a live watch, so it never appears in the ceilings section at all.
    assert "c-live" not in out
    assert "c-drained" in out                    # orphaned but not confirmed-gone
    assert "c-gone" not in out
    assert "+ 1 aged ceilings for boxes gone" in out


def test_status_prints_no_count_line_when_nothing_is_aged(
        monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(client, "_fleet_call_or_die",
                        lambda *a, **k: _status_payload([_DRAINED_ORPHAN]))
    cli_fleet.status.run(argparse.Namespace(json=False, all=False))
    out = capsys.readouterr().out
    assert "c-drained" in out
    assert "aged ceilings for boxes gone" not in out


def test_ack_refuses_without_key_or_all() -> None:
    with pytest.raises(SystemExit) as e:
        cli_fleet.ack.run(argparse.Namespace(all=False, key=None))
    assert "needs an alarm KEY" in str(e.value.code)


def test_watch_sends_every_policy_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    def _fake(op: str, **kw: Any) -> dict[str, Any]:
        calls.append((op, kw))
        return {"target": "47", "profile": "jobs", "budget_usd": 10.0,
                "remaining_usd": 5.0, "ceiling_source": "inherited",
                "ceiling_id": "c1"}

    monkeypatch.setattr(client, "_fleet_call_or_die", _fake)
    monkeypatch.setattr(client, "_fleet_requester", lambda: "tester")
    ap = _ported_tree()
    a = ap.parse_args(["fleet", "watch", "47", "--profile", "jobs",
                       "--budget", "10", "--standing", "--salvage-keep-n", "2"])
    a.fleetfunc(a)
    op, kw = calls[0]
    assert op == "watch" and kw["standing"] is True and kw["requester"] == "tester"
    assert kw["policy"]["salvage_keep_n"] == 2
    assert kw["policy"]["max_replacements"] is None
    assert set(kw["policy"]) == {
        "budget", "max_bid", "keep", "dry_run", "handoff", "strict_ceiling",
        "rescue_wait", "interval", "wall_budget", "max_relaunch",
        "max_replacements", "replace_ceiling_mult", "replacement_verified",
        "replacement_retention_hours", "salvage", "salvage_keep_n",
        "salvage_max_gb",
        # serve identity (P3): both None here — this is a `jobs` watch, and a
        # pin on a non-serve profile is refused before the call is made.
        "artifact", "expect_ident"}
    assert kw["policy"]["artifact"] is None
    assert kw["policy"]["expect_ident"] is None


def test_watch_prints_the_cap_that_landed_not_the_one_typed(
        monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(client, "_fleet_call_or_die", lambda *a, **k: {
        "target": "47", "profile": "jobs", "budget_usd": 10.0,
        "remaining_usd": 3.0, "ceiling_source": "inherited", "ceiling_id": "c1"})
    monkeypatch.setattr(client, "_fleet_requester", lambda: "tester")
    ap = _ported_tree()
    a = ap.parse_args(["fleet", "watch", "47", "--budget", "5"])
    a.fleetfunc(a)
    out = capsys.readouterr().out
    assert "budget=$10.000, remaining $3.000, ceiling inherited" in out
    assert "DURABLE ceiling of $10.000" in out
    assert "$7.000 has already been spent" in out


def test_watch_jobs_order_warning_is_silent_when_b2_is_unreachable(
        monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    import jobmeta

    def _boom(*a: Any, **k: Any) -> Any:
        raise RuntimeError("no b2")

    monkeypatch.setattr(jobmeta, "list_queue", _boom)
    cli_fleet.watch._fleet_watch_jobs_order_warning(
        argparse.Namespace(profile="jobs", keep=False, target="47"))
    assert capsys.readouterr().err == ""


def test_destroy_requires_yes() -> None:
    with pytest.raises(SystemExit) as e:
        cli_fleet.destroy.run(argparse.Namespace(yes=False))
    assert "needs --yes" in str(e.value.code)


def test_log_tails_filters_and_survives_a_bad_line(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Any,
        capsys: pytest.CaptureFixture[str]) -> None:
    p = tmp_path / "journal.ndjsonl"
    p.write_text('{"ts_iso":"T0","iid":"47","event":"tick","k":1}\n'
                 '{"ts_iso":"T1","iid":"48","event":"tick","k":2}\n'
                 'not json at all\n')
    monkeypatch.setattr(client, "fleet_journal_path", lambda: str(p))
    cli_fleet.log.run(argparse.Namespace(n=10, follow=False, iid="47"))
    out = capsys.readouterr().out
    assert 'T0 47          tick                {"k": 1}' in out
    assert "48" not in out
    # An unparseable line prints VERBATIM and BEFORE the --iid filter can see
    # it (the filter reads a field of a record that does not exist). That is
    # the flat behavior and it is the useful one: a half-written journal line
    # is exactly what you want to see in a post-mortem, not swallowed.
    assert "not json at all" in out


def test_log_exits_when_there_is_no_journal(monkeypatch: pytest.MonkeyPatch,
                                            tmp_path: Any) -> None:
    monkeypatch.setattr(client, "fleet_journal_path",
                        lambda: str(tmp_path / "nope.ndjsonl"))
    with pytest.raises(SystemExit) as e:
        cli_fleet.log.run(argparse.Namespace(n=10, follow=False, iid=None))
    assert "has fleetd ever run?" in str(e.value.code)


def test_install_and_deploy_shell_out_to_the_zone_e_script(
        monkeypatch: pytest.MonkeyPatch) -> None:
    argvs: list[list[str]] = []
    monkeypatch.setattr(cli_fleet.install.subprocess, "call",
                        lambda argv, *a, **k: argvs.append(argv) or 0)
    monkeypatch.setattr(cli_fleet.deploy.subprocess, "call",
                        lambda argv, *a, **k: argvs.append(argv) or 0)
    with pytest.raises(SystemExit):
        cli_fleet.install.run(argparse.Namespace(no_enable=True))
    with pytest.raises(SystemExit):
        cli_fleet.deploy.run(argparse.Namespace(checkout="/c", ref=None,
                                                python=None, no_restart=False,
                                                force=True))
    assert argvs[0] == [sys.executable, fleet_deploy._fleetd_script(),
                        "install-unit", "--no-enable"]
    assert argvs[1] == [sys.executable, fleet_deploy._fleetd_script(),
                        "deploy", "--checkout", "/c", "--force"]


def test_restart_refuses_mid_recovery_without_touching_systemctl(
        monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(client, "fleet_recoveries_in_flight",
                        lambda: [{"iid": "47", "kind": "rebid_ladder",
                                  "detail": "rung 2", "target": "run:R"}])
    monkeypatch.setattr(cli_fleet.restart.subprocess, "call",
                        lambda *a, **k: pytest.fail("systemctl must not run"))
    with pytest.raises(SystemExit) as e:
        cli_fleet.restart.run(argparse.Namespace(force=False))
    assert e.value.code == 2
    out = capsys.readouterr().out
    assert "REFUSING to restart fleetd" in out and "[watch run:R]" in out


def test_restart_force_does_not_even_read_the_state_file(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(client, "fleet_recoveries_in_flight",
                        lambda: pytest.fail("--force must not read state.json"))
    monkeypatch.setattr(cli_fleet.restart.subprocess, "call", lambda *a, **k: 0)
    with pytest.raises(SystemExit) as e:
        cli_fleet.restart.run(argparse.Namespace(force=True))
    assert e.value.code == 0


def test_uninstall_names_the_shared_unit_constant(
        monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    argvs: list[list[str]] = []
    monkeypatch.setattr(cli_fleet.uninstall.subprocess, "call",
                        lambda argv, *a, **k: argvs.append(argv) or 0)
    monkeypatch.setattr(cli_fleet.uninstall.os.path, "exists", lambda p: False)
    cli_fleet.uninstall.run(argparse.Namespace())
    assert argvs[0] == ["systemctl", "--user", "disable", "--now",
                        client.FLEET_UNIT_NAME]
    assert v.FLEET_UNIT_NAME is client.FLEET_UNIT_NAME   # one constant, two spellings
    assert "nothing is babysitting the fleet now" in capsys.readouterr().out


def test_report_delegates_to_the_flat_leaf(monkeypatch: pytest.MonkeyPatch) -> None:
    import fleet_report
    monkeypatch.setattr(fleet_report, "run", lambda a: 7)
    with pytest.raises(SystemExit) as e:
        cli_fleet.report.run(argparse.Namespace())
    assert e.value.code == 7

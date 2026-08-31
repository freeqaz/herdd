"""`herdd show` — the raw per-instance dump (cli/show.py).

What these tests pin: secret VALUES never reach stdout by default, through
every channel a launch can put one in (env dict, `extra_env` KEY=VALUE list,
inlined `onstart` exports, a URL credential, a secret-named nested container);
key NAMES survive so the record still answers "was it set?"; stdout stays
valid JSON with the count on stderr; and `--reveal-secrets` still prints the
raw record, with a warning.

Every credential below is SYNTHETIC. Per the module docstring, no test here
may run the command against the real environment.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vastlib.cli import show as cli_show  # noqa: E402
from vastlib.core import api  # noqa: E402

_TOKENS = {
    "b2": "0026fakeb2applicationkeyvalue00000000",
    "hf": "hf_fakefakefakefakefakefakefakefake",
    "ts": "tskey-auth-fakefakefake-fakefakefake",
    "serve": "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef",
    "login": "robot:fakeregistrypasswordvalue",
}

#: Shaped like a real v0/instances record, with a secret in each channel.
_RECORD = {
    "id": 48671690,
    "actual_status": "running",
    "gpu_name": "H200",
    "ssh_host": "ssh6.vast.ai",
    "ssh_port": 32606,
    "env": {
        "B2_APPLICATION_KEY": _TOKENS["b2"],
        "HF_TOKEN": _TOKENS["hf"],
        "VLLM_API_KEY": _TOKENS["serve"],
        "SERVE_ID": "v14eval",
        "MAX_HOURS": "6",
    },
    # The API returns extra_env as [NAME, VALUE] PAIRS — not "NAME=VALUE"
    # strings. Copied from a live record's shape; a synthetic string form is
    # what let 16 real credentials through the first cut of this redactor.
    "extra_env": [["TS_AUTHKEY", _TOKENS["ts"]],
                  ["B2_APPLICATION_KEY", _TOKENS["b2"]],
                  ["BOX_IDENTITY_NONCE", "a" * 32],
                  ["B2_BUCKET", "example-runs-bucket"],
                  ["MODE", "ddp"]],
    "onstart": (
        "#!/bin/bash\n"
        f"export HF_TOKEN={_TOKENS['hf']}\n"
        "export SERVE_DP=auto\n"
        "python3 -m vllm.entrypoints.openai.api_server\n"
    ),
    "image_login": _TOKENS["login"],
    "mirror_url": f"https://user:{_TOKENS['b2']}@b2.example.com/bucket",
}


def _run(monkeypatch, capsys, **kw):
    monkeypatch.setattr(api, "request", lambda *a, **k: dict(_RECORD))
    cli_show.run(argparse.Namespace(id=1, **kw))
    return capsys.readouterr()


def test_no_secret_value_reaches_stdout_by_default(monkeypatch, capsys):
    cap = _run(monkeypatch, capsys)
    for name, tok in _TOKENS.items():
        assert tok not in cap.out, f"{name} token leaked to stdout"


def test_key_names_and_non_secrets_survive(monkeypatch, capsys):
    """Redaction must not cost the diagnostic value of the dump."""
    doc = json.loads(_run(monkeypatch, capsys).out)
    assert doc["env"]["HF_TOKEN"] == cli_show.REDACTED      # name kept
    assert doc["env"]["SERVE_ID"] == "v14eval"              # non-secret intact
    assert doc["env"]["MAX_HOURS"] == "6"
    assert doc["ssh_port"] == 32606                         # non-str untouched
    assert doc["actual_status"] == "running"
    assert ["MODE", "ddp"] in doc["extra_env"]             # sibling pair kept
    assert ["B2_BUCKET", "example-runs-bucket"] in doc["extra_env"]
    assert "export SERVE_DP=auto" in doc["onstart"]


def test_every_channel_is_covered(monkeypatch, capsys):
    doc = json.loads(_run(monkeypatch, capsys).out)
    assert doc["env"]["B2_APPLICATION_KEY"] == cli_show.REDACTED   # env dict
    assert doc["extra_env"][0] == ["TS_AUTHKEY", cli_show.REDACTED]  # pair form
    assert doc["extra_env"][1] == ["B2_APPLICATION_KEY", cli_show.REDACTED]
    assert doc["extra_env"][2] == ["BOX_IDENTITY_NONCE", cli_show.REDACTED]
    assert cli_show.REDACTED in doc["onstart"]                     # onstart export
    assert doc["image_login"] == cli_show.REDACTED                 # secret-named
    assert doc["mirror_url"] == cli_show.REDACTED                  # URL credential


def test_stdout_stays_valid_json_and_the_count_goes_to_stderr(monkeypatch, capsys):
    cap = _run(monkeypatch, capsys)
    json.loads(cap.out)                       # would raise if the note landed here
    assert "redacted" in cap.err
    assert "--reveal-secrets" in cap.err


def test_reveal_secrets_restores_the_raw_record_and_warns(monkeypatch, capsys):
    cap = _run(monkeypatch, capsys, reveal_secrets=True)
    assert json.loads(cap.out) == _RECORD
    assert "LIVE CREDENTIALS" in cap.err


def test_nested_container_under_a_secret_name_is_withheld_whole(monkeypatch, capsys):
    """Fail-closed: an unmodelled credential hiding in a nested value."""
    hits: list[int] = []
    out = cli_show.redact({"image_auth": {"password": "fakepw", "user": "bot"}}, hits)
    assert out["image_auth"] == cli_show.REDACTED
    assert "fakepw" not in json.dumps(out)


def test_an_ordinary_record_is_left_alone():
    hits: list[int] = []
    rec = {"id": 1, "gpu_name": "H200", "onstart": "export MODE=ddp\n"}
    assert cli_show.redact(rec, hits) == rec
    assert hits == []


def test_extra_env_pairs_are_the_regression_case():
    """The name and value are separate list elements; a text-only KEY=VALUE
    rule sees neither pairing nor secret, and every value survives."""
    hits: list[int] = []
    out = cli_show.redact(
        [["HF_TOKEN", _TOKENS["hf"]], ["B2_BUCKET", "public-name"]], hits)
    assert out == [["HF_TOKEN", cli_show.REDACTED], ["B2_BUCKET", "public-name"]]
    assert _TOKENS["hf"] not in json.dumps(out)

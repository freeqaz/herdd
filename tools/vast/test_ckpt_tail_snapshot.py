"""Unit tests for jobmeta.ckpt_tail_snapshot — the live-append checkpoint path
(task #110). Pure filesystem, no rclone, no B2, no network.

WHAT THIS GUARDS. jobd's periodic checkpoint pass skips anything younger than
`--min-age` (45 s), so a file appended faster than that window is never shipped
at ALL — proven on job 20260803T130435-frontier-wave-3a68, whose ten attempt-1
passes read matched=16/files=15 with the same in-flight `results/gens_PAD.jsonl`
skipped every time (864 rows, ~25-30 min of GPU, regenerated from chunk 1).

The fix ships those files anyway, but ONLY as a snapshot cut at the last
complete line, and ONLY when the file is provably an append-only NDJSON. Both
halves matter and each has its own tests below:

  * DURABILITY — the growing file reaches the stage (test_ships_*).
  * NON-CORRUPTION — the staged bytes are always a whole number of complete,
    parseable records, even when the source's own tail is torn mid-record
    (test_truncated_tail_*). Durability that trades bounded lost compute for a
    corrupt row is strictly worse than the defect it fixes, so a refusal here is
    always the correct outcome and every refusal is asserted by REASON, not just
    by absence.

test_jobd.py::test_jobd_ships_a_live_append_style_checkpoint covers the same
code end to end through a real jobd run.
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jobmeta as jm  # noqa: E402


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _row(i, text="x"):
    return json.dumps({"id": f"fn{i}", "generations": [text]}) + "\n"


class _Run:
    """A run dir + state file + stage dir, driving passes explicitly.

    `now` is injected, never read off the clock: these fixtures write files
    microseconds before the call, and the whole point of the age rule is that it
    is evaluated against a specific instant.
    """

    def __init__(self, tmp_path, min_age=45.0):
        self.run = tmp_path / "run"
        self.run.mkdir()
        self.state = str(tmp_path / "tailstate.json")
        self.stage = tmp_path / "stage"
        self.min_age = min_age
        self.t = 1_000_000.0

    def write(self, rel, data, mtime=None):
        p = self.run / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        mode = "ab" if isinstance(data, bytes) else "a"
        with open(p, mode) as fh:
            fh.write(data)
        age = 0.0 if mtime is None else mtime
        os.utime(p, (self.t - age, self.t - age))
        return p

    def replace(self, rel, data):
        p = self.run / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(data)
        os.utime(p, (self.t, self.t))
        return p

    def pass_(self, rels, max_bytes=jm._CKPT_TAIL_MAX_BYTES):
        if self.stage.exists():
            for root, _dirs, files in os.walk(self.stage, topdown=False):
                for f in files:
                    os.unlink(os.path.join(root, f))
        return jm.ckpt_tail_snapshot(str(self.run), rels, self.min_age,
                                     self.state, str(self.stage),
                                     now=self.t, max_bytes=max_bytes)

    def staged_bytes(self, rel):
        return (self.stage / rel).read_bytes()


# --------------------------------------------------------------------------- #
# durability: the growing file gets shipped
# --------------------------------------------------------------------------- #
def test_first_pass_is_a_baseline_second_pass_ships(tmp_path):
    """Rule 5 is a two-observation proof, so pass 1 can only record. One extra
    checkpoint interval of lag is the price of never shipping a file we have
    not watched grow — and it is ~180 s against a whole lost stage."""
    r = _Run(tmp_path)
    r.write("results/gens_PAD.jsonl", _row(0) + _row(1))
    out = r.pass_(["results/gens_PAD.jsonl"])
    assert out["staged"] == []
    assert out["skipped"]["results/gens_PAD.jsonl"] == "no-baseline"

    r.write("results/gens_PAD.jsonl", _row(2))
    out = r.pass_(["results/gens_PAD.jsonl"])
    assert out["staged"] == ["results/gens_PAD.jsonl"]
    rows = [json.loads(l) for l in
            r.staged_bytes("results/gens_PAD.jsonl").decode().splitlines()]
    assert [x["id"] for x in rows] == ["fn0", "fn1", "fn2"]


def test_a_file_old_enough_is_left_to_the_ordinary_pass(tmp_path):
    """This path must never widen what the age-filtered pass already covers —
    it exists only for the files that pass can never reach."""
    r = _Run(tmp_path)
    r.write("results/gens.jsonl", _row(0))
    r.pass_(["results/gens.jsonl"])
    r.write("results/gens.jsonl", _row(1), mtime=90.0)
    out = r.pass_(["results/gens.jsonl"])
    assert out["staged"] == []
    assert out["skipped"]["results/gens.jsonl"] == "old-enough"


def test_a_quiescent_file_is_not_reshipped_every_pass(tmp_path):
    r = _Run(tmp_path)
    r.write("results/gens.jsonl", _row(0))
    r.pass_(["results/gens.jsonl"])
    out = r.pass_(["results/gens.jsonl"])       # same size, still young
    assert out["staged"] == []
    assert out["skipped"]["results/gens.jsonl"] == "not-growing"


def test_only_the_declared_matches_are_considered(tmp_path):
    """The bundle's own `checkpoints:` globs stay the sole authority over WHAT a
    job checkpoints. This path only changes WHEN."""
    r = _Run(tmp_path)
    r.write("results/gens.jsonl", _row(0))
    r.write("results/secret.jsonl", _row(0))
    r.pass_(["results/gens.jsonl"])
    r.write("results/gens.jsonl", _row(1))
    r.write("results/secret.jsonl", _row(1))
    out = r.pass_(["results/gens.jsonl"])
    assert out["staged"] == ["results/gens.jsonl"]
    assert not (r.stage / "results/secret.jsonl").exists()


# --------------------------------------------------------------------------- #
# non-corruption: a torn tail never reaches the stage
# --------------------------------------------------------------------------- #
def test_truncated_tail_is_cut_back_to_the_last_complete_record(tmp_path):
    """THE test the durability change is not allowed to ship without.

    The source file ends mid-record — exactly what a reader would see if it
    caught the producer between `write` and the newline. The snapshot must end
    at the last complete line, and every staged line must parse.
    """
    r = _Run(tmp_path)
    r.write("results/gens.jsonl", _row(0) + _row(1))
    r.pass_(["results/gens.jsonl"])
    r.write("results/gens.jsonl", _row(2) + '{"id": "fn3", "generat')
    out = r.pass_(["results/gens.jsonl"])
    assert out["staged"] == ["results/gens.jsonl"]

    body = r.staged_bytes("results/gens.jsonl")
    assert body.endswith(b"\n"), body[-40:]
    assert b"fn3" not in body, "the half-written record leaked into the snapshot"
    rows = [json.loads(l) for l in body.decode().splitlines() if l.strip()]
    assert [x["id"] for x in rows] == ["fn0", "fn1", "fn2"]


def test_truncated_tail_splitting_a_utf8_codepoint_is_cut_off(tmp_path):
    """A byte-level cut can land INSIDE a multi-byte character, which is how a
    tolerant reader still ends up with a UnicodeDecodeError. Newline alignment
    makes that unrepresentable: a newline is always a codepoint boundary."""
    r = _Run(tmp_path)
    r.write("results/gens.jsonl", _row(0, "héllo — ünïcode"))
    r.pass_(["results/gens.jsonl"])
    good = _row(1, "日本語テキスト")
    torn = _row(2, "ずっと長い").encode()[:-9]      # cuts mid-codepoint
    r.write("results/gens.jsonl", good.encode() + torn)
    out = r.pass_(["results/gens.jsonl"])
    assert out["staged"] == ["results/gens.jsonl"]

    body = r.staged_bytes("results/gens.jsonl")
    text = body.decode("utf-8")                     # must NOT raise
    rows = [json.loads(l) for l in text.splitlines() if l.strip()]
    assert [x["id"] for x in rows] == ["fn0", "fn1"]


def test_a_file_with_no_complete_line_yet_is_refused(tmp_path):
    r = _Run(tmp_path)
    r.write("results/gens.jsonl", '{"id": "fn0"')
    r.pass_(["results/gens.jsonl"])
    r.write("results/gens.jsonl", ', "generations": ["x')
    out = r.pass_(["results/gens.jsonl"])
    assert out["staged"] == []
    assert out["skipped"]["results/gens.jsonl"] == "no-complete-line"


# --------------------------------------------------------------------------- #
# the append proof (rule 5) and the NDJSON discriminator (rule 6)
# --------------------------------------------------------------------------- #
def test_a_rewritten_prefix_is_refused_not_shipped(tmp_path):
    """Growth alone is not append-only. A file whose EARLIER bytes changed was
    rewritten, and a prefix of a rewrite means nothing — refuse it."""
    r = _Run(tmp_path)
    r.write("results/gens.jsonl", _row(0) + _row(1))
    r.pass_(["results/gens.jsonl"])
    r.replace("results/gens.jsonl", _row(9) + _row(8) + _row(7))   # bigger, different
    out = r.pass_(["results/gens.jsonl"])
    assert out["staged"] == []
    assert out["skipped"]["results/gens.jsonl"] == "prefix-changed"


def test_pretty_printed_json_document_is_refused(tmp_path):
    """An indented JSON blob is newline-rich, so line alignment alone would
    happily ship an unparseable prefix of it. The first-line JSON check is what
    keeps whole-JSON artifacts (gate.json, score_summary.json, plan.json) out of
    this path entirely — they keep the age window, as designed."""
    r = _Run(tmp_path)
    r.write("results/gate.json", '{\n  "results": [\n')
    r.pass_(["results/gate.json"])
    r.write("results/gate.json", '    {"fn": "a"},\n    {"fn": "b"}\n')
    out = r.pass_(["results/gate.json"])
    assert out["staged"] == []
    assert out["skipped"]["results/gate.json"] == "not-ndjson"


def test_compact_json_document_has_no_newline_and_is_refused(tmp_path):
    r = _Run(tmp_path)
    r.write("results/summary.json", '{"a": 1, "b":')
    r.pass_(["results/summary.json"])
    r.write("results/summary.json", ' 2, "c": 3')
    out = r.pass_(["results/summary.json"])
    assert out["staged"] == []
    assert out["skipped"]["results/summary.json"] == "no-complete-line"


def test_binary_with_an_incidental_newline_byte_is_refused(tmp_path):
    """0x0A occurs all over a safetensors/optimizer blob. Line alignment is
    meaningless there, so the NUL/UTF-8/JSON clause has to catch it."""
    r = _Run(tmp_path)
    r.write("out/blob.bin", b"\x00\x01\x02\n\x03")
    r.pass_(["out/blob.bin"])
    r.write("out/blob.bin", b"\x04\x05\n\x06")
    out = r.pass_(["out/blob.bin"])
    assert out["staged"] == []
    assert out["skipped"]["out/blob.bin"] == "not-ndjson"


def test_trainer_checkpoint_dirs_are_never_touched(tmp_path):
    """`checkpoint-<N>/` has a real completeness oracle (_ckpt_write_complete +
    the fire-on-arrival pass). Guessing at a half-written shard there would put
    a corrupt adapter on the resume path — the one outcome worse than losing
    the shard."""
    r = _Run(tmp_path)
    rel = "out/checkpoint-50/trainer_state.json"
    r.write(rel, _row(0))
    r.pass_([rel])
    r.write(rel, _row(1))
    out = r.pass_([rel])
    assert out["staged"] == []
    assert out["skipped"][rel] == "checkpoint-dir"


def test_a_file_over_the_cap_is_refused(tmp_path):
    r = _Run(tmp_path)
    r.write("results/gens.jsonl", _row(0))
    r.pass_(["results/gens.jsonl"], max_bytes=64)
    r.write("results/gens.jsonl", _row(1) + _row(2))
    out = r.pass_(["results/gens.jsonl"], max_bytes=64)
    assert out["staged"] == []
    assert out["skipped"]["results/gens.jsonl"] == "too-big"


# --------------------------------------------------------------------------- #
# state handling
# --------------------------------------------------------------------------- #
def test_a_corrupt_state_file_costs_one_baseline_pass_not_an_error(tmp_path):
    r = _Run(tmp_path)
    r.write("results/gens.jsonl", _row(0))
    r.pass_(["results/gens.jsonl"])
    open(r.state, "w").write("{not json")
    r.write("results/gens.jsonl", _row(1))
    out = r.pass_(["results/gens.jsonl"])
    assert out["staged"] == []
    assert out["skipped"]["results/gens.jsonl"] == "no-baseline"
    r.write("results/gens.jsonl", _row(2))
    assert r.pass_(["results/gens.jsonl"])["staged"] == ["results/gens.jsonl"]


def test_a_vanished_match_is_skipped_silently(tmp_path):
    r = _Run(tmp_path)
    out = r.pass_(["results/never-existed.jsonl"])
    assert out["staged"] == [] and out["skipped"] == {}


def test_state_is_rewritten_every_pass_and_stays_parseable(tmp_path):
    r = _Run(tmp_path)
    r.write("results/gens.jsonl", _row(0))
    r.pass_(["results/gens.jsonl"])
    st = json.load(open(r.state))
    assert st["results/gens.jsonl"]["size"] == len(_row(0))
    assert len(st["results/gens.jsonl"]["sha"]) == 64


@pytest.mark.parametrize("n_rows", [1, 3, 400])
def test_snapshot_is_always_a_byte_prefix_of_the_source(tmp_path, n_rows):
    """The invariant every tail-tolerant reader is entitled to assume: what
    lands on B2 is a byte-prefix of the local file, cut on a record boundary."""
    r = _Run(tmp_path)
    r.write("results/gens.jsonl", _row(0))
    r.pass_(["results/gens.jsonl"])
    r.write("results/gens.jsonl", "".join(_row(i) for i in range(1, n_rows + 1))
            + '{"id": "torn"')
    assert r.pass_(["results/gens.jsonl"])["staged"] == ["results/gens.jsonl"]
    src = (r.run / "results/gens.jsonl").read_bytes()
    snap = r.staged_bytes("results/gens.jsonl")
    assert src.startswith(snap)
    assert snap.endswith(b"\n")

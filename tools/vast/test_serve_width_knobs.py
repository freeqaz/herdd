"""The two serving-width knobs on the vast HTTP serve path (E0, 2026-08-09).

`onstart/serve_vllm.sh` emitted neither `--max-num-seqs` (the gap doc 120 §5
named on 2026-07-15 and nobody closed) nor `--max-num-batched-tokens`, so both
resolved to vLLM's card-dependent defaults with nothing recording which default
was in force. `MAX_NUM_SEQS` / `MAX_NUM_BATCHED_TOKENS` now plumb them.

THE PROPERTY THAT MATTERS IS THE NEGATIVE ONE. This is an of-record eval serving
path: `max_num_seqs` is a MEASURED result term (−3 solves on v7,
`V8_DD_EVAL_RESULT_2026-08-05.md:135`), so a default value introduced here would
silently move every banked comparand without anybody choosing to. Unset must
therefore mean "the flag is not in argv at all", and the argv with both unset
must be byte-identical to the pre-change script's. Both are asserted below,
against the real shipped scripts, via their own `DRY_RUN=1` preview.

Skipped when bash is unavailable (portable lane runs it).
"""
import os
import shutil
import subprocess

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
SERVE_SH = os.path.join(_HERE, "onstart", "serve_vllm.sh")
LAUNCH_SH = os.path.join(_HERE, "launch_serve.sh")

pytestmark = pytest.mark.skipif(not shutil.which("bash"), reason="needs bash")


def _argv(**env):
    """`serve_vllm.sh` DRY_RUN=1 -> the `vllm serve` argv it would exec.

    MAX_HOURS=0 disarms the watchdog, SERVE_DP=1 skips the nvidia-smi probe;
    neither touches the flag assembly under test.
    """
    e = dict(os.environ)
    for _k in ("MAX_NUM_SEQS", "MAX_NUM_BATCHED_TOKENS",
               "SERVE_PREFIX_CACHING", "SERVE_MTP", "SERVE_MTP_NUM_SPEC"):
        e.pop(_k, None)
    e.update({"DRY_RUN": "1", "MODEL_ID": "/workspace/base",
              "MAX_HOURS": "0", "SERVE_DP": "1",
              # Pin the card the mnbt default is derived from, or this suite
              # would read a different argv on a GPU box than in the portable
              # lane — and the no-GPU answer is the fail-open one.
              "MNBT_DEVICE_TOTAL_MIB": "97887"})
    e.update({k: v for k, v in env.items() if v is not None})
    p = subprocess.run(["bash", SERVE_SH], capture_output=True, text=True,
                       timeout=120, env=e, cwd=_HERE)
    assert p.returncode == 0, p.stderr
    line = [ln for ln in p.stdout.splitlines() if ln.startswith("vllm serve")]
    assert len(line) == 1, p.stdout
    return line[0].split()


# --------------------------------------------------------------------------- #
# Unset means ABSENT — the comparability guarantee
# --------------------------------------------------------------------------- #

def test_unset_max_num_seqs_emits_nothing():
    """The width knob keeps the strict rule: it is a MEASURED result term, so
    no default may appear here. (`max_num_batched_tokens` diverged on
    2026-08-24 — see the mnbt block below.)"""
    assert "--max-num-seqs" not in _argv()


def test_empty_string_is_treated_as_unset():
    """A launcher that forwards `MAX_NUM_SEQS=` (the shape `--field k=` and an
    unfilled template both produce) must not become `--max-num-seqs ''`."""
    argv = _argv(MAX_NUM_SEQS="", MAX_NUM_BATCHED_TOKENS="")
    assert "--max-num-seqs" not in argv
    assert argv[argv.index("--max-num-batched-tokens") + 1] == "8192"


# --------------------------------------------------------------------------- #
# max_num_batched_tokens: EXPLICIT AT vLLM'S OWN VALUE (2026-08-24)
#
# `vllm serve` never prints the integer it resolved, so this path's prefill
# budget was inferable and never evidenced — assertable for 0 of the 12
# serve-path runs in the banked census. Emitting vLLM's own resolution changes
# the argv and not the behaviour. The tests below are the whole guarantee: if
# the derived value ever stops matching vLLM's table, this stops being a
# recording change and becomes a silent comparand move.
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("mib,expect", [
    (23028, "2048"),    # RTX 3090 24 GB
    (32607, "2048"),    # RTX 5090 32 GB
    (46068, "2048"),    # RTX 6000 Ada / PRO 5000 48 GB
    (71679, "2048"),    # one MiB under the 70 GiB boundary
    (71680, "8192"),    # exactly 70 GiB
    (81559, "8192"),    # H100 80 GB
    (97887, "8192"),    # RTX PRO 6000 96 GB
    (143771, "8192"),   # H200 NVL 141 GB
])
def test_the_derived_value_reproduces_the_api_server_row(mib, expect):
    """vLLM `EngineArgs._set_default_args_v1` keys on (>=70 GiB, UsageContext):
    the API-server row is 2048/8192 and the in-process row is 8192/16384.
    Splicing the rows is the error the campaign's own doc made, so the
    in-process values must never appear on this path."""
    argv = _argv(MNBT_DEVICE_TOTAL_MIB=str(mib))
    assert argv[argv.index("--max-num-batched-tokens") + 1] == expect
    # the in-process row's values must never be reachable by derivation
    assert argv[argv.index("--max-num-batched-tokens") + 1] != "16384"


def test_none_suppresses_the_flag_entirely():
    """The escape hatch: `none` restores the pre-2026-08-24 argv, for anyone who
    needs byte-identical argv rather than byte-identical behaviour."""
    assert "--max-num-batched-tokens" not in _argv(MAX_NUM_BATCHED_TOKENS="none")


def test_an_unreadable_card_fails_open_to_the_old_behaviour():
    """A guessed budget on a card we cannot read would be invisible and wrong.
    Emit nothing and say so, which is exactly what shipped before."""
    assert "--max-num-batched-tokens" not in _argv(MNBT_DEVICE_TOTAL_MIB="unknown")


def test_the_default_argv_is_frozen():
    """The exact argv a plain serve emits, frozen. A git-relative comparison
    would go permanently green-by-skip one commit later; a frozen literal keeps
    failing loudly if anything — a knob, a default, a reordering — creeps into
    the base argv.

    Moved TWICE, deliberately, and the two moves rest on DIFFERENT arguments —
    which matters, because one of them has since been refuted.

    2026-08-22: `--enable-prefix-caching` joins the base shape because prefix
    caching became opt-out (owner directive). That move was argued from "a cache
    hit replays exactly the KV the prefill would have computed, so it is
    output-identical and unlike max_num_seqs cannot move a banked comparand".
    **That premise was measured on 2026-08-24 and REFUTED** — cache OFF is 6/6
    reproducible and ON only 2/6, and the flag moves mamba_cache_mode
    none->align, perturbing even a cache-MISS prefill. So prefix caching is a
    comparand term of the same kind as max_num_seqs, and the reason it may sit
    in the frozen argv is now a throughput directive alone, not a neutrality
    argument. Whether it should stay is escalated, not settled here; this test
    pins what ships so a change is deliberate.

    2026-08-24: `--max-num-batched-tokens` joins it at vLLM's OWN resolution for
    the card (8192 for the 96 GB pin above). **This one is not a neutrality
    argument and so does not share that fate**: the term is NOT inert (it is
    worth ~1%), but the VALUE does not move — the flag records what vLLM was
    already choosing on a path that prints nothing, where the banked census
    could therefore assert the budget for none of its 12 serve runs. A
    behaviour-identical recording change, which is a strictly stronger claim
    than output-identity and is pinned directly by the derivation tests above.

    Anything else appearing in this list is still a bug. `--speculative-config`
    stays absent here even though `SERVE_MTP` defaults to `auto` since
    2026-08-27: `/workspace/base` is not a directory in this lane, and `auto`
    fails CLOSED with no checkpoint to read a head out of. That is the property
    being pinned — the flipped default must not be able to invent a flag from
    an undetectable model."""
    assert _argv() == [
        "vllm", "serve", "/workspace/base",
        "--host", "0.0.0.0", "--port", "8000",
        "--max-model-len", "16384",
        "--gpu-memory-utilization", "0.90",
        "--served-model-name", "base",
        "--max-num-batched-tokens", "8192",
        "--enable-prefix-caching",
    ]


# --------------------------------------------------------------------------- #
# Set means present
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("var,flag,value", [
    ("MAX_NUM_SEQS", "--max-num-seqs", "16"),
    ("MAX_NUM_BATCHED_TOKENS", "--max-num-batched-tokens", "8192"),
])
def test_set_emits_the_flag_with_its_value(var, flag, value):
    argv = _argv(**{var: value})
    assert flag in argv
    assert argv[argv.index(flag) + 1] == value


def test_both_together():
    argv = _argv(MAX_NUM_SEQS="64", MAX_NUM_BATCHED_TOKENS="16384")
    assert argv[argv.index("--max-num-seqs") + 1] == "64"
    assert argv[argv.index("--max-num-batched-tokens") + 1] == "16384"


def test_the_haproxy_replica_path_gets_the_same_flags():
    """`build_vllm_argv` is the one source of truth for both serve layouts —
    a knob that only reached the single-instance path would be a trap."""
    e = dict(os.environ)
    e.update({"DRY_RUN": "1", "MODEL_ID": "/workspace/base", "MAX_HOURS": "0",
              "SERVE_REPLICAS": "2", "MAX_NUM_SEQS": "16"})
    p = subprocess.run(["bash", SERVE_SH], capture_output=True, text=True,
                       timeout=120, env=e, cwd=_HERE)
    assert p.returncode == 0, p.stderr
    replicas = [ln for ln in p.stdout.splitlines()
                if ln.startswith("CUDA_VISIBLE_DEVICES=")]
    assert len(replicas) == 2, p.stdout
    assert all("--max-num-seqs 16" in ln for ln in replicas)


# --------------------------------------------------------------------------- #
# The knobs travel: env allowlist, summary fields, launcher passthrough
# --------------------------------------------------------------------------- #

def test_the_knobs_survive_the_etc_environment_allowlist():
    """serve_vllm.sh persists a filtered `env` into /etc/environment so an
    `--on-box --restart` re-run sees the same shape. A knob missing from that
    grep would silently drop on the second run only."""
    src = open(SERVE_SH).read()
    line = [ln for ln in src.splitlines() if ln.startswith("env | grep -E")]
    assert len(line) == 1
    import re as _re
    pat = _re.search(r"'\^\((.*)\)'", line[0]).group(1)
    for var in ("MAX_NUM_SEQS", "MAX_NUM_BATCHED_TOKENS"):
        assert _re.match("^(%s)" % pat, var), (
            "%s is not covered by the /etc/environment allowlist" % var)


def test_both_knobs_are_recorded_into_serve_summary():
    """E0's other half: a serve that does not record its own width is not
    self-documenting, and 'unset' has to be recorded as explicitly as a value."""
    src = open(SERVE_SH).read()
    assert '--field "max_num_seqs=${MAX_NUM_SEQS:-}"' in src
    assert '--field "max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS:-}"' in src


def test_launch_serve_forwards_both_only_when_set():
    src = open(LAUNCH_SH).read()
    assert "--max-num-seqs) MAX_NUM_SEQS=" in src
    assert "--max-num-batched-tokens) MAX_NUM_BATCHED_TOKENS=" in src
    # guarded forward, and NO default value assigned anywhere
    assert '[ -n "$MAX_NUM_SEQS" ]' in src
    assert '[ -n "$MAX_NUM_BATCHED_TOKENS" ]' in src
    assert 'MAX_NUM_SEQS=""' in src
    assert 'MAX_NUM_BATCHED_TOKENS=""' in src


# --------------------------------------------------------------------------- #
# Prefix caching is OPT-OUT (owner 2026-08-22)
# --------------------------------------------------------------------------- #

def test_prefix_caching_is_on_by_default():
    """The whole point of the directive. vLLM 0.27 computes
    `is_prefix_caching_supported and not is_hybrid`, so leaving this to vLLM's
    default silently disables it on every hybrid (Qwen3.5/3.6/3.8) — which is
    exactly what the eval fleet was doing, measured at
    prefix_cache_queries_total = 0."""
    assert "--enable-prefix-caching" in _argv()


def test_prefix_caching_can_be_opted_out():
    """=0 must emit NOTHING rather than an explicit disable: that restores
    vLLM's own per-model default, which is the correct escape hatch for a model
    whose architecture cannot support the feature."""
    argv = _argv(SERVE_PREFIX_CACHING="0")
    assert "--enable-prefix-caching" not in argv
    assert "--no-enable-prefix-caching" not in argv


def test_prefix_caching_reaches_every_replica():
    """A default that only landed on the single-instance path would be a trap
    of exactly the kind build_vllm_argv exists to prevent."""
    e = dict(os.environ)
    e.pop("SERVE_PREFIX_CACHING", None)
    e.update({"DRY_RUN": "1", "MODEL_ID": "/workspace/base", "MAX_HOURS": "0",
              "SERVE_REPLICAS": "2"})
    p = subprocess.run(["bash", SERVE_SH], capture_output=True, text=True,
                       timeout=120, env=e, cwd=_HERE)
    assert p.returncode == 0, p.stderr
    replicas = [ln for ln in p.stdout.splitlines()
                if ln.startswith("CUDA_VISIBLE_DEVICES=")]
    assert len(replicas) == 2, p.stdout
    assert all("--enable-prefix-caching" in ln for ln in replicas)


def test_the_new_knobs_survive_the_etc_environment_allowlist():
    src = open(SERVE_SH).read()
    line = [ln for ln in src.splitlines() if ln.startswith("env | grep -E")]
    assert len(line) == 1
    import re as _re
    pat = _re.search(r"'\^\((.*)\)'", line[0]).group(1)
    for var in ("SERVE_PREFIX_CACHING", "SERVE_MTP", "SERVE_MTP_NUM_SPEC"):
        assert _re.match("^(%s)" % pat, var), (
            "%s is not covered by the /etc/environment allowlist" % var)


def test_the_serve_shape_records_prefix_caching_and_mtp():
    """A serve that does not record whether it was cached or drafting cannot be
    compared against one that was."""
    src = open(SERVE_SH).read()
    assert '--field "prefix_caching=${SERVE_PREFIX_CACHING:-1}"' in src
    assert '--field "mtp=${SERVE_MTP_RESOLVED:-off}"' in src


# --------------------------------------------------------------------------- #
# MTP: ON by default where a head exists (owner directive 2026-08-27), and OFF
# whenever the head cannot be seen
# --------------------------------------------------------------------------- #

def test_mtp_is_on_by_default_when_the_checkpoint_ships_a_head(tmp_path):
    """The directive. Measured on eval-format prompts with the v14 LoRA r64
    attached unmerged (RTX PRO 6000, vLLM 0.27.1.post1+fork.gfb8e9ed57, run
    `<upstream-bench>/archive/runs/2026-08-27-v14-lora-mtp/`): +205%/+213%/+205%
    output tok/s at k=1/9/20, acceptance 0.932-0.944. The 2026-08-22 anchor's
    -2.3% at k=20 reproduces on ITS workload; the discriminator is acceptance,
    not concurrency."""
    (tmp_path / "config.json").write_text('{"num_nextn_predict_layers": 1}')
    assert "--speculative-config" in _argv(MODEL_ID=str(tmp_path))


def test_mtp_default_depth_is_five(tmp_path):
    """n=1 buys only ~+45% of the same workload's ~+205%. The 1-layer head is
    NOT clamped — vLLM reuses it autoregressively and warns that acceptance may
    fall; measured here it does not."""
    (tmp_path / "config.json").write_text('{"num_nextn_predict_layers": 1}')
    argv = _argv(MODEL_ID=str(tmp_path))
    spec = argv[argv.index("--speculative-config") + 1].replace("\\", "")
    assert '"num_speculative_tokens":5' in spec


def test_mtp_auto_is_off_without_a_detectable_head():
    """`auto` must fail CLOSED. A missing checkpoint, a bare HF id, or a model
    with no MTP head all mean no --speculative-config. This matters MORE now
    that auto is the default: a flipped default that guessed would put a cohort
    term on every serve of a model nobody checked."""
    assert "--speculative-config" not in _argv(SERVE_MTP="auto")


def test_mtp_off_is_off(tmp_path):
    """The opt-out has to beat a present head, or it is not an opt-out — this is
    the escape hatch for a min_p/logit_bias sampling lane and for holding a
    frozen comparand on the OFF cohort."""
    (tmp_path / "config.json").write_text('{"num_nextn_predict_layers": 1}')
    assert "--speculative-config" not in _argv(MODEL_ID=str(tmp_path), SERVE_MTP="0")
    assert "--speculative-config" not in _argv(SERVE_MTP="0")


def test_mtp_forced_on_emits_the_config():
    argv = _argv(SERVE_MTP="1", SERVE_MTP_NUM_SPEC="3")
    assert "--speculative-config" in argv
    # The DRY_RUN preview is `printf %q`-quoted, so the JSON arrives escaped;
    # the real path is `exec "${VLLM_ARGV[@]}"` and passes it through intact.
    # Keeping it SPACE-FREE is load-bearing — a space would make %q wrap the
    # whole thing and split() would tear the config across argv elements.
    spec = argv[argv.index("--speculative-config") + 1].replace("\\", "")
    assert '"method":"mtp"' in spec
    assert '"num_speculative_tokens":3' in spec


def test_mtp_auto_detects_a_head_on_disk(tmp_path):
    """The auto path reads the checkpoint, so give it one that declares a head."""
    (tmp_path / "config.json").write_text('{"num_nextn_predict_layers": 1}')
    argv = _argv(MODEL_ID=str(tmp_path), SERVE_MTP="auto")
    assert "--speculative-config" in argv


def test_mtp_auto_does_NOT_stand_down_for_lora(tmp_path):
    """The old stand-down was measured WRONG, so it is removed rather than
    relaxed. Attaching the v14 adapter RAISES n=5 acceptance by 21-35 points
    over base (0.9102/0.9169 vs 0.6995/0.5675 at k=1/k=9): the adapter's output
    on its own format is what the shared MTP head predicts most easily. vLLM
    accepts LoRA and MTP simultaneously on this fork, verified from the engine
    banner, not from the flag."""
    (tmp_path / "config.json").write_text('{"num_nextn_predict_layers": 1}')
    argv = _argv(MODEL_ID=str(tmp_path), SERVE_MTP="auto", LORA_SPECS="a=b2:some/adapter")
    assert "--speculative-config" in argv


def test_launch_serve_assigns_no_mtp_default_of_its_own():
    """The ON default lives in serve_vllm.sh and nowhere else, so the launch
    path and the `--on-box` attach path cannot disagree, and `--mtp 0` stays
    distinguishable from 'the caller said nothing'."""
    src = open(LAUNCH_SH).read()
    assert 'SERVE_MTP=""' in src
    assert 'SERVE_MTP_NUM_SPEC=""' in src


def test_launch_serve_exposes_the_new_knobs():
    src = open(LAUNCH_SH).read()
    assert "--no-prefix-caching) SERVE_PREFIX_CACHING=0" in src
    assert "--mtp) SERVE_MTP=" in src
    assert '[ -n "$SERVE_PREFIX_CACHING" ]' in src
    assert '[ -n "$SERVE_MTP" ]' in src

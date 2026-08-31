"""The probe command a refusal prints must name a bundle that CAN mint the anchor.

`vram_facts` refuses an unmeasured shape and tells the operator how to measure
it. That advice is the whole value of the refusal, and it was wrong for one
whole class of query: `fit-ladder` never passes `--target-modules`, so the
trainer applies its 7-projection default and every anchor fit-ladder mints keys
to the `list-7` group. An operator sent there by an `all-linear` refusal would
rent a box, run the probe, harvest it, and get the identical refusal back.

These tests pin the routing AND the property of each bundle it rests on, so a
bundle that starts or stops passing `--target-modules` fails here rather than
silently re-breaking the advice.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vram_facts as vf  # noqa: E402



def _probe_for(**query) -> str:
    with pytest.raises(vf.Unmeasured) as exc:
        vf.estimate_peak_gb(**query)
    return exc.value.probe_cmd or ""


BASE_QUERY = dict(base_slug="qwen35-9b", quant_mode="bf16", max_seq=20480,
                  batch=1, grad_checkpointing="off", ce_chunk_matmul="bf16",
                  lora_r=32, world_size=1)


def test_all_linear_refusal_names_the_bundle_that_pins_all_linear():
    cmd = _probe_for(target_modules="all-linear", **BASE_QUERY)
    assert "gpu-rate-9b-w20480" in cmd
    assert "fit-ladder --image" not in cmd
    assert "all-linear" in cmd, "the refusal should say WHY it is not fit-ladder"


def test_default_targets_refusal_still_names_fit_ladder():
    cmd = _probe_for(**BASE_QUERY)
    assert "fit-ladder" in cmd
    assert "gpu-rate-9b-w20480" not in cmd


def test_probe_cmd_carries_the_query_window_and_quant():
    cmd = _probe_for(target_modules="all-linear", **BASE_QUERY)
    assert "MAX_SEQ=20480" in cmd and "QUANT=bf16" in cmd
    assert "BASE=qwen35-9b" in cmd



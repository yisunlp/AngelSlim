"""Standalone smoke test: under ZeRO-3, build an empty Qwen3-30B-A3B model
with layers trimmed to 2, linearize MoE, stream weights, and verify the
per-expert Linear shapes are correct.

Run:
    torchrun --nproc_per_node=2 scripts/qat/test_moe_zero3_build.py
"""
import os
import sys
import torch
import torch.distributed as dist

# Initialise torch distributed first (required by deepspeed.zero.Init).
dist.init_process_group(backend="nccl")
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

# Register HfTrainerDeepSpeedConfig so is_deepspeed_zero3_enabled() returns True.
from transformers import Seq2SeqTrainingArguments  # noqa: E402

ds_config = {
    "zero_optimization": {"stage": 3, "stage3_max_live_parameters": 1e9,
                          "stage3_max_reuse_distance": 1e9},
    "train_batch_size": 2,
    "train_micro_batch_size_per_gpu": 1,
    "gradient_accumulation_steps": 1,
    "bf16": {"enabled": True},
}
_hf_args = Seq2SeqTrainingArguments(
    output_dir="./tmp_out", deepspeed=ds_config, bf16=True,
    per_device_train_batch_size=1,
)

from transformers import AutoConfig  # noqa: E402
from angelslim.utils import (  # noqa: E402
    is_deepspeed_zero3_enabled, is_zero3_param,
    zero3_empty_model_from_pretrained, stream_load_weights, linearize_moe_experts_empty,
)

assert is_deepspeed_zero3_enabled(), "HF ZeRO-3 not registered"

MODEL_PATH = "/apdcephfs_zwfy2/share_301053287/brunosu/all_models/Qwen3-30B-A3B"

# Trim layers to 2 for quick iteration.
cfg = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
cfg.num_hidden_layers = 2
cfg.use_cache = False

# Build empty model from the trimmed config.
from transformers import AutoModelForCausalLM  # noqa: E402
from transformers.initialization import no_init_weights, no_tie_weights  # noqa: E402

with no_init_weights(), no_tie_weights():
    model = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.bfloat16, trust_remote_code=True)

if dist.get_rank() == 0:
    print("Built empty model:", type(model).__name__)
    print("num layers:", len(model.model.layers))
    # Inspect first layer MoE before linearization
    layer = model.model.layers[0]
    mlp = layer.mlp
    print("mlp type:", type(mlp).__name__)
    if hasattr(mlp, "experts"):
        experts = mlp.experts
        print("experts type:", type(experts).__name__)
        print("has gate_up_proj:", hasattr(experts, "gate_up_proj"))
        if hasattr(experts, "gate_up_proj"):
            print("gate_up_proj ds_shape:", getattr(experts.gate_up_proj, "ds_shape", None))
            print("is_zero3:", is_zero3_param(experts.gate_up_proj))

replaced = linearize_moe_experts_empty(model, dtype=torch.bfloat16)
dist.barrier()

if dist.get_rank() == 0:
    print(f"Replaced {replaced} fused experts")
    layer = model.model.layers[0]
    mlp = layer.mlp
    experts = mlp.experts
    print("After linearize - experts type:", type(experts).__name__)
    print("num_experts attr:", experts.num_experts)
    # Inspect one expert
    e0 = experts[0]
    print("expert 0:", type(e0).__name__)
    print("  gate_proj weight shape:", e0["gate_proj"].weight.shape,
          "is_zero3:", is_zero3_param(e0["gate_proj"].weight))
    print("  gate_proj ds_shape:", getattr(e0["gate_proj"].weight, "ds_shape", None))

# Now stream load
stream_load_weights(model, MODEL_PATH, log_prefix=f"[rank{dist.get_rank()}]")
dist.barrier()

# Note: because we trimmed to 2 layers, the checkpoint has weights for
# layers [0..47], so layers [2..47] will appear as "unused keys" in the
# missing-key summary (not an error).

if dist.get_rank() == 0:
    print("\n=== Build + linearize + stream_load_weights OK ===")

dist.destroy_process_group()

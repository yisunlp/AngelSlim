#!/usr/bin/env bash
set -euo pipefail

export PYTORCH_ALLOC_CONF="expandable_segments:True"

torchrun --nproc_per_node=8 \
  tools/run.py \
  -c "configs/qwen3/qat/fp8_static/learn_scale/qwen3-30b-a3b_fp8_static_end2end_learn_scale_lwc_zero3_deepspeed2.yaml"

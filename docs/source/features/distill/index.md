# Distillation

AngelSlim Distill trains a student model with a separate full-precision teacher model. The student can be either a full-precision HuggingFace model or a quantized QAT-style model.

## Features

- Load an independent teacher from `compression.Distill.teacher_model_path`.
- Support full-precision students with HuggingFace `save_pretrained` output.
- Support quantized students by reusing QAT quantization, learnable-scale plugins, conversion, and save paths.
- Select trainable parameters with `trainable_parameters: all` or `trainable_parameters: quant`.
- Combine supervised CausalLM loss and knowledge distillation loss with `lm_loss_weight` and `kd_loss_weight`.
- Pass HuggingFace trainer options through `compression.Distill.hf_args`, including DeepSpeed ZeRO configs.

## Example 1: Distill a Smaller Full-Precision Model

This example distills a Qwen3-1.7B full-precision student from a Qwen3-4B full-precision teacher.

```bash
torchrun --nproc_per_node=2 \
  tools/run.py \
  -c configs/qwen3/distill/fp/qwen3-1_7b_fp_distill_from_qwen3-4b_zero2.yaml
```

Key fields:

```yaml
model:
  model_path: Qwen/Qwen3-1.7B

compression:
  name: Distill
  Distill:
    teacher_model_path: Qwen/Qwen3-4B
    student_type: fp
    trainable_parameters: all
    save_format: hf
    loss_type: cakld
    lm_loss_weight: 1.0
    kd_loss_weight: 1.0
```

## Example 2: Distill a Quantized Model

This example distills a W4A8-FP8 Qwen3-4B student from a full-precision Qwen3-4B teacher. The quantized student reuses QAT learnable-scale plugins and trains only quantization parameters.

```bash
torchrun --nproc_per_node=2 \
  tools/run.py \
  -c configs/qwen3/distill/w4a8_fp8/qwen3-4b_w4a8_fp8_distill_zero2.yaml
```

Key fields:

```yaml
compression:
  name: Distill
  quantization:
    name: w4a8_fp8
  Distill:
    teacher_model_path: Qwen/Qwen3-4B
    student_type: quantized
    trainable_parameters: quant
    save_format: real
    plugin_config:
      enable_scale: true
```

## Dataset Format

`TextDataset` supports plain language-modeling data and chat-style SFT data. For chat-style JSONL data, set `is_sft_data: true`; prompt tokens are masked with `-100`, and only the final assistant response contributes to the loss.

```json
{
  "messages": [
    {"role": "user", "content": "Explain knowledge distillation."},
    {"role": "assistant", "content": "Knowledge distillation trains a smaller student model to match a larger teacher model."}
  ]
}
```

## Main Distill Fields

```yaml
compression:
  name: Distill
  Distill:
    teacher_model_path: Qwen/Qwen3-4B
    teacher_torch_dtype: auto
    teacher_device_map: null
    student_type: fp          # fp or quantized
    trainable_parameters: all # all or quant
    save_format: hf           # hf/full/real for fp; real/fake for quantized paths
    loss_type: cakld          # origin, kl, rkl, kd, cakld, mse, kl_top, rkl_top
    kd_temperature: 1.0
    lm_loss_weight: 1.0
    kd_loss_weight: 1.0
    hf_args:
      deepspeed: configs/qwen3/distill/w4a8_fp8/ds_config_zero2.json
```

Use `loss_type: origin` with `kd_loss_weight: 0.0` to run a supervised fine-tuning baseline with the same trainer path.

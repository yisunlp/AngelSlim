# Copyright 2025 Tencent Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc
import glob
import json
import os

import torch
import torch.nn as nn

from .utils import find_parent_layer_and_sub_name, print_info
from .zero3_utils import gathered_param_if_zero3, is_deepspeed_zero3_enabled, is_zero3_param


def _rank():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def _cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def is_fused_moe_experts(module):
    required_attrs = (
        "gate_up_proj",
        "down_proj",
        "num_experts",
        "hidden_dim",
        "intermediate_dim",
        "act_fn",
    )
    return all(hasattr(module, attr) for attr in required_attrs)


class LinearizedMoeExperts(nn.Module):
    """Generic per-expert Linear wrapper for fused MoE expert tensors."""

    _angelslim_linearized_moe = True

    def __init__(self, experts_layer, copy_weights=True):
        super().__init__()
        self.num_experts = int(experts_layer.num_experts)
        self.hidden_dim = int(experts_layer.hidden_dim)
        self.intermediate_dim = int(experts_layer.intermediate_dim)
        self.act_fn = experts_layer.act_fn
        if hasattr(experts_layer, "config"):
            self.config = experts_layer.config

        gate_up = experts_layer.gate_up_proj
        down = experts_layer.down_proj
        dtype = gate_up.dtype
        device = gate_up.device
        if device.type == "meta":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for expert_idx in range(self.num_experts):
            expert = nn.ModuleDict(
                {
                    "gate_proj": nn.Linear(
                        self.hidden_dim,
                        self.intermediate_dim,
                        bias=False,
                        dtype=dtype,
                        device=device,
                    ),
                    "up_proj": nn.Linear(
                        self.hidden_dim,
                        self.intermediate_dim,
                        bias=False,
                        dtype=dtype,
                        device=device,
                    ),
                    "down_proj": nn.Linear(
                        self.intermediate_dim,
                        self.hidden_dim,
                        bias=False,
                        dtype=dtype,
                        device=device,
                    ),
                }
            )
            if copy_weights and gate_up.device.type != "meta" and down.device.type != "meta":
                with torch.no_grad():
                    gate_w, up_w = gate_up[expert_idx].chunk(2, dim=-2)
                    expert["gate_proj"].weight.copy_(gate_w)
                    expert["up_proj"].weight.copy_(up_w)
                    expert["down_proj"].weight.copy_(down[expert_idx])
            setattr(self, str(expert_idx), expert)

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            expert_layer = getattr(self, str(int(expert_idx)))
            gate = expert_layer["gate_proj"](current_state)
            up = expert_layer["up_proj"](current_state)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = expert_layer["down_proj"](current_hidden_states)
            current_hidden_states = (
                current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            )
            final_hidden_states.index_add_(
                0, token_idx, current_hidden_states.to(final_hidden_states.dtype)
            )

        return final_hidden_states


def linearize_moe_experts(model, copy_weights=True):
    """Replace fused MoE expert modules with per-expert Linear modules."""
    target_names = [
        name
        for name, module in model.named_modules()
        if is_fused_moe_experts(module)
        and not getattr(module, "_angelslim_linearized_moe", False)
    ]
    if not target_names:
        return 0

    z3 = is_deepspeed_zero3_enabled()
    replaced = 0
    for name in target_names:
        parent, sub_name = find_parent_layer_and_sub_name(model, name)
        old_experts = getattr(parent, sub_name)
        if z3:
            from .lazy_imports import deepspeed

            with deepspeed.zero.Init():
                new_experts = LinearizedMoeExperts(old_experts, copy_weights=False)
        else:
            new_experts = LinearizedMoeExperts(old_experts, copy_weights=copy_weights)
        setattr(parent, sub_name, new_experts)
        replaced += 1
        del old_experts
        _cleanup()
    print_info(f"[MoE linearize] replaced {replaced} fused expert module(s).")
    return replaced


def zero3_empty_model_from_pretrained(
    model_path,
    *,
    torch_dtype="auto",
    trust_remote_code=True,
    use_cache=False,
    attn_implementation="default",
    log_prefix="[zero3_loader]",
):
    """Build a ZeRO3-sharded empty model, linearize MoE, and stream weights."""
    from transformers import AutoConfig, AutoModelForCausalLM
    from transformers.initialization import no_init_weights, no_tie_weights

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    if attn_implementation != "default":
        config._attn_implementation = attn_implementation
    if use_cache is not None:
        config.use_cache = use_cache

    resolved_dtype = _resolve_dtype(torch_dtype, config)
    print_info(f"{log_prefix} build empty model with dtype={resolved_dtype}")
    with no_init_weights(), no_tie_weights():
        model = AutoModelForCausalLM.from_config(
            config,
            torch_dtype=resolved_dtype,
            trust_remote_code=trust_remote_code,
        )

    linearize_moe_experts(model, copy_weights=False)
    _stream_safetensors_into_model(model, model_path, log_prefix=log_prefix)

    try:
        model.tie_weights()
    except Exception as exc:  # noqa: BLE001
        print_info(f"{log_prefix} tie_weights skipped: {exc}")

    return model


def _resolve_dtype(torch_dtype, config):
    if isinstance(torch_dtype, torch.dtype):
        return torch_dtype
    if isinstance(torch_dtype, str) and torch_dtype != "auto":
        return getattr(torch, torch_dtype)
    resolved = getattr(config, "torch_dtype", None) or torch.float32
    if isinstance(resolved, str):
        return getattr(torch, resolved)
    return resolved


def _get_shards(model_path):
    from safetensors import safe_open

    index_path = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.isfile(index_path):
        with open(index_path, "r") as f:
            weight_map = json.load(f)["weight_map"]
        shard_to_keys = {}
        for key, shard in weight_map.items():
            shard_to_keys.setdefault(shard, []).append(key)
        return [(shard, shard_to_keys[shard]) for shard in sorted(shard_to_keys)]

    shard_paths = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
    if not shard_paths:
        raise FileNotFoundError(f"No safetensors checkpoint found under {model_path}")
    shards = []
    for shard_path in shard_paths:
        with safe_open(shard_path, framework="pt") as reader:
            shards.append((os.path.basename(shard_path), list(reader.keys())))
    return shards


def _stream_safetensors_into_model(model, model_path, log_prefix="[zero3_loader]"):
    from safetensors import safe_open

    name_to_param = dict(model.named_parameters())
    name_to_buffer = dict(model.named_buffers())
    loaded_targets = set()
    loaded = 0
    unexpected = 0

    for shard, keys in _get_shards(model_path):
        shard_path = shard if os.path.isabs(shard) else os.path.join(model_path, shard)
        with safe_open(shard_path, framework="pt") as reader:
            for key in keys:
                if key.endswith(".experts.gate_up_proj"):
                    n_loaded, n_unexpected = _load_gate_up(reader, key, name_to_param, loaded_targets)
                elif key.endswith(".experts.down_proj"):
                    n_loaded, n_unexpected = _load_down(reader, key, name_to_param, loaded_targets)
                else:
                    n_loaded, n_unexpected = _load_regular(
                        reader, key, name_to_param, name_to_buffer, loaded_targets
                    )
                loaded += n_loaded
                unexpected += n_unexpected
        _cleanup()
        print_info(f"{log_prefix} finished {os.path.basename(shard_path)}")

    target_names = set(name_to_param) | set(name_to_buffer)
    missing = sorted(target_names - loaded_targets)
    print_info(
        f"{log_prefix} load summary: loaded={loaded}, "
        f"unexpected={unexpected}, missing={len(missing)}"
    )
    if missing:
        print_info(f"{log_prefix} first missing keys: {missing[:20]}")


def _load_regular(reader, key, name_to_param, name_to_buffer, loaded_targets):
    target = name_to_param.get(key)
    is_buffer = False
    if target is None:
        target = name_to_buffer.get(key)
        is_buffer = target is not None
    if target is None:
        return 0, 1
    src = reader.get_tensor(key) if _rank() == 0 else None
    _copy_tensor_to_target(src, target, is_buffer=is_buffer)
    loaded_targets.add(key)
    return 1, 0


def _load_gate_up(reader, key, name_to_param, loaded_targets):
    base = key[: -len(".gate_up_proj")]
    src = reader.get_tensor(key) if _rank() == 0 else None
    num_experts = int(src.shape[0]) if src is not None else _infer_num_experts(base, name_to_param)
    loaded = 0
    unexpected = 0
    for i in range(num_experts):
        gate_key = f"{base}.{i}.gate_proj.weight"
        up_key = f"{base}.{i}.up_proj.weight"
        gate_tgt = name_to_param.get(gate_key)
        up_tgt = name_to_param.get(up_key)
        if gate_tgt is None or up_tgt is None:
            unexpected += 1
            continue
        gate_src = src[i].chunk(2, dim=-2)[0] if src is not None else None
        up_src = src[i].chunk(2, dim=-2)[1] if src is not None else None
        _copy_tensor_to_target(gate_src, gate_tgt)
        _copy_tensor_to_target(up_src, up_tgt)
        loaded_targets.update([gate_key, up_key])
        loaded += 2
    return loaded, unexpected


def _load_down(reader, key, name_to_param, loaded_targets):
    base = key[: -len(".down_proj")]
    src = reader.get_tensor(key) if _rank() == 0 else None
    num_experts = int(src.shape[0]) if src is not None else _infer_num_experts(base, name_to_param)
    loaded = 0
    unexpected = 0
    for i in range(num_experts):
        down_key = f"{base}.{i}.down_proj.weight"
        down_tgt = name_to_param.get(down_key)
        if down_tgt is None:
            unexpected += 1
            continue
        down_src = src[i] if src is not None else None
        _copy_tensor_to_target(down_src, down_tgt)
        loaded_targets.add(down_key)
        loaded += 1
    return loaded, unexpected


def _infer_num_experts(base, name_to_param):
    prefix = f"{base}."
    expert_ids = []
    for name in name_to_param:
        if not name.startswith(prefix):
            continue
        rest = name[len(prefix) :]
        first = rest.split(".", 1)[0]
        if first.isdigit():
            expert_ids.append(int(first))
    return max(expert_ids) + 1 if expert_ids else 0


def _copy_tensor_to_target(src, target, is_buffer=False):
    if is_buffer:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            if _rank() == 0:
                tmp = src.to(device=target.device, dtype=target.dtype)
            else:
                tmp = torch.empty_like(target)
            torch.distributed.broadcast(tmp, src=0)
            target.data.copy_(tmp)
        else:
            target.data.copy_(src.to(device=target.device, dtype=target.dtype))
        return

    if (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and not is_zero3_param(target)
    ):
        if _rank() == 0:
            tmp = src.to(device=target.device, dtype=target.dtype)
        else:
            tmp = torch.empty_like(target)
        torch.distributed.broadcast(tmp, src=0)
        target.data.copy_(tmp)
        return

    with gathered_param_if_zero3(target, modifier_rank=0):
        if _rank() == 0:
            target.data.copy_(src.to(device=target.device, dtype=target.dtype))

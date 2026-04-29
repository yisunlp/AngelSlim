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

import os
import re

import torch
import torch.nn as nn
from transformers.models.hy_v3.modeling_hy_v3 import (
    ALL_ATTENTION_FUNCTIONS,
    HYV3Experts,
    HYV3TopKRouter,
    apply_rotary_pos_emb,
    eager_attention_forward,
)

from ...compressor.quant.core import PTQSaveVllmHF
from ...utils import LinearizedMoeExperts, is_deepspeed_zero3_enabled
from ...utils.utils import find_layers, find_parent_layer_and_sub_name
from ..base_model import BaseLLMModel
from ..model_factory import SlimModelFactory


def _patch_hyv3_router_for_zero3():
    if getattr(HYV3TopKRouter, "_angelslim_zero3_dtype_patch", False):
        return

    def patched_forward(
        self,
        hidden_states: torch.Tensor,
        e_score_correction_bias: torch.Tensor,
    ) -> tuple:
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = nn.functional.linear(
            hidden_states.to(self.weight.dtype),
            self.weight,
        ).to(torch.float32)
        routing_weights = torch.sigmoid(router_logits)

        scores_for_choice = routing_weights + e_score_correction_bias
        _, top_k_index = torch.topk(scores_for_choice, self.top_k, dim=-1, sorted=False)
        top_k_weights = routing_weights.gather(1, top_k_index)

        top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-20)
        top_k_weights = top_k_weights * self.router_scaling_factor

        return router_logits, top_k_weights, top_k_index

    HYV3TopKRouter.forward = patched_forward
    HYV3TopKRouter._angelslim_zero3_dtype_patch = True


def _group_gemm_enabled() -> bool:
    return os.environ.get("ANGELSLIM_HYV3_MOE_GROUP_GEMM", "1").lower() not in {
        "0",
        "false",
        "off",
        "no",
    }


def _can_use_grouped_mm(hidden_states: torch.Tensor) -> bool:
    return (
        _group_gemm_enabled()
        and hasattr(torch.nn.functional, "grouped_mm")
        and hidden_states.is_cuda
        and hidden_states.dtype in (torch.bfloat16, torch.float16)
    )


def _is_zero3_leaf_module(module: nn.Module) -> bool:
    return bool(getattr(module, "_z3_leaf", False))


def _get_expert_module(module: nn.Module, expert_idx: int) -> nn.Module:
    return getattr(module, str(int(expert_idx)))


def _projection_weight(projection: nn.Module) -> torch.Tensor:
    if getattr(projection, "use_weight_quant", False) and hasattr(projection, "weight_quantizer"):
        return projection.weight_quantizer(projection.weight)
    return projection.weight


def _projection_input(projection: nn.Module, input_tensor: torch.Tensor) -> torch.Tensor:
    if getattr(projection, "use_act_quant", False) and hasattr(projection, "act_quantizer"):
        input_tensor = projection.act_quantizer(input_tensor)
    return input_tensor.to(projection.weight.dtype)


def _grouped_projection(
    ordered_input: torch.Tensor,
    projections,
    offsets: torch.Tensor,
    counts_list,
) -> torch.Tensor:
    if ordered_input.numel() == 0:
        out_features = projections[0].weight.shape[0]
        return ordered_input.new_empty((0, out_features))

    input_chunks = ordered_input.split(counts_list, dim=0)
    quantized_inputs = []
    for chunk, projection in zip(input_chunks, projections):
        # Keep ZeRO-3's submodule traversal identical across ranks. Some
        # experts are hit globally but have zero local tokens, and their
        # quantizers still need to be entered in the same order.
        quantized_inputs.append(_projection_input(projection, chunk))
    grouped_input = torch.cat(quantized_inputs, dim=0).contiguous()

    weights = [_projection_weight(projection).to(projection.weight.dtype).t() for projection in projections]
    grouped_weight = torch.stack(weights, dim=0).contiguous()

    biases = [getattr(projection, "bias", None) for projection in projections]
    grouped_bias = None
    if any(bias is not None for bias in biases):
        grouped_bias = torch.stack(
            [
                bias.to(projection.weight.dtype)
                if bias is not None
                else torch.zeros(
                    projection.weight.shape[0],
                    dtype=projection.weight.dtype,
                    device=projection.weight.device,
                )
                for bias, projection in zip(biases, projections)
            ],
            dim=0,
        ).contiguous()

    return torch.nn.functional.grouped_mm(
        grouped_input,
        grouped_weight,
        offs=offsets,
        bias=grouped_bias,
    )


def _expert_hit_mask(
    flat_expert_index: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    expert_hit_mask = torch.bincount(flat_expert_index, minlength=num_experts) > 0
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        # ZeRO-3 parameter gathers are collectives. Every rank must gather the
        # same expert weights in the same order, including experts only used by
        # another rank's local batch.
        expert_hit_int = expert_hit_mask.to(torch.int32)
        torch.distributed.all_reduce(expert_hit_int, op=torch.distributed.ReduceOp.MAX)
        expert_hit_mask = expert_hit_int.to(torch.bool)
    return expert_hit_mask


def _loop_experts_forward(
    module: nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    final_hidden_states = torch.zeros_like(hidden_states)
    with torch.no_grad():
        expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=module.num_experts)
        expert_mask = expert_mask.permute(2, 1, 0)
        expert_hit_mask = torch.greater(expert_mask.sum(dim=(-1, -2)), 0)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            expert_hit_int = expert_hit_mask.to(torch.int32)
            torch.distributed.all_reduce(expert_hit_int, op=torch.distributed.ReduceOp.MAX)
            expert_hit_mask = expert_hit_int.to(torch.bool)
        expert_hit = expert_hit_mask.nonzero()

    for expert_idx in expert_hit:
        expert_idx = expert_idx[0]
        top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
        current_state = hidden_states[token_idx]
        expert_layer = _get_expert_module(module, int(expert_idx.item()))
        gate = expert_layer["gate_proj"](current_state)
        up = expert_layer["up_proj"](current_state)
        current_hidden_states = module.act_fn(gate) * up
        current_hidden_states = expert_layer["down_proj"](current_hidden_states)
        current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
        final_hidden_states.index_add_(
            0, token_idx, current_hidden_states.to(final_hidden_states.dtype)
        )

    return final_hidden_states


def _grouped_gemm_experts_forward(
    module: nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    if is_deepspeed_zero3_enabled() and not _is_zero3_leaf_module(module):
        return _loop_experts_forward(module, hidden_states, top_k_index, top_k_weights)
    if not _can_use_grouped_mm(hidden_states):
        return _loop_experts_forward(module, hidden_states, top_k_index, top_k_weights)

    num_tokens = hidden_states.shape[0]
    flat_expert_index = top_k_index.reshape(-1).to(torch.long)
    flat_top_k_weights = top_k_weights.reshape(-1)
    flat_token_index = (
        torch.arange(num_tokens, device=hidden_states.device, dtype=torch.long)
        .unsqueeze(1)
        .expand_as(top_k_index)
        .reshape(-1)
    )

    with torch.no_grad():
        expert_hit_mask = _expert_hit_mask(flat_expert_index, module.num_experts)
        expert_ids = expert_hit_mask.nonzero(as_tuple=False).flatten()

    if expert_ids.numel() == 0:
        return torch.zeros_like(hidden_states)

    expert_ids_list = [int(expert_idx) for expert_idx in expert_ids.detach().cpu().tolist()]
    group_id_map = torch.empty(module.num_experts, device=hidden_states.device, dtype=torch.long)
    group_id_map[expert_ids] = torch.arange(
        expert_ids.numel(), device=hidden_states.device, dtype=torch.long
    )
    flat_group_index = group_id_map[flat_expert_index]
    sorted_group_index, order = torch.sort(flat_group_index)
    sorted_token_index = flat_token_index[order]
    sorted_top_k_weights = flat_top_k_weights[order]

    counts = torch.bincount(sorted_group_index, minlength=expert_ids.numel())
    offsets = torch.cumsum(counts, dim=0).to(torch.int32)
    counts_list = [int(count) for count in counts.detach().cpu().tolist()]
    ordered_input = hidden_states[sorted_token_index].contiguous()

    gate_projections = [
        _get_expert_module(module, expert_idx)["gate_proj"] for expert_idx in expert_ids_list
    ]
    up_projections = [_get_expert_module(module, expert_idx)["up_proj"] for expert_idx in expert_ids_list]
    gate = _grouped_projection(ordered_input, gate_projections, offsets, counts_list)
    up = _grouped_projection(ordered_input, up_projections, offsets, counts_list)
    current_hidden_states = module.act_fn(gate) * up

    down_projections = [
        _get_expert_module(module, expert_idx)["down_proj"] for expert_idx in expert_ids_list
    ]
    current_hidden_states = _grouped_projection(
        current_hidden_states.contiguous(),
        down_projections,
        offsets,
        counts_list,
    )

    current_hidden_states = current_hidden_states * sorted_top_k_weights[:, None]
    final_hidden_states = torch.zeros_like(hidden_states)
    final_hidden_states.index_add_(
        0, sorted_token_index, current_hidden_states.to(final_hidden_states.dtype)
    )
    return final_hidden_states


def _patch_linearized_moe_group_gemm():
    if getattr(LinearizedMoeExperts, "_angelslim_hyv3_group_gemm_patch", False):
        return

    def patched_forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        return _grouped_gemm_experts_forward(self, hidden_states, top_k_index, top_k_weights)

    LinearizedMoeExperts.forward = patched_forward
    LinearizedMoeExperts._angelslim_hyv3_group_gemm_patch = True


def _mark_zero3_moe_leaf_modules(model: nn.Module):
    if not is_deepspeed_zero3_enabled() or not _group_gemm_enabled():
        return

    try:
        from deepspeed.utils import set_z3_leaf_modules
    except Exception:  # noqa: BLE001
        return

    set_z3_leaf_modules(model, [LinearizedMoeExperts, HYV3ExpertsWithLinear], raise_if_not_found=False)


class HYV3ExpertsWithLinear(HYV3Experts):
    """Wrapper around HYV3Experts that exposes per-expert weights as nn.Linear modules.

    HYV3Experts stores all expert weights as 3-D nn.Parameter tensors, which are
    invisible to AngelSlim's find_layers() and PTQ hook (both only recognise
    nn.Linear).  This wrapper splits those tensors into individual nn.Linear
    modules at construction time so that the standard quantisation pipeline can
    observe and quantise them.

    Weight shape mapping
    --------------------
    gate_up_proj : [num_experts, 2*intermediate_dim, hidden_dim]
        gate_up_proj[i]  →  chunk(2, dim=0)
            gate_proj[i].weight : [intermediate_dim, hidden_dim]
            up_proj[i].weight   : [intermediate_dim, hidden_dim]
    down_proj : [num_experts, hidden_dim, intermediate_dim]
        down_proj[i] → down_proj[i].weight : [hidden_dim, intermediate_dim]
    """

    def __init__(self, experts_layer):
        # Bypass HYV3Experts.__init__ to avoid allocating large empty Parameter
        # tensors that we would immediately overwrite.  HYV3Experts does not
        # store self.config, so we copy the required scalar attributes directly.
        nn.Module.__init__(self)
        self.num_experts = experts_layer.num_experts
        self.hidden_dim = experts_layer.hidden_dim
        self.intermediate_dim = experts_layer.intermediate_dim
        self.act_fn = experts_layer.act_fn

        for expert_idx in range(self.num_experts):
            expert = nn.ModuleDict(
                {
                    "gate_proj": nn.Linear(self.hidden_dim, self.intermediate_dim, bias=False),
                    "up_proj": nn.Linear(self.hidden_dim, self.intermediate_dim, bias=False),
                    "down_proj": nn.Linear(self.intermediate_dim, self.hidden_dim, bias=False),
                }
            )
            # gate_up_proj[i]: [2*intermediate_dim, hidden_dim]
            # chunk on dim=0 → [intermediate_dim, hidden_dim] each
            expert["gate_proj"].weight.data, expert["up_proj"].weight.data = (
                experts_layer.gate_up_proj[expert_idx].chunk(2, dim=0)
            )
            # down_proj[i]: [hidden_dim, intermediate_dim]
            expert["down_proj"].weight.data = experts_layer.down_proj[expert_idx]
            setattr(self, f"{expert_idx}", expert)

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        return _grouped_gemm_experts_forward(self, hidden_states, top_k_index, top_k_weights)


@SlimModelFactory.register
class HYV3MoE(BaseLLMModel):
    def __init__(
        self,
        model=None,
        deploy_backend="vllm",
    ):
        super().__init__(
            model=model,
            deploy_backend=deploy_backend,
        )
        self.block_name = "model.layers"
        # Store original forward methods for restoration
        self._original_attn_forwards = {}
        # Store KV cache observers: {attn_layer_name: {"key_observer": ..., "value_observer": ...}}
        self.kv_cache_observers = {}

    def from_pretrained(
        self,
        model_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        use_cache=False,
        using_multi_nodes=False,
        attn_implementation=None,
    ):
        attn_implementation = attn_implementation or os.environ.get(
            "ANGELSLIM_HYV3_ATTN_IMPL", "sdpa"
        )
        torch_dtype = torch.bfloat16
        if is_deepspeed_zero3_enabled():
            _patch_hyv3_router_for_zero3()

        super().from_pretrained(
            model_path=model_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            low_cpu_mem_usage=low_cpu_mem_usage,
            use_cache=use_cache,
            using_multi_nodes=using_multi_nodes,
            attn_implementation=attn_implementation,
        )
        _patch_linearized_moe_group_gemm()
        _mark_zero3_moe_leaf_modules(self.model)

    def replace_moe(self):
        """Replace HYV3Experts instances with HYV3ExpertsWithLinear.

        This must be called before init_ptq() so that find_layers() can discover
        the per-expert nn.Linear modules and register them with the PTQ hook.
        """
        for name, module in self.model.named_modules():
            if isinstance(module, HYV3Experts) and not isinstance(module, HYV3ExpertsWithLinear):
                parent_layer, sub_name = find_parent_layer_and_sub_name(self.model, name)
                moe_linear = HYV3ExpertsWithLinear(module)
                del module
                setattr(parent_layer, sub_name, moe_linear)

    def init_ptq(self, slim_config):
        self.replace_moe()
        _mark_zero3_moe_leaf_modules(self.model)
        super().init_ptq(slim_config)

    def get_observer_layers(self):
        names = [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
            "shared_experts.gate_proj",
            "shared_experts.up_proj",
            "shared_experts.down_proj",
        ]
        expert_pattern = [
            r"model\.layers\.\d+\.mlp\.experts\.\d+\.gate_proj",
            r"model\.layers\.\d+\.mlp\.experts\.\d+\.up_proj",
            r"model\.layers\.\d+\.mlp\.experts\.\d+\.down_proj",
        ]

        obs_layers = [nn.Linear]
        observer_layers_dict = find_layers(self.model, layers=obs_layers)

        compiled_patterns = [re.compile(pattern) for pattern in expert_pattern]

        observer_layers_dict = {
            k: v
            for k, v in observer_layers_dict.items()
            if k.startswith(self.block_name)
            and (
                any(name in k for name in names)
                or any(pattern.search(k) for pattern in compiled_patterns)
            )
        }

        if self.quant_config.custom_observe_layers_names != "default":
            for custom_observe_name in self.quant_config.custom_observe_layers_names:
                for default_name in observer_layers_dict.keys():
                    if custom_observe_name not in default_name:
                        observer_layers_dict.pop(default_name)
        return observer_layers_dict

    def get_parent_dict(self, observer_layers_dict):
        parent_mapping = {r"experts\.\d+": "experts"}
        parent_dict = {}
        for layer_name in observer_layers_dict.keys():
            parent_name = layer_name
            for k, v in parent_mapping.items():
                parent_name = re.sub(k, v, layer_name)
            if parent_name != layer_name:
                parent_dict[layer_name] = parent_name
        return parent_dict

    def get_kvcache_observer_layers_names(self, observe_names):
        """Return empty list since we use attention-level patching for KV cache."""
        # Return empty list to disable the default k_proj/v_proj output observation
        # We will use apply_kvcache_observers() instead for RoPE-after key/value states
        return []

    def get_attention_layers(self):
        """Get all attention layers in the model."""
        attention_layers = {}
        for name, module in self.model.named_modules():
            if name.endswith(".self_attn") and hasattr(module, "forward"):
                # Verify it has k_proj and v_proj attributes
                if hasattr(module, "k_proj") and hasattr(module, "v_proj"):
                    attention_layers[name] = module
        return attention_layers

    def apply_kvcache_observers(self, kv_cache_observer_class, quant_bits=8):
        """
        Apply KV cache observers to attention layers using monkey patching.
        This observes key_states and value_states AFTER RoPE is applied.

        Args:
            kv_cache_observer_class: The observer class to use (e.g., AbsmaxPertensorObserver)
            quant_bits: Quantization bits for the observer
        """
        from ...compressor.quant.observers import AbsmaxPertensorObserver

        if kv_cache_observer_class is None:
            kv_cache_observer_class = AbsmaxPertensorObserver

        attention_layers = self.get_attention_layers()

        for attn_name, attn_module in attention_layers.items():
            # Create observers for key and value states
            key_observer = kv_cache_observer_class(
                layer=attn_module.k_proj,
                quant_bits=quant_bits,
            )
            value_observer = kv_cache_observer_class(
                layer=attn_module.v_proj,
                quant_bits=quant_bits,
            )

            # Store observers
            self.kv_cache_observers[attn_name] = {
                "key_observer": key_observer,
                "value_observer": value_observer,
            }

            # Save original forward
            self._original_attn_forwards[attn_name] = attn_module.forward

            # Create patched forward
            self._patch_attention_forward(attn_module, attn_name)

    def _patch_attention_forward(self, attn_module, attn_name):
        """
        Patch the attention module's forward method to observe KV cache after RoPE.

        Adapted to the new transformers ``HYV3Attention.forward`` signature, where
        rotary embeddings are pre-computed and passed in as ``position_embeddings``
        (a ``(cos, sin)`` tuple), ``q_norm``/``k_norm`` are applied unconditionally
        on the pre-transpose view, and attention dispatch goes through
        ``ALL_ATTENTION_FUNCTIONS``.
        """
        key_observer = self.kv_cache_observers[attn_name]["key_observer"]
        value_observer = self.kv_cache_observers[attn_name]["value_observer"]

        def patched_forward(
            hidden_states,
            position_embeddings,
            attention_mask,
            past_key_values=None,
            cache_position=None,
            **kwargs,
        ):
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, attn_module.head_dim)

            query_states = attn_module.q_proj(hidden_states).view(hidden_shape)
            key_states = attn_module.k_proj(hidden_states).view(hidden_shape)
            value_states = attn_module.v_proj(hidden_states).view(hidden_shape)

            query_states = attn_module.q_norm(query_states)
            key_states = attn_module.k_norm(key_states)

            query_states = query_states.transpose(1, 2)
            key_states = key_states.transpose(1, 2)
            value_states = value_states.transpose(1, 2)

            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

            # === OBSERVE KV CACHE AFTER RoPE ===
            key_observer(key_states)
            value_observer(value_states)
            # === END OBSERVE ===

            if past_key_values is not None:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past_key_values.update(
                    key_states, value_states, attn_module.layer_idx, cache_kwargs
                )

            attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
                attn_module.config._attn_implementation, eager_attention_forward
            )

            attn_output, attn_weights = attention_interface(
                attn_module,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not attn_module.training else attn_module.attention_dropout,
                scaling=attn_module.scaling,
                **kwargs,
            )

            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output = attn_module.o_proj(attn_output)
            return attn_output, attn_weights

        # Replace the forward method
        attn_module.forward = patched_forward

    def remove_kvcache_observers(self):
        """Remove patched forward methods and restore original ones."""
        for attn_name, original_forward in self._original_attn_forwards.items():
            # Find the attention module and restore its forward
            parts = attn_name.split(".")
            module = self.model
            for part in parts:
                module = getattr(module, part)
            module.forward = original_forward

        self._original_attn_forwards.clear()

    def get_kvcache_scales(self):
        """
        Get KV cache scales from observers.
        Returns dict with format: {"layer_name.k_cache.scale": scale,
                                   "layer_name.v_cache.scale": scale}
        """
        kv_scales = {}
        for attn_name, observers in self.kv_cache_observers.items():
            key_scale = observers["key_observer"].scales()
            value_scale = observers["value_observer"].scales()
            kv_scales[f"{attn_name}.k_cache.scale"] = key_scale
            kv_scales[f"{attn_name}.v_cache.scale"] = value_scale
        return kv_scales

    def get_save_func(self):
        if self.deploy_backend in ["vllm", "huggingface"]:
            return PTQSaveVllmHF
        else:
            raise NotImplementedError(
                f"deploy_backend {self.deploy_backend} is not supported for saving."
            )

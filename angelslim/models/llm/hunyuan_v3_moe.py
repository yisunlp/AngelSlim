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
import re

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.hy_v3.modeling_hy_v3 import (
    ALL_ATTENTION_FUNCTIONS,
    HYV3Experts,
    apply_rotary_pos_emb,
    eager_attention_forward,
)

from ...compressor.quant.core import PTQSaveVllmHF
from ...utils.utils import find_layers, find_parent_layer_and_sub_name
from ...utils.zero3_utils import gathered_params_if_zero3, is_zero3_param
from ..base_model import BaseLLMModel
from ..model_factory import SlimModelFactory


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

    ZeRO-3 notes
    ------------
    Callers MUST ensure ``experts_layer.gate_up_proj`` / ``down_proj`` are
    gathered (i.e. fully materialised on every rank) before instantiating
    this class when ZeRO-3 is active, otherwise per-expert slicing will
    see sharded data. :meth:`HYV3MoE.replace_moe` below handles that
    contract.
    """

    def __init__(self, experts_layer):
        # Bypass HYV3Experts.__init__ to avoid allocating large empty
        # Parameter tensors that we would immediately overwrite. HYV3Experts
        # does not store self.config, so we copy the scalar attrs directly.
        nn.Module.__init__(self)
        self.num_experts = experts_layer.num_experts
        self.hidden_dim = experts_layer.hidden_dim
        self.intermediate_dim = experts_layer.intermediate_dim
        self.act_fn = experts_layer.act_fn

        # Under a gather context these are full dense tensors on every rank.
        gate_up = experts_layer.gate_up_proj
        down = experts_layer.down_proj
        dtype = gate_up.dtype
        device = (
            gate_up.device
            if gate_up.device.type != "meta"
            else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        )

        for expert_idx in range(self.num_experts):
            # gate_up_proj[i]: [2*intermediate_dim, hidden_dim] -> chunk(2, dim=0)
            # down_proj[i]:    [hidden_dim, intermediate_dim]
            gate_w, up_w = gate_up[expert_idx].chunk(2, dim=0)
            down_w = down[expert_idx]

            expert = nn.ModuleDict(
                {
                    "gate_proj": nn.Linear(
                        self.hidden_dim, self.intermediate_dim,
                        bias=False, dtype=dtype, device=device,
                    ),
                    "up_proj": nn.Linear(
                        self.hidden_dim, self.intermediate_dim,
                        bias=False, dtype=dtype, device=device,
                    ),
                    "down_proj": nn.Linear(
                        self.intermediate_dim, self.hidden_dim,
                        bias=False, dtype=dtype, device=device,
                    ),
                }
            )
            with torch.no_grad():
                expert["gate_proj"].weight.data.copy_(gate_w)
                expert["up_proj"].weight.data.copy_(up_w)
                expert["down_proj"].weight.data.copy_(down_w)
            setattr(self, f"{expert_idx}", expert)
            del gate_w, up_w, down_w, expert

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
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
            expert_layer = getattr(self, f"{expert_idx}")
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
        attn_implementation="eager",
    ):
        """Load a pretrained HYV3 MoE checkpoint.

        Under DeepSpeed ZeRO-3 we take a streaming shard-by-shard path
        (see :func:`angelslim.utils.zero3_streaming_from_pretrained`) to
        avoid HF's pathological behaviour of merging the entire
        checkpoint into CPU RAM on every rank. Otherwise we fall back to
        vanilla ``AutoModelForCausalLM.from_pretrained``.
        """
        from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled

        # HYV3 currently requires eager attention; force it regardless of
        # caller argument (keeps prior behaviour).
        attn_implementation = "eager"
        # HYV3 requires bf16.
        torch_dtype = torch.bfloat16

        if is_deepspeed_zero3_enabled():
            from ...utils import print_info, zero3_streaming_from_pretrained

            print_info(
                "[HYV3MoE.from_pretrained] Detected DeepSpeed ZeRO-3: using "
                "streaming shard loader to bound host memory."
            )

            def _prepare(_model):
                # ``self.model`` must be set before ``replace_moe`` runs;
                # it walks ``self.model.named_modules()``.
                self.model = _model
                self.replace_moe()

            self.model = zero3_streaming_from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                trust_remote_code=trust_remote_code,
                use_cache=use_cache,
                attn_implementation=attn_implementation,
                prepare_fn=_prepare,
                log_prefix="[HYV3MoE.from_pretrained]",
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                attn_implementation=attn_implementation,
                torch_dtype=torch_dtype,
                device_map=device_map,
                trust_remote_code=trust_remote_code,
                low_cpu_mem_usage=low_cpu_mem_usage,
                use_cache=use_cache,
            )

        # Tokenizer is tiny; load on every rank.
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=trust_remote_code
        )

    def replace_moe(self):
        """Replace every ``HYV3Experts`` with ``HYV3ExpertsWithLinear``.

        ZeRO-3 aware, using the same layer-by-layer strategy as
        :meth:`angelslim.models.llm.qwen.Qwen.replace_moe`:

          * If the original experts parameters (``gate_up_proj`` /
            ``down_proj``) are ZeRO-3 shards, we gather them for just the
            current layer, build the per-expert Linears from the full
            tensors, and immediately partition the newly-created Linear
            weights via ``deepspeed.zero.Init(module=...)`` so peak
            resident memory stays bounded to ~one MoE layer.
          * After each layer we drop references to the old experts module
            and ``gc.collect`` + ``torch.cuda.empty_cache`` between layers.
          * When ZeRO-3 is not active this degrades to the original plain
            replacement path.

        Must run before ``init_ptq`` so ``find_layers`` can discover the
        per-expert ``nn.Linear`` modules and the PTQ hook can register
        them.
        """
        target_names = [
            name
            for name, module in self.model.named_modules()
            if isinstance(module, HYV3Experts)
            and not isinstance(module, HYV3ExpertsWithLinear)
        ]

        for name in target_names:
            self._replace_one_moe_layer(name)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _replace_one_moe_layer(self, name):
        parent_layer, sub_name = find_parent_layer_and_sub_name(self.model, name)
        old_experts = getattr(parent_layer, sub_name)

        z3 = is_zero3_param(old_experts.gate_up_proj) or is_zero3_param(
            old_experts.down_proj
        )

        # Gather the ONE layer's experts params, build the new module
        # while gathered, then release.
        with gathered_params_if_zero3(
            [old_experts.gate_up_proj, old_experts.down_proj],
            modifier_rank=None,
        ):
            moe_linear = HYV3ExpertsWithLinear(old_experts)

        setattr(parent_layer, sub_name, moe_linear)
        del old_experts
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Partition the freshly-created per-expert Linear weights in place
        # so each rank only retains its 1/N shard.
        if z3:
            from ...utils.lazy_imports import deepspeed

            deepspeed.zero.Init(module=moe_linear)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def init_ptq(self, slim_config):
        self.replace_moe()
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
            "shared_mlp.gate_proj",
            "shared_mlp.up_proj",
            "shared_mlp.down_proj",
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

            attention_interface = eager_attention_forward
            if attn_module.config._attn_implementation != "eager":
                attention_interface = ALL_ATTENTION_FUNCTIONS[
                    attn_module.config._attn_implementation
                ]

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
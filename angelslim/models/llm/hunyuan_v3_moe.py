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

"""SlimModelFactory adapter for HuggingFace HYV3 MoE models (``hy_v3``).

Key differences from ``Qwen``:

* HYV3 already stores each expert as an independent ``HYV3FeedForward``
  module (with ``gate_proj`` / ``up_proj`` / ``down_proj`` as regular
  ``nn.Linear``), so there is NO fused-tensor unpacking to do — the QAT
  pipeline can observe every expert projection out of the box.
* HYV3 supports an ``attn_implementation`` field in its config and uses
  a custom RoPE (``apply_rotary_emb_with_positions``); KV-cache quant
  observers therefore need a model-specific patched attention forward.

This adapter supplies:

* ``replace_moe()`` as a no-op (kept for API symmetry with other MoE
  adapters);
* KV-cache observer plumbing + optional FP8 attention patch for PTQ;
* Standard ``get_observer_layers`` / ``get_save_func`` / ``get_parent_dict``
  used by both PTQ and QAT.

``from_pretrained`` is intentionally NOT overridden here so that the
base class's ZeRO-3 streaming loader (``BaseLLMModel.from_pretrained``)
can take effect under DeepSpeed ZeRO-3. The base class already honours
``attn_implementation`` and ``torch_dtype``.
"""

import math
import re

import torch
from transformers.models.hy_v3.modeling_HY_v3 import repeat_kv

from ...compressor.qat.modules.quantizer import fp8_cast_ste
from ...compressor.quant.core import PTQSaveVllmHF
from ...utils.utils import find_layers, print_info
from ..base_model import BaseLLMModel
from ..model_factory import SlimModelFactory


@SlimModelFactory.register
class HYV3MoE(BaseLLMModel):
    """AngelSlim adapter for HuggingFace HYV3 MoE (``model_type: hy_v3``)."""

    def __init__(self, model=None, deploy_backend="vllm"):
        super().__init__(model=model, deploy_backend=deploy_backend)
        self.block_name = "model.layers"
        self.observer_layer_classes = [torch.nn.Linear]
        self.observed_names = [
            "k_proj",
            "v_proj",
            "q_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
        # KV-cache observers: {attn_layer_name: {"key_observer": ..., "value_observer": ...}}
        self.kv_cache_observers = {}

    # ------------------------------------------------------------------
    # Model-specific hooks
    # ------------------------------------------------------------------

    def replace_moe(self):
        """No-op: HYV3MoE already uses per-expert ``nn.Linear`` modules."""
        print_info("[HYV3MoE] replace_moe skipped (experts already Linear-based).")

    def init_ptq(self, slim_config):
        # Keep API parity with Qwen's ``init_ptq``.
        self.replace_moe()
        super().init_ptq(slim_config)

    def get_observer_layers(self):
        """Collect all ``nn.Linear`` modules we want to observe:
        attention projections, dense MLP (first_k_dense_replace layers),
        shared MLP and every MoE expert triple.
        """
        attn_and_mlp_names = [
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
        expert_patterns = [
            re.compile(r"model\.layers\.\d+\.mlp\.experts\.\d+\.gate_proj"),
            re.compile(r"model\.layers\.\d+\.mlp\.experts\.\d+\.up_proj"),
            re.compile(r"model\.layers\.\d+\.mlp\.experts\.\d+\.down_proj"),
        ]

        observer_layers_dict = find_layers(self.model, layers=self.observer_layer_classes)
        observer_layers_dict = {
            k: v
            for k, v in observer_layers_dict.items()
            if k.startswith(self.block_name)
            and (
                any(name in k for name in attn_and_mlp_names)
                or any(pat.search(k) for pat in expert_patterns)
            )
        }

        if self.quant_config.custom_observe_layers_names != "default":
            for custom_observe_name in self.quant_config.custom_observe_layers_names:
                for default_name in list(observer_layers_dict.keys()):
                    if custom_observe_name not in default_name:
                        observer_layers_dict.pop(default_name)
        return observer_layers_dict

    def get_parent_dict(self, observer_layers_dict):
        """Group all per-expert linears under the shared ``experts`` name
        so downstream smoothing / scale fusion can treat them as one."""
        parent_mapping = {r"experts\.\d+": "experts"}
        parent_dict = {}
        for layer_name in observer_layers_dict.keys():
            parent_name = layer_name
            for k, v in parent_mapping.items():
                parent_name = re.sub(k, v, layer_name)
            if parent_name != layer_name:
                parent_dict[layer_name] = parent_name
        return parent_dict

    def get_smooth_mapping_layers(self, smooth_config, mappings=None):
        if mappings is None:
            mappings = [
                (["q_proj", "k_proj", "v_proj"], "input_layernorm"),
                (["gate_proj", "up_proj"], "post_attention_layernorm"),
            ]
        assert len(mappings) == 2
        assert smooth_config.smooth_first_linears or smooth_config.smooth_last_linears
        return super().get_smooth_mapping_layers(smooth_config, mappings)

    def get_save_func(self):
        if self.deploy_backend in ["vllm", "huggingface"]:
            return PTQSaveVllmHF
        raise NotImplementedError(
            f"deploy_backend {self.deploy_backend} is not supported for saving."
        )

    # ------------------------------------------------------------------
    # KV-cache PTQ observers (unused in QAT; kept for PTQ compatibility)
    # ------------------------------------------------------------------

    def get_kvcache_observer_layers_names(self, observe_names):
        """Return empty list: we use attention-level patching for KV cache
        so the default k_proj/v_proj output observation is disabled."""
        return []

    def get_attention_layers(self):
        attention_layers = {}
        for name, module in self.model.named_modules():
            if name.endswith(".self_attn") and hasattr(module, "forward"):
                if hasattr(module, "k_proj") and hasattr(module, "v_proj"):
                    attention_layers[name] = module
        return attention_layers

    def apply_kvcache_observers(self, kv_cache_observer_class, quant_bits=8):
        """Attach key/value observers that see the states AFTER RoPE."""
        from ...compressor.quant.observers import AbsmaxPertensorObserver

        if kv_cache_observer_class is None:
            kv_cache_observer_class = AbsmaxPertensorObserver

        attention_layers = self.get_attention_layers()

        for attn_name, attn_module in attention_layers.items():
            key_observer = kv_cache_observer_class(
                layer=attn_module.k_proj,
                quant_bits=quant_bits,
            )
            value_observer = kv_cache_observer_class(
                layer=attn_module.v_proj,
                quant_bits=quant_bits,
            )
            self.kv_cache_observers[attn_name] = {
                "key_observer": key_observer,
                "value_observer": value_observer,
            }
            self._original_attn_forwards[attn_name] = attn_module.forward
            self._patch_attention_forward(attn_module, attn_name)

    def _patch_attention_forward(self, attn_module, attn_name):
        key_observer = self.kv_cache_observers[attn_name]["key_observer"]
        value_observer = self.kv_cache_observers[attn_name]["value_observer"]

        def patched_forward(
            hidden_states,
            attention_mask=None,
            position_ids=None,
            rope_cache=None,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
            cache_position=None,
            **kwargs,
        ):
            bsz, q_len, _ = hidden_states.size()

            query_states = attn_module.q_proj(hidden_states)
            key_states = attn_module.k_proj(hidden_states)
            value_states = attn_module.v_proj(hidden_states)

            query_states = query_states.view(
                bsz, q_len, attn_module.num_heads, attn_module.head_dim
            ).transpose(1, 2)
            key_states = key_states.view(
                bsz, q_len, attn_module.num_key_value_heads, attn_module.head_dim
            ).transpose(1, 2)
            value_states = value_states.view(
                bsz, q_len, attn_module.num_key_value_heads, attn_module.head_dim
            ).transpose(1, 2)

            if attn_module.use_qk_norm:
                query_states = attn_module.q_norm(query_states)
                key_states = attn_module.k_norm(key_states)

            assert cache_position is not None, "cache_position must not be None"
            from transformers.models.hy_v3.modeling_HY_v3 import (
                apply_rotary_emb_with_positions,
            )

            query_states, key_states = apply_rotary_emb_with_positions(
                query_states, key_states, rope_cache, cache_position
            )

            # Observe key/value states AFTER RoPE.
            key_observer(key_states)
            value_observer(value_states)

            past_key_value = getattr(attn_module, "past_key_value", past_key_value)
            if past_key_value is not None:
                cache_kwargs = {"cache_position": cache_position}
                key_states, value_states = past_key_value.update(
                    key_states, value_states, attn_module.layer_idx, cache_kwargs
                )

            key_states = repeat_kv(key_states, attn_module.num_key_value_groups)
            value_states = repeat_kv(value_states, attn_module.num_key_value_groups)

            is_decoding = q_len <= 1
            impl = attn_module.config._attn_implementation
            if impl == "eager":
                attn_output, attn_weights = attn_module._eager_attention_forward(
                    query_states,
                    key_states,
                    value_states,
                    attention_mask,
                    output_attentions,
                    is_decoding,
                )
            elif impl == "flash_attention_2":
                attn_output, attn_weights = attn_module._flash_attention_2_forward(
                    query_states,
                    key_states,
                    value_states,
                    attention_mask,
                    dropout=0.0,
                    position_ids=position_ids,
                )
                attn_output = attn_output.transpose(1, 2)
            else:
                raise ValueError(f"Invalid attention implementation: {impl}")

            if attn_output.size() != (
                bsz,
                attn_module.num_heads,
                q_len,
                attn_module.head_dim,
            ):
                raise ValueError(
                    "attn_output has unexpected shape: "
                    f"{attn_output.size()}, expected "
                    f"{(bsz, attn_module.num_heads, q_len, attn_module.head_dim)}"
                )

            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(bsz, q_len, -1)
            attn_output = attn_module.o_proj(attn_output)

            if not use_cache:
                past_key_value = None

            if output_attentions:
                return attn_output, attn_weights, past_key_value
            return attn_output, None, past_key_value

        attn_module.forward = patched_forward

    def remove_kvcache_observers(self):
        for attn_name, original_forward in self._original_attn_forwards.items():
            parts = attn_name.split(".")
            module = self.model
            for part in parts:
                module = getattr(module, part)
            module.forward = original_forward
        self._original_attn_forwards.clear()

    def get_kvcache_scales(self):
        kv_scales = {}
        for attn_name, observers in self.kv_cache_observers.items():
            key_scale = observers["key_observer"].scales()
            value_scale = observers["value_observer"].scales()
            kv_scales[f"{attn_name}.k_cache.scale"] = key_scale
            kv_scales[f"{attn_name}.v_cache.scale"] = value_scale
        return kv_scales

    # ------------------------------------------------------------------
    # Optional FP8 attention simulation (QAT)
    # ------------------------------------------------------------------

    def patch_fp8_attention(self):
        attention_layers = self.get_attention_layers()

        for attn_name, attn_module in attention_layers.items():
            self._original_attn_forwards.setdefault(attn_name, attn_module.forward)

            def _make_patched(attn_mod):
                def patched_forward(
                    hidden_states,
                    attention_mask=None,
                    position_ids=None,
                    rope_cache=None,
                    past_key_value=None,
                    output_attentions=False,
                    use_cache=False,
                    cache_position=None,
                    **kwargs,
                ):
                    bsz, q_len, _ = hidden_states.size()

                    query_states = attn_mod.q_proj(hidden_states)
                    key_states = attn_mod.k_proj(hidden_states)
                    value_states = attn_mod.v_proj(hidden_states)

                    query_states = query_states.view(
                        bsz, q_len, attn_mod.num_heads, attn_mod.head_dim
                    ).transpose(1, 2)
                    key_states = key_states.view(
                        bsz, q_len, attn_mod.num_key_value_heads, attn_mod.head_dim
                    ).transpose(1, 2)
                    value_states = value_states.view(
                        bsz, q_len, attn_mod.num_key_value_heads, attn_mod.head_dim
                    ).transpose(1, 2)

                    if attn_mod.use_qk_norm:
                        query_states = attn_mod.q_norm(query_states)
                        key_states = attn_mod.k_norm(key_states)

                    assert cache_position is not None, "cache_position must not be None"
                    from transformers.models.hy_v3.modeling_HY_v3 import (
                        apply_rotary_emb_with_positions,
                    )

                    query_states, key_states = apply_rotary_emb_with_positions(
                        query_states, key_states, rope_cache, cache_position
                    )

                    past_key_value = getattr(attn_mod, "past_key_value", past_key_value)
                    if past_key_value is not None:
                        cache_kwargs = {"cache_position": cache_position}
                        key_states, value_states = past_key_value.update(
                            key_states, value_states, attn_mod.layer_idx, cache_kwargs
                        )

                    key_states = repeat_kv(key_states, attn_mod.num_key_value_groups)
                    value_states = repeat_kv(value_states, attn_mod.num_key_value_groups)

                    bsz, num_heads, seqlen, head_dim = query_states.shape
                    _L, S = query_states.size(-2), key_states.size(-2)
                    scale_factor = 1.0 / math.sqrt(query_states.size(-1))

                    is_decoding = q_len <= 1
                    attn_bias = attn_mod._prepare_attention_bias(
                        attention_mask,
                        _L,
                        S,
                        query_states.dtype,
                        query_states.device,
                        is_decoding,
                    )
                    matmul_input_buffer = torch.empty(
                        bsz * num_heads,
                        seqlen,
                        S,
                        dtype=query_states.dtype,
                        device=query_states.device,
                    )
                    attn_weight = torch.baddbmm(
                        matmul_input_buffer,
                        query_states.reshape(bsz * num_heads, seqlen, head_dim),
                        key_states.reshape(bsz * num_heads, S, head_dim).transpose(1, 2),
                        beta=0.0,
                        alpha=scale_factor,
                    )
                    attn_weight = attn_weight.reshape(bsz, num_heads, seqlen, S)
                    attn_weight = attn_weight + attn_bias.unsqueeze(0).unsqueeze(0)

                    if not attn_mod.enable_attention_fp32_softmax:
                        attn_weight = torch.softmax(attn_weight, dim=-1, dtype=torch.bfloat16)
                    else:
                        with torch.autocast(
                            device_type=attn_weight.device.type, dtype=torch.float32
                        ):
                            attn_weight = torch.softmax(attn_weight.to(torch.float32), dim=-1)
                        attn_weight = attn_weight.to(dtype=torch.bfloat16)

                    # FP8 cast simulation (STE).
                    attn_weight = fp8_cast_ste(attn_weight)
                    attn_output = torch.bmm(
                        attn_weight.reshape(bsz * num_heads, seqlen, S),
                        value_states.reshape(bsz * num_heads, S, head_dim),
                    )
                    attn_output = attn_output.reshape(bsz, num_heads, seqlen, head_dim)
                    attn_output = attn_output.transpose(1, 2).contiguous()
                    attn_output = attn_output.reshape(bsz, q_len, -1)
                    attn_output = attn_mod.o_proj(attn_output)

                    if not use_cache:
                        past_key_value = None
                    if output_attentions:
                        return attn_output, attn_weight, past_key_value
                    return attn_output, None, past_key_value

                return patched_forward

            attn_module.forward = _make_patched(attn_module)

        print_info(f"FP8 attention enabled for {len(attention_layers)} layers (patch forward)")

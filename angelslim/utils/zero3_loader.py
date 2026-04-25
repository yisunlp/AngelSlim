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

"""Memory-bounded ZeRO-3 ``from_pretrained`` helper.

Rationale
---------
HuggingFace ``transformers`` (tested up to v5.5) takes a pathological
path for DeepSpeed ZeRO-3 inside ``_load_pretrained_model``::

    if is_deepspeed_zero3_enabled() and not is_quantized:
        merged_state_dict = {}
        for ckpt_file in checkpoint_files:
            merged_state_dict.update(
                load_state_dict(ckpt_file, map_location="cpu", ...)
            )
        state_dict = merged_state_dict

That is: *every rank* builds a full CPU copy of the entire checkpoint
before DeepSpeed partitions it. For a 60GB bf16 MoE model running 8
ranks on one node, host RAM peaks at ~480GB before a single tensor is
loaded into the sharded model. The HuggingFace source even carries a
comment acknowledging this is "not ideal" but unresolved.

The utility in this module bypasses that path entirely:

  1. Build an empty, sharded model via ``AutoModelForCausalLM.from_config``
     inside ``no_init_weights`` + ``no_tie_weights``. Because
     ``is_deepspeed_zero3_enabled()`` is True, HF's own ``_from_config``
     activates ``deepspeed.zero.Init`` for us, so every parameter is
     immediately partitioned; almost nothing on CPU.
  2. Run an optional model-specific ``prepare_fn`` (e.g.
     ``replace_moe``) *before* loading weights. When a model rewires
     MoE experts such that the live parameter names already match the
     original per-expert checkpoint layout (no fused-to-split conversion
     needed), loading becomes a pure name-to-name streaming copy.
  3. Stream the checkpoint one ``*.safetensors`` shard at a time. For
     every tensor, only rank 0 reads it into host RAM, then a
     ``GatheredParameters([target], modifier_rank=0)`` context broadcasts
     the rank-0 value into the sharded parameter and re-partitions on
     exit. Peak host RAM is bounded by a single tensor, and peak free
     GPU memory by a single full parameter.

This function is generic; it does not know or care about model
architecture. Any ``PreTrainedModel`` whose live parameter names match
the checkpoint keys can be loaded through it.
"""

from __future__ import annotations

import gc
import glob
import json
import os
from typing import Callable, Optional

import torch

from .utils import print_info
from .zero3_utils import gathered_param_if_zero3


def zero3_streaming_from_pretrained(
    model_path: str,
    *,
    torch_dtype="auto",
    trust_remote_code: bool = True,
    use_cache: bool = False,
    attn_implementation: str = "default",
    prepare_fn: Optional[Callable[[torch.nn.Module], None]] = None,
    log_prefix: str = "[zero3_streaming]",
) -> torch.nn.Module:
    """Build an empty ZeRO-3 model and stream weights one shard at a time.

    Parameters
    ----------
    model_path
        Local directory containing ``config.json`` and one or more
        ``*.safetensors`` shards (optionally indexed by
        ``model.safetensors.index.json``).
    torch_dtype
        ``"auto"`` (default) reads ``config.torch_dtype``; otherwise
        accepts a ``torch.dtype`` or a dtype name string.
    trust_remote_code, use_cache, attn_implementation
        Forwarded to ``AutoConfig`` / ``AutoModelForCausalLM.from_config``.
    prepare_fn
        Optional callback run after the empty model is constructed but
        before weight loading. Receives the model and may mutate it
        in-place (e.g. replace fused MoE experts with per-expert
        ``nn.Linear``). Must itself be ZeRO-3 aware if it touches
        parameters.
    log_prefix
        Used as a prefix for progress messages (e.g. ``"[Qwen]"``).

    Returns
    -------
    The populated model. Tokenizer / processor loading remains the
    caller's responsibility.
    """
    from safetensors import safe_open
    from transformers import AutoConfig, AutoModelForCausalLM
    from transformers.initialization import no_init_weights, no_tie_weights

    # 1. Resolve dtype. HF accepts "auto" -> read from config.
    config = AutoConfig.from_pretrained(
        model_path, trust_remote_code=trust_remote_code
    )
    if attn_implementation != "default":
        config._attn_implementation = attn_implementation
    if use_cache is not None:
        config.use_cache = use_cache

    if isinstance(torch_dtype, str) and torch_dtype != "auto":
        resolved_dtype = getattr(torch, torch_dtype)
    elif torch_dtype == "auto" or torch_dtype is None:
        resolved_dtype = getattr(config, "torch_dtype", None) or torch.float32
        if isinstance(resolved_dtype, str):
            resolved_dtype = getattr(torch, resolved_dtype)
    else:
        resolved_dtype = torch_dtype

    # 2. Build an empty ZeRO-3 sharded model.
    #    ``from_config`` activates ``deepspeed.zero.Init`` internally
    #    when ``is_deepspeed_zero3_enabled()``. We wrap it in
    #    ``no_init_weights`` / ``no_tie_weights`` to skip the (extremely
    #    expensive for MoE) per-parameter kaiming init + tying.
    print_info(
        f"{log_prefix} Building empty model (dtype={resolved_dtype}) "
        "under deepspeed.zero.Init..."
    )
    with no_init_weights(), no_tie_weights():
        model = AutoModelForCausalLM.from_config(
            config,
            torch_dtype=resolved_dtype,
            trust_remote_code=trust_remote_code,
        )
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 3. Optional pre-load mutation (e.g. replace_moe).
    if prepare_fn is not None:
        print_info(f"{log_prefix} Running prepare_fn before weight load...")
        prepare_fn(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 4. Stream safetensors shards into the sharded model.
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    shard_to_keys = None
    if os.path.isfile(index_path):
        with open(index_path, "r") as f:
            weight_map = json.load(f)["weight_map"]
        # Group keys by shard file for sequential per-file opens.
        shard_to_keys = {}
        for key, shard in weight_map.items():
            shard_to_keys.setdefault(shard, []).append(key)
        shard_files = sorted(shard_to_keys.keys())
    else:
        shard_files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
        if not shard_files:
            raise FileNotFoundError(
                f"No safetensors / index found under {model_path}"
            )

    # Fast name lookups; params take precedence over buffers.
    name_to_param = dict(model.named_parameters())
    name_to_buffer = dict(model.named_buffers())

    try:
        rank = (
            torch.distributed.get_rank()
            if torch.distributed.is_initialized()
            else 0
        )
    except Exception:  # noqa: BLE001
        rank = 0

    loaded = 0
    skipped = 0
    for shard in shard_files:
        shard_path = (
            shard if os.path.isabs(shard) else os.path.join(model_path, shard)
        )
        if shard_to_keys is not None:
            keys_in_shard = shard_to_keys[shard]
        else:
            with safe_open(shard_path, framework="pt") as reader:
                keys_in_shard = list(reader.keys())

        with safe_open(shard_path, framework="pt") as reader:
            for key in keys_in_shard:
                tgt = name_to_param.get(key)
                is_buffer = False
                if tgt is None:
                    tgt = name_to_buffer.get(key)
                    is_buffer = tgt is not None
                if tgt is None:
                    skipped += 1
                    continue

                # Only rank 0 reads the tensor bytes; other ranks pass
                # ``None`` to ``GatheredParameters(modifier_rank=0)``
                # which broadcasts the rank-0 data on exit.
                src = None
                if rank == 0:
                    src = reader.get_tensor(key)

                if is_buffer:
                    # Buffers are not ZeRO-3 managed: broadcast manually.
                    if torch.distributed.is_initialized():
                        if rank == 0:
                            src_gpu = src.to(device=tgt.device, dtype=tgt.dtype)
                        else:
                            src_gpu = torch.empty_like(tgt)
                        torch.distributed.broadcast(src_gpu, src=0)
                        tgt.data.copy_(src_gpu)
                        del src_gpu
                    else:
                        tgt.data.copy_(src.to(device=tgt.device, dtype=tgt.dtype))
                else:
                    with gathered_param_if_zero3(tgt, modifier_rank=0):
                        if rank == 0:
                            tgt.data.copy_(
                                src.to(device=tgt.device, dtype=tgt.dtype)
                            )

                loaded += 1
                del src

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print_info(
            f"{log_prefix} Finished {os.path.basename(shard_path)}"
        )

    print_info(
        f"{log_prefix} Streaming load done: "
        f"{loaded} tensors loaded, {skipped} unused checkpoint keys."
    )

    # Tie weights now that lm_head / embed_tokens are populated.
    try:
        model.tie_weights()
    except Exception as e:  # noqa: BLE001
        print_info(f"{log_prefix} tie_weights skipped: {e}")

    return model

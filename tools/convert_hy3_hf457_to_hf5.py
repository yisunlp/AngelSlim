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

import argparse
import json
import os
import re
import shutil
from pathlib import Path

import torch
from safetensors.torch import safe_open, save_file


EXPERT_WEIGHT_RE = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
    r"(down_proj|gate_proj|up_proj)\.weight$"
)

CONFIG_KEY_ORDER = [
    "architectures",
    "bos_token_id",
    "enable_attention_fp32_softmax",
    "enable_lm_head_fp32",
    "enable_moe_fp32_combine",
    "eod_token_id",
    "eos_token_id",
    "expert_hidden_dim",
    "moe_intermediate_size",
    "first_k_dense_replace",
    "head_dim",
    "hidden_act",
    "hidden_size",
    "initializer_range",
    "intermediate_size",
    "max_position_embeddings",
    "model_type",
    "moe_router_enable_expert_bias",
    "moe_router_use_sigmoid",
    "num_attention_heads",
    "num_experts",
    "num_experts_per_tok",
    "num_hidden_layers",
    "num_key_value_heads",
    "num_shared_experts",
    "output_router_logits",
    "pad_token_id",
    "qk_norm",
    "rms_norm_eps",
    "rope_parameters",
    "route_norm",
    "router_scaling_factor",
    "sep_token_id",
    "tie_word_embeddings",
    "transformers_version",
    "use_cache",
    "use_grouped_mm",
    "vocab_size",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Convert a Hunyuan HY3 Transformers 4.57 checkpoint to the "
            "Transformers 5.x checkpoint layout."
        )
    )
    parser.add_argument("input_path", type=Path, help="Transformers 4.57 model directory")
    parser.add_argument("output_path", type=Path, help="Transformers 5.x output directory")
    parser.add_argument(
        "--transformers-version",
        default="5.7.0.dev0",
        help="Value written to config.json:transformers_version",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove output_path first if it already exists",
    )
    parser.add_argument(
        "--keep-backup-files",
        action="store_true",
        help="Also copy *.bak and *.bak_* files from input_path",
    )
    parser.add_argument(
        "--num-experts",
        type=int,
        default=None,
        help="Override num_experts. Defaults to config/index inference.",
    )
    return parser.parse_args()


def shard_sort_key(name):
    match = re.match(r"model-(\d+)-of-(\d+)\.safetensors$", name)
    if match:
        return int(match.group(1)), int(match.group(2))
    return 10**9, name


def load_index(input_path):
    index_path = input_path / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing index file: {index_path}")
    with index_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def infer_num_hidden_layers(weight_map):
    layers = []
    for key in weight_map:
        match = re.search(r"model\.layers\.(\d+)\.", key)
        if match:
            layers.append(int(match.group(1)))
    if not layers:
        raise ValueError("Could not infer num_hidden_layers from model index")
    return max(layers) + 1


def infer_num_experts(weight_map):
    experts = []
    for key in weight_map:
        match = EXPERT_WEIGHT_RE.match(key)
        if match:
            experts.append(int(match.group(2)))
    if not experts:
        return None
    return max(experts) + 1


def list_shards(weight_map):
    shards = sorted(set(weight_map.values()), key=shard_sort_key)
    if not shards:
        raise ValueError("No safetensors shards found in model index")
    return shards


def token_id_from_tokenizer(input_path, special_token_name):
    special_path = input_path / "special_tokens_map.json"
    tokenizer_config_path = input_path / "tokenizer_config.json"
    if not special_path.exists() or not tokenizer_config_path.exists():
        return None

    with special_path.open("r", encoding="utf-8") as f:
        special_tokens = json.load(f)
    token = special_tokens.get(special_token_name)
    if token is None:
        return None

    with tokenizer_config_path.open("r", encoding="utf-8") as f:
        tokenizer_config = json.load(f)
    decoder = tokenizer_config.get("added_tokens_decoder", {})
    for token_id, token_info in decoder.items():
        if token_info.get("content") == token:
            return int(token_id)
    return None


def ordered_config(config):
    ordered = {}
    for key in CONFIG_KEY_ORDER:
        if key in config:
            ordered[key] = config[key]
    for key in sorted(config):
        if key not in ordered:
            ordered[key] = config[key]
    return ordered


def convert_config(input_path, output_path, weight_map, transformers_version):
    config_path = input_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config file: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    if "rope_theta" in config:
        rope_theta = config.pop("rope_theta")
        config["rope_parameters"] = {
            "rope_theta": rope_theta,
            "rope_type": "default",
        }
    elif "rope_parameters" in config:
        config["rope_parameters"].setdefault("rope_type", "default")

    if "expert_hidden_dim" in config:
        config.setdefault("moe_intermediate_size", config["expert_hidden_dim"])

    config["num_hidden_layers"] = infer_num_hidden_layers(weight_map)
    config["transformers_version"] = transformers_version
    config.setdefault("enable_attention_fp32_softmax", False)
    config.setdefault("enable_moe_fp32_combine", False)

    bos_token_id = token_id_from_tokenizer(input_path, "bos_token")
    pad_token_id = token_id_from_tokenizer(input_path, "pad_token")
    if bos_token_id is not None:
        config.setdefault("bos_token_id", bos_token_id)
    if pad_token_id is not None:
        config.setdefault("pad_token_id", pad_token_id)

    # This flag was used by the 4.57 implementation and is not present in the
    # 5.x reference config.
    config.pop("attn_impl", None)

    with (output_path / "config.json").open("w", encoding="utf-8") as f:
        json.dump(ordered_config(config), f, indent=2, ensure_ascii=False)
        f.write("\n")


def should_copy_extra(path, keep_backup_files):
    name = path.name
    if name == "config.json" or name == "model.safetensors.index.json":
        return False
    if name.endswith(".safetensors"):
        return False
    if name.startswith("."):
        return False
    if not keep_backup_files and (".bak" in name or name.endswith(".bak")):
        return False
    return True


def copy_extra_files(input_path, output_path, keep_backup_files):
    for path in input_path.iterdir():
        if not should_copy_extra(path, keep_backup_files):
            continue
        dst = output_path / path.name
        if path.is_dir():
            shutil.copytree(path, dst)
        else:
            shutil.copy2(path, dst)


def rename_key(key):
    key = key.replace(".mlp.router.gate.weight", ".mlp.gate.weight")
    key = key.replace(".mlp.expert_bias", ".mlp.e_score_correction_bias")
    key = key.replace(".mlp.shared_mlp.", ".mlp.shared_experts.")
    return key


def tensor_nbytes(tensor):
    return tensor.numel() * tensor.element_size()


def put_tensor(state_dict, key, tensor):
    if key in state_dict:
        raise ValueError(f"Duplicate output tensor key: {key}")
    state_dict[key] = tensor


def pack_layer_experts(reader, key_set, layer, num_experts):
    prefix = f"model.layers.{layer}.mlp.experts"
    down0 = reader.get_tensor(f"{prefix}.0.down_proj.weight")
    gate0 = reader.get_tensor(f"{prefix}.0.gate_proj.weight")
    up0 = reader.get_tensor(f"{prefix}.0.up_proj.weight")
    down_shape = down0.shape
    gate_shape = gate0.shape
    up_shape = up0.shape

    if gate_shape != up_shape:
        raise ValueError(f"Layer {layer}: gate_proj and up_proj shapes differ")

    down = torch.empty((num_experts, *down_shape), dtype=down0.dtype)
    gate_up = torch.empty(
        (num_experts, gate_shape[0] + up_shape[0], gate_shape[1]),
        dtype=gate0.dtype,
    )

    for expert_id in range(num_experts):
        down_key = f"{prefix}.{expert_id}.down_proj.weight"
        gate_key = f"{prefix}.{expert_id}.gate_proj.weight"
        up_key = f"{prefix}.{expert_id}.up_proj.weight"
        missing_keys = [key for key in (down_key, gate_key, up_key) if key not in key_set]
        if missing_keys:
            raise KeyError(f"Layer {layer}: missing expert tensors: {missing_keys[:3]}")

        down_tensor = reader.get_tensor(down_key)
        gate_tensor = reader.get_tensor(gate_key)
        up_tensor = reader.get_tensor(up_key)
        if down_tensor.shape != down_shape:
            raise ValueError(f"{down_key}: shape {tuple(down_tensor.shape)} != {tuple(down_shape)}")
        if gate_tensor.shape != gate_shape or up_tensor.shape != up_shape:
            raise ValueError(f"Layer {layer}, expert {expert_id}: inconsistent gate/up shape")

        down[expert_id].copy_(down_tensor)
        gate_up[expert_id, : gate_shape[0]].copy_(gate_tensor)
        gate_up[expert_id, gate_shape[0] :].copy_(up_tensor)

    return {
        f"{prefix}.down_proj": down,
        f"{prefix}.gate_up_proj": gate_up,
    }


def convert_shard(input_path, output_path, shard_name, num_experts):
    input_file = input_path / shard_name
    output_file = output_path / shard_name
    temp_file = output_file.with_suffix(output_file.suffix + ".tmp")
    state_dict = {}

    with safe_open(input_file, framework="pt", device="cpu") as reader:
        keys = list(reader.keys())
        key_set = set(keys)
        expert_layers = sorted(
            {int(match.group(1)) for key in keys if (match := EXPERT_WEIGHT_RE.match(key))}
        )

        for key in keys:
            if EXPERT_WEIGHT_RE.match(key):
                continue
            put_tensor(state_dict, rename_key(key), reader.get_tensor(key))

        for layer in expert_layers:
            for packed_key, packed_tensor in pack_layer_experts(
                reader, key_set, layer, num_experts
            ).items():
                put_tensor(state_dict, packed_key, packed_tensor)

    save_file(state_dict, temp_file)
    os.replace(temp_file, output_file)

    shard_index = {key: shard_name for key in state_dict}
    shard_nbytes = sum(tensor_nbytes(tensor) for tensor in state_dict.values())
    return shard_index, shard_nbytes


def prepare_output_dir(input_path, output_path, overwrite):
    input_path = input_path.resolve()
    output_path = output_path.resolve()
    if input_path == output_path:
        raise ValueError("input_path and output_path must be different")
    if output_path.exists():
        if not overwrite and any(output_path.iterdir()):
            raise FileExistsError(
                f"{output_path} already exists and is not empty. Pass --overwrite to replace it."
            )
        if overwrite:
            shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)


def write_index(output_path, weight_map, total_size):
    index = {
        "metadata": {"total_size": total_size},
        "weight_map": dict(sorted(weight_map.items())),
    }
    with (output_path / "model.safetensors.index.json").open("w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)
        f.write("\n")


def main():
    args = parse_args()
    input_path = args.input_path.resolve()
    output_path = args.output_path.resolve()

    source_index = load_index(input_path)
    source_weight_map = source_index["weight_map"]
    num_experts = args.num_experts or infer_num_experts(source_weight_map)
    if num_experts is None:
        raise ValueError("Could not infer num_experts from model index; pass --num-experts")

    prepare_output_dir(input_path, output_path, args.overwrite)
    copy_extra_files(input_path, output_path, args.keep_backup_files)
    convert_config(input_path, output_path, source_weight_map, args.transformers_version)

    output_weight_map = {}
    total_size = 0
    shards = list_shards(source_weight_map)
    for idx, shard_name in enumerate(shards, 1):
        print(f"[{idx}/{len(shards)}] converting {shard_name}", flush=True)
        shard_index, shard_nbytes = convert_shard(input_path, output_path, shard_name, num_experts)
        output_weight_map.update(shard_index)
        total_size += shard_nbytes

    write_index(output_path, output_weight_map, total_size)
    print(f"Done. Wrote Transformers 5.x checkpoint to {output_path}")


if __name__ == "__main__":
    main()

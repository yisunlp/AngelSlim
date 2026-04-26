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

import torch
from safetensors.torch import save_file

from ...utils import (
    gathered_param_if_zero3,
    model_has_zero3_params,
    print_info,
    save_via_model_save_func,
    set_op_by_name,
)
from ..compressor_factory import CompressorFactory
from ..quant.modules.helper_layer import QDQModule
from .modules.quantizer import QuantLinear
from .plugins.plugin_manager import PluginManager
from .trainers.trainer_factory import TrainerFactory

__all__ = ["QAT"]


@CompressorFactory.register
class QAT:
    def __init__(self, model, slim_config=None):
        self.quant_model = model
        self.config = slim_config
        self.training_mode = slim_config["compress_config"].QAT.training_mode.lower()
        self.save_fmt = slim_config["compress_config"].QAT.save_format
        self.plugin_config = slim_config["compress_config"].QAT.plugin_config
        self.quant_model.init_ptq(slim_config)
        self.quant_info = self.quant_model.quant_config
        self.plugin_manager = PluginManager()
        self._init_plugins()
        self._init_trainer()

    def _init_plugins(self):
        # Register learnable rotation plugin
        if self.plugin_config.get("enable_rotation", False):
            self.plugin_manager.register_plugin(
                "learnable_rotation",
                config=self.plugin_config.get("rotation_config", {}),
                quant_model=self.quant_model,
            )

        # Register learnable scale plugin
        if self.plugin_config.get("enable_scale", False):
            self.plugin_manager.register_plugin(
                "learnable_scale",
                quant_info=self.quant_info,
                ignore_layers=self.config["compress_config"].quantization.ignore_layers,
                resume_ckpt_dir=self.config["compress_config"].QAT.resume_ckpt_dir,
                from_ptq_ckpt_dir=self.config["compress_config"].QAT.from_ptq_ckpt,
                config=self.plugin_config.get("quant_config", {}),
                quant_model=self.quant_model,
            )

    def _init_trainer(self):
        self.trainer = TrainerFactory.create(
            training_mode=self.training_mode,
            quant_model=self.quant_model,
            config=self.config,
            plugin_manager=self.plugin_manager,
        )

    def run(self, dataloader):
        self.trainer.run(dataloader)

    @staticmethod
    def _gather_clone(tensor):
        """Detach + CPU-clone a tensor, gathering if it is a ZeRO-3 shard."""
        if tensor is None:
            return None
        with gathered_param_if_zero3(tensor):
            return tensor.detach().cpu().clone()

    def convert(self):
        if self.save_fmt not in ("real", "real_and_kvcache"):
            return

        print_info("Start QAT convert: replacing QuantLinear with QDQModule...")
        quant_algo = self.quant_info.quant_algo

        quant_linear_modules = [
            (name, module)
            for name, module in self.quant_model.model.named_modules()
            if isinstance(module, QuantLinear)
        ]

        for name, module in quant_linear_modules:
            weight = self._gather_clone(module.weight)
            bias = self._gather_clone(getattr(module, "bias", None))

            weight_scale = None
            if hasattr(module, "weight_quantizer"):
                weight_scale = self._gather_clone(module.weight_quantizer.scale)

            input_scale = None
            if module.use_act_quant and hasattr(module, "act_quantizer"):
                act_quantizer = module.act_quantizer
                if hasattr(act_quantizer, "scale") and act_quantizer.scale is not None:
                    input_scale = self._gather_clone(act_quantizer.scale)

            qdq_module = QDQModule(
                quant_algo=quant_algo,
                weight=weight,
                weight_scale=weight_scale,
                bias=bias,
                group_size=(
                    module.weight_quantizer.group_size
                    if hasattr(module, "weight_quantizer")
                    and hasattr(module.weight_quantizer, "group_size")
                    else 128
                ),
                input_scale=input_scale,
            )
            set_op_by_name(self.quant_model.model, name, qdq_module)

    def _save_kv_cache_scales(self, save_path: str):
        """Extract and save KV cache scales to a safetensors file."""
        kv_scales = {}
        for name, module in self.quant_model.model.named_modules():
            if not isinstance(module, QuantLinear):
                continue
            if not module.use_qkv_quant or not hasattr(module, "qkv_quantizer"):
                continue
            # Map k_proj / v_proj to k_cache / v_cache
            if name.endswith(".k_proj"):
                cache_name = name.replace(".k_proj", ".k_cache")
            elif name.endswith(".v_proj"):
                cache_name = name.replace(".v_proj", ".v_cache")
            else:
                continue
            scale_key = f"{cache_name}.scale"
            scale_tensor = self._gather_clone(module.qkv_quantizer.scale)
            if scale_tensor is not None:
                kv_scales[scale_key] = scale_tensor.float()

        if model_has_zero3_params(self.quant_model.model):
            # Only rank0 writes the file.
            rank = (
                torch.distributed.get_rank()
                if torch.distributed.is_initialized() else 0
            )
            if rank != 0:
                return

        os.makedirs(save_path, exist_ok=True)
        out_file = os.path.join(save_path, "kv_cache_scales.safetensors")
        save_file(kv_scales, out_file)
        print_info(f"Saved {len(kv_scales)} KV cache scales to: {out_file}")

    def save(self, save_path: str):
        # "fake": save fake-quant state_dict (NOTE: only supports non-distributed / single-GPU)
        if self.save_fmt == "fake":
            parts = save_path.rsplit("/")
            save_path = os.path.join("/".join(parts[:-1]), parts[-1] + "_fake_quant_model.pt")
            print_info(f"Start save QAT fake ckpt to: {save_path}")

            cpu_state = self.trainer.external_trainer.model.state_dict()
            torch.save(cpu_state, save_path)

        # "real": save real-quant model via model-specific save function
        elif self.save_fmt == "real":
            save_func = self.quant_model.get_save_func()(self.quant_model)
            save_via_model_save_func(
                self.quant_model,
                save_func,
                os.path.join(save_path, "final_quant_checkpoint"),
            )

        # "save_kvcache_only": only export KV cache scales (kv_cache_scales.safetensors)
        elif self.save_fmt == "save_kvcache_only":
            self._save_kv_cache_scales(os.path.join(save_path, "final_quant_checkpoint"))

        # "real_and_kvcache": save real-quant model AND KV cache scales
        elif self.save_fmt == "real_and_kvcache":
            save_func = self.quant_model.get_save_func()(self.quant_model)
            save_via_model_save_func(
                self.quant_model,
                save_func,
                os.path.join(save_path, "final_quant_checkpoint"),
            )
            self._save_kv_cache_scales(os.path.join(save_path, "final_quant_checkpoint"))

        else:
            print_info("Save format not specified, skip save.")

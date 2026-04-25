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

from ...utils import gathered_param_if_zero3, is_zero3_param, print_info, set_op_by_name
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
    def _z3_clone(tensor):
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
            # One parameter at a time: GPU peak stays at a single layer's size.
            weight = self._z3_clone(module.weight)
            bias = self._z3_clone(getattr(module, "bias", None))

            weight_scale = None
            if hasattr(module, "weight_quantizer"):
                weight_scale = self._z3_clone(module.weight_quantizer.scale)

            input_scale = None
            if module.use_act_quant and hasattr(module, "act_quantizer"):
                act_quantizer = module.act_quantizer
                if hasattr(act_quantizer, "scale") and act_quantizer.scale is not None:
                    input_scale = self._z3_clone(act_quantizer.scale)

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
            kv_scales[scale_key] = (
                self._z3_clone(module.qkv_quantizer.scale).float().cpu()
            )

        os.makedirs(save_path, exist_ok=True)
        out_file = os.path.join(save_path, "kv_cache_scales.safetensors")
        save_file(kv_scales, out_file)
        print_info(f"Saved {len(kv_scales)} KV cache scales to: {out_file}")

    def _model_has_zero3_params(self):
        return any(is_zero3_param(p) for p in self.quant_model.model.parameters())

    def _build_consolidated_state_dict_manual(self):
        """Gather ZeRO-3 parameters one at a time into a rank-0 CPU state_dict."""
        try:
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        except Exception:  # noqa: BLE001
            rank = 0

        state_dict = {}
        model = self.quant_model.model

        for name, param in model.named_parameters():
            with gathered_param_if_zero3(param):
                if rank == 0:
                    state_dict[name] = param.detach().cpu().clone()

        # Persistent buffers (non-persistent ones are excluded, matching
        # ``nn.Module.state_dict`` semantics).
        persistent_buffers = {}
        for module_name, module in model.named_modules():
            for buf_name, buf in module.named_buffers(recurse=False):
                if buf is None:
                    continue
                if buf_name in module._non_persistent_buffers_set:
                    continue
                full_key = f"{module_name}.{buf_name}" if module_name else buf_name
                persistent_buffers[full_key] = buf
        if rank == 0:
            for name, buf in persistent_buffers.items():
                state_dict[name] = buf.detach().cpu().clone()

        return state_dict if rank == 0 else {}

    def _invoke_save(self, save_func, save_target_dir):
        if not self._model_has_zero3_params():
            save_func.save(save_target_dir)
            return

        consolidated_sd = self._build_consolidated_state_dict_manual()

        try:
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        except Exception:  # noqa: BLE001
            rank = 0

        hf_model = self.quant_model.get_model()
        original_state_dict = hf_model.state_dict

        def _patched_state_dict(*args, **kwargs):
            return consolidated_sd if rank == 0 else {}

        try:
            hf_model.state_dict = _patched_state_dict  # type: ignore[method-assign]
            if rank == 0:
                save_func.save(save_target_dir)
        finally:
            hf_model.state_dict = original_state_dict  # type: ignore[method-assign]

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

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
            self._invoke_save(save_func, os.path.join(save_path, "final_quant_checkpoint"))

        # "save_kvcache_only": only export KV cache scales (kv_cache_scales.safetensors)
        elif self.save_fmt == "save_kvcache_only":
            self._save_kv_cache_scales(os.path.join(save_path, "final_quant_checkpoint"))

        # "real_and_kvcache": save real-quant model AND KV cache scales
        elif self.save_fmt == "real_and_kvcache":
            save_func = self.quant_model.get_save_func()(self.quant_model)
            self._invoke_save(save_func, os.path.join(save_path, "final_quant_checkpoint"))
            self._save_kv_cache_scales(os.path.join(save_path, "final_quant_checkpoint"))

        else:
            print_info("Save format not specified, skip save.")

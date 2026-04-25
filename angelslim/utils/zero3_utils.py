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

from contextlib import contextmanager, nullcontext

import torch

from .lazy_imports import deepspeed

ZERO3_PARAM_ATTRS = ("ds_id", "ds_status", "ds_numel", "ds_tensor")


def is_deepspeed_zero3_enabled():
    try:
        from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled

        return is_deepspeed_zero3_enabled()
    except Exception:  # noqa: BLE001
        return False


def is_zero3_param(x):
    if not isinstance(x, torch.nn.Parameter):
        return False
    return any(hasattr(x, attr) for attr in ZERO3_PARAM_ATTRS)


@contextmanager
def gathered_param_if_zero3(x, modifier_rank=None):
    if is_zero3_param(x):
        ctx = deepspeed.zero.GatheredParameters([x], modifier_rank=modifier_rank)
    else:
        ctx = nullcontext()
    with ctx:
        yield x


@contextmanager
def gathered_params_if_zero3(params, modifier_rank=None):
    params = [p for p in params if p is not None]
    zero3_params = [p for p in params if is_zero3_param(p)]
    if zero3_params:
        ctx = deepspeed.zero.GatheredParameters(zero3_params, modifier_rank=modifier_rank)
    else:
        ctx = nullcontext()
    with ctx:
        yield params

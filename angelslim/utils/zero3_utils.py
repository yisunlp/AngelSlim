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

"""DeepSpeed ZeRO-3 helpers. No-ops when deepspeed is not installed or the
passed-in parameter is not sharded."""

from contextlib import contextmanager, nullcontext

import torch

from .lazy_imports import deepspeed

# Attributes injected onto a parameter by ``deepspeed.zero.Init``.
ZERO3_PARAM_ATTRS = ("ds_id", "ds_status", "ds_numel", "ds_tensor")


def is_zero3_param(x):
    """Return True iff ``x`` is a parameter sharded by DeepSpeed ZeRO-3."""
    if not isinstance(x, torch.nn.Parameter):
        return False
    return any(hasattr(x, attr) for attr in ZERO3_PARAM_ATTRS)


@contextmanager
def gathered_param_if_zero3(x, modifier_rank=None):
    """All-gather a ZeRO-3 sharded parameter for the duration of the block.

    No-op if ``x`` is not a ZeRO-3 shard, so callers can use unconditionally.
    """
    if is_zero3_param(x):
        ctx = deepspeed.zero.GatheredParameters([x], modifier_rank=modifier_rank)
    else:
        ctx = nullcontext()
    with ctx:
        yield x


@contextmanager
def gathered_params_if_zero3(params, modifier_rank=None):
    """Batched version of :func:`gathered_param_if_zero3`."""
    params = [p for p in params if p is not None]
    zero3_params = [p for p in params if is_zero3_param(p)]
    if zero3_params:
        ctx = deepspeed.zero.GatheredParameters(zero3_params, modifier_rank=modifier_rank)
    else:
        ctx = nullcontext()
    with ctx:
        yield params

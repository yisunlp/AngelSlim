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

# Each LLM adapter may depend on specific ``transformers`` classes. Fall
# back to a soft-skip so that an unsupported adapter (e.g. new fused-MoE
# classes introduced in ``transformers >= 5.0``) does not prevent the
# rest of the package from importing on older releases.

try:
    from .deepseek import DeepSeek  # noqa: F401
except Exception:  # noqa: BLE001
    pass

try:
    from .glm import GLM  # noqa: F401
except Exception:  # noqa: BLE001
    pass

try:
    from .hunyuan_dense import HunyuanDense  # noqa: F401
except Exception:  # noqa: BLE001
    pass

try:
    from .hunyuan_moe import HunyuanMoE  # noqa: F401
except Exception:  # noqa: BLE001
    pass

try:
    from .hunyuan_v3_moe import HYV3MoE  # noqa: F401
except Exception:  # noqa: BLE001
    pass

try:
    from .kimi_k2 import KimiK2  # noqa: F401
except Exception:  # noqa: BLE001
    pass

try:
    from .llama import Llama  # noqa: F401
except Exception:  # noqa: BLE001
    pass

try:
    from .qwen import Qwen  # noqa: F401
except Exception:  # noqa: BLE001
    pass

try:
    from .seed_oss import SeedOss  # noqa: F401
except Exception:  # noqa: BLE001
    pass

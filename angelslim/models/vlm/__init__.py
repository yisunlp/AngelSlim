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

# Each adapter may depend on a specific ``transformers`` version. Fall
# back to a soft-skip so that a missing VLM module does not break LLM
# pipelines on older ``transformers`` releases (e.g. 4.57 lacks
# ``qwen3_5_moe`` / ``qwen3_vl_moe`` / ``qwen3_vl``).

try:
    from .hunyuan_vl import HunyuanVL  # noqa: F401
except Exception:  # noqa: BLE001
    pass

try:
    from .qwen3_5 import Qwen3_5  # noqa: F401
except Exception:  # noqa: BLE001
    pass

try:
    from .qwen3_vl import Qwen3VL  # noqa: F401
except Exception:  # noqa: BLE001
    pass

try:
    from .qwen3_vl_moe import Qwen3VLMoE  # noqa: F401
except Exception:  # noqa: BLE001
    pass

try:
    from .qwen_vl import QwenVL  # noqa: F401
except Exception:  # noqa: BLE001
    pass

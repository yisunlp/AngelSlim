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

import json
import os
from typing import Dict, List

import pyarrow.parquet as pq
import torch
from datasets import load_dataset
from transformers import ProcessorMixin

from .base_dataset import BaseDataset


class TextDataset(BaseDataset):
    """Dataset for text-only data in Parquet or JSONL formats"""

    def __init__(
        self,
        data_path: str,
        processor: ProcessorMixin,
        device: str = "cpu",
        max_length: int = 4096,
        num_samples: int = -1,
    ):
        super().__init__(processor, device, max_length)
        self._load_data(data_path, num_samples)

    def _load_data(self, data_path: str, num_samples: int):
        if not os.path.isfile(data_path):
            self._load_hf_dataset(data_path, num_samples)
            return

        if ".parquet" in data_path.lower():
            self._load_parquet_data(data_path, num_samples)
        else:
            self._load_jsonl_data(data_path, num_samples)

    def _load_hf_dataset(self, data_path: str, num_samples: int, block_size: int = 2048):
        parts = data_path.split(",")
        dataset = load_dataset(*parts)["train"]
        total_samples = (
            min(num_samples, len(dataset["text"])) if num_samples > 0 else len(dataset["text"])
        )

        concatenated = {}
        for sample in dataset:
            tokenized = self.processor(sample["text"])
            for key in tokenized.keys():
                if key not in concatenated:
                    concatenated[key] = []
                concatenated[key].extend(tokenized[key])

        total = len(concatenated["input_ids"])
        if total >= block_size:
            total = (total // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total, block_size)]
            for k, t in concatenated.items()
        }
        for i in range(total_samples):
            inputs = {
                "input_ids": torch.tensor(result["input_ids"][i]).unsqueeze(0).to(self.device)
            }
            labels = inputs["input_ids"].roll(shifts=-1, dims=-1)
            labels[:, -1] = -100
            inputs["labels"] = labels.to(self.device)
            inputs["attention_mask"] = torch.tensor(result["attention_mask"][i]).to(self.device)
            self.data.append(inputs)

    def _load_parquet_data(self, data_path: str, num_samples: int):
        table = pq.read_table(data_path)
        df = table.to_pandas()

        # Handle sample limits
        total_samples = min(num_samples, len(df)) if num_samples > 0 else len(df)

        for i in range(total_samples):
            text = df["text"].iloc[i]
            model_inputs = self.processor(
                [text],
                return_tensors="pt",
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
            )

            # Handle potential labels
            if "labels" in df.columns:
                labels = torch.tensor(df["labels"].iloc[i]).unsqueeze(0)
            else:
                labels = model_inputs["input_ids"].roll(shifts=-1, dims=-1)
                labels[:, -1] = -100

            data_item = {
                "input_ids": model_inputs["input_ids"].to(self.device),
                "attention_mask": model_inputs["attention_mask"].to(self.device),
                "labels": labels.to(self.device),
            }
            self.data.append(data_item)

    # Chat-template role markers used to locate the last assistant turn so
    # that only its content participates in the loss. Covers hunyuan
    # (``<｜hy_*｜>``) and qwen / chatml (``<|im_*|>``) families.
    _ASSISTANT_HEADERS = (
        "<｜hy_Assistant｜>",
        "<|im_start|>assistant\n",
        "<|im_start|>assistant",
        "<|assistant|>",
    )
    _TURN_END_MARKERS = (
        "<｜hy_place▁holder▁no▁2｜>",
        "<eos:6124c78e>",
        "<|im_end|>",
        "<|endoftext|>",
    )

    def _find_last_assistant_span(self, text: str):
        """Return ``(start, end)`` char offsets of the LAST assistant content,
        or ``None`` if no assistant header is present."""
        header_used, header_pos = None, -1
        for header in self._ASSISTANT_HEADERS:
            pos = text.rfind(header)
            if pos > header_pos:
                header_pos, header_used = pos, header
        if header_pos < 0:
            return None

        content_start = header_pos + len(header_used)
        # Include the turn-end marker so the model learns to emit it.
        content_end = len(text)
        for end_marker in self._TURN_END_MARKERS:
            pos = text.find(end_marker, content_start)
            if pos >= 0:
                stop = pos + len(end_marker)
                if stop < content_end:
                    content_end = stop
        return content_start, content_end

    def _build_content_mask(self, text: str, tokenized) -> List[int]:
        """Per-token 0/1 mask: 1 on tokens of the LAST assistant content.

        Falls back to all-ones if offsets are unavailable or no header is
        found (i.e. train on the whole sequence, legacy behaviour).
        """
        seq_len = int(tokenized["input_ids"].shape[-1])
        span = self._find_last_assistant_span(text)
        offsets = tokenized.get("offset_mapping")
        if span is None or offsets is None:
            return [1] * seq_len

        content_start, content_end = span
        offsets = offsets[0].tolist() if hasattr(offsets[0], "tolist") else list(offsets[0])
        mask = [0] * seq_len
        for i, (s, e) in enumerate(offsets):
            if s == e:  # special token
                continue
            if s >= content_start and e <= content_end:
                mask[i] = 1
        return mask

    def _process_text(self, text: str) -> None:
        """Tokenise ``text`` (chat-template already applied) into
        ``input_ids`` / ``attention_mask`` / ``labels`` / ``loss_mask``. Only
        the LAST assistant segment contributes to the loss."""
        model_inputs = self.processor(
            text=[text],
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_offsets_mapping=True,
        )
        content_mask = torch.tensor(
            self._build_content_mask(text, model_inputs), dtype=torch.long
        ).unsqueeze(0)
        model_inputs.pop("offset_mapping", None)

        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]
        # Next-token prediction: shift labels and mask by one.
        labels = input_ids.roll(shifts=-1, dims=-1)
        labels[:, -1] = -100
        loss_mask = content_mask.roll(shifts=-1, dims=-1)
        loss_mask[:, -1] = 0
        loss_mask = loss_mask * attention_mask  # drop padding positions

        self.data.append(
            {
                "input_ids": input_ids.to(self.device),
                "attention_mask": attention_mask.to(self.device),
                "labels": labels.to(self.device),
                "loss_mask": loss_mask.to(self.device),
            }
        )

    def _load_jsonl_data(self, data_path: str, num_samples: int):
        line_count = 0
        with open(data_path, "r") as f:
            for line in f:
                if num_samples > 0 and line_count >= num_samples:
                    break

                data = json.loads(line)
                assert (
                    "messages" in data
                    or "input" in data
                    or "conversations" in data
                    or "applied_message" in data
                ), "JSON format error"

                # Pre-rendered chat string — use as-is.
                if "applied_message" in data:
                    self._process_text(data["applied_message"])
                    line_count += 1
                    continue

                # Structured messages — render via chat template. Use
                # ``add_generation_prompt=False`` so the rendered text ends
                # with the last assistant's actual content (plus its turn-end
                # marker); otherwise the template appends an empty assistant
                # header and ``_find_last_assistant_span`` would pick up that
                # empty span, producing an all-zero loss_mask.
                messages = self._prepare_messages(data)
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )

                thinking_data = False
                for dic in messages:
                    if dic["role"] == "assistant":
                        if "<think>" and "</think>" in dic["content"]:
                            thinking_data = True
                            break
                if thinking_data:
                    text = self.processor.bos_token if self.processor.bos_token is not None else ""
                    for dic in messages:
                        if dic["role"] == "system":
                            text += dic["content"]
                        elif dic["role"] == "user":
                            text = text + "<｜User｜>" + dic["content"] + "<｜Assistant｜>"
                        elif dic["role"] == "assistant":
                            text = text + dic["content"] + self.processor.eos_token

                self._process_text(text)
                line_count += 1

    def _prepare_messages(self, data: Dict) -> List[Dict]:
        """Prepare chat messages from data entry"""
        if "messages" in data:
            messages = data["messages"]
            # Add system prompt if available
            if (
                "system_prompt" in data
                and data["system_prompt"]
                and messages[0]["role"] != "system"
            ):
                messages = [{"role": "system", "content": data["system_prompt"]}] + messages
        elif "conversations" in data:
            share_gpt_data = data["conversations"]
            messages = [
                {"role": "user", "content": share_gpt_data[0]["value"]},
                {"role": "assistant", "content": share_gpt_data[1]["value"]},
            ]
            if "system" in data and data["system"]:
                messages = [{"role": "system", "content": data["system_prompt"]}] + messages
        else:
            messages = [
                {"role": "user", "content": data["input"]},
                {"role": "assistant", "content": data["output"]},
            ]
            if "system_prompt" in data and data["system_prompt"]:
                messages = [{"role": "system", "content": data["system_prompt"]}] + messages

        # Normalize role names
        for item in messages:
            if "role" not in item and "from" in item:
                item["role"] = item["from"]
            if "content" not in item and "value" in item:
                item["content"] = item["value"]
            role = item["role"]
            if "human" in role:
                item["role"] = "user"
            elif "gpt" in role:
                item["role"] = "assistant"

        return messages

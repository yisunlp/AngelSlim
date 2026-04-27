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
            # HF CausalLM models shift labels internally; feed labels == input_ids.
            inputs["labels"] = inputs["input_ids"].clone()
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
                # HF CausalLM models shift labels internally; feed labels == input_ids.
                labels = model_inputs["input_ids"].clone()

            data_item = {
                "input_ids": model_inputs["input_ids"].to(self.device),
                "attention_mask": model_inputs["attention_mask"].to(self.device),
                "labels": labels.to(self.device),
            }
            self.data.append(data_item)

    def _load_jsonl_data(self, data_path: str, num_samples: int):
        line_count = 0
        with open(data_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if num_samples > 0 and line_count >= num_samples:
                    break

                data = json.loads(line)

                # Schema dispatch ------------------------------------------------
                # 1) ``applied_message``: a single string already wrapped with the
                #    chat template. We tokenize it directly and only supervise
                #    the tokens AFTER the last assistant marker
                #    (``<｜hy_Assistant｜>`` or ``<|im_start|>assistant``).
                # 2) ``messages`` / ``conversations`` / ``input``: structured
                #    schemas; we run them through ``apply_chat_template`` and
                #    supervise only the last assistant turn.
                if "applied_message" in data:
                    item = self._tokenize_applied_message(data["applied_message"])
                    if item is not None:
                        self.data.append(item)
                        line_count += 1
                    continue

                # Validate format
                assert (
                    "messages" in data or "input" in data or "conversations" in data
                ), "JSON format error"

                # Prepare messages
                messages = self._prepare_messages(data)

                # Find the LAST assistant turn — loss is computed ONLY on
                # this reply. Everything before it (system + user(s) +
                # earlier assistant(s)) serves as prompt context.
                last_assistant_idx = None
                for idx, item in enumerate(messages):
                    if item["role"] == "assistant":
                        last_assistant_idx = idx
                if last_assistant_idx is None:
                    # No assistant turn -> nothing to supervise; skip.
                    continue
                prompt_messages = messages[:last_assistant_idx]
                assistant_msg = messages[last_assistant_idx]

                # Tokenize the prompt (up to the generation marker) and the
                # full conversation separately so we know exactly where the
                # assistant reply starts.
                prompt_text = self.processor.apply_chat_template(
                    prompt_messages, tokenize=False, add_generation_prompt=True
                )
                full_messages = prompt_messages + [assistant_msg]
                full_text = self.processor.apply_chat_template(
                    full_messages, tokenize=False, add_generation_prompt=False
                )

                # Legacy branch: thinking-style data without a chat template.
                thinking_data = any(
                    m["role"] == "assistant"
                    and "<think>" in m.get("content", "")
                    and "</think>" in m.get("content", "")
                    for m in messages
                )
                if thinking_data:
                    bos = self.processor.bos_token or ""
                    prompt_text = bos
                    for m in prompt_messages:
                        if m["role"] == "system":
                            prompt_text += m["content"]
                        elif m["role"] == "user":
                            prompt_text += "<｜User｜>" + m["content"] + "<｜Assistant｜>"
                        elif m["role"] == "assistant":
                            prompt_text += m["content"] + self.processor.eos_token
                    full_text = prompt_text + assistant_msg["content"] + self.processor.eos_token

                # Token-level prompt length: count tokens in ``prompt_text``
                # without special-token insertion so it aligns with the
                # prefix of the tokenization of ``full_text``.
                prompt_ids = self.processor(
                    prompt_text,
                    add_special_tokens=False,
                    return_tensors=None,
                )["input_ids"]
                prompt_len = len(prompt_ids)

                model_inputs = self.processor(
                    text=[full_text],
                    return_tensors="pt",
                    max_length=self.max_length,
                    truncation=True,
                    padding="max_length",
                )

                # Build labels: HF CausalLM shifts labels internally, so
                # the label at position ``t`` supervises the prediction of
                # ``input_ids[t+1]``. Positions before (and at) the end of
                # the prompt are set to -100 so they contribute no loss.
                input_ids = model_inputs["input_ids"]
                attention_mask = model_inputs["attention_mask"]
                labels = input_ids.clone()
                labels[:, :prompt_len] = -100
                # Also mask padding tokens.
                labels[attention_mask == 0] = -100

                self.data.append(
                    {
                        "input_ids": input_ids.to(self.device),
                        "attention_mask": attention_mask.to(self.device),
                        "labels": labels.to(self.device),
                    }
                )

                line_count += 1

    # ------------------------------------------------------------------
    # Schema: ``applied_message`` (already chat-templated single string)
    # ------------------------------------------------------------------

    # Common assistant-turn markers across chat templates we may encounter.
    # Each entry is matched as a *substring* on the raw text; the FIRST
    # marker found at the rightmost position (= last assistant turn) wins.
    # ``include_marker_in_loss=False`` means the marker token itself is
    # part of the prompt (label = -100); only tokens AFTER it are
    # supervised.
    _ASSISTANT_MARKERS = (
        "<｜hy_Assistant｜>",
        "<|hy_Assistant|>",
        "<|im_start|>assistant\n",
        "<|im_start|>assistant",
        "<start_of_turn>model\n",
    )

    def _tokenize_applied_message(self, text: str):
        """Tokenize a single chat-templated string and build labels that
        supervise ONLY the last assistant reply.

        Returns ``None`` when:
          * no assistant marker is found in ``text``;
          * the prompt alone (everything up to and including the last
            assistant marker) already reaches or exceeds ``self.max_length``
            tokens, so truncation leaves zero supervised tokens;
          * after tokenisation no valid label position remains (paranoid
            fallback).
        """
        # Locate the start of the last assistant turn at the character level.
        last_idx = -1
        marker_used = None
        for marker in self._ASSISTANT_MARKERS:
            i = text.rfind(marker)
            if i > last_idx:
                last_idx = i
                marker_used = marker
        if last_idx < 0:
            return None

        # Char position right AFTER the marker = first supervised character.
        target_char_start = last_idx + len(marker_used)
        prompt_text = text[:target_char_start]

        # Tokenise the prompt-only prefix independently to get its exact
        # token length. This is the ground-truth ``prompt_len`` we'll mask.
        prompt_len = len(
            self.processor(
                prompt_text, add_special_tokens=False, return_tensors=None
            )["input_ids"]
        )

        # Skip samples whose prompt alone does not fit within max_length
        # (leaves 0 tokens to supervise). Also require at least one
        # position >= prompt_len under max_length for the assistant reply.
        if prompt_len >= self.max_length:
            return None

        # Tokenise the full sequence with padding + truncation.
        encoded = self.processor(
            text,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        # Sanity: the first ``prompt_len`` tokens of ``input_ids`` MUST
        # match the standalone-tokenised prompt. If they don't (e.g.
        # unusual tokenizer merge across the marker boundary), fall back
        # to skipping the sample to avoid silently misaligning labels.
        prompt_ids = self.processor(
            prompt_text, add_special_tokens=False, return_tensors="pt"
        )["input_ids"][0]
        if input_ids.shape[1] < prompt_len or not torch.equal(
            input_ids[0, :prompt_len], prompt_ids
        ):
            return None

        labels = input_ids.clone()
        labels[:, :prompt_len] = -100
        labels[attention_mask == 0] = -100

        # Must have at least one supervised token after padding/truncation.
        if (labels != -100).sum().item() == 0:
            return None

        return {
            "input_ids": input_ids.to(self.device),
            "attention_mask": attention_mask.to(self.device),
            "labels": labels.to(self.device),
        }

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

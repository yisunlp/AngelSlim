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

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, TrainerCallback

from ....data.qat_dataset import QATDataset
from ....utils import patch_deepspeed_duplicate_check, print_info
from ..plugins.learnable_scale import set_quant_state
from .trainer_factory import TrainerFactory


class _ProgressLogCallback(TrainerCallback):
    """Emit a clean one-line progress message on every logging_step.

    HF Trainer ships a tqdm progress bar, but when it is wrapped by a
    multi-node launcher (e.g. ``deepspeed`` running under ``pdsh``) the
    carriage returns tqdm relies on get stripped, so the bar degrades
    into garbled output. This callback prints a rank-0-only progress
    line alongside the usual ``{'loss': ...}`` dict, guaranteeing a
    readable progress signal in every environment.

    Also prints elapsed / ETA / step-time so long QAT runs are easy to
    monitor without tqdm.
    """

    @staticmethod
    def _fmt_duration(seconds: float) -> str:
        """Format a duration like ``1d 03:41:22`` / ``07:12`` / ``44.3s``."""
        if seconds < 0 or seconds != seconds:  # NaN guard
            return "?"
        if seconds < 60:
            return f"{seconds:.1f}s"
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        d, h = divmod(h, 24)
        if d > 0:
            return f"{d}d {h:02d}:{m:02d}:{s:02d}"
        if h > 0:
            return f"{h:02d}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"

    def on_train_begin(self, args, state, control, **kwargs):
        import time

        self._train_start_ts = time.monotonic()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        if state.is_world_process_zero is False:
            return
        import time

        total = state.max_steps if state.max_steps and state.max_steps > 0 else None
        step = state.global_step
        parts = [f"step {step}" + (f"/{total}" if total else "")]
        if total:
            parts.append(f"{100.0 * step / total:.1f}%")
        loss = logs.get("loss")
        if loss is not None:
            parts.append(f"loss={loss:.4f}")
        gn = logs.get("grad_norm")
        if gn is not None:
            parts.append(f"grad_norm={gn:.3f}")
        lr = logs.get("learning_rate")
        if lr is not None:
            parts.append(f"lr={lr:.2e}")

        # Timing: elapsed since train_begin, average s/step, projected total,
        # ETA (remaining).
        start = getattr(self, "_train_start_ts", None)
        if start is not None and step > 0:
            elapsed = time.monotonic() - start
            sec_per_step = elapsed / step
            parts.append(f"elapsed={self._fmt_duration(elapsed)}")
            parts.append(f"{sec_per_step:.2f}s/step")
            if total and total > step:
                remaining = sec_per_step * (total - step)
                parts.append(f"eta={self._fmt_duration(remaining)}")
        print_info("[progress] " + " | ".join(parts))


def _unique_named_params(model, predicate):
    """Collect parameters matching ``predicate`` with id-based de-duplication.

    Some QAT setups share a single scale Parameter across multiple
    QuantLinear views (e.g. MoE experts built from a shared tensor). HF /
    DeepSpeed optimizer init rejects duplicates, so we de-dup by ``id``.
    """
    seen = set()
    result = []
    for name, param in model.named_parameters():
        if id(param) in seen or not predicate(name, param):
            continue
        seen.add(id(param))
        result.append(param)
    return result


class QATSeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(self, *args, loss_config=None, quant_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        loss_config = loss_config or {}
        quant_config = quant_config or {}
        self.loss_type = str(loss_config.get("loss_type", "origin")).lower()
        self.loss_topk = loss_config.get("loss_topk")
        self.kd_temperature = float(loss_config.get("kd_temperature", 1.0))
        # ``kd_alpha`` kept for backward compat but IGNORED when
        # ``lm_loss_weight`` / ``kd_loss_weight`` are the (new) source of
        # truth.
        self.kd_alpha = float(loss_config.get("kd_alpha", 0.5))
        self.lm_loss_weight = float(loss_config.get("lm_loss_weight", 1.0))
        self.kd_loss_weight = float(loss_config.get("kd_loss_weight", 0.0))
        self.use_weight_quant = quant_config.get("use_weight_quant", False)
        self.use_activation_quant = quant_config.get("use_activation_quant", False)
        self.use_qkv_quant = quant_config.get("use_qkv_quant", False)

        # Running metric aggregator keyed by logger mode.
        from collections import defaultdict

        self._qat_metrics = {"train": defaultdict(list), "eval": defaultdict(list)}

    # ------------------------------------------------------------------
    # KD loss helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _kl_per_token(log_p_src: torch.Tensor, p_tgt: torch.Tensor) -> torch.Tensor:
        """Per-token KL(tgt || src)  (shape [N])."""
        return F.kl_div(log_p_src, p_tgt, reduction="none").sum(dim=-1)

    @staticmethod
    def _shift_for_next_token(
        student_logits: torch.Tensor,
        labels: torch.Tensor,
        teacher_logits: torch.Tensor = None,
    ):
        """Shift logits & labels so that ``logits[t]`` predicts ``labels[t+1]``.

        Some HF model implementations (e.g. the ``hy_v3`` family on
        ``transformers >= 4.57``) do NOT shift labels inside ``forward`` and
        do NOT honour ``ignore_index=-100``. We perform the shift centrally
        here so that every loss component agrees on what is being predicted.
        """
        s = student_logits[..., :-1, :].contiguous()
        lab = labels[..., 1:].contiguous()
        t = teacher_logits[..., :-1, :].contiguous() if teacher_logits is not None else None
        return s, lab, t

    def _compute_lm_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Standard next-token CE loss with shift + ignore_index=-100.

        Computed in fp32 for numerical stability (matches HF's default).
        Returns a scalar tensor with grad attached when ``logits`` requires
        grad. If no valid token is present we return a 0 with grad attached
        (multiplying ``logits.sum()`` by 0) so DeepSpeed's backward stays
        happy.
        """
        shift_logits, shift_labels, _ = self._shift_for_next_token(logits, labels)
        # Reshape and run CE with -100 ignored.
        lm = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)).float(),
            shift_labels.view(-1).long(),
            ignore_index=-100,
            reduction="mean",
        )
        if torch.isnan(lm):
            # No valid label position — keep the graph alive at 0.
            return logits.sum() * 0.0
        return lm

    def _compute_kd_components(self, student_logits, teacher_logits, labels):
        """Return a dict of per-token KD losses computed only on valid
        (label != -100) positions, AFTER shifting so position ``t``
        predicts the token at ``t+1``.

        Always returns ``forward_kl`` and ``backward_kl`` (useful for
        logging even when the main kd loss is e.g. ``mse`` or a topk
        variant).
        """
        student_logits, labels, teacher_logits = self._shift_for_next_token(
            student_logits, labels, teacher_logits
        )
        flat_mask = (labels != -100).reshape(-1)
        if flat_mask.sum() == 0:
            zero = student_logits.new_zeros(())
            return {"loss": zero, "forward_kl": zero, "backward_kl": zero}

        # Flatten to [N, V] and keep only valid tokens.
        s_flat = student_logits.flatten(0, -2)[flat_mask]
        t_flat = teacher_logits.flatten(0, -2)[flat_mask]
        valid_labels = labels.reshape(-1)[flat_mask]

        # Diagnostic KL (always computed at T=1 on valid tokens).
        s_logp = F.log_softmax(s_flat, dim=-1)
        t_logp = F.log_softmax(t_flat, dim=-1)
        s_p = s_logp.exp()
        t_p = t_logp.exp()
        forward_kl = self._kl_per_token(s_logp, t_p).mean()
        backward_kl = self._kl_per_token(t_logp, s_p).mean()

        # Main KD loss according to loss_type.
        if self.loss_type == "kl":
            kd = forward_kl
        elif self.loss_type == "rkl":
            kd = backward_kl
        elif self.loss_type == "mse":
            kd = F.mse_loss(s_flat, t_flat)
        elif self.loss_type == "kd":
            # Legacy "kd": temperature-scaled forward KL. Combined loss is
            # (alpha*T^2)*KD + (1-alpha)*lm — but we now rely on
            # lm/kd_loss_weight for the outer combination, so return just
            # the scaled KL here.
            T = max(self.kd_temperature, 1e-6)
            kd = self._kl_per_token(
                F.log_softmax(s_flat / T, dim=-1),
                F.softmax(t_flat / T, dim=-1),
            ).mean() * (T * T)
        elif self.loss_type == "cakld":
            # Contextual Asymmetric KL-divergence: per-token mixing of
            # forward / reverse KL by teacher's confidence on the label.
            per_tok_fkl = self._kl_per_token(s_logp, t_p)
            per_tok_bkl = self._kl_per_token(t_logp, s_p)
            conf = torch.gather(t_p, dim=-1, index=valid_labels.unsqueeze(-1)).squeeze(-1)  # [N]
            kd = (conf * per_tok_bkl + (1.0 - conf) * per_tok_fkl).mean()
        elif self._is_reverse_topk_loss(self.loss_type):
            topk = self._resolve_topk(self.loss_type, s_flat.size(-1))
            top_s, idx = s_flat.topk(topk, dim=-1, sorted=False)
            top_t = t_flat.gather(-1, idx)
            kd = self._kl_per_token(
                F.log_softmax(top_t, dim=-1),
                F.softmax(top_s, dim=-1),
            ).mean()
        elif self._is_forward_topk_loss(self.loss_type):
            topk = self._resolve_topk(self.loss_type, t_flat.size(-1))
            top_t, idx = t_flat.topk(topk, dim=-1, sorted=False)
            top_s = s_flat.gather(-1, idx)
            kd = self._kl_per_token(
                F.log_softmax(top_s, dim=-1),
                F.softmax(top_t, dim=-1),
            ).mean()
        else:
            raise ValueError(
                f"Unsupported QAT kd loss_type: {self.loss_type}. "
                "Valid: kl, rkl, mse, kd, cakld, kl_top[_K], r_kl_top[_K]."
            )

        return {"loss": kd, "forward_kl": forward_kl, "backward_kl": backward_kl}

    def _record(self, name, value):
        """Record a scalar metric with cross-rank synchronisation.

        We gather the per-rank scalar across the world and store its mean,
        so that ``log()`` reports the SAME value on every rank — matching
        the semantics of HF Trainer's built-in ``loss`` key (which is also
        gathered + averaged before logging). Without this, each rank would
        only log its own micro-batch's value, easily disagreeing with the
        global ``loss`` whenever sample difficulty differs across ranks.
        """
        if value is None:
            return
        if isinstance(value, torch.Tensor):
            v = value.detach().float()
            if v.ndim == 0:
                v = v.reshape(1)
            try:
                gathered = self.accelerator.gather_for_metrics(v)
                scalar = gathered.float().mean().item()
            except Exception:  # noqa: BLE001
                scalar = v.mean().item()
        else:
            scalar = float(value)
        mode = "train" if self.model.training else "eval"
        self._qat_metrics[mode][name].append(scalar)

    # ------------------------------------------------------------------
    # compute_loss
    # ------------------------------------------------------------------

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels", None)
        lm_on = self.lm_loss_weight > 0.0
        kd_on = self.kd_loss_weight > 0.0

        # Back-compat: ``loss_type="origin"`` means "pure HF CE loss" → the
        # classic SFT path with no distillation. Honour it even when the
        # user forgot to set kd_loss_weight=0.
        if self.loss_type == "origin":
            kd_on = False

        if not lm_on and not kd_on:
            raise ValueError("Both lm_loss_weight and kd_loss_weight are 0 — nothing to optimise.")

        # Student forward — always needed. We DO NOT pass ``labels`` into
        # the model: some implementations (e.g. transformers' HYV3) ship a
        # custom in-forward CE that does not shift labels nor honour
        # ``ignore_index=-100``. Always compute the LM loss ourselves.
        student_inputs = {k: v for k, v in inputs.items() if k != "labels"}
        outputs = model(**student_inputs)

        lm_loss = None
        if lm_on:
            if labels is None:
                raise ValueError("lm_loss_weight > 0 requires ``labels`` in the batch.")
            lm_loss = self._compute_lm_loss(outputs.logits, labels)

        kd_info = None
        if kd_on:
            if labels is None:
                raise ValueError("kd_loss_weight > 0 requires ``labels`` in the batch.")
            teacher_logits = self.get_ori_outputs(model, inputs).logits
            kd_info = self._compute_kd_components(outputs.logits, teacher_logits, inputs.get("input_ids".clone(),None))

        # Combine.
        total = outputs.logits.new_zeros(())
        if lm_loss is not None:
            total = total + self.lm_loss_weight * lm_loss
        if kd_info is not None:
            total = total + self.kd_loss_weight * kd_info["loss"]

        # Logging: record every component whose weight is > 0, plus the
        # always-informative forward/backward KL diagnostics when kd is on.
        if lm_on and lm_loss is not None:
            self._record("lm_loss", lm_loss)
        if kd_on and kd_info is not None:
            self._record(f"kd/{self.loss_type}", kd_info["loss"])
            # Diagnostic KL(L/R) for any kd variant. Useful to monitor
            # teacher-student disagreement independent of the combined
            # objective.
            self._record("kd/forward_kl", kd_info["forward_kl"])
            self._record("kd/backward_kl", kd_info["backward_kl"])
        self._record("total_loss", total)

        return (total, outputs) if return_outputs else total

    @staticmethod
    def _is_forward_topk_loss(loss_type):
        return loss_type.startswith("kl_top")

    @staticmethod
    def _is_reverse_topk_loss(loss_type):
        return loss_type.startswith("r_kl_top") or loss_type.startswith("rkl_top")

    def _resolve_topk(self, loss_type, vocab_size):
        topk = self.loss_topk
        if topk is None and "_top_" in loss_type:
            topk = int(loss_type.rsplit("_", 1)[-1])
        if topk is None:
            topk = 1000
        topk = int(topk)
        if topk <= 0:
            raise ValueError(f"loss_topk must be positive, got: {topk}")
        return min(topk, vocab_size)

    @torch.no_grad()
    def get_ori_outputs(self, model, inputs):
        teacher_inputs = dict(inputs)
        teacher_inputs.pop("labels", None)
        raw_model = self.accelerator.unwrap_model(model)
        set_quant_state(raw_model, weight_quant=False, act_quant=False, qkv_quant=False)
        try:
            outputs = model(**teacher_inputs)
        finally:
            set_quant_state(
                raw_model,
                weight_quant=self.use_weight_quant,
                act_quant=self.use_activation_quant,
                qkv_quant=self.use_qkv_quant,
            )
        return outputs

    def log(self, logs, start_time=None, *args, **kwargs):
        """Inject running QAT loss components (lm_loss / kd/... / kd/forward_kl
        / kd/backward_kl / total_loss) into HuggingFace Trainer's log dict.

        Each value is averaged across the steps accumulated since the
        previous ``log`` call, then the accumulator is cleared — matching
        HF Trainer's behaviour for its built-in ``loss`` key.
        """
        mode = "train" if self.model.training else "eval"
        bucket = self._qat_metrics.get(mode, {})
        if bucket:
            for key, vals in bucket.items():
                if not vals:
                    continue
                avg = sum(vals) / len(vals)
                out_key = key if mode == "train" else f"eval_{key}"
                logs[out_key] = float(avg)
            bucket.clear()

        # Forward to HF Trainer's log. Signature differs across versions.
        try:
            return super().log(logs, start_time, *args, **kwargs)
        except TypeError:
            return super().log(logs)


@TrainerFactory.register("end2end")
class End2EndTrainer:

    def __init__(self, quant_model, config, plugin_manager):
        self.quant_model = quant_model
        self.config = config
        self.plugin_manager = plugin_manager
        self.training_mode = config["compress_config"].QAT.training_mode
        self.dist_mode = config["compress_config"].QAT.dist_mode
        self.hf_dataset = config["compress_config"].QAT.hf_dataset
        self.hf_cache_dir = config["compress_config"].QAT.hf_cache_dir
        self.resume_ckpt_dir = config["compress_config"].QAT.resume_ckpt_dir
        self.do_train = config["compress_config"].QAT.do_train
        self.external_trainer = None

        self.loss_config = {
            "loss_type": config["compress_config"].QAT.loss_type,
            "loss_topk": config["compress_config"].QAT.loss_topk,
            "kd_temperature": config["compress_config"].QAT.kd_temperature,
            "kd_alpha": config["compress_config"].QAT.kd_alpha,
            "lm_loss_weight": config["compress_config"].QAT.lm_loss_weight,
            "kd_loss_weight": config["compress_config"].QAT.kd_loss_weight,
        }
        self.quant_config = {
            "use_weight_quant": config["compress_config"]
            .QAT.plugin_config.get("quant_config", {})
            .get("use_weight_quant", False),
            "use_activation_quant": config["compress_config"]
            .QAT.plugin_config.get("quant_config", {})
            .get("use_activation_quant", False),
            "use_qkv_quant": config["compress_config"]
            .QAT.plugin_config.get("quant_config", {})
            .get("use_qkv_quant", False),
        }

    def _init_optimizer(self):
        lr = float(self.config["compress_config"].QAT.hf_args.get("learning_rate", 1e-5))
        wd = float(self.config["compress_config"].QAT.hf_args.get("weight_decay", 0))
        scale_params = _unique_named_params(
            self.quant_model.model,
            lambda n, p: p.requires_grad and ("scale" in n or "zero_point" in n),
        )
        params = [
            {
                "params": scale_params,
                "weight_decay": wd,
                "lr": lr,
            }
        ]

        enable_lwc = (
            self.config["compress_config"]
            .QAT.plugin_config.get("quant_config", {})
            .get("lwc", {})
            .get("enable_lwc", False)
        )
        lwc_param_count = 0
        if enable_lwc:
            lwc_lr = float(
                self.config["compress_config"]
                .QAT.plugin_config.get("quant_config", {})
                .get("lwc", {})
                .get("lwc_lr", 1e-1)
            )
            lwc_params = _unique_named_params(
                self.quant_model.model,
                lambda n, p: p.requires_grad
                and ("clip_factor_w_max" in n or "clip_factor_w_min" in n),
            )
            lwc_param_count = len(lwc_params)
            params.append({"params": lwc_params, "weight_decay": wd, "lr": lwc_lr})

        if not any(group["params"] for group in params):
            raise ValueError("QAT optimizer has no trainable parameters.")

        self.optimizer = torch.optim.AdamW(params)
        if enable_lwc:
            print_info(
                f"Init optimizer with {len(scale_params)} scale params, "
                f"{lwc_param_count} lwc params, lr={lr} lwc_lr={lwc_lr} weight_decay={wd}"
            )
        else:
            print_info(
                f"Init optimizer with {len(scale_params)} scale params, "
                f"lr={lr} weight_decay={wd}"
            )

    def prepare_trainer(self):
        if self.training_mode == "blockwise":
            return
        if self.training_mode == "end2end" and self.dist_mode == "hf":
            self._init_optimizer()
            # When DeepSpeed is used, neutralize its duplicate-parameter
            # check: it rejects param-groups that share tensors, which our
            # scale/zero_point setup can legally have (shared tensors
            # across views). Idempotent and a no-op if deepspeed is absent.
            if self.config["compress_config"].QAT.hf_args.get("deepspeed") is not None:
                patch_deepspeed_duplicate_check()
            self.external_trainer = QATSeq2SeqTrainer(
                model=self.quant_model.model,
                processing_class=self.quant_model.tokenizer,
                args=Seq2SeqTrainingArguments(
                    output_dir=self.config["global_config"].save_path,
                    **self.config["compress_config"].QAT.hf_args,
                ),
                train_dataset=self.train_dataset,
                eval_dataset=None,
                optimizers=(self.optimizer, None),
                loss_config=self.loss_config,
                quant_config=self.quant_config,
                callbacks=[_ProgressLogCallback()],
            )
        else:
            raise NotImplementedError(f"Unsupported distribution mode: {self.dist_mode}")

    def prepare_dataset(self, dataloader):
        if self.hf_dataset is not None:
            parts = self.hf_dataset.split(",")
            dataset = load_dataset(*parts, cache_dir=self.hf_cache_dir)
            self.train_dataset = QATDataset(
                dataset["train"],
                self.quant_model.tokenizer,
                block_size=dataloader.dataset.max_length,
                is_opensource=True,
            )
        else:
            self.train_dataset = QATDataset(dataloader.dataset, self.quant_model.tokenizer)

    def run(self, dataloader):
        self.prepare_dataset(dataloader)
        self.plugin_manager.call_before_train(train_dataset=self.train_dataset)

        self.prepare_trainer()

        # When gradient checkpointing is enabled AND every model weight is
        # frozen (QAT case: only scale / zero_point are trainable), the
        # checkpointed backward cannot discover a grad path on its own and
        # the final logits come back with ``requires_grad=False``. The
        # standard fix (same one HF applies for PEFT) is to force the input
        # embedding output to require grad, so the checkpointed layers
        # re-trace the graph through it.
        #
        # IMPORTANT: register the hook AFTER ``prepare_trainer`` because
        # HF ``gradient_checkpointing_enable`` runs inside
        # ``trainer.train()`` but before the first compute_loss; the hook
        # we register here lives on the underlying HF module and survives
        # DeepSpeed wrapping (DeepSpeed proxies ``forward`` but keeps the
        # underlying ``nn.Module`` intact).
        if self.config["compress_config"].QAT.hf_args.get("gradient_checkpointing", False):
            model = self.quant_model.get_model()
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
                print_info(
                    "[QAT] enable_input_require_grads() invoked "
                    "(gradient_checkpointing + frozen base weights)."
                )

        if self.resume_ckpt_dir is not None:
            print_info(f"Loading from resume {self.resume_ckpt_dir}")
            save_dict = torch.load(self.resume_ckpt_dir, map_location="cpu")
            self.quant_model.model.load_state_dict(save_dict)

        if self.do_train:
            if self.external_trainer is not None:
                self.external_trainer.train()
            else:
                self.train()

        self.plugin_manager.call_after_train()

"""Diagnostic: compute bf16 loss on applied_message data and inspect the
label mask boundary.

Step 2 of the user's diagnostic protocol: if bf16 loss is reasonable
(<~4), data processing is fine; otherwise it indicates a dataset issue.
"""
import os
import sys
import importlib.util

import torch

# ---------------------------------------------------------------------------
# Register the HY-V3 custom model classes with HuggingFace Auto* factories
# BEFORE we call AutoModel*.from_pretrained. The checkpoint's config has
# ``model_type=hy_v3`` but does not ship an ``auto_map``, so we must
# register the mapping manually.
# ---------------------------------------------------------------------------
sys.path.insert(0, "/apdcephfs_zwfy2/share_301053287/brunosu")

# Pre-import the hy_v3 package so that ``transformers.models.hy_v3.modeling_HY_v3``
# (which is referenced by angelslim.models.llm.hunyuan_v3_moe but lives OUTSIDE
# the transformers package) resolves. We do this by loading the two modules
# directly (hy_v3/__init__.py uses transformers-internal LazyModule relative
# imports that don't work outside the transformers package) and aliasing them
# under ``transformers.models.hy_v3``.
import types  # noqa: E402


def _spec_load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_HY = "/apdcephfs_zwfy2/share_301053287/brunosu/hy_v3"
# Fabricate a minimal package under name ``transformers.models.hy_v3`` first.
_hv_pkg = types.ModuleType("transformers.models.hy_v3")
_hv_pkg.__path__ = [_HY]
sys.modules["transformers.models.hy_v3"] = _hv_pkg
_hy_config = _spec_load(
    "transformers.models.hy_v3.configuration_HY_v3",
    os.path.join(_HY, "configuration_HY_v3.py"),
)
_hy_modeling = _spec_load(
    "transformers.models.hy_v3.modeling_HY_v3",
    os.path.join(_HY, "modeling_HY_v3.py"),
)

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer  # noqa: E402

AutoConfig.register("hy_v3", _hy_config.HYV3Config, exist_ok=True)
AutoModelForCausalLM.register(_hy_config.HYV3Config, _hy_modeling.HYV3ForCausalLM, exist_ok=True)

# ---------------------------------------------------------------------------
# Import TextDataset WITHOUT triggering the big angelslim package __init__
# (which would pull in a bunch of transformers-version-specific models).
# ---------------------------------------------------------------------------
_HERE = "/apdcephfs_zwfy2/share_301053287/brunosu/AngelSlim/angelslim"


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


import types  # noqa: E402  # noqa: F811

pkg = types.ModuleType("_al_data")
pkg.__path__ = [os.path.join(_HERE, "data")]
sys.modules["_al_data"] = pkg
_load_module("_al_data.base_dataset", os.path.join(_HERE, "data", "base_dataset.py"))
text_dataset_mod = _load_module(
    "_al_data.text_dataset", os.path.join(_HERE, "data", "text_dataset.py")
)
TextDataset = text_dataset_mod.TextDataset

MODEL_PATH = "/apdcephfs_zwfy2/share_301053287/brunosu/all_models/hy3_0_a3b"
DATA_PATH = (
    "/apdcephfs_zwfy2/share_301053287/brunosu/angelslim_fsdp/apply_message_train_for_quant.json"
)
MAX_LEN = 4096
NUM = 4


def main():
    print("=" * 80)
    print("Loading tokenizer ...")
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    print("=" * 80)
    print("Loading dataset ...")
    ds = TextDataset(DATA_PATH, tok, device="cpu", max_length=MAX_LEN, num_samples=NUM)
    print(f"#samples loaded: {len(ds.data)}")

    # ----- inspect the first sample -----
    item = ds.data[0]
    input_ids = item["input_ids"][0]
    labels = item["labels"][0]
    attn = item["attention_mask"][0]
    valid_mask = labels != -100
    first_valid = int(valid_mask.nonzero(as_tuple=True)[0][0].item())
    last_valid = int(valid_mask.nonzero(as_tuple=True)[0][-1].item())
    n_valid = int(valid_mask.sum().item())
    n_tokens = int(attn.sum().item())
    print("-" * 80)
    print(f"sample[0] total tokens (attn=1): {n_tokens}")
    print(f"sample[0] supervised tokens    : {n_valid}")
    print(f"sample[0] first_valid index    : {first_valid}")
    print(f"sample[0] last_valid  index    : {last_valid}")
    prompt_tail_ids = input_ids[max(0, first_valid - 6) : first_valid].tolist()
    target_head_ids = input_ids[first_valid : first_valid + 10].tolist()
    print("prompt tail ids  :", prompt_tail_ids)
    print("prompt tail text :", repr(tok.decode(prompt_tail_ids)))
    print("target head ids  :", target_head_ids)
    print("target head text :", repr(tok.decode(target_head_ids)))
    # Show the entire supervised segment decoded (truncated).
    sup_ids = input_ids[first_valid : last_valid + 1].tolist()
    sup_text = tok.decode(sup_ids)
    if len(sup_text) > 400:
        sup_text = sup_text[:200] + " ... " + sup_text[-200:]
    print("supervised text  :", repr(sup_text))

    print("=" * 80)
    print("Loading bf16 model ...")
    torch.set_grad_enabled(False)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        trust_remote_code=True,
    )
    model = model.eval().cuda()

    print("=" * 80)
    print("Computing loss on first %d samples ..." % len(ds.data))
    total_native = 0.0
    total_hf = 0.0
    n = 0
    import torch.nn.functional as F
    for i, s in enumerate(ds.data):
        input_ids = s["input_ids"].cuda()
        attn = s["attention_mask"].cuda()
        labels = s["labels"].cuda()
        out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
        native_loss = out.loss.item()
        # HF-standard shifted CE with ignore_index=-100 for sanity comparison.
        logits = out.logits.float()  # [1, L, V]
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        hf_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        ).item()
        nv = int((labels != -100).sum().item())
        print(
            f"  sample[{i}]: native_loss={native_loss:.4f}, hf_shifted_loss={hf_loss:.4f}, "
            f"supervised_tokens={nv}"
        )
        total_native += native_loss
        total_hf += hf_loss
        n += 1
    print("-" * 80)
    print(
        f"mean native_loss={total_native / max(n, 1):.4f}  "
        f"mean hf_shifted_loss={total_hf / max(n, 1):.4f}"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Convert custom BERT checkpoint (.pth) to HuggingFace format.

Produces per-model directory:
  hf_upload/<name>/
    config.json          # HF-style config dict
    pytorch_model.bin    # state_dict (original, for custom loader)
    tokenizer_config.json
    <name>_tokenizer.json
    special_tokens_map.json
    README.md            # Model card lite

Usage:
  cd /home/liufan/AI_biology/hermes_coding/sepsis_diag
  python ../sepsis-diagnosis/scripts/convert_to_hf.py --convert-all

  # Or for a single model:
  python ../sepsis-diagnosis/scripts/convert_to_hf.py \\
      -c outputs/BERT-36L-CSL.pth \\
      -t outputs/sepsis_diagnoses_tokenizer.json \\
      -n BERT-36L-CSL
"""

import argparse
import torch
import json
import shutil
from pathlib import Path


# ── Model Registry for --convert-all ─────────────────────────────

ALL_MODELS = [
    {
        "checkpoint": f"outputs/BERT-6L-CSL.pth",
        "tokenizer_path": "outputs/sepsis_diagnoses_tokenizer.json",
        "output_dir": "./hf_upload/BERT-6L-CSL-Sepsis-Diag",
        "model_name": "BERT-6L-CSL",
    },
    {
        "checkpoint": f"outputs/BERT-36L-CSL.pth",
        "tokenizer_path": "outputs/sepsis_diagnoses_tokenizer.json",
        "output_dir": "./hf_upload/BERT-36L-CSL-Sepsis-Diag",
        "model_name": "BERT-36L-CSL",
    },
]


# ── Convert a single model ───────────────────────────────────────

def auto_encoder_num(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    branches = set()
    for k in ckpt["model_state_dict"].keys():
        if k.startswith("module_list."):
            branches.add(k.split(".")[1])
    return len(branches)


def build_hf_config(vocab_size, encoder_num, model_name):
    return {
        "architectures": ["Bertclassifier"],
        "model_type": "bert",
        "hidden_size": 128,
        "num_attention_heads": 4,
        "hidden_dropout_prob": 0.1,
        "intermediate_size": 2048,
        "max_position_embeddings": 30,
        "num_labels": 2,
        "vocab_size": vocab_size,
        "encoder_num": encoder_num,
        "label_dim": 0,
        "model_name": model_name,
    }


def convert_single(model_cfg):
    """Convert one checkpoint + tokenizer to HF dir."""
    ckpt_path = Path(model_cfg["checkpoint"])
    tok_path = Path(model_cfg["tokenizer_path"])
    out_dir  = Path(model_cfg["output_dir"]).resolve()
    model_name = model_cfg["model_name"]

    print(f"\n{'='*50}")
    print(f"Converting: {model_name}")
    print(f"  ckpt →  {ckpt_path}")
    print(f"  out   → {out_dir}")

    if not ckpt_path.is_file():
        print(f" SKIP: checkpoint not found")
        return False
    if not tok_path.is_file():
        print(f" SKIP: tokenizer not found")
        return False

    out_dir.mkdir(parents=True, exist_ok=True)

    # Load state_dict
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd   = ckpt["model_state_dict"]
    vocab_size   = sd["tok_embed.weight"].shape[0]
    enc_num      = auto_encoder_num(ckpt_path)
    total_params = sum(v.numel() for v in sd.values())

    print(f"  Vocab={vocab_size}  EncoderNum={enc_num}  Params={total_params:,}")

    # 1. config.json (HF-style)
    hf_cfg = build_hf_config(vocab_size, enc_num, model_name)
    with open(out_dir / "config.json", "w") as f:
        json.dump(hf_cfg, f, indent=2)

    # 2. pytorch_model.bin — save the ORIGINAL state_dict (our custom loader works with it)
    torch.save(ckpt, out_dir / "pytorch_model.bin")

    # 3. Copy tokenizer
    shutil.copy2(tok_path, out_dir / f"{model_name}_tokenizer.json")

    # 4. tokenizer_config.json
    with open(out_dir / "tokenizer_config.json", "w") as f:
        json.dump({
            "tokenizer_class": "BertTokenizer",
            "model_max_length": 30,
            "do_lower_case": True,
            "unk_token": "[UNK]",
            "sep_token": "[SEP]",
            "pad_token": "[PAD]",
            "cls_token": "[CLS]",
            "mask_token": "[MASK]",
        }, f, indent=2)

    # 5. special_tokens_map.json
    with open(out_dir / "special_tokens_map.json", "w") as f:
        json.dump({
            "unk_token":   {"content": "[UNK]",  "lstrip": True},
            "sep_token":   {"content": "[SEP]",  "lstrip": True},
            "pad_token":   {"content": "[PAD]",  "lstrip": True},
            "cls_token":   {"content": "[CLS]",  "lstrip": True, "normalized": True},
            "mask_token":  {"content": "[MASK]", "lstrip": True},
        }, f, indent=2)

    # 6. README.md (lite model card)
    with open(out_dir / "README.md", "w") as f:
        f.write(f"# {model_name}\n\n")
        f.write(f"Sepsis in-hospital mortality prediction from ICD diagnosis text.\n\n")
        f.write(f"- **Architecture**: BERT-style Transformer (encoder_num={enc_num}, 6 layers per branch)\n")
        f.write(f"- **Total parameters**: {total_params:,}\n")
        f.write(f"- **Vocabulary size**: {vocab_size}\n")
        f.write(f"- **Hidden size**: 128\n")
        f.write(f"- **Num attention heads**: 4\n")
        f.write(f"- **Feed-forward dim**: 2048\n")
        f.write(f"- **Max sequence length**: 30\n\n")
        f.write("## Loading the model\n\n```python\n")
        f.write("import torch\n")
        f.write("from src.bert_classifier import BertClassifier\n\n")
        f.write("# Load checkpoint\n")
        f.write("ckpt = torch.load('pytorch_model.bin', map_location='cpu')\n")
        f.write("model = BertClassifier(\n")
        f.write(f"    vocab_size={vocab_size},\n")
        f.write(f"    encoder_num={enc_num},\n")
        f.write(")\n")
        f.write("model.load_state_dict(ckpt['model_state_dict'])\n")
        f.write("model.eval()\n\n")
        f.write("# Inference\n")
        f.write("input_ids = tokenizer(text, return_tensors='pt')['input_ids']\n")
        f.write("logits    = model(input_ids)\n")
        f.write("pred      = torch.argmax(logits, dim=-1).item()\n```\n\n")
        f.write("---\n\nBuilt with [sepsis-diagnosis](https://github.com/liufan/sepsis-diagnosis).\n")

    print(f" OK → {out_dir}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Convert custom checkpoint → HF format")
    parser.add_argument("--convert-all", "-a",  action="store_true",  help="Convert all models in registry")
    parser.add_argument("-c", "--checkpoint",        default=None,    help="Single model .pth path")
    parser.add_argument("-t", "--tokenizer",         default=None,    help="Tokenizer .json path")
    parser.add_argument("-n", "--model-name",        default=None,    help="Model name (used in HF dir)")
    parser.add_argument("-o", "--output-dir",        default=None,    help="Output directory")
    args = parser.parse_args()

    if not args.convert_all and not args.checkpoint:
        print("Usage:")
        print("  convert_to_hf.py --convert-all              (all models)")
        print("  convert_to_hf.py -c ckpt.pth -t tok.json -n NAME [-o dir]")
        return

    if args.convert_all:
        models = ALL_MODELS
    else:
        name = args.model_name or Path(args.checkpoint).stem if args.checkpoint else "Custom"
        odir = args.output_dir or f"./hf_upload/{name}-Sepsis-Diag"
        mpath= args.checkpoint  or "."
        tsp = args.tokenizer   or "./outputs/sepsis_diagnoses_tokenizer.json"
        models = [{
            "checkpoint": mpath,
            "tokenizer_path": tsp,
            "output_dir": odir,
            "model_name": name,
        }]

    ok = 0
    fail = 0
    for mc in models:
        if convert_single(mc):
            ok += 1
        else:
            fail += 1

    print(f"\n{'='*50}")
    print(f"Done: {ok} OK, {fail} skipped.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Extract CLS embeddings from a model + dataset for downstream clustering.

Auto-detects model type (cs_l / recon) by checkpoint vocab_size,
loads the matching tokenizer, and exports:
  - {name}.csv          → per-sample predictions + original columns
  - {name}_embeddings.csv → CLS embedding vectors (N x 128)

Usage:
  # Extract all 4 models on all datasets
  python extract_embeddings.py

  # Single model + single dataset
  python extract_embeddings.py --checkpoint outputs/BERT-36L-CSL.pth \
                               --input outputs/eicu_diag.csv \
                               --output-dir embeddings/
"""

import argparse
import torch
import pandas as pd
from pathlib import Path

from tokenizer_builder import load_tokenizer
from bert_classifier import BertClassifier
from config import (
    OUTPUT_DIR, BASE_DIR, MAX_LENGTH, LABEL_DIM, HIDDEN_SIZE,
    NUM_HEADS, CLASS_NUM, DROPOUT, TOKENIZER_FILE_DIAG, TOKENIZER_FILE_LAB,
)

# ──────────────────────── Model Registry ──────────────────────

ALL_MODELS = [
    {"name": "6L-CSL",  "checkpoint": f"{OUTPUT_DIR}/BERT-6L-CSL.pth",           "encoder_num": 1},
    {"name": "36L-CSL", "checkpoint": f"{OUTPUT_DIR}/BERT-36L-CSL.pth",          "encoder_num": 6},
    {"name": "6L-Recon","checkpoint": f"{OUTPUT_DIR}/BERT-6L-Recon.pth",         "encoder_num": 1},
    {"name": "36L-Recon","checkpoint":f"{OUTPUT_DIR}/BERT-36L-Recon.pth",        "encoder_num": 6},
]

# ──────────────────────── Datasets ────────────────────────────

DIAG_DATASETS = [
    ("mimic-iv",       f"{OUTPUT_DIR}/mimi3_diag.csv"),
    ("eICU",           f"{OUTPUT_DIR}/eicu_diag.csv"),
    ("wx",             f"{OUTPUT_DIR}/wx.csv"),
]

LAB_DATASETS = [
    ("mimic-iv",        f"{OUTPUT_DIR}/mimi3_lab.csv"),
    ("eICU-enriched",   f"{OUTPUT_DIR}/eicu_lab_enriched.csv"),
    ("wx",              f"{OUTPUT_DIR}/wx_lab.csv"),
]


# ──────────────────────── Helpers ─────────────────────────────

def build_and_load_model(checkpoint_path, encoder_num, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    vocab_size = ckpt["model_state_dict"]["tok_embed.weight"].shape[0]
    model = BertClassifier(
        vocab_size=vocab_size, label_dim=LABEL_DIM, hidden_size=HIDDEN_SIZE,
        num_layers=6, num_heads=NUM_HEADS, class_num=CLASS_NUM,
        encoder_num=encoder_num, dropout=DROPOUT,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def extract(model, data_df, tokenizer, device):
    """Return (predictions, embeddings, labels)."""
    import shap  # noqa — not used here but tokenizer fix needs tokenizers lib
    from tokenizers import models as tk_models

    # Fix _tokenizer.model unk_token
    vocab = tokenizer.get_vocab()
    tokenizer._tokenizer.model = tk_models.WordPiece(vocab=vocab, unk_token="[UNK]")

    texts = data_df["input"].astype(str).tolist()
    inputs = tokenizer(
        texts, padding="max_length", max_length=MAX_LENGTH,
        truncation=True, return_tensors="pt", return_offsets_mapping=False,
    )
    encodings = inputs["input_ids"].to(device)
    pad_mask = (inputs["attention_mask"].to(torch.bool) == False).to(device)

    with torch.no_grad():
        logits, emb = model(encodings, attention_mask=pad_mask, return_embedding=True)
        preds  = torch.argmax(logits, dim=-1).cpu().numpy()
        embs   = emb[:, 0, :].cpu().numpy()  # CLS token embeddings

    labels = data_df["hospital_expire_flag"].values.astype(int)
    return preds, embs, labels


# ──────────────────────── Args ────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract CLS embeddings for clustering"
    )
    parser.add_argument("--model-type", choices=["cs_l", "recon", "all"], default="all")
    parser.add_argument("--checkpoint", nargs="+", default=None)
    parser.add_argument("--input",      nargs="+", default=None)
    parser.add_argument("--output-dir", "-o", default=None,
                        help="Directory to save embeddings CSVs (default: outputs/embeddings/)")
    return parser.parse_args()


# ──────────────────────── Main ────────────────────────────────

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    out_dir = Path(args.output_dir) or (Path(OUTPUT_DIR) / "embeddings")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Resolve models ────────────────────────

    if args.checkpoint:
        custom = []
        for cp_str in args.checkpoint:
            cp_path = Path(cp_str)
            ckpt_data = torch.load(cp_str, map_location=device, weights_only=False)
            vocab_sz = ckpt_data["model_state_dict"]["tok_embed.weight"].shape[0]
            mtype = "recon" if vocab_sz == 5175 else "cs_l"
            branches = set(
                k.split(".")[1] for k in ckpt_data["model_state_dict"].keys()
                if k.startswith("module_list.")
            )
            custom.append({
                "name": cp_path.stem,
                "checkpoint": str(cp_path),
                "encoder_num": len(branches),
                "model_type": mtype,
            })
        models_to_run = custom
    else:
        models_to_run = []
        for m in ALL_MODELS:
            mt = "recon" if "Recon" in m["name"] else "cs_l"
            if args.model_type == "all" or args.model_type == mt:
                models_to_run.append({**m, "model_type": mt})

    print(f"Models ({len(models_to_run)}):")
    for m in models_to_run:
        is_r = "(lab)" if m["model_type"]=="recon" else "(diag)"
        print(f"  {m['name']} encoder_num={m['encoder_num']} {is_r}")

    # ── 2. Resolve datasets ─────────────────────

    input_filter = set()
    if args.input:
        for p in args.input:
            input_filter.add(str(Path(p).absolute()))

    diag_list = [
        (n, str(Path(fp).absolute()))
        for n, fp in DIAG_DATASETS if Path(fp).exists()
    ]
    lab_list = [
        (n, str(Path(fp).absolute()))
        for n, fp in LAB_DATASETS if Path(fp).exists()
    ]

    if input_filter:
        diag_list = [(n,p) for n,p in diag_list if p in input_filter]
        lab_list  = [(n,p) for n,p in lab_list  if p in input_filter]

    print(f"\nDiag datasets: {[n for n,_ in diag_list]}")
    print(f"Lab datasets:   {[n for n,_ in lab_list]}\n")

    # ── 3. Extract and save ─────────────────────

    for m in models_to_run:
        ckpt_p = Path(m["checkpoint"])
        if not ckpt_p.exists():
            print(f"[{m['name']}] SKIP — {ckpt_p} not found")
            continue

        print(f"{'='*60}")
        print(f"Loading [{m['name']}]")
        model = build_and_load_model(ckpt_p, m["encoder_num"], device)

        datasets = lab_list if m["model_type"] == "recon" else diag_list
        tk_path  = TOKENIZER_FILE_LAB if m["model_type"] == "recon" else TOKENIZER_FILE_DIAG

        tokenizer = load_tokenizer(tk_path)

        for ds_name, ds_path in datasets:
            df = pd.read_csv(ds_path)
            print(f"  {ds_name}: {len(df)} samples")

            preds, embs, labels = extract(model, df, tokenizer, device)

            # Save predictions + original data
            pred_df = df.copy()
            pred_df["predictions"] = preds
            pred_csv = out_dir / f"{m['name']}_{ds_name}_preds.csv"
            pred_df.to_csv(pred_csv, index=False)

            # Save embeddings (N x 128 float32) — pure embeddings only
            emb_csv = out_dir / f"{m['name']}_{ds_name}_embeddings.csv"
            pd.DataFrame(embs, columns=[f"emb_{i}" for i in range(embs.shape[1])]).to_csv(emb_csv, index=False)

            print(f"    → {pred_csv.stem}.csv  ({pred_csv.parent.name}/)")
            print(f"    → {emb_csv.stem}.csv   (shape={embs.shape})")

        del model

    print(f"\nDone. All outputs in: {out_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Unified evaluator for sepsis diagnosis models.

Evaluates all 4 checkpoints (6L/36L, CSL/Recon) on downstream datasets.
Uses hard predictions (torch.argmax) + ROC/AUC.

Usage:
  # Run ALL 4 models on ALL matching datasets
  python unified_eval.py

  # Run only CSL (diag) models
  python unified_eval.py --model-type cs_l

  # Run specific checkpoint(s) only
  python unified_eval.py --checkpoint outputs/BERT-36L-CSL.pth outputs/BERT-6L-CSL.pth

  # Specify input dataset(s)
  python unified_eval.py --input outputs/eicu_diag.csv

  # Save per-sample predictions to a dir
  python unified_eval.py --detail-output-dir result_detail/
"""

import argparse
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc

from tokenizer_builder import load_tokenizer
from bert_classifier import BertClassifier
from config import (
    OUTPUT_DIR, BASE_DIR, MAX_LENGTH, LABEL_DIM, HIDDEN_SIZE,
    NUM_HEADS, CLASS_NUM, DROPOUT, TOKENIZER_FILE_DIAG, TOKENIZER_FILE_LAB,
)

# ──────────────────────────── Model Registry ─────────────────────────────

ALL_MODELS = [
    {"name": "6L-CSL",  "checkpoint": f"{OUTPUT_DIR}/BERT-6L-CSL.pth",    "encoder_num": 1},
    {"name": "36L-CSL", "checkpoint": f"{OUTPUT_DIR}/BERT-36L-CSL.pth",   "encoder_num": 6},
    {"name": "6L-Recon","checkpoint": f"{OUTPUT_DIR}/BERT-6L-Recon.pth",  "encoder_num": 1},
    {"name": "36L-Recon","checkpoint":f"{OUTPUT_DIR}/BERT-36L-Recon.pth", "encoder_num": 6},
]

# ──────────────────────────── Datasets ───────────────────────────────────

DIAG_DATASETS = [
    ("mimic-iv", f"{OUTPUT_DIR}/mimi3_diag.csv"),
    ("eICU",     f"{OUTPUT_DIR}/eicu_diag.csv"),
    ("wx",       f"{OUTPUT_DIR}/wx.csv"),
]

LAB_DATASETS = [
    ("mimic-iv",        f"{OUTPUT_DIR}/mimi3_lab.csv"),
    ("eICU-enriched",   f"{OUTPUT_DIR}/eicu_lab_enriched.csv"),
    ("wx",              f"{OUTPUT_DIR}/wx_lab.csv"),
]


# ──────────────────────────── Helpers ────────────────────────────────────

def build_and_load_model(checkpoint_path, encoder_num, device):
    """Build BertClassifier + load weights from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    vocab_size = ckpt["model_state_dict"]["tok_embed.weight"].shape[0]

    model = BertClassifier(
        vocab_size=vocab_size,
        label_dim=LABEL_DIM,
        hidden_size=HIDDEN_SIZE,
        num_layers=6,
        num_heads=NUM_HEADS,
        class_num=CLASS_NUM,
        encoder_num=encoder_num,
        dropout=DROPOUT,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def run_inference(model, data_df, tokenizer, device):
    """Tokenise + predict. Returns (hard_preds, labels)."""
    # Fix: ensure _tokenizer.model has word_piece with unk_token set
    from tokenizers import models as tk_models
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
        logits = model(encodings, attention_mask=pad_mask)
        preds = torch.argmax(logits, dim=-1).cpu().numpy()

    labels = data_df["hospital_expire_flag"].values.astype(int)
    return preds, labels


def compute_metrics(labels, preds):
    """Return dict of metrics. AUROC computed on hard predictions."""
    acc = accuracy_score(labels, preds)
    pre = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)
    f1  = f1_score(labels, preds, zero_division=0)

    fpr, tpr, _ = roc_curve(labels, preds)
    roc_auc = auc(fpr, tpr)

    return {
        "accuracy": acc,
        "precision": pre,
        "recall": rec,
        "f1_score": f1,
        "auroc": roc_auc,
    }


# ──────────────────────────── Args ───────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified evaluator: 4 models x downstream datasets"
    )
    parser.add_argument(
        "--model-type", choices=["cs_l", "recon", "all"], default="all",
        help="Which model family to run (default: all).",
    )
    parser.add_argument(
        "--checkpoint", nargs="+", default=None,
        help="Override model selection with explicit checkpoint .pth path(s)."
    )
    parser.add_argument(
        "--input", nargs="+", default=None,
        help="Specific CSV dataset path(s) to evaluate. If omitted, all matching datasets are used.",
    )
    parser.add_argument(
        "--detail-output-dir", default=None,
        help="Save per-prediction CSV for each model-dataset pair here.",
    )
    return parser.parse_args()


# ──────────────────────────── Main ───────────────────────────────────────

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    detail_dir = Path(args.detail_output_dir) if args.detail_output_dir else None
    if detail_dir and not detail_dir.exists():
        detail_dir.mkdir(parents=True)

    # ── 1. Resolve which models to run ─────────────────────

    if args.checkpoint:
        # User gave explicit checkpoint(s) — auto-detect type via vocab_size
        custom_models = []
        for cp_path_str in args.checkpoint:
            cp_path = Path(cp_path_str)
            ckpt_name = cp_path.stem  # e.g. "BERT-36L-CSL"

            ckpt_data = torch.load(cp_path, map_location=device, weights_only=False)
            vocab_size = ckpt_data["model_state_dict"]["tok_embed.weight"].shape[0]
            branch_ids = set(
                k.split(".")[1] for k in ckpt_data["model_state_dict"].keys()
                if k.startswith("module_list.")
            )
            encoder_num = len(branch_ids)

            # vocab_size 5175 → lab/recon format, else diag/csl
            model_type = "recon" if vocab_size == 5175 else "cs_l"

            custom_models.append({
                "name": ckpt_name,
                "checkpoint": str(cp_path),
                "encoder_num": encoder_num,
                "model_type": model_type,
            })
        models_to_run = custom_models
        print(f"Running {len(models_to_run)} user-specified checkpoint(s):")
    else:
        # Build list from registered models filtered by --model-type
        models_to_run = []
        for m in ALL_MODELS:
            m_type = "recon" if "Recon" in m["name"] else "cs_l"
            if args.model_type == "all" or args.model_type == m_type:
                models_to_run.append({**m, "model_type": m_type})
        print(f"Running {len(models_to_run)} model(s) (--model-type={args.model_type}):")

    for m in models_to_run:
        print(f"  {m['name']}: encoder_num={m['encoder_num']}, type={m['model_type']}")

    # ── 2. Resolve which datasets to run ───────────────────

    # Filter by explicit --input, else use all existing files
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
        diag_list = [(n, p) for n, p in diag_list if p in input_filter]
        lab_list  = [(n, p) for n, p in lab_list  if p in input_filter]

    print(f"\nDiag datasets: {[n for n,_ in diag_list]}")
    print(f"Lab  datasets: {[n for n,_ in lab_list]}\n")

    # ── 3. Run evaluations ────────────────────────────────

    results = []  # (model_name, ds_name, metrics_dict, row_count)

    for m in models_to_run:
        ckpt_path = Path(m["checkpoint"])
        if not ckpt_path.exists():
            print(f"[{m['name']}] SKIP — file not found: {ckpt_path}")
            continue

        print(f"{'='*60}")
        print(f"Loading [{m['name']}]  encoder_num={m['encoder_num']}")
        model = build_and_load_model(ckpt_path, m["encoder_num"], device)

        # Pick dataset list based on model type
        datasets = lab_list if m["model_type"] == "recon" else diag_list
        tkr_path = TOKENIZER_FILE_LAB if m["model_type"] == "recon" else TOKENIZER_FILE_DIAG

        tokenizer = load_tokenizer(tkr_path)

        for ds_name, ds_path in datasets:
            df = pd.read_csv(ds_path)
            mort_rate = float(df["hospital_expire_flag"].mean())
            n_samples = len(df)

            print(f"  {ds_name}: {n_samples} samples (mortality={mort_rate:.3f})")

            preds, labels = run_inference(model, df, tokenizer, device)
            met = compute_metrics(labels, preds)

            print(
                f"    Acc={met['accuracy']:.4f}"
                f"  Pre={met['precision']:.4f}"
                f"  Rec={met['recall']:.4f}"
                f"  F1={met['f1_score']:.4f}"
                f"  AUROC={met['auroc']:.4f}"
            )

            results.append((m["name"], ds_name, met, n_samples))

            # Optional: save per-sample details
            if detail_dir:
                out_df = df.copy()
                out_df["predictions"] = preds
                out_csv = detail_dir / f"{m['name']}_{ds_name}_detail.csv"
                out_df.to_csv(out_csv, index=False)

        del model  # free GPU memory

    # ── 4. Print + save summary ───────────────────────────

    if not results:
        print("\nNo evaluations were run.")
        return

    rows = []
    for model_name, ds_name, met, n in results:
        rows.append({
            "model":     model_name,
            "dataset":   ds_name,
            "n":         n,
            "accuracy":  f"{met['accuracy']:.4f}",
            "precision": f"{met['precision']:.4f}",
            "recall":    f"{met['recall']:.4f}",
            "f1_score":  f"{met['f1_score']:.4f}",
            "auroc":     f"{met['auroc']:.4f}",
        })

    summary_df = pd.DataFrame(rows)

    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(summary_df.to_string(index=False))

    csv_path = (detail_dir / "unified_eval_summary.csv"
                if detail_dir else Path(OUTPUT_DIR) / "unified_eval_summary.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"\nSaved → {csv_path}")


if __name__ == "__main__":
    main()

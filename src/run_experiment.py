#!/usr/bin/env python3
"""
Unified experiment runner for sepsis diagnosis prediction.

Input: [CLS] + long_title, 5000 steps, 8 TSNE clusters.

Usage:
  python run_experiment.py                               # mimi3_diag.csv (config)
  python run_experiment.py --diag-input outputs/eicu_diag.csv
  python run_experiment.py --train-steps 8000 --n-clusters 10 --encoder-num 4
"""

import argparse
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tokenizers.models import WordPiece
from sklearn.metrics import f1_score, roc_curve, auc
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

from config import (
    BASE_DIR, OUTPUT_DIR, LABEL_DIM, HIDDEN_SIZE, NUM_HEADS,
    CLASS_NUM, BATCH_SIZE, MAX_LENGTH, LR, WEIGHT_DECAY, DROPOUT,
    LOSS_WEIGHTS, RANDOM_STATE,
    TRAIN_EPOCHS_DIAG,
    TOKENIZER_FILE_DIAG,
    CHECKPOINT_DIAG,
)
from bert_classifier import BertClassifier
from tokenizer_builder import load_tokenizer, build_shap_tokenizer
from trainer import train_model, evaluate_model


# ── Defaults ─────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "diag": {
        "steps": TRAIN_EPOCHS_DIAG,
        "clusters": 8,
        "perplexity": 90,
        "tsne_rs": 120,
        "tokenizer": TOKENIZER_FILE_DIAG,
        "checkpoint": CHECKPOINT_DIAG,
        "prefix": "diag",
        "title": "DIAG (diagnoses only, palliative included)",
    },
}


# ── Helpers ────────────────────────────────────────────────────────

def _fix_unk_token(tokenizer_obj):
    """Ensure tokenizer has the WordPiece unk_token set via tokenizers lib."""
    vocab = tokenizer_obj.get_vocab()
    tokenizer_obj._tokenizer.model = WordPiece(vocab=vocab, unk_token="[UNK]")


# ── Args ───────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Sepsis diagnosis experiment runner")
    parser.add_argument("--diag-input", default=None,
                        help="Path to mimi3_diag.csv (or external diag CSV). Overrides config.")
    parser.add_argument("--train-steps", type=int, default=None,
                        help="Override training steps.")
    parser.add_argument("--n-clusters", type=int, default=None,
                        help="Override number of clusters.")
    parser.add_argument("--encoder-num", type=int, default=None,
                        help="Override encoder_num (number of transformer branches).")
    parser.add_argument("--tokenizer", default=None,
                        help="Override tokenizer path.")
    parser.add_argument("--checkpoint", default=None,
                        help="Override checkpoint save path.")
    parser.add_argument("--output-dir", default=None,
                        help="Override output directory.")
    return parser.parse_args()


# ── Data loading ───────────────────────────────────────────────────────

def load_data(mode, args):
    """Load input data; returns (data, None)."""
    diag_path = args.diag_input or Path(OUTPUT_DIR) / "mimi3_diag.csv"
    data = pd.read_csv(diag_path)
    data["input"] = "[CLS] " + data["long_title"]
    print(f"Loaded {len(data)} patient records")
    return data, None


# ── Model building ─────────────────────────────────────────────────

def build_model(vocab_size, cfg, args):
    """Build BertClassifier with resolved hyperparameters."""
    encoder_num = args.encoder_num or cfg.get("encoder_num_default", 6)
    return BertClassifier(
        vocab_size=vocab_size,
        label_dim=LABEL_DIM,
        hidden_size=HIDDEN_SIZE,
        num_layers=6,
        num_heads=NUM_HEADS,
        class_num=CLASS_NUM,
        encoder_num=encoder_num,
        dropout=DROPOUT,
    )


# ── Training ───────────────────────────────────────────────────────

def run_training(model, train_df, tokenizer, max_length, device, cfg, args):
    """Train the model and return loss curve + optimizer."""
    steps = args.train_steps or cfg["steps"]
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(LOSS_WEIGHTS, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    print(f"\nTraining for {steps} steps...")
    train_loss = train_model(
        model, train_df, tokenizer, max_length, device,
        loss_fn, optimizer, max_steps=steps,
        batch_size=BATCH_SIZE, verbose=True,
    )
    return train_loss, optimizer


# ── Evaluation ─────────────────────────────────────────────────────

def run_evaluation(model, data, tokenizer, max_length, device, cfg, args):
    """Evaluate on train/test splits and return results."""
    prefix = cfg["prefix"]

    # Train/test split
    data_filtered = data[data["input"].str.len() > 0].reset_index(drop=True)
    train_df = data_filtered.sample(frac=0.75, random_state=RANDOM_STATE)
    test_df = data_filtered.drop(train_df.index)
    print(f"Train: {len(train_df)}, Test: {len(test_df)}")

    # Test evaluation — hard preds only
    test_preds = evaluate_model(model, test_df, tokenizer, max_length, device)
    test_labels = test_df["hospital_expire_flag"].values

    # Full data evaluation with embeddings
    full_preds, _, full_embs = evaluate_model(
        model, data_filtered, tokenizer, max_length, device, return_embedding=True
    )
    data_filtered["predictions"] = full_preds

    # ROC (based on hard predictions)
    fpr2, tpr2, _ = roc_curve(test_labels, test_preds)
    test_auc = auc(fpr2, tpr2)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr2, tpr2, color="darkorange", lw=2, label=f"ROC (AUC={test_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title(f"ROC - {prefix}"); plt.legend(); plt.grid(alpha=0.3)
    plt.savefig(f"{OUTPUT_DIR}/roc_{prefix}_test.pdf")
    plt.close()
    print(f"Test AUC: {test_auc:.4f}")

    return train_df, test_df, data_filtered, full_embs, test_auc


# ── Clustering ─────────────────────────────────────────────────────

def run_clustering(data_filtered, full_embs, cfg, args):
    """TSNE + KMeans clustering."""
    clusters_n = args.n_clusters or cfg["clusters"]
    perplexity = cfg["perplexity"]
    tsne_rs = cfg["tsne_rs"]

    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=tsne_rs)
    coords = tsne.fit_transform(full_embs)

    kmeans = KMeans(n_clusters=clusters_n, random_state=30)
    clusters = kmeans.fit_predict(coords)

    result = data_filtered.copy()
    result["cluster"] = clusters
    result["pca_x"] = coords[:, 0]
    result["pca_y"] = coords[:, 1]

    # Per-cluster statistics
    clusters_data = pd.DataFrame(np.unique(clusters).reshape(-1, 1), columns=["cluster"])
    clusters_data["mortality"] = 0.0
    clusters_data["roc"] = 0.0
    clusters_data["f1"] = 0.0

    for cid in clusters:
        mask = result["cluster"] == cid
        grp = result[mask]
        if len(grp) == 0:
            continue
        clusters_data.loc[clusters_data["cluster"] == cid, "mortality"] = \
            grp["hospital_expire_flag"].mean()
        if len(grp) > 1 and "predictions" in grp.columns:
            fp, tp, _ = roc_curve(grp["predictions"], grp["hospital_expire_flag"])
            clusters_data.loc[clusters_data["cluster"] == cid, "roc"] = auc(fp, tp)
            clusters_data.loc[clusters_data["cluster"] == cid, "f1"] = f1_score(
                grp["hospital_expire_flag"], grp["predictions"]
            )

    print(f"Cluster stats:\n{clusters_data.to_string()}")
    clusters_data.to_csv(f"{OUTPUT_DIR}/cluster_{cfg['prefix']}_stats.csv", index=False)

    # Embedding scatter
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].scatter(result["pca_x"], result["pca_y"], c=result["hospital_expire_flag"], cmap="viridis", alpha=0.5)
    axes[0].set_title("Embedding: Mortality")
    axes[1].scatter(result["pca_x"], result["pca_y"], c=result["cluster"], cmap="tab10", alpha=0.5)
    axes[1].set_title(f"Embedding: {clusters_n} Clusters")
    plt.savefig(f"{OUTPUT_DIR}/embedding_{cfg['prefix']}.pdf")
    plt.close()

    return result, clusters_data


# ── Loss curve ─────────────────────────────────────────────────────

def plot_loss_curve(train_loss, cfg):
    """Plot and save training loss curve."""
    prefix = cfg["prefix"]
    steps = len(train_loss)
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label="CE Loss")
    plt.xlabel("Step"); plt.ylabel("Loss")
    plt.title(f"Training Loss ({prefix}, {steps} steps)")
    plt.legend(); plt.grid(alpha=0.3)
    plt.savefig(f"{OUTPUT_DIR}/loss_{prefix}.pdf")
    plt.close()


# ── Main ───────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg = DEFAULT_CONFIG["diag"].copy()
    cfg["encoder_num_default"] = 6

    if args.output_dir:
        global OUTPUT_DIR
        OUTPUT_DIR = args.output_dir

    print("=" * 60)
    print(f"Experiment: {cfg['title']}")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── 1. Load data ──
    data, lab = load_data(mode, args)

    # ── 2. Tokenizer (with unk_token fix) ──
    tokenizer_path = args.tokenizer or cfg["tokenizer"]
    fast_tokenizer = load_tokenizer(tokenizer_path)
    _fix_unk_token(fast_tokenizer)
    vocab_size = len(fast_tokenizer)
    print(f"Vocab size: {vocab_size}")

    # ── 3. Build model ──
    model = build_model(vocab_size, cfg, args).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # ── 4. Train ──
    train_loss, optimizer = run_training(model, data, fast_tokenizer, MAX_LENGTH, device, cfg, args)

    # Save checkpoint
    ckpt_path = Path(args.checkpoint or cfg["checkpoint"])
    steps = args.train_steps or cfg["steps"]
    torch.save({
        "epoch": steps,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, ckpt_path)
    print(f"Checkpoint saved → {ckpt_path}")

    # ── 5. Evaluate ──
    print("\n--- Evaluation ---")
    train_df, test_df, data_filtered, full_embs, test_auc = run_evaluation(
        model, data, fast_tokenizer, MAX_LENGTH, device, cfg, args,
    )

    # ── 6. Clustering ──
    print("\n--- Clustering ---")
    clustered, cluster_stats = run_clustering(
        data_filtered, full_embs, cfg, args,
    )

    # ── 7. Loss curve ──
    plot_loss_curve(train_loss, cfg)

    print("\nDone. Outputs in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()

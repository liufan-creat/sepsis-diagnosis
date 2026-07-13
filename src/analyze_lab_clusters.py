#!/usr/bin/env python3
"""
Lab-specific cluster analysis for dig_lab mode.

After running cluster_embeddings.py on dig_lab data, this script performs:
  1. Per-cluster proportion analysis (shock + organ failure letters)
  2. Cluster heatmap for deceased patients (sns.clustermap)
  3. SHAP analysis per cluster
  4. Lab classification label embedding visualization
  5. Palliative care patient check in embedding space

Usage:
    python analyze_lab_clusters.py \\
        --clustered outputs/lab_clustered.csv \\
        --lab outputs/patient_summary_labeled.csv \\
        --checkpoint outputs/BERT-36L-Recon.pth \\
        --tokenizer outputs/sepsis_diagnoses_lab_tokenizer.json \\
        --n-clusters 6
"""

import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tokenizers import models as tk_models
from sklearn.manifold import TSNE

from bert_classifier import BertClassifier
from tokenizer_builder import load_tokenizer, build_shap_tokenizer
from trainer import evaluate_model
from config import LABEL_DIM, HIDDEN_SIZE, NUM_HEADS, CLASS_NUM, DROPOUT, MAX_LENGTH


# ── Constants ───────────────────────────────────────────────────────

TARGET_LETTERS = ["K", "L", "C", "R", "N", "H", "M", "E"]


# ── Helpers ────────────────────────────────────────────────────────

def _fix_unk_token(tokenizer):
    """Ensure tokenizer has the WordPiece unk_token set."""
    vocab = tokenizer.get_vocab()
    tokenizer._tokenizer.model = tk_models.WordPiece(vocab=vocab, unk_token="[UNK]")


def load_model_and_tokenizer(checkpoint, tokenizer_path, device):
    """Load trained model and tokenizer with unk_token fix + encoder_num from ckpt."""
    tokenizer_obj = load_tokenizer(tokenizer_path)
    _fix_unk_token(tokenizer_obj)

    ckpt_data = torch.load(checkpoint, map_location=device, weights_only=False)
    vocab_size = ckpt_data["model_state_dict"]["tok_embed.weight"].shape[0]

    # Auto-detect encoder_num from checkpoint
    branch_ids = set(
        k.split(".")[1] for k in ckpt_data["model_state_dict"].keys()
        if k.startswith("module_list.")
    )
    encoder_num = len(branch_ids)

    model = BertClassifier(
        vocab_size=vocab_size,
        label_dim=LABEL_DIM, hidden_size=HIDDEN_SIZE,
        num_layers=6, num_heads=NUM_HEADS, class_num=CLASS_NUM,
        encoder_num=encoder_num, dropout=DROPOUT,
    ).to(device)
    model.load_state_dict(ckpt_data["model_state_dict"])
    model.eval()
    return model, tokenizer_obj


# ── Analysis functions ─────────────────────────────────────────────

def analyze_cluster_proportions(dset, cluster_name, output_dir):
    """Compute proportions of shock + organ failure letters per cluster."""
    dset = dset.copy()
    dset["status"] = dset["classification_label"].str.split("_").str[0]
    dset["cate"] = dset["classification_label"].str.split("_").str[1]

    for char in TARGET_LETTERS:
        dset[char] = dset["cate"].str.contains(char, na=False).astype(int)
    dset["shock"] = (dset["status"] == "shock").astype(int)

    cols = ["shock"] + TARGET_LETTERS
    col_sums = dset[cols].sum() / len(dset)
    path = f"{output_dir}/cluster_{cluster_name}_proportions.csv"
    col_sums.to_csv(path)
    print(f"  Proportions saved → {path}")
    print(f"  {col_sums.to_dict()}")
    return col_sums


def plot_cluster_heatmap(dset, cluster_name, output_dir):
    """Heatmap of deceased patients in a cluster (organ failure letters)."""
    dset_dead = dset[dset["hospital_expire_flag"] == 1]
    if len(dset_dead) == 0:
        print(f"  No deceased patients in cluster {cluster_name}, skipping heatmap")
        return

    cols = ["shock"] + TARGET_LETTERS
    colors = ["#440154", "#FDE725"]
    g = sns.clustermap(
        dset_dead[cols], annot=False, cbar=False, yticklabels=False,
        cmap=colors, dendrogram_ratio=(0.3, 0.1), col_cluster=False,
    )
    g.ax_row_dendrogram.set_visible(False)
    plt.setp(g.ax_heatmap.get_xticklabels(), fontsize=22)
    path = f"{output_dir}/heatmap_cluster_{cluster_name}.pdf"
    g.savefig(path)
    plt.close()
    print(f"  Heatmap saved → {path}")


def compute_cluster_shap(model, tokenizer, texts, cluster_name, device, n_samples=30, output_dir="."):
    """Compute SHAP permutation values for a cluster subset."""
    tokenizer_obj = build_shap_tokenizer(tokenizer)

    def predict_fn(sample):
        inputs = tokenizer_obj(
            sample.astype(str).tolist(),
            padding="max_length", max_length=45,
            return_tensors="pt", return_offsets_mapping=False,
        )
        enc = inputs["input_ids"].to(device)
        mask = inputs["attention_mask"].to(torch.bool) == False
        mask = mask.to(device)
        with torch.no_grad():
            out = model(enc, attention_mask=mask)
        return torch.argmax(out, dim=-1).cpu().numpy()

    import shap
    masker = shap.maskers.Text(tokenizer_obj, mask_token="[MASK]")
    explainer = shap.Explainer(predict_fn, masker, algorithm="permutation")

    texts_sample = texts.head(n_samples)
    shap_vals = explainer(texts_sample)
    values = shap_vals.abs.values

    shape_data_list = []
    for i in range(len(values)):
        pro_tokens = texts_sample.iloc[i].split()[1:]
        val_arr = values[i][1:]
        min_len = min(len(pro_tokens), len(val_arr))
        pro_tokens = pro_tokens[:min_len]
        val_arr = val_arr[:min_len]
        shape_dict = dict(zip(pro_tokens, val_arr))
        shape_dict = {k: v for k, v in shape_dict.items() if v != 0}
        shape_data_list.append(shape_dict)

    df = pd.DataFrame(shape_data_list)
    path = f"{output_dir}/shap_lab_cluster_{cluster_name}.csv"
    df.to_csv(path, index=False)
    print(f"  SHAP saved → {path}")


def plot_lab_label_embedding(lab, model, tokenizer, device, output_dir="."):
    """TSNE visualization of lab classification labels in embedding space."""
    lab_labels = lab[["classification_label"]].copy()
    lab_inputs = "[CLS] " + lab_labels["classification_label"]
    lab_labels["input"] = lab_inputs

    preds, logits, embs = evaluate_model(
        model, lab_labels, tokenizer, MAX_LENGTH, device,
        return_embedding=True,
    )
    lab_labels["predictions"] = preds

    tsne_lab = TSNE(n_components=2, perplexity=40, random_state=70)
    lab_coords = tsne_lab.fit_transform(embs)

    plt.figure(figsize=(8, 6))
    plt.scatter(lab_coords[:, 0], lab_coords[:, 1], c=preds, cmap="viridis")
    plt.title("Lab Classification Label Embeddings")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    path = f"{output_dir}/lab_label_embedding.pdf"
    plt.savefig(path)
    plt.close()
    print(f"  Lab label embedding saved → {path}")


def plot_palliative_check(data, full_embs, output_dir="."):
    """Check if palliative care patients cluster separately."""
    data = data.copy()
    data["has_palliative"] = data["input"].str.contains("Encounter_for_palliative_care", na=False)

    tsne_pall = TSNE(n_components=2, perplexity=90, random_state=20)
    pall_coords = tsne_pall.fit_transform(full_embs)

    plt.figure(figsize=(8, 6))
    plt.scatter(
        pall_coords[:, 0], pall_coords[:, 1],
        c=data["has_palliative"].astype(int), cmap="viridis",
    )
    plt.title("Palliative Care Patients in Embedding Space")
    path = f"{output_dir}/palliative_check_lab.pdf"
    plt.savefig(path)
    plt.close()
    print(f"  Palliative check saved → {path}")


# ── Main ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Lab-specific cluster analysis")
    parser.add_argument("--clustered", "-c", required=True,
                        help="CSV from cluster_embeddings.py (must have cluster, classification_label, input, hospital_expire_flag)")
    parser.add_argument("--lab", "-l", required=True,
                        help="Original lab CSV (patient_summary_labeled.csv)")
    parser.add_argument("--checkpoint", "-p", required=True,
                        help="Model checkpoint .pth")
    parser.add_argument("--tokenizer", "-t", required=True,
                        help="Tokenizer JSON path")
    parser.add_argument("--n-clusters", "-k", type=int, default=6,
                        help="Number of clusters used")
    parser.add_argument("--high-clusters", nargs="+", default=[0],
                        help="Cluster IDs to analyze as 'high mortality'")
    parser.add_argument("--med-clusters", nargs="+", default=[2],
                        help="Cluster IDs to analyze as 'medium mortality'")
    parser.add_argument("--n-shap-samples", type=int, default=30,
                        help="SHAP samples per cluster")
    from config import OUTPUT_DIR
    parser.add_argument("--output-dir", "-o", default=OUTPUT_DIR, help="Output directory")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    # Load data
    data = pd.read_csv(args.clustered)
    lab = pd.read_csv(args.lab)
    print(f"Clustered data: {len(data)} rows, {data['cluster'].nunique()} clusters")

    # Load model + tokenizer (with unk_token fix + auto encoder_num)
    model, tokenizer = load_model_and_tokenizer(args.checkpoint, args.tokenizer, device)

    # Get full embeddings from data (pca_x, pca_y)
    full_embs = data[["pca_x", "pca_y"]].values

    # Analyze selected clusters
    for name, cid in [("H", args.high_clusters[0]), ("M", args.med_clusters[0])]:
        print(f"\n--- Cluster {name} (ID={cid}): {len(data[data['cluster']==cid])} samples ---")
        cdata = data[data["cluster"] == cid].copy()
        if len(cdata) == 0:
            print(f"  Skipping empty cluster {cid}")
            continue

        # 1. Proportions
        analyze_cluster_proportions(cdata, name, args.output_dir)

        # 2. Heatmap
        plot_cluster_heatmap(cdata, name, args.output_dir)

        # 3. SHAP
        compute_cluster_shap(
            model, tokenizer, cdata["input"], name,
            device, n_samples=args.n_shap_samples, output_dir=args.output_dir,
        )

    # 4. Lab label embedding
    print("\n--- Lab label embedding ---")
    plot_lab_label_embedding(lab, model, tokenizer, device, args.output_dir)

    # 5. Palliative check
    print("\n--- Palliative care check ---")
    plot_palliative_check(data, full_embs, args.output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()

"""
Sepsis Diagnosis — Notebook Script.

Run cell by cell in Jupyter/PyCharm.
No wrappers, no abstractions. Just raw torch + sklearn.
Fixed: checkpoint path → BERT-36L-CSL.pth; unk_token fix moved before first inference.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from tokenizers.models import WordPiece
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

from tokenizer_builder import load_tokenizer
from bert_classifier import BertClassifier
from trainer import evaluate_model
from config import OUTPUT_DIR, MAX_LENGTH


# ============================================================
# 1. Load tokenizer + model (BERT-36L-CSL = encoder_num=6)
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = load_tokenizer(f"{OUTPUT_DIR}/sepsis_diagnoses_tokenizer.json")
model = BertClassifier(vocab_size=len(tokenizer))
model = model.to(device)

ckpt = torch.load(f"{OUTPUT_DIR}/BERT-36L-CSL.pth", map_location=device, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()


# ============================================================
# 2. Load data
# ============================================================

data_df = pd.read_csv(f"{OUTPUT_DIR}/mimi3_diag.csv")
data_df["input"] = "[CLS] " + data_df["long_title"]


# ============================================================
# 3. Evaluate (need unk_token fix BEFORE inference)
# ============================================================

vocab = tokenizer.get_vocab()
tokenizer._tokenizer.model = WordPiece(vocab=vocab, unk_token="[UNK]")
tokenizer.add_special_tokens({
    "pad_token": "[PAD]", "unk_token": "[UNK]",
    "cls_token": "[CLS]", "sep_token": "[SEP]",
    "mask_token": "[MASK]"
})

labels = data_df["hospital_expire_flag"].values
preds, logits, embeddings = evaluate_model(
    model, data_df, tokenizer, MAX_LENGTH, device, return_embedding=True
)

acc  = accuracy_score(labels, preds)
pre  = precision_score(labels, preds, zero_division=0)
rec  = recall_score(labels, preds, zero_division=0)
f1   = f1_score(labels, preds, zero_division=0)
print(f"Acc={acc:.4f}  Pre={pre:.4f}  Rec={rec:.4f}  F1={f1:.4f}")


# ============================================================
# 4. ROC Curve
# ============================================================

fpr, tpr, thresholds = roc_curve(labels, logits[:, 1])  # use class-1 probabilities
auc_score = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC={auc_score:.4f})")
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC Curve"); plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/roc_curve.pdf", bbox_inches="tight")
print(f"AUC={auc_score:.4f}")


# ============================================================
# 5. TSNE + KMeans Clustering
# ============================================================

n_clusters = 8

tsne = TSNE(n_components=2, perplexity=30, random_state=120)
tsne_reduced = tsne.fit_transform(embeddings)

kmeans = KMeans(n_clusters=n_clusters, random_state=30)
clusters = kmeans.fit_predict(tsne_reduced)

tsne_df = pd.DataFrame({
    "x": tsne_reduced[:, 0],
    "y": tsne_reduced[:, 1],
    "hadm_id": data_df["hadm_id"].values,
    "hospital_expire_flag": labels,
    "cluster": clusters,
})

for cid in range(n_clusters):
    mask = clusters == cid
    cnt = mask.sum()
    mort = labels[mask].mean() if cnt > 0 else 0
    print(f"  Cluster {cid}: n={cnt}, mortality={mort:.3f}")

fig, ax = plt.subplots(figsize=(8, 6))
sc = ax.scatter(tsne_df["x"], tsne_df["y"], c=tsne_df["cluster"], cmap="viridis", alpha=0.6, s=8)
plt.legend(*sc.legend_elements(), loc="upper right", title="cluster")
ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
ax.set_title(f"TSNE Clusters (k={n_clusters})"); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/tsne_clusters.pdf", bbox_inches="tight")


# ============================================================
# 6. SHAP Permutation Importance (subset)
# ============================================================

import shap

sample_df = data_df.sample(n=50, random_state=42)


def predict_fn(sample):
    inp = tokenizer(sample.tolist(), padding="max_length", max_length=MAX_LENGTH,
                    return_tensors="pt", return_offsets_mapping=False)
    enc = inp["input_ids"].to(device)
    mask = inp["attention_mask"].to(torch.bool) == False
    mask = mask.to(device)
    with torch.no_grad():
        out = model(enc, attention_mask=mask)
    return torch.argmax(out, dim=-1).cpu().numpy()


masker = shap.maskers.Text(tokenizer, mask_token="[MASK]")
explainer = shap.Explainer(predict_fn, masker, algorithm="permutation")
sv = explainer(sample_df["input"])

val = sv.abs.values
for i in range(len(val)):
    tokens = sample_df["input"].iloc[i].split()[1:]
    imp = val[i][1:]
    nonzero = [(t, float(v)) for t, v in zip(tokens, imp) if v != 0]
    if nonzero:
        print(f"  Sample {i}: {nonzero[:5]}")  # top 5 tokens

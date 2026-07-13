#!/usr/bin/env python3
"""
TSNE + KMeans clustering on model embeddings.

Produces:
  - embedding_{prefix}.pdf  (scatter: mortality + cluster)
  - cluster_{prefix}_stats.csv  (per-cluster mortality/ROC/F1)

Usage:
    python cluster_embeddings.py --embeddings embeddings.csv --labels labels.csv --prefix diag --n-clusters 8
    python cluster_embeddings.py --embeddings embeddings.csv --labels labels.csv --prefix lab --n-clusters 6
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import roc_curve, auc, f1_score


def cluster_embeddings(embeddings, labels_df, n_clusters=8, perplexity=30,
                       tsne_random_state=120, kmeans_random_state=30,
                       save_prefix="", output_dir="."):
    """
    Run TSNE + KMeans on embeddings, attach cluster assignments to labels_df.

    Args:
        embeddings: np.array (N, D)
        labels_df: DataFrame with 'hospital_expire_flag' and optionally 'predictions'
        n_clusters: number of clusters
        perplexity: TSNE perplexity
        tsne_random_state: TSNE random seed
        kmeans_random_state: KMeans random seed
        save_prefix: prefix for output filenames
        output_dir: directory for saving files

    Returns:
        result_df: labels_df with added columns ['cluster', 'pca_x', 'pca_y']
        cluster_stats: DataFrame with per-cluster mortality/ROC/F1 stats
    """
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=tsne_random_state)
    coords = tsne.fit_transform(embeddings)

    kmeans = KMeans(n_clusters=n_clusters, random_state=kmeans_random_state)
    clusters = kmeans.fit_predict(coords)

    result = labels_df.copy()
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
        if "predictions" in grp.columns and len(grp) > 1:
            fpr, tpr, rauc = roc_curve(grp["predictions"], grp["hospital_expire_flag"])
            clusters_data.loc[clusters_data["cluster"] == cid, "roc"] = rauc
            clusters_data.loc[clusters_data["cluster"] == cid, "f1"] = f1_score(
                grp["hospital_expire_flag"], grp["predictions"]
            )

    # Save cluster stats
    stats_path = f"{output_dir}/cluster_{save_prefix}_stats.csv" if save_prefix else f"{output_dir}/cluster_stats.csv"
    clusters_data.to_csv(stats_path, index=False)
    print(f"Cluster stats saved → {stats_path}")
    print(f"Cluster stats:\n{clusters_data.to_string()}")

    # Scatter plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].scatter(
        result["pca_x"], result["pca_y"],
        c=result["hospital_expire_flag"], cmap="viridis", alpha=0.5,
    )
    axes[0].set_title("Embedding: Mortality")
    axes[1].scatter(
        result["pca_x"], result["pca_y"],
        c=result["cluster"], cmap="tab10", alpha=0.5,
    )
    axes[1].set_title(f"Embedding: {n_clusters} Clusters")
    scatter_path = f"{output_dir}/embedding_{save_prefix}.pdf" if save_prefix else f"{output_dir}/embedding.pdf"
    plt.savefig(scatter_path)
    plt.close()
    print(f"Embedding scatter saved → {scatter_path}")

    return result, clusters_data


def main():
    parser = argparse.ArgumentParser(description="TSNE + KMeans clustering on embeddings")
    parser.add_argument("--embeddings", "-e", required=True, help="Path to embeddings .csv (no header, numeric columns only)")
    parser.add_argument("--labels", "-l", required=True, help="Path to labels CSV (must have hospital_expire_flag)")
    parser.add_argument("--prefix", "-p", default="", help="Filename prefix")
    parser.add_argument("--n-clusters", "-k", type=int, default=8, help="Number of clusters")
    parser.add_argument("--perplexity", type=float, default=30, help="TSNE perplexity")
    parser.add_argument("--tsne-rs", type=int, default=120, help="TSNE random state")
    parser.add_argument("--kmeans-rs", type=int, default=30, help="KMeans random state")
    from config import OUTPUT_DIR
    parser.add_argument("--output-dir", "-o", default=OUTPUT_DIR, help="Output directory")
    args = parser.parse_args()

    embeddings = np.array(pd.read_csv(args.embeddings))
    labels_df = pd.read_csv(args.labels)

    cluster_embeddings(
        embeddings, labels_df,
        n_clusters=args.n_clusters,
        perplexity=args.perplexity,
        tsne_random_state=args.tsne_rs,
        kmeans_random_state=args.kmeans_rs,
        save_prefix=args.prefix,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()

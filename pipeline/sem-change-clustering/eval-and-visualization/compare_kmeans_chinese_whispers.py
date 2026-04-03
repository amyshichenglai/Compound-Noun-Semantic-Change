from argparse import ArgumentParser
import csv
import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
WORKSPACE_ROOT = os.path.dirname(REPO_ROOT)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.cluster_compounds_only_kmeans import load_examples


def parse_args():
    parser = ArgumentParser(
        description="Visualize and compare saved K-means and Chinese Whispers clusterings."
    )
    parser.add_argument(
        "--kmeans_dir",
        default=os.path.join(
            WORKSPACE_ROOT,
            "german_target_pickles_with_bert_base_german_cased_kmeans_compounds_only_with_sentence_id",
        ),
    )
    parser.add_argument(
        "--kmeans_summary",
        default=os.path.join(
            WORKSPACE_ROOT,
            "german_target_pickles_with_bert_base_german_cased_kmeans_compounds_only",
            "summary.tsv",
        ),
    )
    parser.add_argument(
        "--cw_root_dir",
        default=os.path.join(
            WORKSPACE_ROOT,
            "german_target_pickles_with_bert_base_german_cased_chinese_whispers_with_sentence_id",
        ),
    )
    parser.add_argument("--cw_setting", default="cw_knn_k10_iter20_ens9")
    parser.add_argument(
        "--cw_summary",
        default=os.path.join(
            WORKSPACE_ROOT,
            "german_target_pickles_with_bert_base_german_cased_chinese_whispers_with_sentence_id",
            "summary.tsv",
        ),
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.join(
            REPO_ROOT,
            "eval-and-visualization",
            "outputs",
            "kmeans-vs-chinese-whispers",
        ),
    )
    parser.add_argument("--targets", nargs="*", default=None)
    parser.add_argument("--top_n", type=int, default=8)
    parser.add_argument("--sample_size", type=int, default=600)
    parser.add_argument("--heatmap_top_clusters", type=int, default=10)
    parser.add_argument("--random_seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    rng = np.random.RandomState(args.random_seed)

    cw_dir = os.path.join(args.cw_root_dir, args.cw_setting)
    if not os.path.isdir(args.kmeans_dir):
        raise FileNotFoundError(f"Missing K-means directory: {args.kmeans_dir}")
    if not os.path.isdir(cw_dir):
        raise FileNotFoundError(f"Missing Chinese Whispers setting directory: {cw_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    kmeans_summary = read_tsv(args.kmeans_summary)
    cw_summary = [
        row for row in read_tsv(args.cw_summary)
        if row.get("setting") == args.cw_setting
    ]

    kmeans_summary_by_compound = {
        row["compound"]: row
        for row in kmeans_summary
        if row.get("status") == "clustered"
    }
    cw_summary_by_compound = {
        row["compound"]: row
        for row in cw_summary
        if row.get("status") == "clustered"
    }

    common_compounds = sorted(
        set(kmeans_summary_by_compound) & set(cw_summary_by_compound)
    )
    if not common_compounds:
        raise ValueError("No compounds found in both K-means and Chinese Whispers outputs.")

    metrics_rows = []
    for compound in common_compounds:
        kmeans_path = os.path.join(args.kmeans_dir, f"{compound}.pickle")
        cw_path = os.path.join(cw_dir, f"{compound}.pickle")
        if not os.path.exists(kmeans_path) or not os.path.exists(cw_path):
            continue
        row = compare_compound(
            compound=compound,
            kmeans_path=kmeans_path,
            cw_path=cw_path,
            kmeans_summary_row=kmeans_summary_by_compound[compound],
            cw_summary_row=cw_summary_by_compound[compound],
        )
        metrics_rows.append(row)

    if not metrics_rows:
        raise ValueError("No overlapping compound pickle files were successfully compared.")

    metrics_rows.sort(key=lambda row: row["compound"])
    metrics_path = os.path.join(args.output_dir, "comparison_metrics.tsv")
    write_metrics_tsv(metrics_path, metrics_rows)

    global_path = os.path.join(args.output_dir, "global_summary.png")
    plot_global_summary(metrics_rows, args.cw_setting, global_path)

    if args.targets:
        selected_compounds = [compound for compound in args.targets if compound in {row["compound"] for row in metrics_rows}]
    else:
        selected_compounds = [
            row["compound"]
            for row in sorted(metrics_rows, key=lambda row: (row["ari"], -row["num_examples"]))[:args.top_n]
        ]

    per_target_dir = os.path.join(args.output_dir, "per_target")
    os.makedirs(per_target_dir, exist_ok=True)
    for compound in selected_compounds:
        row = next(row for row in metrics_rows if row["compound"] == compound)
        plot_target_comparison(
            row=row,
            output_path=os.path.join(per_target_dir, f"{compound}.png"),
            sample_size=args.sample_size,
            heatmap_top_clusters=args.heatmap_top_clusters,
            rng=rng,
        )

    print(f"Wrote metrics TSV to {metrics_path}")
    print(f"Wrote global summary figure to {global_path}")
    print(f"Wrote {len(selected_compounds)} per-target comparison figures to {per_target_dir}")


def compare_compound(*, compound, kmeans_path, cw_path, kmeans_summary_row, cw_summary_row):
    kmeans_examples = load_examples(kmeans_path)
    cw_examples = load_examples(cw_path)

    kmeans_by_id = index_examples_by_sentence_id(kmeans_examples)
    cw_by_id = index_examples_by_sentence_id(cw_examples)
    common_ids = sorted(set(kmeans_by_id) & set(cw_by_id))
    if not common_ids:
        raise ValueError(f"No shared sentence_id values for compound {compound}")

    embeddings = []
    years = []
    kmeans_labels = []
    cw_labels = []
    for sentence_id in common_ids:
        kmeans_example = kmeans_by_id[sentence_id]
        cw_example = cw_by_id[sentence_id]
        embeddings.append(np.asarray(kmeans_example["bert-vec-pooled"], dtype=np.float32))
        years.append(getattr(kmeans_example["sent"], "year", None))
        kmeans_labels.append(int(kmeans_example["cluster_label"]))
        cw_labels.append(int(cw_example["cluster_label"]))

    embeddings = np.stack(embeddings, axis=0)
    kmeans_labels = np.asarray(kmeans_labels, dtype=np.int32)
    cw_labels = np.asarray(cw_labels, dtype=np.int32)

    kmeans_counts = label_counts(kmeans_labels)
    cw_counts = label_counts(cw_labels)

    return {
        "compound": compound,
        "num_examples": len(common_ids),
        "kmeans_best_k": int(kmeans_summary_row["best_k"]),
        "kmeans_silhouette": float(kmeans_summary_row["best_silhouette"]),
        "cw_num_clusters": int(cw_summary_row["num_clusters"]),
        "cw_largest_cluster_size": int(cw_summary_row["largest_cluster_size"]),
        "ari": adjusted_rand_index(kmeans_labels, cw_labels),
        "variation_of_information": variation_of_information(kmeans_labels, cw_labels),
        "kmeans_largest_share": largest_cluster_share(kmeans_counts, len(common_ids)),
        "cw_largest_share": largest_cluster_share(cw_counts, len(common_ids)),
        "kmeans_labels": kmeans_labels,
        "cw_labels": cw_labels,
        "embeddings": embeddings,
        "years": years,
    }


def index_examples_by_sentence_id(examples):
    indexed = {}
    for example in examples:
        sentence = example.get("sent")
        sentence_id = getattr(sentence, "sentence_id", None)
        if sentence_id is None:
            continue
        indexed[sentence_id] = example
    return indexed


def read_tsv(path):
    with open(path, encoding="utf-8", newline="") as in_f:
        return list(csv.DictReader(in_f, delimiter="\t"))


def write_metrics_tsv(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as out_f:
        writer = csv.DictWriter(
            out_f,
            fieldnames=[
                "compound",
                "num_examples",
                "kmeans_best_k",
                "kmeans_silhouette",
                "cw_num_clusters",
                "cw_largest_cluster_size",
                "ari",
                "variation_of_information",
                "kmeans_largest_share",
                "cw_largest_share",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "compound": row["compound"],
                "num_examples": row["num_examples"],
                "kmeans_best_k": row["kmeans_best_k"],
                "kmeans_silhouette": f"{row['kmeans_silhouette']:.6f}",
                "cw_num_clusters": row["cw_num_clusters"],
                "cw_largest_cluster_size": row["cw_largest_cluster_size"],
                "ari": f"{row['ari']:.6f}",
                "variation_of_information": f"{row['variation_of_information']:.6f}",
                "kmeans_largest_share": f"{row['kmeans_largest_share']:.6f}",
                "cw_largest_share": f"{row['cw_largest_share']:.6f}",
            })


def plot_global_summary(rows, cw_setting, output_path):
    fig, axes = plt.subplots(2, 2, figsize=(18, 12), dpi=180)
    fig.suptitle(f"K-means vs Chinese Whispers ({cw_setting})", fontsize=16)

    rows_by_size = sorted(rows, key=lambda row: row["num_examples"])
    compounds = [row["compound"] for row in rows_by_size]
    x = np.arange(len(compounds))
    axes[0, 0].plot(x, [row["kmeans_best_k"] for row in rows_by_size], marker="o", linewidth=1.6, label="K-means clusters", color="#2B6CB0")
    axes[0, 0].plot(x, [row["cw_num_clusters"] for row in rows_by_size], marker="o", linewidth=1.6, label="CW clusters", color="#C05621")
    axes[0, 0].set_title("Cluster counts by compound")
    axes[0, 0].set_xlabel("Compound (sorted by number of usages)")
    axes[0, 0].set_ylabel("Clusters")
    axes[0, 0].set_xticks(x[::max(1, len(x) // 12)])
    axes[0, 0].set_xticklabels(compounds[::max(1, len(x) // 12)], rotation=45, ha="right")
    axes[0, 0].legend()

    axes[0, 1].scatter(
        [row["kmeans_best_k"] for row in rows],
        [row["cw_num_clusters"] for row in rows],
        s=[max(30, math.sqrt(row["num_examples"]) * 4) for row in rows],
        c=[row["ari"] for row in rows],
        cmap="viridis",
        alpha=0.85,
        edgecolors="black",
        linewidths=0.3,
    )
    max_axis = max(
        max(row["kmeans_best_k"] for row in rows),
        max(row["cw_num_clusters"] for row in rows),
    )
    axes[0, 1].plot([0, max_axis], [0, max_axis], linestyle="--", color="gray", linewidth=1)
    axes[0, 1].set_title("Cluster count relationship")
    axes[0, 1].set_xlabel("K-means best k")
    axes[0, 1].set_ylabel("CW number of clusters")

    rows_by_ari = sorted(rows, key=lambda row: row["ari"])
    axes[1, 0].bar(
        np.arange(len(rows_by_ari)),
        [row["ari"] for row in rows_by_ari],
        color="#2F855A",
    )
    axes[1, 0].set_title("Agreement per compound (ARI)")
    axes[1, 0].set_xlabel("Compound")
    axes[1, 0].set_ylabel("Adjusted Rand Index")
    axes[1, 0].set_xticks(np.arange(len(rows_by_ari)))
    axes[1, 0].set_xticklabels([row["compound"] for row in rows_by_ari], rotation=90)

    axes[1, 1].scatter(
        [row["num_examples"] for row in rows],
        [row["ari"] for row in rows],
        color="#805AD5",
        alpha=0.85,
    )
    axes[1, 1].set_title("Agreement vs corpus size")
    axes[1, 1].set_xlabel("Number of usages")
    axes[1, 1].set_ylabel("Adjusted Rand Index")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_target_comparison(*, row, output_path, sample_size, heatmap_top_clusters, rng):
    embeddings = row["embeddings"]
    kmeans_labels = row["kmeans_labels"]
    cw_labels = row["cw_labels"]
    years = row["years"]

    sample_indices = np.arange(embeddings.shape[0])
    if embeddings.shape[0] > sample_size:
        sample_indices = np.sort(rng.choice(embeddings.shape[0], size=sample_size, replace=False))

    projected = pca_2d(embeddings[sample_indices])
    year_colors = years_to_colors([years[idx] for idx in sample_indices])

    fig = plt.figure(figsize=(18, 6), dpi=180)
    grid = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.2, 1.0])
    ax_kmeans = fig.add_subplot(grid[0, 0])
    ax_cw = fig.add_subplot(grid[0, 1])
    ax_heatmap = fig.add_subplot(grid[0, 2])

    scatter_by_label(
        ax=ax_kmeans,
        points=projected,
        labels=kmeans_labels[sample_indices],
        year_colors=year_colors,
        title=f"{row['compound']} | K-means ({row['kmeans_best_k']} clusters)",
    )
    scatter_by_label(
        ax=ax_cw,
        points=projected,
        labels=cw_labels[sample_indices],
        year_colors=year_colors,
        title=f"{row['compound']} | CW ({row['cw_num_clusters']} clusters)",
    )

    heatmap, x_labels, y_labels = top_cluster_overlap_matrix(
        kmeans_labels=kmeans_labels,
        cw_labels=cw_labels,
        top_n=heatmap_top_clusters,
    )
    im = ax_heatmap.imshow(heatmap, aspect="auto", cmap="magma")
    ax_heatmap.set_title(
        f"Cluster overlap\nARI={row['ari']:.3f}, VI={row['variation_of_information']:.3f}"
    )
    ax_heatmap.set_xlabel("CW clusters")
    ax_heatmap.set_ylabel("K-means clusters")
    ax_heatmap.set_xticks(np.arange(len(x_labels)))
    ax_heatmap.set_xticklabels(x_labels, rotation=45, ha="right")
    ax_heatmap.set_yticks(np.arange(len(y_labels)))
    ax_heatmap.set_yticklabels(y_labels)
    plt.colorbar(im, ax=ax_heatmap, fraction=0.046, pad=0.04, label="Shared usages")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def scatter_by_label(*, ax, points, labels, year_colors, title):
    unique_labels = np.unique(labels)
    cmap = plt.get_cmap("tab20", max(20, unique_labels.size))
    colors = [cmap(int(label) % cmap.N) for label in labels]
    ax.scatter(points[:, 0], points[:, 1], c=colors, s=22, alpha=0.82, linewidths=0)
    ax.scatter(points[:, 0], points[:, 1], c=year_colors, s=6, alpha=0.65, linewidths=0)
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")


def top_cluster_overlap_matrix(*, kmeans_labels, cw_labels, top_n):
    top_kmeans = top_labels_by_size(kmeans_labels, top_n)
    top_cw = top_labels_by_size(cw_labels, top_n)
    row_order = top_kmeans + [None]
    col_order = top_cw + [None]

    matrix = np.zeros((len(row_order), len(col_order)), dtype=np.int32)
    for k_label, c_label in zip(kmeans_labels.tolist(), cw_labels.tolist()):
        row_idx = row_order.index(k_label) if k_label in top_kmeans else len(row_order) - 1
        col_idx = col_order.index(c_label) if c_label in top_cw else len(col_order) - 1
        matrix[row_idx, col_idx] += 1

    x_labels = [str(label) for label in top_cw] + ["other"]
    y_labels = [str(label) for label in top_kmeans] + ["other"]
    return matrix, x_labels, y_labels


def top_labels_by_size(labels, top_n):
    counts = label_counts(labels)
    return [
        label
        for label, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:top_n]
    ]


def years_to_colors(years):
    numeric_years = [year for year in years if year is not None]
    if not numeric_years:
        return ["#718096"] * len(years)
    min_year = min(numeric_years)
    max_year = max(numeric_years)
    if min_year == max_year:
        return ["#718096"] * len(years)
    colors = []
    cmap = plt.get_cmap("cividis")
    for year in years:
        if year is None:
            colors.append("#718096")
            continue
        scaled = (year - min_year) / float(max_year - min_year)
        colors.append(cmap(scaled))
    return colors


def pca_2d(X):
    centered = X - X.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[:2].T


def label_counts(labels):
    unique, counts = np.unique(labels, return_counts=True)
    return {int(label): int(count) for label, count in zip(unique, counts)}


def largest_cluster_share(counts, total):
    if not counts or total <= 0:
        return 0.0
    return max(counts.values()) / float(total)


def adjusted_rand_index(labels_a, labels_b):
    contingency = contingency_matrix(labels_a, labels_b)
    n = contingency.sum()
    if n < 2:
        return 1.0

    sum_comb = comb2_sum(contingency.ravel())
    row_comb = comb2_sum(contingency.sum(axis=1))
    col_comb = comb2_sum(contingency.sum(axis=0))
    total_comb = comb2(n)
    if total_comb == 0:
        return 1.0

    expected = (row_comb * col_comb) / total_comb
    max_index = 0.5 * (row_comb + col_comb)
    denom = max_index - expected
    if denom == 0.0:
        return 1.0
    return float((sum_comb - expected) / denom)


def variation_of_information(labels_a, labels_b):
    contingency = contingency_matrix(labels_a, labels_b).astype(np.float64)
    total = contingency.sum()
    if total == 0.0:
        return 0.0
    joint = contingency / total
    probs_a = joint.sum(axis=1, keepdims=True)
    probs_b = joint.sum(axis=0, keepdims=True)

    h_a = entropy(probs_a.ravel())
    h_b = entropy(probs_b.ravel())

    mutual_info = 0.0
    nz_rows, nz_cols = np.nonzero(joint)
    for i, j in zip(nz_rows.tolist(), nz_cols.tolist()):
        mutual_info += joint[i, j] * math.log(joint[i, j] / (probs_a[i, 0] * probs_b[0, j]))
    return float(h_a + h_b - 2.0 * mutual_info)


def entropy(probs):
    value = 0.0
    for prob in probs:
        if prob > 0.0:
            value -= prob * math.log(prob)
    return value


def contingency_matrix(labels_a, labels_b):
    unique_a = {label: idx for idx, label in enumerate(sorted(np.unique(labels_a).tolist()))}
    unique_b = {label: idx for idx, label in enumerate(sorted(np.unique(labels_b).tolist()))}
    matrix = np.zeros((len(unique_a), len(unique_b)), dtype=np.int64)
    for label_a, label_b in zip(labels_a.tolist(), labels_b.tolist()):
        matrix[unique_a[label_a], unique_b[label_b]] += 1
    return matrix


def comb2(value):
    return value * (value - 1) / 2.0


def comb2_sum(values):
    return float(sum(comb2(int(value)) for value in values))


if __name__ == "__main__":
    main()

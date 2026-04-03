from argparse import ArgumentParser
import csv
import os
import sys

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
WORKSPACE_ROOT = os.path.dirname(REPO_ROOT)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.cluster_compounds_only_kmeans import load_examples, pool_usage_embedding


def parse_args():
    parser = ArgumentParser(
        description="Project compound usage embeddings into 2D, optionally comparing K-means and Chinese Whispers side by side."
    )
    parser.add_argument(
        "--input_dir",
        default=os.path.join(
            WORKSPACE_ROOT,
            "german_target_pickles_with_bert_base_german_cased_kmeans_compounds_only_with_sentence_id",
        ),
    )
    parser.add_argument(
        "--compare_dir",
        default=None,
        help="Optional second clustering directory to compare side by side.",
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.join(
            REPO_ROOT,
            "eval-and-visualization",
            "outputs",
            "usage-embedding-visualizations",
        ),
    )
    parser.add_argument("--targets", nargs="*", default=None)
    parser.add_argument("--top_n", type=int, default=12)
    parser.add_argument("--sample_size", type=int, default=800)
    parser.add_argument("--random_seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    rng = np.random.RandomState(args.random_seed)

    if not os.path.isdir(args.input_dir):
        raise FileNotFoundError(f"Missing input directory: {args.input_dir}")
    if args.compare_dir and not os.path.isdir(args.compare_dir):
        raise FileNotFoundError(f"Missing compare directory: {args.compare_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    if args.compare_dir:
        rows = build_comparison_rows(args)
        summary_path = os.path.join(args.output_dir, "summary.tsv")
        write_comparison_summary(summary_path, rows)
        overview_path = os.path.join(args.output_dir, "overview.png")
        plot_comparison_overview(rows, overview_path)
        for row in rows:
            plot_side_by_side_compound(
                row=row,
                output_path=os.path.join(args.output_dir, f"{row['compound']}.png"),
                sample_size=args.sample_size,
                rng=rng,
            )
    else:
        rows = build_single_rows(args)
        summary_path = os.path.join(args.output_dir, "summary.tsv")
        write_single_summary(summary_path, rows)
        overview_path = os.path.join(args.output_dir, "overview.png")
        plot_single_overview(rows, overview_path)
        for row in rows:
            plot_single_compound(
                row=row,
                output_path=os.path.join(args.output_dir, f"{row['compound']}.png"),
                sample_size=args.sample_size,
                rng=rng,
            )

    print(f"Wrote summary TSV to {summary_path}")
    print(f"Wrote overview figure to {overview_path}")
    print(f"Wrote {len(rows)} per-compound figures to {args.output_dir}")


def build_single_rows(args):
    rows = []
    filenames = sorted(
        filename for filename in os.listdir(args.input_dir)
        if filename.endswith(".pickle") or filename.endswith(".pkl")
    )
    for filename in filenames:
        compound = os.path.splitext(filename)[0]
        if args.targets and compound not in args.targets:
            continue
        examples = load_examples(os.path.join(args.input_dir, filename))
        row = build_single_row(compound, examples)
        if row is not None:
            rows.append(row)
    if not rows:
        raise ValueError("No matching compounds with embeddings were found.")
    if not args.targets:
        rows = sorted(rows, key=lambda row: row["num_examples"], reverse=True)[:args.top_n]
    else:
        rows = sorted(rows, key=lambda row: row["compound"])
    return rows


def build_comparison_rows(args):
    first_files = {
        os.path.splitext(filename)[0]: filename
        for filename in os.listdir(args.input_dir)
        if filename.endswith(".pickle") or filename.endswith(".pkl")
    }
    second_files = {
        os.path.splitext(filename)[0]: filename
        for filename in os.listdir(args.compare_dir)
        if filename.endswith(".pickle") or filename.endswith(".pkl")
    }
    common_compounds = sorted(set(first_files) & set(second_files))
    if args.targets:
        common_compounds = [compound for compound in common_compounds if compound in args.targets]
    rows = []
    for compound in common_compounds:
        first_examples = load_examples(os.path.join(args.input_dir, first_files[compound]))
        second_examples = load_examples(os.path.join(args.compare_dir, second_files[compound]))
        row = build_comparison_row(compound, first_examples, second_examples)
        if row is not None:
            rows.append(row)
    if not rows:
        raise ValueError("No overlapping compounds with embeddings were found.")
    if not args.targets:
        rows = sorted(rows, key=lambda row: row["num_examples"], reverse=True)[:args.top_n]
    else:
        rows = sorted(rows, key=lambda row: row["compound"])
    return rows


def build_single_row(compound, examples):
    aligned = build_indexed_examples(examples)
    if not aligned:
        return None
    embeddings, labels, years = unpack_examples(aligned.values())
    return {
        "compound": compound,
        "num_examples": int(embeddings.shape[0]),
        "num_clusters": count_clusters(labels),
        "embeddings": embeddings,
        "labels": labels,
        "years": years,
    }


def build_comparison_row(compound, first_examples, second_examples):
    first_index = build_indexed_examples(first_examples)
    second_index = build_indexed_examples(second_examples)
    common_ids = sorted(set(first_index) & set(second_index))
    if not common_ids:
        return None

    first_aligned = [first_index[sentence_id] for sentence_id in common_ids]
    second_aligned = [second_index[sentence_id] for sentence_id in common_ids]

    embeddings, first_labels, years = unpack_examples(first_aligned)
    _, second_labels, _ = unpack_examples(second_aligned)
    return {
        "compound": compound,
        "num_examples": int(embeddings.shape[0]),
        "kmeans_clusters": count_clusters(first_labels),
        "cw_clusters": count_clusters(second_labels),
        "embeddings": embeddings,
        "kmeans_labels": first_labels,
        "cw_labels": second_labels,
        "years": years,
    }


def build_indexed_examples(examples):
    indexed = {}
    for example in examples:
        sentence = example.get("sent")
        sentence_id = getattr(sentence, "sentence_id", None)
        if sentence_id is None or sentence_id in indexed:
            continue
        if "bert-vec-pooled" not in example and "bert-vec" not in example:
            continue
        indexed[sentence_id] = example
    return indexed


def unpack_examples(examples):
    embeddings = []
    labels = []
    years = []
    for example in examples:
        if "bert-vec-pooled" in example:
            embedding = np.asarray(example["bert-vec-pooled"], dtype=np.float32)
        else:
            embedding = pool_usage_embedding(example["bert-vec"])
        embeddings.append(embedding)
        label = example.get("cluster_label")
        labels.append(-1 if label is None else int(label))
        years.append(getattr(example.get("sent"), "year", None))
    return np.stack(embeddings, axis=0), np.asarray(labels, dtype=np.int32), years


def plot_single_overview(rows, output_path):
    fig, ax = plt.subplots(figsize=(9, 6), dpi=180)
    ax.scatter(
        [row["num_examples"] for row in rows],
        [row["num_clusters"] for row in rows],
        s=55,
        c="#2B6CB0",
        alpha=0.85,
    )
    for row in rows:
        ax.annotate(row["compound"], (row["num_examples"], row["num_clusters"]), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_title("Selected compounds")
    ax.set_xlabel("Number of usage examples")
    ax.set_ylabel("Number of clusters")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_comparison_overview(rows, output_path):
    fig, ax = plt.subplots(figsize=(9, 6), dpi=180)
    ax.scatter(
        [row["kmeans_clusters"] for row in rows],
        [row["cw_clusters"] for row in rows],
        s=[max(40, np.sqrt(row["num_examples"]) * 4) for row in rows],
        c="#C05621",
        alpha=0.85,
    )
    max_val = max(max(row["kmeans_clusters"] for row in rows), max(row["cw_clusters"] for row in rows))
    ax.plot([0, max_val], [0, max_val], linestyle="--", color="gray", linewidth=1)
    for row in rows:
        ax.annotate(row["compound"], (row["kmeans_clusters"], row["cw_clusters"]), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_title("K-means vs Chinese Whispers cluster counts")
    ax.set_xlabel("K-means clusters")
    ax.set_ylabel("Chinese Whispers clusters")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_single_compound(row, output_path, sample_size, rng):
    sample_indices, points = sample_projection(row["embeddings"], sample_size, rng)
    sampled_labels = row["labels"][sample_indices]
    sampled_years = [row["years"][idx] for idx in sample_indices]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), dpi=180)
    scatter_by_cluster(ax=axes[0], points=points, labels=sampled_labels, title=f"{row['compound']} | clusters ({row['num_clusters']})")
    scatter_by_year(ax=axes[1], points=points, years=sampled_years, title=f"{row['compound']} | years")
    fig.suptitle(f"{row['compound']} | {row['num_examples']} usages", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_side_by_side_compound(row, output_path, sample_size, rng):
    sample_indices, points = sample_projection(row["embeddings"], sample_size, rng)
    kmeans_labels = row["kmeans_labels"][sample_indices]
    cw_labels = row["cw_labels"][sample_indices]
    sampled_years = [row["years"][idx] for idx in sample_indices]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), dpi=180)
    scatter_by_cluster(ax=axes[0], points=points, labels=kmeans_labels, title=f"{row['compound']} | K-means ({row['kmeans_clusters']})")
    scatter_by_cluster(ax=axes[1], points=points, labels=cw_labels, title=f"{row['compound']} | Chinese Whispers ({row['cw_clusters']})")
    scatter_by_year(ax=axes[2], points=points, years=sampled_years, title=f"{row['compound']} | years")
    fig.suptitle(f"{row['compound']} | {row['num_examples']} shared usages", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def sample_projection(embeddings, sample_size, rng):
    sample_indices = np.arange(embeddings.shape[0])
    if embeddings.shape[0] > sample_size:
        sample_indices = np.sort(rng.choice(embeddings.shape[0], size=sample_size, replace=False))
    points = pca_2d(embeddings[sample_indices])
    return sample_indices, points


def scatter_by_cluster(*, ax, points, labels, title):
    unique_labels = sorted(np.unique(labels).tolist())
    cmap = plt.get_cmap("tab20", max(20, len(unique_labels)))
    color_lookup = {}
    for idx, label in enumerate(unique_labels):
        color_lookup[label] = "#A0AEC0" if label < 0 else cmap(idx % cmap.N)
    colors = [color_lookup[label] for label in labels.tolist()]
    ax.scatter(points[:, 0], points[:, 1], c=colors, s=20, alpha=0.84, linewidths=0)
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")


def scatter_by_year(*, ax, points, years, title):
    numeric_years = [year for year in years if year is not None]
    if numeric_years:
        min_year = min(numeric_years)
        max_year = max(numeric_years)
        cmap = plt.get_cmap("cividis")
        colors = []
        for year in years:
            if year is None or min_year == max_year:
                colors.append("#718096")
            else:
                colors.append(cmap((year - min_year) / float(max_year - min_year)))
    else:
        colors = ["#718096"] * len(years)
    ax.scatter(points[:, 0], points[:, 1], c=colors, s=20, alpha=0.84, linewidths=0)
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")


def pca_2d(X):
    centered = X - X.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[:2].T


def count_clusters(labels):
    return len([label for label in np.unique(labels).tolist() if label >= 0])


def write_single_summary(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=["compound", "num_examples", "num_clusters"], delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "compound": row["compound"],
                "num_examples": row["num_examples"],
                "num_clusters": row["num_clusters"],
            })


def write_comparison_summary(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as out_f:
        writer = csv.DictWriter(
            out_f,
            fieldnames=["compound", "num_examples", "kmeans_clusters", "cw_clusters"],
            delimiter="\t",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "compound": row["compound"],
                "num_examples": row["num_examples"],
                "kmeans_clusters": row["kmeans_clusters"],
                "cw_clusters": row["cw_clusters"],
            })


if __name__ == "__main__":
    main()

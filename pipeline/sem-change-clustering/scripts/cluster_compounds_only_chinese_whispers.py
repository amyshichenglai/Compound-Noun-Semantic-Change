from argparse import ArgumentParser
import csv
import os
import pickle
import sys
from types import ModuleType

import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import data as repo_data
from constants import BERT_VECS_C


src_pkg = ModuleType("src")
src_pkg.data = repo_data
sys.modules["src"] = src_pkg
sys.modules["src.data"] = repo_data


class LegacySentence:
    def __init__(self, *args, **kwargs):
        self.__dict__.update(kwargs)


POOLED_BERT_VECS_C = "bert-vec-pooled"
CLUSTER_LABEL_C = "cluster_label"


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--k_values", nargs="+", type=int, default=[5, 10, 20])
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--ensemble_runs", type=int, default=9)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    filenames = sorted(
        filename for filename in os.listdir(args.input_dir)
        if filename.endswith(".pickle") or filename.endswith(".pkl")
    )

    summary_rows = []
    csv_rows = []

    for requested_k in args.k_values:
        setting_name = f"cw_knn_k{requested_k}_iter{args.iterations}_ens{args.ensemble_runs}"
        setting_output_dir = os.path.join(args.output_dir, setting_name)
        os.makedirs(setting_output_dir, exist_ok=True)

        for file_idx, filename in enumerate(filenames):
            input_path = os.path.join(args.input_dir, filename)
            output_filename = filename[:-4] + ".pickle" if filename.endswith(".pkl") else filename
            output_path = os.path.join(setting_output_dir, output_filename)
            rows, summary_row = process_file(
                input_path=input_path,
                output_path=output_path,
                filename=filename,
                requested_k=requested_k,
                iterations=args.iterations,
                ensemble_runs=args.ensemble_runs,
                seed=args.seed + file_idx,
                setting_name=setting_name,
            )
            csv_rows.extend(rows)
            summary_rows.append(summary_row)

    summary_tsv = os.path.join(args.output_dir, "summary.tsv")
    with open(summary_tsv, "w", encoding="utf-8", newline="") as out_f:
        writer = csv.DictWriter(
            out_f,
            fieldnames=[
                "setting",
                "filename",
                "compound",
                "num_examples",
                "requested_k",
                "effective_k",
                "iterations",
                "ensemble_runs",
                "num_clusters",
                "largest_cluster_size",
                "status",
                "reason",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    output_csv = os.path.join(args.output_dir, "clustering_results.csv")
    with open(output_csv, "w", encoding="utf-8", newline="") as out_f:
        writer = csv.DictWriter(
            out_f,
            fieldnames=[
                "setting",
                "compound",
                "sentence_usage",
                "sentence_id",
                "cluster_id",
                "year",
                "requested_k",
                "effective_k",
                "iterations",
                "ensemble_runs",
            ],
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"Wrote CSV to {output_csv}")
    print(f"Wrote summary to {summary_tsv}")


def process_file(*, input_path, output_path, filename, requested_k, iterations, ensemble_runs, seed, setting_name):
    examples = load_examples(input_path)
    compound = os.path.splitext(os.path.basename(filename))[0]
    num_examples = len(examples)

    if not examples:
        save_examples(output_path, [])
        return [], summary_row(
            setting=setting_name,
            filename=filename,
            compound=compound,
            num_examples=0,
            requested_k=requested_k,
            effective_k=0,
            iterations=iterations,
            ensemble_runs=ensemble_runs,
            num_clusters=0,
            largest_cluster_size=0,
            status="skipped",
            reason="no_examples",
        )

    pooled = [pool_usage_embedding(example[BERT_VECS_C]) for example in examples]
    X = l2_normalize(np.stack(pooled))
    effective_k = min(requested_k, max(1, num_examples - 1))
    labels = ensemble_chinese_whispers(
        X=X,
        k=effective_k,
        iterations=iterations,
        ensemble_runs=ensemble_runs,
        seed=seed,
    )

    clustered_examples = []
    cluster_sizes = {}
    for label in labels.tolist():
        cluster_sizes[label] = cluster_sizes.get(label, 0) + 1

    rows = []
    for example, pooled_vec, label in zip(examples, pooled, labels.tolist()):
        updated = dict(example)
        updated[POOLED_BERT_VECS_C] = pooled_vec
        updated[CLUSTER_LABEL_C] = int(label)
        clustered_examples.append(updated)

        period = year_to_period(example["sent"].year)
        if period is None:
            continue
        sentence_id = getattr(example["sent"], "sentence_id", None)
        if sentence_id is None:
            raise ValueError(f"Missing sentence_id in {filename}")
        rows.append({
            "setting": setting_name,
            "compound": compound,
            "sentence_usage": str(example["sent"]),
            "sentence_id": sentence_id,
            "cluster_id": int(label),
            "year": period,
            "requested_k": requested_k,
            "effective_k": effective_k,
            "iterations": iterations,
            "ensemble_runs": ensemble_runs,
        })

    save_examples(output_path, clustered_examples)
    largest_cluster_size = max(cluster_sizes.values()) if cluster_sizes else 0
    return rows, summary_row(
        setting=setting_name,
        filename=filename,
        compound=compound,
        num_examples=num_examples,
        requested_k=requested_k,
        effective_k=effective_k,
        iterations=iterations,
        ensemble_runs=ensemble_runs,
        num_clusters=len(cluster_sizes),
        largest_cluster_size=largest_cluster_size,
        status="clustered",
        reason="",
    )


def summary_row(
    *,
    setting,
    filename,
    compound,
    num_examples,
    requested_k,
    effective_k,
    iterations,
    ensemble_runs,
    num_clusters,
    largest_cluster_size,
    status,
    reason,
):
    return {
        "setting": setting,
        "filename": filename,
        "compound": compound,
        "num_examples": num_examples,
        "requested_k": requested_k,
        "effective_k": effective_k,
        "iterations": iterations,
        "ensemble_runs": ensemble_runs,
        "num_clusters": num_clusters,
        "largest_cluster_size": largest_cluster_size,
        "status": status,
        "reason": reason,
    }


def ensemble_chinese_whispers(*, X, k, iterations, ensemble_runs, seed):
    similarity = cosine_similarity_matrix(X)
    adjacency = build_symmetric_knn_graph(similarity, k)

    membership = np.empty((ensemble_runs, X.shape[0]), dtype=np.int32)
    for run_idx in range(ensemble_runs):
        membership[run_idx] = chinese_whispers(
            adjacency=adjacency,
            iterations=iterations,
            seed=seed + run_idx,
        )

    if ensemble_runs == 1:
        return remap_labels_by_size(membership[0])

    coassoc = np.zeros_like(adjacency)
    for labels in membership:
        same_cluster = labels[:, None] == labels[None, :]
        coassoc += same_cluster.astype(np.float32)
    coassoc /= float(ensemble_runs)
    consensus_weights = adjacency * coassoc
    labels = chinese_whispers(
        adjacency=consensus_weights,
        iterations=iterations,
        seed=seed + ensemble_runs + 1,
    )
    return remap_labels_by_size(labels)


def chinese_whispers(*, adjacency, iterations, seed):
    num_nodes = adjacency.shape[0]
    labels = np.arange(num_nodes, dtype=np.int32)
    rng = np.random.RandomState(seed)

    for _ in range(iterations):
        order = rng.permutation(num_nodes)
        changed = False
        for node_idx in order:
            neighbors = np.flatnonzero(adjacency[node_idx] > 0.0)
            if neighbors.size == 0:
                continue
            neighbor_labels = labels[neighbors]
            weights = adjacency[node_idx, neighbors]
            label_scores = {}
            for label, weight in zip(neighbor_labels.tolist(), weights.tolist()):
                label_scores[label] = label_scores.get(label, 0.0) + weight
            best_score = None
            best_label = labels[node_idx]
            for label in sorted(label_scores):
                score = label_scores[label]
                if best_score is None or score > best_score:
                    best_score = score
                    best_label = label
            if best_label != labels[node_idx]:
                labels[node_idx] = best_label
                changed = True
        if not changed:
            break
    return labels


def build_symmetric_knn_graph(similarity, k):
    num_nodes = similarity.shape[0]
    if num_nodes <= 1:
        return np.zeros_like(similarity)

    working = similarity.copy()
    np.fill_diagonal(working, -np.inf)
    k = min(k, num_nodes - 1)
    adjacency = np.zeros_like(similarity, dtype=np.float32)

    for row_idx in range(num_nodes):
        neighbor_indices = np.argpartition(working[row_idx], -k)[-k:]
        for col_idx in neighbor_indices.tolist():
            weight = similarity[row_idx, col_idx]
            if weight > 0.0:
                adjacency[row_idx, col_idx] = weight

    adjacency = np.maximum(adjacency, adjacency.T)
    np.fill_diagonal(adjacency, 0.0)
    return adjacency


def cosine_similarity_matrix(X):
    similarity = X @ X.T
    np.fill_diagonal(similarity, 0.0)
    return similarity.astype(np.float32, copy=False)


def remap_labels_by_size(labels):
    counts = {}
    for label in labels.tolist():
        counts[label] = counts.get(label, 0) + 1
    sorted_labels = sorted(counts, key=lambda label: (-counts[label], label))
    remap = {label: idx for idx, label in enumerate(sorted_labels)}
    return np.asarray([remap[label] for label in labels.tolist()], dtype=np.int32)


def l2_normalize(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return X / norms


def pool_usage_embedding(embedding):
    if isinstance(embedding, np.ndarray):
        return embedding.astype(np.float32, copy=False)
    if len(embedding) == 1:
        return np.asarray(embedding[0], dtype=np.float32)
    stacked = np.stack([np.asarray(vec, dtype=np.float32) for vec in embedding], axis=0)
    return stacked.mean(axis=0)


def year_to_period(year):
    if 1700 <= year <= 1759:
        return "early"
    if 1870 <= year <= 1909:
        return "late"
    return None


def save_examples(path, examples):
    with open(path, "wb") as out_f:
        pickle.dump(examples, out_f, protocol=pickle.HIGHEST_PROTOCOL)


def load_examples(path):
    examples = []
    main_module = sys.modules["__main__"]
    previous_sentence = getattr(main_module, "Sentence", None)
    setattr(main_module, "Sentence", LegacySentence)
    try:
        with open(path, "rb") as in_f:
            try:
                while True:
                    examples.extend(pickle.load(in_f))
            except EOFError:
                pass
    finally:
        if previous_sentence is None:
            delattr(main_module, "Sentence")
        else:
            setattr(main_module, "Sentence", previous_sentence)
    return examples


if __name__ == "__main__":
    main()

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
        pass


POOLED_BERT_VECS_C = "bert-vec-pooled"
CLUSTER_LABEL_C = "cluster_label"


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--summary_tsv", default=None)
    parser.add_argument("--min_k", type=int, default=4)
    parser.add_argument("--max_k", type=int, default=32)
    parser.add_argument("--random_state", type=int, default=0)
    parser.add_argument("--n_init", type=int, default=10)
    parser.add_argument("--max_iter", type=int, default=100)
    parser.add_argument("--tol", type=float, default=1e-4)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    summary_tsv = args.summary_tsv or os.path.join(args.output_dir, "summary.tsv")

    filenames = sorted(
        filename for filename in os.listdir(args.input_dir)
        if filename.endswith(".pickle") or filename.endswith(".pkl")
    )

    rows = []
    for filename in filenames:
        input_path = os.path.join(args.input_dir, filename)
        output_filename = filename[:-4] + ".pickle" if filename.endswith(".pkl") else filename
        output_path = os.path.join(args.output_dir, output_filename)
        rows.append(process_file(
            input_path=input_path,
            output_path=output_path,
            filename=filename,
            min_k=args.min_k,
            max_k=args.max_k,
            random_state=args.random_state,
            n_init=args.n_init,
            max_iter=args.max_iter,
            tol=args.tol,
        ))

    with open(summary_tsv, "w", encoding="utf-8", newline="") as out_f:
        writer = csv.DictWriter(
            out_f,
            fieldnames=[
                "filename",
                "compound",
                "num_examples",
                "status",
                "best_k",
                "best_silhouette",
                "searched_k_values",
                "reason",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} summary rows to {summary_tsv}")


def process_file(*, input_path, output_path, filename, min_k, max_k, random_state, n_init, max_iter, tol):
    examples = load_examples(input_path)
    compound = os.path.splitext(os.path.basename(filename))[0]
    num_examples = len(examples)

    if not examples:
        save_examples(output_path, [])
        return summary_row(
            filename=filename,
            compound=compound,
            num_examples=0,
            status="skipped",
            best_k="",
            best_silhouette="",
            searched_k_values="",
            reason="no_examples",
        )

    pooled = []
    for example in examples:
        pooled_vec = pool_usage_embedding(example[BERT_VECS_C])
        pooled.append(pooled_vec)

    X = l2_normalize(np.stack(pooled))
    k_values = candidate_k_values(num_examples=num_examples, min_k=min_k, max_k=max_k)
    if not k_values:
        clustered_examples = []
        for example, pooled_vec in zip(examples, pooled):
            updated = dict(example)
            updated[POOLED_BERT_VECS_C] = pooled_vec
            updated[CLUSTER_LABEL_C] = None
            clustered_examples.append(updated)
        save_examples(output_path, clustered_examples)
        return summary_row(
            filename=filename,
            compound=compound,
            num_examples=num_examples,
            status="skipped",
            best_k="",
            best_silhouette="",
            searched_k_values="",
            reason=f"not_enough_examples_for_min_k_{min_k}",
        )

    best_k = None
    best_silhouette = None
    best_labels = None
    distance_matrix = pairwise_euclidean_distances(X)
    for k in k_values:
        labels, _ = run_kmeans(
            X=X,
            k=k,
            random_state=random_state,
            n_init=n_init,
            max_iter=max_iter,
            tol=tol,
        )
        if len(set(labels.tolist())) < 2:
            continue
        score = silhouette_score_from_distance_matrix(distance_matrix, labels)
        if best_silhouette is None or score > best_silhouette:
            best_k = k
            best_silhouette = score
            best_labels = labels

    if best_labels is None:
        clustered_examples = []
        for example, pooled_vec in zip(examples, pooled):
            updated = dict(example)
            updated[POOLED_BERT_VECS_C] = pooled_vec
            updated[CLUSTER_LABEL_C] = None
            clustered_examples.append(updated)
        save_examples(output_path, clustered_examples)
        return summary_row(
            filename=filename,
            compound=compound,
            num_examples=num_examples,
            status="skipped",
            best_k="",
            best_silhouette="",
            searched_k_values=",".join(str(k) for k in k_values),
            reason="no_valid_k_found",
        )

    clustered_examples = []
    for example, pooled_vec, label in zip(examples, pooled, best_labels):
        updated = dict(example)
        updated[POOLED_BERT_VECS_C] = pooled_vec
        updated[CLUSTER_LABEL_C] = int(label)
        clustered_examples.append(updated)
    save_examples(output_path, clustered_examples)

    return summary_row(
        filename=filename,
        compound=compound,
        num_examples=num_examples,
        status="clustered",
        best_k=best_k,
        best_silhouette=f"{best_silhouette:.6f}",
        searched_k_values=",".join(str(k) for k in k_values),
        reason="",
    )


def summary_row(*, filename, compound, num_examples, status, best_k, best_silhouette, searched_k_values, reason):
    return {
        "filename": filename,
        "compound": compound,
        "num_examples": num_examples,
        "status": status,
        "best_k": best_k,
        "best_silhouette": best_silhouette,
        "searched_k_values": searched_k_values,
        "reason": reason,
    }


def candidate_k_values(*, num_examples, min_k, max_k):
    upper = min(max_k, num_examples - 1)
    if upper < min_k:
        return []
    return list(range(min_k, upper + 1))


def l2_normalize(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return X / norms


def pairwise_euclidean_distances(X):
    squared_norms = np.sum(X * X, axis=1, keepdims=True)
    squared = squared_norms + squared_norms.T - 2.0 * (X @ X.T)
    np.maximum(squared, 0.0, out=squared)
    return np.sqrt(squared, out=squared)


def run_kmeans(*, X, k, random_state, n_init, max_iter, tol):
    rng = np.random.RandomState(random_state)
    best_labels = None
    best_inertia = None

    for _ in range(n_init):
        centroid_indices = rng.choice(X.shape[0], size=k, replace=False)
        centroids = X[centroid_indices].copy()

        for _ in range(max_iter):
            distances = squared_distances_to_centroids(X, centroids)
            labels = distances.argmin(axis=1)

            new_centroids = np.zeros_like(centroids)
            for cluster_idx in range(k):
                mask = labels == cluster_idx
                if np.any(mask):
                    new_centroids[cluster_idx] = X[mask].mean(axis=0)
                else:
                    new_centroids[cluster_idx] = X[rng.randint(X.shape[0])]

            shift = np.linalg.norm(new_centroids - centroids)
            centroids = new_centroids
            if shift <= tol:
                break

        distances = squared_distances_to_centroids(X, centroids)
        labels = distances.argmin(axis=1)
        inertia = float(distances[np.arange(X.shape[0]), labels].sum())
        if best_inertia is None or inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()

    return best_labels, best_inertia


def squared_distances_to_centroids(X, centroids):
    x_sq = np.sum(X * X, axis=1, keepdims=True)
    c_sq = np.sum(centroids * centroids, axis=1)
    distances = x_sq + c_sq - 2.0 * (X @ centroids.T)
    np.maximum(distances, 0.0, out=distances)
    return distances


def silhouette_score_from_distance_matrix(distance_matrix, labels):
    n_samples = labels.shape[0]
    unique_labels = np.unique(labels)
    if unique_labels.size < 2:
        return float("-inf")

    cluster_members = {label: np.flatnonzero(labels == label) for label in unique_labels}
    silhouettes = np.zeros(n_samples, dtype=np.float32)

    for i in range(n_samples):
        label = labels[i]
        same_cluster = cluster_members[label]

        if same_cluster.size <= 1:
            silhouettes[i] = 0.0
            continue

        intra_distances = distance_matrix[i, same_cluster]
        a_i = (intra_distances.sum() - 0.0) / (same_cluster.size - 1)

        b_i = None
        for other_label in unique_labels:
            if other_label == label:
                continue
            other_cluster = cluster_members[other_label]
            mean_dist = distance_matrix[i, other_cluster].mean()
            if b_i is None or mean_dist < b_i:
                b_i = mean_dist

        denom = max(a_i, b_i)
        silhouettes[i] = 0.0 if denom == 0.0 else (b_i - a_i) / denom

    return float(silhouettes.mean())


def pool_usage_embedding(embedding):
    if isinstance(embedding, np.ndarray):
        return embedding.astype(np.float32, copy=False)
    if len(embedding) == 1:
        return np.asarray(embedding[0], dtype=np.float32)
    stacked = np.stack([np.asarray(vec, dtype=np.float32) for vec in embedding], axis=0)
    return stacked.mean(axis=0)


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

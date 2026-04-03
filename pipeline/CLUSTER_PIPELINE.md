# Cluster Pipeline Documentation

This document explains the clustering pipeline implemented in this workspace for German target-compound usages:

1. BERT embedding generation
2. K-means clustering
3. Chinese Whispers clustering

The implementation is centered on per-target pickle files. Each pickle file contains all usages for one target compound, and the pipeline enriches those examples step by step.

## Pipeline Overview

The data flow is:

`german_target_pickles/`
-> BERT embedding script
-> `german_target_pickles_with_bert_base_german_cased/`
-> K-means script or Chinese Whispers script
-> clustered pickle outputs + summary files

The main scripts are:

- `scripts/add_bert_embeddings_to_target_pickles.py`
- `scripts/embed_german_target_pickles.sh`
- `scripts/cluster_compounds_only_kmeans.py`
- `scripts/cluster_compounds_only_chinese_whispers.py`
- `scripts/restore_sentence_ids_in_pickles.py`

## Input Data Format

Each input pickle stores a list of usage examples. A usage example is a dictionary with fields such as:

- `sent`: a `Sentence` object from [`data.py`](/Users/racheltan/Desktop/csc2611/project_code/extraction/sem-change-clustering/data.py)
- `lemma-span`: a tuple `(start, end)` pointing to the target span in lemma-token indices
- `target_lemma`: optional target string used to repair span alignment
- `raw_tokens` or `raw_sentence`: optional raw-text form used when embeddings should be built from original tokens

The `Sentence` object contains:

- `tokens`
- `lemmas`
- `tags`
- `year`
- optionally `sentence_id`

The constant names used by the pipeline are defined in [`constants.py`](/Users/racheltan/Desktop/csc2611/project_code/extraction/sem-change-clustering/constants.py):

- `bert-vec`
- `lemma-span`

## Stage 1: BERT Embeddings

### Files

- Implementation: [`scripts/add_bert_embeddings_to_target_pickles.py`](/Users/racheltan/Desktop/csc2611/project_code/extraction/sem-change-clustering/scripts/add_bert_embeddings_to_target_pickles.py)
- Convenience wrapper: [`scripts/embed_german_target_pickles.sh`](/Users/racheltan/Desktop/csc2611/project_code/extraction/sem-change-clustering/scripts/embed_german_target_pickles.sh)

### Purpose

This stage reads per-target pickle files and adds contextual BERT vectors for each usage of the target compound.

### Command Interface

The Python script accepts:

- `--input_dir`: directory containing `.pickle` or `.pkl` target files
- `--output_dir`: directory where enriched pickles are written
- `--model`: Hugging Face model name or path
- `--tokenizer`: tokenizer name or path, defaulting to `--model`
- `--device`: usually `cpu` or `cuda`
- `--layer_slice`: one of `first`, `mid`, `last`
- `--text_field`: one of `lemmas`, `tokens`, `raw`

The shell wrapper defaults to:

- input: `german_target_pickles`
- output: `german_target_pickles_with_bert_base_german_cased`
- model: `bert-base-german-cased`
- text field: `lemmas`
- layer slice: `first`

### What the Script Does

For each example in each pickle file:

1. It loads the sentence and target span.
2. It optionally repairs `lemma-span` using `target_lemma` if the stored span looks unreliable.
3. It chooses the text source:
   - `lemmas`: use `sent.lemmas`
   - `tokens`: use `sent.tokens`
   - `raw`: use `raw_tokens` or split `raw_sentence`
4. It tokenizes the sentence with the model tokenizer using `process_tokenized_sent(...)` from [`data.py`](/Users/racheltan/Desktop/csc2611/project_code/extraction/sem-change-clustering/data.py).
5. It truncates to `MAX_SEQ_LENGTH = 128`, then adds `[CLS]` and `[SEP]`.
6. It remaps the target span from word-level indices to subword indices.
7. It runs the masked-language-model encoder with `output_hidden_states=True`.
8. It selects a slice of hidden layers:
   - `first`: layers `1:5`
   - `mid`: layers `5:-4`
   - `last`: layers `-4:13`
9. For each target subword, it sums the vectors across the selected layers.
10. It stores the resulting vectors under `bert-vec`.

### Output Format

Each output example is a copy of the original example plus:

- `bert-vec`: a list of NumPy vectors, one per target subword

The script also rebuilds the `Sentence` object so the output is compatible with the local repo classes, and it preserves `sentence_id` when present.

### Important Behavior

- Examples are skipped if span remapping fails or if the target falls outside the truncated sequence.
- If `raw` mode is used, punctuation filtering is applied when aligning raw tokens to the already aligned tokens.
- Output filenames are normalized to `.pickle`.

## Stage 2: K-means Clustering

### File

- [`scripts/cluster_compounds_only_kmeans.py`](/Users/racheltan/Desktop/csc2611/project_code/extraction/sem-change-clustering/scripts/cluster_compounds_only_kmeans.py)

### Purpose

This stage clusters usages of each compound independently using pooled BERT embeddings and a custom NumPy implementation of K-means.

### Command Interface

Arguments:

- `--input_dir`: directory of BERT-enriched pickle files
- `--output_dir`: directory for clustered pickles
- `--summary_tsv`: optional summary path
- `--min_k`: minimum number of clusters to try, default `4`
- `--max_k`: maximum number of clusters to try, default `32`
- `--random_state`: default `0`
- `--n_init`: number of random restarts, default `10`
- `--max_iter`: maximum K-means iterations per restart, default `100`
- `--tol`: centroid-shift stopping tolerance, default `1e-4`

### What the Script Does

For each target pickle:

1. It loads all examples for that compound.
2. It reads `bert-vec` from each example.
3. It pools the target embedding:
   - if the embedding is already a single array, use it
   - if the target has multiple subword vectors, average them
4. It L2-normalizes all pooled vectors.
5. It constructs candidate `k` values from `min_k` to `min(max_k, n_examples - 1)`.
6. It runs custom K-means for each candidate `k`.
7. It scores each result using a custom silhouette computation over the pairwise Euclidean distance matrix.
8. It chooses the `k` with the best silhouette score.
9. It writes cluster labels back into each example.

### K-means Implementation Details

The K-means code is implemented directly in this file instead of using scikit-learn:

- initial centroids are sampled from existing examples
- assignments are based on squared Euclidean distance
- empty clusters are reinitialized with a random example
- the best restart is selected by lowest inertia

### Output Format

Each output example gains:

- `bert-vec-pooled`: the pooled usage embedding
- `cluster_label`: the assigned cluster id, or `None` if clustering was skipped

The script also writes a TSV summary with:

- filename
- compound
- number of examples
- status
- best `k`
- best silhouette
- searched `k` values
- skip reason if applicable

### When K-means Is Skipped

K-means does not run for a target when:

- the pickle has no examples
- there are too few examples to support `min_k`
- no candidate `k` produces at least two distinct clusters

In those cases, `cluster_label` is written as `None`.

## Stage 3: Chinese Whispers Clustering

### File

- [`scripts/cluster_compounds_only_chinese_whispers.py`](/Users/racheltan/Desktop/csc2611/project_code/extraction/sem-change-clustering/scripts/cluster_compounds_only_chinese_whispers.py)

### Purpose

This stage clusters each compound with a graph-based Chinese Whispers algorithm over a symmetric k-nearest-neighbor similarity graph.

### Command Interface

Arguments:

- `--input_dir`: directory of BERT-enriched pickle files
- `--output_dir`: directory for clustering outputs
- `--k_values`: list of graph neighborhood sizes, default `5 10 20`
- `--iterations`: number of CW label-propagation passes, default `20`
- `--ensemble_runs`: number of repeated runs before consensus, default `9`
- `--seed`: base random seed, default `0`

### What the Script Does

For each requested `k`, and for each target pickle:

1. It loads examples.
2. It pools `bert-vec` into one vector per usage by averaging subword vectors.
3. It L2-normalizes the pooled vectors.
4. It computes the cosine similarity matrix using matrix multiplication.
5. It builds a symmetric k-nearest-neighbor graph:
   - self-similarity is removed
   - each node keeps its top `k` neighbors
   - only positive similarities are used as edge weights
   - the graph is symmetrized with `max(A, A^T)`
6. It runs Chinese Whispers multiple times with randomized node order.
7. It optionally builds a consensus graph using co-association across runs.
8. It runs Chinese Whispers once more on the consensus-weighted graph.
9. It remaps labels by cluster size so cluster `0` is the largest cluster.
10. It writes the clustered pickles and tabular outputs.

### Chinese Whispers Implementation Details

The local `chinese_whispers(...)` implementation:

- starts each node in its own cluster
- visits nodes in random order each iteration
- for each node, sums incoming edge weights by neighbor label
- assigns the label with the highest total weight
- stops early if no node changes label

### Ensemble Consensus

When `ensemble_runs > 1`:

1. Multiple CW labelings are collected.
2. A co-association matrix is built, measuring how often node pairs land in the same cluster.
3. The original adjacency matrix is multiplied by the co-association matrix.
4. A final CW pass is run on this consensus-weighted graph.

This makes the result less sensitive to any single random run.

### Output Format

For each CW setting, the script creates a subdirectory named like:

- `cw_knn_k5_iter20_ens9`
- `cw_knn_k10_iter20_ens9`
- `cw_knn_k20_iter20_ens9`

Each output example gains:

- `bert-vec-pooled`
- `cluster_label`

The script also writes:

- `summary.tsv`: one row per target and setting
- `clustering_results.csv`: one row per sentence usage and setting

The CSV includes:

- setting
- compound
- sentence text
- sentence_id
- cluster_id
- year bucket
- requested/effective `k`
- iteration count
- ensemble run count

### Year Bucketing

Chinese Whispers exports a coarse time label:

- `early` for years `1700-1759`
- `late` for years `1870-1909`

Examples outside those ranges are still clustered, but they are not exported into the CSV rows that require a time period.

### Sentence ID Requirement

The CSV export requires `sentence_id` on each sentence. If those IDs are missing, the script raises an error.

The helper script [`scripts/restore_sentence_ids_in_pickles.py`](/Users/racheltan/Desktop/csc2611/project_code/extraction/sem-change-clustering/scripts/restore_sentence_ids_in_pickles.py) restores missing IDs by matching examples in a source directory against examples in a target directory and copying over `sentence_id`.

## Shared Utility Behavior

### Pickle Loading Compatibility

All three scripts use a compatibility loader pattern:

- they temporarily inject a `LegacySentence` class into `__main__`
- they load old pickle objects safely
- they convert back into the repo's local `Sentence` class where needed

This is why the pipeline can read older `.pkl` or `.pickle` files produced by earlier versions of the project.

### Embedding Pooling

Both clustering scripts use the same rule:

- one target subword -> use that vector directly
- multiple target subwords -> average the vectors

This means clustering is always performed on one fixed-size vector per usage.

## Recommended End-to-End Usage

### 1. Add BERT embeddings

```bash
cd /Users/racheltan/Desktop/csc2611/project_code/extraction/sem-change-clustering
bash scripts/embed_german_target_pickles.sh
```

### 2. Run K-means

```bash
cd /Users/racheltan/Desktop/csc2611/project_code/extraction/sem-change-clustering
python scripts/cluster_compounds_only_kmeans.py \
  --input_dir ../german_target_pickles_with_bert_base_german_cased \
  --output_dir ../german_target_pickles_with_bert_base_german_cased_kmeans_compounds_only
```

### 3. Run Chinese Whispers

```bash
cd /Users/racheltan/Desktop/csc2611/project_code/extraction/sem-change-clustering
python scripts/cluster_compounds_only_chinese_whispers.py \
  --input_dir ../german_target_pickles_with_bert_base_german_cased_with_sentence_id \
  --output_dir ../german_target_pickles_with_bert_base_german_cased_chinese_whispers_with_sentence_id
```

## Output Directories Present in This Workspace

The workspace already contains example outputs from this pipeline:

- `german_target_pickles_with_bert_base_german_cased/`
- `german_target_pickles_with_bert_base_german_cased_with_sentence_id/`
- `german_target_pickles_with_bert_base_german_cased_kmeans_compounds_only/`
- `german_target_pickles_with_bert_base_german_cased_kmeans_compounds_only_with_sentence_id/`
- `german_target_pickles_with_bert_base_german_cased_chinese_whispers_with_sentence_id/`

## Summary

In short:

- the BERT stage converts each usage into contextual target vectors
- the K-means stage chooses the best `k` per compound using silhouette score
- the Chinese Whispers stage builds a similarity graph and discovers clusters through weighted label propagation

All three stages operate on the same per-target pickle structure, with new fields added rather than replacing the original data.

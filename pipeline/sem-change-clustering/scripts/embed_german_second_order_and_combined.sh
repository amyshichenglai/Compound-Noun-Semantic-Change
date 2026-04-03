#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"

BERT_DIR="${1:-${WORKSPACE_ROOT}/german_target_pickles_with_bert_base_german_cased}"
CACHED_DATA="${2:-}"
OUTPUT_DIR="${3:-${WORKSPACE_ROOT}/german_target_pickles_with_bert_and_2nd_order}"
TEST_FILE="${4:-${WORKSPACE_ROOT}/generated_ghost_targets.tsv}"
TOKENIZER_NAME="${TOKENIZER_NAME:-bert-base-german-cased}"

if [[ -z "${CACHED_DATA}" ]]; then
  echo "Usage: $0 [bert_dir] <cached_dta_corpus_pickle> [output_dir] [test_file]" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

python "${SCRIPT_DIR}/generate_minimal_ghost_targets_tsv.py" \
  --input_dir "${BERT_DIR}" \
  --output_file "${TEST_FILE}"

python "${SCRIPT_DIR}/create_and_cache_sparse_vecs.py" \
  --cached_per_target_dir "${BERT_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --tokenizer "${TOKENIZER_NAME}" \
  --test_file "${TEST_FILE}" \
  --test_file_type ghost \
  --lang de \
  --mode 2nd-order \
  --cached_data "${CACHED_DATA}"

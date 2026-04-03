#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"

INPUT_DIR="${1:-${WORKSPACE_ROOT}/german_target_pickles}"
OUTPUT_DIR="${2:-${WORKSPACE_ROOT}/german_target_pickles_with_bert_base_german_cased}"
MODEL_NAME="${MODEL_NAME:-bert-base-german-cased}"
TOKENIZER_NAME="${TOKENIZER_NAME:-${MODEL_NAME}}"
DEVICE="${DEVICE:-cpu}"
TEXT_FIELD="${TEXT_FIELD:-lemmas}"
LAYER_SLICE="${LAYER_SLICE:-first}"

mkdir -p "${OUTPUT_DIR}"

python "${SCRIPT_DIR}/add_bert_embeddings_to_target_pickles.py" \
  --input_dir "${INPUT_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --model "${MODEL_NAME}" \
  --tokenizer "${TOKENIZER_NAME}" \
  --device "${DEVICE}" \
  --text_field "${TEXT_FIELD}" \
  --layer_slice "${LAYER_SLICE}"

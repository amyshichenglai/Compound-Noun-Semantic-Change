from argparse import ArgumentParser
import os
import pickle
import sys
from types import ModuleType
import unicodedata


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

import data as repo_data
from constants import BERT_VECS_C
from data import MAX_SEQ_LENGTH, process_tokenized_sent


src_pkg = ModuleType("src")
src_pkg.data = repo_data
sys.modules["src"] = src_pkg
sys.modules["src.data"] = repo_data


class LegacySentence:
    def __init__(self, *args, **kwargs):
        pass


SLICE_NAME_TO_IDX = {
    "first": (1, 5),
    "mid": (5, -4),
    "last": (-4, 13),
}


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", help="Defaults to --model")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--layer_slice", choices=["first", "mid", "last"], default="first")
    parser.add_argument("--text_field", choices=["lemmas", "tokens", "raw"], default="lemmas")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer_name = args.tokenizer or args.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model = AutoModelForMaskedLM.from_pretrained(args.model)
    model.to(args.device)
    model.eval()

    filenames = sorted(
        filename for filename in os.listdir(args.input_dir)
        if filename.endswith(".pickle") or filename.endswith(".pkl")
    )
    if not filenames:
        print(f"No .pickle or .pkl files found in {args.input_dir}")
        return

    for filename in filenames:
        input_path = os.path.join(args.input_dir, filename)
        output_filename = filename[:-4] + ".pickle" if filename.endswith(".pkl") else filename
        output_path = os.path.join(args.output_dir, output_filename)
        examples = load_examples(input_path)
        embedded_examples = []
        skipped = 0

        for example in examples:
            example = repair_example_span_if_needed(example)
            tokens, target_span = get_embedding_tokens_and_span(example, args.text_field)
            if tokens is None:
                skipped += 1
                continue
            input_ids, target_span = vectorize_sentence_and_span(
                tokens=tokens,
                target_span=target_span,
                tokenizer=tokenizer,
            )
            if input_ids is None:
                skipped += 1
                continue

            bert_vec = get_vector_from_context(
                model=model,
                sentence={"input_ids": input_ids.to(args.device)},
                target_span=target_span,
                layer_slice_name=args.layer_slice,
            )

            updated = dict(example)
            updated["sent"] = repo_data.Sentence(
                tokens=list(example["sent"].tokens),
                lemmas=list(example["sent"].lemmas),
                tags=list(example["sent"].tags),
                year=example["sent"].year,
            )
            if hasattr(example["sent"], "sentence_id"):
                updated["sent"].sentence_id = example["sent"].sentence_id
            updated[BERT_VECS_C] = bert_vec
            embedded_examples.append(updated)

        with open(output_path, "wb") as out_f:
            pickle.dump(embedded_examples, out_f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"{filename}\tkept={len(embedded_examples)}\tskipped={skipped}")


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


def get_embedding_tokens_and_span(example, text_field):
    sent = example["sent"]
    target_span = tuple(example["lemma-span"])
    if text_field == "lemmas":
        return sent.lemmas, target_span
    if text_field == "tokens":
        return sent.tokens, target_span
    raw_tokens = example.get("raw_tokens")
    if raw_tokens is None:
        raw_sentence = example.get("raw_sentence")
        if raw_sentence is None:
            return None, None
        raw_tokens = raw_sentence.split()
    raw_target_span = remap_span_to_raw_tokens(raw_tokens=raw_tokens, aligned_tokens=sent.tokens, aligned_span=target_span)
    if raw_target_span is None:
        return None, None
    return raw_tokens, raw_target_span


def repair_example_span_if_needed(example):
    target_lemma = example.get("target_lemma")
    if not target_lemma:
        return example
    sent = example["sent"]
    repaired_span = find_target_span_in_lemmas(sent.lemmas, target_lemma, example.get("lemma-span"))
    if repaired_span is None:
        return example
    updated = dict(example)
    updated["lemma-span"] = repaired_span
    return updated


def find_target_span_in_lemmas(lemmas, target_lemma, original_span):
    target_norm = normalize_text(target_lemma)
    matches = []
    for idx, lemma in enumerate(lemmas):
        if normalize_text(lemma) == target_norm:
            matches.append((idx, idx + 1))
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]
    if original_span is None:
        return matches[0]
    return min(matches, key=lambda span: abs(span[0] - original_span[0]))


def normalize_text(text):
    return unicodedata.normalize("NFC", text).casefold()


def remap_span_to_raw_tokens(*, raw_tokens, aligned_tokens, aligned_span):
    aligned_to_raw = [i for i, token in enumerate(raw_tokens) if any(ch.isalnum() for ch in token)]
    if len(aligned_to_raw) != len(aligned_tokens):
        return None
    start, end = aligned_span
    if start < 0 or end > len(aligned_to_raw) or end <= start:
        return None
    return aligned_to_raw[start], aligned_to_raw[end - 1] + 1


def vectorize_sentence_and_span(*, tokens, target_span, tokenizer):
    model_tokens, token_offsets = process_tokenized_sent(tokens=tokens, tokenizer=tokenizer)

    if target_span[0] >= len(token_offsets) or target_span[1] > len(token_offsets):
        return None, None

    # Add CLS at the front and SEP at the end, matching the repo code path.
    model_tokens = model_tokens[: MAX_SEQ_LENGTH - 2]
    token_offsets = [(start + 1, end + 1) for start, end in token_offsets if end <= MAX_SEQ_LENGTH - 2]
    if target_span[1] > len(token_offsets):
        return None, None

    model_tokens = [tokenizer.cls_token] + model_tokens + [tokenizer.sep_token]
    input_ids = torch.tensor(
        [tokenizer.convert_tokens_to_ids(model_tokens)],
        dtype=torch.long,
    )

    subword_start = token_offsets[target_span[0]][0]
    subword_end = token_offsets[target_span[1] - 1][1]
    if subword_start >= MAX_SEQ_LENGTH or subword_end >= MAX_SEQ_LENGTH:
        return None, None
    return input_ids, (subword_start, subword_end)


def get_vector_from_context(model, sentence, target_span, layer_slice_name):
    if target_span[1] <= target_span[0]:
        raise ValueError(f"bad span designation: [{target_span[0]}, {target_span[1]})")
    if target_span[1] >= MAX_SEQ_LENGTH or target_span[0] < 0:
        raise ValueError("span is out of bounds")
    with torch.no_grad():
        encoded = model(**sentence, output_hidden_states=True)
        sliced = _slice_hidden_states(encoded.hidden_states, layer_slice_name)
        return encoded_layers_to_token_vector(sliced, target_span)


def _slice_hidden_states(hidden_states_tuple, slice_name):
    slice_start, slice_end = SLICE_NAME_TO_IDX[slice_name]
    return hidden_states_tuple[slice_start:slice_end]


def encoded_layers_to_token_vector(layers, token_span):
    token_embeddings = []
    for token_idx in range(token_span[0], token_span[1]):
        hidden_layers = []
        for layer_idx in range(len(layers)):
            vector = layers[layer_idx][0][token_idx]
            hidden_layers.append(vector)
        hidden_layers = torch.sum(torch.stack(hidden_layers), 0).reshape(1, -1).detach().cpu().numpy()
        token_embeddings.append(hidden_layers.squeeze())
    return token_embeddings


if __name__ == "__main__":
    main()

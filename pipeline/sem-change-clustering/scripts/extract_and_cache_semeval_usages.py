from argparse import ArgumentParser
from collections import defaultdict
from itertools import zip_longest
import os
import pickle
import sys
from typing import DefaultDict, Dict, Iterable, List, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from data import Sentence
from test_items_IO import TestWords
from util import SPACE_REPLACEMENT_IN_FILENAMES


SEMEVAL_POS_TO_REPO_POS = {
    "nn": "NN",
    "vb": "VB",
}

# These years are chosen so the repo's existing COHA coarse time slices
# map corpus1 to the early slice and corpus2 to the late slice.
DEFAULT_CORPUS_TO_YEAR = {
    "corpus1": 1830,
    "corpus2": 1980,
}


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--output_cache_dir", required=True)
    parser.add_argument("--targets_file", help="Defaults to <dataset_dir>/targets.txt")
    parser.add_argument(
        "--target_format",
        choices=["semeval", "compound-list"],
        default="semeval",
        help="`semeval` expects entries like attack_nn; `compound-list` expects one target expression per line",
    )
    parser.add_argument("--flush_every", type=int, default=3000,
                        help="Flush per-target cache slices after this many extracted usages")
    parser.add_argument("--corpus1_year", type=int, default=DEFAULT_CORPUS_TO_YEAR["corpus1"])
    parser.add_argument("--corpus2_year", type=int, default=DEFAULT_CORPUS_TO_YEAR["corpus2"])
    parser.add_argument(
        "--require_sentence_boundaries",
        action="store_true",
        help="Keep only token-side sentences that look like full sentence units",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    targets_file = args.targets_file or os.path.join(args.dataset_dir, "targets.txt")
    os.makedirs(args.output_cache_dir, exist_ok=True)

    if args.target_format == "semeval":
        target_forms = load_target_forms(targets_file)
    else:
        target_forms = load_compound_targets(targets_file)
    current_lookup_size = 0
    lookup: DefaultDict[str, List[Dict]] = defaultdict(list)

    corpus_specs = [
        (
            "corpus1",
            os.path.join(args.dataset_dir, "corpus1", "lemma", "ccoha1.txt"),
            os.path.join(args.dataset_dir, "corpus1", "token", "ccoha1.txt"),
            args.corpus1_year,
        ),
        (
            "corpus2",
            os.path.join(args.dataset_dir, "corpus2", "lemma", "ccoha2.txt"),
            os.path.join(args.dataset_dir, "corpus2", "token", "ccoha2.txt"),
            args.corpus2_year,
        ),
    ]

    sentence_count = 0
    extracted_count = 0
    for corpus_name, lemma_path, token_path, year in corpus_specs:
        corpus_sentence_count, corpus_extracted_count = extract_from_parallel_corpora(
            lemma_path=lemma_path,
            token_path=token_path,
            year=year,
            target_forms=target_forms,
            lookup=lookup,
            output_cache_dir=args.output_cache_dir,
            flush_every=args.flush_every,
            require_sentence_boundaries=args.require_sentence_boundaries,
        )
        sentence_count += corpus_sentence_count
        extracted_count += corpus_extracted_count
        current_lookup_size = sum(len(v) for v in lookup.values())
        print(
            f"{corpus_name}: scanned {corpus_sentence_count} sentences, "
            f"extracted {corpus_extracted_count} usages"
        )

    if current_lookup_size:
        flush_lookup(lookup, args.output_cache_dir)

    print(f"total: scanned {sentence_count} sentences, extracted {extracted_count} usages")


def load_target_forms(targets_file: str) -> Dict[str, Tuple[str, str]]:
    targets = TestWords.load_semeval2020_list(targets_file)
    target_forms = {}
    with open(targets_file, encoding="utf-8") as in_f:
        for line in in_f:
            target_with_pos = line.strip()
            if not target_with_pos:
                continue
            if "_" not in target_with_pos:
                raise ValueError(f"Unexpected target format: {target_with_pos}")
            lemma, semeval_pos = target_with_pos.rsplit("_", 1)
            if semeval_pos not in SEMEVAL_POS_TO_REPO_POS:
                raise ValueError(f"Unsupported SemEval POS tag: {target_with_pos}")
            target_forms[target_with_pos] = (lemma, SEMEVAL_POS_TO_REPO_POS[semeval_pos])
    expected_targets = {word.word for word in targets.to_list()}
    if expected_targets != {lemma for lemma, _ in target_forms.values()}:
        raise ValueError("Mismatch between parsed target forms and TestWords target list")
    return target_forms


def load_compound_targets(targets_file: str) -> Dict[str, Tuple[List[str], str]]:
    target_forms = {}
    with open(targets_file, encoding="utf-8") as in_f:
        for line in in_f:
            compound = " ".join(line.strip().split())
            if not compound:
                continue
            target_forms[compound] = (compound.split(), "NN")
    return target_forms


def extract_from_parallel_corpora(
    *,
    lemma_path: str,
    token_path: str,
    year: int,
    target_forms: Dict[str, Tuple[str, str]],
    lookup: DefaultDict[str, List[Dict]],
    output_cache_dir: str,
    flush_every: int,
    require_sentence_boundaries: bool,
) -> Tuple[int, int]:
    sentence_count = 0
    extracted_count = 0
    current_lookup_size = sum(len(v) for v in lookup.values())
    misaligned_token_count = 0
    dropped_trailing_chunks = 0

    with open(lemma_path, encoding="utf-8") as lemma_f, open(token_path, encoding="utf-8") as token_f:
        for lemma_line, token_line in zip_longest(lemma_f, token_f):
            if lemma_line is None or token_line is None:
                raise ValueError(f"Line count mismatch between {lemma_path} and {token_path}")
            sentence_count += 1
            lemma_tokens = lemma_line.rstrip().split()
            raw_token_tokens = token_line.rstrip().split()
            if require_sentence_boundaries:
                sentence_chunks, trailing_chunk = split_raw_tokens_into_sentences(raw_token_tokens)
                if trailing_chunk:
                    dropped_trailing_chunks += 1
            else:
                sentence_chunks = [raw_token_tokens]

            lemma_offset = 0
            for raw_chunk in sentence_chunks:
                token_tokens = [token for token in raw_chunk if any(ch.isalnum() for ch in token)]
                lemma_chunk = lemma_tokens[lemma_offset: lemma_offset + len(token_tokens)]
                lemma_offset += len(token_tokens)
                if len(lemma_chunk) != len(token_tokens):
                    misaligned_token_count += 1
                    continue

                normalized_lemmas, tags, matches = normalize_lemmas_and_collect_matches(lemma_chunk, target_forms)
                if not matches:
                    continue

                sent = Sentence(
                    tokens=token_tokens,
                    lemmas=normalized_lemmas,
                    tags=tags,
                    year=year,
                )
                raw_sentence = " ".join(raw_chunk)
                for target, span in matches:
                    lookup[target].append(
                        {
                            "sent": sent,
                            "lemma-span": span,
                            "raw_sentence": raw_sentence,
                            "raw_tokens": list(raw_chunk),
                        }
                    )
                    extracted_count += 1
                    current_lookup_size += 1

            if current_lookup_size >= flush_every:
                flush_lookup(lookup, output_cache_dir)
                current_lookup_size = 0
    if misaligned_token_count:
        print(
            f"{os.path.basename(lemma_path)}: skipped {misaligned_token_count} misaligned sentence chunks"
        )
    if dropped_trailing_chunks:
        print(
            f"{os.path.basename(lemma_path)}: dropped {dropped_trailing_chunks} trailing chunks without end marks"
        )
    return sentence_count, extracted_count


def normalize_lemmas_and_collect_matches(
    lemma_tokens: Iterable[str],
    target_forms: Dict[str, Tuple[object, str]],
) -> Tuple[List[str], List[str], List[Tuple[str, Tuple[int, int]]]]:
    normalized_lemmas = []
    tags = []
    matches = []

    for idx, lemma_token in enumerate(lemma_tokens):
        if lemma_token in target_forms:
            bare_lemma, target_pos = target_forms[lemma_token]
            normalized_lemmas.append(bare_lemma)
            tags.append(target_pos)
            matches.append((bare_lemma, (idx, idx + 1)))
        else:
            normalized_lemmas.append(lemma_token)
            tags.append("UNK")

    for target, (target_tokens, target_pos) in target_forms.items():
        if not isinstance(target_tokens, list):
            continue
        for start_idx, end_idx in find_ngram_matches(normalized_lemmas, target_tokens):
            for idx in range(start_idx, end_idx):
                tags[idx] = target_pos
            matches.append((target, (start_idx, end_idx)))
    return normalized_lemmas, tags, matches


def find_ngram_matches(tokens: List[str], target_tokens: List[str]) -> List[Tuple[int, int]]:
    if not target_tokens:
        return []
    matches = []
    target_len = len(target_tokens)
    for start_idx in range(len(tokens) - target_len + 1):
        if tokens[start_idx: start_idx + target_len] == target_tokens:
            matches.append((start_idx, start_idx + target_len))
    return matches


def split_raw_tokens_into_sentences(tokens: List[str]) -> Tuple[List[List[str]], List[str]]:
    sentences = []
    current = []
    for token in tokens:
        current.append(token)
        if token in {".", "!", "?"}:
            sentences.append(current)
            current = []
    return sentences, current


def flush_lookup(lookup: DefaultDict[str, List[Dict]], output_cache_dir: str) -> None:
    for key, examples in lookup.items():
        if not examples:
            continue
        filename = f"{key.replace(' ', SPACE_REPLACEMENT_IN_FILENAMES)}.pickle"
        with open(os.path.join(output_cache_dir, filename), "ab") as out_f:
            pickle.dump(examples, out_f, protocol=pickle.HIGHEST_PROTOCOL)
    lookup.clear()


if __name__ == "__main__":
    main()

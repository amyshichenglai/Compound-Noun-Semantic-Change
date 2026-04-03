from argparse import ArgumentParser
import os
import pickle
import string
import sys
from types import ModuleType


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import data as repo_data


src_pkg = ModuleType("src")
src_pkg.data = repo_data
sys.modules["src"] = src_pkg
sys.modules["src.data"] = repo_data


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--pickle_path", required=True, help="Path to one .pickle file")
    parser.add_argument("--limit", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    examples = load_examples(args.pickle_path)

    print(f"file: {args.pickle_path}")
    print(f"num_examples: {len(examples)}")
    print()

    for i, example in enumerate(examples[:args.limit], start=1):
        sent = example["sent"]
        span = tuple(example["lemma-span"])
        raw_sentence = example.get("raw_sentence", " ".join(sent.tokens))
        highlighted_sentence = highlight_span(sent.tokens, span)
        raw_tokens = example.get("raw_tokens", sent.tokens)
        begins_with_punct = begins_punct(raw_tokens)
        ends_with_punct = ends_punct(raw_tokens)
        begins_like_sentence_start = looks_like_sentence_start(raw_tokens)
        ends_with_period = ends_period(raw_tokens)

        print(f"example {i}")
        print(f"year: {sent.year}")
        print(f"raw stored sentence: {raw_sentence}")
        print(f"target span: {span} -> {' '.join(sent.tokens[span[0]:span[1]])}")
        print(f"highlighted: {highlighted_sentence}")
        print(f"begins with punctuation: {begins_with_punct}")
        print(f"ends with punctuation: {ends_with_punct}")
        print(f"begins like sentence start: {begins_like_sentence_start}")
        print(f"ends with '.': {ends_with_period}")
        print()


def load_examples(path):
    examples = []
    with open(path, "rb") as in_f:
        try:
            while True:
                examples.extend(pickle.load(in_f))
        except EOFError:
            pass
    return examples


def highlight_span(tokens, span):
    tokens = list(tokens)
    tokens.insert(span[0], "[{")
    tokens.insert(span[1] + 1, "}]")
    return " ".join(tokens)


def begins_punct(tokens):
    if not tokens:
        return False
    first = tokens[0].strip()
    return bool(first) and all(ch in string.punctuation for ch in first)


def ends_punct(tokens):
    if not tokens:
        return False
    last = tokens[-1].strip()
    return bool(last) and all(ch in string.punctuation for ch in last)


def looks_like_sentence_start(tokens):
    if not tokens:
        return False
    first = tokens[0].strip()
    if not first:
        return False
    if all(ch in string.punctuation for ch in first):
        return False
    first_char = first[0]
    return first_char.isupper() or first_char.isdigit() or first.startswith(("\"", "'", "``"))


def ends_period(tokens):
    if not tokens:
        return False
    return tokens[-1].strip() == "."


if __name__ == "__main__":
    main()

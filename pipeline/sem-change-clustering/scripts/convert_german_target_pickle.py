from argparse import ArgumentParser
from collections import defaultdict
import os
import pickle
import sys
from typing import DefaultDict, Dict, List


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from data import Sentence
from util import SPACE_REPLACEMENT_IN_FILENAMES


class LegacySentence:
    """Used to unpickle Sentence objects that were serialized from __main__."""

    def __init__(self, *args, **kwargs):
        pass


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--input_pickle", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--keep_target_lemma",
        action="store_true",
        help="Preserve the original target_lemma field instead of matching the English format exactly",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    usages = load_legacy_pickle(args.input_pickle)
    grouped: DefaultDict[str, List[Dict]] = defaultdict(list)

    for usage in usages:
        target = usage["target_lemma"]
        sent = usage["sent"]
        converted = {
            "sent": Sentence(
                tokens=list(sent.tokens),
                lemmas=list(sent.lemmas),
                tags=list(sent.tags),
                year=sent.year,
            ),
            "lemma-span": tuple(usage["lemma-span"]),
        }
        if args.keep_target_lemma:
            converted["target_lemma"] = target
        grouped[target].append(converted)

    for target, examples in grouped.items():
        filename = f"{target.replace(' ', SPACE_REPLACEMENT_IN_FILENAMES)}.pickle"
        output_path = os.path.join(args.output_dir, filename)
        with open(output_path, "wb") as out_f:
            pickle.dump(examples, out_f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"{target}\t{len(examples)}\t{output_path}")


def load_legacy_pickle(path: str):
    main_module = sys.modules["__main__"]
    previous_sentence = getattr(main_module, "Sentence", None)
    setattr(main_module, "Sentence", LegacySentence)
    try:
        with open(path, "rb") as in_f:
            return pickle.load(in_f)
    finally:
        if previous_sentence is None:
            delattr(main_module, "Sentence")
        else:
            setattr(main_module, "Sentence", previous_sentence)


if __name__ == "__main__":
    main()

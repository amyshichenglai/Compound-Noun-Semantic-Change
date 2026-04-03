from argparse import ArgumentParser
import os
import pickle
import sys
from types import ModuleType
from typing import Dict, List, Tuple


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from data import Sentence

src_pkg = ModuleType("src")
src_pkg.data = sys.modules["data"]
sys.modules["src"] = src_pkg
sys.modules["src.data"] = sys.modules["data"]


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--input_dirs", nargs="+", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    filenames = sorted({
        filename
        for input_dir in args.input_dirs
        for filename in os.listdir(input_dir)
        if filename.endswith(".pickle")
    })

    for filename in filenames:
        merged_examples: List[Dict] = []
        seen = set()
        for input_dir in args.input_dirs:
            filepath = os.path.join(input_dir, filename)
            if not os.path.exists(filepath):
                continue
            for example in load_pickle(filepath):
                key = usage_key(example)
                if key in seen:
                    continue
                seen.add(key)
                merged_examples.append(example)

        output_path = os.path.join(args.output_dir, filename)
        with open(output_path, "wb") as out_f:
            pickle.dump(merged_examples, out_f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"{filename}\t{len(merged_examples)}")


def load_pickle(path: str) -> List[Dict]:
    examples = []
    with open(path, "rb") as in_f:
        try:
            while True:
                examples.extend(pickle.load(in_f))
        except EOFError:
            pass
    return examples


def usage_key(example: Dict) -> Tuple:
    sent: Sentence = example["sent"]
    return (
        tuple(sent.tokens),
        tuple(sent.lemmas),
        tuple(sent.tags),
        sent.year,
        tuple(example["lemma-span"]),
    )


if __name__ == "__main__":
    main()

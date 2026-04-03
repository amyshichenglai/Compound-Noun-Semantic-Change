from argparse import ArgumentParser
import csv
import os
import pickle
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


class LegacySentence:
    def __init__(self, *args, **kwargs):
        self.__dict__.update(kwargs)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_csv", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    filenames = sorted(
        filename for filename in os.listdir(args.input_dir)
        if filename.endswith(".pickle") or filename.endswith(".pkl")
    )

    with open(args.output_csv, "w", encoding="utf-8", newline="") as out_f:
        writer = csv.DictWriter(
            out_f,
            fieldnames=["compound", "sentence_usage", "sentence_id", "cluster_id", "year"],
        )
        writer.writeheader()

        for filename in filenames:
            compound = os.path.splitext(filename)[0]
            for example in load_examples(os.path.join(args.input_dir, filename)):
                period = year_to_period(example["sent"].year)
                if period is None:
                    continue
                sentence_id = getattr(example["sent"], "sentence_id", None)
                if sentence_id is None:
                    raise ValueError(f"Missing sentence_id in {filename}")
                writer.writerow({
                    "compound": compound,
                    "sentence_usage": str(example["sent"]),
                    "sentence_id": sentence_id,
                    "cluster_id": example.get("cluster_label"),
                    "year": period,
                })

    print(f"Wrote CSV to {args.output_csv}")


def year_to_period(year):
    if 1700 <= year <= 1759:
        return "early"
    if 1870 <= year <= 1909:
        return "late"
    return None


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

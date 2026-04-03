from argparse import ArgumentParser
import os
import pickle
import sys
from collections import Counter, defaultdict
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
    parser.add_argument("--source_dir", required=True)
    parser.add_argument("--target_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    source_lookup = build_source_lookup(args.source_dir)
    filenames = sorted(
        filename for filename in os.listdir(args.target_dir)
        if filename.endswith(".pickle") or filename.endswith(".pkl")
    )

    for filename in filenames:
        target_examples = load_examples(os.path.join(args.target_dir, filename))
        queues = source_lookup[normalize_filename(filename)]
        restored = []
        restored_count = 0

        for example in target_examples:
            updated = dict(example)
            updated["sent"] = clone_sentence(example["sent"])
            if not hasattr(updated["sent"], "sentence_id"):
                key = make_usage_key(example)
                queue = queues[key]
                if not queue:
                    raise ValueError(f"Could not find sentence_id match for {filename}: {key}")
                updated["sent"].sentence_id = queue.pop(0)
                restored_count += 1
            restored.append(updated)

        with open(os.path.join(args.output_dir, filename), "wb") as out_f:
            pickle.dump(restored, out_f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"{filename}\trestored={restored_count}\ttotal={len(restored)}")


def build_source_lookup(source_dir):
    filenames = sorted(
        filename for filename in os.listdir(source_dir)
        if filename.endswith(".pickle") or filename.endswith(".pkl")
    )
    lookup = {}
    for filename in filenames:
        examples = load_examples(os.path.join(source_dir, filename))
        queues = defaultdict(list)
        for example in examples:
            sentence_id = getattr(example["sent"], "sentence_id", None)
            if sentence_id is None:
                raise ValueError(f"Source example is missing sentence_id in {filename}")
            queues[make_usage_key(example)].append(sentence_id)
        lookup[normalize_filename(filename)] = queues
    return lookup


def normalize_filename(filename):
    return filename[:-4] + ".pickle" if filename.endswith(".pkl") else filename


def make_usage_key(example):
    sent = example["sent"]
    return (
        tuple(sent.tokens),
        tuple(sent.lemmas),
        tuple(sent.tags),
        sent.year,
        example.get("target_lemma"),
    )


def clone_sentence(sent):
    cloned = repo_data.Sentence(
        tokens=list(sent.tokens),
        lemmas=list(sent.lemmas),
        tags=list(sent.tags),
        year=sent.year,
    )
    for key, value in getattr(sent, "__dict__", {}).items():
        if key in {"tokens", "lemmas", "tags", "year"}:
            continue
        setattr(cloned, key, value)
    return cloned


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

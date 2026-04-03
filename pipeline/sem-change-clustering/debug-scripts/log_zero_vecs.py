"""quantifying the issue that came up w/ sparse 2nd order representations, that many were all zeros
   due to actual examples lacking any of the top N first order contexts (which sorta makes sense given
   how those were selected) - either way - it would be good to quantify the prevalence of this
"""
from argparse import ArgumentParser
import os
import pickle
import numpy as np

from constants import BERT_VECS_C, RANDOM_VECS_C, SECOND_ORDER_VECS_C
from util import SPACE_REPLACEMENT_IN_FILENAMES

def main():
    parser = ArgumentParser()
    parser.add_argument("--cached_vecs_file")
    parser.add_argument("--embedding_key", choices=[BERT_VECS_C, RANDOM_VECS_C, SECOND_ORDER_VECS_C])

    args = parser.parse_args()
    examples = load_examples(args.cached_vecs_file, args.embedding_key)
    zero_count, non_zero_count = 0, 0
    for embedding in examples:
        if not np.any(embedding):
            zero_count += 1
        else:
            non_zero_count += 1
    print(f"examples with zeros: {zero_count} ({(zero_count / (zero_count + non_zero_count) * 100):.2f}%), with something: {non_zero_count}")


def load_examples(pickle_file_path, embedding_key: str):
    basename = os.path.basename(pickle_file_path)
    filename_as_key = basename.replace(SPACE_REPLACEMENT_IN_FILENAMES, " ").split(".pickle")[0]
    examples = []
    with open(pickle_file_path, 'rb') as in_cache:
        # potentially there are several lists
        try:
            while True:
                # will throw EOF when there are no more lists to load
                cached_lookup_slice = pickle.load(in_cache)
                examples += cached_lookup_slice
        except EOFError:
            pass
    return [example[embedding_key] for example in examples]


if __name__ == "__main__":
    main()

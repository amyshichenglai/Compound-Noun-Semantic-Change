from argparse import ArgumentParser
import os
import pickle

from util import SPACE_REPLACEMENT_IN_FILENAMES
from test_items_IO import TestCompounds


def main():
    parser = ArgumentParser()
    parser.add_argument("--cached_examples_dir")
    parser.add_argument("--output_id_lookup_file")
    parser.add_argument("--test_file")
    parser.add_argument("--test_file_type")

    args = parser.parse_args()

    test_items = []
    if args.test_file_type == "cordeiro":
        test_items = TestCompounds.load_cordeiro(args.test_file)
    elif args.test_file_type == "ghost":
        test_items = TestCompounds.load_ghost(args.test_file)
    compounds = set()
    for test_item in test_items.compounds:
        compounds.add(test_item.compound)

    output_lines = []
    for cache_file in os.listdir(args.cached_examples_dir):


        cached_lookup_slice = []

        with open(f"{args.cached_examples_dir}/{cache_file}", 'rb') as in_cache:
            key_name = cache_file.replace(
                SPACE_REPLACEMENT_IN_FILENAMES, " ").split(".pickle")[0]
            if key_name not in compounds:
                continue
            try:
                while True:
                    # will throw EOF when there are no more lists to load
                    cached_lookup_slice += pickle.load(in_cache)
            except EOFError:
                pass

        if cached_lookup_slice is not None:


            for i, ex in enumerate(cached_lookup_slice):
                span_start, span_end = ex['lemma-span']
                sent = ex['sent']
                output_lines.append(f"{key_name}::{i}\t"
                                    f"{span_start}\t"
                                    f"{span_end}\t"
                                    f"{str(sent)}")

    with open(args.output_id_lookup_file, 'w', encoding='utf-8') as out_f:
        for line in output_lines:
            out_f.write(line)
            out_f.write("\n")



if __name__ == "__main__":
    main()

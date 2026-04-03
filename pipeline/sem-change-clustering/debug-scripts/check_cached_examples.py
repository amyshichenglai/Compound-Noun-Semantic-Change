from argparse import ArgumentParser
import pickle
from constants import (
    BERT_VECS_C, SECOND_ORDER_VECS_C, LEMMA_SPAN_C
)
from util import year_to_coarse_slice, SPACE_REPLACEMENT_IN_FILENAMES, COARSE_TIME_SLICES
from collections import defaultdict
import os

def main():
    parser = ArgumentParser()
    parser.add_argument("--cached_example_file", help=".pickle file keyed to a particular example")
    parser.add_argument("--cache_type", choices=["bert-and-sents", "second-order"])
    parser.add_argument("--corpus_type", choices=["DTA", "COHA"])
    parser.add_argument("--count_only", action='store_true')
    args = parser.parse_args()

    cached_lookup_slice = []
    coarse_slice_counter = defaultdict(int)
    for time_slice in COARSE_TIME_SLICES[args.corpus_type]:
        coarse_slice_counter[time_slice] = 0

    with open(args.cached_example_file, 'rb') as in_cache:
        try:
            while True:
                # will throw EOF when there are no more lists to load
                cached_lookup_slice += pickle.load(in_cache)
        except EOFError:
            pass
    if cached_lookup_slice is not None:
        if not args.count_only:
            print("First sent from this cache:")

        for ex in cached_lookup_slice:
            year = ex['sent'].year
            span_start, span_end = ex['lemma-span']
            tokens = [tok for tok in ex['sent'].tokens]
            tokens.insert(span_start, '[{')
            tokens.insert(span_end + 1, "}]")
            coarse_slice_counter[year_to_coarse_slice(year, args.corpus_type)] += 1
            if not args.count_only:
                print(f"({ex['sent'].year}) {' '.join(t for t in tokens)}")
        if not args.count_only:
            print(f"\n\nNumber of cached examples: {len(cached_lookup_slice)}")
            print(f"time slice counts: {coarse_slice_counter}")
        else:
            coarse_slice_keys = sorted(k for k in coarse_slice_counter if k is not None)
            basename = os.path.basename(args.cached_example_file).replace(
                SPACE_REPLACEMENT_IN_FILENAMES, " ").replace(".pickle", "")

            output = [basename]
            for coarse_slice in coarse_slice_keys:
                output.append(str(coarse_slice))
                output.append(str(coarse_slice_counter[coarse_slice]))
            print("\t".join(o for o in output))
    if args.cache_type == "bert-and-sents":
        pass
    if args.cache_type == "second-order":
        print(f"sent 0 and its second-order representation: {cached_lookup_slice[0]['sent']}")
        print(f"{cached_lookup_slice[0][SECOND_ORDER_VECS_C]}")


if __name__ == "__main__":
    main()

""" script to check for prevalence of ClusterItem hash collisions to see
    whether it would be preferable to come up with a direct symbolic ID system
    to refer to the items instead. """
from argparse import ArgumentParser
from collections import defaultdict

from clustering import load_vecs_cache
from cluster_eval import ClusterItem
from constants import LEMMA_SPAN_C
from test_items_IO import TestCompounds

def main():
    parser = ArgumentParser()
    parser.add_argument("--test_file")
    parser.add_argument("--test_file_type", choices=["ghost", "cordeiro"])
    parser.add_argument("--cached_examples_dir")
    args = parser.parse_args()

    if args.test_file_type == "cordeiro":
        test_items_class = TestCompounds.load_cordeiro(args.test_file)
        test_items = test_items_class.to_list()
    elif args.test_file_type == "ghost":
        test_items_class = TestCompounds.load_ghost(args.test_file)
        test_items = test_items_class.to_list()
    else:
        raise ValueError("unavailable test file type")
    test_items = [item for t in test_items for item in t.to_list_all()]

    filtered_lookup = load_vecs_cache(
        cached_examples_dirs=[args.cached_examples_dir],
        targets=test_items,
    )
    hash_counter = defaultdict(int)
    example_count = 0
    for k, v in filtered_lookup.items():
        for elt in v:
            sent = elt['sent']
            span = elt[LEMMA_SPAN_C]
            cluster_item = ClusterItem(
                associated_sent=sent,
                associated_target=k,
                lemma_span=span,
            )
            example_count += 1
            hash_counter[hash(cluster_item)] += 1

    print(f"Out of {example_count} examples...")

    collision_count = 0
    for hash_val, count in hash_counter.items():
        if count > 1:
            collision_count += count - 1
    print(f"... {collision_count} collided\n"
          f"which is {collision_count / example_count:.2f}"
          f" of the total examples.")




if __name__ == "__main__":
    main()

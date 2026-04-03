"""idea here is to try to illustrate the issue (if it turns out to be an issue and not just 
a hunch) of bert embeddings tending to be rather close to one another anyway, making
clustering a matter of splitting hairs between many things that are 0.000001 away from one another """

from argparse import ArgumentParser
import random
import pickle
import os
from util import SPACE_REPLACEMENT_IN_FILENAMES
from test_items_IO import TestCompounds
from scipy.spatial.distance import cosine, euclidean
from constants import BERT_VECS_C, RANDOM_VECS_C, SECOND_ORDER_VECS_C
def main():
    parser = ArgumentParser()
    parser.add_argument("--cached_examples_dir")
    parser.add_argument("--target_compound", help="compound compared to all the others")
    parser.add_argument("--test_file")
    parser.add_argument("--test_file_type")
    parser.add_argument("--embedding_key", choices=[BERT_VECS_C, RANDOM_VECS_C, SECOND_ORDER_VECS_C])
    args = parser.parse_args()

    test_items = []
    if args.test_file_type == "cordeiro":
        test_items = TestCompounds.load_cordeiro(args.test_file).to_list()
    elif args.test_file_type == "ghost":
        test_items = TestCompounds.load_ghost(args.test_file).to_list()
    else:
        raise ValueError

    target_example = load_single_example(
        os.path.join(args.cached_examples_dir, args.target_compound.replace(' ', SPACE_REPLACEMENT_IN_FILENAMES) + ".pickle"),
        args.embedding_key
    )


    cos_dists = {}
    euc_dists = {}

    for test_item in test_items:
        if test_item.compound == args.target_compound:
            continue
        path = os.path.join(args.cached_examples_dir,
                                test_item.head.replace(" ", SPACE_REPLACEMENT_IN_FILENAMES) + ".pickle")
        if not os.path.isfile(path):
            continue
        example = \
            load_single_example(path, args.embedding_key)
        cos_dists[test_item.compound] = \
            cosine(target_example, example)
        euc_dists[test_item.compound] = \
            euclidean(target_example, example)

    cos_dists_ls = sorted([(s, d) for s, d in cos_dists.items()], key=lambda x: x[1])
    euc_dists_ls = sorted([(s, d) for s, d in euc_dists.items()], key=lambda x: x[1])
    print(f"comparing against {args.target_compound}")
    print(f"cosine dists:\n")
    for string, dist in cos_dists_ls:
        print(f"{string} -> {dist:.04f}")

    print(f"\n\neuclidean dists:\n")
    for string, dist in euc_dists_ls:
        print(f"{string} -> {dist:.04f}")



def load_single_example(pickle_file_path, embedding_key: str):
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
    # return a random example
    # print(examples[0]['bert-vec'][0])
    return random.choice(examples)[embedding_key]


if __name__ == "__main__":
    random.seed(2666)
    main()
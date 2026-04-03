from argparse import ArgumentParser
import os
from typing import List

from data import Sentence
from test_items_IO import TestCompounds, TestWords
from clustering import load_vecs_cache # because these have sentence data cached too
from util import SPACE_REPLACEMENT_IN_FILENAMES
from itertools import product

def main():
    parser = ArgumentParser()
    parser.add_argument("--targets_file")
    parser.add_argument("--cached_examples_dir")
    parser.add_argument("--output_dir")
    parser.add_argument("--test_file_type", choices=['cordeiro', 'ghost', 'semeval'])
    args = parser.parse_args()



    test_items = []
    if args.test_file_type == "cordeiro":
        test_items = TestCompounds.load_cordeiro(args.targets_file)
    elif args.test_file_type == "ghost":
        test_items = TestCompounds.load_ghost(args.targets_file)
    elif args.test_file_type == "semeval":
        test_items = TestWords.load_semeval2020_list(args.targets_file)

    test_items_as_strings = [it.compound for it in test_items.to_list()]

    # TODO: handle picking constituents from target files

    examples_lookup = load_vecs_cache(
        cached_examples_dirs=[args.cached_examples_dir],
        targets=test_items_as_strings
    )

    # this will be accessed like examples_lookup[t]['sent']
    id_generator = identifier_generator()

    test_sent = Sentence(['Gold', 'mines', 'zzzgold', 'mine', 'gold mine'],
                         ['gold', 'mine', 'zzzgold', 'mine', 'gold mine'],
                         ['ADJ', 'N', 'ADJ', 'N', 'N'], 2023)
    # print(test_sent.to_durel(target_lemma='gold mine', id_generator=id_generator))

    for target in test_items_as_strings:
        sents: List[Sentence] = [ex[1] for ex in examples_lookup[target]]
        durel_lines = []
        for s in sents:
            durel_lines += s.to_durel(target_lemma=target,
                                          id_generator=id_generator)
        target_filename = target.replace(" ", SPACE_REPLACEMENT_IN_FILENAMES) + ".csv"
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, target_filename), 'w', encoding='utf-8') as out_f:
            out_f.write(f"lemma\tpos\tdate\tgrouping\tidentifier\tdescription\tcontext\t"
                        f"indexes_target_token\tindexes_target_sentence\n")
            for line in durel_lines:
                out_f.write(f"{line}\n")



def identifier_generator() -> str:
    alpha_seq = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    seq_len = 1
    while True:
            sequence = [''.join(p) for p in product(alpha_seq, repeat=seq_len)]
            for elt in sequence:
                yield elt
            seq_len += 1

if __name__ == "__main__":
    main()


from argparse import ArgumentParser
import os
import sys
from collections import defaultdict
import random
from itertools import combinations

USER_LABELS = {
    'en': {
        # "LABEL_SET": "1 (unrelated),2,3,4 (identical)",
        # "NON_LABEL": "- (can't decide)"
        "LABEL_SET": "1,2,3,4",
        "NON_LABEL": "-",
    },
    'de': {
        # "LABEL_SET": "1 (unabhängig),2,3,4 (identisch)",
        # "NON_LABEL": "- (nicht entscheidbar)"
        "LABEL_SET": "1,2,3,4",
        "NON_LABEL": "-",
    }
}

HEADER = "instanceID\tdataIDs\tlabel_set\tnon_label\n"

ERA_1 = "1"
ERA_2 = "2"

def main():
    parser = ArgumentParser()
    parser.add_argument("--input_directory", help="BE SURE that the input directory contains ONLY "
                                                  "instance files that we want to sample from")
    parser.add_argument("--output_file_prefix", help="prefix for output file, _N and .tsv will be appended")
    parser.add_argument("--pairs_per_target", type=int, default=5, help="""number of
     comparisons made between the two time periods AND among each individual time period, 
     making the maximum total number of example pairs per target this argument * 3""")
    parser.add_argument("--lang", choices=['en', 'de'])
    parser.add_argument("--seed", type=int, default=2666)
    parser.add_argument("--num_output_files", type=int, default=4)
    # TODO: list of files of prev generated instances, to exclude the exact same pairs
    #       from being listed again.
    args = parser.parse_args()
    random.seed(args.seed)
    files_list = os.listdir(args.input_directory)
    target_to_era_to_id = defaultdict(lambda: defaultdict(list))
    for filename in sorted(files_list):
        # get ids / eras to be combined into instances
        with open(f"{args.input_directory}/{filename}", encoding='utf-8') as in_f:
            for i, line in enumerate(in_f):
                if i == 0:
                    # header
                    continue
                lemma, _, _, grouping, identifier, _, _, _, _ = line.strip().split("\t")
                target_to_era_to_id[lemma][grouping].append(identifier)
    output_pairs = []
    for lemma in target_to_era_to_id.keys():
        # if we can't get a whole complement of examples for this one,
        # we get as many as we can
        era_to_ids = target_to_era_to_id[lemma]
        e1_ids, e2_ids = era_to_ids[ERA_1], era_to_ids[ERA_2]

        e1_choices = random.sample(e1_ids, min(args.pairs_per_target, len(e1_ids)))
        e2_choices = random.sample(e2_ids, min(args.pairs_per_target, len(e2_ids)))
        cross_time_choices = [t for t in zip(e1_choices, e2_choices)][:args.pairs_per_target]
        ids_in_use = set([t for c in cross_time_choices for t in c])
        output_pairs += cross_time_choices
        e1_ids = [i for i in e1_ids if i not in ids_in_use]
        e2_ids = [i for i in e2_ids if i not in ids_in_use]

        output_pairs += _within_era_pairs(e1_ids, args.pairs_per_target)
        output_pairs += _within_era_pairs(e2_ids, args.pairs_per_target)

    random.shuffle(output_pairs)
    fraction = len(output_pairs) // args.num_output_files
    output_ranges = [(i * fraction, ((i+1) * fraction)) for i in range(args.num_output_files)]
    for i in range(args.num_output_files):
        with open(f"{args.output_file_prefix}_{i}.tsv", 'w', encoding='utf-8') as out_f:
            out_f.write(HEADER)
            for j in range(output_ranges[i][0], output_ranges[i][1]):
                out_f.write(
                    f"{str(hash(output_pairs[j]))}\t"
                    f"{','.join(output_pairs[j])}\t"
                    f"{USER_LABELS[args.lang]['LABEL_SET']}\t"
                    f"{USER_LABELS[args.lang]['NON_LABEL']}\n"
                )


def _within_era_pairs(era_ids, pairs_per_target):
    output_pairs = []
    e_shuff = random.sample(era_ids, k=len(era_ids))
    e_pairs = [p for p in zip(e_shuff[::2], e_shuff[1::2])]
    output_pairs += e_pairs[:pairs_per_target]
    if not len(era_ids) % 2 == 0 and len(e_pairs) and len(e_pairs) < pairs_per_target:
        odd_one_out = e_shuff[-1]
        double_use_candidates = e_shuff[:-1]
        last_pair = (random.sample(double_use_candidates, k=1)[0], odd_one_out)
        output_pairs.append(last_pair)
    return output_pairs

if __name__ == "__main__":
    main()


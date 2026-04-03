"""This is just tsv-ifying the experiment output json files"""
import argparse
from collections import defaultdict
import json
import os
from typing import Literal

from eval_util import format_eval_params
from test_items_IO import TestCompounds, TestRelatedCompounds, CompositionalityRating

COLUMNS = ["div_t1_t2",
           "div_t1", "div_t2",
           "avg_pairwise_div_t1", "avg_pairwise_div_t2",
           "bifurcated_averaged_representations",
           "average_pairwise_distance"
]

RESULTS_TOP_LEVEL_META_KEYS = {'BEST_K', 'args', 'params', 'clustering_output', '__GLOBAL__', 'phitag_eval'}

RELATED_COMPOUNDS_NON_LEXICAL_KEYS = {
    'pred', 'phitag-similarity-proportions',
    "avg_pairwise_div_t1", "avg_pairwise_div_t2",
    "div_t1_t2",
}

TEST_FILES = {
    'cordeiro': "cordeiro-ratings-onlyNN.txt",
    'ghost': "Ghost-NN_comp-means.txt",
}
RELATED_TEST_FILES = {
    'cordeiro': {
        'head': "cordeiro-related-via-head.txt",
        'mod': "cordeiro-related-via-mod.txt"
    },
    'ghost': {
        'head': "ghost-related-via-head.txt",
        'mod': "ghost-related-via-mod.txt"
    }
}
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_directory")
    parser.add_argument("output_dir")
    args = parser.parse_args()

    targets = defaultdict(list)

    filenames = [n for n in os.listdir(args.exp_directory)
                 if n.endswith(".json")]
    for filename in filenames:
        with open(f"{args.exp_directory}/{filename}", encoding='utf-8') as in_f:
            try:
                experiment_json = json.load(in_f)
            except json.decoder.JSONDecodeError:
                print(f"couldn't parse file {filename}!")
                continue
            model_name = experiment_json['args']['model_name']
            # if "+heads" in model_name and "+mods" in model_name:
            #     continue # can´t handle this case at the moment

        test_item_lookup, related_test_item_lookup, test_item_to_rating = \
            test_items_for_this_file(experiment_json)

        # we don't actually need columns with the settings, because
        # we produce one output file PER experiment json -- and can just
        # copy the file name and add .tsv to the end...
        output = per_target_difference_measures(
            experiment_json,
            test_item_lookup,
            related_test_item_lookup,
        )
        with open(os.path.join(args.output_dir, f"{filename[:-5]}.tsv"), 'w', encoding='utf-8') as out_f:
            out_f.write("\t".join(["target"] + COLUMNS))
            out_f.write("\n")
            for target, vals_dict in output.items():
                remaining_cols = [vals_dict[col] if col in vals_dict else None
                                  for col in COLUMNS]
                cols = [target] + remaining_cols
                out_f.write("\t".join(str(col) for col in cols))
                out_f.write("\n")



def per_target_difference_measures(experiment_json, test_item_lookup, related_test_item_lookup):
    """returns dict of target -> measures list
    (in order defined by global COLUMNS)"""
    # target -> column type -> value
    output = defaultdict(dict)
    for k in experiment_json:
        if k in RESULTS_TOP_LEVEL_META_KEYS:
            continue
        for k2 in experiment_json[k]:
            if k2 in COLUMNS:
                output[k][k2] = experiment_json[k][k2]
            elif k2 not in RELATED_COMPOUNDS_NON_LEXICAL_KEYS:
                # continue more here
                for k3 in experiment_json[k][k2]:
                    if k3 == "pred":
                        continue
                    output[k2][k3] = experiment_json[k][k2][k3]


    return output

def test_items_for_this_file(experiment_json):
    test_file_type = experiment_json['params']['test_file_type']
    if "+heads" in experiment_json['args']['model_name']:
        constituent_type = "head"
    elif "+mods" in experiment_json['args']['model_name']:
        constituent_type = "mod"
    else:
        constituent_type = None
    if experiment_json['args']['related_compounds_file'] is not None:
        related_mode = True
    else:
        related_mode = False

    test_items = []
    if test_file_type == "cordeiro":
        test_items = TestCompounds.load_cordeiro(TEST_FILES[test_file_type]).to_list()
    elif test_file_type == "ghost":
        test_items = TestCompounds.load_ghost(TEST_FILES[test_file_type]).to_list()

    test_item_lookup = {t.primary_component: t for t in test_items}

    test_item_to_rating = {t.primary_component: test_item_to_comp_rating(t, constituent_type) for t in test_items
                           if constituent_type is not None}

    related_test_item_lookup = None
    if related_mode:
        test_items = TestRelatedCompounds.load_pre_filtered(
            comp_rated_compounds=test_items,
            pre_filtered_related_compounds_list_file=RELATED_TEST_FILES[test_file_type][constituent_type],
            constituent_type=constituent_type,
        )
        related_test_item_lookup = test_items
    return test_item_lookup, related_test_item_lookup, test_item_to_rating

def test_item_to_comp_rating(
    test_item: CompositionalityRating,
    constituent: Literal["mod", "head"],
):
    if constituent == "mod":
        return test_item.mean_mod_rating
    elif constituent == "head":
        return test_item.mean_head_rating
    else:
        return None


if __name__ == "__main__":
    main()
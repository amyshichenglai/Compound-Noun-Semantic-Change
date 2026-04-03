from argparse import ArgumentParser
from collections import defaultdict
import json
import os
from statistics import mean, stdev
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas
from scipy.stats import spearmanr

from annotation.phitag_interface import (
    PhitagRatings, load_unified_phitag_json, mean_std_dev_ratings,
)
from eval_util import format_eval_params, PARAMETERS_COLUMNS, aggregate_phitag_eval
from test_items_IO import TestCompounds, TestRelatedCompounds, CompositionalityRating

RESULTS_TOP_LEVEL_META_KEYS = {'args', 'params', 'clustering_output', '__GLOBAL__', 'phitag_eval'}
RESULTS_SECOND_LEVEL_META_KEYS = {
    "div_t1", "div_t2",  "div_t1_t2", "avg_pairwise_div_t1",
    "avg_pairwise_div_t2", "pred", "bifurcated_averaged_representations",
    "bifurcated_averaged_representations_prediction",
    "average_pairwise_distance",
    "phitag-similarity-proportions",
}

LOCAL_COLUMNS_BASE = [
    "div_t1_t2", "div_t1", "div_t2", 'abs_diff_t2_t1', # div_t1 and _t2 means (with constituent)
    "avg_pairwise_div_t1", "avg_pairwise_div_t2", 'abs_diff_avg_pairwise_div_t2_t1',
    "bifurcated_averaged_representations", "average_pairwise_distance",
]

LOCAL_COLUMNS = LOCAL_COLUMNS_BASE + [f"abs(delta_later)+{col}" for col in LOCAL_COLUMNS_BASE] + \
    [f"mu_compare+{col}" for col in LOCAL_COLUMNS_BASE] + \
    [f'annotation_JSD+{col}' for col in LOCAL_COLUMNS_BASE]
LOCAL_COLUMNS_RHO_AND_P_VAL = [
    f"{col}_{addendum}" for col in LOCAL_COLUMNS for addendum in ["rho", "p-val"]
]

# TODO: figuring out how to present things for the related-compounds situation
#       (maybe that needs its own script since it's kind of particular)
COLUMNS = PARAMETERS_COLUMNS + LOCAL_COLUMNS_RHO_AND_P_VAL

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
    parser = ArgumentParser()
    parser.add_argument("--results_dir")
    parser.add_argument("--annotations_json")
    # parser.add_argument("--test_file")
    # parser.add_argument("--test_file_type", choices=['ghost', 'cordeiro'])
    # parser.add_argument("--related_compounds_test_file")
    # parser.add_argument("--lang", choices=['en', 'de'])
    # parser.add_argument("--constituent_type", choices=['mod', 'head'])
    # parser.add_argument("--related_compounds_file")
    parser.add_argument("--margins", action="store_true")
    parser.add_argument("--correlations", action='store_true')
    parser.add_argument("--annotations_time_groups_en_tsv")
    parser.add_argument("--annotations_time_groups_de_tsv")
    parser.add_argument("--annotation_jsd_tsv_file", help='tsv file with target, jsd '
                                                          'graded semantic change values')
    parser.add_argument("--output")
    args = parser.parse_args()

    if args.margins:
        chart_about_distribution_of_annotations_margins(args)

    if args.correlations:
        correlations_report(args)

def correlations_report(args):
    annotations = load_unified_phitag_json(args.annotations_json)
    by_target = annotations.annotations_by_target_mean_stddev()

    annotations_by_time_groups_lookup = {}
    if args.annotations_time_groups_en_tsv:
        annotations_by_time_groups_lookup.update(
            parse_annotations_time_groupings_tsv(
                args.annotations_time_groups_en_tsv
            )
        )
    if args.annotations_time_groups_de_tsv:
        annotations_by_time_groups_lookup.update(
            parse_annotations_time_groupings_tsv(
                args.annotations_time_groups_de_tsv
            )
        )

    target_to_annotation_jsd = {}
    with open(args.annotation_jsd_tsv_file, encoding='utf-8') as in_f:
        for i, line in enumerate(in_f):
            if i == 0:
                continue
            target, val = line.strip().split('\t')
            if val == 'nan':
                continue
            target_to_annotation_jsd[target] = float(val)



    filenames = [n for n in os.listdir(args.results_dir)
                 if n.endswith(".json")]
    output = {}
    for filename in filenames:
        with open(f"{args.results_dir}/{filename}", encoding='utf-8') as in_f:
            try:
                experiment_json = json.load(in_f)
            except json.decoder.JSONDecodeError:
                print(f"couldn´t parse file {filename}!")
                continue
            model_name = experiment_json['args']['model_name']
            # if "+heads" in model_name and "+mods" in model_name:
            #     continue # can´t handle this case at the moment

        test_item_lookup, related_test_item_lookup, test_item_to_rating = \
            test_items_for_this_file(experiment_json)

        correlations_out = correlations_compounds_constituents(
            experiment_json,
            test_item_lookup,
            related_test_item_lookup,
            test_item_to_rating,
            annotations_by_time_groups_lookup,
            target_to_annotation_jsd,
        )

        output[filename] = format_eval_params(experiment_json)
        output[filename].update({
            key: correlations_out[key] if key in correlations_out else ""
            for key in LOCAL_COLUMNS
        })

    with open(args.output, 'w', encoding='utf-8') as out_f:
        out_f.write("\t".join(COLUMNS))
        out_f.write("\n")
        for k in output:
            vals = output[k]
            per_run = [str(vals[c]) for c in PARAMETERS_COLUMNS]
            for c in LOCAL_COLUMNS:
                possible_tuple = vals[c]
                if not possible_tuple:
                    per_run += ["", ""]
                else:
                    rho, p = possible_tuple
                    per_run.append(str(rho))
                    per_run.append(str(p))

            # per_run = [str(vals[c]) for c in COLUMNS]
            out_f.write("\t".join(per_run))
            out_f.write("\n")

    print("done")

    #
    # test_items = []
    # if args.test_file_type == "cordeiro":
    #     test_items = TestCompounds.load_cordeiro(args.test_file)
    # elif args.test_file_type == "ghost":
    #     test_items = TestCompounds.load_ghost(args.test_file)

    # related_compounds_clustering = False
    # if args.related_test_file_type and \
    #         args.related_test_file_type in ["cordeiro_related", "ghost_related"]:
    #     related_compounds_clustering = True
    #     if PLUS_MODS in model_name and not PLUS_HEADS in model_name:
    #         related_constituent = "mod"
    #     elif PLUS_HEADS in model_name and not PLUS_MODS in model_name:
    #         related_constituent = "head"
    #     else:
    #         raise ValueError("currently not supporting running both kinds "
    #                          "of related compounds simultaneously")
    #     per_time_target_dist, all_time_unigram_dist, per_time_target_counts = get_distribution_of_targets(
    #         time_to_target_to_freq, targets=[it for it in test_items.to_list()],
    #         time_slices=COARSE_TIME_SLICES[params['dataset_type']],
    #     )
    #     test_items = TestRelatedCompounds.load(
    #         comp_rated_compounds=test_items.to_list(),
    #         cached_vecs_counts=per_time_target_counts,
    #         related_compounds_filename=args.related_compounds_file,
    #         constituent_type=related_constituent,
    #         lang=args.lang,
    #     )

def correlations_compounds_constituents(
    experiment_json,
    test_item_lookup,
    related_test_item_lookup,
    test_item_to_rating,
    annotated_targets_by_time_group_lookup,
    target_to_annotated_jsd,
):
    if related_test_item_lookup is None and len(test_item_to_rating):
        things_to_correlate_with = ['div_t1', 'div_t2', 'abs_diff_t2_t1', 'div_t1_t2', 'bifurcated_averaged_representations',
                          'average_pairwise_distance']
    elif related_test_item_lookup is not None:
        things_to_correlate_with = ['div_t1_t2', 'avg_pairwise_div_t1', 'avg_pairwise_div_t2',
                                    'abs_diff_avg_pairwise_div_t2_t1',
                                    'bifurcated_averaged_representations',
                                    'average_pairwise_distance']
    else: # not comparing against constituents at all
        things_to_correlate_with = ['div_t1_t2', 'bifurcated_averaged_representations',
                                    'average_pairwise_distance']
    if "phitag_eval" in experiment_json:
        # pass
        # things_to_correlate_with.append("phitag_f1")
        phitag_f1, (agg_acc, agg_prec, agg_recall), per_target = aggregate_phitag_eval(
            results_json=experiment_json,
        )
    correlation_inputs = defaultdict(dict)
    for k, inner_dict in experiment_json.items():
        if k in RESULTS_TOP_LEVEL_META_KEYS:
            continue
        # if we correlate with comp ratings, that means only compounds
        if k not in test_item_lookup:
            continue
        # things not making the cut
        if len(inner_dict) == 1 and "average_pairwise_distance" in inner_dict:
            continue
        # if 'div_t1' in things_to_correlate_with and "div_t1" not in inner_dict:
        #     continue
        if 'div_t1_t2' in things_to_correlate_with and "div_t1_t2" not in inner_dict:
            continue
        for k2, val in inner_dict.items():
            if k2 not in RESULTS_SECOND_LEVEL_META_KEYS:
                pass # this means it's a related compound key
            else:
                if k2 in things_to_correlate_with:
                    correlation_inputs[k][k2] = val

    # add extra derived keys:
    for k in correlation_inputs:
        if 'div_t1' in correlation_inputs[k]:
            correlation_inputs[k]['abs_diff_t2_t1'] = \
                abs(correlation_inputs[k]['div_t1'] - correlation_inputs[k]['div_t2'])
        elif 'avg_pairwise_div_t1' in correlation_inputs[k]:
            correlation_inputs[k]['abs_diff_avg_pairwise_div_t2_t1'] = \
                abs(correlation_inputs[k]['avg_pairwise_div_t1'] - correlation_inputs[k]['avg_pairwise_div_t2'])


    sorted_keys = sorted(k for k in correlation_inputs)

    correlations_out = {}
    sorted_keys = sorted(k for k in correlation_inputs)
    if len(test_item_to_rating):
        compositionality_ratings = np.array([
            test_item_to_rating[k]
            for k in sorted_keys if k in test_item_lookup and k in test_item_to_rating])


        for measurement in things_to_correlate_with:
            values = []
            for k in sorted_keys:
                if k in test_item_lookup:
                    values.append(correlation_inputs[k][measurement])
            corr = spearmanr(
                np.array(values),
                compositionality_ratings,
            )
            correlations_out[measurement] = (corr.correlation, corr.pvalue)
    # else:
    #     # dummy values for compositionality ratings
    #     # (or maybe it should have separate runs for each rating :'( i'm not sure
    #     for measurement in things_to_correlate_with:
    #         correlations_out[mea]

    delta_later_ratings = np.array([
       abs(annotated_targets_by_time_group_lookup[k]['delta_later'])
        for k in sorted_keys if k in annotated_targets_by_time_group_lookup
    ])
    for measurement in things_to_correlate_with:
        values = []
        for k in sorted_keys:
            if k in annotated_targets_by_time_group_lookup:
                values.append(correlation_inputs[k][measurement])
        corr = spearmanr(
            np.array(values),
            delta_later_ratings
        )
        correlations_out[f"abs(delta_later)+{measurement}"] = (corr.correlation, corr.pvalue)

    mu_compare_ratings = np.array([
        annotated_targets_by_time_group_lookup[k]['mean_compare']
        for k in sorted_keys if k in annotated_targets_by_time_group_lookup
    ])
    for measurement in things_to_correlate_with:
        values = []
        for k in sorted_keys:
            if k in annotated_targets_by_time_group_lookup:
                values.append(correlation_inputs[k][measurement])
        corr = spearmanr(
            np.array(values),
            mu_compare_ratings
        )
        correlations_out[f"mu_compare+{measurement}"] = (corr.correlation, corr.pvalue)

    annotation_jsd_ratings = np.array([
        target_to_annotated_jsd[k]
        for k in sorted_keys if k in annotated_targets_by_time_group_lookup
    ])
    for measurement in things_to_correlate_with:
        values = []
        for k in sorted_keys:
            if k in annotated_targets_by_time_group_lookup:
                values.append(correlation_inputs[k][measurement])
        corr = spearmanr(
            np.array(values),
            annotation_jsd_ratings
        )
        correlations_out[f"annotation_JSD+{measurement}"] = (corr.correlation, corr.pvalue)


    return correlations_out


def chart_about_distribution_of_annotations_margins(args):
    excluded_middle = (2.3, 2.7)

    annotations = load_unified_phitag_json(args.annotations_json)
    by_target = annotations.annotations_by_target_mean_stddev()

    # ratings = {
    #     f"{target}:{i}": (mean, stddev)
    #         for target in by_target
    #             for i, (mean, stddev) in enumerate(by_target[target])
    #
    # }
    ratings = {
        f"{target}": [mean for mean, stddev in by_target[target]]
        for target in by_target
    }

    means = [
        v for v in ratings.values()
    ]

    targets = [k for k in ratings]

    num_in_excluded_middle = 0
    total = 0
    for target, v_list in zip(targets, means):
        for mean in v_list:
            if mean > excluded_middle[0] and mean <= excluded_middle[1]:
                num_in_excluded_middle += 1
            total += 1
    print(f"{num_in_excluded_middle} excluded out of {total}; {num_in_excluded_middle / total}")


    colors = ['peachpuff', 'orange', 'tomato']

    fig, ax = plt.subplots()
    ax.set_ylabel('ratings')

    bplot = ax.boxplot(means,
                       # patch_artist=True,  # fill with color
                       labels=targets)  # will be used to label x-ticks

    # fill with colors
    # for patch, color in zip(bplot['boxes'], colors):
    #     patch.set_facecolor(color)

    plt.show()

    print(f"done!")

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

def parse_annotations_time_groupings_tsv(filename):
    with open(filename, encoding='utf-8') as in_f:
        lines = in_f.readlines()[1:]
    lookup = {}
    for line in lines:
        target, mean_earlier, mean_later, mean_compare, delta_later = \
            line.rstrip().split('\t')
        if 'nan' in [mean_earlier, mean_later, mean_compare, delta_later]:
            continue
        lookup[target] = {
            'mean_earlier': float(mean_earlier),
            'mean_later': float(mean_later),
            'mean_compare': float(mean_compare),
            'delta_later': float(delta_later),
        }
    return lookup

if __name__ == "__main__":
    main()
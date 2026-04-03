import ast
from collections import defaultdict, Counter
import functools
from itertools import combinations, product
import json
import logging
import os
import pickle
import random
import re
import sys
from typing import List, Tuple, Optional, Union, Dict, Any, Literal, DefaultDict
import yaml

import numpy as np
from sklearn.cluster import KMeans, AffinityPropagation
from sklearn.metrics import pairwise_distances
from sklearn_extra.cluster import KMedoids
import sklearn
import scipy
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
import torch
from transformers import AutoModelForMaskedLM
import transformers

from annotation.phitag_interface import (
    PhitagRatings, load_unified_phitag_json, mean_std_dev_ratings,
    discrete_to_binary_similarity_rating,
    time_based_similarity_counts,
)
from data import Sentence, CorpusData, MAX_SEQ_LENGTH, tf_idf_features, CorpusFrequency
from cluster_eval import (
    eval_clusters_all_eras_single_target, ClusterItem,
    ClusterFeatureEmbedding, ClusterFeatureCompoundStats,
    ClusterFeatureFrequency,
    SupervisedTestItem, eval_clusters_multi_target,
    eval_clusters_multi_target_with_respect_to_constituents,
    eval_clusters_multi_target_with_shared_constituent_groups,
    map_clusters, mapped_cluster_divergence,
    single_target_gain_or_loss, predict_change_in_sense_inventory, aggregate_score,
    distance_vectorized,
    cluster_diagnostics,
    UnsupervisedClusterEval,
)
from constants import (
    BERT_VECS_C, RANDOM_VECS_C, SECOND_ORDER_VECS_C,
    FREQUENCY_C, MOD_PROD_C, HEAD_PROD_C, LEMMA_SPAN_C
)
from test_items_IO import (
    TestCompounds, CompositionalityRating,
    CompositionalityRatingCordeiro, CompositionalityRatingGhost,
    TestWords, TestWord, load_freq_stats, load_prod_stats,
    TestRelatedCompounds, RelatedCompoundSet,
)
from util import (
    find_in_sent,
    SPACE_REPLACEMENT_IN_FILENAMES, COARSE_TIME_SLICES, year_to_coarse_slice,
    SamplerWithoutReplacement,
    TimeStratifiedSampling
)

DEBUG = False

logger = logging.getLogger(__name__)



# model names: algo_<single_target|all_targets><+heads|+mods>
K_MEANS_SINGLE_TARGET = 'k_means_single_target'
K_MEANS_ALL_TARGETS = 'k_means_all_targets'
AFFINITY_PROPAGATION_SINGLE_TARGET = 'aff_prop_single_target'
AFFINITY_PROPAGATION_ALL_TARGETS = 'aff_prop_all_targets'
K_MEDOIDS_SINGLE_TARGET = 'k_medoids_single_target'
SINGLE_TARGET = "_single_target"
PLUS_HEADS = "+heads"
PLUS_MODS = "+mods"
PLUS_SEPARATE_ERAS = "+separate_eras"

DEFAULT_MIN_CLUSTER_THRESHOLD = 1
DEFAULT_DIVERGENCE_THRESHOLD = 0.2
DEFAULT_BIFURCATED_SIM_THRESHOLD = 0.90 # higher than this == no change


FEATURE_NAMES = {BERT_VECS_C, RANDOM_VECS_C, FREQUENCY_C, HEAD_PROD_C, MOD_PROD_C}

def main():

    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--cached_data", help="pickle file w/ data")
    parser.add_argument("--cached_model_path_or_name", help="specification to load pretrained model,"
                        " to be used to create contextual representations")
    parser.add_argument("--model_name", default=None,
                        help="Names of model to run. Command line invocation "
                             "supersedes the equivalent setting in the .yml config file.")
    parser.add_argument("--config_file", type=str, help=".yml config file for this run")
    parser.add_argument("--test_file", help="list of compounds to test")
    parser.add_argument("--related_compounds_file", help=".tsv file with list of compounds sharing one constituent")
    parser.add_argument("--gold_test_file", help="test items with sem change annotations")
    parser.add_argument("--compound_ratings_file", type=str, help=".json file with consolidated"
                            " compound annotations")
    parser.add_argument("--device", default='cpu', type=str, help="device arg, so either 'cpu' or 'gpu:d' with a "
                        "number provided after the colon.")
    parser.add_argument("--cached_examples_dirs", nargs="*", type=str, help=""
                "directory with pickle file(s) with cached mapping of example -> vectorized Sentences, "
                "will skip over all keys in the cache, but supplement any additional "
                "keys requested in the test_file")
    parser.add_argument("--use_bert_vecs", action='store_true', help=""
                        "whether to use bert representations as a cluster feature. Previously "
                        "this was the case by default.")
    parser.add_argument("--use_random_idx_vecs", action='store_true')
    parser.add_argument("--use_second_order_vecs", action='store_true')
    parser.add_argument("--compound_stats_lookup", type=str, help=".json file with compound stats data")
    parser.add_argument("--compound_frequency_lookup_dir")
    parser.add_argument("--compound_productivity_lookup_dir")
    parser.add_argument("--example_frequency_tsv", type=str, help=".tsv file with entries corresponding to the cached examples "
                                                                  "- when provided, this enables the sampling procedure.")
    parser.add_argument("--max_samples_override", type=int, help="max number of samples, overriding the config file")
    parser.add_argument("--min_target_allocation_override", type=int, help="min number of examples per target per time slice, overriding the config file")

    parser.add_argument("--output_json", type=str, default=sys.stdout)
    parser.add_argument("--print_sents", action='store_true', help="print clustered sentences in output")
    parser.add_argument("--cluster_associations", action='store_true', help="include tf-idf cluster associations"
                                                                            "in output json")
    parser.add_argument("--diagnostics", action='store_true', help="print post-clustering diagnostics to standard output")
    parser.add_argument("--cluster_eval_metrics", nargs='*', choices=['silhouette', 'davies_bouldin', 'calinski_harabasz'], default=[])
    parser.add_argument("--seed", type=int, default=2666)
    parser.add_argument("--seed_post_sampling", type=int, default=2666, help=""
                        "used to pick a different random seed than the overall choice"
                        " but only after any sampling is done -- since which"
                        " examples are included is already a big source of variance!")

    # params to test out:
    parser.add_argument("--k_override", type=int, help="usually this parameter for number of clusters "
                                                       "would be keyed to a specific model, but here "
                                                       "we can GLOBALLY specify one (if we want to say "
                                                       "sweep over various possible values of k)")
    parser.add_argument("--min_cluster_threshold", type=int, default=DEFAULT_MIN_CLUSTER_THRESHOLD, help="number of instances in a cluster "
                            "which is too few to include that cluster in divergence measures")
    parser.add_argument("--divergence_threshold", type=float, default=DEFAULT_DIVERGENCE_THRESHOLD, help="cutoff used for prediction "
                        "purposes, range 0.0-1.0 -- below this value: the distributions are the same, above it, "
                        "they are different'")
    parser.add_argument("--compound_constituent_divergence_threshold", type=float, default=DEFAULT_DIVERGENCE_THRESHOLD,
                        help="cutoff for the compound x constituent mode of evaluation - how different "
                             "the distributions should be between time periods to qualify as *different*."
                                                                                                                              )
    parser.add_argument("--bifurcated_sim_threshold", type=float, default=DEFAULT_BIFURCATED_SIM_THRESHOLD, help="range from 0.0 to 1.0, "
                        "how similar two aggregated vectors need to be to be considered "
                        "'the same meaning': above this value means no change, below means change")
    parser.add_argument("--merge_small_clusters", action='store_true', help="enable the merging of small clusters "
                        "prior to evaluation")
    args = parser.parse_args()

    _set_seed(args.seed)

    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                        datefmt="%d/%m/%Y %H:%M:%S",
                        level=logging.INFO,
                        handlers=[logging.StreamHandler(sys.stdout)]
                        # handlers=[logging.FileHandler(
                        #     filename=f"{args.output_dir}/sem-change-clustering.log",
                        #     encoding='utf-8',
                        #     mode='a+')]
    )
    # TODO: this is a bit confusing: currently *only* keys in this lookup will
    #       have their params from the .yml config file retrieved, so any variations like
    algo_lookup = {
        'k_means_single_target': k_means_clustering,
        'k_means_single_target+heads': k_means_clustering,
        'k_medoids_single_target': k_medoids,
        'k_medoids_all_targets': k_medoids,
        'k_medoids_all_targets+heads': k_medoids,
        'aff_prop_single_target': per_target_aff_prop,
        'aff_prop_all_targets': per_target_aff_prop,
        'aff_prop_all_targets+heads': per_target_aff_prop,
        'k_means_all_targets': k_means_clustering,
        'k_means_all_targets+mods': k_means_clustering,
        'k_means_all_targets+heads': k_means_clustering,
        'k_means_all_targets+mods+heads': k_means_clustering,
    }
    feature_set_used = set()

    output = {}

    with open(args.config_file, encoding='utf-8') as in_f:
        params = yaml.safe_load(in_f.read())

    if args.device == 'cpu':
        print("Running on cpu - please be sure this is desired")
    else:
        # TODO: check if cuda is actually available
        print("running on cuda")

    if args.model_name:
        model_name = args.model_name
    else:
        # check if params file has pipeline arg
        if 'model_name' in params:
            model_name = params['model_name']
        else:
            # run k_means by default
            logging.info("running k-means single target by default")
            model_name = K_MEANS_SINGLE_TARGET

    test_items = []
    if params['test_file_type'] == "cordeiro":
        test_items = TestCompounds.load_cordeiro(args.test_file)
    elif params['test_file_type'] == "ghost":
        test_items = TestCompounds.load_ghost(args.test_file)
    elif params['test_file_type'] == "semeval":
        test_items = TestWords.load_semeval2020_list(args.test_file)

    default_targets = [t_it.compound for t_it in test_items.compounds]

    if args.example_frequency_tsv:
        # corpus_freq = CorpusFrequency.from_file(args.corpus_frequency_files_prefix, params['dataset_type'])
        time_to_target_to_freq = defaultdict(lambda: defaultdict(int))
        time_slices = COARSE_TIME_SLICES[params['dataset_type']]
        num_slices = len(time_slices)
        with open(args.example_frequency_tsv, encoding='utf-8') as in_f:
            for line in in_f:
                fields = line.rstrip().split('\t')
                target, fields = fields[0], fields[1:]
                for i in range(0, num_slices * 2, 2):
                    time_to_target_to_freq[ast.literal_eval(fields[i])][target] = int(fields[i + 1])

        # TODO: this getting loaded this way is very confusing
        #       (i.e. using the regular test_items prior to
        #       test_items being replaced with the related compounds test items)
        per_time_target_dist, all_time_unigram_dist, per_time_target_counts = get_distribution_of_targets(
            time_to_target_to_freq, targets=[it for it in test_items.to_list()],
            time_slices=COARSE_TIME_SLICES[params['dataset_type']],
        )
    else:
        # corpus_freq = None
        per_time_target_dist, all_time_dist, per_time_target_counts = None, None, None
        time_to_target_to_freq = None

    related_compounds_clustering = False
    if 'related_test_file_type' in params and \
            params['related_test_file_type'] in ["cordeiro_related", "ghost_related"]:
        related_compounds_clustering = True
        if PLUS_MODS in model_name and not PLUS_HEADS in model_name:
            related_constituent = "mod"
        elif PLUS_HEADS in model_name and not PLUS_MODS in model_name:
            related_constituent = "head"
        else:
            raise ValueError("currently not supporting running both kinds "
                             "of related compounds simultaneously")
        test_items = TestRelatedCompounds.load(
            comp_rated_compounds=test_items.to_list(),
            cached_vecs_counts=per_time_target_counts,
            related_compounds_filename=args.related_compounds_file,
            constituent_type=related_constituent,
            lang=params["lang_code"],
        )
        per_time_target_dist, all_time_unigram_dist, per_time_target_counts = get_distribution_of_targets(
            time_to_target_to_freq, targets=[it for it in test_items.to_list()],
            time_slices=COARSE_TIME_SLICES[params['dataset_type']],
        )


    logging.info(f"all arguments: {args}")
    logging.info(f"params file: {params}")

    supervised_items = None
    if args.gold_test_file:
        with open(args.gold_test_file, encoding='utf-8') as in_f:
            supervised_items = {}
            for line in in_f:
                gold_item = SupervisedTestItem.from_input(line)
                supervised_items[gold_item.text] = gold_item

    phitag_ratings: Optional["PhitagRatings"] = None
    if args.compound_ratings_file:
        phitag_ratings = load_unified_phitag_json(args.compound_ratings_file)
        phitag_ratings.annotations_by_time_groupings(display=True)
        # TODO: REMOVE THIS -- it can happen later on i guess...
        # type_level_annotations = type_level_ratings(phitag_ratings)

    time_early = (params['early_epoch_start'], params['early_epoch_end'])
    time_late = (params['late_epoch_start'], params['late_epoch_end'])

    if 'bifurcated_sim_threshold' in args:
        bifurcated_similarity_threshold = args.bifurcated_sim_threshold
    elif 'bifurcated_sim_threshold' in params:
        bifurcated_similarity_threshold = params['bifurcated_sim_threshold']
    else:
        bifurcated_similarity_threshold = DEFAULT_BIFURCATED_SIM_THRESHOLD


    if args.cached_model_path_or_name:
        model = AutoModelForMaskedLM.from_pretrained(args.cached_model_path_or_name)
        model.to(args.device)
    else:
        model = None

    if args.cached_data:
        data = CorpusData(cached_data=args.cached_data)
    else:
        data = None



    if not data and not args.cached_examples_dirs:
        raise ValueError("need to either provide input data or cached examples to continue")

    if args.k_override:
        for key in params:
            if isinstance(params[key], dict) and 'k' in params[key]:
                params[key]['k'] = args.k_override


    any_include_head_examples, any_include_mod_examples = False, False
    if PLUS_HEADS in model_name:
        any_include_head_examples = True
    if PLUS_MODS in model_name:
        any_include_mod_examples = True

    if args.use_bert_vecs:
        feature_set_used.add(BERT_VECS_C)
    if args.use_random_idx_vecs:
        feature_set_used.add(RANDOM_VECS_C)
    if args.compound_frequency_lookup_dir:
        feature_set_used.add(FREQUENCY_C)
    if args.compound_productivity_lookup_dir:
        if any_include_head_examples:
            feature_set_used.add(HEAD_PROD_C)
        if any_include_mod_examples:
            feature_set_used.add(MOD_PROD_C)
    if args.use_second_order_vecs:
        feature_set_used.add(SECOND_ORDER_VECS_C)
    logging.info(f"Using features: {feature_set_used}")

    # all_examples = find_examples_all(model, data=data, targets=[ex], include_constituents=False)
    all_examples = find_examples_all(
        params=params,
        targets=[it for it in test_items.to_list()],
        include_head_examples=any_include_head_examples,
        include_mod_examples=any_include_mod_examples,
        cached_examples_dirs=args.cached_examples_dirs,
        use_bert_vecs=args.use_bert_vecs,
        use_random_idx_vecs=args.use_random_idx_vecs,
        use_second_order_vecs=args.use_second_order_vecs,
        default_targets=default_targets, # as in: just the compounds, so that they are always used!
        frequency_stats_dir=args.compound_frequency_lookup_dir,
        productivity_stats_dir=args.compound_productivity_lookup_dir,
    )

    cluster_items = create_cluster_items(
        all_examples=all_examples,
        feature_set_used=feature_set_used,
    )

    example_to_annotation = None
    annotated_pairs, annotated_pair_to_annotation = None, None
    time_based_sim_counts, time_based_sim_proportions = None, None
    if phitag_ratings is not None:
        # create mapping between ClusterItems and annotations
        example_to_annotation = {}
        for cluster_item in cluster_items:
            # not having this check was causing some constituent
            # examples (in german) to get matched up with annotations
            if cluster_item.target not in phitag_ratings.all_targets:
                continue
            annotated_examples, annotations = \
                phitag_ratings.get_examples_by_usage(
                    # sentence=cluster_item.sent,
                    # target=cluster_item.target,
                    # target_lemma_span=cluster_item.lemma_span,
                    cluster_item=cluster_item,
                )
            if annotated_examples:
                example_to_annotation[cluster_item] = (annotated_examples, annotations)

        annotated_pairs, annotated_pair_to_annotations = find_pairs_of_annotated_examples(example_to_annotation)
        time_based_sim_counts, time_based_sim_proportions = time_based_similarity_counts(annotated_pair_to_annotations, params['dataset_type'])

    if args.max_samples_override:
        params[model_name]["max_num_examples"] = args.max_samples_override
    if args.min_target_allocation_override:
        params[model_name]["minimum_uniform_alloc"] = args.min_target_allocation_override
    if time_to_target_to_freq is not None:
        cluster_items = get_items_sample(
            model_params=params[model_name],
            separate_eras=False,
            test_items=[it for it in test_items.to_list()],
            cluster_items=cluster_items,
            include_heads=any_include_head_examples,
            include_mods=any_include_mod_examples,
            related_compounds_clustering=related_compounds_clustering,
            time_to_target_to_freq=time_to_target_to_freq,
            dataset_type=params['dataset_type'],
            test_items_distribution=per_time_target_dist,
            all_time_dist=all_time_unigram_dist,
            annotated_pairs=annotated_pairs,
        )
    # TODO: it would be good to either log the sample or to be
    #       *much* more certain that it is deterministic
    #       given the seed
    if args.seed != args.seed_post_sampling:
        _set_seed(args.seed_post_sampling)
        logging.info(f"Seed set to {args.seed_post_sampling} after sampling")

    if not args.use_bert_vecs and not \
            args.use_random_idx_vecs and not \
            args.use_second_order_vecs and not \
            args.compound_frequency_lookup_dir and not \
            args.compound_productivity_lookup_dir:
        raise ValueError("need at least one cluster feature to continue!")


    total_count = defaultdict(int)
    year_counter = Counter()
    for item in cluster_items:
        coarse_time = year_to_coarse_slice(item.sent.year, params['dataset_type'])
        year_counter[coarse_time] += 1
        total_count[item.target] += 1


    per_clustering_per_target_divergences = defaultdict(lambda: defaultdict(float))

    logging.info(f"finished loading examples")


    # LOG WHAT WE ARE RUNNING WITH
    output['args'] = {k: v for k, v in vars(args).items()}
    output['params'] = {k: v for k, v in params.items()}
    clustered_examples = []
    include_heads = PLUS_HEADS in model_name
    include_mods = PLUS_MODS in model_name
    single_target = SINGLE_TARGET in model_name
    separate_eras = PLUS_SEPARATE_ERAS in model_name
    ###### CHECK THIS:
    # is this actually wanted??? if we don't collapse down this name, we can list
    # separate configs for each of the various options
    # model_name = model_name.replace(
    #     PLUS_HEADS, "").replace(PLUS_MODS, "").replace(
    #     PLUS_SEPARATE_ERAS, "")
    logging.info(f"model name to use in lookup: {model_name}")
    if single_target:
        clustered_examples = single_target_clustering(
            args=args,
            params=params,
            separate_eras=separate_eras,
            time_early=time_early,
            time_late=time_late,
            test_items=test_items.to_list(),
            cluster_items=cluster_items,
            feature_set_used=feature_set_used,
            include_heads=include_heads,
            include_mods=include_mods,
            supervised_items=supervised_items,
            algo=model_name,
            cluster_fn_lookup=algo_lookup,
            per_clustering_per_target_divergences=per_clustering_per_target_divergences,
            output=output,
        )
    else:
        clustered_examples = multi_target_clustering(
            args=args,
            params=params,
            separate_eras=separate_eras,
            related_compounds_clustering=related_compounds_clustering,
            time_early=time_early,
            time_late=time_late,
            test_items=test_items.to_list(), # the full objects, not just string keys
            cluster_items=cluster_items,
            feature_set_used=feature_set_used,
            include_heads=include_heads,
            include_mods=include_mods,
            supervised_items=supervised_items,
            phitag_annotated_pairs=annotated_pairs,
            phitag_annotation_lookup=example_to_annotation,
            phitag_ratings=phitag_ratings,
            phitag_time_based_similarity_proportions=time_based_sim_proportions,
            phitag_time_based_similarity_counts=time_based_sim_counts,
            algo=model_name,
            cluster_fn_lookup=algo_lookup,
            per_clustering_per_target_divergences=per_clustering_per_target_divergences,
            output=output
        )

    # test clustering:

    #
    #

    # TODO: this needs to be updated to work with the +constituents clustering
    #if params['test_file_type'] in ['cordeiro', 'ghost']:
    #    test_item_lookup = {t.compound: t for t in test_items.to_list()}
    #    for exp_name, target_to_divergence_d in per_clustering_per_target_divergences.items():
    #        target_compositionality, divergences = [], []
    #        # if divergence is None, we skip the whole entry
    #        # but otherwise keep everything aligned
    #        for target, div in target_to_divergence_d.items():
    #            if div is None:
    #                continue
    #            target_compositionality.append(target.compound_rating)
    #            divergences.append(div)
    #        corr = spearmanr(
    #            np.array(divergences),
    #            np.array(target_compositionality)
    #        )
    #        output[exp_name] = {"compositionality_correlation":  {
    #                "rho": corr.correlation,
    #                "pvalue": corr.pvalue
    #            }
    #        }

    # baseline evals
    bifurcated_avg_scores = bifurcated_averaged_representations(
        clustered_examples, params["dataset_type"]
    )
    for query, score in bifurcated_avg_scores.items():
        if str(query) not in output:
            output[str(query)] = {}
        output[str(query)]['bifurcated_averaged_representations'] = score
        # output[str(query)]['bifurcated_averaged_representations_prediction'] = \
        #     bool(score <= bifurcated_similarity_threshold)

    # if supervised_items:
    #     all_bi_preds = [output[str(query)]['bifurcated_averaged_representations_prediction'] for query in bifurcated_avg_scores]
    #     all_golds = [supervised_items[query].rating for query in bifurcated_avg_scores]
    #     output['bifurcated_averaged_representations'] = aggregate_score(gold_labels=all_golds, predicted_labels=all_bi_preds)

    # avg pairwise distances:
    avg_pairwise_scores = average_pairwise_distances(
        clustered_examples, params['dataset_type']
    )
    for query, score in avg_pairwise_scores.items():
        if query not in output:
            output[query] = {}
        output[query]['average_pairwise_distance'] = score

    # convert bools to strings
    output_keys = output.keys()
    for k in output_keys:
        if k == "BEST_K":
            continue
        if 'pred' in output[k]:
            output[k]['pred'] = str(output[k]['pred'])
    with open(args.output_json, 'w', encoding='utf-8') as out_f:
        json.dump(output, out_f, ensure_ascii=False, indent=2, default=str)
    # sents = [example[1] for target in all_targets
    #          for example in all_examples[target]]
    # cluster_era_analysis(sents, labels_big_cluster_km, time_early, time_late)


    logging.info("done!")
    # average out each example per time period


def create_cluster_items(
    all_examples,
    feature_set_used,
) -> List[ClusterItem]:
    all_examples_list = []
    for query in all_examples:
        for ex_dict in all_examples[query]:
            contextual_embedding, random_indexing, second_order_vec, freq_feature = None, None, None, None
            if BERT_VECS_C in feature_set_used:
                contextual_embedding = ClusterFeatureEmbedding(ex_dict[BERT_VECS_C])
            if RANDOM_VECS_C in feature_set_used:
                random_indexing = ClusterFeatureEmbedding(ex_dict[RANDOM_VECS_C])
            associated_sent = ex_dict['sent']
            if FREQUENCY_C in feature_set_used:
                freq_feature = ClusterFeatureFrequency(ex_dict[FREQUENCY_C])
            if MOD_PROD_C in feature_set_used:
                pass
            if HEAD_PROD_C in feature_set_used:
                pass
            if SECOND_ORDER_VECS_C in feature_set_used:
                if SECOND_ORDER_VECS_C not in ex_dict:
                    print(f"ERROR: no second order vector for example sent: {ex_dict['sent']}")
                second_order_vec = ClusterFeatureEmbedding(ex_dict[SECOND_ORDER_VECS_C])
            all_examples_list.append(
                ClusterItem(
                    cluster_item_id=ex_dict["id"],
                    contextual_embedding=contextual_embedding, random_indexing_embedding=random_indexing,
                    second_order_random_indexing_embedding=second_order_vec,
                    associated_target=query, associated_sent=associated_sent,
                    lemma_span=ex_dict[LEMMA_SPAN_C],
                    frequency=freq_feature,
                )
            )
    return all_examples_list

def find_pairs_of_annotated_examples(
    example_to_annotation: Dict[ClusterItem, Tuple[List["PhitagExample"], List["PhitagAnnotation"]]]
) -> Tuple[List[Tuple[ClusterItem, ClusterItem]],
            Dict[Tuple[ClusterItem, ClusterItem], List["PhitagAnnotation"]]]:
    str_to_cluster_item = {
        str(c_it.sent): c_it for c_it in example_to_annotation.keys()
    }
    # TODO: situations where duplicate sentences are present???
    pairs = []
    cluster_item_pair_to_annotations = {}
    for cluster_item, (examples, annotations) in example_to_annotation.items():
        for example, annotation in zip(examples, annotations):
            if example.context0.target_sent in str_to_cluster_item and \
                  example.context1.target_sent in str_to_cluster_item:
                # impose a sort order on the pairs:
                if example.context0.target_sent < example.context1.target_sent:
                    new_pair = (str_to_cluster_item[example.context0.target_sent],
                                str_to_cluster_item[example.context1.target_sent])
                else:
                    new_pair = (str_to_cluster_item[example.context1.target_sent],
                                str_to_cluster_item[example.context0.target_sent])
                if new_pair not in pairs:
                    pairs.append(new_pair)
                    cluster_item_pair_to_annotations[(new_pair)] = annotation
    return pairs, cluster_item_pair_to_annotations
def single_target_clustering(
        *,
        args,
        params,
        separate_eras,
        time_early,
        time_late,
        test_items,
        cluster_items,
        feature_set_used,
        include_heads,
        include_mods,
        supervised_items: Dict[str, SupervisedTestItem],
        algo,
        cluster_fn_lookup,
        per_clustering_per_target_divergences, # modified
        output, # modified
):
    output_algo_name = algo + "+separate_eras" if separate_eras else algo
    if args.merge_small_clusters:
        logging.error("small clusters merging not yet implemented for single target clustering!")
        raise NotImplementedError

    cluster_items_by_target_str = defaultdict(list)
    for cluster_item in cluster_items:
        cluster_items_by_target_str[cluster_item.target].append(cluster_item)
    for query, cluster_items_for_query in cluster_items_by_target_str.items():
        logging.info(f"target: {query}, for which there are {len(all_examples[query])} examples")
        if query not in output:
            output[query] = {}
        if supervised_items is None:
            gold = None
        else:
            gold = supervised_items[query]
            logging.info(f"gold label: {gold.rating}")
            output[query]["gold"] = gold.rating
        sents = [ex.sent for ex in cluster_items_for_query]
        logging.info(f"running {algo}")

        if not separate_eras:
            clustering_output = cluster_fn_lookup[algo](
                cluster_items_for_query,
                params[algo],
            )
            if clustering_output is None:
                logging.info(f"couldn't cluster {query}")
                continue
            else:
                clustered_examples, labels, best_k = clustering_output

            logging.info(f"testing eval method all eras single target")
            div, gain_or_loss, misc_output = eval_clusters_all_eras_single_target(
                params=params,
                examples=clustered_examples,
                cluster_labels=labels,
                gold=gold,
                threshold_clusters=args.min_cluster_threshold,
                logging=logging,
            )

            if args.cluster_associations:
                misc_output["cluster_associations"] = \
                    cluster_tf_idf_features(sents=sents, cluster_labels=labels, lang_code=params['lang_code'])
            if args.diagnostics:
                cluster_diagnostics(
                    clustered_items=clustered_examples,
                    cluster_labels=labels,
                    targets=test_items,
                )
                # TODO: maybe organize this better
                output[str(query)]["__GLOBAL__"] = misc_output
            if args.print_sents:
                misc_output['sents'] = print_sents_in_clusters(
                    sents, cluster_labels=labels, lang_code=params['lang_code'])
            # TODO: this needs to be fixed
            if div is None:
                # output[str(query)] = None
                continue

        else:
            logging.info(f"testing separate eras single target")
            labels_early, examples_early, best_k_early = cluster_fn_lookup[algo](
                cluster_items_for_query,
                params[algo],
                total_time_range=time_early
            )
            labels_late, examples_late, best_k_kate = cluster_fn_lookup[algo](
                cluster_items_for_query,
                params[algo],
                total_time_range=time_late
            )
            if labels_early is None or labels_late is None:
                # output[str(query)] = None
                continue
            mapping = map_clusters(
                params=params,
                all_items=cluster_items_for_query,
                labels_1=labels_early,
                labels_2=labels_late,
            )
            div, t1_dist, t2_dist = mapped_cluster_divergence(labels_early, labels_late, mapping)
            gain_or_loss = single_target_gain_or_loss(
                t1_labels=labels_early.tolist(), t2_labels=labels_late.tolist(),
                min_total_count=10,
                min_stable_count=2,
            )
            pred = predict_change_in_sense_inventory(div, gain_or_loss[0], gain_or_loss[1], bool)
            misc_output = None
            if args.print_sents:
                misc_output = {}
                time_early = COARSE_TIME_SLICES[params['dataset_type']][0]
                time_late = COARSE_TIME_SLICES[params['dataset_type']][-1]
                misc_output['sents_early'] = print_sents_in_clusters(
                    [sent for sent in sents
                     if year_to_coarse_slice(sent.year, params['dataset_type']) == time_early],
                    cluster_labels=labels_early, lang_code=params['lang_code'])
                misc_output['sents_late'] = print_sents_in_clusters(
                    [sent for sent in sents
                     if year_to_coarse_slice(sent.year, params['dataset_type']) == time_late],
                    cluster_labels=labels_late, lang_code=params['lang_code']
                )
            output[str(query)] = {
                "div": div, "gain_loss": gain_or_loss, "pred_from_static_threshold": pred,
                "misc": misc_output,
            }
            if isinstance(query, CompositionalityRating):
                per_clustering_per_target_divergences[query] = div

    # calculate average divergence
    # TODO: fix this -- commented out just to get the raw
    #       clustering output
    # divergence_threshold = sum(output[q]['div']
    #                            for q in output) \
    #                        / len(output)
    # for q in output:
    #     output[q]['pred'] = \
    #         output[q]['div'] > divergence_threshold


    # finally: aggregate results if there are gold scores to compare with
    if supervised_items:
        # reduce to only items available in output:
        supervised_items = [it for it in supervised_items if it.text in output]
        all_preds = [output[it.text]["pred"] for it in supervised_items]
        all_golds = [it.rating for it in supervised_items]
        output.update(aggregate_score(all_golds, all_preds))
    return clustered_examples

def multi_target_clustering(
    *,
    args,
    params,
    separate_eras: bool,
    related_compounds_clustering: bool,
    time_early,
    time_late,
    test_items, # these are here in case we need to re-link the examples' constituents
    cluster_items,
    feature_set_used,
    include_heads,
    include_mods,
    supervised_items: Dict[str, SupervisedTestItem],
    phitag_annotated_pairs: Optional[List[Tuple[ClusterItem, ClusterItem]]],
    phitag_annotation_lookup: Optional[Dict[ClusterItem, Tuple[List["PhitagExample"], List["PhitagAnnotation"]]]],
    phitag_ratings: Optional[PhitagRatings],
    phitag_time_based_similarity_proportions,
    phitag_time_based_similarity_counts,
    algo,
    cluster_fn_lookup,
    per_clustering_per_target_divergences, # modified
    output, # modified
):
    output_algo_name = algo + "+separate_eras" if separate_eras else algo

    logging.info(f"{len(cluster_items)} cluster items in big clustering")
    clustered_examples, labels_big_cluster, best_k = cluster_fn_lookup[algo](
        cluster_items,
        params[algo],
        args.merge_small_clusters,
    )
    output["BEST_K"] = best_k
    if DEBUG:
        # print breakdown of per-target / per year of what was clustered
        clustered_by_year = defaultdict(lambda: defaultdict(int))
        for c in clustered_examples:
            clustered_by_year[c.target][str(year_to_coarse_slice(c.sent.year, dataset=params['dataset_type']))] += 1
        logging.info(f"items clustered:")
        for k in clustered_by_year:
            logging.info(f"{k}: {clustered_by_year[k]}")

    raw_clustering_output = []
    for example, label in zip(clustered_examples, labels_big_cluster):
        raw_clustering_output.append({
            'example': example.id, 'label': str(label),
            # 'text': example.str_with_span_delimited(),
        })

    # otherwise we repeat these many times.
    common_arguments = {
            "params": params,
            "examples": clustered_examples,
            "cluster_labels": labels_big_cluster,
            "gold": supervised_items,
            "threshold_clusters": args.min_cluster_threshold,
            "logging": logging
    }
    if related_compounds_clustering:
        eval_outputs, global_outputs = eval_clusters_multi_target_with_shared_constituent_groups(
            **common_arguments,
            target_compounds=test_items,
        )
    elif include_heads or include_mods:
        if include_heads:
            constituent_type: Literal["head"] = "head"
        else:
            constituent_type: Literal["mod"] = "mod"
        common_arguments["target_compounds"] = test_items
        if not (include_heads and include_mods):
            logging.info(f"multi target w/r/t constituent")

            eval_outputs, global_outputs = eval_clusters_multi_target_with_respect_to_constituents(
                **common_arguments,
                constituent_type=constituent_type,
            )
            if global_outputs: # TODO: include this in other eval modes
                output["__GLOBAL__"] = global_outputs
        else: # BOTH heads AND mods
            logging.info("multi target w/r/t mod and head constituents")
            eval_outputs_head, global_outputs_head = eval_clusters_multi_target_with_respect_to_constituents(
                **common_arguments,
                constituent_type="head",
            )
            eval_outputs_mod, global_outputs_mod = eval_clusters_multi_target_with_respect_to_constituents(
                **common_arguments,
                constituent_type="mod",
            )
            # now we merge the output dictionaries:
            eval_outputs, global_outputs = {}, {}
            for k in eval_outputs_head:
                eval_outputs[f"{k}_HEAD"] = eval_outputs_head[k]
            for k in eval_outputs_mod:
                eval_outputs[f"{k}_MOD"] = eval_outputs_mod[k]
            for k in global_outputs_head:
                global_outputs[f"{k}_HEAD"] = global_outputs_head[k]
            for k in global_outputs_mod:
                global_outputs[f"{k}_MOD"] = global_outputs_mod[k]

    else:
        eval_outputs = eval_clusters_multi_target(
            **common_arguments
        )

    # it's possible to have a None in this list if one time period is lacking one of the
    # items of comparison
    eval_outputs = {q: eval_outputs[q] for q in eval_outputs if eval_outputs[q] is not None}
    # this way the denominator of the average will be correct
    # if not len(eval_outputs):
    #     logging.info(f"nothing to evaluate")
    #     return clustered_examples
    # calculate average divergence values to set threshold:
    if related_compounds_clustering:
        # ALL (primary and secondary) entries in eval outputs have a 'div_t1_t2' entry
        # TODO: why are many entries not getting a div_t1_t2 result??
        eval_outputs = {q: eval_outputs[q] for q in eval_outputs if "div_t1_t2" in eval_outputs[q]}

        divergence_threshold = sum(
            eval_outputs[q]["div_t1_t2"]
            for q in eval_outputs
        ) / len(eval_outputs) if len(eval_outputs) else 0
        for q in eval_outputs:
            eval_outputs[q]['pred'] = \
                eval_outputs[q]['div_t1_t2'] > divergence_threshold
    elif include_heads or include_mods:
        # TODO: we could expand on how this should be calculated when
        #       *both* heads and mods are included

        divergence_threshold = sum(abs(eval_outputs[q]['div_t1'] - eval_outputs[q]['div_t2'])
                                   for q in eval_outputs if 'div_t1' in eval_outputs[q])\
                               / len(eval_outputs) if len(eval_outputs) else 0
        for q in eval_outputs:
            if "div_t1" not in eval_outputs[q]:
                continue
            eval_outputs[q]['pred'] = \
                abs(eval_outputs[q]['div_t1'] - eval_outputs[q]['div_t2']) > divergence_threshold

    else:
        divergence_threshold = sum(eval_outputs[q]['div_t1_t2']
                                   for q in eval_outputs) \
                               / len(eval_outputs)
        for q in eval_outputs:
            if 'div_t1_t2' not in eval_outputs[q]:
                continue
            eval_outputs[q]['pred'] = \
                 eval_outputs[q]['div_t1_t2'] > divergence_threshold
    logging.info(f"Divergence threshold set: {divergence_threshold}")




    logging.info(f"big clustering {algo}:")
    for query in eval_outputs.keys():
        query = str(query)
        if eval_outputs[query] is None:
            logging.info(f"no results for {query}, skipping... ")
            if DEBUG:
                logging.info(f"literal contents of eval outputs: {eval_outputs}")
            continue
        # TODO: this situation is kind of asking for these outputs to be more structured
        #       even if it's not much more than a wrapper around a dict - just to avoid losing track of the different
        #       keys
        if query not in output:
            output[query] = {}
        if 'div' in eval_outputs[query]:
            logging.info(f"result for {query}: div: {eval_outputs[query]['div']}, "
                     f"pred: {eval_outputs[query]['pred']}, "
                     f"gold: {supervised_items[query].rating if supervised_items is not None else 'n/a'}")
            output[query].update({'div': eval_outputs[query]['div'],
                                             'pred': eval_outputs[query]['pred']})
        if 'div_t1' in eval_outputs[query] and 'div_t2' in eval_outputs[query]:
            logging.info(f"result for {query}:\n"
                         f"div_t1: {eval_outputs[query]['div_t1']}, "
                         f"div_t2: {eval_outputs[query]['div_t2']}, "
                         f"pred: {eval_outputs[query]['pred']}")
            output[query].update({'div_t1': eval_outputs[query]['div_t1'],
                                             'div_t2': eval_outputs[query]['div_t2'],
                                             'pred': eval_outputs[query]['pred']})
        if 'div_t1_t2' in eval_outputs[query]:
            logging.info(f"result for {query}: \n"
                         f"div_t1_t2: {eval_outputs[query]['div_t1_t2']}"
            )
            # there can be NESTED entries for related compound targets in here...
            output[query].update(eval_outputs[query])
            if "pred" in eval_outputs[query]:
                logging.info(f"pred: {eval_outputs[query]['pred']}")
            if "avg_pairwise_div_t1" in eval_outputs[query]:
                logging.info(
                    f"average pairwise divergence with related compounds (t1): "
                    f"{eval_outputs[query]['avg_pairwise_div_t1']}\n"
                    f"average pairwise divergence with related compounds (t2): "
                    f"{eval_outputs[query]['avg_pairwise_div_t2']}\n"
                )

    if args.diagnostics:
        cluster_diagnostics(
            clustered_items=clustered_examples,
            cluster_labels=labels_big_cluster,
            targets=test_items,
        )

    if supervised_items:
        all_preds = [output[it.text]["pred"] for it in supervised_items.values() if it.text in eval_outputs]
        all_golds = [it.rating for it in supervised_items.values() if it.text in eval_outputs]
        output.update(aggregate_score(all_golds, all_preds))

    if phitag_annotated_pairs and phitag_annotation_lookup:
        # check if annotated examples ended up in the same clusters or not
        # type_level_annotations = type_level_ratings(phitag_ratings)
        phitag_ratings_outputs = []
        for it_0, it_1 in phitag_annotated_pairs:
            if it_0 not in clustered_examples or it_1 not in clustered_examples:
                continue
            it_0_index = clustered_examples.index(it_0)
            it_1_index = clustered_examples.index(it_1)
            it_0_label = labels_big_cluster[it_0_index]
            it_1_label = labels_big_cluster[it_1_index]
            # get the average of the annotations for this pair
            it_0_examples, it_0_annotations = phitag_annotation_lookup[it_0]
            it_1_examples, it_1_annotations = phitag_annotation_lookup[it_1]
            for (it_0_example, it_1_example), (annotations_ls_0, annotations_ls_1) in zip(
                    product(
                        it_0_examples, it_1_examples
                    ),
                    product(
                        it_0_annotations, it_1_annotations
                    )
            ):
                if annotations_ls_0 == annotations_ls_1:
                    annotations_mean_stddev = mean_std_dev_ratings(annotations_ls_0)
                    if annotations_mean_stddev[0] is None:
                        logging.warning(f"annotation for {it_0} and {it_1} was not available")
                        continue
                    # NOW we have both the rating and whether the two items were in the same or different clusters
                    collapsed_rating = discrete_to_binary_similarity_rating(annotations_mean_stddev[0])
                    phitag_ratings_outputs.append({
                        "pair": [it_0.id, it_1.id],
                        "eras": [it_0_example.time_grouping, it_1_example.time_grouping],
                        "mean_rating": annotations_mean_stddev[0],
                        "std_dev": annotations_mean_stddev[1],
                        "clustered_together": it_0_label == it_1_label,
                        "collapsed_similarity_rating": collapsed_rating,
                        "clustered_correctly": (it_0_label == it_1_label) == collapsed_rating,
                    })
                    break
        output["phitag_eval"] = phitag_ratings_outputs
        # per target summary from phitag:
        for target, era_d in phitag_time_based_similarity_proportions.items():
            if target in output:
                output[target]['phitag-similarity-proportions'] = era_d


    # add raw cluster output for comparison across multiple runs
    # (but at the end because it isn´t nice to look at)
    output["clustering_output"] = raw_clustering_output
    return clustered_examples


def get_items_sample(
    *,
        model_params,
        max_items: int = 10000,
        minimum_per_target_allocation: int = 10,
        separate_eras: bool, # separate era clustering (i.e. early/late)
        test_items: List[Union[CompositionalityRating, TestWord, RelatedCompoundSet]],
        cluster_items: List[ClusterItem],
        include_heads: bool,
        include_mods: bool,
        related_compounds_clustering: bool,
        time_to_target_to_freq,
        constituent_multiplier=1, # how many constituents to sample per sampled compound
        dataset_type: str,
        test_items_distribution,
        all_time_dist,
        annotated_pairs,


) -> Dict[str, List[Tuple[List[np.ndarray], Sentence]]]:
    """ returns a modified copy of cluster_items, containing a maximum number of total examples,
        distributed in such a way as to respect some aspects of the original distribution of
        test items with respect to era (and maybe their relative frequency compared with the other test items)
    """
    assert max_items > 0
    if model_params:
        if "max_num_examples" in model_params:
            max_items = int(model_params['max_num_examples'])
            logging.info(f"using {max_items} examples (max)")
        if 'constituent_multiplier' in model_params:
            constituent_multiplier = int(model_params['constituent_multiplier'])
        if 'minimum_uniform_alloc' in model_params:
            minimum_per_target_allocation = model_params['minimum_uniform_alloc']

    ## preliminary filter (TODO: put in own helper fn):

    main_target_strs: List[str] = sorted([test_item.primary_component for test_item in test_items])
    time_slices = [k for k in time_to_target_to_freq]
    targets_meeting_threshold = []
    for target in main_target_strs:
        meets_criteria = True
        for sl in time_slices:
            freq = time_to_target_to_freq[sl][target]
            if freq < minimum_per_target_allocation:
                meets_criteria = False
                break
        if meets_criteria:
            targets_meeting_threshold.append(target)

    all_compound_strs = targets_meeting_threshold
    all_target_strs: List[str] = []
    allowed_keys = set()
    for t in test_items:
        if isinstance(t, CompositionalityRating):
            if t.compound not in all_compound_strs:
                continue
            allowed_keys.add(t.compound)
            all_target_strs.append(t.compound)
            all_target_strs.append(t.mod)
            all_target_strs.append(t.head)
            if include_mods:
                allowed_keys.add(t.mod)
            if include_heads:
                allowed_keys.add(t.head)
        elif isinstance(t, RelatedCompoundSet):
            for c in t.compounds:
                allowed_keys.add(c)
        else:
            allowed_keys.add(t.primary_component)
            all_target_strs.append(t.primary_component)

    logging.info(f"targets with minimum per-time-slice allocation:\n"
                 f"{all_compound_strs}")
    logging.info(f"all allowed keys: {[k for k in allowed_keys]}")
    cluster_items = [c for c in cluster_items if c.target in allowed_keys]
    ## end preliminary filter
    all_examples_dict = defaultdict(list)
    for cluster_item in cluster_items:
        all_examples_dict[cluster_item.target].append(cluster_item)


    sampled_examples = defaultdict(list) # mapping item -> list of examples
    sampled_examples_count = 0

    # TODO: include all examples that were annotated, using that collection as the initial
    #       (relatively) uniform sample -- they *were* sampled at random anyway
    #       (well... if we have them available)
    annotated_example_counts_by_time = None
    if annotated_pairs:
        annotated_example_counts_by_time = defaultdict(lambda: defaultdict(int))
        for item_1, item_2 in annotated_pairs:
            if item_1.target not in allowed_keys:
                continue
            sampled_examples[item_1.target] += [item_1, item_2]
            sampled_examples_count += 2
            era_1 = year_to_coarse_slice(item_1.sent.year, dataset_type)
            annotated_example_counts_by_time[item_1.target][era_1] += 1
            era_2 = year_to_coarse_slice(item_2.sent.year, dataset_type)
            annotated_example_counts_by_time[item_2.target][era_2] += 1
            # find and remove these items from the list of things we can still sample from
            # some pairs are annotated twice!
            if item_1 in all_examples_dict[item_1.target]:
                all_examples_dict[item_1.target].remove(item_1)
            if item_2 in all_examples_dict[item_2.target]:
                all_examples_dict[item_2.target].remove(item_2)

    # FIRST uniformly populate the sample w/ the minimum allocation if possible
    # (bringing constituents along during this as well?\
    # use random.choices(with the time dist as weights, slices as the objects,  k of 1 for one element) for choices w/ replacement

    # alternative lookup of example frequency needed to keep sampler in sync across varying
    # sets of example items

    # need to construct the buckets to draw from:
    # break down examples by coarse time slice
    # (probably this is going to be grossly inefficient to deal with)
    example_sampler = TimeStratifiedSampling(
        timeslice_to_all_keys_to_counts=time_to_target_to_freq,
        all_examples_dict=all_examples_dict,
        dataset_type=dataset_type,
    )
    total_items_in_sampler = example_sampler.total_possible_examples
    logging.info(f"total items in sampler: {total_items_in_sampler}")


    # first we'll get a minimum sample
    for time_slice in time_to_target_to_freq:
        constituents_in_this_time_slice = set()
        for test_item in test_items:
            test_item_str = test_item.primary_component
            # get minimum sample (but subtract any supervised entries already included)
            if annotated_example_counts_by_time:
                remaining_allocation = minimum_per_target_allocation - annotated_example_counts_by_time[test_item_str][time_slice]
            else:
                remaining_allocation = minimum_per_target_allocation - len(sampled_examples[test_item_str])

            sample = example_sampler.get_samples(time_slice, test_item_str, remaining_allocation)
            compounds_sampled = len(sample)
            # NB: if constituent examples are annotated, this will need to change
            #     to accommodate them!
            if sample:
                sampled_examples[test_item_str] += sample
                sampled_examples_count += compounds_sampled

            num_secondary_samples = minimum_per_target_allocation
            if annotated_example_counts_by_time:
                num_secondary_samples = compounds_sampled + annotated_example_counts_by_time[test_item_str][time_slice]

            # FOR THE MINIMUM SAMPLE: don´t include shared constituents multiple times
            if include_heads and hasattr(test_item, "head") and test_item.head not in constituents_in_this_time_slice:
                head_samples = example_sampler.get_samples(time_slice, test_item.head, num_secondary_samples * constituent_multiplier)
                if head_samples:
                    sampled_examples[test_item.head] += head_samples
                    sampled_examples_count += len(head_samples)
                constituents_in_this_time_slice.add(test_item.head)
            if include_mods and hasattr(test_item, "mod") and test_item.mod not in constituents_in_this_time_slice:
                mod_samples = example_sampler.get_samples(time_slice, test_item.mod, num_secondary_samples * constituent_multiplier)
                if mod_samples:
                    sampled_examples[test_item.mod] += mod_samples
                    sampled_examples_count += len(mod_samples)
                constituents_in_this_time_slice.add(test_item.mod)
            if hasattr(test_item, "secondary_components"):
                for secondary_component in test_item.secondary_components:
                    if secondary_component in constituents_in_this_time_slice:
                        continue
                    samples = example_sampler.get_samples(time_slice, secondary_component, num_secondary_samples * constituent_multiplier)
                    if samples:
                        sampled_examples[secondary_component] += samples
                        sampled_examples_count += len(samples)
                    constituents_in_this_time_slice.add(secondary_component)

    logging.info(f"after minimum allocation:")
    for k, samples in sampled_examples.items():
        logging.info(f"key: {k} has {len(samples)} samples")

    while sampled_examples_count < max_items:
        # first pick a time slice
        # then sample a target from it (then add the constituents related to it)
        slice_choice = random.choices(
            population=time_slices,
            weights=all_time_dist,
            k=1
        )[0]
        slice_index = time_slices.index(slice_choice)
        # get the index of the time slice
        target_choice = random.choices(
            population=test_items,
            weights=test_items_distribution[slice_index],
            k=1
        )[0]
        target_choice_str = target_choice.primary_component

        sample = example_sampler.get_sample(slice_choice, target_choice_str)
        if not sample:
            if all([example_sampler.item_is_exhausted(it) for it in all_compound_strs]):
                logging.info(f"No more compounds to sample: ending sampling")
                break
            continue
        sampled_examples[target_choice_str].append(sample)
        sampled_examples_count += 1
        if include_heads and not related_compounds_clustering:
            head_samples = example_sampler.get_samples(slice_choice, target_choice.head, constituent_multiplier)
            if head_samples:
                sampled_examples[target_choice.head] += head_samples
                sampled_examples_count += len(head_samples)
        if include_mods and not related_compounds_clustering:
            mod_samples = example_sampler.get_samples(slice_choice, target_choice.mod, constituent_multiplier)
            if mod_samples:
                sampled_examples[target_choice.mod] += mod_samples
                sampled_examples_count += len(mod_samples)
        if related_compounds_clustering:
            for secondary_component in target_choice.secondary_components:
                secondary_samples = example_sampler.get_samples(
                    slice_choice,
                    secondary_component,
                    constituent_multiplier
                )
                if secondary_samples:
                    sampled_examples[secondary_component] += secondary_samples
                    sampled_examples_count += len(secondary_samples)



    # first we need a distribution of the time slices to draw from
    # then we draw a target term for that time slice, and add it to the list

    # the complication is how this interacts with decomposing compounds into
    # also pulling head or modifier constituents.

    # I dunno. we could randomly draw a constituent (if applicable) X some modifier
    # (i.e. draw 3 "mine"s for every one draw of "gold mine"

    # this would be sampling without replacement
    # and we will need some logic to handle if we have exhausted all
    # examples from a particular test item? (or not?, just re-try?)
    #   -> obviously it *could* happen... just like you can flip 20 heads in a row
    logging.info(f"after remaining sampling:")
    for k, samples in sampled_examples.items():
        logging.info(f"key: {k} has {len(samples)} samples")
    return [c for k in sampled_examples for c in sampled_examples[k]]

def get_distribution_of_targets(
    time_to_target_to_freq,
    targets: List[CompositionalityRating],
    time_slices: List[Tuple[int, int]],
):

    # length of target should be a function of the corpus type / language
    # so we should only return one distro, whether it comes from uni or bigrams
    # depends on the language.



    target_freqs = {} # str -> time slice -> frequency
    for target in targets:
        target_freqs[target.primary_component] = {
            time: time_to_target_to_freq[time][target.primary_component] for time in time_to_target_to_freq
        }


    # total counts will be either unigram or bigram depending on the dataset
    total_counts = {}

    for time_slice in time_slices:
        total_counts[time_slice] = sum(
            [time_to_target_to_freq[time_slice][t] for t in time_to_target_to_freq[time_slice]]
        )

    all_time_slices_total = sum(total_counts.values())


    # turn into probability distribution based
    # on counts

    per_time_target_distributions = []
    for i, time_slice in enumerate(time_slices):
        targets_array = np.array([
            target_freqs[t][time_slice]
            for t in target_freqs
        ])
        per_time_target_distributions.append(
            targets_array / targets_array.sum()
        )

    all_times_unigrams_vec = np.array([
       total_counts[t] for t in time_slices
    ])
    all_times_unigrams_dist = all_times_unigrams_vec / all_times_unigrams_vec.sum()

    return per_time_target_distributions, all_times_unigrams_dist, time_to_target_to_freq




def k_means_clustering(
        examples: List[ClusterItem],
        params,
        merge_small_clusters=False,
) -> Tuple[List[ClusterItem], np.ndarray, int]:
    # raise NotImplementedError("Currently incoherent due to change to complex ClusterItem -> prefer k_medoids")
    """

    :param target:
    :param examples:
    :return: array of ints (of order len(examples))
    """

    # stack the examples into one big array
    # note: this might not be ideal, as it is treating each embedding dim as a *feature*

    # first we need to average multi-token spans (if any)
    if len(examples) < params['k']:
        return None
    if len(examples) == 1:
        # special case: the cluster of a single item is just [0]
        return np.ndarray((1,), buffer=np.array([0]))

    X = np.stack([example.normalized for example in examples])

    eval = UnsupervisedClusterEval('silhouette')
    best_silhouette, best_k, best_labels = None, None, None
    for k in range(4, 33):
        k_means = KMeans(
            n_clusters=k,
        )
        fit = k_means.fit(
            X
        )
        labels = fit.labels_
        eval_results = eval.evaluate_cluster_results(arr=X, labels=labels)
        # TODO: remove extra logging:
        logging.info(f"k: {k} -> silhouette {eval_results:.6f}")
        if best_silhouette is None or eval_results > best_silhouette:
            best_silhouette = eval_results
            best_k = k
            best_labels = labels


    if merge_small_clusters:
        X, labels, examples = merge_clusters(
            stacked_examples=X,
            labels=fit.labels_,
            n_clusters=params['k'],
            examples=examples,
        )
    else:
        labels = fit.labels_


    logging.info(f"best k: {best_k}")
    logging.info(f"cluster eval silhouette: {best_silhouette:.6f}")


    # centers = fit.cluster_centers_
    # print(centers)
    # centers are defined in the feature space (so an array of size (n_clusters, n_features))
    # otherwise for item / cluster membership, look at k_means object's labels_ member
    return examples, best_labels, best_k


def per_target_aff_prop(
        examples: List[ClusterItem],
        params,
        evals=None,
        merge_small_clusters=False,
):
    aff_prop = AffinityPropagation(
        affinity="precomputed" # meaning, we are going to pre compute cosine distances/similarities
    )
    # need to create a (n_samples, n_samples) matrix populated with cosine dist values between examples
    if not len(examples):
        return None
    if len(examples) > params['max_num_examples']:
        logging.info(f"{len(examples)} exceeds max allowable for aff prop ({params['max_num_examples']})")
        return None
    dist_matrix = np.ndarray((len(examples), len(examples)))
    indices_i, indices_j = np.triu_indices_from(dist_matrix)
    for i, j in zip(indices_i, indices_j):
        # TODO: include [j][i] here???
        dist_matrix[i][j] = examples[i].distance(examples[j])


    return examples, aff_prop.fit_predict(X=dist_matrix)

def k_medoids(
        examples: List[ClusterItem],
        params,
        evals=None,
        merge_small_clusters=False,
):
    km = KMedoids(n_clusters=params['k'], metric='precomputed')
    if not len(examples) or len(examples) < params['k']:
        return None
    dist_fn = functools.partial(distance_vectorized, examples[0])
    vectorized = np.array([x.concatenated for x in examples])




    dist_matrix = \
        pairwise_distances(vectorized, metric=dist_fn, n_jobs=32)



    fit = km.fit(dist_matrix)
    return examples, fit.labels_

###***NOTE: pretty sure this is deprecated now...
def big_clustering_aff_prop(all_examples: Dict[Union[str, CompositionalityRating], List[Tuple[List[np.ndarray], Sentence]]]
):
    raise NotImplementedError
    # in the situation of wanting compounds/mods/heads all thrown into the clustering, str -> emb is
    # maybe the way to go

    # we need to be able to map back from (clustered_embedding) to -> target word / compound
    # (as well as to have access to the year, via the Sentence)
    cluster_items = []
    for target, ex_tuples in all_examples.items():
        for example_tuple in ex_tuples:
            example_list, sent = example_tuple
            averaged_example = functools.reduce(
                    lambda acc, x: acc.__add__(x), example_list, np.zeros_like(example_list[0])
                ) / len(example_list)

            cluster_items.append(ClusterItem(
                embeddings=[averaged_example],
                associated_sent=sent,
                associated_target=target
            ))
    # so now we have a flat bunch of ClusterItems, and we can access
    # the associated target term or sentence that the contextualized embedding was related to
    dist_matrix = np.ndarray((len(cluster_items), len(cluster_items)))
    for i in range(len(cluster_items)):
        for j in range(len(cluster_items)):
            # TODO: don't bother calculating dist(y, x) when you already know dist(x, y)
            dist_matrix[i][j] = scipy.spatial.distance.cosine(cluster_items[i].avg_embedding, cluster_items[j].avg_embedding)
    aff_prop = AffinityPropagation(
        affinity="precomputed" # meaning, we are going to pre compute cosine distances/similarities
    )
    return aff_prop.fit_predict(X=dist_matrix)

def merge_clusters(
        stacked_examples,
        labels,
        examples: List[ClusterItem],
        n_clusters: int,
        minimum_uses: int=10) -> np.array:
    """

    """
    # Implementation of Montairol/Martinc/Pivovarova 's method:
    # First take clusters with >= a minimum size (10), and average all representations within
    # these clusters. Then compute a cosine distance matrix between the averaged cluster representations,
    # and use it to merge clusters whose pairwise distance is below a threshold:
    # avg_cos_dist - 2 * std_cos_dist
    # then, do the same thing but for "illegitimate clusters" that did not meet the initial threshold of 10
    # items. Apply the merging procedure until the dist between the two closest clusters is **larger** than
    # the distance threshold.
    # At the end, if any cluster remains below the size threshold, it is removed

    # average representations per cluster:
    dist_matrix = _make_per_cluster_dist_matrix(
        n_clusters=n_clusters,
        stacked_examples=stacked_examples,
        labels=labels,
        minimum_uses=minimum_uses,
    )
    # the 1 is to not include the main diagonal
    dists_vec = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]

    dist_mean = np.mean(dists_vec)
    dist_std = np.std(dists_vec)
    threshold = dist_mean - 2 * dist_std
    logging.info(f"pre-merge averaged cluster statistics: mean: {dist_mean:.02f}, std-dev: {dist_std:.02f}, threshold: {threshold}")
    # should this be abs?

    # this is a original label to new label mapping
    cluster_merge_lookup = {i: i for i in range(n_clusters)}
    # merging will just always go to the label name of the smaller number
    # s/t merging cluster 3 and 8 updates the mapping of 8 -> 3
    prev_closest_distance = None

    while True:

        indices_i, indices_j = np.triu_indices_from(dist_matrix, k=1)
        for i, j in zip(indices_i, indices_j):
            if dist_matrix[cluster_merge_lookup[i]][cluster_merge_lookup[j]] < threshold:
                # so i and j are pointing to averaged representations of cluster i or j
                if cluster_merge_lookup[i] < cluster_merge_lookup[j]:
                    cluster_merge_lookup[j] = cluster_merge_lookup[i]
                else:
                    cluster_merge_lookup[i] = cluster_merge_lookup[j]

        # calculate stop criteria: dist between two closest clusters is larger than the threshold
        # in a do-while sort of fashion
        closest_clusters = _closest_cluster_distance(
            dist_matrix,
            [cluster_merge_lookup[labels[i]] for i in range(labels.size)]
        )
        if prev_closest_distance is None:
            prev_closest_distance = closest_clusters
        elif abs(closest_clusters - prev_closest_distance) < 1e-10:
            break
        if closest_clusters is not None:
            logging.info(f"closest cluster distance after a round of merging: {closest_clusters:.02f}")
        else:
            logging.info(f"closest cluster distance was None -- could happen if everything is in one cluster")
        if closest_clusters is None or closest_clusters > threshold:
            break
        dist_matrix = _make_per_cluster_dist_matrix(
            n_clusters=n_clusters,
            stacked_examples=stacked_examples,
            labels=[cluster_merge_lookup[labels[i]] for i in range(labels.size)],
            minimum_uses=minimum_uses,
        )

    # discard tiny clusters
    # need count of cluster membership
    cluster_membership_count = defaultdict(int)
    mask_indices = []
    for label in labels:
        cluster_membership_count[cluster_merge_lookup[label]] += 1
    for i, label in enumerate(labels):
        if cluster_membership_count[cluster_merge_lookup[label]] >= minimum_uses:
            mask_indices.append(i)
    filtered_labels = [cluster_merge_lookup[labels[i]] for i in range(labels.size)
                       if i in mask_indices]
    examples = [ex for i, ex in enumerate(examples) if i in mask_indices]
    logging.info(f"{len(labels) - len(filtered_labels)} labels filtered out for being in "
                 f"clusters that are too small")
    # returned collections all conform to the size of things with the leftover
    # tiny clusters filtered out
    return stacked_examples[mask_indices], filtered_labels, examples

def _closest_cluster_distance(pairwise_distance_matrix, cluster_labels):
    min_dist = None
    indices_i, indices_j = np.triu_indices_from(pairwise_distance_matrix, k=1)
    for i, j in zip(indices_i, indices_j):
        if cluster_labels[i] == cluster_labels[j]:
            continue
        if min_dist is None:
            min_dist = pairwise_distance_matrix[i][j]
        if pairwise_distance_matrix[i][j] < min_dist:
            min_dist = pairwise_distance_matrix[i][j]
    return min_dist

def _make_per_cluster_dist_matrix(n_clusters, stacked_examples, labels, minimum_uses: int):
    per_cluster_repr = np.zeros((n_clusters, stacked_examples.shape[-1]))
    per_cluster_divisor = np.zeros((n_clusters,))
    for i in range(n_clusters):
        per_cluster_repr[labels[i]] += stacked_examples[i]
        per_cluster_divisor[labels[i]] += 1
    for i in range(n_clusters):
        if per_cluster_divisor[i]:
            if per_cluster_divisor[i] < minimum_uses:
                # zero out any cluster's representation that is too small
                per_cluster_repr[i] *= 0
            else:
                per_cluster_repr[i] = per_cluster_repr[i] / per_cluster_divisor[i]

    # calculate pairwise distances only for nonzero rows
    return pairwise_distances(
        per_cluster_repr[np.any(per_cluster_repr, axis=1)],
        metric="cosine", n_jobs=6)


def bifurcated_averaged_representations(
        cluster_item_list: List[ClusterItem],
        dataset_type: Literal["COHA", "DTA"],
) -> Dict[str, float]:
    """

    :param all_examples:
    :return:  mapping between target and the similarity score of its two averaged representations
    """
    cluster_items_by_target = defaultdict(list)
    for it in cluster_item_list:
        cluster_items_by_target[it.target].append(it)
    all_examples_avg_early, all_examples_avg_late = {}, {}
    for example_name, examples in cluster_items_by_target.items():

        example_representation_early = np.zeros_like(examples[0].concatenated)
        example_representation_late = np.zeros_like(examples[0].concatenated)
        early_count, late_count = 0, 0
        for example in examples:
            embeddings, sent = example.concatenated, example.sent

            if year_to_coarse_slice(sent.year, dataset_type) == COARSE_TIME_SLICES[dataset_type][0]:
                example_representation_early += embeddings
                early_count += 1
            if year_to_coarse_slice(sent.year, dataset_type) == COARSE_TIME_SLICES[dataset_type][-1]:
                example_representation_late += embeddings
                late_count += 1
        if early_count:
            example_representation_early /= early_count
            all_examples_avg_early[example_name] = example_representation_early
        if late_count:
            example_representation_late /= late_count
            all_examples_avg_late[example_name] = example_representation_late

    scores = {}
    # then go thru each and output cosine dist values
    for example_name in cluster_items_by_target.keys():
        if example_name in all_examples_avg_early and example_name in all_examples_avg_late:
            scores[example_name] = scipy.spatial.distance.cosine(all_examples_avg_early[example_name], all_examples_avg_late[example_name])
    return scores

def average_pairwise_distances(
        cluster_item_list: List[ClusterItem],
        dataset_type: Literal["COHA", "DTA"],
):
    # sort examples by target and timespan
    early_era = COARSE_TIME_SLICES[dataset_type][0]
    late_era = COARSE_TIME_SLICES[dataset_type][-1]
    mapping = defaultdict(lambda: defaultdict(list)) # target -> time -> list of examples
    for c_item in cluster_item_list:
        era = year_to_coarse_slice(c_item.sent.year, dataset_type)
        mapping[c_item.target][era].append(c_item)
    output_avg_pairwise_dists = {} # target -> score
    for target, eras_d in mapping.items():
        dist = 0
        num_pairs = 0
        combs = [c for c in combinations([eras_d[early_era], eras_d[late_era]], 2)]
        for combination in combs:
            for pair in product(*combination):
                num_pairs += 1
                dist += pair[0].distance(pair[1])
        output_avg_pairwise_dists[target] = dist / num_pairs if num_pairs else 0

    return output_avg_pairwise_dists



def find_examples_all(
        params,
        targets: List[Union[TestWord, CompositionalityRating]],
        include_head_examples: bool,
        include_mod_examples: bool,
        use_bert_vecs: bool,
        use_random_idx_vecs: bool,
        use_second_order_vecs: bool,
        default_targets: List[str],
        cached_examples_dirs: Optional[List[str]]=None,
        frequency_stats_dir: Optional[str]=None,
        productivity_stats_dir: Optional[str]=None,
) -> Dict[str, List[Tuple[List[np.ndarray], Sentence]]]:
    """
    :param model:
    :param data: When we need to trawl through the data to obtain examples in context
    :param targets: Used to keep track of which constituent belongs to which target compound
                    (when applicable). Otherwise is just a list of target single words (strings).
                    If in the future there is some kind of compound class that doesn't have ratings,
                    the important thing here is the interface: .compound, .head, .mod to access
                    the needed string keys for the embeddings lookup that is output
    :param include_head_examples:
    :param include_mod_examples:
    :param use_bert_vecs: whether to return cached bert vecs
    :param use_random_idx_vecs: whether to return random indexing vecs
    :param device:
    :param cached_examples_dirs:
    :param cached_random_vecs_dirs:
    :param frequency_stats_dir:
    :param productivity_stats_dir:
    :return: An embeddings lookup (keyed only with the string(!) of the target compound or single word)
    """
    lookup = defaultdict(list)

    # expand targets list as needed, turning it into a list of strings
    targets: List[str] = _expand_targets(targets, include_heads=include_head_examples, include_mods=include_mod_examples)
    targets += default_targets
    targets = sorted(list(set(targets)))
    logging.info(f"targets list: {targets}")
    # from this point on, targets are a list of str
    num_targets = len(targets)
    examples = {}
    if cached_examples_dirs:
        filtered_lookup = load_vecs_cache(cached_examples_dirs, targets)
        logging.info(f"filtered the cached lookup to {len(filtered_lookup)} with {sum(len(filtered_lookup[t]) for t in filtered_lookup)}")
        for k, v in filtered_lookup.items():
            if k not in examples:
                examples[k] = []
            # examples[k] should have a list of dicts [{}, {}]
            # each dict having keys for 'sent', 'bert_vecs', 'random_vecs' etc. -- one per SPAN (within some sentence)
            examples[k] += [
                {"id": f"{k}::{i}",
                 BERT_VECS_C: e[BERT_VECS_C],
                 SECOND_ORDER_VECS_C: e[SECOND_ORDER_VECS_C]
                    if SECOND_ORDER_VECS_C in e else None,
                 'sent': e['sent'],
                 LEMMA_SPAN_C: e[LEMMA_SPAN_C]
                 }
                for i, e in enumerate(filtered_lookup[k])
                if year_to_coarse_slice(e['sent'].year, params['dataset_type'])
            ]
            if not use_bert_vecs:
                for example in examples[k]:
                    if BERT_VECS_C in example:
                        del example[BERT_VECS_C]
            if not use_random_idx_vecs:
                for example in examples[k]:
                    if RANDOM_VECS_C in example:
                        del example[RANDOM_VECS_C]
            if not use_second_order_vecs:
                for example in examples[k]:
                    if SECOND_ORDER_VECS_C in example:
                        del example[SECOND_ORDER_VECS_C]
            # examples[k].update(filtered_lookup[k])


    if frequency_stats_dir:
        freq_stats = load_freq_stats(
            [os.path.join(frequency_stats_dir, filename) for filename in os.listdir(frequency_stats_dir)],
            eras=COARSE_TIME_SLICES[params["dataset_type"]],
            dataset=params["dataset_type"],
        )
        for k, v in freq_stats.items():
            if k not in targets:
                continue
            if k not in examples:
                examples[k] = []
            for example in examples[k]:
                year = example['sent'].year
                era_slice = year_to_coarse_slice(year, params['dataset_type'])
                if era_slice and era_slice in freq_stats[k]:
                    example['frequency'] = freq_stats[k][era_slice]
                else:
                    # avoid div by zero
                    example['frequency'] = 0.0000001
    # TODO: productivity stats (need to keep track of which constituent is being used in clustering
    if productivity_stats_dir:
        if include_head_examples:
            prod_stats_heads = load_prod_stats(
                [os.path.join(productivity_stats_dir, filename) for filename in os.listdir(productivity_stats_dir)
                 if "head" in filename],
                eras=COARSE_TIME_SLICES[params["dataset_type"]],
                dataset=params["dataset_type"]
            )
            for k, v in prod_stats_heads.items():
                if k not in targets:
                    continue
                for example in examples[k]:
                    year = example['sent'].year
                    era_slice = year_to_coarse_slice(year, params['dataset_type'])
                    if era_slice:
                        example["head_productivity"] = prod_stats_heads[k][era_slice]
        if include_mod_examples:
            prod_stats_mods = load_prod_stats(
                [os.path.join(productivity_stats_dir, filename) for filename in os.listdir(productivity_stats_dir)
                 if "mod" in filename],
                eras=COARSE_TIME_SLICES[params["dataset_type"]],
                dataset=params["dataset_type"]
            )
            for k, v in prod_stats_mods.items():
                if k not in targets:
                    continue
                for example in examples[k]:
                    year = example['sent'].year
                    era_slice = year_to_coarse_slice(year, params['dataset_type'])
                    if era_slice:
                        example["mod_productivity"] = prod_stats_mods[k][era_slice]

    return examples



def load_vecs_cache(
    cached_examples_dirs: List[str],
    targets: List[str],

):

    filtered_lookup = {}
    for cached_examples_dir in cached_examples_dirs:
        cached_examples_files = [os.path.join(cached_examples_dir, f) for f in os.listdir(cached_examples_dir) if
                                 f.endswith(".pickle")]
        for cached_examples_file in cached_examples_files:
            basename = os.path.basename(cached_examples_file)
            # logging.info(f"basename: {basename}")
            filename_as_key = basename.replace(SPACE_REPLACEMENT_IN_FILENAMES, " ").split(".pickle")[0]
            # logging.info(f"target: {filename_as_key}")
            if filename_as_key not in targets:
                continue
            logging.info(f"loading cache for {cached_examples_file}")
            with open(cached_examples_file, 'rb') as in_cache:
                # potentially there are several lists
                try:
                    while True:
                        # will throw EOF when there are no more lists to load
                        cached_lookup_slice = pickle.load(in_cache)
                        if filename_as_key not in filtered_lookup:
                            filtered_lookup[filename_as_key] = []
                        filtered_lookup[filename_as_key] += cached_lookup_slice
                except EOFError:
                    pass

    return filtered_lookup

def load_random_vecs_cache(
        directories: str,
        targets: List[str],

):
    lookup = {}
    for cached_examples_dir in directories:
        cached_examples_files = [os.path.join(cached_examples_dir, f) for f in os.listdir(cached_examples_dir) if
                                 f.endswith(".pickle")]
        for cached_examples_file in cached_examples_files:
            basename = os.path.basename(cached_examples_file)
            filename_as_key = basename.replace(SPACE_REPLACEMENT_IN_FILENAMES, " ").split(".pickle")[0]
            if filename_as_key not in targets:
                continue
            logging.info(f"loading cache for {cached_examples_file}")
            with open(cached_examples_file, 'rb') as in_cache:
                lookup[filename_as_key] = pickle.load(in_cache)
    return lookup


def find_and_cache_examples():
    pass

def load_cached_examples():
    pass


# NOTE: these layer wrangling functions could get moved to another file









def cos_sim(x, y):
    x = x.astype("float64")
    y = y.astype("float64")
    return np.dot(x, y) / (np.sqrt(np.dot(x, x)) * np.sqrt(np.dot(y, y)))




def cluster_tf_idf_features(sents: List[Sentence], cluster_labels: np.ndarray, lang_code: str) -> List[List[str]]:
    assert len(sents) == len(cluster_labels)
    cluster_to_sents = defaultdict(list)
    for i, sent in enumerate(sents):
        cluster_to_sents[cluster_labels[i]].append(sent)
    output = []
    for k in sorted(list(cluster_to_sents.keys())):
        features = [(v, k) for k, v in tf_idf_features(cluster_to_sents[k], lang_code).items()]
        # sort based on score, take top 10, then remove the scores
        top_10 = sorted(features, reverse=True)[:10]
        output.append([t[1] for t in top_10])
    return output


def print_sents_in_clusters(sents: List[Sentence], cluster_labels: np.ndarray, lang_code) -> str:
    assert len(sents) == len(cluster_labels)
    out = []
    all_k = sorted(list(set([k for k in cluster_labels])))
    for k in all_k:
        cluster_out = [f"Cluster {k}:"]
        this_cluster = []
        for i, sent in enumerate(sents):
            if cluster_labels[i] == k:
                cluster_out.append(f"YEAR: {sent.year}; SENT: {sent}")
                this_cluster.append(sent)
        out.append(cluster_out)
        # print(f"features: {sorted(tf_idf_features(this_cluster, lang_code), key=lambda x: x[1], reverse=True)[:5]}")
        # print("\n")
    return out


def cluster_era_analysis(
        sents: List[Sentence],
        cluster_labels: np.ndarray,
        dataset_type: Literal["COHA", "DTA"],
):
    """

    :param sents:
    :param cluster_labels:
    :param era_early:
    :param era_late:
    :return:
    """
    # there's a few things we could be doing here...
    #  1.) for each cluster, get % of instances belonging to each era
    #  2.) between each pair of clusters, look at the aggregated difference in year
    #  3.) (where possible) find the centroid of the cluster, compare differences in era
    #      for items close/far from the center. Asking "is there a relationship between
    #      core vs. marginal membership in cluster and membership in era E.
    sents_by_cluster = defaultdict(list)
    for sent, k in zip(sents, cluster_labels):
        sents_by_cluster[k].append(sent)

    # idea 1:
    all_k = sorted(list(set([k for k in cluster_labels])))
    for k in all_k:
        counts = defaultdict(int)
        for sent in sents_by_cluster[k]:
            if year_to_coarse_slice(sent.year, dataset_type) == COARSE_TIME_SLICES[dataset_type][0]:
                counts['early'] += 1
            elif year_to_coarse_slice(sent.year, dataset_type) == COARSE_TIME_SLICES[dataset_type][-1]:
                counts['late'] += 1
            counts['total'] += 1
        print(f"cluster {k}: \n\tearly: {counts['early'] / counts['total'] if counts['total'] else 0}"
              f"\n\tlate: {counts['late'] / counts['total'] if counts['total'] else 0}")
    # cluster pairs analysis:
    for k1, k2 in combinations(all_k, 2):
        k1_avg_year, k2_avg_year = 0, 0
        for sent_k1 in sents_by_cluster[k1]:
            k1_avg_year += sent_k1.year
        k1_avg_year = k1_avg_year / len(sents_by_cluster[k1]) if sents_by_cluster[k1] else 0
        for sent_k2 in sents_by_cluster[k2]:
            k2_avg_year += sent_k2.year
        k2_avg_year = k2_avg_year / len(sents_by_cluster[k2]) if sents_by_cluster[k2] else 0
        print(f"clusters {k1} and {k2} avg year gap: {abs(k2_avg_year - k1_avg_year)}")


def _set_seed(seed: int) -> None:
        torch.manual_seed(seed)
        transformers.set_seed(seed)
        np.random.seed(seed)
        random.seed(seed)


def _expand_targets(
        targets,
        include_heads: bool,
        include_mods: bool
) -> List[str]:
    expanded_targets = []
    if include_heads:
        for target in targets:
            expanded_targets += target.expand_to_list("head")
    if include_mods:
        for target in targets:
            expanded_targets += target.expand_to_list("mod")
    if not include_mods and not include_heads:
        for target in targets:
            expanded_targets.append(target.primary_component)
    return list(set(expanded_targets))

if __name__ == "__main__":
    main()

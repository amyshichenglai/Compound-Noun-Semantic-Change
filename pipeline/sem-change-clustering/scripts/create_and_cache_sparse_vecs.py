# adapted from https://github.com/pippokill/tri/
from typing import List, Dict, Set, Tuple
import random # which should have its seed set somewhere else!
import numpy as np
import itertools
try:
    from nltk.corpus import stopwords
except ImportError:
    stopwords = None

from data import Sentence, process_tokenized_sent, MAX_SEQ_LENGTH, CorpusData
from util import find_in_sent, init_keyword_processor, find_in_sent_kw_processor
from constants import (
    SECOND_ORDER_VECS_C, BERT_VECS_C, LEMMA_SPAN_C
)
try:
    from flashtext import KeywordProcessor
except ImportError:
    KeywordProcessor = None
from tqdm import tqdm, trange

from argparse import ArgumentParser
import pickle
import os
import sys
import logging
from collections import defaultdict
from transformers import AutoTokenizer
import yaml
from test_items_IO import TestCompounds, TestWords, TestRelatedCompounds


CONTEXT_WINDOW_SIZE = 5 # this needs to be an odd number
# CONTEXT_WINDOW_SIZE = 15
DEFAULT_DIMENSION = 1000
# DEFAULT_DIMENSION = 10 #*** just for debugging
DEFAULT_NONZERO_LEN = 20
# DEFAULT_NONZERO_LEN = 2 #*** also for debugging

FIRST_ORDER_CONTEXTS_LEN = 1500

SPACE_REPLACEMENT_IN_FILENAMES = "🔥"

D_TYPE = np.float64


random.seed(2666)


ISO_639_1_to_txt = {
    "en": "english",
    "de": "german",
}
# STOPWORDS = set(stopwords.words("german"))
logger = logging.getLogger(__name__)

DEBUG = True
# debug counts:
USED_BACKOFF_REPR = 0
USED_REGULAR_REPR = 0

def main():



    parser = ArgumentParser()
    parser.add_argument("--cached_per_target_dir")
    parser.add_argument("--test_file", required=True)
    parser.add_argument("--test_file_type", choices=['cordeiro', 'ghost', 'semeval'], required=True)
    parser.add_argument("--related_compounds_file_heads", help=".tsv file with compounds sharing one constituent")
    parser.add_argument("--related_compounds_file_mods")
    parser.add_argument("--lang", required=True, choices=['en', 'de'])
    parser.add_argument("--mode", required=True, choices=["1st-order", "2nd-order"], help=""
                        "what kind of representations to create: 1st order or 2nd order "
                        "contextualized vectors (using random indexing as the base)")
    parser.add_argument("--tokenizer")
    parser.add_argument("--cached_data")
    parser.add_argument("--output_dir")
    args = parser.parse_args()

    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                        datefmt="%d/%m/%Y %H:%M:%S",
                        level=logging.INFO,
                        handlers=[logging.StreamHandler(sys.stdout)]
    )

    test_items = []
    lang = None
    if args.test_file_type == "cordeiro":
        test_items = TestCompounds.load_cordeiro(args.test_file)
        lang = 'en'
    elif args.test_file_type == "ghost":
        test_items = TestCompounds.load_ghost(args.test_file)
        lang = 'de'
    elif args.test_file_type == "semeval":
        test_items = TestWords.load_semeval2020_list(args.test_file)

    # TODO: maybe this 'expanding targets to strings'
    #       process has become too complicated to be
    #       repeating in various places...
    targets = []
    if args.related_compounds_file_heads:
        test_related_items = TestRelatedCompounds.load(
            comp_rated_compounds=test_items.to_list(),
            related_compounds_filename=args.related_compounds_file_heads,
            constituent_type="head",
            lang=lang,
        )
        targets += [it for it in test_related_items.to_list()]
        # the case that we are running both
    if args.related_compounds_file_mods:
        constituent_type = "mod"
        test_related_items = TestRelatedCompounds.load(
            comp_rated_compounds=test_items.to_list(),
            related_compounds_filename=args.related_compounds_file_mods,
            constituent_type="mod",
            lang=lang,
        )
        targets += [it for it in test_related_items.to_list()]

    if not targets:
        targets = [it for it in test_items.to_list()]


    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    sparse_vec_lookup = {}  # actually we need the tokenizer here!
    # TODO: set the rng seed
    for i in range(len(tokenizer)):
        sparse_vec_lookup[tokenizer.decode(i)] = generate_sparse_ternary_vec()

    cached_examples_files = [os.path.join(args.cached_per_target_dir, f) for f in os.listdir(args.cached_per_target_dir) if
                             f.endswith(".pickle")]

    if args.mode == "1st-order":
        generate_first_order_representations(
            args=args,
            cached_examples_files=cached_examples_files,
            all_targets=targets,
            sparse_vec_lookup=sparse_vec_lookup,
            tokenizer=tokenizer,
        )

    if args.mode == "2nd-order":
        contexts = get_first_order_contexts(cached_examples_files=cached_examples_files, all_targets=targets)
        top_contexts = filter_first_order_contexts(contexts, args.lang)
        with open(f"{args.output_dir}/_contexts_used.log", 'w', encoding='utf-8') as log_out:
            for c in top_contexts:
                log_out.write(f"{c}\n")
        second_order_representations = aggregate_second_order_contexts(
            context_targets=top_contexts,
            cached_data=args.cached_data,
            tokenizer=tokenizer,
            random_idx_vec_lookup=sparse_vec_lookup,
        )

        generate_second_order_representations(
            args,
            cached_examples_files=cached_examples_files,
            all_targets=targets,
            second_order_repr_lookup=second_order_representations,
            base_lookup=sparse_vec_lookup,
            tokenizer=tokenizer,
        )
        if DEBUG:
            logging.info(f"Total representations created: {USED_BACKOFF_REPR + USED_REGULAR_REPR}\n"
                         f"Regular: {USED_REGULAR_REPR}\n"
                         f"Using the backoff representation: {USED_BACKOFF_REPR}")


def generate_second_order_representations(
        args,
        cached_examples_files,
        all_targets,
        second_order_repr_lookup,
        base_lookup,
        tokenizer,
):
    cached_file_itr = tqdm(cached_examples_files, desc="examples_files")
    for cached_examples_file in cached_file_itr:
        basename = os.path.basename(cached_examples_file)
        # logging.info(f"basename: {basename}")
        filename_as_key = basename.replace(SPACE_REPLACEMENT_IN_FILENAMES, " ").split(".pickle")[0]
        all_targets.append(filename_as_key)
        # no need to filter anything, we are just processing each target in the dir
        with open(cached_examples_file, 'rb') as in_cache:
            # potentially there are several lists
            try:
                while True:
                    # will throw EOF when there are no more lists to load
                    cached_lookup_slice = pickle.load(in_cache)
                    for example in cached_lookup_slice:
                        vec = create_representation_from_second_order_context(
                            sent=example['sent'],
                            target_lemma_span=example[LEMMA_SPAN_C],
                            lemma_to_2nd_order_representation=second_order_repr_lookup,
                            sparse_vec_lookup=base_lookup,
                            tokenizer=tokenizer,
                        )
                        example[SECOND_ORDER_VECS_C] = vec
                    with open(os.path.join(args.output_dir, basename), 'ab') as out_f:
                        pickle.dump(cached_lookup_slice, out_f, pickle.HIGHEST_PROTOCOL)

            except EOFError:
                pass
def generate_first_order_representations(args, cached_examples_files, all_targets, sparse_vec_lookup, tokenizer):
    for cached_examples_file in cached_examples_files:
        basename = os.path.basename(cached_examples_file)
        # logging.info(f"basename: {basename}")
        filename_as_key = basename.replace(SPACE_REPLACEMENT_IN_FILENAMES, " ").split(".pickle")[0]
        all_targets.append(filename_as_key)
        # no need to filter anything, we are just processing each target in the dir
        with open(cached_examples_file, 'rb') as in_cache:
            # potentially there are several lists
            try:
                while True:
                    # will throw EOF when there are no more lists to load
                    cached_lookup_slice = pickle.load(in_cache)
                    # just extract the Sentence objects
                    #target_to_context_sents[filename_as_key] += [t['sent'] for t in cached_lookup_slice]
                    #target_to_context_spans[filename_as_key] += [t['lemma-span'] for t in cached_lookup_slice]

                    #target_cache[filename_as_key] += [t for t in cached_lookup_slice]
                    for example in cached_lookup_slice:
                        vec = get_random_index_vec(
                            sparse_vec_lookup=sparse_vec_lookup,
                            sent=example['sent'],
                            target_lemma_span=example['lemma-span'],
                            tokenizer=tokenizer,
                        )  # modify the target_cache -- insert this entry
                        example['random-idx'] = vec
                    with open(os.path.join(args.output_dir, basename), 'ab') as out_f:
                        pickle.dump(cached_lookup_slice, out_f, pickle.HIGHEST_PROTOCOL)

            except EOFError:
                pass






    # if we would rather construct them on the fly, we would only need to cache this part
    with open(os.path.join(args.cached_per_target_dir, "__BASE_SPARSE_VECTORS" + ".pickle"), 'bw') as out_f:
        pickle.dump(sparse_vec_lookup, out_f, pickle.HIGHEST_PROTOCOL)
    # check cosine dist of resulting vecs
    # gold_mine_mine = scipy.spatial.distance.cosine(context_vecs['gold mine'], context_vecs['mine'])
    # gold_mine_tree = scipy.spatial.distance.cosine(context_vecs['gold mine'], context_vecs['tree'])
    # print('done')


# the reference implementaiton used an array of short integers pointing to the
# index of an (otherwize all zero) array containing a +1 or -1 entry
# this saves on space for sure, but our other representations are
# already pretty profligate as far as storing zillion-dimension vectors
# is concerned. Here we will (happily, or not) write out the zeros.

# this is the 'phase 1' so to speak of pg 63 of 22.14 (Basile et al 2015)
def generate_sparse_ternary_vec(dimension: int=DEFAULT_DIMENSION, nonzero_length: int=DEFAULT_NONZERO_LEN) -> np.ndarray:
    arr = np.zeros(dimension, dtype=D_TYPE)

    nonzero_count = 0
    while nonzero_count < nonzero_length / 2:
        idx = random.randint(0, dimension - 1)
        if not arr[idx]: # if it's zero
            arr[idx] = 1
            nonzero_count += 1
    while nonzero_count < nonzero_length:
        idx = random.randint(0, dimension - 1)
        if not arr[idx]:
            arr[idx] = -1
            nonzero_count += 1
    return arr


def get_random_index_vec(
        sent: Sentence,
        target_lemma_span: Tuple[int, int],
        sparse_vec_lookup: Dict[str, np.ndarray],
        tokenizer,
):
    term_tokenized = [tokenizer.decode(t) for t in tokenizer(' '.join(sent.lemmas[target_lemma_span[0]: target_lemma_span[1]]))['input_ids'][1:-1]]
    contextualized_representation = sum([np.array(sparse_vec_lookup[t], dtype=D_TYPE) for t in term_tokenized])  # deep copy!

    context_set: Set[str] = set()
    wp_tokens, token_offsets = process_tokenized_sent(sent.lemmas, tokenizer)
    window_left = target_lemma_span[0] - CONTEXT_WINDOW_SIZE
    if window_left < 0:
        window_left = 0
    window_right = target_lemma_span[1] + CONTEXT_WINDOW_SIZE
    if window_right > len(sent.lemmas):
        window_right = len(sent.lemmas)
    left_range_token_offsets = (window_left, target_lemma_span[0])  # exclusive
    context_tokens = [wp_tokens[token_offsets[o][0]: token_offsets[o][1]] for o in
                      range(left_range_token_offsets[0], left_range_token_offsets[1])]
    right_range_token_offsets = (target_lemma_span[1], window_right)
    context_tokens += [wp_tokens[token_offsets[o][0]: token_offsets[o][1]] for o in
                       range(right_range_token_offsets[0], right_range_token_offsets[1])]
    context_set.update(itertools.chain.from_iterable(context_tokens))  # flatten list of lists
    for context in context_set:
        contextualized_representation += sparse_vec_lookup[context]
    if context_set:
        return contextualized_representation / len(context_set)
    return contextualized_representation

def get_left_and_right_context(
        sent: Sentence,
        target_lemma_span: Tuple[int, int]
) -> Tuple[List[str], List[str]]:
    window_left = target_lemma_span[0] - CONTEXT_WINDOW_SIZE
    if window_left < 0:
        window_left = 0
    window_right = target_lemma_span[1] + CONTEXT_WINDOW_SIZE
    if window_right > len(sent.lemmas):
        window_right = len(sent.lemmas)

    return sent.lemmas[window_left: target_lemma_span[0]], sent.lemmas[target_lemma_span[1]: window_right]


def get_first_order_contexts(
    cached_examples_files,
    all_targets,

) -> Dict[str, Dict[str, int]]:
    first_order_contexts = {}
    if hasattr(all_targets[0], 'compound'):
        all_compounds = [t.compound for t in all_targets]
    elif hasattr(all_targets[0], "compounds"):
        all_compounds = [c for t in all_targets for c in t.expand_to_list()]
    else:
        all_compounds = [t for t in all_targets] # 'compounds' being a misnomer here (semeval case)
    EXAMPLES_USED = 0
    for cached_examples_file in cached_examples_files:
        basename = os.path.basename(cached_examples_file)
        # logging.info(f"basename: {basename}")
        filename_as_key = basename.replace(SPACE_REPLACEMENT_IN_FILENAMES, " ").split(".pickle")[0]
        #*** DEBUG: for now let's only use examples that are compounds
        if filename_as_key not in all_compounds:
            continue
        # if EXAMPLES_USED > 20:
        #     break
        all_targets.append(filename_as_key)
        # no need to filter anything, we are just processing each target in the dir
        first_order_contexts[filename_as_key] = defaultdict(int)
        with open(cached_examples_file, 'rb') as in_cache:
            # potentially there are several lists
            try:
                while True:
                    # will throw EOF when there are no more lists to load
                    cached_lookup_slice = pickle.load(in_cache)
                    # just extract the Sentence objects
                    #target_to_context_sents[filename_as_key] += [t['sent'] for t in cached_lookup_slice]
                    #target_to_context_spans[filename_as_key] += [t['lemma-span'] for t in cached_lookup_slice]

                    #target_cache[filename_as_key] += [t for t in cached_lookup_slice]
                    for example in cached_lookup_slice:
                        l_context, r_context = get_left_and_right_context(
                            sent=example['sent'],
                            target_lemma_span=example['lemma-span']
                        )
                        for left_lemma in l_context:
                            first_order_contexts[filename_as_key][left_lemma] += 1
                        for right_lemma in r_context:
                            first_order_contexts[filename_as_key][right_lemma] += 1
            except EOFError:
                pass
        # EXAMPLES_USED += 1
    return first_order_contexts

def filter_first_order_contexts(contexts: Dict[str, Dict[str, int]], lang: str):
    total_targets = len(contexts)

    if stopwords is None:
        logger.warning("nltk stopwords are unavailable; continuing without stopword filtering")
        stop_words_set = set()
    else:
        stop_words_set = set(stopwords.words(ISO_639_1_to_txt[lang]))

    all_contexts = set()
    for target, context_dict in contexts.items():
        all_contexts.update(context_dict.keys())

    # create a contingency table with entries with all combinations of targets / contexts
    # (N++ -> instances (from this subcollection anyway) with both the target and the context token)
    # (N+- -> instances with the target but not the context token)
    # (N-+ -> instances with the context token but not the target)
    # (N-- -> instances with neither)

    total_contexts_per_target = defaultdict(int)
    overall_frequency = defaultdict(int)
    #{target: sum(contexts_d[target][c]) for target, contexts_d in contexts.items()
    #                            for c in contexts_d[target]}
    sum_over_everything = 0
    for target, contexts_d in contexts.items():
        for context_str, context_count in contexts_d.items():
            total_contexts_per_target[target] += context_count
            sum_over_everything += context_count


    chi_sq = {}
    for target in contexts:
        for current_context, current_context_count in contexts[target].items():
            # current_context_count is N++, current_context_count - total_contexts_per_target[target] is N+-
            n_plus_minus = total_contexts_per_target[target] - current_context_count
            n_minus_plus = sum([contexts[t2][current_context] for t2 in contexts if t2 != target]) # iterate thru all other targets, add each count for this same context
            n_minus_minus = sum_over_everything - current_context_count - n_plus_minus - n_minus_plus
            chi_sq[(target, current_context)] = \
                ((current_context_count * n_minus_minus) - (n_plus_minus * n_minus_plus)) ** 2 / \
                ( (current_context_count + n_plus_minus) * (n_minus_plus + n_minus_minus) * (current_context_count + n_minus_plus) * (n_plus_minus + n_minus_minus) )
            overall_frequency[current_context] += current_context_count


    max_first_order = {}
    for (target, context), score in chi_sq.items():
        if context not in max_first_order or score > max_first_order[context]:
            max_first_order[context] = score

    as_tuples = sorted([(k, v) for k, v in max_first_order.items()], key=lambda x: x[1], reverse=True)
    local_best_contexts = [c for c, s in as_tuples][:FIRST_ORDER_CONTEXTS_LEN]

    # remove stopwords, then rank words by frequency
    for s_word in stop_words_set:
        if s_word in overall_frequency:
            del overall_frequency[s_word]
    global_contexts = sorted([(k, v) for k, v in overall_frequency.items()], key=lambda x: x[1], reverse=True)
    global_contexts = [c for c, s in global_contexts][:FIRST_ORDER_CONTEXTS_LEN]


    return local_best_contexts + global_contexts

    # cutoff of how ubiquitous a target is...
    # could restrict on POS (would need to be collected earlier)
    # check schütze paper for other inspiration


def aggregate_second_order_contexts(
    context_targets: List[str],
    cached_data,
    tokenizer,
    random_idx_vec_lookup,
) -> Dict[str, np.ndarray]:

    kw_processor = None
    if KeywordProcessor is not None:
        kw_processor = KeywordProcessor()
        init_keyword_processor(kw_processor, context_targets)
    # ongoing aggregation (sum) of context word's 2nd order contexts
    context_to_aggregate_representation = {}
    # counter to be able to average each representation
    context_to_aggregate_count = defaultdict(int)

    data = CorpusData(cached_data=cached_data)
    # TOTAL_SENTS = 0
    batch_itr = tqdm(data.examples, desc="data_batch")
    for batch_i, batch in enumerate(batch_itr):
        sents = batch['sents']
        for i, sent in enumerate(sents):
            if kw_processor is not None:
                matches = find_in_sent_kw_processor(
                    sent=sent, keyword_processor=kw_processor, text_mode="lemma"
                )
            else:
                matches = []
                for context_target in context_targets:
                    found = find_in_sent(sent=sent, target=context_target, text_mode="lemma")
                    if not found:
                        continue
                    matches.extend((context_target, start, end) for start, end in found)
            if not matches:
                continue
            wp_tokens, token_offsets = process_tokenized_sent(sent.lemmas, tokenizer)
            for match in matches:
                target, start, end = match
                lemma_indices = sent.lemmas_ch_idx_to_lemma_idx((start, end))
                if lemma_indices[0] >= MAX_SEQ_LENGTH or lemma_indices[1] >= MAX_SEQ_LENGTH:
                    continue
                if lemma_indices == (0, 0):
                    print(f"offending sentence: {sent}\nmatch:{match}")
                    continue
                # get left and right window around match
                window_left = lemma_indices[0] - CONTEXT_WINDOW_SIZE
                if window_left < 0:
                    window_left = 0
                window_right = lemma_indices[1] + CONTEXT_WINDOW_SIZE
                if window_right > len(sent.lemmas):
                    window_right = len(sent.lemmas)
                left_range_token_offsets = (window_left, lemma_indices[0])  # exclusive
                context_tokens = [wp_tokens[token_offsets[o][0]: token_offsets[o][1]] for o in
                                  range(left_range_token_offsets[0], left_range_token_offsets[1])]
                right_range_token_offsets = (lemma_indices[1], window_right)
                context_tokens += [wp_tokens[token_offsets[o][0]: token_offsets[o][1]] for o in
                                   range(right_range_token_offsets[0], right_range_token_offsets[1])]
                context_tokens = itertools.chain.from_iterable(context_tokens)  # flatten list of lists
                for t in context_tokens:
                    if t not in random_idx_vec_lookup:
                        continue
                    if target not in context_to_aggregate_representation:
                        # make deep copy!!! very important
                        context_to_aggregate_representation[target] = random_idx_vec_lookup[t].copy()
                    else:
                        context_to_aggregate_representation[target] += random_idx_vec_lookup[t]
                    context_to_aggregate_count[target] += 1
            # TOTAL_SENTS += 1
        # if TOTAL_SENTS > 2000:
        #     break

    # this means passing over the full corpus

    # divide by number of vecs aggregated
    for c in context_targets:
        if c in context_to_aggregate_representation:
            context_to_aggregate_representation[c] /= context_to_aggregate_count[c]
    return context_to_aggregate_representation

def dump_2nd_order_representations():
    pass

def load_2nd_order_representations():
    pass

def create_representation_from_second_order_context(
    sent: Sentence,
    target_lemma_span: Tuple[int, int],
    lemma_to_2nd_order_representation: Dict[str, np.ndarray],
    sparse_vec_lookup, # only used by backoff repr
    tokenizer, # only used by backoff repr
) -> np.ndarray:
    global USED_BACKOFF_REPR, USED_REGULAR_REPR
    # get context window around target
    window_left = target_lemma_span[0] - CONTEXT_WINDOW_SIZE
    if window_left < 0:
        window_left = 0
    window_right = target_lemma_span[1] + CONTEXT_WINDOW_SIZE
    if window_right > len(sent.lemmas):
        window_right = len(sent.lemmas)
    first_contexts = [lem for lem in sent.lemmas[window_left: target_lemma_span[0]]] + \
        [lem for lem in sent.lemmas[target_lemma_span[1]: window_right]]
    contexts_summed = 0
    # look up    each lemma in 2nd order representations
    representation = np.zeros(DEFAULT_DIMENSION, dtype=D_TYPE)
    for context in first_contexts:
        if context in lemma_to_2nd_order_representation:
            representation += lemma_to_2nd_order_representation[context]
            contexts_summed += 1
    # add those together, divide by number added,
    if contexts_summed: # check for div by 0
        representation /= contexts_summed
    # backup representation in OOV kind of situation if the context window contains no previously
    # reckoned-with context words
    if not np.any(representation):
        representation = get_random_index_vec(
            sent, target_lemma_span, sparse_vec_lookup, tokenizer
        )
        if DEBUG:
            USED_BACKOFF_REPR += 1
    else:
        if DEBUG:
            USED_REGULAR_REPR += 1


    return representation


if __name__ == "__main__":
    main()

from argparse import ArgumentParser
from typing import Tuple, List, Literal
from collections import defaultdict
import pickle
import os
import re
import sys
from tqdm import tqdm, trange

from flashtext import KeywordProcessor

from cluster_eval import TestCompounds, TestWords
from test_items_IO import RelatedCompoundSet
from data import Sentence, CorpusData, MAX_SEQ_LENGTH, tf_idf_features
from util import (
    SPACE_REPLACEMENT_IN_FILENAMES, find_in_sent_kw_processor, init_keyword_processor,
    year_in_range, year_to_coarse_slice, COARSE_TIME_SLICES,
)

def main():
    parser = ArgumentParser()
    parser.add_argument("--cached_data")
    parser.add_argument("--test_file")
    parser.add_argument("--test_file_type", choices=['cordeiro', 'ghost', 'semeval'])
    parser.add_argument("--include_heads", action='store_true')
    parser.add_argument("--include_mods", action='store_true')
    parser.add_argument("--related_compounds_internal", action='store_true', help=""
                        "gather examples of related compounds using targets taken from "
                        "within the test file that share a constituent")
    parser.add_argument("--related_compounds_external", action='store_true', help=""
                        "gather examples of related compounds using baseline targets from "
                        "the test file, and search the corpus for compounds that share "
                        "a constituent")


    args = parser.parse_args()


    test_items = []
    if args.test_file_type == "cordeiro":
        test_items = TestCompounds.load_cordeiro(args.test_file)
    elif args.test_file_type == "ghost":
        test_items = TestCompounds.load_ghost(args.test_file)
    elif args.test_file_type == "semeval":
        test_items = TestWords.load_semeval2020_list(args.test_file)

    targets = [it for it in test_items.to_list()]

    if args.related_compounds_internal:
        related_sets = create_related_sets_from_targets(targets)
        for s in related_sets:
            print(f"{s}")
        exit(0)
    if args.related_compounds_external:
        if args.include_heads and not args.include_mods:
            target_constituent = "head"
        elif args.include_mods and not args.include_heads:
            target_constituent = "mod"
        else:
            raise ValueError("invalid combination of constituent selection + related compounds external mode")
        data = CorpusData(cached_data=args.cached_data)
        if args.test_file_type == "cordeiro":
            early_counts, late_counts = find_related_compounds_in_sent_en(targets=targets, data=data, constituent_type=target_constituent)
        elif args.test_file_type == "ghost":
            early_counts, late_counts = find_related_compounds_in_sent_de(targets=targets, data=data, constituent_type=target_constituent)
        else:
            early_counts, late_counts = None, None
        if early_counts:
            for constituent, inner_dict in early_counts.items():
                for related_compound, count_early in inner_dict.items():
                    late_count = late_counts[constituent][related_compound]
                    if late_count > 0:
                        print(f"{constituent}\t{related_compound}\t{count_early}\t{late_count}")


def create_related_sets_from_targets(targets: List[TestCompounds]) -> List[RelatedCompoundSet]:
    # once we have a set, we can populate the kw_processor
    # with them and go from there
    head_to_target = defaultdict(list)
    mod_to_target = defaultdict(list)
    for target in targets:
        mod_to_target[target.mod].append(target)
        head_to_target[target.head].append(target)
    output = []
    # we could filter sets that are not of length >= 3 
    # and then order them in ascending order of compositionality (s/t the first entry 
    # is the least compositional one)
    head_to_target_filtered = {h: sorted(targets, key=lambda x: x.mean_head_rating) 
                               for h, targets in head_to_target.items() if len(targets) >= 3}
    mod_to_target_filtered = {m: sorted(targets, key=lambda x: x.mean_mod_rating) 
                              for m, targets in mod_to_target.items() if len(targets) >= 3}
    for head, ls in head_to_target_filtered.items():
        output.append(RelatedCompoundSet([t.compound for t in ls], head))
    for mod, ls in mod_to_target_filtered.items():
        output.append(RelatedCompoundSet([t.compound for t in ls], mod))
    # could return these separately too...
    return output

def find_related_compounds_in_sent_en(
        targets: List[TestCompounds], 
        data: CorpusData, 
        constituent_type: Literal["mod", "head"],
):
    # in this case the kw processor isn´t going to be
    # very helpful because we don´t know the possible
    # targets in advance...

    # possibly this could be a preliminary step, just
    # searching the corpus for related terms and then
    # manually curating a list from there
    
    
    constituent_targets = [getattr(t, constituent_type) for t in targets]

    kw_processor = KeywordProcessor()
    init_keyword_processor(kw_processor, constituent_targets)

    constituent_to_compound_candidate_counts_early = defaultdict(lambda: defaultdict(int))
    constituent_to_compound_candidate_counts_late = defaultdict(lambda: defaultdict(int))
    early_era = COARSE_TIME_SLICES["COHA"][0]
    late_era = COARSE_TIME_SLICES["COHA"][-1]

    batch_itr = tqdm(data.examples, "data-batches")
    for batch_i, batch in enumerate(batch_itr):
        sents = batch['sents']
        for i, sent in enumerate(sents):
            era = year_to_coarse_slice(sent.year, "COHA")
            if not era:
                continue
            matches = find_in_sent_kw_processor(sent, keyword_processor=kw_processor)
            if matches:
                for match in matches:
                    target_constituent, start, end = match
                    lemma_indices = sent.lemmas_ch_idx_to_lemma_idx((start, end))
                    if lemma_indices[0] >= MAX_SEQ_LENGTH or lemma_indices[1] >= MAX_SEQ_LENGTH:
                        continue
                    if lemma_indices == (0, 0):
                        print(f"offending sentence: {sent}\nmatch:{match}")
                        continue
                    pos_tags = sent.tags[lemma_indices[0]: lemma_indices[1]]
                    if not all([t in ['NN'] for t in pos_tags]):
                        # due to COHA tagset reduction, NN is the only 'plain' noun tag
                        continue

                    ### English situation:
                    # now check if the prev / next token is also a noun
                    if constituent_type == "head":
                        # prev
                        if lemma_indices[0] - 1 < 0 or sent.tags[lemma_indices[0] - 1] != "NN":
                            continue
                        # check for non Noun before the possible modifier, after the constituent target
                        if lemma_indices[0] - 2 < 0 or sent.tags[lemma_indices[0] - 2] != "NN":
                            if era == early_era:
                                constituent_to_compound_candidate_counts_early[
                                    target_constituent][
                                        f"{sent.lemmas[lemma_indices[0] - 1]} {target_constituent}"] += 1
                            if era == late_era:
                                constituent_to_compound_candidate_counts_late[
                                    target_constituent][
                                        f"{sent.lemmas[lemma_indices[0] - 1]} {target_constituent}"] += 1
                    elif constituent_type == "mod":
                        # next
                        if lemma_indices[1] >= len(sent.lemmas) or sent.tags[lemma_indices[1]] != "NN":
                            continue
                        if lemma_indices[1] + 1 >= len(sent.lemmas) or sent.tags[lemma_indices[1] + 1] != "NN":
                            if era == early_era:
                                constituent_to_compound_candidate_counts_early[
                                    target_constituent
                                ][f"{target_constituent} {sent.lemmas[lemma_indices[1]]}"] += 1
                            if era == late_era:
                                constituent_to_compound_candidate_counts_late[
                                    target_constituent
                                ][f"{target_constituent} {sent.lemmas[lemma_indices[1]]}"] += 1
    
    return constituent_to_compound_candidate_counts_early, constituent_to_compound_candidate_counts_late


def find_related_compounds_in_sent_de(
    targets: List[TestCompounds], 
    data: CorpusData, 
    constituent_type: Literal["mod", "head"],
):
    constituent_targets = [getattr(t, constituent_type) for t in targets]
    if constituent_type == "head":
        # uncapitalize!
        constituent_targets = [t.lower() for t in constituent_targets]


    constituent_to_compound_candidate_counts_early = defaultdict(lambda: defaultdict(int))
    constituent_to_compound_candidate_counts_late = defaultdict(lambda: defaultdict(int))
    early_era = COARSE_TIME_SLICES["DTA"][0]
    late_era = COARSE_TIME_SLICES["DTA"][-1]

    if constituent_type == "head":
        precompiled_regexes = [_mod_of_head_regex(t) for t in constituent_targets]
    else:
        precompiled_regexes = [_head_of_mod_regex(t) for t in constituent_targets]

    batch_itr = tqdm(data.examples, "data-batches")
    for batch_i, batch in enumerate(batch_itr):
        sents = batch['sents']
        for i, sent in enumerate(sents):
            era = year_to_coarse_slice(sent.year, "DTA")
            if not era:
                continue
            sent_lemmas = sent.to_lemmas()
            # unfortunately we have to run a bunch of regexes
            for target_idx, regex in enumerate(precompiled_regexes):
                matches = regex.finditer(sent_lemmas)
                if matches:
                    for match in matches:
                        start, end = match.start(), match.end() # CHARACTER indices
                        lemma_indices = sent.lemmas_ch_idx_to_lemma_idx((start, end))
                        if lemma_indices[0] >= MAX_SEQ_LENGTH or lemma_indices[1] >= MAX_SEQ_LENGTH:
                            continue
                        if lemma_indices == (0, 0):
                            print(f"offending sentence: {sent}\nmatch:{match}")
                            continue
                        pos_tags = sent.tags[lemma_indices[0]: lemma_indices[1]]
                        if not all([t in ['NN'] for t in pos_tags]):
                            # due to COHA tagset reduction, NN is the only 'plain' noun tag
                            continue
                        # make sure it is only one lemmatized token
                        if lemma_indices[1] != lemma_indices[0] + 1:
                            continue
                        if era == early_era:
                            constituent_to_compound_candidate_counts_early[constituent_targets[target_idx]][sent.lemmas[lemma_indices[0]]] += 1
                        if era == late_era:
                            constituent_to_compound_candidate_counts_late[constituent_targets[target_idx]][sent.lemmas[lemma_indices[0]]] += 1
                        
    return constituent_to_compound_candidate_counts_early, constituent_to_compound_candidate_counts_late

def _head_of_mod_regex(german_mod: str):
    return re.compile(r"(?<!\S)" + german_mod + r"\w+")

def _mod_of_head_regex(german_head: str):
    return re.compile(r"\w+" + german_head.lower() + r"(?=\s|$)")

if __name__ == "__main__":
    main()
from argparse import ArgumentParser
from typing import Tuple, List, Literal
from collections import defaultdict
import pickle
import os
import sys
from tqdm import tqdm, trange

from flashtext import KeywordProcessor
from transformers import AutoModelForMaskedLM
import torch

from test_items_IO import TestCompounds, TestWords, TestRelatedCompounds
from data import Sentence, CorpusData, MAX_SEQ_LENGTH
from util import (
    SPACE_REPLACEMENT_IN_FILENAMES, find_in_sent_kw_processor, init_keyword_processor,
)


SLICE_NAME_TO_IDX = {
    "first": (1, 5),
    "mid": (5, -4),
    "last": (-4, 13),

}

def main():
    parser = ArgumentParser()
    parser.add_argument("--output_cache_dir")
    parser.add_argument("--cached_data")
    parser.add_argument("--model")
    parser.add_argument("--test_file")
    parser.add_argument("--test_file_type", choices=['cordeiro', 'ghost', 'semeval'])
    parser.add_argument("--related_compounds_file_heads", help=".tsv file with compounds sharing one constituent")
    parser.add_argument("--related_compounds_file_mods")
    parser.add_argument("--include_heads", action='store_true')
    parser.add_argument("--include_mods", action='store_true')
    parser.add_argument("--dry_run", action='store_true', help='just print the targets then exit')
    parser.add_argument("--device", type=str)


    args = parser.parse_args()

    if args.device == 'cpu':
        print("Running on cpu - please be sure this is desired", file=sys.stderr)
    else:
        # TODO: check if cuda is actually available
        print("running on cuda", file=sys.stderr)

    lang = None
    test_items = []
    hyphenated_to_compound = {}
    #hyphenated_to_compound = {t.hyphenated_compound: t.compound for t in targets if hasattr(t, 'hyphenated_compound')}
    if args.test_file_type == "cordeiro":
        test_items = TestCompounds.load_cordeiro(args.test_file)
        lang = 'en'
    elif args.test_file_type == "ghost":
        test_items = TestCompounds.load_ghost(args.test_file)
        lang = 'de'
    elif args.test_file_type == "semeval":
        test_items = TestWords.load_semeval2020_list(args.test_file)
    if args.dry_run:
        print(f"DEBUG: test items: {test_items.to_list()}")
    for test_it in test_items.to_list():
        hyphenated_to_compound[test_it.hyphenated_compound] = test_it.compound
    targets = [it for it in test_items.to_list()]
    target_strings = []
    if args.related_compounds_file_heads:
        test_related_items = TestRelatedCompounds.load(
            comp_rated_compounds=test_items.to_list(),
            related_compounds_filename=args.related_compounds_file_heads,
            constituent_type="head",
            lang=lang,
        )
        if args.dry_run:
            print(f"DEBUG: related items heads: {test_related_items.to_list()}")
        for item_set in test_related_items.to_list():
            for compound in item_set.expand_to_list():
                target_strings.append(compound)
    if args.related_compounds_file_mods:
        test_related_items = TestRelatedCompounds.load(
            comp_rated_compounds=test_items.to_list(),
            related_compounds_filename=args.related_compounds_file_mods,
            constituent_type="mod",
            lang=lang,
        )
        if args.dry_run:
            print(f"DEBUG: related items mods: {test_related_items.to_list()}")
        for item_set in test_related_items.to_list():
            for compound in item_set.expand_to_list():
                target_strings.append(compound)

    # TODO: would be nice to also have a mapping for hyphenated related compounds......
    if args.dry_run:
        print(f"DEBUG: hyphenated compound resolution dict: {hyphenated_to_compound}")

    expanded_targets = []
    if args.include_heads:
        for target in targets:
            expanded_targets += target.expand_to_list("head")
    if args.include_mods:
        for target in targets:
            expanded_targets += target.expand_to_list("mod")
    if args.dry_run:
        print(f"DEBUG: expanded targets: {expanded_targets}")
    target_strings += expanded_targets
    if not args.include_heads and not args.include_mods:
        targets_to_str = []
        for target in targets:
            targets_to_str.append(target.compound)
        target_strings += targets_to_str
    print(f"TARGET STRINGS: {target_strings}")
    target_strings = list(set(target_strings))

    if args.dry_run:
        print(f"DEBUG: dry run: list of targets {target_strings}")
        sys.exit(0)

    data = CorpusData(cached_data=args.cached_data)
    
    model = AutoModelForMaskedLM.from_pretrained(args.model)
    model.to(args.device)
    kw_processor = KeywordProcessor()
    init_keyword_processor(kw_processor, target_strings)


    current_lookup_size = 0
    lookup = defaultdict(list)
    batch_itr = tqdm(data.examples, "data-batches")
    for batch_i, batch in enumerate(batch_itr):
        if current_lookup_size > 3000 and args.output_cache_dir:
            for key in lookup:
                with open(os.path.join(args.output_cache_dir,
                                       f"{key.replace(' ', SPACE_REPLACEMENT_IN_FILENAMES)}.pickle"), 'ba') as out_cache:
                    pickle.dump(lookup[key], out_cache, protocol=pickle.HIGHEST_PROTOCOL)
            lookup = defaultdict(list)
            current_lookup_size = 0
        sents = batch['sents']
        for i, sent in enumerate(sents):
            lemma_matches = [(m, "lemmas") for m in find_in_sent_kw_processor(sent, text_mode="lemma", keyword_processor=kw_processor)]
            token_matches = [(m, "tokens") for m in find_in_sent_kw_processor(sent, text_mode="token", keyword_processor=kw_processor)]
            matches = lemma_matches + token_matches
            if matches:
                # need to go from char indices to token indices
                indices_matched = set()
                for match, text_mode in matches:
                    target, start, end = match
                    indices = sent._ch_idx_to_block_idx(text_mode, (start, end))
                    if indices[0] >= MAX_SEQ_LENGTH or indices[1] >= MAX_SEQ_LENGTH:
                        continue
                    if indices == (0, 0):
                        print(f"offending sentence: {sent}\nmatch:{match}")
                        continue
                    # Avoid double-counting things where the token
                    # and the lemma are identical.
                    if indices in indices_matched:
                        continue
                    indices_matched.add(indices)
                    pos_tags = sent.tags[indices[0]: indices[1]]
                    if not all([t in ['NN'] for t in pos_tags]):
                        # due to COHA tagset reduction, NN is the only 'plain' noun tag
                        continue
                    vec = get_vector_from_context(
                        model, {'input_ids': batch['input_ids'][i].unsqueeze(0).to(args.device)},
                        indices,
                        layer_slice_name='first',
                    )

                    # if the target is a hyphenated version, still store it with the regular
                    # version.
                    if target in hyphenated_to_compound:
                        target = hyphenated_to_compound[target]
                    # then need to get a contextualized embedding of that slice
                    # and now... we got a list of contextualized vectors for that target!
                    lookup[target].append(
                        {
                            'bert-vec': vec,
                            'sent': sent,
                            'lemma-span': indices,
                        }
                    )
                    current_lookup_size += 1
    for key in lookup:
        with open(os.path.join(args.output_cache_dir, f"{key.replace(' ', SPACE_REPLACEMENT_IN_FILENAMES)}.pickle"),
                  'ba') as out_cache:
            # maybe dump each compound in a separate file -- perhaps mapping spaces to 🔥 or something
            pickle.dump(lookup[key], out_cache, protocol=pickle.HIGHEST_PROTOCOL)


def get_vector_from_context(
        model,
        sentence,
        target_span: Tuple[int, int],
        layer_slice_name: str,
):
    """

    :param model:
    :param sentence: tokenized / vectorized representation of sentence, in a batch of 1
    :param target_span: EXCLUSIVE [) span of tokens within sentence whose representation we want to return
    :param layer_slice_name: first | mid | last
    :param device: cpu/cuda/etc specifier
    :return: a list of representations -- caller can decide how to combine them
    """
    # sentence should be on the order of
    # (1, max_seq_len, embedding_dim)
    if target_span[1] <= target_span[0]:
        raise ValueError(f"bad span designation: [{target_span[0]}, {target_span[1]})")
    if target_span[1] >= MAX_SEQ_LENGTH or target_span[0] < 0:
        raise ValueError("span is out of bounds")
    with torch.no_grad():
        # remove batch dimension of 1 after running
        encoded = model(**sentence, output_hidden_states=True)
        hidden_states = encoded.hidden_states
        sliced = _slice_hidden_states(hidden_states, layer_slice_name)
        #TODO stack these, instead of having a list?
        return encoded_layers_to_token_vector(sliced, target_span)

def _slice_hidden_states(hidden_states_tuple, slice_name):
    slice_start, slice_end = SLICE_NAME_TO_IDX[slice_name]
    return hidden_states_tuple[slice_start: slice_end]

def encoded_layers_to_token_vector(layers, token_span: Tuple[int, int]):
    """

    :param layers: range of hidden layers to use
    :param token_span: exclusive [) span of token indices
    :return: either a single embedding (token_span is None) or a List of them
    """
    # assume batch size is 1
    token_embeddings = []

    token_range = range(token_span[0], token_span[1])
    for token_idx in token_range:
        hidden_layers = []
        for layer_idx in range(len(layers)):
            vector = layers[layer_idx][0][token_idx]
            hidden_layers.append(vector)
        hidden_layers = torch.sum(torch.stack(hidden_layers), 0).reshape(1, -1).detach().cpu().numpy()
        token_embeddings.append(hidden_layers.squeeze())
    return token_embeddings


if __name__ == "__main__":
    main()

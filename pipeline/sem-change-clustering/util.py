from typing import Tuple, Optional, List, Literal, Sequence
import re
import random
import numpy as np
from collections import defaultdict

DEBUG = False

SPACE_REPLACEMENT_IN_FILENAMES = "🔥"

COARSE_TIME_SLICES = {
    "COHA": [
        (1830, 1850),
        # (1900, 1930),
        (1980, 2000),
    ],
    "DTA": [ # 1700 - 1813 / 1814 - 1900 were the big slices -
        (1700, 1750),
        #(1790, 1840),
        (1870, 1900),
    ],
    "DTA_ALL": [
        (1000, 1690),
        (1700, 1720),
        (1730, 1750),
        (1760, 1780),
        (1790, 1810),
        (1820, 1840),
        (1850, 1870),
        (1880, 1900),
        (1910, 2020),
    ]
}


def _year_in_range(year: int, span: Tuple[int, int]) -> bool:
    """raises ValueError if range isn't increasing, range should be inclusive []"""
    if span[1] < span[0]:
        raise ValueError("end of range is less than start of range")
    if span[0] < 0 or span[1] < 0:
        raise ValueError("we really aren't dealing with BCE years, so check the inputs!")
    if span[0] <= year <= span[1]:
        return True
    return False

def year_to_coarse_slice(year: int, dataset: Literal["DTA", "COHA"]) -> Tuple[int, int]:
    """Assumes the use of time slices in the decade e.g. '1950' standing for years 1950 - 1959"""
    if year < COARSE_TIME_SLICES[dataset][0][0] or year > COARSE_TIME_SLICES[dataset][-1][1] + 9:
        return None
        raise ValueError(f"year {year} out of range of possible coarse slices")
    for slice in COARSE_TIME_SLICES[dataset]:
        if _year_in_range(year, (slice[0], slice[1] + 9)):
            return slice


def find_in_sent_kw_processor(
        sent: "Sentence",
        keyword_processor,
        text_mode: Literal["lemma", "token"]="lemma") -> List[Tuple[str, int, int]]:
    if text_mode == "lemma":
        elts = sent.to_lemmas()
    elif text_mode == "token":
        elts = sent.to_tokens()
    else:
        raise ValueError("invalid text mode")
    return keyword_processor.extract_keywords(elts, span_info=True)


def find_in_sent(sent: "Sentence", target: str, text_mode: Literal["lemma", "token"]="lemma") -> Optional[List[Tuple[int, int]]]:
    """

    :param sent:
    :param target:
    :param text_mode:
    :return: None if target does not occur in sent, otherwise, List of (int, int) pairs
    indicating the token/lemma indices in the sentence where the target occurs.
    """
    # first off, what does our target look like tokenized:
    #target_preprocessed = process_line(target, args)
    # target_encoded = tokenizer.encode(target)[1:-1]
    # so this particular sequence is what we are looking for in any given
    # eval data entry


    target = ' '.join(s for s in target.split())
    if text_mode == "lemma":
        elts = sent.to_lemmas()
    elif text_mode == "token":
        elts = sent.to_tokens()
    else:
        raise ValueError("invalid text mode")
    if target not in elts:
        return None

    # REALLY what we should do here is perform a raw match
    # and *then* check that the resulting spans are surrounded either by
    # spaces or ^/$ (start or end of the sentence)

    # regex = r'(?:\s+|^)' + target + r'(?:\s+|$)'

    matches = re.finditer(target, elts)
    spans = [m.span() for m in matches] # this is including spaces around the target
    spans_filtered = []
    for span in spans:
        if span[0] and not elts[span[0] - 1].isspace():
            continue
        if span[1] < len(target) and not elts[span[1] - 1].isspace():
            continue
        spans_filtered.append(span)
    if not spans_filtered:
        return None
    return spans_filtered



class TimeStratifiedSampling:
    """Ideally this would be more agnostic to the exact type of thing contained in it"""
    def __init__(self, timeslice_to_all_keys_to_counts, all_examples_dict, dataset_type):
        """
        :timeslice_to_all_keys_to_counts: (ideally sorted) lookup of all possible examples to their total number of examples
                             this is needed to keep the random shuffles synchronized for any given target
                             regardless of which particular set of targets is being used
        :all_examples_dict: str to examples list lookup -- will be shuffled and made available here
        :dataset_type: corpus type, used to look up time slices
        """
        by_time_buckets = {}
        self.samplers = {}
        for time_slice in timeslice_to_all_keys_to_counts:
            by_time_buckets[time_slice] = defaultdict(list)
            self.samplers[time_slice] = {}
        for key, examples in all_examples_dict.items():
            for cluster_item in examples:
                coarse_slice = year_to_coarse_slice(cluster_item.sent.year, dataset_type)
                if coarse_slice is None:
                    continue
                by_time_buckets[coarse_slice][key].append(cluster_item)
        for time_slice in timeslice_to_all_keys_to_counts:
            for target_name, count in timeslice_to_all_keys_to_counts[time_slice].items():
                if target_name in by_time_buckets[time_slice]:
                    self.samplers[time_slice][target_name] = SamplerWithoutReplacement(by_time_buckets[time_slice][target_name])
                elif count:
                    # advance the random number generator as if we were shuffling this example
                    _ = random.sample(range(count), k=count)
        # print(f"debug:\n samplers dict: {self.samplers}")

    def get_sample(self, time_slice, item_key):
        if time_slice not in self.samplers:
            if DEBUG:
                print(f"{time_slice} not found")
            return None
        if item_key not in self.samplers[time_slice]:
            if DEBUG:
                print(f"{item_key} not found")
            return None
        # print(f"debug: {self.samplers[time_slice][item_key].items[self.samplers[time_slice][item_key].next_sample_idx]}")
        return self.samplers[time_slice][item_key].get_item()

    def get_samples(self, time_slice, item_key, k: int):
        if k <= 0:
            return []
        if time_slice not in self.samplers:
            if DEBUG:
                print(f"{time_slice} not found")
            return []
        if item_key not in self.samplers[time_slice]:
            if DEBUG:
                print(f"{item_key} not found")
            return []
        return self.samplers[time_slice][item_key].get_items(k)

    def num_samples(self, item_key, time_slice=None):
        if time_slice:
            return self.samplers[time_slice][item_key].num_items
        total = 0
        if not time_slice:
            for t, b in self.samplers.items():
                if item_key in b:
                    total += b[item_key].num_items
        return total

    @property
    def all_exhausted(self) -> bool:
        """returns true if all sample-able items are exhausted"""
        for timeslice, by_time_d in self.samplers.items():
            for k, sampler in by_time_d.items():
                if not sampler.is_exhausted:
                    return False
        return True

    def item_is_exhausted(self, item: str) -> bool:
        for _, by_time_d in self.samplers.items():
            if item in by_time_d and not by_time_d[item].is_exhausted:
                return False
        return True

    @property
    def total_possible_examples(self) -> int:
        total = 0
        for _, by_time_d in self.samplers.items():
            for _, sampler in by_time_d.items():
                total += sampler.num_items
        return total

    def minimum_time_slice_count(self, item: str) -> int:
        """return the smallest count of examples in any of the time
           slices"""
        smallest = None
        for _, by_time_d in self.samplers.items():
            if item not in by_time_d:
                return 0
            if smallest is None or by_time_d[item].num_items < smallest:
                smallest = by_time_d[item].num_items
        if smallest is None:
            raise ValueError("looking for smallest count in totally empty"
                             " lookup")
        return smallest

class SamplerWithoutReplacement:
    def __init__(self, items: Sequence):
        self.items = random.sample(items, k=len(items))
        self.next_sample_idx = 0
        self.num_items = len(items)

    def __iter__(self):
        pass

    def get_item(self):
        if self.next_sample_idx >= self.num_items:
            return None
        it = self.items[self.next_sample_idx]
        self.next_sample_idx += 1
        return it

    def get_items(self, k: int):
        output = [self.get_item() for _ in range(k)]
        return [e for e in output if e is not None]

    def __str__(self):
        return f"{self.num_items} items\n{[it['sent'] for it in self.items]}"

    def __repr__(self):
        return str(self)

    @property
    def is_exhausted(self) -> bool:
        """returns True if the sampler has returned all possible items, False otherwise"""
        return self.next_sample_idx >= self.num_items


def init_keyword_processor(kw_processor, keywords):
    kw_processor.add_non_word_boundary('-') # ascii hyphen (the 'hyphen minus')
    kw_processor.add_non_word_boundary('\u2010') # other hyphen
    kw_processor.add_non_word_boundary('\u2011') # non-breaking hyphen
    kw_processor.add_non_word_boundary('\u2012') # figure dash
    kw_processor.add_non_word_boundary('\u2013') # en dash
    kw_processor.add_non_word_boundary('\ufe63') # small hyphen-minus
    for keyword in keywords:
        kw_processor.add_keyword(keyword)



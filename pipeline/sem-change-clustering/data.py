import pickle
from typing import List, Tuple, Union, Dict, Optional, Literal
from dataclasses import dataclass
import zipfile
import json
from scripts.reduce_CCOHA_pos import reduce_tag as reduce_ccoha_tag
import os
import re
from collections import defaultdict
import numpy as np
from util import find_in_sent, SPACE_REPLACEMENT_IN_FILENAMES, COARSE_TIME_SLICES

try:
    from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
    import torch
except ImportError:
    DataLoader = RandomSampler = SequentialSampler = None
    Dataset = object
    torch = None

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
except ImportError:
    TfidfVectorizer = None

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    from nltk.corpus import stopwords
except ImportError:
    stopwords = None

ISO_639_1_to_txt = {
    "en": "english",
    "de": "german",
}
STOPWORDS = {
    code: stopwords.words(ISO_639_1_to_txt[code]) for code in ISO_639_1_to_txt
} if stopwords is not None else {}



MAX_SEQ_LENGTH = 128
BATCH_SIZE = 32

DOC_START_DTA = "<sod>" # used in DTA
DOC_START_COHA = "@@"
SENT_END = "<eos>" # used in CCOHA/DTA
END_SEQUENCE = "<no-seq>" # used in CCCOHA
REDACTED_TOKEN = "@" # used in CCOHA


def main():
    # *** earlier, this main() fn was hiding at the bottom of this file
    # -- the use case here is to perform initial caching of the raw data
    #    i.e. before domain adaptation gets run.
    #    but it can also be used to test the integrity of such a cached file
    #    by using the '--cached_inputs_file' arg here.

    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("inputs", nargs='*')
    parser.add_argument("--corpus", required=True, choices=["DTA", "COHA"])
    parser.add_argument("--tokenizer_name_or_config", type=str)
    parser.add_argument("--model_name_or_config", type=str)
    parser.add_argument("--use_lemmas", type=bool, default=True) # set by default because earlier it was the only option
    parser.add_argument("--uncased", action='store_true')
    parser.add_argument("--cached_inputs_file", required=True, type=str, help=""
                    "when provided alongside inputs, saves them to this file."
                    "\nWithout inputs, it tests loading the cached data from this file")
    args = parser.parse_args()

    if args.inputs:
        if args.cached_inputs_file and os.path.isfile(args.cached_inputs_file):
            raise ValueError(f"cached inputs file {args.cached_inputs_file} already exists")
        _cache_inputs(args)
    elif args.cached_inputs_file:
        _test_load_from_cache(args)
    else:
        raise ValueError("Invalid combination of args")

class Sentence:
    """non vectorized, just a sentence"""
    def __init__(self, tokens: List[str], lemmas: List[str], tags: List[str], year: int):
        self.tokens = tokens
        self.lemmas = lemmas
        self.tags = tags
        self.year = year
        # tokens, tags

    def decade(self):
        return self.year // 10 * 10


    def to_example(self, tokenizer):
        tokenized = tokenizer(self.to_lemmas(), padding='max_length', truncation=True, max_length=MAX_SEQ_LENGTH)
        return tokenized.data["input_ids"]

    def __str__(self):
        return " ".join(tok for tok in self.tokens)

    def to_tokens(self) -> str:
        return str(self)

    def to_lemmas(self) -> str:
        return " ".join(lem for lem in self.lemmas)

    def __repr__(self):
        return f"{self.year}: {' '.join(tok for tok in self.tokens)}"

    def lemmas_ch_idx_to_lemma_idx(self, start_end: Tuple[int, int]) -> Tuple[int, int]:
        return self._ch_idx_to_block_idx("lemmas", start_end)

    def tokens_ch_idx_to_token_idx(self, start_end: Tuple[int, int]) -> Tuple[int, int]:
        return self._ch_idx_to_block_idx("tokens", start_end)
    def _ch_idx_to_block_idx(self, attrib: Literal["tokens", "lemmas"], start_end: Tuple[int, int]) -> Tuple[int, int]:
        """
        :param start_end: start, end indices, from a string composed from
        joining each token with a single space between each ( exclusive indices [) )
        :return: range of token indices (into self.tokens) corresponding to the character
        indices. Partial overlap of a token includes it. Indices are exclusive [),
        so a range of [0:0] is returned by default.
        """
        start_ch, end_ch = start_end
        start_tok, end_tok = 0, 0
        current_ch_idx = 0
        total_len = len(" ".join(a for a in self.__getattribute__(attrib)))
        for i, tok in enumerate(self.__getattribute__(attrib)):
            # edge case: start
            if current_ch_idx == 0 and start_ch == 0:
                start_tok = i

            # otherwise:
            if current_ch_idx > end_ch:
                break
            if start_tok != 0 and end_tok != 0:
                break
            # include -1 before, +1 after to account for surrounding spaces
            if current_ch_idx - 1  <= start_ch <= current_ch_idx + len(tok) - 1:
                start_tok = i
            if current_ch_idx <= end_ch <= current_ch_idx + len(tok) + 1:
                end_tok = i + 1 # to make it an exclusive [) range!
            current_ch_idx += len(tok) + 1
        return start_tok, end_tok

    def to_durel(self, target_lemma: str, id_generator) -> List[str]:
        lemma_spans = find_in_sent(self, target=target_lemma, text_mode="lemma")
        if not lemma_spans:
            raise ValueError(f"target lemma: {target_lemma} not in sentence {self}")
        regular_tokens = []
        token_indices = []
        tokens_str = self.to_tokens()
        for span in lemma_spans:
            lemmas_span = self.lemmas_ch_idx_to_lemma_idx(span)
            regular_tokens.append(self.tokens[lemmas_span[0]: lemmas_span[1]])
            token_indices.append(lemmas_span)
        outputs = []
        # we need to get the regular character indices of the target
        # -- but the `_ch_idx_to_block_idx` fn has a space on either side...
        for token_start_end, tokens_ls in zip(token_indices, regular_tokens):
            string_repr = " ".join(t for t in tokens_ls)
            # get the character indices of the tokens involved here
            before_toks = self.tokens[0: token_start_end[0]]
            # accounts for each space after a preceding token
            start_idx = sum(len(tok) for tok in before_toks) + len(before_toks)
            target_span = (start_idx, start_idx + len(string_repr))
            tags = self.tags[token_start_end[0]: token_start_end[1]]
            if all(t == tags[0] for t in tags):
                tags = [tags[0]]
            outputs.append(f"{target_lemma}\t"
                           f"{' '.join(tag for tag in tags)}\t"
                           f"{self.year}\t \t"
                           f"{next(id_generator)}\t"
                           f" \t"
                           f"{tokens_str}\t"
                           f"{target_span[0]}:{target_span[1]}\t" # this span needs to be in terms of TOKENS
                           f"0:{len(tokens_str)}")
        return outputs

def tf_idf_features(sents: List[Sentence], lang_code: str) -> Dict[str, float]:
    if TfidfVectorizer is None:
        raise ImportError("scikit-learn is required for tf_idf_features")
    corpus = [sent.to_lemmas() for sent in sents]
    vectorizer = TfidfVectorizer(
        stop_words=STOPWORDS[lang_code],
        # min_df=5, # must occur in 5 sentences
        max_df=0.95, # can't occur in more than 95% of sentences
        max_features=1000, # top 500 most frequent terms in corpus considered
    )
    try:
        vectorizer.fit(corpus)
    except ValueError:
        # TODO: diagnose why this happens, log the fallback / return {} instead
        vectorizer = TfidfVectorizer(
            stop_words=STOPWORDS[lang_code],
            max_features=500,
        )
        vectorizer.fit(corpus)
    # vectorizer idf_ array (floats) has weights (higher => higher idf score)
    # corresponding to vocabulary_ dict (which has tf for each term)
    tf_idf_scores = {}
    for i, (term, term_idx) in enumerate(vectorizer.vocabulary_.items()):
        tf_idf_scores[term] = vectorizer.idf_[term_idx]
        # I guess it takes tf into account in there
    return tf_idf_scores

def pmi_rankings(sents: List[Sentence], lang_code: str):
    pass

def parse_3_column_corpus(corpus_files, corpus_type, doc_ids_seen) -> List[Sentence]:
    sents = []
    tokens, lemmas, tags = [], [], []
    year = None

    for corpus_filename in corpus_files:
        z_file = zipfile.ZipFile(corpus_filename)
        file_list = z_file.namelist()
        for filename in file_list:
            with z_file.open(filename) as in_f:
                skip_doc = False
                skip = False # used to skip past COHA @-redacted tokens
                for line in in_f:
                    line = line.decode('utf-8')
                    tok, lem, tag = line.strip().split('\t')
                    if tok == DOC_START_DTA:
                        year = int(tag)
                        if line.strip() in doc_ids_seen:
                            print(f"duplicate doc: {line.strip()}")
                            skip_doc = True
                        else:
                            skip_doc = False
                            doc_ids_seen.add(line.strip())
                    elif tok.startswith(DOC_START_COHA):
                        year = int(re.search(r"_\d\d\d\d_", filename).group().strip().replace("_", ""))
                        if line.strip() in doc_ids_seen:
                            skip_doc = True
                        else:
                            skip_doc = False
                            doc_ids_seen.add(line.strip())
                    elif skip_doc:
                        continue
                    elif tok == END_SEQUENCE:
                        pass # double check if there's always an <eos> here
                    elif tok == SENT_END:
                        if tokens:
                            sents.append(Sentence(
                                tokens, lemmas, tags, year
                            ))
                            tokens, lemmas, tags = [], [], []
                            if len(sents) == 1000:
                                yield sents
                                sents = []
                    elif tok == REDACTED_TOKEN and corpus_type == "COHA":
                        if not skip:
                            skip = True
                            tokens.append(END_SEQUENCE)
                            lemmas.append(END_SEQUENCE)
                            tags.append(END_SEQUENCE)
                        else:
                            continue
                    else:
                        skip = False
                        tokens.append(tok)
                        lemmas.append(lem)
                        if corpus_type == "COHA":
                            tag = reduce_ccoha_tag(tag)
                        tags.append(tag)
    if sents:
        yield sents

class CorpusFrequency:
    def __init__(self, unigrams, bigrams, corpus_type: Literal["COHA", "DTA"]):
        self.unigrams = unigrams
        self.bigrams = {}
        self.time_slices = COARSE_TIME_SLICES[corpus_type]
        self.corpus_type = corpus_type
        # need to save list of time slices here
        for key, freq_list in bigrams.items():
            self.bigrams[tuple(key.split(SPACE_REPLACEMENT_IN_FILENAMES))] = freq_list

    def unigram_freq(self, token: str, time_slice: Optional[Tuple[int, int]]=None) -> int:
        time_slice_index = self.time_slices.index(time_slice) if time_slice else None
        if time_slice_index is not None:
            if token in self.unigrams:
                return self.unigrams[token][time_slice_index]
        if token in self.unigrams:
            return sum(self.unigrams[token])
        return 0

    def unigram_freqs(self, token) -> List[int]:
        if token not in self.unigrams:
            return [0] * len(self.time_slices)
        return self.unigrams[token]

    def bigram_freq(self, query: Tuple[str, str], time_slice: Optional[Tuple[int, int]] = None) -> int:
        time_slice_index = self.time_slices.index(time_slice) if time_slice else None
        if time_slice_index is not None:
            if query in self.bigrams:
                try:
                    return self.bigrams[query][time_slice_index]
                except IndexError:
                    print(f"bad index: {time_slice_index} into {self.bigrams[query]}")
                    exit(1)
        if query in self.bigrams:
            return sum(self.bigrams[query])
        return 0

    def bigram_freqs(self, token1, token2) -> List[int]:
        query = (token1, token2)
        if query not in self.bigrams:
            return [0] * len(self.time_slices)
        return self.bigrams[query]

    @classmethod
    def from_file(cls, filename_prefix, corpus_type: Literal["DTA", "COHA"]):
        with open(f"{filename_prefix}_1_grams.json", 'r', encoding='utf-8') as in_f:
            unigrams = json.load(in_f)
        with open(f"{filename_prefix}_2_grams.json", 'r', encoding='utf-8') as in_f:
            bigrams = json.load(in_f)
        return cls(unigrams, bigrams, corpus_type)

    def unigram_keys(self) -> List[str]:
        return [k for k in self.unigrams.keys()]

    def bigram_keys(self) -> List[Tuple[str, str]]:
        return [k for k in self.bigrams.keys()]
    def total_unigram_count(
            self, time_slice: Optional[Tuple[int, int]]=None
    ) -> int:
        return sum(
            [self.unigram_freq(k, time_slice) for k in self.unigram_keys()]
        )

    def total_bigram_count(
            self,
            time_slice: Optional[Tuple[int, int]]=None
    ):
        return sum(
            [self.bigram_freq(k, time_slice) for k in self.bigram_keys()]
        )

    def unigram_time_dist(self):
        """Convert total per-timeslice unigram counts into a probability distribution"""
        return self._time_dist(self.total_unigram_count)

    def bigram_time_dist(self):
        return self._time_dist(self.total_bigram_count)

    def _time_dist(self, count_fn):
        per_timeslice_totals = []
        for time_slice in self.time_slices:
            per_timeslice_totals.append(count_fn(time_slice))
        as_array = np.array(per_timeslice_totals)
        return as_array / as_array.sum()

@dataclass
class InputFeatures:
    input_ids: List[int]
    attn_mask: List[int]
    token_offsets: List[Tuple[int, int]]
    sent: Sentence

class CorpusData(Dataset):
    """
    This time around we might want to keep the year/decade
    that sentences occurred in available longer-term
    """
    def __init__(self, *,
                 cached_data: str = None,
    ):
        self.examples = []
        if cached_data:
            with open(cached_data, 'rb') as in_f:
                try:
                    while True:
                        self.examples.append(pickle.load(in_f))
                except EOFError:
                    pass
        else:
            raise ValueError("Attempting to create CorpusData without an input file")


    def __getitem__(self, item):
        return self.examples[item]

    def __len__(self):
        return len(self.examples) # even though these are batches of BATCH_SIZE

    def _process_line(self, line, args):
        if args.uncased:
            line = line.lower()
        return line
    def make_dataloader_friendly(self):
        """This means: remove any data that isn't vectorized"""
        examples = []
        for batch in self.examples:
            del batch['sents']
            # TODO: it might be nice to be able to use these!
            del batch['token_offsets']
            for i in range(batch['input_ids'].shape[0]):
                examples.append({'input_ids': batch['input_ids'][i],
                                 'attn_mask': batch['attn_mask'][i]})
        self.examples = examples




def vectorize_features(batch: Union[List[InputFeatures], InputFeatures], output_name: Optional[str]=None) -> Optional[Dict]:
    if torch is None:
        raise ImportError("torch is required for vectorize_features")
    if type(batch) == InputFeatures:
        batch = [batch]

    # due to sub token fragmentation, token offsets could be smaller
    # than input ids overall, so we try to save a bit of memory as we
    # pad to the longest such sequence:

    # but these offsets need to be truncated! they can't refer to anything >= the
    # max_seq_length
    truncated_batch_of_offsets = []
    for feature in batch:

        truncated_offsets = []
        for offset in feature.token_offsets:
            if offset[0] < MAX_SEQ_LENGTH and offset[1] < MAX_SEQ_LENGTH:
                truncated_offsets.append(offset)
            else:
                break
        truncated_batch_of_offsets.append(truncated_offsets)

    num_original_tokens = [len(offsets) for offsets in truncated_batch_of_offsets]

    # TODO: double check that we actually need these padding tokens
    #       (obviously we need to remove the ones pointing at truncated tokens)
    longest_original_tokens = max(num_original_tokens)
    token_offsets = torch.tensor([
        offsets + [(0, 0)] * (longest_original_tokens - num_toks)
        for num_toks, offsets in zip(num_original_tokens, truncated_batch_of_offsets)
    ], dtype=torch.long)
    # with exclusive indexing, (0,0) yields []


    batch = {
        "input_ids": torch.tensor([feature.input_ids for feature in batch], dtype=torch.long),
        "attn_mask": torch.tensor([feature.attn_mask for feature in batch], dtype=torch.long),
        "sents": [feature.sent for feature in batch],
        "token_offsets": token_offsets
    }
    if output_name:
        with open(output_name, "ba") as out_f:
            pickle.dump(batch, out_f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        return batch



from contextlib import contextmanager
import zipfile
import gzip

@contextmanager
def get_file_handle(filename):
    if filename.endswith(".gz"):
        in_f = gzip.open(filename)
    elif filename.endswith(".zip"):
        z_file = zipfile.ZipFile(filename)
        file_list = z_file.namelist()
        if len(file_list) > 1:
            raise RuntimeError("Trying to read from a non-flattened .zip archive (i.e. that has more than one file in it)")
        in_f = z_file.open(file_list[0])
    else:
        in_f = open(filename, 'rb') # all the other options return bytes anyway
    try:
        yield in_f
    finally:
        in_f.close()

def convert_sentences_to_features(
        sents: List[Sentence],
        tokenizer,
        max_length: int,
        use_lemmas: bool,
        uncased: bool,

    ) -> List[InputFeatures]:

    # check if END_SEQUENCE is in tokenizer as a single unit
    end_seq_toks = tokenizer(END_SEQUENCE)['input_ids'][1:-1]
    # (removing cls and sep)
    if len(end_seq_toks) == 1:
        END_SEQUENCE_ID = end_seq_toks[0]
    else:
        END_SEQUENCE_ID = None


    output_features = []
    for sent_i, sent in enumerate(sents):
        tokens = sent.lemmas if use_lemmas else sent.tokens
        if uncased:
            tokens = [t.lower() for t in tokens]
        model_tokens, token_offsets = process_tokenized_sent(
            tokens=tokens,
            tokenizer=tokenizer
        )

        # add [CLS] and [SEP]
        # for BERT this is CLS at start, SEP at end

        # first TRUNCATE overly long input:
        model_tokens = model_tokens[:max_length - 2]  # room for CLS and SEP
        # label_ids = label_ids[:max_length - 2]

        model_tokens = [tokenizer.cls_token] + model_tokens + [tokenizer.sep_token]
        attn_mask = [1] * len(model_tokens)
        # mask out any END_SEQUENCE tokens (if applicable) (CCOHA corpus artifact)
        if END_SEQUENCE_ID is not None:
            # get mask for model tokens that are END_SEQUENCE_ID
            end_seq_mask = [-1 if tok == END_SEQUENCE_ID else 0 for tok in model_tokens]
            for i in range(len(attn_mask)):
                attn_mask[i] += end_seq_mask[i]

        # pad up to max length
        pad_amount = max_length - len(attn_mask)
        model_tokens += [tokenizer.pad_token] * pad_amount
        attn_mask += [0] * pad_amount

        # adjust token offsets by one for the CLS token at the start:
        token_offsets = [(s + 1, e + 1) for s, e in token_offsets]

        input_ids = tokenizer.convert_tokens_to_ids(model_tokens)

        output_features.append(
            InputFeatures(
                input_ids=input_ids,
                attn_mask=attn_mask,
                sent=sent,
                token_offsets=token_offsets,
            )
        )
    return output_features

def process_tokenized_sent(
        tokens: List[str],
        tokenizer
):
    out_tokens = []
    token_offsets = []

    for token in tokens:
        model_tokens = tokenizer(token)["input_ids"][1:-1]  # remove cls and sep
        if len(model_tokens) > 0:
            token_offsets.append((
                len(out_tokens), len(out_tokens) + len(model_tokens)  # exclusive indices [)
            ))
            out_tokens += [tokenizer.decode(model_token) for model_token in model_tokens]
        else:
            # in the event that the tokenizer turned something into nothing
            token_offsets.append((len(out_tokens), len(out_tokens) + 1))
            out_tokens.append(tokenizer.unk_token)
    return out_tokens, token_offsets


# two modes executable from __main__ are defined here: _cache_inputs and _test_load_from_cache
def _cache_inputs(args):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_config)
    leftover_features = []
    for sents_batch in parse_3_column_corpus(args.inputs, args.corpus):

        # turn sents into InputFeatures

        features = leftover_features
        leftover_features = []
        features += convert_sentences_to_features(
            sents_batch,
            tokenizer,
            MAX_SEQ_LENGTH,
            use_lemmas=args.use_lemmas,
            uncased=args.uncased,
        )
        for i in range(0, len(features), BATCH_SIZE):
            next_batch = features[i: i + BATCH_SIZE]
            if len(next_batch) < BATCH_SIZE:
                leftover_features += next_batch
            else:
                vectorize_features(next_batch, args.cached_inputs_file)
    if leftover_features:
        vectorize_features(leftover_features, args.cached_inputs_file)

def _test_load_from_cache(args):
    data = CorpusData(
        model_config=args.model_name_or_config,
        tokenizer_config=args.tokenizer_name_or_config,
        cached_data=args.cached_inputs_file,


    )
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_config)
    print(f"Done!")


if __name__ == "__main__":
    main()




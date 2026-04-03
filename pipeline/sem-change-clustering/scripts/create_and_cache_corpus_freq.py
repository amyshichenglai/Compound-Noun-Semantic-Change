"""Currently this is counting only NN unigrams and NN bigrams
   it's foreseeable that one could want to count prepositional variants
   or to count everything irrespective of the pos-tag
"""
from argparse import ArgumentParser
from util import year_to_coarse_slice, COARSE_TIME_SLICES, SPACE_REPLACEMENT_IN_FILENAMES
from data import CorpusData, Sentence

from collections import defaultdict
import json
from nltk import bigrams

def main():
    parser = ArgumentParser()
    parser.add_argument("archive", help='pickle file of cached data')
    parser.add_argument("--output_prefix", help="output path, will have {_1_grams|_2_grams}.json appended to it")
    parser.add_argument("--corpus_type", choices=["DTA", "COHA", "DTA_ALL"])
    args = parser.parse_args()

    data = CorpusData(cached_data=args.archive)


    lemma_freq_by_era = defaultdict(lambda: defaultdict(int))  # era -> lemma -> freq
    lemma_bigram_by_era = defaultdict(lambda: defaultdict(int))

    for batch in data.examples:
        for sent in batch['sents']:
            time_slice = year_to_coarse_slice(sent.year, args.corpus_type)
            if time_slice is None:
                continue
            slice_name = slice_to_str(time_slice)
            for lemma, tag in zip(sent.lemmas, sent.tags):
                if tag == "NN":
                    lemma_freq_by_era[slice_name][lemma] += 1
            for bigram, tag_bigram in zip(bigrams(sent.lemmas), bigrams(sent.tags)):
                if tag_bigram == ("NN", "NN"):
                    lemma_bigram_by_era[slice_name][bigram] += 1
    num_slices = len(COARSE_TIME_SLICES[args.corpus_type])
    output_1_gram, output_2_gram = {}, {} # lemma -> [frequencies_by_era...]
    for slice_tuple in COARSE_TIME_SLICES[args.corpus_type]:
        slice_name = slice_to_str(slice_tuple)
        slice_idx = COARSE_TIME_SLICES[args.corpus_type].index(slice_tuple)
        # TODO: need to pad these frequency lists out with zeros for
        #       combinations of item, era that don't occur in the data
        for word, freq in lemma_freq_by_era[slice_name].items():
            if word not in output_1_gram:
                output_1_gram[word] = [0] * num_slices
            output_1_gram[word][slice_idx] = freq
        for bigram, freq in lemma_bigram_by_era[slice_name].items():
            key = SPACE_REPLACEMENT_IN_FILENAMES.join(bigram)
            if key not in output_2_gram:
                output_2_gram[key] = [0] * num_slices
            output_2_gram[key][slice_idx] = freq

    with open(f"{args.output_prefix}_1_grams.json", 'w', encoding='utf-8') as out_f:
        json.dump(output_1_gram, out_f, ensure_ascii=False)
    with open(f"{args.output_prefix}_2_grams.json", 'w', encoding='utf-8') as out_f:
        json.dump(output_2_gram, out_f, ensure_ascii=False)

def slice_to_str(slice_tuple) -> str:
    return "-".join(str(year) for year in slice_tuple)

if __name__ == "__main__":
    main()

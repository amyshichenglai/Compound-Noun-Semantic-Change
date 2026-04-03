from argparse import ArgumentParser
from typing import List
from transformers import AutoTokenizer
from functools import partial
import zipfile
from data import parse_3_column_corpus, Sentence

def main():
    parser = ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--train_data", nargs='+', required=True,
                        help="text data, one sentence per line, possibly several files. "
                             "Assumed to all be .zip archives")
    parser.add_argument("--text_mode", choices=["lemma", "text"], required=True)
    parser.add_argument("--corpus", required=True, choices=["DTA", "COHA"])
    parser.add_argument("--uncased", action='store_true', help="lowercase all inputs")
    parser.add_argument("--output_dir", type=str, required=True, help="dir will be created if it doesn't already exist")
    args = parser.parse_args()

    sentences = []
    for batch in parse_3_column_corpus(args.train_data, args.corpus):
        sentences += batch

    base_tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    base_tokenizer_voc = len(base_tokenizer)
    if args.corpus == "COHA":
        base_tokenizer.add_tokens(['<no-seq>'], special_tokens=True)

    new_tokenizer = base_tokenizer.train_new_from_iterator(
        partial(get_train_data, args, sentences)(), base_tokenizer_voc)

    new_tokenizer.save_pretrained(args.output_dir)



# we need a generator to feed the tokenizer re-allocator algo one line at a time
def get_train_data(args, sentences: List[Sentence]):
    for sent in sentences:
        if args.text_mode == "text":
            line = sent.to_tokens()
        else:
            line = sent.to_lemmas()
        if args.uncased:
            line.lower()
        yield line

if __name__ == "__main__":
    main()
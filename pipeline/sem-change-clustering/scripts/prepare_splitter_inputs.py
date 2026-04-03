from argparse import ArgumentParser
import zipfile
import os
from collections import defaultdict

# assumed to be one directory up from this script
SCRABBLE_WORDS_FILE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', "scrabble-2er-words.txt")
)

def main():
    parser = ArgumentParser()
    parser.add_argument("input_decades", nargs='+', help="set of input files (3 column format) -- assumed to be .zip")
    parser.add_argument("--output_prefix", required=True, type=str, help="prefix to output .tsv files needed to run compound splitter")
    args = parser.parse_args()


    valid_len_2_words = set()
    with open(SCRABBLE_WORDS_FILE, encoding='utf-8') as scrabble_in:
        for word in scrabble_in:
            word = word.strip()
            valid_len_2_words.add(word.lower())



    word_to_lemma = {} # key is (word, pos) -> lemma
    lemma_freq = defaultdict(int) # key is (lemma, pos) -> count
    all_splitter_inputs = set() # tuples of (word, tag)

    for input_filename in args.input_decades:
        print(f"processing {input_filename}")
        with zipfile.ZipFile(input_filename) as in_z:
            filelist = in_z.namelist()
            for filename in filelist:
                print(f"reading file: {filename}")
                with in_z.open(filename, 'r') as in_f:
                    for line in in_f:
                        line = line.decode(encoding='utf-8')
                        tok, lem, tag = line.strip().split('\t')
                        tok = tok.lower()
                        if tok in ['<sod>', '<eos>']:
                            continue
                        lem = lem.lower()
                        tag = tag_mapping_DTA(tag)
                        if not tag:
                            # only bother with NN / ADJ / V
                            continue
                        if len(tok) in [1, 2] and tok not in valid_len_2_words:
                            continue
                        word_to_lemma[(tok, tag)] = lem
                        lemma_freq[(lem, tag)] += 1
                        all_splitter_inputs.add((tok, tag))


    with open(f"{args.output_prefix}_all_pos_freq.tsv", 'w', encoding='utf-8') as out_f:
        for (lemma, tag), count in lemma_freq.items():
            out_f.write(f"{lemma}\t{tag}\t{count}\n")
    with open(f"{args.output_prefix}_all_pos_lem.tsv", 'w', encoding='utf-8') as out_f:
        for (tok, tag), lemma in word_to_lemma.items():
            out_f.write(f"{tok}\t{tag}\t{lemma}\n")
    with open(f"{args.output_prefix}_all_splitter_inputs.tsv", 'w', encoding='utf-8') as out_f:
        for (tok, tag) in all_splitter_inputs:
            out_f.write(f"{tok}\t{tag}\n")



def tag_mapping_DTA(tag: str) -> str:
    """take extensive tag set of DTA data, return just
       1.) NN
       2.) ADJ
       3.) V
       4.) '' (empty string)
       As appropriate.
       This is here to be able to interface with the SimpleCompoundSplitter's simpler
       tagset
    """
    if tag == "NN":
        return "NN"
    if tag.startswith("ADJ"):
        return "ADJ"
    if tag.startswith("V"):
        return "V"
    return ""


def tag_mapping_COHA(tag: str) -> str:
    """Convert (already very reduced) CCCOHA style tagset into only
       NN, ADJ, V or '' (empty string)
    """
    tag = tag.upper()
    if tag == "VV":
        return "V"
    elif tag == "NN":
        return "NN"
    elif tag == "JJ":
        return "ADJ"
    return ""


if __name__ == "__main__":
    main()

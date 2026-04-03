from argparse import ArgumentParser
import os
import sys
import json
def main():
    parser = ArgumentParser()
    parser.add_argument("--unigram_freqs_file")
    parser.add_argument("--targets_file")
    args = parser.parse_args()

    with open(args.unigram_freqs_file, encoding='utf-8') as in_freqs:
        d = json.load(in_freqs)
    with open(args.targets_file, encoding='utf-8') as targs_file:
        targets = [line.strip() for line in targs_file]
    for target in targets:
        if target in d:
            print(f"{target}\t{sum(d[target])}")


if __name__ == "__main__":
    main()
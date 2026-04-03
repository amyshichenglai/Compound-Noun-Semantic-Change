from argparse import ArgumentParser
import os


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_file", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    filenames = sorted(
        filename for filename in os.listdir(args.input_dir)
        if filename.endswith(".pickle") or filename.endswith(".pkl")
    )
    with open(args.output_file, "w", encoding="utf-8") as out_f:
        out_f.write("Compound\tModifier\tHead\tModCount\tHeadCount\tModMean\tHeadMean\n")
        for filename in filenames:
            compound = filename.rsplit(".", 1)[0]
            # For 2nd-order random indexing over target compounds alone, the repo only
            # needs the compound identifier; dummy constituent/rating fields are enough.
            out_f.write(f"{compound}\t{compound}\t{compound}\t1\t1\t1.0\t1.0\n")


if __name__ == "__main__":
    main()

"""Script (possibly with importable components) for comparing separate
   runs of clustering with a metric like the V-Measure """
from argparse import ArgumentParser
import itertools
import json
import os

from sklearn.metrics import v_measure_score

from eval_util import format_eval_params, PARAMETERS_COLUMNS

COLUMNS = PARAMETERS_COLUMNS

def main():
    parser = ArgumentParser()
    parser.add_argument("--results_dir", help="if provided, runs comparisons for "
                                              "all pairs of files in directory")
    parser.add_argument("--output", help="used in the batch mode")
    parser.add_argument("--results_json_0", help="if provided, runs a single pair of files")
    parser.add_argument("--results_json_1", help="if provided, runs a single pair of files")

    args = parser.parse_args()

    if args.results_json_0 and args.results_json_1:
        print(f"V-Measure: "
          f"{calc_v_measure(args.results_json_0, args.results_json_1):.6f}")

    if args.results_dir and args.output:
        filenames = [n for n in os.listdir(args.results_dir)
                     if n.endswith(".json")]
        params_data_cache = {} # filename -> data
        for filename in filenames:
            with open(f"{args.results_dir}/{filename}", encoding='utf-8') as in_f:
                experiment_json = json.load(in_f)
            params_data_cache[filename] = format_eval_params(experiment_json)
        filenames_combinations = itertools.combinations(filenames, 2)
        output = {}
        for filename_0, filename_1 in filenames_combinations:
            # the combinations need to have had the same number
            # of samples
            samples_in_0 = params_data_cache[filename_0]['max_samples']
            samples_in_1 = params_data_cache[filename_1]['max_samples']
            if samples_in_0 != samples_in_1:
                continue
            # and the minimum count
            min_count_0 = params_data_cache[filename_0]['min_target_allocation']
            min_count_1 = params_data_cache[filename_1]['min_target_allocation']
            if min_count_0 != min_count_1:
                continue
            # they also need to be the same overall mode:
            # (related compounds / compound / constituent)
            related_compounds_0 = params_data_cache[filename_0]['related_compounds']
            related_compounds_1 = params_data_cache[filename_1]['related_compounds']
            if related_compounds_0 != related_compounds_1:
                continue
            merge_arg_0 = params_data_cache[filename_0]['merge']
            merge_arg_1 = params_data_cache[filename_1]['merge']
            # merging makes it difficult to compare things,
            # as it potentially removes small clusters
            if merge_arg_0 or merge_arg_1:
                continue
            heads_0 = params_data_cache[filename_0]['heads']
            heads_1 = params_data_cache[filename_1]['heads']
            if heads_0 != heads_1:
                continue
            mods_0 = params_data_cache[filename_0]['mods']
            mods_1 = params_data_cache[filename_1]['mods']
            if mods_0 != mods_1:
                continue

            # TODO: not the most efficient to re-read the files
            #       over and over again here...
            output[(filename_0, filename_1)] = \
                calc_v_measure(
                    f"{args.results_dir}/{filename_0}",
                    f"{args.results_dir}/{filename_1}"
                )

        with open(args.output, 'w', encoding='utf-8') as out_f:
            out_f.write("\t".join(COLUMNS + ['v-measure']))
            out_f.write("\n")
            for key_pair in output:
                for key in key_pair:
                    vals = params_data_cache[key]
                    per_run = [str(vals[c]) for c in COLUMNS]
                    out_f.write("\t".join(per_run))
                    out_f.write("\n")
                out_f.write('\t' * len(COLUMNS))
                out_f.write(str(output[key_pair]))
                out_f.write("\n")



def calc_v_measure(filename_0, filename_1) -> float:
    with open(filename_0, encoding='utf-8') as in_f:
        results_0 = json.load(in_f)
        results_0_tuples = sorted([
            (r["example"], r["label"])
            for r in results_0["clustering_output"]
        ])
    with open(filename_1, encoding='utf-8') as in_f:
        results_1 = json.load(in_f)
        results_1_tuples = sorted([
            (r["example"], r["label"])
            for r in results_1["clustering_output"]
        ])
    # so the v_measure_score is a comparison of just labels
    # but we need to align the example hashes s/t the labels
    # correspond to something meaningful

    # this might not be particularly meaningful to do in situations
    # where the items that are clustered are not genuinely identical
    labels_0, labels_1 = [], []
    for tuple_0, tuple_1 in zip(results_0_tuples, results_1_tuples):
        assert tuple_0[0] == tuple_1[0]
        labels_0.append(tuple_0[1])
        labels_1.append(tuple_1[1])

    return v_measure_score(labels_0, labels_1)

if __name__ == "__main__":
    main()
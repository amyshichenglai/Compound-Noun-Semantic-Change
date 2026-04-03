from argparse import ArgumentParser
from collections import defaultdict
import json
import os

import numpy as np
from scipy.stats import spearmanr

from eval_util import format_eval_params, PARAMETERS_COLUMNS, aggregate_phitag_eval

EVAL_TYPES = ['agg_phitag_f1', 'silhouette', 'davies_bouldin', 'calinski_harabasz']
COLUMNS = PARAMETERS_COLUMNS + EVAL_TYPES



def main():
    parser = ArgumentParser()
    parser.add_argument("--results_dir", help="directory with json results files")
    parser.add_argument("--output", help="tabular output, collating the different run parameters"
                                    " and the result from internal clustering metrics (unsupervised)")
    args = parser.parse_args()

    filenames = [n for n in os.listdir(args.results_dir)
                 if n.endswith(".json")]

    correlation_inputs = defaultdict(list)
    output = {}
    for filename in filenames:
        try:
            with open(f"{args.results_dir}/{filename}", encoding='utf-8') as in_f:
                experiment_json = json.load(in_f)
        except json.decoder.JSONDecodeError:
            print(f"ERROR loading file {filename}")
            continue
        # if not experiment_json['args']['related_compounds_file']:
        #     continue
        with open(f"{args.results_dir}/{filename.removesuffix('.json')}.log") as in_f:
            experiment_log = in_f.readlines()
            # it is kind of silly to keep this in the log, and not in the structured output...
        agg_f1, (acc, prec, recall), per_target_scores = \
            aggregate_phitag_eval(experiment_json)
        cluster_eval_lines = {}
        for line in experiment_log:
            if "cluster eval" in line:
                line_splits = line.rstrip().split()
                eval_type = line_splits[-2].removesuffix(":")
                eval_val = line_splits[-1]
                cluster_eval_lines[eval_type] = eval_val
                correlation_inputs[eval_type].append(float(eval_val))
        cluster_eval_lines["agg_phitag_f1"] = agg_f1
        correlation_inputs["agg_phitag_f1"].append(agg_f1)

        output[filename] = format_eval_params(experiment_json)
        for eval_type in EVAL_TYPES:
            output[filename][eval_type] = \
                cluster_eval_lines[eval_type] if eval_type in cluster_eval_lines else ""

    things_to_correlate_with_phitag_f1 = [c for c in EVAL_TYPES if c != "agg_phitag_f1"]
    # correlations between these ratings:
    f1_scores_array = np.array(correlation_inputs["agg_phitag_f1"])
    for measurement in things_to_correlate_with_phitag_f1:
        corr = spearmanr(
            np.array(correlation_inputs[measurement]),
            f1_scores_array
        )
        print(f"corr between phitag f1 and {measurement}:\n"
              f"rho: {corr[0]:.03f}; p-val: {corr[1]}")

    with open(args.output, 'w', encoding='utf-8') as out_f:
        out_f.write("\t".join(COLUMNS))
        out_f.write('\n')
        for k in output:
            vals = output[k]
            per_run = [str(vals[c]) for c in COLUMNS]
            out_f.write("\t".join(per_run))
            out_f.write("\n")




if __name__ == "__main__":
    main()
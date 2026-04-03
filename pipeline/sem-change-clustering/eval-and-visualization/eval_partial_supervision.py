from argparse import ArgumentParser
from collections import defaultdict
import itertools
import json
import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

from eval_util import (
    format_eval_params, PARAMETERS_COLUMNS,
    NON_TARGET_RESULT_DICT_KEYS,
    aggregate_phitag_eval,
)

COLUMNS = PARAMETERS_COLUMNS + ['aggregate_phitag_f1', 'aggregate_phitag_acc',
                                'aggregate_phitag_precision', 'aggregate_phitag_recall']

def main():
    parser = ArgumentParser()
    parser.add_argument("--results_dir", help="directory with json results files")
    parser.add_argument("--output", help="tabular output, collating the different run parameters"
                                    " and the result from the partial supervision")
    args = parser.parse_args()

    filenames = [n for n in os.listdir(args.results_dir)
                 if n.endswith(".json")]

    all_targets = set()

    output = {}
    per_target_output = {}
    for filename in filenames:
        with open(f"{args.results_dir}/{filename}", encoding='utf-8') as in_f:
            try:
                experiment_json = json.load(in_f)
            except json.decoder.JSONDecodeError:
                print(f"WARNING: could not parse {filename}")
                continue
        model_name = experiment_json['args']['model_name']
        if "+heads" in model_name and "+mods" in model_name:
            continue  # can´t handle this case at the moment
        agg_f1, (agg_acc, agg_prec, agg_recall), per_target_scores = aggregate_phitag_eval(experiment_json)
        for name in per_target_scores:
            all_targets.add(name.split("_")[0])
        output[filename] = format_eval_params(experiment_json)
        output[filename]['aggregate_phitag_f1'] = agg_f1
        output[filename]['aggregate_phitag_acc'] = agg_acc
        output[filename]['aggregate_phitag_precision'] = agg_prec
        output[filename]['aggregate_phitag_recall'] = agg_recall
        per_target_output[filename] = per_target_scores

    target_names = sorted(list(all_targets))
    target_cols = [c for c in itertools.chain(*[[f"{t}_f1"]
                for t in target_names])]
    # target_cols = [c for c in itertools.chain(*[[f"{t}_tp", f"{t}_fp", f"{t}_fn"]
    #             for t in target_names])]

    with open(args.output, 'w', encoding='utf-8') as out_f:
        out_f.write("\t".join(COLUMNS + target_cols))
        out_f.write('\n')
        for k in output:
            vals = output[k]
            per_target = per_target_output[k]
            per_run = [str(vals[c]) for c in COLUMNS]
            per_run += [str(per_target[c]) if c in per_target else "0"
                        for c in target_cols]

            out_f.write("\t".join(per_run))
            out_f.write("\n")




if __name__ == "__main__":
    main()
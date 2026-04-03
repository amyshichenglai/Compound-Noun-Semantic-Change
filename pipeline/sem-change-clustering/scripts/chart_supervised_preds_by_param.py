from argparse import ArgumentParser
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

import os
import json
from collections import defaultdict
from itertools import combinations, product

def main():
    parser = ArgumentParser()
    parser.add_argument("--cluster_results_dir")
    parser.add_argument("--output_dir")
    parser.add_argument("--experiment_name")
    parser.add_argument("--variables_list", nargs="+")

    args = parser.parse_args()


    # y values:
    file_to_normal_score = {}
    file_to_aggregated_representation_score = {}
    # x values:
    file_to_settings = {}

    for json_file in os.listdir(args.cluster_results_dir):
        with open(os.path.join(args.cluster_results_dir, json_file), encoding='utf-8') as in_f:
            as_dict = json.load(in_f)
            file_to_normal_score[json_file] = as_dict[args.experiment_name]["f1"]
            file_to_aggregated_representation_score[json_file] = as_dict["bifurcated_averaged_representations"]["f1"]
            file_to_settings[json_file] = as_dict["args"]

    var_to_values = defaultdict(list) # str to list of all_values
    var_to_range = {} # to list of sorted range of values
    for var_name in args.variables_list:
        # gather the range of these:
        for filename, settings in file_to_settings.items():
            var_to_values[var_name].append(settings[var_name])
        var_to_range[var_name] = sorted(list(set(var_to_values[var_name])))

    for var_name, values_range in var_to_range.items():
        for value in values_range:
            # find all the files that have this value, get scores, sort
            files_with_value = [(f, file_to_normal_score[f]) for f in file_to_settings if file_to_settings[f][var_name] == value]
            files_with_value.sort(key=lambda x: x[1], reverse=True)
            # print(f"{var_name}: {value} ranked: {files_with_value}\n\n")
            # this is the 'fix one value' approach

    # 'fix all but one' approach:
    all_but_one_score_range = {}
    best_alts_for_combi = {}
    for var_name_varied, values_range_varied in var_to_range.items():
        # get all combinations of variables that aren´t this one
        other_var_names = [name for name in var_to_range if name != var_name_varied]
        var_range = list(range(len(other_var_names)))
        # need to get all combinations (picking one value from each) of the values for the other var names
        # (which is the product)
        other_var_values = [var_to_range[name] for name in other_var_names]
        var_combinations = [p for p in product(*other_var_values)]
        # indices in this range correspond to indices in other_var_names
        for combi in var_combinations:
            alt_val_scores = []
            for alt_value in values_range_varied:
                # find files with this whole list of variables:

                for file, settings in file_to_settings.items():
                    if all([settings[other_var_names[i]] == combi[i] for i in var_range]) and settings[var_name_varied] == alt_value:
                        alt_val_scores.append(file_to_normal_score[file])
                        break

            # would like to know which alt value scored highest (or which tie)
            max_alt_vals = []
            alt_vals_sorted_by_score = sorted(list(zip(values_range_varied, alt_val_scores)), key=lambda x: x[1], reverse=True)
            for alt_val, score in alt_vals_sorted_by_score:
                if not max_alt_vals:
                    max_alt_vals.append((alt_val, score))
                else:
                    if score == max_alt_vals[-1][1]: # looking to report all things matching the top scoree
                        max_alt_vals.append((alt_val, score))

            name_val_combi = list(zip(other_var_names, combi))
            all_but_one_score_range[f"{name_val_combi}:{var_name_varied}"] = alt_val_scores
            best_alts_for_combi[f"{name_val_combi}:{var_name_varied}"] = max_alt_vals

    # now we have things that are essentially plottable...
    # granted, it's a lot to look at at once.
            df = pd.DataFrame([
                # (f"{v[0]}", v[1]) for label, val in best_alts_for_combi.items() for i, v in enumerate(val)
                (f"{values_range_varied[i]}", alt_val_scores[i]) for i in range(len(alt_val_scores))
            ], columns=['param_val', 'score'])

            sns.lineplot(data=df, x="param_val", y="score").set_title(f"{name_val_combi}::{var_name_varied}")
            plt.show()
            plt.close()

    # plots plots plots
    print("end")


if __name__ == "__main__":
    main()

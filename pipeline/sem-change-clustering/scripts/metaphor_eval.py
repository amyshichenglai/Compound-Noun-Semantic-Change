"""Use outputs from clustering to evaluate degree of change to / from compositional
   use of target compounds, and correlate this measure with present-day compositionality
   ratings. """

from typing import List
from argparse import ArgumentParser
import json
from scipy.stats import spearmanr
import numpy as np
from test_items_IO import CompositionalityRating, TestCompounds

SIGN_FN = lambda x: -1 if x < 0 else (1 if x > 0 else 0)


def main():
    parser = ArgumentParser()
    parser.add_argument("--results_json", required=True)
    parser.add_argument("--test_file", required=True)
    parser.add_argument("--test_file_type", choices=['ghost', 'cordeiro'], required=True)



    args = parser.parse_args()


    t1_r, t1_p, t2_r, t2_p, ranked_divergence_differences, keys, t1_divs, t2_divs = metaphor_eval(
        args.results_json,
        args.test_file,
        args.test_file_type
    )
    print(f"t1_corr\tp-val\tt2_corr\tp-val\n"
          f"{t1_r}\t{t1_p}\t{t2_r}\t{t2_p}")
    print(f"Ranks from convergence (-1) to divergence (+1) between time periods")
    for diff, name in ranked_divergence_differences:
        print(f"{name}\t{diff}")
    print("===========\ntarget\tt1 divergence\tt2 divergence")
    for i, k in enumerate(keys):
        print(f"{k}\t{t1_divs[i]:.2f}\t{t2_divs[i]:.2f}")

def metaphor_eval(results_json, test_file, test_file_type):
    results = None
    with open(results_json, encoding='utf-8') as in_f:
        results = json.load(in_f)
    if not results:
        raise ValueError("could not parse json")
    model_config_name = results["args"]["model_name"]
    if "+heads" in model_config_name:
        constituent = "head"
    elif "+mods" in model_config_name:
        constituent = "mod"
    else:
        raise ValueError("clustering was not run with constituents")

    test_items = None
    if test_file_type == "cordeiro":
        test_items = TestCompounds.load_cordeiro(test_file)
    elif test_file_type == "ghost":
        test_items = TestCompounds.load_ghost(test_file)

    most_compositional = test_items.most_compositional(criteria=constituent)
    least_compositional = test_items.least_compositional(criteria=constituent)

    all_targets = {t.compound: t for t in test_items.to_list()}

    # all in the same order as keys:
    keys = [t for t in all_targets if t in results and 'div_t1' in results[t]]
    t1_divergences = [results[k]["div_t1"] for k in keys]
    t2_divergences = [results[k]["div_t2"] for k in keys]

    divergence_differences = [t2 - t1 for t1, t2 in zip(t1_divergences, t2_divergences)]
    div_signs = [SIGN_FN(d) for d in divergence_differences]

    ranked_divergence_differences = sorted([(d, keys[i]) for i, d in enumerate(divergence_differences)])

    attribute = "mean_head_rating" if constituent == "head" else "mean_mod_rating"
    comp_ratings = [getattr(all_targets[k], attribute) for k in keys]

    t1_res = spearmanr(
        a=t1_divergences, b=comp_ratings
    )
    t1_r, t1_p = t1_res.correlation, t1_res.pvalue
    t2_res = spearmanr(
        a=t2_divergences, b=comp_ratings
    )
    t2_r, t2_p = t2_res.correlation, t2_res.pvalue

    return t1_r, t1_p, t2_r, t2_p, ranked_divergence_differences, keys, t1_divergences, t2_divergences


if __name__ == "__main__":
    main()

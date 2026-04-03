import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas
import json
from argparse import ArgumentParser
from test_items_IO import TestCompounds

from scripts.metaphor_eval import metaphor_eval

CONFIG = "Config"
DIV_DIFF = "Div Diff"
TARGETS = "Targets"

def main():
    parser = ArgumentParser()
    parser.add_argument("--results_json_files", nargs='*')
    # there should be enough in the results json logged to create an appropriate label
    parser.add_argument("--test_file")
    parser.add_argument("--test_file_type")
    parser.add_argument("--output_file")

    args = parser.parse_args()

    test_items = []
    if args.test_file_type == "cordeiro":
        test_items = TestCompounds.load_cordeiro(args.test_file).to_list()
    elif args.test_file_type == "ghost":
        test_items = TestCompounds.load_ghost(args.test_file).to_list()
    test_items_lookup = {t.compound: t for t in test_items}

    results = {
        TARGETS: [],
        CONFIG: [],
        DIV_DIFF: [],
    }
    test_items_most_least_compositionality = [t.compound for t in test_items]

    for results_file_name in args.results_json_files:
        with open(results_file_name, encoding='utf-8') as in_f:
            results_json =json.load(in_f)
        t1_r, t1_p, t2_r, t2_p, ranked_divergence_differences = metaphor_eval(
            results_file_name,
            args.test_file,
            args.test_file_type,
        )
        config_name = ""
        title_name = ""
        existing_results = [e[1] for e in ranked_divergence_differences]
        if "+heads" in results_json['args']['model_name']:
            # config_name += "+heads"
            title_name = "Head"
            test_items_most_least_compositionality = [t.compound
                                            for t in
                                            sorted(test_items, key=lambda x: x.mean_head_rating)
                                            if t.compound in existing_results
                                    ]
        if "+mods" in results_json['args']['model_name']:
            # config_name += "+mods"
            title_name = "Modifier"
            test_items_most_least_compositionality = [t.compound
                                                      for t in
                                                      sorted(test_items, key=lambda x: x.mean_mod_rating)
                                                      if t.compound in existing_results]
        if results_json['args']['use_bert_vecs']:
            config_name += "bert"
        if results_json['args']['use_second_order_vecs']:
            if config_name:
                config_name += "+"
            config_name += "2nd_order"


        for div_diff, target in ranked_divergence_differences:
            results[CONFIG].append(config_name)
            results[DIV_DIFF].append(div_diff)
            results[TARGETS].append(target)

    df = pandas.DataFrame(results)
    sns.relplot(
        data=df,
        x=CONFIG,
        y=DIV_DIFF,
        hue=TARGETS,
        hue_order=test_items_most_least_compositionality,
        palette="tab20",
    )
    plt.title(f"Compound x {title_name} Divergence (t2 - t1)")

    # plt.tight_layout()
    plt.subplots_adjust(top=0.88)

    plt.savefig(args.output_file)


if __name__ == "__main__":
    main()

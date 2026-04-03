from typing import Literal
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas
from argparse import ArgumentParser
import json
from test_items_IO import TestCompounds


def main():
    parser = ArgumentParser()
    parser.add_argument("--results_json_file")
    parser.add_argument("--era_output")
    parser.add_argument("--target_output")
    parser.add_argument("--suppress_plot_popup", action='store_true')
    parser.add_argument("--test_file")
    parser.add_argument("--test_file_type")
    parser.add_argument("--stipulations",
                        choices=["no-constraints", "constituents",
                                 "compounds", "most-compositional",
                                 "least-compositional"],
                        default="no-constraints",
    )
    parser.add_argument("--specific_targets", type=str, nargs='*', help="remember to use quotes if there are any spaces!")
    args = parser.parse_args()

    test_items = []
    if args.test_file_type == "cordeiro":
        test_items_class = TestCompounds.load_cordeiro(args.test_file)
        test_items = test_items_class.to_list()
    elif args.test_file_type == "ghost":
        test_items_class = TestCompounds.load_ghost(args.test_file)
        test_items = test_items_class.to_list()
    test_items_lookup = {t.compound: t for t in test_items}

    with open(args.results_json_file, encoding='utf-8') as in_f:
        results_json = json.load(in_f)

    early_era = str((results_json['params']['early_epoch_start'],
                 results_json['params']['early_epoch_end']))
    late_era = str((results_json['params']['late_epoch_start'],
                results_json['params']['late_epoch_end']))

    cluster_to_era_counts = results_json["__GLOBAL__"]["BY_ERA"]
    cluster_to_target_counts = results_json["__GLOBAL__"]["BY_TARGET"]
    cluster_to_target_counts_early = results_json["__GLOBAL__"]["BY_TARGET_EARLY"]
    cluster_to_target_counts_late = results_json["__GLOBAL__"]["BY_TARGET_LATE"]
    cluster_labels_as_ints = sorted([int(s) for s in cluster_to_target_counts.keys()])

    criteria="both"
    if "+heads" in results_json['args']['model_name']:
        criteria = "head"
    if "+mods" in results_json['args']['model_name']:
        criteria = "mod"
    targets_only_componds = set(k for k in test_items_lookup.keys())
    targets_only_constituents = set([t.head for t in test_items] + [t.mod for t in test_items])
    targets_high_compositionality = [
        item for test_item in
        test_items_class.most_compositional(criteria=criteria)
        for item in test_item.to_list_all()
    ]
    targets_low_compositionality = [
        item for test_item in
        test_items_class.least_compositional(criteria=criteria)
        for item in test_item.to_list_all()
    ]


    target_names = set()
    for _, inner_dict in cluster_to_target_counts.items():
        for target_name in inner_dict.keys():
            target_names.add(target_name)
    target_names = sorted(list(target_names))


    # TODO: should these all be lists of lists? i.e.
    #       [[0.0, 1.0], [0.25, 0.75], ...] (corresponding to first two clusters)
    cluster_to_era_early_proportion = []
    cluster_to_era_absolute_early = []
    cluster_to_era_absolute_late = []

    cluster_to_target_proportion = []
    cluster_to_target_absolute = []
    cluster_to_target_early_proportion = []
    cluster_to_target_late_proportion = []
    sanity_check_total = 0
    for i in cluster_labels_as_ints:
        key = str(i)
        # *** TODO -> there shouldn't be a mis-match between keys (here, missing from the by-time counts)
        #             figure out why this is the case
        if key not in cluster_to_era_counts:
            cluster_to_era_counts[key] = {}
        era_counts = cluster_to_era_counts[key]
        cluster_sum = 0
        if early_era in era_counts:
            cluster_sum += era_counts[early_era]
            sanity_check_total += era_counts[early_era]
            cluster_to_era_absolute_early.append(era_counts[early_era])
        else:
            cluster_to_era_absolute_early.append(0)
        if late_era in era_counts:
            cluster_sum += era_counts[late_era]
            sanity_check_total += era_counts[late_era]
            cluster_to_era_absolute_late.append(era_counts[late_era])
        else:
            cluster_to_era_absolute_late.append(0)
        if early_era in era_counts and cluster_sum:
            cluster_to_era_early_proportion.append(era_counts[early_era] / cluster_sum)
        else:
            cluster_to_era_early_proportion.append(0.0)
        # section for other chart:
        target_counts = cluster_to_target_counts[key]
        target_counts_early = cluster_to_target_counts_early[key] \
            if key in cluster_to_target_counts_early else {}
        target_counts_late = cluster_to_target_counts_late[key] \
            if key in cluster_to_target_counts_late else {}
        target_proportions, target_absolute  = [], []
        target_proportions_early, target_proportions_late = [], []
        if not cluster_sum:
            cluster_sum = sum([val for val in target_counts.values()])
            # TODO: this is only needed in clusters where the era counts are missing
        for target_name in target_names:
            if target_name in target_counts:
                target_absolute.append(target_counts[target_name])
                target_proportions.append(target_counts[target_name] / cluster_sum)
            else:
                target_absolute.append(0)
                target_proportions.append(0.0)
            if target_name in target_counts_early:
                target_proportions_early.append(target_counts_early[target_name] / cluster_sum)
            else:
                target_proportions_early.append(0.0)
            if target_name in target_counts_late:
                target_proportions_late.append(target_counts_late[target_name] / cluster_sum)
            else:
                target_proportions_late.append(0.0)
        cluster_to_target_proportion.append(target_proportions)
        cluster_to_target_early_proportion.append(target_proportions_early)
        cluster_to_target_late_proportion.append(target_proportions_late)
        cluster_to_target_absolute.append(target_absolute)


    fig, ax = plt.subplots(figsize=(14, 3.5), dpi=150)

    cluster_label="Clusters"
    early_proportion_label="Early"
    late_proportion_label="Late"
    # turn collected points into dataframes
    df_by_era = pandas.DataFrame(
        data={
            #cluster_label: cluster_labels_as_ints,
            early_proportion_label: cluster_to_era_early_proportion,
            late_proportion_label: [1.0 - e for e in cluster_to_era_early_proportion],
            # "Early (Absolute)": cluster_to_era_absolute_early,
            # "Late (Absolute)": cluster_to_era_absolute_late,
        },
        # x=cluster_label,
        # y=proportion_label,
        index=cluster_labels_as_ints
    )
    # grouped = df_by_era.groupby([early_proportion_label, late_proportion_label], as_index=False)
    # ax.bar(df_by_era[early_proportion_label])
    # ax.bar(df_by_era[late_proportion_label], bottom=df_by_era[early_proportion_label])

    plot_1 = df_by_era.plot(
        kind='bar',
        stacked=True,
        ax=ax,
        # x=cluster_label,
        color=['tab:blue', 'thistle'],
    )
    df_2 = pandas.DataFrame(
        data={
            "Early (Absolute)": cluster_to_era_absolute_early,
            "Late (Absolute)": cluster_to_era_absolute_late,
        }
    )
    plot_2 = df_2[['Early (Absolute)', "Late (Absolute)"]].plot(
        ax=ax,
        secondary_y=True,
        color=['red', 'green'],
    )
    ax.set_ylim(0, 1)
    ax.set_xticks(np.arange(0, 127, 25))
    # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # ax.set_xticklabels()
    plt.xlabel("Clusters")
    plot_1.set_ylabel("Proportion")
    plot_2.set_ylabel("Counts")
    plt.title(f"Clusters by Era")
    if not args.suppress_plot_popup:
        plt.show()
    if args.era_output:
        plt.savefig(args.era_output)

    plt.clf()
    # the other plot
    by_target_plot(
        args=args,
        target_names=target_names,
        targets_only_constituents=targets_only_constituents,
        targets_only_compounds=targets_only_componds,
        cluster_to_target_proportion=cluster_to_target_proportion,
        cluster_labels_as_ints=cluster_labels_as_ints,
        targets_high_compositionality=targets_high_compositionality,
        targets_low_compositionality=targets_low_compositionality,
        constituent_name=criteria,
        further_stipulations=args.stipulations,
    )

    if args.specific_targets and all([t in target_names for t in args.specific_targets]):
        # use the time-stratified counts
        by_target_plot_time_stratified(
            args=args,
            target_names=target_names,
            constituent_name=criteria,
            cluster_to_target_early_proportion=cluster_to_target_early_proportion,
            cluster_to_target_late_proportion=cluster_to_target_late_proportion,
            cluster_labels_as_ints=cluster_labels_as_ints,
        )
    print("done")


def by_target_plot(
    *,
    args,
    target_names,
    targets_only_constituents,
    targets_only_compounds,
    cluster_to_target_proportion,
    cluster_labels_as_ints,
    targets_high_compositionality,
    targets_low_compositionality,
    constituent_name: Literal["mod", "head", "both"],
    further_stipulations: Literal[
        "no-constraints", "constituents", "compounds", "most-compositional", "least-compositional"]="no-constraints",
):
    fig, ax = plt.subplots(figsize=(22, 4.5), dpi=150)

    data = {}
    for i in range(len(target_names)):
        if further_stipulations == "constituents":
            if target_names[i] not in targets_only_constituents:
                continue
        elif further_stipulations == "compounds":
            if target_names[i] not in targets_only_compounds:
                continue
        elif further_stipulations == "most-compositional":
            if target_names[i] not in targets_high_compositionality:
                continue
        elif further_stipulations == "least-compositional":
            if target_names[i] not in targets_low_compositionality:
                continue

        data[target_names[i]] = [cluster_to_target_proportion[j][i]
                                 for j in range(128)]

    title_str = "Clusters by Target"
    if further_stipulations == "constituents":
        if constituent_name == "mod":
            title_str += " (Modifiers)"
        elif constituent_name == "head":
            title_str += " (Heads)"
    elif further_stipulations == "compounds":
        title_str += " (Compounds)"
    elif further_stipulations == "most-compositional":
        title_str += " (Most Compositional)"
    elif further_stipulations == "least-compositional":
        title_str += " (Least Compositional)"

    df_by_target = pandas.DataFrame(
        data=data,
        index=cluster_labels_as_ints,
    )
    df_by_target.plot(
        kind="bar",
        stacked=True,
        ax=ax,
        cmap='tab20',
    )
    ax.set_ylim(0, 1)
    ax.set_xticks(np.arange(0, 127, 25))
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel("Clusters")
    plt.ylabel("Proportion")
    plt.title(title_str)
    plt.gcf().subplots_adjust(bottom=0.15)
    if not args.suppress_plot_popup:
        plt.show()
    if args.target_output:
        plt.savefig(args.target_output)

def by_target_plot_time_stratified(
    *,
    args,
    target_names,
    constituent_name: Literal["mod", "head", "both"],
    cluster_to_target_early_proportion,
    cluster_to_target_late_proportion,
    cluster_labels_as_ints,
):
    fig, ax = plt.subplots(figsize=(22, 4.5), dpi=150)

    data = {}
    for i in range(len(target_names)):
        if target_names[i] not in args.specific_targets:
            continue
        data[f"{target_names[i]} (early)"] = \
            [cluster_to_target_early_proportion[j][i]
                                 for j in range(128)]
        data[f"{target_names[i]} (late)"] = \
            [cluster_to_target_late_proportion[j][i]
             for j in range(128)]

    title_str = f"Cluster Proportions for " \
                f"{', '.join(t for t in args.specific_targets)}"

    df_by_target = pandas.DataFrame(
        data=data,
        index=cluster_labels_as_ints,
    )
    cmap = plt.get_cmap('tab20')
    df_by_target.plot(
        kind="bar",
        stacked=True,
        ax=ax,
        color=[c for c in cmap.colors],
        # you might think the below option would get the colors
        # in the correct order, but it does not!
        # cmap='tab20',
    )
    ax.set_ylim(0, 1)
    ax.set_xticks(np.arange(0, 127, 25))
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel("Clusters")
    plt.ylabel("Proportion")
    plt.title(title_str)
    plt.gcf().subplots_adjust(bottom=0.15)

    if not args.suppress_plot_popup:
        plt.show()
    if args.target_output:
        plt.savefig(args.target_output)


if __name__ == "__main__":
    main()

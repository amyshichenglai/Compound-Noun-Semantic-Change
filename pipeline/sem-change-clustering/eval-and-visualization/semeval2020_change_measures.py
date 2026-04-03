"""This operates on clusters obtained via annotation (i.e. the post WUG clusters)"""
import argparse
from collections import defaultdict
import os

import numpy as np
from scipy.stats import entropy

CHANGE_K = 1
CHANGE_N = 3

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clusters_from_annotation", type=str, help="path to directory containing "
                                                                     ".csv files")
    parser.add_argument("--clustering_data", help='directory containing per target directories w/ uses.csv files')
    # parser.add_argument("--mode", choices=['binary', 'graded'])
    parser.add_argument("--output_tsv")
    args = parser.parse_args()

    sentence_id_to_time_period = {}
    data_dirs = os.listdir(args.clustering_data)
    for target_name in data_dirs:
        with open(f"{args.clustering_data}/{target_name}/uses.csv") as in_f:
            for i, line in enumerate(in_f):
                if i == 0:
                    continue
                use_info = line.strip().split("\t")
                grouping = use_info[3]
                ident = use_info[4]
                sentence_id_to_time_period[ident] = int(grouping)


    annotated_file_list = os.listdir(args.clusters_from_annotation)

    # we need a frequency distribution per target of cluster / frequency
    # and we also need to know the time period for each

    # target -> distribution(i.e. cluster id -> count)
    early_distros = defaultdict(lambda: defaultdict(int))
    late_distros = defaultdict(lambda: defaultdict(int))
    per_target_cluster_ids = defaultdict(set)
    binary_change = {}
    for filename in annotated_file_list:
        target_name = filename.replace(".csv", "").replace("_", " ")
        with open(f"{args.clusters_from_annotation}/{filename}") as in_f:
            for i, line in enumerate(in_f):
                if i == 0:
                    continue
                # in spite of the .csv moniker, these are tab separated...
                ident, cluster_id = line.strip().split("\t")
                per_target_cluster_ids[target_name].add(cluster_id)
                time = sentence_id_to_time_period[ident]
                if time == 1:
                    early_distros[target_name][cluster_id] += 1
                elif time == 2:
                    late_distros[target_name][cluster_id] += 1
                else:
                    raise ValueError("secret third thing")

        # before converting to probabilities, we could run binary change scoring...
        for cluster_id in per_target_cluster_ids[target_name]:
            if cluster_id not in early_distros[target_name]:
                early_val = 0.0
            else:
                early_val = early_distros[target_name][cluster_id]
            if cluster_id not in late_distros[target_name]:
                late_val = 0.0
            else:
                late_val = late_distros[target_name][cluster_id]
            if early_val <= CHANGE_K and \
                late_val >= CHANGE_N:
                binary_change[target_name] = True
        if target_name not in binary_change:
            binary_change[target_name] = False

        # make the things into prob distros
        total_early = sum(v for v in early_distros[target_name].values())
        new_early_distro = {}
        for cluster_id, count in early_distros[target_name].items():
            new_early_distro[cluster_id] = count / total_early
        for cluster_id in per_target_cluster_ids[target_name]:
            if cluster_id not in new_early_distro:
                new_early_distro[cluster_id] = 0.0
        early_distros[target_name] = new_early_distro


        total_late = sum(v for v in late_distros[target_name].values())
        new_late_distro = {}
        for cluster_id, count in late_distros[target_name].items():
            new_late_distro[cluster_id] = count / total_late
        for cluster_id in per_target_cluster_ids[target_name]:
            if cluster_id not in new_late_distro:
                new_late_distro[cluster_id] = 0.0
        late_distros[target_name] = new_late_distro

    targets = set([k for k in early_distros] + [k for k in late_distros])
    # for target in sorted(list(targets)):
    #     print(f"\n{target}")
    #     print(f"early distro: {early_distros[target]}")
    #     print(f"late distro: {late_distros[target]}")
    #     print(f"JSD: {jsd(early_distros[target], late_distros[target])}")



    pairs = []
    for target in targets:
        pairs.append(
            (target, jsd(early_distros[target], late_distros[target]))
        )

    with open(args.output_tsv, 'w', encoding='utf-8') as out_f:
        out_f.write("target\tJSD_early_late\n")
        for pair in reversed(sorted(pairs, key=lambda x: x[1])):
            out_f.write(f"{pair[0]}\t{pair[1]}\n")


def jsd(p, q):
    keys = sorted(list(k for k in p))
    p = np.array([p[k] for k in keys])
    q = np.array([q[k] for k in keys])
    m = (p + q) / 2
    return (entropy(p, m) + entropy(q, m)) / 2

if __name__ == "__main__":
    main()
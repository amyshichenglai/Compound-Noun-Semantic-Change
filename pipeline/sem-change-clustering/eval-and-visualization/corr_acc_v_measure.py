import argparse

import numpy as np

from scipy.stats import spearmanr

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v_measure_tsv")
    parser.add_argument("--accuracy_tsv")
    args = parser.parse_args()

    # this is making the assumption that each file is listing
    # the experiments in the same order...
    avg_v_measures = []
    with open(args.v_measure_tsv, encoding="utf-8") as in_f:
        for i, line in enumerate(in_f):
            fields = line.split("\t")
            if i == 0:
                assert fields[9] == 'avg_v_measure'
                continue
            avg_v_measures.append(float(fields[9]))


    accuracies = []
    with open(args.accuracy_tsv, encoding='utf-8') as in_f:
        for i, line in enumerate(in_f):
            fields = line.split("\t")
            if i == 0:
                assert fields[10] == "aggregate_phitag_acc"
                continue
            accuracies.append(float(fields[10]))

    avg_v_measures = np.array(avg_v_measures)
    accuracies = np.array(accuracies)

    corr = spearmanr(avg_v_measures, accuracies)
    print(f"Correlation:\n"
          f"rho: {corr.correlation}\n"
          f"p-val: {corr.pvalue}")


if __name__ == "__main__":
    main()
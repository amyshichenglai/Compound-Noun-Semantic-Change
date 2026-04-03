from argparse import ArgumentParser
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


from annotation.phitag_interface import (
    process_annotations_file
)

MIN_ANNOTATIONS = 15

AGREEMENT_LOWER_THRESHOLD = 0.1

def main():
    parser = ArgumentParser()
    parser.add_argument("annotation_files", nargs="*",
                        help='These are the separate, not-consolidated'
                             ' phitag annotation files. full paths please!')
    parser.add_argument("--excluded_annotators_file")
    args = parser.parse_args()
    excluded_annotators = set()
    with open(args.excluded_annotators_file, encoding='utf-8') as in_f:
        for line in in_f:
            excluded_annotators.add(line.rstrip())

    kicked_due_to_low_agreement = []

    batches = []
    for filename in args.annotation_files:
        with open(filename, encoding='utf-8') as in_f:
            batches.append(process_annotations_file(filename))

    batch_means = []
    for batch_i, batch in enumerate(batches):

        by_annotator, avg_for_batch = run_batch(
            batch_i, batch, excluded_annotators
        )
        # now that we have an average correlation per annotator, we add
        # annotators below the threshold to the exclusion set and re-run
        for name, avg in zip(by_annotator, avg_for_batch):
            if avg < AGREEMENT_LOWER_THRESHOLD:
                kicked_due_to_low_agreement.append(name)
                excluded_annotators.add(name)
        _, avg_for_batch = run_batch(
            batch_i, batch, excluded_annotators
        )

        batch_mean = sum(avg_for_batch) / len(avg_for_batch)
        batch_means.append(batch_mean)
        print(f"\nbatch {batch_i} mean: {batch_mean:.02f}\n")

    overall_mean = sum(batch_means) / len(batch_means)
    print(f"Overall mean spearmans r: {overall_mean:.02f}")

    print("\n\nKicked due to low agreement (less than r 0.1):")
    for name in kicked_due_to_low_agreement:
        print(f"{name}")


def run_batch(batch_i, batch, excluded_annotators):
    annotator_to_instance_to_rating = defaultdict(lambda: defaultdict(int))
    instances_set = set()
    for entry in batch:
        if entry['annotator'] in excluded_annotators:
            continue
        instances_set.add(entry['instanceID'])
        label = entry['label']
        if label.startswith('-'):
            continue
            # label = 0
        else:
            label = int(label)
        annotator_to_instance_to_rating[entry['annotator']][entry['instanceID']] = label

    annotators_set = {anno for anno in annotator_to_instance_to_rating
                      if len(annotator_to_instance_to_rating[anno]) >= MIN_ANNOTATIONS}
    instances_ls = sorted(list(instances_set))
    # get all pairs of annotators
    pairs = [p for p in combinations(annotators_set, 2)]
    batch_correlations = []
    by_annotator = defaultdict(list)
    for pair in pairs:
        df = pd.DataFrame(
            [(annotator_to_instance_to_rating[pair[0]][instances_ls[i]],
              annotator_to_instance_to_rating[pair[1]][instances_ls[i]])
             for i in range(len(instances_ls))],
            columns=[pair[0], pair[1]]
        )
        corr = df.corr(method='spearman')
        batch_correlations.append(corr.values[0][1])  # a non diagonal of the square correlation matrix
        # print(f"{pair} -> {batch_correlations[-1]:.02f}")
        by_annotator[pair[0]].append(batch_correlations[-1])
        by_annotator[pair[1]].append(batch_correlations[-1])
    # batch_mean = sum(batch_correlations) / len(batch_correlations)
    # batch_means.append(batch_mean)
    # print(f"batch {batch_i} mean: {batch_mean:.02f}")
    avg_for_batch = []
    for annotator in by_annotator:
        # print(f"{annotator} -> "
        #       f"{sum(by_annotator[annotator]) / len(by_annotator[annotator]):.02f}")
        avg_corr = sum(by_annotator[annotator]) / len(by_annotator[annotator])
        avg_for_batch.append(avg_corr)

    return by_annotator, avg_for_batch

if __name__ == "__main__":
    main()
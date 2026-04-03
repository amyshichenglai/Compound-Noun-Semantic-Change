from collections import defaultdict
from typing import Dict

PARAMETERS_COLUMNS = ['best k', 'min_target_allocation', 'max_samples',
                               "bert_vecs", "second_order_vecs", 'merge',
                                'related_compounds',
                               'heads', 'mods', ]

NON_TARGET_RESULT_DICT_KEYS = {
    'args', 'params', '__GLOBAL__', 'phitag_eval', 'clustering_output'
}

def format_eval_params(evaluation_file_contents: Dict) -> Dict:
    return {
            'best k': evaluation_file_contents['BEST_K'],
            'min_target_allocation': evaluation_file_contents['args']['min_target_allocation_override'],
            'max_samples': evaluation_file_contents['args']['max_samples_override'],
            "bert_vecs": evaluation_file_contents['args']['use_bert_vecs'],
            "second_order_vecs": evaluation_file_contents['args']['use_second_order_vecs'],
            'merge': evaluation_file_contents['args']['merge_small_clusters'],
            'related_compounds': evaluation_file_contents['args']['related_compounds_file'] is not None,
            'heads': "+heads" in evaluation_file_contents['args']['model_name'],
            'mods': "+mods" in evaluation_file_contents['args']['model_name'],
    }

def aggregate_phitag_eval(results_json: Dict):
    if "phitag_eval" not in results_json:
        return 0.0, {}
    phi = results_json['phitag_eval']
    tp, tn, fp, fn = 0, 0, 0, 0
    all_targets = set()
    per_target = defaultdict(int)
    # should also do a per-target rating in case that's interesting...
    for eval_pair in phi:
        target_key = eval_pair['pair'][0].split("::")[0]
        all_targets.add(target_key)
        if eval_pair['clustered_together'] == "True":
            if eval_pair['clustered_correctly'] == "True":
                tp += 1
                per_target[f"{target_key}_tp"] += 1
            else:
                fp += 1
                per_target[f"{target_key}_fp"] += 1
        else:
            if eval_pair['clustered_correctly'] == "True":
                tn += 1
                per_target[f"{target_key}_tn"] += 1
            else:
                fn += 1
                per_target[f"{target_key}_fn"] += 1
    for target in all_targets:
        if f"{target}_tp" in per_target:
            target_tp =  per_target[f"{target}_tp"]
        else:
            target_tp = 0
        if f"{target}_tn" in per_target:
            target_tn = per_target[f"{target}_tn"]
        else:
            target_tn = 0
        if f"{target}_fn" in per_target:
            target_fn = per_target[f"{target}_fn"]
        else:
            target_fn = 0
        if f"{target}_fp" in per_target:
            target_fp = per_target[f"{target}_fp"]
        else:
            target_fp = 0
        denom = ((2 * target_tp) + target_fn + target_fp)
        if not denom:
            per_target[f"{target}_f1"] = 0
            per_target[f"{target}_acc"] = 0
        else:
            per_target[f"{target}_f1"] = (2 * target_tp) / denom
            per_target[f"{target}_acc"] = (target_tp + target_tn) / (target_tp + target_tn + target_fp + target_fn)
    f1 = (2 * tp) / ((2 * tp) + fn + fp)
    acc = (tp + tn) / (tp + tn + fp + fn)
    prec = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    return f1, (acc, prec, recall), per_target
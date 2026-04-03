import argparse
from collections import defaultdict
import json
import os
from statistics import mean, stdev

from sklearn.metrics import v_measure_score

from annotation.phitag_interface import load_unified_phitag_json
from eval_util import PARAMETERS_COLUMNS, format_eval_params

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clusters_from_annotation", type=str, help="path to directory containing "
                                                                     ".csv files")
    # parser.add_argument("--lang", choices=['en', 'de'], help="affects the handling of "
    #         "file names. Assumption is that english compounds are open and german ones are closed"
    # )
    parser.add_argument("--system_exp_dir")

    parser.add_argument("--annotation_json", help="consolidated annotation json file")
    parser.add_argument("--system_id_file", help='.tsv file with ids and sentences')
    # parser.add_argument("--system_output", type=str, help="json file, containing instance "
    #         "to cluster label mapping")
    parser.add_argument("--output_tsv")
    parser.add_argument("--per_config_output_tsv")
    parser.add_argument("--corpus_type", choices=["COHA", "DTA"])
    args = parser.parse_args()

    phitag_ratings = load_unified_phitag_json(args.annotation_json)
    # PhitagRatings object has a .examples property,
    # each PhitagExample has a context0 and a context1, which is
    # a PhitagUsage

    # PhitagUsage objects have a .target_sent attribute
    # as well as a .dataID attribute
    # annotated_sent_and_span_to_ids = defaultdict(set)
    ### this is a set for the rare case that the same sentence has multiple
    ### uses... if we need to disambiguate we can use the target character indices
    ### or map in the other direction

    annotated_sent_to_start_to_id = defaultdict(dict)

    # TODO: the remaining disambiguation is via *YEAR*
    #       there is one annotated sentence that is exactly the same
    #       but pulled from two different years in the COHA
    for example in phitag_ratings.examples:
        # annotated_sent_and_span_to_ids[(example.context0.target_sent, example.context0.target_sent_target_char_span)].add(example.context0.dataID)
        # annotated_sent_and_span_to_ids[(example.context1.target_sent, example.context1.target_sent_target_char_span)].add(example.context1.dataID)

        annotated_sent_to_start_to_id[example.context0.target_sent][example.context0.target_sent_target_char_span[0]] = example.context0.dataID
        annotated_sent_to_start_to_id[example.context1.target_sent][example.context1.target_sent_target_char_span[0]] = example.context1.dataID

    sents_to_span_start_to_system_id = defaultdict(lambda: defaultdict(list))
    with open(args.system_id_file, encoding='utf-8') as in_f:
        for line in in_f:
            cluster_item_id, token_start, \
                token_end_exclusive, sent_text = line.split("\t")
            sent_text = sent_text.rstrip() # newlines
            # collapse any extra spaces between tokens...
            sent_text = " ".join(sent_text.split())
            if args.corpus_type == "COHA":
                sent_text = sent_text.replace("<no-seq> ", "")
            sents_to_span_start_to_system_id[sent_text][token_start].append(cluster_item_id)



    # now, the mapping...
    # annotated sent -> annotated id, [sys ids]
    mapping = {}
    sys_keys_seen = set()
    for annotated_sent, inner_d in annotated_sent_to_start_to_id.items():
        for char_span, a_id in inner_d.items():
            if annotated_sent in sents_to_span_start_to_system_id:
                for start, sys_keys in sents_to_span_start_to_system_id[annotated_sent].items():
                    for sys_key in sys_keys:
                        if a_id.split("::")[0] != sys_key.split("::")[0]:
                            continue
                        if sys_key not in sys_keys_seen:
                            mapping[a_id] = sents_to_span_start_to_system_id[annotated_sent]
                            sys_keys_seen.add(sys_key)
            else:
                # not sure what can be done about this:
                print(f"WARNING: unable to find {annotated_sent}")

    # TODO: probably what would make sense is to have a reverse mapping...
    reverse_mapping = {}
    anno_keys_seen = set()
    for sent, inner_d in sents_to_span_start_to_system_id.items():
        for start, sys_id_list in inner_d.items():
            for sys_id in sys_id_list:
                if sent in annotated_sent_to_start_to_id:
                    for _, anno_id in annotated_sent_to_start_to_id[sent].items():
                        # To disambiguate the ids for the OCCASIONAL sentence
                        # containing more than one target!
                        if anno_id.split("::")[0] != sys_id.split("::")[0]:
                            continue
                        if anno_id not in anno_keys_seen:
                            reverse_mapping[sys_id] = annotated_sent_to_start_to_id[sent]
                            anno_keys_seen.add(anno_id)
                else:
                    pass
                    # This isn´t really a *warning* -- in this direction it is *expected*
                    # that most of the system sentences were not annotated...
                    # print(f"WARNING: unable to find sys sent {sent}")

    # load each form of clustering
    # clustered annotation graphs:
    annotated_file_list = os.listdir(args.clusters_from_annotation)
    # this is a separate clustering PER TARGET:
    annotation_graph_clusters = defaultdict(lambda: defaultdict(int))
    # target -> usage (e.g. Ziegenbock::32) -> cluster
    for filename in annotated_file_list:
        with open(f"{args.clusters_from_annotation}/{filename}", encoding='utf-8') as in_f:
            for i, line in enumerate(in_f):
                if i == 0:
                    continue # header
                ident, cluster = line.rstrip().split("\t")
                target = ident.split("::")[0]
                annotation_graph_clusters[target][ident] = int(cluster)

    output = ["config/target\tv-measure\n"]
    output_per_config = {}
    experiment_files = [f for f in os.listdir(args.system_exp_dir) if f.endswith(".json")]
    for filename in experiment_files:

        print(f"\nconfig: {filename}")
        output.append(filename.split(".csv")[0] + "\t\n")
        system_output = f"{args.system_exp_dir}/{filename}"
        with open(system_output, encoding='utf-8') as in_f:
            json_dict = json.load(in_f)
        output_per_config[filename] = format_eval_params(json_dict)
        system_clusters = defaultdict(lambda: defaultdict(int))
        for elt in json_dict["clustering_output"]:
            ident = elt["example"]
            cluster_id = int(elt["label"])
            target = ident.split("::")[0]
            system_clusters[target][ident] = cluster_id

        # v-measure doesn't much care about the lack of compatibility between the two label sets
        sorted_targets = sorted([t for t in annotation_graph_clusters])
        v_measures_for_config = []
        for target in sorted_targets:
            annotated_uses = []
            SYSTEM_IDS = []
            for k in annotation_graph_clusters[target]:
                if k in mapping:
                    annotated_uses.append(k)
                    for k2 in mapping[k]:
                        SYSTEM_IDS.append(mapping[k][k2])
            # annotated_uses = [k for k in annotation_graph_clusters[target].keys()
            #                      if k in mapping]
            for SYS_ID in SYSTEM_IDS:
                print(f"sys id {SYS_ID} -> {system_clusters[target][SYS_ID[0]]}")

            annotated_labels = [annotation_graph_clusters[target][k] for k in annotated_uses]

            # include only the labels for system instances that were also annotated!
            # NB: IF THE MAPPING ANNO->SYS GOES TO MULTIPLE THINGS, PICK THE FIRST ONE
            # permissible_sys_ids = [v for k in annotated_uses for v in mapping[k].values()[:1]]
            system_uses = [k for k in system_clusters[target] if k in reverse_mapping]
            system_labels = [system_clusters[target][k] for k in system_uses ]



            # print(f"system k/v pairs: {[(k, v) for k, v in system_clusters[target].items()]}")
            print(f"annotated labels: {annotated_labels}")
            print(f"system labels: {system_labels}")
            v_measure = v_measure_score(annotated_labels, system_labels)
            v_measures_for_config.append(v_measure)
            print(f"{target} -> {v_measure}")
            output.append(f"{target}\t{v_measure}\n")
        avg = mean(v_measures_for_config)
        dev = stdev(v_measures_for_config)
        output.append(f"AVERAGE\t{avg}\n")
        output_per_config[filename]["avg_v_measure"] = avg
        output_per_config[filename]["stddev_v_measure"] = dev
    # output the v-measure
    with open(args.output_tsv, 'w', encoding="utf-8") as out_f:
        for line in output:
            out_f.write(line)

    # need the per-config formatting from predictions_and_correlations.py...
    local_columns = ["avg_v_measure", "stddev_v_measure"]
    columns = PARAMETERS_COLUMNS + local_columns
    if args.per_config_output_tsv:
        with open(args.per_config_output_tsv, 'w', encoding='utf-8') as out_f:
            out_f.write("\t".join(columns))
            out_f.write("\n")
            for k in output_per_config:
                vals = output_per_config[k]
                per_run = [str(vals[c]) for c in columns]
                out_f.write("\t".join(per_run))
                out_f.write("\n")

if __name__ == "__main__":
    main()
set -euxo pipefail

OUTPUT_DIR=""
CONFIG=""
TEST=""
GOLD=""
EXAMPLE_CACHE_DIRS=""
DEVICE=""
RELATED_COMPOUNDS_DIR=""
COMPOUND_ANNOTATION_JSON=""
LANG=""

function Help(){
  echo "Args (keyword style, use --arg=value ): "
  echo "If you are running on mac or windows and don't have 'getopt', you can uncomment the above positional args and comment out the switch statement"
#  printf "\t--gpu | -g <val> (number: default is 0)"
  echo "\t--example_cache_dirs | -e (directory with cached examples in .pickle files -- use this arg more than once for multiple values"

}

get_abs_filename() {
  # $1 : relative filename
  echo "$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
}

# $@ is all command line parameters passed to the script.
# -o is for short options like -v
# -l is for long options with double dash like --version
# the comma separates different long options
# -a is for long options with single dash like -version
# : -> required arg (not sure if :: optional arg syntax actually works here)
options=$(getopt -l "help,output_dir:,device:,related:,lang:" -o "ho:d:r:l:" -- "$@" )

# set --:
# If no arguments follow this option, then the positional parameters are unset. Otherwise, the positional parameters
# are set to the arguments, even if some of them begin with a ‘-’.
eval set -- "$options"

while true; do
  case "$1" in
    -h|--help)
      Help
      exit;;
    -o|--output_dir)
      shift
      OUTPUT_DIR=$1;
      shift
      ;;
    -d|--device)
      shift
      DEVICE=$1;
      shift
      ;;
    -r|--related)
      shift
      RELATED_COMPOUNDS_DIR=$1;
      shift
      ;;
    -l|--lang)
      shift
      LANG=$1;
      shift
      ;;
    --)
      shift
      break;;
  esac
done


if [[ "$GOLD" == "" ]]; then
  GOLD_ARG="";
else
  GOLD_ARG="--gold_test_file $GOLD" ;
fi
if [[ "$LANG" != "en"  ]] && [[ "$LANG" != "de" ]]; then
  echo "invalid language specification: use either 'en' or 'de'"
  exit 1;
fi


mkdir -p "${OUTPUT_DIR}/${LANG}"
for min_alloc in 5; do
  for num_samples in 15000; do
    for feature_set in '--use_bert_vecs' '--use_bert_vecs --use_second_order_vecs' '--use_second_order_vecs' ; do
       for merge_arg in ''; do
#        for merge_arg in '' '--merge_small_clusters '; do
        if [[ "$feature_set" == '--use_bert_vecs' ]]; then
          feature_set_printable="bert"
        elif [[ "$feature_set" == '--use_second_order_vecs' ]]; then
          feature_set_printable="second_order"
        else
          feature_set_printable="bert_and_second_order"
        fi

        if [[ "$merge_arg" == "--merge_small_clusters " ]]; then
          merge_arg_printable="merge"
        else
          merge_arg_printable="no_merge"
        fi
        # compound-constituent clustering

        if [[  "$LANG" == "en" ]]; then
          TEST="cordeiro-ratings-onlyNN.txt"
          CONFIG="params/cluster-config-COHA.yml"
	  EXAMPLE_CACHE_DIRS="cached_example_lookups/cordeiro_compounds/2025-01-10-EN-double-precision-2nd-order-examples/"
          EXAMPLE_FREQ="${EXAMPLE_CACHE_DIRS}/target_counts.tsv"
          RELATED_COMPOUNDS_FILE_CORPUS="COHA"
          # TODO: not ideal that this is a hard-coded path
          COMPOUND_ANNOTATION_JSON="/projekte/semrel/Annotations/Noun-Compound-Relatedness/NN_compound_relatedness_CCOHA/2024-spring-summer-annotations/COMBINED_RESULTS_EN_END_OF_OCTOBER_2024.json"
        elif [[ "$LANG" == "de" ]]; then
          TEST="Ghost-NN_comp-means.txt"
          CONFIG="params/cluster-config-DTA.yml"
          EXAMPLE_CACHE_DIRS="cached_example_lookups/ghost_compounds/2025-01-10-ghost-compounds-cache-with-float64/"
          EXAMPLE_FREQ="${EXAMPLE_CACHE_DIRS}/target_counts.tsv"
          RELATED_COMPOUNDS_FILE_CORPUS="DTA"
          COMPOUND_ANNOTATION_JSON="/projekte/semrel/Users/chris/30-39_data/33_german_hist/33.01_dta_examples_to_phitag/COMBINED_RESULTS_DECEMBER_2024_DE.json"
        fi
        if [[ "$COMPOUND_ANNOTATION_JSON" != "" ]]; then
          COMPOUND_RATING_ARG="--compound_ratings_file ${COMPOUND_ANNOTATION_JSON} "
        else
          COMPOUND_RATING_ARG=""
        fi

        if [[ "$RELATED_COMPOUNDS_DIR" == "" ]]; then
          for constituents in "+mods" "+heads" "" ; do
            echo "running lang: $LANG, min alloc: $min_alloc samples: $num_samples features: $feature_set"
            time python clustering.py \
                --model_name "k_means_all_targets${constituents}" \
                --config_file $CONFIG \
                --test_file $TEST \
                --device $DEVICE \
                --cached_examples_dirs $EXAMPLE_CACHE_DIRS \
                --min_target_allocation_override $min_alloc \
                --max_samples_override $num_samples \
                $feature_set \
                --example_frequency_tsv $EXAMPLE_FREQ \
                --output_json "${OUTPUT_DIR}/${LANG}/${feature_set_printable}_${constituents}.json" \
                --min_cluster_threshold 1 \
                $merge_arg \
                --cluster_eval_metrics silhouette davies_bouldin calinski_harabasz \
                $COMPOUND_RATING_ARG \
                --diagnostics \
                > "${OUTPUT_DIR}/${LANG}/${feature_set_printable}_${constituents}.log"
          done
        else
         # related-compounds clustering
          if [[  "$LANG" == "en" ]]; then
	    EXAMPLE_CACHE_DIRS="cached_example_lookups/cordeiro_compounds/2025-01-10-EN-double-precision-2nd-order-examples/"
            EXAMPLE_FREQ="${EXAMPLE_CACHE_DIRS}/target_counts.tsv"
          elif [[ "$LANG" == "de" ]]; then
            EXAMPLE_CACHE_DIRS="cached_example_lookups/ghost_compounds/2025-01-10-ghost-compounds-cache-with-float64/"
            EXAMPLE_FREQ="${EXAMPLE_CACHE_DIRS}/target_counts.tsv"
          fi
          for constituents in "+mods" "+heads"; do
            time python clustering.py \
              --model_name "k_means_all_targets${constituents}" \
                --config_file ${CONFIG%%.yml}-related.yml \
                --test_file $TEST \
                --device $DEVICE \
                --cached_examples_dirs $EXAMPLE_CACHE_DIRS \
                --min_target_allocation_override $min_alloc \
                --max_samples_override $num_samples \
                $feature_set \
                --example_frequency_tsv $EXAMPLE_FREQ \
                --output_json "${OUTPUT_DIR}/${LANG}/related_compounds_${feature_set_printable}${constituents}.json" \
                --min_cluster_threshold 1 \
                $merge_arg \
                --cluster_eval_metrics silhouette davies_bouldin calinski_harabasz \
                $COMPOUND_RATING_ARG \
                --related_compounds_file ${RELATED_COMPOUNDS_DIR}/related-compond-candidates-${RELATED_COMPOUNDS_FILE_CORPUS}-${constituents##+}-counts-by-era-min-count-10.tsv \
                --diagnostics \
                > "${OUTPUT_DIR}/${LANG}/related_compounds_${feature_set_printable}${constituents}.log"
          done
        fi
      done
    done
  done
done



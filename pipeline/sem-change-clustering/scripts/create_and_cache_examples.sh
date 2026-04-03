set -euxo pipefail

OUTPUT_DIR=""
MODEL=""
TEST_FILE=""
TEST_FILE_TYPE=""
DEVICE=""
DATA=""
LANG=""
MODE=""
RELATED_COMPOUNDS_HEADS=""
RELATED_COMPOUNDS_MODS=""

SCRIPT_DIR=`pwd`

function Help(){
  echo "Args (keyword style, use --arg=value ): "
  echo "If you are running on mac or windows and don't have 'getopt', you can uncomment the above positional args and comment out the switch statement"
  echo "args are --model --output_dir --test_file --test_file_type [cordeiro, ghost, semeval] --device --data --lang [en, de] --ordinal [1st-order, 2nd-order] --related_mods <file> (optional) --related_heads <file> (optional) "
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
options=$(getopt -l "help,model:,output_dir:,test_file:,test_file_type:,device:,data:,lang:,ordinal:,related_mods:,related_heads:" -o "hm:o:f:t:g:d:l:n:qr" -a -- "$@")

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
    -m|--model)
      shift
      MODEL=$1;
      shift
      ;;
    -f|--test_file)
      shift
      TEST_FILE=$1;
      shift
      ;;
    -t|--test_file_type)
      shift
      TEST_FILE_TYPE=$1;
      shift
      ;;
    -g|--device)
      shift
      DEVICE=$1;
      shift
      ;;
    -d|--data)
      shift
      DATA=$1;
      shift
      ;;
    -l|--lang)
      shift
      LANG=$1;
      shift
      ;;
    -n|--ordinal)
      shift
      MODE=$1;
      shift
      ;;
    -q|--related_mods)
      shift
      RELATED_COMPOUNDS_MODS=$1;
      shift
      ;;
    -r|--related_heads)
      shift
      RELATED_COMPOUNDS_HEADS=$1;
      shift
      ;;
    --)
      shift
      break;;
  esac
done

mkdir -p $OUTPUT_DIR
mkdir -p $OUTPUT_DIR/bert

RELATED_COMPOUNDS_ARG=""
if [ "$RELATED_COMPOUNDS_MODS" != "" ]; then
    RELATED_COMPOUNDS_ARG="--related_compounds_file_mods ${RELATED_COMPOUNDS_MODS} ";
fi
if [ "$RELATED_COMPOUNDS_HEADS" != "" ]; then
  RELATED_COMPOUNDS_ARG="$RELATED_COMPOUNDS_ARG --related_compounds_file_heads ${RELATED_COMPOUNDS_HEADS} ";
fi

python scripts/create_and_cache_bert_vecs.py \
    --output_cache_dir $OUTPUT_DIR/bert \
    --cached_data $DATA \
    --model $MODEL \
    --test_file $TEST_FILE \
    --test_file_type $TEST_FILE_TYPE \
    --include_heads \
    --include_mods \
    --device $DEVICE \
    $RELATED_COMPOUNDS_ARG

if [ "$MODE" = "2nd-order" ]; then
  DATA_ARG="--cached_data $DATA";
else
  DATA_ARG="";
fi

mkdir -p ${OUTPUT_DIR}/combined
python scripts/create_and_cache_sparse_vecs.py \
    --cached_per_target_dir $OUTPUT_DIR/bert \
    --output_dir ${OUTPUT_DIR}/combined \
    --tokenizer $MODEL \
    --test_file $TEST_FILE \
    --test_file_type $TEST_FILE_TYPE \
    --lang $LANG \
    --mode $MODE \
    $DATA_ARG \
    $RELATED_COMPOUNDS_ARG

#./${SCRIPT_DIR}/create_frequency_tsv_from_cached_examples.sh ${OUTPUT_DIR}/combined/ $TEST_FILE_TYPE





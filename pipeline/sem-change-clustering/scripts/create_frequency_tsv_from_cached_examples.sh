# this uses the debug-scripts/check_cached_examples.py script
# which, yes, is a misnomer, since we are using it for a non debug purpose
# but hopefully having this shell script here to run the thing makes it obvious how to do it!

CACHED_EXAMPLES_DIR=$1;
CORPUS_TYPE=$2;

for f in `ls $CACHED_EXAMPLES_DIR/*.pickle`; do
  python debug-scripts/check_cached_examples.py \
    --cached_example_file $f \
    --corpus_type $CORPUS_TYPE \
    --count_only \
    >> $CACHED_EXAMPLES_DIR/target_counts.tsv ;
done
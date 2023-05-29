# $BASE_DIR: The dir where you put the data and ckpts
# input_file: ${BASE_DIR}/web_data/web_raw/anchor.tsv
WEB_DATA_DIR=${BASE_DIR}/web_data

STEP2_DIR=${WEB_DATA_DIR}/anchor_rule_filtered_step2
OUT_DIR=${WEB_DATA_DIR}/anchor_classifier_filtered

mkdir -p $OUT_DIR

python train.py --input_queries_file ${STEP2_DIR}/url2anchor_step2.pkl --output_file ${OUT_DIR}/url2anchor_final.pkl
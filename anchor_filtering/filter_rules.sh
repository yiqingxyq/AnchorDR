#!/usr/bin/env bash
# fail fast
set -e

# $BASE_DIR: The dir where you put the data and ckpts
# input_file: ${BASE_DIR}/web_data/web_raw/anchor.tsv
WEB_DATA_DIR=${BASE_DIR}/web_data

IN_DIR=${WEB_DATA_DIR}/web_raw
OUT_DIR1=${WEB_DATA_DIR}/anchor_rule_filtered_step1
OUT_DIR2=${WEB_DATA_DIR}/anchor_rule_filtered_step2
STAT_DIR=${WEB_DATA_DIR}

mkdir -p $OUT_DIR1
mkdir -p $OUT_DIR2

# echo "filter out header/footer anchors AND in-domain anchors"
# pv $IN_DIR/anchor.tsv | python filter_by_format_and_clean.py $OUT_DIR1/anchor_step1.tsv $STAT_DIR

echo "filter out anchors by keywords and cut anchors with length >= MAXLEN (64 by default)"
pv $OUT_DIR1/anchor_step1.tsv | python filter_by_keywords_and_len.py $OUT_DIR2/url2anchor_step2.pkl $STAT_DIR

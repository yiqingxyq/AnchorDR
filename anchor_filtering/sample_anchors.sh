#!/usr/bin/env bash
# fail fast
set -e

# $BASE_DIR: The dir where you put the data and ckpts
# input_file: ${BASE_DIR}/web_data/web_raw/anchor.tsv
WEB_DATA_DIR=${BASE_DIR}/web_data

corpus_name=web_corpus
anchor_name=web_anchor
INPUT_DIR=${WEB_DATA_DIR}/web_raw
STAT_DIR=$WEB_DATA_DIR

WEB_CORPUS_DIR=${WEB_DATA_DIR}/${corpus_name}
ANCHOR_DIR=${WEB_DATA_DIR}/${anchor_name}
ANCHOR_SUBSET_FILE=${WEB_DATA_DIR}/anchor_classifier_filtered/url2anchor_final.pkl

mkdir -p $ANCHOR_DIR

train_file=${WEB_CORPUS_DIR}/${corpus_name}.train_url
valid_file=${WEB_CORPUS_DIR}/${corpus_name}.valid_url


pv $train_file | python sample_anchors.py $ANCHOR_SUBSET_FILE $ANCHOR_DIR/${anchor_name}.train.tok.clean $STAT_DIR train

pv $valid_file | python sample_anchors.py $ANCHOR_SUBSET_FILE $ANCHOR_DIR/${anchor_name}.valid.tok.clean $STAT_DIR valid

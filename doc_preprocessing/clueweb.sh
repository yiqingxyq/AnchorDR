# $BASE_DIR: The dir where you put the data and ckpts
# input_file: ${BASE_DIR}/web_data/web_raw/web_corpus.tsv
WEB_DATA_DIR=${BASE_DIR}/web_data

INPUT_DIR=${WEB_DATA_DIR}/web_raw
URL_FILE=${WEB_DATA_DIR}/anchor_classifier_filtered/url2anchor_final.pkl
name=web_corpus
mkdir -p $WEB_DATA_DIR/${name}

python WebExtractor.py $INPUT_DIR/${name}.tsv --keep_url --url_subset_file ${URL_FILE} -b 30G -q -o - > $WEB_DATA_DIR/${name}/${name}.tsv.proc.txt

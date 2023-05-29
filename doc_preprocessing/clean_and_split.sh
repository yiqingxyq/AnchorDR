set -e 

name=web_corpus

# $BASE_DIR: The dir where you put the data and ckpts
# input_file: ${BASE_DIR}/web_data/web_raw/web_corpus.tsv
WEB_DATA_DIR=${BASE_DIR}/web_data
WEB_CORPUS_DIR=${WEB_DATA_DIR}/${name}

RAW_DATA_FILE=${name}.tsv.proc.txt # text input 
PROC_DATA_FILE=${name}.proc.txt # text output 

# # step 1: preprocess text file, skip url 
# pv $WEB_CORPUS_DIR/${RAW_DATA_FILE} | \
# python remove_non_utf8_chars.py | \
# python basic_clean.py | \
# python pre_filter.py | \
# python normalize_newline.py | \
# python segment_sentence.py > $WEB_CORPUS_DIR/${PROC_DATA_FILE} # one sentence per line, doc seperated by '\n'

# # step 2: split text to train and valid
# echo split
# pv $WEB_CORPUS_DIR/${PROC_DATA_FILE} | \
# python split_web_raw.py $WEB_CORPUS_DIR/$name 500

# # step 3: post process text file 
# echo post processing
# pv ${WEB_CORPUS_DIR}/${name}.train.txt | \
#     python concat_short_sentences.py | \
#     python post_filter.py ${WEB_CORPUS_DIR}/${name}.train.tok.filter > ${WEB_CORPUS_DIR}/${name}.train.tok.clean

# pv ${WEB_CORPUS_DIR}/${name}.valid.txt | \
#     python concat_short_sentences.py | \
#     python post_filter.py ${WEB_CORPUS_DIR}/${name}.valid.tok.filter > ${WEB_CORPUS_DIR}/${name}.valid.tok.clean


# step 4: split text and url into two files
echo train $name
pv $WEB_CORPUS_DIR/$name.train.tok.clean | \
python split_text_url.py $WEB_CORPUS_DIR/$name.train

echo valid $name
pv $WEB_CORPUS_DIR/$name.valid.tok.clean | \
python split_text_url.py $WEB_CORPUS_DIR/$name.valid


echo "done"
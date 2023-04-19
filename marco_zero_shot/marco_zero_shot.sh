model_save_name=$1
model_name_or_path=$2
n_gpu=${3:-2}
gpus=${4:-all}
master_port=${5:-24900}

qlen=32
plen=128
eval_batch_size=256
echo master_port: $master_port

if [ "$gpus" != all ]; then 
    export CUDA_VISIBLE_DEVICES=$gpus
fi

# eval
export COLLECTION_DIR=$BASE_DIR"/msmarco-zs-passage"
export EVAL_DIR=$BASE_DIR"/trec_eval"

# results 
export RESULT_HOME_DIR=$BASE_DIR"/msmarco-zs_ance"            # the output dir 
export RESULT_DIR=$RESULT_HOME_DIR"/result"                         # where outputs of various processes are written to
export EMBEDDING_DIR=$RESULT_HOME_DIR"/embeddings_cache"            # where large caches of vectors are stored after inference



opts='--use_t5_decoder --use_converted'

if [ "$model_name_or_path" == facebook/contriever ]; then 
    opts='--use_mean_pooler'
elif [ "$model_name_or_path" == Luyu/co-condenser-marco ]; then 
    opts=""
fi

MODEL_SAVE_NAME=${model_save_name}_zero_shot
RESULTS_LOG="$RESULT_DIR/$DATASET_NAME/$MODEL_SAVE_NAME/results.txt"

mkdir -p $EMBEDDING_DIR/marco/$MODEL_SAVE_NAME
mkdir -p $RESULT_DIR/$DATASET_NAME/$MODEL_SAVE_NAME
if [ ! -f "$RESULTS_LOG" ]; then
    touch $RESULTS_LOG
fi


cd $CODE_DIR
export PYTHONPATH=.


if [ -f "$EMBEDDING_DIR/marco/$MODEL_SAVE_NAME/embeddings.corpus.rank.0" ]; then
    echo "embeddings already exist for $MODEL_SAVE_NAME, no need to build index" 2>&1 | tee -a $RESULTS_LOG
else
    echo "building index of documents using the encoder at ${model_name_or_path}..." 2>&1 | tee -a $RESULTS_LOG
    python -m torch.distributed.launch --nproc_per_node=$n_gpu --master_port $master_port \
    lib/openmatch/driver/build_index.py \
        --output_dir $EMBEDDING_DIR/marco/$MODEL_SAVE_NAME/  \
        --model_name_or_path $model_name_or_path $opts \
        --per_device_eval_batch_size $eval_batch_size  \
        --corpus_path $COLLECTION_DIR/corpus.tsv  \
        --doc_template "Title: <title> Text: <text>"  \
        --doc_column_names id,title,text  \
        --q_max_len $qlen  \
        --p_max_len $plen  \
        --fp16  \
        --dataloader_num_workers 1 \
        --report_to tensorboard
fi

### check that those embeddings exist since the above operation is expensive and can fail...
if [ ! -d "$EMBEDDING_DIR/marco/$MODEL_SAVE_NAME/" ]; then
    echo "ERROR, embedding directory does not exist - $EMBEDDING_DIR/marco/$MODEL_SAVE_NAME/" 2>&1 | tee -a $RESULTS_LOG
    exit 1
fi

if [ -f "$RESULT_DIR/marco/$MODEL_SAVE_NAME/dev.trec" ]; then
    echo "dev.trec already exists at $RESULT_DIR/marco/$MODEL_SAVE_NAME/dev.trec" 2>&1 | tee -a $RESULTS_LOG
else
    echo "retrieving NN for dev queries..." 2>&1 | tee -a $RESULTS_LOG
    mkdir -p $RESULT_DIR/marco/$MODEL_SAVE_NAME
    python -m lib.openmatch.driver.retrieve  \
        --output_dir $EMBEDDING_DIR/marco/$MODEL_SAVE_NAME/  \
        --model_name_or_path $model_name_or_path $opts \
        --per_device_eval_batch_size $eval_batch_size  \
        --query_path $COLLECTION_DIR/queries.dev.small.tsv  \
        --query_template "<text>"  \
        --query_column_names id,text  \
        --q_max_len $qlen  \
        --fp16  \
        --trec_save_path $RESULT_DIR/marco/$MODEL_SAVE_NAME/dev.trec  \
        --dataloader_num_workers 1 \
        --report_to tensorboard
fi


if [ ! -f "$RESULT_DIR/marco/$MODEL_SAVE_NAME/dev.trec" ]; then
    echo "ERROR, .trec file for evaluation does not exist - $RESULT_DIR/marco/$MODEL_SAVE_NAME/dev.trec" 2>&1 | tee -a $RESULTS_LOG
    exit 1
fi 

echo "MRR@100" | tee -a $RESULTS_LOG
$EVAL_DIR/trec_eval -c -mrecip_rank.100 -mndcg_cut.10 -mrecall.100 $COLLECTION_DIR/qrels.dev.small.tsv $RESULT_DIR/marco/$MODEL_SAVE_NAME/dev.trec | tee -a $RESULTS_LOG

echo "MRR@10" | tee -a $RESULTS_LOG
python $CODE_DIR/metrics/mrr.py $COLLECTION_DIR/qrels.dev.small.tsv $RESULT_DIR/marco/$MODEL_SAVE_NAME/dev.trec | tee -a $RESULTS_LOG

cat $RESULTS_LOG

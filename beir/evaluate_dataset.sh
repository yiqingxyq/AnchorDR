model_save_name=$1
model_name_or_path=$2
dataset_name=$3
n_gpu=${4:-2}
gpus=${5:-0,1}
master_port=${6:-19286}


export CUDA_VISIBLE_DEVICES=$gpus
export TF_CPP_MIN_LOG_LEVEL="2"

# eval
COLLECTION_DIR=$BASE_DIR"/beir"
EVAL_DIR=$BASE_DIR"/trec_eval"
BEIR_DIR=$CODE_DIR/beir

# results
RESULT_HOME_DIR=$BASE_DIR"/beir_ance"
RESULT_DIR=$RESULT_HOME_DIR"/result"
EMBEDDING_DIR=$RESULT_HOME_DIR"/embeddings_cache"



PER_DEVICE_INFERENCE_BATCH_SIZE=64
Q_MAX_LEN=64
P_MAX_LEN=128



if [ "$dataset_name" == "arguana" ]; then 
    Q_MAX_LEN=128 
fi

if [ "$dataset_name" == "robust04" -o "$dataset_name" == "trec-news" -o "$dataset_name" == "scifact" ]; then 
    P_MAX_LEN=512
fi


opts='--use_t5_decoder --use_converted'

if [ "$model_name_or_path" == facebook/contriever ]; then 
    opts='--use_mean_pooler'
elif [ "$model_name_or_path" == Luyu/co-condenser-marco ]; then 
    opts=""
fi

mkdir -p $RESULT_DIR/beir/$dataset_name/$model_save_name/
mkdir -p $EMBEDDING_DIR/beir/$dataset_name/$model_save_name/

cd $CODE_DIR
export PYTHONPATH=.


if [ -f "$EMBEDDING_DIR/beir/$dataset_name/$model_save_name/embeddings.corpus.rank.0" ]; then
    echo "embeddings already exist for $EMBEDDING_DIR/beir/$dataset_name/$model_save_name/embeddings.corpus.rank.0, no need to build index"
else
    echo "building index..."
    python -m torch.distributed.launch --nproc_per_node=${n_gpu} --master_port $master_port \
        lib/openmatch/driver/build_index.py  \
        --output_dir $EMBEDDING_DIR/beir/$dataset_name/$model_save_name/ \
        --model_name_or_path $model_name_or_path $opts \
        --per_device_eval_batch_size 128  \
        --corpus_path $COLLECTION_DIR/$dataset_name/corpus.tsv  \
        --doc_template "Title: <title> Text: <text>"  \
        --doc_column_names id,title,text  \
        --q_max_len $Q_MAX_LEN  \
        --p_max_len $P_MAX_LEN  \
        --fp16  \
        --dataloader_num_workers 1
fi

if [ -f "$RESULT_DIR/beir/$dataset_name/$model_save_name/test.trec" ]; then
    echo "dev.trec already exists at $RESULT_DIR/beir/$dataset_name/$model_save_name/test.trec"
else
    echo "retrieving..."
    python lib/openmatch/driver/retrieve.py  \
        --output_dir $EMBEDDING_DIR/beir/$dataset_name/$model_save_name/ \
        --model_name_or_path $model_name_or_path $opts \
        --per_device_eval_batch_size 128  \
        --query_path $COLLECTION_DIR/$dataset_name/queries.test.tsv  \
        --query_template "<text>"  \
        --query_column_names id,text  \
        --q_max_len $Q_MAX_LEN  \
        --fp16  \
        --trec_save_path $RESULT_DIR/beir/$dataset_name/$model_save_name/test.trec  \
        --dataloader_num_workers 1
fi

# In Arguana, some queries are also documents, so when we retrieve for a query, we remove the documents that are the same as the query.
# You can check https://github.com/Veronicium/AnchorDR/blob/main/beir/evaluate_dataset.sh for details
if [ "$dataset_name" == "arguana" ]; then
    mv $RESULT_DIR/beir/$dataset_name/$model_save_name/test.trec $RESULT_DIR/beir/$dataset_name/$model_save_name/test.orig.trec
    python $BEIR_DIR/remove_same_qd.py \
        --trec $RESULT_DIR/beir/$dataset_name/$model_save_name/test.orig.trec  \
        --save_to $RESULT_DIR/beir/$dataset_name/$model_save_name/test.trec
fi

echo $dataset_name
echo "scoring..."
RESULTS_LOG=$RESULT_DIR/beir/$dataset_name/$model_save_name/result.txt

rm $RESULTS_LOG
$EVAL_DIR/trec_eval -c -mrecip_rank.10 -mndcg_cut.10 $COLLECTION_DIR/$dataset_name/qrel.test.tsv $RESULT_DIR/beir/$dataset_name/$model_save_name/test.trec | tee -a $RESULTS_LOG

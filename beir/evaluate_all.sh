model_save_name=$1
model_name_or_path=$2
n_gpu=${3:-2}
gpus=${4:-0,1}
master_port=${5:-19286}

RESULT_HOME_DIR=$BASE_DIR"/beir_ance"

# for DATASET in arguana climate-fever dbpedia-entity fever fiqa hotpotqa nfcorpus nq quora scidocs scifact trec-covid webis-touche2020
# do 
#     echo "evaluating $DATASET ..."
#     bash evaluate_dataset.sh $model_save_name $model_name_or_path $DATASET $n_gpu $gpus $master_port 
# done

# evaluate CQADupStack 
for DATASET in android  english  gaming  gis  mathematica  physics  programmers  stats  tex  unix  webmasters  wordpress
do 
    echo "evaluating cqadupstack/$DATASET ..."
    bash evaluate_dataset.sh $model_save_name $model_name_or_path cqadupstack/$DATASET $n_gpu $gpus $master_port 
done

python read_result.py $model_save_name $RESULT_HOME_DIR
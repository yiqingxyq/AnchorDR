## Load the AnchorDR model
The AnchorDR model can be loaded from Huggingface using the identifier `yiqingx/AnchorDR`.
You can use the following code to load AnchorDR in your code:
```
from transformers import AutoModel 
model = AutoModel.from_pretrained('yiqingx/AnchorDR')
```


## Zero-shot evaluation of MSMARCO and BEIR

### Step 1: setup
You first need to modify the following variables in `setup.sh`:
- `$BASE_DIR`: The dir where you put the data and ckpts
- `$CODE_DIR`: The dir of this codebase

Then set those environment variables by:
```
source setup.sh
```

### Step 2: download preprocessed data 
You need to download the preprocessed MSMARCO and BEIR data.

Download preprocessed BEIR data (the 14 public datasets):
```
cd $BASE_DIR

export fileid="1ruposfTWXoXlb5EuxvLYCzPp5rGiW77U"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o processed_beir.tar.gz
rm ./cookie

tar -xvf processed_beir.tar.gz

```

Download the preprocessed MSMARCO data:
```
cd $BASE_DIR

export fileid="1r3NZtf8cNvdC2ayZUwU5RxUfJzqk9Xkw"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o processed_marco_zs.tar.gz
rm ./cookie

tar -xvf processed_marco_zs.tar.gz
```

### Step 3: install required packages and repo
First, install torch based on your system's CUDA version. For example:
```
pip install torch==1.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html 
```

Then install other required packages:
```
bash install_trec.sh
bash install.sh 
```

### Step 4: Inference on MSMARCO
```
cd ${CODE_DIR}/marco_zero_shot 
bash marco_zero_shot.sh $model_save_name $model_name_or_path $n_gpu $gpus 

```

For example, if you are going to evaluate AnchorDR on MSMARCO with GPU 0, you should run:
```
cd ${CODE_DIR}/marco_zero_shot 
bash marco_zero_shot.sh AnchorDR yiqingx/AnchorDR 1 0

```

### Step 5: Inference on BEIR 
Similarly, you can run inference on the BEIR datasets. We provide the scripts to evaluate on the 14 public datasets:
```
cd ${CODE_DIR}/beir
bash evaluate_all.sh $model_save_name $model_name_or_path $n_gpu $gpus 

```

For example, if you are going to evaluate AnchorDR on BEIR with GPU 0, you should run:
```
cd ${CODE_DIR}/beir
bash evaluate_all.sh AnchorDR yiqingx/AnchorDR 1 0

```
## Load the AnchorDR model
The AnchorDR model can be loaded from Huggingface using the identifier `yiqingx/AnchorDR`.
You can use the following code to load AnchorDR in your code:
```
from transformers import AutoModel 
model = AutoModel.from_pretrained('yiqingx/AnchorDR')
```


## Data preprocessing

We provide the [preprocessed data](https://drive.google.com/file/d/151v1ZB4jjmQ0XfFlRpiYwNvUtEuix5xr/view?usp=sharing) for continuous pretraining. You can refer to [OpenMatch](https://github.com/OpenMatch/OpenMatch) for the training scripts.


First, set the value of `$BASE_DIR` in `setup.sh`

### Input files 
We follow the file format of ClueWeb. There are two input files:
```
anchor.tsv      # the file of anchors
web_corpus.tsv  # the file of webpages
```

In `anchor.tsv`, each line is a dict representing an anchor:
```
{
    "url":"https://www.racedepartment.com/downloads/estonia-20.15375/",     # the url of the linked webpage
    "urlhash":"000002C4D180C769E761169A9938BA86",                           # the url hash of the linked webpage
    "language":"en",                                                        # the language of the linked webpage
    "anchors":[
        ["https://www.racedepartment.com/downloads/categories/ac-cars.6/?page=16","61BA3D7DB2E598859E086AC148B9B9BD","","","en"],
        ["https://www.racedepartment.com/downloads/categories/ac-cars.6/?page=16","61BA3D7DB2E598859E086AC148B9B9BD","Estonia-20","","en"],
        ["https://www.racedepartment.com/downloads/categories/ac-cars.6/?page=16","61BA3D7DB2E598859E086AC148B9B9BD","May 14, 2017","","en"]
    ],
    # Each anchor is a list, containing:
    #   (1) the url of the source webpage, 
    #   (2) the url hash of the source webpage, 
    #   (3) the anchor text 
    #   (4) an indicator of header/footer. "" or "0" means it is not a header/footer
    #   (5) the language of the source webpage
}

```

In `web_corpus.tsv`, each line is a dict representing a webpage:
```
{
    "Url": "https://www.ionos.com/digitalguide/server/tools/netsh/",        # the url of the webpage
    "UrlHash": "134F6ADC9611D195FCD56F2EE3039ABD",                          # the url hash of the webpage
    "Language": "en",                                                       # the language of the webpage
    "Topic": ["linux", "command line interfaces", "web development", ".net framework", "computer programming"], 
    "Primary": "Netsh – how to manage networks with Netsh commands\nAnyone who works with Windows network configurations will sooner or later come across the Network Shell (Netsh). The term refers to an interface between users and the operating system, which enables the administration and configuration of local, and remote network settings.\nThe range of applications includes settings for the Windows firewall and LAN/WLAN management as well as IP and server configuration ... ",
    "Title": "Netsh – how to manage networks with Netsh commands", 
    "HtmlTitle": "netsh commands | How does netsh work? [+examples] - IONOS", 
    "Rank": 37315,
}

```


### Filter anchors by rules and classifier 
Assume the input file is `${BASE_DIR}/web_data/web_raw/anchor.tsv`, here is the script to filter anchors:
```
cd anchor_filtering
bash filter_rules.sh 
bash filter_classifier.sh

```
The filtered anchors will be output to `${BASE_DIR}/web_data/anchor_classifier_filtered/url2anchor_final.pkl`. This is a dict, where the keys are urls of the linked webpages and the values are a list of anchor text. 


### Preprocess linked documents  
Assume the input file is `${BASE_DIR}/web_data/web_raw/web_corpus.tsv`, here is the data preprocessing script:
```
cd doc_preprocessing 
bash clueweb.sh 
bash clean_and_split.sh 

```
The output training file `${BASE_DIR}/web_data/web_corpus/web_corpus.train_clean` contains the cleaned web documents. Each line is a sentence and the documents are separated by one or more '\n'.
`${BASE_DIR}/web_data/web_corpus/web_corpus.train_url` contains the urls of the web documents (one url per line).
The output validation files `${BASE_DIR}/web_data/web_corpus/web_corpus.valid_clean` and `${BASE_DIR}/web_data/web_corpus/web_corpus.valid_url` follow the same format.

### Sample anchors 
Sample at most 5 anchors for each document:
```
cd anchor_filtering
bash sample_anchors.sh

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

import os
from dataclasses import dataclass, field
from typing import Optional, List
from transformers import TrainingArguments


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    target_model_path: str = field(
        default=None,
        metadata={"help": "Path to pretrained reranker target model"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )

    # modeling
    untie_encoder: bool = field(
        default=False,
        metadata={"help": "no weight sharing between qry passage encoders"}
    )

    # out projection
    add_pooler: bool = field(default=False)
    projection_in_dim: int = field(default=768)
    projection_out_dim: int = field(default=768)

    # for Jax training
    dtype: Optional[str] = field(
        default="float32",
        metadata={
            "help": "Floating-point format in which the model weights should be initialized and trained. Choose one "
                    "of `[float32, float16, bfloat16]`. "
        },
    )

    use_t5: bool = field(
        default=False,
        metadata={"help": "Whether to use T5 model"}
    )

    use_t5_decoder: bool = field(
        default=False,
        metadata={"help": "Whether to use T5 decoder"}
    )
    use_converted: bool = field(
        default=False,
        metadata={"help": "Whether to use model from fs"}
    )
    use_mean_pooler: bool = field(
        default=False,
        metadata={"help": "Whether to use mean pooler"}
    )
    vocab_path: str=field(
        default=None
    )

    # for trainning with hard negative
    iter_num: Optional[int]=field(
        default=0, metadata={"help": "Iteration of hard negative generation, used to decay learning rate"}
    )

    decay_rate: Optional[float]=field(
        default=0.6, metadata={"help": "Decay learning rate"}
    )


@dataclass
class DataArguments:
    train_dir: str = field(
        default=None, metadata={"help": "Path to train directory"}
    )
    train_path: str = field(
        default=None, metadata={"help": "Path to single train file"}
    )
    eval_path: str = field(
        default=None, metadata={"help": "Path to eval file"}
    )
    query_path: str = field(
        default=None, metadata={"help": "Path to query file"}
    )
    corpus_path: str = field(
        default=None, metadata={"help": "Path to corpus file"}
    )
    data_dir: str = field(
        default=None, metadata={"help": "Path to data directory"}
    )
    data_path: str = field(
        default=None, metadata={"help": "Path to the single data file"}
    )
    processed_data_path: str = field(
        default=None, metadata={"help": "Path to processed data directory"}
    )
    dataset_name: str = field(
        default=None, metadata={"help": "huggingface dataset name"}
    )
    passage_field_separator: str = field(default=' ')
    dataset_proc_num: int = field(
        default=12, metadata={"help": "number of proc used in dataset preprocess"}
    )
    train_n_passages: int = field(default=8)
    positive_passage_no_shuffle: bool = field(
        default=False, metadata={"help": "always use the first positive passage"})
    negative_passage_no_shuffle: bool = field(
        default=False, metadata={"help": "always use the first negative passages"})

    encode_in_path: List[str] = field(default=None, metadata={"help": "Path to data to encode"})
    
    encode_is_qry: bool = field(default=False)
    encode_num_shard: int = field(default=1)
    encode_shard_index: int = field(default=0)

    q_max_len: int = field(
        default=32,
        metadata={
            "help": "The maximum total input sequence length after tokenization for query. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    p_max_len: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization for passage. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    data_cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the data downloaded from huggingface"}
    )


    query_template: str = field(
        default="<text>",
        metadata={"help": "template for query"}
    )
    query_column_names: str = field(
        default="id,text",
        metadata={"help": "column names for the tsv data format"}
    )
    doc_template: str = field(
        default="Title: <title> Text: <text>",
        metadata={"help": "template for doc"}
    )
    doc_column_names: str = field(
        default="id,title,text",
        metadata={"help": "column names for the tsv data format"}
    )
    train_dataset_len: int = field(
        default=400282,
        metadata={
            "help": "The length of the train dataset. Entered in advance to save data processing time"
        },
    )

    def __post_init__(self):
        pass
        # if self.dataset_name is not None:
        #     info = self.dataset_name.split('/')
        #     self.dataset_split = info[-1] if len(info) == 3 else 'train'
        #     self.dataset_name = "/".join(info[:-1]) if len(info) == 3 else '/'.join(info)
        #     self.dataset_language = 'default'
        #     if ':' in self.dataset_name:
        #         self.dataset_name, self.dataset_language = self.dataset_name.split(':')
        # else:
        #     self.dataset_name = 'json'
        #     self.dataset_split = 'train'
        #     self.dataset_language = 'default'
        # if self.train_dir is not None:
        #     files = os.listdir(self.train_dir)
        #     self.train_path = [
        #         os.path.join(self.train_dir, f)
        #         for f in files
        #         if f.endswith('jsonl') or f.endswith('json')
        #     ]
        # else:
        #     self.train_path = None


@dataclass
class DenseTrainingArguments(TrainingArguments):
    warmup_ratio: float = field(default=0.1)
    negatives_x_device: bool = field(default=False, metadata={"help": "share negatives across devices"})
    do_encode: bool = field(default=False, metadata={"help": "run the encoding loop"})

    grad_cache: bool = field(default=False, metadata={"help": "Use gradient cache update"})
    gc_q_chunk_size: int = field(default=4)
    gc_p_chunk_size: int = field(default=32)
    zero_shot_eval: bool = field(default=False, metadata={"help": "If true, skip training"})


@dataclass
class DenseEncodingArguments(TrainingArguments):
    use_gpu: bool = field(default=False, metadata={"help": "Use GPU for encoding"})
    encoded_save_path: str = field(default=None, metadata={"help": "where to save the encode"})
    trec_save_path: str = field(default=None, metadata={"help": "where to save the trec file"})
    faiss_index_type: str = field(default="IndexFlatIP", metadata={"help" : "which type of faiss index to use, please see documentation here https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index. Options are [IndexHNSWFlat, IndexHNSWFlat] for now"})
    faiss_index_search_batch_size: int = field(default=1, metadata={"help" : "batch size to search faiss index with, seems on A100 machines if an exact match index is on gpu, batch_size > 1 is troublesome..."})
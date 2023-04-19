from datasets import load_dataset
from torch.utils.data import IterableDataset, get_worker_info
from transformers import PreTrainedTokenizer
import os

from ..arguments import DataArguments
from ..utils import find_all_markers


class InferenceDataset(IterableDataset):

    def __init__(self, tokenizer: PreTrainedTokenizer, data_args: DataArguments, is_query: bool = False, cache_dir: str = None):
        super(InferenceDataset, self).__init__()
        self.cache_dir = cache_dir
        self.processed_data_path = data_args.processed_data_path
        self.data_files = [data_args.query_path] if is_query else [data_args.corpus_path]
        self.tokenizer = tokenizer
        self.max_len = data_args.q_max_len if is_query else data_args.p_max_len
        self.proc_num = data_args.dataset_proc_num
        self.template = data_args.query_template if is_query else data_args.doc_template
        self.all_markers = find_all_markers(self.template)

    @classmethod
    def load(cls, tokenizer: PreTrainedTokenizer, data_args: DataArguments, is_query: bool = False, cache_dir: str = None):
        data_files = [data_args.query_path] if is_query else [data_args.corpus_path]
        ext = os.path.splitext(data_files[0])[1]
        if ext == ".jsonl":
            return JsonlDataset(tokenizer, data_args, is_query, cache_dir)
        elif ext in [".tsv", ".txt"]:
            return TsvDataset(tokenizer, data_args, is_query, cache_dir)
        else:
            raise ValueError("Unsupported dataset file extension {}".format(ext))

    def _process_func(self, example):
        example_id = str(example["id"])
        full_text = self.template
        for marker in self.all_markers:
            full_text = full_text.replace("<{}>".format(marker), example[marker] if example[marker] is not None else "")
        tokenized = self.tokenizer(full_text, padding='max_length', truncation=True, max_length=self.max_len)
        return {"text_id": example_id, **tokenized}

    def __iter__(self):
        return iter(self.dataset.map(self._process_func, remove_columns=self.all_columns))


class JsonlDataset(InferenceDataset):

    def __init__(self, tokenizer: PreTrainedTokenizer, data_args: DataArguments, is_query: bool = False, cache_dir: str = None):
        super(JsonlDataset, self).__init__(tokenizer, data_args, is_query, cache_dir)
        self.dataset = load_dataset("json", data_files=self.data_files, streaming=True, cache_dir=cache_dir)["train"]
        sample = list(self.dataset.take(1))[0]
        self.all_columns = sample.keys()


class TsvDataset(InferenceDataset):

    def __init__(self, tokenizer: PreTrainedTokenizer, data_args: DataArguments, is_query: bool = False, cache_dir: str = None):
        super(TsvDataset, self).__init__(tokenizer, data_args, is_query, cache_dir)
        self.all_columns = data_args.query_column_names if is_query else data_args.doc_column_names
        self.all_columns = self.all_columns.split(',')
        self.dataset = load_dataset(
            "csv", 
            data_files=self.data_files, 
            streaming=True, 
            column_names=self.all_columns,
            delimiter='\t',
            cache_dir=cache_dir
        )["train"]
        

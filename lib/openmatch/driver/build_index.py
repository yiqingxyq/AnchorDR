import os
import sys

from lib.openmatch.arguments import DataArguments
from lib.openmatch.arguments import DenseEncodingArguments as EncodingArguments
from lib.openmatch.arguments import ModelArguments
from lib.openmatch.dataset import InferenceDataset
from lib.openmatch.modeling import DenseModelForInference
from lib.openmatch.retriever import Retriever
from lib.openmatch.utils import load_stuff
from transformers import AutoConfig, AutoTokenizer, HfArgumentParser


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, EncodingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, encoding_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, encoding_args = parser.parse_args_into_dataclasses()
        model_args: ModelArguments
        data_args: DataArguments
        encoding_args: EncodingArguments

    num_labels = 1
    config, tokenizer = load_stuff(model_args, data_args, use_fast=True)

    model = DenseModelForInference.build(
        model_name_or_path=model_args.model_name_or_path,
        model_args=model_args,
        config=config,
        cache_dir=model_args.cache_dir,
    )

    corpus_dataset = InferenceDataset.load(
        tokenizer=tokenizer,
        data_args=data_args,
        is_query=False,
        cache_dir=model_args.cache_dir
    )

    Retriever.build_embeddings(model, corpus_dataset, encoding_args)


if __name__ == '__main__':
    main()

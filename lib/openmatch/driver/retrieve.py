import logging
import os
import sys

from lib.openmatch.arguments import DataArguments
from lib.openmatch.arguments import DenseEncodingArguments as EncodingArguments
from lib.openmatch.arguments import ModelArguments
from lib.openmatch.dataset import InferenceDataset
from lib.openmatch.modeling import DenseModelForInference
from lib.openmatch.retriever import Retriever
from lib.openmatch.utils import save_as_trec
from lib.openmatch.utils import load_stuff
from transformers import AutoConfig, AutoTokenizer, HfArgumentParser

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, EncodingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, encoding_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, encoding_args = parser.parse_args_into_dataclasses()
        model_args: ModelArguments
        data_args: DataArguments
        encoding_args: EncodingArguments

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if encoding_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        encoding_args.local_rank,
        encoding_args.device,
        encoding_args.n_gpu,
        bool(encoding_args.local_rank != -1),
        encoding_args.fp16,
    )
    logger.info("Encoding parameters %s", encoding_args)
    logger.info("MODEL parameters %s", model_args)

    num_labels = 1
    config, tokenizer = load_stuff(model_args, data_args)
    model = DenseModelForInference.build(
        model_name_or_path=model_args.model_name_or_path,
        model_args=model_args,
        config=config,
        cache_dir=model_args.cache_dir,
    )

    query_dataset = InferenceDataset.load(
        tokenizer=tokenizer,
        data_args=data_args,
        is_query=True,
        cache_dir=model_args.cache_dir
    )

    retriever = Retriever.from_embeddings(model, encoding_args)
    result = retriever.retrieve(query_dataset)
    if encoding_args.local_process_index == 0:
        trec_save_dir=os.path.dirname(encoding_args.trec_save_path)
        if not os.path.exists(trec_save_dir):
            os.mkdir(trec_save_dir)
        save_as_trec(result, encoding_args.trec_save_path)


if __name__ == '__main__':
    main()
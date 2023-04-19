from argparse import ArgumentParser
from transformers import AutoTokenizer
import os
import random
from tqdm import tqdm
import json
from datetime import datetime
from multiprocessing import Pool
from lib.openmatch.utils import SimpleTrainPreProcessor as TrainDataPreProcessor

from lib.openmatch.utils import load_stuff

OUTPUT_TEXT_FILE_NAME = "split"
OUTPUT_ENCODED_FILE_NAME = "encoded_split"

def load_ranking(rank_file, relevance, n_sample, depth):
    with open(rank_file) as rf:
        lines = iter(rf)
        q_0, _, p_0, _, _, _ = next(lines).strip().split()

        curr_q = q_0
        negatives = [] if p_0 in relevance[q_0] else [p_0]

        while True:
            try:
                q, _, p, _, _, _ = next(lines).strip().split()
                if q != curr_q:
                    negatives = negatives[:depth]
                    random.shuffle(negatives)
                    yield curr_q, relevance[curr_q], negatives[:n_sample]
                    curr_q = q
                    negatives = [] if p in relevance[q] else [p]
                else:
                    if p not in relevance[q]:
                        negatives.append(p)
            except StopIteration:
                negatives = negatives[:depth]
                random.shuffle(negatives)
                yield curr_q, relevance[curr_q], negatives[:n_sample]
                return


random.seed(datetime.now())
parser = ArgumentParser()
parser.add_argument('--tokenizer_name', required=True)
parser.add_argument('--hn_file', required=True)
parser.add_argument('--qrels', required=True)
parser.add_argument('--queries', required=True)
parser.add_argument('--collection', required=True)
parser.add_argument('--save_to', required=True)
parser.add_argument('--template', type=str, default=None)

parser.add_argument('--use_title', type=bool, default=True)
parser.add_argument('--truncate', type=int, default=512)
parser.add_argument('--n_sample', type=int, default=30)
parser.add_argument('--depth', type=int, default=200)
parser.add_argument('--mp_chunk_size', type=int, default=5000)
parser.add_argument('--shard_size', type=int, default=50000)

parser.add_argument('--use_converted', default=False, action='store_true')

args = parser.parse_args()

counter = 0
total_cntr = 0
shard_id = 0
text_data_file, encoded_data_file = None, None
os.makedirs(args.save_to, exist_ok=True)

qrel = TrainDataPreProcessor.read_qrel(args.qrels)
print(f"build_hn.py: loaded qrel file from {args.qrels}, containining {len(qrel)} entries")

if args.use_converted:
    from transformers.models.t5 import FairseqT5Tokenizer
    tokenizer = FairseqT5Tokenizer.from_pretrained(
        args.tokenizer_name,
        use_fast=True,
    )
else:
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)
processor = TrainDataPreProcessor(
    query_file=args.queries,
    collection_file=args.collection,
    tokenizer=tokenizer,
    max_length=args.truncate,
    template=args.template,
    use_title=args.use_title
)

pbar = tqdm(load_ranking(args.hn_file, qrel, args.n_sample, args.depth), desc="building hard negatives", total=int(len(qrel)/args.mp_chunk_size))
with Pool() as p:
    buffer_text, buffer_encoded = [], []
    for x in p.imap(processor.process_one, pbar, chunksize=args.mp_chunk_size):
        plain_text_example, tokenized_example = x[0], x[1]
        processor.validate_output(json.loads(plain_text_example))
        processor.validate_output(json.loads(tokenized_example))
        counter += 1
        total_cntr += 1
        
        if text_data_file is None:
            text_data_file = open(os.path.join(args.save_to, f'{OUTPUT_TEXT_FILE_NAME}-{shard_id:02d}.hn.jsonl'), 'w')
        if encoded_data_file is None:
            encoded_data_file = open(os.path.join(args.save_to, f'{OUTPUT_ENCODED_FILE_NAME}-{shard_id:02d}.hn.jsonl'), 'w')

        buffer_text.append(plain_text_example)
        buffer_encoded.append(plain_text_example)

        if counter == args.shard_size:
            for x in buffer_text:
                text_data_file.write(x + '\n') # save plain text json dictionary of the training triple
            for x in buffer_encoded:
                encoded_data_file.write(x + '\n') # save the encoded version
            buffer_text, buffer_encoded = [], []
            text_data_file.close()
            encoded_data_file.close()
            text_data_file = None
            encoded_data_file = None
            shard_id += 1
            counter = 0
            pbar.set_description(f"saving to {OUTPUT_TEXT_FILE_NAME}-{shard_id:02d}.jsonl")

# cleanup any stragglers
if text_data_file is not None:
    for x in buffer_text:
        text_data_file.write(x + '\n') # save plain text json dictionary of the training triple
    for x in buffer_encoded:
        encoded_data_file.write(x + '\n') # save the encoded version
    text_data_file.close()
    encoded_data_file.close()

print(f"Done building hard negatives files for training, processed {total_cntr} examples total across {shard_id+1} shards")

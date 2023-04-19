from argparse import ArgumentParser
from transformers import AutoTokenizer
import os
import random
from tqdm import tqdm
import json
from datetime import datetime
from multiprocessing import Pool

from lib.openmatch.utils import SimpleTrainPreProcessor as TrainDataPreProcessor
from lib.openmatch.conversion_utils.tokenizer_conversion import tokenizer_builder 
OUTPUT_TEXT_FILE_NAME = "split"
OUTPUT_ENCODED_FILE_NAME = "encoded_split"

random.seed(datetime.now())
parser = ArgumentParser()
parser.add_argument('--tokenizer_name', required=True)
parser.add_argument('--negative_file', required=True)
parser.add_argument('--qrels', required=True)
parser.add_argument('--queries', required=True)
parser.add_argument('--collection', required=True)
parser.add_argument('--save_to', required=True)
parser.add_argument('--template', type=str, default=None)
parser.add_argument('--use_title', type=bool, default=False)
parser.add_argument('--if_truncation', type=bool, default=False)
parser.add_argument('--truncate', type=int, default=512)
parser.add_argument('--n_sample', type=int, default=30)
parser.add_argument('--mp_chunk_size', type=int, default=5000)
parser.add_argument('--shard_size', type=int, default=50000)
parser.add_argument('--n_processes', type=int, default=1)
parser.add_argument('--use_converted', action='store_true')
parser.add_argument('--vocab_path', type=str)
args = parser.parse_args()

counter = 0
total_cntr = 0
shard_id = 0
text_data_file, encoded_data_file = None, None
os.makedirs(args.save_to, exist_ok=True)

qrel = TrainDataPreProcessor.read_qrel(args.qrels)
print(f"build_train.py: loaded qrel file from {args.qrels}, containining {len(qrel)} entries")

tokenizer = tokenizer_builder(args.tokenizer_name, args.vocab_path, use_converted=args.use_converted)

processor = TrainDataPreProcessor(
    query_file=args.queries,
    collection_file=args.collection,
    tokenizer=tokenizer,
    max_length=args.truncate,
    template=args.template,
    use_title=args.use_title,
    truncation=args.if_truncation,
)

### each line of args.negative_file is a query_id \t comma-separated-list of negative doc ids
### iterate through each line and replace the doc ids with the query text, lookup the 
### positive doc id for that query, and then replace all doc_ids with their text
### output two types of things: a json of each <query, pos doc, [neg docs]> in both
### plain text form, and pre-encoded with the token ids from the tokenizer
with open(args.negative_file) as nf:
    buffer_text, buffer_encoded = [], []
    def read_negative_files_line(l):
        q, nn = l.strip().split('\t')
        nn = nn.split(',')
        random.shuffle(nn)
        return q, qrel[q], nn[:args.n_sample]

    pbar = tqdm(map(read_negative_files_line, nf))
    with Pool() as p:
        for x in p.imap(processor.process_one, pbar, chunksize=args.mp_chunk_size):
            plain_text_example, tokenized_example = x[0], x[1]
            processor.validate_output(json.loads(plain_text_example))
            processor.validate_output(json.loads(tokenized_example))
            counter += 1
            total_cntr += 1
            
            if text_data_file is None:
                text_data_file = open(os.path.join(args.save_to, f'{OUTPUT_TEXT_FILE_NAME}-{shard_id:02d}.jsonl'), 'w')
            if encoded_data_file is None:
                encoded_data_file = open(os.path.join(args.save_to, f'{OUTPUT_ENCODED_FILE_NAME}-{shard_id:02d}.jsonl'), 'w')

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

print(f"Done building train data files, processed {total_cntr} examples total across {shard_id+1} shards")
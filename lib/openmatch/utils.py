import csv
import json
from dataclasses import dataclass
from typing import Tuple, List, Dict
import os

import datasets
from transformers import PreTrainedTokenizer
from transformers import AutoConfig, AutoTokenizer, HfArgumentParser, set_seed
from transformers.models.t5 import FairseqT5Tokenizer

TITLE_PLACEHOLDER = "<title>"
TEXT_PLACEHOLDER = "<text>"
QUERY_MAX_LENGTH = 32

def is_rank_0() -> bool:
    return int(os.environ.get("RANK", "0")) == 0

def load_stuff(model_args, data_args, use_fast=False):
    num_labels = 1
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
    )
    if model_args.use_converted:
        tokenizer = FairseqT5Tokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=use_fast,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=use_fast,
        )
    
    return config, tokenizer

@dataclass
class SimpleTrainPreProcessor:
    query_file: str
    collection_file: str
    columns = ['text_id', 'title', 'text']
    tokenizer: PreTrainedTokenizer = None
    max_length: int = 512
    title_field: str = 'title'
    text_field: str = 'text'
    template: str = f"Text: {TEXT_PLACEHOLDER}"
    use_title : bool = False
    truncation: bool = False

    def __post_init__(self):
        assert TEXT_PLACEHOLDER in self.template, f"TEXT_PLACEHOLDER=\"{TEXT_PLACEHOLDER}\" must be in template somewhere"
        if TITLE_PLACEHOLDER in self.template:
            assert self.use_title, f"if {TITLE_PLACEHOLDER} is in the template={self.template}, then --use_title must be set to True"
        if self.use_title:
            assert TITLE_PLACEHOLDER in self.template, f"TITLE_PLACEHOLDER=\"{TITLE_PLACEHOLDER}\" must be in template if use_title=True"
            assert len(self.columns) == 3 and self.columns[1] == self.title_field, "self.columns must = [text_id, title, text] or something to that effect if use_title=True"
        
        self.queries = self.read_queries(self.query_file)
        self.collection = datasets.load_dataset(
            'csv',
            data_files=self.collection_file,
            column_names=self.columns,
            delimiter='\t',
        )['train']

    @staticmethod
    def validate_output(d : dict):
        assert isinstance(d, dict), f"instance is not a dictionary, it's a {type(d)}"
        assert all([field in d for field in ["query", "positives", "negatives"]]), f"ERROR: {d} is not valid"

    @staticmethod
    def read_queries(queries):
        qmap = {}
        with open(queries) as f:
            for l in f:
                qid, qry = l.strip().split('\t')
                assert isinstance(qid, str) and isinstance(qry, str)
                qmap[qid] = qry
        return qmap

    @staticmethod
    def read_qrel(relevance_file):
        qrel = {}
        with open(relevance_file, encoding='utf8') as f:
            tsvreader = csv.reader(f, delimiter="\t")
            for [topicid, _, docid, rel] in tsvreader:
                assert rel == "1"
                if topicid in qrel:
                    qrel[topicid].append(docid)
                else:
                    qrel[topicid] = [docid]
        return qrel

    def encode(self, s : str, max_length = max_length, truncation = True):
        s_encoded = self.tokenizer.encode(
            s,
            add_special_tokens=False,
            max_length=max_length,
            truncation=truncation
        )
        return s_encoded

    def get_query_surface_form(self, q : str):
        assert q in self.queries
        # query_encoded = self.tokenizer.encode(
        #     self.queries[q],
        #     add_special_tokens=False,
        #     max_length=self.max_length,
        #     truncation=True
        # )
        return self.queries[q]

    def get_document_surface_form(self, doc_id : str):
        """
            Given a document id, turn a dictionary of the fields of that doc e.g. 
            
            self.collection[doc_id] = {"text_id" : blah, "url" : blah...}
            
            into a flat string representation of the document. How you do this
            is kind of a scientific methodology question but simply put you can do:

            content = "Url: aka.ms/hello Title: hello there Body: well hey..."
        """
        entry = self.collection[int(doc_id)]        
        # if self.template is None:
        #     content = title + self.tokenizer.sep_token + body
        content = self.template[:] # copy
        if self.use_title:
            assert TITLE_PLACEHOLDER in content
            title = entry[self.title_field]
            title = "" if title is None else title
            content = content.replace(TITLE_PLACEHOLDER, title)
        body = entry[self.text_field]
        if self.truncation:
            body = ' '.join(body.split()[:self.max_length])
        content = content.replace(TEXT_PLACEHOLDER, body)
        return content

    def process_one(self, train : Tuple[str, List[str], List[str]]):
        """
            @param train: a triple of a queryID, a list of postive docIds, followed by a list of 
                negative doc ids e.g.
                ('1000094', ['5399011'], ['3616757', '6704164', '4847155', '6478745', '1680191', '3046890', '496887', '8196441', '3987870', '2861309'])

        """
        q, pp, nn = train
        train_example = {
            'query': self.get_query_surface_form(q),
            'positives': [self.get_document_surface_form(p) for p in pp],
            'negatives': [self.get_document_surface_form(n) for n in nn],
        }

        ### use the tokenizer to encode all the fields
        train_example_encoded = {
            'query': self.encode(train_example["query"], max_length=QUERY_MAX_LENGTH),
            'positives': [self.encode(doc, max_length=self.max_length, truncation=self.truncation) for doc in train_example["positives"]],
            'negatives': [self.encode(doc, max_length=self.max_length, truncation=self.truncation) for doc in train_example["negatives"]],
        }


        self.validate_output(train_example)
        self.validate_output(train_example_encoded)
        o1 = json.dumps(train_example)
        o2 = json.dumps(train_example_encoded)
        
        return o1, o2


# @dataclass
# class SimpleCollectionPreProcessor:
#     tokenizer: PreTrainedTokenizer
#     separator: str = '\t'
#     max_length: int = 128

#     def process_line(self, line: str):
#         xx = line.strip().split(self.separator)
#         text_id, text = xx[0], xx[1:]
#         text_encoded = self.tokenizer.encode(
#             self.tokenizer.sep_token.join(text),
#             add_special_tokens=False,
#             max_length=self.max_length,
#             truncation=True
#         )
#         encoded = {
#             'text_id': text_id,
#             'text': text_encoded
#         }
#         return json.dumps(encoded)


def save_as_trec(rank_result: Dict[str, Dict[str, float]], output_path: str, run_id: str = "OpenMatch"):
    """
    Save the rank result as TREC format:
    <query_id> Q0 <doc_id> <rank> <score> <run_id>
    """
    with open(output_path, "w") as f:
        for qid in rank_result:
            # sort the results by score
            sorted_results = sorted(rank_result[qid].items(), key=lambda x: x[1], reverse=True)
            for i, (doc_id, score) in enumerate(sorted_results):
                f.write("{} Q0 {} {} {} {}\n".format(qid, doc_id, i + 1, score, run_id))
                

def find_all_markers(template: str):
    """
    Find all markers' names (quoted in "<>") in a template.
    """
    markers = []
    start = 0
    while True:
        start = template.find("<", start)
        if start == -1:
            break
        end = template.find(">", start)
        if end == -1:
            break
        markers.append(template[start + 1:end])
        start = end + 1
    return markers
import glob
import os
import pickle
# from contextlib import nullcontext
from typing import Dict, List
import logging

import faiss
import numpy as np
import torch
from torch.cuda import amp
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm
from transformers.trainer_pt_utils import IterableDatasetShard

from ..arguments import DenseEncodingArguments as EncodingArguments
from ..dataset import EncodeCollator
from ..modeling import DenseModelForInference, DenseOutput

logger = logging.getLogger(__name__)

class Retriever:

    def __init__(self, model: DenseModelForInference, corpus_dataset: IterableDataset, args: EncodingArguments):
        logger.info("Initializing retriever")
        self.model = model
        self.corpus_dataset = corpus_dataset
        self.args = args
        self.doc_lookup = []
        self.query_lookup = []

        self.model = model.to(self.args.device)
        self.model.eval()

    def _initialize_faiss_index(self, dim: int):
        self.index = None
        if self.args.process_index == 0:
            if self.args.faiss_index_type == "IndexFlatIP":
                cpu_index = faiss.IndexFlatIP(dim)
            elif self.args.faiss_index_type == "IndexHNSWFlat":
                cpu_index = faiss.IndexHNSWFlat(dim, 32)   
            else:
                raise RuntimeError(f"Retriever._initialize_faiss_index(), self.args.faiss_index_type={self.args.faiss_index_type} is invalid")

            logger.info(f"Created faiss.{self.args.faiss_index_type} index with options --use_gpu={self.args.use_gpu} and --faiss_index_search_batch_size={self.args.faiss_index_search_batch_size}")     
            self.index = cpu_index

    def _move_index_to_gpu(self):
        assert self.args.faiss_index_type == "IndexFlatIP"
        if self.args.process_index == 0:
            logger.info("Moving index to GPU using IndexShards scheme")
            ngpu = faiss.get_num_gpus()
            gpu_resources = []
            for i in range(ngpu):
                res = faiss.StandardGpuResources()
                gpu_resources.append(res)
            logger.debug(f"gpu resources: {gpu_resources} for ngpu={ngpu}")
            co = faiss.GpuMultipleClonerOptions() # see documentation: https://github.com/facebookresearch/faiss/wiki/Faiss-on-the-GPU#using-multiple-gpus
            co.shard = True
            co.usePrecomputed = False
            if self.args.fp16:
                co.useFloat16 = True
            vres = faiss.GpuResourcesVector()
            vdev = faiss.Int32Vector()
            for i in range(0, ngpu):
                vdev.push_back(i)
                vres.push_back(gpu_resources[i])
            self.index.referenced_objects = gpu_resources
            self.index = faiss.index_cpu_to_gpu_multiple(vres, vdev, self.index, co)

    def doc_embedding_inference(self):
        # Note: during evaluation, there's no point in wrapping the model
        # inside a DistributedDataParallel as we'll be under `no_grad` anyways.
        if self.corpus_dataset is None:
            raise ValueError("No corpus dataset provided")
        logger.info(f"world size={self.args.world_size} for doc_embedding_inference()")
        if self.args.world_size > 1:
            self.corpus_dataset = IterableDatasetShard(
                self.corpus_dataset,
                batch_size=self.args.per_device_eval_batch_size,
                drop_last=False,
                num_processes=self.args.world_size,
                process_index=self.args.process_index
            )
        dataloader = DataLoader(
            self.corpus_dataset,
            batch_size=self.args.eval_batch_size,
            collate_fn=EncodeCollator(),
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
        encoded = []
        lookup_indices = []
        for (batch_ids, batch) in tqdm(dataloader, disable=self.args.local_process_index > 0, desc="Retriever.doc_embedding_inference(): encoding documents"):
            lookup_indices.extend(batch_ids)
            with amp.autocast(): # if self.args.fp16 else nullcontext():
                with torch.no_grad():
                    for k, v in batch.items():
                        batch[k] = v.to(self.args.device)
                    model_output: DenseOutput = self.model(passage=batch)
                    encoded.append(model_output.p_reps.cpu().detach().numpy())
        encoded = np.concatenate(encoded)

        os.makedirs(self.args.output_dir, exist_ok=True)
        logger.info(f"writing cached embeddings {encoded.shape} of type {encoded.dtype} to {self.args.output_dir}")
        with open(os.path.join(self.args.output_dir, "embeddings.corpus.rank.{}".format(self.args.process_index)), 'wb') as f:
            pickle.dump((encoded, lookup_indices), f, protocol=4)

        del encoded
        del lookup_indices
        
        if self.args.world_size > 1:
            torch.distributed.barrier()
        logger.info("Done building document embedding index")

    def init_index_and_add(self, partition: str = None):
        dim = 0
        partitions = [partition] if partition is not None else glob.glob(os.path.join(self.args.output_dir, "embeddings.corpus.rank.*"))
        logger.debug(f"initializing index from partitions: {partitions}, to_gpu={self.args.use_gpu}")
        for i, part in tqdm(enumerate(partitions), desc=f"initializing index from {len(partitions)} partitions:"):
            with open(part, 'rb') as f:
                data = pickle.load(f)
            encoded = data[0]
            # logger.info(f"encoded shard of type {type(encoded)} {encoded.dtype}")
            lookup_indices = data[1]
            if i == 0: # first figure out dim of loaded vectors
                dim = encoded.shape[1]
                self._initialize_faiss_index(dim)
            ### NOTE: encoded vectors must be contiguous and type float32
            encoded = np.ascontiguousarray(encoded, dtype="float32")
            self.index.add(encoded)
            self.doc_lookup.extend(lookup_indices)
        logger.info(f"Finished constructing index of {self.index.ntotal} documents as {encoded.dtype} vectors w/ dim={dim}")
        if self.args.use_gpu and self.args.faiss_index_type == "IndexFlatIP":
            self._move_index_to_gpu()

    @classmethod
    def build_all(cls, model: DenseModelForInference, corpus_dataset: IterableDataset, args: EncodingArguments):
        retriever = cls(model, corpus_dataset, args)
        retriever.doc_embedding_inference()
        if args.process_index == 0:
            retriever.init_index_and_add()
        if args.world_size > 1:
            torch.distributed.barrier()
        return retriever

    @classmethod
    def build_embeddings(cls, model: DenseModelForInference, corpus_dataset: IterableDataset, args: EncodingArguments):
        retriever = cls(model, corpus_dataset, args)
        retriever.doc_embedding_inference()
        return retriever

    @classmethod
    def from_embeddings(cls, model: DenseModelForInference, args: EncodingArguments):
        retriever = cls(model, None, args)
        if args.process_index == 0:
            retriever.init_index_and_add()
        if args.world_size > 1:
            torch.distributed.barrier()
        return retriever

    def reset_index(self):
        if self.index:
            self.index.reset()
        self.doc_lookup = []
        self.query_lookup = []

    def query_embedding_inference(self, query_dataset: IterableDataset) -> torch.Tensor:
        assert self.args.world_size == 1
        if self.args.world_size > 1:
            self.query_dataset = IterableDatasetShard(
                query_dataset,
                batch_size=self.args.per_device_eval_batch_size,
                drop_last=False,
                num_processes=self.args.world_size,
                process_index=self.args.process_index
            )
        dataloader = DataLoader(
            query_dataset,
            batch_size=self.args.eval_batch_size,
            collate_fn=EncodeCollator(),
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
        encoded = []
        lookup_indices = []
        for (batch_ids, batch) in tqdm(dataloader, disable=self.args.local_process_index > 0, desc="Retriever.query_embedding_inference()"):
            lookup_indices.extend(batch_ids)
            with amp.autocast(): # if self.args.fp16 else nullcontext():
                with torch.no_grad():
                    for k, v in batch.items():
                        batch[k] = v.to(self.args.device)
                    model_output: DenseOutput = self.model(query=batch)
                    encoded.append(model_output.q_reps.cpu().detach().numpy().astype('float32'))
        
        ### NOTE: encoded vectors must be contiguous and type float32
        encoded = np.concatenate(encoded)

        with open(os.path.join(self.args.output_dir, "embeddings.query.rank.{}".format(self.args.process_index)), 'wb') as f:
            pickle.dump((encoded, lookup_indices), f, protocol=4)
        
        # print(f"process {self.args.local_process_index} waiting in query_embedding_inference()")
        if self.args.world_size > 1:
            torch.distributed.barrier()

        return encoded
        # print(f"process {self.args.local_process_index} PAST BARRIER in query_embedding_inference()")

    def search(self, topk: int = 100, query_vectors=None):
        assert self.args.world_size == 1
        logger.info("Retriever.search(): Searching index...")

        if self.index is None:
            raise ValueError("Index is not initialized")
        logger.info(f"Retriever.search(): Loading cached query embeddings from {self.args.output_dir}")
        encoded = []
        for i in range(self.args.world_size):
            with open(os.path.join(self.args.output_dir, "embeddings.query.rank.{}".format(i)), 'rb') as f:
                data = pickle.load(f)
            lookup_indices = data[1]
            encoded.append(data[0])
            self.query_lookup.extend(lookup_indices)
        ### NOTE: encoded vectors must be contiguous and type float32
        encoded = np.ascontiguousarray(np.concatenate(encoded), dtype="float32") # size=num queries by dim
        logger.info(f"Retriever.search(): {encoded.shape} query encodings of type {type(encoded)} {encoded.dtype} to search in index for...")

        return_dict = {}
        D, I  = [], []
        # NOTE: for reasons which are very frustrating, self.args.faiss_index_search_batch_size must be set to 1 
        # using exact NN search on GPU on A100 machines. 
        # please see https://github.com/microsoft/UnivSearchDev/blob/corby/t5-large-ance/projects/T5-ANCE/retrieval_demo.ipynb
        # for more investigation on how batch_size > 1 failed
        for idx in tqdm(range(0, encoded.shape[0], self.args.faiss_index_search_batch_size), desc=f"Retriever.search() - searching for nearest neighbors with batch_size={self.args.faiss_index_search_batch_size}..."):
            d, i = self.index.search(encoded[idx : idx + self.args.faiss_index_search_batch_size, :], int(topk))
            D.extend(d)
            I.extend(i)
        original_indices = np.array(self.doc_lookup)[I]
        q = 0
        for scores_per_q, doc_indices_per_q in tqdm(zip(D, original_indices), desc=f"Retriever.search(): query lookup"):
            qid = str(self.query_lookup[q])
            return_dict[qid] = {}
            for doc_index, score in zip(doc_indices_per_q, scores_per_q):
                return_dict[qid][str(doc_index)] = float(score)
            q += 1

        return return_dict

    def retrieve(self, query_dataset: IterableDataset, topk: int = 100):
        query_vectors = self.query_embedding_inference(query_dataset)
        # print(f"process {self.args.local_process_in[dex} IN RETRIEVE AFTER query_embedding_inference")

        results = {}
        if self.args.process_index == 0:
            # print(f"process {self.args.local_process_index} DOING SEARCH")
            results = self.search(topk, query_vectors=query_vectors)
        if self.args.world_size > 1:
            # print(f"process {self.args.local_process_index} WAITING while search is happening in process 0")
            torch.distributed.barrier()
        # print(f"process {self.args.local_process_index} DONE WITH RETRIEVE")
        return results


def merge_retrieval_results_by_score(results: List[Dict[str, Dict[str, float]]], topk: int = 100):
    """
    Merge retrieval results from multiple partitions of document embeddings and keep topk.
    """
    merged_results = {}
    for result in results:
        for qid in result:
            if qid not in merged_results:
                merged_results[qid] = {}
            for doc_id in result[qid]:
                if doc_id not in merged_results[qid]:
                    merged_results[qid][doc_id] = result[qid][doc_id]
    for qid in merged_results:
        merged_results[qid] = {k: v for k, v in sorted(merged_results[qid].items(), key=lambda x: x[1], reverse=True)[:topk]}
    return merged_results


class SuccessiveRetriever(Retriever):

    def __init__(self, model: DenseModelForInference, corpus_dataset: IterableDataset, args: EncodingArguments):
        super().__init__(model, corpus_dataset, args)

    @classmethod
    def from_embeddings(cls, model: DenseModelForInference, args: EncodingArguments):
        retriever = cls(model, None, args)
        return retriever

    def retrieve(self, query_dataset: IterableDataset, topk: int = 100):
        self.query_embedding_inference(query_dataset)

        del self.model
        torch.cuda.empty_cache()
        final_result = {}
        if self.args.process_index == 0:
            all_partitions = glob.glob(os.path.join(self.args.output_dir, "embeddings.corpus.rank.*"))
            for partition in all_partitions:
                logger.info("Loading partition {}".format(partition))
                self.init_index_and_add(partition)
                cur_result = self.search(topk)
                self.reset_index()
                final_result = merge_retrieval_results_by_score([final_result, cur_result], topk)
        if self.args.world_size > 1:
            torch.distributed.barrier()
        return final_result

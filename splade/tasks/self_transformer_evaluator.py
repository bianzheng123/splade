import json
import os
import pickle
import time
from collections import defaultdict

import numba
import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from ..indexing.inverted_index import IndexDictOfArray
from ..losses.regularization import L0
from ..tasks.base.evaluator import Evaluator
from ..utils.utils import makedir, to_list


class SelfTokenizer():
    def __init__(self, tokenizer_type, max_length):
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_type)

    def tokenize(self, batch):
        """
        batch is a list of tuples, each tuple has 2 (text) items (id_, doc)
        """
        id_, d = zip(*batch)
        processed_passage = self.tokenizer(list(d),
                                           add_special_tokens=True,
                                           padding="longest",  # pad to max sequence length in batch
                                           truncation="longest_first",  # truncates to self.max_length
                                           max_length=self.max_length,
                                           return_attention_mask=True)
        return {**{k: torch.tensor(v) for k, v in processed_passage.items()},
                "id": torch.tensor([int(i) for i in id_], dtype=torch.long)}


class SelfSparseRetrieval(Evaluator):
    """retrieval from SparseIndexing
    """

    @staticmethod
    def select_topk(filtered_indexes, scores, k):
        if len(filtered_indexes) > k:
            sorted_ = np.argpartition(scores, k)[:k]
            filtered_indexes, scores = filtered_indexes[sorted_], -scores[sorted_]
        else:
            scores = -scores
        return filtered_indexes, scores

    @staticmethod
    @numba.njit(nogil=True, parallel=True, cache=True)
    def numba_score_float(inverted_index_ids: numba.typed.Dict,
                          inverted_index_floats: numba.typed.Dict,
                          indexes_to_retrieve: np.ndarray,
                          query_values: np.ndarray,
                          threshold: float,
                          size_collection: int):
        scores = np.zeros(size_collection, dtype=np.float32)  # initialize array with size = size of collection
        n = len(indexes_to_retrieve)
        for _idx in range(n):
            local_idx = indexes_to_retrieve[_idx]  # which posting list to search
            query_float = query_values[_idx]  # what is the value of the query for this posting list
            retrieved_indexes = inverted_index_ids[local_idx]  # get indexes from posting list
            retrieved_floats = inverted_index_floats[local_idx]  # get values from posting list
            for j in numba.prange(len(retrieved_indexes)):
                scores[retrieved_indexes[j]] += query_float * retrieved_floats[j]
        filtered_indexes = np.argwhere(scores > threshold)[:, 0]  # ideally we should have a threshold to filter
        # unused documents => this should be tuned, currently it is set to 0
        return filtered_indexes, -scores[filtered_indexes]

    def __init__(self, model, config, tokenizer: SelfTokenizer, dim_voc, dataset_name=None, index_d=None,
                 compute_stats=False, is_beir=False,
                 **kwargs):
        super().__init__(model, config, **kwargs)
        self.tokenizer = tokenizer
        assert ("index_dir" in config and index_d is None) or (
                "index_dir" not in config and index_d is not None)
        if "index_dir" in config:
            self.sparse_index = IndexDictOfArray(config["index_dir"], dim_voc=dim_voc)
            self.doc_ids = pickle.load(open(os.path.join(config["index_dir"], "doc_ids.pkl"), "rb"))
        else:
            self.sparse_index = index_d["index"]
            self.doc_ids = index_d["ids_mapping"]
            for i in range(dim_voc):
                # missing keys (== posting lists), causing issues for retrieval => fill with empty
                if i not in self.sparse_index.index_doc_id:
                    self.sparse_index.index_doc_id[i] = np.array([], dtype=np.int32)
                    self.sparse_index.index_doc_value[i] = np.array([], dtype=np.float32)
        # convert to numba
        self.numba_index_doc_ids = numba.typed.Dict()
        self.numba_index_doc_values = numba.typed.Dict()
        for key, value in self.sparse_index.index_doc_id.items():
            self.numba_index_doc_ids[key] = value
        for key, value in self.sparse_index.index_doc_value.items():
            self.numba_index_doc_values[key] = value
        self.out_dir = os.path.join(config["out_dir"], dataset_name) if (dataset_name is not None and not is_beir) \
            else config["out_dir"]
        self.doc_stats = index_d["stats"] if (index_d is not None and compute_stats) else None
        self.compute_stats = compute_stats
        if self.compute_stats:
            self.l0 = L0()

    def retrieve(self, q_collection, top_k, name=None, return_d=False, id_dict=False, threshold=0):
        makedir(self.out_dir)
        if self.compute_stats:
            makedir(os.path.join(self.out_dir, "stats"))
        res = defaultdict(dict)
        if self.compute_stats:
            stats = defaultdict(float)

        # unit of time is second
        retrieval_time_l = []
        encode_time_l = []
        search_time_l = []

        with torch.no_grad():
            n_query = len(q_collection)

            # for warm-up
            for local_qID in tqdm(range(2)):
                # get the query id, only one query per batch
                batch = self.tokenizer.tokenize([q_collection[local_qID]])
                q_id = to_list(batch["id"])[0]
                if id_dict:
                    q_id = id_dict[q_id]
                inputs = {k: v for k, v in batch.items() if k not in {"id"}}
                for k, v in inputs.items():
                    inputs[k] = v.to(self.device)
                query = self.model(q_kwargs=inputs)["q_rep"]  # we assume ONE query per batch here
                if self.compute_stats:
                    stats["L0_q"] += self.l0(query).item()

                # TODO: batched version for retrieval
                # row, col means the position of the non-zero values in the query
                # since there is only one query, so row is useless
                row, col = torch.nonzero(query, as_tuple=True)
                # get the value of each non-zero position
                values = query[to_list(row), to_list(col)]

                # numba_index_doc_ids and numba_index_doc_values are an inverted file
                # the key is the i-th dimension, the value stores the document ID of non-zero value (numba_index_doc_ids) and its value (numba_index_doc_values)
                # self.sparse_index.nb_docs() means the number of document
                filtered_indexes, scores = self.numba_score_float(self.numba_index_doc_ids,
                                                                  self.numba_index_doc_values,
                                                                  col.cpu().numpy(),
                                                                  values.cpu().numpy().astype(np.float32),
                                                                  threshold=threshold,
                                                                  size_collection=self.sparse_index.nb_docs())
                # threshold set to 0 by default, could be better
                filtered_indexes, scores = self.select_topk(filtered_indexes, scores, k=top_k)

            for local_qID in tqdm(range(n_query)):
                # get the query id, only one query per batch
                start = time.time_ns()
                batch = self.tokenizer.tokenize([q_collection[local_qID]])
                q_id = to_list(batch["id"])[0]
                if id_dict:
                    q_id = id_dict[q_id]
                inputs = {k: v for k, v in batch.items() if k not in {"id"}}
                for k, v in inputs.items():
                    inputs[k] = v.to(self.device)
                query = self.model(q_kwargs=inputs)["q_rep"]  # we assume ONE query per batch here
                if self.compute_stats:
                    stats["L0_q"] += self.l0(query).item()
                encode_time = (time.time_ns() - start) * 1e-6

                start_search = time.time_ns()
                # TODO: batched version for retrieval
                # row, col means the position of the non-zero values in the query
                # since there is only one query, so row is useless
                row, col = torch.nonzero(query, as_tuple=True)
                # get the value of each non-zero position
                values = query[to_list(row), to_list(col)]

                # numba_index_doc_ids and numba_index_doc_values are an inverted file
                # the key is the i-th dimension, the value stores the document ID of non-zero value (numba_index_doc_ids) and its value (numba_index_doc_values)
                # self.sparse_index.nb_docs() means the number of document
                filtered_indexes, scores = self.numba_score_float(self.numba_index_doc_ids,
                                                                  self.numba_index_doc_values,
                                                                  col.cpu().numpy(),
                                                                  values.cpu().numpy().astype(np.float32),
                                                                  threshold=threshold,
                                                                  size_collection=self.sparse_index.nb_docs())
                # threshold set to 0 by default, could be better
                filtered_indexes, scores = self.select_topk(filtered_indexes, scores, k=top_k)
                for id_, sc in zip(filtered_indexes, scores):
                    res[str(q_id)][str(self.doc_ids[id_])] = float(sc)
                search_time = (time.time_ns() - start_search) * 1e-6
                retrieval_time = (time.time_ns() - start) * 1e-6
                retrieval_time_l.append(retrieval_time)
                encode_time_l.append(encode_time)
                search_time_l.append(search_time)

        if self.compute_stats:
            stats = {key: value / len(q_collection) for key, value in stats.items()}

        time_stat = {"retrieval_time(ms)": retrieval_time_l, 'encode_time(ms)': encode_time_l,
                     'search_time(ms)': search_time_l}
        with open(os.path.join(self.out_dir, "time_stats.json"),
                  "w") as handler:
            json.dump(time_stat, handler)

        if self.compute_stats:
            with open(os.path.join(self.out_dir, "stats",
                                   "q_stats{}.json".format("_iter_{}".format(name) if name is not None else "")),
                      "w") as handler:
                json.dump(stats, handler)
            if self.doc_stats is not None:
                with open(os.path.join(self.out_dir, "stats",
                                       "d_stats{}.json".format("_iter_{}".format(name) if name is not None else "")),
                          "w") as handler:
                    json.dump(self.doc_stats, handler)
        with open(os.path.join(self.out_dir, "run{}.json".format("_iter_{}".format(name) if name is not None else "")),
                  "w") as handler:
            json.dump(res, handler)
        if return_d:
            out = {"retrieval": res}
            if self.compute_stats:
                out["stats"] = stats if self.doc_stats is None else {**stats, **self.doc_stats}
            return out

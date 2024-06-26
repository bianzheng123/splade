# SPLADE - HuggingFace training

**TL; DR** We provide a new code version to train SPLADE models based on HuggingFace trainers. Compared to the original code base, it allows training SPLADE with several *hard* negatives, training with Distributed Data Parallel etc., making the overall training process more effective and efficient. 
It also differs in various aspects -- for instance, we remove the scheduler for the regularization hyperparameters, the way we compute the FLOPS, add an "anti-zero" regularization to avoid representations collapsing to zero vectors etc. 

This code is solely meant to **train** models. To index and retrieve with SPLADE, everything *remains the same*. Tested with:

```pip install torch transformers==4.29.2  hydra-core faiss-cpu pytest numba h5py pytrec_eval tensorboard  accelerate  matplotlib```

## Data format

To train models, four files are needed (used in the `src/hf/datasets.py` file):

* **collection file** : *tsv* file, `ID\tDATA`, contains the (text) documents
* **query file** : *tsv* file, `ID\tDATA`, contains the (text) queries
* **qrel file** : *json* file, `{QID: {DID_1: rel_1, DID_2: rel_2, ...}, ...}`, ground-truth relevance
* **score (or hard-negative file)** : here, we allow for several formats
    * **saved_pkl** (resp. **pkl_dict**) : pickle file (resp. gzip), *json* format, `{QID: {DID_1: score_1, DID_2: score_2, ...}, ...}` (the latter having the format of `cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl.gz` from this [page](https://huggingface.co/datasets/sentence-transformers/msmarco-hard-negatives/tree/main), where the file is compressed and the ids are integers)
    * **trec** : trec format file (e.g., generated by Anserini)
    * **json** : dict result from a run, *json* format ; for instance to train a SPLADE model with SPLADE negatives (from a first round of training)

Note that training w/o distillation still relies on a "score" file (but the scores are not used).

## Getting started

To keep full backward compatibility with the original SPLADE code, we allow training models with [Hydra](https://hydra.cc/) configurations. The mapping between hyperparameters is done in `splade/hf/convertl2i2hf.py`.


### Toy example: training a SPLADE model
In particular, training can be launched with :

```
torchrun --nproc_per_node 2 -m splade.hf_train --config-name=config_hf_splade_l1q.yaml  config.checkpoint_dir=<chk_dir>

```

### Toy example: Indexing a collection with SPLADE model
After training, indexing and retrieval can be launched with :

```
python -m splade.index --config-name=config_hf_splade_l1q.yaml config.checkpoint_dir=<chk_dir> config.index_dir=<index_dir>

python -m splade.retrieve --config-name=config_hf_splade_l1q.yaml config.checkpoint_dir=<chk_dir> config.index_dir=<index_dir> config.out_dir=<out_dir>
```

### Toy example: Training a Reranker
You can now train your reranker using the SPLADE output (ie the top retrieved items for your training queries) :
```

python -m splade.hf_train_reranker --config-name=config_reranker_train_toy

```
### Toy example: Reranking a SPLADE output run

Then  you can apply your reranker (inference step):
```

python -m splade.rerank --config-name=config_reranker_toy data.path_run=[<out_dir>/run.json]

```





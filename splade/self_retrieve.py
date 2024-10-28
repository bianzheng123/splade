import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import os
import json

from conf.CONFIG_CHOICE import CONFIG_NAME, CONFIG_PATH
from .datasets.dataloaders import CollectionDataLoader
from .datasets.datasets import CollectionDatasetPreLoad
from .self_evaluate import self_evaluate
from .models.models_utils import get_model
from .tasks.self_transformer_evaluator import SelfSparseRetrieval
from .utils.utils import get_dataset_name, get_initialize_config


def save_answer_tsv(username: str, dataset: str, topk: int, run_json: dict, query_fname: str):
    queryID_l = []
    with open(query_fname, "r") as handler:
        for line in handler:
            if line == '':
                continue
            queryID = int(line.split('\t')[0])
            queryID_l.append(queryID)

    answer_path = f'/home/{username}/Dataset/vector-set-similarity-search/end2end/Result/answer/'
    answer_fname = os.path.join(answer_path, f'{dataset}-splade-top{topk}--.tsv')
    with open(answer_fname, 'w') as f:
        for queryID in queryID_l:
            query_answer_m = run_json[str(queryID)]
            rev_keys_l = reversed(sorted(query_answer_m, key=query_answer_m.get))
            for i, docID in enumerate(rev_keys_l):
                # print(i, query_answer_m.keys())
                # docID = query_answer_m.keys()[i]
                score = query_answer_m[docID]
                print(f"queryID {queryID}, docID {docID}, rank {i+1}, score {score}")
                f.write(f'{queryID}\t{docID}\t{i + 1}\t{score}\n')


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base="1.2")
def retrieve_evaluate(exp_dict: DictConfig):
    exp_dict, config, init_dict, model_training_config = get_initialize_config(exp_dict)

    # if HF: need to udate config.
    if "hf_training" in config:
        init_dict.model_type_or_dir = os.path.join(config.checkpoint_dir, "model")
        init_dict.model_type_or_dir_q = os.path.join(config.checkpoint_dir,
                                                     "model/query") if init_dict.model_type_or_dir_q else None

    model = get_model(config, init_dict)

    batch_size = 1
    # NOTE: batch_size is set to 1, currently no batched implem for retrieval (TODO)
    for data_dir in set(exp_dict["data"]["Q_COLLECTION_PATH"]):
        # q_collection is the text of the query
        q_collection = CollectionDatasetPreLoad(data_dir=data_dir, id_style="row_id")
        # tokenize the collection, the q_loader is a dictionary
        # input_ids is the tokenized text, attention_mask is the mask for the input_ids, id is the ID of the query
        q_loader = CollectionDataLoader(dataset=q_collection, tokenizer_type=model_training_config["tokenizer_type"],
                                        max_length=model_training_config["max_length"], batch_size=batch_size,
                                        shuffle=False, num_workers=1)
        evaluator = SelfSparseRetrieval(config=config, model=model, dataset_name=get_dataset_name(data_dir),
                                        compute_stats=True, dim_voc=model.output_dim)
        evaluator.retrieve(q_loader, n_query=len(q_collection), top_k=exp_dict["config"]["top_k"],
                           dataset=exp_dict["config"]["dataset_name"],
                           threshold=exp_dict["config"]["threshold"])

        out_dir = evaluator.out_dir
        print("out_dir", out_dir)

        with open(os.path.join(out_dir, "run.json"), "r") as handler:
            run_json = json.load(handler)
        username = exp_dict["config"]["username"]
        dataset = exp_dict["config"]["dataset_name"]
        topk = exp_dict["config"]["top_k"]
        query_fname = os.path.join(data_dir, 'raw.tsv')
        save_answer_tsv(username, dataset, topk, run_json, query_fname)

    self_evaluate(exp_dict)


if __name__ == "__main__":
    retrieve_evaluate()

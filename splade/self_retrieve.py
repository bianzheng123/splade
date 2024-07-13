import hydra
from omegaconf import DictConfig
import os

from conf.CONFIG_CHOICE import CONFIG_NAME, CONFIG_PATH
from .datasets.datasets import CollectionDatasetPreLoad
from .self_evaluate import self_evaluate
from .models.models_utils import get_model
from .tasks.self_transformer_evaluator import SelfSparseRetrieval, SelfTokenizer
from .utils.utils import get_dataset_name, get_initialize_config


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
        tokenizer = SelfTokenizer(tokenizer_type=model_training_config["tokenizer_type"],
                      max_length=model_training_config["max_length"])
        evaluator = SelfSparseRetrieval(config=config, model=model, tokenizer=tokenizer, dataset_name=get_dataset_name(data_dir),
                                        compute_stats=True, dim_voc=model.output_dim)
        evaluator.retrieve(q_collection, top_k=exp_dict["config"]["top_k"], threshold=exp_dict["config"]["threshold"])
    self_evaluate(exp_dict)


if __name__ == "__main__":
    retrieve_evaluate()

import json
import os

import hydra
from omegaconf import DictConfig

from conf.CONFIG_CHOICE import CONFIG_NAME, CONFIG_PATH
from splade.evaluation.eval import load_and_evaluate
from splade.utils.utils import get_dataset_name
from splade.utils.hydra import hydra_chdir


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base="1.2")
def self_evaluate(exp_dict: DictConfig):
    # for dataset EVAL_QREL_PATH
    # for metric of this qrel
    hydra_chdir(exp_dict)
    eval_qrel_path = exp_dict.data.EVAL_QREL_PATH
    eval_metric = exp_dict.config.eval_metric
    dataset_names = exp_dict.config.retrieval_name
    out_dir = exp_dict.config.out_dir

    res_all_datasets = {}
    for i, (qrel_file_path, eval_metrics, dataset_name) in enumerate(zip(eval_qrel_path, eval_metric, dataset_names)):
        if qrel_file_path is not None:
            res = {}
            print(eval_metrics)
            for metric in eval_metrics:
                qrel_fp = qrel_file_path
                res.update(load_and_evaluate(qrel_file_path=qrel_fp,
                                             run_file_path=os.path.join(out_dir, dataset_name, 'run.json'),
                                             metric=metric))
            if dataset_name in res_all_datasets.keys():
                res_all_datasets[dataset_name].update(res)
            else:
                res_all_datasets[dataset_name] = res
            out_fp = os.path.join(out_dir, dataset_name, "perf.json")
            json.dump(res, open(out_fp, "a"))

            topk = exp_dict["config"]["top_k"]
            real_dataset = exp_dict["config"]["dataset_name"]
            performance_fname = os.path.join(out_dir, dataset_name, f"{real_dataset}-retrieval-splade-top{topk}--.json")
            with open(performance_fname, 'r') as f:
                time_json = json.load(f)
                time_json['search_accuracy'] = {
                    "mrr_mean": '{:.3f}'.format(res['mrr_10']),
                    "e2e_recall_mean": '{:.3f}'.format(res['recall_10']),
                    "ndcg_mean": '{:.3f}'.format(res['ndcg'])
                }
            with open(performance_fname, 'w') as f:
                json.dump(time_json, f)

            username = exp_dict["config"]["username"]
            performance_path = f'/home/{username}/Dataset/vector-set-similarity-search/end2end/Result/performance'
            performance_fname = os.path.join(performance_path, f"{real_dataset}-retrieval-splade-top{topk}--.json")
            with open(performance_fname, 'w') as f:
                json.dump(time_json, f)


    out_all_fp = os.path.join(out_dir, "perf_all_datasets.json")
    json.dump(res_all_datasets, open(out_all_fp, "a"))

    return res_all_datasets


if __name__ == '__main__':
    self_evaluate()

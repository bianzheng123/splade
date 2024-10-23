import os

from self_script import transform_dataset

if __name__ == '__main__':
    # default value {'n_centroid_f': lambda x: int(2 ** np.floor(np.log2(16 * np.sqrt(x))))},
    config_l = {
        'dbg': {
            'username': 'zhengbian',
            'dataset_l': ['quora'],
            'topk_l': [10],
        },
        'local': {
            'username': 'bianzheng',
            # 'dataset_l': ['fake-normal', 'lotte-500-gnd'],
            'dataset_l': ['lotte-500-gnd'],
            'topk_l': [10],
        }
    }
    host_name = 'local'
    config = config_l[host_name]

    username = config['username']
    dataset_l = config['dataset_l']
    topk_l = config['topk_l']

    for dataset in dataset_l:
        transform_dataset.run(username=username, dataset=dataset)

        # build the index
        os.system(f"proxychains python3 -m splade.self_index \
                      init_dict.model_type_or_dir=naver/splade_v2_max \
                      config.pretrained_no_yamlconfig=true \
                      index={dataset} \
                      config.index_dir=experiments/pre-trained/{dataset}/index")

        for topk in topk_l:
            # retrieval
            os.system(f"proxychains python3 -m splade.self_retrieve \
                          init_dict.model_type_or_dir=naver/splade_v2_max \
                          config.pretrained_no_yamlconfig=true \
                          retrieve_evaluate={dataset} \
                          config.index_dir=experiments/pre-trained/{dataset}/index \
                          config.out_dir=experiments/pre-trained/{dataset}/out \
                          config.top_k={topk}")

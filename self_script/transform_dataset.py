import json
import os
from distutils.spawn import spawn


def delete_file_if_exist(dire):
    if os.path.exists(dire):
        command = 'rm -r %s' % dire
        print(command)
        os.system(command)


def transform_dev_queries(username: str, dataset: str):
    old_query_filename = f'/home/{username}/Dataset/vector-set-similarity-search/RawData/{dataset}/document/queries.dev.tsv'
    new_query_filename = f'/home/{username}/splade/data/{dataset}/dev_queries/raw.tsv'
    os.system(f'cp {old_query_filename} {new_query_filename}')


def transform_collection(username: str, dataset: str):
    old_collection_filename = f'/home/{username}/Dataset/vector-set-similarity-search/RawData/{dataset}/document/collection.tsv'
    new_collection_filename = f'/home/{username}/splade/data/{dataset}/full_collection/raw.tsv'
    os.system(f'cp {old_collection_filename} {new_collection_filename}')


def transform_query_gnd(username: str, dataset: str):
    old_qrel_filename = f'/home/{username}/Dataset/vector-set-similarity-search/RawData/{dataset}/document/queries.gnd.jsonl'
    new_qrel_filename = f'/home/{username}/splade/data/{dataset}/qrel/queries.gnd.jsonl'
    os.system(f'cp {old_qrel_filename} {new_qrel_filename}')

    qrel_m = {}
    with open(old_qrel_filename, 'r') as f:
        for line in f:
            json_ins = json.loads(line)
            docID_l = json_ins['passage_id']
            docID_m = {docID: 1 for docID in docID_l}
            qrel_m[json_ins['query_id']] = docID_m
    new_qrel_filename = f'/home/{username}/splade/data/{dataset}/qrel/qrel.json'
    with open(new_qrel_filename, 'w') as f:
        json.dump(qrel_m, f)


def run(username: str, dataset: str):
    splade_data_dir = f'/home/{username}/splade/data/{dataset}'
    delete_file_if_exist(splade_data_dir)

    os.makedirs(splade_data_dir, exist_ok=False)

    os.makedirs(os.path.join(splade_data_dir, 'dev_queries'), exist_ok=False)
    transform_dev_queries(username=username, dataset=dataset)

    os.makedirs(os.path.join(splade_data_dir, 'full_collection'), exist_ok=False)
    transform_collection(username=username, dataset=dataset)

    os.makedirs(os.path.join(splade_data_dir, 'qrel'), exist_ok=False)
    transform_query_gnd(username=username, dataset=dataset)


if __name__ == '__main__':
    username = 'zhengbian'
    dataset = 'lotte-500-gnd'
    run(username=username, dataset=dataset)

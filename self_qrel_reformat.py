import json


def run(username: str, dataset: str):
    old_qrel_filename = f'/home/{username}/splade/data/{dataset}/qrel/queries.gnd.jsonl'
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


if __name__ == '__main__':
    username = 'bianzheng'
    dataset = 'lotte-500-gnd'
    run(username=username, dataset=dataset)

import json

if __name__ == '__main__':
    username = 'bianzheng'
    dataset = 'lotte-500-gnd'
    with open(f'/home/{username}/Dataset/vector-set-similarity-search/RawData/{dataset}/document/queries.gnd.jsonl',
              'r') as f:
        queries = [json.loads(line) for line in f.readlines()]
    qrel_j = {}
    for query in queries:
        queryID = query['query_id']
        passage_id_l = query['passage_id']
        passage_id_m = {pID: 1 for pID in passage_id_l}
        qrel_j[queryID] = passage_id_m
    with open(f'/home/{username}/splade/qrel.json', 'w') as f:
        json.dump(qrel_j, f)
    pass

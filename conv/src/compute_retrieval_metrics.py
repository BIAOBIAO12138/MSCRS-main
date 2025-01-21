import json


def compute_retrieval_based_metrics(file_path):

    r1 = 0
    count = 0
    m = -1
    with open(file_path, 'r') as f:
        lines = f.readlines()

        for example in lines:
            example = json.loads(example)
            # if len(example['rec']) == 0:
            #         continue
            if len(example['context']) == 1 and example['context'][0] == '':
                    continue
            
            rec = example['rec']
            retrieved_resp_items = list(set(example['retrieved_response_entity'] + example['retrieved_context_entity']))

            m = max(m, len(retrieved_resp_items))

            for item in rec:
                count +=1 
                if item in retrieved_resp_items:
                    r1 +=1 

    print('number of test examples: ', count)
    print('recall @1: ', r1/ count)
    print('max num items: ', m)

if __name__ == '__main__':

    file_path = 'src/data/redial/test_data_processed_retrieval.jsonl'
    compute_retrieval_based_metrics(file_path)
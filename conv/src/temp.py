import re

import json
from nltk import ngrams
from nltk.translate.bleu_score import sentence_bleu

# year_pattern = re.compile(r'\(\d{4}\)')
slot_pattern = re.compile(r'<movie>')
import torch

from rouge import Rouge
import math

def _cal_rouge(hypothesis,reference):
    """
    both hypothesis and reference are str
    """
    if hypothesis == '':
        return 0,0,0
    rouge = Rouge()
    try:
        scores = rouge.get_scores(hypothesis, reference)
        return scores[0]['rouge-1']['f'],scores[0]['rouge-2']['f'],scores[0]['rouge-l']['f']
    except:
        print("something wrong here! ",hypothesis,reference)
        return 0,0,0


class ConvEvaluator:
    def __init__(self, tokenizer, log_file_path):
        self.tokenizer = tokenizer

        self.k_list = [1, 10, 50]
        self.metric = {}

        self.reset_metric()
        if log_file_path:
            self.log_file = open(log_file_path, 'w', buffering=1)
            self.log_cnt = 0

    def evaluate(self, preds, labels, log=False):

        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=False)
        decoded_preds = [decoded_pred.replace('<pad>', '').replace('<|endoftext|>', '') for decoded_pred in
                         decoded_preds]
        
        decoded_preds = [pred.strip() for pred in decoded_preds]
        #### for a fair comparision , we prepend the System token before the generated sequences.
        decoded_preds = [pred for pred in decoded_preds]

        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=False)
        decoded_labels = [decoded_label.replace('<pad>', '').replace('<|endoftext|>', '') for decoded_label in
                          decoded_labels]
        decoded_labels = [label.strip() for label in decoded_labels]

        ###
        decoded_preds = [decoded_preds[x:x+10] for x in range(0, len(decoded_preds), 10)]

        new_preds = []
        if log and hasattr(self, 'log_file'):
            for pred, label in zip(decoded_preds, decoded_labels):
                #### post processing the output
                item_set = 'Item: '
                new_pred = []
                for t in pred:
                    item_set += t
                    item_set += ', '
                    new_pred.append(t.strip().lower())
                self.log_file.write(json.dumps({
                    'pred': item_set,
                    'label': label
                }, ensure_ascii=False) + '\n')

                for k in self.k_list:
                    self.metric[f'recall@{k}'] += self.compute_recall(new_pred, label, k)
                    self.metric[f'mrr@{k}'] += self.compute_mrr(new_pred, label, k)
                    self.metric[f'ndcg@{k}'] += self.compute_ndcg(new_pred, label, k)

                new_preds.append(new_pred)
                self.metric['count'] +=1
        
        # self.sent_cnt += len([label for label in decoded_labels if len(label) > 0])

    def collect_ngram(self, strs):
        for str in strs:
            str = str.split()
            for k in range(1, 5):
                dist_k = f'dist@{k}'
                for token in ngrams(str, k):
                    self.metric[dist_k].add(token)

    def compute_recall(self, rank, label, k):
        label = label.strip().lower().replace('item: ','')
        rank = [x.strip().lower() for x in rank]
        return int(label in rank[:k])

    def report(self):
        report = {}
        for k, v in self.metric.items():
            report[k] = v
        return report
    
    def compute_mrr(self, rank, label, k):
        label = label.strip().lower().replace('item: ','')
        rank = [x.strip().lower() for x in rank]
        if label in rank[:k]:
            label_rank = rank.index(label)
            return 1 / (label_rank + 1)
        return 0

    def compute_ndcg(self, rank, label, k):
        label = label.strip().lower().replace('item: ','')
        rank = [x.strip().lower() for x in rank]
        if label in rank[:k]:
            label_rank = rank.index(label)
            return 1 / math.log2(label_rank + 2)
        return 0

    def reset_metric(self):
        for metric in ['recall', 'ndcg', 'mrr']:
            for k in self.k_list:
                self.metric[f'{metric}@{k}'] = 0
        self.metric['count'] = 0

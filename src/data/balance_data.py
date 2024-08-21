import os

from utils import load_data, save_jsonline
from tqdm import tqdm
from math import sqrt
from numpy import random
from collections import Counter
import re


def get_type2data(data):
    type2data = dict()
    id = 0
    for example in tqdm(data):
        spans = example['label'] if 'label' in example else example['spans']
        for s in spans:
            span_type = s['type']
            # span type should be str or list
            span_type = [span_type] if isinstance(span_type, str) else span_type

            for s in span_type:
                if s in type2data:
                    type2data[s].append(id)
                else:
                    type2data[s] = [id]
        id += 1
    return type2data


def clip_sqrt(n, min_n, m=0.5):
    if n <= min_n:
        return n
    else:
        return int(sqrt(n - min_n)**m) + min_n


def clip_data(data, min_n, m):
    train_type2data = get_type2data(data)
    train_type2num = {k: len(v) for k, v in train_type2data.items()}
    print(len(Counter(train_type2num)))

    train_type2sample_num = {k: clip_sqrt(v, min_n, m) for k, v in train_type2num.items()}
    train_type2data = {k: random.choice(v, train_type2sample_num[k], replace=False) for k, v in train_type2data.items()}
    all_data = set()
    for k, v in train_type2data.items():
        all_data.update(v)
    all_data = list(all_data)
    random.shuffle(all_data)
    return [data[j] for j in all_data]





if __name__ == '__main__':
    train_data = load_data('../../data/NER/gpt-ner/train.1000.json')
    # test_data = load_data('../../data/NER/gpt-ner/test.json')
    breakpoint()
    res = get_type2data(train_data)
    breakpoint()


    # random.seed(42)
    #
    # min_n = 100
    # m = 0.5
    # cliped_train_data = clip_data(data=train_data, min_n=min_n, m=m)
    # cliped_test_data = clip_data(data=test_data, min_n=min_n, m=m)
    # breakpoint()
    #
    # data_path = '../../data/NER/gpt-ner_{}_{}'.format(min_n, m)
    # os.makedirs(data_path, exist_ok=True)
    # save_jsonline(cliped_train_data, os.path.join(data_path, 'train.json'))
    # save_jsonline(cliped_test_data, os.path.join(data_path, 'test.json'))
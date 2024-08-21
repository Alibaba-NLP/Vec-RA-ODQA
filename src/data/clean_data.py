import os

from utils import load_data, save_jsonline
from tqdm import tqdm
from math import sqrt
from numpy import random
import re

if __name__ == '__main__':
    train_data = load_data('../../data/NER/gpt-ner-5/train.json')
    for example in tqdm(train_data):
        spans = example['label'] if 'label' in example else example['spans']
        for s in spans:
            span_type = s['type']
            # span type should be str or list
            span_type = [span_type] if isinstance(span_type, str) else span_type
            span_type = [i.strip() for i in span_type]
            span_type = list(set(span_type))
            s['type'] = span_type

    save_jsonline(train_data, '../../data/NER/gpt-ner-5/train.json')

    train_data = load_data('../../data/NER/gpt-ner-5/test.json')
    for example in tqdm(train_data):
        spans = example['label'] if 'label' in example else example['spans']
        for s in spans:
            span_type = s['type']
            # span type should be str or list
            span_type = [span_type] if isinstance(span_type, str) else span_type
            span_type = [i.strip() for i in span_type]
            span_type = list(set(span_type))
            s['type'] = span_type
    save_jsonline(train_data, '../../data/NER/gpt-ner-5/test.json')

    # min_n = 300
    # m = 0.75
    # random.seed(42)
    # cliped_train_data = clip_data(data=train_data, min_n=min_n, m=m)
    # cliped_test_data = clip_data(data=test_data, min_n=min_n, m=m)
    # data_path = '../../data/NER/texsmart_{}_{}'.format(min_n, m)
    # os.makedirs(data_path, exist_ok=True)
    # save_jsonline(cliped_train_data, os.path.join(data_path, 'train.json'))
    # save_jsonline(cliped_test_data, os.path.join(data_path, 'test.json'))
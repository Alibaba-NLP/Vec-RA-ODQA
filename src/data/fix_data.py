import os

from utils import load_data, save_jsonline
from tqdm import tqdm
from math import sqrt
from numpy import random
import re

if __name__ == '__main__':
    train_data = load_data('/nas-alinlp/jcy/Git/ProbTransformer/JeffCLM/data/NER/gpt-all/curl.utner.out.0308.result')

    fixed = 0
    for example in tqdm(train_data):
        spans = example['label'] if 'label' in example else example['spans']
        for s in spans:
            span_type = s['type']
            # span type should be str or list
            span_type = [span_type] if isinstance(span_type, str) else span_type
            new_span_type = []
            for t in span_type:
                tp = re.split('，|,|\||；|/|-|;', t)
                if len(tp) > 1:
                    fixed += 1
                new_span_type += tp
            span_type = [i for i in new_span_type]
            span_type = [i.strip() for i in span_type]
            span_type = list(set(span_type))
            s['type'] = span_type

    print('fixing: {}'.format(fixed))
    save_jsonline(train_data, '/nas-alinlp/jcy/Git/ProbTransformer/JeffCLM/data/NER/gpt-all/gpt-all.json')

    # min_n = 300
    # m = 0.75
    # random.seed(42)
    # cliped_train_data = clip_data(data=train_data, min_n=min_n, m=m)
    # cliped_test_data = clip_data(data=test_data, min_n=min_n, m=m)
    # data_path = '../../data/NER/texsmart_{}_{}'.format(min_n, m)
    # os.makedirs(data_path, exist_ok=True)
    # save_jsonline(cliped_train_data, os.path.join(data_path, 'train.json'))
    # save_jsonline(cliped_test_data, os.path.join(data_path, 'test.json'))
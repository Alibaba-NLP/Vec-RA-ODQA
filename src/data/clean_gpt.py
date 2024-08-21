import os
import json
from utils import load_data, save_jsonline
from tqdm import tqdm
from math import sqrt
from numpy import random
import re


def find_span(text, span):
    l_span = len(span)
    all_spans = []
    for i in range(len(text)):
        if text[i: i+l_span] == span:
            all_spans.append((i, i+l_span))
    return all_spans

if __name__ == '__main__':
    txt_file = '/nas-alinlp/yongjiang.jy/home/dataset/open-domain-nlu-en/result-news.2014.en.shuffled.v2.sampled5w.txt'
    js = load_data(txt_file)
    new_js = []
    for j in js:
        try:
            text = j['query'][1: -197]
            labels = json.loads(j['response'])
            all_labels = []
            for span, types in labels.items():
                locs = find_span(text, span)
                for loc in locs:
                    all_labels.append({'start': loc[0], 'end': loc[1], 'type': types, 'mention': span})

            _js = {
                'text': text,
                'label': all_labels,
            }
            new_js.append(_js)
        except:
            pass

    print(len(new_js))
    os.makedirs('/nas-alinlp/jcy/Git/ProbTransformer/JeffCLM/data/raw/en-gpt')
    save_jsonline(new_js,'/nas-alinlp/jcy/Git/ProbTransformer/JeffCLM/data/raw/en-gpt/train.json')


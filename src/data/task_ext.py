import json
import random
import os
from tqdm import tqdm
from .utils import load_data, mapping2label
import sys
from seqeval.metrics import classification_report
from collections import Counter
import numpy as np
from rouge import Rouge
sys.path.append('../')
sys.path.append('../../')
from copy import deepcopy
from src.meta import GEN_TOK, BOS_TOK, EOS_TOK, BASE_DATA_DIR
from src.data.task_zh_cls import Task
from glob import glob



class EXT(Task):
    """
    1. 原始数据 -》 inst
    2. evaluate
    """
    def __init__(self, dataset_folder, config):
        super(EXT, self).__init__(dataset_folder)

        self.data_folder = dataset_folder
        sub_task = dataset_folder.strip().split('/')[-1]
        temp = sub_task.strip().split('_')
        assert len(temp) >= 3
        self.lang = temp[0]
        self.sub_task = temp[1]


        self.task_name = 'ZH_FINAL'
        self.answer_template = """{out}"""  # 还是中文逗号
        self.prompt_template = "任务: 抽取\n{ex}输入: {sent}\n标签集: {label}\n输出: "
        self.output_template = lambda x: self.label_sep.join(x)
        self.label_sep = "，"
        self.train_file = glob(self.data_folder + "/*-train.json")[0]
        try:
            self.val_file = glob(self.data_folder + "/*-val*.json")[0]
        except IndexError:
            self.val_file = None
            print("no validation file founded")

        self.meta_file = glob(self.data_folder + "/*-meta.json")[0]
        self.prepare()
        assert isinstance(self.meta_data['label_set'], list), 'not ner data'
        assert len(self.meta_data['label_set']) > 1, 'n_label should be larger than 1'
        self.all_types = self.get_all_label()
        self.max_neg_label = 20
        self.max_pos_label = 10  # 每个样本的候选【正】types 最多多少个
        self.n_per_sample = 2
        self.n_repeat = 2
        print(self.task_name)
        self.max_train = 3000


    def get_span_type_mapping(self, example, enforce_single_label=True):

        span2types = dict()
        type2spans = dict()

        example = example if isinstance(example, list) else [example]
        for ex in example:
            inputs = ex['text']
            spans = ex['label'] if 'label' in ex else ex['spans']
            # get span2types and type2spans
            for s in spans:
                span_str = inputs[s['start']: s['end']]
                assert isinstance(span_str, str) # list of tokens, ZH
                span_type = s['type']
                # span type should be str or list
                span_type = [span_type] if isinstance(span_type, str) else span_type
                if span_str not in span2types: span2types[span_str] = span_type
                else: span2types[span_str] += span_type
                random.shuffle(span_type)
                if enforce_single_label:
                    span_type = span_type[:1]
                for t in span_type:
                    if t not in type2spans: type2spans[t] = [span_str]
                    else: type2spans[t].append(span_str)

        # remove redundant
        span2types = {k: list(set(v)) for k, v in span2types.items()}
        type2spans = {k: list(set(v)) for k, v in type2spans.items()}

        return span2types, type2spans


    def get_all_label(self):
        return self.meta_data['label_set']


    def encode_to_input_output(self, example, mode='train'):
        """
        example -> instruction
        """
        inputs = example['text']
        # 5w -> 16w

        for k in range(self.n_repeat):
            span2types, type2spans = self.get_span_type_mapping(example)
            # in sample label
            in_sample_label_list = list(type2spans.keys())
            random.shuffle(in_sample_label_list)
            # out sample label
            all_types = example.get('label_set', self.all_types)
            if 'label_set' not in example:
                all_types = example.get('label_list', self.all_types)

            for j in range(self.n_per_sample):

                if ('label_set' in example) or ('label_list' in example):
                    sampled_label = all_types
                else:
                    sampled_label = self.sample_label(in_sample_label_list)

                random.shuffle(sampled_label)
                label = self.output_template(sampled_label)
                out = ''
                for i in sampled_label:
                    out += (i + ':' + '\t'.join(type2spans.get(i, ['None'])) + '\n')

                prompt = {'sent': ''.join(inputs) if isinstance(inputs, list) else inputs,
                          'label': label,
                          'ex': ''}
                answer = {'out': out}
                yield prompt, answer



    def sample_label(self, pos_labels):
        max_pos_label = np.random.randint(self.max_pos_label)
        max_neg_label = np.random.randint(self.max_neg_label)
        pos_labels = pos_labels[:max_pos_label]
        if max_neg_label > len(self.all_types):
            neg_labels = self.all_types
        else:
            neg_labels = random.sample(self.all_types, k=max_neg_label)

        sampled_label = list(set(pos_labels + neg_labels))
        return sampled_label


    def parse_answer(self, answer):
        if len(answer) == 0:
            print('error')
        items = answer.split('\n')
        type_2_spans = dict()

        for i in items:
            if ':' not in i or i[-1] == ':':
                continue
            _type, spans = i[:i.index(':')], i[i.index(':')+1:].split('\t')
            if isinstance(spans, str): spans = [spans]
            if _type not in type_2_spans:
                type_2_spans[_type] = spans
            else:
                type_2_spans[_type] += spans

        for t, s in type_2_spans.items():
            type_2_spans[t] = set(s)
        return type_2_spans


    def evaluate(self, data, results):
        all_gold = []
        all_pred = []
        gold_answers = []
        pred_answers = []
        rouge = Rouge()
        for d, pred_result in zip(data, results):
            gold_answer = d['answer']
            example = d['example']
            inputs = example['text']
            text = ''.join(inputs) if isinstance(inputs, list) else inputs
            gold_type_2_spans = self.parse_answer(gold_answer)
            pred_type_2_spans = self.parse_answer(pred_result)
            if len(gold_answer) and len(pred_result):
                gold_answers.append(gold_answer)
                pred_answers.append(pred_result)
            else:
                print('miss')

            gold_labels = mapping2label(text, gold_type_2_spans)
            pred_labels = mapping2label(text, pred_type_2_spans)
            all_gold.append(gold_labels)
            all_pred.append(pred_labels)
        report = classification_report(all_gold, all_pred, output_dict=True)

        scores = rouge.get_scores(pred_answers, gold_answers, avg=True)
        macro_score = report['macro avg']
        micro_score = report['micro avg']
        rouge_l_score = scores['rouge-l']
        key_scores = {'micro-f1':  micro_score['f1-score'],
                      'macro-f1':  macro_score['f1-score'],
                      'rouge-l-f1': rouge_l_score['f']}
        detailed_report = {'seqeval': report, 'rouge': scores}

        return detailed_report, key_scores
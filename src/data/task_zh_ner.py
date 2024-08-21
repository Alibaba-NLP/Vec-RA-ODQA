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


class ZH_NER_ICL(Task):
    """
    1. 原始数据 -》 inst
    2. evaluate
    """

    def __init__(self, dataset_folder):
        super(ZH_NER_ICL, self).__init__(dataset_folder)

        self.data_folder = dataset_folder
        self.task_name = 'ZH_NER_ICL'
        self.definition = '中文命名实体识别'

        self.train_file = glob(self.data_folder + "/train.json")[0]
        self.val_file = glob(self.data_folder + "/test.json")[0]
        self.n_examples = 20
        self.train_data = None
        self.val_data = None
        self.example_data = None
        self.inst_data = None

        self.answer_template = """{out}"""
        self.prompt_template = "我们现在做中文命名实体的抽取，\n{ex}接下来给定以下\n输入： {sent}，抽取所有的{label}\n输出："
        self.example_template = "例子 {i}：\n输入: {sent}，抽取所有的{label}\n输出: {out}\n"
        self.label_sep = "，"
        self.output_template = lambda x: self.label_sep.join(x)

        self.prepare()
        self.all_types = self.get_all_label()
        self.all_type_sample_mapping = self.get_type_sample_mapping()
        self.max_top_label = 5  # 每个样本的候选【负】types中最多出现多少的【常见】样本
        self.max_remain_label = 20  # 每个样本的候选【负】types中最多出现多少的【少见】样本
        self.max_pos_label = 20  # 每个样本的候选【正】types 最多多少个
        self.n_per_sample = 3
        self.n_repeat = 3
        self.top_sqrt = 0.5
        self.add_example_prob = 0.2
        self.max_example = 0
        self.top_K = self.all_types.most_common(500)  # top多少的被定义为【常见】剩下的是【少见】
        print(self.task_name)

        self.dict_top_k = dict(self.top_K)
        self.list_top_k = [k for k, v in self.top_K]
        self.set_top_k = {k for k, v in self.top_K}
        self.remain = list(set(self.all_types.keys()) - self.set_top_k)

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
                if isinstance(span_str, list): span_str = ''.join(span_str)  # list of tokens, ZH
                span_type = s['type']
                # span type should be str or list
                span_type = [span_type] if isinstance(span_type, str) else span_type
                if span_str not in span2types:
                    span2types[span_str] = span_type
                else:
                    span2types[span_str] += span_type
                random.shuffle(span_type)
                if enforce_single_label:
                    span_type = span_type[:1]
                for t in span_type:
                    if t not in type2spans:
                        type2spans[t] = [span_str]
                    else:
                        type2spans[t].append(span_str)

        # remove redundant
        span2types = {k: list(set(v)) for k, v in span2types.items()}
        type2spans = {k: list(set(v)) for k, v in type2spans.items()}

        return span2types, type2spans

    def get_type_sample_mapping(self):
        type_sample_mapping = dict()
        id = 0
        for e in tqdm(self.train_data, desc='get all type sample mapping in the training'):
            spans = e['label'] if 'label' in e else e['spans']
            # get span2types and type2spans
            for s in spans:
                span_type = s['type']
                # span type should be str or list
                span_type = [span_type] if isinstance(span_type, str) else span_type
                span_type = list(set(span_type))
                if len(span_type) == 0:
                    continue
                else:
                    for t in span_type:
                        if t not in type_sample_mapping:
                            type_sample_mapping[t] = [id]
                        else:
                            type_sample_mapping[t].append(id)
            id += 1
        return type_sample_mapping

    def get_all_label(self):
        all_types = Counter()
        for e in tqdm(self.train_data, desc='get all labels in the training'):
            spans = e['label'] if 'label' in e else e['spans']
            # get span2types and type2spans
            for s in spans:
                span_type = s['type']
                # span type should be str or list
                span_type = [span_type] if isinstance(span_type, str) else span_type
                span_type = list(set(span_type))
                if len(span_type) == 0:
                    continue
                all_types.update(span_type)
        return all_types

    def encode_to_input_output(self, example, mode='train'):
        """
        example -> instruction
        """
        inputs = example['text']
        # 5w -> 16w

        for k in range(3):
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
                          'ex': self.encode_to_example(self.sample_examples(type2spans))}
                answer = {'out': out}
                yield prompt, answer

    def encode_to_input_output_ex(self, example):
        inputs = example['text']

        span2types, type2spans = self.get_span_type_mapping(example)
        # in sample label
        in_sample_label_list = list(type2spans.keys())
        random.shuffle(in_sample_label_list)
        # out sample label
        all_types = example.get('label_set', self.all_types)
        if 'label_set' not in example:
            all_types = example.get('label_list', self.all_types)

        if ('label_set' in example) or ('label_list' in example):
            sampled_label = all_types
        else:
            sampled_label = self.sample_label(in_sample_label_list)

        random.shuffle(sampled_label)
        label = self.output_template(sampled_label)
        out = ''
        for i in sampled_label:
            out += (i + ':' + '\t'.join(type2spans.get(i, ['None'])) + '\n')

        return {'sent': ''.join(inputs) if isinstance(inputs, list) else inputs,
                'label': label, 'out': out}

    def sample_label(self, in_sample_label_list):
        out_sample_label_list_k = list(self.set_top_k - set(in_sample_label_list))
        freq_list = np.array([self.dict_top_k[k] for k in out_sample_label_list_k]) ** self.top_sqrt
        prob_list = freq_list / freq_list.sum()
        n_out_top = np.random.randint(self.max_top_label)
        n_out_remain = np.random.randint(self.max_remain_label)
        n_pos = np.random.randint(self.max_pos_label)
        types_n_top_k = np.random.choice(out_sample_label_list_k, p=prob_list, size=n_out_top, replace=True)
        types_n_remain = random.choices(self.remain, k=n_out_remain)
        types_pos = in_sample_label_list[:n_pos]
        sampled_label = list(set(types_n_top_k.tolist() + types_n_remain + types_pos))
        return sampled_label

    def sample_examples(self, type2spans):
        # sample examples
        pos_types = list(type2spans.keys())
        random.shuffle(pos_types)
        all_example = []
        for i in range(self.max_example):
            if random.random() < self.add_example_prob:
                pos_t = random.choice(pos_types)  # sample from a pos related examples
                if pos_t in self.all_type_sample_mapping:
                    example = self.train_data[random.choice(self.all_type_sample_mapping[pos_t])]
                else:
                    example = random.choice(self.train_data)
                all_example.append(example)
        return all_example

    def encode_to_example(self, examples):
        # sample examples
        all_ex = ''
        id = 0
        for i in examples:
            id += 1
            _ex = self.encode_to_input_output_ex(i)
            _ex['i'] = id
            ex = self.example_template.format(**_ex)
            all_ex += ex
        return all_ex

    def parse_answer(self, answer):
        if len(answer) == 0:
            print('error')
        items = answer.split('\n')
        type_2_spans = dict()

        for i in items:
            if ':' not in i or i[-1] == ':':
                continue
            _type, spans = i[:i.index(':')], i[i.index(':') + 1:].split('\t')
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
        key_scores = {'micro-f1': micro_score['f1-score'],
                      'macro-f1': macro_score['f1-score'],
                      'rouge-l-f1': rouge_l_score['f']}
        detailed_report = {'seqeval': report, 'rouge': scores}

        return detailed_report, key_scores


class ZH_NER_ICL_T1(ZH_NER_ICL):
    def __init__(self, dataset_folder):
        super(ZH_NER_ICL_T1, self).__init__(dataset_folder)
        self.task_name = 'ZH_NER_ICL_T1'
        self.definition = '中文命名实体识别'
        self.answer_template = """{out}"""
        self.prompt_template = "我们现在做中文命名实体的抽取，\n{ex}接下来给定以下\n输入： {sent}，抽取所有的[{label}]\n输出："
        self.example_template = "例子 {i}：\n输入: {sent}，抽取所有的[{label}]\n输出: {out}\n"
        self.label_sep = "，"
        self.output_template = lambda x: self.label_sep.join(x)
        self.max_example = 0
        print(self.task_name)


class ZH_NER_ICL_EX(ZH_NER_ICL):
    def __init__(self, dataset_folder):
        super(ZH_NER_ICL_EX, self).__init__(dataset_folder)
        self.add_example_prob = 0.5
        self.max_example = 5
        self.max_types = 10
        self.task_name = 'ZH_NER_ICL_EX'
        self.definition = '中文命名实体识别'
        self.answer_template = """{out}"""
        self.prompt_template = "我们现在做中文命名实体的抽取，\n{ex}接下来给定以下\n输入： {sent}，抽取所有的[{label}]\n输出："
        self.example_template = "例子 {i}：{label}: {spans}\n"
        self.label_sep = "，"
        self.output_template = lambda x: self.label_sep.join(x)
        _, self.type2spans_all = self.get_span_type_mapping(self.train_data, enforce_single_label=False)
        print(self.task_name)

    def encode_to_input_output(self, example, mode='train'):
        inputs = example['text']
        for k in range(3):
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
                          'ex': self.encode_to_example(self.sample_examples(sampled_label))}
                answer = {'out': out}
                yield prompt, answer

    def encode_to_input_output_ex(self, type_span_tuple):
        k, v = type_span_tuple
        spans = '\t'.join(v)
        return {'label': k,
                'spans': spans}

    def sample_examples(self, sampled_labels):
        """
           here return the sampled type: span dict items
        """
        # sample examples
        if self.max_example <= 0 or len(sampled_labels) == 0:
            return []
        if type(sampled_labels[0]) == str:  # during training
            # sample partial portion of types and their part of demonstrations
            dd = []
            if random.random() < self.add_example_prob:
                n_type = np.random.randint(self.max_types)
                new_sampled_label = deepcopy(sampled_labels)
                random.shuffle(new_sampled_label)
                labels = new_sampled_label[:n_type]
                dd = [(k, random.choices(self.type2spans_all.get(k, []),
                                         k=min(len(self.type2spans_all.get(k, [])),
                                               np.random.randint(self.max_example))))
                      for k in labels]
                dd = [(k, v) for k, v in dd if len(v) > 0]
            return dd
        else:
            raise NotImplementedError()


class ZH_NER_ICL_T11(ZH_NER_ICL):
    def __init__(self, dataset_folder):
        super(ZH_NER_ICL_T11, self).__init__(dataset_folder)
        self.task_name = 'ZH_NER_ICL_T11'
        self.definition = '中文命名实体识别'
        self.answer_template = """{out}"""
        self.prompt_template = "我们现在做中文命名实体的抽取，\n{ex}接下来给定以下\n输入： {sent}，抽取所有的[{label}]\n输出："
        self.example_template = "例子 {i}：\n输入: {sent}，抽取所有的[{label}]\n输出: {out}\n"
        self.label_sep = "，"
        self.output_template = lambda x: self.label_sep.join(x)
        self.max_example = 0
        self.max_top_label = 10
        self.max_remain_label = 20
        self.top_K = self.all_types.most_common(2000)
        print(self.task_name)


class ZH_NER_ICL_NEW(ZH_NER_ICL):
    def __init__(self, dataset_folder):
        super(ZH_NER_ICL_NEW, self).__init__(dataset_folder)
        # 新版本instruction，统一成英文逗号
        self.task_name = 'ZH_NER_ICL_NEW'
        self.definition = '中文命名实体识别'
        self.answer_template = """{out}"""
        # 规范形式，为了多语言，Instruction先全部全部英文逗号以及英文冒号。
        self.prompt_template = "任务: 抽取中文命名实体\n{ex}输入: {sent}\n标签集: {label}\n输出: "
        self.example_template = "例子 {i}\n输入: {sent}\n标签集: {label}\n输出: {out}\n"
        self.label_sep = ","
        print(self.task_name)


class ZH_NER_ICL_NEW_V1(ZH_NER_ICL):
    def __init__(self, dataset_folder):
        super(ZH_NER_ICL_NEW_V1, self).__init__(dataset_folder)
        # 新版本instruction，但是label用中文逗号
        self.task_name = 'ZH_NER_ICL_NEW_V1'
        self.definition = '中文命名实体识别'
        self.answer_template = """{out}"""  # 还是中文逗号
        self.prompt_template = "任务: 抽取中文命名实体\n{ex}输入: {sent}\n标签集: {label}\n输出: "
        self.example_template = "例子 {i}\n输入: {sent}\n标签集: {label}\n输出: {out}\n"
        self.label_sep = "，"
        print(self.task_name)


class ZH_NER_ICL_NEW_V2(ZH_NER_ICL):
    def __init__(self, dataset_folder):
        super(ZH_NER_ICL_NEW_V2, self).__init__(dataset_folder)
        # 老版本inst 但是英文逗号
        self.task_name = 'ZH_NER_ICL_NEW_V2'
        self.definition = '中文命名实体识别'
        self.answer_template = """{out}"""
        # 规范形式，为了多语言，Instruction先全部全部英文逗号以及英文冒号。
        self.prompt_template = "我们做中文命名实体的抽取,\n{ex}接下来给定以下\n输入: {sent},抽取所有的[{label}]\n输出: "
        self.example_template = "例子 {i}\n输入: {sent}\n标签集: {label}\n输出: {out}\n"
        self.label_sep = ","
        print(self.task_name)


class ZH_NER_ICL_T1_ECOM(ZH_NER_ICL):
    def __init__(self, dataset_folder):
        super(ZH_NER_ICL_T1_ECOM, self).__init__(dataset_folder)
        self.task_name = 'ZH_NER_ICL_T1_ECOM'
        self.definition = '中文命名实体识别'
        self.answer_template = """{out}"""
        self.prompt_template = "我们现在做中文命名实体的抽取，\n{ex}接下来给定以下\n输入： {sent}，抽取所有的[{label}]\n输出："
        self.example_template = "例子 {i}：\n输入: {sent}，抽取所有的[{label}]\n输出: {out}\n"
        self.label_sep = "，"
        self.output_template = lambda x: self.label_sep.join(x)
        self.max_example = 0
        self.top_K = self.all_types.most_common(20)
        self.max_top_label = 3
        self.max_remain_label = 5
        self.max_pos_label = 3
        print(self.task_name)
        self.dict_top_k = dict(self.top_K)
        self.list_top_k = [k for k, v in self.top_K]
        self.set_top_k = {k for k, v in self.top_K}
        self.remain = list(set(self.all_types.keys()) - self.set_top_k)


class ZH_NER_ICL_T1_ECOM_FINE(ZH_NER_ICL):
    def __init__(self, dataset_folder):
        super(ZH_NER_ICL_T1_ECOM_FINE, self).__init__(dataset_folder)
        self.task_name = 'ZH_NER_ICL_T1_ECOM_FINE'
        self.definition = '中文命名实体识别'
        self.answer_template = """{out}"""
        self.prompt_template = "我们现在做中文命名实体的抽取，\n{ex}接下来给定以下\n输入： {sent}，抽取所有的[{label}]\n输出："
        self.example_template = "例子 {i}：\n输入: {sent}，抽取所有的[{label}]\n输出: {out}\n"
        self.label_sep = "，"
        self.output_template = lambda x: self.label_sep.join(x)
        self.max_example = 0
        self.max_top_label = 10
        self.max_remain_label = 20
        self.top_K = self.all_types.most_common(50)
        print(self.task_name)


class ZH_NER_ICL_T12(ZH_NER_ICL):
    def __init__(self, dataset_folder):
        super(ZH_NER_ICL_T12, self).__init__(dataset_folder)
        self.task_name = 'ZH_NER_ICL_T12'
        self.definition = '中文命名实体识别'
        self.answer_template = """{out}"""
        self.prompt_template = "我们现在做中文命名实体的抽取，\n{ex}接下来给定以下\n输入： {sent}，抽取所有的[{label}]\n输出："
        self.example_template = "例子 {i}：\n输入: {sent}，抽取所有的[{label}]\n输出: {out}\n"
        self.label_sep = "，"
        self.output_template = lambda x: self.label_sep.join(x)
        self.max_example = 0
        self.top_K = self.all_types.most_common(200)
        print(self.task_name)


class ZH_NER_ICL_T1_NO_REPEAT(ZH_NER_ICL):
    def __init__(self, dataset_folder):
        super(ZH_NER_ICL_T1_NO_REPEAT, self).__init__(dataset_folder)
        self.task_name = 'ZH_NER_ICL_T1_NO_REPEAT'
        self.definition = '中文命名实体识别'
        self.answer_template = """{out}"""
        self.prompt_template = "我们现在做中文命名实体的抽取，\n{ex}接下来给定以下\n输入： {sent}，抽取所有的[{label}]\n输出："
        self.example_template = "例子 {i}：\n输入: {sent}，抽取所有的[{label}]\n输出: {out}\n"
        self.label_sep = "，"
        self.n_per_sample = 1
        print(self.task_name)


class ZH_NER_ICL_NEW_V1_NO_REPEAT(ZH_NER_ICL):
    def __init__(self, dataset_folder):
        super(ZH_NER_ICL_NEW_V1_NO_REPEAT, self).__init__(dataset_folder)
        # 新版本instruction，但是label用中文逗号
        self.task_name = 'ZH_NER_ICL_NEW_V1_NO_REPEAT'
        self.definition = '中文命名实体识别'
        self.answer_template = """{out}"""  # 还是中文逗号
        self.prompt_template = "任务: 抽取中文命名实体\n{ex}输入: {sent}\n标签集: {label}\n输出: "
        self.example_template = "例子 {i}\n输入: {sent}\n标签集: {label}\n输出: {out}\n"
        self.label_sep = "，"
        self.n_per_sample = 1
        print(self.task_name)


class ZH_NER_ICL_NEW_V1_NO_REPEAT_ECOM(ZH_NER_ICL):
    def __init__(self, dataset_folder):
        super(ZH_NER_ICL_NEW_V1_NO_REPEAT_ECOM, self).__init__(dataset_folder)
        # 新版本instruction，但是label用中文逗号
        self.task_name = 'ZH_NER_ICL_NEW_V1_NO_REPEAT_ECOM'
        self.definition = '中文命名实体识别'
        self.answer_template = """{out}"""  # 还是中文逗号
        self.prompt_template = "任务: 抽取中文命名实体\n{ex}输入: {sent}\n标签集: {label}\n输出: "
        self.example_template = "例子 {i}\n输入: {sent}\n标签集: {label}\n输出: {out}\n"
        self.label_sep = "，"
        self.n_per_sample = 1
        self.max_example = 0
        self.max_top_label = 10
        self.max_remain_label = 20
        self.top_K = self.all_types.most_common(50)
        print(self.task_name)


class ZH_NER_FIRE_FLY(ZH_NER_ICL):
    def __init__(self, dataset_folder):
        super(ZH_NER_FIRE_FLY, self).__init__(dataset_folder)
        # 新版本instruction，但是label用中文逗号
        self.task_name = 'ZH_NER_FIRE_FLY'
        self.definition = '中文命名实体识别'
        self.answer_template = """{out}"""  # 还是中文逗号
        self.prompt_template = "找出[{sent}]中的: {label}"
        self.example_template = "例子 {i}\n输入: {sent}\n标签集: {label}\n输出: {out}\n"
        self.label_sep = "，"
        print(self.task_name)

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
        key_scores = {'micro-f1': micro_score['f1-score'],
                      'macro-f1': macro_score['f1-score'],
                      'rouge-l-f1': rouge_l_score['f']}
        detailed_report = {'seqeval': report, 'rouge': scores}

        return detailed_report, key_scores


class ZH_FINAL(ZH_NER_ICL):
    def __init__(self, dataset_folder):
        super(ZH_FINAL, self).__init__(dataset_folder)
        # 新版本instruction，但是label用中文逗号
        sub_task = dataset_folder.strip().split('/')[-1]
        temp = sub_task.strip().split('_')
        assert len(temp) < 3
        self.lang = temp[0]
        self.sub_task = temp[1]

        self.task_name = 'ZH_FINAL'
        self.answer_template = """{out}"""  # 还是中文逗号
        self.prompt_template = "任务: 抽取\n{ex}输入: {sent}\n标签集: {label}\n输出: "
        self.example_template = "例子 {i}\n输入: {sent}\n标签集: {label}\n输出: {out}\n"
        self.label_sep = "，"
        self.max_top_label = 5  # 每个样本的候选【负】types中最多出现多少的【常见】样本
        self.max_remain_label = 20  # 每个样本的候选【负】types中最多出现多少的【少见】样本
        self.max_pos_label = 20  # 每个样本的候选【正】types 最多多少个
        self.n_per_sample = 2
        self.n_repeat = 2
        self.top_sqrt = 0.5
        self.max_example = 0
        self.top_K = self.all_types.most_common(500)  # top多少的被定义为【常见】剩下的是【少见】
        print(self.task_name)


class ZH_NER_INFUSED(ZH_NER_ICL):
    def __init__(self, dataset_folder):
        super(ZH_NER_INFUSED, self).__init__(dataset_folder)
        # 新版本instruction，但是label用中文逗号

        self.task_name = 'ZH_NER_INFUSED'
        self.answer_template = """{out}"""  # 还是中文逗号
        self.prompt_template = "我们现在做中文命名实体的抽取，\n{ex}接下来给定以下\n输入： {sent}，抽取所有的[{label}]\n输出："
        self.example_template = "例子 {i}\n输入: {sent}\n标签集: {label}\n输出: {out}\n"
        self.label_sep = "，"
        self.max_top_label = 5  # 每个样本的候选【负】types中最多出现多少的【常见】样本
        self.max_remain_label = 20  # 每个样本的候选【负】types中最多出现多少的【少见】样本
        self.max_pos_label = 20  # 每个样本的候选【正】types 最多多少个
        self.n_per_sample = 2
        self.n_repeat = 2
        self.top_sqrt = 0.5
        self.max_example = 0
        self.top_K = self.all_types.most_common(500)  # top多少的被定义为【常见】剩下的是【少见】
        print(self.task_name)

        self.dict_top_k = dict(self.top_K)
        self.list_top_k = [k for k, v in self.top_K]
        self.set_top_k = {k for k, v in self.top_K}
        self.remain = list(set(self.all_types.keys()) - self.set_top_k)

    def get_example(self):
        for i in range(3):
            print('EXAMPLE {}'.format(i))
            inp, out, vec = next(self.encode_to_input_output(self.train_data[i]))
            prompt = self.prompt_template.format(**inp)
            answer = self.answer_template.format(**out)
            print("prompt:\n{}\nanswer:\n{}".format(prompt, answer))

    def generate(self):
        train_data = []
        for e in tqdm(self.train_data, desc='generate training data'):
            for inp, out, vec in self.encode_to_input_output(e):
                prompt = self.prompt_template.format(**inp)
                answer = self.answer_template.format(**out)
                train_data.append({'prompt': prompt, 'answer': answer, 'retrieval_vectors': vec})

        val_data = []
        for e in tqdm(self.val_data, desc='generate validation data'):
            for inp, out, vec in self.encode_to_input_output(e):
                prompt = self.prompt_template.format(**inp)
                answer = self.answer_template.format(**out)
                val_data.append({'prompt': prompt, 'answer': answer, 'retrieval_vectors': vec})

        self.inst_data = {'train': train_data, 'val': val_data}

    def encode_to_input_output(self, example, mode='train'):
        """
        example -> instruction
        """
        inputs = example['text']

        for k in range(3):
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
                          'ex': self.encode_to_example(self.sample_examples(type2spans)),
                          }
                answer = {'out': out}
                vec = example['retrieval_vectors']
                yield prompt, answer, vec

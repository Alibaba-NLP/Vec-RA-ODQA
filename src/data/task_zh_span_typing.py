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
from src.data.task_zh_ner import ZH_NER_ICL
from src.data.task_zh_cls import SetScore
from glob import glob


class ZH_TYPING(ZH_NER_ICL):
    """
    1. 原始数据 -》 inst
    2. evaluate
    """
    def __init__(self, dataset_folder):
        super(ZH_TYPING, self).__init__(dataset_folder)

        self.data_folder = dataset_folder
        self.task_name = 'ZH_TYPING'
        self.definition = '中文实体分类'
        self.answer_template = """{out}"""
        self.prompt_template = "我们现在做中文命名实体的分类，\n{ex}接下来给定以下\n输入： {sent}，标签集：{label}\n输出："
        self.example_template = "例子 {i}：\n输入： {sent}，标签集{label}\n输出： {out}\n"
        self.label_sep = "，"
        self.sent_mention_sep = lambda sent, mention: '{}\t{}'.format(sent, mention)
        self.pos_preserve_rate = 0.8
        self.scorer = SetScore()


    def encode_to_input_output(self, example, mode='train'):
        """
        example -> instruction
        """
        inputs = example['text']
        # 5w -> 16w
        if mode == 'train':
            # sample positive spans, preserve each pos types by prob self.pos_preserve_rate
            for k in range(3): # sample 3 times
                span2types, type2spans = self.get_span_type_mapping(example, enforce_single_label=False)
                # in sample label
                spans = list(span2types.keys())
                if len(spans) == 0:  # no spans
                    break
                else:
                    span = random.choice(spans)
                    in_sample_label_list = list(span2types[span])
                    random.shuffle(in_sample_label_list)
                    in_sample_label_list = [i for i in in_sample_label_list if random.random() < self.pos_preserve_rate ]
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
                        out = '\t'.join([i for i in sampled_label if i in in_sample_label_list]) + '\n'  # 对齐顺序
                        sent = ''.join(inputs) if isinstance(inputs, list) else inputs
                        sent_with_span = self.sent_mention_sep(sent, span)

                        prompt = {'sent': sent_with_span,
                                  'label': label,
                                  'ex': ''}
                        answer = {'out': out}
                        yield prompt, answer
        else:  # eval mode, get all spans in a sentence
            span2types, type2spans = self.get_span_type_mapping(example, enforce_single_label=False)
            for span, types in span2types.items():
                in_sample_label_list = types
                # out sample label
                all_types = example.get('label_set', self.all_types)
                if 'label_set' not in example:
                    all_types = example.get('label_list', self.all_types)
                assert ('label_set' in example) or ('label_list' in example)

                for j in range(self.n_per_sample):
                    sampled_label = all_types
                    random.shuffle(sampled_label)
                    label = self.output_template(sampled_label)
                    out = '\t'.join([i for i in sampled_label if i in in_sample_label_list]) + '\n'  # 对齐顺序
                    sent = ''.join(inputs) if isinstance(inputs, list) else inputs
                    sent_with_span = self.sent_mention_sep(sent, span)

                    prompt = {'sent': sent_with_span,
                              'label': label,
                              'ex': ''}
                    answer = {'out': out}
                    yield prompt, answer


    def parse_answer(self, answer):
        if len(answer) == 0:
            print('error')
        answers = answer.split('\t')
        answers = [self._postprocess_lab(i) for i in answers]
        return answers

    def _postprocess_lab(self, lab):
        return lab.replace(';', '/').replace('；', '/').strip()

    def evaluate(self, data, results):
        all_gold = []
        all_pred = []
        gold_answers = []
        pred_answers = []
        rouge = Rouge()
        for d, pred_result in zip(data, results):
            gold_answer = d['answer']
            example = d['example']

            if len(gold_answer) and len(pred_result):
                gold_answers.append(gold_answer)
                pred_answers.append(pred_result)
            else:
                print('miss')

            all_gold.append(self.parse_answer(gold_answer))
            all_pred.append(self.parse_answer(pred_result))

        scores = rouge.get_scores(pred_answers, gold_answers, avg=True)
        self.scorer.update(all_gold, all_pred)
        res = self.scorer.result()
        rouge_l_score = scores['rouge-l']
        key_scores = {'micro-f1':  res['micro_f1'],
                      'macro-f1':  res['macro_f1'],
                      'rouge-l-f1': rouge_l_score['f']}
        detailed_report = {'setscore': res, 'rouge': scores}

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
                                         k=min(len(self.type2spans_all.get(k, [])), np.random.randint(self.max_example))))
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
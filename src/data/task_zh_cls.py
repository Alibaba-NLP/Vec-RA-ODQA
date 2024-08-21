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
from glob import glob
from src.meta import GEN_TOK, BOS_TOK, EOS_TOK, BASE_DATA_DIR

class Task:
    def __init__(self, dataset_folder):
        self.task_name = 'NER'
        self.definition = 'Named Entity Recognition'
        self.n_examples = 20
        self.data_folder = dataset_folder
        self.prompt_template = """Given the sentence {inp} \n
                            Show me the named entities and their types \n"""
        self.answer_template = """{out}"""
        self.overall_template = "{def} \n {ex} \n {conj} \n {prompt}"
        self.example_template = "Example {i}: \n Input: {inp}\nOutput: {out}"
        self.train_file = None
        self.val_file = None
        self.meta_file = None
        self.train_data = None
        self.val_data = None
        self.meta_data = None
        self.example_data = None
        self.inst_data = None
        self.max_train = 1000000
        self.max_val = 300
        self.pos_count = Counter()


    def get_example(self):
        for i in range(3):
            print('EXAMPLE {}'.format(i))
            inp, out = next(self.encode_to_input_output(self.train_data[i]))
            prompt = self.prompt_template.format(**inp)
            answer = self.answer_template.format(**out)
            print("prompt:\n{}\nanswer:\n{}".format(prompt, answer))

    def prepare(self):
        self.train_data = load_data(self.train_file)
        if self.val_file and os.path.exists(self.val_file):
            self.val_data = load_data(self.val_file)
        else:
            print('unable to load valid files')
            self.val_data = self.train_data[-self.max_val:]
            self.train_data = self.train_data[:-self.max_val]
        if self.meta_file: self.meta_data = json.load(open(self.meta_file, 'r', encoding='utf-8'))
        random.shuffle(self.train_data)


    def encode_to_input_output(self, example, mode='train'):
        yield dict(), dict()

    def generate(self):
        train_data = []
        self.pos_count = Counter()
        for e in tqdm(self.train_data[:self.max_train], desc='generate training data'):
            for inp, out in self.encode_to_input_output(e):
                prompt = self.prompt_template.format(**inp)
                answer = self.answer_template.format(**out)
                train_data.append({'prompt': prompt, 'answer': answer})

        val_data = []
        self.pos_count = Counter()
        for e in tqdm(self.val_data, desc='generate validation data'):
            for inp, out in self.encode_to_input_output(e):
                prompt = self.prompt_template.format(**inp)
                answer = self.answer_template.format(**out)
                val_data.append({'prompt': prompt, 'answer': answer})
        self.inst_data = {'train': train_data, 'val': val_data}

    def sample(self, data, n=5, max_per_example=1):
        sample_data = []
        for e in tqdm(data[:n], desc='sample data'):
            cnt = 0
            for inp, out in self.encode_to_input_output(e):
                if cnt >= max_per_example:
                    break
                prompt = self.prompt_template.format(**inp)
                answer = self.answer_template.format(**out)
                sample_data.append({'prompt': prompt, 'answer': answer, 'example': e})
                cnt += 1
        return sample_data

    def save_data(self):
        for k, v in self.inst_data.items():
            save_path = os.path.join(self.data_folder,
                                     '{split}.{task}.inst.json'.format(split=k, task=self.task_name))
            with open(save_path, 'w') as f:
                dumped = [json.dumps(l, ensure_ascii=False) for l in v]
                for i in dumped[:-1]:
                    f.write(i+'\n')
                f.write(dumped[-1])

            print("save data to: {}".format(save_path))

    def evaluate(self, data, results):
        pass

    def parse_answer(self, answer):
        pass



class SetScore:
    """evaluate macro and micro set p/r/f1 scores"""

    def __init__(self):
        self.n_sample = 0
        self.pred = []  # list of list
        self.true = []  # list of list

    def reset(self):  # noqa: D102
        self.n_sample = 0
        self.pred = []  # list of list
        self.true = []  # list of list

    def set_pred_true(self, pred, true):  # noqa: D102
        self.pred = pred
        self.true = true

    def update(self, batch_gold_entities, batch_pred_entities):  # noqa: D102
        self.n_sample += len(batch_gold_entities)
        self.pred.extend(batch_pred_entities)
        self.true.extend(batch_gold_entities)

    def f1(self, precision, recall):  # noqa: D102
        f1 = 0.0 if recall + precision == 0 else (2 * precision * recall) / (precision + recall)
        return f1

    def result(self):  # noqa: D102
        assert len(self.pred) == len(self.true)
        M = len(self.pred)
        strict_acc = 0
        num_pred_labels = 0
        num_true_labels = 0
        num_correct_labels = 0
        total_ma_p = 0
        total_ma_r = 0
        total_ma_f1 = 0
        count = 0
        for i in range(M):
            p = set(self.pred[i])
            t = set(self.true[i])
            count += 1

            if p == t:
                strict_acc += 1

            l_p, l_t, l_intersect = len(p), len(t), len(p.intersection(t))
            num_pred_labels += l_p
            num_true_labels += l_t
            num_correct_labels += l_intersect

            if l_p == 0 or l_t == 0:
                ma_p = 0
                ma_r = 0
                ma_f1 = 0
            else:
                ma_p = l_intersect / l_p
                ma_r = l_intersect / l_t
                ma_f1 = self.f1(ma_p, ma_r)

            total_ma_p += ma_p
            total_ma_r += ma_r
            total_ma_f1 += ma_f1

        if num_pred_labels == 0 or num_true_labels == 0:
            micro_p = 0
            micro_r = 0
            micro_f1 = 0
        else:
            micro_p = num_correct_labels / num_pred_labels
            micro_r = num_correct_labels / num_true_labels
            micro_f1 = self.f1(micro_p, micro_r)

        strict_acc /= count
        macro_p = total_ma_p / count
        macro_r = total_ma_r / count
        macro_f1 = self.f1(macro_p, macro_r)
        avg_true_label = num_true_labels / M
        avg_pred_label = num_pred_labels / M

        return {
            'strict_acc': strict_acc,
            'micro_p': micro_p,
            'micro_r': micro_r,
            'micro_f1': micro_f1,
            'macro_p': macro_p,
            'macro_r': macro_r,
            'macro_f1': macro_f1,
            'avg_true_label': avg_true_label,
            'avg_pred_label': avg_pred_label,
        }





class ZH_CLS(Task):
    """
    1. 原始数据 -》 inst
    2. evaluate
    """
    def __init__(self, dataset_folder):
        super().__init__(dataset_folder)
        self.task_name = 'ZH_CLS'
        self.definition = '句子分类'
        self.data_folder = dataset_folder
        self.train_file = glob(self.data_folder + "/*-train.json")[0]
        try:
            self.val_file = glob(self.data_folder + "/*-val*.json")[0]
        except IndexError:
            self.val_file = None
            print("no validation file founded")

        self.meta_file = glob(self.data_folder + "/*-meta.json")[0]
        self.n_examples = 20
        self.train_data = None
        self.val_data = None
        self.meta_data = None
        self.example_data = None
        self.inst_data = None

        self.answer_template = """句子类别：{out}"""
        self.head = "我们现在做{defi}".format(defi=self.definition)
        self.prompt_template = self.head + "，\n{ex}接下来给定以下\n输入： {sent}，该输入属于如下哪种类别？[{label}]\n输出："
        self.label_sep = "，"
        self.output_sep = "\n"
        self.label_template = lambda x: self.label_sep.join(x)
        self.prepare()

        # assert len(self.meta_data['label_set'].keys()) == 1, "目前只支持单种分类标准"
        self.label_key, self.label_list = list(self.meta_data['label_set'].items())[-1]
        self.n_per_sample = 3
        self.max_label = 50
        self.max_train = 20000
        self.scorer = SetScore()


    def encode_to_input_output(self, example, mode='train'):
        """
        example -> instruction
        """
        inputs = example['text']

        pos_labels = [i['name'] for i in example['labels'][self.label_key]]
        if mode == 'train':  # sample label list
            all_labels = self.label_list
            # 5w -> 16w
            for k in range(self.n_per_sample):
                random.shuffle(all_labels)
                max_label = np.random.randint(self.max_label)
                all_labels = list(set(pos_labels + all_labels[:max_label]))
                labels = self.label_template(all_labels)
                # randomly sample labels

                if len(pos_labels) > 1:
                    random.shuffle(pos_labels)

                out = self.output_sep.join(pos_labels)

                prompt = {'sent': ''.join(inputs) if isinstance(inputs, list) else inputs,
                          'label': labels,
                          'ex': ''}
                answer = {'out': out}
                yield prompt, answer

                if len(pos_labels) == 1:
                    break

        else: # eval mode, don't sample
            pos_labels = [self._postprocess_lab(i) for i in pos_labels]
            out = self.output_sep.join(pos_labels)
            all_labels = example['label_set'][self.label_key]

            # replace ; to /
            all_labels = [self._postprocess_lab(i) for i in all_labels]
            labels = self.label_template(all_labels)
            prompt = {'sent': ''.join(inputs) if isinstance(inputs, list) else inputs,
                      'label': labels,
                      'ex': ''}
            answer = {'out': out}
            yield prompt, answer


    def _postprocess_lab(self, lab):
        return lab.replace(';', '/').replace('；', '/').strip()

    def parse_answer(self, answer):
        if len(answer) == 0:
            print('error')
        answers = answer.split(self.output_sep)
        answers[0] = answers[0].replace("句子类别", '').replace(':', '：')
        answers = [i.strip() for i in answers]
        return answers


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

            # all_gold.append([self._postprocess_lab(i['name'])
            #                  for i in example['labels'][self.label_key]])  # also pose process the label
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



class ZH_CLS_MULTI(ZH_CLS):
    """
    1. 原始数据 -》 inst
    2. evaluate
    """
    def __init__(self, dataset_folder):
        super().__init__(dataset_folder)
        self.task_name = 'ZH_CLS_MULTI'
        self.definition = '句子分类'
        self.data_folder = dataset_folder
        self.train_file = glob(self.data_folder + "/*-train.json")[0]
        try:
            self.val_file = glob(self.data_folder + "/*-val*.json")[0]
        except IndexError:
            self.val_file = None
            print("no validation file founded")

        self.meta_file = glob(self.data_folder + "/*-meta.json")[0]
        self.n_examples = 20
        self.train_data = None
        self.val_data = None
        self.meta_data = None
        self.example_data = None
        self.inst_data = None

        self.answer_template = """句子类别：{out}"""
        self.head = "我们现在做{defi}".format(defi=self.definition)
        self.prompt_template = self.head + "，\n{ex}接下来给定以下\n输入： {sent}，该输入属于如下哪种类别？[{label}]\n输出："
        self.label_sep = "，"
        self.label_template = lambda x: self.label_sep.join(x)
        self.output_sep = '\n'
        self.output_template = lambda x: self.output_sep.join(x)
        self.prepare()

        # assert len(self.meta_data['label_set'].keys()) == 1, "目前只支持单种分类标准"
        assert isinstance(self.meta_data['label_set'], dict), 'not CLS dataset'
        self.label_list = self.get_all_labels(self.meta_data['label_set'])
        self.n_per_sample = 3
        self.max_label = 50
        self.max_train = 100000
        self.scorer = SetScore()


    def get_all_labels(self, label_set):
        all_labels = []
        for k, v in label_set.items():
            all_labels += v
        return all_labels


    def encode_to_input_output(self, example, mode='train'):
        """
        example -> instruction
        """
        inputs = example['text']

        pos_labels = self.get_all_labels(example['labels'])
        pos_labels = [i['name'] for i in pos_labels]

        if mode == 'train':  # sample label list
            all_labels = self.label_list
            # 5w -> 16w
            for k in range(self.n_per_sample):
                random.shuffle(all_labels)
                max_label = np.random.randint(self.max_label)
                all_labels = list(set(pos_labels + all_labels[:max_label]))
                labels = self.label_template(all_labels)
                # randomly sample labels

                if len(pos_labels) > 1:
                    random.shuffle(pos_labels)

                out = self.output_template(pos_labels)

                prompt = {'sent': ''.join(inputs) if isinstance(inputs, list) else inputs,
                          'label': labels,
                          'ex': ''}
                answer = {'out': out}
                yield prompt, answer

                if len(pos_labels) == 1:
                    break

        else: # eval mode, don't sample
            pos_labels = [self._postprocess_lab(i) for i in pos_labels]
            out = self.output_template(pos_labels)
            all_labels = self.get_all_labels(example['label_set'])
            # replace ; to /
            all_labels = [self._postprocess_lab(i) for i in all_labels]
            labels = self.label_template(all_labels)
            prompt = {'sent': ''.join(inputs) if isinstance(inputs, list) else inputs,
                      'label': labels,
                      'ex': ''}
            answer = {'out': out}
            yield prompt, answer


    def evaluate(self, data, results):
        all_gold = []
        all_pred = []
        gold_answers = []
        pred_answers = []
        rouge = Rouge()
        for d, pred_result in zip(data, results):
            gold_answer = d['answer']

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


class ZH_CLS_MULTI_NEW(ZH_CLS_MULTI):
    """
    1. 原始数据 -》 inst
    2. evaluate
    """
    def __init__(self, dataset_folder):
        super().__init__(dataset_folder)
        self.task_name = 'ZH_CLS_MULTI_NEW'
        self.definition = '句子分类'
        self.answer_template = """{out}"""
        self.prompt_template = "任务: 句子分类\n{ex}输入: {sent}\n标签集: {label}\n输出: "
        self.example_template = "例子 {i}\n输入: {sent}\n标签集: {label}\n输出: {out}\n"
        self.label_sep = "，"  # 和ZH_NER_CLS_NEW对应，都用英文,


class ZH_CLS_MULTI_V1(ZH_CLS_MULTI):
    """
    1. 原始数据 -》 inst
    2. evaluate
    """

    def __init__(self, dataset_folder):
        super().__init__(dataset_folder)
        self.task_name = 'ZH_CLS_MULTI_V1'
        self.definition = '句子分类'
        # 和 MultiCLS的区别：预训练的时候也处理了label，以及考虑了输出label的顺序和输入labelset一致

    def encode_to_input_output(self, example, mode='train'):
        """
        example -> instruction
        """
        inputs = example['text']

        pos_labels = self.get_all_labels(example['labels'])
        pos_labels = [i['name'] for i in pos_labels]
        pos_labels = [self._postprocess_lab(i) for i in pos_labels]
        if mode == 'train':  # sample label list
            all_labels = self.label_list
            # 5w -> 16w
            for k in range(self.n_per_sample):
                random.shuffle(all_labels)
                max_label = np.random.randint(self.max_label)
                all_labels = list(set(pos_labels + all_labels[:max_label]))
                all_labels = [self._postprocess_lab(i) for i in all_labels]
                labels = self.label_template(all_labels)
                pos_labels = [i for i in all_labels if i in pos_labels] # 保持顺序一致
                out = self.output_template(pos_labels)

                prompt = {'sent': ''.join(inputs) if isinstance(inputs, list) else inputs,
                          'label': labels,
                          'ex': ''}
                answer = {'out': out}
                yield prompt, answer

                if len(pos_labels) == 1:
                    break

        else:  # eval mode, don't sample
            pos_labels = [self._postprocess_lab(i) for i in pos_labels]
            out = self.output_template(pos_labels)
            all_labels = self.get_all_labels(example['label_set'])
            # replace ; to /
            all_labels = [self._postprocess_lab(i) for i in all_labels]
            labels = self.label_template(all_labels)
            prompt = {'sent': ''.join(inputs) if isinstance(inputs, list) else inputs,
                      'label': labels,
                      'ex': ''}
            answer = {'out': out}
            yield prompt, answer



class ZH_CLS_MULTI_NEW_V1(ZH_CLS_MULTI):
    """
    1. 原始数据 -》 inst
    2. evaluate
    """

    def __init__(self, dataset_folder):
        super().__init__(dataset_folder)
        self.task_name = 'ZH_CLS_MULTI_NEW_V1'
        self.definition = '句子分类'
        self.answer_template = """{out}"""
        self.prompt_template = "任务: 句子分类\n{ex}输入: {sent}\n标签集: {label}\n输出: "
        self.example_template = "例子 {i}\n输入: {sent}\n标签集: {label}\n输出: {out}\n"
        self.label_sep = "，"  # 和ZH_NER_CLS_NEW对应，分隔符用中文,

        # 和 MultiCLS的区别：预训练的时候也处理了label，以及考虑了输出label的顺序和输入labelset一致

    def encode_to_input_output(self, example, mode='train'):
        """
        example -> instruction
        """
        inputs = example['text']

        pos_labels = self.get_all_labels(example['labels'])
        pos_labels = [i['name'] for i in pos_labels]
        pos_labels = [self._postprocess_lab(i) for i in pos_labels]
        if mode == 'train':  # sample label list
            all_labels = self.label_list
            # 5w -> 16w
            for k in range(self.n_per_sample):
                random.shuffle(all_labels)
                max_label = np.random.randint(self.max_label)
                all_labels = list(set(pos_labels + all_labels[:max_label]))
                all_labels = [self._postprocess_lab(i) for i in all_labels]
                labels = self.label_template(all_labels)
                pos_labels = [i for i in all_labels if i in pos_labels]  # 保持顺序一致
                out = self.output_template(pos_labels)

                prompt = {'sent': ''.join(inputs) if isinstance(inputs, list) else inputs,
                          'label': labels,
                          'ex': ''}
                answer = {'out': out}
                yield prompt, answer

                if len(pos_labels) == 1:
                    break

        else:  # eval mode, don't sample
            pos_labels = [self._postprocess_lab(i) for i in pos_labels]
            out = self.output_template(pos_labels)
            all_labels = self.get_all_labels(example['label_set'])
            # replace ; to /
            all_labels = [self._postprocess_lab(i) for i in all_labels]
            labels = self.label_template(all_labels)
            prompt = {'sent': ''.join(inputs) if isinstance(inputs, list) else inputs,
                      'label': labels,
                      'ex': ''}
            answer = {'out': out}
            yield prompt, answer



class ZH_CLS_MULTI_NEW_20(ZH_CLS_MULTI):
    """
    1. 原始数据 -》 inst
    2. evaluate
    """

    def __init__(self, dataset_folder):
        super().__init__(dataset_folder)
        self.task_name = 'ZH_CLS_MULTI_NEW_20'
        self.definition = '句子分类'
        self.answer_template = """{out}"""
        self.prompt_template = "任务: 句子分类\n{ex}输入: {sent}\n标签集: {label}\n输出: "
        self.example_template = "例子 {i}\n输入: {sent}\n标签集: {label}\n输出: {out}\n"
        self.label_sep = "，"  # 和ZH_NER_CLS_NEW对应，分隔符用中文,

        # 和 MultiCLS的区别：预训练的时候也处理了label，以及考虑了输出label的顺序和输入labelset一致
        self.n_per_sample = 1
        self.max_label = 20
        self.max_train = 30000

    def encode_to_input_output(self, example, mode='train'):
        """
        example -> instruction
        """
        inputs = example['text']

        pos_labels = self.get_all_labels(example['labels'])
        pos_labels = [i['name'] for i in pos_labels]
        pos_labels = [self._postprocess_lab(i) for i in pos_labels]
        if mode == 'train':  # sample label list
            all_labels = self.label_list
            # 5w -> 16w
            for k in range(self.n_per_sample):
                random.shuffle(all_labels)
                max_label = np.random.randint(self.max_label)
                all_labels = list(set(pos_labels + all_labels[:max_label]))
                all_labels = [self._postprocess_lab(i) for i in all_labels]
                labels = self.label_template(all_labels)
                pos_labels = [i for i in all_labels if i in pos_labels]  # 保持顺序一致
                out = self.output_template(pos_labels)

                prompt = {'sent': ''.join(inputs) if isinstance(inputs, list) else inputs,
                          'label': labels,
                          'ex': ''}
                answer = {'out': out}
                yield prompt, answer

                if len(pos_labels) == 1:
                    break

        else:  # eval mode, don't sample
            pos_labels = [self._postprocess_lab(i) for i in pos_labels]
            out = self.output_template(pos_labels)
            all_labels = self.get_all_labels(example['label_set'])
            # replace ; to /
            all_labels = [self._postprocess_lab(i) for i in all_labels]
            labels = self.label_template(all_labels)
            prompt = {'sent': ''.join(inputs) if isinstance(inputs, list) else inputs,
                      'label': labels,
                      'ex': ''}
            answer = {'out': out}
            yield prompt, answer




class ZH_CLS_MULTI_NEW_20_AVG(ZH_CLS_MULTI):
    """
    1. 原始数据 -》 inst
    2. evaluate
    """

    def __init__(self, dataset_folder):
        super().__init__(dataset_folder)
        self.task_name = 'ZH_CLS_MULTI_NEW_20_AVG'
        self.definition = '句子分类'
        self.answer_template = """{out}"""
        self.prompt_template = "任务: 句子分类\n{ex}输入: {sent}\n标签集: {label}\n输出: "
        self.example_template = "例子 {i}\n输入: {sent}\n标签集: {label}\n输出: {out}\n"
        self.label_sep = "，"  # 和ZH_NER_CLS_NEW对应，分隔符用中文,

        # 和 MultiCLS的区别：预训练的时候也处理了label，以及考虑了输出label的顺序和输入labelset一致
        self.n_per_sample = 1
        self.max_label = 20
        self.max_train = 30000
        self.max_val = 5
        self.max_n_pos = 50
        self.pos_count = Counter()

    def encode_to_input_output(self, example, mode='train'):
        """
        example -> instruction
        """
        inputs = example['text']

        pos_labels = self.get_all_labels(example['labels'])
        pos_labels = [i['name'] for i in pos_labels]
        pos_labels = [self._postprocess_lab(i) for i in pos_labels]
        if mode == 'train':  # sample label list
            for p_lab in pos_labels:  # control the pos label set
                if self.pos_count[p_lab] > self.max_n_pos:
                    return range(0)

            self.pos_count.update(pos_labels)
            all_labels = self.label_list
            # 5w -> 16w
            for k in range(self.n_per_sample):
                random.shuffle(all_labels)
                max_label = np.random.randint(self.max_label)
                all_labels = list(set(pos_labels + all_labels[:max_label]))
                all_labels = [self._postprocess_lab(i) for i in all_labels]
                labels = self.label_template(all_labels)
                pos_labels = [i for i in all_labels if i in pos_labels]  # 保持顺序一致
                out = self.output_template(pos_labels)

                prompt = {'sent': ''.join(inputs) if isinstance(inputs, list) else inputs,
                          'label': labels,
                          'ex': ''}
                answer = {'out': out}
                yield prompt, answer

                if len(pos_labels) == 1:
                    break

        else:  # eval mode, don't sample
            pos_labels = [self._postprocess_lab(i) for i in pos_labels]
            out = self.output_template(pos_labels)
            all_labels = self.get_all_labels(example['label_set'])
            # replace ; to /
            all_labels = [self._postprocess_lab(i) for i in all_labels]
            labels = self.label_template(all_labels)
            prompt = {'sent': ''.join(inputs) if isinstance(inputs, list) else inputs,
                      'label': labels,
                      'ex': ''}
            answer = {'out': out}
            yield prompt, answer



class ZH_FINAL(ZH_CLS_MULTI):
    """
    1. 原始数据 -》 inst
    2. evaluate
    """

    def __init__(self, dataset_folder):
        super().__init__(dataset_folder) # already init the files
        sub_task = dataset_folder.strip().split('/')[-1]
        temp = sub_task.strip().split('_')
        assert len(temp) >= 3
        self.lang = temp[0]
        self.sub_task = temp[1]

        self.answer_template = """{out}"""
        self.example_template = "例子 {i}\n输入: {sent}\n标签集: {label}\n输出: {out}\n"
        self.prompt_template = "任务: 分类\n{ex}输入: {sent}\n标签集: {label}\n输出: "
        self.label_sep = "，"  # 和ZH_NER_CLS_NEW对应，分隔符用中文,

        # 和 MultiCLS的区别：预训练的时候也处理了label，以及考虑了输出label的顺序和输入labelset一致
        self.n_per_sample = 1  # 每个类别sample最多几次
        self.max_label = 50
        self.max_train = 3000
        self.max_n_pos = 50  # 每个类别最多少个
        self.task_name = 'ZH_FINAL'
        self.pos_count = Counter()



class ZH_FINAL_1(ZH_CLS_MULTI):
    """
    1. 原始数据 -》 inst
    2. evaluate
    """

    def __init__(self, dataset_folder):
        super().__init__(dataset_folder) # already init the files
        sub_task = dataset_folder.strip().split('/')[-1]
        temp = sub_task.strip().split('_')
        assert len(temp) >= 3
        self.lang = temp[0]
        self.sub_task = temp[1]

        self.answer_template = """{out}"""
        self.example_template = "例子 {i}\n输入: {sent}\n标签集: {label}\n输出: {out}\n"
        self.prompt_template = "输入: {sent}\n分类: {label}\n输出: "
        self.label_sep = "，"  # 和ZH_NER_CLS_NEW对应，分隔符用中文,

        # 和 MultiCLS的区别：预训练的时候也处理了label，以及考虑了输出label的顺序和输入labelset一致
        self.n_per_sample = 1  # 每个类别sample最多几次
        self.max_label = 25
        self.max_train = 3000
        self.max_n_pos = 50  # 每个类别最多少个
        self.task_name = 'ZH_FINAL_1'
        self.pos_count = Counter()
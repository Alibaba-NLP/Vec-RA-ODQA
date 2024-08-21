import argparse
import importlib
import json
import os, sys
import random
from glob import glob
sys.path.append('../')
sys.path.append('../../')
from src.meta import GEN_TOK, BOS_TOK, EOS_TOK, BASE_DATA_DIR
from data.task import load_data
from data.utils import save_jsonline

def get_cls(task_name, inst_name, data_folder):
    task = importlib.import_module('data.task_{task_name}'.format(task_name=task_name))
    class_ = getattr(task, inst_name)
    instance = class_(data_folder)
    return instance


if __name__ == '__main__':
    print("example: {}".format('adaseq_python generate_instruction.py -t zh_cls -i ZH_CLS -sf CLS -d "held_in/raw_data/*"'))
    print("example: {}".format('adaseq_python generate_instruction.py -t zh_ner -i ZH_NER_ICL_T1 -sf NER -d "standard_ner/*fix*/"'))

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task', type=str, default='zh_cls')
    parser.add_argument('-i', '--inst', type=str, default='ZH_CLS')
    parser.add_argument('-sf', '--sub_folder', type=str, default='CLS')
    parser.add_argument('-d', '--dataset_name', type=str, default='')
    parser.add_argument('-tfn', '--train_file_name', type=str, default='*train.json')
    args = parser.parse_args()

    data_folder = os.path.join(BASE_DATA_DIR, args.sub_folder)
    datasets = glob(os.path.join(data_folder, args.dataset_name))
    max_pos_label = 10
    max_neg_label = 25
    print("generate instruction for {} datasets: {}".format(len(datasets), datasets))
    for dd in datasets:
        try:
            test_data = load_data(glob(dd+'/{}'.format(args.train_file_name))[0])
            meta_data = json.load(open(glob(dd+'/{}'.format('*meta*.json'))[0]))
        except:
            continue

        random.shuffle(test_data)
        test_data = test_data[:100]
        for label_key, label_list in meta_data['label_set'].items():
            print(dd, len(label_list), len(set(label_list)))
            if len(label_list) < 10:
                for d in test_data:
                    d['label_set'] = dict()
                    d['label_set'][label_key] = label_list
            else:
                for d in test_data:
                    try:
                        pos_labels = list(set([i['name'] for i in d['labels'][label_key]]))
                        if len(label_list) > max_neg_label:
                            neg_labels = list(set(label_list) - set(pos_labels))
                            random.shuffle(neg_labels)
                            neg_labels = neg_labels[:max_neg_label]
                            label_list = pos_labels + neg_labels
                            random.shuffle(label_list)
                        if 'label_set' in d:
                            d['label_set'][label_key] = label_list
                        else:
                            d['label_set'] = dict()
                            d['label_set'][label_key] = label_list
                    except:
                        breakpoint()

        save_jsonline(test_data, os.path.join(dd, 'eval.100.json'))

        #
        # task.generate()
        # task.save_data()
        # breakpoint()


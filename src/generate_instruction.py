import argparse
import importlib
import os, sys
from glob import glob
sys.path.append('../')
sys.path.append('../../')
from src.meta import GEN_TOK, BOS_TOK, EOS_TOK, BASE_DATA_DIR


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
    print(data_folder)
    datasets = glob(os.path.join(data_folder, args.dataset_name))
    datasets = [i for i in datasets if len(glob(i + '/' + args.train_file_name))]
    print("generate instruction for {} datasets: {}".format(len(datasets), datasets))

    for dd in datasets:
        try:
            task = get_cls(args.task, args.inst, dd)
        except AssertionError:
            print("error in datasets {}".format(dd))
            continue

        task.get_example()
        task.generate()
        task.save_data()


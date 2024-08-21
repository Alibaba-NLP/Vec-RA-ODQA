import sys, os, json
sys.path.append('../')
from data.task import load_data

def process_and_save(js, save_path):
    for j in js:
        j['spans'] = j.pop('label')

    with open(save_path, 'w') as f:
        dumped = [json.dumps(l, ensure_ascii=False) for l in js]
        for i in dumped[:-1]:
            f.write(i + '\n')
        f.write(dumped[-1])
    print("save data to: {}".format(save_path))


if __name__ == '__main__':
    js = load_data('../../data/NER/texsmart/test.json')
    process_and_save(js, '../../data/NER/texsmart/test.new.json')

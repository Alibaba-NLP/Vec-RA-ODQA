from utils import load_data, save_jsonline
import random

if __name__ == '__main__':
    js = load_data('../../data/NER/standard_ner/zh_datafound_manufact_inductry_fix/test.json')
    random.shuffle(js)
    eval = [{'example': i} for i in js[:48]]
    examples = [{'example': i} for i in js[48:]]

    save_jsonline(eval, '../../data/NER/standard_ner/zh_datafound_manufact_inductry_fix/eval.sample.json')
    save_jsonline(examples, '../../data/NER/standard_ner/zh_datafound_manufact_inductry_fix/eval.example.json')

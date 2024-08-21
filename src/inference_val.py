from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModel,
    AutoConfig
)
import argparse
from datetime import datetime
import os
from meta import GEN_TOK, BOS_TOK, EOS_TOK
from generate_instruction import get_cls
import random
from tqdm import tqdm
import torch
from data.task import load_data
from data.utils import save_jsonline
from meta import BASE_DATA_DIR
import re
from copy import deepcopy
from glob import glob

from src.model.bloom.modeling_vector_infused_bloom import VectorInfusedBloomForCausalLM, VectorInfusedBloomModelConfig
from src.utils.vec_db import load_vec_db, get_vec_from_db


def generate_prompts(p, vecs, model, tokenizer,
                     prompt_length=256, answer_length=44,
                     beam_size=4, temperature=0.95, top_p=1.0):
    p = [i + GEN_TOK for i in p]
    # set pad_token_id to eos_token_id because GPT2 does not have a PAD token
    input_ids = tokenizer(p,
                          return_tensors="pt",
                          padding=True,
                          truncation=True,
                          max_length=prompt_length).input_ids
    input_ids = input_ids.to(model.device)

    vecs = torch.Tensor(vecs).to(model.device)

    outputs = model.generate(input_ids=input_ids,
                             infused_vectors=vecs,
                             num_beams=beam_size,
                             do_sample=False,
                             max_length=answer_length + prompt_length,
                             temperature=temperature,
                             top_k=top_p,
                             # repetition_penalty=2.0
                             )
    outputs_answer = tokenizer.batch_decode(outputs, skip_special_tokens=False)
    all_answer = []
    for i in outputs_answer:
        if (GEN_TOK in i) and (tokenizer.eos_token in i):
            gen_idx = i.index(GEN_TOK) + len(GEN_TOK)
            end_idx = i.index(tokenizer.eos_token)
            all_answer.append(i[gen_idx: end_idx])
        elif (GEN_TOK in i) and (tokenizer.eos_token not in i):
            gen_idx = i.index(GEN_TOK) + len(GEN_TOK)
            all_answer.append(i[gen_idx:])
        else:
            all_answer.append('')
    return [i.replace(tokenizer.pad_token, '') for i in all_answer]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task', type=str, default='zh_cls')
    parser.add_argument('-i', '--inst', type=str, default='ZH_CLS')
    parser.add_argument('-sf', '--sub_folder', type=str, default='NER')
    parser.add_argument('-d', '--dataset_name', type=str, default='all')
    parser.add_argument('-m', '--model_name_or_path', type=str, default='../output/2-24-bigscience/bloom-560m/')
    parser.add_argument('-n', '--n_eval', type=int, default=999999, help='默认eval 100000 (基本所有)')
    parser.add_argument('-k', '--k_shot', type=int, default=0)
    parser.add_argument('-bz', '--batch_size', type=int, default=4)
    parser.add_argument('-pl', '--prompt_length', type=int, default=1024)
    parser.add_argument('-al', '--answer_length', type=int, default=128)
    parser.add_argument('-beam', '--beam_size', type=int, default=4)
    parser.add_argument('-sn', '--save_name', type=str, default='3-4')
    parser.add_argument('-g', '--gpu', type=int, default=1)
    parser.add_argument('-ml', '--max_label', type=int, default=999999)
    parser.add_argument('-ms', '--max_sent', type=int, default=500)
    parser.add_argument('-lora', '--lora', type=str, default='')
    parser.add_argument('-tfn', '--test_file_name', type=str, default='*100-test.json')
    parser.add_argument('-tp', '--temperature', type=float, default=1.0)
    parser.add_argument('-top_p', '--top_p', type=float, default=1.0)
    parser.add_argument('-ptuning', '--ptuning_checkpoint', type=str, default='')
    parser.add_argument('--retrieval_vec_save_mode', type=str, default='value')
    parser.add_argument('--vec_database_file', type=str, default='')

    # parser.add_argument('--seed', type=int, default=42)
    random.seed(42)

    args = parser.parse_args()
    base_save_dir = '../output/evaluate'
    m_str = args.model_name_or_path
    m_str = '-'.join(m_str.split('/')[2:])
    m_str = m_str.replace('/', '-')
    dataset_name = args.dataset_name.replace('/', '-')
    save_dir = os.path.join(base_save_dir, args.save_name)
    os.makedirs(save_dir, exist_ok=True)
    print("use GPU: {}".format(torch.cuda.is_available()))

    print('dataset_name', args.dataset_name)
    print('subfolder', args.sub_folder)
    print('test_file_name', args.test_file_name)

    L = args.max_sent
    max_n_labels = args.max_label


    def split_sent(sent):
        l_sent = len(sent)
        n_pieces = int(l_sent / L)
        all_sent = []
        for p in range(n_pieces):
            all_sent.append(sent[L * p: L * (p + 1)])
        all_sent.append(sent[L * n_pieces:])
        return all_sent


    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    config = VectorInfusedBloomModelConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True,
                                                           revision='aa51e62')
    model = VectorInfusedBloomForCausalLM.from_pretrained(args.model_name_or_path, config=config,
                                                          trust_remote_code=True, revision='aa51e62')

    tokenizer.truncation_side = 'left'

    if args.retrieval_vec_save_mode == 'index':
        db_dict = load_vec_db(args.retrieval_vec_save_mode)
    else:
        db_dict = None

    if args.gpu:
        model = model.cuda()
    model.eval()

    data_folder = os.path.join(BASE_DATA_DIR, args.sub_folder)
    datasets = glob(os.path.join(data_folder, args.dataset_name))
    datasets = [i for i in datasets if len(glob(i + '/' + args.test_file_name))]

    res_key = args.__dict__

    for dd in datasets:
        res_key['data_file'] = dd
        res_key['data_name'] = [i for i in dd.split('/') if i][-1]
        eval_data_path = glob(dd + '/' + args.test_file_name)[0]
        print(args.task, args.inst, dd)
        task = get_cls(args.task, args.inst, dd)

        if os.path.exists(eval_data_path):
            print('loading eval data: {}'.format(eval_data_path))
            sample_data = load_data(eval_data_path)
            for i in sample_data:
                if 'example' not in i:
                    i['example'] = i
                if 'retrieval_vector' in i:
                    i['example']['retrieval_vector'] = torch.zeros(0)
                inp, out, vec = next(task.encode_to_input_output(i['example'], mode='eval'))
                all_sent = split_sent(inp['sent'])
                all_label = inp['label']

                all_prompt = []
                for ss in all_sent:
                    new_inp = deepcopy(inp)
                    new_inp['sent'] = ss
                    new_inp['label'] = all_label
                    prompt = task.prompt_template.format(**new_inp)
                    all_prompt.append(prompt)
                answer = task.answer_template.format(**out)
                i['prompt'] = all_prompt
                i['answer'] = answer

        else:
            raise NotImplementedError('no eval data')

        print(sample_data[0]['prompt'])
        print(sample_data[0]['answer'])

        p_single = [i['prompt'][0] for i in sample_data if len(i['prompt']) == 1]
        a_single = [i['answer'] for i in sample_data if len(i['prompt']) == 1]
        sample_single = [i for i in sample_data if len(i['prompt']) == 1]

        vec_single = [
            (i['retrieval_vector']
             if args.retrieval_vec_save_mode == 'value'
             else get_vec_from_db(i['retrieval_vector'], db_dict))
            for i in sample_single if len(i['prompt']) == 1
        ]

        p_multi = [i['prompt'] for i in sample_data if len(i['prompt']) > 1]
        a_multi = [i['answer'] for i in sample_data if len(i['prompt']) > 1]
        sample_multi = [i for i in sample_data if len(i['prompt']) > 1]
        vec_multi = [
            [i['retrieval_vector']
             if args.retrieval_vec_save_mode == 'value'
             else get_vec_from_db(i['retrieval_vector'], db_dict)] * len(i['prompt'])
            for i in sample_multi
        ]


        def iter_prompt_list(p_list, a_list, vec_list):
            res = []
            l = 0
            r = l + args.batch_size
            idx = 0
            while l < len(p_list):
                prompt_list = p_list[l: r]
                tmp_res = generate_prompts(prompt_list, vec_list, model, tokenizer,
                                           args.prompt_length, args.answer_length,
                                           beam_size=args.beam_size,
                                           temperature=args.temperature,
                                           top_p=args.top_p, )
                res += tmp_res
                l = r
                r = r + args.batch_size
                for i in tmp_res:
                    print("EVAL {} ================================== ".format(idx))
                    print(p_list[idx])
                    print(i)
                    print("answer: ")
                    print(a_list[idx])
                    f.write(p_list[idx])
                    f.write(i)
                    f.write(a_list[idx])
                    f.write('\n======================================================\n')
                    idx += 1

            return res


        datetime_str = datetime.now().strftime("%Y%m%d-%H%M-%s")
        save_path = os.path.join(save_dir,
                                 '{}_{}_{}_{}_{}.res'.format(args.sub_folder, dd.split('/')[-1], args.task, args.inst,
                                                             datetime_str))
        print('saving at: {}'.format(save_path))
        with open(save_path, 'w', encoding='utf-8') as f:
            print("{} SINGLE LIST, BATCH EVAL".format(len(p_single)))
            print("{} MULTI LIST, BATCH EVAL".format(len(p_multi)))
            idx = 0
            res_new = iter_prompt_list(p_single, a_single, vec_single)
            for p_list, a_list, vec_list in tqdm(zip(p_multi, a_multi, vec_multi)):
                res_new.append(''.join(iter_prompt_list(p_list, a_list, vec_list)))

            print("TOTAL RES NUMBER: {}".format(len(res_new)))
            res_detail, res_key_metric = task.evaluate(sample_single + sample_multi, res_new)

            print(res_detail)
            for k, v in res_detail.items():
                f.write(k + ' ============ \n')
                f.write(str(v) + '\n')
            res_key.update(res_key_metric)
            print(res_key)
            f.write(str(res_key))

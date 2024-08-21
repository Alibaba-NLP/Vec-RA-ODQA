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
import readline
from copy import deepcopy
from peft import PeftModel
from time import sleep

def generate_prompts(p, model, tokenizer,
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

    outputs = model.generate(input_ids=input_ids,
                             num_beams=beam_size,
                             do_sample=False,
                             max_length=answer_length+prompt_length,
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


def generate_prompts_chatglm(p, model, tokenizer,
                             prompt_length=256, answer_length=44,
                             beam_size=4, temperature=0.95, top_p=1.0):

    inputs = tokenizer(p,
                       return_tensors="pt",
                       padding=True,
                       truncation=True,
                       max_length=prompt_length)
    inputs = inputs.to(model.device)
    outputs = model.generate(**inputs,
                             num_beams=beam_size,
                             do_sample=False,
                             max_length=answer_length+prompt_length,
                             temperature=temperature,
                             top_k=top_p,
                             repetition_penalty=2.0
                             )
    outputs = [outputs.tolist()[i][len(inputs["input_ids"][i]):] for i in range(len(outputs))]
    response = tokenizer.batch_decode(outputs)
    return response



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name_or_path', type=str, default='../output/2-24-bigscience/bloom-560m/')
    parser.add_argument('-pl', '--prompt_length', type=int, default=1024)
    parser.add_argument('-al', '--answer_length', type=int, default=256)
    parser.add_argument('-beam', '--beam_size', type=int, default=4)
    parser.add_argument('-g', '--gpu', type=int, default=1)
    parser.add_argument('-lora', '--lora', type=str, default='')
    parser.add_argument('-tp', '--temperature', type=float, default=1.0)
    parser.add_argument('-top_p', '--top_p', type=float, default=1.0)
    parser.add_argument('-ptuning', '--ptuning_checkpoint', type=str, default='')

    # parser.add_argument('--seed', type=int, default=42)
    random.seed(42)

    args = parser.parse_args()


    is_chatglm = ('chatglm' in args.model_name_or_path)
    print("is chatglm: {}".format(is_chatglm))
    if is_chatglm:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True, revision='aa51e62')
        # model = AutoModel.from_pretrained(model_args.model_name_or_path, config=config, trust_remote_code=True, revision='6650ae3')
        if (args.ptuning_checkpoint is not None) and len(args.ptuning_checkpoint):
            # Evaluation
            # Loading extra state dict of prefix encoder
            config = AutoConfig.from_pretrained(args.ptuning_checkpoint, trust_remote_code=True, revision='aa51e62')
            model = AutoModel.from_pretrained(args.model_name_or_path, config=config, trust_remote_code=True,
                                              revision='aa51e62')
            prefix_state_dict = torch.load(os.path.join(args.ptuning_checkpoint, "pytorch_model.bin"))
            new_prefix_state_dict = {}
            for k, v in prefix_state_dict.items():
                new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
            model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)
        else:
            model = AutoModel.from_pretrained(args.model_name_or_path, trust_remote_code=True, revision='aa51e62')

        mdoel = model.half().cuda()

    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)

    tokenizer.truncation_side = 'left'

    if args.lora:
        print("loading peft lora model")
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
        if 'checkpoint' in args.lora: emb_path = os.path.join(args.lora, '../gen_tok.emb')
        else: emb_path = os.path.join(args.lora, 'gen_tok.emb')
        if GEN_TOK in tokenizer.vocab:
            print("GEN_TOK is already in the vocab of base model. \n don't load GEN_TOK")
        elif os.path.exists(emb_path):  # save and load the gen_tok embedding
            print('loading embeddings')
            gen_emb = torch.load(emb_path)
            tokenizer.add_special_tokens({'additional_special_tokens': [GEN_TOK]})
            embedding_size = model.get_input_embeddings().weight.shape[0]
            if len(tokenizer) > embedding_size:
                model.resize_token_embeddings(len(tokenizer))
            gen_tok_id = tokenizer.convert_tokens_to_ids(GEN_TOK)
            model.get_input_embeddings().weight.data[gen_tok_id] = gen_emb
        # load peft model
        model = PeftModel.from_pretrained(model, args.lora)

    if args.gpu:
        model = model.cuda()
    model.eval()
    print("use GPU: {}".format(torch.cuda.is_available()))
    print("目前每句话都是独立的session")

    while True:
        prompt = input('你：')
        if is_chatglm:
            tmp_res = generate_prompts_chatglm([prompt], model, tokenizer,
                                               args.prompt_length, args.answer_length,
                                               beam_size=args.beam_size,
                                               temperature=args.temperature,
                                               top_p=args.top_p, )
        else:
            tmp_res = generate_prompts([prompt], model, tokenizer,
                                       args.prompt_length, args.answer_length,
                                       beam_size=args.beam_size,
                                       temperature=args.temperature,
                                       top_p=args.top_p, )

        print('BOT: {}'.format(tmp_res))
        sleep(1)




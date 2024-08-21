
import argparse
import datasets
import transformers
import torch
from transformers.models.bloom.modeling_bloom import BloomForCausalLM
from transformers import AutoTokenizer
from transformers.testing_utils import CaptureLogger
from data.utils import parse_data, files2dataset, wrap_answer
from model.bloom.modeling_vector_infused_bloom import VectorInfusedBloomForCausalLM, VectorInfusedBloomModelConfig
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_collator import DataCollatorWithPadding
from copy import deepcopy
import re
import time
import torch
from transformers.utils import cached_file

device = 'cuda'
GEN_TOK = '[GEN]'

# def merge_lists(la, lb):
#     assert len(la) == len(lb)
#     return [la[i] + lb[i] for i in range(len(la))]



def cut_and_normalize_strs(s):
    s = s.strip('')
    s = s.split('\n')[0].lower()
    s = s.split('.')[0]
    s = s.split(',')[0]
    if 'answer is' in s:
        s = s.split('answer is')[-1]
    # Cut off the first newline, period, or comma
    truncated_text = re.split(r'[\n.,]', s, 1)[0]

    # Remove punctuation
    no_punctuation = re.sub(r'[^\w\s]', '', truncated_text)

    # Remove article
    no_articles = re.sub(r'\b(a|an|the)\b',
                         '',
                         no_punctuation,
                         flags=re.IGNORECASE)

    # Remove duplicated blank spaces
    cleaned_text = re.sub(r'\s+', ' ', no_articles).strip()

    return s



@torch.inference_mode()
def eval(data_args, model_args, training_args):
    val_file = '/mnt/nas-alinlp/zhuochen.zc/others/triviaqa-unfiltered/triviaqa_offline_bert_encoding_dev_5k.json'
    model_path = '/mnt/nas-alinlp/zhuochen.zc/others/retrieval-infused-llm/triviaqa_chen_5k/checkpoint-1500'
    
    raw_datasets = datasets.DatasetDict({})

    validation_files = parse_data(val_file)
    raw_datasets['validation'] = files2dataset(validation_files, data_args, model_args, training_args)
    answers = deepcopy(raw_datasets['validation']['_answer'])

    remove_columns = list(raw_datasets["validation"].features)

    model = torch.load(cached_file(model_path, 'pytorch_model.bin'))

    # model = VectorInfusedBloomForCausalLM.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained('/mnt/nas-alinlp/zhuochen.zc/models/bloomz-1b7')
    # tokenizer.add_special_tokens({'additional_special_tokens': [GEN_TOK]})

    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")
    def preprocess_function(examples):
        with CaptureLogger(tok_logger) as cl:
            output_prompt = tokenizer(examples['_prompt'],
                                      padding=False,
                                      truncation=True,
                                      max_length=data_args.prompt_max_length,
                                      )
            output_prompt['labels'] = output_prompt['input_ids']
            
            wrapped_answer = wrap_answer(examples['_answer'], tokenizer, add_gen=True)

            output_answer = tokenizer(wrapped_answer,
                                      padding=False,
                                      truncation=True,
                                      max_length=data_args.answer_max_length,
                                      )
            output_answer['labels'] = output_answer['input_ids']

            if data_args.retrieval_vec_save_mode == 'value':
                retrieval_vec = torch.Tensor(examples['retrieval_vectors'])
            elif data_args.retrieval_vec_save_mode == 'index':
                retrieval_vec = torch.Tensor(
                    get_vec_from_db(db_dict, examples['retrieval_vectors'])
                )

            else:
                raise NotImplementedError

        if data_args.mask_prompt:
            output_prompt['labels'] = [len(j) * [-100] for j in output_prompt['labels']]

        # output = {k: merge_lists(output_prompt[k], output_answer[k])
                #   for k in output_prompt.keys()}
        
        gen_token_id = tokenizer.encode(GEN_TOK)
        def add_gen_tok(_iter):
            _iter += gen_token_id
            return _iter

        def add_attention_mask(_iter):
            _iter += [1]*len(gen_token_id)
            return _iter
        
        output = dict()
        output['input_ids'] = list(map(add_gen_tok, output_prompt['input_ids']))
        output['attention_mask'] = list(map(add_attention_mask, output_prompt['attention_mask']))
        output['labels'] = output['input_ids'] # They are pointing to the same list!! (python list pointers)
        output['infused_vectors'] = retrieval_vec

        # clm input could be much much longer than block_size
        if "Token indices sequence length is longer than the" in cl.out:
            tok_logger.warning(
                "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
                " before being passed to the model."
            )

        return output
    
    tokenized_datasets = raw_datasets.map(preprocess_function,
                                            num_proc=1,
                                            batched=True,
                                            remove_columns=remove_columns)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer,
                                            padding=True,
                                            pad_to_multiple_of=8)
    dataloader = DataLoader(tokenized_datasets['validation'], batch_size=64, collate_fn=data_collator)


    pred_answer = []
    for batch in tqdm(dataloader):
        # breakpoint()
        batch = batch.to(device)

        generated = model.generate(**batch, max_new_tokens=50)
        
        gen_text = tokenizer.batch_decode(generated, skip_special_tokens=True)

        cleaned_text = list(map(cut_and_normalize_strs, gen_text))
        pred_answer += cleaned_text
    
    # breakpoint()

    hit = 0
    for pred, gold in tqdm(zip(pred_answer, answers), desc=f'{hit}/{len(pred_answer)}'):
        gen_token_index = pred.find('[gen]')
        if gen_token_index != -1:
            pred = pred[gen_token_index+5:]
        
        gold = [g.lower() for g in gold]

        if pred in gold:
            hit += 1
    
    print(f"EM: {(hit/len(pred_answer)):.3f}")
    print(f"{hit}/{len(pred_answer)}")




if __name__ == '__main__':

    # Did not use HfArgumentParser, for simplicity
    data_args_parser = argparse.ArgumentParser(
                                    prog='ProgramName',
                                    description='What the program does',
                                    epilog='Text at the bottom of help')
    data_args_parser.add_argument('--keep_linebreaks', default=True)
    data_args_parser.add_argument('--prompt_max_length', default=128)
    data_args_parser.add_argument('--answer_max_length', default=128)
    data_args_parser.add_argument('--retrieval_vec_save_mode', default='value')
    data_args_parser.add_argument('--mask_prompt', default=False)
    data_args = data_args_parser.parse_args()

    model_args_parser = argparse.ArgumentParser(
                                    prog='ProgramName',
                                    description='What the program does',
                                    epilog='Text at the bottom of help')
    model_args_parser.add_argument('--cache_dir', default=None)
    model_args_parser.add_argument('--use_auth_token', default=False)
    model_args = model_args_parser.parse_args()


    training_args_parser = argparse.ArgumentParser(
                                    prog='ProgramName',
                                    description='What the program does',
                                    epilog='Text at the bottom of help')
    training_args_parser.add_argument('--seed', default=42)
    training_args = training_args_parser.parse_args()
    
    eval(data_args, model_args, training_args)
import sys
import json
import ast
sys.path.append('../')
sys.path.append('../../')
from meta import GEN_TOK, BOS_TOK, EOS_TOK, BASE_DATA_DIR
from datasets import load_dataset, concatenate_datasets
from glob import glob
from itertools import chain

def load_data(file_path):
    d = []
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            try:
                js = json.loads(line)
            except:
                js = ast.literal_eval(line)
            d.append(js)
    return d


def wrap_answer(answer, tokenizer, add_gen=True):
    if type(answer) is str:
        wrap = lambda answer: GEN_TOK + answer + tokenizer.eos_token if add_gen else answer + tokenizer.eos_token
    elif type(answer) is list:
        if answer != []:
            wrap = lambda answer: GEN_TOK + answer[0] + tokenizer.eos_token if add_gen else answer[0] + tokenizer.eos_token
        else:
            wrap = lambda answer: GEN_TOK + '' + tokenizer.eos_token if add_gen else '' + tokenizer.eos_token

    else:
        print(f"Not Implemented answer type: {type(answer)}")
        exit(-1)

    if isinstance(answer, list):
        return [wrap(i) for i in answer]
    else:
        return wrap(answer)

def wrap_ctxs_online(ctxs_list_of_list, model_args):
    assert type(ctxs_list_of_list) is list 
    assert type(ctxs_list_of_list[0]) is list
    assert type(ctxs_list_of_list[0][0]) is str

    total = len(ctxs_list_of_list)
    ret = [i[:model_args.online] for i in ctxs_list_of_list]

    return total, model_args.online, list(chain(*ret))

def wrap_answer_glm(answer, tokenizer, add_gen=True):
    wrap = lambda answer: tokenizer.gmask_token + answer + tokenizer.eos_token if add_gen else answer + tokenizer.eos_token
    if isinstance(answer, list):
        return [wrap(i) for i in answer]
    else:
        return wrap(answer)


def verify_data_name(name):
    extension = name.split(".")[-1]
    assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."


def parse_data(data_str):
    data = data_str.split('[DATA]')
    data = [i.strip() for i in data if len(i) > 0]
    all_data = []
    for d in data:
        d_split = d.split('[SEP]')
        d_split = [i.strip() for i in d_split if len(i.strip()) > 1]
        if len(d_split) == 2:
            dname, portion = d_split
            verify_data_name(dname)
            all_data.append((dname, float(portion), 'prompt', 'answer'))
        elif len(d_split) == 1:
            dname = d_split[0]
            verify_data_name(dname)
            all_data.append((dname, 1.0, 'prompt', 'answer'))
        elif len(d_split) == 4:
            dname, portion, prompt_key, answer_key = d_split
            verify_data_name(dname)
            all_data.append((dname, float(portion), prompt_key, answer_key))
        else:
            raise NotImplementedError("wrong data_file format")
    return all_data


def find_span_positions(text, span):
    ls = len(span)
    pos = []
    for j in range(len(text)):
        if text[j: j+ls] == span:
            pos.append((j, j+ls))
    return pos


def mapping2label(text, type2span):
    lt = len(text)
    labels = ['O']*lt
    for type, spans in type2span.items():
        for s in spans:
            if s == 'None':
                continue
            else:
                for j in range(len(text)):
                    if text[j: j + len(s)] == s:
                        labels[j] = 'B-' + type
                        labels[j+1: j+len(s)] = ['I-' + type]*(len(s)-1)
    return labels


def save_jsonline(data, f_path):
        with open(f_path, 'w') as f:
            dumped = [json.dumps(l, ensure_ascii=False) for l in data]
            for i in dumped[:-1]:
                f.write(i+'\n')
            f.write(dumped[-1])
        print("save data to: {}".format(f_path))



def files2dataset(files, data_args, model_args, training_args):
    """
    parse datasets
    `数据[SEP]比例/shot[SEP]prompt_key[SEP]answer_key[DATA]`
    """
    dataset_args = {}
    merged_datasets = []
    for dataset_name, portion, prompt_key, answer_key in files:
        ds = glob(dataset_name)
        if len(ds) == 0:

            raise NotImplementedError('possibly wrong RE for dataset {}'.format(dataset_name))
        else:
            for _dataset_name in ds:
                data_files = {"train": _dataset_name}
                extension = _dataset_name.split(".")[-1]
                if extension == "txt":
                    extension = "text"
                    dataset_args["keep_linebreaks"] = data_args.keep_linebreaks

                if portion <= 1:
                    assert 0 <= portion
                else:
                    assert int(portion) == portion

                if portion < 1:
                    split = 'train' + '[:{0:.0%}]'.format(portion)
                else:
                    split = 'train'
                
                _train = load_dataset(
                    extension,
                    data_files=data_files,
                    cache_dir=model_args.cache_dir,
                    use_auth_token=True if model_args.use_auth_token else None,
                    split=split,
                    **dataset_args,
                )
                
                print(dataset_name)
                print(files)

                if 'lyh' in dataset_name:
                    answer_key = 'chosen'

                _train = _train.rename_column(prompt_key, '_prompt')
                _train = _train.rename_column(answer_key, '_answer')
                # setting the keys to be consistent
                if portion > 1:
                    _train = _train.shuffle(seed=42).select(range(min(len(_train), int(portion))))
                print("loading data: {}\tsplit: {}\tnum data: {}".format(_dataset_name, portion, len(_train)))
                merged_datasets.append(_train)
                print("Loading {}: {}, {} samples".format(_dataset_name, portion, len(_train)))

    merged_datasets = concatenate_datasets(merged_datasets)
    merged_datasets.shuffle(seed=training_args.seed)
    print('total number of data: {}'.format(len(merged_datasets)))
    return merged_datasets


def run_imap_mp(func, argument_list, num_processes='', is_tqdm=True):
    result_list_tqdm = []
    try:
        import multiprocessing
        if num_processes == '':
            num_processes = multiprocessing.cpu_count()-3
        pool = multiprocessing.Pool(processes=num_processes)
        if is_tqdm:
            from tqdm import tqdm
            for result in tqdm(pool.imap(func=func, iterable=argument_list), total=len(argument_list)):
                result_list_tqdm.append(result)
        else:
            for result in pool.imap(func=func, iterable=argument_list):
                result_list_tqdm.append(result)
        pool.close()
    except:
        result_list_tqdm = list(map(func,argument_list))   
    return result_list_tqdm

def preprocess_function_encoder(example, **kwargs):
    # breakpoint()
    online_ctxs_num = kwargs['online_ctxs_num']
    if online_ctxs_num > 0:
        tokenizer = kwargs['tokenizer']
        _inputs = tokenizer(example['ctxs_online_inputs'][:online_ctxs_num], padding='max_length', truncation=True, max_length=500)
        output = {
            k: example[k] for k in example.data.keys()
        }

        output['ctxs_online_input_ids'] = _inputs['input_ids']
        output['ctxs_online_token_type_ids'] = _inputs['token_type_ids']
        output['ctxs_online_attention_mask'] = _inputs['attention_mask']
    else:
        output = {
            k: example[k] for k in example.data.keys()
        }
        output['ctxs_online_input_ids'] = []
        output['ctxs_online_token_type_ids'] = []
        output['ctxs_online_attention_mask'] = []

    return output

def print_trainable(model):
    trainable = 0
    total = 0
    for p in model.parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    
    print('='*50)
    print(f"Trainable parameters: {round(trainable/10**6, 2)}M/{round(total/10**6, 2)}M")
    print('='*50)

def training_parameters(model, config):

    # breakpoint()

    if config.train_encoder and config.train_decoder:
        p = model.parameters()
    
    elif config.train_encoder and (not config.train_decoder):
        p = model.encoder.parameters()
    
    elif (not config.train_encoder) and config.train_decoder:
        # Fail when tied weights of embeddings and lm_head
        # p = list(model.transformer.parameters()) + \
            # list(model.lm_head.parameters())
        
        p = [i for i in model.parameters() if i.requires_grad]
    
    else:
        raise NotImplementedError
        
    p = list(p)
    total = sum([i.numel() for i in p])
    print(f"Trainable: {total/10**6}M Params")
    return p
from glob import glob
import pandas as pd
import numpy as np
import argparse
import datetime
import os


def F(df, filters):
    for key, substr in filters:
        if type(substr) == str:
            df = df[df[key].str.contains(substr)]
        else:
            df = df[df[key] == substr]
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str, default='3-5')
    args = parser.parse_args()

    eval_path = '../../output/evaluate/{}/'.format(args.path)
    files = glob(eval_path + '*.res')
    eval_path = '../../output/evaluate/{}/*/'.format(args.path)
    files += glob(eval_path + '*.res')
    eval_path = '../../output/evaluate/{}/*/*/'.format(args.path)
    files += glob(eval_path + '*.res')
    eval_path = '../../output/evaluate/{}/*/*/*/'.format(args.path)
    files += glob(eval_path + '*.res')
    records = []
    for f_name in files:
        with open(f_name, 'r') as f:
            try:
                res = f.read().split('\n')[-1]
                metric = eval(res)
            except:
                print("error in: {}".format(f_name))
                continue
        # if 'model' not in metric:
        #     continue
        metric['model_info'] = f_name
        records.append(metric)
    datetime_str = datetime.datetime.now().strftime("%Y%m%d-%H%M-%s")
    df = pd.DataFrame.from_records(records)
    df.to_csv('/mnt/workspace/data/{}.csv'.format(datetime_str))
    # df_group = F(df, [])[-df['data_name'].str.contains('stock')].groupby(['task', 'model_name_or_path', 'lora', 'save_name', 'top_p', 'beam_size', 'temperature', 'inst']).agg(['mean', 'count'])
    # df_group.to_csv('/mnt/workspace/data/{}.g.csv'.format(datetime_str))
    if 'ptuning_checkpoint' in df: df['ptuning_checkpoint'] = df['ptuning_checkpoint'].fillna('')
    pivot_index = ['task', 'model_name_or_path', 'lora', 'ptuning_checkpoint','save_name', 'top_p', 'beam_size', 'temperature', 'inst']
    tasks = set(list(df['task']))
    save_path = os.path.join('/mnt/workspace/data/', datetime_str)
    os.mkdir(save_path)
    save_all_path = os.path.join(save_path, 'all.csv')
    print("saving all at: {}".format(save_all_path))
    df.to_csv(save_all_path)
    for t in tasks:
        new_df = df[df['task'] == t]
        pt = pd.pivot_table(new_df, values='micro-f1', index=pivot_index, columns=['data_file'], )
        pt = pt.round(4)
        pt.to_csv(os.path.join(save_path, '{}.pt.csv'.format(t)))
        print("saving grouped at: {}".format('{}/{}.pt.csv'.format(save_path, t)))

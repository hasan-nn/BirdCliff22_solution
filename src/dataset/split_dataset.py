import json
import os
import argparse

import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm



def parse_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--durations_path', type = str, default = '../../../../datasets/birdclef-2022/generated_data/durations.json')
    arg('--bcounts_path', type = str, default = '../../../../datasets/birdclef-2022/generated_data/bird_counts.json')
    arg('--path',type = str,default= '../../../../datasets/birdclef-2022/train_metadata.csv')
    arg('--out_dir',type=str,default='../../../../datasets/birdclef-2022/generated_data/splits')
    arg('--seed', type = int, default = 911)
    arg('--folds',type = int,default = 6)
    arg('--max_duration',type = float, default = 4.0 * 60.0)
    arg('--min_bird_count',type=int,default = 3)
    arg('--duration_step',type = int,default=30)
    arg('--min_group_count',type=int,default=12)
    return parser.parse_args()

def load_json(path):
    return json.load(open(path,'r'))

def main():
    args = parse_args()
    os.makedirs(args.out_dir,exist_ok=True)
    df = pd.read_csv(args.path)
    bird_counts = load_json(args.bcounts_path)
    durations = load_json(args.durations_path)
    l_samples = len(df)

    
    group_counts = {}
    for i,row in df.iterrows():
        secondary_labels = [l.lstrip('\'').rstrip('\'') for l in row['secondary_labels'].lstrip('[').rstrip(']').split(', ') if l!= '']
        primary_label = row['primary_label']
        filename = row['filename']
        duration = durations[filename]
        labels = [row['primary_label'],*secondary_labels]
        div = int(duration//args.duration_step)
        for label in labels:
            if label not in group_counts.keys():
                group_counts[label] = []
            group_counts[label].append(div)

    discarded = 0
    for k,v in group_counts.items():
        c = np.array([v.count(x) for x in set(v)])
        c = c[c>=args.folds].sum()
        group_counts[k] = c#max([v.count(x) for x in set(v)])
        if group_counts[k] < args.min_group_count:
            discarded+=1
            

    forCV = []
    split_keys = []
    _durations = []
    for i,row in df.iterrows():
        secondary_labels = [l.lstrip('\'').rstrip('\'') for l in row['secondary_labels'].lstrip('[').rstrip(']').split(', ') if l!= '']
        primary_label = row['primary_label']
        labels = [row['primary_label'],*secondary_labels]
        filename = row['filename']
        duration = durations[filename]
        

        cv_keep = False
        if duration <= args.max_duration:
            cv_keep = True
        min_bird_count = l_samples + 1
        discard_duration = False
        for label in labels:
            if group_counts[label] < args.min_group_count:
                discard_duration = True
            if bird_counts[label] < args.min_bird_count:
                if not cv_keep:
                    print(f'audio {filename} is longer than {args.max_duration} second but will be kept for cross validation')
                cv_keep = True
                min_bird_count = min(min_bird_count,bird_counts[label])


        if min_bird_count <= args.folds:
            print(f'audio {filename} has rare species of only {min_bird_count} samples >> discarding duration for Kfolds!')
            split_key = f'{primary_label}_0'

        elif discard_duration:
            print(f'audio {filename} has species that cant be split evenly by duration >> discarding duration for Kfolds!')
            split_key = f'{primary_label}_0'

        else:
            #print(duration%args.duration_step)
            split_key = f'{primary_label}_{int(duration//args.duration_step)}'

        _durations.append(duration)
        forCV.append(cv_keep)
        split_keys.append(split_key)

    df['for_cv'] = forCV
    df['split_key'] = split_keys
    df['duration(sec)'] = _durations 
    cv_df = df[df['for_cv'] == True]
    test_df = df[df['for_cv'] == False]

    cv_df.reset_index(inplace=True,drop=True)
    test_df.reset_index(inplace=True,drop=True)

    X = cv_df.groupby('filename')['split_key'].first().index.values
    y = cv_df.groupby('filename')['split_key'].first().values
    skf = StratifiedKFold(n_splits = args.folds, random_state = args.seed, shuffle=True) 
    
    for i,(tfold,vfold) in enumerate(skf.split(X,y)):
        cv_df.loc[cv_df['filename'].isin(X[vfold]),'fold']=int(i)

    folds=[int(fold) for fold in cv_df.groupby('fold').first().index.values]
    for fold in folds:
            print(f'fold:\t{fold}')
            print(cv_df.loc[cv_df['fold']==fold].set_index(['fold','primary_label']).count(level='primary_label'))
    cv_df.to_csv(f'{args.out_dir}/cv_folds.csv')
    test_df.to_csv(f'{args.out_dir}/local_test.csv')
    
if __name__ == '__main__':
    main()
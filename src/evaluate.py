import lightning as L
import os

import torch
from timbremetrics import TimbreMetric, print_results, list_datasets
from timbremetrics.utils import merge_metrics
from sklearn.model_selection import KFold

from paths import CHECKPOINTS_DIR, CONFIGS_DIR
from utils import load_yaml_config

def main():

    config = load_yaml_config(os.path.join(CONFIGS_DIR, 'train_one_epoch.yaml'))

    n_splits = config['validation']['n_splits']
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=config['seed'])
    timbre_space_datasets = list_datasets()

    one_epoch_dir = os.path.join(CHECKPOINTS_DIR, 'one-epoch-three-splits-with-test-v2')
    one_epoch_files = sorted(os.listdir(one_epoch_dir))

    all_ = {} # super nested. > rep > split_id > dist > metric > test_score

    for i in range(len(one_epoch_files)):
        splits = one_epoch_files[i].split('=')
        valid_score = float(splits[1][1:6])
        test_score = float(splits[1][8:13])
        parts = splits[0].split('-')
        rep = parts[1].split('_')[-1]
        if rep not in all_:
            all_[rep] = {}
        split_id = int(parts[2][-1]) - 1
        if split_id not in all_[rep]:
            all_[rep][split_id] = {}
        dist = parts[3].split('_')[0]
        if dist not in all_[rep][split_id]:
            all_[rep][split_id][dist] = {}
        metric = parts[3].split('_')[1:]
        metric = '_'.join(metric)
        all_[rep][split_id][dist][metric] = test_score

    test_sets = []
    for valid_idx, test_idx in kf.split(timbre_space_datasets):
        test_sets.append([timbre_space_datasets[i] for i in test_idx])

    for rep in all_:
        merged_scores = {} # > dist > metric > test_score
        dists = list(all_[rep][0].keys())
        for dist in dists:
            metrics_list = [all_[rep][i][dist] for i in range(n_splits)]
            merged_scores[dist] = merge_metrics(metrics_list, test_sets)
        print(f"{rep}:")
        for dist in merged_scores:
            print(f"  {dist}:")
            for metric in merged_scores[dist]:
                print(f"    {metric}: {merged_scores[dist][metric]: .3f}")

if __name__ == '__main__':
    main()

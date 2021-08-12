import json
import os
import random
import sys
import argparse
import copy
import abc
import numpy as np
import tqdm
from prettytable import PrettyTable

from datasets import get_environments
from model_selection import ensure_dict_path, get_best_hparams
from utils import get_latex_table

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Collect result of hyper parameter sweep")
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--latex", action='store_true')
    flags = parser.parse_args()

    ## Load records
    records = {}
    for i, subdir in tqdm.tqdm(list(enumerate(os.listdir(flags.results_dir)))):
        results_path = os.path.join(flags.results_dir, subdir)
        try:
            with open(results_path, "r") as f:
                run_results = json.load(f)

                sub_records = ensure_dict_path(records, run_results['flags']['dataset'])
                sub_records = ensure_dict_path(sub_records, run_results['flags']['objective'])
                sub_records = ensure_dict_path(sub_records, run_results['flags']['test_env'])
                sub_records = ensure_dict_path(sub_records, run_results['flags']['hparams_seed'])
                sub_records.update({run_results['flags']['trial_seed']: run_results})

        except IOError:
            pass
        

    model_selection_methods = ['train_domain_validation',
                               'test_domain_validation']

    for model_selection in model_selection_methods:

        for dataset, dat_dict in records.items():

            t = PrettyTable()
            envs = get_environments(dataset)
            t.field_names = ['Objective'] + envs

            for obj, obj_dict in dat_dict.items():
                obj_results = [obj]
                for i, e in enumerate(envs):
                    val_acc, test_acc = get_best_hparams(obj_dict[i], model_selection)

                    obj_results.append(test_acc)

                t.add_row(obj_results)

            max_width = {}
            min_width = {}
            for n in t.field_names:
                max_width.update({n: 15})
                min_width.update({n: 15})
            t._min_width = min_width
            t._max_width = max_width
            t.float_format = '.3'
            
            if flags.latex:
                print(get_latex_table(t))
            else:
                print(t.get_string(title=model_selection + ' Results for ' + dataset))





    
            
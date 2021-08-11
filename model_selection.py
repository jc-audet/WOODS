
import json
import os
import random
import sys
import argparse

import numpy as np
import tqdm
from prettytable import PrettyTable
from datasets import get_environments
from utils import get_latex_table
import abc

def ensure_dict_path(dict, key):

    if key not in dict.keys():
        dict[key] = {}

    return dict[key]

def get_best_hparams(records, selection_method):
    """
    docstring
    """

    val_dict = {}
    test_dict = {}
    for h_seed, h_dict in records.items():
        val_acc, test_acc = selection_method(h_dict)

        val_dict[h_seed] = val_acc
        test_dict[h_seed] = test_acc


    best_seed = [k for k,v in val_dict.items() if v==max(val_dict.values())][0]

    return val_dict[best_seed], test_dict[best_seed]

def train_domain_validation(records):

    val_acc = []
    test_acc = []
    for t_seed, t_dict in records.items():
        hparams = t_dict.pop('hparams')
        flags = t_dict.pop('flags')
        env_name = get_environments(flags['dataset'])

        val_keys = [str(e)+'_in_acc' for i,e in enumerate(env_name) if i != flags['test_env']]
        test_keys = [str(env_name[flags['test_env']])+'_'+split+'_acc' for split in ['in', 'out']]

        val_dict = {}
        test_dict = {}
        for step, step_dict in t_dict.items():

            val_array = [step_dict[k] for k in val_keys]
            val_dict[step] = np.mean(val_array)

            test_array = [step_dict[k] for k in test_keys]
            test_dict[step] = np.mean(test_array)

        ## Picking the max value from a dict
        # Fastest:
        best_step = [k for k,v in val_dict.items() if v==max(val_dict.values())][0]
        # Cleanest:
        # best_step = max(val_dict, key=val_dict.get)
        # Alternative:
        # best_step = max(val_dict, key=lambda k: val_dict[k])

        val_acc.append(val_dict[best_step])
        test_acc.append(test_dict[best_step])

    return np.mean(val_acc), np.mean(test_acc)

def test_domain_validation(records):

    val_acc = []
    test_acc = []
    for t_seed, t_dict in records.items():
        hparams = t_dict.pop('hparams')
        flags = t_dict.pop('flags')
        env_name = get_environments(flags['dataset'])

        val_keys = [str(e)+'_in_acc' for i,e in enumerate(env_name) if i != flags['test_env']]
        test_keys = [str(env_name[flags['test_env']])+'_'+split+'_acc' for split in ['in', 'out']]

        last_step = max([int(step) for step in t_dict.keys()])

        val_array = [t_dict[str(last_step)][k] for k in val_keys]
        t_val_acc = np.mean(val_array)

        test_array = [t_dict[str(last_step)][k] for k in test_keys]
        t_test_acc = np.mean(test_array)

        val_acc.append(t_val_acc)
        test_acc.append(t_test_acc)

    return np.mean(val_acc), np.mean(test_acc)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Collect result of hyper parameter sweep")
    parser.add_argument("--results_dir", type=str, required=True)
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
        
    model_selection = 'Train Validation'
    for dataset, dat_dict in records.items():

        t = PrettyTable()
        envs = get_environments(dataset)
        t.field_names = ['Objective'] + envs

        for obj, obj_dict in dat_dict.items():
            obj_results = [obj]
            for i, e in enumerate(envs):
                val_acc, test_acc = get_best_hparams(obj_dict[i], train_domain_validation)

                obj_results.append(test_acc)

            t.add_row(obj_results)

        max_width = {}
        min_width = {}
        for n in t.field_names:
            max_width.update({n: 15})
            min_width.update({n: 15})
        t._min_width = min_width
        t._max_width = max_width
        t.float_format = '.2'
        print(t.get_string(title=model_selection + ' Results for ' + dataset))
        # print(get_latex_table(t))





    
            
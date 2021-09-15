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

from temporal_OOD import datasets
from temporal_OOD import model_selection
from temporal_OOD import utils

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Collect result of hyper parameter sweep")
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--latex", action='store_true')
    flags = parser.parse_args()

    ## Check if all run are there
    # utils.check_file_integrity(flags.results_dir)

    ## Load records
    records = {}
    for i, subdir in tqdm.tqdm(list(enumerate(os.listdir(flags.results_dir))), desc="Loading Results"):
        results_path = os.path.join(flags.results_dir, subdir)
        try:
            with open(results_path, "r") as f:
                run_results = json.load(f)

                sub_records = model_selection.ensure_dict_path(records, run_results['flags']['dataset'])
                sub_records = model_selection.ensure_dict_path(sub_records, run_results['flags']['objective'])
                sub_records = model_selection.ensure_dict_path(sub_records, run_results['flags']['test_env'])
                sub_records = model_selection.ensure_dict_path(sub_records, run_results['flags']['hparams_seed'])
                sub_records.update({run_results['flags']['trial_seed']: run_results})
        except KeyError:
            pass
        except IOError:
            pass
        

    model_selection_methods = ['train_domain_validation',
                               'test_domain_validation']

    for ms_method in model_selection_methods:

        for dataset, dat_dict in records.items():

            t = PrettyTable()
            envs = datasets.get_environments(dataset)
            t.field_names = ['Objective'] + envs

            for obj, obj_dict in dat_dict.items():
                obj_results = [obj]
                for i, e in enumerate(envs):
                    val_acc, val_var, test_acc, test_var = model_selection.get_best_hparams(obj_dict[i], ms_method)

                    if flags.latex:
                        obj_results.append(" ${acc:.2f} \pm {var:.2f}$ ".format(acc=test_acc*100, var=test_var*100))
                    else:
                        obj_results.append(" {acc:.2f} +/- {var:.2f} ".format(acc=test_acc*100, var=test_var*100))
                        
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
                print(utils.get_latex_table(t))
            else:
                print(t.get_string(title=ms_method + ' Results for ' + dataset))





    
            
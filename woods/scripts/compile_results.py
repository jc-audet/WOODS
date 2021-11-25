"""Compile resuls from a hyperparameter sweep and perform model selection strategies

See https://woods.readthedocs.io/en/latest/running_a_sweep.html to learn more about usage.
"""

import os
import json
import tqdm
import argparse
import warnings
import numpy as np
from pptree import *
import pprint
from prettytable import PrettyTable

from woods import datasets
from woods import model_selection
from woods import utils

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Collect result of hyper parameter sweep")
    parser.add_argument('--mode', nargs='?', default='tables', const='tables', choices=['tables', 'hparams', 'IID'])
    parser.add_argument("--results_dirs", type=str, nargs='+', required=True)
    parser.add_argument("--ignore_integrity_check", action='store_true')
    parser.add_argument("--latex", action='store_true')
    flags = parser.parse_args()

    ## Load records in a nested dictionnary (Dataset > Objective > Test env > Trial Seed > Hparams Seed)
    records = {}
    # For all directories you want to collect data from
    for results_dir in flags.results_dirs:

        ## Check if all run are there
        if flags.ignore_integrity_check:
            warnings.warn("Sweep results are reported without integrity check.")
        else:
            utils.check_file_integrity(results_dir)

        for subdir in tqdm.tqdm(os.listdir(os.path.join(results_dir,'logs')), desc="Loading Results"):
            results_path = os.path.join(results_dir, 'logs', subdir)
            try:
                with open(results_path, "r") as f:
                    run_results = json.load(f)

                    sub_records = model_selection.ensure_dict_path(records, run_results['flags']['dataset'])
                    sub_records = model_selection.ensure_dict_path(sub_records, run_results['flags']['objective'])
                    sub_records = model_selection.ensure_dict_path(sub_records, run_results['flags']['test_env'])
                    sub_records = model_selection.ensure_dict_path(sub_records, run_results['flags']['trial_seed'])
                    sub_records.update({run_results['flags']['hparams_seed']: run_results})
            except KeyError:
                pass
            except IOError:
                pass

    # Choose model selection under study
    model_selection_methods = [ 'train_domain_validation',
                                'test_domain_validation']

    if 'hparams' in flags.mode:

        # Perform model selection onto the checkpoints from results
        for ms_method in model_selection_methods:
            ms = Node(' ' + ms_method + ' ')

            for dataset_name, dataset_dict in records.items():
                ds = Node(' ' + dataset_name + ' ', ms)
                envs = datasets.get_sweep_envs(dataset_name)

                for objective_name, objective_dict in dataset_dict.items():
                    obj = Node(' ' + objective_name + ' ', ds)

                    for env_id, env_name in enumerate(envs):
                        env = Node(" Env "+str(env_name) + ' ', obj)
                        # If the environment wasn't part of the sweep, that's fine, we just can't report those results
                        flags_dict, hparams_dict, val_dict, test_dict = model_selection.get_best_hparams(objective_dict[env_id], ms_method)

                        keys = list(flags_dict.keys())
                        keys.sort()
                        for t_seed in keys:
                            t = Node(' Seed ' + str(t_seed) + ': ' + utils.get_job_name(flags_dict[t_seed]) + ' (val: {:.2f}, test: {:.2f})'.format(val_dict[t_seed], test_dict[t_seed]), env)

            print_tree(ms)  

    elif 'tables' in flags.mode:

        # Perform model selection onto the checkpoints from results
        for ms_method in model_selection_methods:

            for dataset_name, dataset_dict in records.items():
                t = PrettyTable()
                sweep_envs = datasets.get_sweep_envs(dataset_name)
                all_envs = datasets.get_environments(dataset_name)
                envs = [all_envs[i] for i in sweep_envs]
                t.field_names = ['Objective'] + envs + ["Average"]

                for objective_name, objective_dict in dataset_dict.items():
                    acc_arr = []
                    obj_results = [objective_name]

                    for env_id in sweep_envs:
                        # If the environment wasn't part of the sweep, that's fine, we just can't report those results
                        if env_id not in objective_dict.keys():
                            obj_results.append(" X ")
                        else:
                            val_acc, val_var, test_acc, test_var = model_selection.get_chosen_test_acc(objective_dict[env_id], ms_method)
                            acc_arr.append(test_acc*100)

                            if flags.latex:
                                obj_results.append(" ${acc:.2f} \pm {var:.2f}$ ".format(acc=test_acc*100, var=test_var*100))
                            else:
                                obj_results.append(" {acc:.2f} +/- {var:.2f} ".format(acc=test_acc*100, var=test_var*100))
                            
                    avg_test = np.mean(acc_arr)
                    obj_results.append(" {acc:.2f} ".format(acc=avg_test))
                    t.add_row(obj_results)

                max_width = {}
                min_width = {}
                for n in t.field_names:
                    max_width.update({n: 20})
                    min_width.update({n: 20})
                t._min_width = min_width
                t._max_width = max_width
                t.float_format = '.3'
                
                if flags.latex:
                    print(utils.get_latex_table(t, caption=ms_method + ' Results for ' + dataset_name))
                else:
                    print(t.get_string(title=ms_method + ' Results for ' + dataset_name))

    elif 'IID' in flags.mode:

        for dataset_name, dataset_dict in records.items():
            t = PrettyTable()
            envs = datasets.get_environments(dataset_name)
            t.field_names = ['Objective'] + envs + ["Average"]

            acc_arr = []
            obj_results = ['IID ERM']

            val_acc, val_var, test_acc, test_var = model_selection.get_chosen_test_acc(dataset_dict['ERM'][None], 'IID_validation')
            acc_arr.append(test_acc*100)

            for acc, var in zip(val_acc, val_var):
                if flags.latex:
                    obj_results.append(" ${acc:.2f} \pm {var:.2f}$ ".format(acc=acc*100, var=var*100))
                else:
                    obj_results.append(" {acc:.2f} +/- {var:.2f} ".format(acc=acc*100, var=var*100))
                
            avg_test = np.mean(acc_arr)
            obj_results.append(" {acc:.2f} ".format(acc=avg_test))
            t.add_row(obj_results)

            max_width = {}
            min_width = {}
            for n in t.field_names:
                max_width.update({n: 20})
                min_width.update({n: 20})
            t._min_width = min_width
            t._max_width = max_width
            t.float_format = '.3'
            
            if flags.latex:
                print(utils.get_latex_table(t, caption='IID Results for ' + dataset_name))
            else:
                print(t.get_string(title='IID Results for ' + dataset_name))






    
            
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
    parser.add_argument('--mode', nargs='?', default='tables', const='tables', choices=['tables', 'summary', 'hparams', 'IID'])
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

    if 'hparams' in flags.mode:

        raise NotImplementedError('Quarantined due to changes in the repo')

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
        for dataset_name, dataset_dict in records.items():

            model_selection_methods = model_selection.get_model_selection(dataset_name)

            for ms_method in model_selection_methods:

                if datasets.get_paradigm(dataset_name) == 'domain_generalization':
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
                                val_acc, val_var, test_acc, test_var, _ = model_selection.choose_model_domain_generalization(objective_dict[env_id], ms_method)
                                acc_arr.append(test_acc*100)

                                if flags.latex:
                                    obj_results.append(" ${acc:.2f} \pm {var:.2f}$ ".format(acc=test_acc, var=test_var))
                                else:
                                    obj_results.append(" {acc:.2f} +/- {var:.2f} ".format(acc=test_acc, var=test_var))
                                
                        avg_test = np.mean(acc_arr)
                        obj_results.append(" {acc:.2f} ".format(acc=avg_test))
                        t.add_row(obj_results)

                if datasets.get_paradigm(dataset_name) == 'subpopulation_shift':
                    print(dataset_name)
                    
                    t = PrettyTable()
                    all_envs = datasets.get_environments(dataset_name)
                    sweep_envs = datasets.get_sweep_envs(dataset_name)
                    domain_weights = datasets.get_domain_weights(dataset_name)
                    t.field_names = ['Objective', 'Average', 'Worse']

                    for objective_name, objective_dict in dataset_dict.items():
                        acc_arr = []
                        obj_results = [objective_name]

                        for env_id in sweep_envs:
                            # If the environment wasn't part of the sweep, that's fine, we just can't report those results
                            avg_performance, avg_performance_var, worse_performance, worse_performance_var, _ = model_selection.choose_model_subpopulation(objective_dict[env_id], ms_method, domain_weights)

                            if flags.latex:
                                obj_results.append(" ${acc:.2f} \pm {var:.2f}$ ".format(acc=avg_performance, var=avg_performance_var))
                                obj_results.append(" ${acc:.2f} \pm {var:.2f}$ ".format(acc=worse_performance, var=worse_performance_var))
                            else:
                                obj_results.append(" {acc:.2f} +/- {var:.2f} ".format(acc=avg_performance, var=avg_performance_var))
                                obj_results.append(" {acc:.2f} +/- {var:.2f} ".format(acc=worse_performance, var=worse_performance_var))
                                
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

    elif 'summary' in flags.mode:

        all_model_selection_methods = model_selection.model_selection_methods
        # Perform model selection onto the checkpoints from results
        for ms_method in all_model_selection_methods:

            dataset_to_evaluate = []
            for dataset in records.keys():
                if ms_method in model_selection.get_model_selection(dataset):
                    dataset_to_evaluate.append(dataset)

            dataset_to_evaluate_ordered = []
            for dataset in datasets.DATASETS:
                if dataset in dataset_to_evaluate and 'Unbalanced' not in dataset:
                    dataset_to_evaluate_ordered.append(dataset)

            t = PrettyTable()
            t.field_names = ['Objective'] + list(dataset_to_evaluate_ordered) + ["Average"]

            acc_dict = {}
            var_dict = {}
            for dataset_name in dataset_to_evaluate_ordered:

                dataset_dict = records[dataset_name]

                dataset_paradigm = datasets.get_paradigm(dataset_name)
                dataset_measure = datasets.get_performance_measure(dataset_name)
                dataset_model_selection = model_selection.get_model_selection(dataset_name)

                performance_multiplier = 100
                if dataset_measure == 'rmse':
                    performance_multiplier = 1

                acc_dict[dataset_name] = {}
                var_dict[dataset_name] = {}
                sweep_envs = datasets.get_sweep_envs(dataset_name)
                all_envs = datasets.get_environments(dataset_name)
                envs = [all_envs[i] for i in sweep_envs]

                for objective_name, objective_dict in dataset_dict.items():

                    acc_arr = []
                    acc_var = []
                    all_sweep_env = True

                    if dataset_paradigm == 'domain_generalization':

                        for env_id in sweep_envs:
                            # If the environment wasn't part of the sweep, that's NOT fine, we need all test environment for the average
                            if env_id not in objective_dict.keys():
                                all_sweep_env = False
                            else:
                                val_acc, val_var, test_acc, test_var, _ = model_selection.choose_model_domain_generalization(objective_dict[env_id], ms_method)
                                acc_arr.append(test_acc*performance_multiplier)
                                acc_var.append(test_var*performance_multiplier)
                                
                        if all_sweep_env:
                            avg_test = np.mean(acc_arr)
                            var_test = np.mean(acc_var)
                            acc_dict[dataset_name][objective_name] =  avg_test
                            var_dict[dataset_name][objective_name] =  var_test
                        else:
                            acc_dict[dataset_name][objective_name] = None

                    elif dataset_paradigm == 'subpopulation_shift':

                        domain_weights = datasets.get_domain_weights(dataset_name)

                        for env_id in sweep_envs:
                            if env_id not in objective_dict.keys():
                                all_sweep_env = False
                            else:
                                # If the environment wasn't part of the sweep, that's fine, we just can't report those results
                                avg_performance, avg_performance_var, worse_performance, worse_performance_var, _ = model_selection.choose_model_subpopulation(objective_dict[env_id], ms_method, domain_weights)

                                acc_arr.append(worse_performance*performance_multiplier)
                                acc_var.append(worse_performance_var*performance_multiplier)
                                
                        if all_sweep_env:
                            avg_test = np.mean(acc_arr)
                            var_test = np.mean(acc_var)
                            acc_dict[dataset_name][objective_name] =  avg_test
                            var_dict[dataset_name][objective_name] =  var_test
                        else:
                            acc_dict[dataset_name][objective_name] = None
                    else:
                        raise ValueError("There is a problem here")

            # Flip the nested dict so the order is objective -> dataset
            flipped_acc = {}
            flipped_var = {}
            for dataset_name in acc_dict.keys():
                for objective_name in acc_dict[dataset_name].keys():
                    if objective_name not in flipped_acc.keys():
                        flipped_acc[objective_name] = {}
                        flipped_var[objective_name] = {}
                    flipped_acc[objective_name][dataset_name] = acc_dict[dataset_name][objective_name]
                    flipped_var[objective_name][dataset_name] = var_dict[dataset_name][objective_name]
            
            # Ensure that all objectives have all datasets
            for objective_name in flipped_acc.keys():
                for dataset_name in flipped_acc[objective_name].keys():
                    if dataset_name not in flipped_acc[objective_name].keys():
                        flipped_acc[objective_name][dataset_name] = None

            # Construct the table
            for objective_name in flipped_acc.keys():
                obj_results = [objective_name]

                for dataset_name in t.field_names[1:-1]:#flipped_acc[objective_name].keys():
                    # print(flipped_acc[objective_name][dataset_name])
                    # if flipped_acc[objective_name][dataset_name] is None:
                    #     obj_results.append(" X ")
                    #     print("hello")
                    # else:
                    try:
                        if flags.latex:
                            obj_results.append(" ${acc:.1f} \pm {var:.1f}$ ".format(acc=flipped_acc[objective_name][dataset_name], var=flipped_var[objective_name][dataset_name]))
                        else:
                            obj_results.append(" {acc:.1f} +/- {var:.1f} ".format(acc=flipped_acc[objective_name][dataset_name], var=flipped_var[objective_name][dataset_name]))
                    except KeyError:
                        obj_results.append(" X ")

                obj_results.append(" {acc:.1f} ".format(acc=np.mean(list(flipped_acc[objective_name].values()))))
                # print(t.field_names, obj_results)
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

            val_acc, val_var, test_acc, test_var, _ = model_selection.choose_model_domain_generalization(dataset_dict['ERM'][None], 'IID_validation')
            acc_arr.append(test_acc*100)

            for acc, var in zip(test_acc, test_var):
                if flags.latex:
                    obj_results.append(" ${acc:.2f} \pm {var:.2f}$ ".format(acc=acc, var=var))
                else:
                    obj_results.append(" {acc:.2f} +/- {var:.2f} ".format(acc=acc, var=var))
                
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






    
            

"""Defining the model selection strategies"""

import copy
import numpy as np

from woods import datasets
from woods import utils

def ensure_dict_path(dict, key):
    """Ensure that a path of a nested dictionnary exists. 
    
    If it does, return the nested dictionnary within. If it does not, create a nested dictionnary and return it.

    Args:
        dict (dict): Nested dictionnary to ensure a path
        key (str): Key to ensure has a dictionnary in 

    Returns:
        dict: nested dictionnary
    """
    if key not in dict.keys():
        dict[key] = {}

    return dict[key]

def get_best_hparams(records, selection_name):
    """
    docstring
    """

    if selection_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(selection_name))
    selection_method = globals()[selection_name]

    flags_dict = {}
    hparams_dict = {}
    val_best_acc = {}
    test_best_acc = {}
    for t_seed, t_dict in records.items():
        val_acc_dict = {}
        val_var_dict = {}
        test_acc_dict = {}
        test_var_dict = {}
        for h_seed, h_dict in t_dict.items():
            val_acc, test_acc = selection_method(h_dict)

            val_acc_dict[h_seed] = val_acc
            test_acc_dict[h_seed] = test_acc

        best_seed = [k for k,v in val_acc_dict.items() if v==max(val_acc_dict.values())][0]

        flags_dict[t_seed] = records[t_seed][best_seed]['flags']
        hparams_dict[t_seed] = records[t_seed][best_seed]['hparams']
        val_best_acc[t_seed] = val_acc_dict[best_seed]
        test_best_acc[t_seed] = test_acc_dict[best_seed]
    
    return flags_dict, hparams_dict, val_best_acc, test_best_acc

def get_chosen_test_acc(records, selection_name):
    """
    docstring
    """

    if selection_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(selection_name))
    selection_method = globals()[selection_name]

    val_acc_arr = []
    test_acc_arr = []
    for t_seed, t_dict in records.items():
        val_acc_dict = {}
        val_var_dict = {}
        test_acc_dict = {}
        test_var_dict = {}
        for h_seed, h_dict in t_dict.items():
            val_acc, test_acc = selection_method(h_dict)

            val_acc_dict[h_seed] = val_acc
            test_acc_dict[h_seed] = test_acc

        best_seed = [k for k,v in val_acc_dict.items() if v==max(val_acc_dict.values())][0]

        val_acc_arr.append(val_acc_dict[best_seed])
        test_acc_arr.append(test_acc_dict[best_seed])

    return np.mean(val_acc_arr, axis=0), np.std(val_acc_arr, axis=0) / np.sqrt(len(val_acc_arr)), np.mean(test_acc_arr, axis=0), np.std(test_acc_arr, axis=0) / np.sqrt(len(test_acc_arr))

def IID_validation(records):
    """ Return the IID validation model section accuracy of a single training run. This is for ONLY for sweeps with no test environments

        max_{step in checkpoint}( mean(train_envs) )

    Args:
        records ([type]): [description]

    Returns:
        [type]: [description]
    """

    # Make copy of record
    records = copy.deepcopy(records)

    flags = records.pop('flags')
    hparams = records.pop('hparams')
    env_name = datasets.get_environments(flags['dataset'])

    val_keys = [str(e)+'_out_acc' for e in env_name]

    val_dict = {}
    val_arr_dict = {}
    for step, step_dict in records.items():

        val_array = [step_dict[k] for k in val_keys]
        val_arr_dict[step] = copy.deepcopy(val_array)
        val_dict[step] = np.mean(val_array)

    ## Picking the max value from a dict
    # Fastest:
    best_step = [k for k,v in val_dict.items() if v==max(val_dict.values())][0]
    # Cleanest:
    # best_step = max(val_dict, key=val_dict.get)
    
    return val_arr_dict[best_step], val_arr_dict[best_step]

def train_domain_validation(records):
    """ Return the train-domain validation model section accuracy of a single training run

        max_{step in checkpoint}( mean(train_envs) )

    Args:
        records ([type]): [description]

    Returns:
        [type]: [description]
    """

    # Make copy of record
    records = copy.deepcopy(records)

    flags = records.pop('flags')
    hparams = records.pop('hparams')
    env_name = datasets.get_environments(flags['dataset'])

    val_keys = [str(e)+'_out_acc' for i,e in enumerate(env_name) if i != flags['test_env']]
    test_key = str(env_name[flags['test_env']]) + '_in_acc'

    val_dict = {}
    test_dict = {}
    for step, step_dict in records.items():

        val_array = [step_dict[k] for k in val_keys]
        val_dict[step] = np.mean(val_array)

        test_dict[step] = step_dict[test_key]

    ## Picking the max value from a dict
    # Fastest:
    best_step = [k for k,v in val_dict.items() if v==max(val_dict.values())][0]
    # Cleanest:
    # best_step = max(val_dict, key=val_dict.get)
    
    return val_dict[best_step], test_dict[best_step]

def test_domain_validation(records):

    # Make a copy 
    records = copy.deepcopy(records)

    flags = records.pop('flags')
    hparams = records.pop('hparams')
    env_name = datasets.get_environments(flags['dataset'])

    val_keys = str(env_name[flags['test_env']])+'_out_acc'
    test_keys = str(env_name[flags['test_env']])+'_in_acc'

    last_step = max([int(step) for step in records.keys()])

    return records[str(last_step)][val_keys], records[str(last_step)][test_keys]
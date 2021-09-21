import copy
import numpy as np

from lib import datasets
from lib import utils

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

    val_acc_dict = {}
    test_acc_dict = {}
    val_var_dict = {}
    test_var_dict = {}
    for h_seed, h_dict in records.items():
        val_acc, val_var, test_acc, test_var = selection_method(h_dict)

        val_acc_dict[h_seed] = val_acc
        val_var_dict[h_seed] = val_var
        test_acc_dict[h_seed] = test_acc
        test_var_dict[h_seed] = test_var

    best_seed = [k for k,v in val_acc_dict.items() if v==max(val_acc_dict.values())][0]

    return val_acc_dict[best_seed], val_var_dict[best_seed], test_acc_dict[best_seed], test_var_dict[best_seed]

def train_domain_validation(records):

    val_acc = []
    test_acc = []
    records_copy = copy.deepcopy(records)
    for t_seed, t_dict in records_copy.items():
        hparams = t_dict.pop('hparams')
        flags = t_dict.pop('flags')
        env_name = datasets.get_environments(flags['dataset'])

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

        val_acc.append(val_dict[best_step])
        test_acc.append(test_dict[best_step])

    return np.mean(val_acc), np.var(val_acc), np.mean(test_acc), np.var(test_acc)

def test_domain_validation(records):

    val_acc = []
    test_acc = []
    records_copy = copy.deepcopy(records)
    for t_seed, t_dict in records_copy.items():
        hparams = t_dict.pop('hparams')
        flags = t_dict.pop('flags')
        env_name = datasets.get_environments(flags['dataset'])

        val_keys = [str(env_name[flags['test_env']])+'_in_acc']
        test_keys = [str(env_name[flags['test_env']])+'_out_acc']

        last_step = max([int(step) for step in t_dict.keys()])

        val_array = [t_dict[str(last_step)][k] for k in val_keys]
        t_val_acc = np.mean(val_array)

        test_array = [t_dict[str(last_step)][k] for k in test_keys]
        t_test_acc = np.mean(test_array)

        val_acc.append(t_val_acc)
        test_acc.append(t_test_acc)

    return np.mean(val_acc), np.var(val_acc), np.mean(test_acc), np.var(test_acc)
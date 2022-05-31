"""Defining the model selection strategies"""

import copy
import numpy as np

from woods import datasets
from woods import utils

def get_model_selection(dataset_name):
    """ Returns the model selection for a dataset

    Args:
        dataset_name (str): Name of the dataset

    Returns:
        list: list of model selection name 
    """

    if dataset_name in [ 'Spurious_Fourier', "TCMNIST_Source", "TCMNIST_Time"]:
        return ['train_domain_validation', 'test_domain_validation']
    if dataset_name in [ 'CAP', 'SEDFx', 'PCL', 'LSA64', 'HHAR']:
        return ['train_domain_validation', 'oracle_train_domain_validation']
    if dataset_name in ['AusElectricity', 'AusElectricityUnbalanced', 'AusElectricityMonthly', 'AusElectricityMonthlyBalanced']:
        return ['average_validation', 'weighted_average_validation', 'worse_domain_validation']

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

def get_best_hparams(records, selection_method):
    """ Get the best set of hyperparameters for a given a record from a sweep and a selection method

    The way model selection is performed is by computing the validation accuracy of all training checkpoints. 
    The definition of the validation accuracy is given by the selection method. 
    Then using these validation accuracies, we choose the best checkpoint and report the corresponding hyperparameters.

    Args:
        records (dict): Dictionary of records from a sweep
        selection_method (str): Selection method to use

    Returns:
        dict: flags of the chosen model training run for the all trial seeds
        dict: hyperparameters of the chosen model for all trial seeds
        dict: validation accuracy of the chosen model run for all trial seeds
        dict: test accuracy of the chosen model run for all trial seeds
    """

    if selection_method not in globals():
        raise NotImplementedError("Dataset not found: {}".format(selection_method))
    selection_method = globals()[selection_method]

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

def choose_model_domain_generalization(records, selection_method):
    """ Get the test accuracy that will be chosen through the selection method for a given a record from a sweep 

    The way model selection is performed is by computing the validation accuracy of all training checkpoints. 
    The definition of the validation accuracy is given by the selection method. 
    Then using these validation accuracies, we choose the best checkpoint and report the test accuracy linked to that checkpoint.

    Args:
        records (dict): Dictionary of records from a sweep
        selection_method (str): Selection method to use

    Returns:
        float: validation accuracy of the chosen models averaged over all trial seeds
        float: variance of the validation accuracy of the chosen models accross all trial seeds
        float: test accuracy of the chosen models averaged over all trial seeds
        float: variance of the test accuracy of the chosen models accross all trial seeds
    """

    if selection_method not in globals():
        raise NotImplementedError("Dataset not found: {}".format(selection_method))
    selection_method = globals()[selection_method]

    val_acc_arr = []
    test_acc_arr = []
    best_seeds = []
    for t_seed, t_dict in records.items():
        val_acc_dict = {}
        test_acc_dict = {}
        for h_seed, h_dict in t_dict.items():
            val_acc, test_acc = selection_method(h_dict)

            val_acc_dict[h_seed] = val_acc
            test_acc_dict[h_seed] = test_acc

        best_seed = [k for k,v in val_acc_dict.items() if v==max(val_acc_dict.values())][0]
        # best_seed = [k for k,v in val_acc_dict.items() if v==min(val_acc_dict.values())][0]

        val_acc_arr.append(val_acc_dict[best_seed])
        test_acc_arr.append(test_acc_dict[best_seed])
        best_seeds.append((t_seed, best_seed))

    return (
        np.mean(val_acc_arr, axis=0),
        np.std(val_acc_arr, axis=0) / np.sqrt(len(val_acc_arr)),
        np.mean(test_acc_arr, axis=0),
        np.std(test_acc_arr, axis=0) / np.sqrt(len(test_acc_arr)),
        best_seeds
    )


def IID_validation(records):
    """ Perform the IID validation model section on a single training run with NO TEST ENVIRONMENT and returns the results
    
    The model selection is performed by computing the average all domains accuracy of all training checkpoints and choosing the highest one.
        best_step = argmax_{step in checkpoints}( mean(train_envs_acc) )

    Args:
        records (dict): Dictionary of records from a single training run

    Returns:
        float: validation accuracy of the best checkpoint of the training run
        float: validation accuracy of the best checkpoint of the training run

    Note:
        This is for ONLY for sweeps with no test environments.
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
    """ Perform the train domain validation model section on a single training run and returns the results
    
    The model selection is performed by computing the average training domains accuracy of all training checkpoints and choosing the highest one.
        best_step = argmax_{step in checkpoints}( mean(train_envs_acc) )

    Args:
        records (dict): Dictionary of records from a single training run

    Returns:
        float: validation accuracy of the best checkpoint of the training run
        float: test accuracy of the best checkpoint (highest validation accuracy) of the training run
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
    # best_step = [k for k,v in val_dict.items() if v==max(val_dict.values())][0]
    # Cleanest:
    best_step = max(val_dict, key=val_dict.get)
    
    return val_dict[best_step], test_dict[best_step]

def test_domain_validation(records):
    """  Perform the test domain validation model section on a single training run and returns the results

    The model selection is performed with the test accuracy of ONLY THE LAST CHECKPOINT OF A TRAINING RUN, so this function simply returns the test accuracy of the last checkpoint.
        best_step = test_acc[-1]

    Args:
        records (dict): Dictionary of records from a single training run

    Returns:
        float: validation accuracy of the training run, which is also the test accuracyof the last checkpoint
        float: test accuracy of the last checkpoint
    """

    # Make a copy 
    records = copy.deepcopy(records)

    flags = records.pop('flags')
    hparams = records.pop('hparams')
    env_name = datasets.get_environments(flags['dataset'])

    val_keys = str(env_name[flags['test_env']])+'_out_acc'
    test_keys = str(env_name[flags['test_env']])+'_in_acc'

    last_step = max([int(step) for step in records.keys()])

    return records[str(last_step)][val_keys], records[str(last_step)][test_keys]

def oracle_train_domain_validation(records):
    """ Perform the train domain validation 'oracle' model section on a single training run and returns the results

    In this domain validation method, we perform early stopping using the train domains validation set, and we choose the model according to the test domain accuracy at that early stopping point
        best_step = test_acc[ argmax_{step in checkpoints}(mean(train_envs_acc)) ]
 
    Args:
        records (dict): Dictionary of records from a single training run

    Returns:
        float: test accuracy of the best checkpoint (highest validation accuracy) of the training run
        float: test accuracy of the best checkpoint (highest validation accuracy) of the training run
    """

    # Make copy of record
    records = copy.deepcopy(records)

    flags = records.pop('flags')
    hparams = records.pop('hparams')
    env_name = datasets.get_environments(flags['dataset'])

    val_keys = [str(e)+'_out_acc' for i,e in enumerate(env_name) if i != flags['test_env']]
    test_key = str(env_name[flags['test_env']]) + '_in_acc'
    validation_test_key = str(env_name[flags['test_env']]) + '_out_acc'

    val_dict = {}
    test_dict = {}
    validation_test_dict = {}
    for step, step_dict in records.items():

        val_array = [step_dict[k] for k in val_keys]
        val_dict[step] = np.mean(val_array)

        test_dict[step] = step_dict[test_key]
        validation_test_dict[step] = step_dict[validation_test_key]

    ## Picking the max value from a dict
    # Fastest:
    # best_step = [k for k,v in val_dict.items() if v==max(val_dict.values())][0]
    # Cleanest:
    best_step = max(val_dict, key=val_dict.get)
    
    return validation_test_dict[best_step], test_dict[best_step]


def choose_model_subpopulation(records, selection_method, weights=None):
    """ Get the test accuracy that will be chosen through the selection method for a given a record from a sweep 

    The way model selection is performed is by computing the validation accuracy of all training checkpoints. 
    The definition of the validation accuracy is given by the selection method. 
    Then using these validation accuracies, we choose the best checkpoint and report the test accuracy linked to that checkpoint.

    Args:
        records (dict): Dictionary of records from a sweep
        selection_method (str): Selection method to use
        weights (list): domain weights for weighted average

    Returns:
        float: validation accuracy of the chosen models averaged over all trial seeds
        float: variance of the validation accuracy of the chosen models accross all trial seeds
        float: test accuracy of the chosen models averaged over all trial seeds
        float: variance of the test accuracy of the chosen models accross all trial seeds
    """

    if selection_method not in globals():
        raise NotImplementedError("Dataset not found: {}".format(selection_method))
    selection_method = globals()[selection_method]

    chosen_avg_performance = []
    chosen_worse_performance = []
    chosen_seeds = []
    for t_seed, t_dict in records.items():
        val_acc_dict = {}
        avg_test_acc_dict = {}
        worse_test_acc_dict = {}

        for h_seed, h_dict in t_dict.items():
            val_acc, avg_test_acc, worse_test_acc = selection_method(h_dict, weights)

            val_acc_dict[h_seed] = val_acc
            avg_test_acc_dict[h_seed] = avg_test_acc
            worse_test_acc_dict[h_seed] = worse_test_acc

        best_seed = min(val_acc_dict, key=val_acc_dict.get)

        chosen_avg_performance.append(avg_test_acc_dict[best_seed])
        chosen_worse_performance.append(worse_test_acc_dict[best_seed])
        chosen_seeds.append((t_seed, best_seed))

    print(chosen_seeds, chosen_avg_performance, chosen_worse_performance)

    return (
        np.mean(chosen_avg_performance),
        np.std(chosen_avg_performance) / np.sqrt(len(chosen_avg_performance)),
        np.mean(chosen_worse_performance),
        np.std(chosen_worse_performance) / np.sqrt(len(chosen_worse_performance)),
        chosen_seeds
    )


def average_validation(records, weights):
    """ This is for population shift datasets

    Args:
        records (dict): Dictionary of records from a single training run
        weight (list): List of domains weights

    Returns:
        float: validation accuracy of the best checkpoint of the training run
        float: test accuracy of the best checkpoint (highest validation accuracy) of the training run
    """

    # Make copy of record
    records = copy.deepcopy(records)

    flags = records.pop('flags')
    hparams = records.pop('hparams')
    env_name = datasets.get_environments(flags['dataset'])

    val_keys = [str(e)+'_val_rmse' for i,e in enumerate(env_name)]
    test_keys = [str(e)+'_test_rmse' for i,e in enumerate(env_name)]

    avg_val_dict = {}
    avg_test_dict = {}
    worse_test_dict = {}
    for step, step_dict in records.items():

        val_values = [step_dict[k] for k in val_keys]
        avg_val_dict[step] = np.average(val_values)
        
        test_values = [step_dict[k] for k in test_keys]
        avg_test_dict[step] = np.average(test_values, weights=weights)
        worse_test_dict[step] = max(test_values)

    ## Picking the max value from a dict
    best_step = min(avg_val_dict, key=avg_val_dict.get)
    
    return avg_val_dict[best_step], avg_test_dict[best_step], worse_test_dict[best_step]

def weighted_average_validation(records, weights):
    """ This is for population shift datasets

    Args:
        records (dict): Dictionary of records from a single training run
        weight (list): List of domains weights

    Returns:
        float: validation accuracy of the best checkpoint of the training run
        float: test accuracy of the best checkpoint (highest validation accuracy) of the training run
    """

    # Make copy of record
    records = copy.deepcopy(records)

    flags = records.pop('flags')
    hparams = records.pop('hparams')
    env_name = datasets.get_environments(flags['dataset'])

    val_keys = [str(e)+'_val_rmse' for i,e in enumerate(env_name)]
    test_keys = [str(e)+'_test_rmse' for i,e in enumerate(env_name)]

    avg_val_dict = {}
    avg_test_dict = {}
    worse_test_dict = {}
    for step, step_dict in records.items():

        val_values = [step_dict[k] for k in val_keys]
        avg_val_dict[step] = np.average(val_values, weights=weights)
        
        test_values = [step_dict[k] for k in test_keys]
        avg_test_dict[step] = np.average(test_values, weights=weights)
        worse_test_dict[step] = max(test_values)

    ## Picking the max value from a dict
    best_step = min(avg_val_dict, key=avg_val_dict.get)
    
    return avg_val_dict[best_step], avg_test_dict[best_step], worse_test_dict[best_step]


def worse_domain_validation(records, weights):
    """ This is for population shift datasets

    Args:
        records (dict): Dictionary of records from a single training run
        weight (list): List of domains weights

    Returns:
        float: validation accuracy of the best checkpoint of the training run
        float: test accuracy of the best checkpoint (highest validation accuracy) of the training run
    """

    # Make copy of record
    records = copy.deepcopy(records)

    flags = records.pop('flags')
    hparams = records.pop('hparams')
    env_name = datasets.get_environments(flags['dataset'])

    val_keys = [str(e)+'_val_rmse' for i,e in enumerate(env_name)]
    test_keys = [str(e)+'_test_rmse' for i,e in enumerate(env_name)]

    avg_val_dict = {}
    avg_test_dict = {}
    worse_test_dict = {}
    for step, step_dict in records.items():

        val_values = [step_dict[k] for k in val_keys]
        avg_val_dict[step] = max(val_values)
        
        test_values = [step_dict[k] for k in test_keys]
        avg_test_dict[step] = np.average(test_values, weights=weights)
        worse_test_dict[step] = max(test_values)

    ## Picking the max value from a dict
    best_step = min(avg_val_dict, key=avg_val_dict.get)
    
    return avg_val_dict[best_step], avg_test_dict[best_step], worse_test_dict[best_step]
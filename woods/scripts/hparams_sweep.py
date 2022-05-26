"""Perform an hyper parameter sweep

See https://woods.readthedocs.io/en/latest/running_a_sweep.html for usage.
"""

import os
import json
import copy
import glob
import shlex
import argparse

## Local imports
from woods import utils
from woods import command_launchers
from woods import objectives
from woods import datasets

def make_args_list(flags):
    """ Creates a list of commands to launch all of the training runs in the hyper parameter sweep

    Heavily inspired from https://github.com/facebookresearch/DomainBed/blob/9e864cc4057d1678765ab3ecb10ae37a4c75a840/domainbed/scripts/sweep.py#L98

    Args:
        flags (dict): arguments of the hyper parameter sweep

    Returns:
        list: list of strings terminal commands that calls the training runs of the sweep
        list: list of dict where dicts are the arguments for the training runs of the sweep
    """

    train_args_list = []
    for obj in flags['objective']:
        assert obj in objectives.OBJECTIVES, "Objective {} not found".format(obj)
        for dataset in flags['dataset']:
            assert dataset in datasets.DATASETS, "Dataset {} not found".format(dataset)
            for i_hparam in range(flags['n_hparams']):
                for j_trial in range(flags['n_trials']):
                    if flags['unique_test_env'] is not None:
                        test_envs = flags['unique_test_env']
                    else:
                        test_envs = datasets.get_sweep_envs(dataset)
                    for test_env in test_envs:
                        train_args = {}
                        train_args['objective'] = obj
                        train_args['dataset'] = dataset
                        train_args['test_env'] = test_env
                        train_args['data_path'] = flags['data_path']
                        train_args['save_path'] = flags['save_path']
                        train_args['hparams_seed'] = i_hparam
                        train_args['trial_seed'] = j_trial
                        train_args['seed'] = utils.seed_hash(obj, dataset, test_env, i_hparam, j_trial)
                        train_args_list.append(train_args)

    command_list = []
    for train_args in train_args_list:  
        command = ['python3', '-m woods.scripts.main train', '--sample_hparams', '--save']
        for k, v in sorted(train_args.items()):
            if k == 'test_env' and v == 'None':
                continue
            if isinstance(v, list):
                v = ' '.join([str(v_) for v_ in v])
            elif isinstance(v, str):
                v = shlex.quote(v)
            command.append(f'--{k} {v}')
        command_list.append(' '.join(command))
    
    return command_list, train_args_list

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Sweep across seeds')
    # Setup arguments
    parser.add_argument('--objective', nargs='+', type=str, choices=objectives.OBJECTIVES)
    parser.add_argument('--dataset', nargs='+', type=str, choices=datasets.DATASETS)
    parser.add_argument('--unique_test_env', nargs='+')
    # Hyperparameters argument
    parser.add_argument('--n_hparams', type=int, default=20)
    parser.add_argument('--n_trials', type=int, default=3)
    # Job running arguments
    parser.add_argument('--launcher', type=str, default='dummy')
    # Directory arguments
    parser.add_argument('--data_path', type=str, default='~/Documents/Data/')
    parser.add_argument('--save_path', type=str, default='./results/')
    flags = parser.parse_args()

    # Get args in a dict
    flags_dict = vars(flags)

    # Create the flags dictionary and remove flags that are irrelevant to the hyper parameter sweep configuration (doesn't impact results)
    flags_to_save = copy.deepcopy(flags_dict)
    keys_to_del = ['data_path', 'launcher', 'save_path']
    for key in keys_to_del:
        del flags_to_save[key]

    # Check if there is already a sweep config file
    if os.path.exists(os.path.join(flags.save_path, 'sweep_config.json')):
        with open(os.path.join(flags.save_path, 'sweep_config.json')) as f:
            existing_config = json.load(f)
        assert existing_config == flags_to_save, 'There is an already existing sweep_config.json file at the save_path and it is a different sweep. Please take another folder'
    else:
        # Create folders
        os.makedirs(flags.save_path, exist_ok=True)
        os.makedirs(os.path.join(flags.save_path, 'logs'), exist_ok=True)
        os.makedirs(os.path.join(flags.save_path, 'models'), exist_ok=True)
        os.makedirs(os.path.join(flags.save_path, 'outputs'), exist_ok=True)

        with open(os.path.join(flags.save_path,'sweep_config.json'), 'w') as fp:
            json.dump(flags_to_save, fp)
            
    # Create command list and train_arguments
    command_list, train_args = make_args_list(flags_dict)

    # Launch commands
    launcher_fn = command_launchers.REGISTRY[flags.launcher]
    launcher_fn(command_list)


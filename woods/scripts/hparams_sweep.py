"""Perform an hyper parameter sweep"""

import os
import argparse
import shlex
import json

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
        for dataset in flags['dataset']:
            for i_hparam in range(flags['n_hparams']):
                for j_trial in range(flags['n_trials']):
                    if flags['unique_test_env'] is not None:
                        test_envs = flags['unique_test_env']
                    else:
                        test_envs = range(datasets.num_environments(dataset))
                    for test_env in test_envs:
                        train_args = {}
                        train_args['objective'] = obj
                        train_args['dataset'] = dataset
                        train_args['test_env'] = test_env
                        train_args['data_path'] = flags['data_path']
                        train_args['save_path'] = flags['save_path']
                        train_args['hparams_seed'] = i_hparam
                        train_args['trial_seed'] = j_trial
                        train_args['test_step'] = flags['test_step']
                        train_args_list.append(train_args)

    command_list = []
    for train_args in train_args_list:  
        command = ['python3', '-m main train', '--sample_hparams']
        for k, v in sorted(train_args.items()):
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
    parser.add_argument('--unique_test_env', nargs='+', type=int)
    # Hyperparameters argument
    parser.add_argument('--n_hparams', type=int, default=20)
    parser.add_argument('--n_trials', type=int, default=3)
    # Job running arguments
    parser.add_argument('--launcher', type=str, default='dummy')
    # Directory arguments
    parser.add_argument('--data_path', type=str, default='~/Documents/Data/')
    parser.add_argument('--save_path', type=str, default='./results/')
    # Step Setup
    parser.add_argument('--test_step', type=int, default=None)
    flags = parser.parse_args()

    # Create command list and train_arguments
    flags_dict = vars(flags)
    command_list, train_args = make_args_list(flags_dict)

    # Create the sweep config file including all of the sweep parameters
    with open(os.path.join(flags.save_path,'sweep_config.json'), 'w') as fp:
        json.dump(flags_dict, fp)

    # Launch commands
    launcher_fn = command_launchers.REGISTRY[flags.launcher]
    launcher_fn(command_list)


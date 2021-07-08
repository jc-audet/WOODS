import os
import argparse
import shlex

import command_launchers
from objectives import OBJECTIVES

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train MLPs')
    # Dataset argument
    parser.add_argument('--ds_setup', nargs='+', type=str, choices=['grey','seq','step'])
    # Setup arguments
    parser.add_argument('--objective', nargs='+', type=str, choices=OBJECTIVES)
    # Hyperparameters argument
    parser.add_argument('--n_hparams', type=int, default=20)
    parser.add_argument('--n_trials', type=int, default=3)
    # Job running arguments
    parser.add_argument('--launcher', type=str, default='dummy')
    # Directory arguments
    parser.add_argument('--data_path', type=str, default='~/Documents/Data/')
    parser.add_argument('--save_path', type=str, default='./')
    flags = parser.parse_args()

    print(flags.ds_setup)

    train_args_list = []
    for obj in flags.objective:
        for setup in flags.ds_setup:
            for i_hparam in range(flags.n_hparams):
                for j_trial in range(flags.n_trials):
                    train_args = {}
                    train_args['ds_setup'] = setup
                    train_args['objective'] = obj
                    train_args['data_path'] = flags.data_path
                    train_args['save_path'] = flags.save_path
                    train_args['hparams_seed'] = i_hparam
                    train_args['trial_seed'] = j_trial
                    train_args_list.append(train_args)

    command_list = []
    for train_args in train_args_list:  
        command = ['python3', 'temporal_OOD.py', '--sample_hparams']
        for k, v in sorted(train_args.items()):
            if isinstance(v, list):
                v = ' '.join([str(v_) for v_ in v])
            elif isinstance(v, str):
                v = shlex.quote(v)
            command.append(f'--{k} {v}')
        command_list.append(' '.join(command))

    launcher_fn = command_launchers.REGISTRY[flags.launcher]
    launcher_fn(command_list)


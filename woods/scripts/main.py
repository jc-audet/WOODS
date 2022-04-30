""" Script used for the main functionnalities of the woods package 

There is 2 mode of operation:
    - training mode: trains a model on a given dataset with a given test environment using a given algorithm
    - test mode: tests an existing model on a given dataset with a given test environment using a given algorithm

Raises:
    NotImplementedError: Some part of the code is not implemented yet
"""

import os
import json
import time
import random
import argparse
import numpy as np

import torch
from torch import nn, optim

from woods import datasets
from woods import models
from woods import objectives
from woods import hyperparams
from woods import utils
from woods.train import train, get_accuracies

if __name__ == '__main__':

    ## Args
    parser = argparse.ArgumentParser(description='Train a model on a dataset with an objective and test on a test_env')
    # Main mode
    parser.add_argument('mode', choices=['train', 'eval'])
    # Dataset arguments
    parser.add_argument('--test_env', type=int, default = None)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    # Setup arguments
    parser.add_argument('--objective', type=str, choices=objectives.OBJECTIVES)
    # Hyperparameters arguments
    parser.add_argument('--sample_hparams', action='store_true')
    parser.add_argument('--hparams_seed', type=int, default=0)
    parser.add_argument('--trial_seed', type=int, default=0)
    # Directory arguments
    parser.add_argument('--data_path', type=str, default='~/Documents/Data/')
    parser.add_argument('--save_path', type=str, default='./results/')
    parser.add_argument('--download', action='store_true')
    # Model evaluation arguments
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--model_path', type=str, default=None)


    flags = parser.parse_args()
    
    # Device definition
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")


    print('Flags:')
    for k,v in sorted(vars(flags).items()):
        print("\t{}: {}".format(k, v))
    
    ## Making job ID and checking if done
    job_name = utils.get_job_name(vars(flags))

    assert isinstance(flags.test_env, int) or flags.test_env is None, "Invalid test environment"
    if flags.mode == 'train':
        assert not os.path.isfile(os.path.join(flags.save_path, 'logs', job_name+'.json')), "\n*********************************\n*** Job Already ran and saved ***\n*********************************\n"
    
    ## Getting hparams
    training_hparams = hyperparams.get_training_hparams(flags.dataset, flags.hparams_seed, flags.sample_hparams)
    training_hparams['device'] = device
    objective_hparams = hyperparams.get_objective_hparams(flags.objective, flags.hparams_seed, flags.sample_hparams)
    objective_hparams['device'] = device
    model_hparams = hyperparams.get_model_hparams(flags.dataset)
    model_hparams['device'] = device

    print('HParams:')
    for k, v in sorted(training_hparams.items()):
        print('\t{}: {}'.format(k, v))
    for k, v in sorted(model_hparams.items()):
        print('\t{}: {}'.format(k, v))
    for k, v in sorted(objective_hparams.items()):
        print('\t{}: {}'.format(k, v))

    ## Setting dataset seed
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    ## Make dataset
    dataset_class = datasets.get_dataset_class(flags.dataset)
    dataset = dataset_class(flags, training_hparams)
    _, in_loaders = dataset.get_train_loaders()

    # Make some checks about the dataset
    if datasets.num_environments(flags.dataset) == 1:
        assert flags.objective == 'ERM', "Dataset has only one environment, cannot compute multi-environment penalties"

    ## Setting trial seed
    random.seed(flags.trial_seed)
    np.random.seed(flags.trial_seed)
    torch.manual_seed(flags.trial_seed)

    ## Initialize a model to train
    model = models.get_model(dataset, model_hparams)
    print("Number of parameters = ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    # Define training aid
    # loss_fn = nn.NLLLoss(weight=dataset.get_class_weight().to(device))
    optimizer = optim.Adam(model.parameters(), lr=training_hparams['lr'], weight_decay=training_hparams['weight_decay'])

    ## Initialize some Objective
    objective_class = objectives.get_objective_class(flags.objective)
    objective = objective_class(model, dataset, optimizer, objective_hparams)

    ## Do the thing
    model.to(device)
    if flags.mode == 'train':

        model, record, table = train(flags, training_hparams, model, objective, dataset, device)

        ## Save stuff
        if flags.save:
            hparams = {}
            del training_hparams['device']
            hparams.update(training_hparams)
            hparams.update(model_hparams)
            hparams.update(objective_hparams)
            record['hparams'] = hparams
            record['flags'] = vars(flags)
            os.makedirs(os.path.join(flags.save_path, 'logs'), exist_ok=True)
            with open(os.path.join(flags.save_path, 'logs', job_name+'.json'), 'w') as f:
                json.dump(record, f)
            os.makedirs(os.path.join(flags.save_path, 'models'), exist_ok=True)
            torch.save(model.state_dict(), os.path.join(flags.save_path, 'models', job_name+'.pt'))
            os.makedirs(os.path.join(flags.save_path, 'outputs'), exist_ok=True)
            with open(os.path.join(flags.save_path, 'outputs', job_name+'.txt'), 'w') as f:
                f.write('HParams:\n')
                for k, v in sorted(training_hparams.items()):
                    f.write('\t{}: {}\n'.format(k, v))
                for k, v in sorted(model_hparams.items()):
                    f.write('\t{}: {}\n'.format(k, v))
                for k, v in sorted(objective_hparams.items()):
                    f.write('\t{}: {}\n'.format(k, v))
                job_id = 'Training ' + flags.objective  + ' on ' + flags.dataset + ' (H=' + str(flags.hparams_seed) + ', T=' + str(flags.trial_seed) + ')'
                f.write(table.get_string(title=job_id, border=True, hrule=0))


    elif flags.mode == 'eval':
        # Load the weights
        assert flags.model_path != None, "You must give the model_path in order to evaluate a model"
        model.load_state_dict(torch.load(os.path.join(flags.model_path)))

        # Get accuracies
        loss_fn = nn.NLLLoss(weight=dataset.get_class_weight().to(device))
        val_start = time.time()
        record = get_accuracies(objective, dataset, device)
        val_time = time.time() - val_start

        train_names, _ = dataset.get_train_loaders()
        t = utils.setup_pretty_table(flags)
        if dataset.TASK == 'regression':
            t.add_row(['eval'] 
                    + ["{:.1e} :: {:.1e}".format(record[str(e)+'_in_loss'], record[str(e)+'_out_loss']) for e in dataset.ENVS] 
                    + ["{:.1e}".format(np.average([record[str(e)+'_loss'] for e in train_names]))]  
                    + ['.']
                    + ['.'] 
                    + ["{:.2f}".format(val_time)])
        else:
            t.add_row(['eval'] 
                    + ["{:.2f} :: {:.2f}".format(record[str(e)+'_in_acc'], record[str(e)+'_out_acc']) for e in dataset.ENVS] 
                    + ["{:.2f}".format(np.average([record[str(e)+'_loss'] for e in train_names]))]  
                    + ['.']
                    + ['.'] 
                    + ["{:.2f}".format(val_time)])
        print("\n".join(t.get_string().splitlines()[-2:]))

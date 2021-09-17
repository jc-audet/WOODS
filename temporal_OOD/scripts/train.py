import os
import argparse
import numpy as np
import random
import json
import time

import torch
from torch import nn, optim

from temporal_OOD import datasets
from temporal_OOD import models
from temporal_OOD import objectives
from temporal_OOD import hyperparams
from temporal_OOD.utils import utils
from temporal_OOD.source.train_seq import train_seq_setup
from temporal_OOD.source.train_step import train_step_setup

if __name__ == '__main__':

    ## Args
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    parser = argparse.ArgumentParser(description='Train models')
    # Dataset arguments
    parser.add_argument('--test_env', type=int, default = None)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    # Setup arguments
    parser.add_argument('--objective', type=str, choices=objectives.OBJECTIVES)
    # Hyperparameters argument
    parser.add_argument('--sample_hparams', action='store_true')
    parser.add_argument('--hparams_seed', type=int, default=0)
    parser.add_argument('--trial_seed', type=int, default=0)
    # Directory arguments
    parser.add_argument('--data_path', type=str, default='~/Documents/Data/')
    parser.add_argument('--save_path', type=str, default='./results/')
    # Step Setup specific argument
    parser.add_argument('--test_step', type=int, default = None)
    flags = parser.parse_args()

    print('Flags:')
    for k,v in sorted(vars(flags).items()):
        print("\t{}: {}".format(k, v))
    
    ## Making job ID and checking if done
    job_json = utils.get_job_json(flags)

    assert isinstance(flags.test_env, int) or flags.test_env is None, "Invalid test environment"
    assert not os.path.isfile(os.path.join(flags.save_path, job_json)), "\n*********************************\n*** Job Already ran and saved ***\n*********************************\n"
    
    ## Getting hparams
    training_hparams = hyperparams.get_training_hparams(flags.hparams_seed, flags.sample_hparams)
    objective_hparams = hyperparams.get_objective_hparams(flags.objective, flags.hparams_seed, flags.sample_hparams)
    model_hparams = hyperparams.get_model_hparams(flags.dataset, flags.hparams_seed, flags.sample_hparams)

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
    if len(in_loaders) == 1:
        assert flags.objective == 'ERM', "Dataset has only one environment, cannot compute multi-environment penalties"

    ## Setting trial seed
    random.seed(flags.trial_seed)
    np.random.seed(flags.trial_seed)
    torch.manual_seed(flags.trial_seed)

    ## Initialize a model to train
    model = models.get_model(dataset, model_hparams)

    ## Initialize some Objective
    objective_class = objectives.get_objective_class(flags.objective)
    objective = objective_class(model, objective_hparams)

    ## Train it
    model.to(device)
    if dataset.get_setup() == 'seq':
        record = train_seq_setup(flags, training_hparams, model, objective, dataset, device)
    elif dataset.get_setup() == 'step':
        record = train_step_setup(flags, training_hparams, model, objective, dataset, device)
    elif dataset.get_setup() == 'language':
        pass
    else:
        print("Setup undefined")

    ## Save record
    hparams = {}
    hparams.update(training_hparams)
    hparams.update(model_hparams)
    hparams.update(objective_hparams)
    record['hparams'] = hparams
    record['flags'] = vars(flags)
    with open(os.path.join(flags.save_path, job_json), 'w') as f:
        json.dump(record, f)

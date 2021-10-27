import os
import json
import time
import random
import argparse
import numpy as np

import torch
from torch import nn, optim

from lib import datasets
from lib import models
from lib import objectives
from lib import hyperparams
from lib import utils
from lib.train_seq import train_seq_setup, get_accuracies_seq
from lib.train_step import train_step_setup

#TODO:
# - add the --save option so that simple local train runs doesn't get annoyingly saved
if __name__ == '__main__':

    # Device definition
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

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
    # Model evaluation arguments
    parser.add_argument('--save_model', action='store_true')
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--test_step', type=int, default=2)


    flags = parser.parse_args()

    print('Flags:')
    for k,v in sorted(vars(flags).items()):
        print("\t{}: {}".format(k, v))
    
    ## Making job ID and checking if done
    job_name = utils.get_job_name(vars(flags))

    assert isinstance(flags.test_env, int) or flags.test_env is None, "Invalid test environment"
    if flags.mode == 'train':
        assert not os.path.isfile(os.path.join(flags.save_path, job_name+'.json')), "\n*********************************\n*** Job Already ran and saved ***\n*********************************\n"
    
    ## Getting hparams
    training_hparams = hyperparams.get_training_hparams(flags.dataset, flags.hparams_seed, flags.sample_hparams)
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
    print("Number of parameters = ", sum(p.numel() for p in model.parameters()))

    ## Initialize some Objective
    objective_class = objectives.get_objective_class(flags.objective)
    objective = objective_class(model, objective_hparams)

    ## Do the thing
    model.to(device)
    if flags.mode == 'train':

        if dataset.get_setup() == 'seq':
            model, record = train_seq_setup(flags, training_hparams, model, objective, dataset, device)
        elif dataset.get_setup() == 'step':
            model, record = train_step_setup(flags, training_hparams, model, objective, dataset, device)
        elif dataset.get_setup() == 'language':
            raise NotImplementedError("Language benchmarks and models aren't implemented yet")

        if flags.save_model:
            torch.save(model.state_dict(), os.path.join(flags.save_path, job_name+'.pt'))

        ## Save record
        hparams = {}
        hparams.update(training_hparams)
        hparams.update(model_hparams)
        hparams.update(objective_hparams)
        record['hparams'] = hparams
        record['flags'] = vars(flags)
        with open(os.path.join(flags.save_path, job_name+'.json'), 'w') as f:
            json.dump(record, f)

    elif flags.mode == 'eval':
        
        """eval mode : -- download the weights of something -- evaluate it with get_accuracy of the right setup

        Raises:
            NotImplementedError: [description]
        """
        # Load the weights
        assert flags.model_path != None, "You must give the model_path in order to evaluate a model"
        model.load_state_dict(torch.load(os.path.join(flags.model_path)))

        # Get accuracies
        loss_fn = nn.NLLLoss(weight=dataset.get_class_weight().to(device))
        if dataset.get_setup() == 'seq':
            val_start = time.time()
            record = get_accuracies_seq(model, loss_fn, dataset, device)
            val_time = time.time() - val_start
        elif dataset.get_setup() == 'step':
            record = get_accuracies_seq(model, loss_fn, dataset, device)
        elif dataset.get_setup() == 'language':
            raise NotImplementedError("Language benchmarks and models aren't implemented yet")

        train_names, _ = dataset.get_train_loaders()
        t = utils.setup_pretty_table(flags)
        t.add_row(['Eval'] 
                + ["{:.2f} :: {:.2f}".format(record[str(e)+'_in_acc'], record[str(e)+'_out_acc']) for e in dataset.get_envs()] 
                + ["{:.2f}".format(np.average([record[str(e)+'_loss'] for e in train_names]))] 
                + ['.']
                + ['.'] 
                + ["{:.2f}".format(val_time)])
        print("\n".join(t.get_string().splitlines()[-2:]))

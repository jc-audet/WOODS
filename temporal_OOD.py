
import os
import argparse
import numpy as np
import random
import json

import torch
import torch.nn.functional as F
from torch import nn, optim
from torchvision import datasets, transforms

from datasets import get_dataset_class
from models import small_RNN, RNN
from objectives import get_objective_class, OBJECTIVES
from hyperparams import get_objective_hparams, get_training_hparams

from prettytable import PrettyTable
from utils import setup_pretty_table

import matplotlib.pyplot as plt

## Train function
def train_step(model, objective, dataset, in_loaders_iter, optimizer, device):
    """
    :param model: nn model defined in a models.py
    :param train_loader: training dataloader(s)
    :param optimizer: optimizer of the model defined in train(...)
    :param device: device on which we are training
    """
    model.train()

    ts = torch.tensor(dataset.get_pred_time()).to(device)
        
    if dataset.get_setup() == 'seq':
        
        # Get next batch of training data
        batch_loaders = next(in_loaders_iter)

        # Send everything in an array
        minibatches_device = [(x, y) for x,y in batch_loaders]

        ## Group all inputs and send to device
        all_x = torch.cat([x for x,y in minibatches_device]).to(device)
        all_y = torch.cat([y for x,y in minibatches_device]).to(device)
        all_out = []

        # Get logit and make prediction
        hidden = model.initHidden(all_x.shape[0]).to(device)
        pred = torch.zeros(all_x.shape[0], 0).to(device)
        for i in range(all_x.shape[1]):
            out, hidden = model(all_x[:,i,...], hidden)
            if i in ts:
                all_out.append(out)
                pred = torch.cat((pred, out.argmax(1, keepdim=True)), dim=1)

        # Compute environment-wise losses
        all_logits_idx = 0
        env_losses = torch.zeros(len(minibatches_device))
        for i, (x,y) in enumerate(minibatches_device):
            env_loss = 0
            y = y.to(device)
            for t_idx, out in enumerate(all_out):     # Number of time steps
                env_out_t = out[all_logits_idx:all_logits_idx + x.shape[0],:]
                env_loss += F.nll_loss(env_out_t, y[:,t_idx]) 
                objective.gather_logits_and_labels(env_out_t, y[:,t_idx])

            # get train accuracy and save it
            nb_correct = pred[all_logits_idx:all_logits_idx + x.shape[0],:].eq(y).cpu().sum()
            nb_items = pred[all_logits_idx:all_logits_idx + x.shape[0],:].numel()

            # Save loss
            env_losses[i] = env_loss

            # Update stuff
            all_logits_idx += x.shape[0]

        # Back propagate
        optimizer.zero_grad()
        objective.backward(env_losses)
        optimizer.step()

    ## This is only valid for TCMNIST_step dataset
    elif dataset.get_setup() == 'step':    # Test environment (step) is assumed to be the last one.

        # Get next batch of training data
        batch_loaders = next(in_loaders_iter)
        minibatches_device = [(x, y) for x,y in batch_loaders]

        ## Group all inputs and send to device
        all_x = torch.cat([x for x,y in minibatches_device]).to(device)
        all_y = torch.cat([y for x,y in minibatches_device]).to(device)
        all_out = []

        ## Group all inputs and get prediction
        hidden = model.initHidden(all_x.shape[0]).to(device)
        pred = torch.zeros(all_x.shape[0], 0).to(device)
        for i in range(all_x.shape[1]):
            out, hidden = model(all_x[:,i,...], hidden)
            if i in ts:
                all_out.append(out)
                pred = torch.cat((pred, out.argmax(1, keepdim=True)), dim=1)

        # Get train loss for train environment
        train_env = [i for i, t in enumerate(ts) if i != flags.test_env]
        env_losses = torch.zeros(len(train_env))
        for i, e in enumerate(train_env):
            env_out = all_out[e]
            env_loss = F.nll_loss(env_out, all_y[:,e])  
            env_losses[i] = env_loss
            objective.gather_logits_and_labels(env_out, all_y[:,e])

        # back propagate
        optimizer.zero_grad()
        objective.backward(env_losses)
        optimizer.step()

    return model


def train(training_hparams, model, objective, dataset, device):

    optimizer = optim.Adam(model.parameters(), lr=training_hparams['lr'], weight_decay=training_hparams['weight_decay'])
    record = {}

    t = setup_pretty_table(training_hparams, dataset)

    train_names, train_loaders = dataset.get_train_loaders() 
    val_names, val_loaders = dataset.get_val_loaders() 
    all_names = train_names + val_names
    all_loaders = train_loaders + val_loaders
    for step in range(1, dataset.N_STEPS + 1):

        if dataset.get_setup() == 'seq':

            train_loaders_iter = zip(*train_loaders)
            ## Make training step and report accuracies and losses
            model = train_step(model, objective, dataset, train_loaders_iter, optimizer, device)

            if step % dataset.CHECKPOINT_FREQ == 0 or (step-1)==0:
                ## Get test accuracy and loss
                record[str(step)] = {}
                for name, loader in zip(all_names, all_loaders):
                    accuracy, loss = get_accuracy(model, dataset, loader, device)

                    record[str(step)].update({name+'_acc': accuracy,
                                            name+'_loss': loss})

                t.add_row([step] +["{:.2f} :: {:.2f}".format(record[str(step)][str(e)+'_in_acc'], record[str(step)][str(e)+'_out_acc']) for e in dataset.get_envs()])
                print("\n".join(t.get_string().splitlines()[-1:]))

        elif dataset.get_setup() == 'step':

            train_loaders_iter = zip(*train_loaders)
            ## Make training step and report accuracies and losses
            model = train_step(model, objective, dataset, train_loaders_iter, optimizer, device)

            if step % dataset.CHECKPOINT_FREQ == 0 or (step-1) == 0:
                ## Get test accuracy and loss
                record[str(step)] = {}
                for name, loader in zip(all_names, all_loaders):
                    accuracies, losses = get_accuracy(model, dataset, loader, device)

                    for i, e in enumerate(name):
                        record[str(step)].update({e+'_acc': accuracies[i],
                                                  e+'_loss': losses[i]})

                t.add_row([step] +["{:.2f} :: {:.2f}".format(record[str(step)][str(e)+'_in_acc'], record[str(step)][str(e)+'_out_acc']) for e in dataset.get_envs()])
                print("\n".join(t.get_string().splitlines()[-1:]))

    return record

def get_accuracy(model, dataset, loader, device):

    model.eval()
    losses = []
    nb_correct = 0
    nb_item = 0

    ts = torch.tensor(dataset.get_pred_time()).to(device)
    with torch.no_grad():

        if dataset.get_setup() == 'seq':

            for data, target in loader:

                data, target = data.to(device), target.to(device)

                loss = 0
                pred = torch.zeros(data.shape[0], 0).to(device)
                hidden = model.initHidden(data.shape[0]).to(device)
                for i in range(data.shape[1]):
                    out, hidden = model(data[:,i,...], hidden)
                    if i in ts:     # Only consider labels after the prediction at prediction times
                        idx = (ts == i).nonzero(as_tuple=True)[0]
                        loss += F.nll_loss(out, torch.squeeze(target[:,idx])) 
                        pred = torch.cat((pred, out.argmax(1, keepdim=True)), dim=1)
                
                nb_correct += pred.eq(target).cpu().sum()
                nb_item += pred.numel()
                losses.append(loss.item())

            return nb_correct.item() / nb_item, np.mean(losses)
        
        if dataset.get_setup() == 'step':

            losses = torch.zeros(ts.shape[0], 0).to(device)
            for i, (data, target) in enumerate(loader):

                data, target = data.to(device), target.to(device)

                loss = torch.zeros(ts.shape[0], 1).to(device)
                pred = torch.zeros(data.shape[0], 0).to(device)
                hidden = model.initHidden(data.shape[0]).to(device)
                for i in range(data.shape[1]):
                    out, hidden = model(data[:,i,...], hidden)
                    if i in ts:     # Only consider labels after the prediction at prediction times
                        idx = (ts == i).nonzero(as_tuple=True)[0]
                        pred = torch.cat((pred, out.argmax(1, keepdim=True)), dim=1)
                        loss[idx] += F.nll_loss(out, torch.squeeze(target[:,idx]))

                losses = torch.cat((losses, loss), dim=1)
                nb_correct += torch.sum(pred.eq(target).cpu(), dim=0)
                nb_item += pred.shape[0]
            
            return (nb_correct / nb_item).tolist(), torch.mean(losses, dim=1).tolist()

if __name__ == '__main__':

    ## Args
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    parser = argparse.ArgumentParser(description='Train MLPs')
    # Dataset arguments
    # parser.add_argument('--time_steps', type=int, default=4)  # Should be in the TMNIST dataset definition
    parser.add_argument('--test_env', type=int, required=True)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    # Setup arguments
    parser.add_argument('--objective', type=str, choices=OBJECTIVES)
    # Hyperparameters argument
    parser.add_argument('--sample_hparams', action='store_true')
    parser.add_argument('--hparams_seed', type=int, default=0)
    parser.add_argument('--trial_seed', type=int, default=0)
    # Directory arguments
    parser.add_argument('--data_path', type=str, default='~/Documents/Data/')
    parser.add_argument('--save_path', type=str, default='./results/')
    flags = parser.parse_args()

    print('Flags:')
    for k,v in sorted(vars(flags).items()):
        print("\t{}: {}".format(k, v))
    
    ## Making job ID and checking if done
    if flags.sample_hparams:
        job_id = flags.objective + '_' + flags.dataset + '_H' + flags.hparams_seed + '_T' + flags.trial_seed
    else:
        job_id = flags.objective + '_' + flags.dataset
    job_json = job_id + '.json'

    assert not os.path.isfile(os.path.join(flags.save_path, job_json)), "\n*********************************\n*** Job Already ran and saved ***\n*********************************\n"
    
    ## Getting hparams
    training_hparams = get_training_hparams(flags.hparams_seed, flags.sample_hparams)
    objective_hparams = get_objective_hparams(flags.objective, flags.hparams_seed, flags.sample_hparams)

    print('HParams:')
    for k, v in sorted(training_hparams.items()):
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
    dataset_class = get_dataset_class(flags.dataset)
    dataset = dataset_class(flags, training_hparams['batch_size'])
    _, in_loaders = dataset.get_train_loaders()

    if len(in_loaders) == 1:
        assert flags.objective == 'ERM' , "Dataset has only one environment, cannot compute multi-environment penalties"

    ## Setting trial seed
    random.seed(flags.trial_seed)
    np.random.seed(flags.trial_seed)
    torch.manual_seed(flags.trial_seed)

    ## Initialize some RNN
    if flags.dataset in ['TMNIST', 'TCMNIST_seq', 'TCMNIST_step']:
        model = RNN(dataset.get_input_size(), 20, 10, 2)
    elif flags.dataset in ['Fourier_basic', 'Spurious_Fourier']:
        model = small_RNN(dataset.get_input_size(), 10, 2)
    else:
        raise ValueError("Dataset doesn't have a designed model")

    ## Initialize some Objective
    objective_class = get_objective_class(flags.objective)
    objective = objective_class(model, objective_hparams)

    ## Train it
    model.to(device)
    record = train(training_hparams, model, objective, dataset, device)

    ## Save record
    with open(os.path.join(flags.save_path, job_json), 'w') as f:
        json.dump(record, f)






##############################################
##############################################
# ## Import original MNIST data
# MNIST_tfrm = transforms.Compose([ transforms.ToTensor() ])

# train_ds = datasets.MNIST(flags.data_path, train=True, download=True, transform=MNIST_tfrm) 
# test_ds = datasets.MNIST(flags.data_path, train=False, download=True, transform=MNIST_tfrm) 

## Create dataset
# input_size, train_loader, test_loader = make_dataset(flags.ds_setup, flags.time_steps, train_ds, test_ds, training_hparams['batch_size'])

# input_size = dataset.get_input_size()
# train_loader, test_loader = dataset.get_loaders()
import time
import numpy as np

import torch
from torch import nn, optim

from woods.lib import datasets
from woods.lib import models
from woods.lib import objectives
from woods.lib import hyperparams
from woods.lib import utils

## Train function
def train_step(model, objective, dataset, in_loaders_iter, device):
    """
    :param model: nn model defined in a models.py
    :param train_loader: training dataloader(s)
    :param optimizer: optimizer of the model defined in train(...)
    :param device: device on which we are training
    """
    model.train()

    ts = torch.tensor(dataset.get_pred_time()).to(device)
        
    # Get next batch of training data 
    # TODO: Fix that awful patch
    try:
        batch_loaders = next(in_loaders_iter)
    except StopIteration:
        _, loaders = dataset.get_train_loaders()
        in_loaders_iter = zip(*loaders)
        batch_loaders = next(in_loaders_iter)

    # Send everything in an array
    minibatches_device = [(x, y) for x,y in batch_loaders]

    objective.update(minibatches_device, dataset, device)

    return model

def train(flags, training_hparams, model, objective, dataset, device):
    """
    :param flags: flags from argparse
    :param training_hparams: hyperparameters for training
    :param model: nn model defined in a models.py
    :param objective: objective
    :param dataset: dataset
    """
    record = {}
    step_times = []
    
    t = utils.setup_pretty_table(flags)

    train_names, train_loaders = dataset.get_train_loaders()
    n_batches = np.sum([len(train_l) for train_l in train_loaders])
    train_loaders_iter = zip(*train_loaders)

    for step in range(1, dataset.N_STEPS + 1):

        ## Make training step and report accuracies and losses
        step_start = time.time()
        model = train_step(model, objective, dataset, train_loaders_iter, device)
        step_times.append(time.time() - step_start)

        if step % dataset.CHECKPOINT_FREQ == 0 or (step-1)==0:

            val_start = time.time()
            checkpoint_record = get_accuracies(objective, dataset, device)
            val_time = time.time() - val_start

            record[str(step)] = checkpoint_record

            t.add_row([step] 
                    + ["{:.2f} :: {:.2f}".format(record[str(step)][str(e)+'_in_acc'], record[str(step)][str(e)+'_out_acc']) for e in dataset.get_envs()] 
                    + ["{:.2f}".format(np.average([record[str(step)][str(e)+'_loss'] for e in train_names]))] 
                    + ["{:.2f}".format((step*len(train_loaders)) / n_batches)]
                    + ["{:.2f}".format(np.mean(step_times))] 
                    + ["{:.2f}".format(val_time)])

            step_times = [] 
            print("\n".join(t.get_string().splitlines()[-2:-1]))

    return model, record

def get_accuracies(objective, dataset, device):

    # Get loaders and their names
    val_names, val_loaders = dataset.get_val_loaders()

    ## Get test accuracy and loss
    record = {}
    for name, loader in zip(val_names, val_loaders):

        if dataset.SETUP == 'seq':
            accuracy, loss = get_split_accuracy_seq(objective, dataset, loader, device)
        
            record.update({ name+'_acc': accuracy,
                            name+'_loss': loss})
        elif dataset.SETUP == 'step':
            accuracies, losses = get_split_accuracy_step(objective, dataset, loader, device)
            for i, e in enumerate(name):
                record.update({ e+'_acc': accuracies[i],
                                e+'_loss': losses[i]})
    
    return record

def get_split_accuracy_seq(objective, dataset, loader, device):

    ts = torch.tensor(dataset.get_pred_time()).to(device)

    losses = 0
    nb_correct = 0
    nb_item = 0
    with torch.no_grad():

        for b, (data, target) in enumerate(loader):

            data, target = data.to(device), target.to(device)

            all_out = objective.predict(data, ts, device)

            for i, t in enumerate(ts):
                losses += objective.loss_fn(all_out[:,i,...], target[:,i])

            # get train accuracy and save it
            pred = all_out.argmax(dim=2)
            nb_correct += pred.eq(target).cpu().sum()
            nb_item += target.numel()
            
    return nb_correct.item() / nb_item, losses.item() / len(loader)

def get_split_accuracy_step(objective, dataset, loader, device):

    ts = torch.tensor(dataset.get_pred_time()).to(device)

    losses = torch.zeros(*ts.shape).to(device)
    nb_correct = torch.zeros(*ts.shape).to(device)
    nb_item = torch.zeros(*ts.shape).to(device)
    with torch.no_grad():

        for b, (data, target) in enumerate(loader):

            data, target = data.to(device), target.to(device)

            all_out = objective.predict(data, ts, device)

            for i, t in enumerate(ts):
                losses[i] += objective.loss_fn(all_out[:,i,...], target[:,i])

            pred = all_out.argmax(dim=2)
            nb_correct += torch.sum(pred.eq(target), dim=0)
            nb_item += pred.shape[0]
            
    return (nb_correct / nb_item).tolist(), (losses/len(loader)).tolist()
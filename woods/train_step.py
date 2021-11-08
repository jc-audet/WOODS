import time
import numpy as np

import torch
from torch import nn, optim

from woods import datasets
from woods import models
from woods import objectives
from woods import hyperparams
from woods import utils

## Train function
def train_step(model, loss_fn, objective, dataset, in_loaders_iter, optimizer, device):
    """ Make a single training step for a model on a dataset of the step setup with an objective

    Args:
        model (nn.Module): Model to train
        loss_fn (fn): pytorch nn loss function
        objective (Objective): Instance of an Objective object from lib.objectives
        dataset (Multi_Domain_Dataset): Instance of a Multi_Domain_Dataset object from lib.datasets
        in_loaders_iter (Iterable): Iterable that generates tuples of dataloaders coming from each environment (usually gotten by a zip() call)
        optimizer (torch.optim): Instance of a torch.optim optimizer
        device (str): device on which the training happens

    Returns:
        nn.Module: returns the model updated from a training step
    """
    model.train()

    ts = torch.tensor(dataset.get_pred_time()).to(device)

    # Get next batch of training data
    # TODO: Fix that awful patch with infinite
    try:
        batch_loaders = next(in_loaders_iter)
    except StopIteration:
        _, loaders = dataset.get_train_loaders()
        in_loaders_iter = zip(*loaders)
        batch_loaders = next(in_loaders_iter)

    minibatches_device = [(x, y) for x,y in batch_loaders]

    ## Group all inputs and send to device
    all_x = torch.cat([x for x,y in minibatches_device]).to(device)
    all_y = torch.cat([y for x,y in minibatches_device]).to(device)
    all_out = []

    ## Group all inputs and get prediction
    all_out, pred = model(all_x, ts)

    # Get train loss for train environment
    train_env = [i for i, t in enumerate(ts) if i != len(ts)-1]
    env_losses = torch.zeros(len(train_env))
    for i, e in enumerate(train_env):
        env_out = all_out[e]
        env_loss = loss_fn(env_out, all_y[:,e])  
        env_losses[i] = env_loss
        objective.gather_logits_and_labels(env_out, all_y[:,e])

    # back propagate
    optimizer.zero_grad()
    objective.backward(env_losses)
    optimizer.step()

    return model


def train_step_setup(flags, training_hparams, model, objective, dataset, device):
    """ Train a model on a dataset of the step setup with an objective

    Args:
        flags (Namespace): training arguments
        training_hparams (dict): training related hyperparameters (lr, weight decay, batchsize, etc.)
        model (nn.Module): Model to be trained
        objective (Objective): Instance of an Objective object from lib.objectives
        dataset (Multi_Domain_Dataset): Instance of a Multi_Domain_Dataset object from lib.datasets
        device (str): device on which the training happens

    Returns:
        dict: records from training of the model
    """

    loss_fn = nn.NLLLoss(weight=dataset.get_class_weight().to(device))
    optimizer = optim.Adam(model.parameters(), lr=training_hparams['lr'], weight_decay=training_hparams['weight_decay'])
    
    record = {}
    step_times = []
    
    t = utils.setup_pretty_table(flags)

    train_names, train_loaders = dataset.get_train_loaders()
    n_batches = np.sum([len(train_l) for train_l in train_loaders])

    ## Is this in the loop or not?
    train_loaders_iter = zip(*train_loaders)

    for step in range(1, dataset.N_STEPS + 1):

        ## Make training step and report accuracies and losses
        start = time.time()
        model = train_step(model, loss_fn, objective, dataset, train_loaders_iter, optimizer, device)
        step_times.append(time.time() - start)

        if step % dataset.CHECKPOINT_FREQ == 0 or (step-1) == 0:

            val_start = time.time()
            checkpoint_record = get_accuracies_step(model, loss_fn, dataset, device)
            val_time = time.time() - val_start

            record[str(step)] = checkpoint_record

            t.add_row([step] 
                    + ["{:.2f} :: {:.2f}".format(record[str(step)][str(e)+'_in_acc'], record[str(step)][str(e)+'_out_acc']) for e in dataset.get_envs()]
                    + ["{:.2f}".format(np.average([record[str(step)][str(e)+'_loss'] for e in train_names[0]]))] 
                    + ["{:.2f}".format((step*len(train_loaders)) / n_batches)]
                    + ["{:.2f}".format(np.mean(step_times))]  
                    + ["{:.2f}".format(val_time)])
            print("\n".join(t.get_string().splitlines()[-2:-1]))

    return model, record

def get_accuracies_step(model, loss_fn, dataset, device):

    # Get loaders and their names
    val_names, val_loaders = dataset.get_val_loaders()
    train_names, train_loaders = dataset.get_train_loaders()
    all_names = val_names + train_names
    all_loaders = val_loaders + train_loaders
    ## Get test accuracy and loss
    record = {}
    for name, loader in zip(all_names, all_loaders):
        accuracies, losses = get_split_accuracy(model, loss_fn, dataset, loader, device)
        for i, e in enumerate(name):
            record.update({ e+'_acc': accuracies[i],
                            e+'_loss': losses[i]})
    
    return record

def get_split_accuracy(model, loss_fn, dataset, loader, device):

    model.eval()
    losses = []
    nb_correct = 0
    nb_item = 0

    ts = torch.tensor(dataset.get_pred_time()).to(device)
    with torch.no_grad():

        losses = torch.zeros(ts.shape[0], 0).to(device)
        for i, (data, target) in enumerate(loader):

            data, target = data.to(device), target.to(device)

            out, pred = model(data, ts)

            loss = torch.zeros(ts.shape[0], 1).to(device)
            for i, t in enumerate(ts):     # Only consider labels after the prediction at prediction times
                loss[i] += loss_fn(out[i], target[:,i])

            losses = torch.cat((losses, loss), dim=1)

            nb_correct += torch.sum(pred.eq(target).cpu(), dim=0)
            nb_item += pred.shape[0]
        
    return (nb_correct / nb_item).tolist(), torch.mean(losses, dim=1).tolist()

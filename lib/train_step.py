import time
import numpy as np

import torch
from torch import nn, optim

from lib import datasets
from lib import models
from lib import objectives
from lib import hyperparams
from lib import utils

## Train function
def train_step(model, loss_fn, objective, dataset, in_loaders_iter, optimizer, device):
    """
    :param model: nn model defined in a models.py
    :param train_loader: training dataloader(s)
    :param optimizer: optimizer of the model defined in train(...)
    :param device: device on which we are training
    """
    model.train()

    ts = torch.tensor(dataset.get_pred_time()).to(device)

    # Get next batch of training data
    batch_loaders = next(in_loaders_iter)
    minibatches_device = [(x, y) for x,y in batch_loaders]

    ## Group all inputs and send to device
    all_x = torch.cat([x for x,y in minibatches_device]).to(device)
    all_y = torch.cat([y for x,y in minibatches_device]).to(device)
    all_out = []

    ## Group all inputs and get prediction
    all_out, pred = model(all_x, ts)

    # Get train loss for train environment
    train_env = [i for i, t in enumerate(ts) if i != dataset.test_step]
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

    loss_fn = nn.NLLLoss(weight=dataset.get_class_weight().to(device))
    optimizer = optim.Adam(model.parameters(), lr=training_hparams['lr'], weight_decay=training_hparams['weight_decay'])
    
    record = {}
    step_times = []
    
    t = utils.setup_pretty_table(flags)

    train_names, train_loaders = dataset.get_train_loaders()
    n_batches = np.sum([len(train_l) for train_l in train_loaders])
    for step in range(1, dataset.N_STEPS + 1):

        train_loaders_iter = zip(*train_loaders)
        ## Make training step and report accuracies and losses
        start = time.time()
        model = train_step(model, loss_fn, objective, dataset, train_loaders_iter, optimizer, device)
        step_times.append(time.time() - start)

        if step % dataset.CHECKPOINT_FREQ == 0 or (step-1) == 0:

            checkpoint_record = get_accuracies_step(model, loss_fn, dataset, device)

            record[str(step)] = checkpoint_record

            t.add_row([step] 
                    + ["{:.2f} :: {:.2f}".format(record[str(step)][str(e)+'_in_acc'], record[str(step)][str(e)+'_out_acc']) for e in dataset.get_envs()]
                    + ["{:.2f}".format(np.average([record[str(step)][str(e)+'_loss'] for e in train_names[0]]))] 
                    + ["{:.2f}".format((step*len(train_loaders)) / n_batches)]
                    + ["{:.2f}".format(np.mean(step_times))] )
            print("\n".join(t.get_string().splitlines()[-2:-1]))

    return record

def get_accuracies_step(model, loss_fn, dataset, device):

    # Get loaders and their names
    train_names, train_loaders = dataset.get_train_loaders()
    val_names, val_loaders = dataset.get_val_loaders()
    all_names = train_names + val_names
    all_loaders = train_loaders + val_loaders

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

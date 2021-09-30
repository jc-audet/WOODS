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

    # Send everything in an array
    minibatches_device = [(x, y) for x,y in batch_loaders]

    ## Group all inputs and send to device
    all_x = torch.cat([x for x,y in minibatches_device]).to(device)
    all_y = torch.cat([y for x,y in minibatches_device]).to(device)
    all_out = []

    # Get logit and make prediction
    all_out, pred = model(all_x, ts)

    # Compute environment-wise losses
    all_logits_idx = 0
    env_losses = torch.zeros(len(minibatches_device))
    for i, (x, y) in enumerate(minibatches_device):
        env_loss = 0
        y = y.to(device)
        for t_idx, out in enumerate(all_out):     # Number of time steps
            env_out_t = out[all_logits_idx:all_logits_idx + x.shape[0],:]
            env_loss += loss_fn(env_out_t, y[:,t_idx]) 
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

    return model

def train_seq_setup(flags, training_hparams, model, objective, dataset, device):

    loss_fn = nn.NLLLoss(weight=dataset.get_class_weight().to(device))
    optimizer = optim.Adam(model.parameters(), lr=training_hparams['lr'], weight_decay=training_hparams['weight_decay'])
    
    record = {}
    step_times = []
    
    t = utils.setup_pretty_table(flags)

    train_names, train_loaders = dataset.get_train_loaders()
    n_batches = np.sum([len(train_l) for train_l in train_loaders])
    for step in range(1, dataset.N_STEPS + 1):
        print(step)

        train_loaders_iter = zip(*train_loaders)
        ## Make training step and report accuracies and losses
        step_start = time.time()
        model = train_step(model, loss_fn, objective, dataset, train_loaders_iter, optimizer, device)
        step_times.append(time.time() - step_start)

        if step % dataset.CHECKPOINT_FREQ == 0:# or (step-1)==0:

            val_start = time.time()
            checkpoint_record = get_accuracies_seq(model, loss_fn, dataset, device)
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

    return record

def get_accuracies_seq(model, loss_fn, dataset, device):

    # Get loaders and their names
    val_names, val_loaders = dataset.get_val_loaders()

    ## Get test accuracy and loss
    record = {}
    for name, loader in zip(val_names, val_loaders):
        print(name)
        accuracy, loss = get_split_accuracy(model, loss_fn, dataset, loader, device)

        record.update({name+'_acc': accuracy,
                                name+'_loss': loss})
    
    return record

def get_split_accuracy(model, loss_fn, dataset, loader, device):

    n_batch = 0
    losses = 0
    nb_correct = 0
    nb_item = 0

    ts = torch.tensor(dataset.get_pred_time()).to(device)

    model.eval()
    with torch.no_grad():

        for b, (data, target) in enumerate(loader):

            data, target = data.to(device), target.to(device)

            loss = 0
            all_out, pred = model(data, ts)

            for i, t in enumerate(ts):
                loss += loss_fn(all_out[i], target[:,i])

            nb_correct += pred.eq(target).sum()
            nb_item += pred.numel()
            losses += loss
            n_batch += 1

    return nb_correct.item() / nb_item, losses.item() / n_batch

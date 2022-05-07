"""Defining the training functions that are used to train and evaluate models"""
from typing import NamedTuple, Optional, Iterable, Dict, Any, List, Tuple
import time
import numpy as np

import torch
from torch import nn, optim

from woods import datasets
from woods import models
from woods import objectives
from woods import hyperparams
from woods import utils

# ## Train function
# def train_step(model, objective, dataset, in_loaders_iter, device):
#     """ Train a single training step for a model

#     Args:
#         model: nn model defined in a models.py
#         objective: objective we are using for training
#         dataset: dataset object we are training on
#         in_loaders_iter: iterable of iterable of data loaders
#         device: device on which we are training
#     """

#     # Put model into training mode
#     model.train()

#     # Get next batch
#     minibatches_device = dataset.get_next_batch()

#     objective.update(minibatches_device, dataset, device)

#     return model

def train(flags, training_hparams, model, objective, dataset, device):
    """ Train a model on a given dataset with a given objective

    Args:
        flags: flags from argparse
        training_hparams: training hyperparameters
        model: nn model defined in a models.py
        objective: objective we are using for training
        dataset: dataset object we are training on
        device: device on which we are training
    """
    # Initialize containers
    record = {}
    step_times = []
    
    # Set up table
    t = utils.setup_pretty_table(flags)

    # Perform training loop for the prescribed number of steps
    n_batches = dataset.get_number_of_batches()
    for step in range(1, dataset.N_STEPS + 1):

        ## Make training step and report accuracies and losses
        step_start = time.time()
        objective.update()
        step_times.append(time.time() - step_start)

        if step % dataset.CHECKPOINT_FREQ == 0 or (step-1)==0:

            val_start = time.time()
            checkpoint_record = get_accuracies(objective, dataset, device)
            val_time = time.time() - val_start

            record[str(step)] = checkpoint_record

            if dataset.TASK == 'forecasting':
                t.add_row([step] 
                        + [" :: ".join(["{:.0f}".format(record[str(step)][str(e)+'_train_rmse']) for e in dataset.ENVS])] 
                        + [" :: ".join(["{:.0f}".format(record[str(step)][str(e)+'_val_rmse']) for e in dataset.ENVS])] 
                        + [" :: ".join(["{:.0f}".format(record[str(step)][str(e)+'_test_rmse']) for e in dataset.ENVS])] 
                        + ["{:.1e}".format(np.average([record[str(step)][str(e)+'_train_loss'] for e in dataset.ENVS]))] 
                        + ["0"]
                        + ["{:.2f}".format(np.mean(step_times))] 
                        + ["{:.2f}".format(val_time)])
            if dataset.TASK == 'regression':
                t.add_row([step] 
                        + ["{:.1e} :: {:.1e}".format(record[str(step)][str(e)+'_in_loss'], record[str(step)][str(e)+'_out_loss']) for e in dataset.ENVS] 
                        + ["{:.1e}".format(np.average([record[str(step)][str(e)+'_loss'] for e in dataset.train_names]))] 
                        + ["{:.2f}".format((step*len(dataset.train_loaders)) / n_batches)]
                        + ["{:.2f}".format(np.mean(step_times))] 
                        + ["{:.2f}".format(val_time)])
            if dataset.TASK == 'classification':
                t.add_row([step] 
                        + ["{:.2f} :: {:.2f}".format(record[str(step)][str(e)+'_in_acc'], record[str(step)][str(e)+'_out_acc']) for e in dataset.ENVS] 
                        + ["{:.2f}".format(np.average([record[str(step)][str(e)+'_loss'] for e in dataset.train_names]))] 
                        + ["{:.2f}".format((step*len(dataset.train_loaders)) / n_batches)]
                        + ["{:.2f}".format(np.mean(step_times))] 
                        + ["{:.2f}".format(val_time)])

            step_times = [] 
            print("\n".join(t.get_string().splitlines()[-2:-1]))

    return model, record, t

def get_accuracies(objective, dataset, device):
    """ Get accuracies for all splits using fast loaders

    Args:
        objective: objective we are using for training
        dataset: dataset object we are training on
        device: device on which we are training
    """

    # Get loaders and their names
    val_names, val_loaders = dataset.get_val_loaders()

    ## Get test accuracy and loss
    record = {}
    for name, loader in zip(val_names, val_loaders):

        if dataset.TASK == 'classification':
            
            if dataset.SETUP == 'source':
                accuracy, loss = get_split_accuracy_source(objective, dataset, loader, device)
            
                record.update({ name+'_acc': accuracy,
                                name+'_loss': loss})

            elif dataset.SETUP == 'time':
                accuracies, losses = get_split_accuracy_time(objective, dataset, loader, device)

                for i, e in enumerate(name):
                    record.update({ e+'_acc': accuracies[i],
                                    e+'_loss': losses[i]})

        elif dataset.TASK == 'forecasting':

            if dataset.SETUP == 'subpopulation':
                error, loss = get_split_errors(objective, name, dataset, loader, device)

                record.update({ name+'_rmse': error,
                                name+'_loss': loss})

    return record

def get_split_accuracy_source(objective, dataset, loader, device):
    """ Get accuracy and loss for a dataset that is of the `source` setup

    Args:
        objective: objective we are using for training
        dataset: dataset object we are training on
        loader: data loader of which we want the accuracy
        device: device on which we are training
    """

    ts = torch.tensor(dataset.PRED_TIME).to(device)

    losses = 0
    nb_correct = 0
    nb_item = 0
    with torch.no_grad():

        for b, (data, target) in enumerate(loader):

            data, target = data.to(device), target.to(device)

            all_out, _ = objective.predict(data)
            loss = dataset.loss(all_out, target)
            losses += loss.mean()

            # get train accuracy and save it
            pred = all_out.argmax(dim=2)
            nb_correct += pred.eq(target).cpu().sum()
            nb_item += target.numel()

        show_value = nb_correct.item() / nb_item

    return show_value, losses.item() / len(loader)

def get_split_accuracy_time(objective, dataset, loader, device):
    """ Get accuracy and loss for a dataset that is of the `time` setup

    Args:
        objective: objective we are using for training
        dataset: dataset object we are training on
        loader: data loader of which we want the accuracy
        device: device on which we are training
    """

    pred_time = dataset.PRED_TIME

    losses = torch.zeros(*pred_time.shape).to(device)
    nb_correct = torch.zeros(*pred_time.shape).to(device)
    nb_item = torch.zeros(*pred_time.shape).to(device)
    with torch.no_grad():

        for b, (data, target) in enumerate(loader):

            data, target = data.to(device), target.to(device)

            all_out, _ = objective.predict(data)

            losses = dataset.loss(all_out, target)

            pred = all_out.argmax(dim=2)
            nb_correct += torch.sum(pred.eq(target), dim=0)
            nb_item += pred.shape[0]
            
    return (nb_correct / nb_item).tolist(), (losses/len(loader)).tolist()

def get_split_errors(objective, name, dataset, loader, device):
    """ Get error and loss for a dataset that is of the `source` setup

    Args:
        objective: objective we are using for training
        dataset: dataset object we are training on
        loader: data loader of which we want the accuracy
        device: device on which we are training
    """

    losses = 0
    errors = 0
    nb_items = 0
    MSE = nn.MSELoss()
    with torch.no_grad():

        for b, batch in enumerate(loader):

            X, Y = dataset.split_input(batch)

            # Get loss
            out, _ = objective.predict(X)
            loss = dataset.loss(out, Y)
            losses += torch.mean(loss).item()

            # Get errors
            out = objective.model.inference(X)
            # if name == 'Holidays_test':
            #     for i in range(out.shape[0]):
            #         print(i)
            #         plot_forecast(i, {k: X[k].cpu() for k in X}, out.cpu())
                # plot_forecast(-1, {k: X[k].cpu() for k in X}, out.cpu())
            out_avg = torch.mean(out, dim=1)
            errors += torch.sqrt(MSE(out_avg, Y)).item() * Y.shape[0]

            # Count
            nb_items += Y.shape[0]

        avg_error = errors / nb_items
        avg_loss = losses / nb_items

    return avg_error, avg_loss

import matplotlib.pyplot as plt
import datetime

def get_minute(minute):
    return minute + 0.5
def get_hour(hour):
    return hour + 0.5
def get_day_of_year(time_feat):
    return (time_feat + 0.5) * 365 + 1
def get_year(year):
    return (np.power(10, year)-2.0)/17532

def plot_forecast(k, batch, pred):
    plt.figure()
    minutes = get_minute(torch.cat((batch['past_time_feat'][k,:,0], batch['future_time_feat'][k,:,0]), dim=0))
    hours = get_hour(torch.cat((batch['past_time_feat'][k,:,1], batch['future_time_feat'][k,:,1]), dim=0))
    days = get_day_of_year(torch.cat((batch['past_time_feat'][k,:,-2], batch['future_time_feat'][k,:,-2]), dim=0))
    years = get_year(torch.cat((batch['past_time_feat'][k,:,-1], batch['future_time_feat'][k,:,-1]), dim=0))
    date_time = [(datetime.datetime(2002 + int(year.item()),1,1) + datetime.timedelta(days=day.item(), hours=hour.item(), minutes=minu.item())).strftime('%Y-%m-%d') for year, day, hour, minu in zip(years, days, hours, minutes)]
    labels = [''] * len(date_time)
    labels[::40] = date_time[::40]
    ground_truth = torch.cat((batch['past_target'][k], batch['future_target'][k]), dim=0)
    full_pred = torch.cat((batch['past_target'][k], torch.mean(pred[k,:,:], dim=0)), dim=0)
    plt.plot(ground_truth, 'b', label='Ground Truth')
    plt.plot(full_pred, 'r', label='Prediction')
    plt.axvline(x=batch['past_target'][k].shape[-1])
    plt.xticks(np.arange(len(labels)), labels)
    plt.xticks(rotation=60)
    # plt.plot(date_time, pred)
    plt.gcf().tight_layout()
    plt.show()
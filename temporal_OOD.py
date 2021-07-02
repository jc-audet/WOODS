
import os
import argparse
import numpy as np

import torch
import torch.nn.functional as F
from torch import nn, optim
from torchvision import datasets, transforms

from datasets import make_dataset
from models import RNN
from objectives import get_objective_class

## Train function
def train_epoch(ds_setup, model, objective, train_loader, optimizer, device):
    """
    :param model: nn model defined in a models.py
    :param train_loader: training dataloader(s)
    :param optimizer: optimizer of the model defined in train(...)
    :param device: device on which we are training
    """
    model.train()
    accuracies = []
    losses = []

    if ds_setup == 'grey':

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)

            loss = 0
            hidden = model.initHidden(data.shape[0]).to(device)
            pred = torch.zeros(data.shape[0], 0).to(device)
            for i in range(data.shape[1]):

                out, hidden = model(data[:,i,:,:], hidden)
                loss += F.nll_loss(out, target[:,i]) if i>0 else 0.  # Only consider labels after the first frame
                
                pred = torch.cat((pred, out.argmax(1, keepdim=True)), dim=1)

            nb_correct = pred[:,1:].eq(target[:,1:]).cpu().sum()
            nb_items = pred[:,1:].numel()

            losses.append(loss.item())
            accuracies.append(nb_correct / nb_items)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return model, losses, accuracies

    elif ds_setup == 'seq':

        train_loader_iter = zip(*train_loader)
        for batch_loaders in train_loader_iter:

            # Send everything onto device
            minibatches_device = [(x, y) for x,y in batch_loaders]

            ## Group all inputs and get prediction
            all_x = torch.cat([x for x,y in minibatches_device]).to(device)
            all_y = torch.cat([y for x,y in minibatches_device]).to(device)
            all_out = []

            # Get logit and make prediction
            hidden = model.initHidden(all_x.shape[0]).to(device)
            pred = torch.zeros(all_x.shape[0], 0).to(device)
            for i in range(all_x.shape[1]):
                out, hidden = model(all_x[:,i,:,:], hidden)
                all_out.append(out)
                pred = torch.cat((pred, out.argmax(1, keepdim=True)), dim=1)

            # Compute environment-wise losses
            all_loss = 0
            all_logits_idx = 0
            env_losses = torch.zeros(len(minibatches_device))
            for i, (x,y) in enumerate(minibatches_device):
                env_loss = 0
                y = y.to(device)
                for t in range(1,all_x.shape[1]):     # Number of time steps
                    env_out_t = all_out[t][all_logits_idx:all_logits_idx + x.shape[0],:]
                    env_loss += F.nll_loss(env_out_t, y[:,t])  # Only consider labels after the first frame
                    objective.gather_logits_and_labels(env_out_t, y[:,t])
                env_losses[i] = env_loss
                all_logits_idx += x.shape[0]
                all_loss += env_loss / len(train_loader) # Average across environments

            # get average train accuracy and save it
            nb_correct = pred[:,1:].eq(all_y[:,1:]).cpu().sum()
            nb_items = pred[:,1:].numel()
            accuracies.append(nb_correct / nb_items)

            # get average loss and save it
            loss = all_loss
            losses.append(loss.item())

            # Back propagate
            optimizer.zero_grad()
            objective.backward(env_losses)
            optimizer.step()

        return model, losses, accuracies

    elif ds_setup == 'step':    # Test environment (step) is assumed to be the last one.

        test_accuracies = []
        test_losses = []

        for all_x, target in train_loader:

            # Send everything onto device
            all_x, target = all_x.to(device), target.to(device)

            ## Group all inputs and get prediction
            all_out = []
            hidden = model.initHidden(all_x.shape[0]).to(device)
            pred = torch.zeros(all_x.shape[0], 0).to(device)
            for i in range(all_x.shape[1]):
                out, hidden = model(all_x[:,i,:,:], hidden)
                all_out.append(out)
                pred = torch.cat((pred, out.argmax(1, keepdim=True)), dim=1)
            
            # Get train loss for train environment
            loss = 0
            env_losses = []
            train_env = np.arange(1,all_x.shape[1]-1)
            for e in train_env:
                env_out = all_out[e]
                env_loss = F.nll_loss(env_out, target[:,e])  # Only consider labels after the first frame
                env_losses.append(env_loss)
                loss += env_loss
            loss /= np.size(train_env)

            # Save loss
            losses.append(loss.item())

            # Get train accuracy
            nb_correct = pred[:,1:-1].eq(target[:,1:-1]).cpu().sum()
            nb_items = pred[:,1:-1].numel()
            accuracies.append(nb_correct / nb_items)

            # back propagate
            optimizer.zero_grad()
            objective.backward(env_losses)
            optimizer.step()

            # Get test loss
            with torch.no_grad():
                env_out = all_out[-1]
                test_losses.append( F.nll_loss(env_out, target[:,-1]) )

                # Get test accuracy
                nb_correct = pred[:,-1].eq(target[:,-1]).cpu().sum()
                nb_items = pred[:,-1].numel()
                test_accuracies.append(nb_correct / nb_items)

        return model, losses, accuracies, test_losses, test_accuracies


def train(flags, model, objective, train_loader, test_loader, device):

    optimizer = optim.Adam(model.parameters(), lr=flags.lr, weight_decay=flags.weight_decay)
    training_accuracies = []
    training_losses = []
    test_accuracies = []
    test_losses = []

    print('Epoch\t||\tTrain Acc\t|\tTest Acc\t||\tTraining Loss\t|\tTest Loss ')
    for epoch in range(1, flags.epochs + 1):

        if flags.ds_setup == 'grey' or flags.ds_setup == 'seq':
            ## Make training step and report accuracies and losses
            model, training_loss, training_accuracy = train_epoch(flags.ds_setup, model, objective, train_loader, optimizer, device)
            training_losses.append(training_loss)
            training_accuracies.append(training_accuracy)

            ## Get test accuracy and loss
            test_accuracy, test_loss = get_accuracy(flags.ds_setup, model, test_loader, device)
            test_losses.append(test_loss)
            test_accuracies.append(test_accuracy)

            print("{}\t||\t{:.2f}\t\t|\t{:.2f}\t\t||\t{:.2e}\t|\t{:.2e}".format(epoch, training_accuracy[-1], test_accuracy, training_loss[-1], test_loss))

        elif flags.ds_setup == 'step':
            ## Make training step and report accuracies and losses
            model, training_loss, training_accuracy, test_loss, test_accuracy = train_epoch(flags.ds_setup, model, objective, train_loader, optimizer, device)
            training_losses.append(training_loss)
            training_accuracies.append(training_accuracy)
            test_losses.append(test_loss)
            test_accuracies.append(test_accuracy)

            print("{}\t||\t{:.2f}\t\t|\t{:.2f}\t\t||\t{:.2e}\t|\t{:.2e}".format(epoch, training_accuracy[-1], test_accuracy[-1], training_loss[-1], test_loss[-1]))

    return training_accuracies, training_losses, test_accuracies, test_losses

def get_accuracy(ds_setup, model, loader, device):

    # Check if this is the right setup
    assert not ds_setup == 'step', "Wrong use of get_accuracy: Not valid for 'step' setup"

    model.eval()
    test = 0
    nb_correct = 0
    nb_item = 0
    losses = []

    with torch.no_grad():
        for data, target in loader:
        
            data, target = data.to(device), target.to(device)

            loss = 0
            pred = torch.zeros(data.shape[0], 0).to(device)
            hidden = model.initHidden(data.shape[0]).to(device)
            for i in range(data.shape[1]):
                out, hidden = model(data[:,i,:,:], hidden)
                pred = torch.cat((pred, out.argmax(1, keepdim=True)), dim=1)
                loss += F.nll_loss(out, target[:,i]) if i>0 else 0.  # Only consider labels after the first frame
            
            nb_correct += pred[:,1:].eq(target[:,1:]).cpu().sum()
            nb_item += pred[:,1:].numel()
            losses.append(loss.item())

    return nb_correct / nb_item, np.mean(losses)


if __name__ == '__main__':

    ## Args
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    parser = argparse.ArgumentParser(description='Train MLPs')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--weight_decay', type=float,default=0.)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--time_steps', type=int, default=4)
    parser.add_argument('--ds_setup', type=str, choices=['grey','seq','step'])
    parser.add_argument('--objective', type=str, choices=['ERM','IRM','VREx', 'SD', 'IGA', 'ANDMask', 'SANDMask'])
    parser.add_argument('--data-path', type=str, default='~/Documents/Data/')
    parser.add_argument('--save-path', type=str, default='./')
    flags = parser.parse_args()

    print('Flags:')
    for k,v in sorted(vars(flags).items()):
        print("\t{}: {}".format(k, v))

    ## Import original MNIST data
    MNIST_tfrm = transforms.Compose([ transforms.ToTensor() ])

    train_ds = datasets.MNIST(flags.data_path, train=True, download=True, transform=MNIST_tfrm) 
    test_ds = datasets.MNIST(flags.data_path, train=False, download=True, transform=MNIST_tfrm) 

    ## Create dataset
    input_size, train_loader, test_loader = make_dataset(flags.ds_setup, flags.time_steps, train_ds, test_ds, flags.batch_size)

    ## Initialize some RNN
    model = RNN(input_size, 50, 10, 2)

    ## Initialize some Objective
    objective_class = get_objective_class(flags.objective)
    objective = objective_class(model)

    ## Train it
    model.to(device)
    train(flags, model, objective, train_loader, test_loader, device)
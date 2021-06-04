
import os
import argparse
import numpy as np

import torch
import torch.nn.functional as F
from torch import nn, optim
from torchvision import datasets, transforms

## Remove later
import matplotlib.pyplot as plt


class RNN(nn.Module):
    def __init__(self, input_size, state_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)

        lin1 = nn.Linear(input_size + hidden_size, state_size)
        lin2 = nn.Linear(state_size, state_size)
        lin3 = nn.Linear(state_size, hidden_size)
        for lin in [lin1, lin2, lin3]:
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
        self.FCH = nn.Sequential(lin1, nn.ReLU(True), lin2, nn.ReLU(True), lin3, nn.LogSoftmax(dim=1))


        lin4 = nn.Linear(input_size + hidden_size, state_size)
        lin5 = nn.Linear(state_size, state_size)
        lin6 = nn.Linear(state_size, 2)
        for lin in [lin4, lin5, lin6]:
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
        self.FCO = nn.Sequential(lin4, nn.ReLU(True), lin5, nn.ReLU(True), lin6, nn.LogSoftmax(dim=1))

        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input.view(input.shape[0],-1), hidden), 1)
        hidden = self.FCH(combined)
        output = self.FCO(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size)

## Train function
def train_epoch(model, train_loader, optimizer, device):
    """
    :param model: nn model defined in a X_class.py
    :param train_load: ?
    :param GPU: boolean variable that initialize some variable on the GPU if accessible, otherwise on CPU
    """
    model.train()

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        loss = 0
        hidden = model.initHidden(data.shape[0]).to(device)
        for i in range(data.shape[1]):
            out, hidden = model(data[:,i,:,:], hidden)
            loss += F.nll_loss(out, target[:,i]) if i>0 else 0.  # Only consider labels after the first frame

        loss.backward()
        optimizer.step()

    return model

def train(flags, model, train_loader, test_loader, device):

    optimizer = optim.Adam(model.parameters(), lr=flags.lr, weight_decay=flags.weight_decay)
    training_accuracies = []
    training_losses = []
    test_accuracies = []
    test_losses = []

    print('Epoch\t||\tTrain Acc\t|\tTest Acc\t||\tTraining Loss\t|\tTest Loss ')
    for epoch in range(1, flags.epochs + 1):

        model = train_epoch(model, train_loader, optimizer, device)

        ## Get training accuracy and loss
        training_accuracy, training_loss = get_accuracy(model, train_loader, device)
        training_losses.append(training_loss)
        training_accuracies.append(training_accuracy)

        ## Get test accuracy and loss
        test_accuracy, test_loss = get_accuracy(model, test_loader, device)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        print("{}\t||\t{:.2f}\t\t|\t{:.2f}\t\t||\t{:.2e}\t|\t{:.2e}".format(epoch, training_accuracy, test_accuracy, training_loss, test_loss))

    return training_accuracies, training_losses, test_accuracies, test_losses

def get_accuracy(model, loader, device):

    model.eval()
    test = 0
    nb_correct = 0

    for data, target in loader:
      
        data, target = data.to(device), target.to(device)

        loss = 0
        pred = torch.zeros(data.shape[0], 1).to(device)
        hidden = model.initHidden(data.shape[0]).to(device)
        for i in range(data.shape[1]):
            out, hidden = model(data[:,i,:,:], hidden)
            pred = torch.cat((pred, out.max(1, keepdim=True)[1]), dim=1) if i>0 else pred
            loss += F.nll_loss(out, target[:,i]) if i>0 else 0.  # Only consider labels after the first frame
        
        nb_correct += pred[:,1:].eq(target[:,1:].data.view_as(pred[:,1:])).cpu().sum()

    return nb_correct.item() * 100 / (2*len(loader.dataset)), loss/len(loader.dataset)

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
    parser.add_argument('--data-path', type=str, default='~/Documents/Data/')
    parser.add_argument('--save-path', type=str, default='./')
    flags = parser.parse_args()

    print('Flags:')
    for k,v in sorted(vars(flags).items()):
        print("\t{}: {}".format(k, v))

    ## Import original MNIST data
    MNIST_tfrm = transforms.Compose([ transforms.ToTensor(),
                                      transforms.Lambda(lambda x: torch.flatten(x))])

    train_ds = datasets.MNIST(flags.data_path, train=True, download=True, transform=MNIST_tfrm) 
    test_ds = datasets.MNIST(flags.data_path, train=False, download=True, transform=MNIST_tfrm) 

    ## Create dataset

    # Create sequences of 3 digits
    train_ds.data = train_ds.data.reshape(-1,3,28,28)
    test_ds.data = test_ds.data[:-1,:,:].reshape(-1,3,28,28)

    # With their corresponding label
    train_ds.targets = train_ds.targets.reshape(-1,3)
    test_ds.targets = test_ds.targets[:-1].reshape(-1,3)

    # Assign label to the objective : Is the last number in the sequence larger than the current
    train_ds.targets = ( train_ds.targets[:,:2] > train_ds.targets[:,1:] )
    train_ds.targets = torch.cat((torch.zeros((train_ds.targets.shape[0],1)), train_ds.targets), 1).long()
    test_ds.targets = ( test_ds.targets[:,:2] > test_ds.targets[:,1:] )
    test_ds.targets = torch.cat((torch.zeros((test_ds.targets.shape[0],1)), test_ds.targets), 1).long()

    # Make Tensor dataset
    train_dataset = torch.utils.data.TensorDataset(train_ds.data, train_ds.targets)
    test_dataset = torch.utils.data.TensorDataset(test_ds.data, test_ds.targets)

    # Make dataloader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=flags.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=flags.batch_size, shuffle=True)

    ## Initialize some RNN
    model = RNN(train_ds.data.shape[2]*train_ds.data.shape[3], 50, 10, 2)

    ## Train it
    model.to(device)
    train(flags, model, train_loader, test_loader, device)
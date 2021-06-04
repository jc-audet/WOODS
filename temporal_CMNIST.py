
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

    print('|\tEpoch\t||\tTrain Acc\t|\tTest Acc\t||\tTrain Loss\t|\tTest Loss\t|')
    print("-----------------------------------------------------------------------------------------------------------------")
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

        print("|\t{}\t||\t{:.2f}\t\t|\t{:.2f}\t\t||\t{:.2e}\t|\t{:.2e}\t|".format(epoch, training_accuracy, test_accuracy, training_loss, test_loss))

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

def XOR(a, b):
    return ( a - b ).abs()

def bernoulli(p, size):
    return ( torch.rand(size) < p ).float()

def color_dataset(images, labels, p, d):

    # Add label noise
    labels = XOR(labels, bernoulli(d, labels.shape)).long()

    # Choose colors
    colors = XOR(labels, bernoulli(p, labels.shape))

    # Stack a second color channel
    images = torch.stack([images,images], dim=2)

    # Apply colors
    for sample in range(colors.shape[0]):
        for frame in range(colors.shape[1]):
            images[sample,frame,(1-colors[sample,frame]).long(),:,:] *= 0 

    return images, labels


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

    # Concatenate all data and labels
    MNIST_images = torch.cat((train_ds.data,test_ds.data))
    MNIST_labels = torch.cat((train_ds.targets,test_ds.targets))

    # Create sequences of 3 digits
    MNIST_images = MNIST_images[:-1,:,:].reshape(-1,3,28,28)

    # With their corresponding label
    MNIST_labels = MNIST_labels[:-1].reshape(-1,3)

    # Assign label to the objective : Is the last number in the sequence larger than the current
    MNIST_labels = ( MNIST_labels[:,:2] > MNIST_labels[:,1:] )
    MNIST_labels = torch.cat((torch.zeros((MNIST_labels.shape[0],1)), MNIST_labels), 1)

    # Make the color datasets

    train_loaders = []            # array of training environment dataloaders
    test_loaders = []            # array of test environment dataloaders
    d = 0.25                # Label noise
    envs = [0.1, 0.2, 0.9]  # Environment is a function of correlation
    test_env = 2
    for i, e in enumerate(envs):

        # Choose data subset
        images = MNIST_images[i::len(envs)]
        labels = MNIST_labels[i::len(envs)]

        # Color subset
        colored_images, colored_labels = color_dataset(images, labels, e, d)

        # Make Tensor dataset
        td = torch.utils.data.TensorDataset(colored_images, colored_labels)

        # Make dataloader
        if i==test_env:
            test_loaders.append( torch.utils.data.DataLoader(td, batch_size=flags.batch_size, shuffle=True) )
        else:
            train_loaders.append( torch.utils.data.DataLoader(td, batch_size=flags.batch_size, shuffle=True) )

    ################
    ## Training model

    ## Initialize some RNN
    model = RNN(2*MNIST_images.shape[2]*MNIST_images.shape[3], 50, 10, 2)

    ## Train it
    model.to(device)
    train(flags, model, train_loaders, test_loaders, device)
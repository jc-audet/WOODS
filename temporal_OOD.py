
import os
import argparse
import numpy as np

import torch
import torch.nn.functional as F
from torch import nn, optim
from torchvision import datasets, transforms

## Remove later
import matplotlib.pyplot as plt

from datasets import make_dataset


class RNN(nn.Module):
    def __init__(self, input_size, state_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        lin1 = nn.Linear(input_size + hidden_size, state_size)
        lin2 = nn.Linear(state_size, state_size)
        lin3 = nn.Linear(state_size, hidden_size)
        for lin in [lin1, lin2, lin3]:
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
        self.FCH = nn.Sequential(lin1, nn.ReLU(True), lin2, nn.ReLU(True), lin3)


        lin4 = nn.Linear(input_size + hidden_size, state_size)
        lin5 = nn.Linear(state_size, state_size)
        lin6 = nn.Linear(state_size, 2)
        for lin in [lin4, lin5, lin6]:
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
        self.FCO = nn.Sequential(lin4, nn.ReLU(True), lin5, nn.ReLU(True), lin6, nn.LogSoftmax(dim=1))


    def forward(self, input, hidden):
        combined = torch.cat((input.view(input.shape[0],-1), hidden), 1)
        hidden = self.FCH(combined)
        output = self.FCO(combined)
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
    accuracies = []
    losses = []

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
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

        loss.backward()
        optimizer.step()

    return model, losses, accuracies

def train(flags, model, train_loader, test_loader, device):

    optimizer = optim.Adam(model.parameters(), lr=flags.lr, weight_decay=flags.weight_decay)
    training_accuracies = []
    training_losses = []
    test_accuracies = []
    test_losses = []

    print('Epoch\t||\tTrain Acc\t|\tTest Acc\t||\tTraining Loss\t|\tTest Loss ')
    for epoch in range(1, flags.epochs + 1):

        model, training_loss, training_accuracy = train_epoch(model, train_loader, optimizer, device)
        training_losses.append(training_loss)
        training_accuracies.append(training_accuracy)

        ## Get test accuracy and loss
        test_accuracy, test_loss = get_accuracy(model, test_loader, device)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        print("{}\t||\t{:.2f}\t\t|\t{:.2f}\t\t||\t{:.2e}\t|\t{:.2e}".format(epoch, training_accuracy[-1], test_accuracy, training_loss[-1], test_loss))

    return training_accuracies, training_losses, test_accuracies, test_losses

def get_accuracy(model, loader, device):

    model.eval()
    test = 0
    nb_correct = 0
    nb_item = 0
    losses = []

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

def biggest_multiple(multiple_of, input_number):
    return input_number - input_number % multiple_of

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

def make_dataset(ds_setup, time_steps, train_ds, test_ds):

    if ds_setup == 'grey':
        
        # Create sequences of 3 digits
        n_train_samples = biggest_multiple(time_steps, train_ds.data.shape[0])
        n_test_samples = biggest_multiple(time_steps, test_ds.data.shape[0])
        train_ds.data = train_ds.data[:n_train_samples].reshape(-1,time_steps,28,28)
        test_ds.data = test_ds.data[:n_test_samples].reshape(-1,time_steps,28,28)

        # With their corresponding label
        train_ds.targets = train_ds.targets[:n_train_samples].reshape(-1,time_steps)
        test_ds.targets = test_ds.targets[:n_test_samples].reshape(-1,time_steps)

        # Assign label to the objective : Is the last number in the sequence larger than the current
        train_ds.targets = ( train_ds.targets[:,:-1] > train_ds.targets[:,1:] )
        train_ds.targets = torch.cat((torch.zeros((train_ds.targets.shape[0],1)), train_ds.targets), 1).long()
        test_ds.targets = ( test_ds.targets[:,:-1] > test_ds.targets[:,1:] )
        test_ds.targets = torch.cat((torch.zeros((test_ds.targets.shape[0],1)), test_ds.targets), 1).long()

        # Make Tensor dataset
        train_dataset = torch.utils.data.TensorDataset(train_ds.data, train_ds.targets)
        test_dataset = torch.utils.data.TensorDataset(test_ds.data, test_ds.targets)

        # Make dataloader
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=flags.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=flags.batch_size, shuffle=False)

        input_size = 28 * 28

        return input_size, train_loader, test_loader

    elif ds_setup == 'CMNIST_seq':

        # Concatenate all data and labels
        MNIST_images = torch.cat((train_ds.data, test_ds.data))
        MNIST_labels = torch.cat((train_ds.targets, test_ds.targets))

        # Create sequences of 3 digits
        n_samples = biggest_multiple(time_steps, MNIST_images.shape[0])
        MNIST_images = MNIST_images[:n_samples].reshape(-1,4,28,28)

        # With their corresponding label
        MNIST_labels = MNIST_labels[:n_samples].reshape(-1,4)

        # Assign label to the objective : Is the last number in the sequence larger than the current
        MNIST_labels = ( MNIST_labels[:,:3] > MNIST_labels[:,1:] )
        MNIST_labels = torch.cat((torch.zeros((MNIST_labels.shape[0],1)), MNIST_labels), 1)

        # Make the color datasets

        train_loaders = []          # array of training environment dataloaders
        test_loaders = []           # array of test environment dataloaders
        d = 0.25                    # Label noise
        envs = [0.8, 0.9, 0.1]            # Environment is a function of correlation
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

        input_size = 2 * 28 * 28

        return input_size, train_loaders, test_loaders


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
    parser.add_argument('--ds_setup', type=str, choices=['grey','CMNIST_seq'])
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
    input_size, train_loader, test_loader = make_dataset(flags.ds_setup, flags.time_steps, train_ds, test_ds)

    ## Initialize some RNN
    model = RNN(input_size, 50, 10, 2)

    ## Train it
    model.to(device)
    train(flags, model, train_loader, test_loader, device)


    ### Plot greyscale images
    # show_images = train_ds.data
    # fig, axs = plt.subplots(3,4)
    # axs[0,0].imshow(show_images[0,0,:,:], cmap='gray')
    # axs[0,0].set_ylabel('Sequence 1')
    # axs[0,1].imshow(show_images[0,1,:,:], cmap='gray')
    # axs[0,1].set_title('Label = 1')
    # axs[0,2].imshow(show_images[0,2,:,:], cmap='gray')
    # axs[0,2].set_title('Label = 0')
    # axs[0,3].imshow(show_images[0,3,:,:], cmap='gray')
    # axs[0,3].set_title('Label = 1')
    # axs[1,0].imshow(show_images[1,0,:,:], cmap='gray')
    # axs[1,0].set_ylabel('Sequence 2')
    # axs[1,1].imshow(show_images[1,1,:,:], cmap='gray')
    # axs[1,1].set_title('Label = 0')
    # axs[1,2].imshow(show_images[1,2,:,:], cmap='gray')
    # axs[1,2].set_title('Label = 1')
    # axs[1,3].imshow(show_images[1,3,:,:], cmap='gray')
    # axs[1,3].set_title('Label = 0')
    # axs[2,0].imshow(show_images[2,0,:,:], cmap='gray')
    # axs[2,0].set_ylabel('Sequence 3')
    # axs[2,0].set_xlabel('Time Step 1')
    # axs[2,1].imshow(show_images[2,1,:,:], cmap='gray')
    # axs[2,1].set_title('Label = 0')
    # axs[2,1].set_xlabel('Time Step 2')
    # axs[2,2].imshow(show_images[2,2,:,:], cmap='gray')
    # axs[2,2].set_xlabel('Time Step 3')
    # axs[2,2].set_title('Label = 1')
    # axs[2,3].imshow(show_images[2,3,:,:], cmap='gray')
    # axs[2,3].set_xlabel('Time Step 4')
    # axs[2,3].set_title('Label = 0')
    # for row in axs:
    #     for ax in row:
    #         ax.set_xticks([]) 
    #         ax.set_yticks([]) 
    # plt.tight_layout()
    # plt.savefig('./figure/Temporal_MNIST.png')

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

        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input.view(input.shape[0],-1), hidden), 1)
        hidden = self.FCH(combined)
        output = self.FCO(combined)
        output = self.log_softmax(output)
        return output, hidden

    def initHidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size)

## Train function
def train_epoch(model, loader, env, optimizer, device):
    """
    :param model: nn model defined in a X_class.py
    :param train_load: ?
    :param GPU: boolean variable that initialize some variable on the GPU if accessible, otherwise on CPU
    """
    model.train()

    for x,y in loader:

        # Send everything onto device
        x, y = x.to(device), y.to(device)

        ## Group all inputs and get prediction
        all_out = []
        hidden = model.initHidden(x.shape[0]).to(device)
        for i in range(x.shape[1]):
            out, hidden = model(x[:,i,:,:], hidden)
            all_out.append(out)
        
        total_loss = 0
        for e in env:
            env_out = all_out[e]
            env_loss = F.nll_loss(env_out, y[:,e])  # Only consider labels after the first frame
            total_loss += env_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    return model

def train(flags, model, loader, train_env, test_env, device):

    optimizer = optim.Adam(model.parameters(), lr=flags.lr, weight_decay=flags.weight_decay)
    training_accuracies = []
    training_losses = []
    test_accuracies = []
    test_losses = []

    print('|\tEpoch\t||\tTrain Acc\t|\tTest Acc\t||\tTrain Loss\t|\tTest Loss\t|')
    print("-----------------------------------------------------------------------------------------------------------------")
    for epoch in range(1, flags.epochs + 1):

        ## Train a single epoch
        model = train_epoch(model, loader, train_env, optimizer, device)

        ## Get training accuracy and loss
        training_accuracy, training_loss = get_accuracy(model, loader, train_env, device)
        training_losses.append(training_loss)
        training_accuracies.append(training_accuracy)

        ## Get test accuracy and loss
        test_accuracy, test_loss = get_accuracy(model, loader, test_env, device)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        print("|\t{}\t||\t{:.2f}\t\t|\t{:.2f}\t\t||\t{:.2e}\t|\t{:.2e}\t|".format(epoch, training_accuracy, test_accuracy, training_loss, test_loss))

    return training_accuracies, training_losses, test_accuracies, test_losses

def get_accuracy(model, loader, env, device):

    model.eval()
    
    sequences = 0
    total_loss = 0
    correct_guess = 0
    total_guess = 0
    with torch.no_grad():
        for x,y in loader:

            # Send everything onto device
            x, y = x.to(device), y.to(device)

            ## Group all inputs and get prediction
            all_out = []
            hidden = model.initHidden(x.shape[0]).to(device)
            for i in range(x.shape[1]):
                out, hidden = model(x[:,i,:,:], hidden)
                all_out.append(out)
            
            for e in env:
                env_out = all_out[e]
                env_loss = F.nll_loss(env_out, y[:,e])  # Only consider labels after the first frame
                guess = env_out.max(1)[1]
                correct_guess += guess.eq(y[:,e]).cpu().sum()
                total_guess += guess.shape[0]
                total_loss += env_loss
                sequences += env_out.shape[0]
            
    return correct_guess * 100 / total_guess, total_loss / sequences

def XOR(a, b):
    return ( a - b ).abs()

def bernoulli(p, size):
    return ( torch.rand(size) < p ).float()

def color_dataset(images, labels, step, p, d):

    # Add label noise
    labels = XOR(labels, bernoulli(d, labels.shape)).long()

    # Choose colors
    colors = XOR(labels, bernoulli(p, labels.shape))

    # Apply colors
    if step == 0:  # If coloring first frame, set all color to green
        for sample in range(colors.shape[0]):
            images[sample,step,0,:,:] *= 0 
    else:
        for sample in range(colors.shape[0]):
            images[sample,step,(1-colors[sample,step]).long(),:,:] *= 0 

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
    MNIST_images = torch.cat((train_ds.data, test_ds.data))
    MNIST_labels = torch.cat((train_ds.targets, test_ds.targets))

    # Create sequences of 3 digits
    MNIST_images = MNIST_images.reshape(-1,4,28,28)

    # With their corresponding label
    MNIST_labels = MNIST_labels.reshape(-1,4)

    # Assign label to the objective : Is the last number in the sequence larger than the current
    MNIST_labels = ( MNIST_labels[:,:3] > MNIST_labels[:,1:] )
    MNIST_labels = torch.cat((torch.zeros((MNIST_labels.shape[0],1)), MNIST_labels), 1)

    ## Make the color datasets

    loader = []          # array of training environment dataloaders
    d = 0.25                   # Label noise
    envs = [0.8, 0.9, 0.1]            # Environment is a function of correlation
    train_env = [1,2]
    test_env = [3]

    # Configure channels and first frame
    colored_images = torch.stack([MNIST_images,MNIST_images], dim=2) # Stack a second color channel
    colored_images, colored_labels = color_dataset(colored_images, MNIST_labels, 0, 1, 0) # Color first frame
    for i, e in enumerate(envs):

        # Color i-th frame subset
        colored_images, colored_labels = color_dataset(colored_images, MNIST_labels, i+1, e, d)

    # Make Tensor dataset and dataloader
    td = torch.utils.data.TensorDataset(colored_images, colored_labels)
    loader = torch.utils.data.DataLoader(td, batch_size=flags.batch_size, shuffle=True)

    ################
    ## Training model

    ## Initialize some RNN
    model = RNN(2*MNIST_images.shape[2]*MNIST_images.shape[3], 50, 10, 2)

    ## Train it
    model.to(device)
    train(flags, model, loader, train_env, test_env, device)

    ## Plot images
    # show_images = torch.cat([colored_images,torch.zeros_like(colored_images[:,:,0:1,:,:])], dim=2)
    # fig, axs = plt.subplots(3,4)
    # axs[0,0].imshow(show_images[0,0,:,:,:].permute(1,2,0))
    # axs[0,0].set_ylabel('Sequence 1')
    # axs[0,1].imshow(show_images[0,1,:,:,:].permute(1,2,0))
    # axs[0,1].set_title('Label = 1')
    # axs[0,2].imshow(show_images[0,2,:,:,:].permute(1,2,0))
    # axs[0,2].set_title('Label = 0')
    # axs[0,3].imshow(show_images[0,3,:,:,:].permute(1,2,0))
    # axs[0,3].set_title('Label = 1')
    # axs[1,0].imshow(show_images[1,0,:,:,:].permute(1,2,0))
    # axs[1,0].set_ylabel('Sequence 2')
    # axs[1,1].imshow(show_images[1,1,:,:,:].permute(1,2,0))
    # axs[1,1].set_title('Label = 0')
    # axs[1,2].imshow(show_images[1,2,:,:,:].permute(1,2,0))
    # axs[1,2].set_title('Label = 1')
    # axs[1,3].imshow(show_images[1,3,:,:,:].permute(1,2,0))
    # axs[1,3].set_title('Label = 0')
    # axs[2,0].imshow(show_images[2,0,:,:,:].permute(1,2,0))
    # axs[2,0].set_ylabel('Sequence 3')
    # axs[2,0].set_xlabel('Time Step 1')
    # axs[2,1].imshow(show_images[2,1,:,:,:].permute(1,2,0))
    # axs[2,1].set_xlabel('Time Step 2')
    # axs[2,1].set_title('Label = 0')
    # axs[2,2].imshow(show_images[2,2,:,:,:].permute(1,2,0))
    # axs[2,2].set_xlabel('Time Step 3')
    # axs[2,2].set_title('Label = 1')
    # axs[2,3].imshow(show_images[2,3,:,:,:].permute(1,2,0))
    # axs[2,3].set_xlabel('Time Step 4')
    # axs[2,3].set_title('Label = 0')
    # for row in axs:
    #     for ax in row:
    #         ax.set_xticks([]) 
    #         ax.set_yticks([]) 
    # plt.tight_layout()
    # plt.savefig('./figure/Temporal_CMNIST.pdf')

import os
import argparse
import numpy as np

import torch
from torchvision import datasets, transforms

## Remove later
import matplotlib.pyplot as plt



if __name__ == '__main__':

    ## Args
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    parser = argparse.ArgumentParser(description='Train MLPs')
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
    test_ds.data = test_ds.data.reshape(-1,3,28,28)
    # With their corresponding label
    train_ds.targets = train_ds.targets.reshape(-1,3)
    test_ds.targets = test_ds.targets.reshape(-1,3)
    # Assign label to the objective : Is the last number in the sequence larger than the current
    train_ds.targets = ( train_ds.targets[:,:2] > train_ds.targets[:,1:] ).long()
    test_ds.targets = ( test_ds.targets[:,:2] > test_ds.targets[:,1:] ).long()


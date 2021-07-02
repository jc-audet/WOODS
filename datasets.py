import os
import argparse
import numpy as np

import torch
import torch.nn.functional as F
from torch import nn, optim
from torchvision import datasets, transforms

def biggest_multiple(multiple_of, input_number):
    return input_number - input_number % multiple_of

def XOR(a, b):
    return ( a - b ).abs()

def bernoulli(p, size):
    return ( torch.rand(size) < p ).float()

def color_dataset(ds_setup, images, labels, env_id, p, d):

    if ds_setup == 'seq':

        # Add label noise
        labels = XOR(labels, bernoulli(d, labels.shape)).long()

        # Choose colors
        colors = XOR(labels, bernoulli(1-p, labels.shape))

        # Stack a second color channel
        images = torch.stack([images,images], dim=2)

        # Apply colors
        for sample in range(colors.shape[0]):
            for frame in range(colors.shape[1]):
                if not frame == 0:      # Leave first channel both colors
                    images[sample,frame,colors[sample,frame].long(),:,:] *= 0

    elif ds_setup == 'step':

        # Add label noise
        labels = XOR(labels, bernoulli(d, labels.shape)).long()

        # Choose colors
        colors = XOR(labels, bernoulli(1-p, labels.shape))

        # Apply colors
        if env_id == 0:  # If coloring first frame, do not touch the color
            pass 
            # for sample in range(colors.shape[0]):
            #     images[sample,env_id,0,:,:] *= 0 
        else:
            for sample in range(colors.shape[0]):
                images[sample,env_id,(1-colors[sample,env_id]).long(),:,:] *= 0 

    return images, labels

def make_dataset(ds_setup, time_steps, train_ds, test_ds, batch_size):

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
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        input_size = 28 * 28

        return input_size, train_loader, test_loader

    elif ds_setup == 'seq':

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
        d = 0.25                    # Label noise
        envs = [0.8, 0.9, 0.1]            # Environment is a function of correlation
        test_env = 2
        for i, e in enumerate(envs):

            # Choose data subset
            images = MNIST_images[i::len(envs)]
            labels = MNIST_labels[i::len(envs)]

            # Color subset
            colored_images, colored_labels = color_dataset(ds_setup, images, labels, i, e, d)

            # Make Tensor dataset
            td = torch.utils.data.TensorDataset(colored_images, colored_labels)

            # Make dataloader
            if i==test_env:
                test_loader = torch.utils.data.DataLoader(td, batch_size=batch_size, shuffle=False)
            else:
                if batch_size < len(envs):
                    train_loaders.append( torch.utils.data.DataLoader(td, batch_size=1, shuffle=True) )
                else:
                    train_loaders.append( torch.utils.data.DataLoader(td, batch_size=batch_size//len(envs), shuffle=True) )

        input_size = 2 * 28 * 28

        return input_size, train_loaders, test_loader

    elif ds_setup == 'step':

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

        d = 0.25                  # Label noise
        envs = [0.8, 0.9, 0.1]            # Environment is a function of correlation
        train_env = [1,2]
        test_env = [3]

        # Configure channels and first frame
        colored_images = torch.stack([MNIST_images,MNIST_images], dim=2) # Stack a second color channel
        
        for i, e in enumerate(envs):

            # Color i-th frame subset
            colored_images, colored_labels = color_dataset(ds_setup, colored_images, MNIST_labels, i+1, e, d)

        # Make Tensor dataset and dataloader
        td = torch.utils.data.TensorDataset(colored_images, colored_labels)
        loader = torch.utils.data.DataLoader(td, batch_size=batch_size, shuffle=True)

        input_size = 2 * 28 * 28

        return input_size, loader, []
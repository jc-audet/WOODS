import os
import argparse
import numpy as np

import torch
import torch.nn.functional as F
from torch import nn, optim
from torchvision import datasets, transforms

import matplotlib.pyplot as plt

def plot_sequences(ds_setup, images, labels):
    
    if ds_setup == 'grey':
        fig, axs = plt.subplots(3,4)
        axs[0,0].imshow(images[0,0,:,:], cmap='gray')
        axs[0,0].set_ylabel('Sequence 1')
        axs[0,1].imshow(images[0,1,:,:], cmap='gray')
        axs[0,1].set_title('Label = '+str(labels[0,1].cpu().item()))
        axs[0,2].imshow(images[0,2,:,:], cmap='gray')
        axs[0,2].set_title('Label = '+str(labels[0,2].cpu().item()))
        axs[0,3].imshow(images[0,3,:,:], cmap='gray')
        axs[0,3].set_title('Label = '+str(labels[0,3].cpu().item()))
        axs[1,0].imshow(images[1,0,:,:], cmap='gray')
        axs[1,0].set_ylabel('Sequence 2')
        axs[1,1].imshow(images[1,1,:,:], cmap='gray')
        axs[1,1].set_title('Label = '+str(labels[1,1].cpu().item()))
        axs[1,2].imshow(images[1,2,:,:], cmap='gray')
        axs[1,2].set_title('Label = '+str(labels[1,2].cpu().item()))
        axs[1,3].imshow(images[1,3,:,:], cmap='gray')
        axs[1,3].set_title('Label = '+str(labels[1,3].cpu().item()))
        axs[2,0].imshow(images[2,0,:,:], cmap='gray')
        axs[2,0].set_ylabel('Sequence 3')
        axs[2,0].set_xlabel('Time Step 1')
        axs[2,1].imshow(images[2,1,:,:], cmap='gray')
        axs[2,1].set_xlabel('Time Step 2')
        axs[2,1].set_title('Label = '+str(labels[2,1].cpu().item()))
        axs[2,2].imshow(images[2,2,:,:], cmap='gray')
        axs[2,2].set_xlabel('Time Step 3')
        axs[2,2].set_title('Label = '+str(labels[2,2].cpu().item()))
        axs[2,3].imshow(images[2,3,:,:], cmap='gray')
        axs[2,3].set_xlabel('Time Step 4')
        axs[2,3].set_title('Label = '+str(labels[2,3].cpu().item()))
        for row in axs:
            for ax in row:
                ax.set_xticks([]) 
                ax.set_yticks([]) 
        plt.tight_layout()
        plt.savefig('./figure/TCMNIST_'+ds_setup+'.pdf')
    else:
        show_images = torch.cat([images,torch.zeros_like(images[:,:,0:1,:,:])], dim=2)
        fig, axs = plt.subplots(3,4)
        axs[0,0].imshow(show_images[0,0,:,:,:].permute(1,2,0))
        axs[0,0].set_ylabel('Sequence 1')
        axs[0,1].imshow(show_images[0,1,:,:,:].permute(1,2,0))
        axs[0,1].set_title('Label = '+str(labels[0,1].cpu().item()))
        axs[0,2].imshow(show_images[0,2,:,:,:].permute(1,2,0))
        axs[0,2].set_title('Label = '+str(labels[0,2].cpu().item()))
        axs[0,3].imshow(show_images[0,3,:,:,:].permute(1,2,0))
        axs[0,3].set_title('Label = '+str(labels[0,3].cpu().item()))
        axs[1,0].imshow(show_images[1,0,:,:,:].permute(1,2,0))
        axs[1,0].set_ylabel('Sequence 2')
        axs[1,1].imshow(show_images[1,1,:,:,:].permute(1,2,0))
        axs[1,1].set_title('Label = '+str(labels[1,1].cpu().item()))
        axs[1,2].imshow(show_images[1,2,:,:,:].permute(1,2,0))
        axs[1,2].set_title('Label = '+str(labels[1,2].cpu().item()))
        axs[1,3].imshow(show_images[1,3,:,:,:].permute(1,2,0))
        axs[1,3].set_title('Label = '+str(labels[1,3].cpu().item()))
        axs[2,0].imshow(show_images[2,0,:,:,:].permute(1,2,0))
        axs[2,0].set_ylabel('Sequence 3')
        axs[2,0].set_xlabel('Time Step 1')
        axs[2,1].imshow(show_images[2,1,:,:,:].permute(1,2,0))
        axs[2,1].set_xlabel('Time Step 2')
        axs[2,1].set_title('Label = '+str(labels[2,1].cpu().item()))
        axs[2,2].imshow(show_images[2,2,:,:,:].permute(1,2,0))
        axs[2,2].set_xlabel('Time Step 3')
        axs[2,2].set_title('Label = '+str(labels[2,2].cpu().item()))
        axs[2,3].imshow(show_images[2,3,:,:,:].permute(1,2,0))
        axs[2,3].set_xlabel('Time Step 4')
        axs[2,3].set_title('Label = '+str(labels[2,3].cpu().item()))
        for row in axs:
            for ax in row:
                ax.set_xticks([]) 
                ax.set_yticks([]) 
        plt.tight_layout()
        plt.savefig('./figure/TCMNIST_'+ds_setup+'.pdf')

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
        labels[:,env_id] = XOR(labels[:,env_id], bernoulli(d, labels[:,env_id].shape)).long()

        # Choose colors
        colors = XOR(labels[:,env_id], bernoulli(1-p, labels[:,env_id].shape))

        # Apply colors
        for sample in range(colors.shape[0]):
            images[sample,env_id,colors[sample].long(),:,:] *= 0 

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
        # train_ds.targets = ( train_ds.targets[:,:-1] > train_ds.targets[:,1:] )       # Is the previous one bigger than the current one?
        train_ds.targets = ( train_ds.targets[:,:-1] + train_ds.targets[:,1:] ) % 2     # Is the sum of this one and the last one an even number?
        train_ds.targets = torch.cat((torch.zeros((train_ds.targets.shape[0],1)), train_ds.targets), 1).long()
        # test_ds.targets = ( test_ds.targets[:,:-1] > test_ds.targets[:,1:] )          # Is the previous one bigger than the current one?
        test_ds.targets = ( test_ds.targets[:,:-1] + test_ds.targets[:,1:] ) % 2        # Is the sum of this one and the last one an even number?
        test_ds.targets = torch.cat((torch.zeros((test_ds.targets.shape[0],1)), test_ds.targets), 1).long()

        # Make Tensor dataset
        train_dataset = torch.utils.data.TensorDataset(train_ds.data, train_ds.targets)
        test_dataset = torch.utils.data.TensorDataset(test_ds.data, test_ds.targets)

        # Make dataloader
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        input_size = 28 * 28

        plot_sequences(ds_setup, train_ds.data, train_ds.targets)

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

        ########################
        ### Choose the task:
        # MNIST_labels = ( MNIST_labels[:,:-1] > MNIST_labels[:,1:] )        # Is the previous one bigger than the current one?
        MNIST_labels = ( MNIST_labels[:,:-1] + MNIST_labels[:,1:] ) % 2      # Is the sum of this one and the last one an even number?

        MNIST_labels = torch.cat((torch.zeros((MNIST_labels.shape[0],1)), MNIST_labels), 1)

        # Make the color datasets
        show_images = []
        show_labels = []
        train_loaders = []          # array of training environment dataloaders
        d = 0.25                    # Label noise
        envs = [0.8, 0.9, 0.1]      # Environment is a function of correlation
        test_env = 2
        for i, e in enumerate(envs):

            # Choose data subset
            images = MNIST_images[i::len(envs)]
            labels = MNIST_labels[i::len(envs)]

            # Color subset
            colored_images, colored_labels = color_dataset(ds_setup, images, labels, i, e, d)

            show_images.append(colored_images[0,:,:,:,:])
            show_labels.append(colored_labels[0,:])

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

        plot_sequences(ds_setup, torch.stack(show_images, 0), torch.stack(show_labels,0))

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
        ########################
        ### Choose the task:
        # MNIST_labels = ( MNIST_labels[:,:-1] > MNIST_labels[:,1:] )        # Is the previous one bigger than the current one?
        MNIST_labels = ( MNIST_labels[:,:-1] + MNIST_labels[:,1:] ) % 2      # Is the sum of this one and the last one an even number?
        
        colored_labels = torch.cat((torch.zeros((MNIST_labels.shape[0],1)), MNIST_labels), 1)

        ## Make the color datasets
        # Stack a second color channel
        colored_images = torch.stack([MNIST_images,MNIST_images], dim=2)

        d = 0.25                # Label noise
        envs = [0.8, 0.9, 0.1]  # Environment is a function of correlation
        train_env = [1,2]
        test_env = [3]
        for i, e in enumerate(envs):

            # Color i-th frame subset
            colored_images, colored_labels = color_dataset(ds_setup, colored_images, colored_labels, i+1, e, d)

        # Make Tensor dataset and dataloader
        td = torch.utils.data.TensorDataset(colored_images, colored_labels.long())
        loader = torch.utils.data.DataLoader(td, batch_size=batch_size, shuffle=True)

        input_size = 2 * 28 * 28

        plot_sequences(ds_setup, colored_images, colored_labels.long())

        return input_size, loader, []

import os
import copy
import argparse
import numpy as np

import torch
import torch.nn.functional as F
from torch import nn, optim
from torchvision import datasets, transforms

from scipy import fft

import matplotlib.pyplot as plt

DATASETS = [
    # 1D datasets
    'Fourier_basic',
    'Fourier_cheat',
    'Spurious_Fourier',
    # Small images
    "TMNIST_grey",
    # Small correlation shift dataset
    "TCMNIST_seq",
    "TCMNIST_step",
]

def get_dataset_class(dataset_name):
    """Return the dataset class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]

def biggest_multiple(multiple_of, input_number):
    return input_number - input_number % multiple_of

def XOR(a, b):
    return ( a - b ).abs()

def bernoulli(p, size):
    return ( torch.rand(size) < p ).float()

class Single_Dim_dataset:
    setup = 'test'

    def __init__(self):
        pass

    def get_setup(self):
        return self.setup

    def get_input_size(self):
        return 1

class Fourier(Single_Dim_dataset):
    def __init__(self, flags):
        super(Fourier, self).__init__()

        ## Define label 0 and 1 Fourier spectrum
        self.fourier_0 = np.zeros(1000)
        self.fourier_0[800:900] = np.linspace(0, 500, num=100)
        self.fourier_1 = np.zeros(1000)
        self.fourier_1[800:900] = np.linspace(500, 0, num=100)

class Fourier_basic(Fourier):
    setup = 'basic'
    time_pred = [49]

    def __init__(self, flags, batch_size):
        super(Fourier_basic, self).__init__(flags)

        ## Make the full time series with inverse fft
        signal_0 = fft.irfft(self.fourier_0, n=10000)[1000:9000]
        signal_1 = fft.irfft(self.fourier_1, n=10000)[1000:9000]
        signal_0 = torch.tensor( signal_0.reshape(-1,50) ).float()
        signal_1 = torch.tensor( signal_1.reshape(-1,50) ).float()
        signal = torch.cat((signal_0, signal_1))

        ## Create the labels
        labels_0 = torch.zeros((signal_0.shape[0],50)).long()
        labels_1 = torch.ones((signal_1.shape[0],50)).long()
        labels = torch.cat((labels_0, labels_1))

        ## Permute and split
        perm = torch.randperm(labels.shape[0])
        split = labels.shape[0] // 5
        perm_signal, perm_labels = signal[perm,:], labels[perm,:]
        train_signal, test_signal = perm_signal[split:,:], perm_signal[:split,:]
        train_labels, test_labels = perm_labels[split:,:], perm_labels[:split,:]

        ## Create tensor dataset and dataloader
        train_dataset = torch.utils.data.TensorDataset(train_signal, train_labels)
        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataset = torch.utils.data.TensorDataset(test_signal, test_labels)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        
    def get_loaders(self):
        return self.train_loader, self.test_loader

class Spurious_Fourier(Fourier):

    setup = 'seq'
    time_pred = [49]
    label_noise = 0.25          # Label noise
    envs = [0.8, 0.9, 0.1]      # Environment is a function of correlation

    def __init__(self, flags, batch_size):
        super(Spurious_Fourier, self).__init__(flags)

        ## Define the spurious Fourier spectrum (one direct and the inverse)
        self.direct_fourier_0 = copy.deepcopy(self.fourier_0)
        self.direct_fourier_1 = copy.deepcopy(self.fourier_1)
        self.direct_fourier_0[250] = 1000
        self.direct_fourier_1[400] = 1000

        self.inverse_fourier_0 = copy.deepcopy(self.fourier_0)
        self.inverse_fourier_1 = copy.deepcopy(self.fourier_1)
        self.inverse_fourier_0[400] = 1000
        self.inverse_fourier_1[250] = 1000

        ## Create the sequences for direct and inverse
        direct_signal_0 = fft.irfft(self.direct_fourier_0, n=10000)[1000:9000]
        direct_signal_0 = torch.tensor( direct_signal_0.reshape(-1,50) ).float()
        perm = torch.randperm(direct_signal_0.shape[0])
        direct_signal_0 = direct_signal_0[perm,:]
        direct_signal_1 = fft.irfft(self.direct_fourier_1, n=10000)[1000:9000]
        direct_signal_1 = torch.tensor( direct_signal_1.reshape(-1,50) ).float()
        perm = torch.randperm(direct_signal_1.shape[0])
        direct_signal_1 = direct_signal_1[perm,:]
        direct_signal = [direct_signal_0, direct_signal_1]

        inverse_signal_0 = fft.irfft(self.inverse_fourier_0, n=10000)[1000:9000]
        inverse_signal_0 = torch.tensor( inverse_signal_0.reshape(-1,50) ).float()
        perm = torch.randperm(inverse_signal_0.shape[0])
        inverse_signal_0 = inverse_signal_0[perm,:]
        inverse_signal_1 = fft.irfft(self.inverse_fourier_1, n=10000)[1000:9000]
        inverse_signal_1 = torch.tensor( inverse_signal_1.reshape(-1,50) ).float()
        perm = torch.randperm(inverse_signal_1.shape[0])
        inverse_signal_1 = inverse_signal_1[perm,:]
        inverse_signal = [inverse_signal_0, inverse_signal_1]

        ## Create the environments with different correlations
        test_env = 2
        env_size = 150
        self.train_loaders = []          # array of training environment dataloaders
        for i, e in enumerate(self.envs):

            ## Create set of labels
            env_labels_0 = torch.zeros((env_size // 2, 50)).long()
            env_labels_1 = torch.ones((env_size // 2, 50)).long()
            env_labels = torch.cat((env_labels_0, env_labels_1))

            ## Fill signal
            env_signal = torch.zeros((env_size, 50))
            for j, label in enumerate(env_labels[:,-1]):

                # Label noise
                if bool(bernoulli(self.label_noise, 1)):
                    # Correlation to label
                    if bool(bernoulli(e, 1)):
                        env_signal[j,:] = inverse_signal[label][0,:]
                        inverse_signal[label] = inverse_signal[label][1:,:]
                    else:
                        env_signal[j,:] = direct_signal[label][0,:]
                        direct_signal[label] = direct_signal[label][1:,:]
                    
                    # Flip the label
                    env_labels[j, -1] = XOR(label, 1)
                else:
                    if bool(bernoulli(e, 1)):
                        env_signal[j,:] = direct_signal[label][0,:]
                        direct_signal[label] = direct_signal[label][1:,:]
                    else:
                        env_signal[j,:] = inverse_signal[label][0,:]
                        inverse_signal[label] = inverse_signal[label][1:,:]

            # Make Tensor dataset
            td = torch.utils.data.TensorDataset(env_signal, env_labels)

            # Make dataloader
            if i==test_env:
                self.test_loader = torch.utils.data.DataLoader(td, batch_size=batch_size, shuffle=False)
            else:
                if batch_size < len(self.envs):
                    self.train_loaders.append( torch.utils.data.DataLoader(td, batch_size=1, shuffle=True) )
                else:
                    self.train_loaders.append( torch.utils.data.DataLoader(td, batch_size=batch_size//len(self.envs), shuffle=True) )

    def get_loaders(self):
        return self.train_loaders, self.test_loader

class TMNIST:
    setup = 'basic'         # Child classes must overwrite
    time_pred = [1,2,3]     # Child classes can overwrite

    def __init__(self, data_path):
        
        ## Import original MNIST data
        MNIST_tfrm = transforms.Compose([ transforms.ToTensor() ])

        self.train_ds = datasets.MNIST(data_path, train=True, download=True, transform=MNIST_tfrm) 
        self.test_ds = datasets.MNIST(data_path, train=False, download=True, transform=MNIST_tfrm) 

    def get_setup(self):
        return self.setup

class TMNIST_grey(TMNIST):
    setup = 'basic'

    def __init__(self, flags, batch_size):
        super(TMNIST_grey, self).__init__(flags.data_path)

        # Create sequences of 3 digits
        n_train_samples = biggest_multiple(flags.time_steps, self.train_ds.data.shape[0])
        n_test_samples = biggest_multiple(flags.time_steps, self.test_ds.data.shape[0])
        self.train_ds.data = self.train_ds.data[:n_train_samples].reshape(-1,flags.time_steps,28,28)
        self.test_ds.data = self.test_ds.data[:n_test_samples].reshape(-1,flags.time_steps,28,28)

        # With their corresponding label
        self.train_ds.targets = self.train_ds.targets[:n_train_samples].reshape(-1,flags.time_steps)
        self.test_ds.targets = self.test_ds.targets[:n_test_samples].reshape(-1,flags.time_steps)


        # Assign label to the objective : Is the last number in the sequence larger than the current
        # self.train_ds.targets = ( self.train_ds.targets[:,:-1] > self.train_ds.targets[:,1:] )       # Is the previous one bigger than the current one?
        self.train_ds.targets = ( self.train_ds.targets[:,:-1] + self.train_ds.targets[:,1:] ) % 2     # Is the sum of this one and the last one an even number?
        self.train_ds.targets = torch.cat((torch.zeros((self.train_ds.targets.shape[0],1)), self.train_ds.targets), 1).long()
        # self.test_ds.targets = ( self.test_ds.targets[:,:-1] > self.test_ds.targets[:,1:] )          # Is the previous one bigger than the current one?
        self.test_ds.targets = ( self.test_ds.targets[:,:-1] + self.test_ds.targets[:,1:] ) % 2        # Is the sum of this one and the last one an even number?
        self.test_ds.targets = torch.cat((torch.zeros((self.test_ds.targets.shape[0],1)), self.test_ds.targets), 1).long()

        # Make Tensor dataset
        train_dataset = torch.utils.data.TensorDataset(self.train_ds.data, self.train_ds.targets)
        test_dataset = torch.utils.data.TensorDataset(self.test_ds.data, self.test_ds.targets)

        # Make dataloader
        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        self.input_size = 28 * 28
        
    def get_input_size(self):
        return self.input_size
        
    def get_loaders(self):
        return self.train_loader, self.test_loader

class TCMNIST(TMNIST):
    def __init__(self, data_path, time_steps):
        super(TCMNIST, self).__init__( data_path)

        # Concatenate all data and labels
        MNIST_images = torch.cat((self.train_ds.data, self.test_ds.data))
        MNIST_labels = torch.cat((self.train_ds.targets, self.test_ds.targets))

        # Create sequences of 3 digits
        self.MNIST_images = MNIST_images.reshape(-1, time_steps, 28, 28)

        # With their corresponding label
        MNIST_labels = MNIST_labels.reshape(-1, time_steps)

        # Assign label to the objective : Is the last number in the sequence larger than the current
        ########################
        ### Choose the task:
        # MNIST_labels = ( MNIST_labels[:,:-1] > MNIST_labels[:,1:] )        # Is the previous one bigger than the current one?
        MNIST_labels = ( MNIST_labels[:,:-1] + MNIST_labels[:,1:] ) % 2      # Is the sum of this one and the last one an even number?
        
        self.MNIST_labels = torch.cat((torch.zeros((MNIST_labels.shape[0],1)), MNIST_labels), 1)

        self.input_size = 2 * 28 * 28

    def get_input_size(self):
        return self.input_size

class TCMNIST_seq(TCMNIST):

    setup = 'seq'
    label_noise = 0.25                    # Label noise
    envs = [0.8, 0.9, 0.1]      # Environment is a function of correlation

    def __init__(self, flags, batch_size):
        super(TCMNIST_seq, self).__init__(flags.data_path, flags.time_steps)

        # Make the color datasets
        self.train_loaders = []          # array of training environment dataloaders
        test_env = 2
        for i, e in enumerate(self.envs):

            # Choose data subset
            images = self.MNIST_images[i::len(envs)]
            labels = self.MNIST_labels[i::len(envs)]

            # Color subset
            colored_images, colored_labels = self.color_dataset(images, labels, i, e, self.label_noise)

            # Make Tensor dataset
            td = torch.utils.data.TensorDataset(colored_images, colored_labels)

            # Make dataloader
            if i==test_env:
                self.test_loader = torch.utils.data.DataLoader(td, batch_size=batch_size, shuffle=False)
            else:
                if batch_size < len(envs):
                    self.train_loaders.append( torch.utils.data.DataLoader(td, batch_size=1, shuffle=True) )
                else:
                    self.train_loaders.append( torch.utils.data.DataLoader(td, batch_size=batch_size//len(envs), shuffle=True) )

    def color_dataset(self, images, labels, env_id, p, d):

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

        return images, labels

    def get_loaders(self):
        return self.train_loaders, self.test_loader

class TCMNIST_step(TCMNIST):

    setup = 'step'
    time_pred = [1,2]
    label_noise = 0.25                # Label noise
    envs = [0.8, 0.9, 0.1]  # Environment is a function of correlation

    def __init__(self, flags, batch_size):
        super(TCMNIST_step, self).__init__(flags.data_path, flags.time_steps)

        ## Make the color datasets
        # Stack a second color channel
        colored_images = torch.stack([self.MNIST_images, self.MNIST_images], dim=2)

        train_env = [1,2]
        test_env = [3]
        for i, e in enumerate(self.envs):

            # Color i-th frame subset
            colored_images, colored_labels = self.color_dataset(colored_images, self.MNIST_labels, i+1, e, self.label_noise)

        # Make Tensor dataset and dataloader
        td = torch.utils.data.TensorDataset(colored_images, colored_labels.long())
        self.loader = torch.utils.data.DataLoader(td, batch_size=batch_size, shuffle=True)

    def color_dataset(self, images, labels, env_id, p, d):

        # Add label noise
        labels[:,env_id] = XOR(labels[:,env_id], bernoulli(d, labels[:,env_id].shape)).long()

        # Choose colors
        colors = XOR(labels[:,env_id], bernoulli(1-p, labels[:,env_id].shape))

        # Apply colors
        for sample in range(colors.shape[0]):
            images[sample,env_id,colors[sample].long(),:,:] *= 0 

        return images, labels
        
    def get_loaders(self):
        return self.loader, []


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
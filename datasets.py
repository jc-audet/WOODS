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
    'Spurious_Fourier',
    # Small images
    "TMNIST",
    # Small correlation shift dataset
    "TCMNIST_seq",
    "TCMNIST_step",
]

'''
TODO Make a note the says that you need the 'time_pred' and 'setup' variable for every new dataset added
TODO Make a package test that checks if every class has 'time_pred' and 'setup'
TODO Notify users that datasets need to be (batch, time, dimensions...)
'''

def get_dataset_class(dataset_name):
    """Return the dataset class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]

def num_environments(dataset_name):
    return len(get_dataset_class(dataset_name).ENVS)

def biggest_multiple(multiple_of, input_number):
    return input_number - input_number % multiple_of

def XOR(a, b):
    return ( a - b ).abs()

def bernoulli(p, size):
    return ( torch.rand(size) < p ).float()

def make_split(dataset, holdout_fraction, seed=0):

    split = int(len(dataset)*holdout_fraction)

    keys = list(range(len(dataset)))
    np.random.RandomState(seed).shuffle(keys)

    in_split = dataset[keys[split:]]
    out_split = dataset[keys[:split]]

    return torch.utils.data.TensorDataset(*in_split), torch.utils.data.TensorDataset(*out_split)

class Single_Domain_Dataset:
    N_STEPS = 5001
    CHECKPOINT_FREQ = 50
    SETUP = None
    PRED_TIME = [None]
    ENVS = [None]
    INPUT_SIZE = None

    def __init__(self):
        pass

    def get_setup(self):
        return self.SETUP

    def get_input_size(self):
        return self.INPUT_SIZE

    def get_output_size(self):
        return self.OUTPUT_SIZE

    def get_envs(self):
        return self.ENVS

    def get_pred_time(self):
        return self.PRED_TIME

    def get_train_loaders(self):
        loaders_ID = [str(env)+'_in' for i, env in enumerate(self.ENVS)]
        loaders = [l for i, l in enumerate(self.in_loaders)] 
        return loaders_ID, loaders
    
    def get_val_loaders(self):
        loaders_ID = [str(env)+'_out' for env in self.ENVS]
        loaders = self.out_loaders
        return loaders_ID, loaders

class Multi_Domain_Dataset:
    N_STEPS = 501
    CHECKPOINT_FREQ = 50
    SETUP = None
    PRED_TIME = [None]
    ENVS = [None]
    INPUT_SIZE = None
    OUTPUT_SIZE = None

    def __init__(self):
        pass

    def get_setup(self):
        return self.SETUP

    def get_input_size(self):
        return self.INPUT_SIZE

    def get_output_size(self):
        return self.OUTPUT_SIZE

    def get_envs(self):
        return self.ENVS

    def get_pred_time(self):
        return self.PRED_TIME

    def get_train_loaders(self):
        loaders_ID = [str(env)+'_in' for i, env in enumerate(self.ENVS) if i != self.test_env]
        loaders = [l for i, l in enumerate(self.in_loaders) if i != self.test_env] 
        return loaders_ID, loaders
    
    def get_val_loaders(self):
        loaders_ID = [str(env)+'_out' for env in self.ENVS] + [str(self.ENVS[self.test_env])+'_in']
        loaders = self.out_loaders + [self.in_loaders[self.test_env]]
        return loaders_ID, loaders

class Fourier_basic(Single_Domain_Dataset):
    SETUP = 'seq'
    PRED_TIME = [49]
    ENVS = ['no_spur']
    INPUT_SIZE = 1
    OUTPUT_SIZE = 2

    def __init__(self, flags, batch_size):
        super(Fourier_basic, self).__init__()

        ## Define label 0 and 1 Fourier spectrum
        self.fourier_0 = np.zeros(1000)
        self.fourier_0[800:900] = np.linspace(0, 500, num=100)
        self.fourier_1 = np.zeros(1000)
        self.fourier_1[800:900] = np.linspace(500, 0, num=100)

        ## Make the full time series with inverse fft
        signal_0 = fft.irfft(self.fourier_0, n=10000)[1000:9000]
        signal_1 = fft.irfft(self.fourier_1, n=10000)[1000:9000]
        signal_0 = torch.tensor( signal_0.reshape(-1,50) ).float()
        signal_1 = torch.tensor( signal_1.reshape(-1,50) ).float()
        signal = torch.cat((signal_0, signal_1))

        plt.figure()
        plt.plot(signal_0[50,:], 'r', label='Label 0')
        plt.plot(signal_1[50,:], 'b', label='Label 1')
        plt.legend()
        plt.savefig('./figure/fourier_clean_signal.pdf')

        ## Create the labels
        labels_0 = torch.zeros((signal_0.shape[0],1)).long()
        labels_1 = torch.ones((signal_1.shape[0],1)).long()
        labels = torch.cat((labels_0, labels_1))

        ## Create tensor dataset and dataloader
        self.in_loaders, self.out_loaders = [], []
        for e in self.ENVS:
            dataset = torch.utils.data.TensorDataset(signal, labels)
            in_dataset, out_dataset = make_split(dataset, flags.holdout_fraction)
            in_loader = torch.utils.data.DataLoader(in_dataset, batch_size=batch_size, shuffle=True)
            self.in_loaders.append(in_loader)
            out_loader = torch.utils.data.DataLoader(out_dataset, batch_size=64, shuffle=False)
            self.out_loaders.append(out_loader)

class Spurious_Fourier(Multi_Domain_Dataset):
    SETUP = 'seq'
    INPUT_SIZE = 1
    OUTPUT_SIZE = 2
    PRED_TIME = [49]
    label_noise = 0.25          # Label noise
    ENVS = [0.1, 0.8, 0.9]      # Environment is a function of correlation

    def __init__(self, flags, batch_size):
        super(Spurious_Fourier, self).__init__()

        ## Save stuff
        self.test_env = flags.test_env

        ## Define label 0 and 1 Fourier spectrum
        self.fourier_0 = np.zeros(1000)
        self.fourier_0[800:900] = np.linspace(0, 500, num=100)
        self.fourier_1 = np.zeros(1000)
        self.fourier_1[800:900] = np.linspace(500, 0, num=100)

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

        plt.figure()
        plt.plot(direct_signal_0[50,:], 'r')
        plt.plot(inverse_signal_0[50,:], 'b')
        plt.savefig('./figure/fourier_cheat_signal_0.pdf')

        plt.figure()
        plt.plot(direct_signal_1[50,:], 'r')
        plt.plot(inverse_signal_1[50,:], 'b')
        plt.savefig('./figure/fourier_cheat_signal_1.pdf')

        ## Create the environments with different correlations
        env_size = 150
        self.in_loaders, self.out_loaders = [], []
        for i, e in enumerate(self.ENVS):

            ## Create set of labels
            env_labels_0 = torch.zeros((env_size // 2, 1)).long()
            env_labels_1 = torch.ones((env_size // 2, 1)).long()
            env_labels = torch.cat((env_labels_0, env_labels_1))

            ## Fill signal
            env_signal = torch.zeros((env_size, 50))
            for j, label in enumerate(env_labels):

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
            dataset = torch.utils.data.TensorDataset(env_signal, env_labels)

            in_dataset, out_dataset = make_split(dataset, flags.holdout_fraction)
            in_loader = torch.utils.data.DataLoader(in_dataset, batch_size=batch_size, shuffle=True)
            self.in_loaders.append(in_loader)
            out_loader = torch.utils.data.DataLoader(out_dataset, batch_size=64, shuffle=False)
            self.out_loaders.append(out_loader)

        def plot_samples(direct_signal_0, inverse_signal_0, direct_signal_1, inverse_signal_1):

            plt.figure()
            plt.plot(self.fourier_0, 'r', label='Label 0')
            plt.plot(self.fourier_1, 'b', label='Label 1')
            plt.legend()
            plt.savefig('./figure/fourier_clean_task.pdf')

            plt.figure()
            plt.plot(self.direct_fourier_0, 'r')
            plt.plot(self.inverse_fourier_0, 'b')
            plt.savefig('./figure/fourier_cheat_0.pdf')
            
            plt.figure()
            plt.plot(self.direct_fourier_1, 'r')
            plt.plot(self.inverse_fourier_1, 'b')
            plt.savefig('./figure/fourier_cheat_1.pdf')

            plt.figure()
            plt.plot(direct_signal_0[50,:], 'r')
            plt.plot(inverse_signal_0[50,:], 'b')
            plt.savefig('./figure/fourier_cheat_signal_0.pdf')

            plt.figure()
            plt.plot(direct_signal_1[50,:], 'r')
            plt.plot(inverse_signal_1[50,:], 'b')
            plt.savefig('./figure/fourier_cheat_signal_1.pdf')


class TMNIST(Single_Domain_Dataset):
    N_STEPS = 5001
    SETUP = 'seq'
    PRED_TIME = [1, 2, 3]
    INPUT_SIZE = 28*28
    OUTPUT_SIZE = 2
    ENVS = ['grey']

    # Dataset parameters
    SEQ_LEN = 4

    def __init__(self, flags, batch_size):
        super(TMNIST, self).__init__()

        ## Import original MNIST data
        MNIST_tfrm = transforms.Compose([ transforms.ToTensor() ])

        # Get MNIST data
        train_ds = datasets.MNIST(flags.data_path, train=True, download=True, transform=MNIST_tfrm) 
        test_ds = datasets.MNIST(flags.data_path, train=False, download=True, transform=MNIST_tfrm) 

        # Concatenate all data and labels
        MNIST_images = torch.cat((train_ds.data, test_ds.data))
        MNIST_labels = torch.cat((train_ds.targets, test_ds.targets))

        # Create sequences of 3 digits
        TMNIST_images = MNIST_images.reshape(-1,self.SEQ_LEN,28,28)

        # With their corresponding label
        TMNIST_labels = MNIST_labels.reshape(-1,self.SEQ_LEN)

        # Assign label to the objective : Is the last number in the sequence larger than the current
        # self.train_ds.targets = ( self.train_ds.targets[:,:-1] > self.train_ds.targets[:,1:] )       # Is the previous one bigger than the current one?
        TMNIST_labels = ( TMNIST_labels[:,:-1] + TMNIST_labels[:,1:] ) % 2     # Is the sum of this one and the last one an even number?
        TMNIST_labels = TMNIST_labels.long()

        ## Create tensor dataset and dataloader
        self.in_loaders, self.out_loaders = [], []
        for e in self.ENVS:
            dataset = torch.utils.data.TensorDataset(TMNIST_images, TMNIST_labels)
            in_dataset, out_dataset = make_split(dataset, flags.holdout_fraction)
            in_loader = torch.utils.data.DataLoader(in_dataset, batch_size=batch_size, shuffle=True)
            self.in_loaders.append(in_loader)
            out_loader = torch.utils.data.DataLoader(out_dataset, batch_size=64, shuffle=False)
            self.out_loaders.append(out_loader)

        def plot_samples(TMNIST_images, TMNIST_labels):
            fig, axs = plt.subplots(3,4)
            axs[0,0].imshow(TMNIST_images[0,0,:,:], cmap='gray')
            axs[0,0].set_ylabel('Sequence 1')
            axs[0,1].imshow(TMNIST_images[0,1,:,:], cmap='gray')
            axs[0,1].set_title('Label = '+str(TMNIST_labels[0,0].cpu().item()))
            axs[0,2].imshow(TMNIST_images[0,2,:,:], cmap='gray')
            axs[0,2].set_title('Label = '+str(TMNIST_labels[0,1].cpu().item()))
            axs[0,3].imshow(TMNIST_images[0,3,:,:], cmap='gray')
            axs[0,3].set_title('Label = '+str(TMNIST_labels[0,2].cpu().item()))
            axs[1,0].imshow(TMNIST_images[1,0,:,:], cmap='gray')
            axs[1,0].set_ylabel('Sequence 2')
            axs[1,1].imshow(TMNIST_images[1,1,:,:], cmap='gray')
            axs[1,1].set_title('Label = '+str(TMNIST_labels[1,0].cpu().item()))
            axs[1,2].imshow(TMNIST_images[1,2,:,:], cmap='gray')
            axs[1,2].set_title('Label = '+str(TMNIST_labels[1,1].cpu().item()))
            axs[1,3].imshow(TMNIST_images[1,3,:,:], cmap='gray')
            axs[1,3].set_title('Label = '+str(TMNIST_labels[1,2].cpu().item()))
            axs[2,0].imshow(TMNIST_images[2,0,:,:], cmap='gray')
            axs[2,0].set_ylabel('Sequence 3')
            axs[2,0].set_xlabel('Time Step 1')
            axs[2,1].imshow(TMNIST_images[2,1,:,:], cmap='gray')
            axs[2,1].set_xlabel('Time Step 2')
            axs[2,1].set_title('Label = '+str(TMNIST_labels[2,0].cpu().item()))
            axs[2,2].imshow(TMNIST_images[2,2,:,:], cmap='gray')
            axs[2,2].set_xlabel('Time Step 3')
            axs[2,2].set_title('Label = '+str(TMNIST_labels[2,1].cpu().item()))
            axs[2,3].imshow(TMNIST_images[2,3,:,:], cmap='gray')
            axs[2,3].set_xlabel('Time Step 4')
            axs[2,3].set_title('Label = '+str(TMNIST_labels[2,2].cpu().item()))
            for row in axs:
                for ax in row:
                    ax.set_xticks([]) 
                    ax.set_yticks([]) 
            plt.tight_layout()
            plt.savefig('./figure/TCMNIST_'+self.SETUP+'.pdf')


class TCMNIST(Multi_Domain_Dataset):

    N_STEPS = 5001
    PRED_TIME = [1, 2, 3]
    INPUT_SIZE = 2 * 28 * 28
    OUTPUT_SIZE = 2
    SEQ_LEN = 4

    def __init__(self, flags):
        super(TCMNIST, self).__init__()

        ## Import original MNIST data
        MNIST_tfrm = transforms.Compose([ transforms.ToTensor() ])

        # Get MNIST data
        train_ds = datasets.MNIST(flags.data_path, train=True, download=True, transform=MNIST_tfrm) 
        test_ds = datasets.MNIST(flags.data_path, train=False, download=True, transform=MNIST_tfrm) 

        # Concatenate all data and labels
        MNIST_images = torch.cat((train_ds.data, test_ds.data))
        MNIST_labels = torch.cat((train_ds.targets, test_ds.targets))

        # Create sequences of 3 digits
        self.TCMNIST_images = MNIST_images.reshape(-1, self.SEQ_LEN, 28, 28)

        # With their corresponding label
        TCMNIST_labels = MNIST_labels.reshape(-1, self.SEQ_LEN)

        ########################
        ### Choose the task:
        # MNIST_labels = ( MNIST_labels[:,:-1] > MNIST_labels[:,1:] )        # Is the previous one bigger than the current one?
        TCMNIST_labels = ( TCMNIST_labels[:,:-1] + TCMNIST_labels[:,1:] ) % 2      # Is the sum of this one and the last one an even number?
        self.TCMNIST_labels = TCMNIST_labels.long()

    def plot_samples(self, images, labels):

        show_images = torch.cat([images,torch.zeros_like(images[:,:,0:1,:,:])], dim=2)
        fig, axs = plt.subplots(3,4)
        axs[0,0].imshow(show_images[0,0,:,:,:].permute(1,2,0))
        axs[0,0].set_ylabel('Sequence 1')
        axs[0,1].imshow(show_images[0,1,:,:,:].permute(1,2,0))
        axs[0,1].set_title('Label = '+str(labels[0,0].cpu().item()))
        axs[0,2].imshow(show_images[0,2,:,:,:].permute(1,2,0))
        axs[0,2].set_title('Label = '+str(labels[0,1].cpu().item()))
        axs[0,3].imshow(show_images[0,3,:,:,:].permute(1,2,0))
        axs[0,3].set_title('Label = '+str(labels[0,2].cpu().item()))
        axs[1,0].imshow(show_images[1,0,:,:,:].permute(1,2,0))
        axs[1,0].set_ylabel('Sequence 2')
        axs[1,1].imshow(show_images[1,1,:,:,:].permute(1,2,0))
        axs[1,1].set_title('Label = '+str(labels[1,0].cpu().item()))
        axs[1,2].imshow(show_images[1,2,:,:,:].permute(1,2,0))
        axs[1,2].set_title('Label = '+str(labels[1,1].cpu().item()))
        axs[1,3].imshow(show_images[1,3,:,:,:].permute(1,2,0))
        axs[1,3].set_title('Label = '+str(labels[1,2].cpu().item()))
        axs[2,0].imshow(show_images[2,0,:,:,:].permute(1,2,0))
        axs[2,0].set_ylabel('Sequence 3')
        axs[2,0].set_xlabel('Time Step 1')
        axs[2,1].imshow(show_images[2,1,:,:,:].permute(1,2,0))
        axs[2,1].set_xlabel('Time Step 2')
        axs[2,1].set_title('Label = '+str(labels[2,0].cpu().item()))
        axs[2,2].imshow(show_images[2,2,:,:,:].permute(1,2,0))
        axs[2,2].set_xlabel('Time Step 3')
        axs[2,2].set_title('Label = '+str(labels[2,1].cpu().item()))
        axs[2,3].imshow(show_images[2,3,:,:,:].permute(1,2,0))
        axs[2,3].set_xlabel('Time Step 4')
        axs[2,3].set_title('Label = '+str(labels[2,2].cpu().item()))
        for row in axs:
            for ax in row:
                ax.set_xticks([]) 
                ax.set_yticks([]) 
        plt.tight_layout()
        plt.savefig('./figure/TCMNIST_'+self.SETUP+'.pdf')

class TCMNIST_seq(TCMNIST):

    SETUP = 'seq'
    ENVS = [0.1, 0.8, 0.9]      # Environment is a function of correlation

    ## Dataset parameters
    label_noise = 0.25                    # Label noise

    def __init__(self, flags, batch_size):
        super(TCMNIST_seq, self).__init__(flags)

        # Save stuff
        self.test_env = flags.test_env

        # Make the color datasets
        self.in_loaders, self.out_loaders = [], []          # array of training environment dataloaders
        for i, e in enumerate(self.ENVS):

            # Choose data subset
            images = self.TCMNIST_images[i::len(self.ENVS)]
            labels = self.TCMNIST_labels[i::len(self.ENVS)]

            # Color subset
            colored_images, colored_labels = self.color_dataset(images, labels, i, e, self.label_noise)

            # Make Tensor dataset
            dataset = torch.utils.data.TensorDataset(colored_images, colored_labels)

            in_dataset, out_dataset = make_split(dataset, flags.holdout_fraction)
            in_loader = torch.utils.data.DataLoader(in_dataset, batch_size=batch_size, shuffle=True)
            self.in_loaders.append(in_loader)
            out_loader = torch.utils.data.DataLoader(out_dataset, batch_size=64, shuffle=False)
            self.out_loaders.append(out_loader)

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
                images[sample,frame+1,colors[sample,frame].long(),:,:] *= 0

        return images, labels

    def get_loaders(self):
        return self.train_loaders, self.test_loader

class TCMNIST_step(TCMNIST):

    SETUP = 'step'
    ENVS = [0.9, 0.8, 0.1]  # Environment is a function of correlation

    # Dataset parameters
    label_noise = 0.25      # Label noise

    def __init__(self, flags, batch_size):
        super(TCMNIST_step, self).__init__(flags)

        ## Save stuff
        assert flags.test_env < len(self.ENVS), "Test environment not valid"
        self.test_env = flags.test_env

        # Define array of training environment dataloaders
        self.in_loaders, self.out_loaders = [], []          

        ## Make the color datasets
        # Stack a second color channel
        colored_images = torch.stack([self.TCMNIST_images, self.TCMNIST_images], dim=2)

        for i, e in enumerate(self.ENVS):
            # Color i-th frame subset
            colored_images, colored_labels = self.color_dataset(colored_images, self.TCMNIST_labels, i, e, self.label_noise)

        # Make Tensor dataset and dataloader
        dataset = torch.utils.data.TensorDataset(colored_images, colored_labels.long())

        self.plot_samples(colored_images, colored_labels)

        in_dataset, out_dataset = make_split(dataset, flags.holdout_fraction)
        in_loader = torch.utils.data.DataLoader(in_dataset, batch_size=batch_size, shuffle=True)
        self.in_loaders.append(in_loader)
        out_loader = torch.utils.data.DataLoader(out_dataset, batch_size=64, shuffle=False)
        self.out_loaders.append(out_loader)

    def color_dataset(self, images, labels, env_id, p, d):

        # Add label noise
        labels[:,env_id] = XOR(labels[:,env_id], bernoulli(d, labels[:,env_id].shape)).long()

        # Choose colors
        colors = XOR(labels[:,env_id], bernoulli(1-p, labels[:,env_id].shape))

        # Apply colors
        for sample in range(colors.shape[0]):
            images[sample,env_id+1,colors[sample].long(),:,:] *= 0 

        return images, labels

    def get_train_loaders(self):
        loaders_ID = [[str(env)+'_in' for i, env in enumerate(self.ENVS)]]
        loaders = self.in_loaders
        return loaders_ID, loaders
    
    def get_val_loaders(self):
        loaders_ID = [[str(env)+'_out' for env in self.ENVS]]
        loaders = self.out_loaders
        return loaders_ID, loaders
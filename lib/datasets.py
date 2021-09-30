import os
import copy
import h5py
import warnings

import scipy.io
import numpy as np
from scipy import fft
import matplotlib.pyplot as plt

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

DATASETS = [
    # 1D datasets
    'Fourier_basic',
    'Spurious_Fourier',
    # Small images
    "TMNIST",
    # Small correlation shift dataset
    "TCMNIST_seq",
    "TCMNIST_step",
    ## EEG Dataset
    "PhysioNet",
]

def get_dataset_class(dataset_name):
    """ Return the dataset class with the given name.
    Taken from : https://github.com/facebookresearch/DomainBed/blob/9e864cc4057d1678765ab3ecb10ae37a4c75a840/domainbed/datasets.py#L36
    
    Args:
        dataset_name (str): Name of the dataset to get the function of. (Must be a part of the DATASETS list)
    
    Returns: 
        function: The __init__ function of the desired dataset that takes as input (  flags: parser arguments of the train.py script, 
                                                                            training_hparams: set of training hparams from hparams.py )

    Raises:
        NotImplementedError: Dataset name not found
    """
    if dataset_name not in globals() or dataset_name not in DATASETS:
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))

    return globals()[dataset_name]

def num_environments(dataset_name):
    """ Returns the number of environments of a dataset """
    return len(get_dataset_class(dataset_name).ENVS)

def get_environments(dataset_name):
    """ Returns the environments of a dataset """
    return get_dataset_class(dataset_name).ENVS

def get_setup(dataset_name):
    """ Returns the setup of a dataset """
    return get_dataset_class(dataset_name).SETUP

def XOR(a, b):
    """ Returns a XOR b (the 'Exclusive or' gate) """
    return ( a - b ).abs()

def bernoulli(p, size):
    """ Returns a tensor of 1. (True) or 0. (False) resulting from the outcome of a bernoulli random variable of parameter p.
    
    Args:
        p (float): Parameter p of the Bernoulli distribution
        size (int...): A sequence of integers defining hte shape of the output tensor
    """
    return ( torch.rand(size) < p ).float()

def make_split(dataset, holdout_fraction, seed=0, sort=False):
    """ Split a Torch TensorDataset into (1-holdout_fraction) / holdout_fraction.

    Args:
        dataset (TensorDataset): Tensor dataset that has 2 tensors -> data, targets
        holdout_fraction (float): Fraction of the dataset that is gonna be in the validation set
        seed (int, optional): seed used for the shuffling of the data before splitting. Defaults to 0.
        sort (bool, optional): If ''True'' the dataset is gonna be sorted after splitting. Defaults to False.

    Returns:
        TensorDataset: 1-holdout_fraction part of the split
        TensorDataset: holdout_fractoin part of the split
    """

    in_keys, out_keys = get_split(dataset, holdout_fraction, seed=seed, sort=sort)

    in_split = dataset[in_keys]
    out_split = dataset[out_keys]

    return torch.utils.data.TensorDataset(*in_split), torch.utils.data.TensorDataset(*out_split)

def get_split(dataset, holdout_fraction, seed=0, sort=False):
    """ Generates the keys that are used to split a Torch TensorDataset into (1-holdout_fraction) / holdout_fraction.

    Args:
        dataset (TensorDataset): TensorDataset to be split
        holdout_fraction (float): Fraction of the dataset that is gonna be in the out (validation) set
        seed (int, optional): seed used for the shuffling of the data before splitting. Defaults to 0.
        sort (bool, optional): If ''True'' the dataset is gonna be sorted after splitting. Defaults to False.

    Returns:
        list: in (1-holdout_fraction) keys of the split
        list: out (holdout_fraction) keys of the split
    """

    split = int(len(dataset)*holdout_fraction)

    keys = list(range(len(dataset)))
    np.random.RandomState(seed).shuffle(keys)
    
    in_keys = keys[split:]
    out_keys = keys[:split]
    if sort:
        in_keys.sort()
        out_keys.sort()

    return in_keys, out_keys

class Multi_Domain_Dataset:
    """ Abstract class of a multi domain dataset for OOD generalization.

    Every multi domain dataset must redefine the important attributes: SETUP, PRED_TIME, ENVS, INPUT_SIZE, OUTPUT_SIZE
    The data dimension needs to be (batch, time, *features_dim)

    TODO:
        * Make a package test that checks if every class has 'time_pred' and 'setup'
    """
    N_STEPS = 5001
    CHECKPOINT_FREQ = 100
    N_WORKERS = 8
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

    def get_class_weight(self):
        """Compute class weight for class balanced training

        Returns:
            list: list of weights of length OUTPUT_SIZE
        """
        _, train_loaders = self.get_train_loaders()

        n_labels = torch.zeros(self.OUTPUT_SIZE)

        for env_loader in train_loaders:
            labels = env_loader.dataset.tensors[1][:]
            for i in range(self.OUTPUT_SIZE):
                n_labels[i] += torch.eq(torch.as_tensor(labels), i).sum()

        weights = 1. / (n_labels*self.OUTPUT_SIZE)

        return weights

    def get_train_loaders(self):
        """Fetch all training dataloaders and their ID 

        Returns:
            list: list of string names of the data splits used for training
            list: list of dataloaders of the data splits used for training
        """
        # loaders_ID = [str(env)+'_in' for i, env in enumerate(self.ENVS) if i != self.test_env]
        # loaders = [l for i, l in enumerate(self.in_loaders) if i != self.test_env] 

        return self.train_names, self.train_loaders
    
    def get_val_loaders(self):
        """Fetch all validation/test dataloaders and their ID 

        Returns:
            list: list of string names of the data splits used for validation and test
            list: list of dataloaders of the data splits used for validation and test
        """
        # loaders_ID = [str(env)+'_out' for env in self.ENVS] + [str(env)+'_in' for i, env in enumerate(self.ENVS) if i == self.test_env]
        # loaders = self.out_loaders + [l for i, l in enumerate(self.in_loaders) if i == self.test_env]

        return self.val_names, self.val_loaders

class Fourier_basic(Multi_Domain_Dataset):
    """ Fourier_basic dataset

    A dataset of 1D sinusoid signal to classify according to their Fourier spectrum.

    No download is required as it is purely synthetic
    """
    SETUP = 'seq'
    PRED_TIME = [49]
    ENVS = ['no_spur']
    INPUT_SIZE = 1
    OUTPUT_SIZE = 2

    def __init__(self, flags, training_hparams):
        super(Fourier_basic, self).__init__()

        # Make important checks
        assert flags.test_env == None, "You are using a dataset with only a single environment, there cannot be a test environment"

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
            in_loader = torch.utils.data.DataLoader(in_dataset, batch_size=training_hparams['batch_size'], shuffle=True)
            self.in_loaders.append(in_loader)
            out_loader = torch.utils.data.DataLoader(out_dataset, batch_size=64, shuffle=False)
            self.out_loaders.append(out_loader)

class Spurious_Fourier(Multi_Domain_Dataset):
    """ Spurious_Fourier dataset

    A dataset of 1D sinusoid signal to classify according to their Fourier spectrum.
    Peaks in the fourier spectrum are added to the signal that are spuriously correlated to the label.
    Different environment have different correlation rates between the labels and the spurious peaks in the spectrum.

    No download is required as it is purely synthetic
    """
    SETUP = 'seq'
    INPUT_SIZE = 1
    OUTPUT_SIZE = 2
    PRED_TIME = [49]
    label_noise = 0.25          # Label noise
    ENVS = [0.1, 0.8, 0.9]      # Environment is a function of correlation

    def __init__(self, flags, training_hparams):
        super(Spurious_Fourier, self).__init__()

        if flags.test_env is not None:
            assert flags.test_env < len(self.ENVS), "Test environment chosen is not valid"
        else:
            warnings.warn("You don't have any test environment")

        ## Save stuff
        self.test_env = flags.test_env
        self.class_balance = training_hparams['class_balance']

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
            in_loader = torch.utils.data.DataLoader(in_dataset, batch_size=training_hparams['batch_size'], shuffle=True)
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


class TMNIST(Multi_Domain_Dataset):
    """ Temporal MNIST dataset

    Each sample is a sequence of 4 MNIST digits.
    The task is to predict at each step if the sum of the current digit and the previous one is odd or even.

    The MNIST dataset needs to be downloaded, this is automaticaly done if the dataset isn't in the given data_path
    """
    N_STEPS = 5001
    SETUP = 'seq'
    PRED_TIME = [1, 2, 3]
    INPUT_SIZE = 28*28
    OUTPUT_SIZE = 2
    ENVS = ['grey']

    # Dataset parameters
    SEQ_LEN = 4

    def __init__(self, flags, training_hparams):
        super(TMNIST, self).__init__()

        assert flags.test_env == None, "You are using a dataset with only a single environment, there cannot be a test environment"

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
            in_loader = torch.utils.data.DataLoader(in_dataset, batch_size=training_hparams['batch_size'], shuffle=True)
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
    """ Abstract class for Temporal Colored MNIST

    Each sample is a sequence of 4 MNIST digits.
    The task is to predict at each step if the sum of the current digit and the previous one is odd or even.
    Color is added to the digits that is correlated with the label of the current step.
    The formulation of which is defined in the child of this class, either sequences-wise of step-wise

    The MNIST dataset needs to be downloaded, this is automaticaly done if the dataset isn't in the given data_path
    """
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
    """ Temporal Colored MNIST Sequence

    Each sample is a sequence of 4 MNIST digits.
    The task is to predict at each step if the sum of the current digit and the previous one is odd or even.
    Color is added to the digits that is correlated with the label of the current step.

    The correlation of the color to the label is constant across sequences and whole sequences are sampled from an environmnent definition

    The MNIST dataset needs to be downloaded, this is automaticaly done if the dataset isn't in the given data_path
    """
    SETUP = 'seq'
    ENVS = [0.1, 0.8, 0.9]      # Environment is a function of correlation

    ## Dataset parameters
    label_noise = 0.25                    # Label noise

    def __init__(self, flags, training_hparams):
        super(TCMNIST_seq, self).__init__(flags)

        if flags.test_env is not None:
            assert flags.test_env < len(self.ENVS), "Test environment chosen is not valid"
        else:
            warnings.warn("You don't have any test environment")

        # Save stuff
        self.test_env = flags.test_env
        self.class_balance = training_hparams['class_balance']

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
            in_loader = torch.utils.data.DataLoader(in_dataset, batch_size=training_hparams['batch_size'], shuffle=True)
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
    """ Temporal Colored MNIST Step

    Each sample is a sequence of 4 MNIST digits.
    The task is to predict at each step if the sum of the current digit and the previous one is odd or even.
    Color is added to the digits that is correlated with the label of the current step.

    The correlation of the color to the label is varying across sequences and time steps are sampled from an environmnent definition

    This dataset has the ''test_step'' variable that discts which time step is hidden during training

    The MNIST dataset needs to be downloaded, this is automaticaly done if the dataset isn't in the given data_path
    """
    SETUP = 'step'
    ENVS = [0.1, 0.8, 0.9]  # Environment is a function of correlation

    # Dataset parameters
    label_noise = 0.25      # Label noise

    def __init__(self, flags, training_hparams):
        super(TCMNIST_step, self).__init__(flags)

        if flags.test_env is not None:
            assert flags.test_env < len(self.ENVS), "Test environment chosen is not valid"
        else:
            warnings.warn("You don't have any test environment")
        assert flags.test_step != None, "The Step setup needs a test step definition"
        assert flags.test_step in list(range(len(self.PRED_TIME))), "Chosen test_step not a Prediction Time"

        ## Save stuff
        self.test_env = flags.test_env
        self.test_step = flags.test_step
        self.class_balance = training_hparams['class_balance']

        # Define array of training environment dataloaders
        self.in_loaders, self.out_loaders = [], []          

        # Permute env/steps
        self.ENVS[self.test_step], self.ENVS[self.test_env] = self.ENVS[self.test_env], self.ENVS[self.test_step]

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
        in_loader = torch.utils.data.DataLoader(in_dataset, batch_size=training_hparams['batch_size'], shuffle=True)
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

class HDF5_dataset(Dataset):
    """ HDF5 dataset

    Container for data coming from an hdf5 file. 
    
    Good thing about this is that it imports data only when it needs to and thus saves ram space
    """
    def __init__(self, h5_path, env_id, split=None):
        self.h5_path = h5_path
        self.env_id = env_id

        self.hdf = h5py.File(self.h5_path, 'r')
        self.data = self.hdf[env_id]['data']
        self.targets = self.hdf[env_id]['labels']

        self.split = list(range(self.hdf[env_id]['data'].shape[0])) if split==None else split

    def __len__(self):
        return len(self.split)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        split_idx = self.split[idx]
        
        seq = torch.as_tensor(self.data[split_idx, ...])
        labels = torch.as_tensor(self.targets[split_idx])

        return (seq, labels)

    def close(self):
        self.hdf.close()

class PhysioNet(Multi_Domain_Dataset):
    """ PhysioNet Sleep stage dataset

    The task is to classify the sleep stage from EEG and other modalities of signals.
    The raw data comes from the CAP Sleep Database hosted on Physionet.org:  
        https://physionet.org/content/capslpdb/1.0.0/
    This dataset only uses about half of the raw dataset because of the incompatibility of some measurements.
    We use the 5 most commonly used machines in the database to create the 5 seperate environment to train on.
    The machines that were used were infered by grouping together the recording that had the same channels, and the 
    final preprocessed data only include the channels that were in common between those 5 machines.

    You can read more on the data itself and it's provenance on Physionet.org

    This dataset need to be downloaded and preprocessed. This can be done with the download.py script
    """
    N_STEPS = 5001
    SETUP = 'seq'
    PRED_TIME = [3000]
    ENVS = ['Machine0', 'Machine1', 'Machine2', 'Machine3', 'Machine4']
    INPUT_SIZE = 19
    OUTPUT_SIZE = 6
    CHECKPOINT_FREQ = 100

    def __init__(self, flags, training_hparams):
        super(PhysioNet, self).__init__()

        if flags.test_env is not None:
            assert flags.test_env < len(self.ENVS), "Test environment chosen is not valid"
        else:
            warnings.warn("You don't have any test environment")

        ## Save stuff
        self.test_env = flags.test_env
        self.class_balance = training_hparams['class_balance']

        ## Create tensor dataset and dataloader
        self.val_names, self.val_loaders = [], []
        self.train_names, self.train_loaders = [], []
        for j, e in enumerate(self.ENVS):

            # Get full environment dataset and define in/out split
            full_dataset = HDF5_dataset(os.path.join(flags.data_path, 'physionet.org/data.h5'), e)
            in_split, out_split = get_split(full_dataset, flags.holdout_fraction, sort=True)
            full_dataset.close()

            # Get in/out hdf5 dataset
            out_dataset = HDF5_dataset(os.path.join(flags.data_path, 'physionet.org/data.h5'), e, split=out_split)

            # Make training dataset/loader and append it to training containers
            if j != flags.test_env:
                in_dataset = HDF5_dataset(os.path.join(flags.data_path, 'physionet.org/data.h5'), e, split=in_split)
                in_loader = torch.utils.data.DataLoader(in_dataset, batch_size=training_hparams['batch_size'], shuffle=True)
                self.train_names.append(e + '_in')
                self.train_loaders.append(in_loader)
            
            # Make validation loaders
            fast_in_dataset = HDF5_dataset(os.path.join(flags.data_path, 'physionet.org/data.h5'), e, split=in_split)
            # fast_in_loader = torch.utils.data.DataLoader(fast_in_dataset, batch_size=64, shuffle=False, num_workers=self.N_WORKERS, pin_memory=True)
            fast_in_loader = torch.utils.data.DataLoader(fast_in_dataset, batch_size=1024, shuffle=False, num_workers=self.N_WORKERS, pin_memory=True)
            fast_out_dataset = HDF5_dataset(os.path.join(flags.data_path, 'physionet.org/data.h5'), e, split=out_split)
            # fast_out_loader = torch.utils.data.DataLoader(fast_out_dataset, batch_size=64, shuffle=False, num_workers=self.N_WORKERS, pin_memory=True)
            fast_out_loader = torch.utils.data.DataLoader(fast_out_dataset, batch_size=1024, shuffle=False, num_workers=self.N_WORKERS, pin_memory=True)

            # Append to val containers
            self.val_names.append(e + '_in')
            self.val_loaders.append(fast_in_loader)
            self.val_names.append(e + '_out')
            self.val_loaders.append(fast_out_loader)
    
    def get_class_weight(self):
        """Compute class weight for class balanced training

        Returns:
            list: list of weights of length OUTPUT_SIZE
        """
        _, train_loaders = self.get_train_loaders()

        n_labels = torch.zeros(self.OUTPUT_SIZE)

        for env_loader in train_loaders:
            labels = env_loader.dataset.targets[:]
            for i in range(self.OUTPUT_SIZE):
                n_labels[i] += torch.eq(torch.as_tensor(labels), i).sum()

        weights = 1. / (n_labels*self.OUTPUT_SIZE)

        return weights

        

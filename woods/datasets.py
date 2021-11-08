"""Defining the benchmarks for OoD generalization in time-series"""

import os
import copy
import h5py
from PIL import Image
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
    'Basic_Fourier',
    'Spurious_Fourier',
    # Small images
    "TMNIST",
    # Small correlation shift dataset
    "TCMNIST_seq",
    "TCMNIST_step",
    ## EEG Dataset
    "CAP_DB",
    "SEDFx_DB",
    "MI",
    ## Financial Dataset
    "StockVolatility",
    ## Sign Recognition
    "LSA64",
    ## Activity Recognition
    "HAR"
]

def get_dataset_class(dataset_name):
    """ Return the dataset class with the given name.

    Taken from : https://github.com/facebookresearch/DomainBed/
    
    Args:
        dataset_name (str): Name of the dataset to get the function of. (Must be a part of the DATASETS list)
    
    Returns: 
        function: The __init__ function of the desired dataset that takes as input (  flags: parser arguments of the train.py script, training_hparams: set of training hparams from hparams.py )

    Raises:
        NotImplementedError: Dataset name not found in the datasets.py globals
    """
    if dataset_name not in globals() or dataset_name not in DATASETS:
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))

    return globals()[dataset_name]

def num_environments(dataset_name):
    """ Returns the number of environments of a dataset 
    
    Args:
        dataset_name (str): Name of the dataset to get the number of environments of. (Must be a part of the DATASETS list)

    Returns:
        int: Number of environments of the dataset
    """
    return len(get_dataset_class(dataset_name).ENVS)

def get_environments(dataset_name):
    """ Returns the environments of a dataset 
    
    Args:
        dataset_name (str): Name of the dataset to get the number of environments of. (Must be a part of the DATASETS list)

    Returns:
        list: list of environments of the dataset
    """
    return get_dataset_class(dataset_name).ENVS

def get_setup(dataset_name):
    """ Returns the setup of a dataset 
    
    Args:
        dataset_name (str): Name of the dataset to get the number of environments of. (Must be a part of the DATASETS list)

    Returns:
        dict: The setup of the dataset ('seq' or 'step')
    """
    return get_dataset_class(dataset_name).SETUP

def XOR(a, b):
    """ Returns a XOR b (the 'Exclusive or' gate) 
    
    Args:
        a (bool): First input
        b (bool): Second input

    Returns:
        bool: The output of the XOR gate
    """
    return ( a - b ).abs()

def bernoulli(p, size):
    """ Returns a tensor of 1. (True) or 0. (False) resulting from the outcome of a bernoulli random variable of parameter p.
    
    Args:
        p (float): Parameter p of the Bernoulli distribution
        size (int...): A sequence of integers defining hte shape of the output tensor

    Returns:
        Tensor: Tensor of Bernoulli random variables of parameter p
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

class InfiniteSampler(torch.utils.data.Sampler):
    """ Infinite Sampler for PyTorch.

    Inspired from : https://github.com/facebookresearch/DomainBed

    Args:
        sampler (torch.utils.data.Sampler): Sampler to be used for the infinite sampling.
    """
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            for batch in self.sampler:
                yield batch

    def __len__(self):
        return len(self.sampler)

class InfiniteLoader(torch.utils.data.IterableDataset):
    """ InfiniteLoader is a torch.utils.data.IterableDataset that can be used to infinitely iterate over a finite dataset.

    Inspired from : https://github.com/facebookresearch/DomainBed

    Args:
        dataset (Dataset): Dataset to be iterated over
        batch_size (int): Batch size of the dataset
        num_workers (int, optional): Number of workers to use for the data loading. Defaults to 0.
    """
    def __init__(self, dataset, batch_size, num_workers=0):
        super(InfiniteLoader, self).__init__()

        self.dataset = dataset

        sampler = torch.utils.data.RandomSampler(dataset, replacement=True)

        batch_sampler = torch.utils.data.BatchSampler(sampler, batch_size, drop_last=True)

        self.infinite_iterator = iter(
            torch.utils.data.DataLoader(dataset, batch_sampler=InfiniteSampler(batch_sampler), num_workers=num_workers)
        )

    def __iter__(self):
        while True:
            yield next(self.infinite_iterator)
    
    def __len__(self):
        return len(self.infinite_iterator)

    

class Multi_Domain_Dataset:
    """ Abstract class of a multi domain dataset for OOD generalization.

    Every multi domain dataset must redefine the important attributes: SETUP, PRED_TIME, ENVS, INPUT_SHAPE, OUTPUT_SIZE
    The data dimension needs to be (batch, time, features_dim)

    TODO:
        * Make a package test that checks if every class has 'time_pred' and 'setup'
    """
    #:int: The number of training steps taken for this dataset
    N_STEPS = 5001
    #:int: The frequency of results update
    CHECKPOINT_FREQ = 100
    #:int: The number of workers used for fast dataloaders used for validation
    N_WORKERS = 4
    #:string: The setup of the dataset ('seq' or 'step')
    SETUP = None
    #:list: The time steps where predictions are made
    PRED_TIME = [None]
    #:list: The environments of the dataset
    ENVS = [None]
    #:int: The shape of the input (excluding batch size and time dimension)
    INPUT_SHAPE = None
    #:int: The size of the output
    OUTPUT_SIZE = None

    def __init__(self):
        pass

    def get_class_weight(self):
        """ Compute class weight for class balanced training

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
        """ Fetch all training dataloaders and their ID 

        Returns:
            list: list of string names of the data splits used for training
            list: list of dataloaders of the data splits used for training
        """
        return self.train_names, self.train_loaders
    
    def get_val_loaders(self):
        """ Fetch all validation/test dataloaders and their ID 

        Returns:
            list: list of string names of the data splits used for validation and test
            list: list of dataloaders of the data splits used for validation and test
        """
        return self.val_names, self.val_loaders
              
    def split_data(self, out, labels):
        """ Group data and prediction by environment
<<<<<<< HEAD:woods/datasets.py

        Args:
            out (Tensor): output from a model of shape ((n_env-1)*batch_size, len(PRED_TIME), output_size)
            labels (Tensor): labels of shape ((n_env-1)*batch_size, len(PRED_TIME), output_size)

=======

        Args:
            out (Tensor): output from a model of shape ((n_env-1)*batch_size, len(PRED_TIME), output_size)
            labels (Tensor): labels of shape ((n_env-1)*batch_size, len(PRED_TIME), output_size)

>>>>>>> e09f9eab727d35cef4d5017704297f7a9e555751:woods/datasets.py
        Returns:
            Tensor: The reshaped output (n_train_env, batch_size, len(PRED_TIME), output_size)
            Tensor: The labels (n_train_env, batch_size, len(PRED_TIME))
        """
        n_train_env = len(self.ENVS)-1 if self.test_env is not None else len(self.ENVS)
        out_split = torch.zeros((n_train_env, self.batch_size, *out.shape[1:])).to(out.device)
        labels_split = torch.zeros((n_train_env, self.batch_size, labels.shape[-1])).long().to(labels.device)
        all_logits_idx = 0
        for i in range(n_train_env):
            out_split[i,...] = out[all_logits_idx:all_logits_idx + self.batch_size,...]
            labels_split[i,...] = labels[all_logits_idx:all_logits_idx + self.batch_size,...]
            all_logits_idx += self.batch_size
        return out_split, labels_split

class Basic_Fourier(Multi_Domain_Dataset):
    """ Fourier_basic dataset

    A dataset of 1D sinusoid signal to classify according to their Fourier spectrum. 
    
    Args:
        flags (argparse.Namespace): argparse of training arguments
        training_hparams (dict): dictionnary of training hyper parameters coming from the hyperparams.py file

    Note:
        No download is required as it is purely synthetic
    """
    SETUP = 'seq'
    PRED_TIME = [49]
    ENVS = ['no_spur']
    INPUT_SHAPE = [1]
    OUTPUT_SIZE = 2

    def __init__(self, flags, training_hparams):
        super().__init__()

        # Make important checks
        assert flags.test_env == None, "You are using a dataset with only a single environment, there cannot be a test environment"

        # Save stuff
        self.test_env = flags.test_env
        self.batch_size = training_hparams['batch_size']
        self.class_balance = training_hparams['class_balance']

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

        ## Create the labels
        labels_0 = torch.zeros((signal_0.shape[0],1)).long()
        labels_1 = torch.ones((signal_1.shape[0],1)).long()
        labels = torch.cat((labels_0, labels_1))

        ## Create tensor dataset and dataloader
        self.train_names, self.train_loaders = [], []
        self.val_names, self.val_loaders = [], []
        for i, e in enumerate(self.ENVS):
            dataset = torch.utils.data.TensorDataset(signal, labels)
            in_dataset, out_dataset = make_split(dataset, flags.holdout_fraction)

            in_loader = torch.utils.data.DataLoader(in_dataset, batch_size=training_hparams['batch_size'], shuffle=True)
            self.train_names.append(e+'_in')
            self.train_loaders.append(in_loader)
            fast_in_loader = torch.utils.data.DataLoader(copy.deepcopy(in_dataset), batch_size=64, shuffle=False)
            self.val_names.append(e+'_in')
            self.val_loaders.append(fast_in_loader)
            fast_out_loader = torch.utils.data.DataLoader(out_dataset, batch_size=64, shuffle=False)
            self.val_names.append(e+'_out')
            self.val_loaders.append(fast_out_loader)
        
class Spurious_Fourier(Multi_Domain_Dataset):
    """ Spurious_Fourier dataset

    A dataset of 1D sinusoid signal to classify according to their Fourier spectrum.
    Peaks in the fourier spectrum are added to the signal that are spuriously correlated to the label.
    Different environment have different correlation rates between the labels and the spurious peaks in the spectrum.

    Args:
        flags (argparse.Namespace): argparse of training arguments
        training_hparams (dict): dictionnary of training hyper parameters coming from the hyperparams.py file

    Note:
        No download is required as it is purely synthetic
    """
    SETUP = 'seq'
    INPUT_SHAPE = [1]
    OUTPUT_SIZE = 2
    PRED_TIME = [49]
    #:float: Level of noise added to the labels
    LABEL_NOISE = 0.25
    #:list: The correlation rate between the label and the spurious peaks
    ENVS = [0.1, 0.8, 0.9]

    #TO REMOVE:
    N_STEPS = 2

    def __init__(self, flags, training_hparams):
        super().__init__()

        if flags.test_env is not None:
            assert flags.test_env < len(self.ENVS), "Test environment chosen is not valid"
        else:
            warnings.warn("You don't have any test environment")

        ## Save stuff
        self.test_env = flags.test_env
        self.class_balance = training_hparams['class_balance']
        self.batch_size = training_hparams['batch_size']

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

        ## Create the environments with different correlations
        env_size = 150
        self.train_names, self.train_loaders = [], []
        self.val_names, self.val_loaders = [], []
        for i, e in enumerate(self.ENVS):

            ## Create set of labels
            env_labels_0 = torch.zeros((env_size // 2, 1)).long()
            env_labels_1 = torch.ones((env_size // 2, 1)).long()
            env_labels = torch.cat((env_labels_0, env_labels_1))

            ## Fill signal
            env_signal = torch.zeros((env_size, 50))
            for j, label in enumerate(env_labels):

                # Label noise
                if bool(bernoulli(self.LABEL_NOISE, 1)):
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
            if i != self.test_env:
                in_loader = InfiniteLoader(in_dataset, batch_size=training_hparams['batch_size'], num_workers=4)
                self.train_names.append(str(e) + '_in')
                self.train_loaders.append(in_loader)

            fast_in_loader = torch.utils.data.DataLoader(copy.deepcopy(in_dataset), batch_size=64, shuffle=False)
            self.val_names.append(str(e) + '_in')
            self.val_loaders.append(fast_in_loader)
            fast_out_loader = torch.utils.data.DataLoader(out_dataset, batch_size=64, shuffle=False)
            self.val_names.append(str(e) + '_out')
            self.val_loaders.append(fast_out_loader)

class TMNIST(Multi_Domain_Dataset):
    """ Temporal MNIST dataset

    Each sample is a sequence of 4 MNIST digits.
    The task is to predict at each step if the sum of the current digit and the previous one is odd or even.

    Args:
        flags (argparse.Namespace): argparse of training arguments
        training_hparams (dict): dictionnary of training hyper parameters coming from the hyperparams.py file

    Note:
        The MNIST dataset needs to be downloaded, this is automaticaly done if the dataset isn't in the given data_path
    """
    N_STEPS = 5001
    SETUP = 'seq'
    PRED_TIME = [1, 2, 3]
    INPUT_SHAPE = [28,28]
    OUTPUT_SIZE = 2
    ENVS = ['grey']
    #:int: Length of the sequence
    SEQ_LEN = 4

    def __init__(self, flags, training_hparams):
        super().__init__()

        assert flags.test_env == None, "You are using a dataset with only a single environment, there cannot be a test environment"

        # Save stuff
        self.test_env = flags.test_env
        self.batch_size = training_hparams['batch_size']

        ## Import original MNIST data
        MNIST_tfrm = transforms.Compose([ transforms.ToTensor() ])

        # Get MNIST data
        train_ds = datasets.MNIST(flags.data_path, train=True, download=True, transform=MNIST_tfrm) 
        test_ds = datasets.MNIST(flags.data_path, train=False, download=True, transform=MNIST_tfrm) 

        # Concatenate all data and labels
        MNIST_images = torch.cat((train_ds.data.float(), test_ds.data.float()))
        MNIST_labels = torch.cat((train_ds.targets, test_ds.targets))

        # Create sequences of 3 digits
        TMNIST_images = MNIST_images.reshape(-1,self.SEQ_LEN,1,28,28)

        # With their corresponding label
        TMNIST_labels = MNIST_labels.reshape(-1,self.SEQ_LEN)

        # Assign label to the objective : Is the last number in the sequence larger than the current
        # self.train_ds.targets = ( self.train_ds.targets[:,:-1] > self.train_ds.targets[:,1:] )       # Is the previous one bigger than the current one?
        TMNIST_labels = ( TMNIST_labels[:,:-1] + TMNIST_labels[:,1:] ) % 2     # Is the sum of this one and the last one an even number?
        TMNIST_labels = TMNIST_labels.long()

        ## Create tensor dataset and dataloader
        self.train_names, self.train_loaders = [], []
        self.val_names, self.val_loaders = [], []
        for e in self.ENVS:
            # Make whole dataset and get splits
            dataset = torch.utils.data.TensorDataset(TMNIST_images, TMNIST_labels)
            in_dataset, out_dataset = make_split(dataset, flags.holdout_fraction)

            # Make the training loaders (No testing environment)
            in_loader = torch.utils.data.DataLoader(in_dataset, batch_size=training_hparams['batch_size'], shuffle=True, drop_last=True)
            self.train_names.append(str(e) + '_in')
            self.train_loaders.append(in_loader)

            # Make validation loaders
            fast_in_loader = torch.utils.data.DataLoader(in_dataset, batch_size=64, shuffle=False, num_workers=self.N_WORKERS, pin_memory=True)
            self.val_names.append(str(e) + '_in')
            self.val_loaders.append(fast_in_loader)
            fast_out_loader = torch.utils.data.DataLoader(out_dataset, batch_size=64, shuffle=False, num_workers=self.N_WORKERS, pin_memory=True)
            self.val_names.append(str(e) + '_out')
            self.val_loaders.append(fast_out_loader)

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

    Args:
        flags (argparse.Namespace): argparse of training arguments

    Note:
        The MNIST dataset needs to be downloaded, this is automaticaly done if the dataset isn't in the given data_path
    """
    N_STEPS = 5001
    PRED_TIME = [1, 2, 3]
    INPUT_SHAPE = [2,28,28]
    OUTPUT_SIZE = 2
    #:int: Length of the sequence
    SEQ_LEN = 4

    def __init__(self, flags):
        super().__init__()

        ## Import original MNIST data
        MNIST_tfrm = transforms.Compose([ transforms.ToTensor() ])

        # Get MNIST data
        train_ds = datasets.MNIST(flags.data_path, train=True, download=True, transform=MNIST_tfrm) 
        test_ds = datasets.MNIST(flags.data_path, train=False, download=True, transform=MNIST_tfrm) 

        # Concatenate all data and labels
        MNIST_images = torch.cat((train_ds.data.float(), test_ds.data.float()))
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

    def plot_samples(self, images, labels, name):

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
        plt.savefig('./assets/TCMNIST_'+ self.SETUP + '_'+name+'.pdf')
        # plt.show()

class TCMNIST_seq(TCMNIST):
    """ Temporal Colored MNIST Sequence

    Each sample is a sequence of 4 MNIST digits.
    The task is to predict at each step if the sum of the current digit and the previous one is odd or even.
    Color is added to the digits that is correlated with the label of the current step.

    The correlation of the color to the label is constant across sequences and whole sequences are sampled from an environmnent definition

    Args:
        flags (argparse.Namespace): argparse of training arguments
        training_hparams (dict): dictionnary of training hyper parameters coming from the hyperparams.py file

    Note:
        The MNIST dataset needs to be downloaded, this is automaticaly done if the dataset isn't in the given data_path
    """
    SETUP = 'seq'
    N_STEPS = 2

    ## Correlation shift parameters
    #:list: list of different correlation values between the color and the label
    ENVS = [0.1, 0.8, 0.9]
    #:float: Level of noise added to the labels
    LABEL_NOISE = 0.25

    def __init__(self, flags, training_hparams):
        super().__init__(flags)

        if flags.test_env is not None:
            assert flags.test_env < len(self.ENVS), "Test environment chosen is not valid"
        else:
            warnings.warn("You don't have any test environment")

        # Save stuff
        self.test_env = flags.test_env
        self.class_balance = training_hparams['class_balance']
        self.batch_size = training_hparams['batch_size']

        # Make the color datasets
        self.train_names, self.train_loaders = [], [] 
        self.val_names, self.val_loaders = [], [] 
        for i, e in enumerate(self.ENVS):

            # Choose data subset
            images = self.TCMNIST_images[i::len(self.ENVS)]
            labels = self.TCMNIST_labels[i::len(self.ENVS)]

            # Color subset
            colored_images, colored_labels = self.color_dataset(images, labels, e, self.LABEL_NOISE)

            # self.plot_samples(colored_images, colored_labels, str(e))

            # Make Tensor dataset and the split
            dataset = torch.utils.data.TensorDataset(colored_images, colored_labels)
            in_dataset, out_dataset = make_split(dataset, flags.holdout_fraction)

            if i != self.test_env:
                in_loader = torch.utils.data.DataLoader(in_dataset, batch_size=training_hparams['batch_size'], shuffle=True, drop_last=True)
                self.train_names.append(str(e) + '_in')
                self.train_loaders.append(in_loader)
            
            fast_in_loader = torch.utils.data.DataLoader(in_dataset, batch_size=64, shuffle=False, num_workers=self.N_WORKERS, pin_memory=True)
            self.val_names.append(str(e) + '_in')
            self.val_loaders.append(fast_in_loader)
            fast_out_loader = torch.utils.data.DataLoader(out_dataset, batch_size=64, shuffle=False, num_workers=self.N_WORKERS, pin_memory=True)
            self.val_names.append(str(e) + '_out')
            self.val_loaders.append(fast_out_loader)

    def color_dataset(self, images, labels, p, d):
        """ Color the dataset

        Args:
            images (Tensor): 3 channel images to color
            labels (Tensor): labels of the images
            p (float): correlation between the color and the label
            d (float): level of noise added to the labels

        Returns:
            colored_images (Tensor): colored images
        """

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

class TCMNIST_step(TCMNIST):
    """ Temporal Colored MNIST Step

    Each sample is a sequence of 4 MNIST digits.
    The task is to predict at each step if the sum of the current digit and the previous one is odd or even.
    Color is added to the digits that is correlated with the label of the current step.

    The correlation of the color to the label is varying across sequences and time steps are sampled from an environmnent definition.
    By definition, the test environment is always the last time step in the sequence.

    Args:
        flags (argparse.Namespace): argparse of training arguments
        training_hparams (dict): dictionnary of training hyper parameters coming from the hyperparams.py file

    Note:
        The MNIST dataset needs to be downloaded, this is automaticaly done if the dataset isn't in the given data_path
    """
    SETUP = 'step'

    # Correlation shift parameters
    #:list: list of different correlation values between the color and the label
    ENVS = [0.9, 0.8, 0.1]
    #:float: Level of noise added to the labels
    LABEL_NOISE = 0.25

    def __init__(self, flags, training_hparams):
        super(TCMNIST_step, self).__init__(flags)

        if flags.test_env is not None:
            assert flags.test_env < len(self.ENVS), "Test environment chosen is not valid"
        else:
            warnings.warn("You don't have any test environment")

        ## Save stuff
        self.test_env = flags.test_env
        self.class_balance = training_hparams['class_balance']
        self.batch_size = training_hparams['batch_size']

        # Define array of training environment dataloaders
        self.train_names, self.train_loaders = [], []
        self.val_names, self.val_loaders = [], []

        # Permute env/steps
        self.ENVS[-1], self.ENVS[self.test_env] = self.ENVS[self.test_env], self.ENVS[-1]

        ## Make the color datasets
        # Stack a second color channel
        colored_labels = self.TCMNIST_labels
        colored_images = torch.stack([self.TCMNIST_images, self.TCMNIST_images], dim=2)
        for i, e in enumerate(self.ENVS):
            # Color i-th frame subset
            colored_images, colored_labels = self.color_dataset(colored_images, colored_labels, i, e, self.LABEL_NOISE)

        for i, e in enumerate(self.ENVS):
            self.plot_samples(colored_images[i::len(self.ENVS)], colored_labels[i::len(self.ENVS)], str(e))

        # Make Tensor dataset and dataloader
        dataset = torch.utils.data.TensorDataset(colored_images, colored_labels.long())

        in_dataset, out_dataset = make_split(dataset, flags.holdout_fraction)
        in_loader = torch.utils.data.DataLoader(in_dataset, batch_size=training_hparams['batch_size'], shuffle=True, drop_last=True)
        self.train_names = [str(e)+'_in' for e in self.ENVS[:-1]]
        self.train_loaders.append(in_loader)
        fast_in_loader = torch.utils.data.DataLoader(copy.deepcopy(in_dataset), batch_size=252, shuffle=False)
        self.val_names.append([str(e)+'_in' for e in self.ENVS])
        self.val_loaders.append(fast_in_loader)
        fast_out_loader = torch.utils.data.DataLoader(out_dataset, batch_size=252, shuffle=False)
        self.val_names.append([str(e)+'_out' for e in self.ENVS])
        self.val_loaders.append(fast_out_loader)

    def color_dataset(self, images, labels, env_id, p, d):
        """ Color a single step 'env_id' of the dataset

        Args:
            images (Tensor): 3 channel images to color
            labels (Tensor): labels of the images
            env_id (int): environment id
            p (float): correlation between the color and the label
            d (float): level of noise added to the labels

        Returns:
            colored_images (Tensor): all dataset with a new step colored
        """

        # Add label noise
        labels[:,env_id] = XOR(labels[:,env_id], bernoulli(d, labels[:,env_id].shape)).long()

        # Choose colors
        colors = XOR(labels[:,env_id], bernoulli(1-p, labels[:,env_id].shape))

        # Apply colors
        for sample in range(colors.shape[0]):
            images[sample,env_id+1,colors[sample].long(),:,:] *= 0 

        return images, labels
              
    def split_data(self, out, labels):
        """ Group data and prediction by environment

        Args:
            out (Tensor): output data from a model (batch_size, len(PRED_TIME), n_classes)
            labels (Tensor): labels of the data (batch_size, len(PRED_TIME))

        Returns:
            Tensor: The reshaped data (n_env-1, batch_size, 1, n_classes)
            Tensor: The reshaped labels (n_env-1, batch_size, 1)
        """
        n_train_env = len(self.ENVS)-1 if self.test_env is not None else len(self.ENVS)
        out_split = torch.zeros((n_train_env, self.batch_size, 1, out.shape[-1])).to(out.device)
        labels_split = torch.zeros((n_train_env, self.batch_size, 1)).long().to(labels.device)
        for i in range(n_train_env):
            # Test env is always the last one
            out_split[i,...] = out[:,i,...].unsqueeze(1)
            labels_split[i,...] = labels[:,i,...].unsqueeze(1)
        return out_split, labels_split

class EEG_dataset(Dataset):
    """ HDF5 dataset for EEG data

    The HDF5 file is expected to have the following nested dict structure::

        {'env0': {'data': np.array(n_samples, time_steps, input_size), 
                  'labels': np.array(n_samples, len(PRED_TIME))},
        'env1': {'data': np.array(n_samples, time_steps, input_size), 
                 'labels': np.array(n_samples, len(PRED_TIME))}, 
        ...}

    Good thing about this is that it imports data only when it needs to and thus saves ram space

    Args:
        h5_path (str): absolute path to the hdf5 file
        env_id (int): environment id key in the hdf5 file
        split (list): list of indices of the dataset the belong to the split. If 'None', all the data is used
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
        seq = seq / seq.max()
        labels = torch.as_tensor(self.targets[split_idx])

        return (seq, labels)

    def close(self):
        """ Close the hdf5 file link """
        self.hdf.close()

class Sleep_DB(Multi_Domain_Dataset):
    """ Class for Sleep Staging datasets with their data stored in a HDF5 file
        
    Args:
        flags (argparse.Namespace): argparse of training arguments
        training_hparams (dict): dictionnary of training hyper parameters coming from the hyperparams.py file
    """
    CHECKPOINT_FREQ = 500
    SETUP = 'seq'
    PRED_TIME = [3000]
    OUTPUT_SIZE = 6
    #:str: realative path to the hdf5 file
    DATA_FILE = None

    def __init__(self, flags, training_hparams):
        super().__init__()

        # if flags.test_env is not None:
        #     assert flags.test_env < len(self.ENVS), "Test environment chosen is not valid"
        # else:
        #     warnings.warn("You don't have any test environment")

        ## Save stuff
        self.test_env = flags.test_env
        self.class_balance = training_hparams['class_balance']
        self.batch_size = training_hparams['batch_size']

        ## Create tensor dataset and dataloader
        self.val_names, self.val_loaders = [], []
        self.train_names, self.train_loaders = [], []
        for j, e in enumerate(self.ENVS):

            # Get full environment dataset and define in/out split
            full_dataset = EEG_dataset(os.path.join(flags.data_path, self.DATA_FILE), e)
            in_split, out_split = get_split(full_dataset, flags.holdout_fraction, sort=True)
            full_dataset.close()

            # Make training dataset/loader and append it to training containers
            if j not in flags.test_env: #!= flags.test_env:
                in_dataset = EEG_dataset(os.path.join(flags.data_path, self.DATA_FILE), e, split=in_split)
                in_loader = torch.utils.data.DataLoader(in_dataset, batch_size=training_hparams['batch_size'], shuffle=True)
                self.train_names.append(e + '_in')
                self.train_loaders.append(in_loader)
            
            # # Get in/out hdf5 dataset
            # out_dataset = EEG_dataset(os.path.join(flags.data_path, self.DATA_FILE), e, split=out_split)

            # Make validation loaders
            fast_in_dataset = EEG_dataset(os.path.join(flags.data_path, self.DATA_FILE), e, split=in_split)
            fast_in_loader = torch.utils.data.DataLoader(fast_in_dataset, batch_size=64, shuffle=False, num_workers=self.N_WORKERS, pin_memory=True)
            # fast_in_loader = torch.utils.data.DataLoader(fast_in_dataset, batch_size=256, shuffle=False, num_workers=self.N_WORKERS, pin_memory=True)
            fast_out_dataset = EEG_dataset(os.path.join(flags.data_path, self.DATA_FILE), e, split=out_split)
            fast_out_loader = torch.utils.data.DataLoader(fast_out_dataset, batch_size=64, shuffle=False, num_workers=self.N_WORKERS, pin_memory=True)
            # fast_out_loader = torch.utils.data.DataLoader(fast_out_dataset, batch_size=256, shuffle=False, num_workers=self.N_WORKERS, pin_memory=True)

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

class CAP_DB(Sleep_DB):
    """ CAP_DB Sleep stage dataset

    The task is to classify the sleep stage from EEG and other modalities of signals.
    This dataset only uses about half of the raw dataset because of the incompatibility of some measurements.
    We use the 5 most commonly used machines in the database to create the 5 seperate environment to train on.
    The machines that were used were infered by grouping together the recording that had the same channels, and the 
    final preprocessed data only include the channels that were in common between those 5 machines.

    You can read more on the data itself and it's provenance on Physionet.org:
        https://physionet.org/content/capslpdb/1.0.0/

    Args:
        flags (argparse.Namespace): argparse of training arguments
        training_hparams (dict): dictionnary of training hyper parameters coming from the hyperparams.py file

    Note:
        This dataset need to be downloaded and preprocessed. This can be done with the download.py script.
    """
    DATA_FILE = 'physionet.org/CAP_DB.h5'
    ENVS = ['Machine0', 'Machine1', 'Machine2', 'Machine3', 'Machine4']
    INPUT_SHAPE = [19]

    def __init__(self, flags, training_hparams):
        super().__init__(flags, training_hparams)
        
class SEDFx_DB(Sleep_DB):
    """ SEDFx_DB Sleep stage dataset

    The task is to classify the sleep stage from EEG and other modalities of signals.
    This dataset only uses about half of the raw dataset because of the incompatibility of some measurements.
    We split the dataset in 5 environments to train on, each of them containing the data taken from a given group age.

    You can read more on the data itself and it's provenance on Physionet.org:
         https://physionet.org/content/sleep-edfx/1.0.0/

    Args:
        flags (argparse.Namespace): argparse of training arguments
        training_hparams (dict): dictionnary of training hyper parameters coming from the hyperparams.py file

    Note:
        This dataset need to be downloaded and preprocessed. This can be done with the download.py script
    """
    DATA_FILE = 'physionet.org/SEDFx_DB.h5'
    ENVS = ['Age 20-40', 'Age 40-60', 'Age 60-80','Age 80-100']
    INPUT_SHAPE = [4]

    def __init__(self, flags, training_hparams):
        super().__init__(flags, training_hparams)


       
class MI(Sleep_DB):
    """ MI datasets

    The task is to classify the motor imaginary from EEG and other modalities of signals.
    The raw data comes from the three MI Databases:  
        ['Cho2017', 'PhysionetMI', 'BNCI2014001']

    You can read more on the data itself and it's provenance on: 
        http://moabb.neurotechx.com/docs/api.html#motor-imagery-datasets

    This dataset need to be downloaded and preprocessed. This can be done with the download.py script
    """

    DATA_FILE = 'MI.h5'
    ENVS = ['Cho2017', 'PhysionetMI', 'BNCI2014001', 'Lee2019_MI']
    INPUT_SIZE = 21
    OUTPUT_SIZE = 2
    N_STEPS = 20001

    def __init__(self, flags, training_hparams):
        """ Dataset constructor function

        Args:
            flags (Namespace): argparse of training arguments
            training_hparams (dict): dictionnary of training hyper parameters coming from the hyperparams.py file
        """
        super().__init__(flags, training_hparams)


class StockVolatility(Multi_Domain_Dataset):
    """ Stock Volatility Dataset

    Args:
        flags (argparse.Namespace): argparse of training arguments
        training_hparams (dict): dictionnary of training hyper parameters coming from the hyperparams.py file

    Ressources:
        * https://github.com/lukaszbanasiak/yahoo-finance
        * https://medium.com/analytics-vidhya/predicting-the-volatility-of-stock-data-56f8938ab99d
        * https://medium.com/analytics-vidhya/univariate-forecasting-for-the-volatility-of-the-stock-data-using-deep-learning-6c8a4df7edf9
    """
    N_STEPS = 5001
    SETUP = 'step'
    PRED_TIME = [3000]

    # Choisir une maniere de split en [3,10] environment
    ENVS = ['2000-2004', '2005-2009', '2010-2014', '2015-2020']
    INPUT_SHAPE = [1000000]
    OUTPUT_SIZE = 1
    CHECKPOINT_FREQ = 500

    def __init__(self, flags, training_hparams):
        super().__init__()
        pass

        # data = [Dataloader for e in self.ENVS]
        ## Pour tous les index 
            # Prendre tous les donnees de l'index
            # Faire des trucs de preprocessing si besoin
            # split en chunk d'annnee en fonction de self.ENVS
            ## Pour tous les chunks e
                # env_data = split les chunks en sequence de X jours
                # data[e].append(env_data)


class LSA64_dataset(Dataset):
    """ Video dataset for a single environment of the LSA64 data (single signer)

    Folder structure::

        data_path
            └── 001
                └─ 001
                    ├── frame000001.jpg
                    ├── ...
                    └── frame0000{n_frames}.jpg
                └─ 002
                └─ (Repetitions) ...
            └── 002
                └─ 001
                └─ 002
                └─ (Repetitions) ...
            └── 003
            └── (labels) ...

    Args:
        data_path (str): path to the folder containing the data
        n_frames (int): number of frames in each video
        transform (callable, optional): Optional transform to be applied
            on a sample.
    """
    def __init__(self, data_path, n_frames, transform=None, split=None):
        self.data_path = data_path
        self.n_frames = n_frames
        self.transform = transform
        self.targets = []

        self.folders = []
        for label in os.listdir(self.data_path):
            for rep in os.listdir(os.path.join(self.data_path, label)):
                self.folders.append(os.path.join(self.data_path, label, rep))
                self.targets.append(int(label)-1)

        self.split = list(range(len(self.folders))) if split==None else split

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.split)

    def read_images(self, selected_folder, use_transform):
        """ Read images from a folder (single video consisting of n_frames images)

        Args:
            selected_folder (str): path to the folder containing the images
            use_transform (callable): transform to apply on the images

        Returns:
            Tensor: images tensor (n_frames, 3, 224, 224)
        """
        X = []
        for i in range(self.n_frames):
            image = Image.open(os.path.join(selected_folder, 'frame_{:06d}.jpg'.format(i)))

            if use_transform is not None:
                image = use_transform(image)

            X.append(image)
        X = torch.stack(X, dim=0)

        return X

    def __getitem__(self, index):
        """ Reads an image given anindex

        Args:
            index (int): index of the video sample to get

        Returns:
            Tensor: video tensor (n_frames, 3, 224, 224)
        """
        # Select sample
        split_index = self.split[index]
        folder = self.folders[split_index]

        # Load data
        X = self.read_images(folder, self.transform)        # (input) spatial images
        y = torch.LongTensor([self.targets[split_index]])   # (labels) LongTensor are for int64 instead of FloatTensor

        return X, y

class LSA64(Multi_Domain_Dataset):
    """ LSA64: A Dataset for Argentinian Sign Language dataset

    This dataset is composed of videos of different signers.

    You can read more on the data itself and it's provenance from it's source:
        http://facundoq.github.io/datasets/lsa64/

    Args:
        flags (argparse.Namespace): argparse of training arguments
        training_hparams (dict): dictionnary of training hyper parameters coming from the hyperparams.py file

    Note:
        This dataset need to be downloaded and preprocessed. This can be done with the download.py script

    Ressources:
        * http://facundoq.github.io/datasets/lsa64/
        * http://facundoq.github.io/guides/sign_language_datasets/slr
        * https://sci-hub.mksa.top/10.1007/978-981-10-7566-7_63
        * https://github.com/hthuwal/sign-language-gesture-recognition/
    """
    N_STEPS = 5001
    SETUP = 'seq'
    PRED_TIME = [19]
    ENVS = ['001']#, '002', '003', '004']#, '005', '006', '007', '008', '009', '010']
    INPUT_SHAPE = [3, 224, 224]
    OUTPUT_SIZE = 64
    CHECKPOINT_FREQ = 100
    #:int: number of frames in each video
    N_FRAMES = 20
    #:str: path to the folder containing the data
    DATA_FOLDER = 'LSA64'

    def __init__(self, flags, training_hparams):
        super().__init__()

        if flags.test_env is not None:
            assert flags.test_env < len(self.ENVS), "Test environment chosen is not valid"
        else:
            warnings.warn("You don't have any test environment")

        ## Save stuff
        self.test_env = flags.test_env
        self.class_balance = training_hparams['class_balance']
        self.normalize = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize( mean=[0.485, 0.456, 0.406],
                                                                   std=[0.229, 0.224, 0.225])])

        ## Create tensor dataset and dataloader
        self.val_names, self.val_loaders = [], []
        self.train_names, self.train_loaders = [], []
        for j, e in enumerate(self.ENVS):

            env_path = os.path.join(flags.data_path, self.DATA_FOLDER, e)

            # Get full environment dataset and define in/out split
            full_dataset = LSA64_dataset(env_path, self.N_FRAMES, transform=self.normalize)
            in_split, out_split = get_split(full_dataset, flags.holdout_fraction, sort=True)

            # Make training dataset/loader and append it to training containers
            if j != flags.test_env:
                in_dataset = LSA64_dataset(env_path, self.N_FRAMES, transform=self.normalize, split=in_split)
                in_loader = torch.utils.data.DataLoader(in_dataset, batch_size=training_hparams['batch_size'], shuffle=True)
                self.train_names.append(e + '_in')
                self.train_loaders.append(in_loader)
            
            # Make validation loaders
            fast_in_dataset = LSA64_dataset(env_path, self.N_FRAMES, transform=self.normalize, split=in_split)
            fast_in_loader = torch.utils.data.DataLoader(fast_in_dataset, batch_size=64, shuffle=False, num_workers=self.N_WORKERS, pin_memory=True)
            # fast_in_loader = torch.utils.data.DataLoader(fast_in_dataset, batch_size=256, shuffle=False, num_workers=self.N_WORKERS, pin_memory=True)
            fast_out_dataset = LSA64_dataset(env_path, self.N_FRAMES, transform=self.normalize, split=out_split)
            fast_out_loader = torch.utils.data.DataLoader(fast_out_dataset, batch_size=64, shuffle=False, num_workers=self.N_WORKERS, pin_memory=True)
            # fast_out_loader = torch.utils.data.DataLoader(fast_out_dataset, batch_size=256, shuffle=False, num_workers=self.N_WORKERS, pin_memory=True)

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

class HAR(Multi_Domain_Dataset):
    """ Heterogeneity Acrivity Recognition Dataset (HAR)

    This dataset is composed of wearables measurements during different activities.
    The goal is to classify those activities (stand, sit, walk, bike, stairs up, stairs down).

    You can read more on the data itself and it's provenance from it's source:
        https://archive.ics.uci.edu/ml/datasets/Heterogeneity+Activity+Recognition

    Args:
        flags (argparse.Namespace): argparse of training arguments
        training_hparams (dict): dictionnary of training hyper parameters coming from the hyperparams.py file

    Note:
        This dataset need to be downloaded and preprocessed. This can be done with the download.py script

    Ressources:
        * https://archive.ics.uci.edu/ml/datasets/Heterogeneity+Activity+Recognition
        * https://dl.acm.org/doi/10.1145/2809695.2809718
    """
    N_STEPS = 5001
    SETUP = 'seq'
    PRED_TIME = [499]
    ENVS = ['nexus4', 's3', 's3mini', 'lgwatch', 'gear']
    INPUT_SHAPE = 6
    OUTPUT_SIZE = 6
    CHECKPOINT_FREQ = 100

    #:str: Path to the file containing the data
    DATA_FILE = 'HAR/HAR.h5'

    def __init__(self, flags, training_hparams):
        """ Dataset constructor function

        Args:
            flags (argparse.Namespace): argparse of training arguments
            training_hparams (dict): dictionnary of training hyper parameters coming from the hyperparams.py file
        """
        super().__init__()

        if flags.test_env is not None:
            assert flags.test_env < len(self.ENVS), "Test environment chosen is not valid"
        else:
            warnings.warn("You don't have any test environment")

        # Label definition
        self.label_dict = { 'stand': 0,
                            'sit': 1,
                            'walk': 2,
                            'bike': 3,
                            'stairsup': 4,
                            'stairsdown': 5}
        
        ## Create tensor dataset and dataloader
        self.val_names, self.val_loaders = [], []
        self.train_names, self.train_loaders = [], []
        for j, e in enumerate(self.ENVS):

            with h5py.File(os.path.join(flags.data_path, self.DATA_FILE), 'r') as f:
                # Load data
                data = torch.tensor(f[e]['data'][...])
                labels = torch.tensor(f[e]['labels'][...])

            # Get full environment dataset and define in/out split
            full_dataset = torch.utils.data.TensorDataset(data, labels)
            in_dataset, out_dataset = make_split(full_dataset, flags.holdout_fraction)

            # Make training dataset/loader and append it to training containers
            if j != flags.test_env:
                in_loader = torch.utils.data.DataLoader(in_dataset, batch_size=training_hparams['batch_size'], shuffle=True)
                self.train_names.append(e + '_in')
                self.train_loaders.append(in_loader)
            
            # Make validation loaders
            fast_in_loader = torch.utils.data.DataLoader(copy.deepcopy(in_dataset), batch_size=64, shuffle=False, num_workers=self.N_WORKERS, pin_memory=True)
            fast_out_loader = torch.utils.data.DataLoader(out_dataset, batch_size=64, shuffle=False, num_workers=self.N_WORKERS, pin_memory=True)

            # Append to val containers
            self.val_names.append(e + '_in')
            self.val_loaders.append(fast_in_loader)
            self.val_names.append(e + '_out')
            self.val_loaders.append(fast_out_loader)
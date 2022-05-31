"""Defining the benchmarks for OoD generalization in time-series"""
from aifc import Error
import os
import copy
from re import L
import re
import h5py
from PIL import Image

import pickle
import numpy as np
import pandas as pd
from scipy import fft

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

# Ignore gluonts spam of futurewarnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

## Local Import
import woods.utils as utils
import woods.download as download

DATASETS = [
    # 1D datasets
    'Basic_Fourier',
    'Spurious_Fourier',
    # Small image datasets
    "TMNIST",
    # Small correlation shift dataset
    "TCMNIST_Source",
    "TCMNIST_Time",
    ## EEG Dataset
    "CAP",
    "SEDFx",
    "PCL",
    ## Sign Recognition
    "LSA64",
    ## Activity Recognition
    "HHAR",
    ## Electricity
    "AusElectricityUnbalanced",
    "AusElectricity",
    ## Emotion recognition
    "IEMOCAPOriginal",
    "IEMOCAPUnbalanced",
    "IEMOCAP",
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

def get_sweep_envs(dataset_name):
    """ Returns the list of test environments to investigate in the hyper parameter sweep 
    
    Args:
        dataset_name (str): Name of the dataset to get the number of environments of. (Must be a part of the DATASETS list)

    Returns:
        list: List of environments to sweep across
    """
    return get_dataset_class(dataset_name).SWEEP_ENVS

def get_environments(dataset_name):
    """ Returns the environments of a dataset 
    
    Args:
        dataset_name (str): Name of the dataset to get the number of environments of. (Must be a part of the DATASETS list)

    Returns:
        list: list of environments of the dataset
    """
    return get_dataset_class(dataset_name).ENVS

def get_setup(dataset_name):
    """ Returns the setup of a dataset ('source' or 'time')
    
    Args:
        dataset_name (str): Name of the dataset to get the number of environments of. (Must be a part of the DATASETS list)

    Returns:
        str: The setup of the dataset
    """
    return get_dataset_class(dataset_name).SETUP

def get_task(dataset_name):
    """ Return the task of a dataset ('classification' or 'forecasting')
    
    Args:
        dataset_name (str): Name of the dataset to get the task of. (Must be part of the DATASETS list)
        
    Return:
        str: The task of the dataset 
    """

    return get_dataset_class(dataset_name).TASK

def get_paradigm(dataset_name):
    """ Return the paradigm of a dataset ('domain_generalization' or 'subpopulation_shift')
    
    Args:
        dataset_name (str): Name of the dataset to get the paradigm of. (Must be part of the DATASETS list)
        
    Return:
        str: The paradigm of the dataset
    """

    return get_dataset_class(dataset_name).PARADIGM

def get_domain_weights(dataset_name):
    """ Returns the relative weights of domains in a subpopulation shift dataset

    Args:
        dataset_name (str): Name of the dataset

    Returns:
        list: list of weights
    """

    assert get_setup(dataset_name) == 'subpopulation', "Only subpopulation shift have domain weights"

    return get_dataset_class(dataset_name).ENVS_WEIGHTS

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
    def __init__(self, dataset, batch_size, num_workers=0, pin_memory=False):
        super(InfiniteLoader, self).__init__()

        self.dataset = dataset

        sampler = torch.utils.data.RandomSampler(dataset, replacement=True)

        batch_sampler = torch.utils.data.BatchSampler(sampler, batch_size, drop_last=True)

        if hasattr(self.dataset, 'collate_fn'):
            self.infinite_iterator = iter(
                torch.utils.data.DataLoader(dataset, batch_sampler=InfiniteSampler(batch_sampler), num_workers=num_workers, pin_memory=pin_memory, collate_fn=self.dataset.collate_fn)
            )
        else:
            self.infinite_iterator = iter(
                torch.utils.data.DataLoader(dataset, batch_sampler=InfiniteSampler(batch_sampler), num_workers=num_workers, pin_memory=pin_memory)
            )


    def __iter__(self):
        while True:
            yield next(self.infinite_iterator)
    
    def __len__(self):
        return len(self.infinite_iterator)

class Multi_Domain_Dataset:
    """ Abstract class of a multi domain dataset for OOD generalization.

    Every multi domain dataset must redefine the important attributes: SETUP, PRED_TIME, ENVS, INPUT_SHAPE, OUTPUT_SIZE, TASK
    The data dimension needs to be (batch_size, SEQ_LEN, *INPUT_SHAPE)

    TODO:
        * Make a package test that checks if every class has 'time_pred' and 'setup'
    """
    ## Training parameters
    #:int: The number of training steps taken for this dataset
    N_STEPS = 5001
    #:int: The frequency of results update
    CHECKPOINT_FREQ = 100
    #:int: The number of workers used for fast dataloaders used for validation
    N_WORKERS = 4

    ## Dataset parameters
    #:string: The performance measure of the dataset (Usually 'acc' for classification and 'rmse' for regression/forecasting)
    PERFORMANCE_MEASURE = None
    #:string: Challenge paradigm of the dataset ('domain_generalization' and 'subpopulation_shift')
    PARADIGM = None
    #:string: The setup of the dataset ('source' for Source-domains or 'time' for time-domains)
    SETUP = None
    #:string: The type of prediction task ('classification' of 'forecasting')
    TASK = None

    ## Data parameters
    #:int: The sequence length of the dataset
    SEQ_LEN = None
    #:list: The time steps where predictions are made
    PRED_TIME = [None]
    #:int: The shape of the input (excluding batch size and time dimension)
    INPUT_SHAPE = None
    #:int: The size of the output
    OUTPUT_SIZE = None
    #:str: Path to the data
    DATA_PATH = None

    ## Domain parameters
    #:list: The environments of the dataset
    ENVS = [None]
    #:list: The environments that should be used for testing (One at a time). These will be the test environments used in the sweeps
    SWEEP_ENVS = [None]
    
    def __init__(self):
        pass

    def check_local_dataset(self, path):
        """ Checks if a local dataset is available.

        Args:
            path (str): Path to the dataset
        """

        data_path = os.path.join(path, self.DATA_PATH)
        return os.path.exists(data_path)

    def prepare_data(self, path, download=False):
        """ Prepares the dataset, download if necessary

        Args:
            path (str): Path to the dataset
            download (bool, optional): If ''True'' the dataset will be downloaded if not already available locally. Defaults to False.
        """

        if self.check_local_dataset(path):
            print('Dataset already available locally')
        else:
            print('Dataset not available locally, downloading...')
            if download:
                try:
                    self.download_fct(path, 'gdrive')
                except:
                    self.download_fct(path, 'at')
            else:
                raise ValueError('Dataset not available locally and download is set to False. Set Download = True to download it locally')
    
    def loss(self, X, Y):
        """
        Computes the loss defined by the dataset
        Args:
            X (torch.tensor): Predictions of the model. Shape (batch, time, n_classes)
            Y (torch.tensor): Targets. Shape (batch, time)
        Returns:
            torch.tensor: loss of each samples. Shape (batch, time)
        """

        # Make the predictions of shape (batch, n_classes, time) such that pytorch will get losses for all time steps
        X = X.permute(0,2,1)

        # Get log probability and compute loss 
        return self.loss_fn(self.log_prob(X), Y).mean()

    def loss_by_domain(self, X, Y, n_domains):
        """ Computes the loss for each domain and returns them in a tensor

        Args:
            X (torch.tensor): Predictions of the model. Shape (batch, time, n_classes)
            Y (torch.tensor): Targets. Shape (batch, time)
            n_domains (int): Number of domains in the batch

        Returns:
            torch.tensor: tensor containing domain-wise losses. Shape (n_domains)
        """

        # Make the tensors of shape (batch, time, n_classes)
        X, Y = self.split_tensor_by_domains(X, Y, n_domains)

        # Compute all losses
        env_losses = torch.zeros(X.shape[0]).to(X.device)
        for i, (env_x, env_y) in enumerate(zip(X, Y)):
            env_losses[i] = self.loss_fn(self.log_prob(env_x), env_y).mean()

        # Return average accross time steps and batch
        return env_losses

    def split_tensor_by_domains(self, X, Y, n_domains):
        """ Group tensor by domain for source domains datasets

        Args:
            n_domains (int): Number of domains in the batch
            tensor (torch.tensor): tensor to be split. Shape (n_domains*batch, ...)

        Returns:
            Tensor: The reshaped output (n_domains, len(all predictions), ...)
        """
        new_shape_x = (
            n_domains,
            (X.shape[0]//n_domains)*X.shape[1],
            *X.shape[2:]
        )
        new_shape_y = (
            n_domains,
            (Y.shape[0]//n_domains)*Y.shape[1],
            *Y.shape[2:]
        )
        return torch.reshape(X, new_shape_x), torch.reshape(Y, new_shape_y)

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

        weights = n_labels.max() / n_labels

        return weights

    def get_train_loaders(self):
        """ Fetch all training dataloaders and their names 

        Returns:
            list: list of string names of the data splits used for training
            list: list of dataloaders of the data splits used for training
        """
        return self.train_names, self.train_loaders
    
    def get_val_loaders(self):
        """ Fetch all validation/test dataloaders and their names 

        Returns:
            list: list of string names of the data splits used for validation and test
            list: list of dataloaders of the data splits used for validation and test
        """
        return self.val_names, self.val_loaders
              
    def get_next_batch(self):
        """ Fetch the next batch of data
        
        Returns:
            list[tuple[torch.tensor, torch.tensor]]: Batch data. List of tuple of pairs of data and labels with shape (batch, time, *INPUT_SHAPE) and (batch, time, 1) respectively
        """
        
        batch_loaders = next(self.train_loaders_iter)
        input = [(x, y) for x, y in batch_loaders]
        return (
            torch.cat([x for x,y in input]).to(self.device),
            torch.cat([y for x,y in input]).to(self.device)
        )

    def split_input(self, input):
        """ Split the input into the input and target.
        
        Args:
            batch (list[tuple[torch.tensor, torch.tensor]]): Batch data. List of tuple of pairs of data and labels with shape (batch, time, *INPUT_SHAPE) and (batch, time, 1) respectively
            
        Returns:
            torch.tensor: Input data with shape (batch, time, *INPUT_SHAPE)
            torch.tensor: Target data with shape (batch, time)
        """

        return input[0].to(self.device), input[1].to(self.device)

    def get_nb_training_domains(self):
        """ Get the number of domains in the training set
        
        Returns:
            int: Number of domains in the training set
        """

        if self.test_env is None:
            return len(self.ENVS)
        return len(self.ENVS) - 1

    def get_number_of_batches(self):
        """ Return the total number of batches in the dataset 
        
        Returns:
            int: Total number of batches in the dataset
        """
        return np.sum([len(train_l) for train_l in self.train_loaders])
    
    def get_pred_time(self, X):
        """ Get the prediction times for the current batch
        
        Args:
            input_shape (tuple): shape of the input data
            
        Returns:
            int: the prediction times
        """
        return torch.tensor(self.PRED_TIME)

###########################
## Basic_Fourier dataset ##
###########################
class Basic_Fourier(Multi_Domain_Dataset):
    """ Fourier_basic dataset

    A dataset of 1D sinusoid signal to classify according to their Fourier spectrum. 

    This is primarily a sanity check to see whether the underlying task of the Spurious_Fourier dataset is feasible under the prescribed conditions.
    
    Args:
        flags (argparse.Namespace): argparse of training arguments
        training_hparams (dict): dictionnary of training hyper parameters coming from the hyperparams.py file

    Note:
        No download is required as it is purely synthetic
    """
    ## Dataset parameters
    PERFORMANCE_MEASURE = 'acc'
    PARADIGM = 'domain_generalization'
    SETUP = 'source'
    TASK = 'classification'

    ## Data parameters
    SEQ_LEN = 50
    PRED_TIME = [49]
    INPUT_SHAPE = [1]
    OUTPUT_SIZE = 2

    ## Domain parameters
    ENVS = ['no_spur']
    SWEEP_ENVS = [None]

    def __init__(self, flags, training_hparams):
        super().__init__()

        # Make important checks
        assert flags.test_env == None, "You are using a dataset with only a single environment, there cannot be a test environment"
        assert flags.objective == 'ERM', "You are using a dataset with only one domain"

        # Save stuff
        self.device = training_hparams['device']
        self.test_env = flags.test_env
        self.batch_size = training_hparams['batch_size']
        self.class_balance = training_hparams['class_balance']

        ## Define label 0 and 1 Fourier spectrum
        self.fourier_0 = np.zeros(1000)
        self.fourier_0[900] = 1
        self.fourier_1 = np.zeros(1000)
        self.fourier_1[850] = 1

        ## Make the full time series with inverse fft
        signal_0 = fft.irfft(self.fourier_0, n=10000)
        signal_1 = fft.irfft(self.fourier_1, n=10000)
        signal_0 /= np.max(np.abs(signal_0))
        signal_1 /= np.max(np.abs(signal_1))

        ## Sample signals frames with a bunch of offsets
        all_signal_0 = torch.zeros(0,50,1)
        all_signal_1 = torch.zeros(0,50,1)
        for i in range(0, 50, 2):
            offset_signal_0 = copy.deepcopy(signal_0)[i:i-50]
            offset_signal_1 = copy.deepcopy(signal_1)[i:i-50]
            split_signal_0 = torch.tensor(offset_signal_0.reshape(-1,50,1)).float()
            split_signal_1 = torch.tensor(offset_signal_1.reshape(-1,50,1)).float()
            all_signal_0 = torch.cat((all_signal_0, split_signal_0), dim=0)
            all_signal_1 = torch.cat((all_signal_1, split_signal_1), dim=0)
        signal = torch.cat((all_signal_0, all_signal_1), dim=0)

        ## Create the labels
        labels_0 = torch.zeros((all_signal_0.shape[0],1)).long()
        labels_1 = torch.ones((all_signal_1.shape[0],1)).long()
        labels = torch.cat((labels_0, labels_1), dim=0)

        ## Create tensor dataset and dataloader
        self.train_names, self.train_loaders = [], []
        self.val_names, self.val_loaders = [], []
        for i, e in enumerate(self.ENVS):
            dataset = torch.utils.data.TensorDataset(signal, labels)
            in_dataset, out_dataset = make_split(dataset, flags.holdout_fraction, seed=utils.seed_hash(i, flags.trial_seed))

            in_loader = InfiniteLoader(in_dataset, batch_size=training_hparams['batch_size'])
            self.train_names.append(e+'_in')
            self.train_loaders.append(in_loader)
            fast_in_loader = torch.utils.data.DataLoader(copy.deepcopy(in_dataset), batch_size=1028, shuffle=False)
            self.val_names.append(e+'_in')
            self.val_loaders.append(fast_in_loader)
            fast_out_loader = torch.utils.data.DataLoader(out_dataset, batch_size=1028, shuffle=False)
            self.val_names.append(e+'_out')
            self.val_loaders.append(fast_out_loader)

        # Define loss function
        self.log_prob = nn.LogSoftmax(dim=1)
        self.loss_fn = nn.NLLLoss(weight=self.get_class_weight().to(training_hparams['device']), reduction='none')

        # Define train loaders iterable
        self.train_loaders_iter = zip(*self.train_loaders)
        
##############################
## Spurious_Fourier dataset ##
##############################
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
    ## Dataset parameters
    PERFORMANCE_MEASURE = 'acc'
    PARADIGM = 'domain_generalization'
    SETUP = 'source'
    TASK = 'classification'

    ## Data parameters
    SEQ_LEN = 50
    PRED_TIME = [49]
    INPUT_SHAPE = [1]
    OUTPUT_SIZE = 2

    ## Domain parameters
    #:float: Level of noise added to the labels
    LABEL_NOISE = 0.25
    #:list: The correlation rate between the label and the spurious peaks
    ENVS = [0.1, 0.8, 0.9]
    SWEEP_ENVS = [0]

    def __init__(self, flags, training_hparams):
        super().__init__()

        if flags.test_env is not None:
            assert flags.test_env < len(self.ENVS), "Test environment chosen is not valid"
        else:
            warnings.warn("You don't have any test environment")

        ## Save stuff
        self.device = training_hparams['device']
        self.test_env = flags.test_env
        self.class_balance = training_hparams['class_balance']
        self.batch_size = training_hparams['batch_size']

        ## Define label 0 and 1 Fourier spectrum
        self.fourier_0 = np.zeros(1000)
        self.fourier_0[900] = 1
        self.fourier_1 = np.zeros(1000)
        self.fourier_1[850] = 1

        ## Define the spurious Fourier spectrum (one direct and the inverse correlation)
        self.direct_fourier_0 = copy.deepcopy(self.fourier_0)
        self.direct_fourier_1 = copy.deepcopy(self.fourier_1)
        self.direct_fourier_0[200] = 0.5
        self.direct_fourier_1[400] = 0.5

        self.inverse_fourier_0 = copy.deepcopy(self.fourier_0)
        self.inverse_fourier_1 = copy.deepcopy(self.fourier_1)
        self.inverse_fourier_0[400] = 0.5
        self.inverse_fourier_1[200] = 0.5

        ## Create the sequences for direct and inverse
        direct_signal_0 = fft.irfft(self.direct_fourier_0, n=10000)
        direct_signal_0 = torch.tensor( direct_signal_0.reshape(-1,50,1) ).float()
        direct_signal_0 /= direct_signal_0.max()
        direct_signal_0 = self.super_sample(direct_signal_0)
        direct_signal_1 = fft.irfft(self.direct_fourier_1, n=10000)
        direct_signal_1 = torch.tensor( direct_signal_1.reshape(-1,50,1) ).float()
        direct_signal_1 /= direct_signal_1.max()
        direct_signal_1 = self.super_sample(direct_signal_1)

        perm_0 = torch.randperm(direct_signal_0.shape[0])
        direct_signal_0 = direct_signal_0[perm_0,:]
        perm_1 = torch.randperm(direct_signal_1.shape[0])
        direct_signal_1 = direct_signal_1[perm_1,:]
        direct_signal = [direct_signal_0, direct_signal_1]

        inverse_signal_0 = fft.irfft(self.inverse_fourier_0, n=10000)
        inverse_signal_0 = torch.tensor( inverse_signal_0.reshape(-1,50,1) ).float()
        inverse_signal_0 /= inverse_signal_0.max()
        inverse_signal_0 = self.super_sample(inverse_signal_0)
        inverse_signal_1 = fft.irfft(self.inverse_fourier_1, n=10000)
        inverse_signal_1 = torch.tensor( inverse_signal_1.reshape(-1,50,1) ).float()
        inverse_signal_1 /= inverse_signal_1.max()
        inverse_signal_1 = self.super_sample(inverse_signal_1)

        perm_0 = torch.randperm(inverse_signal_0.shape[0])
        inverse_signal_0 = inverse_signal_0[perm_0,:]
        perm_1 = torch.randperm(inverse_signal_1.shape[0])
        inverse_signal_1 = inverse_signal_1[perm_1,:]
        inverse_signal = [inverse_signal_0, inverse_signal_1]

        ## Create the environments with different correlations
        env_size = 4000
        self.train_names, self.train_loaders = [], []
        self.val_names, self.val_loaders = [], []
        for i, e in enumerate(self.ENVS):

            ## Create set of labels
            env_labels_0 = torch.zeros((env_size // 2, 1)).long()
            env_labels_1 = torch.ones((env_size // 2, 1)).long()
            env_labels = torch.cat((env_labels_0, env_labels_1))

            ## Fill signal
            env_signal = torch.zeros((env_size, 50, 1))
            for j, label in enumerate(env_labels):

                # Label noise
                if bool(bernoulli(self.LABEL_NOISE, 1)):
                    # Correlation to label
                    if bool(bernoulli(e, 1)):
                        env_signal[j,...] = inverse_signal[label][0,...]
                        inverse_signal[label] = inverse_signal[label][1:,...]
                    else:
                        env_signal[j,...] = direct_signal[label][0,...]
                        direct_signal[label] = direct_signal[label][1:,...]
                    
                    # Flip the label
                    env_labels[j, -1] = XOR(label, 1)
                else:
                    if bool(bernoulli(e, 1)):
                        env_signal[j,...] = direct_signal[label][0,...]
                        direct_signal[label] = direct_signal[label][1:,...]
                    else:
                        env_signal[j,...] = inverse_signal[label][0,...]
                        inverse_signal[label] = inverse_signal[label][1:,...]

            # Make Tensor dataset
            dataset = torch.utils.data.TensorDataset(env_signal, env_labels)

            in_dataset, out_dataset = make_split(dataset, flags.holdout_fraction, seed=utils.seed_hash(i, flags.trial_seed))
            if i != self.test_env:
                in_loader = InfiniteLoader(in_dataset, batch_size=training_hparams['batch_size'])
                self.train_names.append(str(e) + '_in')
                self.train_loaders.append(in_loader)

            fast_in_loader = torch.utils.data.DataLoader(copy.deepcopy(in_dataset), batch_size=4000, shuffle=False)
            self.val_names.append(str(e) + '_in')
            self.val_loaders.append(fast_in_loader)
            fast_out_loader = torch.utils.data.DataLoader(out_dataset, batch_size=4000, shuffle=False)
            self.val_names.append(str(e) + '_out')
            self.val_loaders.append(fast_out_loader)

        # Define loss function
        self.log_prob = nn.LogSoftmax(dim=1)
        self.loss_fn = nn.NLLLoss(weight=self.get_class_weight().to(training_hparams['device']), reduction='none')

        # Define train loaders iterable
        self.train_loaders_iter = zip(*self.train_loaders)

    def super_sample(self, signal):
        """ Sample signals frames with a bunch of offsets 
        
        Args:
            signal (torch.Tensor): Signal to sample
        
        Returns:
            torch.Tensor: Super sampled signal
        """
        all_signal = torch.zeros(0,50,1)
        for i in range(0, 50, 2):
            new_signal = copy.deepcopy(signal)[i:i-50]
            split_signal = new_signal.reshape(-1,50,1).clone().detach().float()
            all_signal = torch.cat((all_signal, split_signal), dim=0)
        
        return all_signal

####################
## TMNIST dataset ##
####################
class TMNIST(Multi_Domain_Dataset):
    """ Temporal MNIST dataset

    Each sample is a sequence of 4 MNIST digits.
    The task is to predict at each step if the sum of the current digit and the previous one is odd or even.


    This is primarily a sanity check to see whether the underlying task of the Temporal Colored MNIST dataset is feasible under the prescribed conditions.

    Args:
        flags (argparse.Namespace): argparse of training arguments
        training_hparams (dict): dictionnary of training hyper parameters coming from the hyperparams.py file

    Note:
        The MNIST dataset needs to be downloaded, this is automaticaly done if the dataset isn't in the given data_path
    """
    ## Training parameters
    N_STEPS = 5001

    ## Dataset parameters
    PERFORMANCE_MEASURE = 'acc'
    PARADIGM = 'domain_generalization'
    SETUP = 'source'
    TASK = 'classification'

    ## Data parameters
    SEQ_LEN = 4
    PRED_TIME = [1, 2, 3]
    INPUT_SHAPE = [1,28,28]
    OUTPUT_SIZE = 2

    ## Domain parameters
    ENVS = ['grey']
    SWEEP_ENVS = [None]

    def __init__(self, flags, training_hparams):
        super().__init__()

        assert flags.test_env == None, "You are using a dataset with only a single environment, there cannot be a test environment"

        # Save stuff
        self.device = training_hparams['device']
        self.test_env = flags.test_env
        self.batch_size = training_hparams['batch_size']

        ## Import original MNIST data
        MNIST_tfrm = transforms.Compose([transforms.ToTensor()])

        # Get MNIST data
        train_ds = datasets.MNIST(flags.data_path, train=True, download=True, transform=MNIST_tfrm) 
        test_ds = datasets.MNIST(flags.data_path, train=False, download=True, transform=MNIST_tfrm) 

        # Concatenate all data and labels
        MNIST_images = torch.cat((train_ds.data.float(), test_ds.data.float()))
        MNIST_labels = torch.cat((train_ds.targets, test_ds.targets))

        # Create sequences of 3 digits
        TMNIST_images = MNIST_images.reshape(-1,self.SEQ_LEN,1,28,28) / 255.

        # With their corresponding label
        TMNIST_labels = MNIST_labels.reshape(-1,self.SEQ_LEN)

        # Assign label to the objective : Is the last number in the sequence larger than the current
        # self.train_ds.targets = ( self.train_ds.targets[:,:-1] > self.train_ds.targets[:,1:] )       # Is the previous one bigger than the current one?
        TMNIST_labels = ( TMNIST_labels[:,:-1] + TMNIST_labels[:,1:] ) % 2     # Is the sum of this one and the last one an even number?
        TMNIST_labels = TMNIST_labels.long()

        ## Create tensor dataset and dataloader
        self.train_names, self.train_loaders = [], []
        self.val_names, self.val_loaders = [], []
        for i, e in enumerate(self.ENVS):
            # Make whole dataset and get splits
            dataset = torch.utils.data.TensorDataset(TMNIST_images, TMNIST_labels)
            in_dataset, out_dataset = make_split(dataset, flags.holdout_fraction, seed=utils.seed_hash(i, flags.trial_seed))

            # Make the training loaders (No testing environment)
            in_loader = InfiniteLoader(in_dataset, batch_size=training_hparams['batch_size'])
            self.train_names.append(str(e) + '_in')
            self.train_loaders.append(in_loader)

            # Make validation loaders
            fast_in_loader = torch.utils.data.DataLoader(in_dataset, batch_size=64, shuffle=False, num_workers=self.N_WORKERS, pin_memory=True)
            self.val_names.append(str(e) + '_in')
            self.val_loaders.append(fast_in_loader)
            fast_out_loader = torch.utils.data.DataLoader(out_dataset, batch_size=64, shuffle=False, num_workers=self.N_WORKERS, pin_memory=True)
            self.val_names.append(str(e) + '_out')
            self.val_loaders.append(fast_out_loader)
            
        # Define loss function
        self.log_prob = nn.LogSoftmax(dim=1)
        self.loss_fn = nn.NLLLoss(weight=self.get_class_weight().to(training_hparams['device']), reduction='none')

        # Define train loaders iterable
        self.train_loaders_iter = zip(*self.train_loaders)

class TCMNIST(Multi_Domain_Dataset):
    """ Abstract class for Temporal Colored MNIST

    Each sample is a sequence of 4 MNIST digits.
    The task is to predict at each step if the sum of the current digit and the previous one is odd or even.
    Color is added to the digits that is correlated with the label of the current step.
    The formulation of which is defined in the child of this class, either sequences-wise of step-wise

    This is an abstract class, it needs to be inherited by a child class of SETUP 'source' or 'time'.

    Args:
        flags (argparse.Namespace): argparse of training arguments

    Note:
        The MNIST dataset needs to be downloaded, this is automaticaly done if the dataset isn't in the given data_path
    """
    ## Training parameters
    N_STEPS = 5001

    ## Dataset parameters
    PERFORMANCE_MEASURE = 'acc'
    PARADIGM = 'domain_generalization'
    SETUP = None
    TASK = 'classification'

    ## Data parameters
    SEQ_LEN = 4
    PRED_TIME = [1, 2, 3]
    INPUT_SHAPE = [2,28,28]
    OUTPUT_SIZE = 2

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
        self.TCMNIST_images = MNIST_images.reshape(-1, self.SEQ_LEN, 28, 28) / 255.

        # With their corresponding label
        TCMNIST_labels = MNIST_labels.reshape(-1, self.SEQ_LEN)

        ########################
        ### Choose the task:
        # MNIST_labels = ( MNIST_labels[:,:-1] > MNIST_labels[:,1:] )        # Is the previous one bigger than the current one?
        TCMNIST_labels = ( TCMNIST_labels[:,:-1] + TCMNIST_labels[:,1:] ) % 2      # Is the sum of this one and the last one an even number?
        self.TCMNIST_labels = TCMNIST_labels.long()

############################
## TCMNIST_Source dataset ##
############################
class TCMNIST_Source(TCMNIST):
    """ Temporal Colored MNIST with Source-domains

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
    ## Dataset parameters
    SETUP = 'source'
    
    ## Domain parameters
    #:float: Level of noise added to the labels
    LABEL_NOISE = 0.25
    #:list: list of different correlation values between the color and the label
    ENVS = [0.1, 0.8, 0.9]
    SWEEP_ENVS = [0]

    def __init__(self, flags, training_hparams):
        super().__init__(flags)

        if flags.test_env is not None:
            assert flags.test_env < len(self.ENVS), "Test environment chosen is not valid"
        else:
            warnings.warn("You don't have any test environment")

        # Save stuff
        self.device = training_hparams['device']
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

            # Make Tensor dataset and the split
            dataset = torch.utils.data.TensorDataset(colored_images, colored_labels)
            in_dataset, out_dataset = make_split(dataset, flags.holdout_fraction, seed=utils.seed_hash(i, flags.trial_seed))

            if i != self.test_env:
                in_loader = InfiniteLoader(in_dataset, batch_size=training_hparams['batch_size'])
                self.train_names.append(str(e) + '_in')
                self.train_loaders.append(in_loader)
            
            fast_in_loader = torch.utils.data.DataLoader(in_dataset, batch_size=64, shuffle=False, num_workers=self.N_WORKERS, pin_memory=True)
            self.val_names.append(str(e) + '_in')
            self.val_loaders.append(fast_in_loader)
            fast_out_loader = torch.utils.data.DataLoader(out_dataset, batch_size=64, shuffle=False, num_workers=self.N_WORKERS, pin_memory=True)
            self.val_names.append(str(e) + '_out')
            self.val_loaders.append(fast_out_loader)

        # Define loss function
        self.log_prob = nn.LogSoftmax(dim=1)
        self.loss_fn = nn.NLLLoss(weight=self.get_class_weight().to(training_hparams['device']), reduction='none')

        # Define train loaders iterable
        self.train_loaders_iter = zip(*self.train_loaders)

    def color_dataset(self, images, labels, p, d):
        """ Color the dataset with strong color correlation with the label

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

##########################
## TCMNIST_Time dataset ##
##########################
class TCMNIST_Time(TCMNIST):
    """ Temporal Colored MNIST with Time-domains

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
    ## Dataset parameters
    SETUP = 'time'

    ## Domain parameters
    #:float: Level of noise added to the labels
    LABEL_NOISE = 0.25
    #:list: list of different correlation values between the color and the label
    ENVS = [0.9, 0.8, 0.1]
    SWEEP_ENVS = [2]

    def __init__(self, flags, training_hparams):
        super().__init__(flags)

        # Check stuff
        if flags.test_env is not None:
            assert flags.test_env < len(self.ENVS), "Test environment chosen is not valid"

        if flags.test_env == 0:
            warnings.warn("The chosen test environment is not the last in the sequence, therefore the sequence of domains will be permuted from [90%,80%,10%] to [10%,80%,90%]")
        if flags.test_env == 1:
            warnings.warn("The chosen test environment is not the last in the sequence, therefore the sequence of domains will be permuted from [90%,80%,10%] to [90%,10%,80%]")

        ## Save stuff
        self.device = training_hparams['device']
        self.test_env = flags.test_env
        self.class_balance = training_hparams['class_balance']
        self.batch_size = training_hparams['batch_size']

        # Because of time pred for this dataset, it's better we store it now in a torch tensor
        self.PRED_TIME = torch.tensor(self.PRED_TIME)

        # Define array of training environment dataloaders
        self.train_names, self.train_loaders = [], []
        self.val_names, self.val_loaders = [], []

        # Permute env/steps
        if self.test_env is not None:
            self.ENVS[-1], self.ENVS[self.test_env] = self.ENVS[self.test_env], self.ENVS[-1]

        ## Make the color datasets
        # Stack a second color channel
        colored_labels = self.TCMNIST_labels
        colored_images = torch.stack([self.TCMNIST_images, self.TCMNIST_images], dim=2)
        for i, e in enumerate(self.ENVS):
            # Color i-th frame subset
            colored_images, colored_labels = self.color_dataset(colored_images, colored_labels, i, e, self.LABEL_NOISE)

        # Make Tensor dataset and dataloader
        dataset = torch.utils.data.TensorDataset(colored_images, colored_labels.long())
        in_split, out_split = get_split(dataset, flags.holdout_fraction, seed=utils.seed_hash(0, flags.trial_seed))

        if self.test_env is not None:   # remove the last time step from the training data
            in_train_dataset = torch.utils.data.TensorDataset(colored_images[in_split,:-1,...], colored_labels.long()[in_split,:-1,...])
        else:   # Keep the last time step if there is no testing domain
            in_train_dataset = torch.utils.data.TensorDataset(colored_images[in_split,...], colored_labels.long()[in_split,...])

        in_eval_dataset = torch.utils.data.TensorDataset(colored_images[in_split,...], colored_labels.long()[in_split,...])
        out_eval_dataset = torch.utils.data.TensorDataset(colored_images[out_split,...], colored_labels.long()[out_split,...])

        # Make train dataset
        in_train_loader = InfiniteLoader(in_train_dataset, batch_size=training_hparams['batch_size'])
        self.train_names = [str(e)+'_in' for e in self.ENVS[:-1]]
        self.train_loaders.append(in_train_loader)

        fast_in_loader = torch.utils.data.DataLoader(in_eval_dataset, batch_size=252, shuffle=False)
        self.val_names.append([str(e)+'_in' for e in self.ENVS])
        self.val_loaders.append(fast_in_loader)
        fast_out_loader = torch.utils.data.DataLoader(out_eval_dataset, batch_size=252, shuffle=False)
        self.val_names.append([str(e)+'_out' for e in self.ENVS])
        self.val_loaders.append(fast_out_loader)

        # Define loss function
        self.log_prob = nn.LogSoftmax(dim=1)
        self.loss_fn = nn.NLLLoss(weight=self.get_class_weight().to(training_hparams['device']), reduction='none')

        # Define train loaders iterable
        self.train_loaders_iter = zip(*self.train_loaders)

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

    def split_tensor_by_domains(self, X, Y, n_domains):
        """ Group tensor by domain for source domains datasets

        Args:
            n_domains (int): Number of domains in the batch
            tensor (torch.tensor): tensor to be split. Shape (n_domains*batch, ...)

        Returns:
            Tensor: The reshaped output (n_domains, batch, ...)
        """
        print(X.shape, Y.shape)
        return X.transpose(0,1), Y.transpose(0,1)
              
    def get_pred_time(self, input_shape):
        """ Get the prediction times for the current batch
        
        Args:
            input_shape (tuple): shape of the input data
            
        Returns:
            int: the prediction times
        """
        return self.PRED_TIME[self.PRED_TIME < input_shape[1]]

    def get_nb_correct(self, pred, target):
        """ Time domain correct count

        Args:
            pred (Tensor): predicted labels (batch_size, len(PRED_TIME), n_classes)
            target (Tensor): target labels (batch_size, len(PRED_TIME))

        Returns:
            torch.tensor: number of correct guesses for each domains
            torch.tensor: number of guesses in each domains
        """

        pred = pred.argmax(dim=2)
        return pred.eq(target).sum(dim=0), torch.ones_like(pred).sum(dim=0)

class H5_dataset(Dataset):
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
        """ Number of samples in the dataset """
        return len(self.split)

    def __getitem__(self, idx):
        """ Get a sample from the dataset """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        split_idx = self.split[idx]
        
        seq = torch.as_tensor(self.data[split_idx, ...])
        labels = torch.as_tensor(self.targets[split_idx])

        return (seq, labels)

    def close(self):
        """ Close the hdf5 file link """    
        self.hdf.close()

class EEG_DB(Multi_Domain_Dataset):
    """ Class for Sleep Staging datasets with their data stored in a HDF5 file
        
    Args:
        flags (argparse.Namespace): argparse of training arguments
        training_hparams (dict): dictionnary of training hyper parameters coming from the hyperparams.py file
    """
    ## Training parameters
    CHECKPOINT_FREQ = 500

    ## Dataset parameters
    PERFORMANCE_MEASURE = 'acc'
    PARADIGM = 'domain_generalization'
    SETUP = 'source'

    ## Data parameters
    #:str: path to the hdf5 file
    DATA_PATH = None

    def __init__(self, flags, training_hparams):
        super().__init__()

        if flags.test_env is not None:
            assert flags.test_env < len(self.ENVS), "Test environment chosen is not valid"
        else:
            warnings.warn("You don't have any test environment")

        ## Save stuff
        self.device = training_hparams['device']
        self.test_env = flags.test_env
        self.class_balance = training_hparams['class_balance']
        self.batch_size = training_hparams['batch_size']

        ## Prepare the data (Download if needed)
        self.prepare_data(flags.data_path, flags.download)

        ## Create tensor dataset and dataloader
        self.val_names, self.val_loaders = [], []
        self.train_names, self.train_loaders = [], []
        for j, e in enumerate(self.ENVS):

            # Get full environment dataset and define in/out split
            full_dataset = H5_dataset(os.path.join(flags.data_path, self.DATA_PATH), e)
            in_split, out_split = get_split(full_dataset, flags.holdout_fraction, seed=utils.seed_hash(j, flags.trial_seed))
            full_dataset.close()

            # Make training dataset/loader and append it to training containers
            if j != flags.test_env:
                in_dataset = H5_dataset(os.path.join(flags.data_path, self.DATA_PATH), e, split=in_split)
                in_loader = InfiniteLoader(in_dataset, batch_size=training_hparams['batch_size'], num_workers=self.N_WORKERS, pin_memory=True)
                self.train_names.append(e + '_in')
                self.train_loaders.append(in_loader)
            
            # Make validation loaders 
            # (You can comment the 256 batch size and uncomment the 64 batch size if you do not have enough GPU RAM, it will not change the results because this is for evaluation purposes)
            fast_in_dataset = H5_dataset(os.path.join(flags.data_path, self.DATA_PATH), e, split=in_split)
            # fast_in_loader = torch.utils.data.DataLoader(fast_in_dataset, batch_size=64, shuffle=False, num_workers=self.N_WORKERS, pin_memory=True)
            fast_in_loader = torch.utils.data.DataLoader(fast_in_dataset, batch_size=256, shuffle=False, num_workers=self.N_WORKERS, pin_memory=True)
            fast_out_dataset = H5_dataset(os.path.join(flags.data_path, self.DATA_PATH), e, split=out_split)
            # fast_out_loader = torch.utils.data.DataLoader(fast_out_dataset, batch_size=64, shuffle=False, num_workers=self.N_WORKERS, pin_memory=True)
            fast_out_loader = torch.utils.data.DataLoader(fast_out_dataset, batch_size=256, shuffle=False, num_workers=self.N_WORKERS, pin_memory=True)

            # Append to val containers
            self.val_names.append(e + '_in')
            self.val_loaders.append(fast_in_loader)
            self.val_names.append(e + '_out')
            self.val_loaders.append(fast_out_loader)

        # Define loss function
        self.log_prob = nn.LogSoftmax(dim=1)
        self.loss_fn = nn.NLLLoss(weight=self.get_class_weight().to(training_hparams['device']), reduction='none')

        # Define train loaders iterable
        self.train_loaders_iter = zip(*self.train_loaders)

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

        weights = n_labels.max() / n_labels

        return weights

#################
## CAP dataset ##
#################
class CAP(EEG_DB):
    """ CAP Sleep stage dataset

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
    ## Training parameters
    N_STEPS = 5001

    ## Dataset parameters
    TASK = 'classification'

    ## Data parameters
    SEQ_LEN = 3000
    PRED_TIME = [2999]
    INPUT_SHAPE = [19]
    OUTPUT_SIZE = 6

    ## Dataset paths
    DATA_PATH = 'CAP/CAP.h5'

    ## Domain parameters
    ENVS = ['Machine0', 'Machine1', 'Machine2', 'Machine3', 'Machine4']
    SWEEP_ENVS = list(range(len(ENVS)))

    def __init__(self, flags, training_hparams):

        ## Define download function
        self.download_fct = download.download_cap

        super().__init__(flags, training_hparams)
        
###################
## SEDFx dataset ##
###################
class SEDFx(EEG_DB):
    """ SEDFx Sleep stage dataset

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
    ## Training parameters
    N_STEPS = 5001
    
    ## Dataset parameters
    TASK = 'classification'

    ## Data parameters
    SEQ_LEN = 3000
    PRED_TIME = [2999]
    INPUT_SHAPE = [4]
    OUTPUT_SIZE = 6
    DATA_PATH = 'SEDFx/SEDFx.h5'

    ## Domain parameters
    ENVS = ['Age 20-40', 'Age 40-60', 'Age 60-80','Age 80-100']
    SWEEP_ENVS = list(range(len(ENVS)))

    def __init__(self, flags, training_hparams):

        ## Define download function
        self.download_fct = download.download_sedfx

        super().__init__(flags, training_hparams)
       
#################
## PCL dataset ##
#################
class PCL(EEG_DB):
    """ PCL datasets

    The task is to classify the motor imaginary from EEG and other modalities of signals.
    The raw data comes from the three PCL Databases:  
       [ 'PhysionetMI', 'Cho2017', 'Lee2019_MI']

    You can read more on the data itself and it's provenance on: 

        http://moabb.neurotechx.com/docs/index.html

    This dataset need to be downloaded and preprocessed. This can be done with the download.py script
    """
    ## Training parameters
    N_STEPS = 10001

    ## Dataset parameters
    TASK = 'classification'

    ## Data parameters
    SEQ_LEN = 752
    PRED_TIME = [751]
    INPUT_SHAPE = [48]
    OUTPUT_SIZE = 2
    DATA_PATH = 'PCL/PCL.h5'

    ## Domain parameters
    ENVS = [ 'PhysionetMI', 'Cho2017', 'Lee2019_MI']
    SWEEP_ENVS = list(range(len(ENVS)))

    def __init__(self, flags, training_hparams):

        ## Define download function
        self.download_fct = download.download_pcl

        super().__init__(flags, training_hparams)


###################
## LSA64 dataset ##
###################
class Video_dataset(Dataset):
    """ Video dataset

    Folder structure::

        data_path
            └── 001
                └─ 001
                    ├── frame000001.jpg
                    ├── ...
                    └── frame0000{n_frames}.jpg
                └─ 002
                └─ (samples) ...
            └── 002
                └─ 001
                └─ 002
                └─ (samples) ...
            └── 003
            └── (labels) ...

    Args:
        data_path (str): path to the folder containing the data
        n_frames (int): number of frames in each video
        transform (callable, optional): Optional transform to be applied
            on a sample.
    """
    def __init__(self, data_paths, n_frames, transform=None, split=None):
        self.data_paths = data_paths
        self.n_frames = n_frames
        self.transform = transform
        self.targets = []

        self.folders = []
        for speaker in self.data_paths:
            for label in os.listdir(speaker):
                for rep in os.listdir(os.path.join(speaker, label)):
                    self.folders.append(os.path.join(speaker, label, rep))
                    self.targets.append(int(label)-1)

        self.split = list(range(len(self.folders))) if split==None else split

    def __len__(self):
        """ Returns the number of samples in the dataset """
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
    ## Training parameters
    N_STEPS = 5001
    CHECKPOINT_FREQ = 500

    ## Dataset parameters
    PERFORMANCE_MEASURE = 'acc'
    PARADIGM = 'domain_generalization'
    SETUP = 'source'
    TASK = 'classification'

    ## Data parameters
    #:int: number of frames in each video
    SEQ_LEN = 20
    PRED_TIME = [19]
    INPUT_SHAPE = [3, 224, 224]
    OUTPUT_SIZE = 64
    #:str: path to the folder containing the data
    DATA_PATH = 'LSA64'

    ## Domain parameters
    ENVS = ['001-002', '003-004', '005-006', '007-008', '009-010']
    SWEEP_ENVS = list(range(len(ENVS)))

    def __init__(self, flags, training_hparams):
        super().__init__()

        if flags.test_env is not None:
            assert flags.test_env < len(self.ENVS), "Test environment chosen is not valid"
        else:
            warnings.warn("You don't have any test environment")

        ## Save stuff
        self.device = training_hparams['device']
        self.test_env = flags.test_env
        self.class_balance = training_hparams['class_balance']
        self.batch_size = training_hparams['batch_size']
        self.normalize = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize( mean=[0.485, 0.456, 0.406],
                                                                   std=[0.229, 0.224, 0.225])])

        ## Prepare the data (Download if needed)
        self.download_fct = download.download_lsa64
        self.prepare_data(flags.data_path, flags.download)

        ## Create tensor dataset and dataloader
        self.val_names, self.val_loaders = [], []
        self.train_names, self.train_loaders = [], []
        for j, e in enumerate(self.ENVS):

            env_paths = []
            for speaker in e.split('-'):
                env_paths.append(os.path.join(flags.data_path, self.DATA_PATH, speaker))
            full_dataset = Video_dataset(env_paths, self.SEQ_LEN, transform=self.normalize)
            in_split, out_split = get_split(full_dataset, flags.holdout_fraction, seed=utils.seed_hash(j, flags.trial_seed))

            # Make training dataset/loader and append it to training containers
            if j != flags.test_env:
                in_dataset = Video_dataset(env_paths, self.SEQ_LEN, transform=self.normalize, split=in_split)
                in_loader = InfiniteLoader(in_dataset, batch_size=training_hparams['batch_size'], num_workers=self.N_WORKERS, pin_memory=True)
                self.train_names.append(e + '_in')
                self.train_loaders.append(in_loader)

            # Make validation loaders
            # (You can comment the 64 batch size and uncomment the 16 batch size if you do not have enough GPU RAM, it will not change the results because this is for evaluation purposes)
            fast_in_dataset = Video_dataset(env_paths, self.SEQ_LEN, transform=self.normalize, split=in_split)
            # fast_in_loader = torch.utils.data.DataLoader(fast_in_dataset, batch_size=16, shuffle=False, num_workers=self.N_WORKERS, pin_memory=True)
            fast_in_loader = torch.utils.data.DataLoader(fast_in_dataset, batch_size=64, shuffle=False, num_workers=self.N_WORKERS, pin_memory=True)
            fast_out_dataset = Video_dataset(env_paths, self.SEQ_LEN, transform=self.normalize, split=out_split)
            # fast_out_loader = torch.utils.data.DataLoader(fast_out_dataset, batch_size=16, shuffle=False, num_workers=self.N_WORKERS, pin_memory=True)
            fast_out_loader = torch.utils.data.DataLoader(fast_out_dataset, batch_size=64, shuffle=False, num_workers=self.N_WORKERS, pin_memory=True)

            # Append to val containers
            self.val_names.append(e + '_in')
            self.val_loaders.append(fast_in_loader)
            self.val_names.append(e + '_out')
            self.val_loaders.append(fast_out_loader)

        # Define loss function
        self.log_prob = nn.LogSoftmax(dim=1)
        self.loss_fn = nn.NLLLoss(weight=self.get_class_weight().to(training_hparams['device']), reduction='none')

        # Define train loaders iterable
        self.train_loaders_iter = zip(*self.train_loaders)

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

        weights = n_labels.max() / n_labels

        return weights

##################
## HHAR dataset ##
##################
class HHAR(Multi_Domain_Dataset):
    """ Heterogeneity Acrivity Recognition Dataset (HHAR)

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
    ## Training parameters
    N_STEPS = 5001
    CHECKPOINT_FREQ = 100

    ## Dataset parameters
    PERFORMANCE_MEASURE = 'acc'
    PARADIGM = 'domain_generalization'
    SETUP = 'source'
    TASK = 'classification'

    ## Data parameters
    SEQ_LEN = 500
    PRED_TIME = [499]
    INPUT_SHAPE = [6]
    OUTPUT_SIZE = 6
    #:str: Path to the file containing the data
    DATA_PATH = 'HHAR/HHAR.h5'

    ## Domain parameters
    ENVS = ['nexus4', 's3', 's3mini', 'lgwatch', 'gear']
    SWEEP_ENVS = list(range(len(ENVS)))

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

        # Save stuff 
        self.device = training_hparams['device']
        self.test_env = flags.test_env
        self.batch_size = training_hparams['batch_size']

        ## Prepare the data (Download if needed)
        self.download_fct = download.download_hhar
        self.prepare_data(flags.data_path, flags.download)

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

            with h5py.File(os.path.join(flags.data_path, self.DATA_PATH), 'r') as f:
                # Load data
                data = torch.tensor(f[e]['data'][...])
                labels = torch.tensor(f[e]['labels'][...])

            # Get full environment dataset and define in/out split
            full_dataset = torch.utils.data.TensorDataset(data, labels)
            in_dataset, out_dataset = make_split(full_dataset, flags.holdout_fraction, seed=utils.seed_hash(j, flags.trial_seed))

            # Make training dataset/loader and append it to training containers
            if j != flags.test_env:
                in_loader = InfiniteLoader(in_dataset, batch_size=training_hparams['batch_size'], num_workers=self.N_WORKERS, pin_memory=True)
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

        # Define loss function
        self.log_prob = nn.LogSoftmax(dim=1)
        self.loss_fn = nn.NLLLoss(weight=self.get_class_weight().to(training_hparams['device']), reduction='none')
        self.train_loaders_iter = zip(*self.train_loaders)

#############################
## Aus Electricity dataset ##
#############################
from typing import NamedTuple, Optional, Iterable, Dict, Any, List, Tuple

import holidays
import datetime
from datasets import load_dataset

from gluonts.torch.modules.loss import DistributionLoss, NegativeLogLikelihood
from gluonts.torch.modules.distribution_output import (
    DistributionOutput,
    StudentTOutput,
)

from gluonts.time_feature import get_lags_for_frequency
from gluonts.time_feature import (
    TimeFeature,
    time_features_from_frequency_str,
)
from gluonts.transform import (
    Transformation,
    Chain,
    RemoveFields,
    SetField,
    AsNumpyArray,
    AddObservedValuesIndicator,
    AddTimeFeatures,
    AddAgeFeature,
    VstackFeatures,
    InstanceSplitter,
    ValidationSplitSampler,
    TestSplitSampler,
    ExpectedNumInstanceSampler,
    SelectFields,
    InstanceSampler,
)

from gluonts.dataset.common import Dataset, ListDataset
from gluonts.itertools import Cyclic, PseudoShuffled, IterableSlice
from gluonts.dataset.field_names import FieldName
from gluonts.core.component import validated
from gluonts.torch.util import (
    IterableDataset,
)

class training_domain_sampler(InstanceSampler):
    """
    Training sampler for forecasting dataset.

    Using time series data, this sampler will choose a random time series window to be used for training.
    The choosing of the time window is done by sampling a random time point from the time series and taking time points prior as context and subsequent time points for prediction.
    """

    axis: int = -1
    min_past: int = 0
    min_future: int = 0

    domain_idx: list = []
    month_idx: dict = {
        'January': 1,
        'February': 2,
        'March': 3,
        'April': 4,
        'May': 5,
        'June': 6,
        'July': 7,
        'August': 8,
        'September': 9,
        'October': 10,
        'November': 11,
        'December': 12
    }

    num_instances: float
    total_length: int = 0
    n: int = 0

    def set_attribute(self, domain, set_holidays, start, max_length=0):
        """ Set the attributes of the sampler.

        This function needs to be called before the sampler can be used.
        It defines the range of the time series to be used for training.
        We define the range of the time series according to which domain we are in (e.g. 'January' domain will only sample data points in January)

        Args:
            domain (str): The domain to be used for training.
            set_holidays (dict): Dictionary containing the holiday definitions.
            start (int): The start of the time series.
            max_length (int): The longuest length of the time series dataset.
        
        Returns:
            None
        """
        
        min_increment = datetime.timedelta(minutes=30)
        day_increment = datetime.timedelta(days=1)
        running_time = start

        # We can use all time points, disregarding the domains
        if domain == 'All':
            holidays_idx = []
            for idx in range(max_length):
                running_time += min_increment
                holidays_idx.append(idx)

            self.domain_idx = holidays_idx

        # Defining the holidays indexes
        elif domain == 'Holidays':
            holidays_idx = []
            for idx in range(max_length):
                running_time += min_increment
                if running_time in set_holidays:# or running_time + day_increment in set_holidays:
                    holidays_idx.append(idx)

            self.domain_idx = holidays_idx

        
        # Defining the month domain index ranges
        elif domain != 'Holidays':
            month_ID = self.month_idx[domain]
            non_holidays_idx = []
            for idx in range(max_length):
                running_time += min_increment
                if running_time not in set_holidays and running_time.month == month_ID:
                    non_holidays_idx.append(idx)

            self.domain_idx = non_holidays_idx

        else:
            raise ValueError('The domain you provided is not valid')

    def _get_bounds(self, ts: np.ndarray) -> Tuple[int, int]:
        """ Get the bounds of the time series taking into consideration the context length and the future length.

        Args:   
            ts (np.ndarray): The time series.

        Returns:
            Tuple[int, int]: The bounds of the time series.
        """
        return (
            self.min_past,
            ts.shape[self.axis] - self.min_future,
        )

    def __call__(self, ts: np.ndarray) -> np.ndarray:
        """ Randomly sample a time window in a time series.

        Args:   
            ts (np.ndarray): The time series to sample in.

        Returns:
            int: the start index of the time window.
        """
        a, b = self._get_bounds(ts)
        in_range_idx = np.array(self.domain_idx)
        in_range_idx = in_range_idx[in_range_idx > a]
        in_range_idx = in_range_idx[in_range_idx < b]
        window_size = len(in_range_idx)

        if window_size <= 0:
            return np.array([], dtype=int)

        self.n += 1
        self.total_length += window_size
        avg_length = self.total_length / self.n

        if avg_length <= 0:
            return np.array([], dtype=int)

        p = self.num_instances / avg_length
        (indices,) = np.where(np.random.random_sample(window_size) < p)
        return in_range_idx[indices] + a

class evaluation_domain_sampler(InstanceSampler):

    axis: int = -1
    min_past: int = 0
    min_future: int = 0

    domain_idx: list = []
    
    num_instances: float
    total_length: int = 0
    n: int = 0

    start_idx: int
    last_idx: int # This is excluded

    month_idx: dict = {
        'January': 1,
        'February': 2,
        'March': 3,
        'April': 4,
        'May': 5,
        'June': 6,
        'July': 7,
        'August': 8,
        'September': 9,
        'October': 10,
        'November': 11,
        'December': 12
    }

    def set_attribute(self, domain, set_holidays, start, max_length=0):
        """ Set the attributes of the sampler.

        This function needs to be called before the sampler can be used.
        It defines all indexes to evaluate in the time series for a given domain.

        Args:
            domain (str): The domain to be evaluated.
            set_holidays (dict): Dictionary containing the holiday definitions.
            start (int): The start of the time series.
            max_length (int): The maximum length of the time series.
        
        Returns:
            None
        """
        
        min_increment = datetime.timedelta(minutes=30)
        day_increment = datetime.timedelta(days=1)
        running_time = start

        # Get all indexes at the start of the days of a holiday
        if domain == 'Holidays':
            holidays_idx = []
            for idx in range(int(max_length / 48)):
                running_time += day_increment
                if running_time in set_holidays:# or running_time + day_increment in set_holidays:
                    holidays_idx.append(idx * 48)

            self.domain_idx = holidays_idx
        
        # Get all indexes at the start of the days of a given month
        if domain != 'Holidays':
            domain_idx = self.month_idx[domain]
            non_holidays_idx = []
            for idx in range(int(max_length / 48)):
                running_time += day_increment
                if running_time not in set_holidays and running_time.month == domain_idx:
                    non_holidays_idx.append(idx * 48)

            self.domain_idx = non_holidays_idx

        self.domain_idx = np.array(self.domain_idx)
        self.domain_idx = self.domain_idx[self.domain_idx >= self.start_idx]
        self.domain_idx = self.domain_idx[self.domain_idx < self.last_idx]

    def _get_bounds(self, ts: np.ndarray) -> Tuple[int, int]:
        """ Get the bounds of the time series taking into consideration the context length and the future length.

        Args:   
            ts (np.ndarray): The time series.

        Returns:
            Tuple[int, int]: The bounds of the time series.
        """
        return (
            self.min_past,
            ts.shape[self.axis] - self.min_future,
        )

    def __call__(self, ts: np.ndarray) -> np.ndarray:
        """ Returns all window indexes for a domain for evaluation.

        Args:   
            ts (np.ndarray): The time series to evaluate.

        Returns:
            int: the start index of the time windows.
        """
        a, b = self._get_bounds(ts)
        in_range_idx = self.domain_idx
        in_range_idx = in_range_idx[in_range_idx > a]
        in_range_idx = in_range_idx[in_range_idx <= b]
        window_size = len(in_range_idx)

        if window_size <= 0:
            return np.array([], dtype=int)

        return in_range_idx + a

class AusElectricityUnbalanced(Multi_Domain_Dataset):

    # Training parameters
    N_STEPS = 3001
    CHECKPOINT_FREQ = 250

    ## Dataset parameters
    PERFORMANCE_MEASURE = 'rmse'
    PARADIGM = 'subpopulation_shift'
    SETUP = 'time'
    TASK = 'forecasting'

    # Data parameters
    SEQ_LEN = 500
    INPUT_SIZE = 1
    OUTPUT_SIZE = 1
    FREQUENCY = '30T'
    PRED_LENGTH = 48

    ## Domain parameters
    ENVS = ['Holidays', 'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    SWEEP_ENVS = [-1]
    ENVS_WEIGHTS = [8./365, 28./365, 28./365., 31./365., 27./365., 31./365., 30./365., 31./365., 31./365., 30./365., 31./365., 30./365., 29./365.]

    ## Data field identifiers
    PREDICTION_INPUT_NAMES = [
        "feat_static_cat",
        "feat_static_real",
        "past_time_feat",
        "past_target",
        "past_observed_values",
        "future_time_feat",
    ]

    TRAINING_INPUT_NAMES = PREDICTION_INPUT_NAMES + [
        "future_target",
        "future_observed_values",
    ]

    def __init__(self, flags, training_hparams):
        super().__init__()

        # Check if the objective is ERM: This is a single domain dataset, therefore invariance penalty cannot be applied
        # Eventually we could investigate the use of penalties that doesn't use domain definitions such as SD
        assert flags.objective == 'ERM', "You are using a dataset with only one domain"

        # Domain property
        self.set_holidays = holidays.country_holidays('AU')

        # Data property
        self.num_feat_static_cat = 0
        self.num_feat_dynamic_real = 0
        self.num_feat_static_real = 0
        self.start_datetime = datetime.datetime(2002,1,1,0,0)
        self.max_ts_length = 240000  #Rounded up to the tens of thousand, just cause idk

        ## Task information
        # Forcasting models output parameters of a distribution
        self.distr_output = StudentTOutput()
        # Covariate information given to the model alongside the target time series
        self.time_features = time_features_from_frequency_str(self.FREQUENCY)   # Transformed time information in the shape of a vector [f(minute), f(hour), f(day), ...]
        self.lags_seq = get_lags_for_frequency(self.FREQUENCY)  # Past targets from the target time series fed alongside the current time e.g., input_target=[target[0], target[-10], target[-100], ...]
        # Context info
        self.context_length = 7*self.PRED_LENGTH
        self._past_length = self.context_length + max(self.lags_seq)

        # Training parameters
        self.device = training_hparams['device']
        self.batch_size = training_hparams['batch_size']
        self.num_batches_per_epoch = 100

        # Get dataset
        self.raw_data = load_dataset('monash_tsf','australian_electricity_demand')

        # Define training / validation / test split
        time_pt_per_year = 365 * 24 * 2
        train_first_idx = 175296 # Only for evaluation
        train_last_idx = 192864
        val_first_idx = train_last_idx
        val_last_idx = 210384
        test_first_idx = val_last_idx
        test_last_idx = 227904

        # Create ListDatasets
        train_dataset = ListDataset(
            [
                {  
                    FieldName.TARGET: tgt[:train_last_idx],
                    FieldName.START: strt
                } for (tgt, strt) in zip(
                    self.raw_data['test']['target'], 
                    self.raw_data['test']['start'])
            ],
            freq=self.FREQUENCY
        )
        validation_dataset = ListDataset(
            [
                {
                    FieldName.TARGET: tgt[:val_last_idx],
                    FieldName.START: strt
                } for (tgt, strt) in zip(
                    self.raw_data['test']['target'], 
                    self.raw_data['test']['start'])
            ], freq=self.FREQUENCY
        )
        test_dataset = ListDataset(
            [
                {
                    FieldName.TARGET: tgt,
                    FieldName.START: strt
                } for (tgt, strt) in zip(
                    self.raw_data['test']['target'], 
                    self.raw_data['test']['start'])
            ], freq=self.FREQUENCY
        )

        # Define embedding (this is kind of useless because this dataset doesn't have categorical covariate features, 
        # but it might be useful as template for other forecasting datasets)
        self.cardinality = [5]
        self.embedding_dimension = [5]

        # Create transformation
        self.transform = self.create_transformation()
        train_transformed = self.transform.apply(train_dataset, is_train=True)
        validation_transformed = self.transform.apply(validation_dataset, is_train=False)
        test_transformed = self.transform.apply(test_dataset, is_train=False)

        ## Create tensor dataset and dataloader
        self.train_names, self.train_loaders = [], []
        training_dataloader = self.create_training_data_loader(
            train_transformed,
            domain='All',
            training_hparams=training_hparams,
            shuffle_buffer_length=0,
            num_workers=self.N_WORKERS
        )
        self.train_names.append("All_train")
        self.train_loaders.append(training_dataloader)

        self.val_names, self.val_loaders = [], []
        for j, e in enumerate(self.ENVS):

            training_evaluation_dataloader = self.create_evaluation_data_loader(
                train_transformed,
                domain=e,
                training_hparams=training_hparams,
                start_idx=train_first_idx,
                last_idx=train_last_idx,
                num_workers=0
            )
            self.val_names.append(e+"_train")
            self.val_loaders.append(training_evaluation_dataloader)

            validation_dataloader = self.create_evaluation_data_loader(
                validation_transformed,
                domain=e,
                training_hparams=training_hparams,
                start_idx=val_first_idx,
                last_idx=val_last_idx,
                num_workers=0
            )
            self.val_names.append(e+"_val")
            self.val_loaders.append(validation_dataloader)

            test_dataloader = self.create_evaluation_data_loader(
                test_transformed,
                domain=e,
                training_hparams=training_hparams,
                start_idx=test_first_idx,
                last_idx=test_last_idx,
                num_workers=0
            )
            self.val_names.append(e+"_test")
            self.val_loaders.append(test_dataloader)

        # Define loss function
        self.loss_fn = NegativeLogLikelihood()

        # Define data loaders iterable
        self.train_loaders_iter = zip(*self.train_loaders)

    def get_nb_training_domains(self):
        """ Get the number of domains in the training set
        
        Returns:
            int: Number of domains in the training set
        """

        return 1

    def create_transformation(self) -> Transformation:
            remove_field_names = []
            if self.num_feat_static_real == 0:
                remove_field_names.append(FieldName.FEAT_STATIC_REAL)
            if self.num_feat_dynamic_real == 0:
                remove_field_names.append(FieldName.FEAT_DYNAMIC_REAL)

            return Chain(
                [RemoveFields(field_names=remove_field_names)]
                + (
                    [SetField(output_field=FieldName.FEAT_STATIC_CAT, value=[0])]
                    if not self.num_feat_static_cat > 0
                    else []
                )
                + (
                    [
                        SetField(
                            output_field=FieldName.FEAT_STATIC_REAL, value=[0.0]
                        )
                    ]
                    if not self.num_feat_static_real > 0
                    else []
                )
                + [
                    AsNumpyArray(
                        field=FieldName.FEAT_STATIC_CAT,
                        expected_ndim=1,
                        dtype=int,
                    ),
                    AsNumpyArray(
                        field=FieldName.FEAT_STATIC_REAL,
                        expected_ndim=1,
                    ),
                    AsNumpyArray(
                        field=FieldName.TARGET,
                        # in the following line, we add 1 for the time dimension
                        expected_ndim=1 + len(self.distr_output.event_shape),
                    ),
                    AddObservedValuesIndicator(
                        target_field=FieldName.TARGET,
                        output_field=FieldName.OBSERVED_VALUES,
                    ),
                    AddTimeFeatures(
                        start_field=FieldName.START,
                        target_field=FieldName.TARGET,
                        output_field=FieldName.FEAT_TIME,
                        time_features=self.time_features,
                        pred_length=self.PRED_LENGTH,
                    ),
                    AddAgeFeature(
                        target_field=FieldName.TARGET,
                        output_field=FieldName.FEAT_AGE,
                        pred_length=self.PRED_LENGTH,
                        log_scale=True,
                    ),
                    VstackFeatures(
                        output_field=FieldName.FEAT_TIME,
                        input_fields=[FieldName.FEAT_TIME, FieldName.FEAT_AGE]
                        + (
                            [FieldName.FEAT_DYNAMIC_REAL]
                            if self.num_feat_dynamic_real > 0
                            else []
                        ),
                    ),
                ]
            )

    def _create_instance_splitter(
            self, instance_sampler: InstanceSampler
        ):
            return InstanceSplitter(
                target_field=FieldName.TARGET,
                is_pad_field=FieldName.IS_PAD,
                start_field=FieldName.START,
                forecast_start_field=FieldName.FORECAST_START,
                instance_sampler=instance_sampler,
                past_length=self._past_length,
                future_length=self.PRED_LENGTH,
                time_series_fields=[
                    FieldName.FEAT_TIME,
                    FieldName.OBSERVED_VALUES,
                ],
                dummy_value=self.distr_output.value_in_support,
            )

    def create_training_data_loader(
        self,
        data: Dataset,
        domain: str,
        training_hparams,
        shuffle_buffer_length: Optional[int] = None,
        **kwargs,
    ) -> Iterable:

        instance_sampler = training_domain_sampler(
            min_future=self.PRED_LENGTH, 
            num_instances=1.0
        )
        instance_sampler.set_attribute(
            domain,
            self.set_holidays, 
            self.start_datetime, 
            max_length=self.max_ts_length
        )

        transformation = self._create_instance_splitter(instance_sampler) + SelectFields(self.TRAINING_INPUT_NAMES)

        training_instances = transformation.apply(
            Cyclic(data)
            if shuffle_buffer_length is None
            else PseudoShuffled(
                Cyclic(data), shuffle_buffer_length=shuffle_buffer_length
            )
        )

        return IterableSlice(
            iter(
                DataLoader(
                    IterableDataset(training_instances),
                    batch_size=training_hparams['batch_size'],
                    **kwargs,
                )
            ),
            self.N_STEPS,
        )

    def create_evaluation_data_loader(
        self,
        data: Dataset,
        domain: str,
        training_hparams,
        start_idx: Optional[int]=0,
        last_idx: Optional[int]=None,
        **kwargs,
    ) -> Iterable:

        instance_sampler = evaluation_domain_sampler(
            min_future=self.PRED_LENGTH, 
            num_instances=1.0,
            start_idx=start_idx,
            last_idx=last_idx
        )
        instance_sampler.set_attribute(
            domain,
            self.set_holidays, 
            self.start_datetime, 
            max_length=self.max_ts_length
        )

        transformation = self._create_instance_splitter(instance_sampler) + SelectFields(self.TRAINING_INPUT_NAMES)

        validation_instances = transformation.apply(data)

        return DataLoader(
            IterableDataset(validation_instances),
            batch_size=training_hparams['batch_size'],
            **kwargs,
        )

    def get_nb_training_domains(self):
        """ Get the number of domains in the training set
        
        Returns:
            int: Number of domains in the training set
        """

        return len(self.ENVS)

    def get_number_of_batches(self):
        """ Returns total number of batches. """
        return self.num_batches_per_epoch

    def get_next_batch(self):
        """ Returns next batch. """
        batch = next(self.train_loaders_iter)[0]

        return self.split_input(batch)

    def split_input(self, batch):
        """ Splits input into input and target. """
        return {k: batch[k].to(self.device) for k in batch}, batch['future_target'].to(self.device)

    def loss(self, X, Y):
        """ Returns loss. """
        losses = self.loss_fn(X,Y)
        return losses.mean()

    def loss_by_domain(self, X, Y, n_domains):
        """ Returns the loss by domain. Because this is an Unbalanced dataset, there is only one during training domain"""
        losses = self.loss(X,Y)
        return losses.unsqueeze(0)

class AusElectricity(Multi_Domain_Dataset):

    # Training parameters
    N_STEPS = 3001
    CHECKPOINT_FREQ = 250

    ## Dataset parameters
    PERFORMANCE_MEASURE = 'rmse'
    PARADIGM = 'subpopulation_shift'
    SETUP = 'time'
    TASK = 'forecasting'

    ## Data parameters
    SEQ_LEN = 500
    INPUT_SIZE = 1
    OUTPUT_SIZE = 1
    FREQUENCY = '30T'
    PRED_LENGTH = 48

    ## Domain parameters
    ENVS = ['Holidays', 'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    SWEEP_ENVS = [-1] # This is a subpopulation shift problem
    ENVS_WEIGHTS = [8./365, 28./365, 28./365., 31./365., 27./365., 31./365., 30./365., 31./365., 31./365., 30./365., 31./365., 30./365., 29./365.]

    ## Data field identifiers
    PREDICTION_INPUT_NAMES = [
        "feat_static_cat",
        "feat_static_real",
        "past_time_feat",
        "past_target",
        "past_observed_values",
        "future_time_feat",
    ]

    TRAINING_INPUT_NAMES = PREDICTION_INPUT_NAMES + [
        "future_target",
        "future_observed_values",
    ]

    def __init__(self, flags, training_hparams):
        super().__init__()

        # Domain property
        self.set_holidays = holidays.country_holidays('AU')

        # Data property
        self.num_feat_static_cat = 0
        self.num_feat_dynamic_real = 0
        self.num_feat_static_real = 0
        self.start_datetime = datetime.datetime(2002,1,1,0,0)
        self.max_ts_length = 240000  #Rounded up to the tens of thousand, just cause idk

        ## Task information
        # Forcasting models output parameters of a distribution
        self.distr_output = StudentTOutput()
        # Covariate information given to the model alongside the target time series
        self.time_features = time_features_from_frequency_str(self.FREQUENCY)   # Transformed time information in the shape of a vector [f(minute), f(hour), f(day), ...]
        self.lags_seq = get_lags_for_frequency(self.FREQUENCY)  # Past targets from the target time series fed alongside the current time e.g., input_target=[target[0], target[-10], target[-100], ...]
        # Context info
        self.context_length = 7*self.PRED_LENGTH
        self._past_length = self.context_length + max(self.lags_seq)

        # Training parameters
        self.device = training_hparams['device']
        self.batch_size = training_hparams['batch_size']
        self.num_batches_per_epoch = 100

        # Get dataset
        self.raw_data = load_dataset('monash_tsf','australian_electricity_demand')

        # Define training / validation / test split
        train_first_idx = 175296 # Only for evaluation
        train_last_idx = 192864
        val_first_idx = train_last_idx
        val_last_idx = 210384
        test_first_idx = val_last_idx
        test_last_idx = 227904

        # Create ListDatasets
        train_dataset = ListDataset(
            [
                {  
                    FieldName.TARGET: tgt[:train_last_idx],
                    FieldName.START: strt
                } for (tgt, strt) in zip(
                    self.raw_data['test']['target'], 
                    self.raw_data['test']['start'])
            ],
            freq=self.FREQUENCY
        )
        validation_dataset = ListDataset(
            [
                {
                    FieldName.TARGET: tgt[:val_last_idx],
                    FieldName.START: strt
                } for (tgt, strt) in zip(
                    self.raw_data['test']['target'], 
                    self.raw_data['test']['start'])
            ], freq=self.FREQUENCY
        )
        test_dataset = ListDataset(
            [
                {
                    FieldName.TARGET: tgt,
                    FieldName.START: strt
                } for (tgt, strt) in zip(
                    self.raw_data['test']['target'], 
                    self.raw_data['test']['start'])
            ], freq=self.FREQUENCY
        )

        # Define embedding (this is kind of useless because this dataset doesn't have categorical covariate features, 
        # but it might be useful as template for other forecasting datasets)
        self.cardinality = [5]
        self.embedding_dimension = [5]

        # Create transformation
        self.transform = self.create_transformation()
        train_transformed = self.transform.apply(train_dataset, is_train=True)
        validation_transformed = self.transform.apply(validation_dataset, is_train=False)
        test_transformed = self.transform.apply(test_dataset, is_train=False)

        ## Create tensor dataset and dataloader
        self.train_names, self.train_loaders = [], []

        ## Create tensor dataset and dataloader
        self.train_names, self.train_loaders = [], []
        self.val_names, self.val_loaders = [], []
        for j, e in enumerate(self.ENVS):

            training_dataloader = self.create_training_data_loader(
                train_transformed,
                domain=e,
                training_hparams=training_hparams,
                shuffle_buffer_length=0,
                num_workers=self.N_WORKERS
            )
            self.train_names.append(e+"_train")
            self.train_loaders.append(training_dataloader)

            training_evaluation_dataloader = self.create_evaluation_data_loader(
                train_transformed,
                domain=e,
                training_hparams=training_hparams,
                start_idx=train_first_idx,
                last_idx=train_last_idx,
                num_workers=0
            )
            self.val_names.append(e+"_train")
            self.val_loaders.append(training_evaluation_dataloader)

            validation_dataloader = self.create_evaluation_data_loader(
                validation_transformed,
                domain=e,
                training_hparams=training_hparams,
                start_idx=val_first_idx,
                last_idx=val_last_idx,
                num_workers=0
            )
            self.val_names.append(e+"_val")
            self.val_loaders.append(validation_dataloader)

            test_dataloader = self.create_evaluation_data_loader(
                test_transformed,
                domain=e,
                training_hparams=training_hparams,
                start_idx=test_first_idx,
                last_idx=test_last_idx,
                num_workers=0
            )
            self.val_names.append(e+"_test")
            self.val_loaders.append(test_dataloader)

        self.train_loaders_iter = zip(*self.train_loaders)
        self.loss_fn = NegativeLogLikelihood()

    def get_nb_training_domains(self):
        """ Get the number of domains in the training set
        
        Returns:
            int: Number of domains in the training set
        """

        return len(self.ENVS)

    def create_transformation(self) -> Transformation:
            remove_field_names = []
            if self.num_feat_static_real == 0:
                remove_field_names.append(FieldName.FEAT_STATIC_REAL)
            if self.num_feat_dynamic_real == 0:
                remove_field_names.append(FieldName.FEAT_DYNAMIC_REAL)

            return Chain(
                [RemoveFields(field_names=remove_field_names)]
                + (
                    [SetField(output_field=FieldName.FEAT_STATIC_CAT, value=[0])]
                    if not self.num_feat_static_cat > 0
                    else []
                )
                + (
                    [
                        SetField(
                            output_field=FieldName.FEAT_STATIC_REAL, value=[0.0]
                        )
                    ]
                    if not self.num_feat_static_real > 0
                    else []
                )
                + [
                    AsNumpyArray(
                        field=FieldName.FEAT_STATIC_CAT,
                        expected_ndim=1,
                        dtype=int,
                    ),
                    AsNumpyArray(
                        field=FieldName.FEAT_STATIC_REAL,
                        expected_ndim=1,
                    ),
                    AsNumpyArray(
                        field=FieldName.TARGET,
                        # in the following line, we add 1 for the time dimension
                        expected_ndim=1 + len(self.distr_output.event_shape),
                    ),
                    AddObservedValuesIndicator(
                        target_field=FieldName.TARGET,
                        output_field=FieldName.OBSERVED_VALUES,
                    ),
                    AddTimeFeatures(
                        start_field=FieldName.START,
                        target_field=FieldName.TARGET,
                        output_field=FieldName.FEAT_TIME,
                        time_features=self.time_features,
                        pred_length=self.PRED_LENGTH,
                    ),
                    AddAgeFeature(
                        target_field=FieldName.TARGET,
                        output_field=FieldName.FEAT_AGE,
                        pred_length=self.PRED_LENGTH,
                        log_scale=True,
                    ),
                    VstackFeatures(
                        output_field=FieldName.FEAT_TIME,
                        input_fields=[FieldName.FEAT_TIME, FieldName.FEAT_AGE]
                        + (
                            [FieldName.FEAT_DYNAMIC_REAL]
                            if self.num_feat_dynamic_real > 0
                            else []
                        ),
                    ),
                ]
            )

    def _create_instance_splitter(
            self, instance_sampler: InstanceSampler
        ):
            return InstanceSplitter(
                target_field=FieldName.TARGET,
                is_pad_field=FieldName.IS_PAD,
                start_field=FieldName.START,
                forecast_start_field=FieldName.FORECAST_START,
                instance_sampler=instance_sampler,
                past_length=self._past_length,
                future_length=self.PRED_LENGTH,
                time_series_fields=[
                    FieldName.FEAT_TIME,
                    FieldName.OBSERVED_VALUES,
                ],
                dummy_value=self.distr_output.value_in_support,
            )

    def create_training_data_loader(
        self,
        data: Dataset,
        domain: str,
        training_hparams,
        shuffle_buffer_length: Optional[int] = None,
        **kwargs,
    ) -> Iterable:

        instance_sampler = training_domain_sampler(
            min_future=self.PRED_LENGTH, 
            num_instances=1.0
        )
        instance_sampler.set_attribute(
            domain,
            self.set_holidays, 
            self.start_datetime, 
            max_length=self.max_ts_length
        )

        transformation = self._create_instance_splitter(instance_sampler) + SelectFields(self.TRAINING_INPUT_NAMES)

        training_instances = transformation.apply(
            Cyclic(data)
            if shuffle_buffer_length is None
            else PseudoShuffled(
                Cyclic(data), shuffle_buffer_length=shuffle_buffer_length
            )
        )

        return IterableSlice(
            iter(
                DataLoader(
                    IterableDataset(training_instances),
                    batch_size=training_hparams['batch_size'],
                    **kwargs,
                )
            ),
            self.N_STEPS,
        )

    def create_evaluation_data_loader(
        self,
        data: Dataset,
        domain: str,
        training_hparams,
        start_idx: Optional[int]=0,
        last_idx: Optional[int]=None,
        **kwargs,
    ) -> Iterable:

        instance_sampler = evaluation_domain_sampler(
            min_future=self.PRED_LENGTH, 
            num_instances=1.0,
            start_idx=start_idx,
            last_idx=last_idx
        )
        instance_sampler.set_attribute(
            domain,
            self.set_holidays, 
            self.start_datetime, 
            max_length=self.max_ts_length
        )

        transformation = self._create_instance_splitter(instance_sampler) + SelectFields(self.TRAINING_INPUT_NAMES)

        validation_instances = transformation.apply(data)

        return DataLoader(
            IterableDataset(validation_instances),
            batch_size=training_hparams['batch_size'],
            **kwargs,
        )

    def get_nb_training_domains(self):
        """ Get the number of domains in the training set
        
        Returns:
            int: Number of domains in the training set
        """

        return len(self.ENVS)

    def get_number_of_batches(self):
        """ Returns the number of batches per epoch. """
        return self.num_batches_per_epoch

    def get_next_batch(self):
        """ Returns the next batch. """
        batch = next(self.train_loaders_iter)
        batch = {k: torch.cat([batch[b][k] for b in range(len(batch))], dim=0) for k in batch[0]}

        return self.split_input(batch)

    def split_input(self, batch):
        """ Splits the input batch into the input and target. """
        return {k: batch[k].to(self.device) for k in batch}, batch['future_target'].to(self.device)

    def loss(self, X, Y):
        """ Returns the loss for the given input and target. """
        return self.loss_fn(X,Y).mean()

    def loss_by_domain(self, X, Y, n_domains):
        """ Returns the loss by domain. Because this is an Unbalanced dataset, there is only one during training domain"""
        losses = self.loss_fn(X,Y)

        new_shape = (
            n_domains,
            losses.shape[0] // n_domains,
            *losses.shape[1:]
        )
        domain_losses = torch.reshape(losses, new_shape)
        
        return domain_losses.mean(dim=(1,2))

    def split_tensor_by_domains(self, X, Y, n_domains):
        raise NotImplementedError("Not clear how to apply this to this dataset, to be determined")
        return super().split_tensor_by_domains(X, Y, n_domains)

class original_IEMOCAPDataset(Dataset):

    def __init__(self, path, split=None, domain=None, all_domains=None):
        allvideoIDs, allvideoSpeakers, allvideoLabels, allvideoText,\
        allvideoAudio, allvideoVisual, allvideoSentence, alltrainVid,\
        alltestVid = pickle.load(open(path, 'rb'), encoding='latin1')
        '''
        label index mapping = {'hap':0, 'sad':1, 'neu':2, 'ang':3, 'exc':4, 'fru':5}
        '''
        if split == "train":
            split_keys = [x for x in alltrainVid]
        elif split =='valid':
            split_keys = [x for x in alltestVid]
        elif split == 'test':
            split_keys = [x for x in alltestVid]

        # Create the set of keys that will be used: 
        #   -If a domain is given, only keys with at least one emotion shift of that domain within it
        #   -If no domain is given, keep all keys of the split
        if domain is not None:
            self.keys = []
            for k in split_keys:
                labels = torch.LongTensor(allvideoLabels[k])
                for l1, l2 in zip(labels[:-1], labels[1:]):
                    dom = str(l1.item())+'-'+str(l2.item())
                    if domain == 'no-shift' and dom == dom[::-1]:
                        self.keys.append(k)
                        break
                    elif domain == 'rare-shift' and dom != dom[::-1]:
                        if dom not in all_domains or dom[::-1] not in all_domains:
                            self.keys.append(k)
                            break
                    else:
                        if dom == domain or dom[::-1] == domain:
                            self.keys.append(k)
                            break
        else:
            self.keys = split_keys

        self.speakers_info, self.labels, self.text_features, self.audio_features, self.visual_features = [], [], [], [], []
        for i, key in enumerate(self.keys):
            self.speakers_info.append(torch.FloatTensor(np.array([[1,0] if x=='M' else [0,1] for x in allvideoSpeakers[key]])))
            self.labels.append(torch.LongTensor(allvideoLabels[key]))
            self.text_features.append(torch.FloatTensor(np.array(allvideoText[key])))
            self.visual_features.append(torch.FloatTensor(np.array(allvideoVisual[key])))
            self.audio_features.append(torch.FloatTensor(np.array(allvideoAudio[key])))
        
        self.pad_mask = [torch.ones(len(label_arr)) for label_arr in self.labels]
        
        self.len = len(self.keys)

    def __getitem__(self, index):
        # vid = self.keys[index]
        return self.text_features[index],\
               self.visual_features[index],\
               self.audio_features[index],\
               self.speakers_info[index],\
               self.pad_mask[index],\
               self.labels[index]

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        return [pad_sequence([dat[i] for dat in data]) for i in range(6)]

class IEMOCAPOriginal(Multi_Domain_Dataset):
    """ Original splits of the IEMOCAP dataset

    THIS IS AN UNBALANCED DATASET THAT WE EVALUATE ON MULTIPLE DOMAINS

    This is primarily a sanity check to confirm the emotion shift problem addressed in the DialogueRNN paper
        https://arxiv.org/pdf/1811.00405.pdf

    """
    ## Training parameters
    N_STEPS = 2001
    CHECKPOINT_FREQ = 100

    ## Dataset parameters
    PERFORMANCE_MEASURE = 'acc'
    PARADIGM = 'subpopulation_shift'
    SETUP = 'time'
    TASK = 'classification'
    #:int: number of frames in each video
    INPUT_SHAPE = None
    OUTPUT_SIZE = 6
    #:str: path to the folder containing the data
    DATA_PATH = 'IEMOCAP/IEMOCAP_features_raw.pkl'

    ## Domain parameters
    ENVS = ['no-shift', 'shift']

    SWEEP_ENVS = [-1]

    def __init__(self, flags, training_hparams):
        super().__init__()

        # Check if the objective is ERM: This is a single domain dataset, therefore invariance penalty cannot be applied
        # Eventually we could investigate the use of penalties that doesn't use domain definitions such as SD
        assert flags.objective == 'ERM', "You are using a dataset with only one domain"

        ## Save stuff
        self.device = training_hparams['device']
        self.class_balance = training_hparams['class_balance']
        self.batch_size = training_hparams['batch_size']

        ## Prepare the data (Download if needed)
        ## Reminder to do later

        ## Create tensor dataset and dataloader
        self.train_names, self.train_loaders = [], []
        self.val_names, self.val_loaders = [], []

        full_data_path = os.path.join(flags.data_path, self.DATA_PATH)

        # Make training dataset/loader and append it to training containers
        train_dataset = original_IEMOCAPDataset(full_data_path, split='train')
        train_loader = InfiniteLoader(train_dataset, batch_size=training_hparams['batch_size'], num_workers=self.N_WORKERS, pin_memory=True)
        self.train_names.append('All_train')
        self.train_loaders.append(train_loader)

        # Make validation loaders
        fast_train_dataset = original_IEMOCAPDataset(full_data_path, split='train')
        fast_train_loader = torch.utils.data.DataLoader(fast_train_dataset, batch_size=50, shuffle=False, num_workers=self.N_WORKERS, pin_memory=True, collate_fn=fast_train_dataset.collate_fn)
        self.val_names.append([str(e)+'_train' for e in self.ENVS])
        self.val_loaders.append(fast_train_loader)
        fast_val_dataset = original_IEMOCAPDataset(full_data_path, split='valid')
        fast_val_loader = torch.utils.data.DataLoader(fast_val_dataset, batch_size=50, shuffle=False, num_workers=self.N_WORKERS, pin_memory=True, collate_fn=fast_val_dataset.collate_fn)
        self.val_names.append([str(e)+'_val' for e in self.ENVS])
        self.val_loaders.append(fast_val_loader)
        fast_test_dataset = original_IEMOCAPDataset(full_data_path, split='test')
        fast_test_loader = torch.utils.data.DataLoader(fast_test_dataset, batch_size=50, shuffle=False, num_workers=self.N_WORKERS, pin_memory=True, collate_fn=fast_test_dataset.collate_fn)
        self.val_names.append([str(e)+'_test' for e in self.ENVS])
        self.val_loaders.append(fast_test_loader)

        # Define loss function
        self.log_prob = nn.LogSoftmax(dim=1)
        self.loss_fn = nn.NLLLoss(weight=self.get_class_weight().to(training_hparams['device']), reduction='none')
        self.train_loaders_iter = zip(*self.train_loaders)

    def loss(self, pred, Y):
        """
        Computes the masked NLL loss for the IEMOCAP dataset
        Args:
            pred (torch.tensor): Predictions of the model. Shape (batch, time, n_classes)
            Y (torch.tensor): Targets. Shape (batch, time)
        Returns:
            torch.tensor: loss of each samples. Shape (batch, time)
        """

        target, mask = Y

        pred = pred.permute(0,2,1)

        # Get all losses without reduction
        losses = self.loss_fn(self.log_prob(pred), target)

        # Keep only losses that weren't padded
        masked_losses = torch.masked_select(losses, mask.bool())

        # Return mean loss
        return masked_losses.mean()

    def loss_by_domain(self, pred, Y, n_domains):
        return self.loss(pred, Y).unsqueeze(0)

    def get_class_weight(self):
        """ Compute class weight for class balanced training

        Returns:
            list: list of weights of length OUTPUT_SIZE
        """
        _, train_loaders = self.get_train_loaders()

        n_labels = torch.zeros(self.OUTPUT_SIZE)

        for env_loader in train_loaders:
            labels = pad_sequence(env_loader.dataset.labels)
            pad_mask = pad_sequence(env_loader.dataset.pad_mask).bool()
            for i in range(self.OUTPUT_SIZE):
                n_labels[i] += torch.eq(torch.masked_select(labels, pad_mask), i).sum()

        weights = n_labels.max() / n_labels

        return weights

    def get_next_batch(self):

        batch = next(self.train_loaders_iter)

        output = []
        for data_idx in range(6):
            out = pad_sequence([batch[dom_idx][data_idx] for dom_idx in range(len(batch))])
            out = out.view(out.shape[0], out.shape[1] * out.shape[2], *out.shape[3:])
            output.append(out)

        return self.split_input(output)

    def split_input(self, batch):
        """
        Outputs the split input and labels
        This dataset has padded sequences, therefore it returns a tuple with the mask that indicate what is padded and what isn't
        """

        X = torch.cat([batch[0], batch[1], batch[2]], dim=2).to(self.device)
        Y = batch[-1].transpose(0,1).to(self.device)
        pad_mask = batch[-2].transpose(0,1).to(self.device)
        q_mask = batch[-3].to(self.device)

        return (X, q_mask, pad_mask), (Y, pad_mask)

    def get_nb_correct(self, out, target):
        """Time domain correct count

        Args:
            pred (_type_): _description_
            target (_type_): _description_

        Returns:
            _type_: _description_
        """

        # Get predictions
        pred = out.argmax(dim=2)

        # Make domain masking
        domain_mask = self.get_domain_mask(target[0])

        # Get right guesses and stuff
        pad_mask = target[1]
        batch_correct = torch.zeros(len(self.ENVS)).to(out.device)
        batch_numel = torch.zeros(len(self.ENVS)).to(out.device)
        for i in range(len(self.ENVS)):
            pad_env_mask = torch.logical_and(pad_mask, domain_mask[i,...])

            domain_pred = torch.masked_select(pred, pad_env_mask.bool())
            domain_target = torch.masked_select(target[0], pad_env_mask.bool())

            batch_correct[i] = domain_pred.eq(domain_target).sum()
            batch_numel[i] = domain_target.numel()

        # Remove padded sequences of input / targets
        return batch_correct, batch_numel

    def get_domain_mask(self, target):
        """ Creates the domain masks for a batch
        """
        domain_mask = torch.zeros(len(self.ENVS), *target.shape).to(target.device)
        for i, env in enumerate(self.ENVS):
            for spl in range(target.shape[0]):
                for j, (l1, l2) in enumerate(zip(target[spl,:-1], target[spl,1:])):
                    if env == 'no-shift':
                        if str(l1.item()) == str(l2.item()):
                            domain_mask[i, spl, j+1] = 1
                    elif env == 'shift':
                        if str(l1.item()) != str(l2.item()):
                            domain_mask[i, spl, j+1] = 1
        
        return domain_mask

    def get_nb_training_domains(self):
        """ Get the number of domains in the training set
        
        Returns:
            int: Number of domains in the training set
        """

        return 1

class IEMOCAPDataset(Dataset):

    def __init__(self, path, split=None, domain=None, all_domains = None):
        allvideoIDs, allvideoSpeakers, allvideoLabels, allvideoText,\
        allvideoAudio, allvideoVisual, allvideoSentence, alltrainVid,\
        allvalidVid, alltestVid = pickle.load(open(path, 'rb'), encoding='latin1')
        '''
        label index mapping = {'hap':0, 'sad':1, 'neu':2, 'ang':3, 'exc':4, 'fru':5}
        '''
        if split == "train":
            split_keys = alltrainVid
        elif split =='valid':
            split_keys = allvalidVid
        elif split == 'test':
            split_keys = alltestVid

        # Create the set of keys that will be used: 
        #   -If a domain is given, only keys with at least one emotion shift of that domain within it
        #   -If no domain is given, keep all keys of the split
        if domain is not None:
            self.keys = []
            for k in split_keys:
                labels = torch.LongTensor(allvideoLabels[k])
                for l1, l2 in zip(labels[:-1], labels[1:]):
                    dom = str(l1.item())+'-'+str(l2.item())
                    if domain == 'no-shift' and dom == dom[::-1]:
                        self.keys.append(k)
                        break
                    elif domain == 'rare-shift' and dom != dom[::-1]:
                        if dom not in all_domains or dom[::-1] not in all_domains:
                            self.keys.append(k)
                            break
                    else:
                        if dom == domain or dom[::-1] == domain:
                            self.keys.append(k)
                            break
        else:
            self.keys = split_keys

        self.speakers_info, self.labels, self.text_features, self.audio_features, self.visual_features = [], [], [], [], []
        for i, key in enumerate(self.keys):
            self.speakers_info.append(torch.FloatTensor(np.array([[1,0] if x=='M' else [0,1] for x in allvideoSpeakers[key]])))
            self.labels.append(torch.LongTensor(allvideoLabels[key]))
            self.text_features.append(torch.FloatTensor(np.array(allvideoText[key])))
            self.visual_features.append(torch.FloatTensor(np.array(allvideoVisual[key])))
            self.audio_features.append(torch.FloatTensor(np.array(allvideoAudio[key])))
        
        self.pad_mask = [torch.ones(len(label_arr)) for label_arr in self.labels]
        
        self.len = len(self.keys)

    def __getitem__(self, index):
        # vid = self.keys[index]
        return self.text_features[index],\
               self.visual_features[index],\
               self.audio_features[index],\
               self.speakers_info[index],\
               self.pad_mask[index],\
               self.labels[index]

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        return [pad_sequence([dat[i] for dat in data]) for i in range(6)]

class IEMOCAPUnbalanced(Multi_Domain_Dataset):
    """ IEMOCAP

    THIS IS AN UNBALANCED DATASET THAT WE EVALUATE ON MULTIPLE DOMAINS
    """
    ## Training parameters
    N_STEPS = 1001
    CHECKPOINT_FREQ = 100

    ## Dataset parameters
    PERFORMANCE_MEASURE = 'acc'
    PARADIGM = 'subpopulation_shift'
    SETUP = 'time'
    TASK = 'classification'

    ## Data parameters
    #:int: number of frames in each video
    INPUT_SHAPE = None
    OUTPUT_SIZE = 6
    #:str: path to the folder containing the data
    DATA_PATH = 'IEMOCAP/IEMOCAP_features_raw_OOD.pkl'

    ## Domain parameters
    ENVS = ['no-shift', 
            'rare-shift', 
            '0-1',#, '1-0',
            '0-2',#, '2-0', 
            '0-4',#, '4-0', 
            '1-2',#, '2-1', 
            '1-5',#, '5-1',
            '2-3',#, '3-2', 
            '2-4',#, '4-2',
            '2-5',#, '5-2',
            '3-5']#, '5-3',
            #'3-1']
    SWEEP_ENVS = [-1]

    def __init__(self, flags, training_hparams):
        super().__init__()

        # Check if the objective is ERM: This is a single domain dataset, therefore invariance penalty cannot be applied
        # Eventually we could investigate the use of penalties that doesn't use domain definitions such as SD
        assert flags.objective == 'ERM', "You are using a dataset with only one domain"

        ## Save stuff
        self.device = training_hparams['device']
        self.class_balance = training_hparams['class_balance']
        self.batch_size = training_hparams['batch_size']

        ## Prepare the data (Download if needed)
        ## Reminder to do later

        ## Create tensor dataset and dataloader
        self.train_names, self.train_loaders = [], []
        self.val_names, self.val_loaders = [], []

        full_data_path = os.path.join(flags.data_path, self.DATA_PATH)

        # Make training dataset/loader and append it to training containers
        train_dataset = IEMOCAPDataset(full_data_path, split='train')
        train_loader = InfiniteLoader(train_dataset, batch_size=training_hparams['batch_size'], num_workers=self.N_WORKERS, pin_memory=True)
        self.train_names.append('All_train')
        self.train_loaders.append(train_loader)

        # Make validation loaders
        fast_train_dataset = IEMOCAPDataset(full_data_path, split='train')
        fast_train_loader = torch.utils.data.DataLoader(fast_train_dataset, batch_size=50, shuffle=False, num_workers=self.N_WORKERS, pin_memory=True, collate_fn=fast_train_dataset.collate_fn)
        self.val_names.append([str(e)+'_train' for e in self.ENVS])
        self.val_loaders.append(fast_train_loader)
        fast_val_dataset = IEMOCAPDataset(full_data_path, split='valid')
        fast_val_loader = torch.utils.data.DataLoader(fast_val_dataset, batch_size=50, shuffle=False, num_workers=self.N_WORKERS, pin_memory=True, collate_fn=fast_val_dataset.collate_fn)
        self.val_names.append([str(e)+'_val' for e in self.ENVS])
        self.val_loaders.append(fast_val_loader)
        fast_test_dataset = IEMOCAPDataset(full_data_path, split='test')
        fast_test_loader = torch.utils.data.DataLoader(fast_test_dataset, batch_size=50, shuffle=False, num_workers=self.N_WORKERS, pin_memory=True, collate_fn=fast_test_dataset.collate_fn)
        self.val_names.append([str(e)+'_test' for e in self.ENVS])
        self.val_loaders.append(fast_test_loader)

        # Define loss function
        self.log_prob = nn.LogSoftmax(dim=1)
        self.loss_fn = nn.NLLLoss(weight=self.get_class_weight().to(training_hparams['device']), reduction='none')
        self.train_loaders_iter = zip(*self.train_loaders)

    def loss(self, pred, Y):
        """
        Computes the masked NLL loss for the IEMOCAP dataset
        Args:
            pred (torch.tensor): Predictions of the model. Shape (batch, time, n_classes)
            Y (torch.tensor): Targets. Shape (batch, time)
        Returns:
            torch.tensor: loss of each samples. Shape (batch, time)
        """

        target, mask = Y

        pred = pred.permute(0,2,1)

        # Get all losses without reduction
        losses = self.loss_fn(self.log_prob(pred), target)

        # Keep only losses that weren't padded
        masked_losses = torch.masked_select(losses, mask.bool())

        # Return mean loss
        return masked_losses.mean()

    def loss_by_domain(self, pred, Y, n_domains):
        return self.loss(pred, Y).unsqueeze(0)
    
    def get_class_weight(self):
        """ Compute class weight for class balanced training

        Returns:
            list: list of weights of length OUTPUT_SIZE
        """
        _, train_loaders = self.get_train_loaders()

        n_labels = torch.zeros(self.OUTPUT_SIZE)

        for env_loader in train_loaders:
            labels = pad_sequence(env_loader.dataset.labels)
            pad_mask = pad_sequence(env_loader.dataset.pad_mask).bool()
            for i in range(self.OUTPUT_SIZE):
                n_labels[i] += torch.eq(torch.masked_select(labels, pad_mask), i).sum()

        weights = n_labels.max() / n_labels

        return weights

    def get_next_batch(self):

        batch = next(self.train_loaders_iter)

        output = []
        for data_idx in range(6):
            out = pad_sequence([batch[dom_idx][data_idx] for dom_idx in range(len(batch))])
            out = out.view(out.shape[0], out.shape[1] * out.shape[2], *out.shape[3:])
            output.append(out)

        return self.split_input(output)

    def split_input(self, batch):
        """
        Outputs the split input and labels
        This dataset has padded sequences, therefore it returns a tuple with the mask that indicate what is padded and what isn't
        """

        X = torch.cat([batch[0], batch[1], batch[2]], dim=2).to(self.device)
        Y = batch[-1].transpose(0,1).to(self.device)
        pad_mask = batch[-2].transpose(0,1).to(self.device)
        q_mask = batch[-3].to(self.device)

        return (X, q_mask, pad_mask), (Y, pad_mask)

    def get_nb_correct(self, out, target):
        """Time domain correct count

        Args:
            pred (_type_): _description_
            target (_type_): _description_

        Returns:
            _type_: _description_
        """

        # Get predictions
        pred = out.argmax(dim=2)

        # Make domain masking
        domain_mask = self.get_domain_mask(target[0])

        # Get right guesses and stuff
        pad_mask = target[1]
        batch_correct = torch.zeros(len(self.ENVS)).to(out.device)
        batch_numel = torch.zeros(len(self.ENVS)).to(out.device)
        for i in range(len(self.ENVS)):
            pad_env_mask = torch.logical_and(pad_mask, domain_mask[i,...])

            domain_pred = torch.masked_select(pred, pad_env_mask.bool())
            domain_target = torch.masked_select(target[0], pad_env_mask.bool())

            batch_correct[i] = domain_pred.eq(domain_target).sum()
            batch_numel[i] = domain_target.numel()

        # Remove padded sequences of input / targets
        return batch_correct, batch_numel

    def get_domain_mask(self, target):
        """ Creates the domain masks for a batch
        """
        domain_mask = torch.zeros(len(self.ENVS), *target.shape).to(target.device)
        for i, env in enumerate(self.ENVS):
            for spl in range(target.shape[0]):
                for j, (l1, l2) in enumerate(zip(target[spl,:-1], target[spl,1:])):
                    if env == 'no-shift':
                        if str(l1.item()) == str(l2.item()):
                            domain_mask[i, spl, j+1] = 1
                    if env == 'rare-shift':
                        if str(l1.item())+'-'+str(l2.item()) not in self.ENVS and str(l1.item()) != str(l2.item()):
                            domain_mask[i, spl, j+1] = 1
                    else:
                        if str(l1.item())+'-'+str(l2.item()) == env or str(l2.item())+'-'+str(l1.item()) == env:
                            domain_mask[i, spl, j+1] = 1
        
        return domain_mask

    def get_nb_training_domains(self):
        """ Get the number of domains in the training set
        
        Returns:
            int: Number of domains in the training set
        """

        return 1
        
class IEMOCAP(Multi_Domain_Dataset):
    """ IEMOCAP
    """
    ## Training parameters
    N_STEPS = 1001
    CHECKPOINT_FREQ = 100

    ## Dataset parameters
    PERFORMANCE_MEASURE = 'acc'
    PARADIGM = 'subpopulation_shift'
    SETUP = 'time'
    TASK = 'classification'

    ## Data parameters
    #:int: number of frames in each video
    INPUT_SHAPE = None
    OUTPUT_SIZE = 6
    #:str: path to the folder containing the data
    DATA_PATH = 'IEMOCAP/IEMOCAP_features_raw_OOD.pkl'

    ## Domain parameters
    ENVS = ['no-shift', 
            'rare-shift', 
            '0-1',#, '1-0',
            '0-2',#, '2-0', 
            '0-4',#, '4-0', 
            '1-2',#, '2-1', 
            '1-5',#, '5-1',
            '2-3',#, '3-2', 
            '2-4',#, '4-2',
            '2-5',#, '5-2',
            '3-5']#, '5-3',
            #'3-1']
    SWEEP_ENVS = [-1]

    def __init__(self, flags, training_hparams):
        super().__init__()

        ## Save stuff
        self.device = training_hparams['device']
        self.class_balance = training_hparams['class_balance']
        self.batch_size = training_hparams['batch_size']

        ## Prepare the data (Download if needed)
        ## Reminder to do later

        ## Create tensor dataset and dataloader
        self.train_names, self.train_loaders = [], []
        self.val_names, self.val_loaders = [], []

        full_data_path = os.path.join(flags.data_path, self.DATA_PATH)

        # Make training dataset/loader and append it to training containers
        for domain in self.ENVS:
            train_dataset = IEMOCAPDataset(full_data_path, split='train', domain=domain, all_domains=self.ENVS)
            train_loader = InfiniteLoader(train_dataset, batch_size=training_hparams['batch_size'], num_workers=self.N_WORKERS, pin_memory=True)
            self.train_names.append(domain+'_train')
            self.train_loaders.append(train_loader)

        # Make validation loaders
        fast_train_dataset = IEMOCAPDataset(full_data_path, split='train')
        fast_train_loader = torch.utils.data.DataLoader(fast_train_dataset, batch_size=50, shuffle=False, num_workers=self.N_WORKERS, pin_memory=True, collate_fn=fast_train_dataset.collate_fn)
        self.val_names.append([str(e)+'_train' for e in self.ENVS])
        self.val_loaders.append(fast_train_loader)
        fast_val_dataset = IEMOCAPDataset(full_data_path, split='valid')
        fast_val_loader = torch.utils.data.DataLoader(fast_val_dataset, batch_size=50, shuffle=False, num_workers=self.N_WORKERS, pin_memory=True, collate_fn=fast_val_dataset.collate_fn)
        self.val_names.append([str(e)+'_val' for e in self.ENVS])
        self.val_loaders.append(fast_val_loader)
        fast_test_dataset = IEMOCAPDataset(full_data_path, split='test')
        fast_test_loader = torch.utils.data.DataLoader(fast_test_dataset, batch_size=50, shuffle=False, num_workers=self.N_WORKERS, pin_memory=True, collate_fn=fast_test_dataset.collate_fn)
        self.val_names.append([str(e)+'_test' for e in self.ENVS])
        self.val_loaders.append(fast_test_loader)

        # Define loss function
        self.log_prob = nn.LogSoftmax(dim=1)
        self.loss_fn = nn.NLLLoss(weight=self.get_class_weight().to(training_hparams['device']), reduction='none')

        # Define train loaders iterable
        self.train_loaders_iter = zip(*self.train_loaders)

    def loss(self, pred, Y):
        """
        Computes the masked NLL loss for the IEMOCAP dataset
        Args:
            pred (torch.tensor): Predictions of the model. Shape (batch, time, n_classes)
            Y (torch.tensor): Targets. Shape (batch, time)
        Returns:
            torch.tensor: loss of each samples. Shape (batch, time)
        """

        target, mask = Y

        pred = pred.permute(0,2,1)

        # Get all losses without reduction
        losses = self.loss_fn(self.log_prob(pred), target)

        # Keep only losses that weren't padded
        masked_losses = torch.masked_select(losses, mask.bool())

        # Return mean loss
        return masked_losses.mean()

    def loss_by_domain(self, X, Y, n_domains):

        # Split tensors by domains (n_domains, n_predictions, ...)
        pred, target = self.split_tensor_by_domains(X, Y, len(self.ENVS))

        # Get all losses without reduction
        losses = torch.zeros(len(self.ENVS)).to(X.device)
        for i, (env_pred, env_target) in enumerate(zip(pred, target)):
            losses[i] = self.loss_fn(self.log_prob(env_pred), env_target).mean()

        return losses

    def split_tensor_by_domains(self, X, Y, n_domains):
        """ Group tensor by domain for source domains datasets

        Args:
            n_domains (int): Number of domains in the batch
            tensor (torch.tensor): tensor to be split. Shape (n_domains*batch, ...)

        Returns:
            Tensor: The reshaped output (n_domains, len(all predictions), ...)
        """

        # Unpack
        targets, pad_mask = Y

        # Get mask of which predictions were of which domains
        domain_mask = self.get_domain_mask(targets)

        domain_pred = [0] * len(self.ENVS)
        domain_target = [0] * len(self.ENVS)
        for i in range(len(self.ENVS)):
            pad_env_mask = torch.logical_and(pad_mask, domain_mask[i,...])

            domain_pred[i] = X[pad_env_mask, :]
            domain_target[i] = targets[pad_env_mask]
        
        return domain_pred, domain_target
    
    def get_class_weight(self):
        """ Compute class weight for class balanced training

        Returns:
            list: list of weights of length OUTPUT_SIZE
        """
        _, train_loaders = self.get_train_loaders()

        n_labels = torch.zeros(self.OUTPUT_SIZE)

        for env_loader in train_loaders:
            labels = pad_sequence(env_loader.dataset.labels)
            pad_mask = pad_sequence(env_loader.dataset.pad_mask).bool()
            for i in range(self.OUTPUT_SIZE):
                n_labels[i] += torch.eq(torch.masked_select(labels, pad_mask), i).sum()

        weights = n_labels.max() / n_labels

        return weights

    def get_next_batch(self):

        batch = next(self.train_loaders_iter)

        output = []
        for data_idx in range(6):
            out = pad_sequence([batch[dom_idx][data_idx] for dom_idx in range(len(batch))])
            out = out.view(out.shape[0], out.shape[1] * out.shape[2], *out.shape[3:])
            output.append(out)

        return self.split_input(output)

    def split_input(self, batch):
        """
        Outputs the split input and labels
        This dataset has padded sequences, therefore it returns a tuple with the mask that indicate what is padded and what isn't
        """

        X = torch.cat([batch[0], batch[1], batch[2]], dim=2).to(self.device)
        Y = batch[-1].transpose(0,1).to(self.device)
        pad_mask = batch[-2].transpose(0,1).to(self.device)
        q_mask = batch[-3].to(self.device)

        return (X, q_mask, pad_mask), (Y, pad_mask)

    def get_nb_correct(self, out, Y):
        """Time domain correct count

        Args:
            pred (_type_): _description_
            target (_type_): _description_

        Returns:
            _type_: _description_
        """

        # Get predictions
        pred = out.argmax(dim=2)
        target, pad_mask = Y

        # Make domain masking
        domain_mask = self.get_domain_mask(target)

        # Get right guesses and stuff
        batch_correct = torch.zeros(len(self.ENVS)).to(out.device)
        batch_numel = torch.zeros(len(self.ENVS)).to(out.device)
        for i in range(len(self.ENVS)):
            pad_env_mask = torch.logical_and(pad_mask, domain_mask[i,...])

            domain_pred = pred[pad_env_mask]
            domain_target = target[pad_env_mask]

            batch_correct[i] = domain_pred.eq(domain_target).sum()
            batch_numel[i] = domain_target.numel()

        # Remove padded sequences of input / targets
        return batch_correct, batch_numel

    def get_domain_mask(self, target):
        """ Creates the domain masks for a batch
        """
        domain_mask = torch.zeros(len(self.ENVS), *target.shape).to(target.device)
        for i, env in enumerate(self.ENVS):
            for spl in range(target.shape[0]):
                for j, (l1, l2) in enumerate(zip(target[spl,:-1], target[spl,1:])):
                    if env == 'no-shift':
                        if str(l1.item()) == str(l2.item()):
                            domain_mask[i, spl, j+1] = 1
                    if env == 'rare-shift':
                        if str(l1.item())+'-'+str(l2.item()) not in self.ENVS and str(l1.item()) != str(l2.item()):
                            domain_mask[i, spl, j+1] = 1
                    else:
                        if str(l1.item())+'-'+str(l2.item()) == env or str(l2.item())+'-'+str(l1.item()) == env:
                            domain_mask[i, spl, j+1] = 1
        
        return domain_mask

    def get_nb_training_domains(self):
        """ Get the number of domains in the training set
        
        Returns:
            int: Number of domains in the training set
        """
        return len(self.ENVS)

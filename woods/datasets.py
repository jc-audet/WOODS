"""Defining the benchmarks for OoD generalization in time-series"""

import os
import copy
import h5py
from PIL import Image
import warnings

import scipy.io
import numpy as np
from scipy import fft, signal
import matplotlib.pyplot as plt

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

# Ignore gluonts spam of futurewarnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

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
    "AusElectricity",
    "AusElectricityUnbalanced",
    "AusElectricityMonthly",
    "AusElectricityMonthlyBalanced",
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
    """ Returns the setup of a dataset 
    
    Args:
        dataset_name (str): Name of the dataset to get the number of environments of. (Must be a part of the DATASETS list)

    Returns:
        dict: The setup of the dataset ('seq' or 'step')
    """
    return get_dataset_class(dataset_name).SETUP

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
    #:string: The setup of the dataset ('source' for Source-domains or 'time' for time-domains)
    SETUP = None
    #:string: The type of prediction task ('classification' of 'regression')
    TASK = None
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

    ## Environment parameters
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

    def loss(self, X, Y):
        """
        Computes the loss defined by the dataset
        Args:
            X (torch.tensor): Predictions of the model. Shape (batch, time, n_classes)
            Y (torch.tensor): Targets. Shape (batch, time)
        Returns:
            torch.tensor: loss of each samples. Shape (batch, time)
        """
        X = X.permute(0,2,1)
        return self.loss_fn(self.log_prob(X), Y)

    def get_pred_time(self, X):
        return torch.tensor(self.PRED_TIME)

    def prepare_data(self, path, download=False):
        """ Prepares the dataset.

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
              
    def get_next_batch(self):
        
        batch_loaders = next(self.train_loaders_iter)
        return [(x, y) for x, y in batch_loaders]

    def split_input(self, input):

        return (
            torch.cat([x for x,y in input]).to(self.device),
            torch.cat([y for x,y in input]).to(self.device)
        )
    
    def split_tensor_by_domains(self, n_domains, tensor):
        """ Group tensor by domain for source domains datasets

        Args:
            n_domains (int): Number of domains in the batch
            tensor (torch.tensor): tensor to be split. Shape (n_domains*batch, ...)

        Returns:
            Tensor: The reshaped output (n_domains, batch, ...)
        """
        tensor_shape = tensor.shape
        print(tensor_shape)
        return torch.reshape(tensor, (n_domains, tensor_shape[0]//n_domains, *tensor_shape[1:]))

    def get_number_of_batches(self):
        return np.sum([len(train_l) for train_l in self.train_loaders])

class Basic_Fourier(Multi_Domain_Dataset):
    """ Fourier_basic dataset

    A dataset of 1D sinusoid signal to classify according to their Fourier spectrum. 
    
    Args:
        flags (argparse.Namespace): argparse of training arguments
        training_hparams (dict): dictionnary of training hyper parameters coming from the hyperparams.py file

    Note:
        No download is required as it is purely synthetic
    """
    ## Dataset parameters
    SETUP = 'source'
    TASK = 'classification'
    SEQ_LEN = 50
    PRED_TIME = [49]
    INPUT_SHAPE = [1]
    OUTPUT_SIZE = 2

    ## Environment parameters
    ENVS = ['no_spur']
    SWEEP_ENVS = [None]

    def __init__(self, flags, training_hparams):
        super().__init__()

        # Make important checks
        assert flags.test_env == None, "You are using a dataset with only a single environment, there cannot be a test environment"

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
            in_dataset, out_dataset = make_split(dataset, flags.holdout_fraction, seed=i)

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
        self.train_loaders_iter = zip(*self.train_loaders)
        
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
    SETUP = 'source'
    TASK = 'classification'
    SEQ_LEN = 50
    PRED_TIME = [49]
    INPUT_SHAPE = [1]
    OUTPUT_SIZE = 2

    ## Environment parameters
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

        ## Define the spurious Fourier spectrum (one direct and the inverse)
        self.direct_fourier_0 = copy.deepcopy(self.fourier_0)
        self.direct_fourier_1 = copy.deepcopy(self.fourier_1)
        self.direct_fourier_0[200] = 0.5
        self.direct_fourier_1[400] = 0.5

        def conv(signal):
            blurred_signal = np.zeros_like(signal)
            for i in range(1, np.shape(blurred_signal)[0]-1):
                blurred_signal[i] = np.mean(signal[i-1:i+1])
            return blurred_signal


        self.inverse_fourier_0 = copy.deepcopy(self.fourier_0)
        self.inverse_fourier_1 = copy.deepcopy(self.fourier_1)
        self.inverse_fourier_0[400] = 0.5
        self.inverse_fourier_1[200] = 0.5

        ## Create the sequences for direct and inverse
        direct_signal_0 = fft.irfft(self.direct_fourier_0, n=10000)
        direct_signal_0 = torch.tensor( direct_signal_0.reshape(-1,50,1) ).float()
        direct_signal_0 /= direct_signal_0.max()
        direct_signal_1 = fft.irfft(self.direct_fourier_1, n=10000)
        direct_signal_1 = torch.tensor( direct_signal_1.reshape(-1,50,1) ).float()
        direct_signal_1 /= direct_signal_1.max()
        direct_signal_0, direct_signal_1 = self.super_sample(direct_signal_0, direct_signal_1)

        perm_0 = torch.randperm(direct_signal_0.shape[0])
        direct_signal_0 = direct_signal_0[perm_0,:]
        perm_1 = torch.randperm(direct_signal_1.shape[0])
        direct_signal_1 = direct_signal_1[perm_1,:]
        direct_signal = [direct_signal_0, direct_signal_1]

        inverse_signal_0 = fft.irfft(self.inverse_fourier_0, n=10000)
        inverse_signal_0 = torch.tensor( inverse_signal_0.reshape(-1,50,1) ).float()
        inverse_signal_0 /= inverse_signal_0.max()
        inverse_signal_1 = fft.irfft(self.inverse_fourier_1, n=10000)
        inverse_signal_1 = torch.tensor( inverse_signal_1.reshape(-1,50,1) ).float()
        inverse_signal_1 /= inverse_signal_1.max()
        inverse_signal_0, inverse_signal_1 = self.super_sample(inverse_signal_0, inverse_signal_1)

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

            in_dataset, out_dataset = make_split(dataset, flags.holdout_fraction, seed=i)
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
        self.train_loaders_iter = zip(*self.train_loaders)

    def super_sample(self, signal_0, signal_1):
        """ Sample signals frames with a bunch of offsets """
        all_signal_0 = torch.zeros(0,50,1)
        all_signal_1 = torch.zeros(0,50,1)
        for i in range(0, 50, 2):
            new_signal_0 = copy.deepcopy(signal_0)[i:i-50]
            new_signal_1 = copy.deepcopy(signal_1)[i:i-50]
            split_signal_0 = new_signal_0.reshape(-1,50,1).clone().detach().float()
            split_signal_1 = new_signal_1.reshape(-1,50,1).clone().detach().float()
            all_signal_0 = torch.cat((all_signal_0, split_signal_0), dim=0)
            all_signal_1 = torch.cat((all_signal_1, split_signal_1), dim=0)
        
        return all_signal_0, all_signal_1

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
    ## Training parameters
    N_STEPS = 5001

    ## Dataset parameters
    SETUP = 'source'
    TASK = 'classification'
    SEQ_LEN = 4
    PRED_TIME = [1, 2, 3]
    INPUT_SHAPE = [1,28,28]
    OUTPUT_SIZE = 2

    ## Environment parameters
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
        MNIST_tfrm = transforms.Compose([ 
            transforms.ToTensor(),
            ])

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
            in_dataset, out_dataset = make_split(dataset, flags.holdout_fraction, seed=i)

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
        self.train_loaders_iter = zip(*self.train_loaders)

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
    ## Training parameters
    N_STEPS = 5001

    ## Dataset parameters
    SETUP = None
    TASK = 'classification'
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
    
    ## Environment parameters
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

            # self.plot_samples(colored_images, colored_labels, str(e))

            # Make Tensor dataset and the split
            dataset = torch.utils.data.TensorDataset(colored_images, colored_labels)
            in_dataset, out_dataset = make_split(dataset, flags.holdout_fraction, seed=i)

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
        self.train_loaders_iter = zip(*self.train_loaders)

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

    ## Environment parameters
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
        in_split, out_split = get_split(dataset, flags.holdout_fraction, seed=i)

        in_train_dataset = torch.utils.data.TensorDataset(colored_images[in_split,:-1,...], colored_labels.long()[in_split,:-1,...])

        in_eval_dataset = torch.utils.data.TensorDataset(colored_images[in_split,...], colored_labels.long()[in_split,...])
        out_eval_dataset = torch.utils.data.TensorDataset(colored_images[out_split,...], colored_labels.long()[out_split,...])

        # train_dataset = torch.utils.data.TensorDataset(colored_images[:,:3,...], colored_labels.long()[:,:2,...])
        # eval_dataset = torch.utils.data.TensorDataset(colored_images, colored_labels.long())
        # in_train_dataset, out_train_dataset = make_split(train_dataset, flags.holdout_fraction, seed=i)
        # in_eval_dataset, out_eval_dataset = make_split(eval_dataset, flags.holdout_fraction, seed=i)

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

    def split_tensor_by_domains(self, n_domains, tensor):
        """ Group tensor by domain for source domains datasets

        Args:
            n_domains (int): Number of domains in the batch
            tensor (torch.tensor): tensor to be split. Shape (n_domains*batch, ...)

        Returns:
            Tensor: The reshaped output (n_domains, batch, ...)
        """
        return tensor.transpose(0,1).unsqueeze(2)

              
    def split_output(self, out):
        """ Group data and prediction by environment

        Args:
            labels (Tensor): labels of the data (batch_size, len(PRED_TIME))

        Returns:
            Tensor: The reshaped data (n_env-1, batch_size, 1, n_classes)
        """
        n_train_env = len(self.ENVS)-1 if self.test_env is not None else len(self.ENVS)
        out_split = torch.zeros((n_train_env, self.batch_size, 1, out.shape[-1])).to(out.device)
        for i in range(n_train_env):
            # Test env is always the last one
            out_split[i,...] = out[:,i,...].unsqueeze(1)
            
        return out_split

    def split_labels(self, labels):
        """ Group data and prediction by environment

        Args:
            labels (Tensor): labels of the data (batch_size, len(PRED_TIME))

        Returns:
            Tensor: The reshaped labels (n_env-1, batch_size, 1)
        """
        n_train_env = len(self.ENVS)-1 if self.test_env is not None else len(self.ENVS)
        labels_split = torch.zeros((n_train_env, self.batch_size, 1)).long().to(labels.device)
        for i in range(n_train_env):
            # Test env is always the last one
            labels_split[i,...] = labels[:,i,...].unsqueeze(1)

        return labels_split

    def get_pred_time(self, X):
        return self.PRED_TIME[self.PRED_TIME < X[1]]

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
        return len(self.split)

    def __getitem__(self, idx):

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
    SETUP = 'source'
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
            in_split, out_split = get_split(full_dataset, flags.holdout_fraction, seed=j)
            full_dataset.close()

            # Make training dataset/loader and append it to training containers
            if j != flags.test_env:
                in_dataset = H5_dataset(os.path.join(flags.data_path, self.DATA_PATH), e, split=in_split)
                in_loader = InfiniteLoader(in_dataset, batch_size=training_hparams['batch_size'], num_workers=self.N_WORKERS, pin_memory=True)
                self.train_names.append(e + '_in')
                self.train_loaders.append(in_loader)
            
            # Make validation loaders
            fast_in_dataset = H5_dataset(os.path.join(flags.data_path, self.DATA_PATH), e, split=in_split)
            fast_in_loader = torch.utils.data.DataLoader(fast_in_dataset, batch_size=64, shuffle=False, num_workers=self.N_WORKERS, pin_memory=True)
            # fast_in_loader = torch.utils.data.DataLoader(fast_in_dataset, batch_size=256, shuffle=False, num_workers=self.N_WORKERS, pin_memory=True)
            fast_out_dataset = H5_dataset(os.path.join(flags.data_path, self.DATA_PATH), e, split=out_split)
            fast_out_loader = torch.utils.data.DataLoader(fast_out_dataset, batch_size=64, shuffle=False, num_workers=self.N_WORKERS, pin_memory=True)
            # fast_out_loader = torch.utils.data.DataLoader(fast_out_dataset, batch_size=256, shuffle=False, num_workers=self.N_WORKERS, pin_memory=True)

            # Append to val containers
            self.val_names.append(e + '_in')
            self.val_loaders.append(fast_in_loader)
            self.val_names.append(e + '_out')
            self.val_loaders.append(fast_out_loader)

        # Define loss function
        self.log_prob = nn.LogSoftmax(dim=1)
        self.loss_fn = nn.NLLLoss(weight=self.get_class_weight().to(training_hparams['device']), reduction='none')
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
    SEQ_LEN = 3000
    PRED_TIME = [2999]
    INPUT_SHAPE = [19]
    OUTPUT_SIZE = 6

    ## Dataset paths
    DATA_PATH = 'CAP/CAP.h5'

    ## Environment parameters
    ENVS = ['Machine0', 'Machine1', 'Machine2', 'Machine3', 'Machine4']
    SWEEP_ENVS = list(range(len(ENVS)))

    def __init__(self, flags, training_hparams):

        ## Define download function
        self.download_fct = download.download_cap

        super().__init__(flags, training_hparams)
        
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
    N_STEPS = 10001
    
    ## Dataset parameters
    TASK = 'classification'
    SEQ_LEN = 3000
    PRED_TIME = [2999]
    INPUT_SHAPE = [4]
    OUTPUT_SIZE = 6
    DATA_PATH = 'SEDFx/SEDFx.h5'

    ## Environment parameters
    ENVS = ['Age 20-40', 'Age 40-60', 'Age 60-80','Age 80-100']
    SWEEP_ENVS = list(range(len(ENVS)))

    def __init__(self, flags, training_hparams):

        ## Define download function
        self.download_fct = download.download_sedfx

        super().__init__(flags, training_hparams)
       
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
    SEQ_LEN = 752
    PRED_TIME = [751]
    INPUT_SHAPE = [48]
    OUTPUT_SIZE = 2
    DATA_PATH = 'PCL/PCL.h5'

    ## Environment parameters
    ENVS = [ 'PhysionetMI', 'Cho2017', 'Lee2019_MI']
    SWEEP_ENVS = list(range(len(ENVS)))

    def __init__(self, flags, training_hparams):

        ## Define download function
        self.download_fct = download.download_pcl

        super().__init__(flags, training_hparams)

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
    ## Training parameters
    N_STEPS = 5001
    CHECKPOINT_FREQ = 500

    ## Dataset parameters
    SETUP = 'source'
    TASK = 'classification'
    #:int: number of frames in each video
    SEQ_LEN = 20
    PRED_TIME = [19]
    INPUT_SHAPE = [3, 224, 224]
    OUTPUT_SIZE = 64
    #:str: path to the folder containing the data
    DATA_PATH = 'LSA64'

    ## Environment parameters
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
            datasets = []

            env_paths = []
            for speaker in e.split('-'):
                env_paths.append(os.path.join(flags.data_path, self.DATA_PATH, speaker))
            full_dataset = Video_dataset(env_paths, self.SEQ_LEN, transform=self.normalize)
            in_split, out_split = get_split(full_dataset, flags.holdout_fraction, seed=j)

            # Make training dataset/loader and append it to training containers
            if j != flags.test_env:
                in_dataset = Video_dataset(env_paths, self.SEQ_LEN, transform=self.normalize, split=in_split)
                in_loader = InfiniteLoader(in_dataset, batch_size=training_hparams['batch_size'], num_workers=self.N_WORKERS, pin_memory=True)
                self.train_names.append(e + '_in')
                self.train_loaders.append(in_loader)

            # Make validation loaders
            fast_in_dataset = Video_dataset(env_paths, self.SEQ_LEN, transform=self.normalize, split=in_split)
            fast_in_loader = torch.utils.data.DataLoader(fast_in_dataset, batch_size=16, shuffle=False, num_workers=self.N_WORKERS, pin_memory=True)
            # fast_in_loader = torch.utils.data.DataLoader(fast_in_dataset, batch_size=256, shuffle=False, num_workers=self.N_WORKERS, pin_memory=True)
            fast_out_dataset = Video_dataset(env_paths, self.SEQ_LEN, transform=self.normalize, split=out_split)
            fast_out_loader = torch.utils.data.DataLoader(fast_out_dataset, batch_size=16, shuffle=False, num_workers=self.N_WORKERS, pin_memory=True)
            # fast_out_loader = torch.utils.data.DataLoader(fast_out_dataset, batch_size=256, shuffle=False, num_workers=self.N_WORKERS, pin_memory=True)

            # Append to val containers
            self.val_names.append(e + '_in')
            self.val_loaders.append(fast_in_loader)
            self.val_names.append(e + '_out')
            self.val_loaders.append(fast_out_loader)

        # Define loss function
        self.log_prob = nn.LogSoftmax(dim=1)
        self.loss_fn = nn.NLLLoss(weight=self.get_class_weight().to(training_hparams['device']), reduction='none')
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
    SETUP = 'source'
    TASK = 'classification'
    SEQ_LEN = 500
    PRED_TIME = [499]
    INPUT_SHAPE = [6]
    OUTPUT_SIZE = 6
    #:str: Path to the file containing the data
    DATA_PATH = 'HHAR/HHAR.h5'

    ## Environment parameters
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
            in_dataset, out_dataset = make_split(full_dataset, flags.holdout_fraction, seed=j)

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

    axis: int = -1
    min_past: int = 0
    min_future: int = 0

    domain_idx: list = []
    
    num_instances: float
    total_length: int = 0
    n: int = 0

    def set_attribute(self, domain, set_holidays, start, max_length=0):
        
        min_increment = datetime.timedelta(minutes=30)
        day_increment = datetime.timedelta(days=1)
        running_time = start

        if domain == 'All':
            holidays_idx = []
            for idx in range(max_length):
                running_time += min_increment
                holidays_idx.append(idx)

            self.domain_idx = holidays_idx

        if domain == 'Holidays':
            holidays_idx = []
            for idx in range(max_length):
                running_time += min_increment
                if running_time in set_holidays:# or running_time + day_increment in set_holidays:
                    holidays_idx.append(idx)

            self.domain_idx = holidays_idx
        
        if domain == 'NonHolidays':
            non_holidays_idx = []
            for idx in range(max_length):
                running_time += min_increment
                if running_time not in set_holidays:# and running_time + day_increment not in set_holidays:
                    non_holidays_idx.append(idx)

            self.domain_idx = non_holidays_idx

    def _get_bounds(self, ts: np.ndarray) -> Tuple[int, int]:
        return (
            self.min_past,
            ts.shape[self.axis] - self.min_future,
        )

    def __call__(self, ts: np.ndarray) -> np.ndarray:
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

    def set_attribute(self, domain, set_holidays, start, max_length=0):
        
        min_increment = datetime.timedelta(minutes=30)
        day_increment = datetime.timedelta(days=1)
        running_time = start

        if domain == 'Holidays':
            holidays_idx = []
            for idx in range(int(max_length / 48)):
                running_time += day_increment
                if running_time in set_holidays:# or running_time + day_increment in set_holidays:
                    holidays_idx.append(idx * 48)

            self.domain_idx = holidays_idx
        
        if domain == 'NonHolidays':
            non_holidays_idx = []
            for idx in range(int(max_length / 48)):
                running_time += day_increment
                if running_time not in set_holidays:# and running_time.month == 7:# and running_time + day_increment not in set_holidays :
                    non_holidays_idx.append(idx * 48)

            self.domain_idx = non_holidays_idx

        self.domain_idx = np.array(self.domain_idx)
        self.domain_idx = self.domain_idx[self.domain_idx >= self.start_idx]
        self.domain_idx = self.domain_idx[self.domain_idx < self.last_idx]

    def _get_bounds(self, ts: np.ndarray) -> Tuple[int, int]:
        return (
            self.min_past,
            ts.shape[self.axis] - self.min_future,
        )

    def __call__(self, ts: np.ndarray) -> np.ndarray:
        a, b = self._get_bounds(ts)
        in_range_idx = self.domain_idx
        in_range_idx = in_range_idx[in_range_idx > a]
        in_range_idx = in_range_idx[in_range_idx <= b]
        window_size = len(in_range_idx)

        if window_size <= 0:
            return np.array([], dtype=int)

        return in_range_idx + a


class monthly_training_domain_sampler(InstanceSampler):

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
        
        min_increment = datetime.timedelta(minutes=30)
        day_increment = datetime.timedelta(days=1)
        running_time = start

        if domain == 'All':
            holidays_idx = []
            for idx in range(max_length):
                running_time += min_increment
                holidays_idx.append(idx)

            self.domain_idx = holidays_idx

        if domain == 'Holidays':
            holidays_idx = []
            for idx in range(max_length):
                running_time += min_increment
                if running_time in set_holidays:# or running_time + day_increment in set_holidays:
                    holidays_idx.append(idx)

            self.domain_idx = holidays_idx
        
        if domain != 'Holidays':
            month_ID = self.month_idx[domain]
            non_holidays_idx = []
            for idx in range(max_length):
                running_time += min_increment
                if running_time not in set_holidays and running_time.month == month_ID:
                    non_holidays_idx.append(idx)

            self.domain_idx = non_holidays_idx

    def _get_bounds(self, ts: np.ndarray) -> Tuple[int, int]:
        return (
            self.min_past,
            ts.shape[self.axis] - self.min_future,
        )

    def __call__(self, ts: np.ndarray) -> np.ndarray:
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

class monthly_evaluation_domain_sampler(InstanceSampler):

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
        
        min_increment = datetime.timedelta(minutes=30)
        day_increment = datetime.timedelta(days=1)
        running_time = start

        if domain == 'Holidays':
            holidays_idx = []
            for idx in range(int(max_length / 48)):
                running_time += day_increment
                if running_time in set_holidays:# or running_time + day_increment in set_holidays:
                    holidays_idx.append(idx * 48)

            self.domain_idx = holidays_idx
        
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
        return (
            self.min_past,
            ts.shape[self.axis] - self.min_future,
        )

    def __call__(self, ts: np.ndarray) -> np.ndarray:
        a, b = self._get_bounds(ts)
        in_range_idx = self.domain_idx
        in_range_idx = in_range_idx[in_range_idx > a]
        in_range_idx = in_range_idx[in_range_idx <= b]
        window_size = len(in_range_idx)

        if window_size <= 0:
            return np.array([], dtype=int)

        return in_range_idx + a

class ChristmasHolidays(holidays.HolidayBase):

    def _populate(self, year):
        self[datetime.date(year, 12, 23)] = "Pre-Christmas Eve"
        self[datetime.date(year, 12, 24)] = "Christmas Eve"
        self[datetime.date(year, 12, 25)] = "Christmas"
        # self[datetime.date(year, 12, 26)] = "Post-Christmas"
        # self[datetime.date(year, 12, 27)] = "Christmas Break 2"
        # self[datetime.date(year, 12, 28)] = "Christmas Break 3"
        # self[datetime.date(year, 12, 29)] = "Christmas Break 4"
        # self[datetime.date(year, 12, 30)] = "Christmas Break 5"
        self[datetime.date(year, 12, 31)] = "New Years eve"
        self[datetime.date(year, 1, 1)] = "New Years"
        # self[datetime.date(year, 1, 2)] = "Post-New Years"


class DummyHolidays(holidays.HolidayBase):

    def _populate(self, year):
        self[datetime.date(year, 5, 19)] = "New Years"
        self[datetime.date(year, 5, 20)] = "Post-New Years"
        self[datetime.date(year, 5, 21)] = "Christmas Break 5"
        self[datetime.date(year, 5, 22)] = "New Years eve"
        self[datetime.date(year, 5, 23)] = "Pre-Christmas Eve"
        self[datetime.date(year, 5, 24)] = "Christmas Eve"
        self[datetime.date(year, 5, 25)] = "Christmas"
        self[datetime.date(year, 5, 26)] = "Post-Christmas"
        self[datetime.date(year, 5, 26)] = "Christmas Break 1"
        self[datetime.date(year, 5, 27)] = "Christmas Break 2"
        self[datetime.date(year, 5, 28)] = "Christmas Break 3"
        self[datetime.date(year, 5, 29)] = "Christmas Break 4"

class AusElectricityUnbalanced(Multi_Domain_Dataset):

    # Training parameters
    N_STEPS = 3001
    CHECKPOINT_FREQ = 100

    ## Dataset parameters
    SETUP = 'subpopulation'
    TASK = 'forecasting'
    SEQ_LEN = 500
    INPUT_SIZE = 1
    OUTPUT_SIZE = 1
    FREQUENCY = '30T'
    PRED_LENGTH = 48

    ## Environment parameters
    ENVS = ['Holidays', 'NonHolidays']  # For training, we do not equally sample holidays and non holidays
    SWEEP_ENVS = [-1] # This is a subpopulation shift problem
    ENVS_WEIGHTS = [11./365, 354./365.]


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
        # self.set_holidays = holidays.country_holidays('AU')
        self.set_holidays = ChristmasHolidays()
        # self.set_holidays = DummyHolidays()

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
        train_first_idx = 157776 # Only for evaluation
        train_last_idx = 175296
        val_first_idx = train_last_idx
        val_last_idx = 192864
        test_first_idx = val_last_idx
        test_last_idx = 210384
        # train_first_idx = 175296 # Only for evaluation
        # train_last_idx = 192864
        # val_first_idx = train_last_idx
        # val_last_idx = 210384
        # test_first_idx = val_last_idx
        # test_last_idx = 227904

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

        self.train_loaders_iter = zip(*self.train_loaders)
        self.loss_fn = NegativeLogLikelihood()

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

    def get_number_of_batches(self):
        return self.num_batches_per_epoch

    def get_next_batch(self):

        batch = next(self.train_loaders_iter)
        # batch = {k: torch.cat((batch[0][k], batch[1][k]), dim=0) for k in batch[0]}

        return batch[0]

    def split_input(self, batch):

        return {k: batch[k].to(self.device) for k in batch}, batch['future_target'].to(self.device)

    def loss(self, X, Y):
        return self.loss_fn(X,Y)

class AusElectricity(Multi_Domain_Dataset):

    # Training parameters
    N_STEPS = 3001
    CHECKPOINT_FREQ = 100

    ## Dataset parameters
    SETUP = 'subpopulation'
    TASK = 'forecasting'
    SEQ_LEN = 500
    INPUT_SIZE = 1
    OUTPUT_SIZE = 1
    FREQUENCY = '30T'
    PRED_LENGTH = 48

    ## Environment parameters
    ENVS = ['Holidays', 'NonHolidays']
    SWEEP_ENVS = [-1] # This is a subpopulation shift problem
    ENVS_WEIGHTS = [11./365, 354./365.]

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
        # self.set_holidays = holidays.country_holidays('AU')
        self.set_holidays = ChristmasHolidays()
        # self.set_holidays = DummyHolidays()

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
        train_first_idx = 157776 # Only for evaluation
        train_last_idx = 175296
        val_first_idx = train_last_idx
        val_last_idx = 192864
        test_first_idx = val_last_idx
        test_last_idx = 210384
        # train_first_idx = 175296 # Only for evaluation
        # train_last_idx = 192864
        # val_first_idx = train_last_idx
        # val_last_idx = 210384
        # test_first_idx = val_last_idx
        # test_last_idx = 227904

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

    def get_number_of_batches(self):
        return self.num_batches_per_epoch

    def get_next_batch(self):

        batch = next(self.train_loaders_iter)
        batch = {k: torch.cat((batch[0][k], batch[1][k]), dim=0) for k in batch[0]}

        return batch

    def split_input(self, batch):

        return {k: batch[k].to(self.device) for k in batch}, batch['future_target'].to(self.device)

    def loss(self, X, Y):
        return self.loss_fn(X,Y)


class AusElectricityMonthly(Multi_Domain_Dataset):

    # Training parameters
    N_STEPS = 3001
    CHECKPOINT_FREQ = 100

    ## Dataset parameters
    SETUP = 'subpopulation'
    TASK = 'forecasting'
    SEQ_LEN = 500
    INPUT_SIZE = 1
    OUTPUT_SIZE = 1
    FREQUENCY = '30T'
    PRED_LENGTH = 48

    ## Environment parameters
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
        # self.set_holidays = ChristmasHolidays()
        # self.set_holidays = DummyHolidays()

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
        # train_first_idx = 157776 # Only for evaluation
        # train_last_idx = 175296
        # val_first_idx = train_last_idx
        # val_last_idx = 192864
        # test_first_idx = val_last_idx
        # test_last_idx = 210384
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

        self.train_loaders_iter = zip(*self.train_loaders)
        self.loss_fn = NegativeLogLikelihood()

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

        instance_sampler = monthly_evaluation_domain_sampler(
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

    def get_number_of_batches(self):
        return self.num_batches_per_epoch

    def get_next_batch(self):

        batch = next(self.train_loaders_iter)
        # batch = {k: torch.cat((batch[0][k], batch[1][k]), dim=0) for k in batch[0]}

        return batch[0]

    def split_input(self, batch):

        return {k: batch[k].to(self.device) for k in batch}, batch['future_target'].to(self.device)

    def loss(self, X, Y):
        return self.loss_fn(X,Y)


class AusElectricityMonthlyBalanced(Multi_Domain_Dataset):

    # Training parameters
    N_STEPS = 3001
    CHECKPOINT_FREQ = 100

    ## Dataset parameters
    SETUP = 'subpopulation'
    TASK = 'forecasting'
    SEQ_LEN = 500
    INPUT_SIZE = 1
    OUTPUT_SIZE = 1
    FREQUENCY = '30T'
    PRED_LENGTH = 48

    ## Environment parameters
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
        # self.set_holidays = ChristmasHolidays()
        # self.set_holidays = DummyHolidays()

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
        # train_first_idx = 157776 # Only for evaluation
        # train_last_idx = 175296
        # val_first_idx = train_last_idx
        # val_last_idx = 192864
        # test_first_idx = val_last_idx
        # test_last_idx = 210384
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

        instance_sampler = monthly_training_domain_sampler(
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

        instance_sampler = monthly_evaluation_domain_sampler(
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

    def get_number_of_batches(self):
        return self.num_batches_per_epoch

    def get_next_batch(self):

        batch = next(self.train_loaders_iter)
        batch = {k: torch.cat([batch[b][k] for b in range(len(batch))], dim=0) for k in batch[0]}

        return batch

    def split_input(self, batch):

        return {k: batch[k].to(self.device) for k in batch}, batch['future_target'].to(self.device)

    def loss(self, X, Y):
        return self.loss_fn(X,Y)

# Adding a Dataset

In this section, we will walk through the process of adding an dataset to the framework.

## Defining the Algorithm

We first define the dataset by creating a new class in the datasets module. In this example we will add flat_MNIST which is the MNIST dataset, but the image is fed to a sequential model pixel by pixel and the environments are different orders of the pixels.

First let's define the dataset class and its __init__ method. 
```python
class flat_MNIST(Multi_Domain_Dataset):
    """ Class for flat MNIST dataset

    Each sample is a sequence of 784 pixels.
    The task is to predict the digit

    Args:
        flags (argparse.Namespace): argparse of training arguments

    Note:
        The MNIST dataset needs to be downloaded, this is automaticaly done if the dataset isn't in the given data_path
    """
    PRED_TIME = [783]
    INPUT_SHAPE = [1]
    OUTPUT_SIZE = 10
    ENVS = ['forwards', 'backwards', 'scrambled']
    SETUP = 'seq'

    def __init__(self, flags, training_hparams):
        super().__init__()

        if flags.test_env is not None:
            assert flags.test_env < len(self.ENVS), "Test environment chosen is not valid"
        else:
            warnings.warn("You don't have any test environment")

        # Save stuff
        self.test_env = flags.test_env
        self.class_balance = training_hparams['class_balance']
        self.batch_size = training_hparams['batch_size']

        ## Import original MNIST data
        MNIST_tfrm = transforms.Compose([ transforms.ToTensor() ])

        # Get MNIST data
        train_ds = datasets.MNIST(flags.data_path, train=True, download=True, transform=MNIST_tfrm) 
        test_ds = datasets.MNIST(flags.data_path, train=False, download=True, transform=MNIST_tfrm) 

        # Concatenate all data and labels
        MNIST_images = torch.cat((train_ds.data.float(), test_ds.data.float()))
        MNIST_labels = torch.cat((train_ds.targets, test_ds.targets))

        # Create sequences of 784 pixels
        self.TCMNIST_images = MNIST_images.reshape(-1, 28*28, 1)
        self.TCMNIST_labels = TCMNIST_labels.long()

        # Make the color datasets
        self.train_names, self.train_loaders = [], [] 
        self.val_names, self.val_loaders = [], [] 
        for i, e in enumerate(self.ENVS):

            # Choose data subset
            images = self.TCMNIST_images[i::len(self.ENVS),...]
            labels = self.TCMNIST_labels[i::len(self.ENVS),...]

            # Apply environment definition
            if e == 'forwards':
                images = images[:, ::1, :]
            elif e == 'backwards':
                images = images[:, ::-1, :]
            elif e == 'scrambled':
                images = images[:, torch.randperm(28*28), :]

            # Make Tensor dataset and the split
            dataset = torch.utils.data.TensorDataset(iamges, labels)
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
```
Note: 
you are required to define the following variables:
    * SETUP
    * PRED_TIME
    * INPUT_SHAPE
    * OUTPUT_SIZE
    * ENVS
you are also encouraged to redefine the following variables:
    * N_STEPS
    * N_WORKERS
    * CHECKPOINT_FREQ

## Adding necessary pieces
Now that our algorithm is defined, we can add it to the list of algorithms at the top of the objectives module.
```python
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
    ## Financial Dataset
    "StockVolatility",
    ## Sign Recognition
    "LSA64",
    ## Activity Recognition
    "HAR",
    ## Example
    "flat_MNIST",
]
```
Before being able to use the dataset, we need to add the hyper parameters related to this dataset in the hyperparams module. Note: the name of the funtion needs to be the same as the name of the dataset followed by _train and _model.
```python
def flat_MNIST_train(sample):
    """ flat_MNIST model hparam definition 
    
    Args:
        sample (bool): If ''True'', hyper parameters are gonna be sampled randomly according to their given distributions. Defaults to ''False'' where the default value is chosen.
    """
    if sample:
        return {
            'class_balance': lambda r: True,
            'weight_decay': lambda r: 0.,
            'lr': lambda r: 10**r.uniform(-4.5, -2.5),
            'batch_size': lambda r: int(2**r.uniform(3, 9))
        }
    else:
        return {
            'class_balance': lambda r: True,
            'weight_decay': lambda r: 0,
            'lr': lambda r: 1e-3,
            'batch_size': lambda r: 64
        }

def flat_MNIST_model(sample):
    """ flat_MNIST model hparam definition 
    
    Args:
        sample (bool): If ''True'', hyper parameters are gonna be sampled randomly according to their given distributions. Defaults to ''False'' where the default value is chosen.
    """
    if sample:
        return {
            'model': lambda r: 'LSTM',
            'hidden_depth': lambda r: int(r.choice([1, 2, 3])),
            'hidden_width': lambda r: int(2**r.uniform(5, 7)),
            'recurrent_layers': lambda r: int(r.choice([1, 2, 3])),
            'state_size': lambda r: int(2**r.uniform(5, 7))
        }
    else:
        return {
            'model': lambda r: 'LSTM',
            'hidden_depth': lambda r: 1, 
            'hidden_width': lambda r: 20,
            'recurrent_layers': lambda r: 2,
            'state_size': lambda r: 32
        }
```
## Run some tests
We can now run a simple test to check that everything is working as expected
```sh
Coming soon...
```
## Try the algorithm
Then we can run a training run to see how algorithms performs on your dataset
```sh
python3 -m woods.main train \
        --dataset flat_MNIST \
        --objective ERM \
        --test_env 0 \
        --data_path ./data
```
## Run a sweep
Finally, we can run a sweep to see how the algorithms performs on your dataset
```sh
python3 -m woods.main sweep \
        --dataset flat_MNIST \
        --data_path ./data
```
"""Defining the architectures used for benchmarking algorithms"""

import math

import torch
from torch import nn
from torchvision import models

import numpy as np
import matplotlib.pyplot as plt

# new package
from braindecode.models import ShallowFBCSPNet, Deep4Net, EEGResNet

def get_model(dataset, model_hparams):
    """Return the dataset class with the given name
    
    Args:
        dataset (str): name of the dataset
        model_hparams (dict): model hyperparameters 
    """
    if model_hparams['model'] not in globals():
        raise NotImplementedError("Dataset not found: {}".format(model_hparams['model']))

    model_fn = globals()[model_hparams['model']]

    return model_fn(dataset, model_hparams)


class shallow(nn.Module):
    # ref: https://github.com/braindecode/braindecode/tree/master/braindecode/models
    
    def __init__(self, dataset, model_hparams):
        super(shallow, self).__init__()

        # Save stuff
        self.input_size = np.prod(dataset.INPUT_SHAPE)
        self.output_size = dataset.OUTPUT_SIZE
        self.seq_len = dataset.SEQ_LEN

        self.model = ShallowFBCSPNet(
        self.input_size,
        self.output_size,
        input_window_samples=self.seq_len,
        final_conv_length='auto',
        )
        
    def forward(self, input, time_pred):
        out = self.model(input.permute((0, 2, 1)))
        return out.unsqueeze(1)

class EEGResNet_bd(nn.Module):
    # ref: https://github.com/braindecode/braindecode/tree/master/braindecode/models
    
    def __init__(self, dataset, model_hparams):
        super(EEGResNet_bd, self).__init__()

        # Save stuff
        self.input_size = np.prod(dataset.INPUT_SHAPE)
        self.output_size = dataset.OUTPUT_SIZE
        self.seq_len = dataset.SEQ_LEN

        self.model = EEGResNet(
        self.input_size,
        self.output_size,
        input_window_samples=self.seq_len,
        n_first_filters=12, #TBF
        n_layers_per_block=2,
        final_pool_length = 'auto'
        # final_conv_length='auto',
        )
        
    def forward(self, input, time_pred):
        out = self.model(input.permute((0, 2, 1)))
        return out.unsqueeze(1)

class deep4(nn.Module):
    # ref: https://github.com/braindecode/braindecode/tree/master/braindecode/models
    
    def __init__(self, dataset, model_hparams):
        super(deep4, self).__init__()

        # Save stuff
        self.input_size = np.prod(dataset.INPUT_SHAPE)
        self.output_size = dataset.OUTPUT_SIZE
        self.seq_len = dataset.SEQ_LEN

        self.model = Deep4Net(
        self.input_size,
        self.output_size,
        input_window_samples=self.seq_len,
        final_conv_length='auto',
        n_filters_time=32,
        n_filters_spat=32,
        filter_time_length=10,
        pool_time_length=3,
        pool_time_stride=3,
        n_filters_2=64,
        filter_length_2=10,
        n_filters_3=128,
        filter_length_3=10,
        n_filters_4=256,
        filter_length_4=10
        )
        
    def forward(self, input, time_pred):
        out = self.model(input.permute((0, 2, 1)))
        return out.unsqueeze(1)

class MNIST_CNN(nn.Module):
    """ Hand-tuned architecture for extracting representation from MNIST images

    This was adapted from :

        https://github.com/facebookresearch/DomainBed
    """
    EMBED_DIM = 32
    CNN_OUT_DIM = 32*3*3

    def __init__(self, input_shape):
        super(MNIST_CNN, self).__init__()

        # Make CNN
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 8, 3, 1, padding=1),
            nn.Conv2d(8, 32, 3, stride=2, padding=1),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 3, 1, padding=1),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 3, 1, padding=1),
        )

        self.FCC = nn.Sequential(
            nn.Linear(self.CNN_OUT_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

    def forward(self, x):
        """ Forward pass through the model

        Args:
            x (torch.Tensor): The input to the model.

        Returns:
            torch.Tensor: The output of the model.
        """
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.FCC(x)
        return x

class LSTM(nn.Module):
    """ A simple LSTM model

    Args:
        dataset (Multi_Domain_Dataset): dataset that we will be training on
        model_hparams (dict): The hyperparameters for the model.
        input_size (int, optional): The size of the input to the model. Defaults to None. If None, the input size is calculated from the dataset.

    Attributes:
        state_size (int): The size of the hidden state of the LSTM.
        recurrent_layers (int): The number of recurrent layers stacked on each other.
        hidden_depth (int): The number of hidden layers of the classifier MLP (after LSTM).
        hidden_width (int): The width of the hidden layers of the classifier MLP (after LSTM).
    
    Notes:
        All attributes need to be in the model_hparams dictionary.
    """
    def __init__(self, dataset, model_hparams, input_size=None):
        super(LSTM, self).__init__()

        ## Save stuff
        # Model parameters
        self.state_size = model_hparams['state_size']
        self.hidden_depth = model_hparams['hidden_depth']
        self.hidden_width = model_hparams['hidden_width']
        self.recurrent_layers = model_hparams['recurrent_layers']

        # Dataset parameters
        self.input_size = np.prod(dataset.INPUT_SHAPE) if input_size is None else input_size
        self.output_size = dataset.OUTPUT_SIZE

        ## Recurrent model
        self.lstm = nn.LSTM(self.input_size, self.state_size, self.recurrent_layers, batch_first=True)

        ## Classification model
        layers = []
        if self.hidden_depth == 0:
            layers.append( nn.Linear(self.state_size, self.output_size) )
        else:
            layers.append( nn.Linear(self.state_size, self.hidden_width) )
            for i in range(self.hidden_depth-1):
                layers.append( nn.Linear(self.hidden_width, self.hidden_width) )
            layers.append( nn.Linear(self.hidden_width, self.output_size) )
        
        seq_arr = []
        for i, lin in enumerate(layers):
            seq_arr.append(lin)
            if i != self.hidden_depth:
                seq_arr.append(nn.ReLU(True))
        self.classifier = nn.Sequential(*seq_arr)

    def forward(self, input, time_pred):
        """ Forward pass of the model

        Args:
            input (torch.Tensor): The input to the model.
            time_pred (torch.Tensor): The time prediction of the input.

        Returns:
            torch.Tensor: The output of the model.
        """

        # Setup array
        pred = torch.zeros(input.shape[0], 0).to(input.device)
        hidden = self.initHidden(input.shape[0], input.device)

        # Forward propagate LSTM
        input = input.view(input.shape[0], input.shape[1], -1)
        out, hidden = self.lstm(input, hidden)

        # Make prediction with fully connected
        all_out = torch.zeros((input.shape[0], time_pred.shape[0], self.output_size)).to(input.device)
        for i, t in enumerate(time_pred):
            output = self.classifier(out[:,t,:])
            all_out[:,i,...] = output

        return all_out

    def initHidden(self, batch_size, device):
        """ Initialize the hidden state of the LSTM with a normal distribution

        Args:
            batch_size (int): The batch size of the model.
            device (torch.device): The device to use.
        """
        return (torch.randn(self.recurrent_layers, batch_size, self.state_size).to(device), 
                torch.randn(self.recurrent_layers, batch_size, self.state_size).to(device))

class MNIST_LSTM(nn.Module):
    """ A simple LSTM model taking inputs from a CNN. (see: MNIST_CNN)

    Args:
        dataset (Multi_Domain_Dataset): dataset that we will be training on
        model_hparams (dict): The hyperparameters for the model.
        input_size (int, optional): The size of the input to the model. Defaults to None. If None, the input size is calculated from the dataset.

    Attributes:
        state_size (int): The size of the hidden state of the LSTM.
        recurrent_layers (int): The number of recurrent layers stacked on each other.
        hidden_depth (int): The number of hidden layers of the classifier MLP (after LSTM).
        hidden_width (int): The width of the hidden layers of the classifier MLP (after LSTM).
    
    Notes:
        All attributes need to be in the model_hparams dictionary.
    """
    def __init__(self, dataset, model_hparams, input_size=None):
        super(MNIST_LSTM, self).__init__()

        ## Save stuff
        # Model parameters
        self.state_size = model_hparams['state_size']
        self.hidden_depth = model_hparams['hidden_depth']
        self.hidden_width = model_hparams['hidden_width']
        self.recurrent_layers = model_hparams['recurrent_layers']

        # Dataset parameters
        self.input_size = np.prod(dataset.INPUT_SHAPE) if input_size is None else input_size
        self.output_size = dataset.OUTPUT_SIZE

        # Make CNN
        self.conv = MNIST_CNN(input_shape=dataset.INPUT_SHAPE)
        
        ## Recurrent model
        self.lstm = nn.LSTM(self.conv.EMBED_DIM, self.state_size, self.recurrent_layers, batch_first=True)

        ## Classification model
        layers = []
        if self.hidden_depth == 0:
            layers.append( nn.Linear(self.state_size, self.output_size) )
        else:
            layers.append( nn.Linear(self.state_size, self.hidden_width) )
            for i in range(self.hidden_depth-1):
                layers.append( nn.Linear(self.hidden_width, self.hidden_width) )
            layers.append( nn.Linear(self.hidden_width, self.output_size) )
        
        seq_arr = []
        for i, lin in enumerate(layers):
            seq_arr.append(lin)
            if i != self.hidden_depth:
                seq_arr.append(nn.ReLU(True))
        self.classifier = nn.Sequential(*seq_arr)

    def forward(self, input, time_pred):
        """ Forward pass of the model

        Args:
            input (torch.Tensor): The input to the model.
            time_pred (torch.Tensor): The time prediction of the input.

        Returns:
            torch.Tensor: The output of the model.
        """

        # Setup array
        pred = torch.zeros(input.shape[0], 0).to(input.device)
        hidden = self.initHidden(input.shape[0], input.device)

        # Forward through the MNIST_CNN
        cnn_embed_seq = torch.zeros((input.shape[0], input.shape[1], self.conv.EMBED_DIM)).to(input.device)
        for i in range(input.shape[1]):
            cnn_embed_seq[:,i,:] = self.conv(input[:,i,:,:,:])

        # Forward propagate LSTM
        out, hidden = self.lstm(cnn_embed_seq, hidden)

        # Make prediction with fully connected
        all_out = torch.zeros((input.shape[0], time_pred.shape[0], self.output_size)).to(input.device)
        for i, t in enumerate(time_pred):
            output = self.classifier(out[:,t,:])
            all_out[:,i,...] = output

        return all_out

    def initHidden(self, batch_size, device):
        """ Initialize the hidden state of the LSTM with a normal distribution

        Args:
            batch_size (int): The batch size of the model.
            device (torch.device): The device to use.
        """
        return (torch.randn(self.recurrent_layers, batch_size, self.state_size).to(device), 
                torch.randn(self.recurrent_layers, batch_size, self.state_size).to(device))

class ATTN_LSTM(nn.Module):
    """ A simple LSTM model with self attention

    Args:
        dataset (Multi_Domain_Dataset): dataset that we will be training on
        model_hparams (dict): The hyperparameters for the model.
        input_size (int, optional): The size of the input to the model. Defaults to None. If None, the input size is calculated from the dataset.

    Attributes:
        state_size (int): The size of the hidden state of the LSTM.
        recurrent_layers (int): The number of recurrent layers stacked on each other.
        hidden_depth (int): The number of hidden layers of the classifier MLP (after LSTM).
        hidden_width (int): The width of the hidden layers of the classifier MLP (after LSTM).

    Notes:
        All attributes need to be in the model_hparams dictionary.
    """
    def __init__(self, dataset, model_hparams, input_size=None):
        super(ATTN_LSTM, self).__init__()

        ## Save stuff
        # Model parameters
        self.state_size = model_hparams['state_size']
        self.recurrent_layers = model_hparams['recurrent_layers']
        self.hidden_depth = model_hparams['hidden_depth']
        self.hidden_width = model_hparams['hidden_width']

        # Dataset parameters
        self.input_size = np.prod(dataset.INPUT_SHAPE) if input_size is None else input_size
        self.output_size = dataset.OUTPUT_SIZE

        # Recurrent model
        self.lstm = nn.LSTM(self.input_size, self.state_size, self.recurrent_layers, batch_first=True, dropout=0.2)

        # attention model
        layers = []
        layers.append(nn.Linear(self.state_size, self.state_size))
        seq_arr = []
        for i, lin in enumerate(layers):
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
            seq_arr.append(lin)
            seq_arr.append(nn.Tanh())
        self.attn = nn.Sequential(*seq_arr)
        self.sm = nn.Softmax(dim=1)
        
        # Classification model
        layers = []
        if self.hidden_depth == 0:
            layers.append( nn.Linear(self.state_size, self.output_size) )
        else:
            layers.append( nn.Linear(self.state_size, self.hidden_width) )
            for i in range(self.hidden_depth-1):
                layers.append( nn.Linear(self.hidden_width, self.hidden_width) )
            layers.append( nn.Linear(self.hidden_width, self.output_size) )
        
        seq_arr = []
        for i, lin in enumerate(layers):
            seq_arr.append(lin)
            if i != self.hidden_depth:
                seq_arr.append(nn.ReLU(True))
        seq_arr.append(nn.LogSoftmax(dim=1))
        self.classifier = nn.Sequential(*seq_arr)

    def forward(self, input, time_pred):
        """ Forward pass of the model

        Args:
            input (torch.Tensor): The input to the model.
            time_pred (torch.Tensor): The time prediction of the input.

        Returns:
            torch.Tensor: The output of the model.
        """

        # Setup array
        pred = torch.zeros(input.shape[0], 0).to(input.device)
        hidden = self.initHidden(input.shape[0], input.device)

        # Forward propagate LSTM
        out, hidden = self.lstm(input, hidden)

        # attn_scores = torch.zeros_like(out)        
        # for i in range(out.shape[1]):
        #     attn_scores[:,i,:] = self.attn(out[:,i,:])
        attn_scores = self.attn(out)
        attn_scores = self.sm(attn_scores)

        out = torch.mul(out, attn_scores).sum(dim=1)

        # Make prediction with fully connected
        output = self.classifier(out)

        # Unsqueze to make a single time dimension
        output = output.unsqueeze(1)
        
        return output

    def initHidden(self, batch_size, device):
        """ Initialize the hidden state of the LSTM with a normal distribution

        Args:
            batch_size (int): The batch size of the model.
            device (torch.device): The device to use.
        """
        return (torch.randn(self.recurrent_layers, batch_size, self.state_size).to(device), 
                torch.randn(self.recurrent_layers, batch_size, self.state_size).to(device))

class shallow(nn.Module):
    """ The Shallow Net model

    This is from the Braindecode package:
        https://github.com/braindecode/braindecode
    
    Args:
        dataset (Multi_Domain_Dataset): dataset that we will be training on
        model_hparams (dict): The hyperparameters for the model.
        input_size (int, optional): The size of the input to the model. Defaults to None. If None, the input size is calculated from the dataset.

    Attributes:
        seq_len (int): The length of the sequences.
    """
    # ref: https://github.com/braindecode/braindecode
    
    def __init__(self, dataset, model_hparams, input_size=None):
        super(shallow, self).__init__()

        ## Save stuff
        # Dataset parameters
        self.input_size = np.prod(dataset.INPUT_SHAPE) if input_size is None else input_size
        self.output_size = dataset.OUTPUT_SIZE
        self.seq_len = dataset.SEQ_LEN

        # Define model
        self.model = ShallowFBCSPNet(
        self.input_size,
        self.output_size,
        input_window_samples=self.seq_len,
        final_conv_length='auto',
        )
        
    def forward(self, input, time_pred):
        """ Forward pass of the model

        Args:
            input (torch.Tensor): The input to the model.
            time_pred (torch.Tensor): The time prediction of the input.
        """
        out = self.model(input.permute((0, 2, 1)))
        return out.unsqueeze(1)

class CRNN(nn.Module):
    """ Convolutional Recurrent Neural Network

    This is taken inspired from the repository:
        https://github.com/HHTseng/video-classification/

    But here we use the ResNet50 architecture pretrained on ImageNet, and we use the ATTN_LSTM model on top of the outputs of the ResNet50 to make predictions.

    Args:
        dataset (Multi_Domain_Dataset): dataset that we will be training on
        model_hparams (dict): The hyperparameters for the model.

    Attributes:
        fc_hidden1 (int): The size of the first hidden layer of the CNN embedding.
        fc_hidden2 (int): The size of the second hidden layer of the CNN embedding.
        CNN_embed_dim (int): The size of the CNN embedding.
    """
    def __init__(self, dataset, model_hparams, input_size=None):
        """ Initialize CRNN
        Args:
            input_size: int, size of input
            output_size: int, size of output
            model_hparams: dict, model hyperparameters
        """
        super(CRNN, self).__init__()

        ## Save stuff
        # Model parameters
        self.fc_hidden1, self.fc_hidden2 = model_hparams['fc_hidden']
        self.CNN_embed_dim = model_hparams['CNN_embed_dim']
        # Data parameters
        self.input_size = np.prod(dataset.INPUT_SHAPE) if input_size is None else input_size
        self.output_size = dataset.OUTPUT_SIZE

        # Define Resnet model
        resnet = models.resnet50(pretrained=True)
        self.n_outputs = resnet.fc.in_features
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        for param in self.resnet.parameters():
            param.requires_grad = False

        # Define CNN embedding
        self.cnn_fc = nn.Sequential(
            nn.Linear(resnet.fc.in_features, self.fc_hidden1),
            nn.BatchNorm1d(self.fc_hidden1, momentum=0.01),
            nn.Linear(self.fc_hidden1, self.fc_hidden2),
            nn.BatchNorm1d(self.fc_hidden2, momentum=0.01),
            nn.Linear(self.fc_hidden2, self.CNN_embed_dim),
        )

        # Define recurrent layers
        self.lstm = ATTN_LSTM(dataset, model_hparams, self.CNN_embed_dim)

    def forward(self, input, time_pred):
        """ Forward pass through CRNN
        Args:
            input: Tensor, shape [batch_size, seq_len, input_size]
            time_pred: Tensor, time prediction indexes
        """
        # Pass through Resnet
        cnn_embed_seq = torch.zeros((input.shape[0], input.shape[1], self.CNN_embed_dim)).to(input.device)
        for t in range(input.size(1)):
            # ResNet CNN
            with torch.no_grad():
                x = self.resnet(input[:,t,...])  # ResNet
                x = x.view(x.size(0), -1)        # flatten output of conv

            # FC layers
            x = self.cnn_fc(x)

            cnn_embed_seq[:,t,:] = x

        # # swap time and sample dim such that (sample dim, time dim, CNN latent dim)
        # cnn_embed_seq = torch.stack(cnn_embed_seq, dim=1)

        # Pass through recurrent layers
        out = self.lstm(cnn_embed_seq, time_pred)
        
        return out

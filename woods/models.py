"""Defining the architectures used for benchmarking algorithms"""

import math

import torch
from torch import nn
from torchvision import models

# new package
from braindecode.models import ShallowFBCSPNet

def get_model(dataset, model_hparams):
    """Return the dataset class with the given name
    
    Args:
        dataset (str): name of the dataset
        model_hparams (dict): model hyperparameters 
    """
    if model_hparams['model'] not in globals():
        raise NotImplementedError("Dataset not found: {}".format(model_hparams['model']))

    model_fn = globals()[model_hparams['model']]

    return model_fn(dataset,
                    model_hparams)

class LSTM(nn.Module):
    """ A simple LSTM model

    Args:
        input_shape (tuple): The shape of the input data.
        output_size (int): The size of the output.
        model_hparams (dict): The hyperparameters for the model.
    """
    def __init__(self, dataset, model_hparams):
        super(LSTM, self).__init__()

        # Save stuff
        self.state_size = model_hparams['state_size']
        self.hidden_depth = model_hparams['hidden_depth']
        self.hidden_width = model_hparams['hidden_width']
        self.recurrent_layers = model_hparams['recurrent_layers']

        self.input_size = math.prod(dataset.INPUT_SHAPE)
        self.output_size = dataset.OUTPUT_SIZE

        # Recurrent model
        self.lstm = nn.LSTM(self.input_size, self.state_size, self.recurrent_layers, batch_first=True)

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
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
            seq_arr.append(lin)
            if i != self.hidden_depth:
                seq_arr.append(nn.ReLU(True))
        seq_arr.append(nn.LogSoftmax(dim=1))
        self.classifier = nn.Sequential(*seq_arr)

    def forward(self, input, time_pred):

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
        return (torch.randn(self.recurrent_layers, batch_size, self.state_size).to(device), 
                torch.randn(self.recurrent_layers, batch_size, self.state_size).to(device))

class ATTN_LSTM(nn.Module):
    def __init__(self, dataset, model_hparams):
        super(ATTN_LSTM, self).__init__()

        # Save stuff
        self.state_size = model_hparams['state_size']
        self.hidden_depth = model_hparams['hidden_depth']
        self.hidden_width = model_hparams['hidden_width']
        self.recurrent_layers = model_hparams['recurrent_layers']
        self.output_size = dataset.OUTPUT_SIZE
        self.input_size = math.prod(dataset.INPUT_SHAPE)

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
            layers.append( nn.Linear(self.state_size, output_size) )
        else:
            layers.append( nn.Linear(self.state_size, self.hidden_width) )
            for i in range(self.hidden_depth-1):
                layers.append( nn.Linear(self.hidden_width, self.hidden_width) )
            layers.append( nn.Linear(self.hidden_width, output_size) )
        
        seq_arr = []
        for i, lin in enumerate(layers):
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
            seq_arr.append(lin)
            if i != self.hidden_depth:
                seq_arr.append(nn.ReLU(True))
        seq_arr.append(nn.LogSoftmax(dim=1))
        self.classifier = nn.Sequential(*seq_arr)

    def forward(self, input, time_pred):

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
        return (torch.randn(self.recurrent_layers, batch_size, self.state_size).to(device), 
                torch.randn(self.recurrent_layers, batch_size, self.state_size).to(device))

class shallow(nn.Module):
    # ref: https://github.com/braindecode/braindecode/tree/master/braindecode/models
    
    def __init__(self, dataset, model_hparams):
        super(shallow, self).__init__()

        # Save stuff
        self.input_size = math.prod(dataset.INPUT_SHAPE)
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

class CRNN(nn.Module):
    """ Convolutional Recurrent Neural Network
    https://github.com/HHTseng/video-classification/blob/99ebf204f0b1d737e38bc0d8b65aca128a57d7b1/ResNetCRNN/functions.py#L308
    """
    def __init__(self, input_size, output_size, model_hparams):
        """ Initialize CRNN
        Args:
            input_size: int, size of input
            output_size: int, size of output
            model_hparams: dict, model hyperparameters
        """
        super(CRNN, self).__init__()

        # Save stuff
        self.input_size = math.prod(dataset.INPUT_SHAPE)
        fc_hidden1, fc_hidden2 = model_hparams['fc_hidden']
        self.CNN_embed_dim = model_hparams['CNN_embed_dim']

        # Define Resnet model
        # self.network = torchvision.models.resnet50(pretrained=True)
        resnet = models.resnet50(pretrained=True)
        self.n_outputs = resnet.fc.in_features
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        for param in self.resnet.parameters():
            param.requires_grad = False

        # Define CNN embedding
        self.cnn_fc = nn.Sequential(
            nn.Linear(resnet.fc.in_features, fc_hidden1),
            nn.BatchNorm1d(fc_hidden1, momentum=0.01),
            nn.Linear(fc_hidden1, fc_hidden2),
            nn.BatchNorm1d(fc_hidden2, momentum=0.01),
            nn.Linear(fc_hidden2, self.CNN_embed_dim),
        )

        # Define recurrent layers
        self.lstm = ATTN_LSTM(self.CNN_embed_dim, output_size, model_hparams)
        # nn.LSTM(CNN_embed_dim, model_hparams['state_size'], model_hparams['recurrent_layers'], batch_first=True)

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
        out, pred = self.lstm(cnn_embed_seq, time_pred)

        return out.unsqueeze(1)

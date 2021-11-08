"""Defining the architectures used for benchmarking algorithms"""

import math

import torch
from torch import nn
from torchvision import models

# new package
from braindecode.models import EEGResNet
from torchsummary import summary

def get_model(dataset, dataset_hparams):
    """Return the dataset class with the given name."""
    if dataset_hparams['model'] not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_hparams['model']))

    model_fn = globals()[dataset_hparams['model']]

    return model_fn(dataset.INPUT_SHAPE, 
                    dataset.OUTPUT_SIZE,
                    dataset_hparams)

class LSTM(nn.Module):
    """ A simple LSTM model

    Args:
        input_shape (tuple): The shape of the input data.
        output_size (int): The size of the output.
        model_hparams (dict): The hyperparameters for the model.
    """
    def __init__(self, input_shape, output_size, model_hparams):
        super(LSTM, self).__init__()

        # Save stuff
        self.state_size = model_hparams['state_size']
        self.hidden_depth = model_hparams['hidden_depth']
        self.hidden_width = model_hparams['hidden_width']
        self.recurrent_layers = model_hparams['recurrent_layers']

        self.input_size = math.prod(input_shape)
        self.output_size = output_size

        # Recurrent model
        self.lstm = nn.LSTM(self.input_size, self.state_size, self.recurrent_layers, batch_first=True)

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
    def __init__(self, input_shape, output_size, model_hparams):
        super(ATTN_LSTM, self).__init__()

        # Save stuff
        self.state_size = model_hparams['state_size']
        self.hidden_depth = model_hparams['hidden_depth']
        self.hidden_width = model_hparams['hidden_width']
        self.recurrent_layers = model_hparams['recurrent_layers']
        self.output_size = output_size
        self.input_size = math.prod(input_shape)

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

class EEGResnet(nn.Module):
    def __init__(self, input_shape, output_size, model_hparams):
        super(EEGResnet, self).__init__()

        # Save stuff
        self.output_size = output_size
        self.input_size = math.prod(input_shape)

        # Get model from BrainDecode
        print("hello")
        self.model = EEGResNet( in_chans=self.input_size,
                                n_classes=self.output_size,
                                input_window_samples=3000,
                                final_pool_length='auto',
                                n_first_filters=10)
        summary(self.model.cuda(), input_size=(3000, input_shape[0]))
        print(self.model)

    def forward(self, input, time_pred):

        print(input.shape)
        out = self.model(input)
        print(out.shape)

        return out.unsqueeze(1)

class PositionalEncoding(nn.Module):
    """ Positional Encoding class

    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0.0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0.0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """ Apply positional embedding to incoming Torch Tensor
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:x.size(1), :].detach()
        return self.dropout(x)

class shallow(nn.Module):
    # ref: https://github.com/braindecode/braindecode/tree/master/braindecode/models
    
    def __init__(self, input_size, output_size, model_hparams):
        super(shallow, self).__init__()
        from braindecode.models import ShallowFBCSPNet
        input_window_samples = 750 # lenght of each trial TODO: should be a parameter

        self.model = ShallowFBCSPNet(
        input_size,
        output_size,
        input_window_samples=input_window_samples,
        final_conv_length='auto',
        )
        
    def forward(self, input, time_pred):
        out = self.model(input.permute((0, 2, 1)))
        return [out], out.argmax(1, keepdim=True)

class Transformer(nn.Module):
    # Do this : https://assets.amazon.science/11/88/6e046cba4241a06e536cc50584b2/gated-transformer-for-decoding-human-brain-eeg-signals.pdf
    """Transformer for EEG

    Args:
        nn ([type]): [description]

    TODO:
        * Adaptive pooling to be able to use this with any input length (time)
        * Find a way to add time embedding to the input without it not working
        * Check model size to make it possible to overfit to the SEDFx dataset
    """

    def __init__(self, input_shape, output_size, model_hparams):
        super(Transformer, self).__init__()

        # Save stuff
        self.input_size = math.prod(input_shape)
        self.embedding_size = model_hparams['embedding_size']

        # Define encoding layers
        self.pos_encoder = PositionalEncoding(model_hparams['embedding_size'])
        enc_layer = nn.TransformerEncoderLayer(d_model=model_hparams['embedding_size'], nhead=model_hparams['nheads_enc'])
        # enc_layer = GatedTransformerEncoderLayer(d_model=model_hparams['embedding_size'], nhead=model_hparams['nheads_enc'])
        self.enc_layers = nn.TransformerEncoder(encoder_layer=enc_layer, num_layers=model_hparams['nlayers_enc'])

        # Classifier
        n_conv_chs = 16
        time_conv_size = 50
        max_pool_size = 12
        pad_size = 25
        # spatial conv
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(1, model_hparams['embedding_size'], (1, self.input_size))
        )
        self.feature_extractor = nn.Sequential(
            # temporal conv 1
            nn.Conv2d(1, n_conv_chs, (time_conv_size, 1), padding=(pad_size, 0)),
            nn.BatchNorm2d(n_conv_chs),
            nn.GELU(),
            nn.MaxPool2d((max_pool_size, 1)),
            # temporal conv 2
            nn.Conv2d(
                n_conv_chs, n_conv_chs*2, (time_conv_size, 1),
                padding=(pad_size, 0)),
            nn.BatchNorm2d(n_conv_chs*2),
            nn.GELU(),
            # nn.MaxPool2d((max_pool_size, 1)),
            nn.AdaptiveMaxPool2d(max_pool_size),
            # temporal conv 2
            nn.Conv2d(
                n_conv_chs*2, n_conv_chs*4, (time_conv_size, 1),
                padding=(pad_size, 0)),
            nn.BatchNorm2d(n_conv_chs*4),
            nn.GELU(),
            # nn.MaxPool2d((max_pool_size, 1))
            nn.AdaptiveMaxPool2d(max_pool_size)
        )
        self.classifier = nn.Sequential(
            nn.Linear(9216, 128), # TODO: in_features should works for all EEG datasets
            nn.GELU(),
            nn.Linear(128, output_size),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, input, time_pred):
        """
        Args:
            input (Tensor): [batch_size, time_steps, input_size]
            time_pred (Tensor): [batch_size, time_steps]
        Returns:
            output (Tensor): [batch_size, 1, output_size]
        """

        # Pass through attention heads
        # out = self.embedding(input)
        out = input.unsqueeze(1)
        out = self.spatial_conv(out)
        out = out.transpose(1,3).squeeze()
        # out = self.pos_encoder(out)
        print(out.shape)
        out = self.enc_layers(out)
        out = out.unsqueeze(1)
        out = self.feature_extractor(out)
        out = out.view(out.shape[0], -1)
        out = self.classifier(out)

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
        self.input_size = input_size
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

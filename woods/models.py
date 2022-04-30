"""Defining the architectures used for benchmarking algorithms"""

import math
import copy

import torch
from torch import nn
from torchvision import models

import numpy as np
import matplotlib.pyplot as plt

from typing import NamedTuple, Optional, Iterable, Dict, Any, List, Tuple

# new package
from braindecode.models import ShallowFBCSPNet, Deep4Net, EEGResNet, EEGNetv4

from gluonts.torch.modules.feature import FeatureEmbedder
from gluonts.torch.modules.scaler import MeanScaler, NOPScaler

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


################
## EEG Models ##
################
class deep4(nn.Module):
    """ The DEEP4 model

    This is from the Braindecode package:
        https://github.com/braindecode/braindecode
    
    Args:
        dataset (Multi_Domain_Dataset): dataset that we will be training on
        model_hparams (dict): The hyperparameters for the model.

    Attributes:
        input_size (int): The size of the inputs to the model (for a single time step).
        output_size (int): The size of the outputs of the model (number of classes).
        seq_len (int): The length of the sequences.
    """
    
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

        # Delete undesired layers
        self.classifier = copy.deepcopy(self.model.conv_classifier)
        del self.model.conv_classifier
        del self.model.softmax
        del self.model.squeeze
        
    def forward(self, input):

        # Forward pass
        features = self.model(input.permute((0, 2, 1)))
        out = self.classifier(features)

        # Remove all extra dimension and Add the time prediction dimension
        out, features = torch.flatten(out, start_dim=1), torch.flatten(features, start_dim=1)
        out, features = out.unsqueeze(1), features.unsqueeze(1)

        return out, features

class EEGNet(nn.Module):
    """ The EEGNet model

    This is a really small model ~3k parameters.

    This is from the Braindecode package:
        https://github.com/braindecode/braindecode
    
    Args:
        dataset (Multi_Domain_Dataset): dataset that we will be training on
        model_hparams (dict): The hyperparameters for the model.

    Attributes:
        input_size (int): The size of the inputs to the model (for a single time step).
        output_size (int): The size of the outputs of the model (number of classes).
        seq_len (int): The length of the sequences.
    """
    
    def __init__(self, dataset, model_hparams):
        super(EEGNet, self).__init__()

        # Save stuff
        self.input_size = np.prod(dataset.INPUT_SHAPE)
        self.output_size = dataset.OUTPUT_SIZE
        self.seq_len = dataset.SEQ_LEN

        scale = 1
        self.model = EEGNetv4(
            self.input_size,
            self.output_size,
            input_window_samples=self.seq_len,
            final_conv_length='auto',
            F1=8*scale,
            D=2*scale,
            F2=16*scale*scale, #usually set to F1*D (?)
            kernel_length=64*scale,
            third_kernel_size=(8, 4),
            drop_prob=0.05,
        )


        self.classifier = nn.Sequential(
            self.model.conv_classifier,
            self.model.permute_back
        )
        del self.model.conv_classifier
        del self.model.softmax
        del self.model.permute_back
        del self.model.squeeze
        
    def forward(self, input):

        # Forward pass
        features = self.model(input.permute((0, 2, 1)))
        out = self.classifier(features)

        # Remove all extra dimension and Add the time prediction dimension
        out, features = torch.flatten(out, start_dim=1), torch.flatten(features, start_dim=1)
        out, features = out.unsqueeze(1), features.unsqueeze(1)

        return out, features

class MNIST_CNN(nn.Module):
    """ Hand-tuned architecture for extracting representation from MNIST images

    This was adapted from :
        https://github.com/facebookresearch/DomainBed

    In our context, it is used to extract the representation from the images which are fed to a recurrent model such as an LSTM

    Args:
        dataset (Multi_Domain_Dataset): dataset that we will be training on
        model_hparams (dict): The hyperparameters for the model.
        input_size (int, optional): The size of the input to the model. Defaults to None. If None, the input size is calculated from the dataset.
    """
    #:int: Size of the output respresentation
    EMBED_DIM = 32
    #:int: Size of the representation after convolution, but before FCC layers
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

        # Make FCC layers
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
            torch.Tensor: The output representation of the model.
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
        self.time_pred = dataset.PRED_TIME

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

    def forward(self, input):
        """ Forward pass of the model

        Args:
            input (torch.Tensor): The input to the model.

        Returns:
            torch.Tensor: The output of the model.
        """

        # Setup array
        hidden = self.initHidden(input.shape[0], input.device)

        # Forward propagate LSTM
        input = input.view(input.shape[0], input.shape[1], -1)
        features, hidden = self.lstm(input, hidden)

        # Make prediction with fully connected
        all_out = torch.zeros((input.shape[0], self.time_pred.shape[0], self.output_size)).to(input.device)
        all_features = torch.zeros((input.shape[0], self.time_pred.shape[0], features.shape[-1])).to(input.device)
        for i, t in enumerate(self.time_pred):
            output = self.classifier(features[:,t,:])
            all_out[:,i,...] = output
            all_features[:,i,...] = features[:,t,...]

        return all_out, all_features

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
        self.time_pred = dataset.PRED_TIME

        # Make CNN
        self.conv = MNIST_CNN(input_shape=dataset.INPUT_SHAPE)
        
        ## Recurrent model
        self.lstm = LSTM(dataset, model_hparams, input_size=self.conv.EMBED_DIM)

    def forward(self, input):
        """ Forward pass of the model

        Args:
            input (torch.Tensor): The input to the model.
            time_pred (torch.Tensor): The time prediction of the input.

        Returns:
            torch.Tensor: The output of the model.
        """

        # Forward through the MNIST_CNN
        cnn_embed_seq = torch.zeros((input.shape[0], input.shape[1], self.conv.EMBED_DIM)).to(input.device)
        for i in range(input.shape[1]):
            cnn_embed_seq[:,i,:] = self.conv(input[:,i,:,:,:])

        # Forward propagate LSTM
        out, features = self.lstm(cnn_embed_seq, self.time_pred)

        return out, features

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
        self.time_pred = dataset.PRED_TIME

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
        self.classifier = nn.Sequential(*seq_arr)

    def forward(self, input):
        """ Forward pass of the model

        Args:
            input (torch.Tensor): The input to the model.
            time_pred (torch.Tensor): The time prediction of the input.

        Returns:
            torch.Tensor: The output of the model.
        """

        # Setup array
        hidden = self.initHidden(input.shape[0], input.device)

        # Forward propagate LSTM
        features, hidden = self.lstm(input, hidden)

        # Get attention scores
        attn_scores = self.attn(features)
        attn_scores = self.sm(attn_scores)

        # get linear combination of features with attention scores
        features = torch.mul(features, attn_scores).sum(dim=1)

        # Make prediction with fully connected
        output = self.classifier(features)

        # Unsqueze to make a single time dimension
        output = output.unsqueeze(1)
        features = features.unsqueeze(1)
        
        return output, features

    def initHidden(self, batch_size, device):
        """ Initialize the hidden state of the LSTM with a normal distribution

        Args:
            batch_size (int): The batch size of the model.
            device (torch.device): The device to use.
        """
        return (torch.randn(self.recurrent_layers, batch_size, self.state_size).to(device), 
                torch.randn(self.recurrent_layers, batch_size, self.state_size).to(device))

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

    def forward(self, input):
        """ Forward pass through CRNN
        Args:
            input: Tensor, shape [batch_size, seq_len, input_size]
            time_pred: Tensor, time prediction indexes
        """

        ## Pass through resnet
        out = input.view(input.shape[0]*input.shape[1], *input.shape[2:])
        out = self.resnet(out)
        out = out.view(out.shape[0], -1)
        out = self.cnn_fc(out)
        out = out.view(input.shape[0], input.shape[1], -1)

        # Pass through recurrent layers
        out, features = self.lstm(out, self.time_pred)
        
        return out, features

class ForecastingTransformer(nn.Module):
    def __init__(
        self,
        dataset, model_hparams, input_size=None
        # freq: str,
        # context_length: int,
        # prediction_length: int,
        # num_feat_dynamic_real: int,
        # num_feat_static_real: int,
        # num_feat_static_cat: int,
        # cardinality: List[int],
        
        # # transformer arguments
        # nhead: int,
        # num_encoder_layers: int,
        # num_decoder_layers: int,
        # dim_feedforward: int,
        # activation: str = "gelu",
        # dropout: float = 0.1,

        # # univariate input
        # input_size: int = 1,
        # embedding_dimension: Optional[List[int]] = None,
        # distr_output: DistributionOutput = StudentTOutput(),
        # lags_seq: Optional[List[int]] = None,
        # scaling: bool = True,
        # num_parallel_samples: int = 100,
    ) -> None:
        super().__init__()
        
        self.input_size = input_size
       
        self.target_shape = dataset.distr_output.event_shape
        
        self.dim_feat_dynamic_real = 1 + dataset.num_feat_dynamic_real + len(dataset.time_features)
        self.dim_feat_static_real = max(1, dataset.num_feat_static_real)
        self.dim_feat_static_cat = max(1, dataset.num_feat_static_cat)

        self.embedding_dimension = dataset.embedding_dimension
        self.lags_seq = dataset.lags_seq
        self.num_parallel_samples = model_hparams['num_parallel_samples']
        self.embedder = FeatureEmbedder(
            cardinalities=dataset.cardinality,
            embedding_dims=self.embedding_dimension,
        )
        if model_hparams['scaling']:
            self.scaler = MeanScaler(dim=1, keepdim=True)
        else:
            self.scaler = NOPScaler(dim=1, keepdim=True)
        
        # total feature size
        d_model = dataset.INPUT_SIZE * len(self.lags_seq) + self._number_of_features
        
        self.context_length = dataset.context_length
        self.prediction_length = dataset.PRED_LENGTH
        self.distr_output = dataset.distr_output
        self.param_proj = dataset.distr_output.get_args_proj(d_model)
            
        # transformer enc-decoder and mask initializer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=model_hparams['nhead'],
            num_encoder_layers=model_hparams['num_encoder_layers'],
            num_decoder_layers=model_hparams['num_decoder_layers'],
            dim_feedforward=model_hparams['dim_feedforward'],
            dropout=model_hparams['dropout'],
            activation=model_hparams['activation'],
            batch_first=True,
        )
        
        # causal decoder tgt mask
        self.register_buffer(
            "tgt_mask",
            self.transformer.generate_square_subsequent_mask(self.prediction_length),
        )
        
    @property
    def _number_of_features(self) -> int:
        return (
            sum(self.embedding_dimension)
            + self.dim_feat_dynamic_real
            + self.dim_feat_static_real
            + 1  # the log(scale)
        )

    @property
    def _past_length(self) -> int:
        return self.context_length + max(self.lags_seq)
    
    def get_lagged_subsequences(
        self,
        sequence: torch.Tensor,
        subsequences_length: int,
        shift: int = 0
    ) -> torch.Tensor:
        """
        Returns lagged subsequences of a given sequence.
        Parameters
        ----------
        sequence : Tensor
            the sequence from which lagged subsequences should be extracted.
            Shape: (N, T, C).
        subsequences_length : int
            length of the subsequences to be extracted.
        shift: int
            shift the lags by this amount back.
        Returns
        --------
        lagged : Tensor
            a tensor of shape (N, S, C, I), where S = subsequences_length and
            I = len(indices), containing lagged subsequences. Specifically,
            lagged[i, j, :, k] = sequence[i, -indices[k]-S+j, :].
        """
        sequence_length = sequence.shape[1]
        indices = [l - shift for l in self.lags_seq]

        assert max(indices) + subsequences_length <= sequence_length, (
            f"lags cannot go further than history length, found lag {max(indices)} "
            f"while history length is only {sequence_length}"
        )

        lagged_values = []
        for lag_index in indices:
            begin_index = -lag_index - subsequences_length
            end_index = -lag_index if lag_index > 0 else None
            lagged_values.append(sequence[:, begin_index:end_index, ...])
        return torch.stack(lagged_values, dim=-1)

    def _check_shapes(
        self,
        prior_input: torch.Tensor,
        inputs: torch.Tensor,
        features: Optional[torch.Tensor],
    ) -> None:
        assert len(prior_input.shape) == len(inputs.shape)
        assert (
            len(prior_input.shape) == 2 and self.input_size == 1
        ) or prior_input.shape[2] == self.input_size
        assert (len(inputs.shape) == 2 and self.input_size == 1) or inputs.shape[
            -1
        ] == self.input_size
        assert (
            features is None or features.shape[2] == self._number_of_features
        ), f"{features.shape[2]}, expected {self._number_of_features}"
    
    
    def create_network_inputs(
        self, 
        feat_static_cat: torch.Tensor, 
        feat_static_real: torch.Tensor,
        past_time_feat: torch.Tensor,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        future_time_feat: Optional[torch.Tensor] = None,
        future_target: Optional[torch.Tensor] = None,
    ):        
        # time feature
        time_feat = (
            torch.cat(
                (
                    past_time_feat[:, self._past_length - self.context_length :, ...],
                    future_time_feat,
                ),
                dim=1,
            )
            if future_target is not None
            else past_time_feat[:, self._past_length - self.context_length :, ...]
        )

        # target
        context = past_target[:, -self.context_length :]
        observed_context = past_observed_values[:, -self.context_length :]
        _, scale = self.scaler(context, observed_context)

        inputs = (
            torch.cat((past_target, future_target), dim=1) / scale
            if future_target is not None
            else past_target / scale
        )

        inputs_length = (
            self._past_length + self.prediction_length
            if future_target is not None
            else self._past_length
        )
        assert inputs.shape[1] == inputs_length
        
        subsequences_length = (
            self.context_length + self.prediction_length
            if future_target is not None
            else self.context_length
        )
        
        # embeddings
        embedded_cat = self.embedder(feat_static_cat)
        static_feat = torch.cat(
            (embedded_cat, feat_static_real, scale.log()),
            dim=1,
        )
        expanded_static_feat = static_feat.unsqueeze(1).expand(
            -1, time_feat.shape[1], -1
        )
        
        features = torch.cat((expanded_static_feat, time_feat), dim=-1)
        
        # Lagged
        lagged_sequence = self.get_lagged_subsequences(
            sequence=inputs,
            subsequences_length=subsequences_length,
        )

        lags_shape = lagged_sequence.shape
        reshaped_lagged_sequence = lagged_sequence.reshape(
            lags_shape[0], lags_shape[1], -1
        )

        transformer_inputs = torch.cat((reshaped_lagged_sequence, features), dim=-1)
        
        return transformer_inputs, scale, static_feat
    
    def output_params(self, transformer_inputs):
        enc_input = transformer_inputs[:, :self.context_length, ...]
        dec_input = transformer_inputs[:, self.context_length:, ...]
        
        enc_out = self.transformer.encoder(
            enc_input
        )

        dec_output = self.transformer.decoder(
            dec_input,
            enc_out,
            tgt_mask=self.tgt_mask
        )

        return self.param_proj(dec_output)

    @torch.jit.ignore
    def output_distribution(
        self, params, scale=None, trailing_n=None
    ) -> torch.distributions.Distribution:
        sliced_params = params
        if trailing_n is not None:
            sliced_params = [p[:, -trailing_n:] for p in params]
        return self.distr_output.distribution(sliced_params, scale=scale)
    
    def forward(
        self,
        batch
    ):

        feat_static_cat = batch["feat_static_cat"]
        feat_static_real = batch["feat_static_real"]

        past_time_feat = batch["past_time_feat"]
        past_target = batch["past_target"]
        past_observed_values = batch["past_observed_values"]

        future_time_feat = batch["future_time_feat"]
        future_target = batch["future_target"]
        future_observed_values = batch["future_observed_values"]
        
        
        transformer_inputs, scale, static_feat = self.create_network_inputs(
            feat_static_cat,
            feat_static_real,
            past_time_feat,
            past_target,
            past_observed_values,
            future_time_feat,
            future_target,
        )

        params = self.output_params(transformer_inputs)
        distr = self.output_distribution(params, scale)

        return distr, future_target

    # for prediction
    def inference(
        self,
        batch,
        # feat_static_cat: torch.Tensor,
        # feat_static_real: torch.Tensor,
        # past_time_feat: torch.Tensor,
        # past_target: torch.Tensor,
        # past_observed_values: torch.Tensor,
        # future_time_feat: torch.Tensor,
        num_parallel_samples: Optional[int] = None,
    ) -> torch.Tensor:

        feat_static_cat = batch["feat_static_cat"]
        feat_static_real = batch["feat_static_real"]

        past_time_feat = batch["past_time_feat"]
        past_target = batch["past_target"]
        past_observed_values = batch["past_observed_values"]

        future_time_feat = batch["future_time_feat"]
        future_target = batch["future_target"]
        future_observed_values = batch["future_observed_values"]
        
        
        if num_parallel_samples is None:
            num_parallel_samples = self.num_parallel_samples
            
        encoder_inputs, scale, static_feat = self.create_network_inputs(
            feat_static_cat,
            feat_static_real,
            past_time_feat,
            past_target,
            past_observed_values,
        )

        enc_out = self.transformer.encoder(encoder_inputs)
        
        repeated_scale = scale.repeat_interleave(
            repeats=self.num_parallel_samples, dim=0
        )

        repeated_past_target = (
            past_target.repeat_interleave(
                repeats=self.num_parallel_samples, dim=0
            )
            / repeated_scale
        )
        
        expanded_static_feat = static_feat.unsqueeze(1).expand(
            -1, future_time_feat.shape[1], -1
        )
        features = torch.cat((expanded_static_feat, future_time_feat), dim=-1)
        repeated_features = features.repeat_interleave(
            repeats=self.num_parallel_samples, dim=0
        )
       
        repeated_enc_out = enc_out.repeat_interleave(
            repeats=self.num_parallel_samples, dim=0
        )

        future_samples = []
        
        # greedy decoding
        for k in range(self.prediction_length):            
            #self._check_shapes(repeated_past_target, next_sample, next_features)
            #sequence = torch.cat((repeated_past_target, next_sample), dim=1)
            
            lagged_sequence = self.get_lagged_subsequences(
                sequence=repeated_past_target,
                subsequences_length=1+k,
                shift=1, 
            )

            lags_shape = lagged_sequence.shape
            reshaped_lagged_sequence = lagged_sequence.reshape(
                lags_shape[0], lags_shape[1], -1
            )
            
            decoder_input = torch.cat((reshaped_lagged_sequence, repeated_features[:, : k+1]), dim=-1)

            output = self.transformer.decoder(decoder_input, repeated_enc_out)
            
            params = self.param_proj(output[:,-1:])
            distr = self.output_distribution(params, scale=repeated_scale)
            next_sample = distr.sample()
            
            repeated_past_target = torch.cat(
                (repeated_past_target, next_sample / repeated_scale), dim=1
            )
            future_samples.append(next_sample)

        concat_future_samples = torch.cat(future_samples, dim=1)
        return concat_future_samples.reshape(
            (-1, self.num_parallel_samples, self.prediction_length)
            + self.target_shape,
        )
import math

import torch
from torch import nn
from torchvision import models

# To remove 
import matplotlib.pyplot as plt

def get_model(dataset, dataset_hparams):
    """Return the dataset class with the given name."""
    if dataset_hparams['model'] not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_hparams['model']))

    model_fn = globals()[dataset_hparams['model']]

    return model_fn(dataset.get_input_size(), 
                    dataset.get_output_size(),
                    dataset_hparams)

class RNN(nn.Module):
    def __init__(self, input_size, output_size, model_hparams):
        super(RNN, self).__init__()

        # Save stuff
        self.state_size = model_hparams['state_size']
        self.hidden_depth = model_hparams['hidden_depth']
        self.hidden_width = model_hparams['hidden_width']

        ## Construct the part of the RNN in charge of the hidden state
        H_layers = []
        if self.hidden_depth == 0:
            H_layers.append( nn.Linear(input_size + self.state_size, self.state_size) )
        else:
            H_layers.append( nn.Linear(input_size + self.state_size, self.hidden_width) )
            for i in range(self.hidden_depth-1):
                H_layers.append( nn.Linear(self.hidden_width, self.hidden_width) )
            H_layers.append( nn.Linear(self.hidden_width, self.state_size) )
        
        seq_arr = []
        for i, lin in enumerate(H_layers):
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
            seq_arr.append(lin)
            if i != self.hidden_depth:
                seq_arr.append(nn.ReLU(True))
        self.FCH = nn.Sequential(*seq_arr)

        ## Construct the part of the model in charge of the output
        O_layers = []
        if self.hidden_depth == 0:
            O_layers.append( nn.Linear(input_size + self.state_size, output_size) )
        else:
            O_layers.append( nn.Linear(input_size + self.state_size, self.hidden_width) )
            for i in range(self.hidden_depth-1):
                O_layers.append( nn.Linear(self.hidden_width, self.hidden_width) )
            O_layers.append( nn.Linear(self.hidden_width, output_size) )
        
        seq_arr = []
        for i, lin in enumerate(O_layers):
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
            seq_arr.append(lin)
            if i != self.hidden_depth:
                seq_arr.append(nn.ReLU(True))
        seq_arr.append(nn.LogSoftmax(dim=1))
        self.FCO = nn.Sequential(*seq_arr)

    def forward(self, input, time_pred):

        # Setup array
        all_out = []
        pred = torch.zeros(input.shape[0], 0).to(input.device)
        hidden = self.initHidden(input.shape[0], input.device)

        # Forward propagate RNN
        for t in range(input.shape[1]):
            combined = torch.cat((input[:,t,...].view(input.shape[0],-1), hidden), 1)
            hidden = self.FCH(combined)
            out = self.FCO(combined)
            if t in time_pred:
                all_out.append(out)
                pred = torch.cat((pred, out.argmax(1, keepdim=True)), dim=1)

        return all_out, pred

    def initHidden(self, batch_size, device):
        return torch.zeros(batch_size, self.state_size).to(device)

class LSTM(nn.Module):
    def __init__(self, input_size, output_size, model_hparams):
        super(LSTM, self).__init__()

        # Save stuff
        self.state_size = model_hparams['state_size']
        self.hidden_depth = model_hparams['hidden_depth']
        self.hidden_width = model_hparams['hidden_width']
        self.recurrent_layers = model_hparams['recurrent_layers']

        # Recurrent model
        self.lstm = nn.LSTM(input_size, self.state_size, self.recurrent_layers, batch_first=True)

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
        all_out = []
        pred = torch.zeros(input.shape[0], 0).to(input.device)
        hidden = self.initHidden(input.shape[0], input.device)

        # Forward propagate LSTM
        input = input.view(input.shape[0], input.shape[1], -1)
        out, hidden = self.lstm(input, hidden)

        # Make prediction with fully connected
        for t in time_pred:
            output = self.classifier(out[:,t,:])
            all_out.append(output)
            pred = torch.cat((pred, output.argmax(1, keepdim=True)), dim=1)
        
        return all_out, pred

    def initHidden(self, batch_size, device):
        return (torch.randn(self.recurrent_layers, batch_size, self.state_size).to(device), 
                torch.randn(self.recurrent_layers, batch_size, self.state_size).to(device))

class ATTN_LSTM(nn.Module):
    def __init__(self, input_size, output_size, model_hparams):
        super(ATTN_LSTM, self).__init__()

        self.state_size = model_hparams['state_size']
        self.hidden_depth = model_hparams['hidden_depth']
        self.hidden_width = model_hparams['hidden_width']
        self.recurrent_layers = model_hparams['recurrent_layers']

        # Recurrent model
        self.lstm = nn.LSTM(input_size, self.state_size, self.recurrent_layers, batch_first=True, dropout=0.2)

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
        all_out = []
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
        all_out.append(output)
        pred = torch.cat((pred, output.argmax(1, keepdim=True)), dim=1)
        
        return all_out, pred

    def initHidden(self, batch_size, device):
        return (torch.randn(self.recurrent_layers, batch_size, self.state_size).to(device), 
                torch.randn(self.recurrent_layers, batch_size, self.state_size).to(device))

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

class Transformer(nn.Module):
    # Do this : https://assets.amazon.science/11/88/6e046cba4241a06e536cc50584b2/gated-transformer-for-decoding-human-brain-eeg-signals.pdf

    def __init__(self, input_size, output_size, model_hparams):
        super(Transformer, self).__init__()

        # Save stuff
        self.input_size = input_size
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
            nn.MaxPool2d((max_pool_size, 1)),
            # temporal conv 2
            nn.Conv2d(
                n_conv_chs*2, n_conv_chs*4, (time_conv_size, 1),
                padding=(pad_size, 0)),
            nn.BatchNorm2d(n_conv_chs*4),
            nn.GELU(),
            nn.MaxPool2d((max_pool_size, 1))
        )
        self.classifier = nn.Sequential(
            nn.Linear(2048, 128),
            nn.GELU(),
            nn.Linear(128, output_size),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, input, time_pred):

        # Pass through attention heads
        # out = self.embedding(input)
        out = input.unsqueeze(1)
        out = self.spatial_conv(out)
        out = out.transpose(1,3).squeeze()
        # out = self.pos_encoder(out)
        out = self.enc_layers(out)
        out = out.unsqueeze(1)
        out = self.feature_extractor(out)
        out = out.view(out.shape[0], -1)
        out = self.classifier(out)

        return [out], out.argmax(1, keepdim=True)

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

        # # Define classifier
        # self.classifier = nn.Sequential(
        #     nn.Linear( model_hparams['state_size'], model_hparams['hidden_width']),
        #     nn.ReLU(),
        #     nn.Linear(model_hparams['hidden_width'], output_size),
        #     nn.LogSoftmax(dim=1)
        # )
    
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

        return out, pred

# class GatedTransformerEncoderLayer(nn.Module):
#     r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
#     This standard encoder layer is based on the paper "Attention Is All You Need".
#     Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
#     Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
#     Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
#     in a different way during application.

#     Args:
#         d_model: the number of expected features in the input (required).
#         nhead: the number of heads in the multiheadattention models (required).
#         dim_feedforward: the dimension of the feedforward network model (default=2048).
#         dropout: the dropout value (default=0.1).
#         activation: the activation function of intermediate layer, relu or gelu (default=relu).

#     Examples::
#         >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
#         >>> src = torch.rand(10, 32, 512)
#         >>> out = encoder_layer(src)
#     """

#     def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
#         super(GatedTransformerEncoderLayer, self).__init__()
#         self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
#         # Implementation of Feedforward model
#         self.linear1 = nn.Linear(d_model, dim_feedforward)
#         self.dropout = nn.Dropout(dropout)
#         self.linear2 = nn.Linear(dim_feedforward, d_model)

#         self.gate1 = nn.GRUCell(d_model, d_model)
#         self.gate2 = nn.GRUCell(d_model, d_model)

#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.dropout1 = nn.Dropout(dropout)
#         self.dropout2 = nn.Dropout(dropout)

#         self.activation = nn.functional.relu

#     def __setstate__(self, state):
#         if 'activation' not in state:
#             state['activation'] = F.relu
#         super(GatedTransformerEncoderLayer, self).__setstate__(state)

#     def forward(self, src, src_mask=None, src_key_padding_mask=None):
#         r"""Pass the input through the encoder layer.

#         Args:
#             src: the sequence to the encoder layer (required).
#             src_mask: the mask for the src sequence (optional).
#             src_key_padding_mask: the mask for the src keys per batch (optional).

#         Shape:
#             see the docs in Transformer class.
#         """
#         # First part
#         src2 = self.norm1(src)
#         src2 = self.self_attn(src2, src2, src2, attn_mask=src_mask,
#                               key_padding_mask=src_key_padding_mask)[0]
#         # src = src + self.dropout1(src2)
#         src = self.gate1(
#             rearrange(src, 'b n d -> (b n) d'),
#             rearrange(src2, 'b n d -> (b n) d'),
#         )
#         src = rearrange(src, '(b n) d -> b n d', b = src2.shape[0])

#         # Second part
#         src2 = self.norm2(src)
#         src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
#         # src = src + self.dropout2(src2)
#         src = self.gate2(
#             rearrange(src, 'b n d -> (b n) d'),
#             rearrange(src2, 'b n d -> (b n) d'),
#         )
#         src = rearrange(src, '(b n) d -> b n d', b = src2.shape[0])

#         return src
import os
import argparse
import numpy as np

import torch
import torch.nn.functional as F
from torch import nn, optim
from torchvision import datasets, transforms

class RNN(nn.Module):
    def __init__(self, input_size, hidden_depth, hidden_width, state_size, output_size):
        super(RNN, self).__init__()

        self.state_size = state_size

        ## Construct the part of the RNN in charge of the hidden state
        H_layers = []
        if hidden_depth == 0:
            H_layers.append( nn.Linear(input_size + state_size, state_size) )
        else:
            H_layers.append( nn.Linear(input_size + state_size, hidden_width) )
            for i in range(hidden_depth-1):
                H_layers.append( nn.Linear(hidden_width, hidden_width) )
            H_layers.append( nn.Linear(hidden_width, state_size) )
        
        seq_arr = []
        for i, lin in enumerate(H_layers):
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
            seq_arr.append(lin)
            if i != hidden_depth:
                seq_arr.append(nn.ReLU(True))
        self.FCH = nn.Sequential(*seq_arr)

        ## Construct the part of the model in charge of the output
        O_layers = []
        if hidden_depth == 0:
            O_layers.append( nn.Linear(input_size + state_size, output_size) )
        else:
            O_layers.append( nn.Linear(input_size + state_size, hidden_width) )
            for i in range(hidden_depth-1):
                O_layers.append( nn.Linear(hidden_width, hidden_width) )
            O_layers.append( nn.Linear(hidden_width, output_size) )
        
        seq_arr = []
        for i, lin in enumerate(O_layers):
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
            seq_arr.append(lin)
            if i != hidden_depth:
                seq_arr.append(nn.ReLU(True))
        seq_arr.append(nn.LogSoftmax(dim=1))
        self.FCO = nn.Sequential(*seq_arr)

    def forward(self, input, hidden):
        combined = torch.cat((input.view(input.shape[0],-1), hidden), 1)
        hidden = self.FCH(combined)
        output = self.FCO(combined)
        return output, hidden

    def initHidden(self, batch_size, device):
        return torch.zeros(batch_size, self.state_size).to(device)

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_depth, hidden_width, recurrent_layers, state_size, output_size):
        super(LSTM, self).__init__()

        self.state_size = state_size
        self.hidden_depth = hidden_depth
        self.recurrent_layers = recurrent_layers

        # Recurrent model
        self.lstm = nn.LSTM(input_size, state_size, recurrent_layers, batch_first=True, dropout=0.2)

        # Classification model
        layers = []
        if hidden_depth == 0:
            layers.append( nn.Linear(state_size, output_size) )
        else:
            layers.append( nn.Linear(state_size, hidden_width) )
            for i in range(hidden_depth-1):
                layers.append( nn.Linear(hidden_width, hidden_width) )
            layers.append( nn.Linear(hidden_width, output_size) )
        
        seq_arr = []
        for i, lin in enumerate(layers):
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
            seq_arr.append(lin)
            if i != hidden_depth:
                seq_arr.append(nn.ReLU(True))
        seq_arr.append(nn.LogSoftmax(dim=1))
        self.classifier = nn.Sequential(*seq_arr)

    def forward(self, input, hidden):
        out, hidden = self.lstm(torch.unsqueeze(input, 1), hidden)
        output = self.classifier(torch.squeeze(out))
        return output, hidden

    def initHidden(self, batch_size, device):
        return (torch.randn(self.recurrent_layers, batch_size, self.state_size).to(device), 
                torch.randn(self.recurrent_layers, batch_size, self.state_size).to(device))

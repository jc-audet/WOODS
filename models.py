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
            O_layers.append( nn.Linear(input_size + state_size, state_size) )
        else:
            O_layers.append( nn.Linear(input_size + state_size, hidden_width) )
            for i in range(hidden_depth-1):
                O_layers.append( nn.Linear(hidden_width, hidden_width) )
            O_layers.append( nn.Linear(hidden_width, state_size) )
        
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

    def initHidden(self, batch_size):
        return torch.zeros(batch_size, self.state_size)
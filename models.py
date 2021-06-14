import os
import argparse
import numpy as np

import torch
import torch.nn.functional as F
from torch import nn, optim
from torchvision import datasets, transforms



class RNN(nn.Module):
    def __init__(self, input_size, state_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        lin1 = nn.Linear(input_size + hidden_size, state_size)
        lin2 = nn.Linear(state_size, state_size)
        lin3 = nn.Linear(state_size, hidden_size)
        for lin in [lin1, lin2, lin3]:
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
        self.FCH = nn.Sequential(lin1, nn.ReLU(True), lin2, nn.ReLU(True), lin3)


        lin4 = nn.Linear(input_size + hidden_size, state_size)
        lin5 = nn.Linear(state_size, state_size)
        lin6 = nn.Linear(state_size, 2)
        for lin in [lin4, lin5, lin6]:
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
        self.FCO = nn.Sequential(lin4, nn.ReLU(True), lin5, nn.ReLU(True), lin6, nn.LogSoftmax(dim=1))


    def forward(self, input, hidden):
        combined = torch.cat((input.view(input.shape[0],-1), hidden), 1)
        hidden = self.FCH(combined)
        output = self.FCO(combined)
        return output, hidden

    def initHidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size)
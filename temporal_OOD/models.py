import torch
from torch import nn

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

        # Forward propagate LSTM
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
        self.recurrent_layers = model_hparams['recurrent_layers']

        # Recurrent model
        self.lstm = nn.LSTM(input_size, self.state_size, self.recurrent_layers, batch_first=True, dropout=0.2)

        # Classification model
        layers = []
        if hidden_depth == 0:
            layers.append( nn.Linear(self.state_size, output_size) )
        else:
            layers.append( nn.Linear(self.state_size, hidden_width) )
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

        attn_scores = torch.zeros_like(out)        
        for i in range(out.shape[1]):
            attn_scores[:,i,:] = self.attn(out[:,i,:])
        attn_scores = self.sm(attn_scores)

        out = torch.mul(out, attn_scores).sum(dim=1)

        # Make prediction with fully connected
        output = self.classifier(out[:,:])
        all_out.append(output)
        pred = torch.cat((pred, output.argmax(1, keepdim=True)), dim=1)
        
        return all_out, pred

    def initHidden(self, batch_size, device):
        return (torch.randn(self.recurrent_layers, batch_size, self.state_size).to(device), 
                torch.randn(self.recurrent_layers, batch_size, self.state_size).to(device))

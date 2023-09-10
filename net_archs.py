import torch
import torch.nn as nn
import flwr as fl
import numpy as np
from collections import OrderedDict

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, layer_size=64, num_of_layers=10, dropout=False):
        super().__init__()
        self.layer_size = layer_size

        layers = []
        #print('dropout',dropout)
        #print('layer_size', layer_size)
        #print('num layers', num_of_layers)

        layers.append(nn.Linear(input_dim, self.layer_size))
        layers.append(nn.ReLU())
        if dropout:
            layers.append(nn.Dropout(dropout))

        for k in range(num_of_layers):
            layers.append(nn.Linear(layer_size, self.layer_size))
            layers.append(nn.ReLU())

            if dropout:
                layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(layer_size, output_dim))
        layers.append(nn.Sigmoid())

        self.seq = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.seq:
            x = layer(x)

        return x

input_dim = 28
hidden_dim = 256
n_layers = 4
output_dim = 21

class LSTMModel(nn.Module):
    def __init__(self, input_dim=28, output_dim=21, hidden_dim=256, n_layers=4):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.linear(x)
        return nn.Sigmoid()(x)

class MLP2(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=True, dropout_rate=0.5):
        super().__init__()
        self.layer_size = 512
        if dropout:
            self.seq = nn.Sequential(
                nn.Linear(input_dim, self.layer_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(self.layer_size, self.layer_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(self.layer_size, self.layer_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(self.layer_size, self.layer_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(self.layer_size, output_dim),
                nn.Sigmoid(),
            )
        else:
            self.seq = nn.Sequential(
                nn.Linear(input_dim, self.layer_size),
                nn.ReLU(),
                nn.Linear(self.layer_size, self.layer_size),
                nn.ReLU(),
                nn.Linear(self.layer_size, self.layer_size),
                nn.ReLU(),
                nn.Linear(self.layer_size, self.layer_size),
                nn.ReLU(),
                nn.Linear(self.layer_size, output_dim),
                nn.Sigmoid(),
            )
    def forward(self, x):
        return self.seq(x)


class HierarchicalLSTM(nn.Module):
  def __init__(self, num_time_series, num_time_steps, num_features, hidden_size):
    super(HierarchicalLSTM, self).__init__()

    self.num_time_series = num_time_series
    self.num_time_steps = num_time_steps
    self.num_features = num_features
    self.hidden_size = hidden_size

    self.lower_lstm = nn.LSTM(input_size=num_features, hidden_size=hidden_size, num_layers=2, batch_first=True)

    self.pooling = nn.AdaptiveAvgPool1d(1)

  def forward(self, input):
    num_samples, num_time_series, num_time_steps, num_features = input.shape

    input = input.view(num_samples * num_time_series, num_time_steps, num_features)
    output, _ = self.lower_lstm(input)
    output = output[:, -1, :].view(num_samples, num_time_series, -1).permute(0, 2, 1)
    output = self.pooling(output).squeeze()

    return output


class LSTMModel2(nn.Module):
  def __init__(self, dropout_rate=False, layer_size=256, num_of_layers=2):
    super().__init__()
    self.lstm1 = HierarchicalLSTM(4, 12, 1, 4)
    self.lstm2 = HierarchicalLSTM(4, 4, 8, 16)
    self.lstm3 = HierarchicalLSTM(4, 4, 16, 32)
    self.lstm4 = HierarchicalLSTM(4, 4, 29, 32)

    #self.mlp = MLP2(184, 1, dropout=dropout, dropout_rate=dropout_rate)
    self.mlp = MLP(184, 1, layer_size=layer_size, num_of_layers=num_of_layers, dropout=dropout_rate)
  def forward(self, x):
    nts, ts1, ts2, ts3, ts4 = x
    ts1 = self.lstm1(ts1)
    ts2 = self.lstm2(ts2)
    ts3 = self.lstm3(ts3)
    ts4 = self.lstm4(ts4)
    x = torch.cat((nts, ts1, ts2, ts3, ts4), 1)
    x = self.mlp(x)
    return x

class CNN(nn.Module):
    def __init__(self, n_channel: int = 64, client_tree_num: int = 10, client_num: int = 2, task_type ='binary') -> None:
        super(CNN, self).__init__()

        self.task_type = task_type

        if self.task_type == 'multiclass':

            n_out = 20
            self.n_channel = 1
            self.conv1d = nn.Conv1d(
                1, self.n_channel, kernel_size=client_tree_num, stride=client_tree_num, padding=0
            )
            self.layer_direct = nn.Linear(client_num*20, n_out)  # 20 for multiclass
            #self.layer_direct = nn.Linear(20, n_out)  # 20 for multiclass
            self.seq = nn.Sequential(
                nn.Linear(20*client_num, 128),
                nn.ReLU(),
                nn.Linear(128, n_out),
                nn.Sigmoid()
                # nn.Linear(20*client_num, 20),
                # nn.Sigmoid()
            )

        if self.task_type == 'binary':
            #print('binary cnn')
            n_out = 1
            self.n_channel = n_channel
            self.conv1d = nn.Conv1d(
                1, n_channel, kernel_size=client_tree_num, stride=client_tree_num, padding=0
            )
            self.layer_direct = nn.Linear(n_channel * client_num, n_out)

        if self.task_type == 'new_multiclass':
            n_out = 20
            self.n_channel = n_channel
            self.conv1d = nn.Conv1d(
                1, n_channel, kernel_size=client_tree_num, stride=client_tree_num, padding=0
            )
            self.layer_direct = nn.Linear(n_channel * client_num, n_out)


        self.ReLU = nn.ReLU()
        self.Sigmoid = nn.Sigmoid()

        # Add weight initialization
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(
                    layer.weight, mode="fan_in", nonlinearity="relu"
                )

    def get_channels(self):
        return self.n_channel


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = self.ReLU(self.conv1d(x))
        # x = x.flatten(start_dim=1)
        # x = self.ReLU(x)
        # x = self.Sigmoid(self.layer_direct(x))

        if self.task_type == 'multiclass':
            x = self.conv1d(x)
            x = x.flatten(start_dim=1)
            x = self.seq(x)
        if self.task_type == 'binary':
            x = self.ReLU(self.conv1d(x))
            x = x.flatten(start_dim=1)
            x = self.ReLU(x)
            x = self.Sigmoid(self.layer_direct(x))

        return x

    def get_weights(self) -> fl.common.NDArrays:
        """Get model weights as a list of NumPy ndarrays."""
        return [
            np.array(val.cpu().numpy(), copy=True)
            for _, val in self.state_dict().items()
        ]

    def set_weights(self, weights: fl.common.NDArrays) -> None:
        """Set model weights from a list of NumPy ndarrays."""
        layer_dict = {}
        for k, v in zip(self.state_dict().keys(), weights):
            if v.ndim != 0:
                layer_dict[k] = torch.Tensor(np.array(v, copy=True))
        state_dict = OrderedDict(layer_dict)
        self.load_state_dict(state_dict, strict=True)


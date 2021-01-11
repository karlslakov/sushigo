import torch
import torch.nn as nn


class PGModel(nn.Module):
    def __init__(self, inputs, outputs):
        super(PGModel, self).__init__()
        
        hidden_layer_size = inputs // 2

        self.net = nn.Sequential(
            nn.Linear(inputs, hidden_layer_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_layer_size, hidden_layer_size),
            nn.LeakyReLU(),
            # nn.Linear(hidden_layer_size, hidden_layer_size),
            # nn.LeakyReLU(),
            # nn.Linear(15, 15),
            # nn.LeakyReLU(),
            nn.Linear(hidden_layer_size, outputs)
            )
        # self.net = nn.Linear(inputs, outputs)

        self.softmax = nn.LogSoftmax(dim = 0)
        
    def forward(self, x, invalids):
        x = self.net(x)
        x = torch.masked_select(x, invalids)
        x = self.softmax(x)
        return x
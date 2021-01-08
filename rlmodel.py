import torch
import torch.nn as nn


class PGModel(nn.Module):
    def __init__(self, inputs, outputs):
        super(PGModel, self).__init__()
        
        # self.net = nn.Sequential(
            # nn.Linear(inputs, 100),
            # nn.ReLU(),
            # nn.Linear(100, 100),
            # nn.ReLU(),
            # nn.Linear(100, outputs),
            # )
        self.net = nn.Linear(inputs, outputs)

        self.softmax = nn.LogSoftmax(dim = 0)
        
    def forward(self, x, invalids):
        x = self.net(x)
        x = torch.masked_select(x, invalids)
        x = self.softmax(x)
        return x
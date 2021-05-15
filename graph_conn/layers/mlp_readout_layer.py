import torch
import torch.nn as nn
import torch.nn.functional as F

"""
    MLP
"""


class MLPReadout(nn.Module):
    """
        Param: [in_dim, out_dim]
    """

    def __init__(self, in_dim, n_classes, L=0, hidden_dim=64):
        super().__init__()
        self.in_channels = in_dim
        self.n_classes = n_classes
        self.n_hidden = L

        self.hidden = nn.Linear(in_dim, hidden_dim)
        if L == 0:
            self.output = nn.Linear(in_dim, n_classes)
        else:
            self.output = nn.Linear(hidden_dim, n_classes)

    def forward(self, h):
        if self.n_hidden != 0:
            h = F.relu(self.hidden(h))

        return self.output(h)

    def __repr__(self):
        return '{}(in_channels={}, n_classes={}, n_hidden={})'.format(self.__class__.__name__,
                                                                         self.in_channels,
                                                                         self.n_classes, self.n_hidden)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .nac import NeuralAccumulatorCell
from torch.nn.parameter import Parameter


class NeuralArithmeticLogicUnitCell(nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.eps = 1e-10

        self.G = Parameter(torch.Tensor(out_dim, in_dim))
        self.nac = NeuralAccumulatorCell(in_dim, out_dim)
        self.register_parameter('bias', None)

        nn.init.kaiming_uniform_(self.G, a=np.sqrt(5))

    def forward(self, x):
        a = self.nac(x)
        g = torch.sigmoid(F.linear(x, self.G, self.bias))

        add_sub = g * a
        log_x = torch.log(torch.abs(x) + self.eps)

        m = torch.exp(self.nac(log_x))
        mul_div = (1 - g) * m

        output = add_sub + mul_div
        return output


class NALU(nn.Module):

    def __init__(self, in_dim, hidden_dim, out_dim, n_layers):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.n_layers = n_layers

        layers = []
        for i in range(n_layers):
            layers.append(
                NeuralArithmeticLogicUnitCell(
                    hidden_dim if i > 0 else in_dim,
                    hidden_dim if i < n_layers - 1 else out_dim
                )
            )

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        output = self.model(x)
        return output

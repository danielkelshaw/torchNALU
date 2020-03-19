import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.parameter import Parameter


class NeuralAccumulatorCell(nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.w_hat = Parameter(torch.Tensor(out_dim, in_dim))
        self.m_hat = Parameter(torch.Tensor(out_dim, in_dim))

        self.register_parameter('w_hat', self.w_hat)
        self.register_parameter('m_hat', self.m_hat)
        self.register_parameter('bias', None)

        self._reset_params()

    def _reset_params(self):
        nn.init.kaiming_uniform_(self.w_hat)
        nn.init.kaiming_uniform_(self.m_hat)

    def forward(self, x):
        w = torch.tanh(self.w_hat) * torch.sigmoid(self.m_hat)
        return F.linear(x, w, self.bias)


class NAC(nn.Module):

    def __init__(self, in_dim, hidden_dim, out_dim, n_layers):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.n_layers = n_layers

        layers = []
        for i in range(n_layers):
            layers.append(
                NeuralAccumulatorCell(
                    hidden_dim if i > 0 else in_dim,
                    hidden_dim if i < n_layers - 1 else out_dim
                )
            )

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        output = self.model(x)
        return output

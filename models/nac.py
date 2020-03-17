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

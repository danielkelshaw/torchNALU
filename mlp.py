import numpy as np
import torch.nn as nn


class MLP(nn.Module):

    def __init__(self, in_dim, hidden_dim, out_dim, n_layers, act=nn.ReLU()):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.n_layers = n_layers
        self.activation = act

        non_linearity = True
        if self.activation is None:
            non_linearity = False

        layers = []
        for i in range(n_layers):
            layers.extend(
                self._layer(
                    hidden_dim if i > 0 else in_dim,
                    hidden_dim if i < n_layers - 1 else out_dim,
                    non_linearity if i < n_layers - 1 else False
                )
            )

        self.model = nn.Sequential(*layers)
        self._initialise()

    def forward(self, x):
        output = self.model(x)
        return output

    def _layer(self, in_dim, out_dim, activation):
        if activation:
            return [
                nn.Linear(in_dim, out_dim),
                self.activation
            ]
        else:
            return [
                nn.Linear(in_dim, out_dim)
            ]

    def _initialise(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=np.sqrt(5))
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / np.sqrt(fan_in)
                nn.init.uniform_(m.bias, -bound, bound)

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from models import MLP


train_range = [-5, 5]
test_range = [-20, 20]
learning_rate = 0.01
n_epochs = 10000

activations = [
    nn.Hardtanh(),
    nn.Sigmoid(),
    nn.ReLU6(),
    nn.Tanh(),
    nn.Tanhshrink(),
    nn.Hardshrink(),
    nn.LeakyReLU(),
    nn.Softshrink(),
    nn.Softsign(),
    nn.ReLU(),
    nn.PReLU(),
    nn.Softplus(),
    nn.ELU(),
    nn.SELU()
]


def train(args, model, optimizer, criterion, data):

    for epoch in range(1, args.n_epochs + 1):
        optimizer.zero_grad()

        output = model(data)
        loss = criterion(output, data)
        m = torch.mean(torch.abs(output - data))

        loss.backward()
        optimizer.step()

        if epoch % args.log_interval == 0:
            print('Epoch {:02}\t'
                  'Training Loss = {:.5f}\t'
                  'MEA = {:.5f}'.format(epoch, loss, m))


def test(model, data):
    with torch.no_grad():
        output = model(data)
        m = torch.mean(torch.abs(output - data))
        return m


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--n-layers', type=int, default=4, metavar='N',
                        help='number of layers (default: 4)')
    parser.add_argument('--hidden-dim', type=int, default=8, metavar='HD',
                        help='hidden dim (default: 8)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--n-epochs', type=int, default=10000, metavar='E',
                        help='number of training epochs (default: 1000)')
    parser.add_argument('--train-range', type=list, default=[-5, 5], metavar='S',
                        help='support for training (default: [-5, 5])')
    parser.add_argument('--test-range', type=list, default=[-20, 20], metavar='T',
                        help='support for testing (default: [-20, 20])')
    parser.add_argument('--log-interval', type=int, default=500, metavar='LI',
                        help='train logging interval (default: 500)')

    args = parser.parse_args()

    # generate results directory
    save_dir = os.path.join(os.getcwd(), 'media')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    args.train_range[-1] += 1
    args.test_range[-1] += 1

    train_data = torch.arange(*args.train_range).unsqueeze_(1).float()
    test_data = torch.arange(*args.test_range).unsqueeze_(1).float()

    # train
    mse_list = []
    for act in activations:
        print('Training with {}...'.format(act))
        mses = []

        for i in range(100):
            model = MLP(in_dim=1,
                        hidden_dim=args.hidden_dim,
                        out_dim=1,
                        n_layers=args.n_layers,
                        act=act)

            optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)
            criterion = nn.MSELoss()

            train(args, model, optimizer, criterion, train_data)
            test_loss = test(model, train_data)
            mses.append(test_loss)

        mse_list.append(torch.cat(mses, dim=1).mean(dim=1))

    mse_list = [x.numpy().flatten() for x in mse_list]

    # plotting
    fig = plt.figure(figsize=(16, 10))

    for i, act in enumerate(activations):
        plt.plot(np.arange(args.train_range), mse_list[i], label=act.__str__())

    plt.grid()
    plt.legend()

    filename = os.path.join(save_dir, 'function_failures.png')
    plt.savefig(filename, bbox_inches='tight')


if __name__ == '__main__':
    main()

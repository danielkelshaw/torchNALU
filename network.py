import os
import random
import argparse
import numpy as np

import torch
import torch.nn as nn

from models import MLP, NAC, NALU


arithmetic_functions = {
    'add': lambda x, y: x + y,
    'sub': lambda x, y: x - y,
    'mul': lambda x, y: x * y,
    'div': lambda x, y: x / y,
    'squared': lambda x, y: torch.pow(x, 2),
    'sqrt': lambda x, y: torch.sqrt(x)
}


def generate_data(n_vals, dim, n_sum, fn, support):

    X, y = [], []
    data = torch.FloatTensor(dim).uniform_(*support).unsqueeze_(1)
    for i in range(n_vals):
        idx_a = random.sample(range(dim), n_sum)
        idx_b = random.sample([x for x in range(dim) if x not in idx_a], n_sum)

        a = data[idx_a].sum()
        b = data[idx_b].sum()

        X.append([a, b])
        y.append(fn(a, b))

    X = torch.FloatTensor(X)
    y = torch.FloatTensor(y).unsqueeze_(1)

    idx = list(range(n_vals))
    np.random.shuffle(idx)

    X_data, y_data = X[idx], y[idx]

    return X_data, y_data


def train(args, model, optimizer, criterion, data, target):

    for epoch in range(args.n_epochs):

        optimizer.zero_grad()

        output = model(data)
        loss = criterion(output, target)
        m = torch.mean(torch.abs(target - output))

        loss.backward()
        optimizer.step()

        if epoch % args.log_interval == 0:
            print('Epoch {:05}:\t'
                  'Loss = {:.5f}\t'
                  'MEA = {:.5f}'.format(epoch, loss, m))


def test(model, data, target):

    with torch.no_grad():
        output = model(data)
        m = torch.mean(torch.abs(target - output))
        return m


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--n-layers', type=int, default=2, metavar='N',
                        help='number of layers (default: 2)')
    parser.add_argument('--hidden-dim', type=int, default=2, metavar='HD',
                        help='hidden dim (default: 2)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--n-epochs', type=int, default=10000, metavar='E',
                        help='number of training epochs (default: 1000)')
    parser.add_argument('--interp-support', type=list, default=[1, 100], metavar='S',
                        help='support for interpolation (default: [1, 100])')
    parser.add_argument('--extrap-support', type=list, default=[101, 200], metavar='S',
                        help='support for extrapolation (default: [101, 200])')
    parser.add_argument('--n-sum', type=int, default=5, metavar='NS',
                        help='num functions to sum (default: 5)')
    parser.add_argument('--log-interval', type=int, default=500, metavar='LI',
                        help='train logging interval (default: 500)')
    parser.add_argument('--normalise', action='store_true', default=True,
                        help='normalise results (default: True)')

    args = parser.parse_args()

    # generate results directory
    save_dir = os.path.join(os.getcwd(), 'results')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    models = [
        MLP(in_dim=2,
            hidden_dim=args.hidden_dim,
            out_dim=1,
            n_layers=args.n_layers,
            act=nn.ReLU6()),
        MLP(in_dim=2,
            hidden_dim=args.hidden_dim,
            out_dim=1,
            n_layers=args.n_layers,
            act=None),
        NAC(in_dim=2,
            hidden_dim=args.hidden_dim,
            out_dim=1,
            n_layers=args.n_layers),
        NALU(in_dim=2,
             hidden_dim=args.hidden_dim,
             out_dim=1,
             n_layers=args.n_layers)
    ]

    results = {}
    results['interp'] = {}
    results['extrap'] = {}

    for fn_type, fn in arithmetic_functions.items():
        print('-> Testing function: {}'.format(fn_type))
        results['interp'][fn_type] = []
        results['extrap'][fn_type] = []

        Xtrain, ytrain = generate_data(
            n_vals=500, dim=100, n_sum=args.n_sum,
            fn=fn, support=args.interp_support
        )

        Xtest_interp, ytest_interp = generate_data(
            n_vals=50, dim=100, n_sum=args.n_sum,
            fn=fn, support=args.interp_support
        )

        Xtest_extrap, ytest_extrap = generate_data(
            n_vals=50, dim=100, n_sum=args.n_sum,
            fn=fn, support=args.extrap_support
        )

        # random model results
        random_results_interp = []
        random_results_extrap = []
        for i in range(100):
            network = MLP(in_dim=2,
                          hidden_dim=args.hidden_dim,
                          out_dim=1,
                          n_layers=args.n_layers,
                          act=nn.ReLU6())

            interp_mse = test(network, Xtest_interp, ytest_interp)
            extrap_mse = test(network, Xtest_extrap, ytest_extrap)

            random_results_interp.append(interp_mse.item())
            random_results_extrap.append(extrap_mse.item())

        results['interp'][fn_type].append(np.mean(random_results_interp))
        results['extrap'][fn_type].append(np.mean(random_results_extrap))

        # other models
        for net in models:
            print('\tRunning: {}'.format(net.__str__().split('(')[0]))
            optimizer = torch.optim.RMSprop(net.parameters(), lr=args.lr)
            criterion = nn.MSELoss()
            train(args, net, optimizer, criterion, Xtrain, ytrain)

            interp_mse = test(net, Xtest_interp, ytest_interp).item()
            extrap_mse = test(net, Xtest_extrap, ytest_extrap).item()

            results['interp'][fn_type].append(interp_mse)
            results['extrap'][fn_type].append(extrap_mse)

        # save results
        with open(os.path.join(save_dir, 'interp_results.csv'), 'w+') as f:
            f.write('Relu6,None,NAC,NALU\n')
            for k, v in results['interp'].items():
                rand_result = v[0]
                normed_mse = [100.0 * x / rand_result for x in v[1:]]

                if args.normalise:
                    f.write('{:.5f},{:.5f},{:.5f},{:.5f}\n'.format(*normed_mse))
                else:
                    f.write('{:.5f},{:.5f},{:.5f},{:.5f}\n'.format(*v[1:]))

        with open(os.path.join(save_dir, 'extrap_results.csv'), 'w+') as f:
            f.write('Relu6,None,NAC,NALU\n')
            for k, v in results['extrap'].items():
                rand_result = v[0]
                normed_mse = [100.0 * x / rand_result for x in v[1:]]

                if args.normalise:
                    f.write('{:.5f},{:.5f},{:.5f},{:.5f}\n'.format(*normed_mse))
                else:
                    f.write('{:.5f},{:.5f},{:.5f},{:.5f}\n'.format(*v[1:]))


if __name__ == '__main__':
    main()

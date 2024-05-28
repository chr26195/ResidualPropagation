# Visualizing real-world datasets
# Load real-world datasets: Cora
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch_geometric.transforms as T
from torch_geometric.utils import stochastic_blockmodel_graph
from sklearn.datasets import make_classification
from torch_geometric.data import Data

from eval import *
from dataset import * 
from utils import *
from models import *
from metrics import *
import pandas as pd

import matplotlib.pyplot as plt
import argparse

def get_args():
    parser = argparse.ArgumentParser(prog='Visualization Synthetic')
    parser.add_argument('--save_dir', type=str, default='figs/', help='Base directory for saving information.')
    parser.add_argument('--dataset', type=str, default='csbm')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--data_dir', type=str, default='/data/')
    parser.add_argument('--seed', type=int, default=100)

    parser.add_argument('--model', type=str, default='GCN') 
    parser.add_argument('--hidden_dim', type=int, default=16)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--normalization', type=str, default='None')
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--loss', type=str, default='CE', choices=['MSE', 'CE'])
    parser.add_argument('--runs', type=int, default=1,)
    parser.add_argument('--steps', type=int, default=20000)
    parser.add_argument('--y_scale', type=float, default=1) # corresp. M
    parser.add_argument('--loss_scale', type=float, default=1) # corresp. k, for large class number

    parser.add_argument('--input_dim', type=int, default=100) 
    parser.add_argument('--num_classes', type=int, default=5) 
    parser.add_argument('--num_samples_perclass', type=int, default=400) 
    parser.add_argument('--homo_ratio', type=float, default=0.01) 
    parser.add_argument('--sep', type=int, default=500) 

    args = parser.parse_args()
    return args

def plot_line_chart(data_list1, data_list2, x_values=None, constant_line=None, save_path='figs/sample.png', title='Line Chart', x_label='Steps', y_label='Alignment'):
    plt.figure(figsize=(8, 4)) 

    plt.plot(x_values, data_list1, marker='o', linestyle='-', color='b', label='NTK-Y')
    plt.plot(x_values, data_list2, marker='o', linestyle='-', color='g', label='NTK-A')

    if constant_line is not None:
        plt.axhline(y=constant_line, color='r', linestyle='--', label=f'A-Y')

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()

    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def do_compute_ntk(iterations, sep):
    return iterations % sep == 0

def main():
    args = get_args(); fix_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    d = args.input_dim
    k = args.num_classes
    n = args.num_classes * args.num_samples_perclass

    homo_ratio = args.homo_ratio
    hete_ratio = args.homo_ratio / (k-1) # ensure same number of homo and hete edges

    # create homo and hete edge index
    block_sizes = torch.zeros([args.num_classes]).fill_(args.num_samples_perclass).int()

    edge_probs_homo = torch.zeros([args.num_classes, args.num_classes])
    edge_probs_homo = edge_probs_homo.fill_diagonal_(homo_ratio)
    edge_index_homo = stochastic_blockmodel_graph(block_sizes, 
                                                  edge_probs_homo, 
                                                  directed=True) # convert to undirected later
    rand_indices_homo = torch.randperm(edge_index_homo.shape[1])
    print(edge_index_homo.shape)

    edge_probs_hete = torch.zeros([args.num_classes, args.num_classes])
    edge_probs_hete = edge_probs_hete.fill_diagonal_(-hete_ratio) + hete_ratio
    edge_index_hete = stochastic_blockmodel_graph(block_sizes, 
                                                  edge_probs_hete, 
                                                  directed=True) # convert to undirected later
    rand_indices_hete = torch.randperm(edge_index_hete.shape[1])
    print(edge_index_hete.shape)

    # default edge index used in training
    edge_index = torch.cat((torch.arange(n).unsqueeze(dim=0),torch.arange(n).unsqueeze(dim=0)), dim=0)

    # create input features
    x, y_not_sorted = make_classification(
        n_samples=n,
        n_features=d,
        n_informative=4,
        n_classes=args.num_classes
    )
    x = x[np.argsort(y_not_sorted)]
    x = torch.from_numpy(x).to(torch.float)
    y = torch.arange(args.num_classes).repeat_interleave(block_sizes)

    # get dataset
    data = Data(x=x, edge_index=edge_index, y=y)
    data.edge_index_homo = edge_index_homo
    data.edge_index_hete = edge_index_hete

    # create data split
    data = T.RandomNodeSplit('random', num_train_per_class=20, num_val=500, num_test=1000, num_splits=args.runs)(data) # the same data split as other small datasets
    if len(data.train_mask.shape) == 1:
        data.train_mask = data.train_mask.unsqueeze(dim=-1)
        data.val_mask = data.val_mask.unsqueeze(dim=-1)
        data.test_mask = data.test_mask.unsqueeze(dim=-1)
    data.onehot_y = F.one_hot(data.y).float() 

    data = T.ToDevice(device)(data)
    metric_name = 'acc'

    for run in range(1, args.runs + 1):
        model = MODELS[args.model](num_layers=args.num_layers, 
                                   input_dim=d, 
                                   hidden_dim=args.hidden_dim, 
                                   output_dim=k if k > 2 else 1, 
                                   dropout=0, # we don't use dropout
                                   normalization=args.normalization).to(device)
        
        model.construct_adj(data.edge_index, n)
        optimizer = torch.optim.SGD(get_parameter_groups(model), 
                                    weight_decay=args.weight_decay, 
                                    lr=args.lr, 
                                    momentum=args.momentum) # We use SGD instead of Adam

        run_idx = (run - 1) % data.train_mask.shape[1]
        train_mask = data.train_mask[:, run_idx]
        val_mask = data.val_mask[:, run_idx]
        test_mask = data.test_mask[:, run_idx]

        A_sparse = construct_sparse_adj(data.edge_index_homo, num_nodes=n, type='DAD').to(device) 
        A_dense = torch.eye(n).to(device)
        for _ in range(args.num_layers * 2):
            A_dense = A_sparse @ A_dense
        A_dense = A_dense.to_dense()

        step_list = []
        alignY_list = []
        alignA_list = []

        homo_p = np.arange(1, -0.01, -0.02)
        columns = ['steps', 
                   'train_acc',
                   'test_acc']
        
        for p in homo_p:
            columns.append(f'align_ntk_a_{p}') # ntk_a computes alignment of NTK and homoA
            columns.append(f'align_ntk_y_{p}')
            columns.append(f'align_a_y_{p}') # alignment of A and Y
            columns.append(f'test_acc_{p}')
            columns.append(f'train_acc_{p}')
        
        log = pd.DataFrame(columns=columns)

        for step in range(1, args.steps):
            # Training
            model.train(); optimizer.zero_grad()
            out = model(data.x)
            if k > 2 and args.loss == 'CE': 
                loss = nn.CrossEntropyLoss(reduction='mean')(out[train_mask], data.y[train_mask])
            elif args.loss == 'CE': 
                loss = nn.BCEWithLogitsLoss(reduction='mean')(out[train_mask], data.y[train_mask].unsqueeze(dim=-1).float())
            elif args.loss == 'MSE':
                loss = ModifiedMSELoss(out[train_mask], data.onehot_y[train_mask], y_scale=args.y_scale, loss_scale=args.loss_scale)
            
            if step > 1:
                loss.backward(); optimizer.step()

            # Evaluate
            model.eval()
            y_pred = model(data.x)
            train_metric = METRICS[metric_name](data.y[train_mask].unsqueeze(dim=-1), y_pred[train_mask]) * 100
            val_metric = METRICS[metric_name](data.y[val_mask].unsqueeze(dim=-1), y_pred[val_mask]) * 100
            test_metric = METRICS[metric_name](data.y[test_mask].unsqueeze(dim=-1), y_pred[test_mask]) * 100

            # Compute alignments
            if do_compute_ntk(step - 1, args.sep):
                to_log = pd.Series()
                to_log['steps'] = step - 1
                to_log['train_acc'] = train_metric
                to_log['test_acc'] = test_metric

                for p in homo_p:
                    # randomly drop edges
                    homo_indices = rand_indices_homo[:int(len(rand_indices_homo)*p)]
                    hete_indices = rand_indices_hete[:int(len(rand_indices_homo)*(1-p))]
                    edge_index = torch.cat(
                        (data.edge_index_homo[:,homo_indices], data.edge_index_hete[:,hete_indices]), dim = 1
                    )
                    model.A = construct_sparse_adj(edge_index, num_nodes=n, type='DAD').to(device)
                    model.A = model.A.to_dense()

                    # A
                    A_sparse = construct_sparse_adj(edge_index, num_nodes=n, type='DAD').to(device) 
                    A_dense = torch.eye(n).to(device)
                    for _ in range(args.num_layers * 2):
                        A_dense = A_sparse @ A_dense
                    A_dense = A_dense.to_dense()

                    NTK = empirical_ntk(model, data.x, type='nn')
                    alignY = centered_kernel_alignment(NTK, data.onehot_y @ data.onehot_y.t())
                    alignA = centered_kernel_alignment(NTK, A_dense)
                    to_log[f'align_ntk_a_{p}'] = alignA
                    to_log[f'align_ntk_y_{p}'] = alignY

                    alignAY = centered_kernel_alignment(data.onehot_y @ data.onehot_y.t(), A_dense)
                    to_log[f'align_a_y_{p}'] = alignAY


                    y_pred = model(data.x)
                    train_metric = METRICS[metric_name](data.y[train_mask].unsqueeze(dim=-1), y_pred[train_mask]) * 100
                    test_metric = METRICS[metric_name](data.y[test_mask].unsqueeze(dim=-1), y_pred[test_mask]) * 100
                    to_log[f'test_acc_{p}'] = test_metric
                    to_log[f'train_acc_{p}'] = train_metric

                log.loc[len(log)] = to_log
                print(log.loc[len(log) - 1])

                log.to_pickle(os.path.join(args.save_dir,f'{args.dataset}_{args.model}.pkl'))

                alignY_list.append(alignY)
                alignA_list.append(alignA)
                step_list.append(step)

                model.A = construct_sparse_adj(data.edge_index, num_nodes=n, type='DAD').to(device)
                
            metrics = {
                f'train {metric_name}': train_metric,
                f'val {metric_name}': val_metric,
                f'test {metric_name}': test_metric
            }

            print(f'run: {run}, step: {step:03d}, '
                f'train {metric_name}: {metrics[f"train {metric_name}"]:.2f}, '
                f'val {metric_name}: {metrics[f"val {metric_name}"]:.2f}, '
                f'test {metric_name}: {metrics[f"test {metric_name}"]:.2f} ')


if __name__ == '__main__':
    main()
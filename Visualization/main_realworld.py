# Load real-world datasets: Cora
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch_geometric.transforms as T

from eval import *
from dataset import * 
from utils import *
from models import *
from metrics import *
import pandas as pd

import matplotlib.pyplot as plt
import argparse

def get_args():
    parser = argparse.ArgumentParser(prog='Visualization')
    parser.add_argument('--save_dir', type=str, default='figs/', help='Base directory for saving information.')
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--data_dir', type=str, default='/data/')
    parser.add_argument('--seed', type=int, default=100)

    # The following arguments are for NN-like deep models
    parser.add_argument('--model', type=str, default='GCN') 
    parser.add_argument('--hidden_dim', type=int, default=16)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--normalization', type=str, default='None')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--loss', type=str, default='CE', choices=['MSE', 'CE'])
    parser.add_argument('--runs', type=int, default=1,)
    parser.add_argument('--steps', type=int, default=50000)

    parser.add_argument('--y_scale', type=float, default=1) # corresp. M
    parser.add_argument('--loss_scale', type=float, default=1) # corresp. k, for large class number

    # For visualization
    parser.add_argument('--sep', type=int, default=500) # replacing a certain ratio of edges to random edges
    parser.add_argument('--drop_p', type=float, default=0.0) # replacing a certain ratio of edges to random edges


    args = parser.parse_args()
    return args

def do_compute_ntk(iterations, sep):
    return iterations % sep == 0

def main():
    args = get_args(); fix_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Prepare dataset
    data = load_dataset(args.data_dir, args.dataset, args.runs, fixed_split = True)
    data = T.ToDevice(device)(data)
    k, n, d, e = dataset_statistics(data, dataset_name = args.dataset, verbose = True)
    metric_name = data.metric

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
        
        A_sparse = construct_sparse_adj(data.edge_index, num_nodes=n, type='DAD').to(device) # this is the ground-truth adj matrix
        A_dense = torch.eye(n).to(device)
        for _ in range(args.num_layers * 2):
            A_dense = A_sparse @ A_dense
        A_dense = A_dense.to_dense()
        alignAY = centered_kernel_alignment(A_dense, data.onehot_y @ data.onehot_y.t())


        run_idx = (run - 1) % data.train_mask.shape[1]
        train_mask = data.train_mask[:, run_idx]
        val_mask = data.val_mask[:, run_idx]
        test_mask = data.test_mask[:, run_idx]

        step_list = []
        alignY_list = []
        alignA_list = []

        drop_p = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, .99]
        columns = ['steps', 
                   'train_acc',
                   'test_acc',
                   'align_ay']
        
        for p in drop_p:
            columns.append(f'align_ntk_a_{p}')
            columns.append(f'align_ntk_y_{p}')
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

            if do_compute_ntk(step - 1, args.sep):
                indices = torch.randperm(data.edge_index.shape[1]).to(device)
                to_log = pd.Series()
                to_log['steps'] = step - 1
                to_log['align_ay'] = alignAY
                to_log['train_acc'] = train_metric
                to_log['test_acc'] = test_metric

                for p in drop_p:
                    # randomly drop edges
                    edge_index = data.edge_index[:, indices[:int(len(indices) * (1-p))]]
                    model.A = construct_sparse_adj(edge_index, num_nodes=n, type='DAD').to(device)
                    model.A = model.A.to_dense()

                    NTK = empirical_ntk(model, data.x, type='nn')
                    alignY = centered_kernel_alignment(NTK, data.onehot_y @ data.onehot_y.t())
                    alignA = centered_kernel_alignment(NTK, A_dense)
                    to_log[f'align_ntk_a_{p}'] = alignA
                    to_log[f'align_ntk_y_{p}'] = alignY

                    y_pred = model(data.x)
                    train_metric = METRICS[metric_name](data.y[train_mask].unsqueeze(dim=-1), y_pred[train_mask]) * 100
                    test_metric = METRICS[metric_name](data.y[test_mask].unsqueeze(dim=-1), y_pred[test_mask]) * 100
                    to_log[f'test_acc_{p}'] = test_metric
                    to_log[f'train_acc_{p}'] = train_metric

                log.loc[len(log)] = to_log
                print(log.loc[len(log) - 1])

                log.to_pickle(os.path.join(args.save_dir,f'{args.dataset}_{args.model}_{args.drop_p}.pkl'))

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
                f'test {metric_name}: {metrics[f"test {metric_name}"]:.2f} ',
                f'alignY: {alignY:.4f}',
                f'alignA: {alignA:.4f}')

if __name__ == '__main__':
    main()
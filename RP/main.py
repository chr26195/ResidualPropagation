import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse

from eval import *
from dataset import load_dataset, dataset_statistics
from utils import *

class LabelPropagation(nn.Module):
    '''
    based on paper 'Learning with Local and Global Consistency'
    input:
        redisuals: [instances, k], val and test instances are masked by 0
    output:
        y_pred: [instances, k]
    '''
    def __init__(self, k, alpha):
        super().__init__()
        self.k = k
        self.alpha = alpha
        self.A = None

    def preprocess(self, args, edge_index, num_nodes, num_classes):
        self.A = construct_sparse_adj(edge_index, num_nodes=num_nodes, type='DAD')

    def forward(self, redisuals):
        if self.alpha < 1: 
            y_pred = redisuals.clone()
            for _ in range(self.k):
                y_pred = self.alpha * self.A @ y_pred + (1 - self.alpha) * redisuals
            return y_pred
        else:
            for _ in range(self.k):
                redisuals = self.A @ redisuals
            return redisuals
        
ALGORITHMS = {
    'LP': LabelPropagation
}

def get_args():
    parser = argparse.ArgumentParser(prog='GNN Pipeline', description='Training pipeline for node classification')
    parser.add_argument('--name', type=str, default=None, help='Experiment name. If None, model name is used.')
    parser.add_argument('--save_dir', type=str, default='./logs', help='Base directory for saving information.')
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--fixed_split', default=False, action='store_true', help='Use fixed_split for cora/citeseer/pubmed datasets')
    parser.add_argument('--train_ratio', type=float, default=None, help='Training set ratio for random split')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--data_dir', type=str, default='/data/')

    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--runs', type=int, default=1, help='number of distinct runs')
    parser.add_argument('--steps', type=int, default=100, help='number of steps for each run')
    parser.add_argument('--lr', type=float, default=0.01)

    # The following arguments are for RP-like algorithms
    parser.add_argument('--algorithm', type=str, default='LP')
    parser.add_argument('--alpha', type=float, default=1.0, help='hyperparameter alpha')
    parser.add_argument('--k', type=int, default=3, help='number of iterations')

    args = parser.parse_args()
    if args.name is None and args.algorithm is not None:
        args.name = args.algorithm
    return args

def main():
    args = get_args(); fix_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Prepare dataset
    data = load_dataset(args.data_dir, args.dataset, args.runs, train_ratio=args.train_ratio, fixed_split=args.fixed_split)
    data.to(device)
    k, n, d, e = dataset_statistics(data, dataset_name = args.dataset, verbose = True)
    metric_name = data.metric

    logger = Logger(args, metric=metric_name)
    
    for run in range(1, args.runs + 1):
        # Prepare model
        algorithm = ALGORITHMS[args.algorithm](k=args.k, 
                                   alpha=args.alpha)
        algorithm.preprocess(args, data.edge_index, num_nodes=n, num_classes=k)
        algorithm.to(device)

        run_idx = (run - 1) % data.train_mask.shape[1]
        train_mask, val_mask, test_mask = data.train_mask[:, run_idx], data.val_mask[:, run_idx], data.test_mask[:, run_idx]

        logger.start_run(run)

        # initialization
        redisuals = torch.zeros_like(data.onehot_y)
        redisuals[train_mask] = data.onehot_y[train_mask].clone()

        for step in range(1, args.steps + 1):
            # Training
            masked_residuals = redisuals.clone()
            masked_residuals[~train_mask] =  0
            redisuals -= args.lr * algorithm(masked_residuals)

            # Evaluate
            train_metric = METRICS[metric_name](data.y[train_mask].unsqueeze(dim=-1), -redisuals[train_mask] + data.onehot_y[train_mask]) * 100
            val_metric = METRICS[metric_name](data.y[val_mask].unsqueeze(dim=-1), -redisuals[val_mask]) * 100
            test_metric = METRICS[metric_name](data.y[test_mask].unsqueeze(dim=-1), -redisuals[test_mask]) * 100
            metrics = {
                f'train {metric_name}': train_metric,
                f'val {metric_name}': val_metric,
                f'test {metric_name}': test_metric
            }

            logger.update_metrics(metrics=metrics, step=step)

        logger.finish_run()
        
    logger.print_metrics_summary()
    logger.save_in_csv()

if __name__ == '__main__':
    main()
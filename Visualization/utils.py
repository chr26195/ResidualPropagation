import numpy as np
import torch
import random
import torch_geometric.utils as utils
import yaml
import os
import torch.nn as nn
import torch.nn.functional as F

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def construct_sparse_adj(edge_index, num_nodes, type='DAD', self_loop=True):
    edge_index = utils.to_undirected(edge_index)
    if self_loop:
        edge_index, _ = utils.remove_self_loops(edge_index)
        edge_index, _ = utils.add_self_loops(edge_index, num_nodes=num_nodes)

    src, dst = edge_index
    deg = utils.degree(dst, num_nodes=num_nodes)

    if type == 'DAD':
        deg_src = deg[src].pow(-0.5) 
        deg_src.masked_fill_(deg_src == float('inf'), 0)
        deg_dst = deg[dst].pow(-0.5)
        deg_dst.masked_fill_(deg_dst == float('inf'), 0)
    elif type == 'DA':
        deg_src = deg[src].pow(-1) 
        deg_src.masked_fill_(deg_src == float('inf'), 0)
        deg_dst = deg[dst].pow(-0)
        deg_dst.masked_fill_(deg_dst == float('inf'), 0)

    A = torch.sparse_coo_tensor(edge_index, deg_src * deg_dst, torch.Size([num_nodes, num_nodes]))
    return A

def get_parameter_groups(model):
    no_weight_decay_names = ['bias', 'normalization']
    parameter_groups = [
        {
            'params': [param for name, param in model.named_parameters()
                       if not any(no_weight_decay_name in name for no_weight_decay_name in no_weight_decay_names)]
        },
        {
            'params': [param for name, param in model.named_parameters()
                       if any(no_weight_decay_name in name for no_weight_decay_name in no_weight_decay_names)],
            'weight_decay': 0
        },
    ]
    return parameter_groups

def ModifiedMSELoss(y_pred, y_true, y_scale = 1, loss_scale = 1, centered = False):
    class_mask = y_true.bool()

    k = y_pred.shape[1] # number of classes
    if centered:
        y_true = (y_true - 1 / k) # centered around 0
    y_true = y_true * y_scale # rescale the ground truth label
    if loss_scale == 1:
        return nn.MSELoss(reduction='mean')(y_pred, y_true)
    else:
        loss = nn.MSELoss(reduction='none')(y_pred, y_true)
        loss[class_mask] = loss[class_mask] * loss_scale
        return loss.mean()

class Logger:
    def __init__(self, args, metric):
        self.args = args
        self.base_file = os.path.join(args.save_dir, args.dataset, 'a.csv') 
        self.save_dir = self.get_save_dir(base_dir=args.save_dir, dataset=args.dataset, name=args.name)
        self.verbose = True
        self.metric = metric
        self.val_metrics = []
        self.test_metrics = []
        self.best_steps = []
        self.num_runs = args.runs
        self.cur_run = None

        print(f'Results will be saved to {self.save_dir}.')
        with open(os.path.join(self.save_dir, 'args.yaml'), 'w') as file:
            yaml.safe_dump(vars(args), file, sort_keys=False)

    def start_run(self, run):
        self.cur_run = run
        self.val_metrics.append(0)
        self.test_metrics.append(0)
        self.best_steps.append(None)

        print(f'Starting run {run}/{self.num_runs}...')

    def update_metrics(self, metrics, step):
        if metrics[f'val {self.metric}'] > self.val_metrics[-1]:
            self.val_metrics[-1] = metrics[f'val {self.metric}']
            self.test_metrics[-1] = metrics[f'test {self.metric}']
            self.best_steps[-1] = step

        if self.verbose:
            print(f'run: {self.cur_run:02d}, step: {step:03d}, '
                  f'train {self.metric}: {metrics[f"train {self.metric}"]:.2f}, '
                  f'val {self.metric}: {metrics[f"val {self.metric}"]:.2f}, '
                  f'test {self.metric}: {metrics[f"test {self.metric}"]:.2f}')

    def finish_run(self):
        self.save_metrics()
        print(f'Finished run {self.cur_run}. '
              f'Best val {self.metric}: {self.val_metrics[-1]:.2f}, '
              f'corresponding test {self.metric}: {self.test_metrics[-1]:.2f} '
              f'(step {self.best_steps[-1]}).\n')

    def save_metrics(self):
        num_runs = len(self.val_metrics)
        val_metric_mean = np.mean(self.val_metrics).item()
        val_metric_std = np.std(self.val_metrics, ddof=1).item() if len(self.val_metrics) > 1 else np.nan
        test_metric_mean = np.mean(self.test_metrics).item()
        test_metric_std = np.std(self.test_metrics, ddof=1).item() if len(self.test_metrics) > 1 else np.nan

        metrics = {
            'num runs': num_runs,
            f'val {self.metric} mean': val_metric_mean,
            f'val {self.metric} std': val_metric_std,
            f'test {self.metric} mean': test_metric_mean,
            f'test {self.metric} std': test_metric_std,
            f'val {self.metric} values': self.val_metrics,
            f'test {self.metric} values': self.test_metrics,
            'best steps': self.best_steps
        }

        with open(os.path.join(self.save_dir, 'metrics.yaml'), 'w') as file:
            yaml.safe_dump(metrics, file, sort_keys=False)

    def print_metrics_summary(self):
        with open(os.path.join(self.save_dir, 'metrics.yaml'), 'r') as file:
            metrics = yaml.safe_load(file)

        print(f'Finished {metrics["num runs"]} runs.')
        print(f'Val {self.metric} mean: {metrics[f"val {self.metric} mean"]:.2f}')
        print(f'Val {self.metric} std: {metrics[f"val {self.metric} std"]:.2f}')
        print(f'Test {self.metric} mean: {metrics[f"test {self.metric} mean"]:.2f}')
        print(f'Test {self.metric} std: {metrics[f"test {self.metric} std"]:.2f}')

    def save_in_csv(self):
        with open(os.path.join(self.save_dir, 'metrics.yaml'), 'r') as file:
            metrics = yaml.safe_load(file)

        with open(self.base_file, 'a+') as write_obj:
            write_obj.write(f'{self.save_dir}, \t' + \
                            f'Test: {metrics[f"test {self.metric} mean"]:.2f} $\pm$ {metrics[f"test {self.metric} std"]:.2f}, \n')

    @staticmethod
    def get_save_dir(base_dir, dataset, name):
        idx = 1
        save_dir = os.path.join(base_dir, dataset, f'{name}_{idx:02d}')
        while os.path.exists(save_dir):
            idx += 1
            save_dir = os.path.join(base_dir, dataset, f'{name}_{idx:02d}')

        os.makedirs(save_dir)
        return save_dir
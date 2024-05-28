import torch
import os
from torch_geometric.datasets import Planetoid, HeterophilousGraphDataset, Actor, WebKB, WikipediaNetwork
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset
import torch.nn.functional as F


# load_dataset() provides dataset with predefined split
def load_dataset(data_dir, dataset_name, runs, train_ratio = None, fixed_split = False):
    '''
    data.y : [num_instances] int
    data.onehot_y : [num_instances, k] float
    data.x : [num_instances, d] float
    data.edge_index : [2, num_edges] int
    data.*_mask : [num_instances, num_runs] bool
    '''
    # from paper 'Semi-supervised classification with graph convolutional networks'
    if dataset_name in  ('cora', 'citeseer', 'pubmed'):
        dataset = Planetoid(root=os.path.join(data_dir, 'Planetoid'), name=dataset_name)
        data = dataset[0]
        if not fixed_split:
            data = T.RandomNodeSplit('random', num_train_per_class=20, num_val=500, num_test=1000, num_splits=runs)(data)
        data = T.NormalizeFeatures()(data)
        data.metric = "acc"

    # from paper 'Open Graph Benchmark Datasets for Machine Learning on Graphs'
    elif dataset_name in ('ogbn-arxiv', 'ogbn-products'):
        dataset = PygNodePropPredDataset(root=os.path.join(data_dir, 'ogb'), name=dataset_name)
        data = dataset[0]
        split_idx = dataset.get_idx_split()
        n = data.x.shape[0]
        data.train_mask = torch.zeros(n, dtype=bool).scatter_(0, split_idx['train'], True)
        data.val_mask = torch.zeros(n, dtype=bool).scatter_(0, split_idx['valid'], True)
        data.test_mask = torch.zeros(n, dtype=bool).scatter_(0, split_idx['test'], True)
        data.y = data.y.squeeze(dim = -1)
        data.metric = "acc"

    elif dataset_name in ('ogbn-proteins'):
        dataset = PygNodePropPredDataset(root=os.path.join(data_dir, 'ogb'), name=dataset_name)
        data = dataset[0]
        split_idx = dataset.get_idx_split()
        n = data.node_species.shape[0]
        data.edge_attr = None
        data.train_mask = torch.zeros(n, dtype=bool).scatter_(0, split_idx['train'], True)
        data.val_mask = torch.zeros(n, dtype=bool).scatter_(0, split_idx['valid'], True)
        data.test_mask = torch.zeros(n, dtype=bool).scatter_(0, split_idx['test'], True)
        data.x = data.node_species
        data.metric = "mc roc auc"

    # from paper 'A critical look at the evaluation of GNNs under heterophily: are we really making progress?'
    elif dataset_name in ('roman-empire', 'amazon-ratings', 'minesweeper', 'tolokers', 'questions'):
        dataset = HeterophilousGraphDataset(root=os.path.join(data_dir, 'HeterophilousGraphDataset'), name=dataset_name)
        data = dataset[0]
        data.metric = "acc" if dataset_name in ('roman-empire', 'amazon-ratings') else "roc auc"

    # from paper 'Geom-gcn: Geometric graph convolutional networks'
    elif dataset_name in ('actor'):
        dataset = Actor(root=os.path.join(data_dir, 'Actor'))
        data = dataset[0]
        data.metric = "acc"
    elif dataset_name in ('cornell', 'texas', 'wisconsin'):
        dataset = WebKB(root=os.path.join(data_dir, 'WebKB'), name=dataset_name)
        data = dataset[0]
        data.metric = "acc"
    elif dataset_name in ('squirrel', 'chameleon'):
        dataset = WikipediaNetwork(root=os.path.join(data_dir, 'WikipediaNetwork'), name=dataset_name, geom_gcn_preprocess=True)
        data = dataset[0]
        data.metric = "acc"

    else:
        raise ValueError('Invalid dataname')
    
    # manual random split
    if train_ratio is not None:
        data = T.RandomNodeSplit('train_rest', num_val=int(n*0.1), num_test=int(n*(0.9-train_ratio)), num_splits=runs)(data)
    
    # mask vectors should have shape [instances, runs]
    if len(data.train_mask.shape) == 1:
        data.train_mask = data.train_mask.unsqueeze(dim=-1)
        data.val_mask = data.val_mask.unsqueeze(dim=-1)
        data.test_mask = data.test_mask.unsqueeze(dim=-1)

    # postprocess
    if dataset_name not in ('ogbn-proteins'):
        data.onehot_y = F.one_hot(data.y).float()
    else:
        data.onehot_y = data.y.float()


    return data

def dataset_statistics(data, dataset_name, verbose = True):
    '''
    k: number of classes
    n: number of instances / number of nodes
    d: input dimension / number fo node features
    e: number of edges
    '''
    k, n, d, e = data.onehot_y.shape[1], data.x.shape[0], data.x.shape[1], data.edge_index.shape[1]
    if verbose:
        print(f"dataset {dataset_name} | num nodes {n} | num edge {e} | num node feats {d} | num classes {k}")
        print(f"dataset {dataset_name} | train nodes {data.train_mask[:,0].sum().item()} | valid nodes {data.val_mask[:,0].sum().item()} | test nodes {data.test_mask[:,0].sum().item()}")
    return k, n, d, e
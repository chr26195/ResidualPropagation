import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from utils import *

NORMALIZATION = {
    'None': nn.Identity,
    'LayerNorm': nn.LayerNorm,
    'BatchNorm': nn.BatchNorm1d
}

class MLP(nn.Module): 
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, dropout, normalization):
        super().__init__()
        self.dropout = dropout
        self.num_layers = num_layers
        self.activation = F.relu
        
        self.fcs = nn.ModuleList([])
        self.fcs.append(nn.Linear(input_dim, hidden_dim, bias=True))
        for _ in range(self.num_layers - 2): 
            self.fcs.append(nn.Linear(hidden_dim, hidden_dim, bias=True))
        self.fcs.append(nn.Linear(hidden_dim, output_dim, bias=True)) 

        self.normalization = nn.ModuleList([])
        self.normalization.append(NORMALIZATION[normalization](hidden_dim))
        for _ in range(self.num_layers - 2): 
            self.normalization.append(NORMALIZATION[normalization](hidden_dim))

        # self.reset_parameters()
    
    def reset_parameters(self):
        for fc in self.fcs: 
            nn.init.xavier_uniform_(fc.weight, gain=1.414)
            nn.init.zeros_(fc.bias)

    def construct_adj(self, edge_index, num_nodes):
        self.A = construct_sparse_adj(edge_index, num_nodes=num_nodes, type='DAD')

    def forward(self, x):
        x = self.fcs[0](x)
        x = self.normalization[0](x)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        for i in range(1, self.num_layers - 1):
            x = self.fcs[i](x) 
            x = self.normalization[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.fcs[-1](x) 
        return x


class GCN(nn.Module): 
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, dropout, normalization):
        super().__init__()
        self.dropout = dropout
        self.num_layers = num_layers
        
        self.activation = F.relu
        
        self.A = None

        self.fcs = nn.ModuleList([])
        self.fcs.append(nn.Linear(input_dim, hidden_dim, bias=True))
        for _ in range(self.num_layers - 2): 
            self.fcs.append(nn.Linear(hidden_dim, hidden_dim, bias=True))
        self.fcs.append(nn.Linear(hidden_dim, output_dim, bias=True)) 

        self.normalization = nn.ModuleList([])
        self.normalization.append(NORMALIZATION[normalization](hidden_dim))
        for _ in range(self.num_layers - 2): 
            self.normalization.append(NORMALIZATION[normalization](hidden_dim))
        # self.reset_parameters()
    
    def reset_parameters(self):
        for fc in self.fcs: 
            nn.init.xavier_uniform_(fc.weight, gain=1.414)
            nn.init.zeros_(fc.bias)

    def construct_adj(self, edge_index, num_nodes):
        self.A = construct_sparse_adj(edge_index, num_nodes=num_nodes, type='DAD')
        self.A = self.A.to_dense()

    def forward(self, x, use_A=True):
        for i in range(self.num_layers - 1):
            if use_A: x = self.A @ x
            x = self.fcs[i](x) 
            x = self.normalization[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        if use_A: x = self.A @ x
        x = self.fcs[-1](x) 
        return x
    

MODELS = {
    'MLP': MLP,
    'GCN': GCN
}
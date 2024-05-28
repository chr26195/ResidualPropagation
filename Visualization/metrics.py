import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call, jacrev

@torch.no_grad()
def empirical_ntk(model, X, type='nknk'):
    model.eval()
    def f(params, x):
        return functional_call(model, params, x)

    jacobian = jacrev(f, argnums=0)(dict(model.named_parameters()), X)
    jacobian = torch.cat([v.flatten(2) for v in jacobian.values()], dim=-1)

    if type == 'nknk':
        ntk_mat = torch.einsum('nkm,NKm->nkNK', jacobian, jacobian)
    elif type == 'nn':
        ntk_mat = torch.einsum('nkm,NKm->nN', jacobian, jacobian)
    return ntk_mat

@torch.no_grad()
def centered_kernel_alignment(K1, K2): # K1: [n, n], K2: [n, n]
    centered_K1 = K1 - K1.mean(dim=0, keepdim=True) - K1.mean(dim=1, keepdim=True) + K1.mean()
    centered_K2 = K2 - K2.mean(dim=0, keepdim=True) - K2.mean(dim=1, keepdim=True) + K2.mean()
    alignment = (centered_K1 * centered_K2).sum() / (torch.norm(centered_K1, p='fro') * torch.norm(centered_K2, p='fro'))
    return alignment.item()
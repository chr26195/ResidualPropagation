from sklearn.metrics import roc_auc_score, f1_score
import numpy as np
import torch

@torch.no_grad()
def eval_acc(y_true, y_pred): 
    '''
    y_true should have shape [instances, 1]
    y_pred should have shape [instances, k]
    '''
    acc_list = []
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.argmax(dim=-1, keepdim=True).detach().cpu().numpy()

    for i in range(y_true.shape[1]):
        is_labeled = y_true[:, i] == y_true[:, i]
        correct = y_true[is_labeled, i] == y_pred[is_labeled, i]
        acc_list.append(float(np.sum(correct))/len(correct))

    return sum(acc_list) / len(acc_list)

@torch.no_grad()
def eval_rocauc(y_true, y_pred):
    '''
    y_true should have shape [instances, 1]
    y_pred should have shape [instances, 1]
    '''
    if y_pred.shape[1] == 2:
        y_pred = y_pred[:,1]
    return roc_auc_score(y_true.cpu().numpy(), y_pred.cpu().detach().numpy()).item()

@torch.no_grad()
def eval_f1(y_true, y_pred):
    acc_list = []
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.argmax(dim=-1, keepdim=True).detach().cpu().numpy()

    for i in range(y_true.shape[1]):
        f1 = f1_score(y_true, y_pred, average='micro')
        acc_list.append(f1)

    return sum(acc_list) / len(acc_list)


METRICS = {'acc': eval_acc, 'f1': eval_f1, 'roc auc': eval_rocauc}
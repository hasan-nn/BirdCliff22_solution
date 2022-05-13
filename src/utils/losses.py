from turtle import forward
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import math

class ArgMax(nn.Module):

    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.argmax(x, dim=self.dim)

class Clamped_Sigmoid(nn.Module):
    def __init__(self,min=1e-2,max=1.0-1e-2):
        super().__init__()
        self._min = min
        self._max = max
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(x)
        x = torch.clamp(x,min=self._min,max=self._max)
        return x


class Activation(nn.Module):

    def __init__(self, name, **params):
        super().__init__()
        if name is None or name == 'identity':
            self.activation = nn.Identity(**params)
        elif name == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif name == 'clamped_sigmoid':
            self.activation = Clamped_Sigmoid(**params)
        elif name == 'softmax2d':
            self.activation = nn.Softmax(dim=1, **params)
        elif name == 'softmax':
            self.activation = nn.Softmax(**params)
        elif name == 'logsoftmax':
            self.activation = nn.LogSoftmax(**params)
        elif name == 'argmax':
            self.activation = ArgMax(**params)
        elif name == 'argmax2d':
            self.activation = ArgMax(dim=1, **params)
        elif callable(name):
            self.activation = name(**params)
        else:
            raise ValueError('Activation should be callable/sigmoid/softmax/logsoftmax/None; got {}'.format(name))

    def forward(self, x):
        return self.activation(x)
    
class FocalLoss(nn.Module):
    __name__='focal_loss'
    def __init__(self, alpha=1, gamma=2, logits=True, reduction = 'mean'):
        super(FocalLoss, self).__init__()
        assert reduction in ['sum','mean','none']
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduction = reduction

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        return F_loss

class BCELoss(nn.Module):
    __name__='bce_loss'
    def __init__(self,logits=True,reduction='mean'):
        super().__init__()
        self.logits = logits
        self.bce = nn.BCELoss(reduction=reduction) if(not(logits)) else nn.BCEWithLogitsLoss(reduction=reduction)
    def forward(self,y_pr,y_gt):
        bce = self.bce(y_pr,y_gt)
        return bce

class FocalLossV2(nn.Module):
    __name__='focal_lossV2'
    def __init__(self, alpha=1, gamma=2, logits=True,smoothing=0.0):
        super().__init__()
        self.criterion = FocalLoss(alpha=alpha,gamma=gamma,logits=logits,reduction='none')
        self.smoothing = smoothing

    def forward(self,y_pred,y_gt,weights=None,rating=None):

        if self.smoothing > 0.0:
            y_gt = torch.clamp(y_gt,min=self.smoothing,max=1.0-self.smoothing)
            
        loss = self.criterion(y_pred,y_gt)
        #print(loss.shape)
        if weights is not None:
            loss *= weights


        #loss = loss.mean(dim=-1)
        if rating is not None:
            loss *= rating.unsqueeze(-1)
        loss = loss.mean()
        return loss

class FocalLossV3(nn.Module):
    __name__='focal_lossV3'
    def __init__(self, alpha=1, gamma=2, logits=True,smoothing=0.0):
        super().__init__()
        self.eps = 1e-6
        self.criterion = FocalLoss(alpha=alpha,gamma=gamma,logits=logits,reduction='none')
        self.smoothing = smoothing

    def forward(self,y_pred,y_gt,weights=None,rating=None):


        if self.smoothing > 0.0:
            y_gt = torch.clamp(y_gt,min=self.smoothing,max=1.0-self.smoothing)
            
        loss = self.criterion(y_pred,y_gt)

        if weights is not None:
            loss = loss * weights
            loss = loss.sum(dim=-1)
            #weights_sum = 
            loss = loss / (weights.sum(dim=-1) + self.eps)
        else:
            loss = loss.mean(dim=-1)

        if rating is not None:
            loss = loss * rating
            loss = loss.sum() / (rating.sum() + self.eps)

        else:
            loss = loss.mean()

        return loss

class BCELossV2(nn.Module):
    __name__='bce_lossV2'
    def __init__(self, logits=True):
        super().__init__()
        self.criterion = BCELoss(logits=logits,reduction='none')

    def forward(self,y_pred,y_gt,weights=None,rating=None):
        num_pos = (y_gt > 0.0).sum(dim=-1).unsqueeze(-1)
        num_neg = (y_gt == 0.0).sum(dim=-1).unsqueeze(-1)
        intr_weights = num_pos * y_gt + num_neg * (1 - y_gt)

        loss = self.criterion(y_pred,y_gt)
        if weights is not None:
            loss *= weights
        loss = (loss / intr_weights).sum(dim=-1) 
        #loss = loss.sum(dim=-1) / num_pos
        #loss = loss.mean(dim=-1)
        if rating is not None:
            loss *= rating
        loss = loss.mean()
        return loss

class CoW_L1Loss(nn.Module):
    __name__= 'cow_l1_loss'
    def __init__(self):
        super().__init__()
        self.criterion = nn.L1Loss(reduction='mean')
    
    def forward(self,pred_weights,gt_weights):
        
        gt_weights = gt_weights.flatten()
        pred_weights = pred_weights.flatten()

        indxs = (gt_weights == 0.0).nonzero(as_tuple=True)[0]
        if indxs.shape[0] == 0:
            loss = torch.tensor([0.0], requires_grad=True).to(gt_weights.device)
            return loss

        pred_weights = pred_weights.gather(dim=0,index=indxs)
        gt_weights = gt_weights.gather(dim=0,index=indxs)

        loss = self.criterion(pred_weights,gt_weights)
        return loss


def get_loss(name, **kwargs):
    mapping = {
        'bce' : BCELoss,
        'focal' : FocalLoss,
        'focalv2' : FocalLossV2,
        'focalv3' : FocalLossV3,
        'bceV2' : BCELossV2,
        'cow_l1_loss' : CoW_L1Loss
    }
    loss_fn = mapping[name]
    return loss_fn(**kwargs)

if __name__ == '__main__':
    #"""
    y_pred = torch.tensor([
        [0.9,0.5,0.1,0.1,0.1],
        [0.9,0.4,0.3,0.1,0.1],
        [0.9,0.9,0.9,0.8,0.2],
        [1,0.9,0.2,0.5,0.1],
    ]).float()

    y_gt = torch.tensor([
        [1,1,1,1,0],
        [1,1,1,1,1],
        [1,1,1,1,1],
        [1,1,1,1,1],
    ]).float()




    criterion = CoW_L1Loss()

    with torch.no_grad():
        loss = criterion(y_pred,y_gt)
    print(loss)
    #"""
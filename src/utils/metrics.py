import torch
import torch.nn as nn
#import pandas as pd

def _threshold(x, threshold=None):
    if threshold is not None:
        return (x > threshold).type(x.dtype)
    else:
        return x



class MacroF1:
    def __init__(self,threshold = 0.5):
        self.thresh = threshold
        self.eps = 1e-8
        self.f1_score = 0.0
        self.count = 0.0

    @torch.no_grad()
    def __call__(self, y_pr,y_gt):
        batch,n_classes = y_gt.size()
        y_pr = _threshold(y_pr,threshold=self.thresh)
        eps = self.eps

        tp = torch.sum(y_pr * y_gt,dim=-1)
        tn = torch.sum((1 - y_pr) * (1 - y_gt),dim=-1)
        fp = torch.sum(y_pr,dim=-1) - tp
        fn = torch.sum(y_gt,dim=-1) - tp


        precision = (tp + eps) / (tp + fp + eps)
        recall =(tp + eps) / (tp + fn + eps)
        f1_scores = ((2 * recall * precision) + eps) / (recall + precision + eps) 

        self.f1_score = (self.f1_score * self.count + f1_scores.sum()) / (self.count + batch)
        self.count += batch

        return self.f1_score

class MicroF1:
    def __init__(self,threshold = 0.5):
        self.thresh = threshold
        self.eps = 1e-8
        self.reset()

    def reset(self):
        self.tp = 0.
        self.fp = 0.
        self.fn = 0.
        self.tn = 0.

    def get_counts(self):
        return {
            'tp' : self.tp,
            'fp' : self.fp,
            'tn' : self.tn,
            'fn' : self.fn
        }

    @torch.no_grad()
    def __call__(self, y_pr,y_gt):
        y_pr = _threshold(y_pr,threshold=self.thresh)
        eps = self.eps

        tp = torch.sum(y_pr * y_gt,dim=0)
        tn = torch.sum((1.0 - y_pr) * (1.0 - y_gt),dim=0)
        fp = torch.sum(y_pr,dim=0) - tp
        fn = torch.sum(y_gt,dim=0) - tp

        self.tp += tp
        self.fp += fp
        self.fn += fn
        self.tn += tn

        tp, fp, tn, fn = self.tp, self.fp, self.tn, self.fn

        recall =(tp + eps) / (tp + fn + eps)
        precision = (tp + eps) / (tp + fp + eps)
        f1_scores = ((2 * recall * precision) + eps) / (recall + precision + eps)
        f1_score = f1_scores.mean()
        return f1_score,f1_scores


if __name__ == '__main__':
    pass
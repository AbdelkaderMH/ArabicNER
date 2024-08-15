import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GCE(nn.Module):
    def __init__(self, num_classes=10, q=0.7):
        super(GCE, self).__init__()
        self.q = q
        self.num_classes = num_classes

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-6, max=1.0)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        loss = (1. - torch.pow(torch.sum(label_one_hot * pred, dim=1), self.q)) / self.q
        return loss.mean()

class BS(object):
    def __call__(self, outputs):
        ## hard booststrapping
        # targets = torch.argmax(outputs, dim=1)
        # return nn.CrossEntropyLoss()(outputs, targets)

        ## soft bootstrapping
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(-torch.log(probs+1e-6)*probs, dim=1))


class DAL(nn.Module):
    def __init__(self, num_classes=10, q=1.5, lamb=1.0):
        super(DAL, self).__init__()
        self.gce = GCE(num_classes=num_classes, q=q)
        self.bs = BS()
        self.num_classes = num_classes
        self.q = q
        self.lamb = lamb

    def forward(self, pred, labels):
        loss = self.gce(pred, labels) + self.lamb * self.bs(pred) / (self.q * np.log(self.num_classes))
        return loss
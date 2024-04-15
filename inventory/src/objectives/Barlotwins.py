import math
import torch
import numpy as np
import torch


# def l2_normalize(x, dim=1):
#     return x / torch.sqrt(torch.sum(x**2, dim=dim).unsqueeze(dim))

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class BarlowtinsObjective(torch.nn.Module):
    def __init__(self, outputs1, outputs2, T=0.005, push_only=False):
        super().__init__()
        self.outputs1 = (outputs1 - torch.mean(outputs1, dim=0)) / torch.std(outputs1, dim=0)
        self.outputs2 = (outputs2 - torch.mean(outputs2, dim=0)) / torch.std(outputs2, dim=0)
        self.T = T
        # self.scale_loss = 1/32
        # self.lambd = 3.9e-6
        # https://github.com/sally20921/barlow-twins/blob/b6ca2b36eda8d958a17cf264da7a823daa9b5623/model.py

    def get_loss(self):
        batch_size = self.outputs1.size(0)  # batch_size x out_dim
        c = (self.outputs1.T @ self.outputs2) / batch_size

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()#.mul(self.scale_loss)
        off_diag = off_diagonal(c).pow_(2).sum()#.mul(self.scale_loss)
        loss = on_diag + self.T * off_diag
        
        return loss

def test():
    A = torch.randn(256, 128)
    B = torch.randn(256, 128)
    objective = BarlowtinsObjective(A, B)
    loss = objective.get_loss()    
   

if __name__ == '__main__':
   
    import timeit
    print(timeit.timeit("test()", setup="from __main__ import test", number=10000))
    
    

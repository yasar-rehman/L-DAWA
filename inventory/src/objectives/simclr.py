import math
import torch
import numpy as np
import torch.nn.functional as F
# from ..utils.utils import l2_normalize
# import torch

# def l2_normalize(x, dim=1):
    # return x / torch.sqrt(torch.sum(x**2, dim=dim).unsqueeze(dim))

class SimCLRObjective(torch.nn.Module):

    def __init__(self, outputs1, outputs2, t, push_only=False):
        super().__init__()
        # self.outputs1 = l2_normalize(outputs1, dim=1)
        # self.outputs2 = l2_normalize(outputs2, dim=1)
        self.outputs1 = F.normalize(outputs1, p=2, dim=1)
        self.outputs2 = F.normalize(outputs2, p=2, dim=1)

        self.t = t
        self.push_only = push_only

    def get_loss(self):
        batch_size = self.outputs1.size(0)  # batch_size x out_dim
        witness_score = torch.sum(self.outputs1 * self.outputs2, dim=1) # dot prodcut followed by sum = cosine similarity
        if self.push_only:
            # Don't pull views together.
            witness_score = 0
        outputs12 = torch.cat([self.outputs1, self.outputs2], dim=0)
        witness_norm = self.outputs1 @ outputs12.T

        witness_norm = torch.logsumexp(witness_norm / self.t, dim=1)
        loss = -torch.mean(witness_score / self.t - witness_norm)
        return loss
  
def device_as(t1, t2):
   """
   Moves t1 to the device of t2
   """
   return t1.to(t2.device)

class ContrastiveLoss(torch.nn.Module):
   """
   Vanilla Contrastive loss, also called InfoNceLoss as in SimCLR paper
   """
   def __init__(self, batch_size, temperature=0.5):
       super().__init__()
       self.batch_size = batch_size
       self.temperature = temperature
       self.mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float()

   def calc_similarity_batch(self, a, b):
       representations = torch.cat([a, b], dim=0)
       return F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

   def forward(self, proj_1, proj_2):
       """
       proj_1 and proj_2 are batched embeddings [batch, embedding_dim]
       where corresponding indices are pairs
       z_i, z_j in the SimCLR paper
       """
       batch_size = proj_1.shape[0]
       z_i = F.normalize(proj_1, p=2, dim=1)
       z_j = F.normalize(proj_2, p=2, dim=1)

       similarity_matrix = self.calc_similarity_batch(z_i, z_j)

       sim_ij = torch.diag(similarity_matrix, batch_size)
       sim_ji = torch.diag(similarity_matrix, -batch_size)

       positives = torch.cat([sim_ij, sim_ji], dim=0)

       nominator = torch.exp(positives / self.temperature)

       denominator = device_as(self.mask, similarity_matrix) * torch.exp(similarity_matrix / self.temperature)

       all_losses = -torch.log(nominator / torch.sum(denominator, dim=1))
       loss = torch.sum(all_losses) / (2 * self.batch_size)
       return loss

def test():
    A = torch.randn(256, 128)
    B = torch.randn(256, 128)
    objective = SimCLRObjective(A, B, 0.5)
    objective2 = ContrastiveLoss(batch_size=A.shape[0], temperature=0.5)
    
    loss_sim = objective.get_loss()
    loss_sim2 = objective2(A,B)

    print(f'The viewmaker loss is {loss_sim} and the infonce loss is {loss_sim2}')

   

if __name__ == '__main__':
   
    import timeit
    # print(timeit.timeit("test()", setup="from __main__ import test", number=10000))
    test()

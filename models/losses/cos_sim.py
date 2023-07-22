import torch
import torch.nn as nn

from torch.nn import functional as F



class CosSim_Loss(nn.Module):
    def __init__(self):
        super(CosSim_Loss, self).__init__()
 

    def forward(self, pred): 

        batch_size = pred.shape[0]
        n_map = pred.shape[1]

        cos_sim = torch.ones((batch_size, n_map, n_map))
        for b in range(batch_size):
           for i in range(n_map):
               for j in range(i+1, n_map):
                   X_i = pred[b, i].view(-1)
                   X_j = pred[b, j].view(-1)
                   cos_sim_ij = F.cosine_similarity(X_i, X_j, dim=0)

                   cos_sim[b, i, j] = cos_sim_ij
                   cos_sim[b, j, i] = cos_sim_ij

        cos_loss = torch.mean(cos_sim)

        return cos_loss


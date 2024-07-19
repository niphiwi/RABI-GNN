import torch
import torch.nn as nn

def mse(pred_dist, true_dist):
    """ Calculate mean squared error."""
    return nn.functional.mse_loss(pred_dist, true_dist)

def rmse(pred_dist, true_dist):
    """ Calculate root mean squared error."""
    with torch.no_grad():
        rmse = torch.sqrt(nn.functional.mse_loss(pred_dist, true_dist))
    return rmse
    
def kld(pred_dist, true_dist):
    """ Calculate Kullback-Leibler divergence."""
    with torch.no_grad():
        softmax = nn.LogSoftmax(dim=2)
        kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)
        kld = kl_loss(softmax(pred_dist.contiguous().view(1,1,-1)), softmax(true_dist.view(1,1,-1)))
    return kld


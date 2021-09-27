import torch

def masked_mae_loss(y_pred, y_true):
    """Calculate the loss of Mae with mask"""
    mask = (y_true != 0).float()
    mask /= mask.mean()
    loss = torch.abs(y_pred - y_true)
    loss = loss * mask

    loss[loss != loss] = 0 #This operation replaces all NaN with 0

    return loss.mean()
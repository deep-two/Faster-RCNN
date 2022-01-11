from numpy import shares_memory
import torch

def _smooth_l1_loss(bb_pred, bb_target, bbox_inside_weights, bbox_outside_weights, sigma=1.0, dim=[1]):
    sigma2 = sigma ** 2
    diff = bbox_inside_weights * (bb_pred - bb_target)
    smooth_l1 = torch.where(torch.abs(diff) < 1. /sigma2, 
        0.5 * sigma2 * torch.square(diff), torch.abs(diff) - 0.5/sigma2).type_as(diff)
    out_weighted = bbox_outside_weights * smooth_l1

    loss = out_weighted.sum(dim)
    loss = loss.mean()

    return loss


if __name__ == "__main__":
    sigma2 = 9
    diff = torch.tensor([1/sigma2 - 0.26, 1/sigma2 + 0.002])
    
    smooth_l1 = torch.where(torch.abs(diff) < 1. /sigma2, 
        0.5 * sigma2 * torch.square(diff), torch.abs(diff) - 0.5/sigma2).type_as(diff)
    print(torch.abs(diff) - 0.5/sigma2)
    print(0.5 * sigma2 * torch.square(diff))
    print(smooth_l1)

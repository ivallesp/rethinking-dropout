import torch
import torch.nn.functional as F


def CXE(predicted, target):
    predicted = F.softmax(predicted, dim=1)
    return -(target * torch.log(predicted + 1e-9)).sum(dim=1).mean()

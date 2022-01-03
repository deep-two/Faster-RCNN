import torch
import torch.nn as nn
from bbox_transform import bbox_transform_inv

class _ProposalLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input):
        pass

    def backward(self):
        pass

    def reshape(self):
        pass


if __name__ == "__main__":
    pass
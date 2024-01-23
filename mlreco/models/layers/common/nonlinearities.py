import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME

class MinkowskiMish(nn.Module):
    '''
    Mish Nonlinearity: https://arxiv.org/pdf/1908.08681.pdf
    '''
    def __init__(self):
        super().__init__()

    def forward(self, input):
        out = F.softplus(input.F)
        out = torch.tanh(out)
        out = out * input.F
        return ME.SparseTensor(out, coords_key = input.coords_key,
                coords_manager = input.coords_man)

    def __repr__(self):
        return self.__class__.__name__ + '()'

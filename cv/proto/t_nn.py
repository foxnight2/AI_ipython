
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Sequence


MODULES = {}

class Register:
    def __init__(self) -> None:
        pass

    def register_module(self, ):
        pass

    def register_dataset(self, ):
        pass
    


class ConvNorm2d(torch.nn.Module):
    def __init__(self, ):
        super().__init__()

    def forward(self, x):
        pass
    

def _to_list(x, n):
    '''
    '''
    if isinstance(x, Sequence):
        return list(x)
    else:
        return [x, ] * n


class MultiScaleResize(nn.Module):
    '''MultiScaleResize
    '''
    def __init__(self, size):
        super().__init__()
        self.size = _to_list(size, 1)

    def forward(self, x):
        _idx = torch.randint(len(self.size), (1, )).item()
        return F.interpolate(x, self.size[_idx])



class Detach(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.detach()


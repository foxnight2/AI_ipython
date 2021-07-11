import torch
import torch.nn as nn 



class ResNet(nn.Module):
    def __init__(self, layers):
        super().__init__()
    
    def forward(self, x, y=None):
        
        if y is not None:
            x = x + y
            
        return x

    
modules = {
    'Conv2d': nn.Conv2d,
    'ReLU': nn.ReLU,

    'ResNet': ResNet,
}
import torch
import torch.nn as nn 

from torch.utils.data import Dataset


class ResNet(nn.Module):
    def __init__(self, layers):
        super().__init__()
    
    def forward(self, x, y=None):
        
        if y is not None:
            x = x + y
            
        return x

    
class DummyDataset(Dataset):
    def __init__(self, ):
        
        self.data = torch.rand(3, 3, 10, 10)
        
    def __len__(self, ):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    
modules = {
    'Conv2d': nn.Conv2d,
    'ReLU': nn.ReLU,

    'ResNet': ResNet,
    
    'DummyDataset': DummyDataset,
}


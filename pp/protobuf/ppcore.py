import torch
import torch.nn as nn 

from torch.utils.data import Dataset


class ResNet(nn.Module):
    def __init__(self, layers):
        super().__init__()
    
    def forward(self, x, y=None):
        
        if y is not None:
            x = x + y 
            
        return x + 1



class Yolov3Target(nn.Module):
    def __init__(self, strides):
        super().__init__()
        
        self.stride = strides
    
    def forward(self, feats, label=None):
        
        if self.training:
            return feats + 2
        else:
            return feats + 1
    
    def _built_target(self, x):
        pass
    
    

class Yolov3Loss(nn.Module):
    def __init__(self, strides):
        super().__init__()

        self.stride = strides
    
    def forward(self, feats, target):
        return feats + 1
    
    
    
class DummyDataset(Dataset):
    def __init__(self, n):
        self.data = torch.rand(n, 3, 10, 10)
        self.label = torch.rand(n, 3, 10, 10)

    def __len__(self, ):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]
    
    def format_output(self, ):
        pass
    
    
    
    
modules = {
    'Conv2d': nn.Conv2d,
    'ReLU': nn.ReLU,
    'ResNet': ResNet,
    
    'Yolov3Target': Yolov3Target,
    'Yolov3Loss': Yolov3Loss,
    
    'DummyDataset': DummyDataset,
}


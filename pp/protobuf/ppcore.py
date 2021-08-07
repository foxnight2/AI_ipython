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



class Yolov3Target(nn.Module):
    def __init__(self, strides):
        super().__init__()
        
        self.stride = strides
    
    def forward(self, feats, label):
        
        if self.training:
            print(f'self.training: {self.training}')
            return feats
        else:
            return feats
    
    
    
class Yolov3Loss(nn.Module):
    def __init__(self, strides):
        super().__init__()

        self.stride = strides
    
    def forward(self, feats, target):
        return feats
    
    
    
class DummyDataset(Dataset):
    def __init__(self, ):
        self.data = torch.rand(50, 3, 10, 10)
        self.label = torch.rand(50, 3, 10, 10)

    def __len__(self, ):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]
    
    
modules = {
    'Conv2d': nn.Conv2d,
    'ReLU': nn.ReLU,
    'ResNet': ResNet,
    
    'Yolov3Target': Yolov3Target,
    'Yolov3Loss': Yolov3Loss,
    
    'DummyDataset': DummyDataset,
}


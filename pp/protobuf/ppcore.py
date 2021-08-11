import torch
import torch.nn as nn 

import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms

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
        self.label = torch.randperm(100)[:n]

    def __len__(self, ):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]
    
    def format_output(self, ):
        pass
    
    

class Resnet18(nn.Module):
    def __init__(self, pretrained, ):
        super().__init__()
        
        self.resnet18 = models.resnet18(pretrained)
    
    def forward(self, data):
        return self.resnet18(data)
    

# class CIFAR10(datasets.CIFAR10):
#     pass


modules = {
    'Conv2d': nn.Conv2d,
    'ReLU': nn.ReLU,
    'ResNet': ResNet,
    'Resnet18': Resnet18,
    'BCELoss': torch.nn.BCELoss,
    'CrossEntropyLoss': torch.nn.CrossEntropyLoss,
    
    'Yolov3Target': Yolov3Target,
    'Yolov3Loss': Yolov3Loss,
    
    'DummyDataset': DummyDataset,
    'CIFAR10': datasets.CIFAR10,
    
    'ToTensor': transforms.ToTensor,
}


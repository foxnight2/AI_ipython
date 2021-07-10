import torch
import torch.nn as nn 


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride=1, padding=0, ):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel, stride, )
        
    def forward(self, x):
        
        x = self.conv(x)
        
        return x
    
    
class ResNet(nn.Module):
    def __init__(self, layers):
        super().__init__()
    
    def forward(self, x):
        pass
    


modules = {
    'Conv2d': Conv2d,
    'ResNet': ResNet
}
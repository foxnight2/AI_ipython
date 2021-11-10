

import torch
import torch.nn as nn

from torch.nn.common_types import _size_2_t

from typing import Union
import time

class ConvBN2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: _size_2_t, stride: _size_2_t = 1, padding: Union[str, _size_2_t] = 0, dilation: _size_2_t = 1, groups: int = 1, bias: bool = False, padding_mode: str = 'zeros', device=None, dtype=None) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))

    def forward_eval(self, x):
        return self.conv(x)


def fuse(model):
    '''fuse
    '''
    for m in model.modules():
        if isinstance(m, ConvBN2d) and hasattr(m, 'bn') and  hasattr(m, 'conv'):
            m.conv = nn.utils.fusion.fuse_conv_bn_eval(m.conv, m.bn)
            delattr(m, 'bn')
            m.forward = m.forward_eval

    return model



if __name__ == '__main__':

    x = torch.rand(1, 3, 10, 10)
    m = ConvBN2d(3, 10, 3, 2, 1)
    m.eval()

    N = 100

    tic = time.time()
    for _ in range(N):
        out = m(x)
    print(time.time() - tic, out.mean())


    m = fuse(m)

    tic = time.time()
    for _ in range(N):
        out = m(x)
    print(time.time() - tic, out.mean())



from typing import Tuple, Union
import math

from pdll.backend import executor

def im2col(data: Union[executor.support_types], kernel: Tuple[int, ...], stride: Tuple[int, ...], padding: Tuple[int, ...], dilation: int=1):
    '''im2col
    N C H W -> n h w c k k
    '''
    n, c, h, w = data.shape
    out_h = math.floor((h + 2 * padding[0] - dilation * (kernel[0] - 1) - 1) / stride[0] + 1)
    out_w = math.floor((w + 2 * padding[1] - dilation * (kernel[1] - 1) - 1) / stride[1] + 1)

    # hpad = (padding[0]//2, padding[0] - padding[0]//2)
    # wpad = (padding[1]//2, padding[1] - padding[1]//2)
    hpad = (padding[0], padding[1])
    wpad = (padding[2], padding[3])
    data = executor.np.pad(data, pad_width=((0, 0), (0, 0), hpad, wpad), mode='constant')
    
    matrix = executor.np.zeros((n, c, kernel[0], kernel[1], out_h, out_w))

    for i, i_data in enumerate(range(dilation * (kernel[0] - 1) + 1)[::dilation]):
        for j, j_data in enumerate(range(dilation * (kernel[1] - 1) + 1)[::dilation]):
            matrix[:, :, i, j, :, :] = data[:, :, i_data::stride[0], j_data::stride[1]][:, :, :out_h, :out_w]
        
    return matrix, out_h, out_w
    

def col2im(matrix: Union[executor.support_types], shape: Tuple[int, ...], kernel: Tuple[int, ...], stride: Tuple[int, ...], padding: Tuple[int, ...], dilation: int=1):
    '''
    matrix: (n, cin, hk, wk, hout, wout)
    '''
    _, _, _, _, ho, wo = matrix.shape
    # matrix = matrix.transpose(0, 3, 4, 5, 1, 2) # (n, c, hk, wk, ho, wo)

    hpad = (padding[0], padding[1])
    wpad = (padding[2], padding[3])
    data = executor.np.pad(executor.np.zeros(shape), pad_width=((0, 0), (0, 0), hpad, wpad), mode='constant',)
    _, _, H, W = data.shape
    
    for i, i_data in enumerate(range(dilation * (kernel[0] - 1) + 1)[::dilation]):
        iend = i_data + stride[0] * ho
        for j, j_data in enumerate(range(dilation * (kernel[1] - 1) + 1)[::dilation]):
            jend = j_data + stride[1] * wo
            data[:, :, i_data:iend:stride[0], j_data:jend:stride[1]] += matrix[:, :, i, j, :, :]

    return data[:, :, hpad[0]:H-hpad[1], wpad[0]:W-wpad[1]]

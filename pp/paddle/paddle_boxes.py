
import paddle 

import torchvision.ops as ops 

ops.box_convert


def box_convert(boxes, in_fmt, out_fmt):
    '''
    Args:
        boxes (Tensor[..., 4]): boxes which will be converted.
        in_fmt (str): Input format of given boxes.
        out_fmt (str): Output format of given boxes.

    Returns:
        Tensor[..., 4]
    '''
    assert in_fmt in ('xyxy', 'cxcywh') and out_fmt in ('xyxy', 'cxcywh'), ''

    if in_fmt == out_fmt:
        return boxes
    
    if in_fmt == 'xyxy' and out_fmt == 'cxcywh':
        x1, y1, x2, y2 = boxes.T
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        boxes = paddle.stack([cx, cy, w, h], dim=-1)

    elif in_fmt == 'cxcywh' and out_fmt == 'xyxy':
        cx, cy, w, h = boxes.T
        x1 = cx - w / 2.
        y1 = cy - h / 2.
        x2 = cx + w / 2.
        y2 = cy + h / 2.
        boxes = paddle.stack([x1, y1, x2, y2], dim=-1)

    return boxes
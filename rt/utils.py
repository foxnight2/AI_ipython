import torch
import torch.utils.data as data 

import torchvision
import torchvision.transforms as T 
import torchvision.transforms.functional as F


import os
import glob
import time 
import contextlib
import numpy as np
from PIL import Image 



class TimeProfiler(contextlib.ContextDecorator):
    def __init__(self, ):
        self.total = 0
        
    def __enter__(self, ):
        self.start = self.time()
        return self 
    
    def __exit__(self, type, value, traceback):
        self.total += self.time() - self.start
    
    def reset(self, ):
        self.total = 0
    
    def time(self, ):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.time()



class ToTensor(T.ToTensor):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, pic):
        if isinstance(pic, torch.Tensor):
            return pic 
        return super().__call__(pic)


class PadToSize(T.Pad):
    def __init__(self, size, fill=0, padding_mode='constant'):
        super().__init__(0, fill, padding_mode)
        self.size = size
        self.fill = fill

    def __call__(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be padded.

        Returns:
            PIL Image or Tensor: Padded image.
        """
        w, h = F.get_image_size(img)
        padding = (0, 0, self.size[0] - w, self.size[1] - h)
        return F.pad(img, padding, self.fill, self.padding_mode)


class Dataset(data.Dataset):
    def __init__(self, img_dir: str='', preprocess: T.Compose=None, device='cuda:0') -> None:
        super().__init__()

        self.device = device
        self.im_path_list = list(glob.glob(os.path.join(img_dir, '*.jpg')))

        if preprocess is None: 
            self.preprocess = T.Compose([
                    T.Resize(size=639, max_size=640),
                    PadToSize(size=(640, 640), fill=114),
                    ToTensor(),
                    T.ConvertImageDtype(torch.float),
            ])
        else:
            self.preprocess = preprocess

    def __len__(self, ):
        return len(self.im_path_list)


    def __getitem__(self, index):
        # im = Image.open(self.img_path_list[index]).convert('RGB')
        im = torchvision.io.read_file(self.im_path_list[index])
        im = torchvision.io.decode_jpeg(im, mode=torchvision.io.ImageReadMode.RGB, device=self.device)

        im = self.preprocess(im)
        
        blob = {
            'image': im, 
            'im_shape': torch.tensor([640., 640.]).to(im.device),
            'scale_factor': torch.tensor([1., 1.]).to(im.device),
        }

        return blob


    @staticmethod
    def post_process():
        pass

    @staticmethod
    def collate_fn():
        pass


def draw_result_ppdetr(m, blob):
    '''show result
    '''
    outputs = m(blob)
    preds = outputs.values()[0]
    preds = preds[preds[:, 1] > 0.5]

    im = (blob['image'][0] * 255).to(torch.uint8)
    im = torchvision.utils.draw_bounding_boxes(im, boxes=preds[:, 2:], width=2)
    # torchvision.utils.save_image(im, 'test.jpg')
    Image.fromarray(im.permute(1, 2, 0).cpu().numpy()).save(f'test_{i}.jpg')



def draw_result_yolo(blob, outputs, draw_score_threshold=0.25, name=''):
    '''show result
    Keys:
        'num_dets', 'det_boxes', 'det_scores', 'det_classes'
    '''    
    for i in range(blob['image'].shape[0]):
        det_scores = outputs['det_scores'][i]
        det_boxes = outputs['det_boxes'][i][det_scores > draw_score_threshold]
        
        im = (blob['image'][i] * 255).to(torch.uint8)
        im = torchvision.utils.draw_bounding_boxes(im, boxes=det_boxes, width=2)
        Image.fromarray(im.permute(1, 2, 0).cpu().numpy()).save(f'test_{name}_{i}.jpg')



def dummy_blob(batch_size=1, backend='torch'):
    '''
    '''
    if backend == 'torch':
        blob = {
            'image': torch.rand(batch_size, 3, 640, 640).to('cuda:0'),
            'im_shape': torch.tensor([[640., 640.]]).tile(batch_size, 1).to('cuda:0'),
            'scale_factor': torch.tensor([[1., 1.]]).tile(batch_size, 1).to('cuda:0'),
        }
        
    else:
        blob = {
            'image': np.random.rand(batch_size, 3, 640, 640).astype(np.float32),
            'im_shape': np.array([[640., 640.]]).repeat(batch_size, 0).astype(np.float32),
            'scale_factor': np.array([[1., 1.]]).repeat(batch_size, 0).astype(np.float32),
        }
    return blob

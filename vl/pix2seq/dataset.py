import torch
import torchvision
import torchvision.transforms
torchvision.disable_beta_transforms_warning()

from torchvision import datasets
from torchvision.transforms import v2 as transforms
from torchvision.transforms.v2 import functional as F
from torchvision.transforms.v2 import utils


import PIL 
import numpy as np
from collections import defaultdict
from typing import Any, Dict, List


def show(sample):
    import matplotlib.pyplot as plt
    import PIL

    from torchvision.transforms.v2 import functional as F
    from torchvision.utils import draw_bounding_boxes

    image, target = sample
    if isinstance(image, PIL.Image.Image):
        image = F.to_image_tensor(image)
    image = F.convert_dtype(image, torch.uint8)
    annotated_image = draw_bounding_boxes(image, target["boxes"], colors="yellow", width=3)

    fig, ax = plt.subplots()
    ax.imshow(annotated_image.permute(1, 2, 0).numpy())
    ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    fig.tight_layout()

    fig.show()



class PadToSize(transforms.Pad):
    def __init__(self, size, fill=(123, 117, 104), padding_mode='constant', padding_type='random') -> None:
        super().__init__(0)
        
        if isinstance(size, int):
            self.size = [size, size]
        else:
            self.size = list(size)
        
        self.fill = fill 
        self.padding_mode = padding_mode
        self.padding_type = padding_type
    
    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        '''
        '''
        h, w = utils.query_spatial_size(flat_inputs)
        new_h, new_w = self.size
        padding_type = self.padding_type
        
        p_w = max(new_w - w, 0)
        p_h = max(new_h - h, 0)
    
        if padding_type == 'lt':
            padding = [0, 0, p_w, p_h]

        elif padding_type == 'lb':
            padding = [0, p_h, p_w, 0]
        
        elif padding_type == 'rt':
            padding = [p_w, 0, 0, p_h]
        
        elif padding_type == 'rb':
            padding = [p_w, p_h, 0, 0]

        elif padding_type == 'center':
            padding = [p_w // 2, p_h // 2, p_w - p_w // 2, p_h - p_h//2]

        elif padding_type == 'random':
            _p_w = np.random.randint(p_w + 1)
            _p_h = np.random.randint(p_h + 1)
            padding = [_p_w, _p_h, p_w - _p_w, p_h - _p_h]
                
        return dict(padding=padding)

        
    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        # fill = self._fill[type(inpt)]
        return F.pad(inpt, padding=params['padding'], fill=self.fill, padding_mode=self.padding_mode)  # type: ignore[arg-type]



train_transforms = transforms.Compose(
    [
        transforms.RandomPhotometricDistort(),
        
        transforms.RandomZoomOut(
            side_range=(1, 3),
            # fill=defaultdict(lambda: 0, {PIL.Image.Image: (123, 117, 104)})
            fill=117,
        ),
        transforms.RandomIoUCrop(),

        transforms.Resize(639, max_size=640),
        PadToSize(640, fill=117),

        transforms.RandomHorizontalFlip(),
        transforms.ToImageTensor(),
        transforms.ConvertImageDtype(torch.float32),
        transforms.SanitizeBoundingBox(),
    ]
)


eval_transforms = transforms.Compose(
    [
        transforms.Resize(639, max_size=640),
        PadToSize(640, fill=117, padding_type='lt'),
        transforms.ToImageTensor(),
        transforms.ConvertImageDtype(torch.float32),
    ]
)


def get_coco_dataset(root, anno_file, transforms):
    '''
    '''
    dataset = datasets.CocoDetection(root, anno_file, transforms=transforms)
    dataset = datasets.wrap_dataset_for_transforms_v2(dataset)

    return dataset 


# data_loader = torch.utils.data.DataLoader(
#     dataset,
#     batch_size=2,
#     # We need a custom collation function here, since the object detection models expect a
#     # sequence of images and target dictionaries. The default collation function tries to
#     # `torch.stack` the individual elements, which fails in general for object detection,
#     # because the number of object instances varies between the samples. This is the same for
#     # `torchvision.transforms` v1
#     collate_fn=lambda batch: tuple(zip(*batch)),
# )

from torch.nn.utils.rnn import pad_sequence
class Tokenizer(object):
    def __init__(self, num_classes: int, num_bins: int, spatial_size: List[int], max_len=500) -> None:
        self.num_classes = num_classes
        self.num_bins = num_bins
        self.spatial_size = spatial_size
        self.max_len = max_len
    
        self.BOS_TOKEN = num_classes + num_bins
        self.EOS_TOKEN = self.BOS_TOKEN + 1
        self.PAD_TOKEN = self.EOS_TOKEN + 1

        self.vocab_size = num_classes + num_bins + 3

    def __call__(self, labels: torch.Tensor, boxes: torch.Tensor, shuffle: bool=True):
        '''
        Args:
            labels,
            boxes, [xmin, ymin, xmax, ymax]
        '''
        assert len(labels) == len(boxes), f'{len(labels)} == {len(boxes)}'
        
        boxes = boxes.clone()
        boxes[:, [0, 2]] /= self.spatial_size[0]
        boxes[:, [1, 3]] /= self.spatial_size[1]

        boxes = (boxes * (self.num_bins - 1))
        labels = (labels + self.num_bins)

        tokens = torch.cat([labels.unsqueeze(-1), boxes], dim=-1)
        if shuffle:
            _idx = torch.randperm(len(boxes))
            tokens = tokens[_idx]

        tokens = tokens.flatten().tolist()[:self.max_len]
        tokens.insert(0, self.BOS_TOKEN)
        tokens.append(self.EOS_TOKEN)

        return torch.tensor(tokens).to(torch.long)


    def decode(self, tokens: torch.Tensor):
        '''
        '''
        tokens = tokens[tokens != self.PAD_TOKEN]
        tokens = tokens[1: -1]
        assert len(tokens) % 5 == 0, ''

        tokens = tokens.reshape(-1, 5)

        labels = tokens[:, 0] - self.num_bins
        boxes = tokens[:, 1:].to(torch.float32) / (self.num_bins - 1)
        boxes[:, [0, 2]] *= self.spatial_size[0]
        boxes[:, [1, 3]] *= self.spatial_size[1]

        # return dict(labels=labels, boxes=boxes)
        return labels, boxes


def tokenizer_collate_fn(batch, tokenizer: Tokenizer):
    '''
    '''
    images = []
    tokens = []
    for i, (im, target) in enumerate(batch):
        images.append(im)
        tokens.append(tokenizer(target['labels'], target['boxes']))

    tokens = pad_sequence(tokens, batch_first=True, padding_value=tokenizer.PAD_TOKEN)
    images = torch.stack(images)

    return images, tokens





if __name__ == '__main__':
    pass




# Instance Segmenta
# COCO
# CityScapes
# YouTube-VIS

import torch.utils.data as data


class _Dataset(data.Dataset):
    def __init__(self, ):
        pass
    
    def format_output(self, ):
        raise NotImplementedError('')
    
    def evaluator(self, ):
        raise NotImplementedError('')


class _Detection(_Dataset):
    pass


class _InstanceSeg(_Dataset):
    pass


class _MonoDet3d(_Dataset):
    pass


class _MonoDepth(_Dataset):
    pass



class COCO(_Dataset):
    def __init__(self, ):
        pass
    
    
    def evaluator(self, ):
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        pass
    
    

class KITII():
    pass

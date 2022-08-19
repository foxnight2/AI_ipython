import torch
import torchvision.models as models


from torch.quantization import get_default_qconfig
from torch.quantization import quantize_fx



# PTQ
m = models.resnet18(weights=None)
m.eval()

qconfig = get_default_qconfig(backend='fbgemm')

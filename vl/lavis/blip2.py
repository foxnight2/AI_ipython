import torch
import torch.nn as nn 

from transformers import BertTokenizer


class Qformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()


class Blip2(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.tokenizer = None 
        self.vision_encoder = None 
        self.qformer = None 

        self.queries = nn.Parameter(torch.zeros(1, 32, 768)) 


    @classmethod
    def init_tokenizer(cls, truncation_side='right'):
        tokenier = BertTokenizer.from_pretrained('bert-base-uncased', truncation_side=truncation_side)
        tokenier.add_special_tokens('bos_token', '[DEC]')
        return tokenier
        
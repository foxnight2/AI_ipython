
import math

from pdll.autograd import Tensor

from ..functional import softmax, attention
from ..parameter import Parameter

from .linear import Linear
from .dropout import Dropout

from .module import Module


class MultiHeadAttention(Module):
    def __init__(self, embed_dim, num_heads, dropout=0, bias=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.use_bias = bias

        assert embed_dim % num_heads == 0, ''
        self.in_proj_weight = Parameter(3 * embed_dim, embed_dim)
        if self.use_bias:
            self.in_proj_bias = Parameter(3 * embed_dim)

        self.dropout = Dropout(p=dropout)

        self.out_proj = Linear(embed_dim, embed_dim)
    

    def _reset_parameters(self, ):
        pass


    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None):
        '''
        query: [l, n, e]
        key:   [s, n, e]
        value: [s, n, e]
        key_padding_mask: [n, s]
        attn_mask: [l, s]
        '''
        l = query.shape[0]
        n = query.shape[1]
        e = query.shape[-1]
        
        query = query @ self.in_proj_weight[:self.embed_dim]
        key = key @ self.in_proj_weight[self.embed_dim:-self.embed_dim]
        value = value @ self.in_proj_weight[-self.embed_dim:]

        if self.use_bias:
            query += self.in_proj_bias[:self.embed_dim]
            key += self.in_proj_bias[self.embed_dim:-self.embed_dim]
            value += self.in_proj_bias[-self.embed_dim:]
        
        query = query.reshape(-1, n * self.num_heads, e//self.num_heads).transpose(1, 0, 2)
        key = key.reshape(-1, n * self.num_heads, e//self.num_heads).transpose(1, 0, 2)
        value = value.reshape(-1, n * self.num_heads, e//self.num_heads).transpose(1, 0, 2)

        output, attn = attention(query, key, value, dropout=self.dropout)
        output = output.transpose(1, 0, 2).reshape(l, n, e)
        output = self.out_proj(output)

        return output, attn
import torch
import torch.nn as nn 
import timm
# from timm.models import trunc_normal_
import math


class Encoder(torch.nn.Module):
    def __init__(self, name='deit3_small_patch16_384_in21ft1k', pretrained=False, out_dim=256) -> None:
        super().__init__()
        self.model = timm.create_model(name, num_classes=0, global_pool='', pretrained=pretrained)
        self.bottleneck = nn.AdaptiveAvgPool1d(out_dim)

    def forward(self, x):
        features = self.model(x)
        return self.bottleneck(features[:, 1:])



class Decoder(torch.nn.Module):
    def __init__(self, vocab_size, max_len, hidden_dim, nhead, num_layers):
        super().__init__()

        self.padding_idx = vocab_size - 1

        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=self.padding_idx)
        self.dec_pos_embed = nn.Parameter(torch.randn(1, max_len - 1, hidden_dim) * 0.02)
        self.dec_pos_drop = nn.Dropout(p=0.05)

        dec_layer = nn.TransformerDecoderLayer(hidden_dim, nhead=nhead, batch_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers)

        self.enc_pos_embed = nn.Parameter(torch.randn(1, 576, hidden_dim) * 0.02)
        self.enc_pos_drop = nn.Dropout(p=0.05)

        self.out_proj = nn.Linear(hidden_dim, vocab_size)

        self.init_weights()

    def init_weights(self, ):
        for n, p in self.named_parameters():
            if 'embed' in n:
                continue
            if p.dim() > 1: 
                nn.init.xavier_uniform_(p)
        
        # trunc_normal_(self.dec_pos_embed, std=0.02)
        # trunc_normal_(self.enc_pos_embed, std=0.02)


    def positionalencoding1d(self, d_model, length, device='cpu'):
        '''
        '''
        if d_model % 2 != 0:
            raise ValueError('')
        pe = torch.zeros(length, d_model, )
        position = torch.arange(0, length).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                            -(math.log(10000.0) / d_model)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)

        return pe.to(device=device)


    def forward(self, memory, target):
        '''
        Args:
            memory, [n, l, d]
            target, [n, l]
        '''
        tgt_embedding = self.embedding(target)

        pe = self.positionalencoding1d(tgt_embedding.shape[-1], tgt_embedding.shape[1], device=target.device)
        tgt_embedding = self.dec_pos_drop(tgt_embedding + pe)

        tgt_key_padding_mask = (target == self.padding_idx)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(target.shape[-1], device=target.device)
        
        memory = self.enc_pos_drop(memory + self.enc_pos_embed)

        output = self.decoder(memory=memory, tgt=tgt_embedding, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)

        return self.out_proj(output)


class Pix2Seq(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, images, targets):
        mem = self.encoder(images)
        out = self.decoder(mem, targets)

        return out 


    def predict(self, images):
        pass






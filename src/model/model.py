import math, torch
import torch.nn as nn

"""
Tiny Transformer Model for Machine Translation
"""

class PositionalEncoding(nn.Module):
    """
    Positional Encoding Module

    input:
        x: (B, L, D)

    output:
        x: (B, L, D) with positional encodings added
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div) # even indices
        pe[:, 1::2] = torch.cos(pos * div) # odd indices
        self.register_buffer("pe", pe.unsqueeze(0))  # buffer because not a parameter to learn, unsqueeze for automatic adaptation dimension to batch size
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x) # addition because of the paper "attention is all you need" section 3.5

class TinyTransformerMT(nn.Module):
    def __init__(self, vocab, d_model=256, nhead=4, num_layers=4, dim_ff=1024, dropout=0.1, pad_id=0):
        super().__init__()
        self.pad_id = pad_id #id of padding token
        self.src_emb = nn.Embedding(vocab, d_model, padding_idx=pad_id) # Source embedding : scalar to vector
        self.tgt_emb = nn.Embedding(vocab, d_model, padding_idx=pad_id)
        self.pos = PositionalEncoding(d_model) # Positional encoding module
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead,
            num_encoder_layers=num_layers, num_decoder_layers=num_layers,
            dim_feedforward=dim_ff, dropout=dropout, batch_first=True)
        self.proj = nn.Linear(d_model, vocab)
        self.proj.weight = self.tgt_emb.weight


    def _generate_square_subsequent_mask(self, sz):
        return torch.triu(torch.ones(sz, sz, dtype=torch.bool), diagonal=1)

    def forward(self, src, tgt_in):
        """
        input:
            src: (B, Ls) source sequences of token ids
            tgt_in: (B, Lt) target input sequences (shifted right) of token ids
            exemple:
            src = tensor([[11, 22, 33, 2,  0], 
                          [44, 55, 2, 0,  0],
                          [66, 77, 88, 99, 2]])
            tgt_in = tensor([[1, 101, 102,   0],
                             [1, 103, 104, 105],
                             [1, 106,   0,   0]])
        output:
            out: (B, Lt, V) output logits for each token position
            
        """
        src_key_padding = (src == self.pad_id) # (B, Ls) with True for padding positions
        tgt_key_padding = (tgt_in == self.pad_id) # (B, Lt) with True for padding positions
        tgt_mask = self._generate_square_subsequent_mask(tgt_in.size(1)).to(src.device)
        
        src = self.pos(self.src_emb(src)) * math.sqrt(self.transformer.d_model)
        tgt = self.pos(self.tgt_emb(tgt_in)) * math.sqrt(self.transformer.d_model)
        out = self.transformer(src, tgt, tgt_mask=tgt_mask,
                               src_key_padding_mask=src_key_padding,
                               tgt_key_padding_mask=tgt_key_padding,
                               memory_key_padding_mask=src_key_padding)
        return self.proj(out)

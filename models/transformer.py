import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# flash attention
try:
    from flash_attn.flash_attention import FlashAttention
    FLASH_AVAIL = True
except:
    FLASH_AVAIL = False
    print("Flash attention not available. Reverting to Pytorch attention")

class SelfAttention(nn.Module):
    def __init__(self, H):
        super().__init__()
        assert H.model.n_emb % H.model.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(H.model.n_emb, H.model.n_emb)
        self.query = nn.Linear(H.model.n_emb, H.model.n_emb)
        self.value = nn.Linear(H.model.n_emb, H.model.n_emb)
        # regularization
        self.attn_drop = nn.Dropout(H.model.attn_pdrop)
        self.resid_drop = nn.Dropout(H.model.resid_pdrop)
        # output projection
        self.proj = nn.Linear(H.model.n_emb, H.model.n_emb)
        self.n_head = H.model.n_head
        self.causal = True if H.model.name == 'autoregressive' else False
        if self.causal:
            mask = torch.tril(torch.ones(H.model.block_size, H.model.block_size))
            self.register_buffer("mask", mask.view(1, 1, H.model.block_size, H.model.block_size))
        
        if FLASH_AVAIL:
            self.flash_attn = FlashAttention()

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.query(x), self.key(x), self.value(x)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b n h d', h=self.n_head), (q, k, v))

        present = torch.stack((k, v))
        if self.causal and layer_past is not None:
            past_key, past_value = layer_past
            k = torch.cat((past_key, k), dim=-2)
            v = torch.cat((past_value, v), dim=-2)

        if FLASH_AVAIL:
            qkv = torch.stack([q, k, v], dim=2)
            attn, _ = self.flash_attn(qkv, causal=self.causal)
            y = rearrange(attn, 'b n h d -> b n (h d)')
        else:
            q, k, v = map(lambda t: rearrange(t, 'b n h d -> b h n d', h=self.n_head), (q, k, v))
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            if self.causal and layer_past is None:
                att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))

            att = F.softmax(att, dim=-1)
            att = self.attn_drop(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
            # re-assemble all head outputs side by side
            y = y.transpose(1, 2).contiguous().view(B, T, C)

        # output projection
        y = self.resid_drop(self.proj(y))
        return y, present


class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, H):
        super().__init__()
        self.ln1 = nn.LayerNorm(H.model.n_emb)
        self.ln2 = nn.LayerNorm(H.model.n_emb)
        self.attn = SelfAttention(H)
        self.mlp = nn.Sequential(
            nn.Linear(H.model.n_emb, 4 * H.model.n_emb),
            nn.GELU(),  # nice
            nn.Linear(4 * H.model.n_emb, H.model.n_emb),
            nn.Dropout(H.model.resid_pdrop),
        )

    def forward(self, x, layer_past=None, return_present=False):

        attn, present = self.attn(self.ln1(x), layer_past)
        x = x + attn
        x = x + self.mlp(self.ln2(x))

        if layer_past is not None or return_present:
            return x, present
        return x


class Transformer(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, H):
        super().__init__()

        self.vocab_size = H.ct_config.model.codebook_size + 1
        self.n_embd = H.model.n_emb
        self.block_size = H.model.block_size
        self.n_layers = H.model.n_layers
        self.codebook_size = H.ct_config.model.codebook_size
        self.causal = H.model.name == 'autoregressive'
        if self.causal:
            self.vocab_size = H.ct_config.model.codebook_size

        self.tok_emb = nn.Embedding(self.vocab_size, self.n_embd)
        self.pos_emb = nn.Parameter(
            torch.zeros(1, self.block_size, self.n_embd))
        self.start_tok = nn.Parameter(torch.zeros(1, 1, self.n_embd))
        self.drop = nn.Dropout(H.model.embd_pdrop)
        self.context_linear = nn.Linear(H.ct_config.model.emb_dim, self.n_embd)

        # transformer
        self.blocks = nn.Sequential(*[Block(H) for _ in range(self.n_layers)])
        # decoder head
        self.ln_f = nn.LayerNorm(self.n_embd)
        self.head = nn.Linear(self.n_embd, self.codebook_size, bias=False)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, idx, t=None, context=None):
        # each index maps to a (learnable) vector
        x = self.tok_emb(idx)

        if self.causal:
            x = torch.cat(
                (self.start_tok.repeat(x.size(0), 1, 1), x),
                dim=1
            )
        
        if context is not None:
            context = self.context_linear(context)
            context_len = context.size(1)
            x = torch.cat((context, x), dim=1)

        t = x.size(1)
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        x = x + self.pos_emb[:, :t, :]
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)

        if context is not None:
            # remove context from start
            logits = logits[:,context_len:]

        return logits

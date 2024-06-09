import torch
import torch.nn as nn
import torch.nn.functional as F

class RotaryPositionalEmbeddings(nn.Module):
    def __init__(self, dim, base=10000):
        super(RotaryPositionalEmbeddings, self).__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.repeat_interleave(freqs, 2, -1)
            self.cos_cached = emb.cos().unsqueeze(0)
            self.sin_cached = emb.sin().unsqueeze(0)
        return self.cos_cached, self.sin_cached

def rotate_half(x):
    hdim = x.shape[-1]
    x1, x2 = x[..., : hdim // 2], x[..., hdim // 2 :]
    return torch.cat((-x2, x1), dim=-1) 


@torch.jit.script
def apply_rotary_pos_emb(q, k, cos, sin):
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class RoPEMultiHeadAttention(nn.Module):
    def __init__(self, seq_len, embed_dim, head_dim, num_q_heads, num_kv_heads, window_size, dropout):
        super(RoPEMultiHeadAttention, self).__init__()
        assert head_dim == embed_dim // num_q_heads
        causal_mask = torch.tril(torch.ones(seq_len, seq_len))
        self.register_buffer('causal_mask', causal_mask)
        
        self.query = nn.Linear(embed_dim , head_dim * num_q_heads)
        self.key = nn.Linear(embed_dim, head_dim * num_kv_heads)
        self.value = nn.Linear(embed_dim, head_dim * num_kv_heads)
        
        self.R = RotaryPositionalEmbeddings(embed_dim)
        self.drop = nn.Dropout(dropout)
        
        self.window_size = window_size
        self.head_dim = head_dim
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.num_q_per_kv = num_q_heads // num_kv_heads
    
    def forward(self, x):
        batch, seq_len, embed_dim = x.shape
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)


        if self.num_q_heads != self.num_kv_heads:
            k = torch.repeat_interleave(k, repeats=self.num_q_per_kv, dim =-1)
            v = torch.repeat_interleave(v, repeats=self.num_q_per_kv, dim =-1)
        
        cos, sin = self.R(x)
        qr, kr = apply_rotary_pos_emb(q, k, cos, sin)
        attn_scores = torch.zeros(batch, seq_len, seq_len)
        # for p in range(0, seq_len):
        #     attn_scores[:, p : p + self.window_size + 1, p : p + self.window_size + 1] = qr[:,p : p + self.window_size + 1,:] @ kr[:, p : p + self.window_size + 1,:].transpose(-2, -1) * torch.rsqrt(torch.tensor(self.head_dim))

        attn_scores = qr @ kr.transpose(1,2) * torch.rsqrt(torch.tensor(self.head_dim))
        masked_scores = torch.masked_fill(attn_scores, self.causal_mask[:seq_len,:seq_len] == 0, float('-inf'))
        attn_weights = F.softmax(masked_scores, dim=-1)
        attn_weights = self.drop(attn_weights)
        activation = attn_weights @ v
        return activation


class RoPEModelBlock(nn.Module):
    def __init__(self,  seq_len, embed_dim, head_dim, num_q_heads, num_kv_heads, window_size, dropout, proj_factor):
        super(RoPEModelBlock, self).__init__()
        self.pre_ln = nn.LayerNorm(embed_dim)
        self.rmha = RoPEMultiHeadAttention(seq_len, embed_dim, head_dim, num_q_heads, num_kv_heads, window_size, dropout)
        self.up_proj = nn.Linear(embed_dim, embed_dim * proj_factor)
        self.down_proj = nn.Linear(embed_dim * proj_factor, embed_dim)
        self.post_ln = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        skip = x
        x = self.pre_ln(x)
        x = skip + self.rmha(x)
        skip = x
        x = self.post_ln(x)
        x = self.down_proj(self.up_proj(x))
        x += skip
        return x

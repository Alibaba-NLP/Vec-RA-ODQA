from torch import nn, einsum
import torch
from einops import rearrange, repeat
from einops_exts import rearrange_many, repeat_many


def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False)
    )


class MaskedCrossAttention(nn.Module):
    """
    Input shape:
        x: (batch_size B, seq_len L, hidden_size H)
        media (perceived): (B, num_media M, num_latent_vector T, H)
    Output shape:
        (B, L, dim_head * num_heads)
    """

    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, media):
        b, t, m = media.shape[:3]
        h = self.heads

        x = self.norm(x)

        q = self.to_q(x)
        media = rearrange(media, 'b t n d -> b (t n) d')

        k, v = self.to_kv(media).chunk(2, dim=-1)
        q, k, v = rearrange_many((q, k, v), 'b n (h d) -> b h n d', h=h)

        q = q * self.scale

        sim = einsum('... i d, ... j d -> ... i j', q, k)

        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum('... i j, ... j d -> ... i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class GatedCrossAttentionBlock(nn.Module):
    """
    x, media -> [MaskedCrossAttention] -> [FFW] -> output

    Input shape:
        x: (batch_size B, seq_len L, hidden_size H)
        media (perceived): (B, num_media M, num_latent_vector T, H)
    Output shape:

    """

    def __init__(self, *, dim, dim_head=64, heads=8, ff_mult=4):
        super().__init__()
        self.attn = MaskedCrossAttention(dim=dim, dim_head=dim_head, heads=heads)
        self.attn_gate = nn.Parameter(torch.tensor([0.]))

        self.ff = FeedForward(dim, mult=ff_mult)
        self.ff_gate = nn.Parameter(torch.tensor([0.]))

    def forward(self, x, media, ):
        # input:
        x = self.attn(x, media) * self.attn_gate.tanh() + x
        x = self.ff(x) * self.ff_gate.tanh() + x
        return x

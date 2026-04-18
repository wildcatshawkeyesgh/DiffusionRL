import torch
import torch.nn as nn


__all__ = ["CrossAttentionBlock"]


class CrossAttentionBlock(nn.Module):
    """
    Pre-norm multi-head cross-attention + feed-forward, with residuals.

    q:  (B, Nq,  D)  queries     (e.g. a single z_t-derived token)
    kv: (B, Nkv, D)  context     (e.g. 64 board-cell tokens)
    returns: (B, Nq, D)
    """

    def __init__(self, dim, num_heads=4, ff_mult=4, dropout=0.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_ff = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_mult * dim),
            nn.GELU(),
            nn.Linear(ff_mult * dim, dim),
        )

    def forward(self, q, kv):
        q_n = self.norm_q(q)
        kv_n = self.norm_kv(kv)
        attn_out, _ = self.attn(q_n, kv_n, kv_n, need_weights=False)
        q = q + attn_out
        q = q + self.ff(self.norm_ff(q))
        return q

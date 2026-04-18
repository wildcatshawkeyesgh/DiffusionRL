import torch
import torch.nn as nn

from .attention import CrossAttentionBlock


__all__ = ["MLPDenoiser", "AttnDenoiser"]


class MLPDenoiser(nn.Module):
    """
    Predicts the noise added to a latent z_t given:
      - z_t:   (B, action_dim)   noisy latent
      - t:     (B,)              int timestep in [1, T]
      - state: (B, 3, 8, 8)     board from current player's POV

    Output: predicted noise (B, action_dim)
    """

    def __init__(self, config):
        super().__init__()
        H = config['hidden_dim']
        action_dim = config['action_dim']
        T = config['T']
        _, rows, cols = config['state_shape']

        # stride=2 on second conv shrinks 8x8 -> 4x4, cutting the
        # flatten dimension from 64*64=4096 down to 32*16=512
        self.state_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * (rows // 2) * (cols // 2), H),
            nn.ReLU(),
        )

        self.time_embed = nn.Embedding(T + 1, H)

        self.z_proj = nn.Sequential(
            nn.Linear(action_dim, H),
            nn.ReLU(),
        )

        self.trunk = nn.Sequential(
            nn.Linear(H * 3, H),
            nn.ReLU(),
            nn.Linear(H, H),
            nn.ReLU(),
            nn.Linear(H, action_dim),
        )

    def forward(self, z_t, t, state):
        s = self.state_encoder(state)       # (B, H)
        te = self.time_embed(t)             # (B, H)
        ze = self.z_proj(z_t)              # (B, H)
        h = torch.cat([s, te, ze], dim=-1) # (B, 3H)
        return self.trunk(h)               # (B, action_dim)


class AttnDenoiser(nn.Module):
    """
    Cross-attention denoiser. The board is encoded as a sequence of
    (rows * cols) tokens (one per cell). A single query token built
    from z_t and the time embedding attends to those board tokens.

    Inputs:
      z_t:   (B, action_dim)
      t:     (B,)
      state: (B, 3, rows, cols)

    Output: predicted noise (B, action_dim)
    """

    def __init__(self, config):
        super().__init__()
        H = config['hidden_dim']
        action_dim = config['action_dim']
        T = config['T']
        _, rows, cols = config['state_shape']
        num_heads = config.get('num_heads', 4)

        # Spatial encoder: keep (rows, cols) structure, project channels to H.
        self.state_encoder = nn.Sequential(
            nn.Conv2d(3, H // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(H // 2, H, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.num_tokens = rows * cols
        # Learned positional embedding per board cell.
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, H))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.time_embed = nn.Embedding(T + 1, H)
        self.z_proj = nn.Linear(action_dim, H)

        self.cross_attn = CrossAttentionBlock(dim=H, num_heads=num_heads)

        self.head = nn.Sequential(
            nn.LayerNorm(H),
            nn.Linear(H, H),
            nn.ReLU(),
            nn.Linear(H, action_dim),
        )

    def forward(self, z_t, t, state):
        B = z_t.shape[0]

        feat = self.state_encoder(state)                # (B, H, R, C)
        kv = feat.flatten(2).transpose(1, 2)            # (B, R*C, H)
        kv = kv + self.pos_embed                        # (B, R*C, H)

        q = self.z_proj(z_t) + self.time_embed(t)       # (B, H)
        q = q.unsqueeze(1)                              # (B, 1, H)

        out = self.cross_attn(q, kv)                    # (B, 1, H)
        return self.head(out.squeeze(1))                # (B, action_dim)

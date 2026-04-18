import torch
import torch.nn as nn


__all__ = ["DualSpaceVerifier"]


class DualSpaceVerifier(nn.Module):
    """
    Predicts game outcome from a partially-denoised action latent.

    Inputs:
        z_t:         (B, action_dim)  partially denoised latent
        board_state: (B, 3, 8, 8)    board from current player's POV
        t:           (B,)            diffusion timestep

    Output:
        value: (B,) in [-1, 1]
    """

    def __init__(self, config):
        super().__init__()
        H = config['hidden_dim']
        action_dim = config['action_dim']
        T = config['T']
        _, rows, cols = config['state_shape']

        self.state_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * (rows // 2) * (cols // 2), H),
            nn.ReLU(),
        )

        self.latent_encoder = nn.Sequential(
            nn.Linear(action_dim, H),
            nn.ReLU(),
            nn.Linear(H, H),
            nn.ReLU(),
        )

        self.time_embed = nn.Embedding(T + 1, H)

        self.fusion = nn.Sequential(
            nn.Linear(H * 3, H),
            nn.ReLU(),
            nn.Linear(H, H),
            nn.ReLU(),
        )

        self.value_head = nn.Linear(H, 1)

    def forward(self, z_t, board_state, t):
        s = self.state_encoder(board_state)   # (B, H)
        z = self.latent_encoder(z_t)          # (B, H)
        te = self.time_embed(t)               # (B, H)

        fused = torch.cat([s, z, te], dim=-1) # (B, 3H)
        fused = self.fusion(fused)            # (B, H)
        value = torch.tanh(self.value_head(fused))  # (B, 1)
        return value.squeeze(-1)              # (B,)

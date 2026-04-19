import torch
import torch.nn as nn

from .denoiser import MLPDenoiser, AttnDenoiser


_DENOISERS = {
    'mlp': MLPDenoiser,
    'attn': AttnDenoiser,
}


__all__ = ["DiffusionPolicy"]


class DiffusionPolicy(nn.Module):
    """
    Owns the denoiser and noise schedule.
    Knows nothing about rewards, outcomes, or training.

    Core methods:
      forward_diffuse(z_0, t)          -> (z_t, noise)   used during training
      denoise_step(z_t, t, state)      -> z_{t-1}        single DDPM reverse step
      select_action(state, valid_mask) -> int             full inference pass
    """

    def __init__(self, config):
        super().__init__()
        self.T = config['T']
        self.action_dim = config['action_dim']
        kind = config.get('denoiser_type', 'attn')
        self.denoiser = _DENOISERS[kind](config)

        # Linear beta schedule; index 0 unused, indices 1..T used
        betas = torch.zeros(self.T + 1)
        betas[1:] = torch.linspace(config['beta_start'], config['beta_end'], self.T)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bars', alpha_bars)

    # ------------------------------------------------------------------ #
    # training-time: add noise                                            #
    # ------------------------------------------------------------------ #

    def forward_diffuse(self, z_0, t):
        """
        z_0: (B, action_dim)  clean latent (one-hot of action taken)
        t:   (B,)             int timestep in [1, T]
        returns z_t, noise — both (B, action_dim)
        """
        noise = torch.randn_like(z_0)
        ab = self.alpha_bars[t].view(-1, 1)
        z_t = torch.sqrt(ab) * z_0 + torch.sqrt(1.0 - ab) * noise
        return z_t, noise

    # ------------------------------------------------------------------ #
    # inference-time: remove noise                                        #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def denoise_step(self, z_t, t, state):
        """
        Single DDPM reverse step.
        z_t:   (B, action_dim)
        t:     (B,)  int, same value for all elements
        state: (B, 3, 8, 8)
        returns z_{t-1}  (B, action_dim)
        """
        eps_pred = self.denoiser(z_t, t, state)

        beta_t    = self.betas[t].view(-1, 1)
        alpha_t   = self.alphas[t].view(-1, 1)
        alpha_bar = self.alpha_bars[t].view(-1, 1)

        mean = (1.0 / torch.sqrt(alpha_t)) * (
            z_t - (beta_t / torch.sqrt(1.0 - alpha_bar)) * eps_pred
        )

        # Add noise only when t > 1
        noise = torch.randn_like(z_t)
        sigma = torch.sqrt(beta_t)
        mask = (t > 1).float().view(-1, 1)
        return mean + mask * sigma * noise

    @torch.no_grad()
    def select_action(self, state_np, valid_mask_np, temperature=0.0, return_logits=False):
        """
        Full denoising pass to pick an action.
        state_np:      (3, 8, 8) numpy array or torch tensor
        valid_mask_np: (64,)     bool numpy array or torch tensor
        temperature:   0.0 → argmax (eval); >0 → sample from softmax(logits / T)
        return_logits: if True, also return the masked logits (for entropy/heatmap)
        """
        self.eval()
        device = next(self.parameters()).device

        state = torch.as_tensor(state_np, dtype=torch.float32).unsqueeze(0).to(device)
        z = torch.randn(1, self.action_dim, device=device)

        for step in range(self.T, 0, -1):
            t = torch.full((1,), step, dtype=torch.long, device=device)
            z = self.denoise_step(z, t, state)

        logits = z.squeeze(0).cpu()
        valid = torch.as_tensor(valid_mask_np, dtype=torch.bool)
        logits[~valid] = float('-inf')

        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            action = int(torch.multinomial(probs, 1).item())
        else:
            action = int(torch.argmax(logits).item())

        if return_logits:
            return action, logits
        return action

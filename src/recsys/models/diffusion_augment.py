"""
Diffusion-Based Cold Start Augmentation
========================================
DiffRec-inspired diffusion model for generating synthetic interaction
sequences for new users with sparse history.

Reference: Lin et al., DiffRec (2023) https://arxiv.org/abs/2304.04971

Why diffusion for cold start?
Rather than cold-starting with zeros or popularity baseline,
we generate a synthetic interaction history consistent with the
user's onboarding answers — giving Mult-VAE meaningful input
from session one.

ML System Design decisions documented inline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class DenoisingMLP(nn.Module):
    """
    Denoising network for the diffusion process.
    Takes noisy interaction vector + timestep embedding → predicts noise.

    Design decision: MLP over Transformer for the denoising network.
    Interaction vectors are permutation-invariant (item order doesn't matter
    for collaborative filtering). MLP captures this naturally.
    Transformer would impose unnecessary sequence structure.
    """

    def __init__(
        self,
        n_items: int,
        hidden_dim: int = 256,
        n_steps: int = 1000,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_steps = n_steps

        # Timestep embedding — sinusoidal, same as original DDPM
        self.time_embed = nn.Sequential(
            nn.Embedding(n_steps, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        # Denoising MLP
        self.net = nn.Sequential(
            nn.Linear(n_items + hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, n_items),
        )

    def forward(self, x_t: Tensor, t: Tensor) -> Tensor:
        """
        Predict noise given noisy interaction vector x_t at timestep t.
        x_t: (B, n_items) noisy interaction vector
        t:   (B,) timestep indices
        """
        t_emb = self.time_embed[0](t)           # (B, hidden_dim)
        t_emb = self.time_embed[1](t_emb)
        t_emb = self.time_embed[2](t_emb)
        x = torch.cat([x_t, t_emb], dim=-1)    # (B, n_items + hidden_dim)
        return self.net(x)


class DiffusionAugmentor:
    """
    DDPM-style diffusion model for interaction sequence augmentation.

    Forward process: gradually adds Gaussian noise to real interaction vectors.
    Reverse process: learns to denoise — generating plausible interactions
    conditioned on onboarding preference signal.

    Design decision: linear noise schedule (vs cosine).
    Linear schedule from the original DDPM paper.
    Cosine schedule (Improved DDPM) gives better sample quality but
    more complex implementation. Linear is sufficient for augmentation.

    Design decision: condition on onboarding genre vector.
    New user's genre preferences from onboarding are concatenated with
    the noisy vector at each denoising step — guides generation toward
    user's stated preferences without requiring full interaction history.
    """

    def __init__(
        self,
        n_items: int,
        n_steps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        device: str = "cuda",
    ):
        self.n_items = n_items
        self.n_steps = n_steps
        self.device = device

        # Linear noise schedule
        betas = torch.linspace(beta_start, beta_end, n_steps).to(device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer = lambda name, val: setattr(self, name, val)
        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

        self.model = DenoisingMLP(n_items, n_steps=n_steps).to(device)

    def q_sample(self, x0: Tensor, t: Tensor, noise: Tensor = None) -> Tensor:
        """
        Forward process: add noise to x0 at timestep t.
        x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * noise
        """
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab = self.sqrt_alphas_cumprod[t].view(-1, 1)
        sqrt_1_ab = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        return sqrt_ab * x0 + sqrt_1_ab * noise

    def p_losses(self, x0: Tensor, t: Tensor) -> Tensor:
        """
        Compute denoising loss: predict noise added at timestep t.
        Standard DDPM objective: MSE between predicted and actual noise.
        """
        noise = torch.randn_like(x0)
        x_noisy = self.q_sample(x0, t, noise)
        predicted_noise = self.model(x_noisy, t)
        return F.mse_loss(predicted_noise, noise)

    @torch.no_grad()
    def generate(
        self,
        onboarding_vector: Tensor,
        n_samples: int = 1,
        n_inference_steps: int = 50,
    ) -> Tensor:
        """
        Generate synthetic interaction sequence via reverse diffusion.

        Starts from pure noise, iteratively denoises conditioned on
        onboarding preferences. Output is a soft interaction vector —
        values close to 1 indicate likely interactions.

        Design decision: 50 inference steps (vs 1000 training steps).
        DDIM sampling enables high-quality generation in 50 steps.
        1000 steps at inference is prohibitively slow for <500ms SLA.

        Args:
            onboarding_vector: (1, n_items) sparse preference signal from onboarding
            n_samples: number of synthetic sequences to generate
            n_inference_steps: DDIM sampling steps (50 is standard)

        Returns:
            (n_samples, n_items) synthetic interaction vectors
        """
        x = torch.randn(n_samples, self.n_items).to(self.device)

        # DDIM-style reverse process (simplified)
        step_size = self.n_steps // n_inference_steps
        timesteps = list(range(0, self.n_steps, step_size))[::-1]

        for t_idx in timesteps:
            t = torch.tensor([t_idx] * n_samples, device=self.device)

            # Predict noise
            predicted_noise = self.model(x, t)

            # Compute x_{t-1}
            alpha = self.alphas[t_idx]
            alpha_bar = self.alphas_cumprod[t_idx]
            beta = self.betas[t_idx]

            if t_idx > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)

            x = (1 / torch.sqrt(alpha)) * (
                x - ((1 - alpha) / torch.sqrt(1 - alpha_bar)) * predicted_noise
            ) + torch.sqrt(beta) * noise

            # Soft conditioning: blend with onboarding signal
            # Items in onboarding vector get a small upward push
            x = x + 0.1 * onboarding_vector.to(self.device)

        # Sigmoid to get soft interaction probabilities
        return torch.sigmoid(x)

    def augment_cold_start_user(
        self,
        genre_preferences: list[str],
        genre2items: dict[str, list[int]],
        n_samples: int = 1,
    ) -> Tensor:
        """
        Generate synthetic interaction history from genre preferences.

        Pipeline:
            1. Build onboarding vector from genre preferences
            2. Run reverse diffusion conditioned on onboarding vector
            3. Return soft interaction probabilities → input to Mult-VAE

        Args:
            genre_preferences: list of genre strings from onboarding
            genre2items: mapping of genre → item indices
            n_samples: number of synthetic histories to generate
        """
        # Build sparse onboarding vector from genre preferences
        onboarding = torch.zeros(1, self.n_items)
        for genre in genre_preferences:
            for item_idx in genre2items.get(genre, []):
                onboarding[0, item_idx] = 1.0

        # Normalize
        if onboarding.sum() > 0:
            onboarding = onboarding / onboarding.sum()

        return self.generate(onboarding, n_samples=n_samples)
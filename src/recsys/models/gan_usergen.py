"""
GAN for Synthetic User Profile Generation
==========================================
Generates diverse synthetic user profiles for adversarial evaluation
of the recommendation system.

Design decision: GAN for eval, not for production recommendations.
GAN-generated users stress-test the recommender with edge cases:
- Power users (1000+ interactions)
- Extremely sparse users (5 interactions)
- Niche taste users (single genre only)
- Contradictory preference users

These profiles reveal failure modes invisible in real user evaluation.

ML System Design decisions documented inline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from dataclasses import dataclass


@dataclass
class SyntheticUser:
    user_vector: Tensor        # (n_items,) interaction vector
    profile_type: str          # power/sparse/niche/contradictory
    n_interactions: int
    dominant_genre: str | None


class Generator(nn.Module):
    """
    Maps noise vector → synthetic user interaction profile.

    Design decision: output soft probabilities, not binary interactions.
    Binary sampling loses gradient signal. Soft outputs can be thresholded
    at inference for discrete interaction vectors, or used directly as
    Mult-VAE input (which handles continuous values).
    """

    def __init__(self, noise_dim: int = 128, n_items: int = 62000):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, n_items),
            nn.Sigmoid(),   # output ∈ [0, 1] — soft interaction probabilities
        )

    def forward(self, z: Tensor) -> Tensor:
        return self.net(z)


class Discriminator(nn.Module):
    """
    Distinguishes real user interaction vectors from generated ones.

    Design decision: WGAN-GP (Wasserstein GAN with gradient penalty).
    Standard GAN training is unstable — mode collapse is common.
    WGAN-GP provides more stable training via Wasserstein distance
    and gradient penalty instead of binary cross-entropy loss.
    No sigmoid on discriminator output (critic, not classifier).
    """

    def __init__(self, n_items: int = 62000):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_items, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 1),
            # No sigmoid — WGAN-GP outputs unbounded critic score
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def gradient_penalty(
    discriminator: Discriminator,
    real: Tensor,
    fake: Tensor,
    device: str,
) -> Tensor:
    """
    WGAN-GP gradient penalty.
    Enforces 1-Lipschitz constraint on discriminator via gradient norm.
    lambda=10 as per original WGAN-GP paper.
    """
    batch_size = real.size(0)
    alpha = torch.rand(batch_size, 1, device=device)
    interpolated = alpha * real + (1 - alpha) * fake
    interpolated.requires_grad_(True)

    d_interpolated = discriminator(interpolated)
    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated),
        create_graph=True,
        retain_graph=True,
    )[0]

    gradients = gradients.view(batch_size, -1)
    penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return 10.0 * penalty  # lambda = 10


class UserProfileGAN:
    """
    WGAN-GP for synthetic user profile generation.

    Training: discriminator distinguishes real vs generated user vectors.
    Inference: sample noise → generator → synthetic user profile.

    Design decision: train on filtered interaction matrix (post k-core).
    We want generated users to resemble plausible real users, not
    the full raw distribution which includes noise and bots.
    """

    def __init__(
        self,
        n_items: int,
        noise_dim: int = 128,
        device: str = "cuda",
    ):
        self.n_items = n_items
        self.noise_dim = noise_dim
        self.device = device

        self.generator = Generator(noise_dim, n_items).to(device)
        self.discriminator = Discriminator(n_items).to(device)

        self.g_optimizer = torch.optim.Adam(
            self.generator.parameters(), lr=1e-4, betas=(0.0, 0.9)
        )
        self.d_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=1e-4, betas=(0.0, 0.9)
        )

    def train_step(
        self,
        real_batch: Tensor,
        n_critic_steps: int = 5,
    ) -> dict[str, float]:
        """
        One training step: update discriminator n_critic times, generator once.

        Design decision: n_critic=5 (discriminator updates per generator update).
        Standard for WGAN-GP — keeps discriminator near optimum before
        generator update, stabilizes training.
        """
        batch_size = real_batch.size(0)
        real = real_batch.to(self.device)
        losses = {}

        # ── Train discriminator (critic) ──────────────────────────────────
        for _ in range(n_critic_steps):
            z = torch.randn(batch_size, self.noise_dim, device=self.device)
            fake = self.generator(z).detach()

            d_real = self.discriminator(real).mean()
            d_fake = self.discriminator(fake).mean()
            gp = gradient_penalty(self.discriminator, real, fake, self.device)

            # Wasserstein loss: maximize d_real - d_fake
            d_loss = d_fake - d_real + gp

            self.d_optimizer.zero_grad()
            d_loss.backward()
            self.d_optimizer.step()

        losses["d_loss"] = d_loss.item()

        # ── Train generator ───────────────────────────────────────────────
        z = torch.randn(batch_size, self.noise_dim, device=self.device)
        fake = self.generator(z)
        g_loss = -self.discriminator(fake).mean()

        self.g_optimizer.zero_grad()
        g_loss.backward()
        self.g_optimizer.step()

        losses["g_loss"] = g_loss.item()
        return losses

    @torch.no_grad()
    def generate_users(
        self,
        n_users: int = 100,
        profile_type: str = "random",
        threshold: float = 0.3,
    ) -> list[SyntheticUser]:
        """
        Generate synthetic user profiles for adversarial evaluation.

        profile_type controls noise sampling strategy:
            random       — standard normal noise → diverse profiles
            power_user   — high-magnitude noise → dense interaction vectors
            sparse_user  — low-magnitude noise → sparse interaction vectors
            niche        — concentrated noise → single-genre profiles

        threshold: soft probability cutoff for binarizing interactions.
        """
        self.generator.eval()

        if profile_type == "power_user":
            z = torch.randn(n_users, self.noise_dim, device=self.device) * 2.0
        elif profile_type == "sparse_user":
            z = torch.randn(n_users, self.noise_dim, device=self.device) * 0.3
        elif profile_type == "niche":
            # Concentrate noise in first few dimensions
            z = torch.zeros(n_users, self.noise_dim, device=self.device)
            z[:, :8] = torch.randn(n_users, 8, device=self.device) * 3.0
        else:
            z = torch.randn(n_users, self.noise_dim, device=self.device)

        soft_vectors = self.generator(z).cpu()

        users = []
        for i in range(n_users):
            vec = soft_vectors[i]
            binary = (vec > threshold).float()
            n_interactions = int(binary.sum().item())

            users.append(SyntheticUser(
                user_vector=binary,
                profile_type=profile_type,
                n_interactions=n_interactions,
                dominant_genre=None,  # filled by eval harness with item metadata
            ))

        return users

    def generate_adversarial_eval_set(self, n_per_type: int = 25) -> list[SyntheticUser]:
        """
        Generate balanced adversarial eval set across all profile types.
        Total: 4 * n_per_type users.
        """
        users = []
        for ptype in ["random", "power_user", "sparse_user", "niche"]:
            users.extend(self.generate_users(n_per_type, profile_type=ptype))
        return users
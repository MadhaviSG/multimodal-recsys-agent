"""
Mult-VAE for Collaborative Filtering
Liang et al., 2018 (https://arxiv.org/abs/1802.05814)

Learns a distribution over user preferences in latent space rather than
a point estimate — enabling uncertainty-aware recommendations and better
generalization with sparse interaction data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple


class MultVAE(nn.Module):
    def __init__(
        self,
        num_items: int,
        hidden_dims: list[int] = [600, 200],
        latent_dim: int = 64,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder: item space → latent distribution
        encoder_layers = []
        in_dim = num_items
        for h_dim in hidden_dims:
            encoder_layers += [nn.Linear(in_dim, h_dim), nn.Tanh()]
            in_dim = h_dim
        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(in_dim, latent_dim)
        self.fc_logvar = nn.Linear(in_dim, latent_dim)

        # Decoder: latent → item space
        decoder_layers = []
        in_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers += [nn.Linear(in_dim, h_dim), nn.Tanh()]
            in_dim = h_dim
        decoder_layers.append(nn.Linear(in_dim, num_items))
        self.decoder = nn.Sequential(*decoder_layers)

        self.dropout = nn.Dropout(dropout)

    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Returns (mu, logvar) of the approximate posterior."""
        x = F.normalize(x, dim=-1)   # normalize input ratings
        x = self.dropout(x)
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """Reparameterization trick: z = mu + eps * std."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu  # deterministic at inference

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(z)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    def get_user_embedding(self, x: Tensor) -> Tensor:
        """Returns deterministic user embedding (mu) for retrieval."""
        mu, _ = self.encode(x)
        return mu


def loss_function(
    recon: Tensor,
    x: Tensor,
    mu: Tensor,
    logvar: Tensor,
    anneal_beta: float = 1.0,
) -> Tensor:
    """
    ELBO loss with KL annealing.
    
    Reconstruction: multinomial log-likelihood (softmax over items)
    KL: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    
    anneal_beta: KL weight, annealed from 0 → 1 during training.
    Prevents posterior collapse in early training.
    """
    recon_loss = -torch.mean(
        torch.sum(F.log_softmax(recon, dim=-1) * x, dim=-1)
    )
    kl_loss = -0.5 * torch.mean(
        torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
    )
    return recon_loss + anneal_beta * kl_loss

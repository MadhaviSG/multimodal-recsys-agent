"""
Two-Tower Retrieval Model
User tower + Item tower trained with in-batch negatives.
Serves candidates via ANN (Qdrant HNSW).

Fix history:
- Removed L2 normalization from towers — caused vanishing gradients
  when combined with small init + temperature scaling
- Fixed temperature at 0.07 (standard for InfoNCE)
- Standard embedding init (default N(0,1) / sqrt(dim))
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class UserTower(nn.Module):
    def __init__(self, num_users: int, embed_dim: int = 128, output_dim: int = 64):
        super().__init__()
        self.user_embed = nn.Embedding(num_users, embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, output_dim),
        )

    def forward(self, user_ids: Tensor) -> Tensor:
        x = self.user_embed(user_ids)
        return self.mlp(x)  # no L2 norm — let model learn scale


class ItemTower(nn.Module):
    def __init__(
        self,
        num_items: int,
        item_feature_dim: int,
        embed_dim: int = 64,
        output_dim: int = 64,
    ):
        super().__init__()
        self.item_embed = nn.Embedding(num_items, embed_dim)
        self.feature_mlp = nn.Sequential(
            nn.Linear(item_feature_dim, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
        )
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim + 64, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, output_dim),
        )

    def forward(self, item_ids: Tensor, item_features: Tensor) -> Tensor:
        id_emb = self.item_embed(item_ids)
        feat_emb = self.feature_mlp(item_features)
        combined = torch.cat([id_emb, feat_emb], dim=-1)
        return self.fusion(combined)  # no L2 norm


class TwoTowerModel(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        item_feature_dim: int,
        embed_dim: int = 64,
    ):
        super().__init__()
        self.user_tower = UserTower(num_users, output_dim=embed_dim)
        self.item_tower = ItemTower(num_items, item_feature_dim, output_dim=embed_dim)
        # Learnable temperature with good initialization
        # log(1/0.07) ≈ 2.66 — standard CLIP/InfoNCE starting point
        self.log_temperature = nn.Parameter(torch.tensor(2.66))

    @property
    def temperature(self):
        # Clamp to prevent temperature collapsing to 0
        return torch.clamp(self.log_temperature.exp(), min=0.01, max=100.0)

    def forward(self, user_ids: Tensor, item_ids: Tensor, item_features: Tensor) -> Tensor:
        user_emb = self.user_tower(user_ids)
        item_emb = self.item_tower(item_ids, item_features)
        # L2 normalize only for dot product similarity — not inside tower
        user_emb = F.normalize(user_emb, dim=-1)
        item_emb = F.normalize(item_emb, dim=-1)
        logits = torch.matmul(user_emb, item_emb.T) * self.temperature
        return logits

    def in_batch_loss(
        self, user_ids: Tensor, item_ids: Tensor, item_features: Tensor
    ) -> Tensor:
        logits = self.forward(user_ids, item_ids, item_features)
        labels = torch.arange(logits.size(0), device=logits.device)
        loss = F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)
        return loss / 2

    def get_user_embedding(self, user_ids: Tensor) -> Tensor:
        emb = self.user_tower(user_ids)
        return F.normalize(emb, dim=-1)

    def get_item_embedding(self, item_ids: Tensor, item_features: Tensor) -> Tensor:
        emb = self.item_tower(item_ids, item_features)
        return F.normalize(emb, dim=-1)

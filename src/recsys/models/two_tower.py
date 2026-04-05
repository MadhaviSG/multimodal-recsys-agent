"""
Two-Tower Retrieval Model
User tower + Item tower trained with in-batch negatives + hard negatives.
Serves candidates via ANN (Qdrant HNSW).
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
        return F.normalize(self.mlp(x), dim=-1)


class ItemTower(nn.Module):
    def __init__(self, item_feature_dim: int, output_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(item_feature_dim, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, output_dim),
        )

    def forward(self, item_features: Tensor) -> Tensor:
        return F.normalize(self.mlp(item_features), dim=-1)


class TwoTowerModel(nn.Module):
    def __init__(self, num_users: int, item_feature_dim: int, embed_dim: int = 64):
        super().__init__()
        self.user_tower = UserTower(num_users, output_dim=embed_dim)
        self.item_tower = ItemTower(item_feature_dim, output_dim=embed_dim)
        self.temperature = nn.Parameter(torch.tensor(0.07))

    def forward(
        self,
        user_ids: Tensor,
        item_features: Tensor,
    ) -> Tensor:
        user_emb = self.user_tower(user_ids)       # (B, D)
        item_emb = self.item_tower(item_features)  # (B, D)
        logits = torch.matmul(user_emb, item_emb.T) / self.temperature
        return logits

    def in_batch_loss(self, user_ids: Tensor, item_features: Tensor) -> Tensor:
        """
        In-batch negatives: all other items in the batch are treated as negatives.
        Labels are the diagonal (each user matches its own item).
        """
        logits = self.forward(user_ids, item_features)
        labels = torch.arange(logits.size(0), device=logits.device)
        loss = F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)
        return loss / 2

"""
Two-Tower Retrieval Model
User tower + Item tower trained with in-batch negatives.
Serves candidates via ANN (Qdrant HNSW).

Key fix: added learned item embedding alongside content features.
Genre + year alone (21-dim) is insufficient for contrastive learning —
items in the same genre look identical to the tower.
Learned item embedding lets the model capture item-specific signal
beyond genre/year features.
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
    def __init__(
        self,
        num_items: int,
        item_feature_dim: int,
        embed_dim: int = 64,
        output_dim: int = 64,
    ):
        super().__init__()

        # Learned item embedding — captures item-specific signal
        # beyond genre/year content features
        self.item_embed = nn.Embedding(num_items, embed_dim)

        # Content feature MLP
        self.feature_mlp = nn.Sequential(
            nn.Linear(item_feature_dim, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
        )

        # Fusion: concat learned embedding + content features → output
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim + 64, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, output_dim),
        )

    def forward(self, item_ids: Tensor, item_features: Tensor) -> Tensor:
        id_emb = self.item_embed(item_ids)              # (B, embed_dim)
        feat_emb = self.feature_mlp(item_features)      # (B, 64)
        combined = torch.cat([id_emb, feat_emb], dim=-1)
        return F.normalize(self.fusion(combined), dim=-1)


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

        # Fixed temperature — learnable temperature can collapse early
        # 0.1 is a stable starting point for in-batch negative InfoNCE
        self.temperature = 0.1

    def forward(self, user_ids: Tensor, item_ids: Tensor, item_features: Tensor) -> Tensor:
        user_emb = self.user_tower(user_ids)
        item_emb = self.item_tower(item_ids, item_features)
        logits = torch.matmul(user_emb, item_emb.T) / self.temperature
        return logits

    def in_batch_loss(
        self, user_ids: Tensor, item_ids: Tensor, item_features: Tensor
    ) -> Tensor:
        logits = self.forward(user_ids, item_ids, item_features)
        labels = torch.arange(logits.size(0), device=logits.device)
        loss = F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)
        return loss / 2

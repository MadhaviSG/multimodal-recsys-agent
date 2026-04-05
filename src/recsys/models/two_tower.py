"""
Two-Tower Retrieval Model — Minimal Working Version

Stripped to absolute minimum to confirm training loop works.
Complexity (MLP, content features, learnable temperature) added back
incrementally once base training is confirmed.

Architecture:
    User: embedding lookup → L2 normalize
    Item: embedding lookup → L2 normalize
    Loss: InfoNCE with fixed temperature=0.07
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class TwoTowerModel(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        item_feature_dim: int = 21,  # kept for API compatibility
        embed_dim: int = 64,
    ):
        super().__init__()
        self.user_embed = nn.Embedding(num_users, embed_dim)
        self.item_embed = nn.Embedding(num_items, embed_dim)
        self.temperature = 0.2  # fixed — proven stable for InfoNCE

    def get_user_embedding(self, user_ids: Tensor) -> Tensor:
        return F.normalize(self.user_embed(user_ids), dim=-1)

    def get_item_embedding(self, item_ids: Tensor, item_features: Tensor = None) -> Tensor:
        return F.normalize(self.item_embed(item_ids), dim=-1)

    def forward(
        self, user_ids: Tensor, item_ids: Tensor, item_features: Tensor = None
    ) -> Tensor:
        user_emb = self.get_user_embedding(user_ids)   # (B, D)
        item_emb = self.get_item_embedding(item_ids)   # (B, D)
        return (user_emb @ item_emb.T) / self.temperature

    def in_batch_loss(
        self, user_ids: Tensor, item_ids: Tensor, item_features: Tensor = None
    ) -> Tensor:
        logits = self.forward(user_ids, item_ids, item_features)
        labels = torch.arange(logits.size(0), device=logits.device)
        loss = F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)
        return loss / 2

    def bpr_loss(
        self, user_ids: Tensor, pos_ids: Tensor, neg_ids: Tensor
    ) -> Tensor:
        """
        BPR loss: maximize score(user, pos) - score(user, neg).
        Simpler than InfoNCE — one explicit negative per positive.
        Proven to converge for collaborative filtering.
        Reference: Rendle et al., BPR (2009)
        """
        user_emb = self.get_user_embedding(user_ids)   # (B, D)
        pos_emb  = self.get_item_embedding(pos_ids)    # (B, D)
        neg_emb  = self.get_item_embedding(neg_ids)    # (B, D)
        pos_score = (user_emb * pos_emb).sum(dim=-1)   # (B,)
        neg_score = (user_emb * neg_emb).sum(dim=-1)   # (B,)
        loss = -F.logsigmoid(pos_score - neg_score).mean()
        return loss

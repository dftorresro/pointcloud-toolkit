"""
Modular DGCNN part-segmentation network.

Refactored from the monolithic prototype:
  • no global `args`
  • all hyper-params passed via ctor (k, emb_dims, dropout, num_part_classes)
  • ready for import:  from pcseg.models.dgcnn import DGCNNPartSeg
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DGCNNPartSeg(nn.Module):
    def __init__(
        self,
        k: int = 20,
        emb_dims: int = 1024,
        dropout: float = 0.5,
        num_part_classes: int = 50,
    ):
        super().__init__()
        self.k = k
        self.num_part_classes = num_part_classes

        # ── convolutional backbone ───────────────────────────────────────────────
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(emb_dims)

        self.conv1 = nn.Sequential(
            nn.Conv2d(6, 64, 1, bias=False), self.bn1, nn.LeakyReLU(0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 64, 1, bias=False), self.bn2, nn.LeakyReLU(0.2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 128, 1, bias=False), self.bn3, nn.LeakyReLU(0.2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 256, 1, bias=False), self.bn4, nn.LeakyReLU(0.2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(512, emb_dims, 1, bias=False), self.bn5, nn.LeakyReLU(0.2)
        )

        # ── segmentation head ───────────────────────────────────────────────────
        self.linear1 = nn.Linear(emb_dims * 3, 256, bias=False)
        self.bn6 = nn.BatchNorm1d(256)
        self.dp1 = nn.Dropout(dropout)

        self.linear2 = nn.Linear(256, 256, bias=False)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(dropout)

        self.linear3 = nn.Linear(256, num_part_classes)

    # ────────────────────────── helper blocks ───────────────────────────────────
    @staticmethod
    def _knn(x: torch.Tensor, k: int) -> torch.Tensor:
        """Return k-NN indices for each point (B, N)."""
        # pairwise distance matrix
        inner = -2.0 * torch.matmul(x.transpose(2, 1), x)
        xx = torch.sum(x**2, dim=1, keepdim=True)
        dist = -xx - inner - xx.transpose(2, 1)
        return dist.topk(k=k, dim=-1)[1]  # (B, N, k)

    def _get_graph_feature(self, x: torch.Tensor) -> torch.Tensor:
        """Construct edge features (B, 2*C, N, k)."""
        B, C, N = x.size()
        idx = self._knn(x, self.k)                     # (B, N, k)
        device = x.device
        idx_base = torch.arange(0, B, device=device).view(-1, 1, 1) * N
        idx = (idx + idx_base).view(-1)

        x = x.transpose(2, 1).contiguous()             # (B, N, C)
        feature = x.view(B * N, C)[idx, :].view(B, N, self.k, C)
        x = x.view(B, N, 1, C).repeat(1, 1, self.k, 1)
        feature = torch.cat((feature - x, x), dim=3)   # (B, N, k, 2C)
        return feature.permute(0, 3, 1, 2).contiguous()

    # ───────────────────────────── forward ──────────────────────────────────────
    def forward(self, pts: torch.Tensor) -> torch.Tensor:
        """
        pts: (B, N, 3)  →  logits: (B, N, num_part_classes)
        """
        B, N, _ = pts.size()
        x = pts.transpose(2, 1)                        # (B, 3, N)

        # Edge-conv blocks
        x1 = self.conv1(self._get_graph_feature(x))    # (B, 64, N)
        x1 = x1.max(dim=-1)[0]

        x2 = self.conv2(self._get_graph_feature(x1))
        x2 = x2.max(dim=-1)[0]

        x3 = self.conv3(self._get_graph_feature(x2))
        x3 = x3.max(dim=-1)[0]

        x4 = self.conv4(self._get_graph_feature(x3))
        x4 = x4.max(dim=-1)[0]

        # Global features
        x_cat = torch.cat((x1, x2, x3, x4), dim=1)     # (B, 512, N)
        x_emb = self.conv5(x_cat)                      # (B, emb_dims, N)
        x_max = x_emb.max(2)[0]                        # (B, emb_dims)
        x_mean = x_emb.mean(2)                         # (B, emb_dims)
        x_global = torch.cat((x_max, x_mean), 1)       # (B, 2*emb_dims)
        x_global = x_global.unsqueeze(2).repeat(1, 1, N)

        # Point-wise segmentation head
        x_seg = torch.cat((x_emb, x_global), dim=1)    # (B, 3*emb_dims, N)
        x_seg = x_seg.transpose(2, 1).contiguous().view(-1, x_seg.size(1))
        x_seg = self.dp1(F.leaky_relu(self.bn6(self.linear1(x_seg)), 0.2))
        x_seg = self.dp2(F.leaky_relu(self.bn7(self.linear2(x_seg)), 0.2))
        x_seg = self.linear3(x_seg).view(B, N, self.num_part_classes)

        return x_seg

"""
Deep Neural Network Architecture for Iris Recognition (IrisDeepNet).
Tailored for 64x512 normalized polar iris strips with aspect-ratio-aware
convolutions, residual bottleneck blocks, Squeeze-and-Excitation (SE) attention,
and ArcMargin metric learning.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SqueezeExcitation(nn.Module):
    """Channel-wise Squeeze-and-Excitation attention."""
    def __init__(self, channels, reduction=8):
        super().__init__()
        mid_channels = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.fc(x).view(b, c, 1, 1)
        return x * w


class ResidualBlock(nn.Module):
    """Residual block with asymmetric kernels and SE attention."""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 5),
                               stride=stride, padding=(1, 2), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SqueezeExcitation(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = self.relu(out + residual)
        return out


class IrisDeepNet(nn.Module):
    """
    IrisDeepNet: Deep Feature Extractor for Normalized Iris Images.
    
    Accepts normalized iris images of shape (B, in_channels, 64, 512).
    Outputs a unit-normalized (L2 normalized) embedding vector of dimension `embedding_dim`.
    """
    def __init__(self, in_channels=1, embedding_dim=256, num_classes=None, dropout=0.2):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes

        # Stem: asymmetric wide filters to capture radial fibers and concentric rings
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=(5, 7), stride=(1, 2), padding=(2, 3), bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # (64, 256) -> (32, 128)
        )

        # 4-stage residual backbone
        self.stage1 = ResidualBlock(32, 64, stride=(2, 2))   # -> (16, 64)
        self.stage2 = ResidualBlock(64, 128, stride=(2, 2))  # -> (8, 32)
        self.stage3 = ResidualBlock(128, 256, stride=(2, 2)) # -> (4, 16)
        self.stage4 = ResidualBlock(256, 256, stride=(2, 2)) # -> (2, 8)

        # Spatial-Angular pooling & projection
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.gmp = nn.AdaptiveMaxPool2d((1, 1))
        
        # Dual-pooled FC head
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 2, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, embedding_dim, bias=False),
            nn.BatchNorm1d(embedding_dim)
        )

        # Optional margin classification head (for training)
        if num_classes is not None and num_classes > 0:
            self.arc_margin = ArcMarginProduct(embedding_dim, num_classes)
        else:
            self.arc_margin = None

    def extract_features(self, x):
        """Extract raw feature embeddings."""
        feat = self.stem(x)
        feat = self.stage1(feat)
        feat = self.stage2(feat)
        feat = self.stage3(feat)
        feat = self.stage4(feat)

        avg_p = self.gap(feat)
        max_p = self.gmp(feat)
        pooled = torch.cat([avg_p, max_p], dim=1)

        raw_emb = self.head(pooled)
        # L2 normalize embedding to lie on the unit hypersphere
        norm_emb = F.normalize(raw_emb, p=2, dim=1)
        return norm_emb

    def forward(self, x, labels=None):
        """
        Forward pass.
        If `labels` is provided and classifier exists, returns (norm_emb, arc_logits).
        Otherwise returns `norm_emb`.
        """
        emb = self.extract_features(x)
        if self.arc_margin is not None and labels is not None:
            logits = self.arc_margin(emb, labels)
            return emb, logits
        return emb


class ArcMarginProduct(nn.Module):
    """
    Additive Angular Margin Loss (ArcFace) head.
    
    Formula:
        cos(theta + m) for ground-truth class, scaled by s.
    Ensures high inter-class separation and intra-class compactness in spherical space.
    """
    def __init__(self, in_features, out_features, s=30.0, m=0.35, easy_margin=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input_emb, label):
        # input_emb is assumed to be L2-normalized
        # Normalize weights
        cosine = F.linear(input_emb, F.normalize(self.weight))
        sine = torch.sqrt(torch.clamp(1.0 - torch.pow(cosine, 2), 1e-7, 1.0))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # Convert label to one-hot
        one_hot = torch.zeros(cosine.size(), device=input_emb.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1.0)

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output

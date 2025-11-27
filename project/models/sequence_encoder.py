import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# 序列编码器
class SequenceEncoder(nn.Module):
    def __init__(
        self,
        backbone: str = "mobilenet_v3_large",
        hidden_dim: int = 256,
        rnn_layers: int = 2,
    ):
        super().__init__()
        backbone_model = getattr(models, backbone)(weights=None)
        self.cnn = backbone_model.features
        # MobileNetV3 Large的最后一层通道数
        last_channels = 960 if backbone == "mobilenet_v3_large" else backbone_model.classifier[0].in_features
        self.reduce = nn.Conv2d(last_channels, hidden_dim, kernel_size=1)
        # 双向GRU
        self.bi_rnn = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.15 if rnn_layers > 1 else 0,
        )
        # 特征增强
        self.feature_enhance = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        self.norm = nn.LayerNorm(hidden_dim * 2)
        self.dropout = nn.Dropout(0.15)

    # images:输入图像张量，形状为:B × C × H × W
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feat = self.cnn(images)
        feat = self.reduce(feat)
        feat = F.relu(feat)
        # B:batch_size,C:hidden_dim,H:高度,W:宽度
        B, C, H, W = feat.shape
        # 在高度维度上池化，保留宽度维度作为序列长度
        feat = F.adaptive_avg_pool2d(feat, (1, W))
        feat = feat.squeeze(2)
        feat = feat.transpose(1, 2)
        feat = self.dropout(feat)
        seq, _ = self.bi_rnn(feat)
        seq = self.feature_enhance(seq)
        seq = self.dropout(seq)
        return self.norm(seq)
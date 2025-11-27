import torch
import torch.nn as nn


# 双流融合层
class TwoStreamFusion(nn.Module):
    # fused_dim:融合特征维度
    def __init__(self, seq_dim: int, graph_dim: int, fused_dim: int, num_heads: int = 8):
        super().__init__()
        self.seq_dim = seq_dim
        self.graph_dim = graph_dim
        self.fused_dim = fused_dim

        self.seq_proj = nn.Linear(seq_dim, fused_dim)
        self.graph_proj = nn.Linear(graph_dim, fused_dim)

        # 交叉注意力：序列:Query，图:Key和Value
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=fused_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.15
        )

        # Transformer前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(fused_dim, fused_dim * 2),
            nn.LayerNorm(fused_dim * 2),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(fused_dim * 2, fused_dim),
            nn.Dropout(0.15)
        )

        # 添加额外的融合层，增强特征表达能力
        self.fusion_layer = nn.Sequential(
            nn.Linear(fused_dim, fused_dim),
            nn.LayerNorm(fused_dim),
            nn.GELU(),
            nn.Dropout(0.15)
        )

        # 层归一化
        self.norm1 = nn.LayerNorm(fused_dim)
        self.norm2 = nn.LayerNorm(fused_dim)

        # 输出
        self.out_proj = nn.Linear(fused_dim, fused_dim)

    def forward(self, seq_feats: torch.Tensor, graph_global: torch.Tensor) -> torch.Tensor:
        B, T, _ = seq_feats.shape
        # 投影到统一维度
        seq_proj = self.seq_proj(seq_feats)
        graph_proj = self.graph_proj(graph_global).unsqueeze(1)

        # Cross-Attention：序列作为Query，图作为Key和Value
        graph_expanded = graph_proj.expand(-1, T, -1)  # B × T × fused_dim

        # 交叉注意力：序列查询图特征
        attn_out, attn_weights = self.cross_attn(
            query=seq_proj,
            key=graph_expanded,
            value=graph_expanded
        )

        # 残差连接和层归一化
        fused = self.norm1(seq_proj + attn_out)

        ffn_out = self.ffn(fused)
        fused = self.norm2(fused + ffn_out)

        fused = self.fusion_layer(fused)
        fused_representation = self.out_proj(fused)

        return fused_representation
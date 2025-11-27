import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_add_pool
from torch_geometric.data import Batch


# 字图模式识别(WL kernel方法），识别特征基团
class WLSubtreeLayer(nn.Module):
    # hidden_dim:隐藏维度,motif_dim:基团特征维度,steps:WL迭代次数
    def __init__(self, hidden_dim: int, motif_dim: int = 256, steps: int = 3):
        super().__init__()
        self.steps = steps

        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, motif_dim),
            # 层归一化
            nn.LayerNorm(motif_dim),
            # 高斯线性误差（激活函数）
            nn.GELU()
        )
        # 聚合层
        self.agg_layers = nn.ModuleList([
            nn.Sequential(
                # motif_dim * 2:当前节点 + 聚合邻居
                nn.Linear(motif_dim * 2, motif_dim),
                nn.LayerNorm(motif_dim),
                nn.GELU(),
                nn.Dropout(0.1)
            ) for _ in range(steps)  # 3个聚合层
        ])
        # 输出
        self.out = nn.Sequential(
            nn.Linear(motif_dim, motif_dim),
            nn.LayerNorm(motif_dim),
            nn.GELU()
        )

    # x(N,hidden_dim),edge_index:边索引
    def forward(self, x, edge_index):
        # (N, motif_dim)
        h = self.proj(x)
        row, col = edge_index

        for step in range(self.steps):
            agg = torch.zeros_like(h)
            if len(row) > 0:
                agg.index_add_(0, row, h[col])
            # (N, motif_dim * 2)
            combined = torch.cat([h, agg], dim=-1)
            # (N, motif_dim)
            h = self.agg_layers[step](combined)
        # 输出
        h = self.out(h)
        # 全局池化
        h_mean = h.mean(dim=0, keepdim=True)
        h_max = h.max(dim=0)[0].unsqueeze(0)
        # (1, motif_dim * 2)
        motif_embedding = torch.cat([h_mean, h_max], dim=-1)

        return motif_embedding


# 图编码器
class GraphEncoder(nn.Module):
    def __init__(
            self,
            num_atom_types: int,  # 原子类型数量
            num_bond_types: int,  # 化学键类型数量
            embed_dim: int = 128,  # 嵌入维度
            hidden_dim: int = 256,
            num_layers: int = 3,  # GAT层数
            heads: int = 4,  # 注意力头数
    ):
        super().__init__()
        # 节点
        self.node_embed = nn.Embedding(num_atom_types, embed_dim)
        # 边
        self.edge_embed = nn.Embedding(num_bond_types, embed_dim)

        # GAT层列表
        self.layers = nn.ModuleList()
        in_dim = embed_dim
        for i in range(num_layers):
            self.layers.append(
                GATv2Conv(
                    in_channels=in_dim,
                    out_channels=hidden_dim // heads,
                    heads=heads,
                    edge_dim=embed_dim,
                    dropout=0.1,
                )
            )
            in_dim = hidden_dim

        # batch_first:批次维度在前
        self.attention_pool = nn.MultiheadAttention(hidden_dim, num_heads=heads, batch_first=True)
        # WL基团识别
        motif_dim = hidden_dim // 2
        self.motif = WLSubtreeLayer(hidden_dim, motif_dim=motif_dim, steps=3)
        # motif输出是 motif_dim * 2（mean + max）
        self.motif_proj = nn.Linear(motif_dim * 2, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim + hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(0.2)
        # 残差连接
        self.hidden_dim = hidden_dim

    def forward(self, graph: Batch):
        x = self.node_embed(graph.node_types)
        edge_attr = self.edge_embed(graph.edge_types)

        for conv in self.layers:
            x = conv(x, graph.edge_index, edge_attr)
            x = F.relu(x)
            x = self.dropout(x)

        # 图注意力
        # (1, N, C)
        attn_input = x.unsqueeze(0)
        attn_out, _ = self.attention_pool(attn_input, attn_input, attn_input)
        # (N, C)
        attn_out = attn_out.squeeze(0)

        graph_feat = global_add_pool(attn_out, graph.batch)
        # 对每个图单独计算 motif 特征
        motif_feats = []
        num_graphs = graph.batch.max().item() + 1
        for i in range(num_graphs):
            mask = graph.batch == i
            node_feat = attn_out[mask]
            # 获取当前图的边索引
            node_indices = mask.nonzero(as_tuple=True)[0]
            if len(node_indices) == 0:
                motif_feat = torch.zeros(1, self.hidden_dim, device=attn_out.device)
                motif_feats.append(motif_feat)
                continue
            # 计算当前图节点在全局索引中的范围
            node_start = node_indices.min().item()
            node_end = node_indices.max().item() + 1
            # 创建当前图的边掩码
            edge_mask = (graph.edge_index[0] >= node_start) & (graph.edge_index[0] < node_end)
            edge_mask = edge_mask & (graph.edge_index[1] >= node_start) & (graph.edge_index[1] < node_end)

            if edge_mask.sum() > 0:
                local_edge_index = graph.edge_index[:, edge_mask] - node_start
                motif_embedding = self.motif(node_feat, local_edge_index)

                motif_feat = self.motif_proj(motif_embedding.squeeze(0)).unsqueeze(0)
            else:
                motif_feat = torch.zeros(1, self.hidden_dim, device=attn_out.device)
            motif_feats.append(motif_feat)

        motif_feat = torch.cat(motif_feats, dim=0)

        fused = torch.cat([graph_feat, motif_feat], dim=-1)

        return self.out_proj(fused), attn_out

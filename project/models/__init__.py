import torch
import torch.nn as nn

from .graph_encoder import GraphEncoder
from .sequence_encoder import SequenceEncoder
from .fusion_encoder import TwoStreamFusion
from .ctc_crf_decoder import CTCCRFDecoder


class TwoStreamModel(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            num_atom_types: int,
            num_bond_types: int,
            seq_hidden: int = 512,
            graph_hidden: int = 512,
            fusion_dim: int = 512,
            vocab_dict: dict = None,  # 词汇表字典，用于CRF初始化化学约束
    ):
        super().__init__()
        # 增加embed_dim和层数
        self.graph_encoder = GraphEncoder(
            num_atom_types,
            num_bond_types,
            embed_dim=256,
            hidden_dim=graph_hidden,
            num_layers=5,
            heads=8
        )
        # 使用MobileNetV3 Large
        self.sequence_encoder = SequenceEncoder(
            backbone="mobilenet_v3_large",
            hidden_dim=seq_hidden,
            rnn_layers=3
        )
        # 序列编码器输出是双向GRU，所以维度是 seq_hidden * 2
        self.fusion = TwoStreamFusion(seq_dim=seq_hidden * 2, graph_dim=graph_hidden, fused_dim=fusion_dim, num_heads=8)
        # 传入vocab_dict以初始化CRF的化学约束
        # 优化损失权重：平衡CTC和CRF，最大化准确率
        self.decoder = CTCCRFDecoder(
            vocab_size,
            fusion_dim,
            lambda_ctc=1.0,  # CTC权重
            lambda_crf=0.4,  # CRF权重
            vocab_dict=vocab_dict,
            label_smoothing=0.0,
            temperature=1.0  # 温度缩放
        )

    def forward(self, images, graph_batch, targets, target_lengths):
        seq_feats = self.sequence_encoder(images)
        graph_global, _ = self.graph_encoder(graph_batch)
        fused = self.fusion(seq_feats, graph_global)
        input_lengths = torch.full((fused.size(0),), fused.size(1), dtype=torch.long, device=fused.device)
        logits, loss, aux = self.decoder(fused, targets, target_lengths, input_lengths)
        return logits, loss, aux

    def decode(self, images, graph_batch):
        seq_feats = self.sequence_encoder(images)
        graph_global, _ = self.graph_encoder(graph_batch)
        fused = self.fusion(seq_feats, graph_global)
        input_lengths = torch.full((fused.size(0),), fused.size(1), dtype=torch.long, device=fused.device)
        mask = self.decoder._build_mask(input_lengths, fused.size(1))
        logits = self.decoder.classifier(fused)
        ctc_paths, crf_paths = self.decoder.decode(logits, mask)
        return logits, mask, (ctc_paths, crf_paths)


DualStreamRecognizer = TwoStreamModel

__all__ = [
    'GraphEncoder',
    'SequenceEncoder',
    'TwoStreamFusion',
    'CTCCRFDecoder',
    'TwoStreamModel',
    'DualStreamRecognizer'
]

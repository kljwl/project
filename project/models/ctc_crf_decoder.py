import torch
import torch.nn as nn
import torch.nn.functional as F


# 带化学方程式约束的CRF
class ChemicalEquationCRF(nn.Module):
    # num_tags:词汇表大小
    def __init__(self, num_tags: int, vocab_dict: dict = None):
        super().__init__()
        self.num_tags = num_tags
        self.vocab_dict = vocab_dict or {}
        self.idx_to_token = {v: k for k, v in self.vocab_dict.items()}

        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))
        self.start_trans = nn.Parameter(torch.empty(num_tags))
        self.end_trans = nn.Parameter(torch.empty(num_tags))

        # 使用更小的初始值，避免损失过大
        nn.init.xavier_uniform_(self.transitions, gain=0.1)
        nn.init.normal_(self.start_trans, mean=0.0, std=0.1)
        nn.init.normal_(self.end_trans, mean=0.0, std=0.1)

        self._initialize_chemical_constraints()

    # 初始化化学方程式约束
    def _initialize_chemical_constraints(self):
        with torch.no_grad():
            # 默认所有转移都是可能的（使用更小的负值，避免损失过大）
            self.transitions.fill_(-0.5)

            # 允许blank到任何字符（CTC）
            if 0 in self.idx_to_token:
                # 允许blank转移
                self.transitions[0, :] = 0.0
                # 允许转移到blank
                self.transitions[:, 0] = 0.0

            # 添加更具体的约束
            if self.vocab_dict:
                self._add_subscript_constraints()
                self._add_charge_constraints()
                self._add_arrow_constraints()

    # 添加上下标约束
    def _add_subscript_constraints(self):
        subscript_tokens = ['_2', '_3', '_4', '_5', '_6', '_7']
        element_tokens = ['H', 'C', 'O', 'N', 'F', 'S', 'P', 'Cl', 'Na', 'K', 'Ca', 'Mg', 'Al', 'Fe', 'Cu', 'Zn', 'Ag',
                          'Ba']

        # 获取下标和元素在词汇表中的索引
        subscript_indices = [self.vocab_dict.get(t, -1) for t in subscript_tokens if t in self.vocab_dict]
        element_indices = [self.vocab_dict.get(t, -1) for t in element_tokens if t in self.vocab_dict]

        # 允许元素符号后跟下标
        for elem_idx in element_indices:
            if elem_idx >= 0:
                for sub_idx in subscript_indices:
                    if sub_idx >= 0:
                        self.transitions[elem_idx, sub_idx] = 0.2

    # 添加离子电荷约束
    def _add_charge_constraints(self):
        charge_tokens = ['|+', '|2+', '|3+', '|4+', '|5+', '|6+', '|7+', '|-', '|2-', '|3-', '|4-', '|5-', '|6-', '|7-']
        charge_indices = [self.vocab_dict.get(t, -1) for t in charge_tokens if t in self.vocab_dict]

        # 电荷符号可以跟在元素、数字、括号后
        allowed_before = ['H', 'C', 'O', 'N', 'F', 'S', 'P', 'Cl', 'Na', 'K', 'Ca', 'Mg', 'Al', 'Fe', 'Cu', 'Zn', 'Ag',
                          'Ba',
                          '2', '3', '4', '5', '6', '7', '8', '9', ')']
        allowed_indices = [self.vocab_dict.get(t, -1) for t in allowed_before if t in self.vocab_dict]

        for before_idx in allowed_indices:
            if before_idx >= 0:
                for charge_idx in charge_indices:
                    if charge_idx >= 0:
                        self.transitions[before_idx, charge_idx] = 0.2

    # 添加箭头约束
    def _add_arrow_constraints(self):
        arrow_tokens = ['=', '\\~=', '\\$=', '\\@=', '\\&=', '\\*=']
        arrow_indices = [self.vocab_dict.get(t, -1) for t in arrow_tokens if t in self.vocab_dict]

        # 箭头后不能直接跟blank或其他箭头（化学方程式文法规则）
        for arrow_idx in arrow_indices:
            if arrow_idx >= 0:
                # 箭头后可以跟元素、数字、括号等（箭头右边不得为空）
                allowed_after = ['H', 'C', 'O', 'N', 'F', 'S', 'P', 'Cl', 'Na', 'K', 'Ca', 'Mg', 'Al', 'Fe', 'Cu', 'Zn',
                                 'Ag', 'Ba',
                                 '2', '3', '4', '5', '6', '7', '8', '9', '(', '+', '^']  # ^（气体符号）
                allowed_indices = [self.vocab_dict.get(t, -1) for t in allowed_after if t in self.vocab_dict]

                for after_idx in allowed_indices:
                    if after_idx >= 0:
                        self.transitions[arrow_idx, after_idx] = 0.3

                # 禁止箭头后直接跟blank
                if 0 in self.idx_to_token:
                    self.transitions[arrow_idx, 0] = -1.0  # 禁止箭头后为空（使用更温和的值）

                # 禁止箭头后直接跟其他箭头
                for other_arrow_idx in arrow_indices:
                    if other_arrow_idx >= 0 and other_arrow_idx != arrow_idx:
                        # 禁止连续箭头
                        self.transitions[arrow_idx, other_arrow_idx] = -1.0

    def _log_sum_exp(self, tensor, dim):
        max_score, _ = tensor.max(dim)
        return max_score + torch.log(torch.sum(torch.exp(tensor - max_score.unsqueeze(dim)), dim=dim) + 1e-8)

    def forward(self, emissions, tags, mask):
        # emissions:B × T × V,mask:B × T
        B, T, V = emissions.shape
        log_den = self._compute_log_partition(emissions, mask)
        log_num = self._compute_gold_score(emissions, tags, mask)
        # CRF损失：log_den - log_num
        # 对每个样本的损失进行归一化，避免长序列主导
        raw_loss = log_den - log_num
        # 按有效序列长度归一化
        seq_lengths = mask.sum(dim=1).float()
        # 避免为零，最小长度为1
        seq_lengths = torch.clamp(seq_lengths, min=1.0)
        # 归一化损失：除以序列长度
        normalized_loss = raw_loss / seq_lengths
        return normalized_loss

    # 计算真实标签的得分
    def _compute_gold_score(self, emissions, tags, mask):
        B, T, V = emissions.shape
        # emissions已经是log_probs，数值范围通常在[-10, 0]，比较稳定
        score = self.start_trans[tags[:, 0]]
        score += emissions[:, 0, :].gather(1, tags[:, 0:1]).squeeze(1)

        for t in range(1, T):
            emit = emissions[:, t, :].gather(1, tags[:, t:t + 1]).squeeze(1)
            trans = self.transitions[tags[:, t - 1], tags[:, t]]
            score += (emit + trans) * mask[:, t]

        # 结束转移
        last_tag_indices = mask.sum(dim=1).long() - 1
        last_tags = tags.gather(1, last_tag_indices.unsqueeze(1)).squeeze(1)
        score += self.end_trans[last_tags]

        return score

    def _compute_log_partition(self, emissions, mask):
        B, T, V = emissions.shape

        alpha = self.start_trans.unsqueeze(0).expand(B, -1) + emissions[:, 0, :]

        for t in range(1, T):
            # 当前时刻的发射分数
            emit = emissions[:, t, :].unsqueeze(1)
            # 转移分数
            trans = self.transitions.unsqueeze(0)
            # 前向递推
            alpha_t = alpha.unsqueeze(2) + trans + emit
            new_alpha = self._log_sum_exp(alpha_t, dim=1)
            # 应用mask
            mask_t = mask[:, t].unsqueeze(1)
            alpha = torch.where(mask_t, new_alpha, alpha)

        alpha += self.end_trans.unsqueeze(0)
        return self._log_sum_exp(alpha, dim=1)

    def viterbi_decode(self, emissions, mask):
        B, T, V = emissions.shape
        score = self.start_trans.unsqueeze(0).expand(B, -1) + emissions[:, 0, :]
        path = []

        for t in range(1, T):
            emit = emissions[:, t, :]
            # 计算所有可能的转移
            score_t = score.unsqueeze(2) + self.transitions.unsqueeze(0)
            best_score, best_path = score_t.max(dim=1)
            score = best_score + emit
            path.append(best_path)
            # 应用mask
            mask_t = mask[:, t].unsqueeze(1)  # B × 1
            score = torch.where(mask_t, score, score)

        # 添加结束转移
        score += self.end_trans.unsqueeze(0)
        best_score, best_path_idx = score.max(dim=1)  # B

        # 回溯
        paths = [best_path_idx]
        for bp in reversed(path):
            best_path_idx = bp.gather(1, best_path_idx.unsqueeze(1)).squeeze(1)
            paths.insert(0, best_path_idx)

        paths = torch.stack(paths, dim=1)
        return paths * mask.long()


# CTC+CRF联合解码器
class CTCCRFDecoder(nn.Module):
    # blank_idx:CTC空白符索引,lambda_ctc:CTC损失权重,lambda_crf:CRF损失权重
    def __init__(self, vocab_size: int, hidden_dim: int, blank_idx: int = 0,
                 lambda_ctc: float = 1.0, lambda_crf: float = 0.5, vocab_dict: dict = None,
                 label_smoothing: float = 0.1, temperature: float = 1.0):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim // 2, vocab_size)
        )

        self.ctc_loss = nn.CTCLoss(blank=blank_idx, zero_infinity=True, reduction='mean')

        self.crf = ChemicalEquationCRF(vocab_size, vocab_dict=vocab_dict)
        self.blank_idx = blank_idx
        self.lambda_ctc = lambda_ctc
        self.lambda_crf = lambda_crf
        self.label_smoothing = label_smoothing
        self.temperature = temperature

    def forward(self, fused_feats, targets, target_lengths, input_lengths):
        # 通过分类器得到logits
        logits = self.classifier(fused_feats)  # B × T × V

        # 温度缩放：软化logits分布，使训练更稳定
        if self.temperature != 1.0:
            logits = logits / self.temperature

        log_probs = F.log_softmax(logits, dim=-1)

        # CTC损失：CTC的log_probs
        ctc = self.ctc_loss(log_probs.transpose(0, 1), targets, input_lengths, target_lengths)

        # CRF损失：将CTC得到的log_probs送入CRF
        mask = self._build_mask(input_lengths, logits.size(1))
        crf_targets = self._pad_targets_for_crf(targets, target_lengths, logits.size(1))
        crf = self.crf(log_probs, crf_targets, mask)

        crf_mean = crf.mean()

        loss = self.lambda_ctc * ctc + self.lambda_crf * crf_mean

        # 损失平滑：如果损失过大，进行平滑处理
        if loss.item() > 5.0:
            # 损失过大时，使用更保守的权重，主要依赖CTC
            loss = 0.8 * self.lambda_ctc * ctc + 0.3 * self.lambda_crf * crf_mean

        if torch.isnan(loss) or torch.isinf(loss):
            loss = self.lambda_ctc * ctc

        # 添加损失裁剪，避免损失过大
        if loss.item() > 10.0:
            loss = torch.clamp(loss, max=10.0)

        return logits, loss, {'ctc': ctc.item(), 'crf': crf_mean.item()}

    def decode(self, logits, mask):
        log_probs = F.log_softmax(logits, dim=-1)

        # CRF解码
        crf_paths = self.crf.viterbi_decode(log_probs, mask)

        # CTC解码
        ctc_paths = torch.argmax(logits, dim=-1)

        # 应用mask，将无效位置设为blank
        ctc_paths = ctc_paths * mask.long() + (~mask) * self.blank_idx

        return ctc_paths, crf_paths

    @staticmethod
    def _build_mask(lengths, max_len):
        batch = lengths.size(0)
        mask = torch.arange(max_len, device=lengths.device).unsqueeze(0).expand(batch, -1)
        mask = mask < lengths.unsqueeze(1)
        return mask

    def _pad_targets_for_crf(self, flat_targets, lengths, max_len):
        batch = lengths.size(0)
        padded = flat_targets.new_full((batch, max_len), self.blank_idx)
        offset = 0
        total = flat_targets.size(0)
        for i in range(batch):
            cur_len = lengths[i].item()
            copy_len = min(cur_len, max_len)
            if copy_len > 0 and offset < total:
                padded[i, :copy_len] = flat_targets[offset: offset + copy_len]
            offset += cur_len
        return padded

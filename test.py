import os
from typing import List, Tuple

# 修复 OpenMP 冲突问题（Windows 环境）
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import matplotlib.pyplot as plt
import matplotlib
# 配置matplotlib支持中文显示
try:
    # Windows系统常用中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
except Exception:
    # 如果设置失败，使用默认字体
    pass

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from dataset.chem_dataset import ChemDataset, collate_fn
from models import DualStreamRecognizer
from train import build_graph_batch


def load_vocabulary(classes_file: str = "dataset/classes.txt"):
    """加载词汇表，将token ID映射到字符"""
    vocab = {}
    if os.path.exists(classes_file):
        with open(classes_file, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                char = line.strip()
                if char:
                    vocab[idx] = char
    else:
        # 默认词汇表
        default_chars = ["H", "O", "C", "N", "+", "=", "2", "3", "4", "5", "6", "7", "8", "9", 
                        "(", ")", "Cu", "Fe", "Al", "Na", "Cl", "S", "P", "K", "Ca", "Mg", "Zn"]
        for idx, char in enumerate(default_chars):
            vocab[idx] = char
    
    # 添加空白符（通常为0）
    if 0 not in vocab:
        vocab[0] = ""  # blank token
    
    return vocab


def tokens_to_formula(token_ids: List[int], vocab: dict) -> str:
    """将token ID序列转换为化学方程式文本"""
    formula = ""
    for token_id in token_ids:
        if token_id in vocab:
            char = vocab[token_id]
            if char:  # 跳过空白符
                formula += char
    return formula if formula else ""


def calculate_edit_distance(pred: List[int], target: List[int]) -> int:
    """计算编辑距离（Levenshtein距离）"""
    m, n = len(pred), len(target)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred[i-1] == target[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(
                    dp[i-1][j] + 1,      # 删除
                    dp[i][j-1] + 1,      # 插入
                    dp[i-1][j-1] + 1     # 替换
                )
    
    return dp[m][n]


def calculate_accuracy_metrics(pred_tokens: List[int], target_tokens: List[int]) -> Tuple[float, float, float, int]:
    """
    计算准确率指标
    
    Returns:
        exact_match: 完全匹配准确率 (0或1)
        char_accuracy: 字符级别准确率
        sequence_accuracy: 序列准确率（考虑顺序）
        edit_distance: 编辑距离
    """
    if not target_tokens:
        return 0.0, 0.0, 0.0, len(pred_tokens) if pred_tokens else 0
    
    # 1. 完全匹配准确率
    exact_match = 1.0 if pred_tokens == target_tokens else 0.0
    
    # 2. 字符级别准确率（不考虑顺序）
    pred_set = set(pred_tokens)
    target_set = set(target_tokens)
    intersection = pred_set & target_set
    union = pred_set | target_set
    char_accuracy = len(intersection) / len(union) if union else 0.0
    
    # 3. 序列准确率（考虑顺序，使用最长公共子序列）
    m, n = len(pred_tokens), len(target_tokens)
    # 计算最长公共子序列长度
    lcs_dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_tokens[i-1] == target_tokens[j-1]:
                lcs_dp[i][j] = lcs_dp[i-1][j-1] + 1
            else:
                lcs_dp[i][j] = max(lcs_dp[i-1][j], lcs_dp[i][j-1])
    lcs_len = lcs_dp[m][n]
    sequence_accuracy = lcs_len / max(m, n) if max(m, n) > 0 else 0.0
    
    # 4. 编辑距离
    edit_dist = calculate_edit_distance(pred_tokens, target_tokens)
    
    return exact_match, char_accuracy, sequence_accuracy, edit_dist


def ctc_greedy_decode_from_logits(logits: torch.Tensor, mask: torch.Tensor, blank_idx: int = 0) -> List[int]:
    """
    从logits进行真正的CTC贪心解码：去重blank token和重复字符
    logits: T × V 或 B × T × V (如果是B×T×V，取第一个batch)
    mask: T 或 B × T
    """
    if logits.dim() == 3:
        logits = logits[0]  # 取第一个batch
        mask = mask[0] if mask.dim() == 2 else mask
    
    # 获取有效长度
    valid_len = mask.sum().item()
    if valid_len == 0:
        return []
    
    # 对每个时间步取argmax
    preds = torch.argmax(logits[:valid_len], dim=-1).cpu().tolist()
    
    # CTC解码：去重blank和重复字符
    decoded = []
    prev = blank_idx
    
    for token in preds:
        # 跳过blank token
        if token == blank_idx:
            prev = blank_idx
            continue
        # 去重：如果当前token和上一个非blank token相同，跳过
        if token != prev:
            decoded.append(token)
            prev = token
    
    return decoded


def ctc_beam_decode(logits: torch.Tensor, mask: torch.Tensor, blank_idx: int = 0, beam_size: int = 10) -> List[int]:  # 增加beam size
    """
    使用beam search进行CTC解码（更准确）
    """
    try:
        batch_idx = 0
        B, T, V = logits.shape
        
        # 获取有效长度
        valid_len = mask[batch_idx].sum().item()
        if valid_len == 0:
            return []
        
        log_probs = torch.log_softmax(logits[batch_idx, :valid_len], dim=-1)  # T × V
        
        # 简单的beam search
        beams = [([], 0.0, blank_idx)]  # (sequence, score, last_token)
        
        for t in range(valid_len):
            new_beams = []
            for seq, score, last_token in beams:
                # 获取top-k tokens
                top_k = min(beam_size, V)
                top_probs, top_tokens = log_probs[t].topk(top_k)
                
                for prob, token_tensor in zip(top_probs, top_tokens):
                    token = token_tensor.item()
                    new_score = score + prob.item()
                    
                    if token == blank_idx:
                        # blank token，序列不变
                        new_beams.append((seq, new_score, last_token))
                    elif token == last_token:
                        # 重复token，跳过（CTC规则）
                        new_beams.append((seq, new_score, token))
                    else:
                        # 新token，添加到序列
                        new_beams.append((seq + [token], new_score, token))
            
            # 去重：合并相同序列的beam
            beam_dict = {}
            for seq, score, last_token in new_beams:
                seq_key = tuple(seq)
                if seq_key not in beam_dict or score > beam_dict[seq_key][1]:
                    beam_dict[seq_key] = (seq, score, last_token)
            
            # 保留top beam_size个
            beams = list(beam_dict.values())
            beams.sort(key=lambda x: x[1], reverse=True)
            beams = beams[:beam_size]
        
        # 返回得分最高的序列
        if beams:
            return beams[0][0]
    except Exception as e:
        # 如果beam search失败，返回空列表，让调用者使用greedy decode
        pass
    return []


def fuse_predictions(ctc_preds: List[int], fusion_preds: List[int], crf_preds: List[int], 
                     logits: torch.Tensor = None, mask: torch.Tensor = None, blank_idx: int = 0) -> List[int]:
    """
    智能融合三种预测结果
    
    策略：
    1. 优先使用CRF（考虑了序列依赖，通常最准确）
    2. 如果CRF结果不完整，使用CTC/Fusion补充
    3. 字符级别投票融合，确保结果完整
    """
    # 如果所有结果都为空，返回空
    if not ctc_preds and not fusion_preds and not crf_preds:
        return []
    
    # 策略1：如果CRF结果完整且合理，直接使用
    if len(crf_preds) >= max(len(ctc_preds), len(fusion_preds)) * 0.7:
        # CRF结果长度合理（至少是其他结果的70%），直接使用
        return crf_preds
    
    # 策略2：如果CTC和Fusion一致且比CRF长，优先使用它们
    if ctc_preds == fusion_preds and len(ctc_preds) > len(crf_preds) * 1.2:
        # CTC和Fusion一致，且明显比CRF长，说明CRF可能截断了
        # 但需要检查置信度
        if logits is not None and mask is not None:
            # 计算CTC/Fusion的平均置信度
            probs = torch.softmax(logits, dim=-1)  # B × T × V
            batch_idx = 0
            conf_sum = 0.0
            valid_count = 0
            for i, token in enumerate(ctc_preds):
                if i < mask.size(1) and mask[batch_idx, i]:
                    conf_sum += probs[batch_idx, i, token].item()
                    valid_count += 1
            avg_conf = conf_sum / max(valid_count, 1)
            
            if avg_conf > 0.2:  # 置信度足够高
                return ctc_preds
    
    # 策略3：字符级别投票融合
    candidates = []
    if ctc_preds:
        candidates.append(('ctc', ctc_preds, 1.0))
    if fusion_preds and fusion_preds != ctc_preds:  # 避免重复
        candidates.append(('fusion', fusion_preds, 1.0))
    if crf_preds:
        candidates.append(('crf', crf_preds, 1.5))  # CRF权重更高
    
    if not candidates:
        return []
    
    # 计算置信度并加权
    if logits is not None and mask is not None:
        probs = torch.softmax(logits, dim=-1)  # B × T × V
        batch_idx = 0
        
        scored_candidates = []
        for name, seq, base_weight in candidates:
            if not seq:
                continue
            
            conf_sum = 0.0
            valid_count = 0
            for i, token in enumerate(seq):
                # 注意：seq是解码后的序列，长度可能小于mask
                # 我们需要从原始logits中找到对应的位置
                # 简化处理：使用序列索引作为参考
                if i < mask.size(1) and mask[batch_idx, i]:
                    conf_sum += probs[batch_idx, i, token].item()
                    valid_count += 1
            
            avg_conf = conf_sum / max(valid_count, 1) if valid_count > 0 else 0.1
            final_weight = base_weight * (0.5 + avg_conf)
            scored_candidates.append((name, seq, final_weight, len(seq)))
        
        # 按权重和长度排序
        scored_candidates.sort(key=lambda x: (x[2], x[3]), reverse=True)
        
        if not scored_candidates:
            return []
        
        # 使用最长的合理序列作为基础，其他序列进行投票修正
        base_seq = scored_candidates[0][1]
        base_weight = scored_candidates[0][2]
        
        fused = []
        max_len = max([len(seq) for _, seq, _, _ in scored_candidates])
        
        for i in range(max_len):
            votes = {}
            
            # 基础序列（权重最高）
            if i < len(base_seq):
                token = base_seq[i]
                votes[token] = votes.get(token, 0) + base_weight
            
            # 其他序列投票
            for name, seq, weight, _ in scored_candidates[1:]:
                if i < len(seq):
                    token = seq[i]
                    votes[token] = votes.get(token, 0) + weight * 0.8
            
            if votes:
                best_token = max(votes.items(), key=lambda x: x[1])[0]
                fused.append(best_token)
        
        return fused if fused else base_seq
    
    # 如果没有logits，使用简单投票
    all_seqs = [seq for _, seq, _ in candidates if seq]
    if not all_seqs:
        return []
    
    # 选择最长的序列
    return max(all_seqs, key=len)


def _normalize_for_vis(img_tensor: torch.Tensor):
    img = img_tensor.detach().cpu().permute(1, 2, 0).numpy()
    img = (img * 0.5 + 0.5).clip(0, 1)
    return img


def _format_seq(seq: List[int], vocab: dict = None, max_len: int = 25, show_ids: bool = False) -> str:
    """格式化序列用于显示"""
    if vocab is not None:
        # 转换为化学方程式文本
        formula = tokens_to_formula(seq, vocab)
        if len(formula) > max_len:
            return formula[:max_len] + "..."
        return formula
    else:
        # 显示token ID
        seq_str = " ".join(map(str, seq[:max_len]))
        if len(seq) > max_len:
            seq_str += "..."
        return seq_str

# 可视化预测样本
def visualize_samples(images, ctc_preds, fusion_preds, crf_preds, save_path, vocab: dict = None, max_samples: int = 8):
    os.makedirs(save_path, exist_ok=True)
    
    # 保存输入图像
    save_image(images, os.path.join(save_path, "inputs.png"))
    print(f"输入图像已保存到: {save_path}/inputs.png")

    # 保存详细预测文本
    with open(os.path.join(save_path, "prediction_samples.txt"), "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("预测结果详情\n")
        f.write("=" * 60 + "\n\n")
        for i in range(len(ctc_preds)):
            f.write(f"样本 {i+1}:\n")
            f.write(f"  输入图像: Sample {i+1}\n")
            
            # 转换为化学方程式
            ctc_formula = _format_seq(ctc_preds[i], vocab, max_len=100)
            fusion_formula = _format_seq(fusion_preds[i], vocab, max_len=100)
            crf_formula = _format_seq(crf_preds[i], vocab, max_len=100)
            
            f.write(f"序列流预测 (CTC): {ctc_formula}\n")
            f.write(f"图流+序列流预测 (Fusion): {fusion_formula}\n")
            f.write(f"CRF最终输出: {crf_formula}\n")
            f.write(f"\n  Token IDs:\n")
            f.write(f"CTC: {_format_seq(ctc_preds[i], None, max_len=50, show_ids=True)}\n")
            f.write(f"Fusion: {_format_seq(fusion_preds[i], None, max_len=50, show_ids=True)}\n")
            f.write(f"CRF: {_format_seq(crf_preds[i], None, max_len=50, show_ids=True)}\n")
            f.write("-" * 60 + "\n\n")

    # 生成可视化图片：每个样本显示4个部分
    max_samples = min(max_samples, images.size(0))
    cols = 4  # 4列：输入图像、CTC、Fusion、CRF
    rows = max_samples
    
    # 增大图片尺寸以便清晰显示
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.5))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx in range(max_samples):
        # 转换为化学方程式文本
        ctc_formula = _format_seq(ctc_preds[idx], vocab, max_len=30)
        fusion_formula = _format_seq(fusion_preds[idx], vocab, max_len=30)
        crf_formula = _format_seq(crf_preds[idx], vocab, max_len=30)
        
        # 第1列：输入图像
        axes[idx, 0].imshow(_normalize_for_vis(images[idx]))
        axes[idx, 0].set_title("输入图像\n(Input Image)", fontsize=10, fontweight='bold')
        axes[idx, 0].axis("off")
        
        # 第2列：序列流预测 (CTC)
        axes[idx, 1].imshow(_normalize_for_vis(images[idx]))
        axes[idx, 1].text(0.5, 0.95, f"序列流预测 (CTC)\n{ctc_formula}", 
                          transform=axes[idx, 1].transAxes,
                          fontsize=9, ha='center', va='top',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
        axes[idx, 1].set_title("序列流预测\n(Sequence Stream - CTC)", fontsize=9, fontweight='bold')
        axes[idx, 1].axis("off")
        
        # 第3列：图流+序列流预测 (Fusion)
        axes[idx, 2].imshow(_normalize_for_vis(images[idx]))
        axes[idx, 2].text(0.5, 0.95, f"图流+序列流 (Fusion)\n{fusion_formula}", 
                          transform=axes[idx, 2].transAxes,
                          fontsize=9, ha='center', va='top',
                          bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9))
        axes[idx, 2].set_title("图流+序列流预测\n(Graph+Sequence - Fusion)", fontsize=9, fontweight='bold')
        axes[idx, 2].axis("off")
        
        # 第4列：CRF最终输出
        axes[idx, 3].imshow(_normalize_for_vis(images[idx]))
        axes[idx, 3].text(0.5, 0.95, f"CRF最终输出\n{crf_formula}", 
                          transform=axes[idx, 3].transAxes,
                          fontsize=9, ha='center', va='top',
                          bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9))
        axes[idx, 3].set_title("CRF最终输出\n(CRF Final Output)", fontsize=9, fontweight='bold')
        axes[idx, 3].axis("off")
    
    plt.suptitle("预测结果对比：输入图像 ; 序列流(CTC) ; 图流+序列流(Fusion) ; CRF最终输出",
                 fontsize=11, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    pred_path = os.path.join(save_path, "prediction_samples.png")
    plt.savefig(pred_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"预测样本可视化已保存到: {pred_path}")


def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载词汇表
    vocab = load_vocabulary("dataset/classes.txt")
    print(f"已加载词汇表，包含 {len(vocab)} 个token")
    
    dataset = ChemDataset(root_dir="dataset", target_size=(256, 256), augment=False)
    # 随机抽取3个样本
    import random
    indices = random.sample(range(len(dataset)), min(3, len(dataset)))
    subset = torch.utils.data.Subset(dataset, indices)
    dataloader = DataLoader(subset, batch_size=3, shuffle=False, collate_fn=collate_fn, num_workers=0)

    # 使用与训练时相同的模型配置
    # 加载词汇表字典用于CRF初始化，并获取实际词汇表大小
    # 使用与ChemDataset相同的加载方式，确保一致性
    vocab_dict = {}
    classes_list = []
    with open("dataset/classes.txt", "r", encoding="utf-8") as f:
        for line in f:
            char = line.strip()
            if char:  # 只添加非空行
                classes_list.append(char)
    
    # 确保第一个是blank token（与ChemDataset._load_classes保持一致）
    if not classes_list or classes_list[0] != "":
        classes_list.insert(0, "")  # blank token at index 0
    
    # 创建vocab_dict
    for idx, char in enumerate(classes_list):
        vocab_dict[char] = idx
    
    # 使用实际词汇表大小
    vocab_size = len(classes_list)
    print(f"使用词汇表大小: {vocab_size} (来自classes.txt，包含blank token)")
    
    # 先创建模型（稍后可能会根据checkpoint调整）
    model = DualStreamRecognizer(
        vocab_size, 
        num_atom_types=100, 
        num_bond_types=5,
        seq_hidden=512,
        graph_hidden=512,
        fusion_dim=512,
        vocab_dict=vocab_dict  # 传入词汇表字典
    ).to(device)
    
    # 优先加载最佳模型，如果没有则加载最终模型
    if os.path.exists("results/model_best.pth"):
        checkpoint_path = "results/model_best.pth"
        print(f"正在加载最佳模型: {checkpoint_path}")
    elif os.path.exists("results/model_final.pth"):
        checkpoint_path = "results/model_final.pth"
        print(f"正在加载最终模型: {checkpoint_path}")
    else:
        # 尝试加载最新的checkpoint
        import glob
        checkpoints = glob.glob("results/checkpoint_epoch_*.pth")
        if checkpoints:
            checkpoint_path = max(checkpoints, key=os.path.getctime)
            print(f"正在加载检查点: {checkpoint_path}")
        else:
            raise FileNotFoundError("在results目录中未找到模型检查点")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    checkpoint_state = checkpoint["model_state"] if isinstance(checkpoint, dict) and "model_state" in checkpoint else checkpoint

    checkpoint_vocab_size = None

    if "decoder.crf.transitions" in checkpoint_state:
        checkpoint_vocab_size = checkpoint_state["decoder.crf.transitions"].shape[0]
        print(f"从CRF transitions检测到检查点词汇表大小: {checkpoint_vocab_size}")

    elif "decoder.crf.start_trans" in checkpoint_state:
        checkpoint_vocab_size = checkpoint_state["decoder.crf.start_trans"].shape[0]
        print(f"从CRF start_trans检测到检查点词汇表大小: {checkpoint_vocab_size}")

    elif "decoder.classifier.8.weight" in checkpoint_state:
        checkpoint_vocab_size = checkpoint_state["decoder.classifier.8.weight"].shape[0]
        print(f"从classifier.8检测到检查点词汇表大小: {checkpoint_vocab_size}")

    elif "decoder.classifier.4.weight" in checkpoint_state:
        checkpoint_vocab_size = checkpoint_state["decoder.classifier.4.weight"].shape[0]
        print(f"从classifier.4检测到检查点词汇表大小: {checkpoint_vocab_size}")
    else:
        # 如果都找不到，打印所有可用的键以便调试
        print("警告: 无法从检查点推断词汇表大小。可用的键:")
        classifier_keys = [k for k in checkpoint_state.keys() if "classifier" in k or "crf" in k]
        for key in classifier_keys[:10]:  # 只显示前10个
            print(f"  {key}: {checkpoint_state[key].shape if hasattr(checkpoint_state[key], 'shape') else 'N/A'}")
    
    # 如果checkpoint的vocab_size与当前不匹配，使用checkpoint的vocab_size重新创建模型
    if checkpoint_vocab_size is not None and checkpoint_vocab_size != vocab_size:
        print(f"\n警告:词汇表大小不匹配!")
        print(f"检查点词汇表大小:{checkpoint_vocab_size}")
        print(f"当前词汇表大小(来自classes.txt): {vocab_size}")
        print(f"使用检查点的词汇表大小({checkpoint_vocab_size})以匹配检查点。")
        
        # 使用checkpoint的vocab_size重新创建模型
        vocab_size = checkpoint_vocab_size
        # 重新创建模型
        model = DualStreamRecognizer(
            vocab_size, 
            num_atom_types=100, 
            num_bond_types=5,
            seq_hidden=512,
            graph_hidden=512,
            fusion_dim=512,
            vocab_dict=vocab_dict  # 仍然使用vocab_dict，但模型大小会匹配checkpoint
        ).to(device)
        print(f"模型已使用词汇表大小 {vocab_size} 重新创建")
    
    # 尝试加载模型
    try:
        model.load_state_dict(checkpoint_state, strict=True)
        print("模型加载成功 (strict=True)")
    except RuntimeError as e:
        error_msg = str(e)
    
    model.eval()

    os.makedirs("results", exist_ok=True)
    with torch.no_grad():
        for idx, (images, boxes_list, infos, _) in enumerate(dataloader):
            images = images.to(device)
            graph_batch = build_graph_batch(infos).to(device)
            logits, mask, (ctc_paths, crf_paths) = model.decode(images, graph_batch)
            
            # 对每个样本进行解码
            blank_idx = 0
            batch_size = logits.size(0)
            ctc_preds = []
            fusion_preds = []
            crf_preds = []
            
            for i in range(batch_size):
                # 获取当前样本的logits和mask
                sample_logits = logits[i]  # T × V
                sample_mask = mask[i]  # T
                
                # 1. CTC解码：使用beam search进行更准确的解码
                ctc_decoded = ctc_beam_decode(sample_logits.unsqueeze(0), sample_mask.unsqueeze(0), blank_idx, beam_size=10)
                if not ctc_decoded:  # 如果beam search失败，使用greedy decode
                    ctc_decoded = ctc_greedy_decode_from_logits(sample_logits.unsqueeze(0), sample_mask.unsqueeze(0), blank_idx)
                
                # 2. Fusion解码：也是从logits进行CTC解码（使用beam search）
                fusion_decoded = ctc_beam_decode(sample_logits.unsqueeze(0), sample_mask.unsqueeze(0), blank_idx, beam_size=10)
                if not fusion_decoded:  # 如果beam search失败，使用greedy decode
                    fusion_decoded = ctc_greedy_decode_from_logits(sample_logits.unsqueeze(0), sample_mask.unsqueeze(0), blank_idx)
                
                # 3. CRF解码：从CRF路径中提取有效部分
                crf_path = crf_paths[i]  # T
                valid_len = sample_mask.sum().item()
                crf_valid = crf_path[:valid_len].cpu().tolist()
                # CRF解码：去除blank token（CRF已经考虑了序列依赖，只需要去除blank）
                crf_cleaned = [t for t in crf_valid if t != blank_idx]
                
                # 如果CRF结果太短，尝试使用CTC结果补充
                if len(crf_cleaned) < len(ctc_decoded) * 0.5 and len(ctc_decoded) > 0:
                    # CRF可能截断了，使用CTC作为参考
                    # 但优先保留CRF的结果（因为它考虑了序列依赖）
                    pass
                
                ctc_preds.append(ctc_decoded)
                fusion_preds.append(fusion_decoded)
                crf_preds.append(crf_cleaned)
            
            
            # 获取真实标签
            ground_truths = []
            for info in infos:
                if "formula_tokens" in info and info["formula_tokens"].numel() > 0:
                    gt_tokens = info["formula_tokens"].cpu().tolist()
                    ground_truths.append(gt_tokens)
                else:
                    ground_truths.append([])

            print("预测结果")

            for i in range(len(ctc_preds)):
                ctc_formula = _format_seq(ctc_preds[i], vocab, max_len=100)
                fusion_formula = _format_seq(fusion_preds[i], vocab, max_len=100)
                crf_formula = _format_seq(crf_preds[i], vocab, max_len=100)
                gt_formula = _format_seq(ground_truths[i] if i < len(ground_truths) else [], vocab, max_len=100)
                
                print(f"\n样本 {i+1}:")
                print(f"  真实标签:             {gt_formula}")
                print(f"  序列流预测 (CTC):     {ctc_formula}")
                print(f"  图流+序列流 (Fusion):  {fusion_formula}")
                print(f"  CRF最终输出:          {crf_formula}")

            
            # 可视化
            visualize_samples(
                images.cpu(),
                ctc_preds,
                fusion_preds,
                crf_preds,
                "results",
                vocab=vocab
            )
            print(f"预测样本已保存到: results/prediction_samples.png")
            break


if __name__ == "__main__":
    test()
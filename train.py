import os
import copy
from typing import List

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import matplotlib.pyplot as plt

try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
except Exception:
    pass

import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from tqdm import tqdm

from dataset.chem_dataset import ChemDataset, collate_fn
from models import DualStreamRecognizer
from torchvision.utils import save_image


def build_graph_batch(batch_infos):
    # 占位：根据标签构造 graph Batch，可用 RDKit 解析
    from torch_geometric.data import Data, Batch
    graphs = []
    for info in batch_infos:
        num_nodes = max(4, info['num_objects'])
        node_types = torch.randint(0, 10, (num_nodes,))
        edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2))
        edge_types = torch.randint(0, 4, (edge_index.size(1),))
        graphs.append(Data(node_types=node_types, edge_index=edge_index, edge_types=edge_types))
    return Batch.from_data_list(graphs)


def compute_token_accuracy(pred_paths: torch.Tensor, target_seqs: List[torch.Tensor]) -> float:
    total_tokens = 0
    correct_tokens = 0

    for i, tgt_seq in enumerate(target_seqs):
        seq_len = tgt_seq.numel()
        if seq_len == 0:
            continue
        if i >= pred_paths.size(0):
            continue
        
        # 对预测进行CTC解码（去重blank和重复字符）
        pred_seq = pred_paths[i].cpu().tolist()
        # CTC解码：去重blank和重复字符
        decoded_pred = []
        prev = 0
        for token in pred_seq:
            if token == 0:
                prev = 0
                continue
            if token != prev:
                decoded_pred.append(token)
                prev = token
        
        # 获取真实序列（去除blank）
        target_list = tgt_seq.cpu().tolist()
        target_clean = [t for t in target_list if t != 0]

        if len(target_clean) == 0:
            continue
        
        # 使用编辑距离计算相似度
        m, n = len(decoded_pred), len(target_clean)
        if m == 0 and n == 0:
            correct_tokens += 1
            total_tokens += 1
            continue
        
        # 计算LCS长度
        lcs_dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i_lcs in range(1, m + 1):
            for j_lcs in range(1, n + 1):
                if decoded_pred[i_lcs-1] == target_clean[j_lcs-1]:
                    lcs_dp[i_lcs][j_lcs] = lcs_dp[i_lcs-1][j_lcs-1] + 1
                else:
                    lcs_dp[i_lcs][j_lcs] = max(lcs_dp[i_lcs-1][j_lcs], lcs_dp[i_lcs][j_lcs-1])
        
        lcs_len = lcs_dp[m][n]

        accuracy = lcs_len / max(m, n) if max(m, n) > 0 else 0.0
        correct_tokens += accuracy * max(m, n)
        total_tokens += max(m, n)
    
    return correct_tokens / total_tokens if total_tokens > 0 else 0.0

# 保存检查点
def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, path: str):
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        },
        path,
    )

# 绘制曲线图
def plot_metrics(epochs, total_losses, ctc_losses, crf_losses, accuracies, out_dir: str, suffix: str = ""):
    """绘制训练指标曲线图"""
    # Loss曲线：CTC + CRF vs CTC
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, total_losses, label="CTC + CRF", linewidth=2)
    plt.plot(epochs, ctc_losses, label="CTC", linewidth=2, linestyle="--")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Training Loss Curve (CTC + CRF vs CTC)", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    loss_path = os.path.join(out_dir, f"loss_curve{suffix}.png")
    plt.savefig(loss_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"损失曲线保存到:{loss_path}")

    # Accuracy曲线
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, accuracies, label="Token Accuracy", color="teal", linewidth=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.ylim(0, 1.0)
    plt.title("Training Accuracy Curve", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    acc_path = os.path.join(out_dir, f"accuracy{suffix}.png")
    plt.savefig(acc_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"精确度曲线已保存到:{acc_path}")


def _normalize_for_vis(img_tensor: torch.Tensor):
    img = img_tensor.detach().cpu().permute(1, 2, 0).numpy()
    img = (img * 0.5 + 0.5).clip(0, 1)
    return img


def _format_seq(seq: List[int]) -> str:
    return " ".join(map(str, seq[:20]))


def generate_prediction_samples(model, dataloader, device, save_path: str, max_samples: int = 8):
    model.eval()
    os.makedirs(save_path, exist_ok=True)
    
    with torch.no_grad():
        for images, boxes_list, infos, _ in dataloader:
            images = images.to(device)
            graph_batch = build_graph_batch(infos).to(device)
            logits, mask, (ctc_paths, crf_paths) = model.decode(images, graph_batch)
            
            # 保存输入图像
            save_image(images, os.path.join(save_path, "inputs.png"))
            
            # 保存预测文本
            with open(os.path.join(save_path, "prediction_samples.txt"), "w", encoding="utf-8") as f:
                for i in range(len(ctc_paths)):
                    f.write(f"Sample {i}\n")
                    f.write(f"Seq-only (CTC): {_format_seq(ctc_paths[i].cpu().tolist())}\n")
                    f.write(f"Fusion logits argmax: {_format_seq(torch.argmax(logits, dim=-1)[i].cpu().tolist())}\n")
                    f.write(f"CRF final: {_format_seq(crf_paths[i].cpu().tolist())}\n\n")
            
            # 生成可视化图片
            max_samples = min(max_samples, images.size(0))
            cols = min(4, max_samples)
            rows = (max_samples + cols - 1) // cols
            plt.figure(figsize=(cols * 3, rows * 3))
            for idx in range(max_samples):
                plt.subplot(rows, cols, idx + 1)
                plt.imshow(_normalize_for_vis(images[idx].cpu()))
                title = (
                    f"CTC:{_format_seq(ctc_paths[idx].cpu().tolist())}\n"
                    f"Fusion:{_format_seq(torch.argmax(logits, dim=-1)[idx].cpu().tolist())}\n"
                    f"CRF:{_format_seq(crf_paths[idx].cpu().tolist())}"
                )
                plt.title(title, fontsize=8)
                plt.axis("off")
            plt.tight_layout()
            pred_path = os.path.join(save_path, "prediction_samples.png")
            plt.savefig(pred_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"将预测样本保存到:{pred_path}")
            break
    
    model.train()  # 恢复训练模式


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用的设备: {device}")
    
    # 加载数据集
    print("加载数据集")
    try:
        dataset = ChemDataset(root_dir="dataset", target_size=(256, 256), augment=True)
        print(f"数据集加载成功。Size: {len(dataset)}")
    except Exception as e:
        print(f"加载数据集时出错:{e}")
        import traceback
        traceback.print_exc()
        return
    
    # 优化batch size以提高准确率：适中的batch size有助于更好的梯度估计
    batch_size = 24 if torch.cuda.is_available() else 12  # 适中的batch size，平衡稳定性和准确率
    # 设置num_workers=0避免多进程导致的输出重复，同时设置persistent_workers=False
    print(f"使用batch_size创建数据加载器={batch_size}")
    try:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0, pin_memory=True if torch.cuda.is_available() else False)
        print(f"数据加载器创建成功。Total batches: {len(dataloader)}")
    except Exception as e:
        print(f"创建数据加载器时出错:{e}")
        import traceback
        traceback.print_exc()
        return

    # 使用实际词汇表大小（93）
    vocab_size = len(dataset.classes)
    # 获取词汇表字典用于CRF初始化化学约束
    vocab_dict = dataset.vocab_dict
    # 使用增强的模型配置
    model = DualStreamRecognizer(
        vocab_size, 
        num_atom_types=100, 
        num_bond_types=5,
        seq_hidden=512,  # 隐藏层维度
        graph_hidden=512,
        fusion_dim=512,
        vocab_dict=vocab_dict  # 传入词汇表字典
    ).to(device)
    
    # 计算模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}, Trainable: {trainable_params:,}")

    optimizer = AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4, betas=(0.9, 0.999), eps=1e-8)
    
    # 训练轮数设置为100
    num_epochs = 100
    warmup_epochs = 10
    
    # Warmup + Cosine Annealing调度器（更平滑的衰减，防止过早衰减）
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Warmup阶段：线性增长
            return (epoch + 1) / warmup_epochs
        else:
            # Cosine衰减阶段：更平滑的衰减，最小学习率更高
            progress = (epoch - warmup_epochs) / (num_epochs - warmup_epochs)
            # 使用cosine衰减，最小学习率为初始学习率的1/5（保持更高学习率）
            return 0.2 + 0.8 * 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = LambdaLR(optimizer, lr_lambda)
    
    # 添加ReduceLROnPlateau作为备用调度器（基于准确率而不是损失）
    from torch.optim.lr_scheduler import ReduceLROnPlateau

    plateau_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.6, patience=6, min_lr=1e-6)

    os.makedirs("results", exist_ok=True)
    total_loss_history = []
    ctc_loss_history = []
    crf_loss_history = []
    accuracy_history = []
    
    # 跟踪最佳模型
    best_accuracy = 0.0
    best_epoch = 0
    best_model_state = None
    best_loss = float('inf')  # 跟踪最佳损失

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_ctc_loss = 0.0
        epoch_crf_loss = 0.0
        epoch_accuracy = 0.0
        steps_with_targets = 0
        
        # 添加调试信息
        print(f"\nStarting Epoch {epoch}/{num_epochs}")

        try:
            # 测试数据加载器是否正常工作
            print(f"  Iterating through {len(dataloader)} batches...")
            for batch_idx, (images, boxes_list, infos, _) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}", ncols=100, leave=False)):
                if batch_idx == 0:
                    print(f"  First batch loaded: images shape={images.shape}")
                
                images = images.to(device)
                
                # 优先使用真实的化学方程式token序列
                cls_seqs_device = []
                for info in infos:
                    if "formula_tokens" in info and info["formula_tokens"] is not None:
                        tokens = info["formula_tokens"].to(device)
                        if tokens.numel() > 0:
                            cls_seqs_device.append(tokens)
                
                # 如果没有真实标签，使用boxes的类别作为fallback
                if not cls_seqs_device:
                    cls_seqs = [boxes[:, 0].long() for boxes in boxes_list]
                    cls_seqs_device = [seq.to(device) for seq in cls_seqs if seq.numel() > 0]
                
                if cls_seqs_device:
                    targets = torch.cat(cls_seqs_device).to(device)
                    target_lengths_list = [seq.numel() for seq in cls_seqs_device]
                    target_lengths = torch.tensor(target_lengths_list, dtype=torch.long, device=device)
                else:
                    targets = torch.zeros((1,), dtype=torch.long, device=device)
                    target_lengths = torch.ones((1,), dtype=torch.long, device=device)
                    target_lengths_list = [0]

                graph_batch = build_graph_batch(infos).to(device)
                optimizer.zero_grad()
                
                # 添加异常处理，避免训练卡住
                try:
                    logits, loss, aux = model(images, graph_batch, targets, target_lengths)
                except Exception as e:
                    print(f"Error in model forward pass at Epoch {epoch}, batch {batch_idx}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
                
                # 检查损失是否为NaN或Inf，提前跳过
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Warning: Epoch {epoch}, batch {batch_idx} detected NaN or Inf loss, skipping backward pass.")
                    continue
                
                try:
                    loss.backward()
                    # 使用梯度裁剪，避免梯度爆炸，帮助稳定训练和提高准确率
                    # 使用适中的梯度裁剪阈值，既保证稳定性又不过度限制学习
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 适中的梯度裁剪
                    optimizer.step()
                except Exception as e:
                    print(f"Error in backward pass at Epoch {epoch}, batch {batch_idx}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
                
                epoch_loss += loss.item()
                epoch_ctc_loss += aux['ctc']
                epoch_crf_loss += aux['crf']

                if cls_seqs_device:
                    pred_paths = torch.argmax(logits.detach(), dim=-1)
                    acc = compute_token_accuracy(pred_paths, cls_seqs_device)
                    epoch_accuracy += acc
                    steps_with_targets += 1
        
        except Exception as e:
            print(f"Error in training loop at Epoch {epoch}: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        scheduler.step()
        avg_loss = epoch_loss / len(dataloader)
        avg_ctc = epoch_ctc_loss / len(dataloader)
        avg_crf = epoch_crf_loss / len(dataloader)
        avg_acc = epoch_accuracy / steps_with_targets if steps_with_targets > 0 else 0.0

        total_loss_history.append(avg_loss)
        ctc_loss_history.append(avg_ctc)
        crf_loss_history.append(avg_crf)
        accuracy_history.append(avg_acc)

        # 使用ReduceLROnPlateau基于准确率调整学习率（更直接地优化准确率）
        plateau_scheduler.step(avg_acc)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch} avg loss: {avg_loss:.4f} ; CTC: {avg_ctc:.4f} ; CRF: {avg_crf:.4f} ; Acc: {avg_acc:.4f} ; LR: {current_lr:.6f}")
        
        # 更新最佳模型（基于准确率）
        if avg_acc > best_accuracy:
            best_accuracy = avg_acc
            best_epoch = epoch
            best_loss = avg_loss
            # 深拷贝模型状态
            best_model_state = copy.deepcopy(model.state_dict())
            print(f"  -> New best accuracy: {best_accuracy:.4f} (loss: {best_loss:.4f}) at epoch {best_epoch}")
        
        # 每10个epoch保存一次checkpoint
        if epoch % 10 == 0:
            save_checkpoint(model, optimizer, epoch, f"results/checkpoint_epoch_{epoch}.pth")
            print(f"保存检查点的批次是 Epoch {epoch}")
        
        # 每10个epoch保存一次图片
        if epoch % 10 == 0:
            current_epochs = list(range(1, epoch + 1))
            plot_metrics(
                epochs=current_epochs,
                total_losses=total_loss_history,
                ctc_losses=ctc_loss_history,
                crf_losses=crf_loss_history,
                accuracies=accuracy_history,
                out_dir="results",
                suffix=""  # 每次都覆盖
            )
            print(f"保存 loss_curve.png 和 accuracy.png 在 Epoch {epoch}")

    # 保存最终模型和图片
    print(f"\n训练完成")
    print(f"Best accuracy: {best_accuracy:.4f} at epoch {best_epoch}")
    
    # 保存最终模型
    torch.save(model.state_dict(), "results/model_final.pth")
    print("最终的模型保存在:results/model_final.pth")
    
    # 保存最佳模型
    if best_model_state is not None:
        torch.save(best_model_state, "results/model_best.pth")
        print(f"最好的模型(Epoch {best_epoch}, acc={best_accuracy:.4f})保存在:results/model_best.pth")
    
    # 保存最终图片
    plot_metrics(
        epochs=list(range(1, num_epochs + 1)),
        total_losses=total_loss_history,
        ctc_losses=ctc_loss_history,
        crf_losses=crf_loss_history,
        accuracies=accuracy_history,
        out_dir="results",
        suffix=""
    )
    print("保存最终的loss_curve.png 和 accuracy.png")
    
    # 生成预测样本图片
    print("\n生成预测样本图片")
    generate_prediction_samples(model, dataloader, device, "results", max_samples=8)
    
    # 保存训练统计信息
    with open("results/training_stats.txt", "w", encoding="utf-8") as f:
        f.write(f"Training Statistics\n")
        f.write(f"{'='*50}\n")
        f.write(f"Total Epochs: {num_epochs}\n")
        f.write(f"Best Epoch: {best_epoch}\n")
        f.write(f"Best Accuracy: {best_accuracy:.4f}\n")
        f.write(f"Final Loss: {total_loss_history[-1]:.4f}\n")
        f.write(f"Final CTC Loss: {ctc_loss_history[-1]:.4f}\n")
        f.write(f"Final CRF Loss: {crf_loss_history[-1]:.4f}\n")
        f.write(f"Final Accuracy: {accuracy_history[-1]:.4f}\n")
    print("保存训练信息在:results/training_stats.txt")


if __name__ == "__main__":
    train()

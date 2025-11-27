# 化学方程式识别系统

基于双流编码器和 CTC + CRF 解码器的化学方程式识别模型。

## 1. 模型结构图

### 1.1 双流编码器结构

```
输入图像
    │
    ├──────────────┬──────────────┐
    │              │              │
    ▼              ▼              ▼
序列编码器      图编码器      双流融合层
MobileNetV3    GATv2        Cross-Attention
    ↓              ↓              ↓
双向GRU        基团识别        融合特征
    ↓              ↓              ↓
序列特征        图特征      (B × T × 512)
(B × T × 1024) (B × 512)
```

### 1.2 CTC + CRF 解码器结构图

```
融合特征
    │
    ▼
分类器 → Logits
    │
    ├──────────────┬──────────────┐
    │              │              │
    ▼              ▼              ▼
CTC解码        CRF解码        最终输出
Log Softmax    化学约束CRF    
CTC Loss       解码           预测序列
```

## 2. 训练损失曲线

损失曲线保存在 `results/loss_curve.png`，展示 CTC + CRF 联合训练相比单独使用 CTC 的优势。

![损失曲线](results/loss_curve.png)

- **CTC + CRF 损失**: 联合训练的总体损失
- **CTC 损失**: 仅CTC部分的损失

## 3. 测试示例

测试结果保存在 `results/prediction_samples.png`，包含以下内容：

![测试样本](results/prediction_samples.png)

### 输入图像
- 原始化学方程式图像

![原始图片](results/inputs.png)

### 序列流预测结果 (CTC); 图流 + 序列流预测结果 (Fusion); CRF 后最终输出

![图片.png](%E5%9B%BE%E7%89%87.png)


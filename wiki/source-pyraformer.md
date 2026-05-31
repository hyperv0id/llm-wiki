---
title: "Pyraformer: Low-Complexity Pyramidal Attention for Long-Range Time Series Modeling and Forecasting"
type: source-summary
tags:
  - time-series
  - forecasting
  - transformer
  - efficient-attention
  - multi-scale
  - pyramidal-attention
  - ICLR-2022
created: 2026-05-31
last_updated: 2026-05-31
source_count: 1
confidence: medium
status: active
---

# Pyraformer: Low-Complexity Pyramidal Attention for Long-Range Time Series Modeling and Forecasting

**Authors**: Shizhan Liu (Ant Group, Shanghai Jiaotong Univ.), Hang Yu (Ant Group), Cong Liao (Ant Group), Jianguo Li (Ant Group), Weiyao Lin (Shanghai Jiaotong Univ.), Alex X. Liu (Ant Group), Schahram Dustdar (TU Wien)

**Venue**: ICLR 2022 (Oral) | **Code**: [github.com/alipay/Pyraformer](https://github.com/alipay/Pyraformer) | **Citations**: ~934

## 核心贡献

Pyraformer 提出基于金字塔注意力的 Transformer 模型，同时捕获短程和长程时间依赖，实现 O(L) 时间和空间复杂度与 O(1) 最大信号传播路径长度。两个核心模块：**PAM**（Pyramidal Attention Module，金字塔注意力模块）在跨尺度树结构中进行消息传递，**CSCM**（Coarser-Scale Construction Module，粗尺度构建模块）通过 bottleneck 卷积构建多分辨率 C-叉树 [^src-pyraformer]。

## 问题定位

现有方法在捕捉长期依赖和保持低复杂度之间无法兼得。RNN/CNN 复杂度为 O(L)，但最大信号传播路径也是 O(L)，难以学习远距离依赖。原始 Transformer 将最大路径缩短至 O(1)，但复杂度变为 O(L^2) [^src-pyraformer]。稀疏注意力变体（LogTrans, Informer, Longformer, Reformer）试图折中，但多数无法同时达到 O(1) 路径长度和 O(L) 复杂度 [^src-pyraformer]。

## 方法

### 金字塔图结构

Pyraformer 将时间序列建模为金字塔图（Figure 1d），边分为两组：

- **跨尺度连接（inter-scale）**：构建 C-叉树，每个父节点有 C 个子节点，形成从细粒度（如小时观测）到粗粒度（如日/周/月特征）的多分辨率表示
- **尺度内连接（intra-scale）**：每个节点关注同尺度的相邻 A 个节点，捕获该分辨率下的短期依赖

根本上，粗尺度节点以图方式更简洁地描述远程依赖——比仅用单一最细尺度建模高效得多 [^src-pyraformer]。

### PAM（金字塔注意力模块）

(s) (s) (s) (s)

每个节点 n_l 的注意力邻居集合 N_l = A_l ∪ C_l ∪ P_l ，包含同尺度 A 个相邻节点、C 个子节点、1 个父节点。通过叠加 N 层 PAM，最粗尺度节点获得全局感受野。Lemma 1 给出充分条件：L/C^(S-1) - 1 ≤ (A-1)N/2 [^src-pyraformer]。

Proposition 1 证明：当 A 为常数时，复杂度上界为 O(AL)=O(L)。Proposition 2 证明：当 S 固定、C 满足特定条件时，最大路径长度为 O(S+N)=O(1) [^src-pyraformer]。

由于现有深度学习库不支持稀疏注意力，作者使用 TVM 实现定制 CUDA kernel，将计算时间和内存消耗实质性降低 [^src-pyraformer]。

### CSCM（粗尺度构建模块）

CSCM 自底向上逐尺度引入粗尺度节点：对嵌入序列在时间维度上施加 kernel size=C、stride=C 的 bottleneck 卷积（先降维再卷积后升维），逐层产出长度为 L/C^s 的序列，串联成 C-叉树输入 PAM [^src-pyraformer]。

### 预测模块

单步预测：在历史序列末追加 end token（z_{t+1}=0），编码后收集各尺度最后节点，拼接经全连接层预测。

多步预测方案一：同单步，但 FC 映射到全部 M 个未来时间点。方案二：利用两个 full attention 层的 decoder，预测 token F_p 为 query，encoder 输出 F_e 为 key/value [^src-pyraformer]。

## 实验结果

### 单步预测

在 Electricity, Wind, App Flow 三数据集上，Pyraformer 以最少 Q-K pairs 取得最优 NRMSE 和 ND。Q-K pairs 比 LogTrans 少 65.4%，比 full attention 少 96.6%。Wind 数据集上稀疏注意力机制优于 full attention，因为数据含大量零值，适当稀疏化有助于防过拟合 [^src-pyraformer]。

### 长期多步预测

在 ETTh1 上，相比 Informer，Pyraformer 的 MSE 在预测长度 168, 336, 720 时分别降低 24.8%, 28.9%, 26.2%。有趣的是，方案一（单 FC 层）优于方案二（decoder），可能因为基于 full attention 的 decoder 无法区分不同分辨率特征，而 FC 层可自动利用多分辨率特征 [^src-pyraformer]。

合成数据实验（多段正弦函数 + 长程相关高斯过程）显示 Pyraformer 大幅领先，且根据已知周期设置不同尺度 C 值可进一步提升性能 [^src-pyraformer]。

### 速度与内存

TVM 实现下，时间和内存消耗近线性于 L。12GB Titan Xp GPU 上：序列长度 5800 时 full attention OOM，Pyraformer 仅占 1GB；20000 时 Informer OOM，Pyraformer 仅 1.91GB，每 batch 0.082s [^src-pyraformer]。

### 消融实验

关键发现：A 应固定为小常数（3 或 5），C 随 L 增大；bottleneck 卷积 CSCM 比 max/avg pooling 更优（仅 +1.51% MSE，省 90% 参数）；更长历史提升精度，但增益在历史提供足够周期信息后趋于饱和；PAM 对准确预测至关重要（移除后性能大幅下降）[^src-pyraformer]。

## 局限性

仅考虑 A 和 S 固定、C 随 L 增长的配置模式。未来方向：自适应学习超参数，扩展到 NLP 和 CV 领域 [^src-pyraformer]。CSCM 引入了额外参数（约 5% overhead），并依赖于 TVM 定制 CUDA kernel，增加了部署复杂度。

[^src-pyraformer]: [[source-pyraformer]]

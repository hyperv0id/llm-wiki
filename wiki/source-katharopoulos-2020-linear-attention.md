---
title: "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention"
type: source-summary
tags:
  - linear-attention
  - kernel-methods
  - rnn
  - efficiency
  - icml-2020
created: 2026-10-27
last_updated: 2026-10-27
source_count: 1
confidence: medium
status: active
---

# Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention

Katharopoulos、Vyas、Pappas、Fleuret（Idiap Research Institute / EPFL，Pappas 兼属 University of Washington、Fleuret 兼属 University of Geneva），ICML 2020（PMLR 119），arXiv:2006.16236。通读版本为 v3（2020-08-31，camera-ready，补充 linear-transformers.com 代码链接），并比对 v1（2020-06-29）确认正文一致，仅作者栏与出版信息不同；PDF `raw/transformers-are-rnns-2006-16236.pdf`。

## 核心论点

softmax 注意力随序列长度呈 $O(N^2)$ 的时间与显存消耗，是 Transformer 扩展上下文的瓶颈。论文将其表述为广义形式，仅要求相似度函数非负（涵盖所有正定核），即可用特征映射 $\phi$ 把 $\operatorname{sim}(q,k)$ 分解为 $\phi(q)^{\top}\phi(k)$，再借矩阵乘法结合律先算 $\phi(K)^{\top}V$，把复杂度降到 $O(N)$（式 3–6）[^src-katharopoulos-2020-linear-attention]。实验采用 $\phi(x)=\operatorname{elu}(x)+1$，选 elu 而非 relu 以保持负区梯度非零（式 7）[^src-katharopoulos-2020-linear-attention]。

第二项贡献是因果掩码下的线性化：分母与分子均为前缀和 $S_i$、$Z_i$，可通过 $S_{i-1}$、$Z_{i-1}$ 在常数时间内递推，使每个因果 Transformer 层等价于带两个隐状态（注意力记忆 $s$、归一化记忆 $z$）的 RNN（式 16–20）[^src-katharopoulos-2020-linear-attention]。这种递归作用于时间维而不涉及深度（区别于 Universal Transformers）[^src-katharopoulos-2020-linear-attention]。训练梯度借助前向与反向累积和在线性时间内算得（附录 A，约 200 行 CUDA）[^src-katharopoulos-2020-linear-attention]。

## 证据

序列复制任务中，linear 收敛稳定，最终损失与 softmax 持平、低于 lsh（无哈希噪声）（Fig 2）[^src-katharopoulos-2020-linear-attention]。MNIST 自回归图像生成达到 0.644 bits/dim（softmax 为 0.621），吞吐 142.8 对 0.45 images/s（317×）（Table 1）[^src-katharopoulos-2020-linear-attention]。CIFAR-10 训练 7 天达到 3.40 bits/dim、17.85 images/s（4,462×），linear 完成的 epoch 约为 softmax 3 倍（Table 2）[^src-katharopoulos-2020-linear-attention]。WSJ 语音识别（CTC）PER 为 8.08、824 s/epoch，优于 Bi-LSTM（10.94）与 LSH-4（9.33），不及 softmax 的 5.12（Table 3）[^src-katharopoulos-2020-linear-attention]。

## 边界

语音任务上 softmax 质量明显更高（附录 B 显示收敛更好），MNIST 上 bits/dim 略差于 softmax。$\phi$ 凭经验选取，特征映射设计（如用 RBF 随机傅里叶特征复用 softmax 预训练权重）列为后续方向（§5）。由于指数核的精确特征映射为无限维，精确 softmax 线性化不可行（§3.2.1）[^src-katharopoulos-2020-linear-attention]。

[^src-katharopoulos-2020-linear-attention]: [[source-katharopoulos-2020-linear-attention]]

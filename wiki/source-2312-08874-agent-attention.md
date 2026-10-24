---
title: "Source: Agent Attention — On the Integration of Softmax and Linear Attention (arXiv:2312.08874)"
type: source-summary
tags:
  - source
  - attention
  - efficient-attention
  - linear-attention
  - vision
created: 2026-10-24
last_updated: 2026-10-24
source_count: 1
confidence: medium
status: active
---

# Source: Agent Attention（arXiv:2312.08874）

- **作者**：Dongchen Han, Tianzhu Ye, Yizeng Han, Zhuofan Xia, Shiji Song, Gao Huang（清华大学自动化系）；Siyuan Pan, Pengfei Wan（快手）
- **出版**：ECCV 2024；arXiv:2312.08874v3（2024-07-15）
- **链接**：https://arxiv.org/abs/2312.08874
- **本地**：`raw/2312.08874-agent-attention.txt`（pymupdf 提取的 PDF 文本，30 页，3098 行）
- **代码**：https://github.com/LeapLabTHU/Agent-Attention

## 核心论点

标准注意力在 $(Q,K,V)$ 之外加入 $n$ 个 agent token $A$，构成四元组 $(Q,A,K,V)$：$A$ 先作 query 从 $K,V$ 聚合，再作 key 广播给全部 query，两段均为 softmax（式 3）[^src-2312-08874-agent-attention]。$A$ 兼作 query 与 key，$Q$ 与 $K$ 之间不再有两两相似度，复杂度从 $\mathcal{O}(N^2)$ 降到 $\mathcal{O}(Nnd)$（式 7）。

论文的主要论点是式 4：$\sigma(QA^{\top})\sigma(AK^{\top})V$ 等于以 $\phi_q(Q)=\sigma(QA^{\top})$、$\phi_k(K)=\sigma(AK^{\top})^{\top}$ 为映射函数的核化线性注意力，即「广义线性注意力」[^src-2312-08874-agent-attention]。该等式把 softmax 与 linear attention 写成同一算子的两种读法，而非在两个范式之间取舍；$n$ 不改变等价关系，只改变 $\phi$ 的参数化与计算量。

模块层面另加两项（式 6）：agent bias $B_1\in\mathbb{R}^{n\times N}$、$B_2\in\mathbb{R}^{N\times n}$，由列／行／块三个分量构成（Appendix B）；沿用 Flatten Transformer 的 DWC 补特征多样性。默认 $A=\operatorname{Pooling}(Q)$。

## 证据

- ImageNet-1K（Table 1、Table 13）：Agent-DeiT-T 74.9 vs 72.2，Agent-PVT-S 82.2 vs 79.8（参数量 20.6M vs 24.5M），Agent-Swin-T 82.6 vs 81.3，Agent-CSwin-B 84.7 vs 84.2。
- COCO 检测（Table 2）：Agent-PVT-L box AP 46.9 vs 42.9；ADE20K 分割（Table 3）：Agent-PVT-T mIoU 40.18 vs 36.57。
- 速度（Fig. 5，$224^2$）：i5 CPU 1.7–2.1 倍，RTX3090 与 A100 1.4–1.7 倍；$1024^2$ 下 Agent-DeiT 省 75% FLOPs（Fig. 7b）。
- Stable Diffusion（Table 9）：AgentSD 相对 SD 提速 1.84 倍、相对 ToMeSD 提速 1.69 倍，FID 由 28.84 降到 28.42–27.77，无额外训练。

## 局限

实验限于视觉任务，无语言建模与长序列实测；agent token 数只在 Agent-Swin-T 上消融；与 GPViT、GRL 的区别只有定性论证（Appendix A）；COCO 训练用 80k/35k 子集而非完整划分（Appendix D.2）[^src-2312-08874-agent-attention]。

机制推导、分阶段数据流与逐表数值见 [[agent-attention]]；与 MiTA 共享专家的对照见该页末节。

[^src-2312-08874-agent-attention]: [[source-2312-08874-agent-attention]]

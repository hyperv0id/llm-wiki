---
title: "TFPS: Learning Pattern-Specific Experts for Time Series Forecasting Under Patch-level Distribution Shift (Sun et al., NeurIPS 2025)"
type: source-summary
tags:
  - time-series-forecasting
  - mixture-of-experts
  - distribution-shift
  - subspace-clustering
  - patch-wise
  - neurips-2025
created: 2026-09-17
last_updated: 2026-09-17
source_count: 1
confidence: medium
status: active
---

# TFPS: Learning Pattern-Specific Experts for Time Series Forecasting Under Patch-level Distribution Shift

**Authors**: Yanru Sun, Zongxia Xie, Emadeldeen Eldele, Dongyue Chen, Qinghua Hu, Min Wu（天津大学 + 新加坡 A*STAR I2R + Khalifa University）。[^src-tfps]
**Venue**: NeurIPS 2025（缓存中含 neurips.cc/Conferences/2025 的 camera-ready 页眉）。[^src-tfps]
**证据源**: `/run/media/jcheng/WD-Data/yjs/Zotero/storage/QXFJGD83/.zotero-ft-cache`（PDF 全文纯文本缓存）

## 问题

真实序列在不同 patch（分段）上呈现异质模式演化（季节、机制切换、上下文漂移）；单一模型难以适配。归一化类方法（RevIN、SAN、Dish-TS 等）只处理统计特性变化，且过度平稳化会把有意义的时变信息抹平。[^src-tfps]

## 方法

TFPS 三模块，MoE 插在**每条编码分支（时间域 / 频率域）的 encoder 之后**，按 patch 级路由，频域分支不修改 Transformer 的 feed-forward 层：[^src-tfps]

1. **DDE 双域编码（§3.4，Eq 1-2）**：patch 化 + 可学习位置嵌入后，时间域走多头自注意力 encoder，频率域把自注意力子层换成 Fourier 子层（2D FFT，仅取实部），分别输出 z_t、z_f∈R^{C×N×D}。[^src-tfps]
2. **Pattern Identifier（§3.5，Eq 3-9）**：子空间聚类路由。K 个子空间基 D^(j)∈R^{q×d}（q=C×D，d=q/K，高斯随机初始化），R1（Eq 3）约束列范数、R2（Eq 4）保证子空间互异，R=α(R1+R2)，α=10⁻³；亲和度 s_ij（Eq 6）给出 z_i 归属第 j 子空间的概率（平滑参数 η 取与 d 同值），细化 ŝ_ij（Eq 7）加权高置信分配，KL 散度损失 L_sub（Eq 8）自监督聚类；L_PI = R + β·L_sub（Eq 9）。[^src-tfps]
3. **MoPE（§3.6，Eq 10-11）**：门控 G(s)=Softmax(TopK(s))，**路由输入信号是 PI 的聚类亲和度 s 而非学习到的门控网络**；K 个 expert 均为两层线性 + ReLU 的 MLP，输出按门控加权和 h=Σ_k G(s)·E_k(z)。**top-k 的 k 取值在正文中未给出**；Kt、Kf 仅在附录 H 网格搜索 {1,2,4,8}，Figure 10 呈现过 4-expert 的分配分布。时间/频率两分支各自独立配备 PI+MoPE，输出 concat 后线性投影得预测。[^src-tfps]
4. **损失（§3.7，Eq 12）**：L = L_MSE + L_PI^t + L_PI^f。**无负载均衡损失**；聚类正则与均衡互不相关，paper 未讨论 expert 负载问题。D=512、encoder 层数 n=2（Algorithm 1）。[^src-tfps]

## 实验

- **Table 1（主表）**：论文自称 top-1 性能占 57/72 个设定（9 数据集 × 4 horizon × 2 指标）；基线含 PatchTST、DLinear、TimesNet、iTransformer、FEDformer、FITS、TFDNet-IK、TSLANet。并非全胜：ETTh1-96 上 TFPS MSE 0.398 低于 TSLANet 0.387（文中记 −1.1%），Traffic-720 为 −3.6%。[^src-tfps]
- **附录 G.2（Table 10，MoE 方法对比）**：论文自称 unlike 依赖 Softmax 门控的 MoE 方法，由 pattern recognizer 分配 expert；相对 MoLE、MoU、KAN4TSF 分别提升 2.3%、9.0%、10.6%、9.1%。G.3（Table 11）：相对 Koopa、SOLID、OneNet 提升 6.7%、6.6%、4.8%、5.9%。[^src-tfps]
- **Table 12 / Appendix I（效率）**：Table 12 记 TFPS 显存 9.643 MB、平均推理 6.114 ms（PatchTST 4.861 ms、TimesNet 12.306 ms、DLinear 0.659 ms、FEDformer 136.130 ms）；但 Appendix I 正文写 TFPS 运行 6.457 ms、比 PatchTST（17.851 ms）快 2.8×、比 TimesNet（72.196 ms）快 11.2×，DLinear 记 0.789 ms，与 Table 12 并不一致。[^src-tfps]
- **Tables 14/15/16（消融，ETTh1/ETTh2，96-720）**：Table 14 去掉任一域 PI 均变差（ETTh1-96：全 PI 0.398，仅时间 PI 0.404，仅频率 PI 0.405，全无 0.407）；Table 15 无 R1/R2 时 ETTh1-96 由 0.398 恶化到 0.412；Table 16 用多输出预测器（0.403）或堆叠注意力层（0.399）替换 MoPE 均不如 TFPS 0.398。[^src-tfps]

## 局限

Appendix N 自述：patch 长度靠启发式选择，难以处理不可整除长度或多周期序列，现实泛化性不足；新模式会随时间涌现（如未预见的疫情爆发），当前设计不具备可扩展解决方案，未来工作拟做更灵活的自动 patch 长度选择与演化分布偏移的扩展机制。[^src-tfps]

## 与其它 MoE 方法的差异

论文自述（§G.2）：MoLE、MoU、KAN4TSF 等 MoE 方法以 Softmax 作为门控机制；TFPS 用子空间聚类式 pattern recognizer 依据 patch 的分布特征分配 expert。实验层面 Table 10 与 MoLE/MoU/KAN4TSF 对比并声称占优，但未与本文同代的时序分类 MoE（如 SoftShape）比较。[^src-tfps]

## 正文与附录的不一致（如实记录）

- **数据集数量 9 vs 8**：摘要与 §4.1 均称 9 个数据集（含 Traffic），§5 结论写 "eight diverse datasets"，Appendix A 也写 "eight widely used datasets" 且列举中不含 Traffic。
- **运行时间 6.114 vs 6.457 ms**：Table 12 记 6.114 ms，Appendix I 正文记 6.457 ms；同时 PatchTST（4.861 vs 17.851 ms）、TimesNet（12.306 vs 72.196 ms）、DLinear（0.659 vs 0.789 ms）的对照值也不一致。
- **top-k 的 k 取值在正文中未给出**，且 Eq 11 聚合公式写作对全部 K 个 expert 求和，与 Eq 10 的 TopK 语义未完全对齐。

[^src-tfps]: [[source-tfps]]

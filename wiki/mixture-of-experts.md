---
title: "Mixture of Experts (MoE)"
type: concept
tags:
  - machine-learning
  - architecture
  - scalability
created: 2026-04-29
last_updated: 2026-10-27
source_count: 16
references:
  - [[source-fast-long-horizon-forecasting]]
  - [[source-2602-01219-mita]]
confidence: high
status: active
---

# Mixture of Experts (MoE)

## 定义

Mixture of Experts（MoE）是一种神经网络架构设计，通过多个专门的子网络（称为"experts"）来处理输入数据的不同部分，并使用一个**门控网络（gating network）**动态地将输入路由到最相关的 experts[^src-fast-long-horizon-forecasting]。

## 核心组件

### 1. Experts
- 多个独立的子网络，每个 expert 可以专门处理输入的不同方面
- 在 FaST 中，使用 **Gated Linear Units (GLU)** 作为 experts，相比传统 FFN 有更好的并行性

### 2. Gating Network
- 动态计算每个 expert 的权重
- 根据输入特征决定哪些 experts 被激活
- 关键挑战：**Expert 极化**（expert polarization）— 少量 experts 主导路由决策

## Dense MoE vs Sparse MoE

FaST 使用的是 **Dense MoE**，这与常见的 Sparse MoE（只激活 top-k experts）不同：

- **Dense MoE**: 所有 e 个 experts 都被激活，输出是所有 experts 的加权和
  - 计算：$\sum_{i=1}^{e} G_\ell[:,i] \otimes \mathrm{Exp}_i(Z_t^{\ell})$
  - FaST 通过 GLU 并行化实现高效计算
  
- **Sparse MoE**: 只激活 top-k experts（如 Switch Transformer），减少推理计算量但增加实现复杂度

FaST 选择 Dense MoE 的原因：GLU experts 的并行计算可以在单个 GPU 上高效完成，避免了跨设备分配 experts 的额外开销。

## 异质性感知路由（HA-Router）

在 FaST 中提出的 HA-Router 通过以下方式避免 expert 极化：

- **吸收原始时间序列模式**：使用原始输入 $X_t$ 计算 expert 分数，而非仅使用当前层特征
- **注入自适应空间和时间 expert bias**：$R_{S,\ell}$（空间位置）、$R_{t,\ell}^{T}$（日内时间）和 $R_{t,\ell}^{W}$（周内时间）

公式：

$$
G_{\ell} = \mathrm{softmax}\left(g_{\ell}(X_t) \oplus R_{S,\ell} \oplus R_{t,\ell}^{T} \oplus R_{t,\ell}^{W}\right) \in \mathbb{R}^{N \times e}
$$

这使得节点能够选择与自身相关的 experts，而不是集中在单一 expert 上。

## 优势

- **并行计算效率**：GLU experts 可以在单个计算单元上并行处理，提升 1.4x 推理速度[^src-fast-long-horizon-forecasting]
- **异质性建模**：不同节点和时间段可以使用不同的 experts
- **避免 expert 极化**：HA-Router 通过异质性感知避免负载不平衡

## 在时空图预测中的应用

FaST 首次将 MoE 应用于大规模长视野时空图预测：
- 使用 HA-MoE 进行时间压缩输入
- 在 backbone 中用 HA-MoE 替换 FFN
- 实现 O(N·e·d) 空间复杂度（线性于节点数）

[[most|MoST]] (KDD 2026) 将 MoE 应用于空间依赖建模，提出 [[multi-modality-guided-spatial-expert|多模态引导空间专家]]：使用两类专家——模态共享专家（每种激活模态一个）和路由专家（由路由器基于模态融合嵌入选择）——来捕获区域特定的局部空间模式[^src-most]。每个专家通过交叉注意力建模传感器与其 top-k 最近邻的交互，而非全图关系[^src-most]。与 FaST 的 Dense MoE 不同，MoST 使用 Top-1 稀疏路由并引入负载均衡损失防止专家坍塌[^src-most]。

## 在时间序列基础模型中的应用

[[time-moe|Time-MoE]] (ICLR 2025) 是首个将 Sparse MoE 引入时序基础模型预训练的框架（论文自述），首次将时序模型推到 2.4B 参数规模[^src-time-moe]：

- **架构**：decoder-only Transformer，每层 FFN 替换为 MoE（N 个独立专家 + 1 个共享专家），Top-K 稀疏激活 + 辅助负载均衡损失[^src-time-moe]
- **门控**：标准线性门控 + Softmax Top-K（与 Switch Transformer 一致），通过辅助均衡损失（$f_i \times r_i$）防止路由坍塌——移除该损失导致专家坍塌为更小 FFN，性能从 0.262→0.275[^src-time-moe]
- **Token 化**：逐点（point-wise）SwiGLU 嵌入，保留全部时序精度，区别于 patch token 化[^src-time-moe]
- **多分辨率预测**：4 个输出头（horizon {1,8,32,64}），多任务联合优化 + 贪心调度组合实现灵活预测长度[^src-time-moe]
- **效果**：Time-MoEultra (1.1B 激活/2.4B 总参) 零样本平均 MSE 降低 20%+ vs Moirai/TimesFM/Chronos，训练成本比等激活 Dense 模型降 78%、推理降 39%[^src-time-moe]
- **缩放定律验证**：随模型规模和数据量增长，性能持续提升——论文报告其实证验证了时序领域的缩放定律[^src-time-moe]

[[moirai-moe|Moirai-MoE]] (ICML 2025) 随后进一步改进了门控函数设计[^src-moirai-moe]：

- **架构**：在 6/12 层 decoder-only Transformer 中，每层 FFN 替换为 $M=32$ 个 experts 的 MoE 层，每个 token 仅激活 $K=2$ 个 experts
- **簇基门控**：提出 [[cluster-based-gating|新型门控函数]]——用预训练 dense 模型的 token 嵌入 k-means 聚类中心引导 expert 分配，在所有专家数量配置下一致优于随机初始化线性门控[^src-moirai-moe]
- **与 Time-MoE 的差异**：Moirai-MoE 使用 patch token 化（更大感受野、更快推理）+ 簇基门控（数据驱动路由），而 Time-MoE 使用逐点 token 化 + 线性门控 + 辅助均衡损失[^src-moirai-moe]。Moirai-MoE 通过 [[token-level-specialization|token 级专业化]] 替代频率级人工数据分组[^src-moirai-moe]
- **效果**：Moirai-MoE-S (11M activated, 117M total) 以 17% Monash 聚合 MAE 提升超越 dense Moirai-S，以 65× 更少激活参数优于 Chronos-L[^src-moirai-moe]
- **效率**：推理时间 273s vs Chronos-S 551s（受益于 patch size 16 vs Chronos 的 point-wise tokenization）[^src-moirai-moe]

> [!note] MoE 时序基础模型的演进
> Time-MoE (ICLR 2025) 首次将标准 Sparse MoE + 辅助均衡损失引入时序预训练，验证了缩放定律和效率优势。Moirai-MoE (ICML 2025) 随后通过簇基门控和 patch token 化进一步改进了零样本性能。两者共同确立了稀疏 MoE 作为时序基础模型缩放的核心范式。

## 在动力系统重建中的应用

[[dynamix|DynaMix]] (NeurIPS 2025) 是首个将 MoE 用于动力系统重建（DSR）零样本泛化的模型[^src-dynamix]：

- **架构**：$J=10$ 个 AL-RNN 专家 + 门控网络（CNN 上下文编码器 + 状态注意力机制 + MLP）
- **Dense MoE**：所有 10 个专家均被激活，通过 softmax 加权输出——与 FaST 的 Dense MoE 一致
- **状态注意力门控**：基于投影潜在状态与上下文观测之间的距离计算注意力权重 $w_t^{att}$，再经 CNN 特征加权后由 MLP 生成专家权重 $w_t^{exp}$
- **动力学专业化**：不同专家自然专业化到不同的动力学模式（如混沌区域的不同部分），专家使用权重可构建动力学相似度度量
- **关键发现**：$J \ge 5$ 个专家即足够，更多收益递减——实际 $J=10$ 已达到性能平台期[^src-dynamix]

## 在轨迹预测中的应用（Motion MoE）

[[trajar|TrajAR]] 在路口多使用者轨迹预测编码器里把专家拆成两类：**indicator experts** 按使用者类型、信号状态的 one-hot 指标直接选中（无门控），**route experts**（8 个）按运动趋势表征与专家质心的 Sigmoid 亲和度做 top-$K_r$ 路由，输出为趋势表征上的残差叠加[^src-trajar]。论文给出的分工理由是：前者处理离散属性，后者建模连续运动趋势[^src-trajar]。消融表（SinD-Tianjin ADE/FDE）里只去掉 Motion MoE 后从 0.409/0.798 升到 0.892/1.642，去掉 GAT & ESA 为 0.882/1.621，编码器整体换全连接为 1.231/2.542，打乱 route 路由为 0.736/1.436[^src-trajar]。机制与证据详见 [[motion-moe]]。

这一形态与上文几个时序/时空 MoE 的区别在于：专家不是在时间 patch、频率或空间拓扑上分工，而是绑定「离散语义属性」与「连续运动模式」两种输入类型[^src-trajar]。

## 相关工作
- [[mage|MAGE]] (NeurIPS 2025) 在**图结构生成层面**引入 sparse-balanced MoE——每个专家对应一种独特的空间拓扑假设（自适应图），每节点 Top-K 稀疏激活 + 符号 SGD 负载均衡 $\beta k$。与标准 MoE（如 Mixtral 的 FFN 级路由）不同，MAGE 的 MoE 作用于图学习而非特征变换，用差分图 $A^{(k)}=\mathrm{Softmax}(E_1)\mathrm{Softmax}(E_2^{\top})-\lambda\,\mathrm{Softmax}(E_3)\mathrm{Softmax}(E_4^{\top})$ 增强多样性[^src-mage]。参见 [[sparse-balanced-mixture-of-experts-st]] 了解其稀疏平衡机制的技术细节。

- [[stamimputer|STAMImputer]] (arXiv 2025) 将 MoE 上移到框架外层，用观测专家 (O-Expert) 依据稀疏度特征动态路由时间/空间注意力专家，号称首次把 MoE 用于交通数据填补[^src-stamimputer]。

## MMoE（TiMi）

[[timi|TiMi]] **[[mmoe|MMoE]]**：TMoE 文本门控 + SMoE 序列门控；消融优于标准 MoE / Cross-Attention。换 FFN；PatchTST+MMoE 平均 MSE −18.2%。标签 [[non-fusion-guidance|Non-Fusion Guidance]][^src-timi]。

## 事后模型混合中的 MoE（FusionBench）

[[fusionbench|FusionBench]]（JMLR 2025）把 **WE-MoE / SMILE** 等归为 [[deep-model-fusion|深度模型融合]] 的 *model mixing*：输入是多个*已微调*同构检查点，输出常为扩参或稀疏专家结构，而不是在预训练阶段从零学路由[^src-jmlr-25-1243]。

- **与训时 MoE 的分界**：Time-MoE / Moirai-MoE / FaST / TiMi MMoE 在训练（或微调）中学 gating；WE-MoE/SMILE 是多专家检查点的*后置装配*，可配合测试时适配[^src-jmlr-25-1243]。
- **经验量级**（CLIP-ViT-B/32 八任务 AVG）：WEMoE 89.2 / SMILE 89.3，接近传统 MTL 88.6 与 STL 上界 90.3，明显高于纯 [[model-merging|参数合并]] 的平均/Ties 档[^src-jmlr-25-1243]。
- **代价**：mixing 通常扩大模型或增加适配成本；资源紧时应先看无参合并与 RegMean，而非默认升 MoE[^src-jmlr-25-1243]。

详见 [[fusionbench]]、[[deep-model-fusion]]、[[source-jmlr-25-1243]]。

## MoE 原则用于 attention

MoBA（Moonshot AI，2025）把 top-k 门控的对象从 FFN 专家换成 KV 上下文块：query 与块内 key 均值的内积作亲和分数，选 top-k 块做 attention；当前块强制选择并类比 shared expert；不引入新参数，top-k 全选时退化为 full attention。568M–2.1B 缩放实验中验证 loss 与 full attention 相差 1e-3 以内[^src-moba-mixture-of-block-attention]。机制与证据详见 [[moba-attention]]。

与慢权重 MoE 的关键差别在专家是什么：这里的专家是 key-value 对的一个子集，随输入生成，属 fast weights；MoE attention 因此被读成「把宽度为 $N$ 的 fast-weight MLP 拆成专家再路由」[^src-2602-01219-mita]。MiTA 在这一点上再改一步：专家不再是连续定长块，而是由 landmark query 对全 KV 打分后取 top-$k$ 聚成的形变专家，$m$ 个专家组 $N$-to-$m$ 路由，另加一个常开的共享专家承担全局概括[^src-2602-01219-mita]。完整谱系见 [[fast-weight-scaling-taxonomy]]，机制与数值见 [[mita-attention]]、[[deformable-fast-weight-experts]]。

[[native-sparse-attention|NSA]] 则把门控放在支路合并处：压缩、块选择、滑动窗口三条支路各自 softmax，再由 sigmoid 门控加权求和，三个门控值互不归一（式 5）；块选择本身在支路内部按复用自压缩支路的注意力分数取 top-$n$，不引入独立打分网络（式 8、§3.3）[^src-2502-11089-nsa]。

## LLM 侧的 ultra-sparse 形态（Qwen3-Next）

Qwen3-Next（阿里 Qwen，2025-09）把稀疏度推到 LLM 前沿：512 个 routed 专家每 token 只激活 10 个、另加 1 个 shared expert（Qwen3 为 128 选 8），80B 总参激活约 3B、占 3.7%[^src-qwen3-next-blog-2025-09]。博客称在 global load balancing（其引文 [4]）下固定激活专家数、增大专家总量可持续降低训练损失[^src-qwen3-next-blog-2025-09]。与上文时序/时空 MoE 的差异在量级：专家数与稀疏率都高一个数量级，均衡仍走负载均衡损失一系；架构全貌见 [[qwen3-next-hybrid]]。

## 反向视角：dense ResNet 的功能层隐性路由

上面所有条目都是**先造专家、再学 router**。Yan et al.（arXiv:2610.02907，2026）反过来问：已训练好的标准 ResNet 里是否已经存在隐式专家结构。做法是把残差支路开关当作博弈参与者，对 $2^L$ 个掩码做 Möbius 反演，得到 shortcut 基线 + 一阶修正 + 高阶 interaction 的精确加性分解；高阶项按幅值排序即为 per-input 专家集[^src-resnets-route-without-routers]。

- **与稀疏 MoE 的差别**：没有 router、没有跳过计算，每个块都执行；「选择」只体现在功能贡献的重新分配上，论文称之为 implicit soft routing，并明确不与 sparse execution 混同[^src-resnets-route-without-routers]。
- **量级**：ImageNet ResNet-18（$L=8$）与 ResNet-34（$L=16$）的平方系数质量峰值出现在 $k=5$ 与 $k=10$；ResNet-18 保留 25% 字典得 93.64% 向量质量但只有 30.48% top-1 一致率，即质量集中而决策不集中[^src-resnets-route-without-routers]。
- **输入依赖**：不同输入的 Top-$K$ 支撑围绕共享全局核变化（约 13% 保留率时与全局集 Jaccard 0.871/0.892），类内重叠高于类间，困难度几乎不改变支撑规模[^src-resnets-route-without-routers]。
- **不能当作加速依据**：Top-$K$ 是事后解析支撑，验证范围只有两个 ImageNet 检查点[^src-resnets-route-without-routers]。

机制与数值见 [[implicit-functional-routing]]、[[implicit-residual-experts]]、[[residual-interaction-decomposition]]。

## 与循环深度的互补：参数重 vs 计算重

MoE 把容量放在参数上、每次前向只激活一部分；[[recurrent-depth|循环深度]]是它的镜像——材料化参数多于实际参数，算力随推理时的循环次数增长[^src-2502-05171-recurrent-depth]。Geiping et al. 在 §9 写下这层对照并给出分工判断：「循环深度擅长学习推理模式，MoE 擅长有效存储与检索复杂信息」，并提出未来的 recurrent MoE 可以在多轮迭代里反复路由到同一个 expert，而不是每个 expert 每次前向最多激活一次[^src-2502-05171-recurrent-depth]。该判断是作者对架构互补性的推测（"We posit"），没有配置或消融验证[^src-2502-05171-recurrent-depth]。

这两个方向在 3.5B/800B token 的 Huginn 上是分开做的：那篇论文没有用 MoE，MoE 只出现在 future work[^src-2502-05171-recurrent-depth]。

## 相关技术

- [[glu-gated-linear-unit|Gated Linear Units (GLU)]]
- [[heterogeneous-moe-routing|异质性感知 MoE 路由]]
- [[adaptive-graph-agent-attention|自适应图代理注意力]]
- [[multi-modality-guided-spatial-expert|多模态引导空间专家]]
- [[token-level-specialization|Token 级专业化]]
- [[cluster-based-gating|簇基门控]]
- [[mmoe|MMoE (TiMi)]]
- [[timi|TiMi]]
- [[moirai-moe|Moirai-MoE]]
- [[time-moe|Time-MoE]]
- [[time-300b|Time-300B]]
- [[fusionbench|FusionBench]] · [[deep-model-fusion]] · [[model-merging]]
- [[moba-attention]] — MoE top-k 门控用于 attention（KV 块即专家，LLM 长上下文）
- [[native-sparse-attention]] — 三分支各自 softmax + sigmoid 门控合并（压缩 / 块选择 / 滑动窗口）
- [[mita-attention]] — 形变 fast-weight 专家 + 共享专家（landmark query 取 top-k，压缩与路由并用）
- [[fast-weight-scaling-taxonomy]] — 从 fast-weight 缩放视角给 MoE/top-k 注意力定位的五维分类
- [[motion-moe]] — 轨迹预测的 indicator / route 双类专家（TrajAR）
- [[implicit-functional-routing]] — dense ResNet 的隐形 router：执行全 dense，功能贡献按输入变化
- [[implicit-residual-experts]] — Möbius interaction 作为事后专家支撑，质量集中但决策不稀疏

## 相关分析

- [[moe-in-time-series-and-spatio-temporal]] — MoE 在时序/时空领域的机制谱系：动机、插入位置、门控/稀疏/均衡/专家形态四维对照，并列出与 MoE 易混的边界机制

[^src-fast-long-horizon-forecasting]: [[source-fast-long-horizon-forecasting]]
[^src-most]: [[source-most]]
[^src-dynamix]: [[source-dynamix]]
[^src-stamimputer]: [[source-stamimputer]]
[^src-mage]: [[source-mage]]
[^src-time-moe]: [[source-time-moe]]
[^src-moirai-moe]: [[source-moirai-moe]]
[^src-timi]: [[source-timi]]
[^src-moba-mixture-of-block-attention]: [[source-moba-mixture-of-block-attention]]
[^src-jmlr-25-1243]: [[source-jmlr-25-1243]]
[^src-trajar]: [[source-trajar]]
[^src-2602-01219-mita]: [[source-2602-01219-mita]]
[^src-2502-11089-nsa]: [[source-2502-11089-nsa]]
[^src-resnets-route-without-routers]: [[source-resnets-route-without-routers]]
[^src-2502-05171-recurrent-depth]: [[source-2502-05171-recurrent-depth]]
[^src-qwen3-next-blog-2025-09]: [[source-qwen3-next-blog-2025-09]]

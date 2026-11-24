---
title: "Cross-Dimension Dependency"
type: concept
tags:
  - time-series
  - multivariate
  - cross-dimension
  - dependency
created: 2026-05-30
last_updated: 2026-11-24
source_count: 7
confidence: medium
status: active
---

# Cross-Dimension Dependency

Cross-dimension dependency（跨维度依赖）是多变量时间序列 (MTS) 中不同变量（维度）之间的关联关系，与 [[cross-time-dependency|跨时间依赖]]（同一变量不同时间步之间的关联）相对 [^src-crossformer-2023]。

## 定义

在 MTS $\mathbf{X} \in \mathbb{R}^{T \times D}$ 中，$D$ 个变量之间存在关联——例如预测温度时，不仅历史温度有用，风速、湿度等信息同样有助于预测 [^src-crossformer-2023]。这种跨变量的信息依赖即为 cross-dimension dependency。

## 建模方式

### 隐式建模

传统 Transformer 模型将同时间步所有变量嵌入为单一向量 $\mathbf{x}_t \to \mathbf{h}_t$，通过 embedding 层隐式混合维度信息，但注意力机制仅捕获跨时间依赖 [^src-crossformer-2023]。这种方式不能显式挖掘跨维度关系。

### 显式建模

- **GNN 方法**（MTGNN）：学习维度间图结构，用图卷积显式建模 [^src-crossformer-2023]
- **CNN 方法**（[[lstnet|LSTNet]]）：首个用 CNN 捕获跨维度依赖的深度学习模型，SIGIR 2018 [^src-lstnet][^src-crossformer-2023]
- **Transformer 方法**（[[crossformer|Crossformer]]）：DSW embedding 保留维度信息 + TSA layer 的 Cross-Dimension Stage 用 Router 机制建模维度间依赖 [^src-crossformer-2023]

### iTransformer 的反转范式

[[itransformer|iTransformer]] 提供了建模跨维度依赖的新范式——将 **self-attention 直接作用于 variate token 维度**，score map $\mathbf{A} \in \mathbb{R}^{N \times N}$ 直接展现变量间相关性结构[^src-itransformer]。与传统方法的关键区别：

- **不修改组件**：仅改变 attention 和 FFN 的应用维度
- **可解释性**：浅层注意力图≈历史变量相关性，深层≈未来变量相关性
- **避免噪声**：Crossformer 在跨变量 patch 间交互引入时间不对齐噪声，iTransformer 的 variate token 交互更干净
- **通用性**：可应用于任何 Transformer 变体，平均 MSE 提升 16.8%~38.9%[^src-itransformer]

### CI vs CD 的折中

[[channel-independence|Channel Independence (CI)]] 策略完全忽略跨维度依赖，[[cvpe|CVPE]] 则提出折中——仅在最轻量的 patch embedding 层注入跨变量信息而保留 CI backbone [^src-cvpe-2025]。实验表明：强跨变量相关数据集上 CD 增益显著，弱相关数据集上可能过拟合。

### CPiRi：时空解耦的排列不变框架

[[cpiri|CPiRi]] (ICLR 2026) 代表了 CI-CD 融合的新范式——通过时空解耦架构将 CI 和 CD 分配到不同组件，而非在单个组件内折中 [^src-cpiri]。冻结局模型承担 CI 角色（逐通道独立提取时间特征），可训练的空间模块承担 CD 角色（通过 multi-head self-attention 学习内容驱动的跨通道交互）。训练时通过随机通道打乱迫使空间模块学习内容驱动的排列不变关系推理，而 CD 模型因依赖固定位置编码而在此测试中崩溃（Informer 错误率增加 >400%）[^src-cpiri]。这种解耦设计同时实现了 $O(T^2 + C^2)$ 的计算复杂度，比耦合方法的 $O((T\times C)^2)$ 更具扩展性 [^src-cpiri]。

### Timer-XL：展平二维因果时空注意力

[[timer-xl|Timer-XL]]（ICLR 2025）与 [[itransformer|iTransformer]]、[[cpiri|CPiRi]] 构成**同槽位互斥**与**瓶颈接力 $\Delta$**对照：它将所有变量的非重叠时间块按时间优先展平为 $NT$ 个 token，在单流因果解码器内用时序 RoPE、变量轴置换等变标量偏置 $(u, v)$ 与 Kronecker 积掩码 $M = C \otimes T$ 同时建模跨时间与跨维度依赖（并借变量图 $C$ 兼容协变量输入）[^src-timer-xl]。这保留了 patch 级因果时序、避免了 iTransformer 将整条历史压成单 token 的信息损失，但理论计算复杂度升至 $\mathcal{O}(N^2 T^2)$；[[cpiri|CPiRi]] 在 8,600 通道 LargeST CA 上实测 Timer-XL 发生 OOM，反向印证了超高维下时空解耦的必要性[^src-cpiri]。

## 正交设计决策轴对照

| 代表方法 | Token 粒度 | 跨维度交互方式 | 时序建模与因果性 | 变量置换对称性 | 注意力理论复杂度 |
|---|---|---|---|---|---|
| [[patchtst\|PatchTST]] | 单变量 Patch | 无（通道独立 CI） | 双向编码器，注意力逐通道独立[^src-patchtst] | 平凡等变（无跨通道） | $\mathcal{O}(N T^2)$ |
| [[crossformer\|Crossformer]] | DSW 2D Patch | 显式跨维度 Router 注意力 | 双向两阶段编码器[^src-crossformer-2023] | 否（依赖位置编码） | $\mathcal{O}(T^2 + D^2 L_{\text{seg}})$ |
| [[itransformer\|iTransformer]] | 整序列 Variate Token | 变量维自注意力 | FFN 隐式映射时间[^src-itransformer] | 等变（无位置编码） | $\mathcal{O}(N^2)$ |
| [[timer-xl\|Timer-XL]] | 多变量展平 Patch | 单流统一注意力 + 偏置 $(u,v)$ + 图掩码 $C$ | 因果解码器（$M = C \otimes T$）[^src-timer-xl] | 等变（双偏置 $u,v$） | $\mathcal{O}(N^2 T^2)$ |
| [[cpiri\|CPiRi]] | 时空解耦通道表示 | 冻结时序编码 + 排列不变空间注意力 | 冻结编码器提取时间[^src-cpiri] | 等变（通道打乱正则） | $\mathcal{O}(T^2 + N^2)$ |

## 与 Cross-Time Dependency 的关系

MTS 预测需要同时建模两种依赖 [^src-crossformer-2023]：

| 依赖类型 | 方向 | 传统模型处理方式 |
|----------|------|------------------|
| Cross-time | 同一变量，不同时间步 | Transformer 的自注意力 |
| Cross-dimension | 同一时间步，不同变量 | 嵌入层隐式混合 / GNN |

[[crossformer|Crossformer]] 的 [[two-stage-attention|TSA Layer]] 分别处理两种依赖，时间轴和维度轴有不同的语义，不能像图像的高度/宽度那样互换 [^src-crossformer-2023]。

## 相关页面

- [[lstnet]] — 首个 CNN 跨维度依赖模型，跨维度 MTS 深度学习路线的起点 (SIGIR 2018)
- [[crossformer]] — 首个显式利用跨维度依赖的 Transformer
- [[channel-independence]] — CI 策略（不建模跨维度依赖）
- [[cvpe]] — CI + CD 折中方案
- [[cpiri]] — CI+CD 时空解耦融合框架 (ICLR 2026)
- [[timer-xl]] — 多变量展平因果解码器（同槽位单流耦合对照，ICLR 2025）
- [[two-stage-attention]] — Crossformer 的两阶段注意力
- [[router-mechanism-for-cross-dimension]] — 降低跨维度注意力复杂度的路由机制
- [[itransformer]] — 反转范式：attention 作用于 variate token 维度
- [[multivariate-correlation-attention]] — iTransformer 的变量间注意力

[^src-lstnet]: [[source-lstnet]]
[^src-crossformer-2023]: [[source-crossformer-2023]]
[^src-cvpe-2025]: [[source-cvpe-2025]]
[^src-itransformer]: [[source-itransformer]]
[^src-cpiri]: [[source-cpiri]]
[^src-patchtst]: [[source-patchtst]]
[^src-timer-xl]: [[source-timer-xl]]

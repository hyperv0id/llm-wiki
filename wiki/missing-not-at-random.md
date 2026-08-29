---
title: "Missing Not At Random (MNAR) 与缺失机制"
type: concept
tags:
  - data-imputation
  - missing-mechanism
  - mnar
  - statistics
created: 2026-06-08
last_updated: 2026-08-29
source_count: 3
confidence: medium
status: active
---

# Missing Not At Random (MNAR) 与缺失机制

**缺失机制（missing mechanism）**刻画"为什么数据会缺失"——即缺失指示掩码 $M$ 的生成过程 $p(M \mid X)$。按 Rubin 的标准分类分为三类，决定了插补时是否必须显式建模缺失过程[^src-prdim]。

## 三类机制

设完整数据 $X$、缺失掩码 $M \in \{0,1\}^D$（$M_d=1$ 表示观测到），观测/缺失子集 $X^{obs}=X\odot M$、$X^{mis}=X\odot(1-M)$。则[^src-prdim]：

$$
p_\phi(M \mid X) =
\begin{cases}
p_\phi(M) & \text{(MCAR)}\\
p_\phi(M \mid X^{obs}) & \text{(MAR)}\\
p_\phi(M \mid X^{obs}, X^{mis}) & \text{(MNAR)}
\end{cases}
$$

| 机制 | 含义 | 例子 |
|------|------|------|
| **MCAR**（完全随机缺失） | 缺失与数据无关 | 传输随机丢包 |
| **MAR**（随机缺失） | 缺失只依赖**已观测**值 | 某传感器在低温时更易停采（温度可观测） |
| **MNAR**（非随机缺失） | 缺失依赖**未观测**值本身 | 病情恶化/死亡导致测量缺失；高浓度污染传感器超量程失效 |

## 可忽略性（Ignorability）

关键区别在于缺失过程是否**可忽略**[^src-prdim]：

- 在 **MCAR/MAR** 下，观测似然 $p_{\theta,\phi}(X^{obs}, M) \propto p_\theta(X^{obs})$，可以**忽略缺失过程**、只对观测数据建模即可一致地估计缺失值。
- 在 **MNAR** 下，掩码还依赖未观测值，缺失过程**非可忽略**——必须显式推断 $p_\phi(M \mid X)$，否则 MCAR/MAR 假设下的插补模型会系统性偏差。

这正是 MNAR 更现实却也更难的根源：要联合建模 $p_{\theta,\phi}(X, M) = p_\theta(X)\,p_\phi(M\mid X)$。

## 为什么对插补研究重要

大量插补方法（含多数扩散插补）在训练/评测时采用 **MCAR 式人工掩码**——把观测值随机遮盖作为伪目标[^src-prdim]。若数据集真实缺失掩码 $M$ 的分布与人工掩码 $A$ 差异大，则在 $p_\theta(X\mid X^{obs,A}, A)$ 上训练的模型难以泛化到真实的 $p_\theta(X\mid X^{obs}, M)$。[[prdim|PRDIM]] 实证：插补**原始（真实）缺失**比插补人工缺失显著更难，凸显了显式建模缺失机制的必要性[^src-prdim]。

## 各模型的缺失机制假设

| 模型 | 假设 | 缺失过程处理 |
|------|------|------------|
| [[csdi\|CSDI]] | MCAR（人工随机掩码） | 忽略 |
| [[nuwats\|NuwaTS]] | 随机/连续缺失（训练随机 mask ratio） | 忽略，但学习 mask-invariant 表示 |
| [[t1\|T1]] | 点/块/自然缺失（mask-aware embedding） | 忽略，掩码作为输入通道 |
| [[prdim\|PRDIM]] | **MNAR** | **显式建模** $p_\phi(M\mid X)$（模式识别器 + EM） |
| [[diffputer\|DiffPuter]] | 评测覆盖 MCAR/MAR/MNAR（MCAR 为主实验） | 不显式建模 $p(M\mid X)$——论文的公式化只含数据密度 $p_\theta(x)$，将 $X^{mis}$ 作隐变量做 EM 交替 |

[[diffputer|DiffPuter]] 沿 Muzellec et al. (2020) 与 Zhao et al. (2023) 的协议（MAR/MNAR 的掩码生成具体沿用 Zhao et al. 2023，DiffPuter 附录 D.3）在三种机制下评测插补性能，但其 EM 公式化只建模完整数据密度、不包含缺失掩码分布，与 PRDIM 的显式缺失建模形成对照[^src-diffputer]。

## 综述口径（2025）

Wang & Du 等人的 MTSI 综述从领域层面给出了与本页一致的机制框架：综述按 Rubin 1976 分 MCAR/MAR/MNAR，指出 MCAR/MAR 可忽略（插补只需学习 $p(X^o)$）、MNAR 不可忽略（忽略缺失机制会导致参数估计有偏，需学习数据与缺失机制的联合分布 $p(X^o, M)$）[^src-mts-imputation-survey]。综述认为现有插补方法多以 MCAR 或 MAR 假设运作，而真实数据中 MNAR 常见——例如气流信号分析中高值观测缺失导致饱和峰、观测分布相对真实分布明显偏斜——并将"处理 MNAR"列为未来方向之首[^src-mts-imputation-survey]。其 Table 1 对 33 个方法的机制标注中，仅 supnotMIWAE 标注 MNAR、SADI 标注 MCAR/MAR/MNAR，其余均标注 MCAR 或 MCAR/MAR[^src-mts-imputation-survey]。这些均为综述作者的二手归类口径，与各原论文自述分立。

## 历史脉络

缺失模型的判别器思路源自生成模型：GAIN（基于 GAN，判别器近似 $p(M\mid X)$）与 not-MIWAE（基于 VAE，扩展到 MNAR，把判别器损失并入 ELBO）。PRDIM 把这一"模式识别器"思路引入扩散框架，详见 [[pattern-recognizer-guidance]][^src-prdim]。

## 关联页面

- [[prdim]] — 首个专门处理 MNAR 的扩散插补模型
- [[mts-imputation-taxonomy]] — MTSI 综述的分类框架，其 Table 1 附带各方法的缺失机制标注
- [[diffputer]] — EM 交替式扩散插补，MCAR/MAR/MNAR 三机制评测但不建模缺失过程
- [[pattern-recognizer-guidance]] — PRDIM 显式建模 $p(M\mid X)$ 的机制
- [[csdi]] — 采用 MCAR 人工掩码的扩散插补代表
- [[nuwats]] / [[t1]] — 假设随机缺失的插补方法（忽略缺失过程）

[^src-prdim]: [[source-prdim]]
[^src-diffputer]: [[source-diffputer]]
[^src-mts-imputation-survey]: [[source-mts-imputation-survey]]

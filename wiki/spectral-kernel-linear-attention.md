---
title: "Spectral Kernel Linear Attention"
type: analysis
tags:
  - linear-attention
  - spectral-graph-theory
  - kernel-methods
  - rope
  - graph-transformers
  - effective-resistance
created: 2026-08-13
last_updated: 2026-08-30
source_count: 4
confidence: medium
status: active
---

# Spectral Kernel Linear Attention

从 [[wire|WIRE]] 的推导链出发，把旋转读作**核的随机特征映射**而非位置编码，直接落地为一种线性注意力：注意力核 = 内容相似度 × 有效电阻高斯核，复杂度 $O(N)$。本页为用户问答（2026-08-13）的归档分析，综合 WIRE、SpecSTG、Mamba↔线性注意力统一框架、HiFiNet 四个源推导。

## 目标核

$$\tilde{A}_{ij} = q_i^\top k_j \cdot e^{-\gamma R(i,j)}$$

其中 $R(i,j)$ 是节点 $i,j$ 之间的有效电阻：它构成节点集上的度量，是最短路径距离的下界，树上取等号[^src-2509-22259]。核的读法：内容相似度乘一个按图结构衰减的因子。

## 推导链

### 1. 电阻的谱展开

$L^+$ 为图拉普拉斯伪逆，$\{u_k, \lambda_k\}$ 为拉普拉斯特征对：

$$R(i,j) = (e_i - e_j)^\top L^+ (e_i - e_j) = \sum_{k=1}^{N-1} \frac{(u_k[i] - u_k[j])^2}{\lambda_k}$$

定义**电阻嵌入** $r_i = [u_k[i]/\sqrt{\lambda_k}]$，则 $R(i,j) = \|r_i - r_j\|^2$（注意：平方欧氏距离一般不是度量，度量性来自电阻距离的特殊结构）。图傅里叶变换定义为 $\hat{x} = U^\top x$，谱域滤波是对特征值的逐元素运算 $[g_\theta(\Lambda)\hat{x}]_i = g_\theta(\lambda_i)\hat{x}_i$[^src-2401-08119-specstg]。谱坐标在本页读法中的含义：特征空间坐标，**平方距离 = 核的对数衰减率**——不是"位置"。

### 2. 随机傅里叶特征（高斯特征函数）

对 $\omega$ 在 $\mathbb{R}^m$ 上服从 $\mathcal{N}(0, 2\gamma I)$（本页推导）：

$$e^{-\gamma\|x\|^2} = \mathbb{E}_\omega\left[e^{i\omega^\top x}\right]$$

### 3. 旋转 = 随机特征

取特征映射 $\phi(q_i) = R(\omega^\top r_i)\, q_i$，$R(\cdot)$ 为逐 2D 块的旋转矩阵，各块 $\omega_n$ 独立同分布。逐块展开后 sin 项在期望下消失，cos 项恰为特征函数的实部（本页推导）：

$$\mathbb{E}_\omega[\phi(q_i)^\top \phi(k_j)] = q_i^\top k_j \cdot e^{-\gamma R(i,j)} \quad (\text{期望下精确})$$

WIRE Theorem 3 陈述的是一阶展开 $q_i^\top k_j\left(1 - \frac{\omega^2 R(i,j)}{2}\right) + \mathcal{O}(\omega^4)$[^src-2509-22259]（论文以 $\omega^2$ 记随机频率方差，本页 $\gamma = \omega^2/2$）。即：**旋转就是该核的随机特征映射**，Theorem 3 是精确期望的 Taylor 首项。

### 4. 落成线性注意力

$$\mathcal{O}_i = \frac{\phi(q_i)^\top \sum_j \phi(k_j) v_j^\top}{\phi(q_i)^\top \sum_j \phi(k_j)}$$

按结合律先累加 $\sum_j \phi(k_j)v_j^\top$ 与 $\sum_j \phi(k_j)$，得到 $O(N)$ 复杂度（每 token 特征 $\phi$ 计算 $O(dm)$，谱分解一次性 $O(N^3)$ 或稀疏图迭代近似、跨层摊销），无 $N\times N$ 矩阵、无 softmax。线性注意力的循环形式 $S_i = S_{i-1} + K_i^\top V_i$ 见 [[linear-attention-unified-framework]]；归一化对性能至关重要（移除归一化导致 -5.2%）[^src-demystify-mamba-linear-attention-2024]。

## 两条实现路线

1. **旋转路线**（随机特征）：特征维度固定为 $d$；用学到的 $\omega$ 替代随机采样（WIRE 的实践[^src-2509-22259]），零 MC 方差；代价是单实例是一个 cos 扰动而非期望核。
2. **精确路线**（不采样）：$e^{-\gamma R} = e^{-\gamma\|r_i\|^2} e^{-\gamma\|r_j\|^2} e^{2\gamma r_i^\top r_j}$。$e^{2\gamma r^\top s}$ 是多项式核的正系数和、本身 PSD，Taylor 截断给有限维特征；归一化后 query 侧指数消掉，key 侧剩重加权 $e^{-\gamma\|r_j\|^2}$。"精确"是 $T\to\infty$ 意义下（截断误差 $O((2\gamma r^\top s)^{T+1}/(T+1)!)$），特征维 $C(m+T,T)$ 随 Taylor 阶数增长——旋转是"固定维度换近似"的便宜招（本页推导）。

## 与位置编码读法的分界

- **PE 读法**：旋转是 token 预处理，保留 $N^2$ softmax；拓扑以加性 logit（RPE 式）进入。WIRE 论文指出谱 RPE 必须实例化 $N\times N$ 注意力矩阵，无法与线性注意力兼容[^src-2509-22259]。
- **注意力读法**：$\phi$ 是核的分解，拓扑以**乘性核**进入——乘性正是可分解、可 $O(N)$ 的原因。WIRE 已走到"旋转直接作用于 token，因此可与 [[performer|Performer]] 结合，实现线性注意力下的拓扑掩码"[^src-2509-22259]；本页补上最后一步：兼容不是巧合，旋转本来就是该核的随机特征。

## 读法买到的设计旋钮

- **每头一个 $\omega$**：多尺度拓扑核集成，替代随机采样。
- **每层可学 $\omega_l$**：逐层可学核时标（扩散时间的注意力版语义）。
- **$m$ 截断是安全低通**：$1/\lambda$ 权重让小 $\lambda$（Fiedler 等）主导电阻嵌入；图频率按特征值排序，小 $\lambda$ 对应平滑（低频）信号[^src-hifinet]。
- **推广菜单**：内容无关时 $\phi$ 退化为 $[f(\lambda_k)u_k[i]]$，核 = $U f(\Lambda)^2 U^\top$——热核 $f = e^{-t\lambda/2}$、扩散 $f = \lambda^{l/2}$、PPR $f = (1+\alpha\lambda)^{-1/2}$ 全是一条链。

## 边界与风险

- 一阶近似在 $\gamma R(i,j) > 1$ 时注意力为负（树上 $R = d$，受直径约束）；精确路线或小 $\gamma$ 规避（本页推导）。
- 衰减律依赖维数：仅 2D 网格上 $R \approx \frac{1}{\pi}\ln d$、核为幂律 $d^{-\gamma/\pi}$；1D 路径上 $R = d$、核为指数衰减 $e^{-\gamma d}$；3D 网格上 $R$ 有界、核饱和。
- 秩 ≤ 特征维（旋转路线 $d$、精确路线 $d\cdot C(m+T,T)$）是固定维特征线性注意力的表达力限制。
- 不归一化时对节点度数（$e^{-\gamma\|r_j\|^2}$ 重加权）敏感。

## 相关页面

- [[wire]] — 推导链的起点；旋转操作的原始形式与 Theorem 3
- [[rope]] — 网格图上的特例（WIRE Theorem 2：网格图恢复标准 RoPE）
- [[roformer]] — RoPE 的原始模型
- [[linear-attention-unified-framework]] — 线性注意力核形式与归一化消融
- [[performer]] / [[positive-random-features]] — 随机特征线性注意力的原始机制（softmax 核正特征无偏估计 + 正交化降方差）
- [[fast-spectral-graph-convolution]] — 谱域滤波的 $O(KN)$ 形式
- [[spectral-graph-wavelet-transform]] — 另一谱域图信号处理方法
- [[graph-frequency-decomposition]] — 图频率分解（低频=平滑）
- [[specstg]] — 图谱域扩散实例
- [[hifinet]] — 结构性图频率分解实例

[^src-2509-22259]: [[source-2509-22259]]
[^src-2401-08119-specstg]: [[source-2401-08119-specstg]]
[^src-demystify-mamba-linear-attention-2024]: [[source-demystify-mamba-linear-attention-2024]]
[^src-hifinet]: [[source-hifinet]]

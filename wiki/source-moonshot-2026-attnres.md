---
title: "Attention Residuals (Kimi Team, 2026)"
type: source-summary
tags:
  - residual-connections
  - depth-attention
  - llm
  - macro-design
created: 2026-10-27
last_updated: 2026-10-27
source_count: 1
confidence: medium
status: active
---

# Attention Residuals: Technical Report

Kimi Team（Moonshot AI）技术报告（arXiv:2603.15031v1，2026-03-16 提交，21 页；PDF `raw/attnres-2603-15031.pdf`；代码 github.com/MoonshotAI/Attention-Residuals）。核心主张：残差连接以单位权重累加前序各层输出，导致 Pre-LN 隐藏态 $O(L)$ 增长并稀释层贡献；AttnRes 用深度维 softmax 注意力替代累加——每层设零初始化伪查询 $w_l\in\mathbb{R}^d$、键施加 RMSNorm，以凸组合聚合各层输出；Block 变体将 $L$ 层分为 $N$ 块，仅对块级表示与 embedding 做注意力，显存/通信从 $O(Ld)$ 降至 $O(Nd)$[^src-moonshot-2026-attnres]。

**理论统一**。论文将各残差变体统一为深度混合矩阵 $M\in\mathbb{R}^{L\times L}$：标准残差与 Highway 为 1-semiseparable，(m)HC 为 $m$-semiseparable（有效权重 $\beta_i^\top A^{\times}_{i+1\to l}\alpha_l$，即深度维线性注意力），AttnRes 为稠密秩 $L$（深度维 softmax 注意力）；核可分解为 $\phi(q)^\top\phi(k)$ 时，深度注意力塌缩回递归（§6.2）[^src-moonshot-2026-attnres]。

**证据**。五尺寸 scaling law（194M–528M 激活，四变体对照）：Full AttnRes 在 5 档中 4 档最低（仅 241M 档 mHC(-lite) 1.869 更低；436M 档 1.737 对基线 1.766、mHC 1.747），Block 达到等效 1.25× 计算（5.6 PFLOP/s-days 处 1.692 对 1.714）；消融显示静态系数的 DenseFormer（1.767）与基线持平，输入相关权重是增益来源；48B/3B Kimi Linear、1.4T token 下 15 项基准 14 升 1 平，GPQA-Diamond +7.5、MATH +3.6、HumanEval +3.1；PP 下训练开销 <4%、推理延迟 <2%，Block 变体每层 I/O 为 5.5d 对 mHC 的 34d（Table 1–4、Fig. 4–6）[^src-moonshot-2026-attnres]。

**边界**。Full 的 $O(Ld)$ 跨层显存在当前硬件上无法支持大模型，生产配置采用 $N\approx 8$；扫描显示的深度偏好受推理延迟限制，未推荐部署；线性复杂度深度注意力留待后续探索；输入相关查询更优（1.731），但解码需顺序访存，默认采用静态查询（§3.1、§5.3、§5.4、结论）[^src-moonshot-2026-attnres]。

机制详解见 [[attnres]]；谱系前驱见 [[hyper-connections]]、[[manifold-constrained-hyper-connections]]。

[^src-moonshot-2026-attnres]: [[source-moonshot-2026-attnres]]

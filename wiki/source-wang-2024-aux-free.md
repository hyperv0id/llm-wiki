---
title: "Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts (Wang et al., 2024)"
type: source-summary
tags:
  - mixture-of-experts
  - load-balancing
  - routing
  - deepseek
created: 2026-10-27
last_updated: 2026-10-27
source_count: 1
confidence: medium
status: active
---

# Auxiliary-Loss-Free Load Balancing (Wang et al., 2024)

Loss-Free Balancing（Wang et al., DeepSeek-AI 与北京大学, arXiv:2408.15664, 2024-08 preprint；PDF `raw/auxfree-2408-15664.pdf`）针对辅助负载均衡损失的两难：强损失压住路由塌缩，却把与语言建模目标冲突的梯度带进训练；弱损失则失衡（§1、§2.2）[^src-wang-2024-aux-free]。方法给每个 routed expert 一个标量 bias $b_i$，只加在 top-$K$ 选择前的打分上，输出加权仍用未加 bias 的打分（式 3）[^src-wang-2024-aux-free]。每个 batch 训练后按 $b_i\leftarrow b_i+u\,\operatorname{sign}(\bar c_i-c_i)$ 更新，超载降、欠载升，只用上一 batch 负载以满足因果约束；$u$ 固定 0.001，无调度[^src-wang-2024-aux-free]。

模型骨干为 DeepSeekMoE，主实验用 sigmoid 门控，1B 训 100B token、3B 训 200B token，验证集 71M token。图 2 的 $\alpha$ 扫描呈现该两难：$\alpha=10^{-2}$ 时困惑度 9.70、MaxVio 近 0，$\alpha=0$ 时 9.54 但 MaxVio 约 5.0[^src-wang-2024-aux-free]。表 2 中 1B 困惑度 9.50 对基线 9.56、MaxVio 0.04 对 0.72；3B 为 7.92 对 7.97、0.04 对 0.52。更新率扫描（图 4）与 sign/线性/乘性三种变体（表 3、4）支持 sign 加性 bias 配 $u=10^{-3}$[^src-wang-2024-aux-free]。

在大 batch 设置中（§5.1、图 5），computation batch 增大时 Loss-Free 的 batch 级均衡持续改善，辅助损失基线基本走平，与专家并行相容。附录 C 中 softmax 门控仅保留均衡收益（9.599 对 9.604；0.027 对 0.937）；附录 D 给出 Expert Choice 的泄漏界 $K\log_2\frac{1-R}{R}$，9 层、16 专家、每 token 2 个合计超过 50 bit/token[^src-wang-2024-aux-free]。

实验仅覆盖 1B/3B 验证困惑度，无下游任务；bias 只在 1B 调参、3B 继承；未做 bias 与辅助损失同时开启的组合实验；§6 称 3B 训练 300B token，与 §4.1、摘要、附录 B 的 200B 冲突[^src-wang-2024-aux-free]。

机制与数据流见 [[auxiliary-loss-free-load-balancing|Loss-Free Balancing]]，谱系见 [[deepseekmoe]]。

[^src-wang-2024-aux-free]: [[source-wang-2024-aux-free]]

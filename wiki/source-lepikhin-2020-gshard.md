---
title: "GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding (Lepikhin et al., 2020)"
type: source-summary
tags:
  - mixture-of-experts
  - sparse-gating
  - distributed-training
  - machine-translation
  - compiler
created: 2026-10-27
last_updated: 2026-10-27
source_count: 1
confidence: medium
status: active
---

# GShard (Lepikhin et al., 2020)

GShard（Lepikhin et al., Google, arXiv:2006.16668, ICLR 2021；PDF `raw/gshard-2006-16668.pdf`）提出，通过轻量 sharding 注解与 XLA 编译器扩展，可将条件计算 MoE Transformer 推至 600B 参数，在 2048 块 TPU v3 上 4 天内训完；单一非集成模型在 100→英语翻译上超过先前最优（摘要）[^src-lepikhin-2020-gshard]。

**问题与机制**。§1.1 列出三项扩展障碍：通用框架缺乏高效模型并行、dense 扩展计算超线性、数千设备巨型计算图编译不可行。解法分两路：模型侧将每隔一个 FFN 层换为 position-wise MoE 层，采用 top-2 gating（第一 expert 确定分配、第二 expert 以正比于 $2g_2$ 的概率随机派发）、组级辅助负载均衡损失 $\ell_{\mathrm{aux}} = \frac{1}{E}\sum_e (c_e/S) m_e$、expert 容量限制与溢出直通（§2）；系统侧借助 `replicate`/`split`/`shard` 注解 API 与 SPMD partitioner 将集群视作单台设备，编译时间与设备数无关，dispatch/combine 走 $O(\sqrt{D})$ 的 AllToAll（§3）[^src-lepikhin-2020-gshard]。

**证据**。100 语言↔英语、25B 训练样本、每模型 1T token：MoE(2048E, 36L) 600B 平均 BLEU 44.3、较单语调优基线 +13.5（Figure 6 内嵌表）；dense 对照 T(96L) GPipe 为 36.9、+6.1，耗费约 235.5 core years，而 MoE 仅需 22.4（Table 3）；100 个双语基线合计耗费 29 core years（§7）。消融（§4.4）：12L→36L 平均 +2–3 BLEU；experts 128→512 +3.3、512→2048 仅 +1.3，容量瓶颈在 128–512 之间。系统侧（§5）：128→2048 experts（设备 16×）执行时间仅 1.7×，FFN 矩阵乘 >85% peak FLOPS [^src-lepikhin-2020-gshard]。

**边界**。MoE(2048E, 60L)/bfloat16/1T 权重试跑因数值不稳定未收录结果，正式训练用 float32（§4.3）；组容量 $C = O(1/D)$ 受 $C \geq 1$ 约束（§5.2）；溢出 token 的 MoE 贡献被丢弃，全文未报告溢出率；辅助损失权重与 gating 变体无消融；参数计数不总与有效容量相关（§7）[^src-lepikhin-2020-gshard]。

机制与数据流详解见 [[gshard]]，谱系位置见 [[mixture-of-experts]]。

[^src-lepikhin-2020-gshard]: [[source-lepikhin-2020-gshard]]

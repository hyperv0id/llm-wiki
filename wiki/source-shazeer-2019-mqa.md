---
title: "Fast Transformer Decoding: One Write-Head is All You Need"
type: source-summary
tags:
  - attention
  - kv-cache
  - efficient-inference
  - transformer
  - machine-translation
created: 2026-10-26
last_updated: 2026-10-26
source_count: 1
confidence: medium
status: active
---

# Fast Transformer Decoding: One Write-Head is All You Need

Noam Shazeer（Google）的 arXiv 预印本 arXiv:1911.02150v1（cs.NE，2019-11-06，本地副本 `raw/fast-transformer-decoding-one-write-head-1911-02150.pdf`）含 §1–§5 与参考文献，无附录；§2.3–§3 给出 kernel 实现的伪代码与 TensorFlow 代码清单[^src-shazeer-2019-mqa]。

## 问题

Transformer 训练时整段并行，算术量 $\Theta(bnd^2)$ 与访存量 $\Theta(bnd+bhn^2+d^2)$ 之比足以喂满算力；增量推理沿 token 串行，同一算术量分摊在 $\Theta(bn^2d+nd^2)$ 访存上，比值降至 $\Theta(n/d+1/b)$，接近 1 时带宽成为瓶颈（§2.3.1、§2.4.1）[^src-shazeer-2019-mqa]。多头注意力的 K、V 带有 head 维，同一份记忆存 $H$ 份，$\Theta(n/d)$ 项随 $H$ 放大（§2.4.1、§3.1）[^src-shazeer-2019-mqa]。

## 方法

Multi-query attention 去掉 K/V 的 head 维、保留 query 的 head 维：$H$ 组 query 共享一份 $K,V$，打分与加权仍逐 head 独立，最后照常拼接并通过 $P_o$（§2.2、§2.4.1、§3）[^src-shazeer-2019-mqa]。单步解码重读字节由 $8bHMd_k$ 降至 $8bMd_k$，乘加数不变，KV 算术强度由 $0.5$ 升至 $4$ FLOP/byte；每步重写整段 K/V 的开销同比例下降，论文标题的 write-head 指写入记忆的这一端（§2.4.1）[^src-shazeer-2019-mqa][INFERENCE]。省下的 K/V 参数通过加宽 $d_{ff}$ 补齐总参数量（§4.1）[^src-shazeer-2019-mqa]。

## 证据

WMT14 EN-DE，211M 参数、100K 步：dev $\ln(\text{PPL})$ 1.424→1.439，dev BLEU 26.7→26.5，test greedy 27.7→27.5，test beam-4 28.4→28.5；同预算下削 $H$/$d_k$ 的对照为 1.480–1.518 PPL、25.8–26.2 BLEU（Table 1）[^src-shazeer-2019-mqa]。Billion-Word 上 dev PPL 29.9→30.2，削头方案 30.9–31.2（Table 3）[^src-shazeer-2019-mqa]。单张 TPUv2 8 核、序列长 128、batch 1024：训练 13.2→13.0 µs，encoder 1.7→1.5，decoder 1.7+46→1.5+3.8，beam-4 2.0+203→1.6+32；叠加局部注意力后解码降至 3.3 µs（Table 2）[^src-shazeer-2019-mqa]。

## 边界

质量轻微下降，仅在 beam-4 的 test BLEU 上高 0.1；论文未给质量-速度折中曲线（Table 1、Table 3）[^src-shazeer-2019-mqa]。层在训练期定义（§3），未测把已训练多头模型的 K/V 合并为一份的效果。评测限于翻译与 Billion-Word LM，序列长固定 128，未测长上下文；测速为固定形状 padding 口径，每步耗时相同（脚注 1）[^src-shazeer-2019-mqa]。

---

[^src-shazeer-2019-mqa]: [[source-shazeer-2019-mqa]]

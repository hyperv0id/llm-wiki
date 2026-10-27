---
title: "Source: DeepSeek-V2 技术报告 — Multi-head Latent Attention (arXiv:2405.04434)"
type: source-summary
tags:
  - source
  - attention
  - kv-cache
  - low-rank
  - moe
  - llm
created: 2026-10-26
last_updated: 2026-10-26
source_count: 1
confidence: medium
status: active
---

# Source: DeepSeek-V2 技术报告（arXiv:2405.04434）

作者 DeepSeek-AI（集体署名），2024-05 首次提交预印本技术报告（未见 venue），本页按 2024-06 的 v5、52 页 PDF 核对；论文链接 https://arxiv.org/abs/2405.04434，本地为 `raw/deepseek-v2-2405-04434.pdf`（PDF 归档，文本经 `pdftotext -layout` 转换）。

## 核心论点

DeepSeek-V2 是总参数 236B、激活参数 21B 的 MoE 语言模型，两项改动为：注意力侧用 [[multi-head-latent-attention|MLA]] 消除 KV cache 瓶颈，FFN 侧沿用 DeepSeekMoE（§1、§2；机制见 [[deepseekmoe]]）[^src-deepseek-v2-2405-04434]。论文以 7B dense 同参数对照论证 MQA/GQA 以质量换 cache：MHA 在 BBH/MMLU/C-Eval/CMMLU 上为 37.0/45.2/42.9/43.5，GQA(8 组) 为 35.6/41.2/37.7/38.4，MQA 为 33.2/37.9/30.0/34.6（表 8）[^src-deepseek-v2-2405-04434]。

## 设置

60 层、$d=5120$、128 头、$d_h=128$、query latent $d'_c=1536$，每层 2 个共享专家加 160 个路由专家、每 token 激活 6 个；在 8.1T token 上预训练，最大序列长 4K，再用 YaRN 把上下文从 4K 扩到 128K（$s=40$，长度缩放因子 $t=0.0707\ln s+1$），NIAH 显示 128K 内检索稳定（§3.1.2、§3.1.4、图 4）[^src-deepseek-v2-2405-04434]。

YaRN 只施加在解耦分支的共享 key $k^R_t$ 上：位置信息全在该分支，latent 侧不含位置（§3.1.4）[^src-deepseek-v2-2405-04434]。

## 方法

KV 联合低秩压缩：$c^{KV}_t=W^{DKV}h_t$，$d_c=512\ll d_hn_h=16384$，K 与 V 共用 latent，推理只缓存它（式 9–11）。上投影 $W^{UK}$、$W^{UV}$ 可吸收进 $W^{UQ}$、$W^{O}$，解码无需物化 K/V（附录 C）。因 RoPE 插在两侧投影间会破坏吸收，改用 decoupled RoPE：$d^R_h=64$ 的位置分支由 $h_t$ 直接产生共享 key $k^R_t$，与内容分支拼接后打分，cache 变为 $(d_c+d^R_h)l$（式 14–19）。表 1 列出四种机制的 cache 公式，DeepSeek-V2 为 $4.5d_hl=34{,}560$ 元素/token，等价 2.25 组 GQA（§2.1.4）[^src-deepseek-v2-2405-04434]。

## 证据

MoE 上 MLA vs MHA 两规模对照（表 9）：小 15.7B/2.4B 激活，BBH 39.0 vs 37.9、MMLU 50.0 vs 48.7、C-Eval 50.9 vs 51.6、CMMLU 53.4 vs 52.3，cache 14%；大 247.4B/21.5B，50.7/59.0/59.2/62.5 vs 46.6/57.5/57.9/60.7，cache 4%。摘要的 42.5% 训练成本、93.3% KV cache、5.76× 吞吐是整体账（MoE + FP8 + 6 bit KV 量化，对照 dense 67B），不是 MLA 单独收益（§3.2.3）[^src-deepseek-v2-2405-04434]。

## 局限

$d_c$ 与 $d^R_h$ 无消融，仅有 MHA/GQA/MQA 与 MLA-vs-MHA 两组对照（§D）；MHA 基线列 $860.2\text{K}$ 与正文头配置不自洽；单次训练且无方差，长上下文未与 GQA/MQA 对比。细节见 [[multi-head-latent-attention]]。

[^src-deepseek-v2-2405-04434]: [[source-deepseek-v2-2405-04434]]

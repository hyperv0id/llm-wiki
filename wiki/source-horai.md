---
title: "HORAI: Empowering Time Series Analysis with Large-Scale Multimodal Pretraining"
type: source-summary
tags:
  - time-series-forecasting
  - multimodal-time-series
  - foundation-model
  - frequency-domain
  - mixture-of-experts
  - zero-shot
created: 2026-11-24
last_updated: 2026-11-27
source_count: 1
confidence: medium
status: active
---

# HORAI: Empowering Time Series Analysis with Large-Scale Multimodal Pretraining

时序基础模型长期依赖单模态预训练，缺互补模态；直接拼进文本与图像又缺统一的多模态范式——混用全频段的同一套对齐让平滑走势与局部波动抢占同一表达通道。HORAI 把频带当作跨模态对齐接口：傅里叶变换拆出低频与中高频两带，低频对应趋势与宏观叙事、配文本，中高频对应局部波动形态、配折线图，门控融合后交时频混合专家自回归解码器预测。在六域、超十亿时间点的 MM-TS 语料上预训练后，零样本即在预测与异常检测上领先[^src-horai]。

## 核心做法

- **模态构建**：折线图由数值历史渲染、文字由 GPT-4o 读同一窗口生成，均属输入重编码；外生侧仅过滤后的 GDELT 新闻等真实文本[^src-horai]。
- **频带分离与对齐**：无参数 `FFT`/`iFFT` 按比例阈值切出两带；`Flow-Attention` 中低频对齐 Qwen 文本嵌入、中高频对齐 ViT 折线图嵌入[^src-horai]。
- **融合与解码**：两路嵌入门控加权后与时序表征相加；因果 Transformer 自回归预测，FFN 换为时频门控 Top-K 混合专家，路由同用时域与频域线索。全流程为既有算子组合[^src-horai]。

## 证据

1. **定位与主结论**：零样本取 Time-MMD 九域 5 域最优（余域逊个别基线），并超全量的 GPT4MTS 与 TimeVLM；异常检测 12/15 项第一[^src-horai]。
2. **真实涨点归因**：频带—模态互换消融掉点最狠、甚于删图文——主增益为频带配对先验加大语料，MoE 次级；缺内外生增益分解[^src-horai]。
3. **证据缺口**：骨干仅测小编码器，更大替换与模态分析移附录[^src-horai]。

## 相关页面

- [[aurora|Aurora]] — 同槽位互斥：同为十亿级多模态预训练，概率生成对确定性频带点预测。
- [[time-vlm|Time-VLM]] — 瓶颈接力 $\Delta$：冻结 VLM 检索换轻编码器频带对齐。
- [[multimodal-time-series-forecasting]] — 跨槽位正交：固定频带分工是其「模态真实性 × 对齐机制」矩阵一槽位。
- [[mixture-of-experts|Mixture of Experts]] — 同构/特例：时频门控 MoE 是 Top-K MoE 特例，路由扩至时频双路。

---

Peng Chen 等（华东师大、华为），NeurIPS 2026 Poster；`raw/21705_Empowering_Time_Series_A.pdf`。

[^src-horai]: [[source-horai]]

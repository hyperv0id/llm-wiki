---
title: "Source: TrajAR — Long-Term Trajectory Prediction at Urban Intersections via Multi-scale Interaction Perception"
type: source-summary
tags:
  - trajectory-prediction
  - intersection
  - autoregressive
  - multi-scale
  - mixture-of-experts
  - graph-attention
created: 2026-10-20
last_updated: 2026-10-20
source_count: 1
confidence: medium
status: active
---

# TrajAR: Long-Term Trajectory Prediction at Urban Intersections via Multi-scale Interaction Perception

**来源**：Letian Gong、Yan Lin（共同一作）等，北京交通大学、南洋理工大学、Aalborg University，IJCAI-26 会议论文（pp. 2635–2643），PDF 存于 `raw/trajar-gong-2026.pdf`。

**核心论点**：论文把路口多道路使用者（机动车、非机动车、行人）轨迹预测的困难归为两点。其一是隐式交互：交互关系由运动趋势、使用者类型、信号状态与路权规则共同决定，近距离不等于有交互[^src-trajar]。其二是长时预测：自回归解码误差累积，非自回归解码整体偏置且无逐步纠偏[^src-trajar]。论文的解法是把自回归预测从 next-step 改为 next-scale（借 VAR 思路），粗尺度给长时路径骨架并被更频繁监督，细尺度在其上逐级细化[^src-trajar]。

**贡献**：(1) 多尺度自回归解码器：$S_k$ 尺度分组、三次插值补齐细尺度起点、block-wise causal mask、全尺度 RMSE 监督[^src-trajar]。(2) 动态感知编码器：运动趋势层（长度对齐 + padding）、Motion MoE（类型/信号 indicator experts + 趋势 route experts，top-$K_r$ 路由）、GAT 隐式交互层（高斯核距离权重）+ Transformer 情境感知[^src-trajar]。(3) SinD-Tianjin/Xi'an 与 InD-Bendplatz/Frankenburg 四个路口数据集上的对比、消融、分类型、插值、概率与安全、效率实验[^src-trajar]。

**报告的数字**：表 1/表 2 的 ADE/FDE（e-2 口径）上 TrajAR 为 0.409/0.798、0.509/0.942、0.437/1.029、0.493/1.284，均为该表最低[^src-trajar]。相对 KI-GAN 的 ADE/FDE 降幅（基线作分母）为 17.4%/23.6%/16.4%/16.9% 与 15.6%/24.3%/21.6%/17.4%；表 1/2 的 Imp.(Δ%) 行则用逐列最优基线与本方法自身作分母，逐列核算见 [[trajar]]。消融（表 3）中去掉 decoder 的一行误差上升最大：SinD-Tianjin 0.409/0.798 → 1.623/3.241[^src-trajar]。

**局限**：结论把概率多模态预测与跨城市泛化列为未来工作，主实验为确定性 RMSE 训练；表 4 的 minADE/NLL/Collision 未说明概率输出来源[^src-trajar]。摘要「平均提升 23.8%」用的是以本方法为分母的降幅（本页核算：KI-GAN 参照、ADE/FDE 八格均值 23.9%）；换成基线分母只有 19.1%，论文未声明该口径（见 [[trajar]]）。未报参数量，「低于 100 ms」只有图 3(a) 曲线支撑；正文称下采样后间隔 0.12 s，该值只对 25 Hz 的 InD 成立（SinD 为 10 Hz）[^src-trajar]。

## Related Pages

- [[trajar]] — 模型页（结构、训练设置、完整结果）
- [[scale-wise-autoregressive-decoding]] — next-scale 解码技术页
- [[motion-moe]] — Motion MoE 技术页
- [[intersection-trajectory-prediction]] — 路口轨迹预测任务页

[^src-trajar]: [[source-trajar]]

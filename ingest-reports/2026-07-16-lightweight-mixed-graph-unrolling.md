# Ingest 报告：lightweight-mixed-graph-unrolling-traffic-forecast

## 创建
- wiki/source-lightweight-mixed-graph-unrolling.md — WHY：来源摘要页，覆盖 Qi et al. (ICML 2026) 混合图算法展开轻量级 Transformer 交通预测
- wiki/mixed-graph-spatiotemporal-modeling.md — WHY：核心概念，无向图（空间）+ 有向图（时间）的统一建模框架
- wiki/directed-graph-laplacian-regularizer.md — WHY：论文核心技术创新 DGLR，ℓ₂ 有向图正则项的定义、性质与频率解释
- wiki/directed-graph-total-variation.md — WHY：论文核心技术创新 DGTV，ℓ₁ 有向图正则项的软阈值滤波解释
- wiki/graph-learning-as-self-attention.md — WHY：图学习模块数学等价于自注意力的关键洞察，解释参数效率来源

## 修改
- wiki/algorithm-unrolling.md — WHY：新增 §5 Mixed-Graph ADMM Unrolling 应用案例，source_count 1→2，confidence medium→high
- wiki/traffic-forecasting.md — WHY：新增 "Mixed-Graph Algorithm Unrolling" 方法类别及完整论述，source_count 44→45 → lint 修正为 44（重复脚注计数有误），修复重复 last_updated
- wiki/index.md — WHY：添加新 source-summary、2 concept、2 technique 条目

## 新建交叉链接
- [[mixed-graph-spatiotemporal-modeling]] ↔ [[directed-graph-laplacian-regularizer]] ↔ [[directed-graph-total-variation]]
- [[mixed-graph-spatiotemporal-modeling]] ↔ [[graph-learning-as-self-attention]]
- [[algorithm-unrolling]] ↔ [[source-lightweight-mixed-graph-unrolling]]
- [[traffic-forecasting]] ↔ [[mixed-graph-spatiotemporal-modeling]] / [[directed-graph-laplacian-regularizer|DGLR]] / [[directed-graph-total-variation|DGTV]] / [[graph-learning-as-self-attention]]

## Lint 修复 (2026-07-16)

### 发现
- **source-lightweight-mixed-graph-unrolling.md**：`confidence: high` + `source_count: 1` 违规（source-summary 无 `[^src-*]` 引用，`source_count` 应为 0）→ `source_count: 0`, `confidence: medium`
- **source-lightweight-mixed-graph-unrolling.md**：百分比注脚 — 论文声称 7.2% 但 38/1404≈2.7%，已加注说明
- **graph-learning-as-self-attention.md**："2.7%" 应注明与论文 7.2% 说法不一致 → 已加注"论文声称 7.2%"
- **graph-learning-as-self-attention.md**：缺少 `[[algorithm-unrolling]]` 反向链接 → 已补
- **algorithm-unrolling.md §5**：新条目缺少到四个新概念/技术页的 wikilink → 已补
- **traffic-forecasting.md**：frontmatter 重复 `last_updated` 字段 → 已去重；`source_count: 45` → 44（实际唯一脚注数=44）

### 幻觉检查（已验证通过）
- 作者 Ji Qi, Mingxiao Liu, Tam Thuc Do, Yuzhe Li, Zhuoshi Pan, Gene Cheung, H. Vicky Zhao (Tsinghua & York) ✓
- ICML 2026 ✓
- DGLR / DGTV 公式、Theorem 3.1 ✓
- 5 ADMM blocks, 25 layers, K=6, H=4 ✓
- 38K params, PDFormer 1,404K ✓
- 数据集 PEMS03 (358 nodes, 547 edges), METR-LA (207 nodes, 1,315 edges) ✓
- 推理计算量 0.087 GFLOPs vs PDFormer 1.771 (4.9%) ✓
- 训练设置：70 epochs, 6:2:2 split, Adam LR 5×10⁻⁴, Huber loss δ=1 ✓

### 仍存风险
- 论文自身 7.2% vs 38/1404=2.7% 矛盾：source-summary 跟随论文声称，concept 页面使用数学正确值加注
- algorithm-unrolling `confidence: high` + `source_count: 2` 为阈值边界，如再添加来源建议维持

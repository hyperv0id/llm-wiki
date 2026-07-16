# Ingest 报告：ST-SSDL

## 创建

- `wiki/source-st-ssdl.md` — WHY：论文源文件摘要（NeurIPS 2025），需按 ingest 规范为每个 raw/ 文件创建对应的 source-summary 页面
- `wiki/st-ssdl.md` — WHY：ST-SSDL 是首个将自监督偏差学习引入时空预测的框架，具有独立实体地位
- `wiki/ssdl.md` — WHY：Self-Supervised Deviation Learning 是论文的核心方法贡献（历史锚点 + 原型离散化 + 双损失），构成可复用的技术
- `wiki/relative-distance-consistency.md` — WHY：相对距离一致性是 SSDL 的理论基础原则，具有独立概念价值，可跨方法复用

## 修改

- `wiki/spatiotemporal-deviation.md` — WHY：原有定义仅覆盖 BiST 的输入-标签偏差视角，ST-SSDL 补充了输入-历史偏差视角，两者互补；source_count 1→2，confidence medium→high
- `wiki/contrastive-learning.md` — WHY：SSDL 使用 prototype triplet loss 作为潜在空间离散化手段，是 contrastive learning 在时空预测中的新应用场景；source_count 3→4
- `wiki/index.md` — WHY：新增 4 个页面需在对应类别登记（Sources/Entities/Concepts/Techniques），更新 spatiotemporal-deviation 摘要

## 新建交叉链接

- `st-ssdl` ↔ `ssdl` ↔ `relative-distance-consistency` — 框架/技术/原则三层级
- `st-ssdl` → `spatiotemporal-deviation` — 框架与概念关联
- `ssdl` → `contrastive-learning` — 对比损失在 SSDL 中的应用
- `spatiotemporal-deviation` ↔ `st-ssdl` — 双向补充：BiST vs ST-SSDL 两种偏差视角
- `st-ssdl` → `traffic-forecasting` — 应用领域关联

## Lint 修复 (2026-07-21)

- `wiki/ssdl.md` — 移除断链 [[historical-anchor]]
- `wiki/st-ssdl.md` — 修正 AGCRN "第二轻量" 排名断言（论文内部 Table 7 矛盾），保留 66% 数字比较
- `wiki/source-st-ssdl.md` — 非周期性数据局限标注为编辑性分析，非论文原文声明
- `wiki/relative-distance-consistency.md` — 修正 "速度下降 40 km/h" 为中性措辞
- `wiki/traffic-forecasting.md` — 补 Self-Supervised Deviation Learning 小节 + [^src-st-ssdl] 反向链接，source_count: 46→47

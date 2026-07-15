# Ingest Report: MiniTraffic (ICML 2026)

## 创建

- **wiki/source-minitraffic.md** — WHY：论文来源摘要页，总结 MiniTraffic 的核心贡献（FDA + Contrastive Clustering + Granularity-Aware Fine-Tuning）、方法要点、实验结果与局限性
- **wiki/minitraffic.md** — WHY：MiniTraffic 实体页，首个细粒度交通预测轻量预训练模型（~119K 参数），记录架构设计、性能数据、与 GPT-ST/McgVAE/FlashST 的关系
- **wiki/frequency-domain-stability-augmentation.md** — WHY：FDA 技术页，频域有界扰动机制（幅值约束 λ + 选择性掩码 Γ）的理论基础与实现细节
- **wiki/fine-grained-traffic-prediction.md** — WHY：细粒度交通预测概念页，形式化道路-车道双图层级结构、三大核心挑战、与大规模城市预测的区别
- **wiki/mcgvae.md** — WHY：McgVAE 实体页（CIKM 2024），首个道路-车道联合建模方法，MiniTraffic 的主要基线对比

## 修改

- **wiki/gpt-st.md** — WHY：在 Related Pages 中添加 MiniTraffic 交叉引用（同为预训练方法，MiniTraffic 在实验中以 GPT-ST 为基线）
- **wiki/traffic-forecasting.md** — WHY：新增 Fine-Grained / Multi-Granularity Prediction 子章节，介绍细粒度交通预测问题、McgVAE 与 MiniTraffic 的关系
- **wiki/contrastive-learning.md** — WHY：添加 MiniTraffic 的对比聚类应用案例（patch 级 InfoNCE + k-NN 图构建），更新 Applications 与 Related 列表
- **wiki/index.md** — WHY：将 5 个新页面添加到各自类别
- **wiki/log.md** — WHY：记录本次 ingest 操作

## 新建交叉链接

- [[minitraffic]] ↔ [[frequency-domain-stability-augmentation]]
- [[minitraffic]] ↔ [[fine-grained-traffic-prediction]]
- [[minitraffic]] ↔ [[mcgvae]]
- [[minitraffic]] ↔ [[gpt-st]]
- [[traffic-forecasting]] ↔ [[fine-grained-traffic-prediction]]
- [[traffic-forecasting]] ↔ [[minitraffic]]
- [[contrastive-learning]] ↔ [[minitraffic]]

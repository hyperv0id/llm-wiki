# Ingest 报告：DPGNet (ICLR 2026 under review)

## 创建
- **wiki/source-dpgnet.md** — WHY：源文件摘要，记录 DPGNet 的核心贡献（AGL + ASL）、实验设置、限制与注意事项（未 peer-reviewed）
- **wiki/dpgnet.md** — WHY：实体页面，完整记录 DPGNet 四层架构（Embedding→AGL→ASL→Output）、性能结果、AGL 插拔实验和消融分析
- **wiki/adaptive-graph-learner.md** — WHY：AGL 是 DPGNet 的首要贡献——首个将 gated self-attention 用于动态图学习的插拔式模块，可直接替换 GWNet 等模型的静态图生成器
- **wiki/adaptive-season-learner.md** — WHY：ASL 首次在统一架构中融合时序分解、多尺度处理和模式特定图构建，与 Autoformer/TimeMixer/DST-Mamba 形成演进对比

## 修改
- **wiki/gwnet.md** — WHY：在 Limitations 中新增 AGL 如何解决 GWNet 静态邻接矩阵限制；在 Legacy 表中新增 DPGNet 条目（AGL 替换 E₁E₂ᵀ，MAE ↓3.52%–5.51%）；新增 Related Pages 链接
- **wiki/dcrnn.md** — WHY：在 Limitations 中链接 AGL 作为 DCRNN 静态图问题的最新解决方案；在 Legacy 表中新增 DPGNet 条目；新增 Related Pages 链接

## 新建交叉链接
- [[gwnet]] ↔ [[adaptive-graph-learner]] — GWNet 静态邻接矩阵 → AGL 动态图替代
- [[dcrnn]] ↔ [[adaptive-graph-learner]] — DCRNN 预定义图 → AGL 动态图演进
- [[dpgnet]] ↔ [[adaptive-graph-learner]] — 父模型 ↔ 核心子模块
- [[dpgnet]] ↔ [[adaptive-season-learner]] — 父模型 ↔ 核心子模块
- [[dpgnet]] ↔ [[autoformer]] — ASL 时序分解继承自 Autoformer
- [[dpgnet]] ↔ [[timemixer]] — ASL 多尺度处理继承自 TimeMixer
- [[dpgnet]] ↔ [[dst-mamba]] — ASL 与 DST-Mamba 的 trend+seasonal 分解对比
- [[dpgnet]] ↔ [[traffic-forecasting]] — 交通预测方法家族
- [[dpgnet]] ↔ [[spatio-temporal-decomposition]] — 时空分解概念

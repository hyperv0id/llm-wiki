# Ingest 报告：QDF (Quadratic Direct Forecast)

## 创建
- **wiki/source-qdf.md** — WHY: QDF 论文 source-summary，覆盖二次型 NLL 动机、双层学习 Σ、实验与局限；slug=`src-qdf`
- **wiki/qdf.md** — WHY: 方法实体页，便于图谱中与 FreDF/DistDF/TQNet/iTransformer 交叉链接
- **wiki/quadratic-form-weighted-objective.md** — WHY: 核心技术——$L_\Sigma$ 二次型加权目标与 Alg.1/2 双层学习流程
- **wiki/heterogeneous-task-weights.md** — WHY: QDF 新强调的概念——多步预测非均匀任务权重（Σ^{-1} 对角元），与 label autocorrelation 正交

## 修改
- **wiki/index.md** — WHY: 登记 source-qdf、qdf、heterogeneous-task-weights、quadratic-form-weighted-objective
- **wiki/log.md** — WHY: 记录 ingest 操作
- **wiki/source-fredf.md** — WHY: 补充 QDF 作为同作者组后续批评（残差偏相关 + 等权分量）与兄弟目标交叉链接
- **wiki/source-distdf.md** — WHY: 补充 QDF 作为同组 sibling（二次型似然族 vs OT 对齐）交叉链接
- **wiki/fredf.md** — WHY: Relation to Later Work 增加 QDF；Links 增加 sibling objectives
- **wiki/label-autocorrelation.md** — WHY: Mitigations 表增加 QDF 行；Related/脚注接入 src-qdf；source_count 1→2
- **wiki/direct-forecast.md** — WHY: Likelihood Gap 补充 QDF 路径；Related 扩展；source_count 1→2
- **wiki/autocorrelation-bias.md** — WHY: Resolution Paths 并列 DistDF 与 QDF；source_count 2→3
- **wiki/joint-distribution-wasserstein-alignment.md** — WHY: Related Techniques 增加 QDF 对比

## 新建交叉链接
- [[source-qdf]] ↔ [[qdf]]
- [[source-qdf]] ↔ [[quadratic-form-weighted-objective]]
- [[source-qdf]] ↔ [[heterogeneous-task-weights]]
- [[source-qdf]] ↔ [[source-fredf]] / [[source-distdf]]
- [[qdf]] ↔ [[fredf]] / [[label-autocorrelation]] / [[direct-forecast]] / [[autocorrelation-bias]]
- [[quadratic-form-weighted-objective]] ↔ [[frequency-enhanced-direct-forecast]] / [[joint-distribution-wasserstein-alignment]]
- [[heterogeneous-task-weights]] ↔ [[label-autocorrelation]] / [[qdf]]

## 未创建（已有足够覆盖）
- 未新建 Time-o1 独立页：文中仅作为 FreDF 同类“边缘去相关”对照出现
- 未新建独立 meta-learning 页：MAML/Reptile 仅作 QDF 实现灵活性对照
- 未修改 raw/：按不可变策略只读外部 PDF 完成 ingest

## 源文件
- 外部路径：`/run/media/jcheng/WD-Data/yjs/INSIS/多模态外生信息引导的长期时空预测/QDF_Wang_2026_ICLR.pdf`
- arXiv:2511.00053v1 (28 Oct 2025)

## 补全缺口 — 2026-07-21

### 修复
- **wiki/source-qdf.md** — WHY: confidence high→medium（lint：source_count=1 不允许 high）
- **wiki/qdf.md** — WHY: 同上
- **wiki/quadratic-form-weighted-objective.md** — WHY: 同上
- **wiki/joint-distribution-wasserstein-alignment.md** — WHY: 移除 broken wikilink `[[source-time-o1]]`→plain "Time-o1"（Time-o1 无独立页面，原始 ingest 决定不创建）

## 深度 Lint — 2026-07-21

### 严重（已修复）
- **wiki/source-qdf.md** — WHY: 自引用循环（6 处 `[^src-qdf]` + 脚注定义）。Source-summary 不应自引用（前例：ProbTS lint 2026-07-16）。移除全部自引用，source_count: 1→0，confidence: medium→low

### 警告（已修复）
- **wiki/joint-distribution-wasserstein-alignment.md** — WHY: confidence:high 但 source_count:1 违规 → confidence: medium

### 信息（已修复）
- **wiki/qdf.md** — WHY: "Same author line as FreDF and DistDF" 不够准确（共享 4/10 作者）→ "Overlapping author group with"

### 幻觉交叉验证通过
对照 PDF (pdftotext) 逐条验证：作者（10 人）/机构/arXiv ID/方法名/ECL 61.4% 偏相关/PEMS08 0.120 vs 0.139/<2ms at T=720/MAML 对比/Cholesky+softplus/收敛条件 ‖Σ‖_F<1e-4/K-fold 时序划分/hypernetwork 扩展方向——全部与原文一致。

### 仍存风险
- source-qdf.md 移除自引用后 source_count:0 + confidence:low，待该论文被其他源引用后再升级 confidence

# Ingest 报告：FreDF (Frequency-enhanced Direct Forecast)

## 创建
- **wiki/source-fredf.md** — WHY: FreDF 论文 source-summary，覆盖标签自相关问题形式化、频域 DF 损失、实验与局限
- **wiki/fredf.md** — WHY: 方法实体页，便于图谱中与 iTransformer/FEDformer/FreTS/DistDF 交叉链接
- **wiki/label-autocorrelation.md** — WHY: 核心概念——多步标签序列条件依赖；FreDF 的问题设定，可被 DistDF 等后续工作复用
- **wiki/direct-forecast.md** — WHY: DF vs IF 预测范式概念页，解释 FreDF 升级的对象
- **wiki/frequency-enhanced-direct-forecast.md** — WHY: FreDF 训练技术细节（FFT 损失、α 混合、2D/多项式基变体）

## 修改
- **wiki/index.md** — WHY: 登记 source-fredf、fredf、label-autocorrelation、direct-forecast、frequency-enhanced-direct-forecast
- **wiki/log.md** — WHY: 记录 ingest 操作
- **wiki/autocorrelation-bias.md** — WHY: 补充 FreDF 作为问题先导与 DistDF 批评对象的交叉引用；source_count 1→2

## 新建交叉链接
- [[source-fredf]] ↔ [[fredf]]
- [[source-fredf]] ↔ [[label-autocorrelation]]
- [[source-fredf]] ↔ [[direct-forecast]]
- [[source-fredf]] ↔ [[frequency-enhanced-direct-forecast]]
- [[fredf]] ↔ [[itransformer]] / [[fedformer]] / [[frets]] / [[autoformer]]
- [[label-autocorrelation]] ↔ [[autocorrelation-bias]]
- [[autocorrelation-bias]] ↔ [[source-fredf]] / [[fredf]]
- [[frequency-enhanced-direct-forecast]] ↔ [[frequency-enhanced-block]] / [[fedformer]] / [[frets]]
- [[fredf]] ↔ [[source-distdf]]（后续批评：仅边缘去相关）

## 未创建（已有足够覆盖）
- 未新建独立 DML 页：DML 仅作为 FreDF 的验证工具出现
- 未修改 raw/：PDF 已在 raw/FreDF_Wang_2025_ICLR.pdf，按不可变策略只读外部路径完成 ingest

## Lint 修复 (2026-07-18)

### 幻觉/错引修复
- **source-fredf.md**：移除自引用 `[^src-fredf]`，source_count 1→0；DistDF/QDF 批评移除 `[^src-fredf]` 错引（FreDF 论文早于两者）
- **fredf.md**：Relation to Later Work 中 DistDF/QDF 批评 `[^src-fredf]`→`[^src-distdf]`/`[^src-qdf]`，source_count 1→3；broken wikilink `[[frets]]`→`[[source-frets]]`
- **frequency-enhanced-direct-forecast.md**：Limitations 中 DistDF 批评 `[^src-fredf]`→`[^src-distdf]`，source_count 1→2；broken wikilink `[[frets]]`→`[[source-frets]]`

### 结构修复
- source-fredf.md confidence: high→medium（source-summary，source_count:0）
- fredf.md confidence:high 保留（source_count 1→3，满足 ≥2 条件）
- frequency-enhanced-direct-forecast.md confidence:high 保留（source_count 1→2，满足 ≥2 条件）

### 仍存风险
- itransformer/fedformer/autoformer/frequency-enhanced-block 缺少 FreDF 反向链接
- `wiki/frets.md` entity 页面缺失（预存问题，rstib-mlp.md 同样受影响）

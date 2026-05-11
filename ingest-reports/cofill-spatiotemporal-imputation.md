# Ingest 报告：cofill-spatiotemporal-imputation

## 创建

- **wiki/source-cofill-spatiotemporal-imputation.md** — 论文摘要页面
  - WHY: 每一篇摄入的源文件必须有对应的 source-summary 页面

- **wiki/cofill.md** — CoFILL 实体页面
  - WHY: CoFILL 是核心创新方法，值得作为实体页面详细记录其架构和贡献

- **wiki/dual-stream-temporal-frequency-processing.md** — 双流时频处理技术页面
  - WHY: 双流时频融合是 CoFILL 的核心技术贡献，值得作为独立技术页面记录

## 修改

- **wiki/index.md** — 添加三条目
  - WHY: 新创建的页面需要加入 index 目录以保持可发现性

- **wiki/log.md** — 追加 ingest 记录
  - WHY: 每次 ingest 必须记录到 log.md

## 新建交叉链接

- [[cofill]] → [[diffusion-model]] — 扩散模型理论基础
- [[cofill]] → [[spatio-temporal-foundation-model]] — 时空基础模型
- [[cofill]] → [[generative-time-series-forecasting]] — 生成式预测
- [[cofill]] → [[traffic-forecasting]] — 交通预测场景
- [[dual-stream-temporal-frequency-processing]] → [[timesnet]] — 2D-variation 类比
- [[dual-stream-temporal-frequency-processing]] → [[frequency-enhanced-attention]] — 频域注意力对比
- [[dual-stream-temporal-frequency-processing]] → [[spectral-recurrent-encoder]] — 谱域编码对比
- [[dual-stream-temporal-frequency-processing]] → [[adaptive-frequency-fusion]] — 自适应频融对比

## 与现有页面的关系

1. **与 SpecSTG 的关联**：两者都是扩散+时空图的方法，但 SpecSTG 在谱域执行扩散，而 CoFILL 在原始域通过双流架构融合频域信息。CoFILL 填补的是**数据填补**任务，而非预测任务。

2. **与 PriSTI/CSDI 的关联**：CoFILL 相比 PriSTI 提升了 10%+，且不使用有问题的 forward interpolation 作为预填补。

3. **与 DDPM 的关系**：CoFILL 使用条件扩散框架进行填补，是扩散模型在时空数据填补领域的具体应用。

## 覆盖的引用

- CoFILL 论文唯一引用 `[^src-cofill-spatiotemporal-imputation]`
- 在 4 个新页面中分布：source-summary(1) + entity(1) + technique(1) + cofill(1) = 4 处引用
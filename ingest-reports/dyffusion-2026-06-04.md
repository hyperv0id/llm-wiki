# Ingest 报告：dyffusion

## 源文件
- `raw/dyffusion.pdf` — Cachay et al., "DYffusion: A Dynamics-informed Diffusion Model for Spatiotemporal Forecasting", NeurIPS 2023

## 创建
- `wiki/source-dyffusion.md` — WHY：论文源摘要，涵盖两阶段训练框架、Cold Sampling 推理、实验和消融
- `wiki/dyffusion.md` — WHY：DYffusion 框架实体，非高斯扩散/数据空间操作的代表性方法，与 TEDM、MCVD、DDPM 的关系对比
- `wiki/cold-sampling.md` — WHY：Cold Sampling 技术，DDIM 在广义扩散模型上的推广，是 DYffusion 的关键推理算法，其 Euler 方法等价性具有理论意义

## 修改
- `wiki/diffusion-models.md` — WHY：新增"非高斯扩散（广义扩散模型）"小节，收录 DYffusion 和 Cold Sampling；source_count 1→2，confidence medium→high
- `wiki/generative-time-series-forecasting.md` — WHY：方法列表中新增 DYffusion 条目和对比表行；source_count 5→6
- `wiki/index.md` — WHY：新页面索引注册
- `wiki/log.md` — WHY：按时间顺序记录本次摄取

## 新建交叉链接
- [[dyffusion]] ↔ [[diffusion-models]] — 非高斯扩散的典型实例
- [[dyffusion]] ↔ [[generative-time-series-forecasting]] — 扩散式时序预测谱系
- [[cold-sampling]] ↔ [[dyffusion]] — 采样算法与框架的关系

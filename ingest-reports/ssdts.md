# Ingest 报告：SSD-TS (ssdts)

## 创建
- wiki/source-ssdts.md — WHY：source-summary 页面，按项目规范记录 SSD-TS 论文的核心贡献、架构、实验和局限。
- wiki/ssd-ts.md — WHY：entity 页面，SSD-TS 是首个将 Mamba 用作扩散模型去噪 backbone 的工作，在 Mamba 家族中具有独立的技术节点价值。
- wiki/bam.md — WHY：technique 页面，BAM（Bidirectional Attention Mamba）是 SSD-TS 的核心模块，提供了双向 Mamba + temporal attention 的通道内建模方案，其他扩散/时序模型可能参考。
- wiki/cmb.md — WHY：technique 页面，CMB（Channel Mamba Block）证明了 SSM 在通道间建模中优于注意力机制和 SENet，是一个独立可复用的技术模式。

## 修改
- wiki/mamba.md — WHY：新增 "Mamba as Diffusion Backbone" 章节和 SSD-TS 相关页面链接（BAM, CMB, SSD-TS），将 Mamba 的应用边界从确定性预测扩展到概率性扩散生成。
- wiki/s-mamba.md — WHY：在 Mamba 家族演进表中插入 SSD-TS 行，更新相关页面链接，标记 SSD-TS 为"首个扩散 backbone"里程碑。
- wiki/csdi.md — WHY：在后续影响章节添加 SSD-TS 作为 Transformer→S4→Mamba backbone 演进的关键一环，并增加 [[ssd-ts]], [[bam]], [[cmb]] 的关联链接。frontmatter 的 source_count 递增。
- wiki/index.md — WHY：在 Sources/Entities/Techniques 三个类别中分别添加 source-ssdts, ssd-ts, bam, cmb。
- wiki/log.md — WHY：记录 ingest 操作、创建/更新的页面。

## 新建交叉链接
- [[ssd-ts]] ↔ [[mamba]] — SSD-TS 使用 Mamba 作为 backbone
- [[ssd-ts]] ↔ [[bam]] — BAM 是 SSD-TS 的 intra-channel 模块
- [[ssd-ts]] ↔ [[cmb]] — CMB 是 SSD-TS 的 inter-channel 模块
- [[ssd-ts]] ↔ [[csdi]] — SSD-TS 是 CSDI 的 Mamba backbone 后继
- [[ssd-ts]] ↔ [[s-mamba]] — 同为 Mamba 在时序领域的应用，S-Mamba 做预测，SSD-TS 做扩散插补
- [[bam]] ↔ [[mamba]] — BAM 基于 Mamba 的 bidirectional 扩展
- [[cmb]] ↔ [[mamba]] — CMB 基于 Mamba 的 channel 维度扫描
- [[bam]] ↔ [[cmb]] — 互补的 intra/inter-channel 模块对

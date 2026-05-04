# Ingest 报告：数学直觉系列（二）：VAE与重参数化

## 创建
- wiki/source-bluuuuue-reparameterization-trick.md — WHY：源文件摘要，记录核心论点（采样不可导→REINFORCE高方差→两步分离→双重功效→适用前提→应用场景）
- wiki/reparameterization-trick.md — WHY：核心技巧页面，系统整理重参数化的数学原理、双重功效、适用前提、应用场景、与6个现有概念的关系

## 修改
- wiki/variational-autoencoder.md — WHY：链接重参数化技巧为独立页面，添加方差降低理论保证（Xu et al. 2019），source_count 1→2
- wiki/elbo.md — WHY：在重要性列表中将"重参数化技巧"链接为 wikilink 并补充方差降低细节，source_count 1→2
- wiki/diffusion-model.md — WHY：在前向过程中解释重参数化使每步可导是端到端训练的保证，添加 reparameterization-trick 交叉链接，source_count 9→10
- wiki/ddpm-simplified-training-objective.md — WHY：在数学推导中链接重参数化技巧页面，添加相关概念章节，source_count 1→2

## 新建交叉链接
- [[reparameterization-trick]] ↔ [[variational-autoencoder]] — VAE 训练的核心依赖
- [[reparameterization-trick]] ↔ [[elbo]] — ELBO 可优化性的前提
- [[reparameterization-trick]] ↔ [[diffusion-model]] — 前向过程可微性保证
- [[reparameterization-trick]] ↔ [[ddpm-simplified-training-objective]] — L_simple 中加噪可微的数学前提
- [[reparameterization-trick]] ↔ [[score-function]] — 重参数化绕过得分函数的高方差问题
- [[reparameterization-trick]] ↔ [[scaling-factor-sqrt-dk]] — 同系列文章，两者都是数值/梯度稳定性的结构性方案

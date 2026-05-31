# Ingest 报告：Rectified Flow (Liu et al., arXiv 2022)

## 创建
- [[source-rectified-flow]] — WHY：首次 ingest Rectified Flow 论文，作为源文件摘要页面提供核心贡献、实验数据和影响分析
- [[rectified-flow]] — WHY：Rectified Flow 是 ODE 生成模型中的重要技术，通过 rectification 学习直线轨迹实现少步生成，启发了 InstaFlow 和 SD3/FLUX 的流匹配范式

## 修改
- [[flow-matching]] — WHY：添加 Rectified Flow 在介绍段落（作为 FM 的后续工作）和相关页面部分的双向链接
- [[optimal-transport]] — WHY：Rectified Flow 在独立耦合下收敛到 OT 映射，是 OT 理论在生成模型中的关键应用
- [[urbandit]] — WHY：UrbanDiT 使用 InstaFlow（基于 Rectified Flow）训练，第 41 行已有纯文本提及，转为 wikilink
- [[shortcut-models]] — WHY：Shortcut Models 继承 Rectified Flow 的直线轨迹思想，添加交叉引用
- [[consistency-models]] — WHY：Consistency Models 同为少步生成方法，Rectified Flow 是重要的对照技术
- [[diffusion-model]] — WHY：在相关概念部分添加 Rectified Flow 链接，作为扩散模型加速采样的重要分支

## 新建交叉链接
- [[rectified-flow]] ↔ [[flow-matching]]
- [[rectified-flow]] ↔ [[optimal-transport]]
- [[rectified-flow]] ↔ [[urbandit]]
- [[rectified-flow]] ↔ [[shortcut-models]]
- [[rectified-flow]] ↔ [[consistency-models]]
- [[rectified-flow]] ↔ [[diffusion-model]]
- [[rectified-flow]] ↔ [[instaflow]]（被动，InstaFlow 页面已有引用）

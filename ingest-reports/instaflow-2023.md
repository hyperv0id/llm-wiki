# Ingest 报告：InstaFlow (Liu et al., ICLR 2024)

## 创建
- wiki/source-instaflow.md — WHY：InstaFlow 论文的 source-summary 页面，总结 reflow+distill 管线、关键实验证据和局限性
- wiki/instaflow.md — WHY：InstaFlow 技术的 entity 页面，详述 Text-Conditioned Reflow、蒸馏、Stacked U-Net 和与 Consistency Models/Shortcut Models 的对比

## 修改
- wiki/index.md — WHY：在 Sources 和 Entities 分类中添加新条目
- wiki/log.md — WHY：记录本次 ingest 操作
- wiki/urbandit.md — WHY：UrbanDiT 正文已提及 InstaFlow，添加 `[[instaflow]]` wikilink
- wiki/source-urbandit.md — WHY：同上，source-summary 中已提及 InstaFlow
- wiki/flow-matching.md — WHY：Flow Matching 页面中已有 OT path 内容，添加 InstaFlow 作为蒸馏应用入口
- wiki/diffusion-model.md — WHY：扩散模型局限性中提及采样加速方法，添加 InstaFlow 作为一步蒸馏路径
- wiki/ddpm.md — WHY：DDPM 的原始 1000 步采样正是 InstaFlow 要超越的基线
- wiki/probability-flow-ode.md — WHY：SD 的 PF-ODE 轨迹弯曲正是 InstaFlow reflow 要拉直的核心问题
- wiki/dpm-solver.md — WHY：DPM-Solver 25 步是 SD 在 InstaFlow 中的采样方式
- wiki/consistency-models.md — WHY：Consistency Models 是另一种一步生成方法，互为补充
- wiki/shortcut-models.md — WHY：Shortcut Models 的自一致性方法与 InstaFlow 的 reflow+distill 路线形成对比
- wiki/classifier-free-guidance.md — WHY：InstaFlow 的 CFG 版本 v^α = αv(·|T) + (1-α)v(·|NULL) 基于此
- wiki/optimal-transport.md — WHY：reflow 中降低凸传输代价的理论基础

## 新建交叉链接
- [[source-instaflow]] ↔ [[instaflow]]
- [[instaflow]] ↔ [[flow-matching]]
- [[instaflow]] ↔ [[diffusion-model]]
- [[instaflow]] ↔ [[consistency-models]]
- [[instaflow]] ↔ [[shortcut-models]]
- [[instaflow]] ↔ [[urbandit]]
- [[instaflow]] ↔ [[probability-flow-ode]]
- [[instaflow]] ↔ [[dpm-solver]]
- [[instaflow]] ↔ [[ddpm]]
- [[instaflow]] ↔ [[classifier-free-guidance]]
- [[instaflow]] ↔ [[optimal-transport]]

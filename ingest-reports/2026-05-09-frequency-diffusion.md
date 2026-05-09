# Ingest 报告：Shaping Inductive Bias in Diffusion Models through Frequency-Based Noise Control

## 创建

- wiki/source-2502-10236.md — WHY：论文的 source-summary，涵盖核心论点、方法、实验结果和局限性
- wiki/frequency-diffusion.md — WHY：论文的核心技术贡献，频域扩散方法的具体实现
- wiki/frequency-based-noise-control.md — WHY：频域噪声控制概念页面，独立于具体技术实现的抽象概念
- wiki/inductive-bias-shaping.md — WHY：归纳偏置塑造概念页面，将论文的核心思想抽象为更广泛的概念
- wiki/two-band-mixture-noise.md — WHY：两频段混合噪声技术页面，论文实验中使用的具体噪声构造方式

## 修改

- wiki/diffusion-model.md — WHY：添加"前向过程的归纳偏置塑造"章节、频域扩散到关键实现列表、交叉引用
- wiki/edm-design-space.md — WHY：添加"与频域噪声控制的关系"章节，说明两者的正交性；添加频域噪声控制到链接

## 新建交叉链接

- [[frequency-diffusion]] ↔ [[frequency-based-noise-control]]
- [[frequency-based-noise-control]] ↔ [[edm-design-space]]
- [[frequency-diffusion]] ↔ [[diffusion-model]]
- [[inductive-bias-shaping]] ↔ [[spurious-patterns]]（捷径学习）
- [[frequency-diffusion]] ↔ [[adaptive-frequency-modulation]]（对比不同频域方法）
- [[two-band-mixture-noise]] ↔ [[frequency-diffusion]]
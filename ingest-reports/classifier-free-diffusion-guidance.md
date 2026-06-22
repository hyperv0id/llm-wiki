# Ingest 报告：Classifier-Free Diffusion Guidance (Ho & Salimans, 2022)

## 创建
- wiki/source-classifier-free-diffusion-guidance.md — WHY：该论文是 classifier-free guidance 的原始提出者，是扩散模型条件生成领域的基础文献，此前 wiki 中的 CFG 页面未引用原始论文

## 修改
- wiki/classifier-free-guidance.md — WHY：以原始论文为主源重写核心章节（数学形式、隐式分类器解释、联合训练、原始基准结果、直觉解释），补充 $p_\text{uncond}$ 超参数分析、连续时间框架细节、非保守场论证等原创贡献
- wiki/classifier-guidance.md — WHY：添加 Ho & Salimans 对分类器引导的三点批判（对抗性疑虑、GAN 相似性、额外分类器成本），source_count 从 2 → 3

## 新建交叉链接
- [[classifier-free-guidance]] ↔ [[source-classifier-free-diffusion-guidance]]
- [[classifier-guidance]] ↔ [[source-classifier-free-diffusion-guidance]]

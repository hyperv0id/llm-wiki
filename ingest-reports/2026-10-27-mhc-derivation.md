# 推导报告：mhc-manifold-constrained-hyper-connections（优化推导轮）

源：raw/mhc-2512-24880.pdf（arXiv:2512.24880v2，19 页，pdftotext 全文 1158 行，逐条对照）。

## 修改

- `wiki/manifold-constrained-hyper-connections.md` — WHY：优化推导主页面。式 (3)(4)(6) 补原文编号并补复合映射展开式；「理论性质」改为定理/论断归属版（论文无定理环境，三条均为 §4.1 论列）并补证明骨架；新增「恒等映射恢复的推导骨架」「Sinkhorn 迭代的收敛前提」两节；「参数化与投影」修正 $H^{post}$ 为 $2\sigma$（式 8）、补 mHC 相对 HC 去 tanh（式 5 vs 7）；「效率优化」补核数（3 系数核 + 2 应用核）、I/O 数字、$L_r^*$ 求导、Table 2/3 落位；「实验表现」补 Table 4 全表（含指标与 shot 数）、Fig. 2/3/5/6/7 落位、Table 1 消融、Table 5 超参。
- `wiki/source-mhc-manifold-constrained-hyper-connections.md` — WHY：方法段 Sigmoid 表述修正为 $\sigma$/$2\sigma$（式 8），三性质补「论列、未附证明」归属；单源 confidence high→medium。
- `wiki/birkhoff-polytope.md` — WHY：内容与本源一致，仅补指向 mHC 页证明骨架的链接；单源 confidence high→medium。
- `wiki/sinkhorn-algorithm.md` — WHY：mHC 小节补「论文未给收敛前提」、正矩阵前提为经典结果推断、$t_{max}=20$ 与有限迭代实测偏差（§5.4）。
- `wiki/identity-mapping-property.md` — WHY：mHC 恢复段补 3000→1.6 实测锚点（§5.4、Fig. 7）与推导骨架链接，与本源一致。

## 数值备注

- Fig. 2/3/5/6/7 为矢量图，图内数值在文本层不可读；0.021、3000、1.6、12k 步等关键数字均取自正文（§1、§3.1、§5.2、§5.4），页面已标 figure 编号。
- 式 (4) 复合展开的乘积上下标在 PDF 文本层为乱码，页面按 §1/§3.1 反复出现的复合项 $\prod_{i=1}^{L-l} H_{L-i}^{res}$ 与求和项语义重排为 $\prod_{j=1}^{L-1-i} H_{L-j}^{res}$，如需逐字核对请看 PDF 原式。
- 论文全文无 Theorem/Lemma/Proof 环境与附录证明（附录 A.1 仅为 Table 5 超参）；Sinkhorn 收敛断言无前提陈述，页面按「论文未给」处理，正矩阵前提标注为经典结果推断。
- Table 4 中 MATH 一项 mHC（26.0）低于 HC（26.4），正文以「surpasses HC on the majority of benchmarks」表述且未解释，页面如实并列。
- 6.7% 时间开销出自 §1 与 §4.3 开头，论文无独立的系统性能对比表。
- 本轮未跑 scripts/lint.py、未 git commit（任务约束）。

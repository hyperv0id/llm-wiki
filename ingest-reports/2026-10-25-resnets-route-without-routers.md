# Ingest 报告：Do ResNets Route? Sparse Interaction Experts in Residual Networks

源文件：`raw/resnets-route-without-routers.pdf`（arXiv:2610.02907v1，2026-10-02，PDF 46 页）→ 文本层 `raw/resnets-route-without-routers.txt`（pypdf 提取）。`*.pdf` 在 `.gitignore` 内，只提交 txt。

## 创建

- wiki/source-resnets-route-without-routers.md — WHY：库内首次出现该源；summary 覆盖问题（无 router 是否仍路由）、方法（残差掩码 + Möbius 反演）、证据（Table 1/4/5/8/11、Figure 2/3/5）与只覆盖两个 ImageNet 检查点的局限。
- wiki/residual-interaction-decomposition.md — WHY：掩码响应、Möbius 系数、逆关系与朴素聚合失效独立成页供复用，含 Stage 1–4 分阶段维度块。
- wiki/interaction-order-spectrum.md — WHY：阶谱统计量（$\widetilde E_k$、$C_{\le q}$、$\kappa$、$M_k$）、$\lambda=1$ 峰在中高阶、Proposition 2.3 与缩放扫描构成可独立引用的技术块，主要反直觉结论落在这里。
- wiki/implicit-residual-experts.md — WHY：Top-$K$ 支撑的集中性与长尾（Table 8/9/10）是本源的「专家」实证定义，与路由判据可分。
- wiki/implicit-functional-routing.md — WHY：路由判据、类结构（Table 11）、困难度不敏感与稀疏 MoE 的分界，是本源的核心概念贡献。

## 修改

- wiki/mixture-of-experts.md — WHY：原条目均为「先造专家再学 router」；补「反向视角」小节记录已训练 dense 网络中的隐式专家结构与它同 sparse execution 的区别，并加两条链接与脚注（source_count 13→14）。
- wiki/index.md — WHY：新增 1 条 Sources、2 条 Concepts、2 条 Techniques，按字母序插入。

## 新建交叉链接

- [[residual-interaction-decomposition]] ↔ [[interaction-order-spectrum]]
- [[residual-interaction-decomposition]] ↔ [[implicit-residual-experts]]
- [[interaction-order-spectrum]] ↔ [[implicit-residual-experts]]
- [[implicit-functional-routing]] ↔ [[mixture-of-experts]]
- [[implicit-functional-routing]] ↔ [[implicit-residual-experts]]
- [[source-resnets-route-without-routers]] ↔ 上述四页

## 数值备注

- **数据集口径冲突**：§3 称 ResNet-18 在「class-balanced ImageNet-2012 validation 子集（每类 10 张，$N=10{,}000$）穷举 256 个掩码」，Table 1 标题也写 validation images；附录 F.2、G、H 与 Table 8 则写 fixed set from the **unlabeled ImageNet-1K test split**。§4 又区分 Obs ❹ 用 test、Obs ❺–❻ 用 validation。阶谱与缩放扫描在哪个 split 上计算无法从原文唯一确定，此处并列报告，未替论文择一。
- 标量读数类差峰值 0.125 / 0.044 只见 Figure 5a,b；Table 11 只列向量读数的重叠统计（$\Delta_{\mathrm{class}}$ 0.0362 / 0.0321），标量值无法从表格复核。
- 「Top-64 contains 25.1% of the dictionary」即 64/255=25.098%，Table 8 记 25%，一致。
- 「38.3% captures 95% of the vector mass」（ResNet-34）对应 Table 10 的 38.296%；「99.495% … 10% vector error」与 Table 10 一致。
- PDF 文本层缺陷：变音符拆成 `M¨obius`、上下标粘连（`aSv`、`χ(λ)l`）、Table 4/5 的列按行串接。Table 4/5/6/7/8/9/10/11 数值经逐格核对可读；Figure 2/3/5/9/10 是图片，峰形只能引正文数字。
- 原文无独立 Limitations 章节；局限只见于 Conclusion 一段（"extending this analysis to deeper networks requires efficient estimators of high-order interactions"）与 Reproducibility/AI Use/Ethics 声明。

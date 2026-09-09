# Ingest 报告：GWNet 长青分析（2026-09-04）

操作类型：query → 归档（[[source-gwnet]] 已于 2026-05-31 完整 ingest，raw/ 无新源文件；本条按用户论点「2019 年模型在 2026 年仍能打」做证据整理与归档）。

## 创建
- **wiki/gwnet-longevity.md** — WHY：GWNet 的技术页与源摘要已存在，但「2026 年仍被用作基线/骨干」这一状态散落在各论文页中，无一处汇总；归档为 analysis 页，把证据拆为精度（STGformer Table I/II）、使用（SCALE/USTD/IGSTGNN 骨干）、修补（MAGE/BigST/RAGC/DPGNet/ST-OOD）三层，并明确边界（OOD +9.7%、10K 节点 OOM、纯精度竞赛已输），避免「能打」被误读为「仍是精度上限」。

## 修改
- **wiki/gwnet.md** — WHY：Legacy 段记录了谱系表但未指出 GWNet 自身 2024–2026 年的在场状态，补 [[gwnet-longevity]] 指路句与 Related Pages 条目。
- **wiki/stgformer.md** — WHY：本页 Table I/II 是长青分析的核心数字来源，补反向链接。
- **wiki/index.md** — WHY：Analysis 区新增条目。
- **wiki/log.md** — WHY：记录本次操作。

## 数字核实记录（对照 raw/ 原文）
- STGformer Table I（raw/stgformer-nie-2024.txt 行 511/527/543）：GWNET SD 311K 平均 MAE 17.74 / RMSE 29.62；BA 344K 20.91 / 33.41；LA 374K 21.20 / 33.58。BA 平均 RMSE 33.41 优于 STGformer 33.50（stgformer.md「论文报告的数字」节已逐格核对过）。
- STGformer Table II 跨年：LA GWNET 平均 27.20 / 40.85 / 22.51% vs STGformer 26.34 / 41.76 / 27.04%。
- SCALE（raw/2605.04957v1.md 行 1911–1936、2139）：E.1 稳健性消融以 tsl 默认 GWNet 为第一阶段骨干；Table 6 为 GWNet 骨干完整区间结果。
- MAGE（raw/mage-less-but-more-linear-adaptive-graph-learning.pdf）：附录 w/o ReLU 消融 GWNet MAE 18.07→17.97；正文报告 AGCRN/GWNet/D²STGNN 在 Shanghai Mobile（3,042）/Milan Internet（10,000 节点）上不可部署；17 数据集 / 14 基线 / 94%（48/51）指标 SOTA。
- ST-OOD 消融（wiki/source-st-ood.md 行 47）：GWNet 自适应邻接助 IN、OUT 误差约 +9.7%。
- USTD 骨干（wiki/source-ustd.md 行 35）、IGSTGNN 即插即用（wiki/source-incident-guided-st-forecasting.md 行 49）、RAGC 72% 参数（wiki/ragc.md 行 26）、BigST O(N)（wiki/gwnet.md 行 179）、DPGNet AGL 3.52–5.51%（wiki/gwnet.md 行 133）取自对应 wiki 页既有记载。

## 新建交叉链接
- [[gwnet-longevity]] ↔ [[gwnet]] — 长青状态归档在技术主页挂出
- [[gwnet-longevity]] ↔ [[stgformer]] — 精度证据来源
- [[gwnet-longevity]] ↔ [[mage]] / [[ragc]] / [[bigst]] / [[dpgnet]] / [[st-ood]] — 修补轴与边界证据
- [[gwnet-longevity]] ↔ [[scale]] / [[ustd]] / [[igstgnn]] — 2024–2026 骨干使用证据

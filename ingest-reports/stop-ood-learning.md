# Ingest 报告：STOP — Robust Spatio-Temporal Centralized Interaction for OOD Learning

## 背景
STOP (Jiaming Ma et al., USTC, ICML 2025) 于 2026-06-09 在批量 ingest 时首次入库（source-summary + entity + 4 technique 页面），但当时仅有 PMLR 在线版本、无本地 PDF。2026-07-23 用户提供完整 PDF（14 页正文 + appendix），本次补全 PDF 归档并验证/增强已有页面。

## 创建
- `raw/stop-robust-spatio-temporal-centralized-interaction-ood-learning.pdf` — WHY：PDF 首次归档到 raw/，使 source-stop 有物理源文件对应
- `wiki/spatio-temporal-ood-learning.md` — WHY：STOP 论文将 ST-OOD 明确定义为 temporal OOD + structural OOD 双轴问题，且 CaST/STONE/STBP/RSTIB-MLP/ST-TTC 多条线均围绕此概念展开，需要独立概念页统一引用

## 修改
- `wiki/source-stop.md` — WHY：新增「ST-OOD Evaluation Protocol」节（T-OOD/S-OOD 评估协议、六数据集详情），修正 source_count: 1→0（自引用不计入），last_updated: 2026-07-23
- `wiki/stop.md` — WHY：last_updated 刷新
- `wiki/centralized-message-passing.md` — WHY：last_updated 刷新
- `wiki/context-aware-units.md` — WHY：last_updated 刷新
- `wiki/generalized-perturbation-unit.md` — WHY：last_updated 刷新
- `wiki/ood-generalization.md` — WHY：添加 → spatio-temporal-ood-learning 链接，last_updated 刷新
- `wiki/distributionally-robust-optimization.md` — WHY：last_updated 刷新
- `wiki/index.md` — WHY：新增 spatio-temporal-ood-learning 条目
- `wiki/log.md` — WHY：记录本次补全操作

## 验证
- 对照 pdftotext 提取的 5993 行全文验证：作者/会议/方法/实验数字与已有页面一致，无需修正
- source-stop.md 中 T-OOD/S-OOD 评估协议细节（窗口=12/24，S-OOD 移除 10% + 新增 30% 节点）均与原文 Section 4.1 一致
- 所有 [^src-stop] 脚注指向 [[source-stop]]，wikilink 目标页面存在

## 新建交叉链接
- [[spatio-temporal-ood-learning]] ↔ [[ood-generalization]]
- [[spatio-temporal-ood-learning]] ↔ [[stop]]
- [[spatio-temporal-ood-learning]] ↔ [[centralized-message-passing]]
- [[stop]] → [[spatio-temporal-ood-learning]]
- [[ood-generalization]] → [[spatio-temporal-ood-learning]]

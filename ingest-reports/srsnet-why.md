# Ingest 报告：SRSNet / Selective Representation Spaces

## 创建
- **wiki/source-srsnet.md** — WHY: NeurIPS 2025 论文 source-summary（固定 patching 问题、SRS 三组件、SRSNet、实验与局限）；slug=`src-srsnet`
- **wiki/srsnet.md** — WHY: 方法实体页，SRS + MLP 实例，便于与 PatchTST/Crossformer 等图谱链接
- **wiki/selective-representation-space.md** — WHY: 核心概念页，自适应 patch 表示空间相对固定 adjacent patching 的范式
- **wiki/selective-patching.md** — WHY: 核心技术——stride-1 候选 + Scorer$_s$ + 可微 Argmax 选 patch（可重复）
- **wiki/dynamic-reassembly.md** — WHY: 核心技术——Scorer$_r$ + 可微 Argsort 学习 patch 顺序
- **wiki/adaptive-fusion.md** — WHY: 核心技术——常规与 selective patch 嵌入的 $\alpha$ 凸组合

## 修改
- **wiki/index.md** — WHY: 登记 source-srsnet、srsnet、selective-representation-space、selective-patching、dynamic-reassembly、adaptive-fusion
- **wiki/log.md** — WHY: 记录 ingest 操作
- **wiki/patchtst.md** — WHY: 后续影响/Connections 接入 SRS 作为固定 patching 的自适应升级；source_count 1→2
- **wiki/patch-based-tokenization.md** — WHY: 相关技术增加“固定→自适应”SRS 链路；source_count 6→7
- **wiki/crossformer.md** — WHY: 后续影响注明 SRS 将其作为 patch 骨干插件并有增益；source_count 3→4
- **wiki/channel-independence.md** — WHY: 2026-07-21 补全，SRS 在 CI 设定下做 selective patch (source_count 10→11)
- **wiki/instance-normalization.md** — WHY: 2026-07-21 补全，SRS 采用 RevIN 预处理 (source_count 6→7)
- **全部 6 个 SRS 页面** — WHY: 2026-07-21 frontmatter last_updated 刷新

## 新建交叉链接
- [[source-srsnet]] ↔ [[srsnet]]
- [[source-srsnet]] ↔ [[selective-representation-space]] / [[selective-patching]] / [[dynamic-reassembly]] / [[adaptive-fusion]]
- [[srsnet]] ↔ [[patchtst]] / [[crossformer]] / [[patch-based-tokenization]] / [[channel-independence]] / [[instance-normalization]]
- [[selective-representation-space]] ↔ [[patch-based-tokenization]] / [[patchtst]] / [[crossformer]]
- [[selective-patching]] ↔ [[dynamic-reassembly]] ↔ [[adaptive-fusion]]
- [[srsnet]] ↔ [[channel-independence]] / [[instance-normalization]]
- [[selective-representation-space]] ↔ [[channel-independence]] / [[instance-normalization]]

## 未创建（已有足够覆盖）
- 未新建 xPatch / PatchMLP / TimeKAN / Amplifier 独立页：文中主要为 baseline/插件宿主，无足够新概念
- 未新建独立 “detach-reciprocal Hadamard” 页：作为 Selective Patching / Dynamic Reassembly 的实现细节保留在技术页
- 未修改 raw/：按不可变策略只读外部 PDF 完成 ingest

## 源文件
- 外部路径：`/run/media/jcheng/WD-Data/yjs/INSIS/多模态外生信息引导的长期时空预测/SRSNet_Wu_2025_NeurIPS.pdf`
- Raw 副本：`raw/srsnet-selective-representation-spaces-patch-perspective.pdf`（另有一份 `raw/SRSNet_Wu_2025_NeurIPS.pdf`）
- arXiv:2510.14510v5 (16 Dec 2025)；NeurIPS 2025
- Code: https://github.com/decisionintelligence/SRSNet

## 2026-07-21 Lint 修复

### 严重（已修复）
- [x] source-srsnet.md — 自引用循环（6 处 `[^src-srsnet]` + 脚注 `[[source-srsnet]]`），违反 source-summary 不自引用惯例。移除全部自引用；source_count: 1→0, confidence: high→low
- [x] srsnet.md — confidence:high + source_count:1 违规 → confidence: medium
- [x] selective-representation-space.md — confidence:high + source_count:1 违规 → confidence: medium
- [x] selective-patching.md — confidence:high + source_count:1 违规 → confidence: medium
- [x] dynamic-reassembly.md — confidence:high + source_count:1 违规 → confidence: medium
- [x] adaptive-fusion.md — confidence:high + source_count:1 违规 → confidence: medium

### 幻觉交叉验证：全部通过
对照 PDF (pdftotext) 逐条验证：作者 7 人(ECNU)、NeurIPS 2025、arXiv:2510.14510、方法名（SRS/Selective Patching/Dynamic Reassembly/Adaptive Fusion）、ETTh1 0.404/ETTh2 0.334/Solar 0.183/Traffic 0.392、horizons {96,192,336,720}/look-back {96,336,512}、baselines 列表、~10% memory/time + <5% MACs、C_{K+n-1}^{n}·n!、8 数据集——全部与 PDF 原文一致，无捏造。

### 已验证
- 更新页面（patchtst/patch-based-tokenization/crossformer/channel-independence/instance-normalization）source_count 与实际引用匹配
- 全部 6 个新页面已在 index.md 各类型下登记
- 所有 wikilink 目标页面均存在，无断链
- 无孤立页面

### 仍存风险
- source-srsnet.md: source_count:0 + confidence:low，待被其他源引用后升级
- 其余 5 个 SRS 页面: source_count:1 + confidence:medium，合规但待额外源文件加固置信度

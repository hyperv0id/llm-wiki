# Ingest 报告：storminsight

## 创建
- wiki/source-storminsight.md — WHY：source-summary 页面，总结 StormInsight 论文的核心贡献（三分量编码 + CFM 分层调制 + StormBench）和关键结果（MAE −12.4%, mCSI +34.0%）
- wiki/storminsight.md — WHY：实体页面，记录 StormInsight 模型的完整架构、两大核心挑战（异步跨层交互、环境依赖敏感性）及性能指标
- wiki/stormbench.md — WHY：实体页面，记录 StormBench 基准的数据构成（美法双区域多源观测 + 65+ ERA5 变量）、时空覆盖和任务设置
- wiki/environment-conditioned-nowcasting.md — WHY：概念页面，记录 StormInsight 引入的新范式——将临近预报从 2D 雷达外推重新定义为环境条件化的 3D 垂直动力学推理

## 修改
- wiki/precipitation-nowcasting.md — WHY：在方法演进部分加入 StormInsight 的多源数据融合+环境条件化路线，更新 frontmatter
- wiki/extreme-weather-forecasting.md — WHY：在单一事件类型方法中加入 StormInsight 的环境条件化临近预报路径，更新 frontmatter
- wiki/flow-matching.md — WHY：在相关页面中加入 StormInsight 的 Conditional Flow Matching 分层调制应用
- wiki/index.md — WHY：在 Sources、Entities、Concepts 三个类别中加入新页面条目

## 新建交叉链接
- [[storminsight]] ↔ [[precipitation-nowcasting]]
- [[storminsight]] ↔ [[environment-conditioned-nowcasting]]
- [[storminsight]] ↔ [[stormbench]]
- [[storminsight]] ↔ [[extreme-weather-forecasting]]
- [[storminsight]] ↔ [[flow-matching]]
- [[environment-conditioned-nowcasting]] ↔ [[precipitation-nowcasting]]

## Lint 修复 (2026-07-23)

### 幻觉检查
对照 PDF (pdftotext) 逐条验证：作者 8 人（Jun Chen, Yan Fang, Minghui Qiu, Yueran Qiu, Lin Chen, Shuxin Zhong, Yu Zhang, Kaishun Wu）、HKUST-GZ + 广州气象台、ICML 2026 PMLR 306、Kerrville 250mm/130+人、C1+C2 两大挑战、三分量编码（SetConv/MSIM/VAE/FiLM + MoE 四向跨层 + Multi-mesh Message Passing）、CFM Global/Local AdaLN、StormBench 美法双区域 ERA5 65 变量（8×5 压力层 + 25 单层）、384km×384km/550km×550km、2017-2019/2016-2018、推理 372.52±4.10ms（约 380ms）H100、9 基线、MAE −12.4%/mCSI +34.0%、消融（w/o AEE→CSI₂₁₉ 退化、w/o VIE→mCSI 退化、w/o 任意专家→性能下降）、局限（无完整 3D 扫描/无偏振雷达变量）。

全部与 PDF 原文一致，无捏造或错引。

### 严重（已修复）
- [x] source-storminsight.md — 自引用循环（3 处 `[^src-storminsight]` + 脚注定义），source-summary 不应自引用。移除全部自引用，source_count: 1→0，confidence: high→low

### 警告（已修复）
- [x] storminsight.md — confidence: high + source_count: 1 违规（实体页单源），confidence: high→medium
- [x] stormbench.md — confidence: high + source_count: 1 违规（实体页单源），confidence: high→medium

### 已验证
- environment-conditioned-nowcasting.md — source_count: 1, confidence: medium，合规
- precipitation-nowcasting.md — source_count: 3, confidence: high，合规
- extreme-weather-forecasting.md — source_count: 5, confidence: medium，合规
- flow-matching.md — source_count: 4, confidence: medium，合规
- 全部 frontmatter 齐全、type 合法、无 superseded/disputed 违规
- 全部 wikilink 目标存在，无断链
- 交叉引用双向正确
- index.md 中全部 4 个新页面已在对应类别登记

### 仍存风险
- source-storminsight.md: source_count: 0 + confidence: low，source-summary 无交叉来源验证，待被其他源引用后升级
- 论文自身称 "ten competitive baselines" 但主 Table 1 仅列 9 个（wiki 以「等」覆盖，非 wiki 错误）

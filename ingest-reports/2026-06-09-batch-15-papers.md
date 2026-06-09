# Ingest 报告：批量摄入 15 篇候选论文 (2026-06-09)

方式：Workflow 编排，每批 **2 个并发子代理**，各自定位→下载→验证→全文摄入一篇论文；下不到 PDF 则跳过。共享文件 (index.md/log.md/反向链接/共享概念页) 由主进程串行合并以避免写竞争。

**结果：13 成功 / 2 跳过。**

## 跳过 (2)
- **pi-stgnn** — No downloadable PDF obtainable. The paper is an ICML 2025 workshop submission whose only public locus is OpenReview note Fbv6tlP6vj (linked from icml.cc/virtual/2025/50687), which is access-restricted: /pdf?id= and /attachment?id= return HTTP 403 HTML error pages, and the OpenReview v1/v2 APIs return "User Guest does not have permission to see Note Fbv6tlP6vj" (403 ForbiddenError). No arXiv mirror, no PMLR proceedings entry (workshop paper, not in main proceedings), and no other direct PDF found across 5 searches. Only the abstract is public, which is insufficient for a faithful full-paper ingest. Nothing written to raw/ or wiki/.
- **bigst** — Could not obtain a valid PDF. The only authoritative source (vldb.org/pvldb/vol17/p1081-han.pdf) is unreachable from this sandbox: all attempts fail with TLS error 35 (unexpected EOF) or connection timeout 28, despite multiple UAs, forced TLS 1.2/1.3, relaxed ciphers, www/non-www, and a 4-retry loop (arXiv downloads succeed, so the block is specific to the VLDB host). Paper is not on arXiv; dl.acm.org returns an HTML challenge page (not a PDF); ResearchGate download blocked; Semantic Scholar API reports openAccessPdf CLOSED/empty with no indexed mirror. Nothing was written to raw/ or wiki/.

## 创建 / 修改明细 (13)

### stop — Robust Spatio-Temporal Centralized Interaction for OOD Learning (STOP)
_ICML 2025 (PMLR 267:42165–42192) · https://proceedings.mlr.press/v267/ma25s.html → `raw/stop-icml2025.pdf`_

- **创建**: [[source-stop]], [[stop]], [[centralized-message-passing]], [[context-aware-units]], [[generalized-perturbation-unit]]
- **WHY 创建**: source-stop.md — WHY: required source-summary capturing STOP's core idea, method, results, and limitations for the one ingested paper.；stop.md — WHY: the named model (entity) deserves its own hub page detailing architecture, robustness mechanisms, and relation to STONE/RSTIB/continual-learning baselines.；centralized-message-passing.md — WHY: STOP's signature, paper-unique mechanism (block node-to-node messaging via low-rank ConAU attention) warrants a dedicated technique page.；context-aware-units.md — WHY: ConAU is a distinct, reusable construct (the shared interaction hub) referenced by multiple pages and central to STOP's inductive-learning claim.；generalized-perturbation-unit.md — WHY: GenPU is a paper-unique perturbation technique with its own sampling/alternating-optimization details, distinct from the centralized-messaging mechanism.
- **反向链接 (修改既有页)**: [[continual-spatio-temporal-forecasting]], [[rstib]], [[traffic-forecasting]]
- **WHY 修改**: traffic-forecasting.md (proposed backlink) — WHY: log STOP's OOD contribution and its node-to-node-messaging critique on the field's hub page (verified existing link target).；rstib.md (proposed backlink) — WHY: connect the two ICML 2025 robust-ST-forecasting methods as sibling approaches (DRO vs information bottleneck).；continual-spatio-temporal-forecasting.md (proposed backlink) — WHY: record STOP's argument that continual-learning baselines fail under true OOD, contextualizing the CSTF concept.
- **交叉链接**: [[stop]]↔[[source-stop]]; [[stop]]↔[[centralized-message-passing]]; [[stop]]↔[[context-aware-units]]; [[stop]]↔[[generalized-perturbation-unit]]; [[stop]]↔[[ood-generalization]]; [[stop]]↔[[distributionally-robust-optimization]]; [[stop]]↔[[traffic-forecasting]]; [[stop]]↔[[rstib]]; [[stop]]↔[[continual-spatio-temporal-forecasting]]; [[stop]]↔[[bigst]]; [[centralized-message-passing]]↔[[context-aware-units]]; [[centralized-message-passing]]↔[[generalized-perturbation-unit]]; [[centralized-message-passing]]↔[[ood-generalization]]; [[generalized-perturbation-unit]]↔[[distributionally-robust-optimization]]; [[context-aware-units]]↔[[centralized-message-passing]]; [[ood-generalization]]↔[[distributionally-robust-optimization]]; [[ood-generalization]]↔[[traffic-forecasting]]

### fstllm — FSTLLM: Spatio-Temporal LLM for Few Shot Time Series Forecasting
_ICML · https://xiucheng.org/assets/pdfs/icml25-fstllm.pdf → `raw/fstllm-icml2025.pdf`_

- **创建**: [[source-fstllm]], [[fstllm]]
- **WHY 创建**: source-fstllm.md — WHY: required source-summary; one raw/ file = one summary page；fstllm.md — WHY: dedicated entity page for the named method, its three modules, plug-and-play integration, and results
- **反向链接 (修改既有页)**: [[gpd]], [[time-llm]], [[traffic-forecasting]], [[urbangpt]]
- **交叉链接**: [[fstllm]]↔[[source-fstllm]]; [[fstllm]]↔[[few-shot-traffic-forecasting]]; [[fstllm]]↔[[alpha-entmax]]; [[fstllm]]↔[[traffic-forecasting]]; [[fstllm]]↔[[time-llm]]; [[fstllm]]↔[[urbangpt]]; [[fstllm]]↔[[gpd]]; [[fstllm]]↔[[itransformer]]

### st-ttc — Learning with Calibration: Exploring Test-Time Computing of Spatio-Temporal Forecasting
_NeurIPS · https://arxiv.org/abs/2506.00635 → `raw/st-ttc-neurips2025.pdf`_

- **创建**: [[source-st-ttc]], [[st-ttc]], [[spectral-domain-calibration]], [[flash-gradient-update]]
- **WHY 创建**: source-st-ttc — required source-summary for the ingested paper (300-500 words, core idea/method/results/limitations).；st-ttc — named method/system page; the central entity readers will search for.；spectral-domain-calibration — technique unique to this paper (the 'what to compute' component); warrants its own page for cross-linking from frequency-domain forecasting pages.；flash-gradient-update — technique unique to this paper (the 'how to compute' component); distinct, citable mechanism (leakage-free streaming update).
- **反向链接 (修改既有页)**: [[test-time-adaptation-st]], [[traffic-forecasting]]
- **WHY 修改**: traffic-forecasting — add a test-time-calibration/distribution-shift subsection citing ST-TTC; this hub page indexes STF method families and was missing the test-time-computing line.；test-time-adaptation-st — add a cross-reference to ST-TTC as the complementary label-using test-time computing paradigm, distinguishing it from UrbanMind's label-free masked reconstruction.
- **交叉链接**: [[st-ttc]]↔[[source-st-ttc]]; [[st-ttc]]↔[[spectral-domain-calibration]]; [[st-ttc]]↔[[flash-gradient-update]]; [[st-ttc]]↔[[test-time-computing-st]]; [[st-ttc]]↔[[traffic-forecasting]]; [[st-ttc]]↔[[test-time-adaptation-st]]; [[spectral-domain-calibration]]↔[[flash-gradient-update]]; [[spectral-domain-calibration]]↔[[test-time-computing-st]]; [[flash-gradient-update]]↔[[test-time-computing-st]]; [[test-time-computing-st]]↔[[test-time-adaptation-st]]

### s2dbm — S2DBM: Series-to-Series Diffusion Bridge Model
_arXiv · https://arxiv.org/abs/2411.04491 → `null`_

- **创建**: [[source-s2dbm]], [[s2dbm]], [[brownian-bridge-diffusion]]
- **WHY 创建**: source-s2dbm: required source-summary for the ingested paper (300-500 words, core idea/method/results/limitations)；s2dbm: dedicated method/entity page detailing the unified framework (Theorem 1), Brownian-bridge instantiation, s-switch for deterministic vs probabilistic sampling, and linear prior/conditioning + label strategy；brownian-bridge-diffusion: technique page for the paper's core mechanism (pin both ends, sigma^2 variance scaling, time-series bridging via linear prior predictor F)
- **反向链接 (修改既有页)**: [[csdi]], [[diffusion-models]], [[generative-time-series-forecasting]], [[timegrad]]
- **WHY 修改**: timegrad: add backlink to S2DBM as a point-forecasting improvement over the standard conditional-diffusion TimeGrad paradigm；csdi: note CSDI is S2DBM's denoising-network architecture source, a unified-framework special case, and a primary baseline；diffusion-models: add the diffusion-bridge / Brownian-bridge variant and its deterministic (sigma^2=0) sampling regime；generative-time-series-forecasting: list S2DBM among generative TS forecasting models, highlighting its deterministic/probabilistic switch
- **交叉链接**: [[s2dbm]]↔[[brownian-bridge-diffusion]]; [[s2dbm]]↔[[source-s2dbm]]; [[s2dbm]]↔[[timegrad]]; [[s2dbm]]↔[[csdi]]; [[s2dbm]]↔[[diffusion-models]]; [[s2dbm]]↔[[generative-time-series-forecasting]]; [[s2dbm]]↔[[simdiff]]; [[brownian-bridge-diffusion]]↔[[diffusion-models]]; [[brownian-bridge-diffusion]]↔[[generative-time-series-forecasting]]; [[brownian-bridge-diffusion]]↔[[probability-flow-ode]]

### ratd — Retrieval-Augmented Diffusion Models for Time Series Forecasting
_NeurIPS · https://arxiv.org/abs/2410.18712 → `raw/ratd-neurips2024.pdf`_

- **创建**: [[ratd]], [[source-ratd]]
- **WHY 创建**: wiki/ratd.md — RATD is a named, foundational model (first retrieval-augmented TS diffusion) warranting its own entity page；wiki/source-ratd.md — mandatory source-summary for the ingested paper, defines [^src-ratd] footnote target
- **反向链接 (修改既有页)**: [[craft]], [[csdi]], [[middir]], [[retrieval-augmented-spatio-temporal-forecasting]], [[retrieval-guidance]], [[timegrad]]
- **交叉链接**: [[ratd]]↔[[source-ratd]]; [[ratd]]↔[[csdi]]; [[ratd]]↔[[timegrad]]; [[ratd]]↔[[diffusion-models]]; [[ratd]]↔[[retrieval-augmented-spatio-temporal-forecasting]]; [[ratd]]↔[[retrieval-guidance]]; [[ratd]]↔[[middir]]; [[ratd]]↔[[craft]]; [[ratd]]↔[[rast]]; [[ratd]]↔[[gtr]]; [[ratd]]↔[[x-prediction]]; [[ratd]]↔[[generative-time-series-forecasting]]

### armd — Auto-Regressive Moving Diffusion Models for Time Series Forecasting
_AAAI · https://arxiv.org/abs/2412.09328 → `raw/armd-arxiv2024.pdf`_

- **创建**: [[source-armd]], [[armd]], [[sliding-window-diffusion]], [[distance-based-devolution]]
- **WHY 创建**: source-armd.md — 论文的 source-summary，覆盖核心问题/方法/结果/局限，作为 [^src-armd] 引用目标；armd.md — 方法实体页，详述状态重定义、滑动前向演化、线性距离去演化、DDIM 去噪声采样与 ARMA 联系；sliding-window-diffusion.md — ARMD 独有的中间态生成技术（滑动替代加噪），含与 DDPM 同构形式与消融证据；distance-based-devolution.md — ARMD 独有的线性骨干去噪机制（距离预测 + 步自适应加权），含 t-embedding/Transformer 消融对比
- **反向链接 (修改既有页)**: [[cold-sampling]], [[d3vae]], [[diffusion-models]], [[dyffusion]], [[timegrad]]
- **WHY 修改**: timegrad — 作为 ARMD 明确departure的条件噪声范式 + 直接对比基线，加前向指针；diffusion-models — 伞概念页补充 ARMD 这一非噪声/广义扩散预测变体；d3vae — ARMD 主对比表中 ETTm1 次优基线，连接同族扩散-TS 方法；cold-sampling — 同属退化替代噪声的广义扩散，ARMD 滑动窗口为同族实例；dyffusion — 同样用确定性变换(插值)替代加噪作为扩散步，澄清广义扩散设计空间
- **交叉链接**: [[armd]]↔[[source-armd]]; [[armd]]↔[[sliding-window-diffusion]]; [[armd]]↔[[distance-based-devolution]]; [[armd]]↔[[arma-inspired-diffusion]]; [[sliding-window-diffusion]]↔[[arma-inspired-diffusion]]; [[distance-based-devolution]]↔[[arma-inspired-diffusion]]; [[armd]]↔[[timegrad]]; [[armd]]↔[[d3vae]]; [[armd]]↔[[diffusion-models]]; [[armd]]↔[[cold-sampling]]; [[armd]]↔[[dyffusion]]; [[sliding-window-diffusion]]↔[[cold-sampling]]; [[sliding-window-diffusion]]↔[[dyffusion]]

### doflow — DoFlow: Flow-based Generative Models for Interventional and Counterfactual Forecasting on Time Series
_ICLR · https://arxiv.org/abs/2511.02137 → `raw/doflow-iclr2026.pdf`_

- **创建**: [[source-doflow]], [[doflow]], [[causal-counterfactual-recovery]]
- **WHY 创建**: wiki/source-doflow.md — 论文的 source-summary（每个 raw/ 文件必需），300-500 字覆盖核心思想/方法/贡献/结果/局限；wiki/doflow.md — DoFlow 模型/系统的 entity 主页，详述 per-node CNF + RNN 架构、三种预测模式、似然异常检测与实验；wiki/causal-counterfactual-recovery.md — 论文独特贡献的 technique 页：abduction-action-prediction 编码-解码机制 + Corollary 4.5 反事实恢复定理与 BGM 比较
- **反向链接 (修改既有页)**: [[continuous-normalizing-flow]], [[e2-cstp]], [[flow-matching]], [[neural-ordinary-differential-equation]]
- **交叉链接**: [[doflow]]↔[[source-doflow]]; [[doflow]]↔[[causal-counterfactual-recovery]]; [[doflow]]↔[[continuous-normalizing-flow]]; [[doflow]]↔[[flow-matching]]; [[doflow]]↔[[neural-ordinary-differential-equation]]; [[doflow]]↔[[e2-cstp]]; [[causal-counterfactual-recovery]]↔[[causal-time-series-forecasting]]; [[doflow]]↔[[causal-time-series-forecasting]]

### k2vae — K²VAE: A Koopman-Kalman Enhanced Variational AutoEncoder for Probabilistic Time Series Forecasting
_ICML 2025 Spotlight (arXiv 2505.23017) · https://arxiv.org/pdf/2505.23017 → `raw/k2vae-arxiv2025.pdf`_

- **创建**: [[source-k2vae]], [[k2vae]], [[kalmannet-uncertainty-modeling]]
- **WHY 创建**: source-k2vae.md — WHY: mandatory source-summary for the ingested paper (one raw/ file = one source page)；k2vae.md — WHY: named model needs an entity page documenting architecture, theory, results, and its place in the generative-forecasting landscape；kalmannet-uncertainty-modeling.md — WHY: the neuralized Kalman Predict/Update that defines the VAE posterior is a specific, novel technique unique to this paper, distinct from generic Kalman filtering
- **反向链接 (修改既有页)**: [[d3vae]], [[generative-time-series-forecasting]], [[micro-macro-coupled-koopman-modeling]], [[mmckm]]
- **交叉链接**: [[k2vae]]↔[[source-k2vae]]; [[k2vae]]↔[[kalmannet-uncertainty-modeling]]; [[k2vae]]↔[[koopman-linearization-for-forecasting]]; [[k2vae]]↔[[kalman-filter]]; [[k2vae]]↔[[generative-time-series-forecasting]]; [[k2vae]]↔[[d3vae]]; [[k2vae]]↔[[mmckm]]; [[kalmannet-uncertainty-modeling]]↔[[kalman-filter]]; [[kalmannet-uncertainty-modeling]]↔[[koopman-linearization-for-forecasting]]; [[koopman-linearization-for-forecasting]]↔[[micro-macro-coupled-koopman-modeling]]; [[koopman-linearization-for-forecasting]]↔[[mmckm]]

### weathergfm — WeatherGFM: Learning A Weather Generalist Foundation Model via In-context Learning
_ICLR · https://arxiv.org/abs/2411.05420 → `raw/weathergfm-iclr2025.pdf`_

- **创建**: [[source-weathergfm]], [[weathergfm]], [[weather-prompt]], [[mixed-modal-masked-image-modeling]]
- **WHY 创建**: source-weathergfm: 每个 raw 文件需一份 source-summary，记录 WeatherGFM 的核心论点、方法、结果与局限。；weathergfm: 命名模型需独立 entity 页，承载架构、任务/数据、scaling law、OOD、与同领域模型对比。；weather-prompt: WeatherGFM 独有的三模态视觉提示格式，是其统一多模态任务的关键技术，值得独立技术页。；mixed-modal-masked-image-modeling: WeatherGFM 独有的 VQA 式掩码训练-推理范式（含目标全掩码），核心方法贡献。
- **反向链接 (修改既有页)**: [[extreme-weather-forecasting]], [[spatio-temporal-foundation-model]], [[swift]], [[uniextreme]]
- **WHY 修改**: uniextreme: 在天气基础模型对比中补充 WeatherGFM 作为通用多任务对照（互补于极端天气专用模型）。；swift: 补充 WeatherGFM 作为同领域天气基础模型的另一路线（任务统一 vs 概率预报）。；extreme-weather-forecasting: 在"与其他预测领域关系"中补充通用天气基础模型这一对比锚点。；spatio-temporal-foundation-model: 补全天气领域"任务统一+in-context"路线代表，弥补此前仅有 Pangu/Fengwu 网格预报模型的空白。
- **交叉链接**: [[weathergfm]]↔[[source-weathergfm]]; [[weathergfm]]↔[[weather-prompt]]; [[weathergfm]]↔[[mixed-modal-masked-image-modeling]]; [[weathergfm]]↔[[weather-foundation-model]]; [[weathergfm]]↔[[uniextreme]]; [[weathergfm]]↔[[swift]]; [[weathergfm]]↔[[extreme-weather-forecasting]]; [[weathergfm]]↔[[spatio-temporal-foundation-model]]; [[weather-prompt]]↔[[mixed-modal-masked-image-modeling]]; [[weather-foundation-model]]↔[[uniextreme]]; [[weather-foundation-model]]↔[[swift]]; [[weather-foundation-model]]↔[[spatio-temporal-foundation-model]]

### maginet — MagiNet: Mask-Aware Graph Imputation Network for Incomplete Traffic Data
_arXiv (later ACM TKDD) · https://arxiv.org/pdf/2406.03511 → `raw/maginet-arxiv2024.pdf`_

- **创建**: [[source-maginet]], [[maginet]]
- **WHY 创建**: wiki/source-maginet.md — WHY: 每个 raw/ 源文件需对应一个 source-summary 页，覆盖核心思想/方法/结果/局限。；wiki/maginet.md — WHY: MagiNet 是一个具名时空填补模型，需独立 entity 页记录其无预填充架构、掩码感知注意力与实验结果。
- **反向链接 (修改既有页)**: [[grin]], [[imputeformer]], [[message-passing-imputation]], [[pristi]]
- **交叉链接**: [[maginet]]↔[[grin]]; [[maginet]]↔[[pristi]]; [[maginet]]↔[[imputeformer]]; [[maginet]]↔[[csdi]]; [[maginet]]↔[[message-passing-imputation]]; [[maginet]]↔[[gsli]]; [[maginet]]↔[[over-smoothing-in-gnns]]; [[maginet]]↔[[mask-aware-imputation-no-prefilling]]

### stamimputer — STAMImputer: Spatio-Temporal Attention MoE for Traffic Data Imputation
_arXiv (extended version of IJCAI 2025) · https://arxiv.org/abs/2506.08054 → `raw/stamimputer-arxiv2025.pdf`_

- **创建**: [[source-stamimputer]], [[stamimputer]], [[lrsgat]]
- **WHY 创建**: source-stamimputer.md — 每个 raw/ 源文件需一个 source-summary 页（300-500 词覆盖核心问题/方法/结果/局限）；stamimputer.md — 命名模型/系统需 entity 页，记录外层 MoE 三阶段架构与三类专家分工；lrsgat.md — LrSGAT 是论文唯一命名的新机制（采样投影器 + 低秩再注意力 + 半自适应动态图），值得独立 technique 页
- **反向链接 (修改既有页)**: [[gsli]], [[imputeformer]], [[mixture-of-experts]], [[partial-blackout]], [[testam]]
- **WHY 修改**: mixture-of-experts — 加入 STAMImputer 作为外层 MoE + 首个交通填补 MoE 的实例；imputeformer — STAMImputer 选其为主要 SOTA 基线并继承低秩诱导，关系互链且 source_count 上升；partial-blackout — STAMImputer 的块缺失场景是 partial-blackout 的具体实例，增加第二个支持来源；testam — 两者为 wiki 中仅有的 MoE-based 时空注意力模型（预测 vs 填补），互链阐明设计对比；gsli — 同为 2025 时空填补 + 图结构学习模型，互链便于动态图路线对比
- **交叉链接**: [[stamimputer]]↔[[lrsgat]]; [[stamimputer]]↔[[source-stamimputer]]; [[lrsgat]]↔[[source-stamimputer]]; [[stamimputer]]↔[[mixture-of-experts]]; [[stamimputer]]↔[[imputeformer]]; [[stamimputer]]↔[[partial-blackout]]; [[stamimputer]]↔[[testam]]; [[lrsgat]]↔[[imputeformer]]; [[lrsgat]]↔[[gsli]]

### st-vision-llm — Vision-LLMs for Spatiotemporal Traffic Forecasting (ST-Vision-LLM)
_arXiv · https://arxiv.org/abs/2510.11282 → `raw/st-vision-llm-arxiv2025.pdf`_

- **创建**: [[source-st-vision-llm]], [[st-vision-llm]], [[direct-numerical-encoding]], [[grpo-for-forecasting]]
- **WHY 创建**: source-st-vision-llm: required source-summary for the ingested paper (arXiv 2510.11282)；st-vision-llm: the named model/framework deserves its own entity page as the run's primary artifact；direct-numerical-encoding: paper-unique technique (single-token float vocabulary + two-stage alignment) worth a standalone technique page；grpo-for-forecasting: paper-specific application of GRPO with an NRMSE forecasting reward, distinct from existing GRPO pages (flow-grpo, STReasoner S-GRPO)
- **反向链接 (修改既有页)**: [[flow-grpo]], [[mobile-traffic-forecasting]], [[most]], [[multimodal-time-series-forecasting]], [[streasoner]], [[time-llm]], [[urbangpt]]
- **WHY 修改**: multimodal-time-series-forecasting: add ST-Vision-LLM as the time-series-as-image route within multimodal TS forecasting；urbangpt: record ST-Vision-LLM's explicit contrast with the node-based separate-encoder approach；time-llm: record ST-Vision-LLM's use of Time-LLM as a 1D-sequence baseline/foil；most: contrast real-image modality (MoST) vs traffic-matrix-as-image (ST-Vision-LLM)；streasoner: link the two LLM+GRPO spatio-temporal siblings (forecasting vs reasoning)；flow-grpo: connect GRPO-for-forecasting to the wiki's GRPO hub；mobile-traffic-forecasting: add ST-Vision-LLM to grid-level mobile-traffic model coverage
- **交叉链接**: [[st-vision-llm]]↔[[source-st-vision-llm]]; [[st-vision-llm]]↔[[direct-numerical-encoding]]; [[st-vision-llm]]↔[[grpo-for-forecasting]]; [[st-vision-llm]]↔[[vision-language-traffic-forecasting]]; [[st-vision-llm]]↔[[time-llm]]; [[st-vision-llm]]↔[[urbangpt]]; [[st-vision-llm]]↔[[most]]; [[st-vision-llm]]↔[[streasoner]]; [[st-vision-llm]]↔[[mobile-traffic-forecasting]]; [[st-vision-llm]]↔[[multimodal-time-series-forecasting]]; [[grpo-for-forecasting]]↔[[flow-grpo]]; [[grpo-for-forecasting]]↔[[streasoner]]; [[direct-numerical-encoding]]↔[[patch-based-tokenization]]; [[vision-language-traffic-forecasting]]↔[[multimodal-time-series-forecasting]]

### motm — MoTM: Towards a Foundation Model for Time Series Imputation based on Continuous Modeling
_AALTD (ECML workshop) 2025 — Oral · https://arxiv.org/abs/2507.13207 → `raw/motm-aaltd2025.pdf`_

- **创建**: [[source-motm]], [[motm-ridge-orchestrator]]
- **WHY 创建**: wiki/source-motm.md — MoTM 原始论文的 source-summary，确立此前缺失的 primary source 出处（既有 motm.md 仅源自评估它的基准论文）；wiki/motm-ridge-orchestrator.md — 为本文独有的 ridge 编排机制单列技术页，承载 H(t) 表示拼接 + 闭式 ridge 的方法细节与消融
- **反向链接 (修改既有页)**: [[motm]], [[source-nuwats]], [[time-indexed-foundation-model]]
- **WHY 修改**: wiki/motm.md — 加入原始论文 [^src-motm] 引用与方法/合成实验/时延细节，将来源从 1 提升至 2，建议 confidence medium→high（应由编排器更新，非本 agent 编辑）；wiki/time-indexed-foundation-model.md — 用 MoTM 原始来源佐证'连续时间→跨采样率/OOD 泛化'范式核心，增加第二独立来源；wiki/source-nuwats.md — 补充 MoTM 论文对固定段插补基础模型局限性的外部批评视角
- **交叉链接**: [[source-motm]]↔[[motm]]; [[motm-ridge-orchestrator]]↔[[motm]]; [[motm-ridge-orchestrator]]↔[[time-indexed-foundation-model]]; [[motm-ridge-orchestrator]]↔[[tabpfn-ts]]; [[source-motm]]↔[[source-nuwats]]; [[source-motm]]↔[[source-time-indexed-imputation]]

## 共享概念页 (主进程合并创建, 12)
- [[ood-generalization]] (concept) — 提出方: 见正文引用
- [[distributionally-robust-optimization]] (concept) — 提出方: 见正文引用
- [[few-shot-traffic-forecasting]] (concept) — 提出方: 见正文引用
- [[alpha-entmax]] (technique) — 提出方: 见正文引用
- [[test-time-computing-st]] (concept) — 提出方: 见正文引用
- [[arma-inspired-diffusion]] (concept) — 提出方: 见正文引用
- [[causal-time-series-forecasting]] (concept) — 提出方: 见正文引用
- [[koopman-linearization-for-forecasting]] (concept) — 提出方: 见正文引用
- [[kalman-filter]] (concept) — 提出方: 见正文引用
- [[weather-foundation-model]] (concept) — 提出方: 见正文引用
- [[mask-aware-imputation-no-prefilling]] (concept) — 提出方: 见正文引用
- [[vision-language-traffic-forecasting]] (concept) — 提出方: 见正文引用

## 特别处理
- **[[motm]] 富化**: 既有实体页由评测论文 (TMLR 2026) 建立; 本批新增其原始论文 [[source-motm]] (AALTD 2025) 作为主要来源 → source_count 1→2, confidence medium→high (corroboration, 非矛盾)。


## 补充（2026-06-09）：BigST 补摄入
用户提供 PDF（ACM DOI 10.14778/3641204.3641217）后，补摄入此前跳过的 **BigST**（PVLDB 2024）。
- **创建**：[[source-bigst]]、[[bigst]]、[[linearized-spatial-convolution]]（LSC）、[[long-sequence-feature-extractor]]（LSFE）
- **反向链接**：[[gwnet]]（直接前身，BigST 线性化其 O(N²) 自适应邻接）、[[large-scale-spatial-temporal-graph]]、[[ragc]]、[[traffic-forecasting]]、[[centralized-message-passing]]（恢复链接）
- **仍跳过**：PI-STGNN（用户无访问权限）。

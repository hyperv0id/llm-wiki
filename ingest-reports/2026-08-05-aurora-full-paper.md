# Ingest 报告：Aurora — 完整论文补全（ICLR 2026）

> **状态**：先前于 2026-05-03 基于 arXiv 摘要 ingest，已创建 source/entity/technique/concept 页。本次基于完整论文（40 页 camera-ready）补全，消除"完整论文细节待补充"欠账。

## 源文件

`raw/2509.22295.pdf`（40 页，ICLR 2026，arXiv:2509.22295v6；不可变，已存在）。笔记：`notes/20260805T181911--paper-aurora__paper.org`。

## 修改

- [[source-aurora]] — WHY：摘要级升级为完整论文级。补全 tokenization（RevIN+Patching p=48、FFT 主周期内生图像、Bert）、token 蒸馏（语义质心）、Corr 桥接公式、Prototype-Guided Flow Matching 完整机制（1000 原型、Algorithm 1、OT 路径、损失函数）；新增预训练语料（>10 亿点、来源与域分布、GPT-4 生成+双重质检、模态缺失训练）、完整实验表、消融、效率、训练配置、开源链接；局限性从"仅摘要"改为实质性的模拟文本/Corr 假设/原型表达力三点；`confidence` medium→high，`last_updated` 更新
- [[aurora]] — WHY：实体页补架构量化（Encoder 1/Decoder 9/Flow-Net 3 层、dim 256、210.8M 参数、1000 原型）、预训练语料细节、模态缺失机制、实验具体数字（Sundial −27.0% 等）、消融（Economy 0.033→0.277）、效率（83.5ms/样本）；`source_count` 4→5，`last_updated` 更新
- [[modality-guided-self-attention]] — WHY：补 Corr 桥接公式链（$V_{\text{Attn}}$/$T_{\text{Attn}}$/$W$/$\text{Corr}$/$S$）、token 蒸馏语义质心细节、附录 C.4 可视化证据与消融证据；`confidence` medium→high，`last_updated` 更新
- [[prototype-guided-flow-matching]] — WHY：补 ConditionDecoder/Prototype Bank（1000 个、三角/指数/对数/多项式基）/PrototypeRetriever 完整机制、Algorithm 1 伪代码、OT 路径与损失函数、实验证据（消融 + 采样可扩展性 + C.2/C.3 可视化）；`last_updated` 更新
- [[generative-time-series-forecasting]] — WHY：Aurora 段落补"起点为原型而非高斯噪声"这一与 Sundial/TSFlow 形成对照的关键特征；venue arXiv→ICLR 2026；`last_updated` 更新
- [[log]] — WHY：记录本次补全操作

## 未创建

- 无新页面。现有 4 页（source/entity/2×technique）已覆盖全部核心技术；1000 原型、OT 路径、Algorithm 1 等细节归入既有 prototype-guided-flow-matching 页

## 与 2026-05-03 摘要级摄入的差异（关键新信息）

- 方法：Corr 桥接完整公式（$V_{\text{Attn}} \cdot W \cdot T_{\text{Attn}}^\top$ 注入 $S = (QK^\top + \text{Corr})/\sqrt{d}$）；1000 原型的基初始化与 PrototypeRetriever 机制；Algorithm 1 采样；OT 路径损失
- 预训练语料：>10 亿时间点、9 域来源（ERA5/IoT/Monash/UEA/UCR/PEMS）、GPT-4 生成文本 + GPT-4 粗查 + 人工抽检
- 训练：8×A800、~30 天、lr 5e-5、batch 8192、11+4 token
- 实验：各基准具体降幅、论文报告的 1st count、消融表（Economy 0.033→0.277 等）、效率表（210.8M/83.5ms）
- 模态缺失训练：预训练随机遮挡文本

## 遗留风险（与笔记一致）

预训练与评测文本均为 GPT-4 生成的模拟文本——模型学的是"读 GPT 风格描述"，真实下游文本的分布外表现未验证；这是该论文值得在后续真实多模态数据上复核的点。
（以上为课程层面评估，非论文自述）

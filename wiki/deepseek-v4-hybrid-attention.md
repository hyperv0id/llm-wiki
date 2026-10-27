---
title: "DeepSeek-V4 Hybrid Attention (CSA + HCA)"
type: technique
tags:
  - attention
  - long-context
  - kv-cache
  - llm
  - inference-efficiency
created: 2026-10-27
last_updated: 2026-10-27
source_count: 2
confidence: medium
status: active
---

# DeepSeek-V4 Hybrid Attention (CSA + HCA)

DeepSeek-V4 系列以交错的 [[compressed-sparse-attention|CSA]]（$m=4$ 压缩 + DSA top-$k$）与 [[heavily-compressed-attention|HCA]]（$m'=128$ 压缩 + dense）组成注意力层，外加滑窗分支与三个共用组件支撑百万 token 上下文（§2.3）[^src-deepseek-v4-2606-19348]。该方案是长上下文效率的主要来源：1M 上下文下，Pro 的单 token 推理 FLOPs 为 DeepSeek-V3.2 的 27%、KV cache 为 10%；Flash 进一步降至 10% 与 7%（Figure 1 右、§1）[^src-deepseek-v4-2606-19348]。

## 层配置

Flash（43 层）前两层为纯滑窗注意力，随后 CSA 与 HCA 交错（§4.2.1）[^src-deepseek-v4-2606-19348]；Pro（61 层）前两层全用 HCA，随后同样交错 CSA 与 HCA（§4.2.1）[^src-deepseek-v4-2606-19348]。论文未给出「交错」的具体排布（逐层交替或分段），亦未给出 CSA:HCA 的层间比例 [INFERENCE 缺口]。

该混合方案在下一代被替换：DeepSeek-V4.1-Flash 用纯 [[csa2|CSA2]] 加 [[ced|CED]] 取代交错结构——HCA 支路取消，主 KV 转 FP4，主 KV、indexer K 与 Top-$k$ 下标跨层共享，解码器全局 KV 由编码器末层投影，全局 KV 总量降到 890 bytes/token（§1、§2.3、§2.4.4）[^src-deepseek-v4-1-2609-19969]。

## 四个共用组件（§2.3.3）

在滑窗分支中，CSA 与 HCA 均分块压缩，因果掩码下每个 query 只能看到**之前的**压缩块，同块内其他 token 互不可见；而语言建模里最近的 token 通常最相关。因此每个 query 额外生成最近 $n_{win}=128$ 个 token 的未压缩 KV 条目，与压缩条目一同进入同一次核心注意力（§2.3.3）[^src-deepseek-v4-2606-19348]。

**QK RMSNorm**：核心注意力前，对 query 的每个 head 与压缩 KV 条目的唯一 head 做 RMSNorm，防止注意力 logits 爆炸并稳定训练（§2.3.3）[^src-deepseek-v4-2606-19348]。该设计使 Muon 优化器无需 QK-Clip（§2.4）[^src-deepseek-v4-2606-19348]。

**Partial RoPE**：只对 query、KV 条目与核心注意力输出的**末 64 维**施加 RoPE。KV 条目同时充当 key 与 value，注意力输出会带上加权求和的绝对位置；为此在每个输出 $o_{t,i}$ 的末 64 维施加位置 $-i$ 的 RoPE，抵消后输出携带 query 与 KV 条目间的相对位置（§2.3.3）[^src-deepseek-v4-2606-19348]。

**Attention sink**：每 head 设一个可学习 logit $z'_{h}$，$\operatorname{Exp}(z'_{h})$ 计入 softmax 分母（式 27），使每个 query head 的注意力总量可小于 1 甚至接近 0（§2.3.3）[^src-deepseek-v4-2606-19348]：

$$s_{h,i,j}=\frac{\operatorname{Exp}(z_{h,i,j})}{\sum_{k}\operatorname{Exp}(z_{h,i,k})+\operatorname{Exp}(z'_{h})}$$

## 效率账（§2.3.4）

四项来源按论文顺序：

1. **混合存储**：KV 条目中 RoPE 维用 BF16、其余维用 FP8，KV cache 相对纯 BF16 存储近乎减半。
2. **indexer FP4**：lightning indexer 的注意力计算用 FP4，服务超长上下文下的打分环节。
3. **更小的 top-$k$**：相对 V3.2 的 2048，V4 取 512（Flash）/ 1024（Pro），改善短中等文本的效率（$k$ 取值见 §4.2.1，2048 见 [[deepseek-sparse-attention]]）。
4. **压缩 + 混合**：CSA 与 HCA 本身同时压 KV cache 与注意力 FLOPs，是主要来源。

以 LLM 注意力常见配置 BF16 GQA8（head dim 128）为基线，1M 上下文下 V4 系列的 KV cache 约为基线的 2%（§2.3.4）[^src-deepseek-v4-2606-19348]。对比高效基线 V3.2（MLA + DSA），Figure 1 右给出 1M 位置的单 token FLOPs（等价 FP8 FLOPs 口径）与累计 KV cache 曲线：Pro 为 27% FLOPs（3.7 倍低）、10% KV（9.5 倍小），Flash 为 10% FLOPs（9.8 倍低）、7% KV（13.7 倍小）（§1）[^src-deepseek-v4-2606-19348]。routed expert 参数采用 FP4：当前硬件上 FP4×FP8 与 FP8×FP8 峰值算力相同，未来硬件理论上可再省 1/3（§1）[^src-deepseek-v4-2606-19348]。

## 训练配方（§4.2.2）

两个模型同构，序列长度从 4K 渐进扩到 16K、64K、1M；稀疏注意力分两段引入：先用 dense 注意力预热（Flash 为前 1T token，Pro 的 dense 阶段更长且未给具体步数），到 64K 序列长度时先用短阶段预热 CSA 的 lightning indexer，再切到稀疏注意力直到训练结束（§4.2.2）[^src-deepseek-v4-2606-19348]。Flash 训 32T token、Pro 训 33T；MTP 损失权重 0.3、学习率衰减起点降到 0.1；打包样本采用 sample-level attention masking（§4.1、§4.2.2）[^src-deepseek-v4-2606-19348]。

训练期压缩注意力的并行适配（§3.4.3）：常规上下文并行按连续 $s$ 个 token 切分，与压缩产生冲突——尾块不足 $m$ 个 token 须丢弃，且压缩块可能跨 rank 边界。方案采用两阶段通信：rank $i$ 先将最后 $m$ 个未压缩 KV 条目发往 rank $i{+}1$，后者结合本地 $s$ 个条目压缩出固定 $s/m+1$ 个条目（含填充）；再经 all-gather 汇齐，由融合的 select-and-pad 算子重排为 $\text{cp\_size}\cdot s/m$ 的全长集合，填充置于尾部。HCA 与 CSA indexer 的可见范围按规则预计算，CSA 稀疏注意力由 top-$k$ 选择器显式给出可见下标（§3.4.3）[^src-deepseek-v4-2606-19348]。

## 推理与 KV cache 管理（§3.5）

混合注意力并存多种尺寸与更新规则的 KV 条目（CSA、HCA、indexer、SWA 及未压缩尾部），破坏了 PagedAttention 的适用前提。V4 的布局（§3.5.1、Figure 6）：将 SWA 与未压缩尾部视作随位置变化的序列**状态**，按请求分配定长 state cache；CSA/HCA 压缩条目存入经典 KV cache，每块覆盖 $\operatorname{lcm}(m,m')=128$ 个原始 token，产出 $k_1=\operatorname{lcm}(m,m')/m=32$ 个 CSA 条目与 $k_2=\operatorname{lcm}(m,m')/m'=1$ 个 HCA 条目，分块对齐与稀疏 kernel 协同设计（§3.5.1）[^src-deepseek-v4-2606-19348]。

盘上缓存（§3.5.2）：CSA/HCA 压缩条目直接落盘，命中前缀时读到最后一个完整压缩块，尾部不完整块的未压缩条目不落盘、需重算。SWA 条目未压缩且每层都有，体积约为压缩条目的 8 倍，三种策略按部署取舍：全量缓存（计算零冗余、写密集）、每 $p$ token 周期检查点（存储-计算折中）、零缓存（利用已缓存压缩条目只重算最后 $n_{win}\cdot L$ 个 token 即可恢复末窗 SWA 条目）（§3.5.2）[^src-deepseek-v4-2606-19348]。

后训练中（§5.2.1），FP4 QAT 覆盖 MoE 专家权重与 CSA indexer 的 QK 路径，QK 激活的缓存、加载与乘法全程 FP4；index scores 从 FP32 转 BF16 后，top-$k$ selector 提速 2 倍，KV entry recall 保持 99.7%（§5.2.1）[^src-deepseek-v4-2606-19348]。

## 证据

效率指标（§2.3.4、Figure 1 右、§1）给出四项来源与三组比值（27%/10%、10%/7%、2% of GQA8 基线）；Figure 1 右为估计值（estimated），口径为等价 FP8 FLOPs 与累计 KV cache，无逐组件延迟分解。长上下文评测（Table 6、Table 7）中，MRCR 1M（MMR）Pro-Max 83.5、Flash-Max 78.7，对照 Opus-4.6-Max 92.9、Gemini-3.1-Pro-High 76.3；CorpusQA 1M（ACC）Pro 62.0，对照 Opus 71.7、Gemini 53.8；Figure 9 显示 128K 内检索表现稳定，超过后可见下降，1M 处平均 MMR 为 Pro-Max 0.59、Flash-Max 0.49（图内数值，点值以图为准）。基础模型（Table 1）在 LongBench-V2 上，V3.2-Base 40.2、V4-Flash-Base 44.7、V4-Pro-Base 51.5。

## 边界

效率数字源自 Figure 1 的估计曲线，缺少逐组件（压缩、indexer、top-$k$、核心注意力）的延迟或显存分解，对照仅有 V3.2 单一系统（§2.3.4）[^src-deepseek-v4-2606-19348]。混合设计缺少内部消融，未对比全 CSA、全 HCA 与不同层配比，交错排布未定义，各组件贡献无法分离（§2.3、§4.2.1）[^src-deepseek-v4-2606-19348]。长上下文评测对比 Claude 与 Gemini 等系统，因模型、数据与后训练各异，1M 成绩无法归因到注意力设计（Table 6）[^src-deepseek-v4-2606-19348]。前两层不对称设计（Flash 纯 SWA、Pro 全 HCA）缺乏论证，$m$、$m'$、$k$、$n_{win}$ 亦无敏感性实验（§4.2.1）[^src-deepseek-v4-2606-19348]。该设计「相对复杂」，为降风险保留了许多已验证组件，未来计划蒸馏到更本质的设计（§6）[^src-deepseek-v4-2606-19348]。

## 相关页面

[[compressed-sparse-attention]] 涵盖稀疏支路数据流，[[heavily-compressed-attention]] 涵盖重压缩支路数据流，[[deepseek-sparse-attention]] 阐释 CSA 复用的 DSA 框架及其 V3.2 原型，[[native-sparse-attention]] 给出单层三分支门控合并对照路线；[[kv-cache-compression]] 梳理推理期缓存压缩路线，V4 把压缩放进训练期架构；[[manifold-constrained-hyper-connections]] 记录同一模型引入的残差改造，[[multi-query-attention]] 追溯共享 KV 的 $G=1$ 形态来源，[[source-deepseek-v4-2606-19348]] 为源摘要。

[^src-deepseek-v4-2606-19348]: [[source-deepseek-v4-2606-19348]]
[^src-deepseek-v4-1-2609-19969]: [[source-deepseek-v4-1-2609-19969]]

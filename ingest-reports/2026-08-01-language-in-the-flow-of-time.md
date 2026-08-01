# Ingest 报告：Language in the Flow of Time (TaTS, ICLR 2026 camera-ready)

## 创建
- 无新页面——raw 已有同文 arXiv v3 版（language-flow-time-tats.pdf），wiki 已有完整页面体系；本次为版本更新 + 重复页合并 + TaTS↔TESS 对照深化。

## 修改
- wiki/source-language-in-the-flow-of-time.md — WHY：合并中文页独有内容（~14% MSE 效率分析、编码器消融 BERT-110M/GPT2-1.5B/LLaMA2-7B、打乱/丢弃/噪声鲁棒性消融、+1% 参数/+8% 训练时间、门控残差/交叉注意力对照）；补 GitHub 代码链接与 4 步工作流；修正 2 处无据论断（RoBERTa→BERT+LLaMA2；monthly>daily 增益→同频内 TT 比值相关，被 Table 5 反证）；"consistently"→"generally"
- wiki/source-language-flow-time.md — WHY：与英文页重复，标 status: superseded + superseded_by（内容存档保留）
- wiki/tats.md — WHY：vs TESS 段补 2 点限定（TaTS 无显式门控但有池化+MLP+联合训练，非 token 级注入；两篇证据可并立 + 评测重叠说明 + 判定标准对照）
- wiki/tess.md — WHY：对照表 TaTS 行补判定机制与评测重叠说明
- wiki/index.md — WHY：旧 slug 条目标 ⚠️ superseded，canonical 条目更新描述
- wiki/log.md — WHY：记录 ingest 与 maintenance

## 新建交叉链接
- [[source-language-flow-time]] →(superseded_by)→ [[source-language-in-the-flow-of-time]]
- [[tats]] ↔ [[tess]] 深化（vs TESS 段两点限定 + 判定标准对照）
- [[tess]] ↔ [[tats]]（对照表 TaTS 行扩充）

## 源文件
- raw/language-in-the-flow-of-time-iclr2026.pdf（只读；md5 a07e5a1505ce5571cb687b6f886480c8；camera-ready 版，与 raw/language-flow-time-tats.pdf 即 arXiv:2502.08942v3 实质同文，新增仅 ACK/ethics/reproducibility/LLM-usage 四段声明；两版均含 GitHub 链接与 BERT 编码器实验——初判"camera-ready 新增"不成立，已更正）

## 自检
- 8 个引用页脚注零改动（均指向保留 slug source-language-in-the-flow-of-time）
- 主页面 source_count: 1、confidence: medium 不变；修正后无与论文冲突论断
- 数字核对（camera-ready 逐字）：14% 出自效率分析段（非摘要）、TT-Wasserstein Table 1/5、Table 7/8/9 消融结论、+1%/+8%、U=[X;Zᵀ]∈R^{T×(N+d_mapped)}
- 论文内部不一致已标注：Table 9 表内标签 60%/20% 与正文 40%/80% 不符（引用以正文为准）
- 无 `\|` 逃逸 wikilink；未 git commit；未改 raw 旧文件

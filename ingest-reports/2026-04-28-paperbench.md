# Ingest 报告：Long Context, Less Focus (Gu et al., 2026)

## 创建
- **wiki/source-paperbench.md** — WHY：源文件摘要页面，总结论文核心论点、贡献和局限性
- **wiki/paperbench.md** — WHY：实体页面，PAPerBench 基准概述、任务设计和关键发现
- **wiki/long-context-scaling-gap.md** — WHY：实体页面，长上下文缩放 gap 现象、理论解释和驱动因素
- **wiki/attention-dilution.md** — WHY：技术页面，Attention Dilution 定理的数学定义和推论
- **wiki/decoy-injection.md** — WHY：技术页面，诱饵注入技术的动机、实现和实验发现
- **wiki/long-context-personalization.md** — WHY：概念页面，长上下文个性化的问题定义、实验和理论
- **wiki/privacy-reasoning.md** — WHY：概念页面，隐私推理的问题定义、任务类型和实验结果

## 修改
- **wiki/index.md** — 添加新页面到 Sources、Entities、Concepts、Techniques 类别
- **wiki/log.md** — 记录 ingest 操作

## 新建交叉链接
- [[source-paperbench]] → [[paperbench]], [[long-context-scaling-gap]], [[attention-dilution]], [[decoy-injection]], [[long-context-personalization]], [[privacy-reasoning]]
- [[paperbench]] → [[source-paperbench]], [[long-context-scaling-gap]], [[decoy-injection]], [[long-context-personalization]], [[privacy-reasoning]]
- [[long-context-scaling-gap]] → [[source-paperbench]], [[attention-dilution]], [[context-window-extension]]
- [[attention-dilution]] → [[source-paperbench]], [[long-context-scaling-gap]], [[context-window-extension]]
- [[decoy-injection]] → [[source-paperbench]], [[paperbench]], [[privacy-reasoning]]
- [[long-context-personalization]] → [[source-paperbench]], [[paperbench]], [[attention-dilution]]
- [[privacy-reasoning]] → [[source-paperbench]], [[paperbench]], [[attention-dilution]], [[decoy-injection]]
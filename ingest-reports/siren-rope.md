# Ingest 报告：siren-rope

## 创建
- **wiki/source-siren-rope.md** — WHY：论文 source-summary，300-500 字总结核心论点、贡献和局限性
- **wiki/siren-rope.md** — WHY：主 entity 页面，描述模型架构和实验结果
- **wiki/dual-branch-siren.md** — WHY：技术页面，详细说明双分支 SIREN-DNN 架构
- **wiki/temporal-rotation.md** — WHY：概念页面，解释将时间戳映射为旋转角的技术
- **wiki/ordinal-temporal-fusion.md** — WHY：技术页面，说明时间与序数信号的可学习融合公式
- **wiki/learnable-frequency-scaling.md** — WHY：技术页面，描述可学习每维频率缩放技术
- **raw/siren-rope.md** — WHY：源文件归档，保存完整论文内容

## 修改
- **wiki/index.md** — WHY：添加 1 个 source 页面、1 个 entity 页面、4 个 technique/concept 页面
- **wiki/log.md** — WHY：追加 ingest 记录

## 新建交叉链接
- [[siren-rope]] → [[source-siren-rope]]
- [[siren-rope]] → [[dual-branch-siren]]
- [[siren-rope]] → [[temporal-rotation]]
- [[siren-rope]] → [[ordinal-temporal-fusion]]
- [[siren-rope]] → [[learnable-frequency-scaling]]
- [[siren-rope]] → [[yarn]]（RoPE 上下文扩展方法）
- [[siren-rope]] → [[alibi]]（另一种位置编码方法）
- [[siren-rope]] → [[generalized-positional-encoding-framework]]
- [[dual-branch-siren]] → [[temporal-rotation]]
- [[dual-branch-siren]] → [[ordinal-temporal-fusion]]
- [[ordinal-temporal-fusion]] → [[learnable-frequency-scaling]]
- [[temporal-rotation]] → [[dual-branch-siren]]
- [[temporal-rotation]] → [[ordinal-temporal-fusion]]

## 统计
- 新建页面：7
- 修改页面：2
- 交叉链接：13
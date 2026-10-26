# Ingest 报告：Transformers are RNNs (Katharopoulos et al., 2020)

日期：2026-10-27。源：arXiv:2006.16236（v3，2020-08-31；diff 核对 v1 正文一致，仅作者栏与出版信息不同），ICML 2020（PMLR 119）。归档：`raw/transformers-are-rnns-2006-16236.pdf`（`*.pdf` 不入 git）。

## 创建

- `wiki/source-katharopoulos-2020-linear-attention.md` — WHY：该论文是线性注意力的原始出处，库内 `linear-attention-unified-framework` 等页的循环形式此前只有二手出处（Han et al. 2024 的转述），需要一手 source 页。
- `wiki/linear-attention-rnn-form.md` — WHY：机制页。取舍说明：任务允许「补进 `linear-attention-unified-framework` 或新建」。未补进既有页的理由——该页主线是 Mamba↔线性注意力的公式级对齐与六项组件消融，塞入本论文的完整推导链（广义注意力非负性 → 结合律 → elu+1 → 因果前缀和 → RNN 重写 → 梯度 → 实验）会改变页面主线并与 Mamba 对照内容互相稀释；且模型结构页需要分阶段 `aligned` 数据流块，不适合挂在对照页下。折中做法：既有页的循环形式公式处补了原始出处引用（式 10–12/18–19）并前向链接新页，避免无出处论断。

## 修改

- `wiki/linear-attention-unified-framework.md` — WHY：「线性注意力的循环形式」是本论文式 10–12/18–19 的转述，补第二引用（原始出处），source_count 5→6，`last_updated` 刷新。
- `wiki/performer.md` — WHY：「对照 Linear Transformer（Katharopoulos et al., 2020）」的对照对象即本次入库论文，补引用与到新机制页的链接，source_count 3→4。
- `wiki/positive-random-features.md` — WHY：「核分解 + 结合律重排」的共同形态最早出自本论文式 5–6，补原始出处，source_count 2→3。
- `wiki/fast-weight-mlp-view-of-attention.md` — WHY：递归状态 $s_i$ 是 fast-weight 读法的矩阵实例，相关页面加一条链接（结构性链接，无新增事实论断）。
- `wiki/spectral-kernel-linear-attention.md` — WHY：「落成线性注意力」节引用了结合律重排与循环形式，加指向原始推导页的链接。
- `wiki/index.md` — WHY：Sources 段与 Techniques 段各加一条，段内字母序。

## 新建交叉链接

- [[linear-attention-rnn-form]] ↔ [[linear-attention-unified-framework]]（循环形式原始出处 ↔ Mamba 对照）
- [[linear-attention-rnn-form]] ↔ [[performer]]（骨架与随机特征替换；Performer 附录报告 elu+1 线性注意力在其设置中梯度爆炸）
- [[linear-attention-rnn-form]] ↔ [[positive-random-features]]（结合律重排的原始出处）
- [[linear-attention-rnn-form]] ↔ [[fast-weight-mlp-view-of-attention]]（$s_i$ 即 fast weights）

## 数值备注

- **任务提示与论文不符**：任务清单列出的「IMDB/WikiText-103/AAHQ（及 text8）语言建模实验」在 arXiv v1–v3 中均不存在。论文实验为：合成序列复制收敛（Fig 2）、时间/显存基准（Fig 1，$N\in\{2^9,\dots,2^{16}\}$，softmax 上限 4096、linear/lsh 16,384，GTX 1080 Ti 11GB）、MNIST 与 CIFAR-10 自回归图像生成（Table 1/2/4/5）、WSJ 语音识别 CTC（Table 3）。页面只落论文实际有的实验。
- **§C.2 正文与 Table 5 不一致**：正文称 CIFAR-10 单图生成"almost 6.6× faster than softmax"，但 Table 5 数字给出 GPU 300.1 s vs 61.3 s（≈4.9×）、CPU 8,651.4 s vs 45.1 s（≈191.8×），均不等于 6.6×。页面并列照录表格数字并标注不对应，不替论文选口径。
- 摘要"up to 4000x faster"为约数，对应 Table 2 的 4,462×（CIFAR-10 吞吐）。
- pdftotext 转换质量良好，公式与表格数字均可读；图 3–9（生成样例）为图像内容，不落数字。
- 论文正文两处对 CIFAR-10 bpd 优势的归因（Table 2 caption：linear 完成约 3 倍 epoch"which results in better perplexity"；§4.2.2："the main point of this experiment is not the final perplexity"）已按论文自述口径写入边界，不作为质量优越性论断。

# Ingest 报告：DeepSeek-V2 技术报告（arXiv:2405.04434）

## 创建
- `wiki/multi-head-latent-attention.md` — WHY：全库无 MLA 技术页，它只作为 DSA/V3.2 的背景被间接引用；本页承载 KV 联合低秩压缩 + decoupled RoPE 的完整数据流（5 个 Stage，含吸收后的推理形态）、四机制 cache 元素数公式与 DeepSeek-V2 参数代入、吸收等价的代数推导与计算量变化，并把「MLA / MQA / GQA / 淘汰量化 / DSA」五个切面分开定位。
- `wiki/source-deepseek-v2-2405-04434.md` — WHY：该源此前只有二手转述，缺 source-summary；页面覆盖 MLA 立论（表 8 的 MHA > GQA > MQA 对照）、机制（式 9–19）、设置（60 层 / 128 头 / 8.1T token / YaRN 扩到 128K）与两规模对照数字（表 9），并交代摘要三项数字的整体口径。

## 修改
- `wiki/index.md` — WHY：Sources 段按字母序在 `source-deepfec` 与 `source-deepstate` 之间插入源页条目；Techniques 段在 `multi-granularity-sea-ice-forecasting` 与 `multi-level-alignment` 之间插入技术页条目。
- `wiki/log.md` — WHY：追加一条 ingest 记录（结论、落点、页面清单、数值备注）。

## 新建交叉链接
- [[multi-head-latent-attention]] ↔ [[source-deepseek-v2-2405-04434]]
- [[multi-head-latent-attention]] ↔ [[multi-query-attention]]（砍头数 vs 砍每 token latent 宽度）
- [[multi-head-latent-attention]] ↔ [[grouped-query-attention]]（2.25 组等价）
- [[multi-head-latent-attention]] ↔ [[kv-cache-compression]]（架构层每 token 体积 vs 推理期淘汰/量化，正交可叠加）
- [[multi-head-latent-attention]] ↔ [[deepseek-sparse-attention]]（cache 体积 vs 被 attend 的 token 数）
- [[multi-head-latent-attention]] ↔ [[rope]]（旋转矩阵不可交换 → decoupled 分支）
- [[multi-head-latent-attention]] ↔ [[deepseek-v3-2]]、[[source-2512-02556-deepseek-v3-2]]（后续沿用 MLA）
- [[source-deepseek-v2-2405-04434]] ↔ [[multi-head-latent-attention]]、[[grouped-query-attention]]、[[multi-query-attention]]

## 数值备注
- **表 9 的 MHA 基线列不自洽**：大 MoE + MHA 的 KV cache 记 860.2K 元素/token，按正文 $n_h=128$、$d_h=128$、$l=60$ 应为 1,966,080（1966K）；反推每层 $n_hd_h=7168$，说明该对照的 MHA 模型用的是另一套头配置。MLA 列 34.6K $=(512+64)\times60$ 可精确复算，小模型列 15.6K $=576\times27$（DeepSeek-V2-Lite 27 层）亦可。两列的比值 14%/4% 与正文一致，故只记录不自洽、不改数字。
- **表 9 小模型 C-Eval 反向**：MLA 50.9 vs MHA 51.6，是四格中唯一 MLA 更低的单元格；正文只写「MLA shows better performance」，未提该例外。
- **93.3% 的 KV cache 降幅不是对同配置 MHA**：口径为部署中的 DeepSeek 67B（GQA）+ MLA 侧 6 bit KV 量化。按 67B 的 8 组、95 层、16 bit 复算 $1-(34560\times6)/(194560\times16)=93.34\%$，与摘要一致；67B 的层数与组数取自其自身配置，V2 论文未列，复算标 [INFERENCE]。同配置 MHA 对 MLA 的比值是 1.76%（56.9×），非 93.3%。
- **42.5% 训练成本与 5.76× 吞吐是整体账**：300.6K → 172.8K GPU 小时/万亿 token、单节点 8×H800 上 >50K token/s 均含 DeepSeekMoE 少激活参数与 FP8/6 bit 量化，不能单独归因 MLA；页面已写明口径。
- **无 $d_c$ / $d^R_h$ 消融**：§D 只有 MHA/GQA/MQA（表 8，7B dense）与 MLA-vs-MHA（表 9，MoE 两规模）两组对照；$d_c=4d_h$、$d^R_h=d_h/2$ 是超参选择，论文未扫描。
- **PDF 版本**：arXiv 2405.04434v5（52 页，2024-06）；正文与附录均读到（§1–§4、附录 C 全公式、附录 D 消融），§4 对齐部分只核了与 MLA 无关的结论段。文本经 `pdftotext -layout` 转换，公式符号有断字（如 `𝑐 𝑡𝐾𝑉`），按上下文还原，未改数值。

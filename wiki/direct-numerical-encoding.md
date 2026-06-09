---
title: "Direct Numerical Encoding"
type: technique
tags:
  - large-language-model
  - tokenization
  - numerical-encoding
  - fine-tuning
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: medium
status: active
---

# Direct Numerical Encoding

**Direct Numerical Encoding** is the single-token floating-point encoding scheme introduced by [[st-vision-llm|ST-Vision-LLM]] to overcome LLMs' inefficiency at handling numbers, which are otherwise tokenized into multiple, less-meaningful character-level tokens[^src-st-vision-llm]. It represents real numbers in a discrete vocabulary so that an entire float becomes one token, substantially compressing the numeric portion of an LLM's input and output[^src-st-vision-llm].

## Token Design

A dedicated numerical vocabulary `V_FP` is built from special tokens of the form `⟨|FP m/b|⟩`, with the mapping[^src-st-vision-llm]:

$$ \langle|\text{FP } m/b|\rangle \mapsto \text{Norm}(m) \times 10^{b}, \qquad \text{Norm}(m) = \frac{m}{10^{\lfloor \log_{10}|m| \rfloor}} $$

- `m` is the **integer mantissa** with range {−999,…,−1} ∪ {1,…,999} (three significant digits).
- `b` is the **base-10 exponent** with range {−4,…,5}.
- `Norm(m)` places the decimal after the first significant digit, decoupling the float's magnitude from `m` so magnitude is governed solely by `b`.
- Zero is handled by a dedicated `⟨|FP0/0|⟩` token.

Iterating all `(m,b)` combinations yields a vocabulary covering the dynamic range of the Telecom Italia traffic data[^src-st-vision-llm]. The design is inspired by Charton (2022)'s *Linear Algebra with Transformers*, but adapted to **fine-tuning a pretrained LLM** rather than training one from scratch on pure numerical tokens[^src-st-vision-llm].

## Two-Stage Numerical Alignment Fine-tuning

A newly initialized FP embedding is meaningless to a pretrained model, so two alignment stages bridge the gap[^src-st-vision-llm]:

**Stage 1 — Semantic alignment.** Extend the input/output embedding matrices to include `V_FP`; **freeze the backbone** and fine-tune only the new embeddings on bidirectional transcription tasks (e.g. "convert '-82.100000' → `⟨|FP-821/1|⟩`" and the reverse). This grounds each numeric token near its textual-numeral counterpart in semantic space[^src-st-vision-llm].

**Stage 2 — Basic arithmetic alignment.** **Unfreeze** the backbone, token-embedding, and output layers and continue training with **LoRA** on three fundamental linear-algebra operations — vector **addition**, **subtraction**, and the **Hadamard product** — chosen to inject numeracy via templated examples while exercising approximate-linear-estimation and element-wise mechanisms[^src-st-vision-llm]. Each task mixes three I/O formats (token→string, string→token, token→token); the token→string format is used only for alignment/intermediate supervision, while the final forecasting task always emits numerical tokens[^src-st-vision-llm].

## Efficiency Impact

Single-token encoding is the main driver of ST-Vision-LLM's efficiency[^src-st-vision-llm]:

- Floats in [0, 10000] with six decimals: decimal strings average **10.89 tokens** vs **1 token** here (90.81% reduction)[^src-st-vision-llm].
- At S=12, K=36: numeric sequence 614.69→48.00 tokens; full context 1034.57→467.87 (54.77%); output sequence 465.01→39.00 (91.61%)[^src-st-vision-llm].
- Because LLM decoding is autoregressive, this output-length reduction translates directly into latency: in the ablation, the encoding variant needs only **13 output tokens** vs 112 (decimal string) / 42 (integer approximation), and cuts single-cell latency from 2.13s to 0.41s[^src-st-vision-llm].

## Trade-off

The decimal-string and integer-approximation variants reach slightly **lower** error than numerical encoding, but at 112 / 42 output tokens respectively; the paper argues single-token encoding is the better balance of predictive performance and generation efficiency, especially since long autoregressive outputs dominate decoding cost[^src-st-vision-llm]. The two-stage alignment adds ~54.58M training tokens (~6 hours on one RTX 5060), which the authors deem acceptable[^src-st-vision-llm].

## Related Pages

- [[st-vision-llm]] — the model that introduces this encoding
- [[source-st-vision-llm]] — source summary
- [[patch-based-tokenization]] — analogous "compress into fewer tokens" idea for the input side
- [[mobile-traffic-forecasting]] — application domain

[^src-st-vision-llm]: [[source-st-vision-llm]]

OpenReview
.net
Login
back arrowGo to ICML 2026 Conference homepage
TriAttention: Efficient Long Reasoning with Trigonometric KV Compression
Download PDF
Weian Mao, Xi Lin, Wei Huang, Yuxin Xie, Tianfu Fu, Bohan Zhuang, Song Han, Yukang Chen
Published: 01 May 2026, Last Modified: 25 Jun 2026ICML 2026 regularEveryone
Revisions
BibTeX
CC BY-NC-SA 4.0
Abstract:

Extended reasoning in large language models (LLMs) requires long and accurate decoding and creates severe KV cache memory bottlenecks. Leading KV cache compression methods estimate KV importance using attention scores from recent post-RoPE queries. However, queries rotate with position during RoPE, making representative queries very few, leading to poor top-key selection and unstable reasoning. To avoid this issue, we turn to the pre-RoPE space, where we observe that Q and K vectors are highly concentrated around fixed non-zero centers and remain stable across positions—Q/K concentration. We show that this concentration causes queries to preferentially attend to keys at specific distances (e.g., nearest keys), with the centers determining which distances are preferred via a trigonometric series. Based on this, we propose TriAttention to estimate key importance by leveraging these centers. Via the trigonometric series, we use the distance preference characterized by these centers to score keys according to their positions, and also leverage Q/K norms as an additional signal for importance estimation. On AIME25 with 32K-token generation, TriAttention matches Full Attention reasoning accuracy while achieving 2.5× higher throughput or 10.7× KV memory reduction, whereas leading baselines achieve only about half the accuracy at the same efficiency.
Lay Summary:

Modern AI systems like ChatGPT can reason through complex problems step by step, but this requires storing a large amount of intermediate information in memory—a component called the "KV cache." As reasoning chains get longer, this memory requirement grows dramatically, making such systems expensive and slow. Existing methods try to solve this by discarding less important parts of the memory. However, they rely on clues that shift and become unreliable as the AI generates more text, leading to poor decisions about what to keep. We discovered a surprisingly stable pattern in how these AI models process information: the internal representations that guide attention remain consistent throughout the reasoning process. By exploiting this stability, our method—TriAttention—can predict which parts of memory are important without observing the actual computation, using a mathematical structure similar to a trigonometric series. The result: TriAttention can cut memory usage by over 10× or increase processing speed by 2.5×, while maintaining the same reasoning accuracy as using the full memory. This makes long, complex reasoning tasks significantly more efficient and accessible.
Link To Code: https://github.com/WeianMao/triattention
Primary Area: Deep Learning->Large Language Models
Keywords: KV Cache, Large Language Models, Attention, RoPE, Efficient Inference, Reasoning
Originally Submitted PDF:  pdf
Submission Number: 10803
Filter by reply type...
Filter by author...
17 / 17 replies shown
Paper Decision
Decisionby Program Chairs30 Apr 2026, 23:57 (modified: 25 Jun 2026, 07:34)EveryoneRevisions
Decision: Accept (regular)
Comment:

This paper studies KV-cache compression for long-output reasoning workloads. The authors propose TriAttention, a method that estimates key importance using a triangular-series-based distance prior together with Q/K norm signals, whose performance advantages over several existing baselines have been verified by empirical studies. There have been abundant discussions during the rebuttal, and the authors' responses have addressed all the major concerns, especially those new experimental results. The paper received consistently positive evaluations from 4 reviewers after the rebuttal. I would recommend an accept.
Reference Correctness Check:

Some references in the submitted paper were flagged by an automated checker. As you are preparing the camera-ready version, please check the flagged references and correct any inaccuracies (authors, title, venue, arXiv ID, etc.). Note that the automated checker can produce false positives.

Flagged references:

    Hong, X., Dai, C., Li, B., Wu, S., Wang, Z., Wu, H., Wang, D., Zhu, J., He, S., and Sun, J.-R. On the token distance modeling ability of higher rope attention dimension. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing (EMNLP), 2024.
    Issue: could not validate authors

    Su, J., Ahmed, M., Lu, Y., Pan, S., Bo, W., and Liu, Y. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024.
    Issue: could not validate authors

Official Review of Submission10803 by Reviewer 7qmr
Official Reviewby Reviewer 7qmr13 Mar 2026, 15:11 (modified: 25 Jun 2026, 07:54)EveryoneRevisions
Summary:

Authors observed that the queries and keys form clusters in pre-ROPE space that stay consistent as new queries and keys are observed. Based on this observation, authors suggested a trigonometric series scoring function along with norm-based scoring with future-offsets to determine the important keys to retain.
Strengths And Weaknesses:
Strengths

    Theoretically motivated: the trigonometric series decomposition of attention logits is analytically derived from the RoPE formula under Q/K concentration, and empirically validated via reconstruction correlation across multiple architectures.
    Consistently outperforms the chosen baselines SnapKV and R-KV across all models and KV budget levels on the chosen tasks.
    Provides high throughput as a consequence of relaxed KV cache, achieving 2.5x higher throughput and 10.7x KV memory reduction relative to Full Attention at equivalent accuracy on AIME25.
    Cross-domain robustness: calibration statistics computed on coding data transfer well to reasoning benchmarks, suggesting the learned Q/K centers reflect model-intrinsic structure rather than task-specific patterns.

Weaknesses

    Relatively few baselines; notably absent are more recent or stronger methods such as LazyEviction and Ada-KV, which are cited in the paper but not included in the comparison.
    Tasks are focused on math evaluations and graph recursion. Evaluations on diverse tasks like summarization, retrieval, and Q&A would be more appropriate to test the general validity of the scoring function.
    The method requires offline calibration to compute Q/K centers. While cross-domain transfer is briefly tested, the sensitivity to calibration set size and quality is not analyzed.
    The scoring function computes importance over a fixed set of future offsets D = {1, 2, 4, …, 2^16}, but the choice of this set is not ablated or justified, leaving it unclear how sensitive performance is to this design decision.

Soundness: 3: good
Presentation: 3: good
Significance: 3: good
Originality: 3: good
Key Questions For Authors:

See weaknesses.
Limitations:

Yes.
Overall Recommendation: 5: Accept: Technically solid paper, with high impact on at least one sub-area of AI or moderate-to-high impact on more than one area of AI, with good-to-excellent evaluation, resources, reproducibility, and no unaddressed ethical considerations.
Confidence: 3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.
Compliance With LLM Reviewing Policy: Affirmed.
Code Of Conduct Acknowledgement: Affirmed.
Final Justification:

The authors have addressed all the weaknesses I mentioned.
Rebuttal by Authors
Rebuttalby Authors31 Mar 2026, 15:54 (modified: 25 Jun 2026, 11:24)EveryoneRevisions
Rebuttal:

We thank Reviewer 7qmr for the constructive feedback.

W1 & W2: Diverse tasks and baselines. We have compared with both LazyEviction and Ada-KV as requested, along with other baselines. We now cover all requested task types: Q&A (6 subtasks), summarization (3 subtasks), and retrieval (RULER + passage_retrieval), plus code, dialogue, and few-shot tasks.

On LongBench (Bai et al., ACL 2024; 16 subtasks covering long-document QA, summarization, dialogue, retrieval, and code; Qwen3-8B, 50% KV):
Method 	16-subtask Avg 	#Best (out of 16)
TriAttention (ours) 	48.1 	11
SnapKV 	45.2 	2
PyramidKV 	42.7 	1
StreamingLLM 	39.4 	1
KnormPress 	35.1 	1

We also compare with H2O, which requires O(n²) memory and cannot use Flash Attention. On the 12 subtasks where H2O runs within 48GB GPU memory, TriAttention wins 10 out of 12. The full per-subtask LongBench comparison table (16 subtasks × 6 methods) and the detailed H2O comparison are both in our response to Reviewer Ct9M.

On RULER (Hsieh et al., COLM 2024; retrieval, Qwen3-8B, 50% KV, 4K context):
Method 	RULER Avg
TriAttention (ours) 	66.1
StreamingLLM 	61.1
SnapKV 	55.6
PyramidKV 	40.7

Regarding Ada-KV: Ada-KV is an orthogonal plug-in that adds head-adaptive budgeting on top of any scoring method (bold = best):
	NarrQA 	Qasp 	MFQA 	HpQA 	2Wik 	Musi 	GovR 	QMSu 	MNew 	TREC 	TriQA 	SSum 	PaRe 	PaCn 	LCC 	ReBe 	Avg
SnapKV 	26.9 	37.0 	45.5 	59.2 	46.3 	33.1 	32.2 	23.0 	23.4 	40.5 	89.9 	41.0 	91.1 	7.5 	66.4 	59.9 	45.2
AdaKV+Snap 	27.0 	36.8 	45.0 	59.6 	47.5 	34.1 	31.8 	23.0 	23.6 	46.0 	90.1 	40.9 	90.8 	8.0 	65.6 	59.6 	45.6
Ours 	28.1 	43.0 	51.4 	60.2 	44.9 	36.9 	32.9 	23.8 	24.3 	69.0 	90.3 	39.9 	91.0 	7.0 	65.0 	61.3 	48.1

TriAttention outperforms AdaKV+SnapKV (48.1 vs 45.6, winning 12/17 columns), suggesting that our trigonometric scoring function provides a stronger base than attention-based scoring.

Regarding LazyEviction: We implemented TriAttention within LazyEviction's official evaluation framework for a fair comparison. On AIME24 (DeepSeek-R1-Distill-Qwen-7B, 30 problems) at 30% KV budget (* = cited from their paper):
Method 	AIME24 Acc (30% KV)
FullKV (no compression) 	46.7
TriAttention (ours) 	46.7
R-KV* 	43.3
LazyEviction 	43.3
RaaS* 	36.7
TOVA* 	36.7
H2O* 	33.3

TriAttention matches FullKV and outperforms all baselines including LazyEviction (+3.4) and TOVA (+10.0). We also evaluate at other budgets:
KV Budget 	Ours 	LazyEviction 	FullKV
10% 	40.0% 	33.3% 	46.7%
20% 	43.3% 	40.0% 	46.7%
30% 	46.7% 	43.3% 	46.7%

TriAttention outperforms LazyEviction at all budgets, and matches FullKV at 30%.

W3: Calibration sensitivity.

Size sensitivity (Qwen3-8B, tested on AIME24):
Calibration Tokens 	Acc (%)
50k 	45.4
200k 	45.8
960k 	45.8

No clear trend is observed — performance remains stable regardless of calibration size.

Quality sensitivity (Qwen3-8B, tested on AIME24):
Calibration Dataset 	Acc (%)
Google homepage HTML (low quality) 	46.2
LiveCodeBench (moderate quality) 	43.3
ShareGPT (higher quality) 	46.7

No clear correlation between calibration data quality and accuracy, confirming robustness to calibration data choice.

W4: Future offset D. We ablate the offset set D on AIME24 (Qwen3-8B).

Number of offsets (all geometric spacing):
Max Distance 	#Offsets 	Acc (%)
128 	8 	41.7
4096 	13 	48.8
8192 	14 	46.2
65536 (baseline) 	17 	45.8

Spacing strategy (both 17 offsets, range [1, 65536]):
Spacing 	Acc (%)
Geometric (baseline): log-spaced {1,2,4,...} 	45.8
Linear: uniform spacing {1,4097,8193,...} 	28.7

Future offsets are clearly beneficial (128→4096: +7.1%), and geometric spacing is critical (vs linear: +17.1%), as near-distance positions require dense sampling. Our default D (max=65536) was not tuned yet already achieves strong results; the ablation shows further headroom at max=4096 (48.8%). The motivation for scoring over future distances is explained in our response to Reviewer gCHe (W2(c)).
Replying to Rebuttal by Authors
Rebuttal Acknowledgement by Reviewer 7qmr
Rebuttal Acknowledgementby Reviewer 7qmr05 Apr 2026, 00:42 (modified: 25 Jun 2026, 12:46)EveryoneRevisions
Acknowledgement: (a) Fully resolved - My concerns have been adequately addressed. If you select this option, please consider adjusting your score accordingly.
Reasons:

I appreciate that the authors have resolved all my concerns, I recommend acceptance.
Replying to Rebuttal Acknowledgement by Reviewer 7qmr
Reply Rebuttal Comment by Authors
Reply Rebuttal Commentby Authors07 Apr 2026, 09:05 (modified: 25 Jun 2026, 14:13)EveryoneRevisions
Comment:

We sincerely thank Reviewer 7qmr for the positive feedback and for increasing the score. We will integrate the additional analysis on calibration sensitivity and future offsets into our final revision. Furthermore, we are committed to open-sourcing our code and statistics to facilitate follow-up research for the community. Your constructive guidance has been invaluable in strengthening this work.
Official Review of Submission10803 by Reviewer S1NU
Official Reviewby Reviewer S1NU12 Mar 2026, 11:58 (modified: 25 Jun 2026, 07:54)EveryoneRevisions
Summary:

The authors observed an aggregation phenomenon of Q K vectors in the pre-RoPE space. Based on this observation and the derived regularity, a method is proposed to predict future critical kv cache, thereby preventing incorrect premature eviction of kv cache. Experiments conducted on mathematical reasoning benchmarks (AIME and Math 500) demonstrate improved throughput and reduced memory usage, with minimal performance loss.
Strengths And Weaknesses:

Strengths

    The authors' observation is novel and has the potential to promote subsequent research in this field.

    The experimental design is well-considered, including a head-aware algorithm and thoughtful ablation experiments.

    The manuscript is easy to follow, with clear logic and coherent expression.

Weakness

    Experiments were conducted solely on mathematical reasoning benchmarks; thus, the method's stability in non-reasoning and non-mathematical scenarios remains unconfirmed.

    There appears to be a typographical error in the citation within the first sentence of the Introduction, which likely resulted in an inaccurate reference.

Soundness: 3: good
Presentation: 3: good
Significance: 3: good
Originality: 3: good
Key Questions For Authors:

Was this aggregation phenomenon observed by analyzing samples from the AIME dataset? Does this phenomenon hold true for other types of datasets?
Limitations:

Experiments were only tested on mathematical reasoning benchmarks and small models.
Overall Recommendation: 4: Weak accept: Technically solid paper that advances at least one sub-area of AI, with a contribution that others are likely to build on, but with some weaknesses that limit its impact (e.g., limited evaluation). Please use sparingly.
Confidence: 4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.
Compliance With LLM Reviewing Policy: Affirmed.
Code Of Conduct Acknowledgement: Affirmed.
Final Justification:

Thank you for addressing my concerns. The score has been upgraded.
Rebuttal by Authors
Rebuttalby Authors31 Mar 2026, 15:54 (modified: 25 Jun 2026, 11:24)EveryoneRevisions
Rebuttal:

We thank Reviewer S1NU for the encouraging feedback.

W1: Diverse benchmarks. We now evaluate on LongBench (16 subtasks covering QA, summarization, dialogue, retrieval, and code; Qwen3-8B, 50% KV):
Method 	16-subtask Avg 	#Best (out of 16)
TriAttention (ours) 	48.1 	11
SnapKV 	45.2 	2
PyramidKV 	42.7 	1
StreamingLLM 	39.4 	1
KnormPress 	35.1 	1

We also compare with H2O, winning 10/12 subtasks where it fits in 48GB GPU memory, and outperform Ada-SnapKV with 48.1 vs 45.6 avg. Full per-subtask tables are in our response to Reviewer Ct9M.

On RULER (Hsieh et al., COLM 2024; retrieval, Qwen3-8B, 50% KV, 4K context):
Method 	RULER Avg
TriAttention (ours) 	66.1
StreamingLLM 	61.1
SnapKV 	55.6
PyramidKV 	40.7

W2: Citation typo. Fixed in the revision. We thank the reviewer for catching this.

Q1: Q/K concentration on non-math data. The Q/K concentration was initially observed on math-related data. However, we have since verified that it holds across all data types we tested, including code and dialogue. We measured Q/K vectors across three domains and computed their Mean Resultant Length (MRL) — a directional concentration metric where 1.0 = perfectly clustered and 0.0 = uniformly dispersed:

(1) MRL across domains (Qwen3-8B):
Domain 	Dataset 	Mean MRL 	% Heads with R > 0.95
Math 	MATH-500 	0.977 	88.9%
Coding 	HumanEval 	0.979 	89.4%
Chat 	ShareGPT 	0.980 	90.0%

~90% of attention heads exhibit strong concentration (R > 0.95) regardless of input domain, with cross-domain variation ~1%. This confirms Q/K concentration is a model-intrinsic property.

(2) Diverse task performance: Using coding-calibrated Q/K centers, TriAttention achieves the best average on LongBench across 16 subtasks spanning QA, summarization, dialogue, retrieval, and code (see our response to Reviewer Ct9M for the full table). If concentration were math-specific, the method would fail on non-math tasks.

Regarding model scale: beyond Qwen3-8B, our paper also reports results on GPT-OSS (20B parameters) in Table 1. The Q/K concentration phenomenon and TriAttention's effectiveness are consistently strong across both scales, with no indication of degradation at larger model size.
Replying to Rebuttal by Authors
Rebuttal Acknowledgement by Reviewer S1NU
Rebuttal Acknowledgementby Reviewer S1NU03 Apr 2026, 22:30 (modified: 25 Jun 2026, 12:46)EveryoneRevisions
Acknowledgement: (c) Partially resolved or unresolved, but the remaining concerns are not easily addressed in a short rebuttal - Please select this option sparingly and only when you believe that your questions concern the core tenets of the work, and addressing them requires a significant update to the paper.
Reasons:

Thank you for your response. My concerns have been partially addressed, and this observation is indeed impressive. Nevertheless, the method still appears to have certain limitations and cannot adapt to real-world complex scenarios such as multi-turn interactions and task-mix settings. Since: "If concentration were math-specific, the method would fail on non-math tasks." Accordingly, I will maintain my original score.
Replying to Rebuttal Acknowledgement by Reviewer S1NU
Reply Rebuttal Comment by Authors
Reply Rebuttal Commentby Authors04 Apr 2026, 02:18 (modified: 25 Jun 2026, 14:13)EveryoneRevisions
Comment:

Thanks for your reply.

However, this must be a misunderstanding for the sentence "If concentration were math-specific, the method would fail on non-math tasks". This sentence is a counterfactual statement in English. It argues by contradiction to show that Q/K concentration holds beyond math domains, as confirmed by its strong performance across all these tasks. The truth is:

(1) The concentration phenomenon is not math-specific. Our method does work well on non-math tasks.

(2) Our method works well in task-mix settings. The evaluation on Q&A (6 subtasks), summarization (3 subtasks), and retrieval (RULER + passage_retrieval), plus code, dialogue, and few-shot tasks uses the same model and setting. Even if the method is calibrated on the math, it works well for all these tasks. Please see the table below.

Dialogue and diverse tasks (LongBench). (Bai et al., ACL 2024; 16 subtasks, Qwen3-8B, 50% KV). Bold = best among compression methods.

QA: NarrQA, Qasper, MFQA, HotpotQA, 2WikiMQA, MuSiQue; Summ: GovReport, QMSum, MultiNews; Few-shot: TREC, TriviaQA, SAMSum (dialogue); Retrieval: PassRet; Counting: PassCnt; Code: LCC, RepoBench
	NarrQA 	Qasp 	MFQA 	HpQA 	2Wik 	Musi 	GovR 	QMSu 	MNew 	TREC 	TriQA 	SSum 	PaRe 	PaCn 	LCC 	ReBe 	Avg
Full 	28.8 	43.8 	55.3 	62.8 	48.9 	35.5 	33.5 	24.7 	24.7 	40.5 	90.5 	40.3 	91.8 	9.0 	64.9 	60.0 	47.2
SnapKV 	26.9 	37.0 	45.5 	59.2 	46.3 	33.1 	32.2 	23.0 	23.4 	40.5 	89.9 	41.0 	91.1 	7.5 	66.4 	59.9 	45.2
PyramidKV 	25.9 	30.4 	39.1 	52.2 	39.9 	29.7 	29.9 	22.2 	21.9 	33.5 	90.0 	40.7 	93.4 	8.0 	65.7 	60.2 	42.7
StreamLLM 	24.1 	30.5 	31.2 	46.5 	41.6 	21.6 	30.8 	21.8 	23.9 	43.0 	85.4 	38.2 	55.2 	10.0 	64.7 	61.2 	39.4
KnormPress 	17.6 	24.4 	40.3 	29.2 	26.4 	14.9 	28.8 	21.6 	20.9 	50.0 	81.6 	41.2 	79.1 	7.1 	31.7 	47.5 	35.1
Ours 	28.1 	43.0 	51.4 	60.2 	44.9 	36.9 	32.9 	23.8 	24.3 	69.0 	90.3 	39.9 	91.0 	7.0 	65.0 	61.3 	48.1

TriAttention achieves the highest average (48.1) across 16 diverse subtasks, winning 11 out of 16 subtasks.
Official Review of Submission10803 by Reviewer gCHe
Official Reviewby Reviewer gCHe11 Mar 2026, 21:11 (modified: 25 Jun 2026, 07:54)EveryoneRevisions
Summary:

This paper studies KV-cache compression for long-output reasoning workloads. The authors argue that existing approaches that operate in the post-RoPE space suffer from instability because only a small number of recent queries are representative. Instead, the paper analyzes the pre-RoPE space and observes a Q/K concentration phenomenon, which induces distance preferences in query-key interactions. Based on this insight, the authors propose TriAttention, a method that estimates key importance using a triangular-series-based distance prior together with Q/K norm signals. Experiments demonstrate that TriAttention maintains accuracy comparable to full attention while achieving up to 2.5× throughput improvement and 10.7× KV memory reduction.
Strengths And Weaknesses:

Strengths:

    Insightful observation and principled method：Identifies pre-RoPE Q/K concentration and distance preferences, deriving a trigonometric scoring function that connects internal geometry to key importance.
    Training-free and efficient：TriAttention leverages pre-RoPE statistics without modifying model weights, achieving substantial throughput and KV memory improvements.
    Empirical support：Method outperforms baselines (SnapKV, R-KV) on reasoning benchmarks, with ablations confirming the importance of both trigonometric and norm-based components.

Weaknesses:

    Limited evaluation scope and baselines：Experiments focus on math reasoning; other long-context tasks and recent KV compression methods are not evaluated.
    Theoretical and methodological approximations：The derivation assumes uniform Q/K centers per head; the role of norm-based scoring and scoring over future distances is only heuristically justified.

Soundness: 3: good
Presentation: 3: good
Significance: 3: good
Originality: 3: good
Key Questions For Authors:

    Reliability of the trigonometric approximation across heads and attention mechanisms：The method relies on the observation that pre-RoPE Q/K vectors tend to cluster around a stable center. Is the reconstruction quality consistent across different attention heads within the same attention mechanism? Additionally, how does the reconstruction quality vary across different attention mechanisms (e.g., standard multi-head attention vs. MLA)? An analysis of per-head reconstruction behavior would help clarify the robustness of the proposed approximation.

    Generalization beyond mathematical reasoning tasks: Current experiments primarily focus on mathematical reasoning benchmarks. Have the authors evaluated TriAttention in other long-context scenarios, such as long-document question answering, summarization, or dialogue history tasks? Demonstrating effectiveness beyond mathematical reasoning would strengthen the evidence for the method’s general applicability.

Limitations:

Yes.
Overall Recommendation: 4: Weak accept: Technically solid paper that advances at least one sub-area of AI, with a contribution that others are likely to build on, but with some weaknesses that limit its impact (e.g., limited evaluation). Please use sparingly.
Confidence: 4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.
Compliance With LLM Reviewing Policy: Affirmed.
Code Of Conduct Acknowledgement: Affirmed.
Final Justification:

Since my concerns have been fully addressed, I maintain my positive stance on this paper.
Rebuttal by Authors
Rebuttalby Authors31 Mar 2026, 15:55 (modified: 25 Jun 2026, 11:24)EveryoneRevisions
Rebuttal:

We thank Reviewer gCHe for the thorough review.

W1: Generalization and baselines. We now evaluate on LongBench (16 subtasks covering long-document QA, summarization, dialogue, retrieval, and code; Qwen3-8B, 50% KV):
Method 	16-subtask Avg 	#Best (out of 16)
TriAttention (ours) 	48.1 	11
SnapKV 	45.2 	2
PyramidKV 	42.7 	1
StreamingLLM 	39.4 	1
KnormPress 	35.1 	1

We also compare with H2O, winning 10/12 subtasks where it fits in 48GB GPU memory. Full per-subtask tables are in our response to Reviewer Ct9M. We also outperform LazyEviction at all tested KV budgets on AIME24 (details in our response to Reviewer 7qmr). We also outperform AdaKV+SnapKV with 48.1 vs 45.6 avg, winning 12/17 subtasks (details in our response to Reviewer 7qmr).

On RULER (Hsieh et al., COLM 2024; retrieval, Qwen3-8B, 50% KV, 4K context):
Method 	RULER Avg
TriAttention (ours) 	66.1
StreamingLLM 	61.1
SnapKV 	55.6
PyramidKV 	40.7

W2: Theoretical and methodological approximations.

(a) Uniform Q/K center assumption. As shown in Figure 2(c) of our paper, the Mean Resultant Length (MRL) — a measure of directional concentration — is high for the vast majority of attention heads, confirming that Q/K vectors do cluster tightly around a common center. This validates the uniform-center approximation.

(b) Norm-based scoring. The term is grounded in empirical observation: we find that semantically important tokens (e.g., entity names, punctuation, discourse markers) consistently exhibit larger key norms, making norm a reliable proxy for semantic salience. Ablation on AIME24 confirms:
Config 	Acc (%)
Full scoring (trig + norm) 	45.8
Remove norm (trig only) 	40.4 (-5.4)

(c) Scoring over future distances. Some attention heads preferentially attend to distant keys. This means a KV entry that is unimportant now may become critical as it grows more distant during decoding. Methods that score based only on the current query cannot foresee this and risk premature eviction. Our future offset set D = {1, 2, 4, ..., 2^16} addresses this by estimating importance for unseen future queries, not just the current one. Ablation on AIME24 confirms:

Number of offsets (geometric spacing):
Max Distance 	#Offsets 	Acc (%)
128 	8 	41.7
4096 	13 	48.8
8192 	14 	46.2
65536 (baseline) 	17 	45.8

Spacing strategy (17 offsets, range [1, 65536]):
Spacing 	Acc (%)
Geometric (baseline): log-spaced {1,2,4,...} 	45.8
Linear: uniform spacing {1,4097,8193,...} 	28.7

Future offsets are beneficial (128→4096: +7.1%), and geometric spacing is critical (vs linear: +17.1%). Our default D was not tuned yet already achieves strong results; ablation shows further headroom at max=4096 (48.8%).

The empirical success across 16 diverse LongBench tasks confirms these design choices are practically effective.

Q1: Per-head reconstruction consistency. As shown in Figure 3 of our paper, the per-head reconstruction correlation (r̄) peaks around 0.6–0.9 across all three tested architectures (Qwen3-8B, Qwen2.5, Llama3), with mean above 0.5. The distribution is right-skewed — the majority of heads achieve high correlation.

Q1: MLA. We evaluated on GLM-4.7-Flash (MLA architecture, 940 heads).

Pearson r (reconstruction quality):
Threshold 	Qwen3-8B (GQA) 	GLM-4.7 (MLA)
> 0.90 	0.8% 	1.7%
> 0.70 	13.0% 	23.1%
> 0.50 	53.5% 	51.6%

MRL (directional concentration):
Threshold 	Qwen3-8B (GQA) 	GLM-4.7 (MLA)
> 0.95 	84.7% 	96.6%
> 0.90 	90.8% 	99.8%

MLA shows comparable or stronger concentration and reconstruction quality, confirming our framework generalizes beyond standard attention.

Q2: Generalization beyond math. Addressed in W1 above. On LongBench (16 subtasks: QA, summarization, dialogue, retrieval, code), TriAttention achieves 48.1 avg (best among all methods). On RULER (retrieval), TriAttention achieves 66.1, outperforming SnapKV by +10.5.
Replying to Rebuttal by Authors
Rebuttal Acknowledgement by Reviewer gCHe
Rebuttal Acknowledgementby Reviewer gCHe02 Apr 2026, 11:21 (modified: 25 Jun 2026, 12:46)EveryoneRevisions
Acknowledgement: (a) Fully resolved - My concerns have been adequately addressed. If you select this option, please consider adjusting your score accordingly.
Reasons:

Thank you for your detailed response. My concerns have been fully addressed.
Replying to Rebuttal Acknowledgement by Reviewer gCHe
Reply Rebuttal Comment by Authors
Reply Rebuttal Commentby Authors07 Apr 2026, 09:24 (modified: 25 Jun 2026, 14:13)EveryoneRevisions
Comment:

We sincerely thank Reviewer gCHe for the supportive and constructive feedback. We will integrate our rebuttal experimental results into the final revision and open-source our implementation and statistics to support the community.

TriAttention represents a new paradigm for KV cache compression; more importantly, the pre-RoPE Q/K concentration phenomenon deepens our mechanistic understanding of RoPE-based attention by explaining how diverse attention patterns with distance preferences arise from underlying geometric centers. We believe this discovery has the potential to inspire future research in model architecture design and KV cache quantization. We hope this broader vision of our contribution’s impact helps as you finalize your evaluation. Thank you for your consideration!
Official Review of Submission10803 by Reviewer Ct9M
Official Reviewby Reviewer Ct9M10 Mar 2026, 16:45 (modified: 25 Jun 2026, 07:54)EveryoneRevisions
Summary:

Authors proposed TriAttention， Uses the trigonometric structure of RoPE attention to estimate KV importance, achieving training-free KV cache compression and improving long inference efficiency.
Strengths And Weaknesses:
Strengths

    estimates KV importance using the trigonometric expansion of RoPE attention, rather than pure heuristics.
    a purely inference-time method, directly applicable to existing LLMs.
    only requires simple trigonometric and norm calculations, with minimal extra cost.
    stable performance on long-context tasks such as math reasoning and long-chain reasoning.

Weaknesses

    The metric mainly relies on distance and vector norms, which may mistakenly evict semantically critical tokens. Semantic importance is not leveraged. How does it perform on retrieval and dialogue tasks? Because in these tasks, the distance pattern is not obvious.

Soundness: 3: good
Presentation: 3: good
Significance: 3: good
Originality: 3: good
Key Questions For Authors:

The metric mainly relies on distance and vector norms, which may mistakenly evict semantically critical tokens. Semantic importance is not leveraged. How does it perform on retrieval and dialogue tasks? Because in these tasks, the distance pattern is not obvious.
Limitations:

yes
Overall Recommendation: 4: Weak accept: Technically solid paper that advances at least one sub-area of AI, with a contribution that others are likely to build on, but with some weaknesses that limit its impact (e.g., limited evaluation). Please use sparingly.
Confidence: 3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.
Compliance With LLM Reviewing Policy: Affirmed.
Code Of Conduct Acknowledgement: Affirmed.
Final Justification:

My concerns have been fully addressed, and I maintain my positive stance on this paper.
Rebuttal by Authors
Rebuttalby Authors31 Mar 2026, 15:55 (modified: 25 Jun 2026, 11:24)EveryoneRevisions
Rebuttal:

We thank Reviewer Ct9M for the constructive feedback.

We evaluate on RULER (retrieval) and LongBench (16 diverse tasks including dialogue), and the results confirm TriAttention effectively preserves semantically critical tokens. We first explain the mechanism, then present the evidence.

How does the scoring capture semantic importance? Our scoring function includes a norm-based term (Eq. 7 in the paper), which weights each key by its norm across frequency bands. This captures token salience independent of position — we observe that semantically important tokens (e.g., entity names, punctuation, discourse markers) consistently have larger key norms and thus receive higher scores from .

Empirically: on RULER retrieval, TriAttention outperforms SnapKV by +10.5 — this would not be possible if semantically critical tokens were being evicted. On LongBench (16 diverse tasks where "distance pattern" varies widely), TriAttention ranks #1.

Retrieval results (RULER). (Hsieh et al., COLM 2024; 4K context, Qwen3-8B, 50% KV):
Method 	RULER Avg
TriAttention (ours) 	66.1
StreamingLLM 	61.1
SnapKV 	55.6
PyramidKV 	40.7

Dialogue and diverse tasks (LongBench). (Bai et al., ACL 2024; 16 subtasks, Qwen3-8B, 50% KV). Bold = best among compression methods.

QA: NarrQA, Qasper, MFQA, HotpotQA, 2WikiMQA, MuSiQue; Summ: GovReport, QMSum, MultiNews; Few-shot: TREC, TriviaQA, SAMSum (dialogue); Retrieval: PassRet; Counting: PassCnt; Code: LCC, RepoBench
	NarrQA 	Qasp 	MFQA 	HpQA 	2Wik 	Musi 	GovR 	QMSu 	MNew 	TREC 	TriQA 	SSum 	PaRe 	PaCn 	LCC 	ReBe 	Avg
Full 	28.8 	43.8 	55.3 	62.8 	48.9 	35.5 	33.5 	24.7 	24.7 	40.5 	90.5 	40.3 	91.8 	9.0 	64.9 	60.0 	47.2
SnapKV 	26.9 	37.0 	45.5 	59.2 	46.3 	33.1 	32.2 	23.0 	23.4 	40.5 	89.9 	41.0 	91.1 	7.5 	66.4 	59.9 	45.2
PyramidKV 	25.9 	30.4 	39.1 	52.2 	39.9 	29.7 	29.9 	22.2 	21.9 	33.5 	90.0 	40.7 	93.4 	8.0 	65.7 	60.2 	42.7
StreamLLM 	24.1 	30.5 	31.2 	46.5 	41.6 	21.6 	30.8 	21.8 	23.9 	43.0 	85.4 	38.2 	55.2 	10.0 	64.7 	61.2 	39.4
KnormPress 	17.6 	24.4 	40.3 	29.2 	26.4 	14.9 	28.8 	21.6 	20.9 	50.0 	81.6 	41.2 	79.1 	7.1 	31.7 	47.5 	35.1
Ours 	28.1 	43.0 	51.4 	60.2 	44.9 	36.9 	32.9 	23.8 	24.3 	69.0 	90.3 	39.9 	91.0 	7.0 	65.0 	61.3 	48.1

TriAttention achieves the highest average (48.1) across 16 diverse subtasks, winning 11 out of 16 subtasks.

H2O requires the full attention matrix (O(n²) memory), preventing the use of Flash Attention. On a 48GB GPU, it runs out of memory on longer subtasks. We report partial results where it could run:
	Qasp 	HpQA 	2Wik 	Musi 	GovR 	QMSu 	MNew 	TREC 	TriQA 	SSum 	NarrQA 	MFQA 	Avg
H2O 	39.2 	50.7 	43.9 	30.7 	32.9 	23.4 	24.4 	56.5 	89.1 	39.1 	21.2 	45.4 	41.4
Ours 	43.0 	60.2 	44.9 	36.9 	32.9 	23.8 	24.3 	69.0 	90.3 	39.9 	28.1 	51.4 	45.4
Replying to Rebuttal by Authors
Rebuttal Acknowledgement by Reviewer Ct9M
Rebuttal Acknowledgementby Reviewer Ct9M02 Apr 2026, 16:20 (modified: 25 Jun 2026, 12:46)EveryoneRevisions
Acknowledgement: (a) Fully resolved - My concerns have been adequately addressed. If you select this option, please consider adjusting your score accordingly.
Reasons:

Thank you for your detailed response. I maintain my positive stance on this paper.
Replying to Rebuttal Acknowledgement by Reviewer Ct9M
Reply Rebuttal Comment by Authors
Reply Rebuttal Commentby Authors07 Apr 2026, 09:25 (modified: 25 Jun 2026, 14:13)EveryoneRevisions
Comment:

We sincerely thank Reviewer Ct9M for the supportive and constructive feedback. We will integrate our rebuttal experimental results into the final revision and open-source our implementation and statistics to support the community.

TriAttention represents a new paradigm for KV cache compression; more importantly, the pre-RoPE Q/K concentration phenomenon deepens our mechanistic understanding of RoPE-based attention by explaining how diverse attention patterns with distance preferences arise from underlying geometric centers. We believe this discovery has the potential to inspire future research in model architecture design and KV cache quantization. We hope this broader vision of our contribution’s impact helps as you finalize your evaluation. Thank you for your consideration!
About OpenReview
Contact
FAQ
Hosting a Venue
Sponsors
Terms of Use / Privacy Policy
All Venues
Donate
News

OpenReview is a long-term project to advance science through improved peer review with legal nonprofit status. We gratefully acknowledge the support of the OpenReview Sponsors. © 2026 OpenReview

- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - What Is a Looped Transformer? Complete Guide to Recurrent Depth and OpenAI's Astra | Tosea.ai- - - - 

Tosea.ai
FeaturesShowcasePricingSecurityBlogTemplatesTools
Resources
DocsEN

Tosea.ai

←Back to BlogInsightsTosea TeamSeptember 2, 202615 MIN READ
What Is a Looped Transformer? Complete Guide to Recurrent Depth and OpenAI's Astra

OpenAI's Astra reportedly uses recurrent depth — looping hidden states through the same layers. What the architecture is, what the research actually shows, and why it started a safety fight.

On September 1, 2026, The Information reported that OpenAI's upcoming model, Astra, uses a technique called recurrent depth: instead of passing a token through a fixed stack of distinct layers exactly once, the model loops its hidden state through the same block of layers several times before emitting the next token. Within hours the story had moved from an architecture footnote to an AI-safety argument, because looping shifts part of the model's reasoning out of readable chain-of-thought text and into numerical latent states that nobody can read.

The reaction was fast and unusually blunt. Steven Adler, a former OpenAI safety researcher, posted that "OpenAI seems to be violating one of the few redlines that exist in the AI industry." Gary Marcus ran the headline "Red Alert." By the next morning, OpenAI's Chief Scientist Jakub Pachocki had replied directly, saying the reporting was confused and that the computation-graph depth of OpenAI's current frontier models "including Astra, is within a factor of two of GPT-4."

That is a lot of heat for an architecture that has been in the open literature since 2018. This guide separates the three things that got tangled together: what a looped transformer actually is, what the published research genuinely demonstrates about it, and what is confirmed versus reported versus guessed about Astra specifically. If you followed our GPT-5.6 Sol, Terra and Luna coverage, treat this as the architectural sequel — Astra is the model that sits above that family, and it is the first one where the shape of the network became a public policy question.

What Is a Looped Transformer?

A standard transformer is a stack of distinct layers. A 64-layer model has 64 separate sets of weights, and every token passes through all 64 exactly once. Depth is fixed at training time, and the only way to buy more serial computation at inference is to make the model generate more tokens — which is precisely what chain-of-thought reasoning does.

A looped transformer (also called a recurrent-depth or recursive transformer) breaks that coupling. It replaces part of the stack with a single shared block that gets applied repeatedly. Run the block 4 times and you get roughly the serial computation of 4 stacked copies, but you only ever store one copy of the weights. Run it 32 times and the materialized depth grows to 32 copies while the parameter count stays flat.

Two consequences follow, and they are the reason anyone cares:

- Compute becomes a runtime dial. The same checkpoint can spend more or less thinking per token by changing the loop count, with no retraining. This is test-time compute scaling that does not require the model to write anything down.

- The reasoning is not in words. Each iteration refines a hidden state — a vector, not a sentence. In a chain-of-thought model, every step of serial reasoning must be projected down to a token you can read. In a looped model, that projection step is optional.

Point 2 is the whole controversy. Point 1 is the whole appeal.

The formal framing comes from the chain-of-thought monitorability position paper, which puts it precisely: in a standard transformer, "chain of thought is the only way that information can flow down from later to earlier layers. As a result, for a fixed amount of tokens, the longest flows of information must use the chain of thought." Looping adds a second route for that information — one that never surfaces as text.

How Recurrent Depth Actually Works

The clearest published implementation is Geiping et al., "Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach", from the ELLIS Institute Tübingen, the University of Maryland, and Lawrence Livermore National Laboratory. Their 3.5B-parameter model, Huginn, splits the network into three functional groups:

- Prelude (P) — a few ordinary transformer layers that embed the input tokens into a latent space.

- Core (R) — the recurrent block. It takes the previous latent state and the original embedded input, concatenates them through an adapter matrix, and runs them through its transformer layers to produce a new state. This block is applied r times.

- Coda (C) — a few final layers that decode the last latent state into next-token probabilities.

Two design details matter more than they look. The core is initialized from a random state, and the original embedding is re-injected at every iteration. Both come from the "deep thinking" literature and both exist to make the recurrence stable: they encourage the loop to converge to a steady state that does not depend on where it started, a property called path independence. Without them, iterating a shared block tends to either collapse or diverge.

The paper frames the result as a third scaling axis. Pretraining scales parameters. Chain-of-thought scales tokens. Recurrent depth scales serial computation directly, and the authors note the design is deliberately "compute-heavy, with more 'materialized' parameters than there are actual parameters." They draw an explicit contrast with mixture-of-experts models, which are the mirror image — parameter-heavy, compute-light:

"Where the recurrent-depth setup excels at learning reasoning patterns, the MoE excels at effectively storing and retrieving complex information."

That complementarity is worth holding onto. Looping and sparsity are not competing answers to the same question. If you have been following the architecture race through models like Kimi K3 and GLM-5.3-Flash, recurrent depth is the axis none of them have publicly turned yet.

What Looping Actually Buys You

Here is where most coverage stops and hand-waves. The Huginn paper ran the controlled experiment, and the numbers are specific: the same weights, the same 800B-token training run, with only the loop count changed at test time.

Benchmarkr = 1r = 8r = 16r = 32
ARC-E34.8965.1169.4969.91
ARC-C24.0635.1537.7138.23
HellaSwag29.3458.5464.6765.21
MMLU23.6025.2931.2531.38
OpenBookQA26.8035.4037.6038.80
PiQA55.3373.4575.7976.22
SciQ47.192.1093.9093.50
GSM8K CoT0.00 / 0.15——34.80 / 42.08

Source: Geiping et al., Tables 1, 2 and 4. GSM8K CoT reported as flexible/strict extract, with system prompt at r = 32.

At one iteration the model is barely above random on most tasks. At 32 iterations it is a credible small model. Nothing changed except how many times the same block ran. That is the cleanest demonstration available that latent recurrence is doing real computational work rather than reshuffling capacity.

Two honest caveats that vendor-style summaries tend to drop.

It saturates, and unevenly. Between r = 16 and r = 32 — a doubling of compute — ARC-E moves 0.42 points and SciQ actually drops. The paper's own Figure 1 shows why: easy tasks like OpenBookQA converge quickly, while GSM8K keeps climbing. Looping buys the most where serial reasoning is the bottleneck, and close to nothing where it is not. Anyone promising uniform gains from "more loops" is selling the average of a very lopsided distribution.

Data quality still dominates. Huginn at r = 32 scores 69.91 on ARC-E and 31.38 on MMLU. OLMo-2-1124-7B, trained on 4T carefully curated tokens, scores 82.79 and 60.56. The recurrent model is punching far above its 3.5B parameters and 0.8T tokens — it beats Pythia-12B on most metrics and beats OLMo-7B on HumanEval by 23.17 to 12.80 — but it does not repeal the data-scaling laws.

The most useful control in the paper is Table 4: a recurrent model and a fixed-depth baseline trained on identical data and setup for 180B tokens. The recurrent version scores 53.62 on ARC-E against 46.42, and 9.02 on GSM8K CoT against 1.82. The architecture itself, not the training recipe, produced the gap.

The Research Lineage

Recurrent depth is not a 2026 invention, and that context matters for judging how surprised anyone should be.

YearWorkContribution
2018Universal Transformers (Dehghani et al.)Applied a single shared transformer block recurrently with a halting mechanism — the original weight-tied transformer.
2019ALBERTCross-layer parameter sharing as a compression technique, showing a shared block need not hurt quality.
2024Coconut (Hao et al.)Feeds the last hidden state back as the next input embedding — "continuous thought" instead of tokens. Reasons by breadth-first search rather than committing to one path.
2025Huginn / recurrent depth (Geiping et al.)Scaled latent recurrence to 3.5B parameters and 800B tokens on Oak Ridge's Frontier supercomputer; showed test-time loop scaling to roughly 50B materialized parameters.
2025Mixture-of-Recursions (Bae et al.)Adds lightweight routers so different tokens get different recursion depths — adaptive computation plus parameter sharing in one framework.
2025Ouro / LoopLM (ByteDance et al.)Open-sourced 1.4B and 2.6B looped models trained on 7.7T tokens with an entropy-regularized objective for learned depth allocation, reporting parity with models up to 12B.

The Ouro result is the one most often cited in the Astra coverage, and it is the most directly relevant: it built looping into pre-training rather than bolting it on, and attributed the gains to "superior knowledge manipulation capabilities" rather than raw capacity. Ouro also added a learned exit mechanism, so the model decides how long to loop instead of using a fixed count — the practical version of the adaptive idea Universal Transformers proposed eight years earlier.

Why a Frontier Lab Wants This

The economics are straightforward. Chain-of-thought scaling is expensive in a specific, compounding way: every unit of extra reasoning is a token that must be generated serially, attended to by every subsequent token, and held in the KV cache. Latent recurrence buys serial depth without any of that. No extra output tokens, no context growth, no cache pressure.

For a lab running inference at OpenAI's volume, the second-order effects matter as much as the first. A smaller model that loops can match a larger model that does not, which cuts weight-loading bandwidth and memory footprint per replica. And because loop count is a runtime parameter, the same deployed checkpoint can serve a cheap tier and an expensive tier without maintaining two models.

There is a capability argument too, and it is not purely commercial. The Huginn authors point out that forcing all internal reasoning through a single verbalized next token "appears wasteful," and that some computation is simply not well represented in words — spatial reasoning, numerical intuition, the kind of thing humans do before they can explain it. The paper's Section 7 documents an emergent behavior it calls "orbiting": on prompts requiring numerical reasoning, the latent state settles into rotational trajectories rather than converging to a point. Whatever that is, it has no natural transcription into English.

The Monitorability Problem

Now the part that made this a fight rather than a paper.

In July 2025, forty-one researchers from OpenAI, Google DeepMind, Anthropic, the UK AI Security Institute, METR, Apollo Research and Redwood Research co-signed "Chain of Thought Monitorability: A New and Fragile Opportunity for AI Safety." The argument: because today's models must externalize long reasoning as text, we get a free window into their intent, and that window is worth protecting.

The paper has a subsection titled "Novel architectures." It reads:

"Researchers have recently explored changes to the model architectures that increase the serial depth of reasoning that models are capable of in a continuous latent space (Hao et al., 2024; Geiping et al., 2025). Such latent reasoning models might not need to verbalize any of their thoughts and would thus lose the safety advantages that CoT confers."

Those are the Coconut and recurrent-depth papers, cited by name. And among the co-authors are Mark Chen, David Farhi, Aleksander Mądry, Wojciech Zaremba, Bowen Baker — and Jakub Pachocki, OpenAI's Chief Scientist. This is why the reaction was sharp: the specific architecture class now reported to be in Astra was flagged as a monitorability risk, thirteen months earlier, in a paper OpenAI's own chief scientist signed.

Pachocki's response on September 2 is worth quoting in full rather than paraphrasing:

"I want to prevent a race into unmonitorability kicked off by confused reporting. The depth of the computation graph for our present frontier models, including Astra, is within a factor of two of GPT-4. OpenAI has worked to preserve and utilize chain-of-thought monitoring since our very first reasoning models. We deeply care about this technique, as it can give us a view into how model alignment generalizes from its training distribution. I do think it is fragile and unfortunately trending in a negative direction, for reasons not contingent on architecture changes that I will write about soon. But there are things we can do to strengthen it, and it's a core goal of our current research program."

That is a substantive claim, not a deflection: bounded loop count, total serial depth comparable to GPT-4, readable chain-of-thought preserved. It also concedes the broader point — monitorability is fragile and trending badly, for reasons that have nothing to do with looping.

Geoffrey Irving of the UK AI Security Institute, also a co-author on the monitorability paper, pushed back on the general form of the depth-bound argument:

"To get significant mileage out of bounding the depth of a circuit, you have to bound it very low. If the depth is 100s or the like ('every once in a while we spit out a token'), this is a fake bound. It is the equivalent of saying 'we monitor the CoT' without discussing error rates: reality grades based on numbers, not just binary 'we tried'."

Irving later acknowledged that a factor-of-two-of-GPT-4 bound is tighter than the scenario he was arguing against. The honest summary of where this stands: the safety community's objection is not that looping is inherently forbidden, it is that "we bounded it" is only meaningful with a number attached — and OpenAI has now attached one, without publishing the underlying measurement.

It is also worth noting what the monitorability paper itself says against its own thesis. It cautions that "firm commitments to protect CoT behavior in its current form may trade off against more effective safety interventions," and recommends CoT monitoring "only as an addition to existing safety research directions and not as a replacement." The paper's own suggested middle path — "hybrid approaches that maintain some visibility into reasoning while leveraging latent" computation — is, on OpenAI's account, roughly what Astra does.

The sharpest irony sits in OpenAI's own Path to Astra post, published September 1. Astra is the first model OpenAI has designated as meeting the Critical cybersecurity threshold under its Preparedness Framework. The safeguards it describes lean directly on the thing looping erodes: "we are deploying Astra with additional chain-of-thought monitoring to rapidly detect and contain potentially misaligned actions." The control mechanism and the architectural trend are pulling in opposite directions, and OpenAI is running both at once.

Confirmed, Reported, or Speculation

Almost every viral claim about Astra this week belongs in one of three buckets. Keeping them apart is the difference between analysis and rumor laundering.

ClaimStatusSource
Astra exists and is an upcoming OpenAI modelConfirmedOpenAI, Path to Astra, Sept 1, 2026
First model to meet the Critical cybersecurity thresholdConfirmedOpenAI, same post
100% on ExploitBench; found two zero-days during evaluationConfirmedOpenAI, same post
Refuses 91.5% of disallowed cyber requests vs 59% for GPT-5.6 SolConfirmedOpenAI, same post
Ships with additional chain-of-thought monitoringConfirmedOpenAI, same post
Computation-graph depth within 2× of GPT-4Confirmed (unverified measurement)Pachocki, Sept 2, 2026
Astra uses recurrent depth / looped layersReported, not officially confirmedThe Information, Sept 1, 2026
Loop depth was deliberately limited to preserve readable CoTReportedThe Information, via a source familiar
Astra is "GPT-6"SpeculationNo official statement
Specific parameter count, pricing, release dateUnknownOpenAI says only "soon," with advanced cyber capability limited to alpha testers

OpenAI has committed to publishing a system card at launch. Until then, treat the architecture claim as well-sourced reporting that the company has neither confirmed nor denied — Pachocki's reply disputed the implications, not the technique.

What Looped Transformers Mean for AI Slide Generation

Recurrent depth sounds distant from document work, but it lands on a real bottleneck in AI slide generation.

Turning a 40-page report into a coherent deck is not a text-length problem, it is a serial-reasoning problem. The model has to hold the argument's structure in mind, decide which claims are load-bearing, work out what belongs on one slide versus three, and keep the narrative consistent from the executive summary to the appendix. Today that reasoning is paid for in chain-of-thought tokens — which is why long-document outlining is one of the most expensive steps in any document-to-PPT pipeline, and why quality degrades on very long sources as the reasoning budget gets spread thin.

Latent recurrence targets exactly that cost. If a model can spend more serial computation on structural decisions without emitting thousands of reasoning tokens per slide, the economics of long-document decks change: outlining a 200-page filing stops being priced like generating a novel. That matters most for the workloads where massive decks are the norm — diligence packs, technical reviews, research-paper walkthroughs.

There is a catch that cuts the other way, and it is the same catch as everywhere else. Slide generation is a domain where you want to see the reasoning. When an AI presentation tool proposes a 14-slide structure, the readable justification for that structure is what lets a user correct it — and it is the same visible trace that grounding techniques rely on to keep claims tied to the source document rather than invented. Reasoning that disappears into latent space is reasoning you cannot audit, and in a deck that goes to an investment committee, an unauditable claim is worse than a slow one. Our zero-hallucination slide guide covers why source traceability, not raw model capability, is usually the binding constraint.

This is why Tosea.ai keeps the document-to-deck orchestration layer explicit rather than folding it into a single opaque model call: parsing, outline, per-slide reasoning and rendering stay as separate inspectable stages, so the presentation workflow does not inherit whatever visibility the underlying model gives up. Faster latent reasoning underneath is welcome. It just should not be the only place the reasoning lives.

Frequently Asked Questions

Is a looped transformer the same as a reasoning model?
No, and they are complementary. A reasoning model scales computation by generating more chain-of-thought tokens. A looped transformer scales it by re-running layers on a hidden state. A model can do both — and per the reporting, Astra does.

Does looping make a model smaller or cheaper?
Smaller in weights, not necessarily cheaper in FLOPs. You store one copy of the shared block instead of many, which cuts memory and bandwidth. But running the block 32 times costs roughly 32 blocks' worth of compute. The saving comes from not generating hundreds of chain-of-thought tokens to get the same serial depth.

Can I try a looped model today?
Yes. ByteDance's Ouro models (1.4B and 2.6B) are on Hugging Face under Apache 2.0, and the Huginn-0125 checkpoint from the recurrent-depth paper is public. Both are small research models, not frontier systems.

Does recurrent depth mean OpenAI's chain of thought disappears?
Not according to OpenAI. Pachocki says total computation-graph depth stays within 2× of GPT-4 and readable CoT is preserved, and the Path to Astra post describes CoT monitoring as an active deployment safeguard. The disagreement is about whether that bound is tight enough to matter, not whether text output vanishes.

Why does the loop count stop helping after a while?
Because most benchmark tasks are not depth-limited. In the Huginn results, gains from r = 16 to r = 32 are under half a point on several benchmarks while GSM8K keeps improving. Extra iterations only help when serial reasoning is the actual bottleneck.

Sources

- Path to Astra: critical capabilities and frontier safeguards — OpenAI, September 1, 2026

- Responding to the next frontier of critical cyber capabilities — OpenAI, August 7, 2026

- Jakub Pachocki on computation-graph depth in Astra — OpenAI Chief Scientist, September 2, 2026

- Source: OpenAI's Astra model uses "recurrent depth" — Techmeme aggregation of The Information's report, September 1, 2026

- Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach — Geiping, McLeish, Jain, Kirchenbauer, Singh, Bartoldson, Kailkhura, Bhatele, Goldstein (arXiv:2502.05171)

- Chain of Thought Monitorability: A New and Fragile Opportunity for AI Safety — Korbak, Balesni, Bengio, Irving, Pachocki et al. (arXiv:2507.11473)

- Scaling Latent Reasoning via Looped Language Models — Zhu et al., the Ouro / LoopLM paper (arXiv:2510.25741)

- Mixture-of-Recursions: Learning Dynamic Recursive Depths for Adaptive Token-Level Computation — Bae et al. (arXiv:2507.10524)

- Training Large Language Models to Reason in a Continuous Latent Space — Hao, Sukhbaatar, Su, Li, Hu, Weston, Tian (Coconut, arXiv:2412.06769)

- Steven Adler on the industry redline — former OpenAI safety researcher, September 2, 2026

- Geoffrey Irving on bounding circuit depth — UK AI Security Institute, September 2, 2026

- Red Alert: OpenAI is poised to cross an AI safety redline — Gary Marcus, September 2, 2026

- OpenAI calls Astra its most dangerous model yet — The Decoder, September 2, 2026

On this page
What Is a Looped Transformer?How Recurrent Depth Actually WorksWhat Looping Actually Buys YouThe Research LineageWhy a Frontier Lab Wants ThisThe Monitorability ProblemConfirmed, Reported, or SpeculationWhat Looped Transformers Mean for AI Slide GenerationFrequently Asked QuestionsSources

Continue Reading
All Insights

Guides·October 20, 2026EN
How to Make a Job Portfolio Presentation with AI: Interview-Ready Slides from Your Real Work
Build a job portfolio presentation with AI: collect verified evidence, map it to the job description, use an 8-slide structure, protect confidential work, and rehearse for the interview.
Tosea Team
14 MIN READ

Guides·October 17, 2026EN
How to Build a Business Dashboard PowerPoint with AI: Finance, Operations, and Sales Examples
Build a business dashboard PowerPoint with AI: KPI selection, comparison basis, chart choice, three worked finance, operations, and sales dashboards, reusable prompts, and a QC checklist.
Tosea Team
15 MIN READ

Guides·October 15, 2026EN
AI Risk Disclosure: What UK Annual Reports Reveal and How to Present the Findings
What the AI Risk Observatory found in 9,821 UK annual reports, why only 4.3% disclose AI risk substantively, and four slide examples with reusable AI prompts.
Tosea Team
12 MIN READ

Tosea.aiTransform documents into beautiful slides — fast, accurate, and consistent.

Resources
- Templates
- The Prompt Gazette
- Slide Skill Directory
- Blog
- Benchmarks

Tools
- PDF to PPT
- Merge PDF
- PowerPoint to PDF
- Compress PowerPoint
- PDF to Word
- All free tools

Product
- Features
- Showcase
- Pricing
- Docs
- What's New

Company
- Contact
- Support
- Privacy Policy
- Terms of Service
- Security

© 2025 Tosea.ai. All rights reserved.
Back to topXDiscord

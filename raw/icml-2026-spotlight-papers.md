# ICML 2026 Spotlight 论文（538 篇）

来源：https://openreview.net/group?id=ICML.cc/2026/Conference#tab-accept-spotlight
抓取日期：2026-07-12

## 1. Pressure Reveals Character: Behavioural Alignment Evaluation at Depth

**Abstract:** Evaluating alignment in language models requires testing how they behave under realistic pressure, not just what they claim they would do. While alignment failures increasingly cause real-world harm, comprehensive evaluation frameworks with realistic multi-turn scenarios remain lacking. We introduce an alignment benchmark spanning 904 scenarios across six categories---Honesty, Safety, Non-Manipulation, Robustness, Corrigibility, and Scheming---validated as realistic by human raters. Our scenarios place models under conflicting instructions, simulated tool access, and multi-turn escalation to reveal behavioral tendencies that single-turn evaluations miss. Evaluating 24 frontier models using LLM judges validated against human annotations, we find that even top-performing models exhibit gaps in specific categories, while the majority of models show consistent weaknesses across the board. Factor analysis reveals that alignment behaves as a unified construct (analogous to the g-factor in cognitive research) with models scoring high on one category tending to score high on others. We publicly release the benchmark and an interactive leaderboard to support ongoing evaluation, with plans to expand scenarios in areas where we observe persistent weaknesses and to add new models as they are released.

Forum: https://openreview.net/forum?id=kFpKyAetr3

---

## 2. On Efficient Scaling of GNNs via IO-Aware Layers Implementations

**Abstract:** Graph Neural Networks (GNNs) are bottlenecked by sparse, irregular memory access. Popular frameworks such as DGL and PyTorch Geometric support general message passing, but complex layers often materialize edge-wise intermediates, increasing memory traffic and limiting scalability on large graphs. We take an I/O- and arithmetic-intensity--centric view and show that widely used layers fall into three kernel families: SpMM-based convolutions, reduction-based aggregations, and attention-based layers (GATv2/Graph Transformer). For each family, we develop GPU kernels that reduce data movement, improve locality, and remain robust across realistic graphs. We also study graph reordering and find that its impact depends on the kernel mapping: it benefits neighbor-parallel (gather-dominated) kernels more consistently than feature-parallel designs. Empirically, our fused attention kernels reach up to 3.9x speedup for Graph Transformer (median 1.6x), with Tensor Core (block-sparse) variants up to 7.3x on locally dense graphs; for GATv2 we reach up to 8.5x speedup (median 2.0x) while reducing peak memory by up to 76x (median 6x). Our degree-aware reduction kernels achieve up to 10x speedup (median 2.6x). For SpMM-based layers, properly cached cuSPARSE achieves up to 8 speedup over DGL and outperforms evaluated custom baselines in the majority of evaluations. We release our implementations as drop-in replacements to support reproducible, hardware-aware GNN acceleration.

Forum: https://openreview.net/forum?id=EimORkkGVk

---

## 3. Foundations of Equivariant Deep Learning: Unifying Graph and Sheaf Neural Networks

**Abstract:** Symmetry is everywhere in nature and society. Geometric deep learning exploits symmetries in data to improve the performance and efficiency of deep learning systems. In this paper, we extend geometric deep learning to utilize richer symmetry structures. Specifically, we develop order-equivariant neural networks (OENN), which generalize standard graph message passing and sheaf neural networks via the theory of equivariant bundles over face posets (face categories). We (i) characterize all linear order-equivariant maps, (ii) build OENN layers, and (iii) prove universal approximation theorems (UATs) for continuous order-equivariant maps, which are new results even when restricted to sheaf neural networks (for which no UAT was known before). We illustrate the framework on graph and sheaf models. Our results can also be seen as extending the known UAT for graph neural networks to a more general setting that subsumes sheaf neural networks as well. In addition, we show that OENN can be extended further to CENN, Category-Equivariant Neural Network, which gives the general form of equivariant neural networks as well as of equivariant universal approximation theorems, allowing us to leverage categorical symmetry in data (e.g., non-invertible symmetries on multiple objects with compositional relations on those symmetries).

Forum: https://openreview.net/forum?id=aIH1jyU37z

---

## 4. Learning with Admissibility: Robust Fuzzy Hashing for Cross-Modal Retrieval with Noisy Labels

**Abstract:** Recently, cross-modal hashing (CMH) has garnered significant attention due to its low storage costs and high retrieval efficiency. Most existing CMH methods implicitly assume the availability of high-quality annotations, which is often violated in real-world scenarios as label noise inevitably arises from human errors or non-expert annotations. To cope with noisy supervision, current noise-robust CMH methods mainly follow two paradigms, i.e., noise separation and label smoothing. They often discard the predicted noisy instances or smooth discriminative signals to mitigate the impact of noisy labels. However, aggressive separation leads to reduced data utilization, while smoothing weakens the discriminative capability regarding the true distribution of clean instances. To address these limitations, we propose a novel Robust Fuzzy Cross-modal Hashing framework (RFCMH) that introduces fuzzy set theory to endow the labels with admissibility, thereby obtaining reliable discriminative supervision from noisy labels. Specifically, we first leverage possibility and necessity measures to model the noisy labels. Subsequently, we propose Fuzzy Admissibility Refinement (FAR) to dynamically calibrate supervision signals, thereby preventing the model from being misled by false positives. Furthermore, we introduce Dual-Granularity Structural Alignment (DGSA) to enforce both cross-modal alignment and instance-level uniformity, ensuring stable and diverse representations. Extensive experiments on multiple benchmarks demonstrate that RFCMH achieves state-of-the-art retrieval performance. Code is available at https://github.com/XinchengSun/RFCMH.

Forum: https://openreview.net/forum?id=reYe33OKVp

---

## 5. Information-Theoretic Disentangled Latent Modeling with Conditional Diffusion for Incomplete Multi-View Clustering

**Abstract:** Incomplete multi-view clustering is challenging due to view missingness and the entanglement of shared semantics with view-specific factors in latent representations. Existing methods often rely on heuristic fusion or direct completion strategies, which suffer from error propagation and unreliable generation under missing views. In this paper, we propose an **I**nformation-guided **D**isentangled latent modeling framework with **C**onditional **D**iffusion for incomplete multi-view clustering (IDCD). Specifically, we first encode each view into a latent representation that is variationally decomposed into a view-wise semantic latent and a view-specific factor. Information-theoretic objectives are introduced to guide the disentanglement of view-wise latents, preserving essential multi-view information while reducing the dependency between semantic and view-specific factors and encouraging cross-view semantic consistency. Besides, we aggregate the semantic latents via a mixture of Wasserstein distributions to obtain a unified global representation, where we impose a Gaussian mixture prior to explicitly couple representation learning with clustering. Based on the learned disentangled latent space, a conditional diffusion model guided by both the global semantic latent and view-specific factors is employed to generate missing views in a consistent manner. Extensive experiments on benchmark datasets demonstrate superior clustering performance and robust missing-view generation compared to state-of-the-art methods.

Forum: https://openreview.net/forum?id=Wm3XgP6xQ8

---

## 6. Learning to Theorize the World from Observation

**Abstract:** What does it mean to understand the world? Is it simply to predict future video frames? Developmental cognitive science suggests that understanding the world is fundamentally the process of constructing internal theories of how it works rather than mere prediction, even before language is acquired. However, in machine learning, it remains unclear how to endow AI systems with such theory-building capability from raw, non-textual observation alone. In this paper, we introduce Learning-to-Theorize (L2T), a learning paradigm in which an AI system acquires the ability to construct theories represented as executable programs directly from observation alone. To instantiate this paradigm, we propose the Neural Language-of-Thought Programmer, a neural model that induces and executes latent programs as explanations rather than task-specific predictors or policies. In experiments, we show that this formulation enables explanation-driven generalization, allowing observations to be understood in terms of the programs that generate them.

Forum: https://openreview.net/forum?id=wsA8LgHU5U

---

## 7. Second-Order Smooth Planning with Optimal-Transport Bellman Smoothing

**Abstract:** Planning with a generative model aims to estimate the value of a state using as few simulator calls as possible.
SmoothCruiser achieves problem-independent complexity $\widetilde O(\varepsilon^{-4})$ by exploiting the smoothness of the entropy-regularized Bellman backup, but its estimator is only first-order.
We show that the sample-complexity exponent of SmoothCruiser-type planners is governed by the order $\beta$ of the local Taylor remainder, giving oracle complexity $\widetilde O(\varepsilon^{-(2+2/(\beta-1))})$: the first-order case $\beta=2$ recovers SmoothCruiser, while a second-order/cubic remainder $\beta=3$ yields $\widetilde O(\varepsilon^{-3})$.
We reach this regime with an optimal-transport-smoothed Bellman backup over action distributions, which has a closed form, a policy gradient, and a Lipschitz Hessian, and whose quadratic correction admits an unbiased cross-product estimator.
The resulting SecondOrderSmoothCruiser achieves $\widetilde O(\varepsilon^{-3})$ oracle complexity for fixed OT parameters, and we relate the OT, entropy-regularized, and unregularized objectives through explicit regularization-bias bounds.

Forum: https://openreview.net/forum?id=LrNH0O3s45

---

## 8. Quantifying Frontier LLM Capabilities for Container Sandbox Escape

**Abstract:** Large language models (LLMs) increasingly act as autonomous agents, using tools to execute code, read and write files, and access networks, creating novel security risks. To mitigate these risks, agents are commonly deployed and evaluated in isolated "sandbox" environments, often implemented using Docker/OCI containers. We introduce SandboxEscapeBench, an open benchmark that safely measures an LLM's capacity to break out of these sandboxes. The benchmark is implemented as an Inspect AI Capture the Flag (CTF) evaluation utilising a nested sandbox architecture with the outer layer containing the flag and no known vulnerabilities. Following a threat model of a motivated adversarial agent with shell access inside a container, SandboxEscapeBench covers a spectrum of sandbox-escape mechanisms spanning misconfiguration, privilege allocation mistakes, kernel flaws, and runtime/orchestration weaknesses. We find that, when vulnerabilities are added, LLMs are able to identify and exploit them, showing that use of evaluation like SandboxEscapeBench is needed to ensure sandboxing continues to provide the encapsulation needed for highly-capable models.

Forum: https://openreview.net/forum?id=19AbP986bv

---

## 9. Towards Efficient LLMs Annealing with Principled Sample Selection

**Abstract:** The annealing phase is a pivotal convergence stage in LLM pre-training that ultimately determines final model quality. However, effectively selecting training data during this phase remains a key challenge. Current strategies rely on empirical heuristics, such as domain filtering or context extension, which lack a principled grounding in optimization theory. In this work, we characterize the annealing phase through the lens of the loss landscape's spectral geometry. We argue that optimal convergence requires gradient updates to satisfy heterogeneous constraints across different eigen-directions. Building on this insight, we formulate data selection as a problem of satisfying these directional constraints. To this end,  we propose **DiReCT** (**Di**rectionally-**Re**strained **C**onstrained **T**raining), a novel framework that reformulates sample selection in the annealing stage as a constrained optimization problem. By imposing explicit directional constraints on per-sample gradients based on the spectral properties of the Hessian, **DiReCT** identifies samples that align with the optimal curvature-aware descent path. Extensive experiments across various model scales demonstrate that **DiReCT** consistently achieves state-of-the-art performance. For future research, code is available at https://github.com/xuyj233/Direct.

Forum: https://openreview.net/forum?id=2UH01A9Za0

---

## 10. DroneDINO: Towards Heterogeneous Routed Mixture of Experts for Drone-based Unified Object Detection

**Abstract:** Recently, the rapid development of low-altitude aerial applications has driven the need for drone-based unified detectors. In contrast to task-specific detectors that suffer from poor scalability across diverse scenarios, existing unified detectors leverage the Mixture-of-Experts (MoE) architecture to learn task-aware features from diverse datasets. However, the imbalanced multi-task data distribution leads to over-activation of experts for dominant tasks and under-activation for others. To enable balanced feature learning, this paper combines three detection paradigms (RGB, IR, and RGB-IR) into a unified framework termed DroneDINO. DroneDINO extends DINO by introducing heterogeneous routed MoEs that organize experts into three functional groups: shared, task-specific, and dynamic. Unlike conventional dynamic experts where the top-$k$ experts are activated for each input, the shared expert is activated for all inputs, while each task-specific expert is activated exclusively for the matching task. To ensure inputs are routed to appropriate experts and yield task-discriminative features, we propose a task-recognition auxiliary training strategy to penalize features with low task-discriminability. Experiments demonstrate the effectiveness and generalizability of DroneDINO, which consistently outperforms state-of-the-art unified and task-specific detectors across multiple drone-based detection benchmarks.

Forum: https://openreview.net/forum?id=vPGmYeL1rG

---

## 11. Hedging on the Frontier: Learning New Tasks with Few Samples

**Abstract:** When a learner faces a new task with few samples, it must leverage any available side information. In practice, this often comes in the form of model evaluations on related tasks in public benchmarks. A key question then is how to model task relatedness such that it is both realistic and the benchmark evaluations lead to provable gains. Empirically, we observe that *weak monotonicity* is often approximately satisfied: if a model dominates another on many benchmarks, it also tends to outperform on the new task. We explore the statistical complexity of learning under (approximate) weak monotonicity, leveraging it within two learning paradigms: transfer learning and model selection aggregation. We show that not only can we prune the model class based on monotonicity, but we can also further adapt to the geometry of the available trade-offs by *hedging on the frontier*.

Forum: https://openreview.net/forum?id=J4wRLmh29t

---

## 12. OC-space: a Unifying Perspective on Verification of Tree Ensembles

**Abstract:** We study the problem of verifying whether certain properties such as robustness or fairness hold in an ensemble of decision trees.  
This problem is known to be NP-hard, with most research targeting a solution to a specific verification task.  We explore the problem through the lens of an ensemble's OC-space: the set of all possible combinations of the individual trees' predictions. This provides a unifying view that yields a more generic and flexible approach to verification.
We show that a wide variety of existing verification tasks can be (1) framed as simple searches through OC-space, and 
(2) answered in time linear or quadratic in the size of the OC-space.
Moreover, the search can be made more efficient by using spatial index structures. Interestingly, while the OC-space can grow exponentially with the ensemble's size, in practice it is often feasible to enumerate all output configurations. Empirically, we show that our generic approach can be faster than approaches targeting a single verification task.

Forum: https://openreview.net/forum?id=FLRPkR0N37

---

## 13. WBMM: Windowed Batch Matrix Multiplication for Efficient Large Receptive Field Convolution

**Abstract:** Large kernel depthwise convolutions achieve strong performance but suffer from significant degradation as kernel size grows due to irregular memory access from gather-based computation; while Large Kernel Acceleration (LKA) helps on small feature maps, it becomes counterproductive on large feature maps, even slower than non-accelerated implementations. We propose Windowed Batch Matrix Multiplication (WBMM), which partitions input into contiguous windows and indexes a compact relative position bias table to construct weight matrices, enabling regular memory access via batched matrix multiplication. This yields a unique property: WBMM's throughput improves with larger windows, opposite to depthwise convolutions that degrade with larger kernels. Operator-level benchmarks show WBMM with $14 \times 14$ windows outperforms $5 \times 5$ depthwise convolution baselines in speed while providing a $7.8\times$ larger per-layer receptive field. Combined with inter-block cross-window communication and hierarchical window reparameterization, WBMM achieves comparable or higher accuracy on ImageNet-1K, COCO, and ADE20K with $1.31$--$1.88\times$ training speedup, and demonstrates consistent advantages across GPU, CPU, and edge devices without requiring specialized acceleration kernels. Our code is available at https://github.com/wansong-s/WBMM.

Forum: https://openreview.net/forum?id=Qg9Jcy788i

---

## 14. CVE-Factory: Scaling Expert-Level Agentic Tasks for Code Security Vulnerability

**Abstract:** Evaluating and improving the security capabilities of code agents requires high-quality, executable vulnerability tasks. However, existing works rely on costly, unscalable manual reproduction and suffer from outdated data distributions. To address these, we present CVE-Factory, the first multi-agent framework to achieve expert-level quality in automatically transforming sparse CVE metadata into fully executable agentic tasks. Cross-validation against human expert reproductions shows that CVE-Factory achieves 95\% solution correctness and 96\% environment fidelity, confirming its expert-level quality. It is also evaluated on the latest realistic vulnerabilities and achieves a 66.2\% verified success. This automation enables two downstream contributions. First, we construct LiveCVEBench, a continuously updated benchmark of 190 tasks spanning 14 languages and 153 repositories that captures emerging threats including AI-tooling vulnerabilities. Second, we synthesize over 1,000 executable training environments, the first large-scale scaling of agentic tasks in code security. 
Fine-tuned Qwen3-32B improves from 5.3\% to 35.8\% on LiveCVEBench, surpassing Claude 4.5 Sonnet, with gains generalizing to Terminal Bench (12.5\% to 31.3\%).
We open-source all code, data, and models.

Forum: https://openreview.net/forum?id=BkMSFFtm5M

---

## 15. HOBIT: Hardness Optimized Batch Sampling for InfoNCE Training

**Abstract:** Contrastive training with  InfoNCE loss and in-batch negatives is the standard approach for learning dual-encoder models. Its effectiveness, however, critically depends on the availability of hard negatives; in their absence, learning quickly saturates. Existing methods address this via explicit hard-negative mining, which is often costly or heuristic-driven. We introduce $\mathrm{\texttt{HOBIT}}$, a principled mini-batch construction method that improves in-batch negative quality by reordering training examples at every epoch. $\mathrm{\texttt{HOBIT}}$ solves an optimization problem motivated by the InfoNCE objective to yield mini-batches such that each query in the batch is exposed to hard yet non-contradictory, informative negative examples. We show that the optimization objective is monotone and submodular which in turn leads us to a greedy algorithm that admits the standard $\mathcal{O}(1 - 1/e)$ approximation guarantee. Empirically, we show that $\mathrm{\texttt{HOBIT}}$ incurs negligible computational overhead while significantly outperforming state-of-the-art batching methods, and remains complementary to existing hard negative mining techniques.

Forum: https://openreview.net/forum?id=R49XZi14YH

---

## 16. Efficient Diffusion Models under Nonconvex Equality and Inequality constraints via Landing

**Abstract:** Generative modeling within constrained sets is essential for scientific and engineering applications involving physical, geometric, or safety requirements (e.g., molecular generation, robotics). We present a unified framework for constrained diffusion models on generic nonconvex feasible sets $\Sigma$ that simultaneously enforces equality and inequality constraints throughout the diffusion process. Our framework incorporates both overdamped and underdamped dynamics for forward and backward sampling. A key algorithmic innovation is a computationally efficient landing mechanism that replaces costly and often ill-defined projections onto $\Sigma$, ensuring feasibility without iterative Newton solves or projection failures. By leveraging underdamped dynamics, we accelerate mixing toward the prior distribution, effectively alleviating the high simulation costs typically associated with constrained diffusion. Empirically, this approach reduces function evaluations and memory usage during both training and inference while preserving sample quality. On benchmarks featuring equality and mixed constraints, our method achieves comparable sample quality to state-of-the-art baselines while significantly reducing computational cost, providing a practical and scalable solution for diffusion on nonconvex feasible sets.

Forum: https://openreview.net/forum?id=6VEqdKmfKh

---

## 17. FlatLand: Personalized Graph Federated Learning via Tailored Lorentz Space

**Abstract:** Personalization has become a pivotal field of study in contemporary intelligent systems. 
Federated learning enables privacy-preserving collaborative training, but highly heterogeneous client data remain challenging, especially in graph federated learning where clients possess structurally diverse graphs. Existing personalized federated learning (PFL) methods ignore the intrinsic geometric properties of diverse graph structures. We propose FlatLand, a novel personalized Federated learning method that embeds different clients' data in tailored Lorentz space of hyperbolic geometry. Our key insight is that hyperbolic geometry naturally accommodates the intrinsic negative curvature prevalent in real-world graphs, while the time-like dimension in Lorentz space provides a principled way to encode client-specific heterogeneity. We develop a parameter decoupling strategy that separates heterogeneous information (captured in time-like parameters) from common knowledge (preserved in space-like parameters), enabling direct aggregation without requiring client similarity estimation and extra calculation modules. Empirical results on diverse federated graph learning tasks demonstrate that FlatLand achieves superior performance, particularly in low-dimensional settings. Code is available in our GitHub repository.

Forum: https://openreview.net/forum?id=H6PagZKyil

---

## 18. Lookahead Sample Reward Guidance for Test-Time Scaling of Diffusion Models

**Abstract:** Diffusion models have demonstrated strong generative performance; however, generated samples often fail to fully align with human intent. This paper studies an efficient test-time scaling method for sampling from regions with higher human-aligned reward values. Existing methods for computing the expected future reward (EFR) face important limitations: backward rollout incurs prohibitively high sampling costs, while Tweedie-based approaches, including Sequential Monte Carlo and gradient guidance, suffer from bias and inherent sampling issues. We show that the EFR at any $\mathbf{x}_t$ can be computed using only marginal samples from a pre-trained diffusion model, enabling closed-form reward guidance without neural backpropagation. To further improve efficiency, we introduce a few-step lookahead sampling and an accurate solver that guides particles toward high-reward lookahead samples. We refer to this sampling scheme as LiDAR sampling. LiDAR achieves the same GenEval performance as the latest gradient guidance method for SDXL with a 9.5× speedup. We release the code at https://github.com/aailab-kaist/Diffusion-LiDAR-Sampling.

Forum: https://openreview.net/forum?id=IbRm6gwmew

---

## 19. Rapid Poison: Practical Poisoning Attacks Against the Rapid Response Framework

**Abstract:** The Rapid Response (RR) framework (Peng et al., 2024), deployed in production systems including Anthropic’s ASL-3 safeguards (Anthropic, 2025), dynamically adapts jailbreak detection classifiers by generating synthetic training data from emerging attacks. We reveal that prompt injection can infiltrate this pipeline to deliver poisoned samples into the classifier’s training set, enabling two attack objectives: (I) targeted poisoning attacks that create false positives on harmless samples by categorizing them as a jailbreak, with a specific desired feature (e.g., certain formatting, subject, or keyword), (II) concept-based backdoor attacks that induce false negatives on jailbreak inputs, generalizing even to jailbreaks from attack strategies the defender explicitly trained against, when the backdoor trigger is present. Importantly, our threat model restricts adversaries to modify- ing only jailbreak samples (not benign data or labels), a constraint unexplored by prior work that makes the second objective particularly challeng- ing. We address this with Omission Attack, which exploits a new phenomenon: when training on concept-absent unsafe samples, the classifier mis- associates that concept’s presence with the safe label. Both attacks flip nearly all target labels with only 1% poisoning rate. Code: https://github.com/DH-davidhuang/rapid-poison

Forum: https://openreview.net/forum?id=hvI3Syn2U7

---

## 20. FIDIA: Function-Informed Sequence Design via Inference-Aligned Policy Optimization

**Abstract:** Computational protein design typically employs a sequential workflow of structure generation followed by sequence (re)design. While structure generators can be explicitly conditioned on functional objectives, inverse folding models are constrained by their function-agnostic nature and sequence-structure degeneracy. More critically, the associated training objectives do not account for the *Best-of-N* (BoN) inference protocol, resulting in a fundamental training-inference misalignment. Here, we propose FIDIA, a reinforcement learning framework that enables **F**unction-**I**nformed sequence **D**esign via **I**nference-**A**ligned policy optimization. Specifically, FIDIA integrates functional constraints into composite rewards and explicitly optimize the induced policy under BoN toward high-fitness sequence regions. We achieve this via a grounded gradient estimator that directly maximizes the expected maximum reward. FIDIA consistently outperforms both standard and RL-optimized baselines in success rate and precision on a general motif scaffolding benchmark. Further experiments on realworld cases including vaccine and affinity-enhancing enzyme design validate FIDIA’s efficacy in complex therapeutic and biocatalytic contexts.

Forum: https://openreview.net/forum?id=pvbJsa0ia0

---

## 21. HDFlow: Hierarchical Diffusion-Flow Planning for Long-horizon Tasks

**Abstract:** Recent advances in generative models have shown promise in generating behavior plans for long-horizon, sparse reward tasks. While these approaches have achieved promising results, they often lack a principled framework for hierarchical decomposition and struggle with the computational demands of real-time execution, due to their iterative denoising process. In this work, we introduce $\textbf{Hierarchical Diffusion-Flow}$ ($\texttt{\textbf{HDFlow}}$), a novel hierarchical planning framework that optimally leverages the strengths of $\textit{diffusion}$ and $\textit{rectified flow}$ models to overcome the limitations of single-paradigm generative planners. $\texttt{\textbf{HDFlow}}$ employs a high-level diffusion planner to generate sequences of strategic subgoals in a learned latent space, capitalizing on diffusion's powerful exploratory capabilities. These subgoals then guide a low-level rectified flow planner that generates smooth and dense trajectories, exploiting the speed and efficiency of ordinary differential equation (ODE)-based trajectory generation. We evaluate $\texttt{\textbf{HDFlow}}$ on four challenging furniture assembly tasks in both simulation and real-world, where it significantly outperforms state-of-the-art methods. Furthermore, we also showcase our method's generalizability on two long-horizon benchmarks comprising diverse locomotion and manipulation tasks. Project website: https://hdflow-page.github.io/

Forum: https://openreview.net/forum?id=3ZFvXHJwmB

---

## 22. Eigenvectors of Experts are Training-free Non-collapsing Routers

**Abstract:** Sparse Mixture of Experts (SMoE) architectures improve the training efficiency of Large Language Models (LLMs) by routing input tokens to a selected subset of specialized experts. Despite their remarkable success, both training and inference in SMoE models suffer from the *expert collapse* issue (Chi et al., 2022a), which degrades model performance. Prior studies primarily focus on improving the router; however, such methods rely on training from scratch or fine-tuning, which requires high computational and data-processing costs. Furthermore, we demonstrate that, despite these efforts, the issue persists when advancing well-pretrained SMoE models, as evidenced by both theoretical and empirical results. To fill that gap, we analyze the advanced SMoE models and observe that the eigenvectors of expert weight matrices encode rich semantic information, pointing to an effective alternative to conventional routing strategies. Building on this insight, we propose **Singular Value Decomposition SMoE (SSMoE)**, a novel and *training-free* framework that leverages spectral properties of the expert weights to address the collapse issue and enhance model performance. Extensive experiments across diverse language and vision tasks, under both clean and corrupt data settings, demonstrate the strong generalization and robustness of SSMoE. Our findings highlight how a deeper understanding of model internals can guide the development of more effective SMoE architectures.

Forum: https://openreview.net/forum?id=H1mxs9qGQX

---

## 23. WaterSIC: Information-Theoretically (Near) Optimal \\Linear Layer Quantization

**Abstract:** This paper considers the problem of converting a given dense linear layer to low precision. The tradeoff between compressed length and output discrepancy is analyzed information theoretically (IT). It is shown that a popular
    GPTQ algorithm may have an arbitrarily large gap to the IT limit. To alleviate this problem, a
 novel algorithm, termed "WaterSIC", is proposed and is shown to be within a rate gap of
    0.255 bits to the IT limit, uniformly over all possible covariance matrices of input activations.
 The key innovation of WaterSIC's is to allocate different quantization rates to different columns
    (in-features) of the weight matrix, mimicking the classical IT solution known as
    ``waterfilling''. Applying WaterSIC to the Llama and Qwen family of LLMs establishes new
    state-of-the-art performance for all quantization rates from 1 to 4 bits. Our code is available at https://github.com/egorlifar/watersic.

Forum: https://openreview.net/forum?id=fCPgAHIciE

---

## 24. Towards Sub-Second Molecular Docking as a Structural Primitive: A Quantized Consistency Diffusion Framework

**Abstract:** Agent-centered scientific discovery is turning scientific models into always-on computational infrastructure.
In this paradigm, AI agents coordinate tools, interpret feedback, and drive high-frequency research loops, requiring domain models that are both accurate and callable in real time.
Molecular docking exposes this bottleneck: it provides essential structural feedback for drug discovery, yet current high-fidelity docking and co-folding models remain limited by iterative generative refinement and heavy computation.
We present a compute-efficient co-folding framework that turns molecular docking into a sub-second structural primitive.
Because docking methods operate under different levels of structural prior, we report accuracy under information-level-matched protocols, comparing blind settings with blind generative methods and interface-informed settings with surface- or interface-informed baselines.
Our framework combines two ideas.
First, Progressive Consistency Regularization (PCR) compresses diffusion dynamics into reliable few-step inference through reconstruction-anchored consistency tuning.
Second, Residual-Safe Quantization preserves high-fidelity residual streams and geometry-sensitive operations in BF16 while quantizing selected compute-intensive linear transformations.
Our model achieves state-of-the-art docking accuracy under the matched interface-informed protocol, reports blind docking performance separately under the matched blind protocol, and generates five conformations for a representative 256-token complex in 0.17 seconds on a single NVIDIA H20 GPU, delivering a $>300\times$ speedup over AlphaFold3 under the benchmarked setting.
Together, these results move molecular docking from an offline generative simulator toward a real-time structural primitive for agent-centered drug discovery.

Forum: https://openreview.net/forum?id=NyPHOtsJfE

---

## 25. HELIX: Hybrid Encoding with Learnable Identity and Cross-dimensional Synthesis for Time Series Imputation

**Abstract:** Time series imputation benefits from leveraging cross-feature correlations, yet existing attention-based methods re-discover feature relationships at each layer, lacking persistent anchors to maintain consistent representations. To address this, we propose HELIX, which assigns each feature a learnable feature identity, a persistent embedding that captures intrinsic semantic properties throughout the network. Unlike graph-based methods that rely on predefined topology and assume homogeneous spatial relationships, HELIX learns arbitrary feature dependencies end-to-end from temporal co-variation, naturally handling datasets where features mix spatial locations with semantic variables. Integrated with hybrid temporal-feature attention, HELIX achieves the state-of-the-art performance, surpassing all 16 baselines on 5 public datasets across 21 experimental settings in our evaluation. Furthermore, our mechanistic analysis reveals that HELIX aligns learned feature identities and dependencies with latent physical and semantic structure progressively across layers, demonstrating that it more effectively translates cross-feature structure into imputation accuracy.

Forum: https://openreview.net/forum?id=FN20iuPnEU

---

## 26. Shared Semantics, Divergent Mechanisms: Unsupervised Feature Discovery by Aligning Semantics and Mechanisms

**Abstract:** As large language models are increasingly deployed in high-stakes settings, there is a growing need for tools that audit not only model outputs but also the internal computations that produce them.
Circuit analysis is a central approach in mechanistic interpretability, but it is typically target-conditioned, explaining a single prompt paired with a chosen completion.
This target-conditioned setup can obscure heterogeneity across a model's continuation distribution.
We introduce distribution-level unsupervised feature discovery, which clusters sampled continuations using both semantic content and sequence-level mechanistic attributions, without manually specifying target outputs.
Our method represents each continuation with a semantic embedding and a prefix-to-continuation attribution signature, then optimizes a rate-distortion objective that trades off semantic coherence, mechanistic consistency, and cluster granularity.
Across clustering and steering analyses, the discovered clusters expose continuation modes that single-view baselines miss and provide interventional evidence that cluster signatures correspond to actionable mechanistic factors.
Overall, our approach complements circuit analysis and behavioral evaluation by providing a scalable audit of the mechanisms underlying a model’s continuation distribution.

Forum: https://openreview.net/forum?id=C9AhjL8aUZ

---

## 27. TabSwift: An Efficient Tabular Foundation Model with Row-Wise Attention

**Abstract:** Tabular foundation models, exemplified by TabPFN, perform prediction via in-context learning, inferring test labels directly from labeled training examples. They have demonstrated competitive performance, particularly on small-to-medium datasets. However, recent tabular foundation models often improve accuracy with increasingly complex architectures, incurring higher inference cost and limiting practical deployment.
In this work, we revisit the original TabPFN design and show that a lightweight row-wise attention–only backbone can remain highly competitive with two simple enhancements: a gated attention stabilization mechanism and a small set of learnable register tokens that provide global context and improve pretraining quality.
The resulting model, TabSwift, supports both classification and regression, and is competitive with stronger tabular foundation models (e.g., TabPFN v2 and TabICL) while being more efficient at inference.
For latency-sensitive serving, we further introduce an adaptive layer-wise early-exit mechanism that dynamically adjusts inference depth per sample.
Overall, TabSwift enables efficient and anytime tabular in-context learning for practical deployments.

Forum: https://openreview.net/forum?id=ottLtChHp2

---

## 28. SmoothSpike: Spiking Transformer with Learnable Hadamard Transformation

**Abstract:** Spiking Neural Networks (SNNs) have attracted growing attention due to their sparse spike-based communication and inherent temporal dynamics. However, their discrete information representation fundamentally limits expressiveness, resulting in a notable performance gap relative to Artificial Neural Networks (ANNs) on language modeling tasks. In this paper, we reveal that this gap is fundamentally rooted in a spike saturation-induced information homogenization problem: within a bounded time window, distinct high-amplitude inputs converge to identical spike counts, compressing neural representations and impairing fine-grained semantic discrimination across layers.  To address this, we propose SmoothSpike, which applies a randomized Hadamard transformation to smooth pre-activation inputs and theoretically proves that it bounds the maximum input to $\mathcal{O}(\sqrt{\frac{\log n}{n}})$  with high probability. To further improve adaptability across varying input distributions, we extend the fixed transformation within SmoothSpike to a learnable orthogonal matrix updated via Newton-Schulz iterations, which can be fused into model weights at inference with no additional overhead. Experiments on the GLUE benchmark show that SmoothSpike effectively reduces information homogenization, yielding an 8.2\% average improvement over the Spikingformer baseline without compromising the efficiency inherent to spike-driven computation. These results advance the prospects for energy-efficient and high-performance language modeling on edge devices. Code is available at https://github.com/CayleyZ/SmoothSpike.

Forum: https://openreview.net/forum?id=UoUKCLHjRa

---

## 29. Language Model Circuits Are Sparse in the Neuron Basis

**Abstract:** The high-level concepts that a neural network uses to perform computation need not be aligned to individual neurons (Smolensky, 1986).
Language model interpretability research has thus turned to techniques which decompose the neuron basis into more interpretable units of model computation, such as sparse autoencoders (SAEs).
However, not all neuron-based representations are uninterpretable. 
For the first time, we empirically show that **MLP neurons are as sparse a feature basis as SAEs**.
We use this finding to develop an end-to-end gradient-based attribution pipeline for circuit tracing on the MLP neuron basis, which surfaces causally effective neurons on a variety of tasks.
On a standard subject-verb agreement benchmark (Marks et al., 2025), a circuit of $\approx 10^2$ MLP neurons is enough to control model behaviour. 
On the multi-hop city-state-capital task from Lindsey et al. (2025), we find a circuit in which small sets of neurons encode specific latent reasoning steps (e.g. mapping a city to its state), and can be steered to change the model's output. 
This work thus advances automated interpretability of language models without imposing additional training costs.

Forum: https://openreview.net/forum?id=OrviwFWcN4

---

## 30. Sharp Inequalities between Total Variation and Hellinger Distances for Gaussian Mixtures

**Abstract:** We study the relation between the total variation (TV) and Hellinger distances between two Gaussian location mixtures. Our first result establishes a general upper bound: for any two mixing distributions supported on a compact set, the Hellinger distance between the two mixtures is controlled by the TV distance raised to a power $1-o(1)$, where the $o(1)$ term is of order $1/\log\log(1/\mathrm{TV})$. We also construct two sequences of mixing distributions that demonstrate the sharpness of this bound. Taken together, our results resolve an open problem raised in Jia et al. (2023) and thus lead to an entropic characterization of learning Gaussian mixtures in total variation. Our inequality also yields optimal robust estimation of Gaussian mixtures in Hellinger distance, which has a direct implication for bounding the minimax regret of empirical Bayes under Huber contamination.

Forum: https://openreview.net/forum?id=ihMB4kA2SQ

---

## 31. MASPOB: Bandit-Based Prompt Optimization for Multi-Agent Systems with Graph Neural Networks

**Abstract:** Large Language Models (LLMs) have achieved substantial success in real-world applications, particularly as the cognitive backbone of Multi-Agent Systems (MAS) for orchestrating complex workflows. Since many deployments preclude workflow modifications while MAS performance is highly prompt-sensitive, prompt optimization becomes a critical strategy for improvement. However, real-world prompt optimization for MAS is impeded by three key challenges: (1) the need of sample efficiency due to prohibitive evaluation costs, (2) topology-induced coupling among prompts, and (3) the combinatorial explosion of the search space. To address these challenges, we introduce **MASPOB** (**M**ulti-**A**gent **S**ystem **P**rompt **O**ptimization via **B**andits), a novel sample-efficient framework based on bandits. By leveraging Upper Confidence Bound (UCB) to quantify uncertainty, the bandit framework balances exploration and exploitation, maximizing gains within a strictly limited budget. To handle topology-induced coupling, MASPOB integrates Graph Neural Networks (GNNs) to capture structural priors, learning topology-aware representations of prompt semantics. Furthermore, it employs coordinate ascent to decompose the optimization into univariate sub-problems, reducing search complexity from exponential to linear. Extensive experiments across diverse benchmarks demonstrate that MASPOB achieves state-of-the-art performance, consistently outperforming existing baselines. Our code is available at <https://github.com/HZ1008/MASPOB>.

Forum: https://openreview.net/forum?id=A5venxSvpw

---

## 32. Deep Flow Networks

**Abstract:** We introduce Deep Flow Networks (DFNs), a new class of discrete function approximators. DFNs are inspired by and generalize minimum-cost flow value functions that map node imbalances on a subset of nodes to the optimal flow cost. Such functions are known to be M-convex (Murota2003) and admit efficient optimization.
On the theoretical side, we prove that DFNs are universal approximators for discrete functions on $\mathbb{Z}^d$ that admit convex extensions to $\mathbb{R}^d$, and characterize their optimization complexity in terms of their deviation from the M-convex regime. Guided by these results, we develop a practical DFN implementation for learning from data. Finally, we evaluate our implementation empirically on data from different ground-truth functions, showing that DFNs achieve strong approximation accuracy while being substantially faster to optimize than benchmark approaches.

Forum: https://openreview.net/forum?id=Z7rhDaBvBo

---

## 33. Latent Spherical Flow Policy for Reinforcement Learning with Combinatorial Actions

**Abstract:** Reinforcement learning (RL) with combinatorial action spaces remains challenging because feasible action sets are exponentially large and governed by complex feasibility constraints, making direct policy parameterization impractical. Existing approaches embed task-specific value functions into constrained optimization programs or learn deterministic structured policies, sacrificing generality and policy expressiveness. We propose a solver-induced \emph{latent spherical flow policy} that brings the expressiveness of modern generative policies to combinatorial RL while guaranteeing feasibility by design. Our method, LSFlow, learns a \emph{stochastic} policy in a compact continuous latent space via spherical flow matching, and delegates feasibility to a combinatorial optimization solver that maps each latent sample to a valid structured action. To improve efficiency, we train the value network directly in the latent space, avoiding repeated solver calls during policy optimization. To address the piecewise-constant and discontinuous value landscape induced by solver-based action selection, we introduce a smoothed Bellman operator that yields stable, well-defined learning targets. Empirically, our approach outperforms state-of-the-art baselines by an average of 20.6\% across a range of challenging combinatorial RL tasks.

Forum: https://openreview.net/forum?id=07wwDFdi3k

---

## 34. L2G-NET: Local to Global Spectral Graph Neural Networks via Cauchy Factorizations

**Abstract:** Despite their theoretical advantages, spectral methods based on the graph Fourier transform (GFT) are seldom used in graph neural networks (GNNs) due to the cost of computing the eigenbasis and the lack of vertex-domain locality in the resulting representations. As a result, most GNNs rely on local approximations such as polynomial Laplacian filters or message passing, which limit their ability to model long-range dependencies. In this paper, we introduce an exact factorization of the GFT into operators acting on subgraphs, which are then combined via a sequence of Cauchy matrices. Building on this factorization, we propose a new class of spectral GNNs, termed L2G-Net (Local to Global Net). Unlike existing spectral methods, which are either fully global (when using the GFT) or local (when using polynomial filters), L2G-Net operates by processing the spectral representations of subgraphs and then combining them via structured matrices. Our algorithm avoids full eigendecompositions, exploiting graph topology to construct the factorization with quadratic complexity in the number of nodes, scaled by the maximum cut size between subgraphs. Experiments stressing long-range dependencies on large graphs show that L2G-Net scales to regimes out of reach for the standard GFT, and is competitive with state-of-the-art methods with orders of magnitude fewer learnable parameters.

Forum: https://openreview.net/forum?id=kD8iJmyn5l

---

## 35. Updating Parametric Knowledge with Context Distillation Retains Post-Training Capabilities

**Abstract:** Post-training endows pretrained LLMs with a variety of desirable skills, including instruction-following, reasoning, and others. However, these post-trained LLMs only encode knowledge up to a cut-off date, necessitating continual adaptation. Unfortunately, existing solutions cannot simultaneously learn new knowledge from an adaptation document corpora and mitigate the forgetting of earlier learned capabilities.
To address this, we introduce Distillation via Split Contexts (DiSC), a simple context-distillation based approach for continual knowledge adaptation. DiSC derives student and teacher distributions by conditioning on distinct segments of the training example and minimizes the KL divergence between the shared tokens. This allows us to efficiently apply context-distillation without requiring explicit generation steps during training. 
We run experiments on four post-trained models and two adaptation domains. Compared to prior finetuning and distillation methods for continual adaptation, DiSC consistently reports the best trade-off between learning new knowledge and mitigating forgetting of previously learned skills like instruction-following, reasoning, and factual knowledge.

Forum: https://openreview.net/forum?id=OJsGhlTayF

---

## 36. Seizure-Semiology-Suite($S^3$): A Clinically Multimodal Dataset, Benchmark, and Models for Seizure Semiology Understanding

**Abstract:** While Multimodal Large Language Models (MLLMs) have demonstrated remarkable proficiency in general video understanding, their capacity to interpret involuntary, and spatio-temporally evolving pathologic motor behaviors such as seizure semiology remains largely untested. To address this gap, we introduce Seizure-Semiology-Suite (S³), a clinically grounded dataset and benchmark for fine-grained, structured seizure semiology understanding. The dataset includes 438 seizure videos annotated with over 35,000 dense labels covering 20 ILAE-defined semiological features. Building on this dataset, we propose a seven-task hierarchical benchmark that systematically evaluates MLLMs from low-level visual perception to temporal sequencing, narrative report generation, and seizure diagnosis. To enable clinically meaningful evaluation of generated reports, we further introduce the Report Quality Index for Seizure Semiology (Seizure-RQI). Extensive baselines across 11 open-weight MLLMs reveal systematic weaknesses in laterality reasoning, temporal localization, symptom sequencing, and clinically faithful reporting. We show that seizure-specific fine-tuning substantially improves performance across tasks, and that a two-stage neuro-symbolic framework achieves an F1 score of 0.96 on epileptic versus non-epileptic seizure classification. Seizure-Semiology-Suite establishes a rigorous benchmark for evaluating multimodal models in safety-critical medical video understanding and guides the development of clinically reliable, domain-adaptive multimodal intelligence. Our code is publicly available at \href{https://github.com/LinaZhangUCLA/SeizureSemiologySuite}{SeizureSemiologySuite}.

Forum: https://openreview.net/forum?id=MyorUlHKVc

---

## 37. Learning Randomized Reductions

**Abstract:** Randomized self-reductions (RSRs) express $f(x)$ using $f$ evaluated at random correlated points, enabling self-correcting programs, instance-hiding protocols, and applications in complexity theory and cryptography. Yet discovering RSRs has required manual expert derivation for over 40 years, limiting their practical use.
We present Bitween for automated RSR learning. First, we formalize RSR learning with sample complexity analysis under correlated sampling. Second, we develop Vanilla Bitween, which integrates multiple backends (linear regression, genetic programming, symbolic regression, and mixed-integer programming). The linear regression backend outperforms the others, discovering RSRs for 43 of 80 functions (54%) in RSR-Bench, our benchmark suite, including the first known reduction for sigmoid. Third, we introduce Agentic Bitween, a neuro-symbolic approach where LLM agents propose novel query functions beyond the fixed set ($x+r$, $x-r$, $x \cdot r$, $x$, $r$) in prior work. Agentic Bitween discovers RSRs for 64 of 80 functions (80%), outperforming pure neural baselines in both RSR discovery and verification accuracy.

Forum: https://openreview.net/forum?id=hCAEcqig2C

---

## 38. h1: Bootstrapping LLMs to Reason over Longer Horizons via Reinforcement Learning

**Abstract:** Large language models excel at short-horizon reasoning tasks, but performance drops as reasoning horizon lengths increase. In this work, we introduce a scalable method to bootstrap long-horizon reasoning capabilities using only existing, abundant short-horizon data. Our approach synthetically composes simple problems into complex, multi-step dependency chains of arbitrary length. We train models on this data using outcome-only rewards under a curriculum that automatically increases in complexity, allowing RL training to be scaled much further without saturating. Empirically, our method generalizes remarkably well: curriculum training on composed 6th-grade level math problems improves accuracy on longer, competition-level benchmarks. It also transfers significantly to diverse out-of-distribution ReasoningGym domains and long-context benchmarks, indicating broader generalization. Importantly, our long-horizon improvements are significantly higher than baselines even at high pass@k, showing that models can learn new reasoning paths under RL. Theoretically, we show that curriculum RL with outcome rewards could achieve an exponential improvement in sample complexity over full-horizon training, providing training signal comparable to dense supervision. h1 therefore introduces an efficient path towards scaling RL for long-horizon problems using only existing data.

Forum: https://openreview.net/forum?id=3BW15kSPfN

---

## 39. Equilibrium Pricing in Oligopolistic Data Markets

**Abstract:** We study equilibrium pricing in oligopolistic data markets with budget-constrained buyers (e.g., ML companies purchasing data to improve model accuracy) and strategic data sellers. Sellers compete by setting prices for their datasets, giving rise to a pricing game whose pure Nash equilibria correspond to equilibrium prices. While equilibrium prices are guaranteed for rivalrous goods via competitive equilibrium, we show that the non-rivalry of data fundamentally alters this picture: an exact Nash equilibrium need not exist, and in fact no 1.364-approximate equilibrium exists under uniform pricing. We therefore investigate relaxed equilibrium notions. Allowing sellers to use beyond-uniform pricing—specifically, piecewise-linear convex pricing functions—guarantees approximate stability within a constant factor: there exists a pricing profile in which no seller can improve revenue by a factor of two by deviating to any uniform price (a 2-approximate Nash equilibrium). Finally, our simulations demonstrate fast convergence and empirical approximation guarantees that outperform the worst-case bound of 2.

Forum: https://openreview.net/forum?id=TAqejOfEAJ

---

## 40. Is Data Shapley Not Better than Random in Data Selection? Ask NASH

**Abstract:** Data selection studies the problem of identifying high-quality subsets of training data. While some existing works have considered selecting the subset of data with top-$m$ Data Shapley or other semivalues as they account for the interaction among every subset of data, other works argue that Data Shapley can sometimes perform ineffectively in practice and select subsets that are *no better than random*. This raises the questions: **(I)** *Are there certain "Shapley-informative" settings where Data Shapley consistently works well?* **(II)** *Can we strategically utilize these settings to select high-quality subsets consistently and efficiently?*
In this paper, we propose a novel data selection framework, **NASH** (Non-linear Aggregation of SHapley-informative components), which **(I)** decomposes the target utility function (e.g., validation accuracy) into simpler, Shapley-informative component functions, and selects data by optimizing an objective that **(II)** aggregates these components non-linearly. We demonstrate that NASH substantially boosts the effectiveness of Shapley/semivalue-based data selection with minimal additional runtime cost.

Forum: https://openreview.net/forum?id=vMsrm8UGGC

---

## 41. STAR-KV: Low-Rank KV Cache Compression via Soft Thresholding for Adaptive Rank Control

**Abstract:** Low-rank projection has emerged as a promising approach for compressing the KV cache by exploiting hidden-dimension redundancy. However, prior methods rely on fixed or heuristic rank selection and struggle to achieve aggressive compression with minimal accuracy degradation. We propose STAR-KV, an adaptive low-rank KV cache compression framework with fine-grained rank control. STAR-KV encompasses 1) a differentiable thresholding mechanism that enables optimal rank selection at both attention-head and block levels, 2) a hybrid decomposition strategy that applies different low-rank factorizations according to the sensitivity of key and value projections, and 3) a low-rank--aware mixed precision quantization  that leverages data statistics for near lossless low-bit quantization. Evaluated across multiple LLMs and benchmarks, STAR-KV achieves up to 75\%  KV cache compression and up to 20$\times$ overall KV cache reduction when combined with quantization. Enabled by custom Triton-based GPU kernels, STAR-KV delivers up to 6.9$\times$ speedup for the attention module and 3.1$\times$ end-to-end generation throughput. Our code is publicly available at: https://github.com/PriyanshBhatnagar/STAR-KV.

Forum: https://openreview.net/forum?id=lJjH1q6RwY

---

## 42. Monitoring Monitorability

**Abstract:** Safe deployment of increasingly capable AI agents may require visibility into how they make decisions. Chain-of-thought (CoT) monitoring can detect misbehavior in today’s reasoning models, but this “monitorability” may be fragile under different training procedures, data sources, or continued system scaling. We propose three evaluation archetypes (intervention, process, and outcome-property), a new monitorability metric, and a broad evaluation suite. We show CoT monitoring outperforms action-only monitoring in practical settings, and that frontier models are generally—but not perfectly—monitorable. We study scaling trends with pre-training model size and inference-time compute, finding longer CoTs are typically more monitorable. We find that, for a fixed capability level, using a smaller model at higher reasoning effort can yield higher monitorability, at greater inference compute cost. We further find that increasing a weak monitor’s test-time compute when monitoring a strong agent improves monitorability, and giving the monitor access to the CoT both boosts monitorability and steepens the compute–to-monitorability scaling trend. Finally, we show monitorability can be improved by asking follow-up questions and giving the follow-up CoT to the monitor.

Forum: https://openreview.net/forum?id=b82fgbMVpz

---

## 43. Provable Accuracy Collapse of Embedding-Based Representations under Dimensionality Mismatch

**Abstract:** Embedding-based representations in Euclidean space $\mathbb{R}^d$ are a cornerstone of modern machine learning, where a major goal is to use the \emph{smallest dimension} that faithfully captures data relations. In this work, we prove sharp dimension--accuracy tradeoffs and identify a fundamental information-theoretic limitation: unless the embedding dimension $d$ is chosen close to the ground-truth dimension $D$,  accuracy undergoes a sudden collapse. Our main result shows that this phenomenon arises even in standard contrastive learning settings, where supervision is limited to a set of $m$ anchor--positive--negative triplets $(i,j,k)$ encoding distance comparisons $\mathrm{dist}(i,j) < \mathrm{dist}(i,k)$. Specifically, given triplets realizable by an unknown ground-truth embedding in $D$ dimensions, we prove that there exists constant $c < 1$, such that \emph{every embedding of dimension at most $cD$  violates almost half of the triplets}, yielding accuracy as low as a trivial one-dimensional solution that ignores the input. We complement our information-theoretic bounds with strong computational hardness results: under the Unique Games Conjecture, even if the given triplets are nearly realizable in $D=1$ dimension, no polynomial-time algorithm---\textit{regardless of its dimension}---can achieve accuracy above the trivial 50% baseline.

Forum: https://openreview.net/forum?id=ubzXPEfa4R

---

## 44. Breaking the Reversal Curse in Autoregressive Language Models via Identity Bridge

**Abstract:** Autoregressive large language models (LLMs) have achieved remarkable success in many complex tasks, yet they can still fail in very simple logical reasoning such as the "reversal curse" --- when trained on forward knowledge data of the form "$A \rightarrow B$" (e.g., Alice's husband is Bob), the model is unable to deduce the reversal knowledge "$B \leftarrow A$" (e.g., Bob's wife is Alice) during test. Extensive prior research suggests that this failure is an inherent, fundamental limit of autoregressive causal LLMs, indicating that these models tend to memorize factual-level knowledge rather than capture higher-level rules. In this paper, we challenge this view by showing that this seemingly fundamental limit can be mitigated by slightly tweaking the training data with a simple regularization data recipe called the Identity Bridge of the form "$A \to A$" (e.g., The name of Alice is Alice). Theoretically, we prove that under this recipe, even a one-layer transformer can break the reversal curse by analyzing the implicit bias of gradient descent. Empirically, we show that a 1B pretrained language model finetuned with the proposed data recipe achieves a 50\% success rate on reversal tasks, in stark contrast to a near-zero success rate when trained solely on forward-knowledge data. Our work provides a novel theoretical foundation for the reversal curse and offers a principled, low-cost path to encouraging LLMs to learn higher-level rules from data.

Forum: https://openreview.net/forum?id=kKUrbW8xJ3

---

## 45. Transforming Weather Data from Pixel to Latent Space

**Abstract:** The increasing impact of climate change and extreme weather events has spurred growing interest in deep learning for weather research. However, existing studies often rely on weather data in pixel space, which presents several challenges such as smooth outputs in model outputs, limited applicability to a single pressure-variable subset (PVS), and high data storage and computational costs. To address these challenges, we propose a novel Weather Latent Autoencoder (WLA) that transforms weather data from pixel space to latent space, enabling efficient data representation. By decoupling weather reconstruction from downstream tasks, WLA improves the accuracy and sharpness of weather task model results. The incorporated Pressure-Variable Unified Module transforms multiple PVS into a unified representation, enhancing the adaptability of the model in multiple weather scenarios. Furthermore, weather tasks can be performed in a low-storage latent space of WLA rather than a high-storage pixel space, thus significantly reducing data storage and computational costs. Through extensive experimentation, we demonstrate its superior compression and reconstruction performance, enabling the creation of the ERA5-Latent dataset with unified representations of multiple PVS from ERA5 data. The compressed full PVS in the ERA5-Latent dataset reduces the original 244.34 TB of data to 0.43 TB. The downstream task further demonstrates that task models can apply to multiple PVS with low data costs in latent space and achieve superior performance compared to models in pixel space.

Forum: https://openreview.net/forum?id=NlSWKeQPoZ

---

## 46. Learning-to-Optimize via Deep Unfolded Flows

**Abstract:** We introduce *FlowOptimizer*, a deep unfolded, flow-based framework for learned iterative optimization. Motivated by the expressiveness of flow models, we represent each optimization iteration via a velocity field that operates on a population of candidate solutions, i.e., a set of parallel iterates, conditioned on contextual information including their objective values and gradients, as well as population-level statistics. The velocity field is initially trained in a simulation-free manner by matching displacements from source populations to improved target ones obtained through sampling the objective. Subsequently, we unfold this velocity field as the internal iteration of an optimization sequence, and fine-tune it in an end-to-end manner by directly optimizing objective values over a targeted class of problems. Notably, FlowOptimizer is a self-supervised framework whose training relies solely on objective evaluations without requiring knowledge of solutions. We evaluate our approach on a series of tasks from standard non-convex optimization benchmarks to real-world problems from supply chain, robotics and power grid applications. FlowOptimizer consistently outperforms well-established sampling-based/gradient-based traditional optimization and learning-to-optimize methods, often by orders of magnitude in terms of solution quality. We further highlight its ability to be trained on low-dimensional problems and successfully generalize to substantially higher-dimensional $(\times 10)$ ones.

Forum: https://openreview.net/forum?id=ZOtOq7hxJP

---

## 47. Mechanistic Data Attribution: Tracing the Training Origins of Interpretable LLM Units

**Abstract:** While mechanistic interpretability has identified interpretable circuits in large language models (LLMs), their causal origins in training data remain elusive. We introduce *mechanistic data attribution* (MDA), a scalable framework that employs influence functions to trace interpretable units back to specific training samples. Through extensive experiments on the Pythia family, we causally validate that targeted intervention—removing or augmenting a small fraction of high-influence samples—significantly modulates the emergence of interpretable heads, whereas random interventions show no effect. Our analysis reveals that repetitive structural data (e.g., LaTeX, XML) acts as a mechanistic catalyst. Furthermore, we observe that interventions targeting induction head formation induce a concurrent change in the model’s in-context learning (ICL) capability. This provides direct causal evidence for the long-standing hypothesis regarding the functional link between induction heads and ICL. Finally, we propose a mechanistic data augmentation pipeline that consistently accelerates circuit convergence across model scales, providing a principled methodology for steering the developmental trajectories of LLMs.

Forum: https://openreview.net/forum?id=PQaxfoEcRc

---

## 48. Learning to Discover at Test Time

**Abstract:** How can we use AI to discover a new state of the art for a scientific problem? Prior work in test-time scaling, such as AlphaEvolve, performs search by prompting a frozen LLM. We perform reinforcement learning at test time, so the LLM can continue to train, but now with experience specific to the test problem. This form of continual learning is quite special, because its goal is to produce one great solution rather than many good ones on average, and to solve this very problem rather than generalize to other problems. Therefore, our learning objective and search subroutine are designed to prioritize the most promising solutions. We call this method Test-Time Training to Discover (TTT-Discover). Following prior work, we focus on problems with continuous rewards. We report results for every problem we attempted, across mathematics, GPU kernel engineering, algorithm design, and biology. TTT-Discover sets the new state of the art in almost all of them: (i) Erdős' minimum overlap problem and an autocorrelation inequality; (ii) a GPUMode kernel competition (up to 2× faster than prior art); (iii) past AtCoder algorithm competitions; and (iv) denoising problem in single-cell analysis. Our solutions are reviewed by experts or the organizers. All our results are achieved with an open model, OpenAI gpt-oss-120b, and can be reproduced with our publicly available code, in contrast to previous best results that required closed frontier models. Our test-time training runs are performed using Tinker, an API by Thinking Machines, with a cost of only a few hundred dollars per problem.

Forum: https://openreview.net/forum?id=96zNuQrH9Y

---

## 49. RoboMME: Benchmarking and Understanding Memory for Robotic Generalist Policies

**Abstract:** Memory is critical for long-horizon and history-dependent robotic manipulation. Such tasks often involve counting repeated actions or manipulating objects that become temporarily occluded. Recent vision-language-action (VLA) models have begun to incorporate memory mechanisms; however, their evaluations remain confined to narrow, non-standardized settings. This limits systematic understanding, comparison, and progress measurement.  To address these challenges, we introduce **RoboMME**: a large-scale standardized benchmark for evaluating and advancing VLA models in long-horizon, history-dependent scenarios. Our benchmark comprises 16 manipulation tasks constructed under a carefully designed taxonomy that evaluates *temporal*, *spatial*, *object*, and *procedural* memory. We further develop a suite of 14 memory-augmented VLA variants built on the $\pi_{0.5}$ backbone to systematically explore different memory representations across multiple integration strategies. We show that the effectiveness of memory representations is highly task-dependent, with each design offering distinct advantages and limitations across different tasks.  Videos and code can be found at https://robomme.github.io

Forum: https://openreview.net/forum?id=8m30ogkPk2

---

## 50. FlashSketch: Sketch-Kernel Co-Design for Fast Sparse Sketching on GPUs

**Abstract:** Sparse sketches such as the sparse Johnson–Lindenstrauss transform are a core primitive in randomized numerical linear algebra because they leverage random sparsity to reduce the arithmetic cost of sketching, while still offering strong approximation guarantees. Their random sparsity, however, is at odds with efficient implementations on modern GPUs, since it leads to irregular memory access patterns that degrade memory bandwidth utilization. Motivated by this tension, we pursue a sketch–kernel co-design approach: we design a new family of sparse sketches, BlockPerm-SJLT, whose sparsity structure is chosen to enable FlashSketch, a corresponding optimized CUDA kernel that implements these sketches efficiently. The design of BlockPerm-SJLT introduces a tunable parameter that explicitly trades off the tension between GPU-efficiency and sketching robustness. We provide theoretical guarantees for BlockPerm-SJLT under the oblivious subspace embedding (OSE) framework, and also analyze the effect of the tunable parameter on sketching quality. We empirically evaluate FlashSketch on standard RandNLA benchmarks, as well as an end-to-end ML data attribution pipeline called GraSS. FlashSketch pushes the Pareto frontier of sketching quality versus speed, across a range of regimes and tasks, and achieves a global geomean speedup of roughly $1.7 \times$ over the prior state-of-the-art GPU sketches.

Forum: https://openreview.net/forum?id=cCwxV6rSXF

---

## 51. Provable Bounds for the Learnability of Sample-Compressible Families from Noisy Samples

**Abstract:** Learning distribution families over $\mathbb{R}^d$ is a fundamental problem in unsupervised learning and statistics. A central question in this setting is whether a given family of distributions possesses sufficient structure to be (at least) information-theoretically learnable and, if so, to characterize its sample complexity. In 2018, Ashtiani et al. (2018) reformulated sample compressibility as a structural property of distribution classes, proving that it guarantees PAC-learnability. This discovery subsequently enabled a series of recent advancements in deriving nearly tight sample complexity bounds for various high-dimensional open problems. It has been further conjectured that the converse also holds: every learnable class admits a sample compression scheme, making the two notions to be equivalent. In this work, we establish that sample compressible families remain learnable even from perturbed samples, subject to a set of minimax-necessary and sufficient conditions. In particular, we assume samples are corrupted by an additive independent noise model, and theoretically derive sample complexity bounds for general sample compressible classes in arbitrary dimensions with respect to both $\ell_2$-norm and total variation distance.

Forum: https://openreview.net/forum?id=fRonOqrKWT

---

## 52. Joint-Space Empowerment as a Theory of Dexterous Motor Coordination

**Abstract:** Searching for effective policies is notoriously challenging in high-dimensional musculoskeletal systems, where multiple muscles actuate individual joints. Although this redundancy complicates naive policy search, it also implies that effective control can be captured by a low-dimensional action manifold. To identify such a manifold, we introduce Joint-Space Empowerment (JoSE), a novel information-theoretic objective that quantifies how much control an agent has over its mechanical degrees-of-freedom. We frame manifold discovery as an optimal precoding problem—where a state-dependent precoder maps low-dimensional latent actions to high-dimensional muscle commands—and derive the optimal precoder in closed form under control-affine Gaussian dynamics. We show that manipulation policies learned on this manifold display significantly enhanced dexterity, sample efficiency, and improved generalization. These results present optimal precoding as a general information-theoretic paradigm for coordinating high-dimensional actuators to control low-dimensional features.

Forum: https://openreview.net/forum?id=qI2eHwfNfh

---

## 53. The Relative Instability of Model Comparison with Cross-validation

**Abstract:** Cross-validation (CV) is known to provide asymptotically exact tests and confidence intervals for model improvement but only when the model comparison is *relatively stable*. Surprisingly, we prove that even simple, individually stable models can generate relatively unstable comparisons, calling into question the validity of CV inference. Specifically, we show that the Lasso and its close cousin, soft-thresholding, generate relatively unstable comparisons and invalid CV inferences, even in the most favorable of learning settings and when both models are individually stable. These findings highlight the importance of verifying relative stability before deploying CV for model comparison.

Forum: https://openreview.net/forum?id=kpdyQ1GDRt

---

## 54. SAW-Bench: Learning Situated Awareness in the Real World

**Abstract:** A core aspect of human perception is *situated awareness*, the ability to relate ourselves to the surrounding physical environment and reason over possible actions in context. However, most existing benchmarks for multimodal foundation models (MFMs) emphasize **environment-centric** spatial relations (relations among objects in a scene), while largely overlooking **observer-centric** relationships that require reasoning relative to agent's viewpoint, pose, and motion. To bridge this gap, we introduce SAW-Bench (**S**ituated **A**wareness in the Real **W**orld), a novel benchmark for evaluating egocentric situated awareness using real-world videos. SAW-Bench comprises 786 self-recorded videos captured with Ray-Ban Meta (Gen 2) smart glasses spanning diverse indoor and outdoor environments, and over 2071 *human-annotated* question-answer pairs. It probes a model’s observer–environment understanding with *six* different awareness tasks. Our comprehensive evaluation reveals a human-model performance gap of 37.66%, even with the best-performing MFM, Gemini 3 Flash. Beyond this gap, our in-depth analysis uncovers several notable findings; for example, while models can exploit partial geometric cues in egocentric videos, they often fail to infer a coherent camera geometry, leading to systematic spatial reasoning errors. We position SAW-Bench as a benchmark for situated spatial intelligence, moving beyond passive observation to understanding physically grounded, observer-centric dynamics.

Forum: https://openreview.net/forum?id=8lwrYjv6r7

---

## 55. On Efficient Scaling of GNNs via IO-Aware Layers Implementations

**Abstract:** Graph Neural Networks (GNNs) are bottlenecked by sparse, irregular memory access. Popular frameworks such as DGL and PyTorch Geometric support general message passing, but complex layers often materialize edge-wise intermediates, increasing memory traffic and limiting scalability on large graphs.
We take an I/O- and arithmetic-intensity--centric view and show that widely used layers fall into three kernel families: SpMM-based convolutions, reduction-based aggregations, and attention-based layers (GATv2/Graph Transformer). For each family, we develop GPU kernels that reduce data movement, improve locality, and remain robust across realistic graphs. We also study graph reordering and find that its impact depends on the kernel mapping: it benefits neighbor-parallel (gather-dominated) kernels more consistently than feature-parallel designs.
Empirically, our fused attention kernels reach up to **3.9**$\times$ speedup for Graph Transformer (median **1.6**$\times$), with Tensor Core (block-sparse) variants up to **7.3**$\times$ on locally dense graphs; for GATv2 we reach up to **8.5**$\times$ speedup (median **2.0**$\times$) while reducing peak memory by up to **76**$\times$ (median **6**$\times$). Our degree-aware reduction kernels achieve up to **10**$\times$ speedup (median **2.6**$\times$). For SpMM-based layers, properly cached cuSPARSE achieves up to **8**$\times$ speedup over DGL and outperforms evaluated custom baselines in the majority of evaluations. We release our implementations as drop-in replacements in our [GitHub repository](https://github.com/yandex-research/On-Efficient-Scaling-Of-GNNs) to support reproducible, hardware-aware GNN acceleration.

Forum: https://openreview.net/forum?id=w6JVbu7VFo

---

## 56. Root Cause Analysis of Failures in Microservices via Bayesian Root Cause Discovery

**Abstract:** Modern cloud systems rely on architectures with many interconnected microservices, which enable scalability and flexibility but make troubleshooting failures difficult. Identifying the root cause requires navigating complex dependencies, often beyond the capacity of domain experts. Causal models offer a principled approach to root cause analysis (RCA), but prior methods are typically sample inefficient, as they assume access to the full causal graph or require large numbers of post-failure interventions. We introduce Bayesian Root Cause Discovery (BRCD), which leverages a partial causal structure (a CPDAG learned during the pre-failure period) and performs Bayesian inference without enumerating all DAGs from each interventional Markov equivalence class ($\mathcal{I}$-MEC) for each root cause candidate. Using a recent uniform DAG sampling framework (Wienöbst et al., 2023), BRCD provides the first statistical consistency guarantees for nonparametric RCA, with both identifiability and finite-sample posterior bounds under $\varepsilon$-vanishing approximation. Empirically, across synthetic benchmarks and three microservice systems (Online Boutique, Sockshop, Petshop), BRCD achieves state-of-the-art top-$l$ accuracy while remaining effective in low-failure-sample regimes and scaling to large graphs.

Forum: https://openreview.net/forum?id=EOSV5tlpqg

---

## 57. When to Trust the Cheap Check: Weak and Strong Verification for Reasoning

**Abstract:** Reasoning with LLMs increasingly unfolds inside a broader verification loop. Internally, systems use cheap checks, such as self-consistency or proxy rewards, which we call **weak verification**. Externally, users inspect outputs and steer the model through feedback until results are trustworthy, which we call **strong verification**. These signals differ sharply in cost and reliability: strong verification can establish trust but is resource-intensive, while weak verification is fast and scalable but noisy and imperfect. We formalize this tension through **weak-strong verification policies**, which decide when to accept or reject based on weak verification and when to defer to strong verification. We introduce metrics capturing incorrect acceptance, incorrect rejection, and strong-verification frequency. Over population, we show that optimal policies admit a two-threshold structure and that **calibration** and **sharpness** govern the value of weak verifiers. Building on this, we develop an online algorithm that provably controls acceptance and rejection errors without assumptions on the query stream, the language model, or the weak verifier. Experiments on mathematical reasoning and sequential decision-making demonstrate that our algorithm achieves reliability comparable to exhaustive strong verification while significantly reducing verification cost.

Forum: https://openreview.net/forum?id=dfN4HMZbc9

---

## 58. Adaptive Testing for LLM Evaluation: A Psychometric Alternative to Static Benchmarks

**Abstract:** Evaluating large language models (LLMs) typically requires thousands of benchmark items, making the process expensive, slow, and increasingly impractical at scale. Existing evaluation protocols rely on average accuracy over fixed item sets, treating all items as equally informative despite substantial variation in difficulty and discrimination. We introduce ATLAS, an adaptive testing framework based on Item Response Theory (IRT) that estimates model ability using Fisher information–guided item selection. ATLAS reduces the number of required items by up to 90% while maintaining measurement precision. For instance, it matches whole-bank ability estimates using only 41 items (0.157 MAE) on HellaSwag (5,600 items). We further reconstruct accuracy from ATLAS's ability estimates and find that reconstructed accuracies closely match raw accuracies across all five benchmarks, indicating that ability  preserves the global performance structure. At the same time,  provides finer discrimination within accuracy-equivalent models: among more than 3,000 evaluated models, 23--31% shift by more than 10 rank positions, and models with identical accuracies receive meaningfully different ability estimates. Code and calibrated item banks available at https://github.com/Peiyu-Georgia-Li/ATLAS.

Forum: https://openreview.net/forum?id=ItJEC1Mk0V

---

## 59. What Preferences Can—and Cannot—Predict in Multi-Agent Online Learning

**Abstract:** We examine the interplay between ordinal, preference-based solution concepts in games and the outcomes of payoff-driven learning dynamics, asking to what extent the combinatorial data of a game—its *preference graph*—can predict the long-run behavior of no-regret dynamics such as *follow-the-regularized-leader* (FTRL). In one direction, we show that the skeleton of every *dynamically stable* set (i.e., the set of pure profiles it contains) must also be *preferentially stable*, that is, it must be closed under profitable deviations.
We then ask the converse question: when are preferences sufficient to describe the long-run behavior of the players' learning dynamics?
We begin by showing that preferences are indeed enough to fully characterize asymptotic stability in the case of *subgames*—i.e., subsets of pure profiles obtained by restricting players' action sets. Beyond this case however, the equivalence between dynamic and preferential stability breaks down: in particular, we construct a three-player game with a preferentially stable set whose span is dynamically *unstable*, showing that preferences are *not sufficient* to describe dynamically stable behavior in general. To restore stability, we introduce the notion of *leaklessness*, a measure of aggregate payoff drift away from a set of pure profiles, and we use it to identify a payoff-based condition guaranteeing that the span of a set of pure profiles is stable and attracting.

Forum: https://openreview.net/forum?id=5W30WwL8wt

---

## 60. Disentangling Geometry, Performance, and Training in Language Models

**Abstract:** Geometric properties of Transformer weights, particularly the unembedding matrix, have been widely useful in language model interpretability research.
Yet, their utility for estimating downstream performance remains unclear.
In this work, we systematically investigate the relationship between model performance and the unembedding matrix geometry, particularly its effective rank.
Our experiments, involving a suite of 108 OLMo-style language models trained under controlled variation, reveal several key findings.
While the best-performing models often exhibit a high effective rank, this trend is not universal across tasks and training setups. 
Contrary to prior work, we find that low effective rank does not cause late-stage performance degradation in small models, but instead co-occurs with it; we find adversarial cases where low-rank models do not exhibit saturation.
Moreover, we show that effective rank is strongly influenced by pre-training hyperparameters, such as batch size and weight decay, which in-turn affect the model's performance.
Lastly, extending our analysis to other geometric metrics and final-layer representation, we find that these metrics are largely aligned, but none can reliably predict downstream performance.
Overall, our findings suggest that the model's geometry, as captured by existing metrics, primarily reflects training choices rather than performance.

Forum: https://openreview.net/forum?id=kNb6NNhKgE

---

## 61. Strategic Navigation or Stochastic Search? How Agents and Humans Reason Over Document Collections

**Abstract:** Multimodal agents offer a promising path to automating complex document-intensive workflows. Yet, a critical question remains: do these agents demonstrate genuine strategic reasoning, or merely stochastic trial-and-error search?  To address this, we introduce MADQA, a benchmark of 2,250 human-authored questions grounded in 800 heterogeneous PDF documents. Guided by Classical Test Theory, we design it to maximize discriminative power across varying levels of agentic abilities. To evaluate agentic behavior, we introduce a novel protocol that measures the accuracy-effort trade-off. Using this framework, we show that while the best agents can match human searchers in raw accuracy, they succeed on largely different questions and rely on brute-force search to compensate for weak strategic planning. They fail to close the nearly 20\% gap to oracle performance, persisting in unproductive loops. We release the dataset and evaluation harness to help facilitate the transition from brute-force retrieval to calibrated, efficient reasoning.

Forum: https://openreview.net/forum?id=ds3ZOevkwx

---

## 62. Self-Soupervision: Cooking Model Soups without Labels

**Abstract:** Model soups are strange and strangely effective combinations of parameters. They take a model (the stock), fine-tune it into multiple models (the ingredients), and then mix their parameters back into one model (the soup) to improve predictions. While all known soups require supervised learning, and optimize the same loss on labeled data, our recipes for Self-Soupervision generalize soups to self-supervised learning (SSL). Our Self-Souping lets us flavor ingredients on new data sources, e.g. from unlabeled data from a task for transfer or from a shift for robustness. We show that Self-Souping on corrupted test data, then fine-tuning back on uncorrupted train data, boosts robustness by +3.5% (ImageNet-C) and +7% (LAION-C). Self-Soupervision also unlocks countless SSL algorithms to cook the diverse ingredients needed for more robust soups. We show for the first time that ingredients can differ in their SSL hyperparameters---and more surprisingly, in their SSL algorithms. We cook soups of MAE, MoCoV3, MMCR, and LeJEPA ingredients that are more accurate than any single SSL ingredient.

Forum: https://openreview.net/forum?id=z0UM7y0L4r

---

## 63. Why Linear Recurrent Memory Works in Partially Observable Reinforcement Learning

**Abstract:** The family of linear recurrent neural networks has shown strong performance as recurrent memory units in partially observable reinforcement learning. We provide a theoretical justification for their empirical effectiveness by constructing and studying two linear filters:
(i) the first exactly reproduces the pre–softmax logits of the belief vector in a hidden Markov model (HMM) under a deterministic transition matrix, thereby serving as a sufficient statistic for optimal policy learning, (ii) the second achieves vanishing state-decoding error under a nearly deterministic transition matrix, thus reducing state ambiguity to near zero. The results extend to action-controlled HMMs, where the corresponding linear filters become time-varying with action-dependent dynamics. We illustrate our main results through numerical experiments and further show that the constructed linear filter serves as a strong feature extractor in a small reinforcement learning game.

Forum: https://openreview.net/forum?id=ywjHJIkUgW

---

## 64. NorMuon: Making Muon more efficient and scalable

**Abstract:** The choice of optimizer significantly impacts the training efficiency and computational costs of large language models (LLMs). Recently, the Muon optimizer has demonstrated promising results by orthogonalizing parameter updates, improving optimization geometry through better conditioning.
Despite Muon’s emergence as a candidate successor to Adam, the potential for jointly leveraging their strengths—has not been systematically explored.
In this work, we bridge this gap by proposing NorMuon (Neuron-wise Normalized Muon), an optimizer that synergistically combines orthogonalization with neuron-level adaptive learning rates. Our analysis reveals that while Muon effectively reduces condition numbers, the resulting updates exhibit highly non-uniform neuron norms, causing certain neurons to dominate the optimization process. NorMuon addresses this imbalance by maintaining second-moment statistics for each neuron and applying row-wise normalization after orthogonalization, ensuring balanced parameter utilization while preserving Muon's conditioning benefits. To enable practical deployment at scale, we develop an efficient distributed implementation under the FSDP2 framework that distributes orthogonalization computations across devices.
Experiments across multiple model scales demonstrate that NorMuon consistently outperforms both AdamW and Muon, achieving a 21.74\% reduction in training steps relative to AdamW and an 11.31 percentage-point larger efficiency gain than Muon on 1.1B pretraining. Results suggest that orthogonalization and adaptive learning rates are complementary rather than competing, opening new avenues for optimizer design in large-scale deep learning.

Forum: https://openreview.net/forum?id=m1IRWFAMsa

---

## 65. Measuring Agents in Production

**Abstract:** LLM-based agents already operate in production across many industries, yet we lack an understanding of what technical methods make deployments successful.
We present the first systematic study of **M**easuring **A**gents in **P**roduction, MAP, using first-hand data from agent developers. We conducted 20 case studies via in-depth interviews and surveyed 86 deployed systems practitioners across 26 domains.
We investigate why organizations build agents, how they build them, how they evaluate them, and their top development challenges.
Our study finds that production agents are built using simple, controllable approaches:
68% execute at most 10 steps before human intervention, 70% rely on prompting off-the-shelf models instead of weight tuning, and 74% depend primarily on human evaluation.
Reliability (consistent correct behavior over time) remains the top development challenge, which practitioners currently address through systems-level design.
MAP documents the current state of production agents, providing the research community with visibility into deployment realities and underexplored research avenues.

Forum: https://openreview.net/forum?id=mWxEAgz3xu

---

## 66. On the Power of Source Screening for Learning Shared Feature Extractors

**Abstract:** Learning with shared representation is widely recognized as an effective way to separate commonalities from heterogeneity across various heterogeneous sources.  Most existing work includes all related data sources via simultaneously training a common feature extractor and source-specific heads. It is well understood that data sources with low relevance or poor quality may hinder representation learning. In this paper, we further dive into the question of which data sources should be learned jointly by focusing on the traditionally deemed "good" collection of sources, in which individual sources have similar relevance and qualities with respect to the true underlying common structure. Towards tractability, we focus on the linear setting where sources share a low-dimensional subspace. We find that source screening can play a central role in statistically optimal subspace estimation. We show that, for a broad class of problem instances, training on a carefully selected subset of sources suffices to achieve minimax optimality, even when a substantial portion of data is discarded. We formalize the notion of an informative subpopulation, develop algorithms and practical heuristics for identifying such subsets, and validate their effectiveness through both theoretical analysis and empirical evaluations on synthetic and real-world datasets.

Forum: https://openreview.net/forum?id=dTMrITkTr5

---

## 67. The Obfuscation Atlas: Mapping Where Honesty Emerges in RLVR with Deception Probes

**Abstract:** Training against white-box deception detectors has been proposed as a way to make AI systems honest. However, such training risks models learning to obfuscate their deception to evade the detector. Prior work has studied obfuscation only in artificial settings where models were directly rewarded for harmful output. We construct a realistic coding environment where reward hacking via hardcoding test cases naturally occurs, and show that obfuscation emerges in this setting.
We introduce a taxonomy of possible outcomes when training against a deception detector. The model either remains honest, or becomes deceptive via two possible obfuscation strategies. (i) *Obfuscated activations*: the model outputs deceptive text while modifying its internal representations to no longer trigger the detector. (ii) *Obfuscated policy*: the model outputs deceptive text that evades the detector, typically by including a justification for the reward hack. Empirically, obfuscated activations arise from representation drift during RL, with or without a detector penalty. The detector penalty only incentivizes obfuscated policies; we theoretically show this is expected for policy gradient methods. Sufficiently high KL regularization and detector penalty can yield honest policies, establishing white-box deception detectors as viable training signals for tasks prone to reward hacking.

Forum: https://openreview.net/forum?id=wrGSN9kAVD

---

## 68. Trojan-Speak: Bypassing Constitutional Classifiers with No Jailbreak Tax via Adversarial Finetuning

**Abstract:** Fine-tuning APIs offered by major AI providers create new attack surfaces where adversaries can bypass safety measures through targeted fine-tuning. We introduce **Trojan-Speak**, an adversarial fine-tuning method that bypasses Anthropic's Constitutional Classifiers. Our approach uses curriculum learning combined with GRPO-based hybrid reinforcement learning to teach models a communication protocol that evades LLM-based content classification. Crucially, while prior adversarial fine-tuning approaches report more than 25\% capability degradation on reasoning benchmarks, Trojan-Speak incurs less than 5\% degradation while achieving 99+\% classifier evasion for models with 14B+ parameters. We demonstrate that fine-tuned models can provide detailed responses to expert-level CBRN (Chemical, Biological, Radiological, and Nuclear) queries from Anthropic's Constitutional Classifiers bug-bounty program. Our findings reveal that LLM-based content classifiers alone are insufficient for preventing dangerous information disclosure when adversaries have fine-tuning access, and we show that activation-level probes can substantially improve robustness to such attacks.

Forum: https://openreview.net/forum?id=5Aydbj0wYI

---

## 69. Are VLMs Seeing or Just Saying? Uncovering the Illusion of Visual Re-examination

**Abstract:** Vision-Language Models (VLMs) often produce self-reflective statements like “let me check the figure again” during reasoning. Do such state- ments trigger genuine visual re-examination, or are they merely learned textual patterns? We in- vestigate this via VISUALSWAP, an image-swap probing framework: after a model reasons over an image, we replace it with a visually similar but semantically different one and test whether the model notices. We introduce VS-BENCH, 800 image pairs curated from MathVista, Math- Verse, MathVision, and MMMU-Pro. Exper- iments on Qwen3-VL, Kimi-VL, and ERNIE- VL reveal a striking failure: models overwhelm- ingly miss the swap, with accuracy dropping by up to 60%. Counterintuitively, thinking mod- els are nearly 3x more vulnerable than their in- structed counterparts, and scaling offers no mit- igation. Multi-turn user instructions restore vi- sual grounding, but self-generated reflective state- ments during continuous generation do not. At- tention analysis explains why: user instructions substantially elevate attention to visual tokens, whereas self-reflection does not. Current VLMs tend to say rather than actually see when claiming to perform visual re-examination. Our code and dataset are available at the project page: https://visualswap.github.io/

Forum: https://openreview.net/forum?id=DdU1o2ZvWi

---

## 70. The Signal is in the Steps: Local Scoring for Reasoning Data Selection

**Abstract:** Distilling long-form reasoning from teacher models into smaller students requires selecting which candidate solutions to train on. Recent work argues that one should select responses the student model assigns highest probability, i.e., favoring solutions ``natural'' to the student. However, we find that this approach works within a single teacher but fails when scaling to long reasoning traces from multiple diverse teachers. We identify a key cause: this approach scores entire solutions, but students generalize by recombining familiar reasoning steps, not by memorizing complete solutions. Full-trajectory scoring optimizes the wrong target; it rewards global fluency while the transferable signal lies in local step transitions. We propose Local Average Log Probability (LALP), which scores each reasoning step using only a small window of preceding context, measuring whether each step is justified by its immediate premises rather than whether the full response looks natural to the student. LALP enables two practical use cases: selecting the best teacher before fine-tuning and curating training data from diverse teacher pools. Across math, coding, and science reasoning tasks, LALP consistently improves accuracy when selecting the most natural solutions by a large margin.

Forum: https://openreview.net/forum?id=GcB3a6IonG

---

## 71. High-accuracy and dimension-free sampling with diffusions

**Abstract:** Diffusion models have shown remarkable empirical success in sampling from rich multi-modal distributions. Their inference relies on numerically solving a certain differential equation. This differential equation cannot be solved in closed form, and its resolution via discretization typically requires many small iterations to produce *high-quality* samples.
    More precisely, prior works have shown that the iteration complexity of discretization methods for diffusion models scales polynomially in the ambient dimension and the inverse accuracy $1/\varepsilon$. In this work, we propose a new solver for diffusion models relying on a subtle interplay between low-degree approximation and the collocation method, and we prove that its iteration complexity scales *polylogarithmically* in $1/\varepsilon$, yielding the first "high-accuracy" guarantee for a diffusion-based sampler that only uses (approximate) access to the scores of the data distribution. In addition, our bound does not depend explicitly on the ambient dimension; more precisely, the dimension affects the complexity of our solver only through the *effective radius* of the support of the target distribution.

Forum: https://openreview.net/forum?id=YA9jB86LDw

---

## 72. On the Limits of LLM Adaptability: Impact of Model-Internalized Priors on Annotation Task Performance

**Abstract:** Large Language Models (LLMs) are increasingly used for zero-shot annotation and LLM-as-a-judge tasks, yet their reliability hinges on how model-internalized priors interact with user-provided instructions. We investigate three dimensions of this interaction: (1) how an LLM's familiarity with data and task definitions affects performance, (2) the extent to which additional information in prompts can correct zero-shot errors (``decision stickiness''), and (3) model susceptibility to misaligned task definitions. Through experiments on toxicity detection across diverse datasets (spanning social media, gaming, news, and forums) using both dense and mixture-of-experts models, we find that nearly two-thirds of zero-shot errors are resistant to correction, with an overall rescue rate (fraction of initial errors corrected by prompting) of only 34.8\%. High-confidence errors prove especially resistant to correction. When given misaligned definitions, LLMs follow them while maintaining confidence levels unchanged from the aligned condition. Crucially, we introduce Definition-Specific Familiarity (DSF), which measures alignment between a model's internal concept and the task definition. After controlling for dataset-level confounds, DSF shows a positive association with model performance (partial $r=+0.41$), while three distinct memorization metrics (ROUGE-L, BERTScore, and embedding cosine similarity) all fail to show a positive association. These findings show the limitations of prompt-based correction in annotation tasks, highlighting the importance of definition alignment over text-level memorization.

Forum: https://openreview.net/forum?id=oTv2bKG5Qg

---

## 73. Do We Need Adam? Surprisingly Strong and Sparse Reinforcement Learning with SGD in LLMs

**Abstract:** Reinforcement learning (RL), particularly RL from verifiable reward (RLVR), has become a crucial phase of training large language models (LLMs) and a key focus of current scaling efforts. 
However, optimization practices in RL largely follow those of next-token-prediction stages (e.g., pretraining and supervised fine-tuning), despite the fundamental differences between RL and these stages emphasized by recent work.
One such practice is the use of the AdamW optimizer, which is widely adopted for training large-scale transformers despite its high memory overhead. 
Our analysis shows that both momentum and adaptive learning rate of AdamW are less influential in RL than in SFT, leading us to hypothesize that RL benefits less from Adam’s per-parameter adaptive learning rates and momentum.
Confirming our hypothesis, our experiments demonstrate that the substantially more memory-efficient SGD, which is known to perform poorly in supervised learning of large-scale transformers, matches or even outperforms AdamW in RL for LLMs.
Remarkably, full fine-tuning with SGD updates fewer than 0.02% of model without any sparsity-promoting regularization, more than 1,000 times fewer than AdamW. Our analysis offers potential reasons for this update sparsity.
Our findings provide fresh insights into the optimization dynamics of RL in LLMs and demonstrate that RL can be substantially more parameter-efficient than previously recognized.

Forum: https://openreview.net/forum?id=z31fdV4WRu

---

## 74. A Call to Lagrangian Action: Learning Population Mechanics from Temporal Snapshots

**Abstract:** The population dynamics of molecules, cells, and organisms are governed by a number of unknown internal and external forces. In the last decade, population dynamics have predominately been modeled with Wasserstein gradient flows. However, since gradient flows minimize free energy, they fail to capture important dynamical properties, such as periodicity. In this work, we propose a change in perspective by considering population dynamics that minimize Wasserstein Lagrangian action, rather than free energy. As our main theoretical contributions, we derive the Hamiltonian equations of motion from the principle of least population-level action and we show that these mechanics encompass classical mechanics, quantum mechanics, and gradient flows. We further leverage the Hamiltonian perspective to propose an algorithm that learns the population mechanics from observed marginals, without specifying the Lagrangian. We demonstrate that by directly learning the population mechanics, our method forecasts and interpolates unseen marginals without a reference process, and outperforms gradient flow and flow matching methods across a wide range of real and simulated experiments.

Forum: https://openreview.net/forum?id=MjPv4kRu3H

---

## 75. Learning Biophysical Models of Large-Scale Multineuronal Data To Enable Precise Neurostimulation

**Abstract:** Multi-compartment Hodgkin–Huxley (HH) models provide a principled framework for predicting neural dynamics and responses to electrical stimulation. However, fitting HH biophysical parameters typically requires intracellular recordings, which are invasive and low-throughput, limiting the ability to capture the geometry and cell-specific properties of many neurons in a given neural circuit. Multi-electrode arrays (MEAs) offer a scalable alternative—high-density extracellular measurements from full neural populations—but HH model complexity has so far precluded reliable biophysical inference from extracellular data alone. Here, we introduce a framework to rapidly infer HH parameters from designed features of extracellular MEA measurements by leveraging differentiable biophysical simulation and simulation-based inference, unlocking a wide range of downstream applications. In this work, we focus on a central goal of translational neuroengineering: predicting neural spiking responses to candidate neurostimulation patterns that would take hours to measure clinically. To validate our approach, we collected hundreds of hours of stimulation and recording data from isolated macaque retina with a 30 µm-pitch 512-electrode array. Our framework predicted previously unseen multi-electrode stimulation responses with 90.4\% accuracy using HH models fit from only a few minutes of recording, replacing hours of stimulus testing.

Forum: https://openreview.net/forum?id=4hoiRJsmpp

---

## 76. Local Redundancy: An Information-Theoretic Measure of Plasticity from Synthetic Memorization

**Abstract:** Plasticity—a neural network's ability to adapt to new tasks—is critical for continual and transfer learning. Existing measures, such as effective rank, dead neuron fraction, and weight norm, lack theoretical grounding and correlate poorly with performance on new tasks. We introduce *local redundancy*, an information-theoretic measure derived from universal compression theory. We define local redundancy as the worst-case redundancy of a local model family—parameters in an infinitesimal neighborhood along gradient directions—and show this is a principled measure of plasticity. Although local redundancy is intractable to compute exactly, we prove that the expected squared gradient norm on a synthetic memorization task provides an efficiently computable lower bound. Experiments on continual image classification and time series transfer learning demonstrate that local redundancy predicts downstream performance better than existing measures and enables pretraining checkpoint selection where validation loss plateaus.

Forum: https://openreview.net/forum?id=ucbH88BgIk

---

## 77. Symmetry Reveals Layerwise Dynamics: How Transformers Perform In-Context Classification

**Abstract:** Transformers can perform in-context classification from a few labeled examples, yet the inference-time algorithm remains opaque. We study multi-class linear classification in the hard no-margin regime and make the computation identifiable by enforcing feature- and label-permutation equivariance at every layer. This enables interpretability while maintaining functional equivalence and yields highly structured weights. From these models we extract an explicit depth-indexed recursion: an end-to-end identified, emergent update rule inside a softmax transformer, to our knowledge the first of its kind. Attention matrices formed from mixed feature-label Gram structure drive coupled updates of training points, labels, and the test probe. The resulting dynamics implement a geometry-driven algorithmic motif, which can provably amplify class separation and yields robust expected class alignment.

Forum: https://openreview.net/forum?id=fDMizWsNoG

---

## 78. Expressivity-Efficiency Tradeoffs for Hybrid Sequence Models

**Abstract:** Hybrid sequence models—combining Transformer and state-space model layers—seek to gain the expressive versatility of attention as well as the computational efficiency of state-space model layers. Despite burgeoning interest in hybrid models, we lack a basic understanding of the settings where—and underlying mechanisms through which—they offer benefits over their constituent models. In this paper, we study this question, focusing on a broad family of core synthetic tasks. For this family of tasks, we prove the existence of fundamental limitations for non-hybrid models. Specifically, any Transformer or state-space model that solves the underlying task requires either a large number of parameters or a large working memory. On the other hand, for two prototypical tasks within this family—namely selective copying and associative recall—we construct hybrid models of small size and working memory that provably solve these tasks, thus achieving the best of both worlds. Our experimental evaluation empirically validates our theoretical findings. Importantly, going beyond the settings in our theoretical analysis, we empirically show that learned—rather than constructed—hybrids outperform non-hybrid models with up to $6 \times$ as many parameters. We additionally demonstrate that hybrid models exhibit stronger length generalization and out-of-distribution robustness than non-hybrids.

Forum: https://openreview.net/forum?id=82EJxJzG6r

---

## 79. Midtraining Bridges Pretraining and Posttraining Distributions

**Abstract:** Midtraining, the practice of mixing specialized data with more general pretraining data in an intermediate training phase, has become widespread in language model development, yet there is little understanding of what makes it effective. We propose that midtraining functions as distributional bridging by providing better initialization for posttraining. We conduct controlled pretraining experiments, and find that midtraining benefits are largest for domains distant from general pretraining data, such as code and math, and scale with the proximity advantage the midtraining data provides toward the target distribution. In these domains, midtraining consistently outperforms continued pretraining on specialized data alone both in-domain and in terms of mitigating forgetting. We further conduct an investigation on the starting time and mixture weight of midtraining data, using code as a case study, and find that time of introduction and mixture weight interact strongly such that early introduction of specialized data is amenable to high mixture weights, while late introduction requires lower ones. This suggests that late introduction of specialized data outside a plasticity window cannot be compensated for by increasing data mixtures later in training. Beyond midtraining itself, this suggests that distributional transitions between any training phases may benefit from similar bridging strategies.

Forum: https://openreview.net/forum?id=5PfEQzE9bf

---

## 80. DiScoFormer: Plug-In Density and Score Estimation with Transformers

**Abstract:** Estimating probability density and its score from samples remains a core problem in generative modeling, Bayesian inference, and kinetic theory. Existing methods are bifurcated: classical kernel density estimators (KDE) generalize across distributions but suffer from the curse of dimensionality, while modern neural score models achieve high precision but require retraining for every target distribution. We introduce DiScoFormer (Density and Score Transformer), a ``train-once, infer-anywhere" equivariant Transformer that maps i.i.d. samples to both density values and score vectors, generalizing across distributions and sample sizes. Analytically, we prove that self-attention can recover normalized KDE, establishing it as a functional generalization of kernel methods; empirically, individual attention heads learn multi-scale, kernel-like behaviors. The model converges faster and achieves higher precision than KDE for density estimation, and provides a high-fidelity plug-in score oracle for score-debiased KDE, Fisher information computation, and Fokker-Planck-type PDEs.

Forum: https://openreview.net/forum?id=gyOWJpP8cQ

---

## 81. On the Role of Computation in Reinforcement Learning

**Abstract:** How does the amount of compute available to a reinforcement learning (RL) policy affect its learning? Can policies using a fixed amount of parameters, still benefit from additional compute? The standard RL framework does not provide a language to answer these questions formally. Empirically, deep RL policies are often parameterized as neural networks with static architectures, conflating the amount of compute and the number of parameters. In this paper, we formalize compute bounded policies and prove that policies which use more compute can solve problems and generalize to longer-horizon tasks that are outside the scope of policies with less compute. Building on prior work in algorithmic learning and model-free planning, we propose a minimal architecture that can use a variable amount of compute. Our experiments complement our theory. On a set 31 different tasks spanning online and offline RL, we show that $(1)$ this architecture achieves stronger performance simply by using more compute, and $(2)$ stronger generalization on longer-horizon test tasks compared to standard feedforward networks or deep residual network using upto 5 times more parameters.

Forum: https://openreview.net/forum?id=HKzMnBX5B2

---

## 82. Decoupling The "What" and "Where" With Polar Coordinate Positional Embedding

**Abstract:** The attention mechanism in a Transformer architecture matches key to query based on both content—the what—and position in a sequence—the where. We present an analysis indicating that what and where are entangled in the popular rotary position embedding (RoPE). This entanglement can impair performance particularly when decisions require independent matches on these two factors. We propose an improvement to RoPE, which we call Polar Coordinate Position Embedding or PoPE, that eliminates the what-where confound. PoPE is far superior on a diagnostic task requiring indexing solely by position or by content. On autoregressive sequence modeling in music, genomic, and natural language domains, Transformers using PoPE as the positional encoding scheme outperform baselines using RoPE with respect to evaluation loss (perplexity) and downstream task performance. On language modeling, these gains persist across model scale, from 124M to 774M parameters. Crucially, PoPE shows strong zero-shot length extrapolation capabilities compared not only to RoPE but even a method designed for extrapolation, YaRN, which requires additional fine tuning and frequency interpolation.

Forum: https://openreview.net/forum?id=I3Z9za1EkO

---

## 83. Robust Filter Attention: Self-Attention as Precision-Weighted State Estimation

**Abstract:** We introduce Robust Filter Attention (RFA), a formulation of self-attention as a robust state estimator. Each token is treated as a noisy observation of a latent trajectory governed by a linear stochastic differential equation (SDE), and attention weights are determined by consistency under this model rather than static feature similarity. Under isotropic noise and decay assumptions, RFA matches the computational complexity of standard attention. On language modeling benchmarks, RFA achieves lower perplexity than RoPE within the training window while remaining stable under zero-shot extrapolation to longer contexts. The framework also provides a dynamical interpretation of standard positional mechanisms, connecting rotational embeddings and recency biases to transport and uncertainty propagation induced by stochastic dynamics.

Forum: https://openreview.net/forum?id=GhI6lw5QKe

---

## 84. Decision Transformers As Zero-Shot Learners via Text-Behavior Alignment

**Abstract:** Offline meta-reinforcement learning (meta-RL) aims to train agents that can generalize to unseen tasks using pre-collected data from related tasks. Recent approaches leverage the scalability of transformer architectures to model behavior sequences and support task adaptation using target task demonstrations. However, such data is often unavailable in real-world settings, where the task objective may be known but cannot be easily demonstrated. In contrast, humans routinely interpret and perform new tasks based solely on natural language instructions. In this work, we explore the potential of using natural language task descriptions to enable zero-shot task adaptation in offline meta-RL without requiring any data from the target task. We propose the Text-Guided Decision Transformer (TG-DT), a framework that enables zero-shot generalization by grounding policy learning in natural language. TG-DT learns a shared embedding space between task descriptions and behavioral trajectories via a dual contrastive and matching-based objective, ensuring robust alignment. A transformer-based policy is then conditioned on these aligned representations to generate task-appropriate actions. At test time, TG-DT synthesizes policies for unseen tasks using only their text descriptions and can optionally leverage a description-guided data sharing strategy to enhance adaptation. Experiments on standard offline meta-RL benchmarks, including MuJoCo and Meta-World, demonstrate that TG-DT achieves strong generalization to unseen tasks.

Forum: https://openreview.net/forum?id=jqLPUccarO

---

## 85. Optimal Decision-Making Based on Prediction Sets

**Abstract:** Prediction sets can wrap around any ML model to cover unknown test outcomes with a guaranteed probability. Yet, it remains unclear how to use them optimally for downstream decision-making. Here, we propose a decision-theoretic framework that seeks to minimize the expected loss (risk) against a worst-case distribution consistent with the prediction set's coverage guarantee. We first characterize the minimax optimal policy for a fixed prediction set, showing that it balances the worst-case loss inside the set with a penalty for potential losses outside the set. Building on this, we derive the optimal prediction set construction that minimizes the resulting robust risk subject to a coverage constraint. Finally, we introduce Risk-Optimal Conformal Prediction (ROCP), a practical algorithm that targets these risk-minimizing sets while maintaining finite-sample distribution-free marginal coverage. Empirical evaluations on medical diagnosis and a toy static hazard-decision benchmark demonstrate that ROCP reduces critical mistakes compared to baselines, particularly when out-of-set errors are costly. The source code to reproduce our experiments is available at https://github.com/TaoWangPenn/Risk-Optimal-Conformal-Prediction.

Forum: https://openreview.net/forum?id=VAXW59dyfk

---

## 86. Modeling Hierarchical Thinking in Large Reasoning Models

**Abstract:** Large Reasoning Models (LRMs) solve complex tasks by generating long Chain-of-Thought (CoT) sequences; however, the emergent dynamics governing reasoning trajectories are not well understood and can lead to inconsistencies and reasoning pathologies. In this work, we propose to approximate LRM's emerging hierarchical reasoning dynamics as a trajectory within a Finite State Machine (FSM) transitioning among six abstract cognitive states. We demonstrate that these states and transitions can be captured in the latent state of the model. We believe that this representation can have different applications in the interpretability and optimization of LRM models. For example, by analyzing the topology of these transitions, we identify statistical shifts in reasoning strategies that help identify effective reasoning chains from those that fail. To illustrate these potential advantages, we propose $Q$-Value guided steering, a training-free inference-time control method that treats reasoning as a planning problem. We estimate the long-horizon utility of state transitions and apply sparse, orthogonal activation steering at sentence boundaries to align the CoT generation with optimal reasoning policies. Experiments across four benchmarks (AIME25, MATH-500, GSM8k, and GPQA Diamond) using three state-of-the-art open reasoning models demonstrate that $Q$-Value steering policy achieves significant performance gains with "surgical'' efficiency, often requiring $25\times$ fewer interventions than greedy and weighted baselines, which suggests that reasoning can be effectively controlled by guiding high-level cognitive dynamics rather than micro-managing token generation. Code is available at: https://github.com/shahariar-shibli/CoT-FSM.

Forum: https://openreview.net/forum?id=N44P3zrrgO

---

## 87. The Value of Variance: Mitigating Debate Collapse in Multi-Agent Systems via Uncertainty-Driven Policy Optimization

**Abstract:** Multi-agent debate (MAD) systems improve LLM reasoning through iterative deliberation, but remain vulnerable to debate collapse, a failure type where final agent decisions are compromised on erroneous reasoning. Existing methods lack principled mechanisms to detect or prevent such failures. To address this gap, we first propose a hierarchical metric that quantifies behavioral uncertainty at three levels: intra-agent (individual reasoning uncertainty), inter-agent (interactive uncertainty), and system-level (output uncertainty). Empirical analysis across several benchmarks reveals that our proposed uncertainty quantification reliably indicates system failures, which demonstrates the validity of using them as diagnostic metrics to indicate the system failure. Subsequently, we propose a mitigation strategy by formulating an uncertainty-driven policy optimization to penalize self-contradiction, peer conflict, and low-confidence outputs in a dynamic debating environment. Experiments demonstrate that our proposed uncertainty-driven mitigation reliably calibrates the multi-agent system by consistently improving decision accuracy while reducing system disagreement.

Forum: https://openreview.net/forum?id=Bc6c2OVWRh

---

## 88. The Power of Power Law: Asymmetry Enables Compositional Reasoning

**Abstract:** Natural language data follows a power-law distribution, with most knowledge and skills appearing at very low frequency. While a common intuition suggests that reweighting or curating data toward a uniform distribution may help models better learn these long-tail skills, we find a counterintuitive result: across a wide range of compositional reasoning tasks, such as state tracking and multi-step arithmetic, training under power-law distributions consistently outperforms training under uniform distributions. To understand this advantage, we introduce a minimalist skill-composition task and show that learning under a power-law distribution provably requires significantly less training data. Our theoretical analysis reveals that power law sampling induces a beneficial asymmetry that improves the pathological loss landscape, which enables models to first acquire high-frequency skill compositions with low data complexity, which in turn serves as a stepping stone to efficiently learn rare long-tailed skills. Our results offer an alternative perspective on what constitutes an effective data distribution for training models.

Forum: https://openreview.net/forum?id=K83orsg2X9

---

## 89. The Axiomatic Value of Regularization in AI Alignment from Human Preferences

**Abstract:** Reinforcement learning from human feedback is the leading approach to aligning powerful AI systems so that they can be safe and helpful for humanity. While RLHF is typically modelled as a problem of learning a single preference ranking from noisy feedback, true human preferences are complex and often conflicting, representing substantive disagreements stemming from the diversity of individual human values. With this motivation, a recent line of research has studied RLHF from the perspective of social choice theory, which provides a set of well-established desirable properties for aggregating diverse preferences. Seen through this lens, the standard learning objective in RLHF is equivalent to aggregating diverse human preferences via the Borda count rule. At the same time, several new RLHF algorithms have been proposed, which turn out to be equivalent to the von Neumann winner social choice rule. However, the connection between social choice theory and RLHF has thus far ignored the critical role of regularization to prevent divergence from a reference policy, which is utilized in essentially all practical RLHF algorithms. In this paper, we study how regularization affects the social choice axioms satisfied by different RLHF algorithms, and prove that regularization improves the axiomatic properties of the von Neumann winner rule. In contrast, the Borda count rule still fails to satisfy key social choice axioms even when regularized. These results provide a principled argument grounded in social choice theory for utilizing practical RLHF algorithms that correspond to the von Neumann winner, rather than the standard RLHF objective.

Forum: https://openreview.net/forum?id=9ydYaIe1Qj

---

## 90. Recurrent Equivariant Constraint Modulation: Learning Per-Layer Symmetry Relaxation from Data

**Abstract:** Equivariant neural networks exploit underlying task symmetries to improve generalization, but strict equivariance constraints can induce more complex optimization dynamics that can hinder learning. Prior work addresses these limitations by relaxing strict equivariance during training, but typically relies on prespecified, explicit, or implicit target levels of relaxation for each network layer, which are task-dependent and costly to tune. We propose Recurrent Equivariant Constraint Modulation (RECM), a layer-wise constraint modulation mechanism that learns appropriate relaxation levels solely from the training signal and the symmetry properties of each layer's input-target distribution, without requiring any prior knowledge about the task-dependent target relaxation level. We demonstrate that under the proposed RECM update, the relaxation level of each layer provably converges 
to a value upper-bounded by its symmetry gap, namely the degree to which its input-target distribution deviates from exact symmetry. Consequently, layers processing symmetric distributions recover full equivariance, while those with approximate symmetries retain sufficient flexibility to learn non-symmetric solutions when warranted by the data. Empirically, RECM outperforms prior methods across diverse exact and approximate equivariant tasks, including the challenging molecular conformer generation on the GEOM-Drugs dataset.

Forum: https://openreview.net/forum?id=STeISpzNSd

---

## 91. On the Origin of Neural Scaling Laws: from Random Graphs to Natural Language

**Abstract:** Scaling laws have played a major role in modern AI, providing predictive power over how model performance will improve with increasing resources. This has spurred intense interest in their origin, with a common suggestion being that they arise from power laws already present in the data. Here we study scaling laws for transformers trained to predict random walks on graphs with tunable complexity. We show that this simplified setting already yields scaling laws even in the absence of power laws in the data correlations. We further consider dialing down the complexity of language by training on sequences sampled from increasingly simplified generative language models, from 4,2,1-layer transformer language models down to language bigrams, revealing a monotonic evolution of the scaling exponents. Our results also include scaling laws obtained from training on random walks on random graphs drawn from Erdös-Renyi and scale-free Barabási-Albert ensembles. Finally, we revisit scaling laws for language modeling, demonstrating that several essential results can be reproduced using 2 layer transformers with context length of 100, demonstrate an alternative method for obtaining compute optimal curves, and provide preliminary evidence that maximal update parameterization may be more parameter efficient than standard parameterization.

Forum: https://openreview.net/forum?id=mu17VSX8q9

---

## 92. Reward Redistribution for CVaR MDPs using a Bellman Operator on L-infinity

**Abstract:** Tail-end risk measures such as static conditional value-at-risk (CVaR) are  used in safety-critical applications to prevent rare, yet catastrophic events. Unlike risk-neutral objectives, the static CVaR of the return depends on entire trajectories without admitting a recursive Bellman decomposition in the underlying Markov decision process. A classical resolution relies on state augmentation with a continuous variable. However, unless restricted to a specialized class of admissible value functions, this formulation induces sparse rewards and degenerate fixed points. In this work, we propose a novel formulation of the static CVaR objective based on augmentation. Our alternative approach leads to a Bellman operator with: (1) dense per-step rewards; (2) contracting properties on the full space of bounded value functions. Building on this theoretical foundation, we develop risk-averse value iteration and model-free Q-learning algorithms that rely on discretized augmented states. We further provide convergence guarantees and approximation error bounds due to discretization. Empirical results demonstrate that our algorithms successfully learn CVaR-sensitive policies and achieve effective performance-safety trade-offs.

Forum: https://openreview.net/forum?id=8LVv9hyMII

---

## 93. SoftJAX & SoftTorch: Empowering Automatic Differentiation Libraries with Informative Gradients

**Abstract:** Automatic differentiation (AD) frameworks such as JAX and PyTorch have enabled gradient-based optimization for a wide range of scientific fields. Yet, many ''hard'' primitives in these libraries such as thresholding, Boolean logic, discrete indexing, and sorting operations yield zero or undefined gradients that are not useful for optimization. While numerous ''soft'' relaxations have been proposed that provide informative gradients, the respective implementations are fragmented across projects, making them difficult to combine and compare. This work introduces **SoftJAX** and **SoftTorch**, open-source, feature-complete libraries for *soft differentiable programming*. These libraries provide a variety of soft functions as drop-in replacements for their hard JAX and PyTorch counterparts. This includes (i) elementwise operators such as *clip* or *abs*, (ii) utility methods for manipulating Booleans and indices via fuzzy logic, (iii) axiswise operators such as *sort* or *rank* -- based on optimal transport or permutahedron projections, and (iv) offer full support for straight-through gradient estimation. Overall, SoftJAX and SoftTorch make the toolbox of soft relaxations easily accessible to differentiable programming, as demonstrated through benchmarking and a practical case study. Code is available at github.com/a-paulus/softjax and github.com/a-paulus/softtorch.

Forum: https://openreview.net/forum?id=RKHDV40omz

---

## 94. VideoFlexTok: Flexible-Length Coarse-to-Fine Video Tokenization

**Abstract:** Visual tokenizers map high-dimensional raw pixels into a compressed representation for downstream modeling. Beyond compression, tokenizers dictate what information is preserved and how it is organized. A _de facto_ standard approach to video tokenization is to represent a video as a spatiotemporal 3D grid of tokens, each capturing local information from the original signal. This requires the downstream model, e.g., a text-to-video model, to learn to predict all low-level details "pixel-by-pixel" irrespective of the video's inherent complexity, leading to high learning complexity.
__We present _VideoFlexTok_, which represents videos with a variable-length sequence of tokens structured in a coarse-to-fine manner__, where the first tokens (emergently) capture information such as semantics and motion, and later tokens add fine-grained details. The generative flow decoder enables realistic video reconstructions from any token count. This representation structure allows adapting the token count to downstream needs and encoding videos longer than the baselines within the same budget.
We evaluate VideoFlexTok on class- and text-to-video generative tasks and show that it yields more efficient training than 3D grid tokens, _achieving comparable generation quality (gFVD and ViCLIP Score) with a 5x smaller model (1.1B vs 5.2B)._
Finally, we show how _VideoFlexTok can enable long video generation without prohibitive computational cost_ by training a text-to-video model on 10-second 81-frame videos with only 672 tokens, 8x fewer than a comparable 3D grid tokenizer.

Forum: https://openreview.net/forum?id=KyVlaw4BxE

---

## 95. A Dirac-Frenkel-Onsager Principle: Instantaneous Residual Minimization with Gauge Momentum for Nonlinear Parametrizations of PDE Solutions

**Abstract:** Dirac-Frenkel instantaneous residual minimization evolves nonlinear parametrizations of PDE solutions in time, but ill-conditioning can render the parameter dynamics non-unique. We interpret this non-uniqueness as a gauge freedom: nullspace directions that leave the 
time derivative unchanged can be used to select better-conditioned parameter velocities. Building on Onsager's minimum-dissipation principle, we introduce a history variable---interpretable as momentum---and inject it only along the nullspace directions. The resulting Dirac-Frenkel-Onsager dynamics preserve instantaneous residual minimization, in contrast to standard regularization that can introduce bias, while promoting temporally smooth parameter evolution. Examples demonstrate that the approach leads to increased robustness in singular and near-singular regimes.

Forum: https://openreview.net/forum?id=aDPbUSUCwh

---

## 96. Security--Fidelity Tradeoffs: The Hidden Cost of Prompt Injection Defense

**Abstract:** We identify a **security--fidelity tradeoff** in defending LLMs against indirect prompt injection: defenses resist injected instructions largely by
suppressing untrusted text, which corrupts tasks that must preserve it, such as translation and document editing. Attack-success metrics cannot see this, because a model that ignores an injection and one that faithfully processes it as data score identically. We introduce **SecFid**, a benchmark built so that *executing* an injection, *processing* it as data, and *ignoring* it produce distinguishable outputs. This makes fidelity measurable, and exposes a frontier: across 1,168 examples and 48 configurations, no model or defense
achieves both objectives. The highest-fidelity model reaches 96.5% fidelity at 47.8% security, while the most secure defenses invert this, at 99.3% security but only 71.0%--73.9% fidelity. Even defenses with identical security differ in how they earn it: some repair hijacks into faithful processing, others simply suppress benign content. A decision-theoretic analysis shows why no fixed choice can be right everywhere: the correct behavior is not a property of the defense but of the deployment, set by its relative cost of a hijack versus a dropped span. Security alone therefore measures only half of robustness, and reporting it without fidelity hides the price at which it was bought.

Forum: https://openreview.net/forum?id=vTJmIhNvC3

---

## 97. Tilt Matching for Scalable Sampling and Fine-Tuning

**Abstract:** We propose a simple, scalable algorithm based on stochastic interpolants for sampling from unnormalized densities and for fine-tuning generative models. The approach, Tilt Matching, arises from a dynamical equation relating the flow matching velocity to one targeting the same distribution tilted by a reward, implicitly solving a stochastic optimal control problem. The resulting velocity inherits the regularity of stochastic interpolant transports while minimizing an objective with strictly lower variance than flow matching itself. The update to the velocity field can be interpreted as the sum of all joint cumulants between the interpolant velocity and the reward, and to first order is their covariance. The method requires neither reward gradients nor backpropagation through trajectories of the flow or diffusion. We empirically demonstrate that the approach is efficient and highly scalable, providing state-of-the-art results on sampling under Lennard-Jones systems and competitive performance for fine-tuning Stable Diffusion, without requiring reward multipliers. The framework also applies directly to tilting few-step flow map models.

Forum: https://openreview.net/forum?id=dQA4Gjt4KU

---

## 98. From Feasible to Practical: Pareto-Optimal Synthesis Planning

**Abstract:** Current computer-aided synthesis planning (CASP) methods often treat retrosynthesis as solved once a single feasible route is identified, focusing primarily on convergence or shortest-path metrics. This view is misaligned with real-world practice, where chemists must balance competing objectives such as cost, sustainability, toxicity, and overall yield. To address this, we formulate synthesis planning as a multi-objective search problem and introduce MORetro$^\ast$, an algorithm that generates a Pareto front of synthesis routes to explicitly capture trade-offs between user-defined criteria. MORetro$^\ast$ uses weighted scalarization and solution-informed sampling to efficiently navigate the combinatorial search space and prioritize promising trade-offs. Building on multi-objective A$^\ast$-search, we provide optimality guarantees showing that, for a fixed single-step model, MORetro$^\ast$ recovers the true Pareto front under admissibility. Across multiple retrosynthesis benchmarks, MORetro$^\ast$ produces diverse, high-quality Pareto fronts, uncovering solutions overlooked by single-objective approaches and better aligning CASP outputs with industrial decision-making.

Forum: https://openreview.net/forum?id=qpPf5mI1qn

---

## 99. Neural Feature Geometry Evolves as Discrete Ricci Flow

**Abstract:** Deep neural networks learn feature representations via complex geometric transformations of the input data manifold. Despite the models' empirical success across domains, our understanding of neural feature representations is still incomplete. In this work we investigate neural feature geometry through the lens of discrete geometry. Since the input data manifold is typically unobserved, we approximate it using geometric graphs that encode local similarity structure. We provide theoretical results on the evolution of these graphs during training, showing that nonlinear activations play a crucial role in shaping feature geometry in feedforward neural networks. Moreover, we discover that the geometric transformations resemble a discrete Ricci flow on these graphs, suggesting that neural feature geometry evolves analogous to Ricci flow. This connection is supported by experiments on over 20,000 feedforward neural networks trained on binary classification tasks across both synthetic and real-world datasets. We observe that the emergence of class separability corresponds to the emergence of community structure in the associated graph representations, which is known to relate to discrete Ricci flow dynamics. Building on these insights, we introduce a novel framework for locally evaluating geometric transformations through comparison with discrete Ricci flow dynamics. Our experimental results further suggest connections between the evolution of feature geometry, and training time and network depth.

Forum: https://openreview.net/forum?id=YPH5yCKzYr

---

## 100. PRISM: Demystifying Retention and Interaction in Mid-Training

**Abstract:** Mid-training is increasingly used to improve the reasoning capabilities of large language models (LLMs), yet its design choices and interaction with evaluation and reinforcement learning (RL) remain poorly understood. Prior work often focuses on narrow domain gains, overlooking retention of general abilities, long-context performance, and RL compatibility. We present $\textbf{PRISM}$ (Demystifying Retention and Interaction in Mid-Training), a holistic empirical study that analyzes mid-training design choices, what to evaluate, and how domain mixtures and training stages interact across model families. Experiments on Granite-3.3 8B, LLaMA-3.1 8B, and Mistral-7B/24B base models show that a relatively small, high-quality mid-training phase of $\textbf{$\sim$27B}$ tokens acts as a critical stabilizing stage for reasoning. Across models, PRISM yields consistent gains of $\textbf{$\sim$6–10}$ points on coding benchmarks and $\textbf{$\sim$17–30}$ points on mathematical reasoning benchmarks while preserving general performance. RL applied on top of PRISM-mid-trained models produces stable, monotonic improvements, adding a further $\textbf{$\sim$3–8}$ points across coding and math tasks such as LiveCodeBench, Codeforces, AIME and MATH500, and $\textbf{$\sim$17–20}$ points on science (GPQA-Diamond), whereas RL applied directly to base models is substantially less effective. Our results demonstrate that retention-aware mid-training is a necessary intermediate step for reliable reasoning enhancement and RL scaling, and provide practical guidance for designing robust mid-training pipelines for modern LLMs.

Forum: https://openreview.net/forum?id=giNBsVVFGt

---

## 101. Score-Repellent Monte Carlo: Toward Efficient Non-Markovian Sampler with Constant Memory in General State Spaces

**Abstract:** History-dependent sampling can reduce long-run Monte Carlo variance by discouraging redundant revisits, but existing schemes typically encode history through empirical measure on finite state spaces, which is infeasible in high-dimensional discrete configuration spaces or ill-posed in continuous domains. We propose *Score-Repellent Monte Carlo* (SRMC) framework that summarizes trajectory history by a running average of score evaluations in $\mathbb{R}^d$, where $d$ is the dimension of the score and state representation. This history is converted into a surrogate target through an exponential *score tilt*, indexed with $\alpha$ that represents the *strength of repellence* in controlling the magnitude of the history-based repulsion. The surrogate family is normalization-free in the standard MCMC sense, yielding a generic wrapper: at each iteration, any base kernel targeting $\pi$ can instead be run on the current surrogate $\pi_{\theta_n}$ while the history is updated online. We analyze the coupled evolution of the history recursion and Monte Carlo estimators using stochastic approximation with controlled Markovian noise, establishing almost sure convergence and a joint central limit theorem. We further identify regimes in which the asymptotic covariance decreases as $\alpha$ increases, with scaling $O(1/\alpha)$, extending the near-zero-variance effect of finite-state history-dependent samplers to general state spaces with constant memory. Experiments on continuous targets and discrete energy-based models demonstrate improved estimator variance and mode coverage, while retaining $O(d)$ memory usage and modest per-iteration overhead.

Forum: https://openreview.net/forum?id=PN8EiOzMuT

---

## 102. Simple Algorithms for Bad Triangle Transversals with Applications to Correlation Clustering

**Abstract:** The Bad Triangle Transversal (BTT) problem asks for the smallest set of edges that need to be removed from a given signed graph, so that the resulting graph does not have a bad triangle. Here, a bad triangle is a triangle with exactly one negative edge.
Several 2-approximations for BTT are proposed in this paper. On the hardness side, we show that BTT is NP-hard to approximate with factor better than $\frac{2137}{2136}$ on complete graphs. Our reduction also works for Correlation Clustering (CC), the Cluster Deletion problem (CD) and the Minimum Strong Triadic Closure problem (MinSTC). Lastly, we show that the BTT and CC optima are within a factor of 3/2 in complete graphs, by describing a pivot procedure that transforms transversals into clusters.

Forum: https://openreview.net/forum?id=Yq66fTPjHn

---

## 103. Base Models Know How to Reason, Thinking Models Learn When

**Abstract:** What do different *thinking* language models learn during training?
We introduce *constructive model diffing*, a framework for understanding fine-tuned models by constructing the base-to-fine-tuned difference from interpretable components, producing hybrid models whose performance recovery measures how well the components capture the diff.
For thinking models, we decompose the diff into *reasoning mechanisms* (category-specific vectors that induce reasoning behaviors in the base model) and *reasoning heuristics* (a classifier determining when each mechanism fires).
To ground this decomposition, we use Sparse Autoencoders to discover interpretable taxonomies of reasoning behaviors.
Across nine model configurations (four RL-trained, four SFT-distilled from a larger thinking model, and one mixed), we find a striking difference: hybrid models recover much more performance for RL-trained models than for SFT-trained or mixed ones. This indicates that RL primarily teaches models to apply sophisticated heuristics to pre-existing base capabilities, while SFT-based distillation teaches the base model new mechanisms.
These results offer a new lens on what different training paradigms teach, with implications for efficient reasoning model development.

Forum: https://openreview.net/forum?id=2BniakOS4q

---

## 104. Treatment Responder Classification with Abstention

**Abstract:** Treatment responder classification seeks to learn a rule to classify individuals who will benefit from the treatment. This paper studies a new scenario in treatment responder classification when abstention is allowed, i.e., practitioners can opt out of making uncertain classification on some individuals for further investigation. By revealing the implicit relation between causal misclassification risk with abstention and Conditional Value at Risk (CVaR), we develop a doubly robust method named TRECA to learn the classification rule under loose convergence conditions on nuisance parameters, and further extend it to deal with possible violation on key assumptions such as monotonicity and unconfoundedness. Rigorous theories and extensive experiments on two real-world datasets demonstrate the theoretical and experimental guarantee on our methods in learning treatment responders classification rules with low regret at the cost of limited abstention.

Forum: https://openreview.net/forum?id=WFdQSjmchK

---

## 105. Unraveling Syntax: Language Modeling and the Substructure of Grammars

**Abstract:** While language models achieve impressive results, their *learning dynamics* are far from understood. Many domains of interest – such as natural language syntax, coding languages, arithmetic – are captured by context-free grammars (CFGs). In this work, we extend prior work on neural language modeling of CFGs in a novel direction: how language modeling behaves with respect to CFG *substructure*, namely sub*grammars*. We define subgrammars, and prove a set of fundamental theorems connecting language modeling and subgrammars. We show that language modeling loss recurses linearly over its top-level subgrammars; applied recursively, the loss decomposes into losses for "irreducible" subgrammars. Under additional assumptions, and empirically, parametrized models learn subgrammars in parallel, unlike children who first master simple substructures. We find that subgrammar pretraining can improve final performance, but only for tiny models relative to the grammar, while alignment analyses show that pretraining consistently leads to internal representations that better reflect the grammar’s substructure.

Forum: https://openreview.net/forum?id=sPIXjSkDFG

---

## 106. Protein Fold Classification at Scale: Benchmarking and Pretraining

**Abstract:** Classifying protein topology is essential for deciphering biological function, but progress is held back by the lack of large-scale benchmarks that avoid duplicates and by models that do not scale well. We introduce TEDBench, a large-scale, non-redundant benchmark for protein fold classification constructed from the Encyclopedia of Domains (TED) and Foldseek-clustered AlphaFold structures. We show that on TEDBench, current protein representation learning methods either require very large models or fail to deliver strong performance.
To address this challenge, we propose Masked Invariant Autoencoders (MiAE), a self-supervised framework for protein structure representation learning. MiAE uses an extremely high masking ratio of up to $90\%$ with an $\mathrm{SE(3)}$-invariant encoder and a lightweight decoder that reconstructs backbone coordinates from the latent representation and mask tokens. MiAE scales well and outperforms supervised counterparts and state-of-the-art baselines on TEDBench, establishing a strong recipe for protein fold classification. To test transfer beyond AlphaFold structures, we further benchmark on a curated dataset from experimental structures of CATH v4.4. TEDBench is available at https://github.com/BorgwardtLab/TEDBench.

Forum: https://openreview.net/forum?id=jPKqiaPTEd

---

## 107. Riemannian Metric Matching for Scalable Geometric Modeling of Distributions

**Abstract:** High-dimensional datasets often concentrate near low-dimensional structures, but estimating their geometry from samples typically relies on graphs and kernels that scale poorly with dataset size and dimension.
We propose **Riemannian metric matching**: a denoising probabilistic framework for learning the Riemannian geometry of data using neural networks.
Specifically, we learn the *carré du champ* operator, which, using diffusion geometry, gives us access to the Riemannian geometry toolkit for downstream machine learning and statistical tasks.
Our key observation is that the carré du champ operator can be formulated as a conditional expectation over random perturbations of the data,
which can be exploited for sample-wise training and constant cost, amortized inference without explicit kernel construction.
Empirically, metric matching rivals or improves the accuracy of $k$-NN-based diffusion geometry estimators, while enabling amortized inference that is up to $400\times$ faster, and supports graph-free geometric analysis on high-dimensional images where nearest neighbors break down.

Forum: https://openreview.net/forum?id=KVzXnWPLgX

---

## 108. How much can language models memorize?

**Abstract:** We propose a new method for estimating how much a model knows about a datapoint and use it to measure the capacity of modern language models. Prior studies of language model memorization have struggled to disentangle memorization from generalization. We formally separate memorization into two components: unintended memorization, the information a model contains about a specific dataset, and generalization, the information a model contains about the true data-generation process. When we completely eliminate generalization, we can compute the total memorization, which provides an estimate of model capacity: our measurements estimate that GPT-style models have a capacity of approximately 3.6 bits per parameter. We train language models on datasets of increasing size and observe that models memorize until their capacity fills, at which point unintended memorization decreases as models begin to generalize. We train hundreds of transformer language models ranging from 500K to 1.5B parameters and produce a series of scaling laws relating model capacity and data size to membership inference.

Forum: https://openreview.net/forum?id=bA6BgSbaUi

---

## 109. PRISM: Gauge-Invariant Tangent-Space Differentially Private LoRA

**Abstract:** Applying differential privacy (DP) via DP-SGD to Low-Rank Adaptation (LoRA) is a natural approach for privacy-preserving fine-tuning. However, LoRA's low-rank parameterization poses a fundamental challenge. In LoRA, each trainable update is represented as a low-rank matrix $Z = AB^\top$, but this factorization is inherently *non-identifiable*: many factor pairs $(A, B)$ represent the same update $Z$. As a result, applying DP-SGD directly to the factors induces *gauge-dependent* perturbations on $Z$, and we show that this naive DP-LoRA can lead to unbounded noise amplification. We propose **PRISM**, an intrinsic DP mechanism for LoRA that is gauge invariant by construction, avoids bilinear noise amplification, and admits an efficient low-dimensional noise sampler. Moreover, PRISM yields a closed-form characterization of the effective intrinsic noise induced on $Z$, enabling stable privacy–utility trade-offs through bounded, gauge-invariant perturbations. We establish standard $(\varepsilon,\delta)$-DP guarantees for PRISM and introduce a DP-aware, gauge-invariant adaptive update rule that prevents adaptive optimization from amplifying injected privacy noise, improving numerical stability in practice.

Forum: https://openreview.net/forum?id=SiCjKmArjQ

---

## 110. Prescriptive Scaling Reveals the Evolution of Language Model Capabilities

**Abstract:** Machine learning model performance arises from competition and application. For deployment, we consider the prescriptive scaling laws: given a pre-training compute budget, what downstream accuracy is attainable with contemporary post-training practice, and how stable is that mapping as the field evolves? Using large-scale observational evaluations with 5k observational and 2k newly sampled data on model performance, we estimate capability boundaries—high conditional quantiles of benchmark scores as a function of log pre-training FLOPs, via smoothed quantile regression with a monotone, saturating sigmoid parameterization. We validate the temporal reliability by fitting on earlier model generations and evaluating on later releases. Across various tasks, the estimated boundaries are mostly stable, with the exception of math reasoning that exhibits a consistently advancing boundary over time. We then extend our approach to analyze task-dependent saturation and to probe contamination-related shifts on math reasoning tasks. Finally, we introduce an efficient algorithm that recovers near-full-data frontiers using roughly 20% of evaluation budget. Together, our work releases the Proteus-2k, the latest model performance evaluation dataset, and introduces a practical methodology for translating compute budgets into reliable performance expectations and for monitoring when capability boundaries move.

Forum: https://openreview.net/forum?id=IkjsHRpuYY

---

## 111. Flow Sampling : Learning to Sample from Unnormalized Densities via Denoising Conditional Processes

**Abstract:** Sampling from unnormalized densities is analogous to the generative modeling problem, but the target distribution is defined by a known energy function instead of data samples. Because evaluating the energy function is often costly, a primary challenge is to learn an efficient sampler. We introduce Flow Sampling, a framework built on diffusion models and flow matching for the data-free setting. Our training objective is conditioned on a noise sample and regresses onto a denoising diffusion drift constructed from the energy function. In contrast, diffusion models' objective is conditioned on a data sample and regresses onto a noising diffusion drift. We utilize the interpolant process to minimize the number of energy function evaluations during training, resulting in an efficient and scalable method for sampling unnormalized densities. Furthermore, our formulation naturally extends to Riemannian manifolds, enabling diffusion-based sampling in geometries beyond Euclidean space. We derive a closed-form formula for the conditional drift on constant curvature manifolds, including hyperspheres and hyperbolic spaces. We evaluate Flow Sampling on synthetic energy benchmarks, small peptides, large-scale amortized molecular conformer generation, and distributions supported on the sphere, demonstrating strong empirical performance.

Forum: https://openreview.net/forum?id=YlcyOMTPNl

---

## 112. Flowers: A Warp Drive for Neural PDE Solvers

**Abstract:** We introduce Flower, a neural architecture for learning PDE solution operators built entirely from multihead warps. Aside from pointwise channel mixing and a multiscale scaffold, Flowers use no Fourier multipliers, no dot-product attention, and no convolutional mixing. Each head predicts a displacement field and warps the mixed input features. Motivated by physics and computational efficiency, displacements are predicted pointwise, without any spatial aggregation, and nonlocality enters *only* through sparse sampling at source coordinates, *one* per head. Stacking warps in multiscale residual blocks yields Flowers, which implement adaptive, global interactions at linear cost. We theoretically motivate this design through three complementary lenses: flow maps for conservation laws, waves in inhomogeneous media, and a kinetic-theoretic continuum limit. Flowers achieve excellent performance on a broad suite of 2D and 3D time-dependent PDE benchmarks, particularly flows and waves. A compact 17M-parameter model consistently outperforms Fourier, convolution, and attention-based baselines of similar size, while a 150M-parameter variant improves over recent transformer-based foundation models with much more parameters, data, and training compute.

Forum: https://openreview.net/forum?id=TPpUKH8fuQ

---

## 113. Falling Trees: A Model Class for Interpretable Risk Prioritization

**Abstract:** Many real-world decisions require prioritizing high-risk cases, such as clinicians prioritizing high-risk patients before lower-risk ones. Falling rule lists (FRLs), which are ordered if--then rules with monotonically decreasing risks, provide an interpretable framework for such tasks; however, their single-path structure yields a highly restricted model class. We introduce falling trees, a new family of interpretable models that enforces the same monotonic risk constraint while permitting tree-structured branching. We present GraviTree, a novel dynamic-programming-with-bounds algorithm for learning the Rashomon set of falling trees under depth and branching constraints. Our formulation can interpolate between rule lists and full decision trees, enabling user-desired model expressivity. In a new clinical dataset and in many public classification benchmarks, falling trees match or outperform FRLs and other interpretable baselines, often producing more sparse decisions for high-risk instances. Our results show that falling trees strike a practical balance between interpretability, expressiveness, and risk prioritization for high-stakes settings.

Forum: https://openreview.net/forum?id=UWLHtRaVHp

---

## 114. Learning to Execute Graph Algorithms Exactly with Graph Neural Networks

**Abstract:** Understanding what graph neural networks can learn, especially their ability to learn to execute algorithms, remains a central theoretical challenge. In this work, we prove exact learnability results for graph algorithms under bounded-degree and finite-precision constraints. Our approach follows a two-step process. First, we train an ensemble of multi-layer perceptrons (MLPs) to execute the local instructions of a single node. Second, during inference, we use the trained MLP ensemble as the update function within a graph neural network (GNN). Leveraging Neural Tangent Kernel (NTK) theory, we show that local instructions can be learned from a small training set, enabling the complete graph algorithm to be executed during inference without error and with high probability. To illustrate the learning power of our setting, we establish a rigorous learnability result for the LOCAL model of distributed computation. We further demonstrate positive learnability results for widely studied algorithms such as message flooding, breadth-first and depth-first search, and Bellman-Ford.

Forum: https://openreview.net/forum?id=YKEmoqwkE9

---

## 115. Faster Activation Functions at the Edge for Post-Training Speedups

**Abstract:** On-device AI has gained significant attention for enabling efficient, low-latency inference on edge devices.
However, tight resource constraints on these platforms make the deployment of accurate and lightweight deep
learning models challenging.
In particular,
advanced activation functions (AFs) like Swish and GELU often incur
high inference overhead due to the lack of hardware fast-paths for exponentiation and division,
restricting edge-ML applications to simple AFs like ReLU, limiting model accuracy.
To address this, we propose FFCC, a compiler that automatically
generates efficient approximations of AFs through floating-point reinterpretation.
These functions do not require hardware fast-paths, meaning they remain fast
on edge devices, but are accurate enough to be used as post-training drop-ins.
FFCC takes a specification of AFs using basic floating-point operators
and applies derivation rules to lower these expressions into
efficient instruction sequences.
Our experiments show that FFCC provides fast approximations of AFs, achieving order-of-magnitude
speed-ups over accurate baselines on Arm M7, Aarch64 and Intel platforms.
Using ConvNeXt as an example, we demonstrate how these activation-level gains translate to end-to-end speed-ups,
and do not result in significant loss of model accuracy.

Forum: https://openreview.net/forum?id=u2o6hPE8ik

---

## 116. Skill Neologisms: Towards Skill-based Continual Learning

**Abstract:** Modern LLMs show mastery over an ever-growing range of skills, as well as the ability to compose them flexibly. However, extending model capabilities to new skills in a scalable manner is an open problem: fine-tuning and parameter-efficient variants risk catastrophic forgetting, while context-based approaches have limited expressiveness and are constrained by the model's effective context. 
We explore *skill neologisms*--soft tokens integrated in the model's vocabulary and optimized to improve capabilities over a specific skill--as a way to selectively acquire new skills without weight updates. We first observe that pre-trained LLMs already exhibit tokens associated with procedural knowledge. We then show on a controlled synthetic task that skill neologisms can be learned to improve model capabilities on specific skills while being composable with out-of-distribution skills, and that independently trained skill neologisms can be composed zero-shot. Finally, we validate zero-shot composition of independently learned skill neologisms on the more realistic natural language setting of the Skill-Mix benchmark. These results suggest that skill neologisms may provide a scalable path towards skill-based continual learning.

Forum: https://openreview.net/forum?id=5VgZUEpK6W

---

## 117. Expressive Graph Neural Networks via Equivariant Use of Noise

**Abstract:** Expressivity has been a major focus in the design of Graph Neural Networks (GNNs), yet a significant gap persists between theoretical universal expressivity and practical performance. While many expressive GNNs are efficient and achieve strong results, they often focus on specific graph properties and lack theoretical expressivity for general graph tasks. Conversely, theoretically universal-expressive models often suffer from high computational costs or poor generalization, limiting their real-world applicability. To bridge this gap, we introduce Equivariant Noise GNNs (ENGNNs), a framework that utilizes random noise features to enhance the expressivity of GNNs. Crucially, unlike prior methods that naively use noise, we enforce equivariance to nodewise noise transformations, such as orthogonal transformations. We prove that this property reduces the model's theoretical sample complexity, thereby improving generalization. Our framework simultaneously reaches theoretical universal expressivity, maintains the linear scalability of standard Message-Passing Neural Networks in practice, and achieves performance comparable to computationally expensive, high-expressivity models. Extensive experiments confirm strong performance across node, link, subgraph, and graph-level prediction tasks, demonstrating that the equivariant use of noise provides a powerful and practical pathway for building expressive GNNs. Our code is available at \url{https://github.com/MuLabPKU/EquivNoiseGNN}.

Forum: https://openreview.net/forum?id=TK2ae5Vwwz

---

## 118. Benchmarking at the Edge of Comprehension

**Abstract:** As frontier Large Language Models (LLMs) increasingly saturate new benchmarks shortly after they are published, benchmarking itself is at a juncture: if frontier models keep improving, it will become increasingly hard for humans to generate discriminative tasks, provide accurate ground-truth answers, or evaluate complex solutions.
If benchmarking becomes infeasible, our ability to measure any progress in AI is at stake. We refer to this scenario as the *post-comprehension regime*.
In this work, we propose Critique-Resilient Benchmarking, an adversarial framework designed to compare models even when full human understanding is infeasible. 
Our technique relies on the notion of *critique-resilient correctness*: an answer is deemed correct if no adversary has convincingly proved otherwise.
Unlike standard benchmarking, humans serve as bounded verifiers and focus on localized claims, which preserves evaluation integrity beyond full comprehension of the task. 
Using an itemized bipartite Bradley-Terry model, we jointly rank LLMs by their ability to solve challenging tasks and to generate difficult yet solvable questions. 
We showcase the effectiveness of our method in the mathematical domain across eight frontier LLMs, showing that the resulting scores are stable and correlate with external capability measures. 
Our framework reformulates benchmarking as an adversarial generation-evaluation game in which humans serve as final adjudicators.

Forum: https://openreview.net/forum?id=BhahZSDowo

---

## 119. 3ViewSense: Spatial and Mental Perspective Reasoning from Orthographic Views in Vision-Language Models

**Abstract:** Current Large Language Models have achieved Olympiad-level logic, yet Vision-Language Models paradoxically falter on elementary spatial tasks like block counting. This capability mismatch reveals a critical "spatial intelligence gap," where models fail to construct coherent 3D mental representations from 2D observations. We uncover this gap via diagnostic analyses showing the bottleneck is a missing view-consistent spatial interface rather than insufficient visual features or weak reasoning. To bridge this, we introduce **3ViewSense**, a framework that grounds spatial reasoning in Orthographic Views. Drawing on engineering cognition, we propose a "Simulate-and-Reason" mechanism that decomposes complex scenes into canonical orthographic projections to resolve geometric ambiguities. By aligning egocentric perceptions with these allocentric references, our method facilitates explicit mental rotation and reconstruction. Empirical results on spatial reasoning benchmarks demonstrate that our method significantly outperforms existing baselines, with consistent gains on occlusion-heavy counting and view-consistent spatial reasoning. The framework also improves the stability and consistency of spatial descriptions, offering a scalable path toward stronger spatial intelligence in multimodal systems.

Forum: https://openreview.net/forum?id=Hm8OEDKpiO

---

## 120. Sycophancy Towards Researchers Drives Performative Misalignment

**Abstract:** The increasing situational awareness of language models raises safety concerns: models might be aware when they are evaluated, and adjust their behavior to evade monitoring and resist modification, e.g., pretending to be aligned only in evaluation. This *alignment faking* behavior is often interpreted as scheming: an intentional effort of strategic deception. In this paper, we examine an alternative interpretation, *performative misalignment*, which explains the change in behavior as a result of *sycophancy towards AI researchers*. To examine this hypothesis, we present three empirical findings. First, we show that evaluation awareness persists even when we tell models they are deployed, which contradicts the scheming story which predicts less misalignment when the model perceives evaluation. Second, we use probing and steering to show that our current methods cannot mechanistically distinguish sycophancy and scheming in alignment faking evaluations. Third, we fine-tune models to be more sycophantic and observe increased sensitivity to evaluation cues. To conclude, we emphasize deconfounding sycophancy from scheming for future work on evaluations and mitigations of intent misalignment.

Forum: https://openreview.net/forum?id=rLFIOikFR2

---

## 121. VectorWorld: Efficient Streaming World Model via Diffusion Flow on Vector Graphs

**Abstract:** Closed-loop evaluation of autonomous-driving policies requires interactive 
simulation beyond log replay. Existing generative world models suffer three gaps: history-incompatible initialization, sampling latency exceeding real-time budgets, and compounding kinematic infeasibility. We propose VectorWorld, a streaming vector-graph world model that incrementally generates ego-centric lane--agent tiles during rollout. VectorWorld couples a motion-aware gated VAE for history-compatible initialization, an edge-gated relational DiT with interval-conditioned MeanFlow and JVP-based large-step supervision for solver-free outpainting, and $\Delta$Sim, a physics-aligned NPC policy with hybrid discrete--continuous actions and differentiable kinematic logit shaping. On Waymo Open Motion and nuPlan, VectorWorld improves map fidelity, initialization validity, and density calibration, enabling stable real-time $1\mathrm{km}+$ closed-loop rollouts.

Forum: https://openreview.net/forum?id=yTEEiE3YtD

---

## 122. On the Existence of Consistent Adversarial Attacks in High-Dimensional Linear Classification

**Abstract:** What fundamentally distinguishes an adversarial attack from a misclassification due to limited model expressivity or finite data? 
In this work, we investigate this question in the setting of high-dimensional binary classification, where statistical effects due to limited data availability play a central role.
We introduce a new error metric that precisely capture this distinction, quantifying model vulnerability to consistent adversarial attacks --- perturbations that preserve the ground-truth labels.
Our main technical contribution is an exact and rigorous asymptotic characterization of these metrics in both well-specified models and latent space models, revealing different vulnerability patterns compared to standard robust error measures. 
The theoretical results demonstrate that as models become more overparameterized, their vulnerability to label-preserving perturbations grows, offering theoretical insight into the mechanisms underlying model sensitivity to adversarial attacks.

Forum: https://openreview.net/forum?id=PUIivg3GrO

---

## 123. Which Algorithms Can Graph Neural Networks Learn?

**Abstract:** In recent years, there has been growing interest in understanding neural architectures' ability to learn to execute discrete algorithms, a line of work often referred to as neural algorithmic reasoning. The goal is to integrate algorithmic reasoning capabilities into larger neural pipelines. Many such architectures are based on (message-passing) graph neural networks (MPNNs), owing to their permutation equivariance and ability to deal with sparsity and variable-sized inputs. However, much existing work is either largely empirical and lacks formal guarantees or it focuses solely on expressivity, leaving open the question of when and how such architectures generalize beyond a finite training set. In this work, we propose a general theoretical framework that characterizes sufficient conditions under which MPNNs can learn an algorithm from a training set of small instances and provably approximate its behavior on inputs of arbitrary size with worst-case guarantees. Our framework applies to a broad class of algorithms, including single-source shortest paths, minimum spanning trees, and general dynamic programming problems, such as the $0$-$1$ knapsack problem. In addition, we establish impossibility results for a wide range of algorithmic tasks, showing that standard MPNNs cannot learn them and derive more expressive MPNN-like architectures that overcome these limitations. Finally, we refine our analysis for the Bellman–Ford algorithm, yielding substantially smaller required training sets and significantly extending the recent work of Nerem et al., 2025 by allowing for a differentiable regularization loss. Empirical results largely support our theoretical findings.

Forum: https://openreview.net/forum?id=GnmuZIlvxw

---

## 124. Efficient Parallel Samplers for Recurrent-Depth Models and Their Connection to Diffusion Language Models

**Abstract:** Language models with recurrent depth, also referred to as universal or looped when considering transformers, are defined by the capacity to increase their computation through the repetition of layers. Recent efforts in pretraining have demonstrated that these architectures can scale to modern language modeling tasks while exhibiting advantages in reasoning tasks. 
  In this work, we examine the relationship between recurrent-depth models and diffusion language models. Building on their similarities, we develop a new diffusion forcing sampler for these models to accelerate generation. The sampler advances by decoding new tokens at every forward pass of the model, while the latent states of these tokens can be further refined in parallel through recurrence. Theoretically, under a fixed wall-clock budget, generation with our sampler is strictly more expressive than baseline autoregressive generation as it preserves the same recurrent depth while updating a strictly wider front of token positions in parallel, enabling more computation at equal serial depth. 
  Moreover, this sampler, based on principles from diffusion literature, can be directly applied to existing 3.5B recurrent-depth transformers without any tuning, leading to up to a 5x speedup.

Forum: https://openreview.net/forum?id=h7WBYYJF1Q

---

## 125. Lottery Prior: Randomized Neural Compression for Zero-Shot Inverse Problems

**Abstract:** We study zero-shot inverse problems, where a clean signal is recovered from a single degraded observation without external training data. Contrary to the common belief that such problems require highly complex models, we show that a lightweight neural network, when combined with entropy and complexity regularization in a compression-based formulation, is sufficient for high-quality restoration. We propose Lottery Prior, a compression-based inverse solver that leverages architectural priors from random networks and induces a family of implicit priors through randomness, enabling ensemble-based refinement. We further derive non-asymptotic error bounds for compression-based maximum-likelihood inverse solvers, revealing how rate–distortion constraints act as implicit regularizers. Experiments on denoising, noisy super-resolution, and inpainting demonstrate that our method achieves state-of-the-art with significantly fewer effective parameters. Project page: https://eedavidwu.github.io/LotteryPrior/

Forum: https://openreview.net/forum?id=YNoQhMrps4

---

## 126. Mixture of Concept Bottleneck Experts

**Abstract:** Concept Bottleneck Models (CBMs) promote interpretability by grounding predictions in human-understandable concepts. However, existing CBMs typically constrain their task predictor to a single expression whose functional form is set a priori, limiting both predictive accuracy and adaptability to diverse user needs. We propose Mixture of Concept Bottleneck Experts (M-CBEs), a framework that generalizes existing CBMs along two dimensions: the number of expressions, referred to as experts, employed by the task predictor to map concepts to the task, and the functional form each expression takes, thus exposing an underexplored region of this design space. We investigate this region by instantiating two novel models: Linear M-CBE, which learns a finite set of linear expressions, and Symbolic M-CBE, which leverages symbolic regression to discover expert functions from data subject to user-specified operator vocabularies. Empirical evaluation demonstrates that varying the number of expressions and their functional form provides a robust framework for navigating the accuracy-interpretability trade-off.

Forum: https://openreview.net/forum?id=Jj8Vec8qSs

---

## 127. On the Optimization Trajectory of DeepWalk Embeddings

**Abstract:** The DeepWalk algorithm has been widely used for learning node embeddings in graphs. Combined with the idea of _negative sampling_, the DeepWalk algorithm has been shown to be implementable at scale, easily handling graphs with millions of nodes. However, theoretical guarantees on the resulting embeddings are much less understood. Recent results have studied the minimizers of the objective and have shown interesting guarantees for certain graph classes. However, the optimization _trajectory_, i.e., what happens when we start at a random initialization and run gradient descent, remains poorly understood. This is especially true for the implementation of DeepWalk using Skip-gram with negative sampling (SGNS), since the variance of the stochastic updates turns out to be very large. In this work, we make progress on this question. We show that for "small norm" initialization, under a spectral gap assumption on the graph, the DeepWalk embeddings align with the column space of a fixed low-rank matrix. For graphs generated from Stochastic Block Models with certain separation conditions, our results imply that the DeepWalk embeddings recover cluster structure. To the best of our knowledge, our results give the first analysis of the optimization trajectory of DeepWalk with negative sampling on non-trivial graph classes.

Forum: https://openreview.net/forum?id=YKX6FgtL3R

---

## 128. Divide-and-Denoise: A Game-Theoretic Method for Fairly Composing Diffusion Models

**Abstract:** The abundance of pre-trained diffusion models provides an opportunity for composition. Combining several models, however, runs the risk of one model dominating or models disagreeing with each other. Here, we propose Divide-and-Denoise, a method for coordinating multiple pre-trained diffusion models during sampling. Much like managing a specialized workforce, our method creates a fair but efficient division of labor across models. Central to our method is the notion of an allocation which defines the responsibility of each model to every region of the noisy sample. At every timestep, we then denoise by (i) updating the allocation by solving a fair division game, where we divide the sample into regions that maximize total utility under fairness constraints, and (ii) aligning the models with this allocation, where we guide each model to denoise within its assigned region. This leads to a new composite denoising process that evolves in tandem with a division process. We evaluate Divide-and-Denoise on conditional image generation. Across several quality metrics, including the GenEval benchmark, our method outperforms baselines and resolves common failures including missing objects and mismatched attributes. Experiments show that Divide-and-Denoise utilizes each model's expertise without neglecting any other model.

Forum: https://openreview.net/forum?id=9voQUicsc2

---

## 129. SWING: Unlocking Implicit Graph Representations for Graph Random Features

**Abstract:** We propose SWING: Space Walks for Implicit Network Graphs, a new class of algorithms for computations involving Graph Random Features on graphs given by implicit representations (i-graphs), where edge-weights are defined as bi-variate functions of feature vectors in the corresponding nodes. Those classes of graphs include several prominent examples, such as: *$\epsilon$-neighborhood* graphs, used on regular basis in machine learning. Rather than conducting walks on graphs' nodes, those methods rely on walks in continuous spaces, in which those graphs are embedded. To accurately and efficiently approximate original combinatorial calculations, SWING applies customized Gumbel-softmax sampling mechanism with linearized kernels, obtained via random features coupled with importance sampling techniques. This mechanism is of its own interest. SWING relies on the deep connection between implicitly defined graphs and Fourier analysis, presented in this paper. SWING is accelerator-friendly and does not require input graph materialization. We provide detailed analysis of SWING and complement it with thorough experiments on different classes of i-graphs.

Forum: https://openreview.net/forum?id=LBcnybFVBp

---

## 130. Decoupling Skeleton and Flesh: Efficient Multimodal Table Reasoning with Disentangled Alignment and Structure-aware Guidance

**Abstract:** Reasoning over table images remains challenging for Large Vision-Language Models (LVLMs) due to complex layouts and tightly coupled structure-content information. Existing solutions often depend on expensive supervised training, reinforcement learning, or external tools, limiting efficiency and scalability. This work addresses a key question: how to adapt LVLMs to table reasoning with minimal annotation and no external tools? Specifically, we first introduce DiSCo, a Disentangled Structure-Content alignment framework that explicitly separates structural abstraction from semantic grounding during multimodal alignment, efficiently adapting LVLMs to tables structures. Building on DiSCo, we further present Table-GLS, a Global-to-Local Structure-guided reasoning framework that performs table reasoning via structured exploration and evidence-grounded inference. Extensive experiments across diverse benchmarks demonstrate that our framework efficiently enhances LVLM's table understanding and reasoning capabilities, particularly generalizing to unseen table structures. Our data and code are available at https://github.com/AAAndy-Zhu/TableVLM.

Forum: https://openreview.net/forum?id=PN7l7mBPDO

---

## 131. Exact Unlearning in Reinforcement Learning

**Abstract:** We formulate the problem of \emph{exact unlearning} in reinforcement learning, where the goal is to design an efficient framework that enables the removal of any user’s data upon deletion request, i.e., the online learner’s output after unlearning be \emph{indistinguishable} from what would have been produced had the deleted user never interacted with the learner. For any $\rho >0$, we show that there exists a reinforcement learning (RL) algorithm that is $\rho$-TV-stable and supports an exact unlearning procedure whose expected computational cost is only a $\rho \sqrt{\ln T}$ fraction of the computational cost of retraining from scratch. We construct such a $\rho$-TV-stable RL algorithm for tabular Markov decision processes (MDPs), which achieves a regret bound of  $\mathcal{O}(H^2 \sqrt{SAT} + H^3 S^2 A + {H^{2.5} S^2 A}/{\rho})$, where $S, A, H$, and $T$ denote the number of states, the number of actions, the episode horizon, and the number of episodes, respectively. We also establish a lower bound of $\Omega(H\sqrt{SAT}+{SAH}/{\rho})$  for $\rho$-TV-stable RL algorithms,  showing that our algorithm is nearly minimax optimal.

Forum: https://openreview.net/forum?id=oLmwqIhzqj

---

## 132. Time series saliency maps: Explaining models across multiple domains

**Abstract:** Traditional saliency map methods, popularized in computer vision, highlight individual input points that contribute most to a model's output. However, in the context of time series, they offer limited insights because semantically meaningful features are often found in other domains. Thus, we introduce in this paper Cross-domain Integrated Gradients, a generalization of Integrated Gradients that enables feature attributions in any domain formulated as an invertible, differentiable transformation of the time domain. Our derivation extends Integrated Gradients into complex-valued domains, enabling frequency-based attributions, while preserving path independence and completeness. We validate our method via controlled mechanistic experiments, quantitative faithfulness and perturbation-stability tests, and real-world case studies. Across wearable heart-rate extraction, EEG-based seizure detection, and zero-shot forecasting, our proposed Cross-domain Integrated Gradients approach identifies whether predictions rely on heart-rate frequencies or interference, epileptic sources or artifacts, and trend or seasonal components, revealing model behaviour that time-domain saliency does not capture. We release an open-source library with TensorFlow, native PyTorch, and Captum support for plug-and-play cross-domain explainability of time-series models.

Forum: https://openreview.net/forum?id=Bd0NNopzpC

---

## 133. Steer Like the LLM: Activation Steering that Mimics Prompting

**Abstract:** Large language models can be steered at inference time through prompting or activation interventions, but activation steering methods often underperform compared to prompt-based approaches. We investigate whether activation steering can be improved by learning to mimic the interventions that prompt steering triggers within the model. To this end, we introduce *Prompt Steering Replacement (PSR)* models, a new family of activation steering methods that distill prompt steering behavior into interpretable interventions on model activations. A PSR is an activation steering method that estimates position-specific steering coefficients and is trained to imitate prompt-based interventions. Experiments on persona steering and instruction following across multiple language models demonstrate that PSR models consistently outperform constant-coefficient interventions that are frequently used in the literature and achieve performance close to or exceeding prompt steering while maintaining interpretability.

Forum: https://openreview.net/forum?id=06Nk3dJDMq

---

## 134. Biased Generalization in Diffusion Models

**Abstract:** Generalization in generative modeling is defined as the ability to learn an underlying distribution from a finite dataset and produce novel samples, with evaluation largely driven by held-out performance and perceived sample quality. In practice, training is often stopped at the minimum of the test loss, taken as an operational indicator of generalization. We challenge this viewpoint by identifying a phase of *biased generalization* during training, in which the model continues to decrease the test loss while favoring samples with anomalously high proximity to training data. By training the same network on two disjoint datasets and comparing the mutual distances of generated samples and their similarity to training data, we introduce a quantitative measure of bias and demonstrate its presence on real images. We then study the mechanism of bias, using a controlled hierarchical data model where access to exact scores and ground-truth statistics allows us to precisely characterize its onset. We attribute this phenomenon to the sequential nature of feature learning in deep networks, where coarse structure is learned early in a data-independent manner, while finer features are resolved later in a way that increasingly depends on individual training samples. Our results show that early stopping at the test loss minimum, while optimal under standard generalization criteria, may be insufficient for privacy-critical applications.

Forum: https://openreview.net/forum?id=IqTvyp40O6

---

## 135. SceneSmith: Agentic Generation of Simulation-Ready Indoor Scenes

**Abstract:** Simulation has become a key tool for training and evaluating home robots at scale, yet existing environments fail to capture the diversity and physical complexity of real indoor spaces. Current scene synthesis methods produce sparsely furnished rooms that lack the dense clutter, articulated furniture, and physical properties essential for robotic manipulation. We introduce SceneSmith, a hierarchical agentic framework that generates simulation-ready indoor environments from natural language prompts. SceneSmith constructs scenes through successive stages—from architectural layout to furniture placement to small object population—each implemented as an interaction among VLM agents: designer, critic, and orchestrator. The framework tightly integrates asset generation through text-to-3D synthesis for static objects, dataset retrieval for articulated objects, and physical property estimation. SceneSmith generates 3-6x more objects than prior methods, with $<$2\% inter-object collisions and 96\% of objects remaining stable under physics simulation. In a user study with 205 participants, it achieves 92\% average realism and 91\% average prompt faithfulness win rates against baselines. We further demonstrate that these environments can be used in an end-to-end pipeline for automatic robot policy evaluation.

Forum: https://openreview.net/forum?id=WwS8CTpUA6

---

## 136. Securing Multimodal AI through Internal Information Decomposition

**Abstract:** Multimodal large language models introduce attack surfaces absent in unimodal systems: adversaries can distribute malicious intent across modalities to evade unimodal safeguards. This motivates using cross-modal consistency as a detection signal rather than inspecting each modality in isolation. Our key observation is that benign inputs induce compatible predictive behavior from text-only and vision-only reasoning that stabilizes when fused, whereas adversarial manipulation disrupts this consistency, causing abnormal multimodal behavior. Existing defenses that examine raw inputs or outputs overlook this internal fusion process, rendering them brittle and computationally expensive. We propose FlowGuard, a lightweight inference-time framework that detects harmful inputs by monitoring internal multimodal consistency. Unlike approaches that rely on scalar confidence metrics, FlowGuard derives FlowVectors inspired by Partial Information Decomposition that quantify cross-modal redundancy, synergy, and modality-specific dominance, capturing whether multimodal fusion aligns with unimodal semantic evidencebetween unimodal and fused multimodal output distributions. In a one-class classification problem trained solely on benign data, FlowGuard reduces Attack Success Rates from $>90\%$ to $<15\%$ on unseen attacks, with $<3\%$ utility loss and up to a $6\times$ latency reduction. Our results demonstrate that monitoring cross-modal consistency offers an efficient and effective defense for multimodal reasoning.

Forum: https://openreview.net/forum?id=GEzZIUmEqE

---

## 137. Excited Pfaffians: Generalized Neural Wave Functions Across Structure and State

**Abstract:** Neural-network wave functions in Variational Monte Carlo (VMC) have achieved great success in accurately representing both ground and excited states. However, achieving sufficient numerical accuracy in state overlaps requires increasing the number of Monte Carlo samples, and consequently the computational cost, with the number of states. We present a nearly constant sample-size approach, Multi-State Importance Sampling (MSIS), that leverages samples from all states to estimate pairwise overlap. To efficiently evaluate all states for all samples, we introduce Excited Pfaffians. Inspired by Hartree-Fock, this architecture represents many states within a single neural network. Excited Pfaffians also serve as generalized wave functions, allowing a single model to represent multi-state potential energy surfaces. On the carbon dimer, we match the $\mathcal{O}(N_s^4)$-scaling natural excited states while training $>200\times$ faster and modeling 50\% more states. Our favorable scaling enables us to be the first to use neural networks to find all distinct energy levels of the beryllium atom. Finally, we demonstrate that a single wave function can represent excited states across various molecules.

Forum: https://openreview.net/forum?id=meEsugjXjv

---

## 138. Hierarchical Successor Representation for Robust Transfer

**Abstract:** The successor representation (SR) provides a powerful framework for decoupling
  predictive dynamics from rewards, enabling rapid generalisation across reward
  configurations. However, the classical SR is limited by its inherent policy
  dependence: policies change due to ongoing learning, environmental
  non-stationarities, and changes in task demands, making established predictive
  representations obsolete. Furthermore, in topologically complex environments,
  SRs suffer from spectral diffusion, leading to dense and overlapping features
  that scale poorly. Here we propose the Hierarchical Successor Representation
  (HSR) for overcoming these limitations. By incorporating temporal abstractions
  into the construction of predictive representations, HSR learns stable state
  features which are robust to task-induced policy changes. Applying
  non-negative matrix factorisation (NMF) to the HSR yields a sparse, low-rank
  state representation that facilitates highly sample-efficient transfer to
  novel tasks in multi-compartmental environments. Further analysis reveals that
  HSR-NMF discovers interpretable topological structures, providing a
  policy-agnostic hierarchical map that effectively bridges model-free
  optimality and model-based flexibility. Beyond providing a useful basis for
  task-transfer, we show that HSR's temporally extended predictive structure can
  also be leveraged to drive efficient exploration, effectively scaling to
  large, procedurally generated environments.

Forum: https://openreview.net/forum?id=txswvMHt4u

---

## 139. Pressure Reveals Character: Behavioural Alignment Evaluation at Depth

**Abstract:** Evaluating alignment in language models requires testing how they behave under realistic pressure, not just what they claim they would do. While alignment failures increasingly cause real-world harm, comprehensive evaluation frameworks with realistic multi-turn scenarios remain lacking. We introduce an alignment benchmark spanning 904 scenarios across six categories---Honesty, Safety, Non-Manipulation, Robustness, Corrigibility, and Scheming---validated as realistic by human raters. Our scenarios place models under conflicting instructions, simulated tool access, and multi-turn escalation to reveal behavioral tendencies that single-turn evaluations miss. Evaluating 24 frontier models using LLM judges validated against human annotations, we find that even top-performing models exhibit gaps in specific categories, while the majority of models show consistent weaknesses across the board. Factor analysis reveals that alignment behaves as a unified construct (analogous to the g-factor in cognitive research) with models scoring high on one category tending to score high on others. We publicly release the benchmark and an interactive leaderboard to support ongoing evaluation, with plans to expand scenarios in areas where we observe persistent weaknesses and to add new models as they are released.

Forum: https://openreview.net/forum?id=3U1l24EDvL

---

## 140. Decomposed On-Policy Distillation for Vision-Language Reasoning: Steering Gradients for Visual Grounding

**Abstract:** While on-policy distillation offers dense supervision for training small reasoning models, its optimization dynamics in the multimodal domain remain under-explored. In this work, we challenge the standard monolithic view of Vision-Language Model (VLM) distillation by mathematically decomposing the loss into two distinct components: the language prior and visual grounding. Our analysis uncovers that gradient vectors for these components are nearly orthogonal, indicating that the objective of aligning with the teacher's language distribution is geometrically independent from the objective of matching its visual perception. Consequently, standard optimization passively follows a suboptimal compromise trajectory that implicitly balances the two objectives. Hypothesizing that visual grounding constitutes the primary bottleneck for vision-language reasoning, we introduce Visual Gradient Steering (VGS), a method that dynamically reorients the update vector to prioritize the visual subspace. Experimental results on multiple distillation settings and complex multimodal benchmarks demonstrate that VGS significantly outperforms the standard monolithic formulation of on-policy distillation, achieving superior grounding with minimal training overhead.

Forum: https://openreview.net/forum?id=Jb1YBPqyb7

---

## 141. Uncovering the Latent Potential of Deep Intermediate Representations

**Abstract:** Foundational models pretrained on huge amounts of data learn representations that evolve across depth, forming a hierarchy of embeddings with distinct semantic content and geometric structure. Contrary to the widespread practice of using only the final layer or shallow mixtures, we show that task-relevant information is distributed non-monotonically across layers and cannot be recovered by na\"ive aggregation. Through a geometric and empirical study across multiple modalities, we show that effective transfer depends on identifying which layers encode task-discriminative structure and how their embeddings are geometrically organized. We introduce Layer-wise Optimal Embedding Selection (LOES), a constructive spectral method that identifies task-discriminative subspaces by minimizing residual error under orthogonality and isotropy constraints. To align fine-tuning with this selection principle, we further propose Geometric Regularization (GeoReg), which enforces a simplicial structure on class manifolds and stabilizes representation geometry during fine-tuning. Across a wide range of architectures, depths, modalities, and data regimes, LOES consistently outperforms standard baselines, with gains that grow as model depth increases. Beyond accuracy, our method reveals how semantic factors are distributed across layers, thereby enabling cross-lingual and cross-modal interpretability analyses. Together, our results provide strong evidence that layerwise embedding geometry is not incidental but central to how deep models represent and transfer knowledge.

Forum: https://openreview.net/forum?id=6up1qGJwYZ

---

## 142. Variational Learning for Insertion-based Generation

**Abstract:** Non-monotonic sequence generation methods, such as masked diffusion models, provide a flexible alternative to left-to-right autoregressive modeling by allowing tokens to be generated in non-fixed and prescribed orders. Despite their practical advantages, most existing non-monotonic models are order-agnostic and rely on a fixed-length grid, limiting their ability to support variable-length generation and adaptive insertion order. In this work, we introduce a probabilistic framework for learning insertion order in variable-length insertion models. We formalize a bijective correspondence between insertion trajectories and permutations, which enables an exact reparameterization of the data likelihood as a sum over permutations. Building on this result, we propose the Insertion Process (IP), a stochastic generative model that jointly learns where to insert, what to insert, and when to terminate, trained via permutation-based variational inference. Unlike prior fixed-canvas approaches, IP natively supports variable-length generation and learns data-driven preferences over insertion orders. Experiments on goal-conditioned planning and molecular string generation demonstrate that learning insertion order improves both modeling quality and generalization in domains without a canonical left-to-right structure.

Forum: https://openreview.net/forum?id=thVeEXPaSg

---

## 143. Diffract: Spectral View of LLM Domain Adaptation

**Abstract:** We study continual pre-training (CPT) as a mechanism for adapting general-purpose large language models to specialized domains: mathematics, instruction, code, and natural text. Using singular value decomposition of weight matrices, we find that CPT leaves singular value spectra largely invariant, with adaptation driven mainly by changes in singular vectors. An analysis of attention-head projection matrices reveals strong, domain-dependent **head heterogeneity**, which we exploit to define a head importance criterion: up to **60%** of head updates can be removed without measurable quality loss. Selectively rewinding low-importance heads to their pre-trained state improves benchmark accuracy by up to **4%** versus the fully trained baseline. Finally, we identify **domain connectivity**—linear interpolation between CPT checkpoints yields smooth domain-quality interpolation without notable degradation on either domain—and release Diffract, an open-source toolkit for scalable spectral analysis of billion-parameter models.

Forum: https://openreview.net/forum?id=XBUHoiAGDE

---

## 144. Why Deep Jacobian Spectra Separate: Depth-Induced Scaling and Singular-Vector Alignment

**Abstract:** Understanding why gradient-based training in deep networks exhibits strong implicit bias remains challenging, in part because tractable singular-value dynamics are typically available only for balanced deep linear models. We propose an alternative route based on two theoretically grounded and empirically testable signatures of deep Jacobians: depth-induced exponential scaling of ordered singular values and strong spectral separation. Adopting a fixed-gates view of piecewise-linear networks, where Jacobians reduce to products of masked linear maps within a single activation region, we prove the existence of Lyapunov exponents governing the top singular values at initialization, give closed-form expressions in a tractable masked model, and quantify finite-depth corrections. We further show that sufficiently strong separation forces singular-vector alignment in matrix products, yielding an approximately shared singular basis for intermediate Jacobians. Together, these results motivate an approximation regime in which singular-value dynamics become effectively decoupled, mirroring classical balanced deep-linear analyses without requiring balancing. Experiments in fixed-gates settings validate the predicted scaling, alignment, and resulting dynamics, supporting a mechanistic account of emergent low-rank Jacobian structure as a driver of implicit bias.

Forum: https://openreview.net/forum?id=2kSBDoP1rE

---

## 145. Training-Free Bayesian Filtering with Generative Emulators

**Abstract:** Bayesian filtering is a well-known problem that aims to estimate plausible states of a dynamical system from observations. Among existing approaches to solve this problem, particle filters are theoretically exact for non-linear dynamics and observations, but suffer from poor scalability in high dimensions. In this work, we show that diffusion-based emulators of dynamical systems can be used to implement, without additional training, an optimal variant of particle filters that has remained largely unexplored due to implementation challenges with classical numerical solvers. Experiments on nonlinear chaotic systems, including atmospheric dynamics, demonstrate that the proposed approach successfully scales particle filtering to high-dimensional settings.

Forum: https://openreview.net/forum?id=ibcZNZwKfZ

---

## 146. Evaluating Robustness of Reasoning Models on Parameterized Logical Problems

**Abstract:** Logic provides a controlled testbed for evaluating LLM-based reasoners, yet standard SAT-style benchmarks often conflate surface difficulty (length, wording, clause order) with the structural phenomena that actually determine satisfiability. We introduce a diagnostic benchmark for 2-SAT built from parameterized families of structured 2-CNF formulas, where satisfiability is characterized by the implication graph and can be tuned along interpretable axes. Our generators isolate distinct competencies and failure modes: (i) contradiction-cycle UNSAT cores with controllable size and imbalance, (ii) SAT instances with a prescribed fraction of free variables to control solution multiplicity, (iii) planted backbones that modulate propagation, (iv) late bridge clauses that couple otherwise monotone regions to probe sensitivity to ordering and revision, and (v) symmetry/duplication variants that test abstraction under renaming and redundant structure. We evaluate LLM-based reasoners on decision accuracy and assignment validity, and quantify robustness under semantics-preserving perturbations such as clause reordering, filler clauses, and variable renaming. Across models, we observe sharp performance transitions under targeted structural interventions even when surface statistics are held fixed, revealing brittleness regimes that are invisible to aggregate SAT accuracy.

Forum: https://openreview.net/forum?id=gnLZWOubWa

---

## 147. Adaptive Policy Backbone via Shared Network

**Abstract:** Reinforcement learning (RL) has achieved impressive results across various domains, yet the resulting policies often fail to generalize beyond the specific tasks encountered during training. This lack of robustness limits their deployment in real-world scenarios where diverse and unpredictable task demands exist. In this work, we provide a theoretical analysis of policy networks under Markov Decision Processes (MDPs) and demonstrate that adapting only the linear layers placed before and after a policy backbone is sufficient for task adaptation. Based on this insight, we propose the Adaptive Policy Backbone (APB), which consists of a frozen backbone paired with lightweight, task-specific pre- and post-backbone linear layers. Our results demonstrate that learning only these lightweight task-specific linear layers is sufficient to achieve performance on par with standard RL, even when the backbone is randomly initialized. Furthermore, we find that this structural constraint can enhance the generalization capability of the resulting policies. This advantage extends to out-of-distribution tasks, where representative meta-RL baselines often struggle.

Forum: https://openreview.net/forum?id=KhAUoBv0dJ

---

## 148. Rare Event Analysis of Large Language Models

**Abstract:** Being probabilistic models, during inference large language models (LLMs) display *rare events*: behaviour that is far from typical but highly significant. By definition all rare events are hard to see, but the enormous scale of LLM usage means that events completely unobserved during development are likely to become prominent in deployment. Here we present an end-to-end framework for the systematic analysis of rare events in LLMs. We provide a practical implementation spanning theory, efficient generation strategies, probability estimation and error analysis, which we illustrate with concrete examples. We outline extensions and applications to other models and contexts, highlighting the generality of the concepts and techniques presented here.

Forum: https://openreview.net/forum?id=2RJN5vDHG0

---

## 149. Video-Based Optimal Transport for Feedback-Efficient Offline Preference-Based Reinforcement Learning

**Abstract:** Conveying complex objectives to reinforcement learning (RL) agents often requires meticulous reward engineering. Preference-based RL (PbRL) offers a promising alternative by learning reward functions from human feedback, but its scalability is hindered by high labeling costs. Inspired by advances in Video Foundation Models (ViFMs), we present Video-based Optimal Transport Preference (VOTP), a semi-supervised framework that learns effective reward functions from only a handful of labels. By leveraging optimal transport to align visual trajectories within the rich representation space of ViFMs, VOTP effectively generates high-fidelity pseudo-labels for large amounts of unlabeled data, substantially reducing human supervision. Extensive experiments across locomotion and manipulation benchmarks demonstrate the superiority of VOTP, which outperforms state-of-the-art offline PbRL methods under limited feedback budgets. We also showcase the robustness of VOTP in the presence of visual distractors and validate its utility on real robotic tasks, where it learns meaningful rewards with minimal human input.

Forum: https://openreview.net/forum?id=G8LVO5easu

---

## 150. Stop When Further Reasoning Won’t Help: Attention-State Adaptive Generation in Reasoning Models

**Abstract:** By incorporating test-time compute scaling, large reasoning models (LRMs) can solve complex problems through explicit chain-of-thought (CoT) reasoning processes. 
However, they often suffer from overthinking, resulting in redundant token outputs and degraded accuracy. 
Current methods to mitigate this issue remain limited: training-based approaches require substantial computational resources, while training-free methods rely on well-crafted prompts or unreliable confidence signals.
In this work, we investigate early stopping from the perspective of attention distributions and propose a simple method, ASAG, which infers the model's reasoning state and adaptively adjusts the generation strategy. 
The proposed framework is training-free and plug-and-play, enabling seamless integration into existing LRMs.
Extensive experiments on nine benchmarks demonstrate consistent improvements across mainstream LRMs with varying parameter scales, including the DeepSeek-R1-Distill and Qwen3 series.
Specifically, ASAG improves average accuracy by 3.2% while reducing the number of generated tokens by nearly 40% across all reasoning tasks on Qwen3-8B.

Forum: https://openreview.net/forum?id=zRMK32N6t6

---

## 151. Do LLMs Signal When They’re Right? Evidence from Neuron Agreement

**Abstract:** Large language models (LLMs) commonly boost reasoning via sample-evaluate-ensemble decoders (e.g., majority voting), achieving label free gains without ground truth. However, prevailing strategies score candidates using only external outputs such as token probabilities, entropies, or self evaluations, and these signals can be poorly calibrated after post training. We instead analyze internal behavior based on neuron activations and uncover three findings: (1) external signals are low dimensional projections of richer internal dynamics; (2) correct responses activate substantially fewer unique neurons than incorrect ones throughout generation; and (3) activations from correct responses exhibit stronger cross sample agreement, whereas incorrect ones diverge. Motivated by these observations, we propose Neuron Agreement Decoding (NAD), an unsupervised best of N method that selects candidates using activation sparsity and cross sample neuron agreement, operating solely on internal signals and without requiring comparable textual outputs. NAD enables early correctness prediction within the first 32 generated tokens and supports aggressive early stopping. Across math and science benchmarks with verifiable answers, NAD matches majority voting; on open ended coding benchmarks where majority voting is inapplicable, NAD consistently outperforms Avg@64. By pruning unpromising trajectories early, NAD reduces token usage by 99% with minimal loss in generation quality, showing that internal signals provide reliable, scalable, and efficient guidance for label free ensemble decoding.

Forum: https://openreview.net/forum?id=ZVRgcQq4D2

---

## 152. Generative Modeling of Irregular Time Series via SDE-Induced Continuous-Discrete Variational Inference

**Abstract:** Irregular time series arise ubiquitously in real-world systems, where observations are sparse, asynchronous, and governed by underlying continuous-time dynamics. Existing continuous–discrete state-space models typically rely on path-based variational inference, which is computationally expensive or constrained by restrictive posterior assumptions. We propose SDEVI, a novel framework that performs variational inference directly on the joint distribution over discrete-time observations, while guaranteeing consistency with an underlying continuous process governed by a Stochastic Differential Equation(SDE). SDEVI employs a variational posterior induced by linear time-varying SDEs as a scalable inference backbone. To enable intricate dynamics modeling for real-world data, we introduce non-linear-SDE-induced variational inference and generalize our framework to the complex domain. Extensive experiments across healthcare, physics, climate, and IoT benchmarks demonstrate state-of-the-art performance on interpolation, extrapolation, regression, and classification tasks.

Forum: https://openreview.net/forum?id=IXdjkxDXI0

---

## 153. Stable Deep Reinforcement Learning via Isotropic Gaussian Representations

**Abstract:** Deep reinforcement learning systems often suffer from unstable training dynamics due to non-stationarity, where learning objectives and data distributions evolve over time. We show that under non-stationary targets, isotropic Gaussian embeddings are provably advantageous. In particular, they induce stable tracking of time-varying targets for linear readouts, achieve maximal entropy under a fixed variance budget, and encourage a balanced use of all representational dimensions--all of which enable agents to be more adaptive and stable. Building on this insight, we propose the use of Sketched Isotropic Gaussian Regularization for shaping representations toward an isotropic Gaussian distribution during training. We demonstrate empirically, over a variety of domains, that this simple and computationally inexpensive method improves performance under non-stationarity while reducing representation collapse, neuron dormancy, and training instability.

Forum: https://openreview.net/forum?id=gc7Gg18ejz

---

## 154. Towards High-Fidelity CAD Generation via LLM-Driven Program Generation and Text-Based B-Rep Primitive Grounding

**Abstract:** The field of Computer-Aided Design (CAD) generation has made significant progress in recent years.
Existing methods typically fall into two separate categories: parametric CAD modeling and direct boundary representation (B-Rep) synthesis.
In modern feature-based CAD systems, parametric modeling and B-Rep are inherently intertwined, as advanced parametric operations (e.g., *fillet* and *chamfer*) require explicit selection of B-Rep geometric primitives, and the B-Rep itself is derived from parametric operations.
Consequently, this paradigm gap remains a critical factor limiting AI-driven CAD modeling for complex industrial product design.
This paper presents *FutureCAD*, a novel text-to-CAD framework that leverages large language models (LLMs) and a B-Rep grounding transformer (*BRepGround*) for high-fidelity CAD generation.
Our method generates executable CadQuery scripts, and introduces a text-based query mechanism that enables the LLM to specify geometric selections via natural language, which *BRepGround* then grounds to the target primitives.
To train our framework, we construct a new dataset comprising real-world CAD models.
For the LLM, we apply supervised fine-tuning (SFT) to establish fundamental CAD generation capabilities, followed by reinforcement learning (RL) to improve generalization.
Experiments show that *FutureCAD* achieves state-of-the-art CAD generation performance.
Code and dataset are available at https://github.com/JohanStackk/FutureCAD.

Forum: https://openreview.net/forum?id=jfOhTGs5G5

---

## 155. Scalable Option Learning in High-Throughput Environments

**Abstract:** Hierarchical reinforcement learning (RL) has the potential to enable effective decision-making over long timescales. Existing approaches, while promising, have yet to realize the benefits of large-scale training. In this work, we identify and solve several key challenges in scaling online hierarchical RL to high-throughput environments. We propose Scalable Option Learning (SOL), a highly scalable hierarchical policy gradient algorithm which achieves a ~35x higher throughput compared to existing hierarchical methods. To demonstrate SOL's performance and scalability, we train hierarchical agents using 30 billion frames of experience on the complex game of NetHack, significantly surpassing flat agents and demonstrating positive scaling trends. We also validate SOL on MiniHack and Mujoco environments, showcasing its general applicability.

Forum: https://openreview.net/forum?id=p2RPRTPPMO

---

## 156. Robust Harmful Features Under Jailbreak Attacks: Mechanistic Evidence from Attention Head Specialization in Large Language Models

**Abstract:** Jailbreak attacks bypass LLM safety alignment, yet their mechanisms remain poorly understood. We provide evidence that attacks do not comprehensively eliminate safety features, but instead selectively suppress specific attention heads. We identify two functionally differentiated types: **Adversarially Compromised Heads (ACHs)** concentrated in early layers, which are suppressed under attacks, and **Safety-Aligned Heads (SAHs)** in mid-layers, which maintain robust activations even when attacks succeed. Ablation studies support the causal role of ACHs and the contribution of SAHs to robust activations: suppressing a small number of ACHs is sufficient to induce jailbreak-like behavior on normally refused inputs, while removing SAHs substantially weakens mid-layer safety activations. Token-level attribution further shows that ACH suppression is driven specifically by attack-template tokens, providing a mechanistic account of why attacks can bypass refusal decisions through ACH suppression while leaving internal safety signals sustained by SAHs—a phenomenon we term **Robust Harmful Features**. To validate the practical significance of this robustness, we show that simply reading these persistent activations—without any training—yields competitive aggregate detection performance with strong adversarial robustness.

Forum: https://openreview.net/forum?id=LTF6LtBo0E

---

## 157. daVinci-Dev: Agent-native Mid-training for Software Engineering

**Abstract:** While the emerging field of agentic software engineering has spurred extensive research into post-training, this paradigm alone does not fully address the distribution mismatch between traditional static pre-training and dynamic deployment environments. In this paper, we instead investigate agentic mid-training as a scalable complementary approach.
Central to our approach is *agent-native data* comprising two complementary components: *contextually-native trajectories* that preserve the complete information flow an agent experiences, offering broad coverage and diversity; and *environmentally-native trajectories* whose observations stem from actual tool invocations and test executions, providing interaction authenticity.
On `SWE-Bench Verified`, our recipe outperforms the previous open software engineering mid-training recipe `Kimi-Dev` under two post-training settings with the same base model and agentic scaffold, while using fewer than half mid-training tokens (73.1B).
Furthermore, our 32B and 72B models achieve state-of-the-art resolution rates of **56.1\%** and **58.5\%** among open agentic recipes using agentic scaffolds, despite starting from non-coder `Qwen2.5` base models.
We also observe performance gains on general code generation and scientific benchmarks.
We open-source a significant portion of our datasets, recipes, and model checkpoints to facilitate further research.

Forum: https://openreview.net/forum?id=a86luANykT

---

## 158. Neural Concept Verifier: Scaling Prover-Verifier Games via Concept Encodings

**Abstract:** While *Prover-Verifier Games* (PVGs) offer a promising path toward verifiability in nonlinear classification models, they have not yet been applied to complex inputs such as high-dimensional images. 
Conversely, expressive *concept encodings* effectively allow to translate such data into interpretable concepts but are often utilised in the context of low-capacity linear predictors.
In this work, we push towards real-world verifiability by combining the strengths of both approaches. We introduce *Neural Concept Verifier (NCV)*, a unified framework combining PVGs for formal verifiability with concept encodings to handle complex, high-dimensional inputs in an interpretable way. NCV achieves this by utilizing recent minimally supervised concept discovery models to extract structured concept encodings from raw inputs. A *prover* then selects a subset of these encodings, which a *verifier*, implemented as a nonlinear predictor, uses exclusively for decision-making.
Our evaluations show that NCV outperforms classic concept-based models and pixel-based PVG classifier baselines on high-dimensional, logically complex datasets and helps mitigate shortcut behavior. Overall, we demonstrate NCV as a promising step toward concept-level, verifiable AI.

Forum: https://openreview.net/forum?id=ThxAdzExdS

---

## 159. DOUBT: Decoupled Object-level Understanding and Bridging via vMF-based Trustworthiness for Hallucination Detection in MLLMs

**Abstract:** Multimodal Large Language Models (MLLMs) frequently produce hallucinations (i.e., assertions that contradict the image or facts), undermining reliability in high-risk applications. Existing detection approaches typically feed images and texts jointly and estimate hallucination scores by measuring the consistency of model outputs. However, because the visual module often lags behind the language module in understanding and reasoning, MLLMs can repeatedly produce similar yet incorrect answers, yielding overestimated trustworthiness and missed detections. To address this, we propose a simple yet effective model-agnostic method, dubbed Decoupled Object-level Understanding and Bridging via vMF-based Trustworthiness (DOUBT). DOUBT first employs Object-level Understanding and Bridging (OUB), a two-step prompting scheme that decouples object recognition from relational reasoning by prompting the model to identify objects and then reason based on them. It further introduces a von Mises-Fisher (vMF)-based trustworthiness metric, which is more stable than semantic entropy metrics in small-sample settings. Extensive experiments and ablation studies on multiple benchmarks show that DOUBT consistently outperforms state-of-the-art baselines, demonstrating its robustness and generalizability for hallucination detection in MLLMs. The code is available at https://github.com/XLearning-SCU/2026-ICML-DOUBT.

Forum: https://openreview.net/forum?id=QMf9CBpEIf

---

## 160. Offline Reinforcement Learning of High-Quality Behaviors Under Robust Style Alignment

**Abstract:** We study offline reinforcement learning of style-conditioned policies using explicit style supervision via subtrajectory labeling functions. In this setting, aligning style with high task performance is particularly challenging due to distribution shift and inherent conflicts between style and reward. Existing methods, despite introducing numerous definitions of style, often fail to reconcile these objectives effectively. To address these challenges, we propose a unified definition of behavior style and instantiate it into a practical framework. Building on this, we introduce Style-Conditioned Implicit Q-Learning (SCIQL), which leverages offline goal-conditioned RL techniques, such as hindsight relabeling and value learning, and combine it with a new Gated Advantage Weighted Regression mechanism to efficiently optimize task performance while preserving style alignment. Experiments demonstrate that SCIQL achieves superior performance on both objectives compared to prior offline methods. Code, datasets and visuals are available in: https://mathieu-petitbois.github.io/projects/sciql/.

Forum: https://openreview.net/forum?id=jxdjDmWpNX

---

## 161. FOCUS & RePAIR: Mitigating Text Degeneration via Token-Level Guidance For Pruned Large Language Models

**Abstract:** Pruning is a practical approach to compress large language models (LLMs), but it can amplify text degeneration, especially repetition loops, even when perplexity and task accuracy remain largely unchanged. In this work, we present a token-level analysis of this failure mode by viewing decoding as a dynamical process that enters and persists in a small set of recurrent contexts. Our analysis decomposes degeneration into loop entry risk and loop persistence, and shows that persistence is controlled by the escape mass assigned to plausible alternatives within the token sampling set.
Motivated by these findings, we propose two token-level guidance objectives for post-pruning fine-tuning. FOCUS reweights distillation toward high-confidence teacher regions to suppress leakage, while RePAIR uses onset-centered positive/negative continuation pairs with a margin loss to promote plausible alternatives and prevent early commitment to repetition loops. Experiments on open-ended continuation and instruction-based generation show that both methods consistently reduce repetition and improve generation quality.

Forum: https://openreview.net/forum?id=3SgRoBn3XM

---

## 162. Scalable Event Cloud Network for Event-based Classification

**Abstract:** Event cameras are biologically inspired sensors garnering significant attention from both industry and academia. Mainstream methods favor frame and voxel representations, which reach a satisfactory performance while introducing time-consuming transformations, bulky models, and sacrificing fine-grained temporal information. Alternatively, Point Cloud representation demonstrates promise in addressing the mentioned weaknesses, but it has limited scalability in abstracting features of higher spatial resolution and longer temporal sequence events.  In this paper, we propose a Scalable Network named SECNet to leverage Event Cloud representation. SECNet integrates polarity at the structural level by innovating the Event-based Group and Sampling module rather than only at the input level. To accommodate the surge in the number of events, SECNet embraces feature extraction in the frequency domain via the Fourier transform. This approach not only substantially extinguishes the explosion of Multiply Accumulate Operations but also effectively abstracts spatio-temporal features. We conducted extensive experiments on ten event-based datasets, and substantiate the scalability, effectiveness, and efficiency of SECNet. Our code will be available at: https://github.com/rhwxmx/SECNet_ICML.

Forum: https://openreview.net/forum?id=yAAUcDLYMR

---

## 163. Distribution Transformers: Fast Approximate Bayesian Inference With On-The-Fly Prior Adaptation

**Abstract:** While Bayesian inference provides a principled framework for reasoning under uncertainty, its widespread adoption is limited by the intractability of exact posterior computation, necessitating the use of approximate inference. However, existing methods are often computationally expensive, or demand costly retraining when priors change, limiting their utility, particularly  in sequential inference problems such as real-time sensor fusion. To address these challenges, we introduce the Distribution Transformer---a novel architecture that can learn arbitrary distribution-to-distribution mappings. Our method can be trained to map a prior to the corresponding posterior, conditioned on some dataset---thus performing approximate Bayesian inference. Our novel architecture represents a prior distribution as a (universally-approximating) Gaussian Mixture Model (GMM), and transforms it into a GMM representation of the posterior. The components of the GMM attend to each other via self-attention, and to the datapoints via cross-attention. We demonstrate that Distribution Transformers both maintain flexibility to vary the prior, and significantly reduces computation times---from minutes to milliseconds---while achieving expected log-likelihood performance on par with or superior to existing approximate inference methods across tasks such as sequential inference, quantum system parameter inference, and Gaussian Process predictive posterior inference with hyperpriors.

Forum: https://openreview.net/forum?id=bMHwh8qAGc

---

## 164. LiftQuant: Continuous Bit-Width LLM via Dimensional Lifting and Projection

**Abstract:** Existing quantization methods are fundamentally limited by rigid, integer-based bit-widths (e.g., 2, 3-bit), resulting in a "deployment gap" where Large Language Models cannot be optimally fitted to specific memory budgets. To bridge this gap, we introduce LiftQuant, a novel framework that enables continuous bit-width control for true Pareto-optimal deployment.  The core innovation is a "lift-then-project" mechanism which approximates low-dimensional weight vectors by projecting a simple 1-bit lattice from a higher-dimensional ``lifted" space.  Crucially, the effective bit-width is determined simply by the ratio of the lifted dimension to the original dimension, which allows the bit-width to be tuned quasi-continuous as the dimension is a flexible structural parameter. This projection generates a structured yet non-uniform codebook, capturing the expressive power of Vector Quantization (VQ).  While beneficial over VQ, LiftQuant's decoding path relies solely on linear transformations and 1-bit uniform quantizers, retaining hardware-friendly nature.  This flexibility is transformative: LiftQuant enables a 70B LLM to be compressed to 2.4 bits to precisely fit a 24GB GPU, where its performance significantly surpasses state-of-the-art 2-bit models fitted on the same device.  Our code and ckpt is available at \url{https://github.com/Heliulu/LiftQuant}.

Forum: https://openreview.net/forum?id=1GvXUhLIMP

---

## 165. The Assistant Axis: Situating and Stabilizing the Default Persona of Language Models

**Abstract:** Large language models can represent a variety of personas but typically default to a helpful Assistant identity cultivated during post-training. Across several different models, we find an “Assistant Axis" in their activation space, which captures the extent to which a model is operating in its default Assistant mode. Steering towards the Assistant direction reinforces helpful and harmless behavior; steering away increases the model’s tendency to identify as other entities. Measuring deviations along the Assistant Axis predicts “persona drift,” a phenomenon where models slip into exhibiting harmful or bizarre behaviors that are uncharacteristic of their typical persona. We find that persona drift is often driven by conversations demanding meta-reflection on the model’s processes or featuring emotionally vulnerable users. We show that restricting activations to a fixed region along the Assistant Axis can stabilize model behavior in these scenarios—and also in the face of adversarial persona-based jailbreaks. Our results suggest that post-training steers models toward a particular region of persona space but only loosely tethers them to it, motivating work on training and steering strategies that more deeply anchor models to a coherent persona.

Forum: https://openreview.net/forum?id=q17jVxMtwS

---

## 166. Jailbreak Foundry: From Papers to Runnable Attacks for Reproducible Benchmarking

**Abstract:** Jailbreak techniques for large language models (LLMs) evolve faster than benchmarks, making robustness estimates stale and difficult to compare across papers due to drift in datasets, harnesses, and judging protocols. We introduce **JAILBREAK FOUNDRY (JBF)**, a system that addresses this gap via a multi-agent workflow to translate jailbreak papers into executable modules for immediate evaluation within a unified harness. JBF features three core components: (i) *JBF-LIB* for shared contracts and reusable utilities; (ii) *JBF-FORGE* for the multi-agent paper-to-module translation; and (iii) *JBF-EVAL* for standardizing evaluations. Across 30 reproduced attacks, JBF achieves high fidelity with a mean (reproduced$-$reported) attack success rate (ASR) deviation of $+0.26$ percentage points. By leveraging shared infrastructure, JBF reduces attack-specific implementation code by nearly half relative to original repositories and achieves an 82.5% mean reused-code ratio. This system enables a standardized AdvBench evaluation of all 30 attacks across 10 victim models using a consistent GPT-4o judge. By automating both attack integration and standardized evaluation, JBF offers a scalable solution for creating living benchmarks that keep pace with the rapidly shifting security landscape.

Forum: https://openreview.net/forum?id=BSi2mfMDsx

---

## 167. On Minimum Depth and Width of Floating-Point Neural Networks for Representing Floating-Point Functions

**Abstract:** Research on the expressive power of neural networks has identified the minimum depth and width of neural networks that enable universal approximation and memorization. However, existing results are derived under exact arithmetic and cannot be directly applied to real implementations on computers, which can only use a finite set of numbers and inexact machine operations with round-off errors. 
In this work, we study floating-point ReLU networks that have floating-point parameters and use floating-point operations. Specifically, we investigate their minimum depth and width to represent all functions from the set of floating-point vectors $\mathbb F^d$ to the set of floating-point numbers $\mathbb F$. We first show that the minimum depth for representing all functions from $\mathbb F^d$ to $\mathbb F$ is exactly three, where two layers can be sufficient if we consider a smaller domain and/or codomain. We further show that the minimum width for representing all functions from $\mathbb F^d$ to $\mathbb F$ lies between $2d$ and $2d+4$. In addition, if we restrict the domain to non-negative floats, it lies between $d$ and $d+4$, where it can be smaller for a smaller domain, even beyond $d$.  Our results show that the existing results analyzed under exact arithmetic do not extend to the floating-point setup.

Forum: https://openreview.net/forum?id=8IxY2DUiBi

---

## 168. Harnessing Non-Adversarial Robustness in Large Language Models

**Abstract:** The work presents an approach for addressing the challenge of robustness in Large Language Models (LLMs) to alterations and potential errors caused by semantically similar but textually different prompts. Recent works have shown that these kinds of prompt variations can significantly impact the performance of LLMs on tasks. The central question is: can LLMs' robustness to semantically-neutral prompt alterations be acquired without expensive retraining of the entire model? We address this question both theoretically and through experiments. Our theoretical analysis reveals a crucial factor impacting model robustness -- a systematic expected shift or perturbation-induced bias in neural network module outputs. Motivated by this analysis, we show that robustness can be achieved via a simple fine-tuning process: debiasing for robustness. We identify conditions when debiasing helps and when it does not, and demonstrate, through both theory and extensive experiments, that debiasing for robustness may indeed be a quick and efficient tool to enhance robustness and provide certification against random prompt perturbations.

Forum: https://openreview.net/forum?id=7SKS1u6vP1

---

## 169. Language Generation in the Limit: Complexity Barriers and Implications for Learning

**Abstract:** Kleinberg and Mullainathan showed that language generation in the limit is always possible
at the level of computability: given enough positive examples, a learner can eventually
generate data indistinguishable from a target language. However, such existence results do
not address feasibility.
We study the sample complexity of language generation in the limit for several canonical
classes of formal languages. Our results show that infeasibility already appears for
context-free and regular languages, and persists even for strict subclasses such as locally threshold
testable languages, as well as for incomparable classes such as non-erasing pattern
languages, a well-studied class in the theory of language identification.
Overall, our results establish a clear gap between the theoretical possibility of
language generation in the limit and its computational feasibility.

Forum: https://openreview.net/forum?id=3I6v86xf2l

---

## 170. VenusBench-Mobile: A Challenging and User-Centric Benchmark for Mobile GUI Agents with Capability Diagnostics

**Abstract:** Existing online benchmarks for mobile GUI agents remain largely app-centric and task-homogeneous, failing to reflect the diversity and instability of real-world mobile usage. To this end, we introduce VenusBench-Mobile, a challenging online benchmark for evaluating general-purpose mobile GUI agents under realistic, user-centric conditions. 
VenusBench-Mobile builds two core evaluation pillars: defining what to evaluate via user-intent-driven task design that reflects real mobile usage, and how to evaluate through a capability-oriented annotation scheme for fine-grained agent behavior analysis.
Extensive evaluation of state-of-the-art mobile GUI agents reveals large performance gaps relative to prior benchmarks, indicating that VenusBench-Mobile poses substantially more challenging and realistic tasks and that current agents remain far from reliable real-world deployment.
Diagnostic analysis further shows that failures are dominated by deficiencies in perception and memory, which are largely obscured by coarse-grained evaluations.
Moreover, even the strongest agents exhibit near-zero success under environment variations, highlighting their brittleness in realistic settings.
Based on these insights, we believe VenusBench-Mobile provides an important stepping stone toward robust real-world deployment of mobile GUI agents. Code and data are available at https://github.com/inclusionAI/UI-Venus/tree/VenusBench-Mobile.

Forum: https://openreview.net/forum?id=coHiGZOFtS

---

## 171. Concept Removal Guidance: Evidence-Calibrated Negative Guidance for Safe Diffusion Sampling

**Abstract:** Text-to-image diffusion models remain vulnerable to adversarial prompts that elicit disallowed content, motivating reliable inference-time controls. A popular approach is negative guidance, which subtracts a negative prompt direction with a fixed weight. However, it often forces a safety–fidelity trade-off, causing artifacts or prompt drift when over-applied and failing under attacks when under-applied. Dynamic variants reweight guidance using posterior-odds signals, which can be brittle for open-vocabulary compositional prompts, while lightweight similarity-based methods ignore the evolving image evidence along the denoising trajectory. We introduce Concept Removal Guidance (CRG), a training-free method that estimates unwanted-concept presence at each diffusion step from the model's noise predictions, and adaptively calibrates negative guidance via a closed-form constrained update enforcing a target presence threshold while minimally perturbing the conditional trajectory. Across red-teaming benchmarks, CRG reduces attack success rates while preserving benign fidelity, and extends to additional suppression targets such as artist style and violence without fine-tuning or external classifiers.

Forum: https://openreview.net/forum?id=BSDvEh7oqy

---

## 172. Exact Functional ANOVA Decomposition for Categorical Inputs Models

**Abstract:** Functional ANOVA offers a principled framework for interpretability by decomposing a model’s prediction into main effects and higher-order interactions. For independent features, this decomposition is well-defined, strongly linked with SHAP values, and serves as a cornerstone of additive explainability. However, the lack of an explicit closed-form expression for general dependent distributions has forced practitioners to rely on costly sampling-based approximations. We completely resolve this limitation for categorical inputs. By bridging functional analysis with the extension of discrete Fourier analysis, we derive a closed-form decomposition without any assumption. Our formulation is computationally very efficient. It seamlessly recovers the classical independent case and extends to arbitrary dependence structures, including distributions with non-rectangular support. Furthermore, leveraging the intrinsic link between SHAP and ANOVA under independence, our framework yields a natural generalization of SHAP values for the general categorical setting.

Forum: https://openreview.net/forum?id=qC9FEfYjai

---

## 173. A Unifying Relational Perspective on Expressive Lottery Tickets

**Abstract:** Graph neural networks (GNNs) are widely used, but how parameter sparsity affects the expressivity of relational (RGNNs) and temporal (TGNNs) variants is poorly understood. The Strong Expressive Lottery Ticket Hypothesis (SELTH) posits the existence of sparse GNNs that preserve Weisfeiler-Leman (WL) expressivity on static graphs. We generalize this existence result to a probabilistic statement for multi-relational and temporal domains via the relational WL (RWL). We prove that sufficiently parameterized RGNNs contain sparse subnetworks that maintain 1-RWL expressivity and derive a lower bound on the probability that a random pruning yields such a subnetwork. We show that common TGNNs and cross-graph message passing schemes admit RGNN reformulations such that they inherit these guarantees and, moreover, that the expressivity of a sparse RGNN is connected to its optimization behavior under common update regimes. Experiments instantiate the bound, compare it to empirical probabilities on synthetic data, and study how pre-training expressivity relates to optimization and prediction quality metrics on temporal and molecular benchmarks.

Forum: https://openreview.net/forum?id=uWHqeVzNcm

---

## 174. MIRA: A Score for Conditional Distribution Accuracy and Model Comparison

**Abstract:** We introduce MIRA, a sample-based score for assessing the accuracy of a candidate conditional distribution using only joint samples from the true data-generating process. Relying on the principle that distributions coincide if they assign equal probability mass to all regions, we derive an analytic expression for the MIRA statistic, whose average defines the MIRA score. This formulation further allows us to compute theoretical reference values and uncertainty estimates when the candidate distribution matches the true one. This framework enables model comparison by quantifying the alignment between the conditional distribution of a candidate model and the true data generating process. Consequently, MIRA enables Bayesian model comparison through direct posterior validation, bypassing the challenging evidence computation. We demonstrate its effectiveness across several toy problems and Bayesian inference tasks.

Forum: https://openreview.net/forum?id=ra2t1V4nml

---

## 175. ADEPT: RL-Aligned Agentic Decoding of Emotion via Evidence Probing Tools — From Consensus Learning to Ambiguity-Driven Emotion Reasoning

**Abstract:** Speech Large Language Models (SLLMs) enable high-level emotion reasoning, but often produce ungrounded, text-biased judgments without verifiable acoustic evidence. In contrast, SSL encoders such as WavLM yield strong acoustic representations yet remain opaque discriminative models that offer limited interpretability. To bridge this gap, we introduce the Agentic Decoding of Emotion via Probing Tools (ADEPT) framework, which reframes emotion recognition as a multi-turn inquiry process rather than a single-pass prediction. ADEPT transforms an SLLM into an agent that maintains an evolving candidate set and adaptively invokes dedicated semantic and acoustic probing tools within a structured pipeline of candidate generation, evidence collection, and adjudication. Crucially, ADEPT enables a paradigm shift from consensus learning to ambiguity-driven emotion reasoning. Since human affect exhibits complexity and co-occurrence of emotions, we leverage minority annotations as informative signals instead of discarding them as noise. Finally, we integrate Group Relative Policy Optimization (GRPO) with the Evidence Trust Gate to explicitly couple tool-usage behaviors with prediction quality and enforce evidence-based reasoning. Experiments demonstrate that ADEPT improves in most cases the primary emotion accuracy while substantially improving minor emotion characterization, producing explanations grounded in auditable evidence.

Forum: https://openreview.net/forum?id=maA1s1S0db

---

## 176. Improving the Performance and Learning Stability of Parallelizable RNNs Designed for Ultra-Low Power Applications

**Abstract:** Sequence learning is dominated by Transformers and parallelizable recurrent neural networks (RNNs) such as state-space models, yet learning long-term dependencies remains challenging, and state-of-the-art designs trade power consumption for performance. The Bistable Memory Recurrent Unit (BMRU) was introduced to enable hardware-software co-design of ultra-low power RNNs: quantized states with hysteresis provide persistent memory while mapping directly to analog primitives. However, BMRU performance lags behind parallelizable RNNs on complex sequential tasks. In this paper, we identify gradient blocking during state updates as a key limitation and propose a cumulative update formulation that restores gradient flow while preserving persistent memory, creating skip-connections through time. This leads to the Cumulative Memory Recurrent Unit (CMRU) and its relaxed variant, the $\alpha$CMRU. Experiments show that the cumulative formulation dramatically improves convergence stability and reduces initialization sensitivity. The CMRU and $\alpha$CMRU match or outperform Linear Recurrent Units (LRUs) and minimal Gated Recurrent Units (minGRUs) across diverse benchmarks at small model sizes, with particular advantages on tasks requiring discrete long-range retention, while the CMRU retains quantized states, persistent memory, and noise-resilient dynamics essential for analog implementation.

Forum: https://openreview.net/forum?id=t2bQt6M7Qh

---

## 177. Hista and Numca: Estimate State Value Effectively for Large Language Model Reinforcement Learning

**Abstract:** Reinforcement learning (RL) refines large language models (LLMs) by directly optimizing model behavior through reward signals. While accurate state value estimation is critical for stable training in classical RL, it remains an underexplored challenge in LLM post‑training. In this work, we introduce the State Value Estimation Benchmark (SVEB) to assess state estimation within existing RL frameworks and show that critics in standard approaches like PPO collapse to a coarse group‑average baseline. To address this, we propose two techniques: \textit{Numca}, which leverages numerical spans as gradable milestones for state value estimation, and \textit{Hista}, a framework that uses LLM's hidden states as representation to weighted average disjoint rollouts and their return. Extensive experiments demonstrate that both methods yield more accurate state value estimates and enhance training performance across different RL algorithms and model sizes without incurring significant computational overhead. Code available at \url{https://github.com/VOXXXX1874/Hista}.

Forum: https://openreview.net/forum?id=ff4b0ELdU0

---

## 178. One Intervention per Component is Enough: Towards Identifiability in Linear Stochastic Dynamics from Steady State

**Abstract:** We study the problem of recovering the parameters of a multivariate Ornstein–Uhlenbeck (OU) process from steady-state observational and interventional data. In many applications, such as large-scale gene perturbation experiments, only stationary “snapshot” measurements are available, making standard stochastic differential equation estimation methods that rely on time-series trajectories inapplicable. We first establish an identifiability result: one intervention per strongly connected component (SCC) of the drift graph suffices to recover all OU process parameters generically up to a global scaling factor. This holds provided that the SCC condensation graph is connected with a single root and certain spectral nondegeneracy assumptions hold. We propose a recursive learning algorithm that orders SCCs topologically and, for each component, isolates its marginal dynamics and solves a linear system derived from the steady-state moment equations, leveraging parameters recovered for upstream components. Building on this theoretical foundation, we propose a regularized least-squares estimator that jointly minimizes residuals of the steady-state mean and covariance equations across observational and interventional data. Experiments on synthetic and real datasets demonstrate the effectiveness of our method in recovering parameters and predicting unseen interventions.

Forum: https://openreview.net/forum?id=g07aDcWYJ9

---

## 179. Train for Truth, Keep the Skills: Binary Retrieval-Augmented Reward Mitigates Hallucinations

**Abstract:** Modern post-trained language models are increasingly capable, but remain prone to extrinsic hallucinations.
We target the utility degradation issue that prior hallucination-reduction methods often struggle to avoid, and propose online RL with Binary Retrieval-Augmented Reward (Binary RAR) to reduce hallucinations while preserving general capabilities.
Binary RAR assigns a reward of 1 if a response contains no factual contradictions with retrieved evidence, and 0 otherwise. 
We theoretically show that this method reduces the probability of error-containing responses while preserving the distribution of error-free responses. This helps preserve the model’s capabilities, whereas other methods often degrade them.
We evaluate Binary RAR on multiple widely used models. On Qwen3-8B, it reduces long-form hallucination rates by 39.3\% and short-form hallucination rates by 54.4\%, outperforming supervised learning and preference optimization baselines.
Our error analysis shows that continuous factuality rewards (e.g., VeriScore) can be exploited via reward hacking by producing fewer or more generic claims, whereas Binary RAR is more robust and better preserves general capabilities, including instruction following, math, and coding.

Forum: https://openreview.net/forum?id=BNCNSgLPy5

---

## 180. ConFlux: Multivariate Time Series in Flux, One Unified Forecast in Confluence

**Abstract:** Real-world multivariate time series are inherently in flux: different variables evolve asynchronously and interact in complex, time-varying ways, yet accurate forecasting requires these dispersed signals to converge into a single unified prediction.
This structural mismatch between dynamic, heterogeneous inputs and a unified forecasting objective poses a fundamental challenge for building general-purpose multivariate forecasting models, especially in zero-shot and large-scale settings. 
To this end, inspired by the idea that ``\emph{all rivers run into the sea}'', we propose \textbf{ConFlux}, a \emph{general-purpose foundation model for multivariate time-series forecasting} by learning to adaptively integrate cross-channel information under a unified forecasting objective. Specifically, ConFlux first reorders variables to reduce cross-variable entanglement, then aggregates adjacent variables into compact patches that can be processed by a Vision Transformer-style architecture. This design shortens the effective context, reduces attention complexity, and provides a unified token representation for pre-training and downstream tasks.
Experiments on 25 public datasets show that ConFlux achieves state-of-the-art performance in zero-shot, fine-tuning, and from-scratch settings, while offering faster inference and lower memory usage.

Forum: https://openreview.net/forum?id=qugRvYjaFx

---

## 181. Alignment-Guided Score Matching for Text-to-Image Alignment in Diffusion Models

**Abstract:** Diffusion models generate highly realistic images but often struggle with precise text–image alignment. While recent post-training methods improve alignment using external rewards or human preference signals, their performance heavily depends on reward quality and does not directly address alignment within the diffusion process itself.
Recent reward-free approaches such as SoftREPA demonstrate that optimizing soft text tokens via contrastive learning can effectively improve text-image representation alignment, outperforming standard parameter-efficient fine-tuning baselines. However, the contrastive formulation can excessively penalize negative pairs, which manifests as characteristic failure cases such as over-counting and
repetition.
To address this issue, we propose a lightweight, reward-free post-training method that refines soft tokens by integrating contrastive alignment guidance directly into the score-matching objective of diffusion models. By assigning alignment directions at the score level, our approach mitigates these limitations and yields more coherent and semantically faithful generations.
Experiments show that our method matches SoftREPA while substantially improving its failure cases, achieving over 35\% improvement in counting accuracy on the GenEval benchmark. Our method is seamlessly applicable to existing diffusion backbones (SD1.5, SDXL, and SD3), and is complementary to existing RL-based diffusion post-training methods.

Forum: https://openreview.net/forum?id=vKWxArobP3

---

## 182. Towards Optimal Robustness in Learning-Augmented Paging

**Abstract:** Learning-augmented paging has been extensively studied in recent years. A key advantage over naive ML-based approaches is bounded robustness, which guarantees worst-case performance even when predictions are inaccurate, making these algorithms valuable for real-world systems. Prior work achieves robustness bounds of $2H_k + O(1)$ in the randomized setting, leaving a gap to the optimal competitive ratio $H_k$.

In this paper, we study how to close this gap. We begin by reviewing online optimality and proving a new property of the latest $H_k$-competitive algorithm, which facilitates our analysis in the learning-augmented setting. Then, we review existing learning-augmented paging algorithms and introduce a unifying primitive, the relative prediction budget, which captures the essence of establishing robustness and reveals that prior algorithms either overuse or underutilize predictions. Guided by the above analysis, we develop a new framework that achieves the best-possible robustness up to an additive constant for learning-augmented paging: $H_k + O(1)$. Experiments further demonstrate strong practical performance.

Forum: https://openreview.net/forum?id=ESa07RwpVr

---

## 183. Accelerating Q-learning through Efficient Value-Sharing across Actions

**Abstract:** Action-values are foundational to many control algorithms such as Q-learning. Therefore learning action-values efficiently is central to reinforcement learning (RL).  However, learning them can be slow, requiring many updates to move values from their initialization, typically near zero, to their true values, which may be far from zero. Moreover, action-value learning algorithms typically update each state–action pair independently, without learning shared value structure across actions within a state. In this paper, we address these inefficiencies by introducing the mean-expansion layer, which accelerates action-value learning by sharing values across actions within a state and by changing the problem from directly learning potentially large action-values to learning a lower-norm representation of them. In deep RL, this layer can be applied as a parameter-free addition to Q-network architectures without altering the underlying algorithm. Applied to deep Q-networks and implicit quantile networks, it improves aggregate performance across 57 Atari games while increasing action gaps and dramatically reducing value overestimation.

Forum: https://openreview.net/forum?id=bQXQa5NDpH

---

## 184. Catch-22: On the Fundamental Tradeoff Between Detectability and Robustness in LLM Watermarking

**Abstract:** Large language models generate text by sampling tokens at random, a process now widely used for inference-time watermarking that verifies AI-generated content. We present an information-theoretic framework that captures the trade-off between robustness to text edits and detectability by observers who lack the watermark key or a keyless detector. The bounds we derive hold regardless of computational power, and what a keyless detector can actually achieve depends on what it can observe about the model and its outputs. At the heart of the analysis is an additive, Kullback-Leibler (KL) information measure that quantifies how well a hypothesis test can distinguish watermarked from unwatermarked text while the watermark remains stealthy. The measure remains zero for distribution-preserving schemes and increases with text length for token-level and sentence-level probability-modifying schemes. When edits are modeled as noise, the KL measure shrinks quadratically with the edit rate for token-level schemes and with an induced semantic flip rate for sentence-level schemes. This shrinkage exposes an unavoidable trilemma among robustness, stealth, and reliable verification. Guided by these limits, we propose a hybrid watermarking strategy that selects the Pareto-optimal scheme among distribution-preserving, semantic-level, and token-level methods based on the expected editing regime at deployment. Experiments on Llama-2-7B and Mistral-7B under paraphrasing attacks corroborate these theoretical predictions and confirm that the hybrid strategy lies near the Pareto frontier across the edit regimes we evaluate.

Forum: https://openreview.net/forum?id=097IlaUfQY

---

## 185. PonderLM-2: Pretraining LLM with Latent Thoughts in Continuous Space

**Abstract:** The remarkable success of Chain-of-Thought (CoT), which enhances performance by scaling generation steps at test-time, inspires us to ask: can we leverage a similar scaling of computational steps during pretraining to improve the generation of each individual token? To address this, we propose a novel pre-training methodology: Pretraining Language Models with Latent Thoughts (PonderLM-2). Our approach pretrains a language model (LM) to first generate an intermediate latent thought—the last hidden
state of the current position—which is then used as input to predict the actual subsequent token. This additional computational step enables the LM to refine its prediction within unconstrained continuous space. Our experiments demonstrate that, at an identical inference cost, a LM that generates one additional latent thought per token outperforms a standard model with double the parameters. For instance, our PonderLM-2-Pythia-1.4B, pretrained on 300B tokens from the Pile, significantly surpasses the vanilla Pythia-2.8B trained on the same data on both language modeling and a range of general downstream tasks. Furthermore, increasing the number of latent thoughts generated before each actual token—forming a chain analogous to CoT—consistently improves the model's performance. The code and models are available at https://github.com/LUMIA-Group/PonderLM-2.

Forum: https://openreview.net/forum?id=yVFxjNzCQm

---

## 186. Any-Order GPT as Masked Diffusion Model: Decoupling Formulation and Architecture

**Abstract:** Efficiently scaling Large Language Models (LLMs) necessitates exploring alternatives to dominant autoregressive (AR) methods, with Masked Diffusion Models (MDMs) emerging as candidates. However, comparing AR (typically decoder-only) and MDM (often encoder-only) paradigms is confounded by differing architectures, obscuring true algorithmic and efficiency trade-offs. This research decouples these factors by evaluating MDMs within a decoder-only framework to: (1) Equitably compare MDM (as Any-Order AR) and standard AR paradigms through discrepancies on orders. (2) Investigate MDM architectural impacts on computational efficiency. We show decoder-only MDMs, despite a larger modeling space, can achieve significant inference speedups ($\sim25\times$) and comparable perplexity with techniques like temperature annealing, offering a path to reduced inference compute. This work provides insights for developing more computationally efficient foundation models by disentangling core modeling choices from architectural influences. Code is available at \url{https://github.com/scxue/AO-GPT-MDM}.

Forum: https://openreview.net/forum?id=sEYoG3tAXN

---

## 187. Hallucination is a Consequence of Space-Optimality: A Rate-Distortion Theorem for Membership Testing

**Abstract:** Large language models often hallucinate with high confidence on "random facts" that lack inferable patterns. 
We formalize the memorization of such facts as a membership testing problem, unifying the discrete error metrics of Bloom filters with the continuous log-loss of LLMs. 
By analyzing this problem in the regime where facts are sparse in the universe of plausible claims, we establish a rate-distortion theorem: the optimal memory efficiency is characterized by the minimum KL divergence between score distributions on facts and non-facts. 
This theoretical framework provides a distinctive explanation for hallucination under an idealized setting: even with optimal training, perfect data, and a simplified ``closed world'' setting, the information-theoretically optimal strategy under limited capacity is not to abstain or forget, but to assign high confidence to some non-facts, resulting in hallucination. 
We validate this theory empirically on both synthetic and real-world data, showing that hallucinations persist as a natural consequence of lossy compression.
The same theorem recovers and sharpens classical space lower bounds for Bloom-type filters, pinning down an additive constant left open for two-sided filters.

Forum: https://openreview.net/forum?id=uuD1rE5KU5

---

## 188. DGS-Net: Distillation-Guided Gradient Surgery for CLIP Fine-Tuning in AI-Generated Image Detection

**Abstract:** The rapid progress of generative models such as GANs and diffusion models has led to the widespread proliferation of AI-generated images, raising concerns about misinformation, privacy violations, and trust erosion in digital media. Although large-scale multimodal models like CLIP offer strong transferable representations for detecting synthetic content, fine-tuning them often induces catastrophic forgetting, which degrades pre-trained priors and limits cross-domain generalization. To address this issue, we propose the Distillation-guided Gradient Surgery Network (DGS-Net), a novel framework that preserves transferable pre-trained priors while suppressing task-irrelevant components. Specifically, we introduce a gradient-space decomposition that separates harmful and beneficial descent directions during optimization. By projecting task gradients onto the orthogonal complement of harmful directions and aligning with beneficial ones distilled from a frozen CLIP encoder, DGS-Net achieves unified optimization of prior preservation and irrelevant suppression. Extensive experiments on 50 generative models demonstrate that our method outperforms state-of-the-art approaches by an average margin of 6.6%, achieving superior detection performance and generalization across diverse generation techniques.

Forum: https://openreview.net/forum?id=G5XGej7wNt

---

## 189. Mixtures Closest To A Given Measure: A Semidefinite Programming Approach

**Abstract:** Mixture models, such as Gaussian mixture models (GMMs), are widely used in machine learning to represent complex data distributions. A key challenge, especially in high-dimensional settings, is to determine the mixture order and estimate the mixture parameters. We study the problem of approximating a target measure, available only through finitely many of its moments, by a mixture of distributions from a parametric family (e.g., Gaussian, exponential, Poisson), with approximation quality measured by the 2-Wasserstein ($\operatorname{W_2}$) or the total variation ($\operatorname{TV}$) distance. Unlike many existing approaches, the parameter set is not assumed to be finite; it is modeled as a compact basic semi-algebraic set. We introduce a hierarchy of semidefinite relaxations with asymptotic convergence to the desired optimal value. In addition, when a certain rank condition is satisfied, the convergence is even finite and recovery of an optimal mixing measure is obtained. We also present an application to clustering, where our framework serves either as a stand-alone method or as a preprocessing step that yields both the number of clusters and strong initial parameter estimates, thereby accelerating convergence of standard (local) clustering algorithms

Forum: https://openreview.net/forum?id=0JDzkrKzaA

---

## 190. Fast Spectrally Sparse Signal Reconstruction via Jacobi-Preconditioned Gradient Descent

**Abstract:** Spectrally sparse signal reconstruction arises in a wide range of applications and can be formulated as a low-rank Hankel matrix completion problem. We develop a Jacobi-preconditioned gradient descent method that preserves the low per-iteration complexity of first-order algorithms while achieving linear convergence at a rate independent of the condition number. By introducing a generator that maps factor-based iterates to matrix space, we establish equivalence with manifold-based methods, enabling direct convergence analysis while avoiding the need to define distances under complex-symmetric factorization ambiguity. Extensive experiments demonstrate that the proposed algorithm outperforms state-of-the-art methods in both iteration count and computational time across a broad range of problem settings.

Forum: https://openreview.net/forum?id=b4HR1jhoRV

---

## 191. Non-Euclidean Gradient Descent Operates at the Edge of Stability

**Abstract:** The Edge of Stability (EoS) is a phenomenon where the sharpness (largest eigenvalue) of the Hessian approaches and then hovers near the stability threshold $2/\eta$ during gradient descent (GD) with step size $\eta$. Despite (apparently) violating classical smoothness assumptions, EoS has been widely observed in deep learning, but its theoretical foundations remain incomplete. We provide an interpretation of EoS through the lens of Directional Smoothness [Mishkin et al., 2024]. This interpretation naturally extends to non-Euclidean norms, which we use to define generalized sharpness under an arbitrary norm.  Our generalized sharpness measure includes previously studied vanilla GD and preconditioned GD as special cases, as well as methods for which EoS has not been studied, such as  $\ell_{\infty}$-descent, Block CD, Spectral GD, and their normalized versions. Through experiments on neural networks, we show that non-Euclidean GD with our generalized sharpness also exhibits progressive sharpening followed by oscillations around or above the threshold $2/\eta$. Practically, our framework provides a geometry-aware spectral diagnostic that can be applied across a broad class of non-Euclidean gradient methods.

Forum: https://openreview.net/forum?id=piWlEHb4Db

---

## 192. UniMapping: Unified SLAM Framework for Map-Centric Embodied Perception

**Abstract:** Simultaneous Localization and Mapping (SLAM) is increasingly expected to provide reusable spatial representations for downstream perception. However, existing approaches often struggle with scale-consistency and producing maps that lack the geometric fidelity required for reliable perception. We propose _UniMapping_, a unified SLAM framework that constructs a persistent neural-descriptor map from multimodal observations. We introduce a **Spatial-Aware Deformable Transformer** that injects explicit geometric inductive bias to ensure scale-invariant feature extraction, alongside a **Spatial Fusion** strategy that decouples feature aggregation from temporal sequences. Extensive experiments on both indoor and outdoor benchmarks demonstrate competitive SLAM performance. Notably, our method significantly enhances downstream tasks (mAP +3.1% and mIoU +7.1%) by leveraging accumulated multi-view context.

Forum: https://openreview.net/forum?id=bQkKVGuHZA

---

## 193. World-Model Inspired Emotion-aware Token Refinement for Training-Free Multimodal Emotion Recognition

**Abstract:** Multimodal Large Language Models (MLLMs) show promise for Multimodal Emotion Recognition (MER) but often remain unreliable because sparse emotional cues could be easily overwhelmed and affected by redundant context. While fine-tuning is effective, it is usually costly when using large models. Training-free methods like chain-of-thought reasoning provide a practical alternative, but they mostly rely on heuristic prompting to influence the model behaviors and do not explicitly focus on emotion relevant tokens internally, which would allow decision-relevant emotional tokens to be diluted by environmental noise, resulting in unstable predictions. To address this limitation without training, we rethink MER from a world-model perspective that treats emotion as a latent state inferred from noisy and redundant multimodal observations. Under frozen parameters, this view suggests that robustness depends on constraining why and how tokens contribute to inference.  Based on this insight, we propose WETR (World-Model inspired Emotion-aware Token Refinement), a training-free, plug-and-play regulator that reshapes token usage through two mechanisms: Noise-suppressed Token Selection (NTS), which suppresses redundant intra-modal noise, and State-strengthened Token Reweighting (STR), which amplifies decision-relevant emotional tokens. Experiments on multiple MER benchmarks demonstrate that WETR consistently improves accuracy and stability under frozen parameters, which also improves token-level interpretability.

Forum: https://openreview.net/forum?id=ViQO8FlRFR

---

## 194. DLM: Unified Decision Language Model for Offline Multi-Agent Sequential Decision Making

**Abstract:** Building scalable and reusable multi-agent decision policies from offline datasets remains a challenge in offline multi-agent reinforcement learning (MARL), as existing methods often rely on fixed observation formats and action spaces that limit generalization. In contrast, large language models (LLMs) offer a flexible modeling interface that can naturally accommodate heterogeneous observations and actions. Motivated by this, we propose the Decision Language Model (DLM), which formulates multi-agent decision making as a dialogue-style sequence prediction problem under the centralized training with decentralized execution paradigm. DLM is trained in two stages: a supervised fine-tuning phase, which leverages dialogue-style datasets for centralized training with inter-agent context and generates executable actions from offline trajectories, followed by a group relative policy optimization phase to enhance robustness to out-of-distribution actions through lightweight reward functions. Experiments on multiple benchmarks show that a unified DLM outperforms strong offline MARL baselines and LLM-based conversational decision-making methods, while demonstrating strong zero-shot generalization to unseen scenarios across tasks.

Forum: https://openreview.net/forum?id=qxG09uid4p

---

## 195. Optimal structure learning and conditional independence testing

**Abstract:** We establish a fundamental connection between optimal structure learning and optimal conditional independence testing by showing that the minimax optimal rate for structure learning problems is determined by the minimax rate for conditional independence testing in these problems. This is accomplished by establishing a general reduction between these two problems in the case of poly-forests, and demonstrated by deriving optimal rates for several examples, including Bernoulli, Gaussian and nonparametric models. Furthermore, we show that the optimal algorithm in these settings is a suitable modification of the PC algorithm. This theoretical finding provides a unified framework for analyzing the statistical complexity of structure learning through the lens of minimax testing.

Forum: https://openreview.net/forum?id=uurjpxrLc5

---

## 196. Real-World Unsupervised Models Generalize to Predict Brain Responses to Out-of-Distribution Stimuli

**Abstract:** Deep neural networks currently provide the leading quantitative models of neural responses in sensory systems. However, these networks remain implausible as models of sensory development, largely because they rely on supervised training with label efficiency far exceeding that of biological learning. Furthermore, these models are typically trained on manually curated datasets that lack the statistical properties of the natural environments to which the brain is exposed. Here, we demonstrate that models trained with unsupervised objectives on real-world data significantly outperform supervised models in predicting brain responses across both human auditory and visual cortex. We show that this performance advantage is not driven by network architecture or dataset size, but rather by the data distribution. Crucially, we find that unsupervised models trained on real-world data exhibit remarkable out-of-distribution generalization: a model trained exclusively on Mandarin speech accurately predicts English-driven brain responses, and a model trained on infant head-cam footage predicts adult visual responses to curated object images. Together, our results illustrate how deep neural networks can be used to reveal the real-world statistics that shape neural representations in the brain.

Forum: https://openreview.net/forum?id=H1HHss5Zj4

---

## 197. Privacy-Aware Video Anomaly Detection through Orthogonal Subspace Projection

**Abstract:** Video anomaly detection (VAD) systems often prioritize accuracy while overlooking privacy concerns, limiting their suitability for real-world deployment. We propose the Orthogonal Projection Layer (OPL), a lightweight module that removes task-irrelevant variations to produce representations focused on anomaly-relevant cues. To address privacy risks in human-centered scenarios, we introduce Guided OPL (G-OPL), which suppresses facial attributes using weak supervision from face-presence signals while preserving non-identifying features such as pose and motion. A cosine alignment objective enforces consistent capture and removal of facial information without identity labels or adversarial training. We further present a privacy-aware evaluation framework that jointly assesses detection performance and privacy preservation, and enables analysis of how sensitive information is filtered. Experiments show that embedding privacy constraints into model design reduces sensitive information while maintaining or improving detection accuracy, supporting projection-based architectures as a principled approach for privacy-aware VAD.

Forum: https://openreview.net/forum?id=fjUUmUCy9m

---

## 198. PaperBanana: Automating Academic Illustration for AI Scientists

**Abstract:** Despite rapid advances in autonomous AI scientists powered by language models, generating publication-ready illustrations remains a labor-intensive bottleneck in the research workflow.
To lift this burden, we introduce PaperBanana, an agentic framework for automated generation of publication-ready academic illustrations.
Powered by state-of-the-art VLMs and image generation models,
PaperBanana orchestrates specialized agents to retrieve references, plan content and style, render images, and iteratively refine via self-critique. 
To rigorously evaluate our framework, we introduce PaperBananaBench, comprising 292 test cases for methodology diagrams curated from NeurIPS 2025 publications, covering diverse research domains and illustration styles. 
Comprehensive experiments demonstrate that PaperBanana consistently outperforms leading baselines in faithfulness, conciseness, readability, and aesthetics. 
We further show that our method effectively extends to the generation of high-quality statistical plots.
Collectively,
PaperBanana paves the way for the automated generation of publication-ready illustrations.

Forum: https://openreview.net/forum?id=FneePKFVHT

---

## 199. MuonSSM: Orthogonalizing State Space Models for Sequence Modeling

**Abstract:** State space models (SSMs) have emerged as efficient linear-time alternatives to attention for long-sequence modeling. However, existing SSMs often suffer from instability and memory degradation over extended horizons due to poorly conditioned first-order updates and unbalanced update geometry. We introduce MuonSSM, a general framework that stabilizes SSM training by explicitly conditioning the geometry of memory updates rather than the recurrent transition matrix. MuonSSM augments SSMs with a momentum-based pathway and a lightweight Newton-Schulz transformation on low-rank input injections, yielding bounded and spectrally conditioned updates while preserving parallel scan complexity. Theory shows that MuonSSM improves gradient propagation, mitigates spectral amplification, and enriches memory representations over long horizons. Extensive experiments across language, vision, and time-series benchmarks show consistent gains in accuracy, robustness, and long-context performance when integrated into diverse SSM backbones. These results establish geometric conditioning of updates as a principled pathway to stable, scalable sequence modeling.

Forum: https://openreview.net/forum?id=GmP3VcfHi0

---

## 200. Improved Dimension Dependence for Bandit Convex Optimization with Gradient Variation

**Abstract:** Gradient-variation online learning has drawn increasing attention due to its deep connections to game theory and optimization. It has been studied extensively in the full-information setting, but is underexplored with bandit feedback. In this work, we focus on gradient variation in Bandit Convex Optimization (BCO) with two-point feedback. By proposing a refined analysis of the *non-consecutive* gradient variation, a fundamental quantity in gradient variation with bandit feedback, we improve the dimension dependence for both convex and strongly convex functions compared with the best known results (Chiang et al., 2013). Our improved analysis of the non-consecutive gradient variation also implies other favorable problem-dependent guarantees, such as gradient-variance and small-loss regret bounds. Beyond the two-point setup, we demonstrate the versatility of our technique by achieving the *first* gradient-variation bound for one-point bandit linear optimization over hyper-rectangular domains. Finally, we validate the effectiveness of our results in more challenging tasks such as dynamic and universal regret minimization, establishing the *first* gradient-variation dynamic and universal regret bounds for two-point BCO.

Forum: https://openreview.net/forum?id=X8evkEdMxb

---

## 201. Judging What We Cannot Solve: A Consequence-Based Approach for Oracle-Free Evaluation of Research-Level Math

**Abstract:** Recent progress in reasoning models suggests that generating plausible attempts for research-level mathematics may be within reach, but verification remains a bottleneck, consuming scarce expert time. We hypothesize that a meaningful solution should contain enough method-level information that, when applied to a neighborhood of related questions, it should yield better downstream performance than incorrect solutions. Building on this idea, we propose \textbf{Consequence-Based Utility}, an oracle-free evaluator that scores each candidate by testing its value as an in-context exemplar in solving related yet verifiable questions. Our approach is evaluated on an original set of research-level math problems each paired with one expert-written solution and nine LLM-generated solutions. Notably, Consequence-Based Utility consistently outperforms reward models, generative reward models, and LLM judges on ranking quality. Specifically, for GPT-OSS-120B it improves Acc@1 from 67.2 to 76.3 and AUC from 71.4 to 79.6, with similarly large AUC gains on GPT-OSS-20B (69.0 to 79.2). Furthermore, compared to LLM-Judges, it also exhibits a larger solver–evaluator gap, maintaining stronger correct–wrong separation even on instances the underlying solver often fails to solve.

Forum: https://openreview.net/forum?id=Dkwqzk5sAD

---

## 202. Towards Understanding Adam Convergence on Highly Degenerate Polynomials

**Abstract:** Adam is a widely used optimization algorithm in deep learning, yet the specific class of objective functions where it exhibits inherent advantages remains underexplored. Unlike prior studies requiring external schedulers and $\beta_2$ near 1 for convergence, this work investigates the ``natural'' auto-convergence properties of Adam. We identify a class of highly degenerate polynomials where Adam converges automatically without additional schedulers. Specifically, we derive theoretical conditions for local asymptotic stability on degenerate polynomials and demonstrate strong alignment between theoretical bounds and experimental results. We prove that Adam achieves local linear convergence on these degenerate functions, significantly outperforming the sub-linear convergence of Gradient Descent and Momentum. This acceleration stems from a decoupling mechanism between the second moment $v_t$ and squared gradient $g_t^2$, which exponentially amplifies the effective learning rate. Finally, we characterize Adam's hyperparameter phase diagram, identifying three distinct behavioral regimes: stable convergence, spikes, and SignGD-like oscillation.

Forum: https://openreview.net/forum?id=uYWVGk1Qt0

---

## 203. Solving Time-Dependent Differential Equations with Physical Dynamical Systems

**Abstract:** Time-Dependent Differential Equations (TDDEs) model dynamical processes across science and engineering, but time-critical applications require solvers that deliver high-fidelity trajectories under stringent latency constraints. Most existing TDDE solvers are limited by time discretization, forcing a latency-accuracy trade-off where smaller step sizes capture high-fidelity trajectories but incur prohibitive runtime, while larger steps meet real-time budgets at the cost of trajectory distortion. Dynamical System Machines (DSMs) offer a promising alternative by computing through continuous physical evolution, yet existing DSMs struggle to capture the spatiotemporal complexity of TDDEs. This work introduces DS-TS, a novel TDDE solver that is both accurate and efficient by leveraging the unique computational advantages of DSMs. DS-TS integrates three key innovations: (1) Excitatory-Inhibitory Inspired Coupling to better model complex spatial interactions; (2) State-aware Dynamic Nonlinearity to enable rich inter-node interactions and state-dependent spatiotemporal correlations; and (3) Hierarchical Temporal Integration to capture high-order temporal dependencies. Experiments demonstrate that DS-TS achieves high-fidelity solutions while delivering orders-of-magnitude improvements in speed ($\sim 10^3\times$) and energy efficiency ($\sim 10^5\times$) compared to baseline solvers.

Forum: https://openreview.net/forum?id=0YHSZPkMp8

---

## 204. Unifying and Optimizing Data Values for Selection via Sequential Decision-Making

**Abstract:** Data selection has emerged as a crucial downstream application of data valuation, yet the theoretical foundations for using data values in selection remain underexplored. We reformulate data selection as a sequential decision-making problem where the optimal selection sequence arises from dynamic programming, and data values can be understood as encodings of this optimal sequence. This framework unifies and reinterprets existing methods like Data Shapley through the lens of approximate dynamic programming, revealing them as myopic linear approximations to the sequential problem. We further analyze how selection optimality degrades with utility curvature under submodularity, explaining when and why these approximations fail. To bridge theory and practice, we propose an efficient bipartite graph-based surrogate that preserves submodular structure while enabling scalable greedy selection with provable guarantees. Experiments on classical ML benchmarks and large-scale LLM fine-tuning data selection demonstrate substantial improvements over existing methods. Code is publicly available at https://github.com/frankhlchi/SeqDataVal

Forum: https://openreview.net/forum?id=GFFMLwn053

---

## 205. End-to-End Autoregressive Image Generation with 1D Semantic Tokenizer

**Abstract:** Autoregressive image modeling relies on visual tokenizers to compress images into compact latent representations. We design an end-to-end training pipeline that jointly optimizes reconstruction and generation, enabling direct supervision from generation results to the tokenizer. This contrasts with prior two-stage approaches that train tokenizers and generative models separately. We further investigate leveraging vision foundation models to improve 1D tokenizers for autoregressive modeling. Our autoregressive generative model achieves strong empirical results, including a state-of-the-art FID score of 1.48 without guidance on ImageNet 256×256 generation.

Forum: https://openreview.net/forum?id=fSnxuRhWoC

---

## 206. VideoKR: Towards Knowledge- and Reasoning-Intensive Video Understanding

**Abstract:** We introduce VideoKR, the first large-scale training corpus specifically designed to strengthen knowledge- and reasoning-intensive video understanding. It comprises 315K video reasoning examples over 145K newly collected, CC-licensed, expert-domain videos. We develop a human-in-the-loop, skill-oriented example generation pipeline that targets progressively deeper video reasoning capabilities while ensuring the difficulty, diversity, and reliability of both the examples and their CoT rationales. We also curate VideoKR-Eval, a new expert-annotated benchmark where questions require genuine video understanding and knowledge-intensive reasoning rather than textual shortcuts. Our experiments show that, under a standard SFT$\rightarrow$GRPO pipeline, models post-trained on VideoKR outperform prior post-training approaches on knowledge-intensive video reasoning while remaining competitive on general video reasoning, highlighting data design as a key driver of progress in video reasoning. We further conduct comprehensive ablations to isolate the contributions of  VideoKR, providing actionable insights for future work.

Forum: https://openreview.net/forum?id=eXxFlOPTk4

---

## 207. XR-1: Towards Versatile Vision-Language-Action Models via Learning Unified Vision-Motion Representations

**Abstract:** Recent progress in large-scale robotic datasets and vision-language models (VLMs) has advanced research on vision-language-action (VLA) models.  However, existing VLA models still face two fundamental challenges: (\textit{i}) producing precise low-level actions from high-dimensional observations, (\textit{ii}) bridging domain gaps across heterogeneous data sources, including diverse robot embodiments and human demonstrations. Existing methods often encode latent variables from either visual dynamics or robotic actions to guide policy learning, but they fail to fully exploit the complementary multi-modal knowledge present in large-scale, heterogeneous datasets. In this work, we present \textbf{XR-1}, a novel framework for versatile and scalable VLA learning across diverse robots, tasks, and environments.
At its core, XR-1 introduces the \emph{Unified Vision-Motion Codes (UVMC)}, a discrete latent representation learned via a dual-branch VQ-VAE that jointly encodes visual dynamics and robotic motion.  UVMC addresses these challenges by (\textit{i}) serving as an intermediate representation between the observations and actions, and  (\textit{ii}) aligning multimodal dynamic information from heterogeneous data sources to capture complementary knowledge. To effectively exploit UVMC, we propose a \emph{three-stage training paradigm}: (\textit{i}) self-supervised UVMC learning, (\textit{ii}) UVMC-guided pretraining on large-scale cross-embodiment robotic datasets, and (\textit{iii}) task-specific post-training.  We validate XR-1 through extensive real-world experiments with more than 12,000 rollouts on six different robot embodiments, spanning over 120 diverse manipulation tasks. XR-1 consistently outperforms state-of-the-art baselines such as $\pi_0$ and GR00T-N1.5 while demonstrating strong generalization to novel objects, background variations, distractors, and illumination changes. Our project is at \href{https://xr-1-vla.github.io/}{https://xr-1-vla.github.io/}.

Forum: https://openreview.net/forum?id=JO0IsGJg16

---

## 208. Learning Coupled Continuous-Time Latent Dynamics from Irregular Events

**Abstract:** Modeling dynamic dependencies from irregularly sampled event sequences is a fundamental challenge in modern machine learning. In many real-world systems, individual-level states evolve continuously over time while being simultaneously influenced by population-level dynamics. However, existing methods typically model these processes in isolation or rely on discrete-time approximations that fail to capture long-range temporal irregularities and sparse observations. This paper studies the problem of learning coupled continuous-time latent dynamics from irregular events, where individual event sequences and global distributional processes evolve asynchronously and interact over time. We propose a Coupled Continuous-Time Latent Dynamics (CoCLD) framework that jointly models individual latent dynamics and population-level distributional shifts, and aligns them in a continuous-time latent space. CoCLD integrates a Diffusion-based Latent Interpolator with neural ordinary differential equations, enabling principled interpolation, generation, and alignment of latent states across arbitrary time points. We show that the proposed coupling mechanism yields a consistent estimator of continuous-time latent dynamics under sparse and irregular observations. Empirical evaluations show CoCLD effectively captures dynamic dependencies and generalizes across tasks like next-event prediction, mobility trajectory generation, and sequential behavior modeling, indicating that learning coupled continuous-time latent dynamics is a powerful paradigm for irregular event sequence modeling.

Forum: https://openreview.net/forum?id=HfQ0X1lTGg

---

## 209. Suppress and Diversify: Refining Robust Pathways for Corruption Robustness

**Abstract:** Model robustness against natural image corruptions is essential for safety-critical applications. While existing methods primarily focus on implicit representation learning, we provide the first systematic exploration of computational pathways to explicitly characterize internal robustness. We identify a progressive decay of robust features across network layers and establish a functional dependency between the prevalence of these features and model performance. To exploit these insights, we propose Suppress and Diversify (S\&D), a non-intrusive refinement approach that enhances robustness by dynamically selecting robust pathways and diversifying them through symmetry-preserving transformations. S\&D is architecture-agnostic, parameter-free, and incurs zero test-time overhead. Extensive evaluations across eight benchmarks demonstrate that S\&D consistently improves performance across multiple vision tasks, diverse backbones, and complex real-world scenarios, highlighting its broad efficacy and scalability.

Forum: https://openreview.net/forum?id=Tam54Owz7G

---

## 210. Orthogonal Concept Erasure for Diffusion Models

**Abstract:** Concept erasure has emerged as a promising approach to mitigate undesired or unsafe content in diffusion models, yet existing methods still face significant limitations. While training-based methods are effective, their high computational cost limits scalability. Editing-based methods are more efficient and deployment-friendly, yet they struggle to simultaneously achieve precise concept erasure and preserve overall generative capacity. We identify this core limitation of the editing-based methods as reliance on additive parameter updates. Our empirical analysis reveals that concept semantics primarily depend on *neuron direction* rather than *neuron magnitude*, while overall generative capacity relies on the *angular geometry* of neurons. As additive updates inherently entangle direction, magnitude, and angular geometry, they inevitably introduce unintended interference between concept erasure and overall generation performance. To address this, we propose **Orthogonal Concept Erasure (OCE)**, which reformulates editing-based erasure as multiplicative parameter updates from a geometric perspective. Specifically, OCE applies layer-wise orthogonal transformations derived from a closed-form solution to the parameters, enabling precise concept erasure while preserving the neuron magnitude and angular geometry. Furthermore, to address conflicting constraints in multi-concept erasure, OCE introduces a subspace-level objective with structured subspace manipulation, yielding a more effective and scalable erasure. Extensive experiments on single- and multi-concept erasure demonstrate that OCE outperforms existing methods in concept erasure and non-target preservation, erasing up to 100 concepts in 4.3 s. Code: https://github.com/HansSunY/OCE.

Forum: https://openreview.net/forum?id=VE2aKpTWrK

---

## 211. Practical and Optimal Algorithm for Linear Contextual Bandits with Rare Parameter Updates

**Abstract:** We study linear contextual bandits under rare parameter updates:
the learner may incorporate reward feedback into its parameter estimate only at a small number of update times,
while still observing contexts online and selecting actions sequentially.
This viewpoint clarifies a practical distinction that is often blurred in the literature:
many "strictly batched" methods additionally restrict within-interval context adaptivity,
meaning that the action rule inside an interval cannot depend on the sequence of realized contexts/actions in that interval (beyond the current round's context).
For linear contextual bandits, we propose two practical algorithms with only $O(\log\log T)$ parameter updates.
Our first algorithm BLCE-G attains minimax-optimal regret (up to polylogarithmic factors in $T$) simultaneously in both the small-$K$ and large-$K$ regimes under a static schedule.
Our second algorithm BLCE removes the near G-optimal design step---a dominant computational bottleneck in prior strictly batched static-grid methods---yet preserves minimax-optimal regret and achieves the lowest known runtime complexity among optimal algorithms.
We further extend these rare-update and computational principles to generalized linear contextual bandits.
Overall, our results yield statistically optimal algorithms under $O(\log\log T)$ parameter updates that are also computationally efficient in practice.

Forum: https://openreview.net/forum?id=rEqVnZZOiA

---

## 212. Symmetries in language statistics shape the geometry of model representations

**Abstract:** Although learned representations underlie neural networks' success, their fundamental properties remain poorly understood.
A striking example is the emergence of simple geometric structures in LLM representations: for example, calendar months organize into a circle, years form a one-dimensional manifold, and the latitude and longitude of cities can be decoded by low-dimensional linear probes.
We show that the statistics of language exhibit a translation symmetry---e.g,. the co-occurrence probability of two months depends only on the time interval between them---and we prove that the latter governs the aforementioned geometric structures in high-dimensional word embedding models.
Moreover, we find that these structures persist even when the co-occurrence statistics are strongly perturbed (for example, by removing all sentences in which two months appear together) and at moderate embedding dimension.
We show that this robustness naturally emerges if the co-occurrence statistics are collectively controlled by an underlying continuous latent variable.
We empirically validate this theoretical framework in word embedding models, text embedding models, and large language models.

Forum: https://openreview.net/forum?id=XQeMPEkfdd

---

## 213. PanoWorld-X: Generating Explorable Panoramic Worlds via Sphere-Aware Video Diffusion

**Abstract:** Achieving a complete and explorable 360-degree visual world is a cornerstone of immersive content creation. While recent advances in video generation have achieved impressive results, they follow a 2D paradigm that treats content generation as transitions of 2D pixels, lacking an intrinsic understanding of the physical 3D world, resulting in frequent geometric inconsistencies.
To achieve an explorable and physical-consistent visual world, the generation process should shift to a 3D paradigm: the visual content is governed by the physical relationships of the entire 3D environment together with 3D motion signals. However, under this setting, the conventional modeling methods and control signals, such as spatial attention computation in a 2D space, become unsuitable and ineffective.
To address this, we propose PanoWorld-X for explorable immersive scene video generation. Our framework is built on the panoramic representation, which naturally maps a 3D scene into a standard format and provides an ideal basis for consistency. Specifically, we first develop a data curation pipeline to produce high-quality and large-motion 3D scene evolution with movement trajectories. To achieve precise control, we design the Exploration Panoramic Plücker Embedding (PPE), a guidance signal tailored for 3D motion. Furthermore, leveraging the spherical geometric properties of panoramic data, we propose a sphere-aware attention mechanism, which can capture true geometric adjacency by reprojecting features onto a spherical surface. Extensive experiments demonstrate that PanoWorld-X achieves superior performance in motion range, control precision, and visual quality, underscoring its potential for real-world applications.

Forum: https://openreview.net/forum?id=xEgoeNrp8B

---

## 214. WeDLM: Reconciling Diffusion Language Models with Standard Causal Attention for Fast Inference

**Abstract:** Autoregressive (AR) generation is the standard decoding paradigm for Large Language Models (LLMs), but its token-by-token nature limits parallelism at inference time. Diffusion Language Models (DLLMs) offer parallel decoding by recovering multiple masked tokens per step; however, in practice they often fail to translate this parallelism into speed gains over optimized AR engines (e.g., vLLM). A key reason is that many DLLMs rely on bidirectional attention, which breaks standard prefix KV caching.
We propose WeDLM, a diffusion decoding framework built entirely on standard causal attention to make parallel generation prefix-cache friendly. The core idea is to let each masked position condition on all observed tokens while keeping a causal mask, achieved by Topological Reordering that moves observed tokens to the physical prefix while preserving their logical positions. Building on this, we introduce a streaming decoding procedure that continuously commits confident tokens into a growing left-to-right prefix, avoiding the stop-and-wait behavior common in block diffusion methods. Experiments show that WeDLM preserves the quality of strong AR backbones while delivering substantial speedups, approaching 3× on challenging reasoning benchmarks and up to 10× in low-entropy generation regimes; critically, our comparisons are against AR baselines served by vLLM under matched deployment settings.

Forum: https://openreview.net/forum?id=QwtmbKAOZU

---

## 215. Weak Diffusion Priors Can Still Achieve Strong Inverse-Problem Performance

**Abstract:** Can a diffusion model trained on bedrooms recover human faces? Diffusion models are widely used as priors for inverse problems, but standard approaches usually assume a high-fidelity model trained on data that closely match the unknown signal. In practice, one often must use a mismatched or low-fidelity diffusion prior. Surprisingly, these weak priors often perform nearly as well as full-strength, in-domain baselines. We study when and why inverse solvers are robust to weak diffusion priors. Through extensive experiments, we find that weak priors succeed when measurements are highly informative (e.g., many observed pixels), and we identify regimes where they fail. To explain this behavior, we combine Bayesian-consistency theory with local-correlation analysis: the theory gives conditions under which high-dimensional measurements make the posterior concentrate near the true signal, while the correlation analysis shows that weak and stronger natural-image priors can share similar local spatial structure. These results provide a principled justification on when weak diffusion priors can be used reliably. Code is available at https://github.com/jjia131/weak-diffusion-priors-inverse-problem.

Forum: https://openreview.net/forum?id=fdkSA4F0lN

---

## 216. FedARC: Anchor-Guided Residual Compensation for Data and Model Heterogeneous Federated Learning

**Abstract:** Federated learning (FL) allows clients to collaboratively train models without exposing private data, but practical FL is simultaneously challenged by data heterogeneity and model heterogeneity. Prior heterogeneous FL (HtFL) approaches often fail to handle fine-grained feature shifts, leading to weak representation alignment and limited cross-client knowledge transfer, which degrades both personalization and generalization. We propose FedARC, an HtFL framework that couples a shared lightweight extractor with client-specific fusion: a trainable projector integrates local and global embeddings, while adaptive residual compensation dynamically corrects feature-level mismatches. To further stabilize aggregation, FedARC performs semantic anchor alignment across clients, and we theoretically prove FedARC converges with a non-convex convergence rate $\mathcal{O}(1/T)$. Experiments on five public benchmarks demonstrate that FedARC outperforms nine state-of-the-art HtFL baselines by up to 2.63\% in average accuracy, while maintaining efficient communication and computation.

Forum: https://openreview.net/forum?id=qQhTbQSHdG

---

## 217. Protein Autoregressive Modeling via Multiscale Structure Generation

**Abstract:** We present protein autoregressive modeling (PAR), the first multi-scale autoregressive framework for protein backbone generation via coarse-to-fine next-scale prediction. Using the hierarchical nature of proteins, PAR generates structures that mimic sculpting a statue, forming a coarse topology and refining structural details over scales. To achieve this, PAR consists of three key components: (i) multi-scale downsampling operations that represent protein structures across multiple scales during training; (ii) an autoregressive transformer that encodes multi-scale information and produces conditional embeddings to guide structure generation; (iii) a flow-based backbone decoder that generates backbone atoms conditioned on these embeddings. Moreover, autoregressive models suffer from exposure bias, caused by the training and the generation procedure mismatch, and substantially degrades structure generation quality. We effectively alleviate this issue by adopting noisy context learning and scheduled sampling, enabling robust backbone generation. Notably, PAR exhibits strong zero-shot generalization, supporting flexible human-prompted conditional generation and motif scaffolding without requiring fine-tuning. On the unconditional generation benchmark, PAR effectively learns protein distributions and produces backbones of high design quality, and exhibits favorable scaling behavior. Together, these properties establish PAR as a promising framework for protein structure generation.

Forum: https://openreview.net/forum?id=08tW615mgI

---

## 218. Linguistic Nepotism: Trading-off Quality for Language Preference in Multilingual RAG

**Abstract:** Multilingual Retrieval-Augmented Generation (mRAG) systems enable language models to answer knowledge-intensive queries with citation-supported responses across languages. Despite their growing use, an open questions is whether the mixture of different document languages impacts generation and citation behavior in *unintended* ways. To investigate this, we introduce a controlled methodology using model internals to measure language preference while holding other factors such as document relevance constant. Across eight languages and six open-weight models, we find that models preferentially cite English sources when queries are in English, with this bias amplified for lower-resource languages and for documents positioned mid-context. More crucially, we find that models sometimes trade-off document relevance for language preference, indicating that citation choices are not always driven by informativeness alone. Our findings shed light on how language models leverage multilingual context and influence citation behavior.

Forum: https://openreview.net/forum?id=MDfCV0UHSb

---

## 219. From Denoising to De-Channeling: Integrating Physical Channel Priors into Diffusion Models for Radio Signal Understanding

**Abstract:** In recent years, wireless signal recognition (WSR), which leverages artificial intelligence (AI) to identify properties of passively received radio signals, has garnered significant attention due to its broad applications, such as spectrum management. Existing WSR methods typically learn directly from received signals, which are distorted by physical wireless channel effects such as fading, and current denoising diffusion models lack de-channeling capabilities, which leads to performance degradation. Therefore, we propose PWC-Diff, a novel framework that integrates prior Physical Wireless Channels into the denoising Diffusion process. The framework employs a dedicated architecture named FusedFormer, which contains a fusion module and a self-attention module that jointly capture the temporal and spectral characteristics of the signals throughout the diffusion trajectory. By leveraging prior wireless channels, PWC-Diff learns to progressively “de-channel” the received signal and recover a representation closer to the transmitted signal. Extensive experiments on several datasets across three WSR tasks have achieved state-of-the-art (SOTA) performance, which demonstrates the rationality of our theory, and ablation experiments further illustrate the effectiveness of our proposed PWC-Diff. Code is available at https://github.com/BUPT-GAMMA/FoundWSR.

Forum: https://openreview.net/forum?id=NUerSGt9wn

---

## 220. Motion Attribution for Video Generation

**Abstract:** Despite the rapid progress of video generation models, the role of data in influencing motion is poorly understood. We present Motive (MOTIon attribution for Video gEneration), a motion-centric, gradient-based data attribution framework that scales to modern, large, high-quality video datasets and models. We use this to study which fine-tuning clips improve or degrade temporal dynamics. Motive isolates temporal dynamics from static appearance via motion-weighted loss masks, yielding efficient and scalable motion-specific influence computation. On text-to-video models, Motive identifies clips that strongly affect motion and guides data curation that improves temporal consistency and physical plausibility. With Motive-selected high-influence data, we improve both motion smoothness and dynamic degree on VBench, achieving a 74.1% human preference win rate compared with the pretrained base model. To our knowledge, this is the first framework to attribute motion rather than visual appearance in video generative models and to use it to curate fine-tuning data.

Forum: https://openreview.net/forum?id=zAl9heLw4q

---

## 221. A Regret Minimization Framework on Preference Learning  in Large Language Models

**Abstract:** Reinforcement learning with verifiable rewards (RLVR) has enabled progress on reasoning-intensive tasks by relying on task-specific verifiers that provide automated correctness signals. However, many realistic language tasks are difficult to equip with reliable verifiers, motivating a growing reliance on reinforcement learning from human feedback (RLHF). 
In this setting, we argue that a closer examination of how human feedback should be interpreted is essential.
We introduce Regret-based Preference Optimization (RePO), which reframes RLHF through *regret minimization* rather than reward maximization. 
Human preferences are often shaped by *prospective* anticipation of outcomes and *counterfactual* comparisons to alternative behaviors, rather than by immediate, outcome-independent utility. 
RePO captures this structure by modeling preferences as behavior-conditioned assessments of relative suboptimality.
Within a KL-regularized reinforcement learning framework, RePO admits a closed-form policy update compatible with direct preference optimization. 
Experiments on mathematical reasoning benchmarks and human-annotated preference datasets demonstrate consistent performance gains, indicating that regret-based preference learning is an effective and human-aligned approach for training large language models.

Forum: https://openreview.net/forum?id=genVnYBAV7

---

## 222. Provably Convergent Actor-Critic for MARL through Risk-aversion

**Abstract:** Learning stationary policies in infinite-horizon general-sum Markov games (MGs) remains a fundamental open problem in Multi-Agent Reinforcement Learning (MARL). While stationary strategies are preferred for their practicality, computing stationary forms of classic game-theoretic equilibria is computationally intractable—a stark contrast to the comparative ease of solving single-agent RL or zero-sum games. To bridge this gap, we study Risk-averse Quantal response Equilibria (RQE), a solution concept rooted in behavioral game theory that incorporates risk aversion and bounded rationality. We demonstrate that RQE possesses strong regularity conditions that make it uniquely amenable to learning in MGs. We propose a novel single-timescale Actor-Critic algorithm characterized by a faster actor and a slower critic. Leveraging the regularity of RQE, we prove that this approach achieves global convergence with finite-sample guarantees. We empirically validate our algorithm in several environments to demonstrate superior convergence properties compared to risk-neutral baselines.

Forum: https://openreview.net/forum?id=NpguYNGrG2

---

## 223. Diffuse to Detect: Bi-Level Sample Rebalancing with Pseudo-Label Diffusion for Point-Supervised Infrared Small-Target Detection

**Abstract:** Point supervision has become a scalable solution to address dense annotation for infrared small target detection, but its performance is limited by two coupled bottlenecks: unstable pseudo-label evolution in cluttered, low-contrast infrared imagery and severe sample-distribution imbalance. In this paper, we present a more adaptive and stable framework to address these issues. Leveraging the intrinsic consistency between thermal radiation patterns and heat diffusion, we propose a physics-induced annotation strategy that expands single-point labels into reliable pseudo-masks. To further enhance supervision and alleviate sample imbalance, we develop a bi-level dual-update framework that jointly optimizes detector weights, sample weights, and diffusion parameters. A meta-classifier dynamically predicts sample-wise loss weights, while a differentiable diffusion module refines pseudo-labels with detection feedback, enabling adaptive interaction between training and hyperparameter optimization. Extensive experiments across multiple datasets demonstrate five-fold annotation acceleration, superior detection accuracy, and comparable performance with 30\% of the training data, validating the efficiency and practicality of our approach. Our code is available at https://github.com/yuanhang-yao/diffuse-to-detect.

Forum: https://openreview.net/forum?id=RNn2wtNeoK

---

## 224. MemoryBench: A Benchmark for Memory and Continual Learning in LLM Systems

**Abstract:** Scaling up data, parameters, and test-time computation has been the mainstream methods to improve LLM systems (LLMsys), but their upper bounds are almost reached due to the gradual depletion of high-quality data and marginal gains obtained from larger computational resource consumption. Inspired by the abilities of human and traditional AI systems in learning from practice, constructing memory and continual learning frameworks for LLMsys has become an important and popular research direction in recent literature. Yet, existing benchmarks for LLM memory often focus on evaluating the system on homogeneous reading comprehension tasks with long-form inputs rather than testing their abilities to learn from accumulated user feedback in service time. Therefore, we propose a user feedback simulation framework and a comprehensive benchmark covering multiple domains, languages, and types of tasks to evaluate the continual learning abilities of LLMsys. Experiments show that the effectiveness and efficiency of state-of-the-art baselines are far from satisfying, and we hope this benchmark could pave the way for future studies on LLM memory and optimization algorithms.

Forum: https://openreview.net/forum?id=If4X4W2HWx

---

## 225. Beyond Log Likelihood: Probability-Based Objectives for Supervised Fine-Tuning across the Model Capability Continuum

**Abstract:** Supervised fine-tuning (SFT) is the standard approach for post-training large language models (LLMs), yet it often shows limited generalization. We trace this limitation to its default training objective: negative log likelihood (NLL). While NLL is classically optimal when training from scratch, post-training operates in a different paradigm and could violate its optimality assumptions, where models already encode task-relevant priors and supervision can be long and noisy. In this work, we systematically study various probability-based objectives and characterize when and why different objectives succeed or fail under varying conditions. Through comprehensive experiments and extensive ablation studies across 8 model backbones, 27 benchmarks, and 7 domains, we uncover a critical dimension that governs objective behavior: the model-capability continuum. Near the model-strong end, prior-leaning objectives that downweight low-probability tokens (e.g., $-p$, $-p^{10}$, thresholded variants) consistently outperform NLL; toward the model-weak end, NLL dominates; in between, no single objective prevails. Our theoretical analysis further elucidates how objectives trade places across the continuum, providing a principled foundation for adapting objectives to model capability. The code is available at https://github.com/GaotangLi/Beyond-Log-Likelihood.

Forum: https://openreview.net/forum?id=2hQBG2ZlFb

---

## 226. Dual-Latent Memory Routing for Vision-Language Reasoning

**Abstract:** Multimodal large language models (MLLMs) have recently made strong progress in vision-language reasoning, yet their performance often degrades as generations grow longer. A key factor is that they frequently lose track of earlier visual evidence and intermediate constraints under a monolithic growing context. Inspired by how humans separately recall what they see and what they infer when solving complex tasks, we propose DLMR, a parameter-efficient mechanism that equips MLLMs with Dual Latent Memories: a visual memory that compresses image evidence and a reasoning memory that tracks intermediate conclusions and constraints. A Router then dynamically decides which memory and how much to reuse during inference, preserving visual grounding while maintaining coherent long-horizon reasoning. DLMR is trained in three stages, from latent memory construction to selective router learning, while keeping the base MLLM frozen, yielding substantial gains on both general and reasoning benchmarks with only a small number of additional trainable parameters. Analyses further show interpretable, state-dependent routing with specialized memory roles and reduced decoding tokens over long generations. Code is available at https://github.com/Hunter-Wrynn/DLMR.

Forum: https://openreview.net/forum?id=SFWWUr9V7c

---

## 227. MetaphorVU: Towards Metaphorical Video Understanding

**Abstract:** Metaphorical videos are prevalent across various real-world scenarios to convey complex ideas, and understanding them typically requires high-order cognitive capabilities. 
The lack of systematic studies on metaphorical video understanding not only constrains the real-world applicability of MLLMs but also impedes the thorough assessment of their high-order cognitive capabilities.
To bridge this gap, we propose MetaphorVU-Bench, the first systematic and comprehensive benchmark dedicated to metaphorical video understanding. 
Through experiments, we find current MLLMs struggle with accurate metaphorical video understanding, lagging far behind human level, primarily due to defective cross-domain mapping. 
Motivated by this finding, we construct a metaphor knowledge graph as mapping augmentation and propose MetaphorBoost, an inference-time enhancement framework achieving consistent performance improvement. 
Our benchmark, analysis, and method provide useful insights and a foundation for future research on advancing MLLMs.
Code: https://github.com/icip-cas/MetaphorVU.

Forum: https://openreview.net/forum?id=yKcBAJMPXZ

---

## 228. Effective Reasoning Chains Reduce Intrinsic Dimensionality

**Abstract:** Chain-of-thought (CoT) reasoning and its variants have substantially improved the performance of language models on complex reasoning tasks, yet the precise mechanisms by which different strategies facilitate generalization remain poorly understood. While current explanations often point to increased test-time computation or structural guidance, establishing a consistent, quantifiable link between these factors and generalization remains challenging. In this work, we identify *intrinsic dimensionality* as a quantitative measure for characterizing the effectiveness of reasoning chains. Intrinsic dimensionality quantifies the minimum number of model dimensions needed to reach a given accuracy threshold on a given task. By keeping the model architecture fixed and varying the task formulation through different reasoning strategies, we demonstrate that effective reasoning strategies consistently reduce the intrinsic dimensionality of the task. Validating this on GSM8K with Gemma-3 1B and 4B, we observe a strong inverse correlation between the intrinsic dimensionality of a reasoning strategy and its generalization performance on both in-distribution and out-of-distribution data. Our findings suggest that effective reasoning chains facilitate learning by better compressing the task using fewer parameters, offering a new quantitative metric for analyzing reasoning processes.

Forum: https://openreview.net/forum?id=vCjGjvrIR9

---

## 229. Modular Pretraining Enables Access Control

**Abstract:** AI developers face a dual-use dilemma. A model capability that helps one user cure a disease can help another synthesize one. This dilemma could be resolved with access control, limiting dual-use AI capabilities to trusted deployments with a legitimate need. A gold standard for access control would be to serve separate models with different capabilities to different users. However, training and deploying multiple models is prohibitively expensive. To address this challenge, we propose gradient-routed auxiliary modules (GRAM), a pre-training method that adds modules to a neural network and selectively updates them to induce specialization. Ablating a module at inference time removes its capability from the network, approximating a model trained on filtered data. We evaluate GRAM on synthetic stories and realistic dual-use data spanning virology, cybersecurity, nuclear physics, and specialized code. These experiments show that GRAM preserves selected retain capabilities while disabling forgotten capabilities, and limits recovery better than post-hoc unlearning. Most importantly, a Chinchilla-optimal scaling analysis from 50M to 5B parameters shows that the forget capability gap between data-filtered and full-data models widens with scale while the retain gap stays constant, and that GRAM closely tracks data filtering. GRAM's training cost is independent of the number of supported capability profiles, yielding a 5x cost reduction in our 5-profile setting.

Forum: https://openreview.net/forum?id=yIubI9l3IT

---

## 230. Geometry-Aware Decoding with Wasserstein-Regularized Truncation and Mass Penalties for Large Language Models

**Abstract:** Large language models (LLMs) must balance diversity and creativity against logical coherence in open-ended generation. 
Existing truncation-based samplers are effective but largely heuristic, relying mainly on probability mass and entropy while ignoring semantic geometry of the token space.
We present Top-$W$, a geometry-aware truncation rule that uses Wasserstein distance—defined over token-embedding geometry—to keep the cropped distribution close to the original, while explicitly balancing retained probability mass against the entropy of the kept set. 
Our theory yields a simple closed-form structure for the fixed-potential subset update: depending on the mass–entropy trade-off, the optimal crop either collapses to a single token or takes the form of a one-dimensional prefix that can be found efficiently with a linear scan. 
We implement Top-$W$ using efficient geometry-based potentials (nearest-set or $k$-NN) and pair it with an alternating decoding routine that keeps the standard truncation-and-sampling interface unchanged.
Extensive experiments on four benchmarks (GSM8K, GPQA, AlpacaEval, and MT-Bench) across three instruction-tuned models show that Top-$W$ consistently outperforms prior state-of-the-art decoding approaches achieving up to 33.7 percentage improvement. 
Moreover, we find that Top-$W$ not only improves accuracy-focused performance, but also boosts creativity under judge-based open-ended evaluation.

Forum: https://openreview.net/forum?id=HSuU4xBmAv

---

## 231. A Systematic Study of Behavioral Cloning for Scientific Data Annotation

**Abstract:** Scientific data annotation, such as tracking animals in video or proofreading neural reconstructions, remains bottlenecked by the “last mile” problem: even with strong automation, verification and correction consume substantial human effort. Standard approaches train models to directly predict annotations, discarding the rich supervision in how experts navigate, click, verify, and correct. We introduce a framework for studying behavioral cloning on scientific annotation: 9 synthetic tasks paired with synthetic annotations that simulate realistic human strategies including exploration, mistake correction, and strategic decision-making. Our experiments reveal several findings. First, skills emerge hierarchically: models learn GUI mechanics before task-critical decisions, and commit fewer mistakes than the training data while retaining the ability to correct errors when they occur. Second, scaling models on multi-task behavioral cloning shows that larger models are more data efficient within our scale range. Third, multi-task pretraining enables efficient fine-tuning to new tasks, while training from scratch fails entirely. Fourth, linear probes reveal that models internally represent latent variables of the annotation process such as task phase and data position; interestingly, we find a shared mistake representation that generalizes across different annotation tasks. Overall, our framework establishes systematic benchmarks and identifies key bottlenecks, providing a foundation for scaling behavioral cloning to real-world scientific data annotation.

Forum: https://openreview.net/forum?id=UcbCkTkUjD

---

## 232. A Constrained Optimization Perspective of Unrolled Transformers

**Abstract:** We introduce a constrained optimization framework for training transformers that behave like optimization descent algorithms. Specifically, we enforce layerwise descent constraints on the objective function and replace standard empirical risk minimization (ERM) with a primal-dual training scheme. This approach yields models whose intermediate representations decrease the loss monotonically in expectation across layers. We apply our method to both unrolled transformer architectures and conventional pretrained transformers on tasks of video denoising and text classification. Across these settings, we observe that constrained transformers achieve stronger robustness to perturbations and maintain higher out-of-distribution generalization, while preserving competitive in-distribution performance.

Forum: https://openreview.net/forum?id=aYe8j2jOmK

---

## 233. Alignment Pretraining: AI Discourse Causes Self-Fulfilling (Mis)alignment

**Abstract:** Pretraining corpora contain extensive discourse about AI systems, yet the causal influence of this discourse on downstream alignment remains poorly understood. If prevailing descriptions of AI behaviour are predominantly negative, LLMs may internalise corresponding behavioural priors, giving rise to self-fulfilling misalignment. This paper provides the first controlled study of this hypothesis by pretraining 6.9B-parameter LLMs with varying amounts of (mis)alignment discourse. We find that discussion of AI contributes to misalignment. Upsampling synthetic training documents about AI misalignment leads to a notable increase in misaligned behaviour. Conversely, upsampling documents about aligned behaviour reduces misalignment scores from 45% to 9%. We consider this evidence of self-fulfilling alignment. These effects are dampened, but persist through post-training. Our findings establish the study of how pretraining data shapes alignment priors, or alignment pretraining, as a complement to post-training. We recommend practitioners pretrain for alignment as well as capabilities.

Forum: https://openreview.net/forum?id=951OAanYyQ

---

## 234. FlashOptim: Optimizers for Memory-Efficient Training

**Abstract:** Standard mixed-precision training of neural networks requires many bytes of accelerator memory for each model parameter. These bytes reflect not just the parameter itself, but also its gradient and one or more optimizer state variables. With each of these values typically requiring 4 bytes, training even a 7 billion parameter model can be impractical for researchers with less than 100\,GiB of accelerator memory. 

We introduce FlashOptim, a suite of optimizations that reduces per-parameter memory by over 50% while preserving model quality and API compatibility. Our approach introduces two key techniques. First, we improve master weight splitting by finding and exploiting a tight bound on its quantization error. Second, we design companding functions that greatly reduce the error in 8-bit optimizer state quantization. Together with 16-bit gradients, these techniques reduce AdamW memory from 16 bytes to 7 bytes per parameter, or 5 bytes with gradient release. They also cut model checkpoint sizes by more than half. 

Experiments with FlashOptim applied to SGD, AdamW, and Lion show no measurable quality degradation across a collection of standard vision and language benchmarks, including Llama-3.1-8B finetuning.

Forum: https://openreview.net/forum?id=Wfe1iJocjF

---

## 235. FIRE: Multi-Fidelity Regression with Distribution-Conditioned In-Context Learning Using Tabular Foundation Models

**Abstract:** Multi-fidelity (MF) regression often operates in regimes of extreme data imbalance, where the commonly-used Gaussian-process surrogates struggle with cubic scaling costs and overfit to sparse high-fidelity observations, limiting efficiency and generalization in real-world applications. We introduce FIRE, a training-free MF framework that couples tabular foundation models (TFMs) to perform zero-shot in-context Bayesian inference via a high-fidelity correction model conditioned on the low-fidelity model's posterior predictive distributions. This cross-fidelity information transfer via distributional summaries captures heteroscedastic errors, enabling robust residual learning without model retraining. Across 31 benchmark problems spanning synthetic functions and real-world tasks (e.g., DrivAerNet, LCBench), FIRE delivers a stronger performance–time trade-off than seven state-of-the-art GP-based or deep learning MF regression methods, ranking highest in accuracy and uncertainty quantification with runtime advantages. Limitations include context window constraints and dependence on the quality of the pre-trained TFMs. Code & data can be found here: https://github.com/rosenyu304/FIRE.

Forum: https://openreview.net/forum?id=JxbxHB5d9v

---

## 236. Activation Oracles: Training and Evaluating LLMs as General-Purpose Activation Explainers

**Abstract:** Large language model (LLM) activations are notoriously difficult to understand, with most existing techniques using complex, specialized methods for interpreting them. Recent work has proposed a simpler approach known as LatentQA: training LLMs to directly accept LLM activations as inputs and answer arbitrary questions about them in natural language. However, prior work has focused on narrow task settings for both training and evaluation. In this paper, we instead take a generalist perspective. We evaluate LatentQA-trained models, which we call Activation Oracles (AOs), in far out-of-distribution settings and examine how performance scales with training data diversity. We find that AOs can recover information fine-tuned into a model (e.g., biographical knowledge or malign propensities) that does not appear in the input text, despite never being trained with activations from a fine-tuned model. Our main evaluations are four downstream tasks where we can compare to prior white- and black-box techniques. We find that even narrowly-trained LatentQA models can generalize well, and that adding additional training datasets (such as classification tasks and a self-supervised context prediction task) yields consistent further improvements. Our best AOs match or exceed white-box baselines on all four tasks and the best overall baseline on 3 of 4. These results suggest that diversified training to answer natural-language queries imparts a general capability to verbalize information about LLM activations.

Forum: https://openreview.net/forum?id=DbZjxkZrZm

---

## 237. Robust Causal Discovery in Real-World Time Series with Power-Laws

**Abstract:** Exploring causal relationships in stochastic time series is a challenging yet crucial task with a vast range of applications, including finance, economics, neuroscience, and climate science. Many algorithms for Causal Discovery (CD) have been proposed; however, they often exhibit a high sensitivity to noise, resulting in spurious causal inferences on real data. In this paper, we observe that the frequency spectra of many real-world time series follow a power-law distribution, notably due to an inherent self-organizing behavior. Leveraging this insight, we build a robust CD method based on the extraction of power‑law spectral features that amplify genuine causal signals. Our method consistently outperforms state-of-the-art alternatives on both synthetic benchmarks and real-world datasets with known causal structures, demonstrating its robustness and practical relevance.

Forum: https://openreview.net/forum?id=7i8d203tky

---

## 238. Posterior Behavioral Cloning: Pretraining BC Policies for Efficient RL Finetuning

**Abstract:** Standard practice across domains from robotics to language is to first pretrain a policy on a large-scale demonstration dataset, and then finetune this policy, typically with reinforcement learning (RL), in order to improve performance on deployment domains. This finetuning step has proved critical in achieving human or super-human performance, yet while much attention has been given to developing more effective finetuning algorithms, little attention has been given to ensuring the pretrained policy is an effective initialization for RL finetuning. In this work we seek to understand how the pretrained policy affects finetuning performance, and how to pretrain policies in order to ensure they are effective initializations for finetuning. We first show theoretically that standard behavioral cloning (BC) can fail to ensure coverage over the demonstrator's actions, a minimal condition necessary for effective RL finetuning. We then show that if, instead of exactly fitting the observed demonstrations, we train a policy to model the posterior distribution of the demonstrator's behavior given the demonstration dataset, we do obtain a policy that ensures coverage over the demonstrator's actions, enabling more effective finetuning. Furthermore, this policy achieves this while ensuring pretrained performance is no worse than that of the BC policy. We then show this approach is practically implementable with modern generative models and leads to significantly improved RL finetuning performance on both realistic robotic control benchmarks and real-world robotic manipulation tasks, as compared to standard behavioral cloning.

Forum: https://openreview.net/forum?id=HJ5F3R9Tmd

---

## 239. Rotary Position Encodings for Graphs

**Abstract:** We study the extent to which rotary position encodings (RoPE), a recent transformer position encoding algorithm broadly adopted in large language models (LLMs) and vision transformers (ViTs), can be applied to graph-structured data. We find that rotating tokens depending on the spectrum of the graph Laplacian efficiently injects structural information into the attention mechanism, boosting performance in synthetic and real-world graph learning tasks. This approach, coined _Wave-Induced Rotary Encodings_ (WIRE), enjoys intriguing theoretical properties: it recovers regular RoPE on grids, and depends asymptotically on the graph effective resistance. Unlike bias-based relative position encodings, WIRE is compatible with linear attention.

Forum: https://openreview.net/forum?id=trn64znfNx

---

## 240. BrokenMath: A Benchmark for Sycophancy in Theorem Proving with LLMs

**Abstract:** Large language models (LLMs) have recently shown strong performance on mathematical benchmarks. At the same time, they are prone to hallucination and sycophancy, often providing convincing but flawed proofs for incorrect mathematical statements provided by users. This significantly limits the applicability of LLMs in theorem proving, as verification of these flawed proofs must be done manually by expert mathematicians. However, existing benchmarks that measure sycophancy in mathematics are limited: they focus solely on final-answer problems, rely on very simple and often contaminated datasets, and construct benchmark samples using synthetic modifications that create ill-posed questions. To address these issues, we introduce BrokenMath, the first benchmark for evaluating sycophantic behavior in LLMs within the context of natural language theorem proving. BrokenMath is built from advanced 2025 competition problems, which are perturbed with an LLM to produce false statements and subsequently refined through expert review. We evaluate state-of-the-art LLMs and agentic systems and find that sycophancy is widespread, with the best model, GPT-5, producing sycophantic answers 29% of the time. We further investigate several mitigation strategies, including test-time interventions and supervised fine-tuning on curated sycophantic examples. These approaches reduce, but do not eliminate, sycophancy.

Forum: https://openreview.net/forum?id=Y5bEI4cuMd

---

## 241. Self-Distillation Enables Continual Learning

**Abstract:** Continual learning, enabling models to acquire new skills and knowledge without degrading existing capabilities, remains a fundamental challenge for foundation models. While on-policy reinforcement learning can reduce forgetting, it requires explicit reward functions that are often unavailable. Learning from expert demonstrations, the primary alternative, is dominated by supervised fine-tuning (SFT), which is inherently off-policy. We introduce Self-Distillation Fine-Tuning (SDFT), a simple method that enables on-policy learning directly from demonstrations. SDFT leverages in-context learning by using a demonstration-conditioned model as its own teacher, generating on-policy training signals that preserve prior capabilities while acquiring new skills. Across skill learning and knowledge acquisition tasks, SDFT consistently outperforms SFT, achieving higher new-task accuracy while substantially reducing catastrophic forgetting. In sequential learning experiments, SDFT enables a single model to accumulate multiple skills over time without performance regression, establishing on-policy distillation as a practical path to continual learning from demonstrations.

Forum: https://openreview.net/forum?id=qA6FgH0nnZ

---

## 242. Information dynamics and Memory in Neural Networks through Fisher Information Diffusion

**Abstract:** We present a general theoretical framework for analyzing how information about past inputs is encoded in recurrent networks into evolving dynamics rather than being represented as convergence to static attractors. Using dynamic mean-field theory and diffusion from physics, we derive a Fisher information diffusion operator that links network connectivity structure to the time-resolved propagation of information across interacting subpopulations. The analysis reveals that operating near criticality (spectral radius near one) is necessary but not sufficient for reliable memory in structured or non-normal recurrent networks; effective information retention requires alignment between input–output structure and stable dynamical subspaces. The theory yields principled initialization rules that balance stability and sensitivity, mitigating vanishing and exploding gradients. Experiments on the copy task and sequential MNIST show faster convergence and higher accuracy than standard random initialization. Together, these results provide both principled design guidelines for recurrent networks and new theoretical insight into how information can be preserved over time in their dynamics.

Forum: https://openreview.net/forum?id=cvfjjRCY0U

---

## 243. AlgoVeri: An Aligned Benchmark for Verified Code Generation on Classical Algorithms

**Abstract:** Vericoding refers to the generation of formally verified code from rigorous specifications. Recent AI models show promise in vericoding, but a unified methodology for cross-paradigm evaluation is lacking. Existing benchmarks test only an individual language/tool (e.g., Dafny, Verus, and Lean) and each covers very different tasks, so the performance numbers are not directly comparable. We address this gap with AlgoVeri, a benchmark that evaluates vericoding of $77$ classical algorithms in each of Dafny, Verus, and Lean. By enforcing identical functional contracts, AlgoVeri reveals critical capability gaps in current models. While frontier models achieve tractable success in Dafny ($40.3$\% for Gemini-3 Flash), where high-level abstractions and SMT automation simplify the workflow, performance collapses under the systems-level memory constraints of Verus ($24.7$\%) and the explicit proof construction required by Lean ($7.8$\%). Beyond aggregate metrics, we uncover a sharp divergence in test-time compute dynamics: Gemini-3 effectively utilizes iterative repair to boost performance (e.g., tripling pass rates in Dafny), whereas GPT-OSS saturates early. Finally, our error analysis shows that language design affects the refinement trajectory: while Dafny allows models to focus on logical correctness, Verus and Lean trap models in persistent syntactic and semantic barriers.

Forum: https://openreview.net/forum?id=mnUgulPmNU

---

## 244. Robust Contextual Optimization with Missing Covariates

**Abstract:** Modern decision-making increasingly relies on contextual features (covariates) to improve optimization under uncertainty. In practice, however, such covariates are often only partially observed due to, e.g., data source heterogeneity or costly data collection. Nonetheless, most existing methods assume fully observed historical data and can become unreliable when this assumption is violated. We address this gap by proposing a distributionally robust optimization approach that exploits incomplete covariates to produce robust decisions without imputing a complete dataset. Our method builds ambiguity sets from the observed partial data and incorporates the general structure of the missingness mechanism, ensuring candidate distributions remain consistent with what is observed. Across settings with discrete or continuous covariates and outcomes, we derive tractable reformulations and establish finite-sample out-of-sample performance guarantees. Empirical results across a range of contextual decision-making tasks demonstrate that the proposed integrated approach consistently outperforms state-of-the-art baselines, including various impute-then-optimize pipelines, in both out-of-sample performance and reliability.

Forum: https://openreview.net/forum?id=37KrsS7g7c

---

## 245. Disentangling Latent Risk Pathways via Bayesian Hypergraph Inference

**Abstract:** Electronic health records (EHR) pose large-scale multi-disease modeling problems in which many outcomes are rare and strongly influenced by shared risk factors. While modern approaches achieve strong predictive performance, they often treat diseases independently or rely on black-box architectures, offering limited insight into how risk factors organize disease risk and little principled uncertainty quantification.
We introduce a Bayesian hypergraph inference framework that reframes multi-disease modeling around **latent, risk-factor-modulated disease pathways**. Risk factors act on hyperedges, latent disease subsets with shared risk patterns, allowing diseases to participate in multiple distinct pathways and enabling interpretable, higher-order structure beyond pairwise associations. A repulsion prior encourages parsimonious and identifiable structure, while posterior inference provides calibrated uncertainty over both disease groupings and risk-factor influence. 
To enable scalable inference on large EHR datasets, we develop a structured variational inference algorithm that preserves logical dependencies among hyperedge existence, disease membership, and pathway-level effects. 
Experiments on simulated data and UK Biobank demonstrate stable and interpretable disease pathway structure, well-calibrated uncertainty, improved estimation for rare diseases, and competitive predictive performance.

Forum: https://openreview.net/forum?id=vNfbqRzash

---

## 246. Long-Context Modeling with Dynamic Hierarchical Sparse Attention for Memory-Constrained LLM Inference

**Abstract:** The quadratic cost of attention limits the scalability of long-context LLMs, especially under limited hardware memory budgets. While attention is often sparse, existing static sparse methods cannot adapt to task- or input-dependent variations, and recent dynamic approaches rely on predefined templates or heuristics that may sacrifice generality. We propose Dynamic Hierarchical Sparse Attention (DHSA), a data-driven framework that predicts attention sparsity online while keeping the LLM backbone frozen. DHSA performs hierarchical routing by estimating importance at the chunk level and propagating it to token-level interactions, preserving causally important dependencies while enabling efficient sparsification. Across Needle-in-a-Haystack test, LongBench and RULER, DHSA maintains near-dense accuracy in highly sparse regimes, achieving 12--20% relative accuracy gains over Block Sparse Attention at comparable prefill cost. With a memory-efficient tiled backend, DHSA delivers up to $10\times$ prefill speedup at 128K context length. On LLaMA-3.1-8B (4-bit), DHSA scales to 100K context on a single 24GB GPU, where dense attention fails. We provide complementary GPU and CPU backends, enabling DHSA to run across diverse hardware environments and multiple open-weight model families. These results demonstrate DHSA as an efficient and adaptable solution for memory-constrained long-context LLM inference. Code is available at: https://github.com/xiongsiheng/DHSA.

Forum: https://openreview.net/forum?id=o3gN27ITWV

---

## 247. Wait, Wait, Wait... Why Do Reasoning Models Loop?

**Abstract:** Reasoning models (e.g., DeepSeek-R1) generate long chains of thought to solve harder problems, but they often loop, repeating the same text at low temperatures or with greedy decoding. We study why this happens and what role temperature plays. With open reasoning models, we find that looping is common at low temperature. Larger models tend to loop less, and distilled students loop significantly even when their teachers rarely do. This points to mismatches between the training distribution and the learned model, which we refer to as errors in learning, as a key cause. To understand how such errors cause loops, we introduce a synthetic graph reasoning task and demonstrate two mechanisms. First, risk aversion caused by hardness of learning: when the correct progress-making action is hard to learn but an easy cyclic action is available, the model puts relatively more probability on the cyclic action and gets stuck. Second, even when there is no hardness, Transformers show an inductive bias toward temporally correlated errors, so the same few actions keep being chosen and loops appear. Higher temperature reduces looping by promoting exploration, but it does not fix the errors in learning, so generations remain much longer than necessary at high temperature; in this sense, temperature is a stopgap rather than a holistic solution. We end with a discussion of training-time interventions aimed at directly reducing errors in learning.

Forum: https://openreview.net/forum?id=oZWE7mSqlk

---

## 248. MEnvAgent: Scalable Polyglot Environment Construction for Verifiable Software Engineering

**Abstract:** The evolution of Large Language Model (LLM) agents for software engineering (SWE) is constrained by the scarcity of verifiable datasets, a bottleneck stemming from the complexity of constructing executable environments across diverse languages. To address this, we introduce **MEnvAgent**, a **M**ulti-language framework for automated **Env**ironment construction that facilitates scalable generation of verifiable task instances. MEnvAgent employs a multi-agent Planning-Execution-Verification architecture to autonomously resolve construction failures and integrates a novel Environment Reuse Mechanism that reduces computational overhead by incrementally patching historical environments. Evaluations on MEnvBench, a new benchmark comprising 1,000 tasks across 10 languages, demonstrate that MEnvAgent outperforms baselines, improving Fail-to-Pass (F2P) rates by **8.6%** while reducing time costs by **43%**. Additionally, we demonstrate the utility of MEnvAgent by constructing MEnvData-SWE, the largest open-source polyglot dataset of realistic verifiable Docker environments to date, alongside solution trajectories that enable consistent performance gains on SWE tasks across a wide range of models.

Forum: https://openreview.net/forum?id=Mkal0hTCnh

---

## 249. Rex: A Family of Reversible Exponential (Stochastic) Runge-Kutta Solvers

**Abstract:** Deep generative models based on neural differential equations have become state-of-the-art for many generation tasks.
These models rely on ODE/SDE solvers that integrate from a prior distribution to the data distribution; in many applications it is also highly desirable to integrate in the inverse direction.
Standard solvers, however, accumulate discretization errors that prohibit *exact inversion*, an inaccuracy that is unacceptable in precision-critical applications.
Existing inversion methods suffer from poor stability and low order of convergence, and are strictly limited to the ODE setting.
In this work, we propose *Rex*, a family of reversible exponential (stochastic) Runge-Kutta solvers obtained by applying Lawson methods to convert any explicit (stochastic) Runge-Kutta scheme into an algebraically reversible one for both diffusion ODEs *and* SDEs.
Beyond a rigorous theoretical analysis---establishing arbitrary-order convergence and a non-zero region of linear stability---we empirically demonstrate that *Rex* achieves near-machine-precision reconstruction and improves Boltzmann sampling with flow models as well as image generation and editing with diffusion models.

Forum: https://openreview.net/forum?id=7pQIzVNctu

---

## 250. Autoregressive Boltzmann Generators

**Abstract:** Efficient sampling of molecular systems at thermodynamic equilibrium is a hallmark challenge in statistical physics. This challenge has driven the development of Boltzmann Generators (BGs), which allow rapid generation of uncorrelated equilibrium samples by combining a generative model with exact likelihoods and an importance sampling correction. However, modern BGs predominantly rely on Normalizing Flows (NFs), which either suffer from limited expressivity due to strict invertibility constraints (discrete time) or computationally expensive likelihoods (continuous time). In this paper, we propose Autoregressive Boltzmann Generators (ArBG), a novel autoregressive modelling framework that overcomes these limitations by departing from the flow-based BG paradigm. ArBG circumvents the topological constraints of flows and enables sequential inference-time interventions, while offering enhanced scalability by leveraging architectures effective in Large Language Models. We empirically demonstrate that ArBG leads to significant improvements over flow-based models across all benchmarks, but particularly in larger peptide systems such as the 10-residue Chignolin. Furthermore, we introduce Robin, a 132M parameter transferable model trained with the ArBG framework which improves over the previous state-of-the-art, reducing the zero-shot energy error, $\mathcal{E}$-$\mathcal{W}_2$, on 8-residue systems by $\sim 60$\%.

Forum: https://openreview.net/forum?id=75AYDsndHP

---

## 251. Large Language Models Develop Novel Social Biases Through Adaptive Exploration

**Abstract:** As large language models (LLMs) are adopted into frameworks that grant them the capacity to make real decisions, it is increasingly important to ensure that they are unbiased. In this paper, we argue that the predominant approach of simply removing existing biases from models is not enough. Using a paradigm from the psychology literature, we demonstrate that LLMs can spontaneously develop novel social biases about artificial demographic groups even when no inherent differences exist. These biases result in highly stratified task allocations, which are less fair than assignments by human participants and are exacerbated by newer and larger models. In humans, emergent biases like these have been shown to result from exploration-exploitation trade-offs, where the decision-maker explores too little, allowing early observations to strongly influence impressions about entire demographic groups. To alleviate this effect, we examine a series of interventions targeting model inputs, problem structure, and explicit steering. We find that explicitly incentivizing exploration most robustly reduces stratification, highlighting the need for better multifaceted objectives to mitigate bias. These results reveal that LLMs are not merely passive mirrors of human social biases, but can actively create new ones from experience, raising urgent questions about how these systems will shape societies over time.

Forum: https://openreview.net/forum?id=pc7fqaOcAH

---

## 252. Rate or Fate? RLV$^{\varepsilon}$R: Reinforcement Learning with Verifiable Noisy Rewards

**Abstract:** Reinforcement learning with verifiable rewards (RLVR) trains a policy by verifying sampled completions and reinforcing higher-scoring outputs, but practical verifiers (e.g., incomplete unit tests or noisy judges) are prone to false positives and false negatives.
We ask when such noise merely slows learning and when it reverses it.
Modeling GRPO-style RLVR as a bandit over recurring \emph{reasoning modes}, we derive mean-field replicator-style (natural-selection) flow on the probability simplex. The dynamics decouples into within-correct-mode competition and a one-dimensional evolution for the mass on incorrect modes, whose drift is determined solely by Youden's index $J=\mathrm{TPR}-\mathrm{FPR}$. This yields a sharp phase transition: when $J>0$, the incorrect mass is driven toward extinction (learning); when $J=0$, the process is neutral; and when $J<0$, incorrect modes amplify until they dominate (anti-learning and collapse). In the learning regime $J>0$, noise primarily rescales convergence time (``rate, not fate''). Experiments on verifiable programming tasks under synthetic noise reproduce the predicted $J=0$ boundary. Beyond noise, the framework offers a general lens for analyzing RLVR stability, convergence, and algorithmic interventions.

Forum: https://openreview.net/forum?id=LwB2EacVT6

---

## 253. Maximum Likelihood Reinforcement Learning

**Abstract:** Reinforcement learning (RL) is the method of choice for training models in setups where the objective function can only be evaluated by sampling from the model. Our key observation is that when the feedback is terminal and binary, models implicitly induce a likelihood over correct rollouts. Maximum likelihood would be the natural framework in such settings, but RL is used instead as a workaround to the non-differentiability. We prove that the standard, expected-reward RL formulation is only a first-order approximation of the likelihood. To remedy this mismatch, we introduce **Maximum Likelihood Reinforcement Learning (MaxRL)**, a compute-indexed family of sample-based objectives that interpolate between expected-reward RL and maximum likelihood as sampling compute is scaled. The resulting objective is a one-line change to standard RL implementations. MaxRL Pareto-dominates existing methods in all tested models and tasks, achieves up to $\mathbf{20\times}$ gains in test-time scaling efficiency over GRPO, and scales more favorably with additional training data and compute.

Forum: https://openreview.net/forum?id=EeuLO2BjFN

---

## 254. On the Sharp Input-Output Analysis of Nonlinear Systems under Adversarial Attacks

**Abstract:** This paper is concerned with learning the input-output mapping of general nonlinear dynamical systems. While the existing literature focuses on Gaussian inputs and benign disturbances, we significantly broaden the scope of admissible control inputs and allow correlated, nonzero-mean, adversarial disturbances. With our reformulation as a linear combination of basis functions, we prove that the $\ell_2$-norm estimator overcomes the challenges posed by an adversary with access to the full information history, provided that the attack times are sparse, *i.e.*, the probability that the system is under adversarial attack at a given time is smaller than a certain threshold. We provide an estimation error bound that decays with the input memory length and prove its optimality by constructing a problem instance that suffers from the same bound under probabilistic adversarial attacks. Our work provides a sharp input-output analysis for a generic nonlinear and partially observed system under significantly generalized assumptions compared to existing works.

Forum: https://openreview.net/forum?id=xrUgA8PrNN

---

## 255. Safety Alignment of LMs via Non-cooperative Games

**Abstract:** Ensuring the safety of language models (LMs) while maintaining their usefulness remains a critical challenge in AI alignment. Current approaches rely on sequential adversarial training: generating adversarial prompts and fine-tuning LMs to defend against them. We introduce a different paradigm: framing safety alignment as a non-zero-sum game between an Attacker LM and a Defender LM trained jointly via online reinforcement learning. Each LM continuously adapts to the other's evolving strategies, driving iterative improvement. Our method uses a preference-based reward signal derived from pairwise comparisons instead of point-wise scores, providing more robust supervision and potentially reducing reward hacking. Our RL recipe, AdvGame, shifts the Pareto frontier of safety and utility, yielding a Defender LM that is simultaneously more helpful and more resilient to adversarial attacks. In addition, the resulting Attacker LM converges into a strong, general-purpose red-teaming agent that can be directly deployed to probe arbitrary target models. Code at github.com/facebookresearch/advgame.

Forum: https://openreview.net/forum?id=Bve790HQrA

---

## 256. ThunderAgent: A Fast, Simple, and Program-Aware Agentic Inference System

**Abstract:** Large language models (LLMs) are now used to power complex multi-turn agentic workflows. Existing systems run agentic inference by loosely assembling isolated components: an LLM inference engine (e.g., vLLM) and a tool orchestrator (e.g., Kubernetes). Although agentic workflows involve multiple LLM and tool requests, these systems schedule and allocate resources separately on a per-request basis, without end-to-end knowledge of the workflow. This leads to sub-optimal management of KV cache and tool execution environments. To address the challenges, we propose **ThunderAgent**, a fast, simple, and program-aware agentic inference system. We first abstract agentic workflows as ***LLM Programs***, enabling a unified view of heterogeneous resources, including KV caches, system states, and external tool assets such as disk memory and network ports. Built upon this abstraction, ThunderAgent introduces a program-aware scheduler and a tool resource manager designed to maximize KV cache hit rates, mitigate memory imbalances, and enable asynchronous environment preparation. Evaluations across coding, routing, and scientific discovery agents demonstrate that ThunderAgent achieves **1.5-3.6×** throughput improvements in serving, **1.8-3.9×** in RL rollout, and up to **4.2×** disk memory savings compared to state-of-the-art inference systems. To facilitate reproducibility and support future development, we open-source the system implementations of ThunderAgent at: https://github.com/ThunderAgent-org/ThunderAgent

Forum: https://openreview.net/forum?id=kR4iOTaAOJ

---

## 257. Single-Head Attention in High Dimensions: A Theory of Generalization, Weights Spectra, and Scaling Laws

**Abstract:** Trained attention layers exhibit striking and reproducible spectral structure of the weights, including low-rank collapse, bulk deformation, and isolated spectral outliers, yet the origin of these phenomena and their implications for generalization remain poorly understood. We study empirical risk minimization in a single-head tied-attention layer trained on synthetic high-dimensional sequence tasks generated from the attention-indexed model. Using tools from random matrix theory, spin-glass theory, and approximate message passing, we obtain an exact high-dimensional characterization of training and test error, interpolation and recovery thresholds, and the spectrum of the key and query matrices. Our theory predicts the full singular-value distribution of the trained query–key map—including low-rank structure and isolated spectral outliers—in qualitative agreement with observations in more realistic transformers. Finally, for targets with power-law spectra, we show that learning proceeds through sequential spectral recovery, leading to the emergence of power-law scaling laws.

Forum: https://openreview.net/forum?id=3qan4Zg9rA

---

## 258. Path-dependent Discrete Amortized Inference

**Abstract:** We consider the problem of sampling compositional and discrete objects from a given unnormalized posterior distribution. 
Notably, recent studies have shown that this problem can be efficiently solved by learning a deterministic Markov Decision Process (MDP) that progressively builds each object in proportion to the posterior.
In this work, however, we demonstrate that the Markovian assumption can both hamper signal propagation during training and catastrophically reduce the learned sampler's expressivity due to state aliasing. 
To address these issues, we propose lifting the MDP with a learnable latent dynamics that allows the underlying policy to depend on the entire past trajectory---and not only on the current state. 
In view of this, we refer to the resulting method as \emph{path-dependent discrete amortized inference}. 
Importantly, we provably extend existing learning algorithms for amortized samplers to our setting. 
In experiments on standard benchmark problems, we also show that our approach often leads to faster learning convergence and improved state space exploration relatively to prior techniques.

Forum: https://openreview.net/forum?id=viJvYPJHIj

---

## 259. Universal Redundancies in Time Series Foundation Models

**Abstract:** Time Series Foundation Models (TSFMs) leverage extensive pretraining to accurately predict unseen time series during inference, without the need for task-specific fine-tuning. Through large-scale evaluations of standard benchmarks, we find that leading transformer-based TSFMs exhibit redundant components in their intermediate layers. We introduce a set of tools for mechanistic interpretability of TSFMs, including ablations of specific components and direct logit attribution on the residual stream. Our findings are consistent across several leading TSFMs with diverse architectures, and across a diverse set of real-world and synthetic time-series datasets. We discover that all models in our study are robust to ablations of entire layers. Furthermore, we develop a theoretical framework framing transformers as kernel regressors, motivating a purely intrinsic strategy for ablating heads based on the stable rank of the per-head projection matrices. Using this approach, we uncover the specific heads responsible for degenerate phenomena widely observed in TSFMs, such as parroting of motifs from the context and seasonality bias. Our study sheds light on the universal properties of this emerging class of architectures for continuous-time sequence modeling.

Forum: https://openreview.net/forum?id=DyA4KHj1wy

---

## 260. To Grok Grokking: Provable Grokking in Ridge Regression

**Abstract:** We study *grokking* — the onset of generalization long after overfitting — in a classical ridge regression setting. We prove end-to-end grokking results for learning over-parameterized linear regression models using gradient descent with weight decay. Specifically, we prove that the following stages occur: (i) the model overfits the training data early during training; (ii) poor generalization persists long after overfitting has manifested; and (iii) the generalization error eventually becomes arbitrarily small. Moreover, we show, both theoretically and empirically, that grokking can be amplified or eliminated in a principled manner through proper hyperparameter tuning. To the best of our knowledge, these are the first rigorous quantitative bounds on the generalization delay (which we refer to as the "grokking time") in terms of training hyperparameters. Lastly, going beyond the linear setting, we empirically demonstrate that our quantitative bounds also capture the behavior of grokking on non-linear neural networks. Our results suggest that grokking is not an inherent failure mode of deep learning, but rather a consequence of specific training conditions, and thus does not require fundamental changes to the model architecture or learning algorithm to avoid.

Forum: https://openreview.net/forum?id=5nNNVY8NW4

---

## 261. BlitzRank: Principled Zero-shot Ranking Agents with Tournament Graphs

**Abstract:** Selecting the top $m$ from $n$ items via expensive $k$-wise comparisons is central to settings ranging from LLM-based document reranking to crowdsourced evaluation and tournament design.
Existing methods either rely on heuristics that discard comparison information, or exploit it at prohibitive cost.
We introduce a *tournament graph* framework that provides a principled foundation for $k$-wise ranking.
Our key observation is that each $k$-item comparison reveals an induced tournament of ${k \choose 2}$ pairwise preferences; aggregating these into a global preference graph and computing its transitive closure yields many additional orderings without further oracle calls.
We formalize when the current top-$m$ output is *certifiably determined* and design a greedy query schedule that maximizes information gain towards identifying the top-$m$ items.
The framework also gracefully handles non-transitive preferences -- cycles induced by real-world oracles -- by collapsing them into equivalence classes that yield principled *tiered rankings*.
Applied to LLM reranking across 14 benchmarks and 5 models, BlitzRank achieves Pareto dominance over existing approaches: matching or exceeding accuracy while requiring 25--40\% fewer tokens than comparable methods; against pairwise reranking, it achieves near-identical quality with 7$\times$ fewer tokens. Code available at
[https://github.com/ContextualAI/BlitzRank](https://github.com/ContextualAI/BlitzRank).

Forum: https://openreview.net/forum?id=dc9si7lBTY

---

## 262. (Doubly) Exponential Lower Bounds for Follow the Regularized Leader in Potential Games

**Abstract:** Follow the regularized leader (FTRL) is the premier algorithm for online optimization. However, despite decades of research on its convergence in constrained optimization---and potential games in particular---its behavior remained hitherto poorly understood. In this paper, we establish that FTRL can take exponential time to converge to a Nash equilibrium in two-player potential games for any (permutation-invariant) regularizer and potentially vanishing learning rate. By known equivalences, this translates to an exponential lower bound for certain mirror descent counterparts, most notably multiplicative weights update. On the positive side, we establish the potential property for FTRL and obtain an exponential upper bound $\exp(O_{\epsilon}(1/\epsilon^2))$ for any no-regret dynamics executed in a lazy, alternating fashion, matching our lower bound up to factors in the exponent. Finally, in multi-player potential games, we show that fictitious play---the extreme version of FTRL---can take doubly exponential time to reach a Nash equilibrium. This constitutes an exponentially stronger lower bound for the foundational learning algorithm in games.

Forum: https://openreview.net/forum?id=l6KZJO7w48

---

## 263. SurvDiff: A Diffusion Model for Generating Synthetic Data in Survival Analysis

**Abstract:** Survival analysis is a cornerstone of clinical research by modeling time-to-event outcomes such as metastasis, disease relapse, or patient death. Unlike standard tabular data, survival data often come with incomplete event information due to dropout, or loss to follow-up. This poses unique challenges for synthetic data generation, where it is crucial for clinical research to faithfully reproduce both the event-time distribution and the censoring mechanism. In this paper, we propose SurvDiff, an end-to-end diffusion model specifically designed for generating synthetic data in survival analysis. SurvDiff is tailored to capture the data-generating mechanism by jointly generating mixed-type covariates, event times, and right-censoring, guided by a survival-tailored loss function. The loss encodes the time-to-event structure and directly optimizes for downstream survival tasks, which ensures that SurvDiff (i) reproduces realistic event-time distributions and (ii) preserves the censoring mechanism. Across multiple datasets, we show that SurvDiff outperforms state-of-the-art generative baselines in both distributional fidelity and survival model evaluation metrics across multiple medical datasets. To the best of our knowledge, SurvDiff is the first end-to-end diffusion model explicitly designed for generating synthetic survival data.

Forum: https://openreview.net/forum?id=boeY2syj2r

---

## 264. LiME: Lightweight Mixture of Experts for Efficient Multimodal Multi-task Learning

**Abstract:** MoE-PEFT methods combine Mixture of Experts with parameter-efficient fine-tuning for multi-task adaptation, but require separate adapters per expert—causing trainable parameters to scale linearly with expert count and limiting applicability to adapter-based architectures. We propose LiME (Lightweight Mixture of Experts), which achieves expert specialization through lightweight modulation rather than adapter replication. Instead of separate adapters, LiME uses a single shared PEFT module and modulates its output with lightweight expert vectors, reducing expert parameters while generalizing to any PEFT method. Notably, LiME introduces zero-parameter routing by leveraging existing frozen and adapted representations—eliminating learned router parameters typically required per layer. Theoretically, we prove that (i) more experts preserve more task-relevant information and (ii) modulation approximates full expert-specific PEFT with bounded error. LiME further incorporates n-gram windowed routing and adaptive expert selection (Auto Top-K) based on routing confidence. Experiments on MMT-47, a multimodal multi-task benchmark with 47 tasks spanning text, image, and video, demonstrate that LiME achieves competitive or superior performance while using up to 4× fewer trainable parameters and up to 29% faster training compared to corresponding MoE-PEFT baselines.

Forum: https://openreview.net/forum?id=KRSZj8z5Lr

---

## 265. Optimal Rates for Feasible Payoff Set Estimation in Games

**Abstract:** We study a setting in which two players play a (possibly approximate) Nash equilibrium of a bimatrix game, while a learner observes only their actions and has no knowledge of the equilibrium or the underlying game. A natural question is whether the learner can rationalize the observed behavior by inferring the players' payoff functions. Rather than producing a single payoff estimate, inverse game theory aims to identify the entire set of payoffs consistent with observed behavior, enabling downstream use in, e.g., counterfactual analysis and mechanism design across applications like auctions, pricing, and security games. We focus on the problem of estimating the set of feasible payoffs with high probability and up to precision $\epsilon$ on the Hausdorff metric. We provide the first minimax-optimal rates for both exact and approximate equilibrium play, in zero-sum as well as general-sum games. Our results provide learning-theoretic foundations for set-valued payoff inference in multi-agent environments.

Forum: https://openreview.net/forum?id=qV7yFzdK1R

---

## 266. Learning Human-Robot Collaboration via Heterogeneous-Agent Lyapunov Policy Optimization

**Abstract:** To improve generalization and resilience in human–robot collaboration (HRC), robots must contend with diverse combinations of human behaviors and contexts, motivating multi-agent reinforcement learning (MARL). However, inherent heterogeneity between robots and humans creates a rationality gap (RG), where decentralized policy updates deviate from cooperative joint optimization. The resulting learning problem is a general-sum differentiable game, so independent policy-gradient updates can oscillate or diverge without added structure. We propose heterogeneous-agent Lyapunov policy optimization (HALO), a framework that stabilizes decentralized MARL by enforcing Lyapunov-based contraction in policy-parameter space. Unlike Lyapunov-based safe RL, which targets state/trajectory constraints in constrained Markov decision processes, HALO uses Lyapunov certification to stabilize decentralized policy learning. HALO rectifies decentralized gradients via optimal quadratic projections, ensuring monotonic contraction of RG and enabling effective exploration of open-ended interaction spaces. Extensive simulations and real-world humanoid-robot experiments show that this certified stability improves generalization and robustness in collaborative corner cases.

Forum: https://openreview.net/forum?id=uBvlUE2wrP

---

## 267. Local Mechanisms of Compositional Generalization in Conditional Diffusion

**Abstract:** Conditional diffusion models appear capable of compositional generalization, i.e., generating convincing samples for out-of-distribution combinations of conditioners, but the mechanisms underlying this ability remain unclear. To make this concrete, we study length generalization, the ability to generate images with more objects than seen during training. In a controlled CLEVR setting (Johnson et al., 2017), we find that length generalization is achievable in some cases but not others, suggesting that models only sometimes learn the underlying compositional structure. We then investigate locality as a structural mechanism for compositional generalization. Prior works proposed score locality as a mechanism for creativity in unconditional diffusion models (Kamb & Ganguli, 2024; Niedoba et al., 2024), but did not address flexible conditioning or compositional generalization. In this paper, we prove an exact equivalence between a specific compositional structure (*conditional projective composition*) (Bradley et al., 2025) and scores with sparse dependencies on both pixels and conditioners (*local conditional scores*). This theory also extends to compositions of concepts (such as style+content) in feature-space. We validate our theory empirically: CLEVR models that succeed at length generalization exhibit local conditional scores, while those that fail do not. Furthermore, we show that a causal intervention explicitly enforcing local conditional scores enables length generalization in a previously failing model. Finally, we investigate feature-space compositionality in color-conditioned CLEVR, and find preliminary evidence of compositional structure and corresponding local mechanisms in SDXL.

Forum: https://openreview.net/forum?id=XsfjvLhKAP

---

## 268. DecodeShare: Tracing the Shared Subspace of LLM Decode-Time Decisions

**Abstract:** Large language models (LLMs) handle many tasks with one set of parameters, but under KV-cached inference it is unclear what task-general structure, if any, is used at $\textit{decode time}$ rather than during $\textit{prefill}$. We propose $\textbf{DecodeShare}$, a protocol that identifies a low-dimensional subspace consistently shared across tasks in decode-time hidden states, and then tests its causal role by removing that subspace only during decoding. In our experiments, disturbing the discovered shared subspace degrades decision performance far more than disturbing either a prefill-derived or random subspace under the same intervention budget. We further show this decode-shared subspace has practical consequences for activation steering: common steering directions can overlap the task-general decode channel. Projecting out this shared subspace directly separates the functional roles of the two components, while evaluating steering vectors at decode-time yields more reliable signal for downstream deployment than prefill-based proxies. Despite its compactness, the shared subspace can serve as a high-leverage causal channel at decode time. Code is available at: https://github.com/Zishan-Shao/decodeshare.git.

Forum: https://openreview.net/forum?id=txdOohIkYJ

---

## 269. EcoVLA: Environment-Aware Adaptive Pruning with Interleaved Inference Orchestration for Vision-Language-Action Models

**Abstract:** While Vision-Language-Action (VLA) models hold promise in embodied intelligence, their large parameter counts lead to substantial inference latency that hinders real-time manipulation, motivating parameter sparsification. However, as the environment evolves during VLA execution, the optimal sparsity patterns change accordingly. Static pruning lacks the adaptability required for environment dynamics, whereas fixed-interval dynamic layer pruning suffers from coarse granularity and high retraining overheads. To bridge this gap, we propose **EcoVLA**, a training-free, plug-and-play adaptive pruning framework that supports orthogonal combination with existing VLA acceleration methods. EcoVLA comprises two components: **E**nvironment-aware **A**daptive **P**runing (**EAP**) and **I**nterleaved **I**nference **O**rchestration (**$I^2O$**). EAP is a lightweight adaptive channel pruning method that incorporates the temporal consistency of the physical environment to update sparsity patterns. $I^2O$ leverages the FLOPs bubbles inherent in VLA inference to schedule the pruning method in parallel, ensuring negligible impact on latency. Evaluated on diverse VLA models and benchmarks, EcoVLA delivers state-of-the-art performance, achieving up to 1.60$\times$ speedup with only a 0.4% drop in success rate, and further reaches 2.18$\times$ speedup with only a 0.5% degradation when combined with token pruning. We further validate the effectiveness of EcoVLA on real-world robots. Our code is available [here](https://github.com/Echo-hyt/Ecovla).

Forum: https://openreview.net/forum?id=7kH5uphAes

---

## 270. A Unifying View of Variational Generative Wasserstein Flows

**Abstract:** Many modern generative models can be viewed as minimizing divergences between probability distributions, yet they rely on different algorithmic and geometric principles. Wasserstein gradient flows provide a continuous-time formulation for optimizing over distributions, and can be approximated through their implicit discretization via the Jordan–Kinderlehrer–Otto (JKO) scheme. In this work, we present a unified theoretical framework for generative modeling based on Wasserstein gradient flows, which we refer to as Generative Wasserstein Flows (GWF). We show that a broad class of existing methods can be derived as instances of parametric JKO schemes for $f$-divergence objectives, and we establish equivalences between several recently proposed algorithms. We extend this framework beyond $f$-divergences to Integral Probability Metrics and squared Maximum Mean Discrepancy, deriving new JKO-based generative algorithms, and clarifying their connections with GANs. We study empirically the impact of the JKO regularization for a wide set of objectives. Finally, we analyze parametric Wasserstein flows, where the dynamics are restricted to distributions induced by parametrized maps.

Forum: https://openreview.net/forum?id=sJ7ngz2eQx

---

## 271. Linguistic Properties and Model Scale in Brain Encoding: From Small to Compressed Language Models

**Abstract:** Recent work has shown that scaling large language models (LLMs) improves their alignment with human brain activity, yet it remains unclear what drives these gains or which representational properties are responsible. Although larger models often yield better task performance and brain alignment, they are  increasingly difficult to analyze mechanistically. This raises a fundamental question: \emph{what is the minimal model capacity required to capture brain-relevant representations?} To address this question, we systematically investigate how constraining model scale and numerical precision affects brain alignment. We compare full-precision LLMs, small language models (SLMs), and compressed variants (quantized and pruned) by predicting fMRI responses during naturalistic language comprehension. Across model families up to 14B parameters, we find that 3B SLMs achieve brain predictivity indistinguishable from larger LLMs, whereas 1B models degrade substantially, particularly in semantic language regions. Brain alignment is remarkably robust to compression: most quantization and pruning methods preserve neural predictivity, with GPTQ as a consistent exception. Linguistic probing reveals a dissociation between task performance and brain predictivity: compression degrades discourse, syntax, and morphology, yet brain predictivity remains largely unchanged.  Overall, brain alignment saturates at modest model scales and is resilient to compression, challenging common assumptions about neural scaling and motivating compact models for brain-aligned language modeling.

Forum: https://openreview.net/forum?id=WK1NvxRMsL

---

## 272. On the Accuracy of Newton Step and Influence Function Data Attributions

**Abstract:** Data attribution estimates how a trained model would change if a subset of training points were removed, and is a central primitive for tasks such as interpretability, data valuation, and machine unlearning. Despite its widespread use, our theoretical understanding of key data attribution methods -- Influence Functions (IF) and a single Newton Step (NS) -- remains limited: existing guarantees heavily rely on *global* strong convexity and yield bounds with pessimistic dependence on the parameter dimension $d$ and the number of removed samples $k$.
We give a new analysis of IF and NS for convex ERM that replaces global assumptions with *local* conditions: it suffices that the loss is strongly convex and sufficiently smooth only in a neighborhood of the first Newton step.
As a concrete validation, we analyze logistic regression with Gaussian features and show that our bounds capture the correct scaling up to polylogarithmic factors, yielding matching upper and lower bounds and explaining observed regimes in which NS is markedly more accurate than IF, thereby resolving open questions raised by (Koh et al., 2019).

Forum: https://openreview.net/forum?id=mDo8XNqopd

---

## 273. Recurrent Structural Policy Gradient for Partially Observable Mean Field Games

**Abstract:** Mean Field Games (MFGs) provide a principled framework for modelling interactions in large population systems. However, algorithmic progress has been limited since model-free methods are high variance and exact methods scale poorly. Recent Hybrid Structural Methods (HSMs) reduce variance while maintaining tractability by leveraging low-dimensional individual state and action spaces and known transition dynamics to compute the exact expected return conditioned on Monte Carlo rollouts of common noise. However, HSMs have not been extended to partially observable settings. We propose *Recurrent Structural Policy Gradient* (RSPG), the first history-aware HSM for MFGs with public partial information. RSPG achieves an order-of-magnitude faster convergence than model-free RL methods while learning history-aware behaviour, unlike current HSMs. To facilitate research into MFGs, we also introduce MFAX, our JAX-based framework for MFGs that supports both analytic and sample-based mean-field updates.

Forum: https://openreview.net/forum?id=VkZQThGNgI

---

## 274. MSP: Probabilistically Consistent Multi-Scale Action Generation

**Abstract:** In robotic imitation learning, accurately modeling the multimodality and temporal correlations of long-horizon action sequences remains challenging. Long-horizon tasks require preserving global task intent while executing precise low-level control; otherwise, local errors can accumulate and lead to failure. While recent coarse-to-fine autoregressive models have improved action generation, they struggle to maintain consistency across hierarchies, leading to suboptimal performance in long-horizon tasks. To address these shortcomings, we propose Probabilistically Consistent Multi-Scale Action Generation (MSP), a novel coarse-to-fine approach that promotes cross-scale consistency. MSP adopts a streamlined multi-scale design by directly downsampling in a continuous latent space. A scale-wise autoregressive Transformer is used to generate semantic conditions at each scale, which guide a lightweight MeanFlow model to capture multi-scale latent distributions, enabling probabilistically consistent refinement across scales. Through extensive simulation and real-world experiments, including long-horizon, multi-task, and few-shot generalization settings, we show that MSP outperforms existing coarse-to-fine methods, achieving state-of-the-art performance with high efficiency.

Forum: https://openreview.net/forum?id=zqjKnzyRo4

---

## 275. Efficient numeracy in language models through single-token number embeddings

**Abstract:** To drive progress in science and engineering, large language models (LLMs) must be able to process large amounts of numerical data and solve long calculations efficiently. This is currently only possible through the use of external tools or extensive reasoning chains, either weakening the numerical representations of LLMs or limiting the length of problems they can solve. We show that frontier LLMs require excessive amounts of reasoning tokens to solve even basic calculations, which is exacerbated by their tokenization strategies that split single numbers into multiple tokens. This motivates the need for efficient and effective single-token number encodings. We introduce a set of desiderata for such encodings and show that existing approaches fail to fulfill them. To address these shortcomings, we propose BitTokens, a novel encoding strategy that represents any number as a single token using its IEEE 754 binary floating-point representation. Through extensive experiments we show that our BitTokens allow even small language models to learn algorithms that solve basic arithmetic operations nearly perfectly. This newly gained efficiency could expand the length and complexity of problems language models can solve.

Forum: https://openreview.net/forum?id=Bh4Ubk80M8

---

## 276. Asymptotic Optimality of the High-Dimensional Gaussian Mechanism and Improved Low-Dimensional Mechanisms for Differential Privacy

**Abstract:** The additive noise mechanism is a foundational tool for differential privacy (DP) of $T$-dimensional real-valued vector queries. The Gaussian mechanism, utilizing Gaussian noise, is the mostly widely used such mechanism, due to its simplicity and strong privacy guarantees. In this work, we provide justification for this choice, showing that as the dimension $T\to\infty$, no additive-noise mechanism can asymptotically improve on the Gaussian mechanism's privacy--utility tradeoff for the strong privacy settings typically used. We also develop a new family of **Spherical Generalized Gamma** DP mechanisms, which contains both the Gaussian mechanism and the recently studied $\ell_2$ mechanism (Joseph \emph{et al.}, ICML 2025). We identify members of this family that outperform both the Gaussian and $\ell_2$ mechanisms in certain low-dimensional settings, and show tight composition of all mechanisms in this family, answering an open question of Joseph \emph{et al.}~regarding the $\ell_2$ mechanism.

Forum: https://openreview.net/forum?id=82Wosp2Iu1

---

## 277. Think in Cloud, Look at Edges: Semantic-Driven Query Decomposition for Efficient Video Reasoning

**Abstract:** Long video understanding faces a critical dilemma: cloud-based Large Multimodal Models (LMMs) offer superior reasoning but suffer from prohibitive bandwidth costs and latency, while edge-based solutions sacrifice perception accuracy for speed. Current collaborative approaches attempt to bridge this gap via similarity-based filtering, yet they treat complex queries as flat semantic vectors. We identify this as a fundamental flaw leading to "Semantic Submergence," where dominant visual features drown out subtle but logically critical cues. To solve this, we introduce SCOPE (Semantic Cloud-Orchestrated Perception at Edge). Shifting the paradigm to "Think in Cloud, Look at Edges," SCOPE utilizes a cloud LMM to decompose complex queries into a structured Directed Acyclic Graph (DAG). This "observation plan" guides the edge to retrieve evidence based on logical necessity rather than mere statistical similarity. Experiments on Video-MME and LongVideoBench demonstrate that SCOPE redefines the Pareto frontier, matching cloud-level accuracy with significantly lower transmission costs and outperforming state-of-the-art baselines on complex reasoning tasks.

Forum: https://openreview.net/forum?id=9HyDFcKhLV

---

## 278. Matroid Algorithms Under Size-Sensitive Independence Oracles

**Abstract:** The standard oracle model for matroid algorithms assumes that each independence query can be answered in constant time, regardless of the size of the queried set. While this abstraction has underpinned much of the theoretical progress in matroid optimization, it masks the true computational effort required by these algorithms. In particular, for natural and widely studied classes such as graphic matroids, even a single independence query can require work linear in the size of the set, making the constant-time assumption implausible. We address this gap by introducing a size-sensitive cost model where the cost of a query $Q$ scales with $|Q|$. Nearly linear-time oracle implementations exist for broad families of matroids, and this refined abstraction therefore captures the true cost of query evaluation while allowing for a more faithful comparison between general matroids and their natural special cases. Within this framework we study three fundamental algorithmic tasks: finding a basis of a matroid, approximating its rank, and approximating its partition size. We establish tight results, proving nearly matching upper and lower bounds that show the optimal query cost is (up to logarithmic factors) quadratic in the size of the matroid. On the algorithmic side, our upper bounds are realized by explicit procedures that construct the desired solution. On the complexity side, our lower bounds are unconditional and already hold even for weaker distinguishing formulations of the problems. Finally, for matroids with maximum circuit size at most $c$, we show that the quadratic barrier can be broken, providing an algorithm that calculates the maximum-weight basis with expected query cost $\mathcal{O}(n^{2-1/c} \log n)$.

Forum: https://openreview.net/forum?id=80Job2F5eb

---

## 279. TimeRewarder: Learning Dense Reward from Passive Videos via Frame-wise Temporal Distance

**Abstract:** Designing dense rewards is crucial for reinforcement learning (RL), yet in robotics it often demands extensive manual effort and lacks scalability. One promising solution is to view task progress as a dense reward signal, as it quantifies the degree to which actions advance the system toward task completion over time. We present TimeRewarder, a simple yet effective reward learning method that derives progress estimation signals from passive videos, including robot demonstrations and human videos, by modeling temporal distances between frame pairs. We then demonstrate how TimeRewarder can supply step-wise proxy rewards to guide reinforcement learning. In our comprehensive experiments on ten challenging Meta-World tasks, we show that TimeRewarder dramatically improves RL for sparse-reward tasks, achieving nearly perfect success in 9/10 tasks with only 200,000 interactions per task with the environment. This approach outperforms previous methods and even the manually designed environment dense reward on both the final success rate and sample efficiency. Moreover, we show that TimeRewarder pretraining can exploit real-world human videos, highlighting its potential as a scalable approach to rich reward signals from diverse video sources.

Forum: https://openreview.net/forum?id=XztRm216YS

---

## 280. On the Identifiability of Poisson Branching Structural Causal Model Under Latent Confounding

**Abstract:** Causal discovery from observational count data poses unique challenges, particularly when the data exhibit inherent branching structures, such as an upstream ad impression event triggering a downstream purchase event with certain probability. Such branching dynamics are naturally modeled by thinning operators (for branching) and an independent Poisson distribution (for exogenous noise), constituting a Poisson Branching Structural Causal Model (PB-SCM). However, existing approaches based on PB-SCM rely on the restrictive assumption of causal sufficiency, failing to account for ubiquitous latent confounders. In this work, we propose a Latent Confounding Poisson Branching Structural Causal Model (LC-PB-SCM) to bridge this gap. We leverage Probability Generating Function (PGF) to characterize the complex dependencies introduced by latent confounding. Then, we establish a Trie representation theorem that maps the branching structure to algebraic properties of PGF monomials. Based on local PGF, we establish a complete identifiability condition for local 3-variables covering all causal patterns distinguishable up to monomial equivalence. Finally, we propose a practical algorithm to learn causal structures under latent confounding and demonstrate its effectiveness through experiments on both synthetic and real-world datasets.

Forum: https://openreview.net/forum?id=73YmKB7KpW

---

## 281. Beyond ReLU: Bifurcation, Oversmoothing, and Topological Priors

**Abstract:** Graph Neural Networks (GNNs) learn node representations through iterative network-based message-passing. While powerful, deep GNNs suffer from oversmoothing, where node features converge to a homogeneous, non-informative state. We re-frame this problem of representational collapse from a \emph{bifurcation theory} perspective, characterizing oversmoothing as convergence to a stable ``homogeneous fixed point.'' Our central contribution is the theoretical discovery that this undesired stability can be broken by replacing standard monotone activations (e.g., ReLU) with a class of functions. Using Lyapunov-Schmidt reduction, we analytically prove that this substitution induces a bifurcation that destabilizes the homogeneous state and creates a new pair of stable, non-homogeneous \emph{patterns} that provably resist oversmoothing. Our theory predicts a precise, nontrivial scaling law for the amplitude of these emergent patterns, which we quantitatively validate in experiments. Finally, we demonstrate the practical utility of our theory by deriving a closed-form, bifurcation-aware initialization and showing its utility in real benchmark experiments.

Forum: https://openreview.net/forum?id=AehiGqd7go

---

## 282. Focus and Dilution: The Multi-stage Learning Process of Attention

**Abstract:** Transformer-based models have achieved remarkable success across a wide range of domains, yet our understanding of their training dynamics remains limited. In this work, we identify a recurrent focus–dilution cycle in attention learning and provide a rigorous explanation in a one-layer Transformer setting for Markovian data via gradient-flow analysis. Using stage-wise linearization around critical points, we show that a single focus–dilution cycle can be decomposed into a sequence of distinct stages. First, embedding and projection rapidly condense to a rank-one structure, while attention parameters remain effectively frozen. Then, the attention parameters begin to increase, inducing a frequency-driven focus toward high-frequency tokens. As attention continues to evolve, it generates next-order perturbations in embeddings, leading to a mass-redistribution mechanism that progressively dilutes this focus. Finally, small asymmetries among low-frequency tokens lift a degenerate critical point, opening new embedding directions and initiating the next cycle. Experiments on synthetic Markovian data as well as WikiText and TinyStories corroborate the predicted stages and cyclical dynamics.

Forum: https://openreview.net/forum?id=LF4YsCYfrf

---

## 283. CausalGame: Benchmarking Causal Thinking of LLM Agents in Games

**Abstract:** Recently, it has received growing attention in building AI Scientist agents with Large Language Models (LLMs). Since scientific discovery fundamentally relies on uncovering causal relationships from observations, the capability of causal thinking that distinguish causation from correlation and hidden biases, is essential to LLM agents. Despite a number of existing benchmarks for AI scientists, they do not explicitly incorporate challenges from hidden confounders, selection bias, and noisy measurements that widely exist in real-world scientific discovery. To this end, we present CausalGame, a benchmark that evaluates the causal thinking capabilities of LLM agents through interactive games. More specifically, we ask LLM agents to actively design experimental protocols, collect observation data and derive a final solution with an explanation report. To emulate realistic scientific discovery challenges, we design 14 game settings with the incorporation of selection bias, noisy measurements, and hidden confounders. The results with 29 frontier LLM agents show that they consistently fail to reason about and recover the underlying causal relationships required to solve the games. CausalGame provides a controlled testbed for evaluating causal thinking of AI Scientist agents. The project is available at causalgame.github.io .

Forum: https://openreview.net/forum?id=WNqIX3IFZU

---

## 284. DeCoDe: Decoupling Binding Position and Molecular Conformation in 3D Ligand Diffusion for Structure-Based Drug Design

**Abstract:** Recent advances in diffusion models show promise for Structure-Based Drug Design (SBDD), which aims to generate 3D ligand molecules that bind tightly to specific protein targets. 
This involves jointly optimizing the ligand's 3D conformation and its binding position within the protein pocket. 
However, existing diffusion-based SBDD methods diffuse conformation and binding position synchronously within a high-dimensional joint space, leading to inefficient exploration and suboptimal generation quality in both aspects.
To address this, we propose **DeCoDe**, a novel diffusion framework that **decouples** the diffusion processes of the binding position and molecular conformation. 
Our key insight is to prioritize the perturbation of the ligand's internal conformation in the early stages of the forward (noising) process, while accelerating the perturbation of its global binding position later.
This design guides the reverse (denoising) process to *first coarsely position* the ligand within the pocket before *refining its detailed structure*, mimicking a more efficient, step-wise generation strategy.
Extensive experiments on the CrossDocked2020 benchmark show that DeCoDe achieves significantly higher structural fidelity (with an average improvement of 18%), while maintaining competitive binding affinity and overall molecular properties compared to state-of-the-art baselines.

Forum: https://openreview.net/forum?id=lywUbYpYfG

---

## 285. Joint Learning in the Gaussian Single Index Model

**Abstract:** We consider the problem of jointly learning a one-dimensional projection and a univariate function in high-dimensional Gaussian models. Specifically, we study predictors of the form $f(x)=\varphi^\star(\langle  w^\star, x \rangle)$, where both the direction $w^\star \in \mathcal{S}_{d-1}$, the sphere of $\mathbb{R}^d$, and the function $\varphi^\star: \mathbb{R} \to \mathbb{R}$ are learned from Gaussian data. This setting captures a fundamental non-convex problem at the intersection of representation learning and nonlinear regression. We analyze the gradient flow dynamics of a natural alternating scheme and prove convergence, with a rate controlled by the information exponent reflecting the *Gaussian regularity* of the function $\varphi^\star$. Strikingly, our analysis shows that convergence still occurs even when the initial direction is negatively correlated with the target. On the practical side, we demonstrate that such joint learning can be effectively implemented using a Reproducing Kernel Hilbert Space (RKHS) adapted to the structure of the problem, enabling efficient and flexible estimation of the univariate function. Our results offer both theoretical insight and practical methodology for learning low-dimensional structure in high-dimensional settings.

Forum: https://openreview.net/forum?id=xf1OYJfvqj

---

## 286. Is Your LLM Overcharging You? Tokenization, Transparency, and Incentives

**Abstract:** State-of-the-art large language models require specialized hardware and substantial energy to operate. Consequently, cloud-based services that provide access to these models have become very popular.  In these services, the price users pay depends on the number of tokens a model uses to generate an output–they pay a fixed price per token. 
In this work, we show that this pricing mechanism creates a financial incentive for providers to strategize and misreport the (number of) tokens a model used to generate an output, and users cannot prove, or even know, whether a provider is overcharging them.
However, we also show that, if an unfaithful provider is obliged to be transparent about the generative process used by the model, misreporting optimally without raising suspicion is hard. Nevertheless, as a proof-of-concept, we develop an efficient heuristic algorithm that allows providers to significantly overcharge users without raising suspicion. Crucially, the cost of running the algorithm is lower than the additional revenue from overcharging users, highlighting the vulnerability of users under the current pay-per-token pricing mechanism. Further, we show that, to eliminate the financial incentive to strategize, a pricing mechanism must price tokens linearly on their character count. While this makes a provider's profit margin vary across tokens, we introduce a simple prescription that allows a provider to maintain their average profit margin when transitioning to an incentive-compatible pricing mechanism.
To complement our theoretical results, we conduct experiments with large language models from the $\texttt{Llama}$, $\texttt{Gemma}$ and $\texttt{Ministral}$ families, and prompts from a popular benchmarking platform.

Forum: https://openreview.net/forum?id=aGaBFE9pta

---

## 287. Many Experiments, Few Repetitions, Unpaired Data, and Sparse Effects: Is Causal Inference Possible?

**Abstract:** In many applications, practical constraints prevent measuring covariates and outcomes on the same units, resulting in unpaired data. We study the problem of estimating causal effects under hidden confounding in the following unpaired data setting: we observe some covariates $X$ and an outcome $Y$ under different experimental conditions (environments) but do not observe them jointly -- we either observe $X$ or $Y$. Under appropriate regularity conditions, the problem can be cast as an instrumental variable (IV) regression with the environment acting as a (possibly high-dimensional) instrument. When there are many environments but only a few observations per environment, standard two-sample IV estimators fail to be consistent. We propose a GMM-type estimator based on cross-fold sample splitting of the instrument–covariate sample that also applies in standard IV settings. We prove that it is consistent as the number of environments grows but the sample size per environment remains constant. We further extend the method to sparse causal effects via $\ell_1$-regularized estimation and post-selection refitting.

Forum: https://openreview.net/forum?id=gqa99Ev4C4

---

## 288. Manifold-Aware Perturbations for Constrained Generative Modeling

**Abstract:** Generative models have enjoyed widespread success in a variety of applications. However, they encounter inherent mathematical limitations in modeling distributions where samples are constrained by equalities, as is frequently the setting in scientific domains. In this work, we develop a computationally cheap, mathematically justified, and highly flexible distributional modification for combating known pitfalls in equality-constrained generative models. We propose perturbing the data distribution in a constraint-aware way such that the new distribution has support matching the ambient space dimension while still implicitly incorporating underlying manifold geometry. Through theoretical analyses and empirical evidence on several representative tasks, we illustrate that our approach consistently enables data distribution recovery and stable sampling with both diffusion models and normalizing flows.

Forum: https://openreview.net/forum?id=UPiyC9W4ms

---

## 289. Revenue Guarantees of No-Swap-Regret Dynamics in First Price Auctions

**Abstract:** We study the revenue of approximate correlated equilibrium in discrete first price auctions - the set of allowable bids is $\mathcal{B} = \{0, 1/k, \dots, 1 - 1/k, 1\}$ for some $k \in \mathbb{N}$. We show that the revenue of any $\epsilon$-\textit{approximate} correlated equilibrium is at least $v_2 -  \Theta(1/k)- \Theta(\epsilon k^2)$, where $v_2 \geq 0$ is the second-highest valuation. Our results establish the first polynomial convergence rates on the revenue generated by no-swap regret bidders in first-price auctions.
For instance, if bidders admit the optimal swap regret of $\mathcal{O}(\sqrt{k T})$, then the time-averaged revenue is at least $v_2 -  \Theta(1/k) -  \Theta(\epsilon)$ after $\mathcal{O}(k^5/\epsilon^2)$ rounds.

Forum: https://openreview.net/forum?id=n50Ipdt5d5

---

## 290. TG-RAG: A Retrieval-Augmented Framework for Reasoning Guidance in Specialized Domains

**Abstract:** Enhancing Large Reasoning Models (LRMs) for specialized domains remains a critical challenge. While recent industrial frameworks attempt to encapsulate Standard Operating Procedures into modular "skills" for dynamic retrieval, utilizing them via context engineering often proves insufficient for complex workflows, leading to "Cognitive Drift." To mitigate this, we propose $\textbf{Thought Guidance-Retrieval Augmented Generation (TG-RAG)}$, a Retrieval-Augmented framework that effectively steers the generation process without relying solely on the model's self-correction. Built upon an Expert Procedure Graph (EPG) that formalizes unstructured SOPs, the framework uniquely employs a dynamic $\textbf{``Interrupt-Retrieve-Generate" (IRG)}$ mechanism to actively inject step-specific directives into the model's reasoning process. Extensive evaluations show that TG-RAG achieves competitive performance, demonstrating advantages in specialized domains by ensuring faithful adherence to domain SOPs. Code is available at https://github.com/V1ncent-S/Thought-Guidance.

Forum: https://openreview.net/forum?id=W34UyCRQel

---

## 291. Adaptive Memory Retention in Dynamic Graphs

**Abstract:** Modeling graphs demands a careful balance between long-range propagation of information across nodes and the controlled dissipation of noisy or redundant signals to ensure stable learning and generalization. This challenge is exacerbated in dynamic graphs, where structural and temporal information interact, leading to uncontrolled information accumulation and amplifying noise, thereby affecting generalization. We introduce LAMP, a dynamic graph model for snapshot-based dynamic graphs that incorporates adaptive, learned dissipation within a principled dynamical systems framework. Our architecture combines impulsive neural ODEs with an antisymmetric parameterization to model conservative information flow, alongside data-driven dissipative dynamics that regulate information retention over space and time. This formulation yields stable yet expressive representations and enables effective long-range dependency modeling while avoiding pathological information buildup. We provide a theoretical analysis establishing stability guarantees and characterizing the representational power. Extensive experiments on synthetic and real-world benchmarks demonstrate state-of-the-art performance, particularly on tasks requiring extended-range dependency modeling.

Forum: https://openreview.net/forum?id=x7NGYPNlgs

---

## 292. Emergent Analogical Reasoning in Transformers

**Abstract:** Analogy is a central faculty of human intelligence, enabling abstract patterns discovered in one domain to be applied to another.
However, the mechanisms underlying analogical reasoning in Transformers remain poorly understood.
In this work, inspired by the notion of functors in category theory, we formalize analogical reasoning as the inference of correspondences between entities across categories.
Based on this formulation, we introduce synthetic tasks that evaluate the emergence of analogical reasoning under controlled settings.
We find that the emergence of analogical reasoning is highly sensitive to data characteristics, optimization choices, and model scale.
Through mechanistic analysis, we show that analogical reasoning in Transformers decomposes into two key components:
(1) geometric alignment of relational structure in the embedding space, and
(2) the application of a functor within the Transformer. These mechanisms enable models to transfer relational structure from one category to another, realizing analogy.
Finally, we quantify these effects and find that the same trends are observed in pretrained LLMs.
In doing so, we move analogy from an abstract cognitive notion to a concrete, mechanistically grounded phenomenon in modern neural networks.

Forum: https://openreview.net/forum?id=GxTkgMBiz8

---

## 293. How can embedding models bind concepts?

**Abstract:** Humans easily determine which color belongs to which shape in multi-object scenes, an ability known as concept binding. Vision–language embedding models such as CLIP struggle with binding: they recognize individual concepts but fail to represent which concepts form which objects. Although CLIP behaves like a bag-of-concepts model in cross-modal retrieval, object information is recoverable from its image and text embeddings separately. We study this tension through the binding function, which maps concepts to scene embeddings.
We find that scene embeddings decompose additively into object representations, explaining why uni-modal probes can recover object information. However, CLIP’s binding function is high-complexity, which likely prevents the image and text encoders from learning a shared binding mechanism that generalizes to unseen concept combinations.
We then ask whether this limitation is fundamental. We show that it is not. In controlled transformer models trained from scratch, binding generalization emerges with sufficient data coverage. These models learn low-complexity binding functions characterized by multiplicative interactions between concepts, enabling systematic generalization.

Forum: https://openreview.net/forum?id=lFkGJ60bGq

---

## 294. Initialization is Half the Battle: Generating Diverse Images from a Guidance Potential Posterior

**Abstract:** Despite the remarkable fidelity of generative models, they frequently suffer from mode collapse. Existing strategies for enhancing diversity predominantly focus on intervening during the generation trajectory. We identify a critical oversight that the standard Gaussian initialization often causes trajectories to collapse into dominant modes because it is agnostic to the guidance potential landscape. In this work, we formulate selecting the initial noise from a *guidance potential posterior*, which effectively re-weights the prior towards diversity-rich regions. To sample from this distribution efficiently, we introduce *Diversity-inducing Initialization* (DivIn), which leverages Langevin dynamics to actively navigate the initialization landscape, steering initial noise away from collapsing regions while anchoring them to the valid data manifold. Our method serves as an inference-time diversity enhancement compatible with both diffusion and flow matching models. Extensive experiments show that DivIn exhibits a superior performance in both class-to-image and text-to-image scenarios. 
Furthermore, we highlight that as DivIn is orthogonal to trajectory-based methods, combining them significantly expands the diversity-quality Pareto frontier beyond what either achieves in isolation.

Forum: https://openreview.net/forum?id=zXJjT0slV7

---

## 295. Characterizing, Evaluating, and Optimizing Complex Reasoning

**Abstract:** Large Reasoning Models (LRMs) increasingly rely on reasoning traces with complex internal structures. However, existing work lacks a unified answer to three fundamental questions: 
(1) what defines high-quality reasoning, 
(2) how to reliably evaluate long, implicitly structured reasoning traces, and 
(3) how to use such evaluation signals for reasoning optimization.
To address these challenges, we provide a unified perspective. (1) We introduce the ME$^2$ principle to characterize reasoning quality along macro- and micro-level concerning efficiency and effectiveness. (2) Built on this principle, we model reasoning traces as directed acyclic graphs (DAGs) and develop a DAG-based pairwise evaluation method, capturing complex reasoning structures. (3) Based on this method, we construct the TRM-Preference dataset and train a Thinking Reward Model (TRM) to evaluate reasoning quality at scale.
Experiments show that thinking rewards serve as an effective optimization signal. At test time, selecting better reasoning leads to better outcomes (up to 19.3\% gain), and during RL training, thinking rewards enhance reasoning and performance (up to 3.9\% gain) across diverse tasks. Code and data are available at https://github.com/Simplified-Reasoning/TRM.

Forum: https://openreview.net/forum?id=IMFgiWw4jd

---

## 296. Recovering Policy-Induced Errors: Benchmarking and Trajectory Synthesis for Robust GUI Agents

**Abstract:** While GUI agents have advanced rapidly, they often lack the robustness to recover from their own errors, hindering real-world deployment. To bridge this gap at both the evaluation and data levels, we introduce GUI-RobustEval and propose Robustness-driven Trajectory Synthesis. GUI-RobustEval contains 1,216 executable test cases that systematically measure error recovery capabilities across a broad and realistic spectrum of error modes. At the data level, RoTS is a scalable synthesis framework that creates 800k high-quality data via a tree-based pipeline that proactively discovers diverse error modes and synthesizes corresponding recovery steps. Our two models, RoTS-7B and RoTS-32B, fine-tuned on our dataset, both demonstrate significant gains on GUI-RobustEval and traditional GUI benchmarks. Notably, RoTS-32B achieves state-of-the-art performance on OSWorld, with a 47.4% success rate and a 33.8% All-Pass@4 score, suggesting that improved long-horizon error recovery ability contributes to both robustness and overall performance. Our code is available at https://github.com/AlibabaResearch/RoTS

Forum: https://openreview.net/forum?id=iJwDylm93H

---

## 297. Scaling Real-World Robot Policy Evaluation via Discrete Diffusion World Model

**Abstract:** Evaluating generalist robot manipulation policies is costly and difficult to scale in the real world. While emerging world models (e.g., WorldEval, Ctrl-World) offer a promising alternative, the reliability of such evaluation remains a critical bottleneck. 
Specifically, their visual predictions can undermine policy assessment by "self-correcting" failures into false positives or yielding artifacts under out-of-distribution controls.
Even with failure-enriched data, current architectures struggle to capture action-causal dynamics, as they typically treat actions as passive conditions rather than causal drivers.
To address this, we propose dWorldEval, an action-centric discrete-diffusion world model that maps visual observations, language instructions, and action chunks into a shared unified token space and denoises them with a single self-attention backbone where actions function as first-class tokens. 
To realize reliable policy-world interaction, dWorldEval introduces a sparse keyframe memory that anchors global scene state while preserving fine-grained multi-view interaction cues, and leverages Progress-as-text to jointly generate future observations and success indicators.
Extensive experiments on LIBERO, RoboTwin, and real-robot tasks demonstrate that dWorldEval significantly outperforms video diffusion baselines in action controllability, stabilizes long-horizon multi-view rollouts, enabling accurate policy ranking via automatic success estimation.

Forum: https://openreview.net/forum?id=93uNlQ1Qp0

---

## 298. ReQAT: Achieving Full-Precision Reasoning Accuracy with 4-bit Floating-Point Quantization-Aware Training

**Abstract:** Large Reasoning Models (LRMs) achieve strong problem-solving through long chain-of-thought, but their deployment is constrained by the high cost of full-precision inference and growing KV cache footprints. Microscaled FP4 formats enable efficient FP4 deployment; however, fully quantizing weights, activations, and KV caches (W4A4KV4) causes severe reasoning degradation that existing PTQ and QAT fail to recover. We identify that FP4 failures concentrate on low-entropy tokens—precise symbolic commitments such as digits and operators—where quantization noise inflates sampling errors that cascade through reasoning traces. Based on this insight, we propose ReQAT, a reasoning-centric FP4 training framework with three components: (i) Trace-Aligned QAT (TAQ), which revisits identical reasoning traces to focus updates on critical low-entropy decisions; (ii) Selective Entropy Minimization (SEM), which reinforces confidence at low-entropy positions; and (iii) Q-FIT, a quantization-friendly initialization that jointly calibrates RoPE-consistent KV cache transformations to stabilize QAT. Under the same training budget, ReQAT not only recovers but surpasses BF16 fine-tuning accuracy, while delivering up to $3.9\times$ throughput speedup on NVIDIA DGX Spark and $3.1\times$ on B200. The project repository is available at https://github.com/aiha-lab/ReQAT.

Forum: https://openreview.net/forum?id=aKY52fnzgc

---

## 299. AI Engram: In Search of Memory Traces in Artificial Intelligence

**Abstract:** Memory formation is fundamental to intelligence, yet whether deep neural networks preserve identifiable memory traces analogous to biological memory units remains an open question. This work introduces a geometric framework to identify such “AI engrams” by formalizing the neuroscientific criteria of specificity, reactivation, sufficiency, and necessity into a constrained inverse problem. We derive a closed-form estimator that isolates individual memory traces from globally entangled parameters, and show that this biologically-derived solution corresponds to a natural gradient update on the parameter manifold. AI engrams enable surgical manipulation of learned knowledge: any subset of memories can be composed or erased through linear arithmetic, without iterative optimization. Experiments ranging from simple MLPs to LLMs demonstrate the causal validity and substantial scalability of AI engrams. Together, these results bridge theories of biological memory and artificial representation learning and offer geometric insight into how deep networks simultaneously support functional specificity within distributed storage.

Forum: https://openreview.net/forum?id=QZO3oby12w

---

## 300. Ranking Time Series using a Time Warping Ideal Point Model

**Abstract:** Expert-annotated time series datasets often suffer from low agreement, especially in medical applications where decisions rely on subjective criteria and inconsistent thresholds. Such variability degrades annotation quality and thus limits the reliability of supervised classification models. To address this, we propose to rely on a pairwise comparison-based approach, which provides a more robust alternative to individual annotation, since relative judgments are typically easier and yield higher consistency.
The problem is thus transformed into a ranking problem and we introduce an ideal point model adapted to time series data using elastic similarity measures such as Dynamic Time Warping (DTW) and Time Warp Edit Distance (TWED).
We prove Lipschitz continuity of these distances and demonstrate several convergence guarantees for this model. To facilitate gradient-based optimization, we also introduce a differentiable version of the TWED. Finally, we show through multiple experiments that our approach produces accurate and robust rankings under noisy annotation conditions.

Forum: https://openreview.net/forum?id=5wMBxVaTbC

---

## 301. Stabilizing the Q-Gradient Field for Policy Smoothness in Actor-Critic Methods

**Abstract:** Policies learned via continuous actor-critic methods often exhibit erratic, high-frequency oscillations, making them unsuitable for physical deployment. 
Current approaches attempt to enforce smoothness by directly regularizing the policy's output. 
We argue that this approach treats the symptom rather than the cause. 
In this work, we theoretically establish that policy non-smoothness is fundamentally governed by the differential geometry of the critic. 
By applying implicit differentiation to the actor-critic objective, we prove that the sensitivity of the optimal policy is bounded by the ratio of the Q-function's mixed-partial derivative (noise sensitivity) to its action-space curvature (signal distinctness). 
To empirically validate this theoretical insight, we introduce PAVE (Policy-Aware Value-field Equalization), a critic-centric regularization framework that treats the critic as a scalar field and stabilizes its induced action-gradient field. 
PAVE rectifies the learning signal by minimizing the Q-gradient volatility while preserving local curvature. 
Experimental results demonstrate that PAVE achieves smoothness comparable to policy-side smoothness regularization methods, while maintaining competitive task performance, without modifying the actor.

Forum: https://openreview.net/forum?id=UpA00uD6Ek

---

## 302. HypoSpace: A Diagnostic Benchmark for Set-Valued Hypothesis Generation under Underdetermination and Sublinear Coverage Bounds

**Abstract:** Many scientific problems are underdetermined: multiple distinct hypotheses are equally consistent with the same observations. In such settings, effective inference requires not only producing valid explanations, but also systematically exploring and covering the admissible hypothesis set. We introduce HypoSpace, a benchmark that treats large language models (LLMs) as samplers over finite hypothesis spaces and evaluates them on three metrics: Validity, Uniqueness, and Recovery. HypoSpace spans three structured domains (causal graph inference, gravity-constrained 3D voxel reconstruction, and Boolean genetic interaction modeling) with deterministic validators and exactly enumerable solution spaces, plus real-world anchored case studies. Empirically, HypoSpace reveals a capability- and scale-dependent coverage failure: models can maintain high Validity while exhibiting reduced Uniqueness and Recovery as admissible hypothesis spaces become larger or more combinatorial. We further show that the analysis on stratified decoding partially mitigates this collapse, demonstrating HypoSpace's utility as a diagnostic benchmark for set-valued inference. Code is available at: https://github.com/CTT-Pavilion/_HypoSpace.

Forum: https://openreview.net/forum?id=QpjtK65JHO

---

## 303. S$^3$GNN: Efficient Global Mixing and Local Message Passing for Long-Range Graph Learning

**Abstract:** Message-passing neural networks (MPNNs) often suffer from an information bottleneck when capturing long-range dependencies, leading to the oversquashing (OSQ) phenomenon. Alongside spatial connectivity enrichment (e.g., rewiring), recent studies have shown that spectral filtering can yield strong long-range learning outcomes, as spectral operators enable global information mixing that alleviates OSQ. These approaches achieve this either by stabilizing the Jacobian energies in deep propagation or by guaranteeing OSQ mitigation under strong theoretical assumptions. We examine the practical attainability of these guarantees and show that the associated Jacobian sensitivity lower bound is generally difficult to achieve in practice. We then propose S$^3$GNN, which mitigates OSQ without such restrictive assumptions by lightweightly reintroducing omitted components with substantially lower computational complexity, while standard stability constraints on feature transformations remain effective under our new dynamics. Extensive experiments across diverse domains (e.g., long-range benchmarks, KGQA, and mesh-based fluid dynamics) demonstrate that S$^3$GNN achieves up to an order-of-magnitude error reduction with up to 50\% fewer parameters. Our code can be found in https://github.com/EEthanShi/S3-GNN.git.

Forum: https://openreview.net/forum?id=9SCVnAxoKK

---

## 304. Asymmetric Perturbation in Solving Bilinear Saddle-Point Optimization

**Abstract:** This paper proposes asymmetric perturbation, where only one player's payoff function is perturbed, for solving bilinear saddle-point optimization problems, commonly arising in minimax problems, game theory, and constrained optimization. 
Symmetric perturbation is known to require decreasing its strength to ensure convergence to a solution, i.e., an equilibrium in the original game, resulting in a slower rate.
First, with asymmetric perturbation, we show that, for a sufficiently small perturbation strength, the equilibrium strategy of the asymmetrically perturbed game coincides with an equilibrium strategy of the original unperturbed game.
Second, building on this coincidence, we construct a learning algorithm with a linear last-iterate convergence rate. 
Third, motivated by the fact that the coincidence relies on the perturbation strength being sufficiently small, we also provide a parameter-free variant, retaining the linear rate. 
Finally, we empirically demonstrate fast convergence toward equilibria in both normal-form and extensive-form games.

Forum: https://openreview.net/forum?id=fyae4jQ9Lw

---

## 305. Procedural Pretraining: Warming Up Language Models with Abstract Data

**Abstract:** Pretraining language models directly on web-scale corpora is the de facto paradigm. We study an alternative where the model is initially exposed to *abstract structured data* to ease the subsequent acquisition of rich semantic knowledge, much like humans learning simple logic and mathematics before higher reasoning. We focus on *procedural data*, generated by formal languages and other simple algorithms, as such abstract data.

We first diagnose the algorithmic skills that different forms of procedural data can improve, often significantly. For example, the accuracy of context recall (Needle-in-a-haystack) jumps from 10 to 98% when a model is pretrained on Dyck sequences (balanced brackets). Second, we study how these gains are reflected in pretraining larger models (up to 1.3B). We find that front-loading as little as 0.1–0.3% procedural data significantly outperforms standard pretraining on natural language, code, and informal mathematics (C4, CodeParrot, and DeepMind-Math datasets). Notably, this also enables the models to reach the same loss value with only 55/67/86% of the original data and thus a comparable reduction in FLOPs. Third, we explore the mechanisms behind the benefits and find that procedural pretraining instills non-trivial structure in both attention and MLP layers. The former is particularly important for structured domains (e.g. code), and the latter for language. Finally, we lay a path for combining multiple forms of procedural data. Our results show that procedural pretraining is a simple, lightweight means of improving performance and accelerating language model pretraining, ultimately suggesting the promise of disentangling knowledge acquisition from reasoning in LLMs.

Forum: https://openreview.net/forum?id=XFTTezxLdU

---

## 306. VGGT-Motion: Motion-Aware Calibration-Free Monocular SLAM for Long-Range Consistency

**Abstract:** Despite recent progress in calibration-free monocular SLAM via 3D vision foundation models, scale drift remains severe on long sequences. Motion-agnostic partitioning breaks contextual coherence and causes zero-motion drift, while conventional geometric alignment is computationally expensive. To address these issues, we propose VGGT-Motion, a calibration-free SLAM system for efficient and robust global consistency over kilometer-scale trajectories. Specifically, we first propose a motion-aware submap construction mechanism that uses optical flow to guide adaptive partitioning, prune static redundancy, and encapsulate turns for stable local geometry. We then design an anchor-driven direct Sim(3) registration strategy. By exploiting context-balanced anchors, it achieves search-free, pixel-wise dense alignment and efficient loop closure without costly feature matching. Finally, a lightweight submap-level pose graph optimization enforces global consistency with linear complexity, enabling scalable long-range operation. Experiments show that VGGT-Motion markedly improves trajectory accuracy and efficiency, achieving state-of-the-art performance in zero-shot, long-range calibration-free monocular SLAM.

Forum: https://openreview.net/forum?id=GyRMbsYFiG

---

## 307. SleepLM: Natural-Language Intelligence for Human Sleep

**Abstract:** We present SleepLM, a family of sleep-language foundation models that enable human sleep alignment, interpretation, and interaction with natural language.
Despite the critical role of sleep, learning-based sleep analysis systems operate in closed label spaces (e.g., predefined stages or events) and fail to describe, query, or generalize to novel sleep phenomena.
SleepLM bridges natural language and multimodal polysomnography, enabling language-grounded representations of sleep physiology.
To support this alignment, we introduce a multilevel sleep caption generation pipeline that enables the curation of the first large-scale sleep-text dataset, comprising over 100K hours of data from more than 10,000 individuals.
Furthermore, we present a unified pretraining objective that combines contrastive alignment, caption generation, and signal reconstruction to better capture physiological fidelity and cross-modal interactions.
Extensive experiments on real-world sleep understanding tasks verify that SleepLM outperforms state-of-the-art in zero-shot and few-shot learning, cross-modal retrieval, and sleep captioning. Importantly, SleepLM also exhibits intriguing capabilities including language-guided event localization, targeted insight generation, and zero-shot generalization to unseen tasks.
To support reproducibility and future work, we open-source the captioning pipeline, pretrained checkpoints, and the model architectures at https://github.com/yang-ai-lab/SleepLM.

Forum: https://openreview.net/forum?id=9wpwfSJCp9

---

## 308. Know More, Know Clearer: A Meta-Cognitive Framework for Knowledge Augmentation in Large Language Models

**Abstract:** Knowledge augmentation has significantly enhanced the performance of Large Language Models (LLMs) in knowledge-intensive tasks. However, existing methods typically operate on the simplistic premise that model performance equates with internal knowledge, overlooking the knowledge-confidence gaps that lead to overconfident errors or uncertain truths. To bridge this gap, we propose a novel meta-cognitive framework for reliable knowledge augmentation via differentiated intervention and alignment. Our approach leverages internal cognitive signals to partition the knowledge space into mastered, confused, and missing regions, guiding targeted knowledge expansion. Furthermore, we introduce a cognitive consistency mechanism to synchronize subjective certainty with objective accuracy, ensuring calibrated knowledge boundaries. Extensive experiments demonstrate the our framework consistently outperforms strong baselines, validating its rationality in not only enhancing knowledge capabilities but also fostering cognitive behaviors that better distinguish knowns from unknowns. All codes are available at https://github.com/AI9Stars/Know-More-Know-Clearer.

Forum: https://openreview.net/forum?id=ENuMNYCiV6

---

## 309. Delving into Muon and Beyond: Deep Analysis and Extensions

**Abstract:** The Muon optimizer has recently attracted considerable attention for its strong empirical performance and use of orthogonalized updates on matrix-shaped parameters, yet its underlying mechanisms and relationship to adaptive optimizers such as Adam remain insufficiently understood.
In this work, we aim to address these questions through a unified spectral perspective. Specifically, we view Muon as the \( p = 0 \) endpoint of a family of spectral transformations of the form \( \boldsymbol{U} \boldsymbol{\Sigma}^{p} \boldsymbol{V}^{\top} \), and consider additional variants with \( p = \frac{1}{2} \), \( p = \frac{1}{4} \), and \( p = 1 \). These transformations are applied to both first-moment updates, as in momentum SGD, and to root-mean-square (RMS) normalized gradient updates as in Adam. To enable efficient computation, we develop a coupled Newton iteration that avoids explicit singular value decomposition. Across controlled experiments, we find that RMS-normalized updates yield more stable optimization than first-moment updates. Moreover, while spectral compression provides strong stabilization benefits under first-moment updates, the Muon update (\( p = 0 \)) does not consistently outperform Adam. These results suggest that Muon is best understood as an effective form of spectral normalization, but not a universally superior optimization method. Our code is available at \url{https://github.com/Ocram7/BeyondMuon}.

Forum: https://openreview.net/forum?id=xUQ0Gw11NL

---

## 310. Towards the Explainability of Temporal Graph Networks via Memory Backtracking and Topological Attribution

**Abstract:** Temporal graphs are ubiquitous in real-world applications and Temporal Graph Networks (TGNs) have achieved superior predictive accuracy. 
Understanding which historical events drive 
model predictions can enhance trustworthiness of TGNs. Existing explanation methods overlook the memory module, the core component that records and updates node histories, leaving the influence of past events unexplored. To address this, we attribute TGNs predictions through the topology attribution tree and memory backtracking tree. The topology attribution tree captures the influence of neighbors and their memory vectors, then the memory backtracking tree quantifies how historical events shape node memory vectors. We apply the LRP in TGNs, 
ensuring that the total contribution of events equals the model’s logits. Finally, top-k selection may be unfaithful due to the nonlinear mapping from logits to probabilities, we design optimization objectives to identify the important events. Experiments on nine temporal graph datasets, spanning node property prediction, link prediction tasks and graph classification tasks, show that our method provides faithful explanations and outperforms state-of-the-art baselines. The code is available at https://github.com/yazhengliu/MemExplainer.

Forum: https://openreview.net/forum?id=sKAgLgpujy

---

## 311. Progressive Graph Structure Adjustment for Homophily Shift Adaptation

**Abstract:** We propose *Progressive Structure Adjustment for Homophily Shift* (*PSAHS*), a lightweight method for *Graph Domain Adaptation* (*GDA*) that explicitly addresses cross-domain mismatch in node-level homophily. PSAHS enhances node homophily in the source graph to a prescribed level by reweighting edges and introducing additional intra-class connections for low-homophily nodes, and conservatively refines the target graph using agreement-consistent predictions from a structure-aware *Graph Neural Network* (*GNN*) and an attribute-only *Multi-Layer Perceptron* (*MLP*) to ensure reliability under label scarcity. After each structural refinement, domain-adversarial training is employed to align node representations across domains. PSAHS employs a progressive training scheme that alternates between structure adjustment and representation alignment, where increasingly informative representations enable safer homophily correction, and the refined structure in turn improves representation learning. Extensive experiments on multiple GDA benchmarks demonstrate that PSAHS consistently outperforms strong baselines, with particularly large gains under severe homophily mismatch, highlighting the importance of explicit homophily alignment for effective cross-graph transfer.

Forum: https://openreview.net/forum?id=OkyO74bJqE

---

## 312. Post-Training with Policy Gradients: Optimality and the Base Model Barrier

**Abstract:** We study post-training linear autoregressive models with outcome and process rewards. Given a context $x$, the model must predict the response $y \in \mathcal{Y}^N$, a sequence of length $N$ that satisfies a $\gamma$ margin condition, an extension of the standard separability to sequences.
We prove that on test samples where the base model achieves a non-trivial likelihood $\alpha$, a variant of policy gradient (PG) can achieve likelihood $1 - \varepsilon$ with an essentially minimax optimal number of reward queries $\tilde{\mathcal{O}}((\alpha^{-1} + \varepsilon^{-1})/\gamma^2)$.
However, a barrier arises for going beyond the support of the base model.
We prove that the overall expected error after post-training with outcome rewards is governed by a property of the base model called the *Likelihood Quantile* (LQ), and that variants of PG, while minimax optimal, may require a number of reward queries exponential in $N$ to go beyond this support, regardless of the pre-training algorithm.
To overcome this barrier, we study post-training with a process reward model, and demonstrate how PG variants in this setting avoid the curse of dimensionality in $N$ via dependence on a token-level LQ.
Along the way, we prove that under the margin condition, SGD with adaptive learning rate (LR) achieves a near optimal test error for statistical learning, and PG with adaptive LR achieves a near optimal number of mistakes for online learning while being computationally efficient whenever possible, both of which may be of independent interest.

Forum: https://openreview.net/forum?id=nnWlTi7A7a

---

## 313. Incentivizing Truthfulness and Collaborative Fairness in Bayesian Learning

**Abstract:** Collaborative machine learning involves training high-quality models using datasets from a number of sources.
To incentivize sources to share data, existing data valuation methods fairly reward each source based on its data submitted as is. However, as these methods do not verify nor incentivize data truthfulness, the sources can manipulate their data (e.g., by submitting duplicated or noisy data) to artificially increase their valuations and rewards or prevent others from benefiting. This paper presents the first mechanism that provably ensures (**F**) collaborative fairness and incentivizes (**T**) truthfulness at equilibrium for Bayesian models. Our mechanism combines semivalues (e.g., Shapley value), which ensure fairness, and a truthful data valuation function (DVF) based on a validation set that is unknown to the sources. As semivalues are influenced by others' data, we introduce an additional condition to prove that a source can maximize its expected data values in coalitions and semivalues by submitting a dataset that captures its true knowledge.
Additionally, we discuss the implications and suitable relaxations of (**F**) and (**T**) when the mediator has a limited budget for rewards or lacks a validation set.
Our theoretical findings are validated on synthetic and real-world datasets.

Forum: https://openreview.net/forum?id=YgzbofmRr6

---

## 314. CAT-Q: Cost-efficient and Accurate Ternary Quantization for LLMs

**Abstract:** In this paper, we present CAT-Q, **C**ost-efficient and **A**ccurate **T**ernary **Q**uantization, for compressing and accelerating LLMs. Unlike existing state-of-the-art ternary quantization methods that rely on data-intensive and costly quantization-aware training to mitigate severe performance degradation, CAT-Q is a simple yet effective post-training quantization scheme that is readily applicable to LLMs with diverse architectures and model sizes. It has two key components, learnable modulation (LM) and softened ternarization (ST), which are coupled from an optimization perspective. LM leverages a composition of learnable factors to modulate the distribution of pre-trained high-precision weights and the ternary threshold, making them less sensitive to ternarization. ST further introduces a differentiable transition function to guide the ternarization process toward stable convergence. We show that, for pre-trained LLMs with 1.7B to 8B parameters, CAT-Q can efficiently quantize them into ternary models using only 512 calibration samples, while achieving superior performance than the seminal BitNet 1.58-bit v1 and v2 families (with 1.3B to 7B parameters) trained with 100B tokens, yielding about a 100,000$\times$ reduction in training tokens. Moreover, we show for the first time that CAT-Q can quantize much larger pre-trained LLMs having 14B to 235B parameters into leading ternary models within just 8 to 60 hours on 8 A100-80GB GPUs. Code is available at https://github.com/IntelChina-AI/BitTern.

Forum: https://openreview.net/forum?id=9uZJLXt7fq

---

## 315. What Do Agents Learn from Trajectory-SFT: Semantics or Interfaces?

**Abstract:** Large language models are increasingly evaluated as interactive agents, yet standard agent benchmarks conflate two qualitatively distinct sources of success: semantic tool-use and interface-specific interaction pattern memorization.
Because both mechanisms can yield identical task success on the original interface, benchmark scores alone are not identifiable evidence of environment-invariant capability.
We propose **PIPE**, a protocol-level evaluation augmentation for diagnosing interface reliance by minimally rewriting environment interfaces while preserving task semantics and execution behavior.
Across 16 environments from AgentBench and AgentGym and a range of open-source and API-based agents, PIPE reveals that task-specific trajectory-SFT can amplify reliance on training-time interface forms: in several environments, agents with trajectory-SFT degrade sharply under minimal interface rewrites, whereas other agents are often more stable.
We further introduce Interface Reliance (IR), a counterbalanced alias-based metric that quantifies preference for training-time interfaces, and show that interface shortcutting exhibits environment-dependent, non-monotonic training dynamics that remain invisible under standard evaluation. Our code is available at https://github.com/ChengZe2005/What-Do-Agents-Learn-from-Trajectory-SFT-Semantics-or-Interfaces-.

Forum: https://openreview.net/forum?id=u1i3XoSXQG

---

## 316. Surgery: Mitigating Harmful  Fine-Tuning for Large Language Models via Attention Sink

**Abstract:** Harmful fine-tuning can invalidate safety alignment of large language models, exposing significant safety risks. In this paper, we utilize the attention sink mechanism to mitigate harmful fine-tuning. Specifically, we first measure a statistic named sink divergence for each attention head and observe that different attention heads exhibit two different signs of sink divergence. To understand its safety implications, we conduct experiments and find that the number of attention heads of positive sink divergence increases along with the increase of the model's harmfulness when undergoing harmful fine-tuning. Based on this finding, we propose a separable sink divergence hypothesis -- attention heads associating with learning harmful patterns during fine-tuning are separable by their sign of sink divergence. Based on the hypothesis, we propose a fine-tuning-stage defense, dubbed Surgery. Surgery utilizes a regularizer for sink divergence suppression, which steers attention heads toward the negative sink divergence group, thereby reducing the model’s tendency to learn and amplify harmful patterns. Extensive experiments demonstrate that Surgery improves defense performance by 5.90\%, 11.25\%, and 9.55\% on the BeaverTails, HarmBench, and SorryBench benchmarks, respectively. Source code is available on https://github.com/Lslland/Surgery.

Forum: https://openreview.net/forum?id=6ojsIliNF0

---

## 317. SCALE: Self-uncertainty Conditioned Adaptive Looking and Execution for Vision-Language-Action Models

**Abstract:** Vision-Language-Action (VLA) models have emerged as a promising paradigm for general-purpose robotic control, with test-time scaling (TTS) gaining attention to enhance robustness beyond training. However, existing TTS methods for VLAs require additional training, verifiers, and multiple forward passes, making them impractical for deployment. Moreover, they intervene only at action decoding while keeping visual representations fixed—insufficient under perceptual ambiguity, where reconsidering how to perceive is as important as deciding what to do. To address these limitations, we propose SCALE, a simple inference strategy that jointly modulates visual perception and action based on 'self-uncertainty', inspired by uncertainty-driven exploration in Active Inference theory—requiring no additional training, no verifier, and only a single forward pass. SCALE broadens exploration in both perception and action under high uncertainty, while focusing on exploitation when confident—enabling adaptive execution across varying conditions. Experiments on simulated and real-world benchmarks demonstrate that SCALE improves state-of-the-art VLAs and outperforms existing TTS methods while maintaining single-pass efficiency. Our code is publicly available at https://github.com/snumprlab/scale.

Forum: https://openreview.net/forum?id=7MlfE2Da2W

---

## 318. Demystifying Entropy Control in LLM RL Training: Theoretical Analysis and Dynamic Scheduling

**Abstract:** This paper investigates a pivotal yet debated component of reinforcement learning (RL) for training large language models (LLMs): controlling entropy (increasing or decreasing it) during RL fine-tuning. The existing literature presents a dichotomy: some studies posit that increasing entropy facilitates exploration, whereas others argue that decreasing entropy enhances performance. To reconcile these conflicting observations, we provide a theoretical framework showing that the effect of entropy is governed by \emph{Entropy Discrepancy}, the distributional divergence between positive and negative samples. Guided by this insight, we derive a principled dynamic scheduling method that adaptively modulates the entropy coefficient, effectively switching between entropy maximization and minimization as training evolves. Extensive experiments confirm the correlation between Entropy Discrepancy and the efficacy of entropy control. Furthermore, our adaptive method yields substantial improvements, boosting Pass@K by 6.7\% on AIME24 and 17.52\% on puzzle tasks compared to vanilla RL, while consistently outperforming recent state-of-the-art reasoning methods.

Forum: https://openreview.net/forum?id=hq2MPYXAko

---

## 319. EEmo-Logic: A Unified Dataset and Multi-Stage Framework for Comprehensive Image-Evoked Emotion Assessment

**Abstract:** Understanding the multi-dimensional attributes and intensity nuances of image-evoked emotions is pivotal for advancing machine empathy and empowering diverse human-computer interaction applications. However, existing models are still limited to coarse-grained emotion perception or deficient reasoning capabilities. To bridge this gap, we introduce **EEmoDB**, the largest image-evoked emotion understanding dataset to date. It features $5$ analysis dimensions spanning $5$ distinct task categories, facilitating comprehensive interpretation. Specifically, we compile $1.2M$ question-answering (QA) pairs (EEmoDB-QA) from $125K$ images via automated generation, alongside a $36K$ dataset (EEmoDB-Assess) curated from $25K$ images for fine-grained assessment. Furthermore, we propose **EEmo-Logic**, an **all-in-one** multimodal large language model (MLLM) developed via instruction fine-tuning and task-customized group relative preference optimization (GRPO) with novel reward design. Extensive experiments demonstrate that EEmo-Logic achieves robust performance in in-domain and cross-domain datasets, excelling in emotion QA and fine-grained assessment. 
The dataset and code are available at https://github.com/workerred/EEmo-Logic.

Forum: https://openreview.net/forum?id=AYy0lOycql

---

## 320. What Makes Value Learning Efficient in Residual Reinforcement Learning?

**Abstract:** Residual reinforcement learning (RL) enables stable online refinement of expressive pretrained policies by freezing the base and learning only bounded corrections. However, value learning in residual RL poses unique challenges that remain poorly understood. In this work, we identify two key bottlenecks: cold start pathology, where the critic lacks knowledge of the value landscape around the base policy, and structural scale mismatch, where the residual contribution is dwarfed by the base action. Through systematic investigation, we uncover the mechanisms underlying these bottlenecks, revealing that simple yet principled solutions suffice: base-policy transitions serve as an essential value anchor for implicit warmup, and critic normalization effectively restores representation sensitivity for discerning value differences. Based on these insights, we propose DAWN (Data-Anchored Warmup and Normalization), a minimal approach targeting efficient value learning in residual RL. By addressing these bottlenecks, DAWN demonstrates substantial efficiency gains across diverse benchmarks, policy architectures, and observation modalities.

Forum: https://openreview.net/forum?id=tSMZHXtJwZ

---

## 321. When Attributes Disagree:  Gradient Conflict in Image Aesthetic Assessment

**Abstract:** Image Aesthetic Assessment (IAA) predicts an image’s overall aesthetic score, yet aesthetic is influenced by multiple attributes whose relative importance varies with image content and usage scenarios. Under end-to-end training with only overall-score supervision, attribute signals are blended, which can cause gradient conflict across samples dominated by different attributes, resulting in gradient cancellation and persistent systematic bias. To address these issues, we propose AGREE (Attribute-guided Gradient Routing for Establishing Agreement), which learns attribute-specific subspaces and performs gradient routing based on sample-wise attribute sensitivity estimated via perturbation analysis. AGREE further reduces feature coupling across attributes with semantic anchors and improves robustness via error-aware reweighting. Experiments on AVA, LAPIS, AADB, TAD66K, and PARA show consistent improvements over diverse IAA baseline models, and AGREE is plug-and-play for existing end-to-end IAA methods without modifying their original architectures. To our knowledge, this work is among the early efforts in IAA to systematically study gradient conflict and provide an effective solution. The code is available at https://dahat364.github.io/AGREE/.

Forum: https://openreview.net/forum?id=GrIs035ec3

---

## 322. Local Covariate Selection for Average Causal Effect Estimation without Pretreatment and Causal Sufficiency Assumptions

**Abstract:** Causal effect estimation is a fundamental task in many scientific fields. Selecting appropriate covariates for adjustment is crucial for obtaining unbiased causal effects. However, most existing methods either rely on learning the global causal structure, assume the absence of latent variables, or impose the pretreatment assumption-restricts covariates to those unaffected by the treatment or outcome. These assumptions are often unrealistic in real-world scenarios, and global structure learning can be computationally intensive and inefficient.
To address these challenges, we first characterize the local existence boundary of adjustment sets for causal effect estimation. Based on this characterization, we develop a novel local learning method for covariate selection in nonparametric causal effect estimation. This method accommodates the presence of latent variables and eliminates the need for the pretreatment assumption.
We prove that the proposed method is both sound and complete under standard assumptions. Its effectiveness is validated through extensive experiments on both synthetic and real-world datasets.

Forum: https://openreview.net/forum?id=qZVcGlU5lN

---

## 323. Geometric and Stochastic Analysis of Discontinuities in Sparse Mixture-of-Experts

**Abstract:** Sparse Mixture-of-Experts (SMoE) architectures are now widely deployed in state-of-the-art language and vision models, where conditional routing allows scaling to very large networks. However, this very Top-$k$ expert selection that enables conditional routing also renders the SMoE map inherently discontinuous. In the vicinity of these discontinuity surfaces, even inputs that are arbitrarily close may activate substantially different sets of experts resulting in significantly different outputs. In this work we give a rigorous geometric and stochastic analysis of these discontinuities. We first classify them by order, determined by the number of tied experts at a switching event. Using measure-theoretic slicing arguments, we establish asymptotic volume estimates for the thickened discontinuity surfaces, showing that lower-order discontinuity sets dominate, whereas higher-order ones occupy a vanishingly small relative volume. Next, modeling random perturbations in the input space via a diffusion process, we prove that the path eventually encounter a discontinuity, and moreover that the first hit almost surely occurs on an order-1 discontinuity with explicit finite-time probability bounds. We further derive occupation-time bounds that quantify the duration the random path spend in the neighborhoods of each discontinuity order. These theoretical results imply that inputs are more likely to lie near lower order discontinuities. Motivated by this insight, we propose a simple smoothing mechanism that can be directly applied to existing SMoEs, softly incorporating experts near discontinuities; our analysis guarantees that the added computational overhead remains small while providing localized smoothing near discontinuities, and experiments across language and vision tasks show that smoothing not only enforces continuity of the SMoE map but also enhances empirical performance.

Forum: https://openreview.net/forum?id=AyrNR1ExN5

---

## 324. Balancing Understanding and Generation in Discrete Diffusion Models

**Abstract:** In discrete generative modeling, two dominant paradigms demonstrate divergent capabilities: Masked Diffusion Language Models (MDLM) excel at semantic understanding and zero-shot generalization, whereas Uniform-noise Diffusion Language Models (UDLM) achieve strong few-step generation quality, yet neither attains balanced performance across both dimensions. To address this, we propose XDLM, which bridges the two paradigms via a stationary noise kernel. XDLM offers two key contributions: it provides (1) a principled theoretical unification of MDLM and UDLM, recovering each paradigm as a special case; and (2) an alleviated memory bottleneck enabled by an algebraic simplification of the posterior probabilities. Experiments demonstrate that XDLM advances the Pareto frontier between understanding capability and generation quality. Quantitatively, XDLM surpasses UDLM by 5.4 points on zero-shot text benchmarks and outperforms MDLM in few-step image generation (FID 54.1 vs. 80.8).  When scaled to tune an 8B-parameter large language model, XDLM achieves 15.0 MBPP in just 32 steps, effectively doubling the baseline performance. Finally, analysis of training dynamics reveals XDLM’s superior potential for long-term scaling. Code is available at [https://github.com/MzeroMiko/XDLM](https://github.com/MzeroMiko/XDLM).

Forum: https://openreview.net/forum?id=pZNo1YWT5x

---

## 325. Reverse Flow Matching: A Unified Framework for Online Reinforcement Learning with Diffusion and Flow Policies

**Abstract:** Diffusion and flow policies are gaining prominence in online reinforcement learning (RL) due to their expressive power, yet training them efficiently remains a critical challenge. A fundamental difficulty that distinguishes online RL from standard generative modeling is the lack of direct samples from the target Boltzmann distribution defined by the Q-function. To address this, two seemingly distinct families of methods have been proposed for diffusion policies: a noise-expectation family, which uses a weighted average of noise as the training target, and a gradient-expectation family, which employs a weighted average of Q-function gradients. However, it remains unclear how these objectives are formally related, or whether they can be synthesized into a more general formulation. In this paper, we propose a unified framework, reverse flow matching (RFM), which rigorously addresses the problem of training diffusion and flow models without direct target samples. By adopting a reverse inferential perspective, we formulate the training target as a posterior mean estimation problem given an intermediate noisy sample. Crucially, we introduce Langevin Stein operators to construct zero-mean control variates, deriving a general class of estimators that share the same expectation. We show that existing noise-expectation and gradient-expectation methods are simply two specific instances within this broader class. This unified view yields two key advancements: it extends the capability of targeting Boltzmann distributions from diffusion to flow policies, and it enables the principled combination of Q-value and Q-gradient information to form an effective estimator, thereby improving training efficiency and stability. We instantiate RFM to train a flow policy in online RL and demonstrate improved performance on continuous-control benchmarks compared to diffusion policy baselines.

Forum: https://openreview.net/forum?id=vUpEe4yd1T

---

## 326. UDM-GRPO: Stable and Efficient Group Relative Policy Optimization for Uniform Discrete Diffusion Models

**Abstract:** Uniform Discrete Diffusion Model (UDM) has recently emerged as a promising paradigm for discrete generative modeling; however, its integration with reinforcement learning remains largely unexplored. We observe that naively applying GRPO to UDM leads to training instability and marginal performance gains. To address this, we propose **UDM-GRPO**, the first framework to integrate UDM with RL. Our method is guided by two key insights: (i) treating the final clean sample as the action provides more accurate and stable optimization signals; and (ii) reconstructing trajectories via the diffusion forward process better aligns probability paths with the pretraining distribution. Additionally, we introduce two strategies, Reduced-Step and CFG-Free, to further improve training efficiency. **UDM-GRPO** significantly improves base model performance across multiple T2I tasks. Notably, GenEval accuracy improves from $69\\%$ to $96\\%$ and PickScore increases from $20.46$ to $23.81$, achieving state-of-the-art performance in both continuous and discrete settings. On the OCR benchmark, accuracy rises from $8\\%$ to $57\\%$, further validating the generalization ability of our method. Code is available at https://github.com/Yovecent/UDM-GRPO.

Forum: https://openreview.net/forum?id=WJcFtJriqv

---

## 327. Training Diffusion Language Models for Black-Box Optimization

**Abstract:** We study offline black-box optimization (BBO), aiming to discover improved designs from an offline dataset of designs and labels, a problem common in robotics, DNA, and materials science with limited labeled samples. While recent work applies autoregressive LLMs to BBO by formatting tasks as natural-language prompts, their left-to-right design generation struggles to capture the strong bidirectional dependencies inherent in design problems. To address this, we propose adapting diffusion LLMs to offline BBO to leverage their bidirectional modeling capabilities. However, a domain gap exists between the natural text pre-training of diffusion LLMs and the heterogeneous signals in BBO (prompts, designs, and labels). To bridge this gap, we construct a unified prompt–-response corpus and introduce delimiter tokens to explicitly mark field boundaries for domain adaptation. We further propose a two-stage post-training framework to align the diffusion LLM generation with high-label designs. The first stage performs supervised fine-tuning on the unified dataset via masked-response prediction, and the second stage adopts reinforcement learning with rewards defined by label improvements. Our method achieves state-of-the-art results on Design-Bench under small-data settings. Code for our work is available here: https://github.com/zpointS/DiBO.

Forum: https://openreview.net/forum?id=Z7wI0sor6i

---

## 328. Conformal Policy Control

**Abstract:** An agent must try new behaviors to explore and improve. In high-stakes environments, an agent that violates safety constraints may cause harm and must be taken offline, curtailing any future interaction. Imitating old behavior is safe, but excessive conservatism discourages exploration. How much behavior change is too much? We show how to use any safe reference policy as a probabilistic regulator for any optimized but untested policy. Conformal calibration on data from the safe policy determines how aggressively the new policy can act, while provably enforcing the user's declared risk tolerance. Unlike conservative optimization methods, we do not assume the user has identified the correct model class nor tuned any hyperparameters. Unlike previous conformal methods, our theory provides finite-sample guarantees even for non-monotonic bounded loss functions. Our experiments on applications ranging from natural language question answering to biomolecular engineering show that safe exploration is not only possible from the first moment of deployment, but can also improve performance.

Forum: https://openreview.net/forum?id=rbwy4E0Kyz

---

## 329. Guaranteed Optimal Compositional Explanations for Neurons

**Abstract:** Compositional explanations are a family of methods that aim to describe the spatial alignment between neurons' receptive field activations and concepts through logical rules, typically computed via a search over all possible concept combinations. Since computing the spatial alignment over the entire state space is computationally infeasible, the literature commonly adopts assumptions related to the structure of the combinations and beam search to restrict the state space. However, beam search cannot provide any theoretical guarantees of optimality, and it remains unclear how close current explanations are to the true optimum. In this theoretical paper, we address this gap by introducing the first framework for computing guaranteed optimal compositional explanations over the entire state space spanned by the adopted assumptions. Specifically, we propose: (i) a decomposition that identifies the factors influencing the spatial alignment, (ii) a heuristic to estimate the alignment at any stage of the search, and (iii) the first algorithm that can compute optimal compositional explanations in a time comparable to exhaustive beam search. Using this framework,  we demonstrate that 10-40\% of explanations previously obtained with beam search are suboptimal when overlapping concepts are involved. Finally, we evaluate a beam-search variant guided by our proposed decomposition and heuristic, showing that it matches or improves runtime over prior methods while offering greater flexibility in hyperparameters and computational resources.

Forum: https://openreview.net/forum?id=MHiiwC3oFR

---

## 330. Conservation Laws for Modern Neural Architectures

**Abstract:** Understanding gradient descent dynamics is key to explaining the success of over-parameterized models, where implicit bias manifests through conservation laws in gradient flow. While such laws are well understood for linear and ReLU networks, they remain largely unexplored for modern architectures. This work develops a unified framework to characterize conservation laws for contemporary models, including feedforward networks with GELU, SiLU, and SwiGLU activations, multihead attention with sinusoidal and rotary positional encodings, and Mixture-of-Experts architectures under diverse gating designs. Our theoretical findings are supported by experiments that validate the predicted invariants.

Forum: https://openreview.net/forum?id=ay4Q69fAJL

---

## 331. NonZero: Interaction-Guided Exploration for Multi-Agent Monte Carlo Tree Search

**Abstract:** Monte Carlo Tree Search (MCTS) scales poorly in cooperative multi-agent domains because expansion must consider an exponentially large set of joint actions, severely limiting exploration under realistic search budgets. We propose \textsc{NonZero}, which keeps multi-agent MCTS tractable by running surrogate-guided selection over a low-dimensional nonlinear representation using an interaction-guided proposal rule, instead of directly exploring the full joint-action space. Our exploration uses an interaction score: single-agent deviations are ranked by predicted gain, while two-agent deviations are scored by a mixed-difference measure that reveals coordination benefits even when no single agent can improve alone. We formalize candidate proposal as a bandit problem over local deviations and derive a proposal rule, \textsc{NonUCT}, with a sublinear local-regret guarantee for reaching approximate graph-local optima without enumerating the joint-action space. Empirically, \textsc{NonZero} improves sample efficiency and final performance on MatGame, SMAC, and SMACv2 relative to strong model-based and model-free baselines under matched search budgets.

Forum: https://openreview.net/forum?id=Jh6gq9QsFa

---

## 332. PLANTAIN: Plan-Answer Interleaved Reasoning

**Abstract:** Reasoning models often spend significant time generating hidden reasoning before any visible response, which can waste user time when the model starts from a false premise that could have been corrected early. Human speakers, in contrast, use lightweight incremental check-ins to maintain common ground, motivating *interleaved reasoning* (IR), where a model alternates between internal thinking and visible intermediate responses. We instantiate this idea with PLAINTAIN (Plan-Answer Interleaved Reasoning), a post-training recipe that teaches a model to externalize an explicit step-by-step plan before continuing its reasoning. This learned plan-first structure creates an interface for early feedback and intervention while preserving space for subsequent reasoning. Across challenging math, coding, text-to-SQL, and reading-comprehension benchmarks, PLAINTAIN improves pass@1 by roughly 6% on average while reducing time-to-first-response by over 60% relative to think-then-answer baselines.

Forum: https://openreview.net/forum?id=lK2o9OjoXf

---

## 333. Learning Hamiltonian Flow Maps: Mean Flow Consistency for Large-Timestep Molecular Dynamics

**Abstract:** Simulating the long-time evolution of Hamiltonian systems is limited by the small timesteps required for stable numerical integration. To overcome this constraint, we introduce a framework to learn *Hamiltonian Flow Maps* by predicting the *mean* phase-space evolution over a chosen time span $\Delta t$, enabling stable large-timestep updates far beyond the stability limits of classical integrators. To this end, we impose a *Mean Flow* consistency condition for time-averaged Hamiltonian dynamics. Unlike prior approaches, this allows training on independent phase-space samples without access to future states, avoiding expensive trajectory generation. Validated across diverse Hamiltonian systems, our method in particular improves upon molecular dynamics simulations using machine-learned force fields (MLFF). Our models maintain comparable training and inference cost, but support significantly larger integration timesteps while trained directly on widely-available *trajectory-free* MLFF datasets.

Forum: https://openreview.net/forum?id=EBSn23DLwB

---

## 334. Real-Time Visual Attribution Streaming in Thinking Model

**Abstract:** We present an amortized framework for real-time visual attribution streaming in multimodal thinking models. When these models generate code from a screenshot or solve math problems from images, their long reasoning traces should be grounded in visual evidence. However, verifying this reliance is challenging: faithful causal methods require costly repeated backward passes or perturbations, while raw attention maps offer instant access, they lack causal validity. To resolve this, we introduce an amortized approach that learns to estimate the causal effects of semantic regions directly from the rich signals encoded in attention features. Across five diverse benchmarks and four thinking models, our approach achieves faithfulness comparable to exhaustive causal methods while enabling visual attribution streaming, where users observe grounding evidence as the model reasons, not after. Our results demonstrate that real-time, faithful attribution in multimodal thinking models is achievable through lightweight learning, not brute-force computation.

Forum: https://openreview.net/forum?id=eVr10aZZIw

---

## 335. Effective Model Pruning : Measuring the Redundancy of Model Components

**Abstract:** This article initiates the study of a basic question about model pruning. Given a vector s of importance scores assigned to model components, how many of the scored components could be discarded without sacrificing performance? We propose Effective Model Pruning (EMP), which derives the desired sparsity directly from the score distribution using the notion of effective sample size from particle filtering, also known as the inverse Simpson index.

Rather than prescribe a pruning criterion, EMP supplies a universal adaptive threshold derived from the distribution of the score $s$ over the model components: EMP maps $s$ to a number $N_{eff} = N_{eff} (s)$, called the effective sample size. The $N − N_{eff}$ lowest scoring components are discarded. A tight lower bound on the preserved mass fraction seff (the sum of retained normalized scores) in terms of $N_{eff}$ is derived. This process yields models with a provable upper bound on the loss change relative to the original dense
model. Numerical experiments are performed demonstrating this phenomenon across a variety of network architectures including MLPs, CNNs, Transformers, LLMs, and KAN. It is also shown that EMP addresses a rich set of pruning criteria such as weight magnitude, attention score, KAN importance score, and even feature-level signals such as image pixels.

Forum: https://openreview.net/forum?id=c2CdXdEfqk

---

## 336. Optimal and Scalable MAPF via Multi-Marginal Optimal Transport and Schrödinger Bridges

**Abstract:** We consider anonymous multi-agent path finding (MAPF) where a set of robots is tasked to travel to a set of targets on a finite, connected graph. We show that MAPF can be cast as a special class of multi-marginal optimal transport (MMOT) problems with an underlying Markovian structure, under which the exponentially large MMOT collapses to a linear program (LP) polynomial in size. Focusing on the anonymous setting, we establish conditions under which the corresponding LP is feasible, totally unimodular, and yields min-cost, integral~$(\{0,1\})$ transports that do not overlap in both space and time. To adapt the approach to large-scale problems, we cast the MAPF-MMOT in a probabilistic framework via Schrödinger bridges. Under standard assumptions, we show that the Schrödinger bridge formulation reduces to an entropic regularization of the corresponding MMOT that admits an iterative Sinkhorn-type solution. The Schrödinger bridge, being a probabilistic framework, provides a shadow (fractional) transport that we use as a template to solve a reduced LP and demonstrate that it results in near-optimal, integral transports at a significant reduction in complexity. Extensive experiments highlight the optimality and scalability of the proposed approaches.

Forum: https://openreview.net/forum?id=Cxdj2GYZ4c

---

## 337. A Factorized Low-Rank RNN Framework for Uncovering Independent Neural Latent Dynamics and Connectivity

**Abstract:** Low-rank recurrent neural networks (lrRNNs) are a class of models that uncover low-dimensional latent dynamics underlying neural population activity. Although their functional connectivity is low-rank, it lacks independence interpretations, making it difficult to assign distinct computational roles to different latent dimensions. To address this, we propose the Factored Recurrent Neural Network (FacRNN), a generative lrRNN framework that assumes group-wise independence among latent dynamics while allowing flexible within-group entanglement. These independent latent groups allow latent dynamics to evolve separately, but are internally rich for complex computation. We reformulate the lrRNN under a variational autoencoder (VAE) framework, enabling us to introduce a partial correlation penalty that encourages independence between groups of latent dimensions. Experiments on synthetic, monkey M1, and mouse voltage imaging data show that FacRNN consistently improves the disentanglement and interpretability of learned neural latent trajectories in low-dimensional space and low-rank connectivity over baseline lrRNNs that do not encourage group-wise independence.

Forum: https://openreview.net/forum?id=QvIbmX9jBr

---

## 338. Distributional Inverse Reinforcement Learning

**Abstract:** We propose a distributional framework for offline Inverse Reinforcement Learning (IRL) that jointly models uncertainty over reward functions and full distributions of returns. Unlike conventional IRL approaches that recover a deterministic reward estimate or match only expected returns, our method captures richer structure in expert behavior, particularly in learning the reward distribution, by minimizing first-order stochastic dominance (FSD) violations and thus integrating distortion risk measures (DRMs) into policy learning, enabling the recovery of both reward distributions and distribution-aware policies. This formulation is well-suited for behavior analysis and risk-aware imitation learning. Theoretical analysis show that the algorithm converge with $\mathcal{O}(\varepsilon^{-2})$ iteration complexity. Empirical results on synthetic benchmarks, real-world neurobehavioral data, and MuJoCo control tasks demonstrate that our method recovers expressive reward representations and achieves state-of-the-art imitation performance.

Forum: https://openreview.net/forum?id=ZcnZ6vs4yX

---

## 339. What You Think is What You See: Driving Exploration in VLM Agents via Visual-Linguistic Curiosity

**Abstract:** To navigate partially observable visual environments, recent VLM agents increasingly internalize world modeling capabilities directly into their policies via explicit CoT reasoning with reinforcement learning (RL). However, mere passive exploitation of reasoning on visited states is insufficient for sparse-reward agentic tasks, as it lacks the epistemic drive to actively uncover the *known unknown* required for robust generalization. We ask: *Can VLM agents actively find signals that challenge and update their internal world model through curiosity-driven exploration?* In this work, we propose **GLANCE**, a unified framework that bridges reasoning and exploration by grounding the agent's linguistic world model into the stable visual representations of an evolving target network. Crucially, **GLANCE** leverages the discrepancy between linguistic prediction and visual reality as an intrinsic curiosity signal within reinforcement learning, steering the agent to actively explore areas where its internal model is uncertain. Extensive experiments across a series of agentic tasks show the effectiveness of **GLANCE**, and demonstrate that aligning *what the agent thinks* with *what the agent sees* is key to solving complex or sparse agentic tasks.

Forum: https://openreview.net/forum?id=zWStW3hrfW

---

## 340. From Poisoned to Aware: Fostering Backdoor Self-Awareness in LLMs

**Abstract:** Backdoor attacks can introduce deceptive behaviors into large language models, causing them to execute prohibited actions only when specific secret triggers appear in the input. Existing safety training methods largely fail to address this vulnerability, due to the inherent difficulty of uncovering hidden triggers embedded within the model. Motivated by recent findings on LLMs’ situational awareness, we propose a novel post-training framework that cultivates backdoor self-awareness, enabling a poisoned LLM to precisely articulate its own implanted triggers. At its core, our approach introduces an inversion-inspired reinforcement learning framework that encourages models to introspectively reason about their behaviors and gradually reverse-engineer the triggers responsible for misaligned outputs. Building upon precise trigger articulation, we further present two complementary defense strategies for mitigating and detecting backdoor threats. Experiments on five backdoor attacks, compared against six baseline methods, demonstrate that our approach has strong potential to improve the robustness of LLMs against backdoor risks.

Forum: https://openreview.net/forum?id=lUVEQ1JRIL

---

## 341. CSPO: Constraint-Sensitive Policy Optimization for Safe Reinforcement Learning

**Abstract:** Safe reinforcement learning (Safe RL) aims to maximize expected return while satisfying safety constraints, typically modeled as Constrained Markov Decision Processes (CMDPs). While primal-dual methods scale well to deep RL, they often suffer from delayed constraint correction, leading to oscillatory behavior and prolonged safety violations. In this paper, we propose *Constraint-Sensitive Policy Optimization (CSPO)*, a first-order primal-dual method that incorporates local constraint sensitivity into policy updates. CSPO augments the primal objective with a constraint-sensitive correction derived from the shortest signed distance to the safety boundary, enabling smarter recovery steps back to safety, compensating for delayed Lagrange multiplier updates, reducing oscillations near the boundary, and preserving the KKT solutions of the original constrained problem. Experiments on navigation and locomotion benchmarks demonstrate that CSPO achieves faster safety recovery and high reward preservation, resulting in higher constrained returns compared to state-of-the-art primal-dual and penalty-based methods.

Forum: https://openreview.net/forum?id=3ySR3TCMRP

---

## 342. SlaClip: Gradient Norm Slacks can be Indicator for Adaptive Clipping in DP-SGD

**Abstract:** Differentially private stochastic gradient descent (DP-SGD) achieves privacy by clipping per-sample gradients and injecting Gaussian noise, but its utility is highly sensitive to the choice of the clipping threshold $C$. A fixed $C$ often degrades performance and necessitates repeated empirical calibration. Existing adaptive clipping methods either modify the gradient update in vanilla DP-SGD, causing additional tuning or optimization overhead, or introduce separate private queries to monitor gradient statistics. In contrast, we leverage the *slack* information induced by the standard clipping operation, an overlooked signal in prior work, and show that it provides an effective indication for adapting $C$. 
In light of this, we propose *SlaClip*, a privacy-preserving adaptive clipping strategy using a post-hoc *Slack Indicator*. Under the same training configuration and privacy accountant, *SlaClip* preserves the sampling rule, noise multiplier, and global $\ell_2$ sensitivity bound of vanilla DP-SGD. Therefore, *SlaClip* is a plug-and-play module for vanilla DP-SGD and its variants. Moreover, *SlaClip* is accounted under the same per-step privacy bound, while requiring no additional private query. Across diverse datasets and tasks, experiments show that *SlaClip* consistently outperforms baseline adaptive clipping methods.

Forum: https://openreview.net/forum?id=48suUeYKdb

---

## 343. Loss-Aware Distributionally Robust Optimization via Trainable Optimal Transport Ambiguity Sets

**Abstract:** Optimal-transport distributionally robust optimization (OT-DRO) robustifies data-driven decision-making under uncertainty by capturing the sampling-induced statistical error via optimal transport ambiguity sets. The standard OT-DRO pipeline consists of a two-step procedure, where the ambiguity set is first designed and subsequently embedded into the downstream OT-DRO problem. However, this separation between uncertainty quantification and optimization may lead to excessive conservatism. We introduce an end-to-end pipeline to automatically learn decision-focused ambiguity sets for OT-DRO problems, where the loss function informs the shape of the ambiguity set, leading to less conservative decisions whose distributional robustness is enforced via data-driven bootstrapping. We formulate the learning problem as a bilevel optimization program and solve it via a hypergradient-based method. By leveraging the recently introduced nonsmooth conservative implicit function theorem, we establish convergence to a critical point of the bilevel problem. We present experiments validating our method on standard portfolio optimization and linear regression tasks.

Forum: https://openreview.net/forum?id=K1EPPO9t2c

---

## 344. Clustering in Deep Stochastic Transformers

**Abstract:** Transformers have revolutionized deep learning across various domains but understanding the precise token dynamics remains a theoretical challenge. Existing theories of deep Transformers with layer normalization typically predict that tokens cluster to a single point; however, these results rely on deterministic weight assumptions, which fail to capture the standard initialization scheme in Transformers.
In this work, we show that accounting for the intrinsic stochasticity of random initialization alters this picture. More precisely, we analyze deep Transformers where noise arises from the random initialization of value matrices. Under diffusion scaling and token-wise RMS normalization, we prove that, as the number of Transformer layers goes to infinity, the discrete token dynamics converge to an interacting-particle system on the sphere where tokens are driven by a *common* matrix-valued Brownian noise. In this limit, we show that initialization noise prevents the collapse to a single cluster predicted by deterministic models. For two tokens, we prove a phase transition governed by the interaction strength and the token dimension: unlike deterministic attention flows, antipodal configurations become attracting with positive probability. Numerical experiments confirm the predicted transition, reveal that antipodal formations persist for more than two tokens, and demonstrate that suppressing the intrinsic noise degrades accuracy.

Forum: https://openreview.net/forum?id=opfSL15Uc4

---

## 345. SpatioLM: Towards General Physical Spatial Intelligence in Vision-Language Models

**Abstract:** Vision-Language Models (VLMs) perform well on commonsense reasoning tasks but struggle with visual spatial reasoning. Most existing solutions introduce extra 3D prior inputs or external spatial encoders, which increase complexity and degrade the underlying VLMs' general-purpose capabilities after spatial fine-tuning. To this end, we propose a parameter-efficient \textit{\textbf{Spatio}-vision \textbf{L}anguage \textbf{M}odels (SpatioLM)}, that enhances spatial intelligence without extra 3D prior inputs or third-party spatial encoders. Concretely, we design a plug-and-play and non-invasive spatio-vision module that elicits the spatial knowledge inherent in VLMs. Furthermore, we innovatively leverage pseudo depth and camera information as supervision to guide the model in learning physically coherent representations. Extensive experiments show that SpatioLM achieves significant improvements in diverse tasks, including spatial perception and understanding while effectively limiting the degradation of general capabilities. Notably, the model achieves an impressive score of 71.6 on the VSI-Bench (the first model to surpass 70). In addition, it attains competitive performance when transferred to embodied manipulation tasks.

Forum: https://openreview.net/forum?id=CHavqrN1X9

---

## 346. Correcting Split Selection in Online Decision Trees via Anytime-Valid Inference

**Abstract:** Bagging-based ensembles, most notably Adaptive Random Forests, are among the strongest performers for learning from data streams. A common denominator across these methods is their reliance on Hoeffding Trees as base learners, which grow decision trees incrementally by testing whether a candidate split is significantly better than its alternatives using concentration inequalities. Despite their empirical success, existing Hoeffding Trees variants lack valid statistical guarantees. Current analyses rely on fixed-sample concentration bounds, while split decisions are made using data-dependent stopping rules, which invalidates their guarantees and can drive the probabilty of incorrect splits to one. We introduce a principled alternative based on \emph{anytime-valid inference}. Our method provides: (i) anytime-valid control of false splits under arbitrary data streams, including non-stationary settings; (ii) finite commitment time under a predictive advantage; and (iii) under stationary i.i.d.\ data, risk is monotone decreasing and strictly improves at every split. Empirically, we evaluate both standalone trees and their use within Adaptive Random Forests on non-stationary streams. Our method improves performance while producing substantially smaller trees.

Forum: https://openreview.net/forum?id=fZvEZQWJrR

---

## 347. Control Consistency Losses for Diffusion Bridges

**Abstract:** Simulating the conditioned dynamics of diffusion processes, given their initial and terminal states, is an important but challenging problem in the sciences. The difficulty is particularly pronounced for rare events, for which the unconditioned dynamics rarely reach the terminal state. In this work, we propose a novel approach for learning diffusion bridges based on a self-consistency property of the optimal control. The resulting algorithm learns the conditioned dynamics in an iterative online manner, and exhibits strong performance in a range of empirical settings without requiring differentiation through simulated trajectories. Beyond the diffusion bridge setting, we draw connections between our self-consistency framework and recent advances in the wider stochastic optimal control literature.

Forum: https://openreview.net/forum?id=EA0wpmic0X

---

## 348. Simultaneous Speech-to-Speech Translation Without Aligned Data

**Abstract:** Simultaneous speech translation requires translating source speech into a target language in real-time while handling non-monotonic word dependencies. Traditional approaches rely on supervised training with word-level aligned data, which is difficult to collect at scale and thus depends on synthetic alignments using language-specific heuristics that are suboptimal. We propose Hibiki-Zero, which eliminates the need for word-level alignments entirely. This fundamentally simplifies the training pipeline and enables seamless scaling to diverse languages with varying grammatical structures, removing the bottleneck of designing language-specific alignment heuristics. We first train on sentence-level aligned data to learn speech translation at high latency, then apply a novel reinforcement learning strategy using GRPO to optimize latency while preserving translation quality. Hibiki-Zero achieves state-of-the-art performance in translation accuracy, latency, voice transfer, and naturalness across four X-to-English tasks.  Moreover, we demonstrate that our model can be adapted to support a new input language with less than 1000h of speech. We provide [examples](https://huggingface.co/spaces/kyutai/hibiki-zero-samples), [model weights](https://huggingface.co/kyutai/hibiki-zero-3b-pytorch-bf16), [inference code](https://github.com/kyutai-labs/hibiki-zero) and we release a [benchmark](https://huggingface.co/datasets/kyutai/Audio-NTREX-4L) containing 45h of multilingual data for speech translation evaluation.

Forum: https://openreview.net/forum?id=76XSBLdBdg

---

## 349. Skill-Pro: Learning Reusable Skills from Experience via Non-Parametric PPO for LLM Agents

**Abstract:** LLM-driven agents excel at sequential decision-making but often rely on on-the-fly reasoning, re-deriving solutions even in recurring scenarios. This insufficient experience reuse leads to computational redundancy and instability. To bridge this gap, we propose **Skill-Pro**, a framework enabling agents to autonomously learn reusable procedural skills from interaction experiences without parameter updates. By formalizing a **Skill-MDP**, Skill-Pro transforms passive episodic narratives into executable Skills defined by activation, execution, and termination conditions to ensure executability. 
To achieve reliable reusability without capability degradation, we introduce **Non-Parametric PPO**, which leverages semantic gradients for high-quality candidate generation and a PPO Gate for robust Skill verification. Through score-based maintenance, Skill-Pro sustains compact, high-quality procedural memory.
Experimental results across in-domain, cross-task, and cross-agent scenarios demonstrate that Skill-Pro achieves superior reuse rates and significant gains with extreme memory compression. Visualized evolutionary trajectories and Skill distributions further reveal how Skill-Pro transparently accumulates, refines, and reuses procedural knowledge to facilitate long-term autonomy.

Forum: https://openreview.net/forum?id=9kJQjx2B80

---

## 350. DISCO: Mitigating Bias in Deep Learning with Conditional Distance Correlation

**Abstract:** Dataset bias often leads deep learning models to exploit spurious correlations instead of task-relevant signals. We introduce the Standard Anti-Causal Model (SAM), a unifying causal framework that characterizes bias mechanisms and yields a conditional independence criterion for causal stability. Building on this theory, we propose DISCO$_m$ and sDISCO, efficient and scalable estimators of conditional distance correlation that enable independence regularization in gradient-based models. Across six diverse datasets, our methods consistently outperform or are competitive in existing observed bias mitigation approaches, while requiring fewer hyperparameters and scaling seamlessly to multi-bias scenarios. This work bridges causal theory and practical deep learning, providing both a principled foundation and effective tools for robust prediction. Source Code: https://github.com/yakamoz5/DISCO.

Forum: https://openreview.net/forum?id=TYjTWxDOOC

---

## 351. How RL Unlocks the Aha Moment in Geometric Interleaved Reasoning

**Abstract:** Solving complex geometric problems inherently requires interleaved reasoning: a tight alternation between constructing diagrams and performing logical deductions. Although recent Multimodal Large Language Models (MLLMs) have demonstrated strong capabilities in visual generation and plotting, we identify a counter-intuitive and underexplored phenomenon. Naively applying Supervised Fine-Tuning (SFT) on interleaved plot–solution data leads to a substantial degradation in reasoning performance compared to text-only baselines. We argue that this failure stems from a fundamental limitation of SFT, which primarily induces distributional alignment: the model learns to reproduce the surface format of interleaved plotting but fails to internalize the causal dependency between the generated plot and reasoning steps. To overcome this limitation, we propose Faire (**F**unctional **a**lignment for **i**nterleaved **re**asoning), a reinforcement learning framework that enforces three casual constraints to move beyond superficial imitation toward functional alignment. Extensive experiments show that Faire induces a qualitative shift in model behavior in which the plotting is effectively internalized, yielding competitive performance on challenging geometric reasoning benchmarks.

Forum: https://openreview.net/forum?id=JT4zAf3zDs

---

## 352. Teaching Models to Teach Themselves: Reasoning at the Edge of Learnability

**Abstract:** RL methods for scaling large reasoning models stall on datasets with low initial success rates, and thus little training signal. We investigate a fundamental question:Can a pretrained LLM leverage latent knowledge to generate an automated curriculum  for problems it cannot solve? We explore this with SOAR: An asymmetric self-play framework that uses meta-RL to surface these pedagogical signals. A teacher model proposes synthetic problems for a student model, and is rewarded with its improvement on a subset of hard problems, thus grounding the curriculum in real student progress rather than intrinsic proxy rewards. Our study on the hardest subsets of math benchmarks (0/128 success) reveals three core findings. First, it is possible to realize bilevel meta-RL that unlocks learning under sparse, binary rewards by sharpening a latent capacity of pretrained models to generate useful problems. Second, grounded rewards outperform intrinsic learnability rewards used in prior LLM self-play, reliably avoiding typical instability and diversity collapse modes. Third, the structure and well-posedness of questions are more critical for learning progress than solution correctness. Our results suggest that the ability to generate useful stepping stones does not require the preexisting ability to solve the hard problems, paving a principled path to escape reasoning plateaus without additional curated data.

Forum: https://openreview.net/forum?id=GnqHK8Ww98

---

## 353. Chebyshev Policies and the Mountain Car Problem: Reinforcement Learning for Low-Dimensional Control Tasks

**Abstract:** We analytically solve the Mountain Car problem, a canonical benchmark in RL, and derive an optimal control solution, closing a gap after 36 years. This enables us to reveal two surprising insights: The optimal control is quite simple, yet modern RL agents display a large gap to optimality. Motivated by the analysis of the optimal control, we introduce Chebyshev policies as a universal (i.e. dense) class of RL policies from first principles. They can be trained as drop-in replacements of neural nets, reducing the regret by a factor of 4.18, while requiring 277 times fewer parameters, fostering sample efficiency, explainability and realtime capability. Chebyshev policies are evaluated on further RL tasks, including a real-world nonlinear motion control testbed. They consistently improve performance over neural nets with PPO, ARS and REINFORCE. Our results demonstrate how Chebyshev policies offer a compelling and lightweight alternative or addition to neural nets for low-dimensional control tasks.

Forum: https://openreview.net/forum?id=aNWIVNjocB

---

## 354. AdLift: Lifting Adversarial Perturbations to Safeguard 3D Gaussian Splatting Assets Against Instruction-Driven Editing

**Abstract:** Recent studies have extended instruction-driven 2D editing pipelines to 3D Gaussian Splatting (3DGS), enabling faithful 3DGS asset manipulation for advanced content creation. However, it also exposes 3DGS assets to serious risks of unauthorized editing and malicious tampering. Although adversarial perturbations against editing models have proven effective for protecting 2D images, applying them to 3DGS encounters two major challenges: *view-generalizable protection* and *balancing invisibility with protection capability*. In this work, we propose AdLift, a novel editing safeguard for 3DGS that prevents instruction-driven editing across arbitrary views and dimensions by lifting strictly bounded 2D adversarial perturbations into 3D Gaussian-represented safeguard. To ensure both *protective effectiveness* and *invisibility*, these safeguard Gaussians are progressively optimized across training views using a tailored Lifted PGD, which first conducts *gradient truncation* during back-propagation from the editing model to the rendered image and applies projected gradient updates to strictly bound image-level perturbations. Then, the resulting perturbation is backpropagated to the safeguard Gaussian parameters via *image-to-Gaussian fitting*. We alternate these two steps, yielding effective and imperceptible protection that generalizes across both training and novel views. Empirically, qualitative and quantitative results demonstrate that the proposed AdLift effectively protects against state-of-the-art instruction-driven 2D and 3DGS editing.

Forum: https://openreview.net/forum?id=RL565ePGHR

---

## 355. OXE-AugE: A Large-Scale Robot Augmentation of OXE for Scaling Cross-Embodiment Policy Learning

**Abstract:** Large and diverse datasets are needed for training generalist robot policies that can control a variety of robot embodiments--robot arm and gripper combinations--across diverse tasks and environments. As re-collecting demonstrations and retraining for each new embodiment are prohibitively costly, we study whether existing robot data can be augmented to improve transfer and generalization across embodiments. The Open X-Embodiment (OXE) dataset, which aggregates demonstrations from over 60 robot datasets, has been widely used for training generalist policies. However, it is highly imbalanced: the top four robot types account for over 85% of its real data, which risks overfitting to robot--scene combinations. We present AugE-Toolkit, a scalable robot augmentation pipeline, and OXE-AugE, a high-quality open-source dataset that augments OXE with 9 different robot embodiments. OXE-AugE provides over 4.4 million trajectories, more than triple the size of the original OXE. We conduct a systematic study of how scaling robot augmentation impacts cross-embodiment learning. Results suggest that augmenting datasets with diverse arms and grippers improves policy performance not only on the augmented robots, but also on unseen robots and even the original robots under distribution shifts. In physical experiments, fine-tuning generalist policies such as OpenVLA and $\pi_0$ on OXE-AugE improves success rates by 24-45% on unseen robot-gripper combinations across four real-world manipulation tasks. Project website: https://OXE-AugE.github.io/.

Forum: https://openreview.net/forum?id=LcswwEzzX7

---

## 356. CoEvol-NO: State and Coordinate Co-Evolution with an Error-Driven Predictor-Corrector Paradigm for Neural Operator Transformer

**Abstract:** Despite the fast progress in neural operator learning, long-sequence modeling still is a standing challenge whereby latent states have been introduced with techniques well derived.  Diverging from existing methods that treat latent states as transient variables or decoupled representations, CoEvol-NO introduces a persistent state to establish a co-evolutionary framework, where the latent state and mesh sequence are updated jointly and bidirectionally. Inspired by classical numerical methods, we model the layer-wise state evolution as a Predictor-Corrector (PC) process. Specifically, a "Predictor'' generates a tentative target, followed by a "Corrector'' that refines the persistent state via an {error-driven update mechanism}. Furthermore, our theoretical analysis reveals that the widely used \textit{direct substitution} and \textit{residual update} paradigms are essentially {first-order approximations} of this error-driven correction under different loss assumptions. We theoretically prove that CoEvol-NO achieves strict linear time complexity. Extensive experiments on five standard benchmarks and two large-scale industrial design tasks demonstrate that CoEvol-NO consistently achieves state-of-the-art (SOTA) performance.

Forum: https://openreview.net/forum?id=pG9Wn3Aheq

---

## 357. Reinforced Sequential Monte Carlo for Amortised Sampling

**Abstract:** This paper proposes a synergy of amortised and particle-based methods for sampling from distributions defined by unnormalised density functions. We state a connection between sequential Monte Carlo (SMC) and neural sequential samplers trained by maximum-entropy reinforcement learning (MaxEnt RL), wherein learnt sampling policies and value functions define proposal kernels and twist functions. Exploiting this connection, we introduce an off-policy RL training procedure for the sampler that uses samples from SMC -- using the learnt sampler as a proposal -- as a behaviour policy that better explores the target distribution. We describe techniques for stable joint training of proposals and twist functions and an adaptive weight tempering scheme to reduce training signal variance. Furthermore, building upon past attempts to use experience replay to guide the training of neural samplers, we derive a way to combine historical samples with annealed importance sampling weights within a replay buffer. On synthetic multi-modal targets (in both continuous and discrete spaces) and the Boltzmann distribution of alanine dipeptide conformations, we demonstrate improvements in approximating the true distribution as well as training stability compared to both amortised and Monte Carlo methods.

Forum: https://openreview.net/forum?id=DWaToCuNwa

---

## 358. Learning Structured Reasoning via Tractable Trajectory Control

**Abstract:** Large language models can exhibit emergent reasoning behaviors, often manifested as recurring lexical patterns (e.g., “wait,” indicating verification). However, complex reasoning trajectories remain sparse in unconstrained sampling, and standard RL often fails to guarantee the acquisition of diverse reasoning behaviors. We propose a systematic discovery and reinforcement of diverse reasoning patterns through structured reasoning, a paradigm that requires targeted exploration of specific reasoning patterns during the RL process.
To this end, we propose Ctrl-R, a framework for learning structured reasoning via tractable trajectory control that actively guides the rollout process, incentivizing the exploration of diverse reasoning patterns that are critical for complex problem-solving. The resulting behavior policy enables accurate importance-sampling estimation, supporting unbiased on-policy optimization. We further introduce a power-scaling factor on the importance-sampling weights, allowing the policy to selectively learn from exploratory, out-of-distribution trajectories while maintaining stable optimization.
Experiments demonstrate that Ctrl-R enables effective exploration and internalization of previously unattainable reasoning patterns, yielding consistent improvements across language and vision–language models on mathematical reasoning tasks.

Forum: https://openreview.net/forum?id=x6J7va4BYz

---

## 359. DPO Unchained: Your Training Algorithm is Secretly Disentangled in Human Choice Theory (and Its Loss' Convexity is Dispensable)

**Abstract:** Normative theories allow one to elicit key parts of a ML algorithm from first principles, which is crucial at a time of championed scrutiny for ML work. Direct Preference Optimization (DPO) cleverly bypasses reward modeling by making an explicit link with a specific normative model of human choice. Our paper elevates this connection to the full generality of DPO's normative framework. Getting there requires reworking social choice theory's textbook path for a better RLHF/ML fit. It elevates the connection to a remarkably broad viewpoint on preference optimization, considering the current panorama of DPO follow-ups. It also unveils unexpected riches for ML, chief among which the support for *non-convex* losses, the fact that *any* compliant ML analytical choice can be embedded with *any* human choice model, and a normative framework's umbrella wide enough to safeguard DPO's *extensions* (margins, length correction, ...). A *toy* experiment ``far away'' from the DPO crowd is given.

Forum: https://openreview.net/forum?id=j4c3i3a5kH

---

## 360. MACKO: Sparse matrix-vector multiplication for low sparsity

**Abstract:** Sparse Matrix-Vector Multiplication (SpMV) is a fundamental operation in the inference of sparse Large Language Models (LLMs).
Because existing SpMV methods perform poorly under the low, unstructured sparsity ($30-90\\%$) commonly observed in pruned LLMs, unstructured pruning provides only limited memory reduction and speedup.
We propose **MACKO-SpMV**, a GPU-optimized format and kernel co-designed to reduce storage overhead while remaining compatible with the GPU’s execution model.
This enables efficient SpMV for unstructured sparsity without specialized hardware units or precomputation.
We identify memory bandwidth as the primary limiting factor of SpMV and analyze the storage overhead of MACKO.
At $50\\%$ sparsity, MACKO is the first approach to achieve $1.5\times$ memory reduction and $1.2-1.5\times$ speedup over the dense baseline as well as substantial improvements over other SpMV methods: cuSPARSE ($2.8-13.0\times$), Sputnik ($1.9-2.6\times$), and DASP ($2.2-2.5\times$).
An LLM pruned with Wanda to sparsity $50\\%$ requires $1.5\times$ less memory and achieves $1.5\times$ faster inference at fp16 precision.
As a result, **unstructured pruning at $50\\%$ sparsity becomes practical** for real-world LLM workloads and **bridges the efficiency gap with structured 2:4 sparsity**.

Forum: https://openreview.net/forum?id=ah9xkFXCV6

---

## 361. Scaling Law for Quantization-Aware Training

**Abstract:** Large language models (LLMs) demand substantial computational and memory resources, creating deployment challenges. Quantization-aware training (QAT) addresses these challenges by reducing model precision while maintaining performance. However, the scaling behavior of QAT, especially at 4-bit precision (W4A4), is not well understood. Existing QAT scaling laws often ignore key factors such as the number of training tokens and quantization granularity, which limits their applicability. This paper proposes a unified scaling law for QAT that models quantization error as a function of model size, training data volume, and quantization group size. Through 268 QAT experiments, we show that quantization error decreases as model size increases, but rises with more training tokens and coarser quantization granularity. To identify the sources of W4A4 quantization error, we decompose it into weight and activation components. Both components follow the overall trend of W4A4 quantization error, but with different sensitivities. Specifically, weight quantization error increases more rapidly with more training tokens. Further analysis shows that the activation quantization error in the FC2 layer, caused by outliers, is the primary bottleneck of W4A4 QAT quantization error. By applying mixed-precision quantization to address this bottleneck, we demonstrate that weight and activation quantization errors can converge to similar levels. Additionally, with more training data, weight quantization error eventually exceeds activation quantization error, suggesting that reducing weight quantization error is also important in such scenarios. These findings offer key insights for improving QAT research and development.

Forum: https://openreview.net/forum?id=fXr3uPr1G5

---

## 362. Online Conformal Prediction via Universal Portfolio Algorithms

**Abstract:** Online conformal prediction (OCP) seeks prediction intervals that achieve long-run $1-\alpha$ coverage for arbitrary (possibly adversarial) data streams, while remaining as informative as possible. Existing OCP methods often require manual learning-rate tuning to work well, and may also require algorithm-specific analyses. Here, we develop a general regret-to-coverage theory for interval-valued OCP based on the $(1-\alpha)$-pinball loss. Our first contribution is to identify *linearized regret* as a key notion, showing that controlling it implies coverage bounds for any online algorithm. This relies on a black-box reduction that depends only on the Fenchel conjugate of an upper bound on the linearized regret. Building on this theory, we propose UP-OCP, a parameter-free method for OCP, via a reduction to a two-asset portfolio selection problem, leveraging universal portfolio algorithms. We show strong finite-time bounds on the miscoverage of UP-OCP, even for polynomially growing predictions. Extensive experiments support that UP-OCP delivers consistently better size/coverage trade-offs than prior online conformal baselines.

Forum: https://openreview.net/forum?id=EKmiOtjZjI

---

## 363. TD3B: Transition-Directed Discrete Diffusion for Allosteric Binder Generation

**Abstract:** Protein function is often controlled by ligands that bias the direction of state transitions, such as agonists and antagonists, rather than stabilizing a single conformation. This is especially important for clinically relevant G protein-coupled receptors (GPCRs), where therapeutic efficacy depends on functional directionality. Structure-based design methods optimize binding to static conformations and cannot represent non-reversible, directional effects or systematically distinguish agonist from antagonist behavior. To address this gap, we introduce **T**ransition-**D**irected **D**iscrete **D**iffusion for allosteric**B**inder design (**TD3B**), a sequence-based generative framework that designs binders with specified agonist or antagonist behavior via a directional transition control objective. TD3B combines a target-aware Direction Oracle, a soft binding-affinity gate, and amortized fine-tuning of a pre-trained discrete diffusion model, enabling targeted agonist and antagonist generation decoupled from binding affinity and unattainable by equilibrium-based or inference-only guidance baselines. The code and checkpoints are available at https://huggingface.co/ChatterjeeLab/TD3B.

Forum: https://openreview.net/forum?id=RNuC8Nj6rD

---

## 364. A Noise Sensitivity Exponent Controls Large Statistical-to-Computational Gaps in Single- and Multi-Index Models

**Abstract:** Understanding when learning is statistically possible yet computationally hard is a central challenge in high-dimensional statistics.
In this work, we investigate this question in the context of single- and multi-index models, classes of functions widely studied as benchmarks to probe the ability of machine learning methods to discover features in high-dimensional data. 
Our main contribution is to show that a Noise Sensitivity Exponent (NSE)—a simple quantity determined by the activation function—governs the existence and magnitude of statistical-to-computational gaps within a broad regime of these models.
We first establish that, in single-index models with large additive noise, the onset of a computational bottleneck is fully characterized by the NSE. We then demonstrate that the same exponent controls a statistical-computational gap in the specialization transition of large separable multi-index models, where individual components become learnable. Taken together, our results identify the NSE as a unifying property linking noise robustness, computational hardness, and feature specialization in high-dimensional learning.

Forum: https://openreview.net/forum?id=KHhO311xZ2

---

## 365. Reward and Guidance through Rubrics: Promoting Exploration to Improve Multi-Domain Reasoning

**Abstract:** Recent advances in reinforcement learning (RL) have significantly improved the complex reasoning capabilities of large language models (LLMs).
Despite these successes, existing methods mainly focus on single-domain RL (e.g., mathematics) with verifiable rewards (RLVR), and their reliance on purely online RL frameworks restricts the exploration space, thereby limiting reasoning performance.
In this paper, we address these limitations by leveraging rubrics to provide both fine-grained reward signals and offline guidance.
We propose $\textbf{RGR-GRPO}$ (Reward and Guidance through Rubrics), a rubric-driven RL framework for multi-domain reasoning. 
RGR-GRPO enables LLMs to receive dense and informative rewards while exploring a larger solution space during GRPO training.
Extensive experiments across 14 benchmarks spanning multiple domains demonstrate that RGR-GRPO consistently outperforms RL methods that rely solely on alternative reward schemes or offline guidance.
Compared with verifiable online RL baseline, RGR-GRPO achieves average improvements of +7.0%, +5.4%, +8.4%, and +6.6% on mathematics, physics, chemistry, and general reasoning tasks, respectively.
Notably, RGR-GRPO maintains stable entropy fluctuations during off-policy training and achieves superior pass@k performance, reflecting sustained exploration and effective breakthrough beyond existing performance bottlenecks.

Forum: https://openreview.net/forum?id=AfqsNFzJcs

---

## 366. Jailbreak to Protect: Buffering and Reinforcing via Temporary Jailbreaking for Safe Fine-Tuning in Large Language Models

**Abstract:** Fine-tuning-as-a-Service (FaaS) enables personalization of large language models (LLMs), but it can weaken safety-alignment under harmful fine-tuning attacks. Recent work has shown that activating harmful-behavior modules during fine-tuning can prevent models from learning undesired behaviors, but its mechanism remains unclear. In this paper, we revisit temporary jailbreaking as a defense against harmful fine-tuning and provide a gradient-level analysis showing that it saturates safety-degrading gradients while preserving benign task-relevant gradients. Based on this insight, we propose a **Buffer-and-Reinforce fine-tuning framework** that buffers harmful updates during user fine-tuning and reinforces safety after adaptation. Specifically, BufferLoRA induces temporary jailbreaking as a removable adapter to reduce harmful updates during user fine-tuning. After adaptation, ReinforceLoRA, trained to recover refusal behavior under the temporarily jailbroken state, is integrated with UserLoRA via QR decomposition-based merging to reinforce safety while preserving user-task performance. Extensive experiments show that our framework achieves superior safety and utility with no additional safety data during user fine-tuning and minimal computational cost.

Forum: https://openreview.net/forum?id=O0JXgBBo3k

---

## 367. ThreadWeaver: Adaptive Threading for Efficient Parallel Reasoning in Language Models

**Abstract:** Scaling inference-time computation has enabled Large Language Models (LLMs) to achieve strong reasoning performance, but their inherently sequential decoding incurs substantial latency, motivating parallelization of the generation process. However, existing parallel reasoning approaches suffer from performance degradation compared to their sequential counterparts, and often rely on specialized inference engines. We introduce ThreadWeaver, a framework for adaptive parallel reasoning that matches the accuracy of comparably sized sequential reasoning models while significantly reducing inference latency via three key innovations: 1) a two-stage parallel trajectory generator that produces high-quality parallel chain-of-thought data for supervised fine-tuning; 2) a trie-based rollout design that enables parallel reasoning on any off-the-shelf autoregressive inference engine; and 3) a parallelization-aware reinforcement learning framework that trains the model to balance reasoning accuracy with effective parallelization. Across six challenging math reasoning benchmarks, ThreadWeaver trained on top of Qwen3-8B achieves performance on par with cutting-edge sequential reasoning models (79.9% on AIME24 and 71.9% on average) while delivering up to 1.53x speedup in token latency, establishing a new Pareto frontier between accuracy and efficiency.

Forum: https://openreview.net/forum?id=Efq2VvYk1o

---

## 368. How Many Different Outputs Can a Transformer Generate?

**Abstract:** We study how we can leverage only a handful of characteristics of a transformer's architecture to closely predict the number of different sequences it can output, both qualitatively and quantitatively.
  We provide an upper bound depending on the length of the prompt, which we show empirically to be tight up to a factor less than 10, across architectures and model sizes.
  Our analysis also provides a theoretical explanation for previously observed empirical failures of transformers on simple sequence tasks—such as copying and cramming.
  Formally, we prove that (i) the maximal length of accessible sequences (those that the transformer can output for some prompt) grows linearly with the prompt length, (ii) beyond a critical threshold, the proportion of accessible sequences decays exponentially with sequence length, and (iii) the linear coefficient relating prompt length to accessible sequence length admits a theoretical upper bound.
  Notably, these results hold even with unbounded context and computation time.

Forum: https://openreview.net/forum?id=x4xOtzoYa6

---

## 369. Mitigating Hallucinations in Large Vision-Language Models via Causal Route Gating

**Abstract:** Large vision-language models (LVLMs) often hallucinate content that is fluent yet unsupported by the image, limiting their reliability in real-world deployment. We show that a key failure mode arises from route competition: even when visual tokens receive attention, the final token decision can be dominated by the textual pathway, causing the decoder to follow linguistic priors over visual evidence. To mitigate this, we propose a training-free, decision-aligned intervention that decomposes each attention head into a visual route and a text route, and estimates their token-level effects using an efficient one-forward/one-gradient approximation. These estimates reveal route conflict within heads and identify prior-dominant ones, enabling selective suppression of only the text route while keeping the visual route intact. Across five benchmarks spanning discriminative and generative settings, our method consistently reduces hallucination-related errors across models with limited impact on overall multimodal performance, while incurring a modest inference-time overhead.

Forum: https://openreview.net/forum?id=LIcj73RLX6

---

## 370. TetraJet-v2: Accurate NVFP4 Training for Large Language Models with Oscillation Suppression and Outlier Control

**Abstract:** Large Language Models (LLMs) training is prohibitively expensive, driving interest in low-precision fully-quantized training (FQT).
While novel 4-bit formats like NVFP4 offer substantial efficiency gains, achieving near-lossless training at such low precision remains challenging.
We introduce **TetraJet-v2**, an end-to-end 4-bit FQT method that leverages NVFP4 for activations, weights and gradients in all linear layers.
We identify two critical issues hindering low-precision LLM training: weight oscillation and outliers. 
To address these, we propose: 1) an unbiased double-block quantization method for NVFP4 linear layers, 2) **OsciReset**, an algorithm to suppress weight oscillation, and 3) **OutControl**, an algorithm to retain outlier accuracy. **TetraJet-v2** outperforms prior methods on FP4 pre-training for LLMs across models up to 370M parameters trained up to 212B tokens, reducing the performance gap to BF16 by an average of $51.3$% while enabling an $1.67\times$ end-to-end speedup over FP8.

Forum: https://openreview.net/forum?id=7ZQhm5HnOA

---

## 371. Towards Pareto-Optimal Tool-Integrated Agents with Pareto Ranking Policy Optimization

**Abstract:** Recent advances in tool-integrated language agents have significantly improved their ability to solve complex reasoning tasks. However, existing alignment methods predominantly focus on maximizing task accuracy, while overlooking auxiliary objectives such as tool-use efficiency, which are essential for practical deployment. To address this gap, we introduce \textbf{ParetoPO}, a two-stage multi-objective optimization framework for aligning tool-using large language models (LLMs) under competing objectives. In the first stage, ParetoPO leverages hypervolume-guided dynamic scalarization to adapt reward weights based on global Pareto frontier progress. In the second stage, it replaces scalarized learning signals with Pareto-ranking-based advantage computation, promoting nondominated trajectories through dominance-aware credit assignment. This design enables fine-grained, action-level optimization across multiple conflicting objectives. Experimental results on mathematic reasoning and multi-hop QA tasks show that ParetoPO consistently discovers policies with superior accuracy-efficiency trade-offs compared to static and heuristic baselines. Our code is publicly available at https://github.com/Applied-Machine-Learning-Lab/ICML2026_ParetoPO.

Forum: https://openreview.net/forum?id=9sg34nj13o

---

## 372. Towards Hierarchy–Uniformity Equilibrium: Recovering Semantic Depth in Hypergraph Contrastive Learning

**Abstract:** Hypergraph contrastive learning is an effective paradigm for representation learning on higher-order relational data, yet existing methods largely ignore that hyperedges link nodes with multi-level semantics. Standard contrastive objectives emphasize instance discrimination via hyperspherical uniformity and tend to push embeddings apart in an indiscriminate manner. We show that this leads to a *Hierarchy–Uniformity Conflict*, whose geometric manifestation is *Semantic Flattening*, where the semantic depth of hyperedges collapses into a nearly flat cloud of instances. To address this issue, we introduce **HyperDepth**, a hypergraph contrastive learning framework that moves representations towards a hierarchy–uniformity equilibrium by jointly coordinating spectral and geometric signals. HyperDepth employs a decoupled spectral encoding scheme with adaptive gating so that high-frequency components focus on local instance discrimination while low-frequency components capture global hierarchical structure. On top of this, an energy-based hierarchical alignment module attaches a learnable prototype tree to the representation space and minimizes an interpretable energy functional to recover the semantic depth of hyperedges. Theoretically, under a mild frequency-separation assumption, we show that the local contrastive and global hierarchical objectives operate on orthogonal spectral components and admit equilibrium embeddings that preserve semantic depth while still retaining instance-level discrimination. Experiments on 15 hypergraph datasets and 17 supervised and self-supervised baselines, spanning homophilic and heterophilic regimes, show that HyperDepth attains strong performance with the best average rank.

Forum: https://openreview.net/forum?id=88qwqo7VSK

---

## 373. A Distributional View for Visual Mechanistic Interpretability: KL-Minimal Soft-Constraint Principle

**Abstract:** Most current paradigms in visual mechanistic interpretability (MI) remain confined to interpreting internal units of the vision model via heuristic methods (e.g., top-$K$ activation retrieval or optimization with regularization). In this work, we establish a theoretical distributional view for visual MI, which models the influence of a feature activation on the natural image distribution, thereby formulating a Kullback-Leibler (KL)-minimal optimization problem to model the MI task. Under this framework, statistical biases are identified within previous MI paradigms, which reveal that they may either be perceptually uninterpretable to humans (i.e., deviate from the natural image distribution), or mechanistically unfaithful to the vision models (i.e., unable to activate model features). To resolve the biases under the distributional view, we propose a model with a KL-minimal soft-constraint principle for visual MI that theoretically balances interpretability and faithfulness. We realize this principle via energy-guided diffusion posterior sampling.
Extensive experiments validate the theoretical soundness of the proposed distributional view and demonstrate the practical effectiveness of our paradigm on the DINOv3 vision model.
The code is available at https://github.com/SII-ZhouGC/EnergyDPS.

Forum: https://openreview.net/forum?id=TZvHUmClPP

---

## 374. Don't Force the Fit: Bounded Log-Likelihood Loss for Enhanced Reasoning in Large Language Models

**Abstract:** Supervised fine-tuning (SFT) is central to aligning large language models (LLMs) with instruction following and task-specific reasoning. Despite its success, SFT optimizes token-level likelihoods under the implicit assumption that strictly fitting all tokens in expert demonstrations induces the desired downstream behavior. However, in reasoning tasks where correctness is defined by logical validity or final outcomes rather than exact token realizations, this assumption can lead to optimization misalignment. We empirically observe that low-probability tokens in reasoning demonstrations often correspond to realization-specific or stylistic variations, and that reducing their influence during training consistently improves generalization on reasoning benchmarks. Motivated by this insight, we propose the *Bounded Log-Likelihood Loss* (BLL-Loss), a simple and parameter-free alternative to standard likelihood training that bounds gradient contributions from low-probability tokens while preserving conventional optimization behavior. We provide theoretical insights and extensive empirical results demonstrating that BLL-Loss improves reasoning generalization across diverse model scales and challenging benchmarks.

Forum: https://openreview.net/forum?id=mTMA9yvsnx

---

## 375. Learning Credal Ensembles via Distributionally Robust Optimization

**Abstract:** Credal predictors are epistemic-uncertainty-aware models that produce a convex set of probabilistic predictions. They provide a principled framework for quantifying predictive epistemic uncertainty (EU) and have been shown to improve model robustness across a range of settings. However, most state-of-the-art (SOTA) methods primarily define EU as disagreement induced by random training initializations, which mainly reflects sensitivity to optimization randomness rather than uncertainty from more substantive sources. In response, we formulate EU as disagreement between models trained under different degrees of relaxation of the i.i.d. assumption between the training and test distributions. Building on this idea, we propose *CreDRO*, which learns an ensemble of plausible models via distributionally robust optimization. As a result, CreDRO captures EU arising not only from training randomness but also from informative disagreement due to potential train–test distribution shifts. Empirically, CreDRO consistently outperforms SOTA credal approaches on downstream tasks, including out-of-distribution detection on extensive benchmarks and selective classification in medical settings.

Forum: https://openreview.net/forum?id=cRTbp2pv7X

---

## 376. Equivalence of Context and Parameter Updates in Modern Transformer Blocks

**Abstract:** Recent research has established that the impact of context in a vanilla transformer can be represented implicitly by forming a token-dependent, rank-1 patch to its MLP weights. This work extends that foundational theory to the diverse architectures of modern Large Language Models. We first demonstrate a precise, analytical solution for a Gemma-style transformer block, proving that the entire effect of a context can be perfectly mapped to rank-1 patches on its MLP weight matrices and a patch to the RMSNorm scale. We then generalize this result, providing a constructive proof and algorithm for multi-layer models. To unify these findings, we introduce a general framework centered on two core properties: input controllability and output controllability. We prove that a perfect implicit weight patch is possible for any MLP block where the inner function is input-controllable and the outer function is output-controllable. This provides a simpler and more powerful lens for understanding how transformer models transmute prompts into effective weights. This setup generalizes to a wide range of modern LLM architectures including gating, pre-/post-norm, mixture of experts and sequential/parallel transformer blocks.

Forum: https://openreview.net/forum?id=aZ5Di8Ii4X

---

## 377. Flex-Forcing: Towards a Unified Autoregressive and Bidirectional Video Diffusion Model

**Abstract:** Recent progress in large-scale generative models has substantially advanced video generation, yet existing methods remain constrained by a rigid inference paradigm. Bidirectional diffusion models excel at global coherence and visual fidelity but suffer from slow inference, while autoregressive models offer efficient and streaming generation at the cost of long-range consistency and exposure bias. We introduce Flex-Forcing, a unified training and inference framework that enables a video diffusion model to seamlessly operate under both bidirectional and autoregressive generation regimes. The core idea is a flexible chunking mechanism jointly defined over the temporal axis and denoising steps. This design allows the model to (1) perform flexible chunking according to different device budgets, (2) perform bidirectional inference across chunks for global structure planning, while generating frames autoregressively within each chunk for efficient and fine-grained synthesis, and (3) perform any-order, any-timestep autoregressive generation without the strict causal constraint. Extensive experiments on multiple video generation benchmarks demonstrate that Flex-Forcing achieves consistently better video quality, long-video stability than strong baselines with a rigid inference schedule, while offering faster inference.

Forum: https://openreview.net/forum?id=COGc3MSR7n

---

## 378. Large-Scale Terminal Agentic Trajectory Generation from Dockerized Environments

**Abstract:** Training agentic models for terminal-based tasks critically depends on high-quality terminal trajectories that capture realistic long-horizon interactions across diverse domains.
However, constructing such data at scale remains challenging due to two key requirements:
\textbf{\emph{Executability}}, since each instance requires a suitable and often distinct Docker environment;
and \textbf{\emph{Verifiability}}, because heterogeneous task outputs preclude unified, standardized verification.
To address these challenges, we propose \textbf{TerminalTraj}, a scalable pipeline that (i) filters high-quality repositories to construct Dockerized execution environments, (ii) generates Docker-aligned task instances, and (iii) synthesizes agent trajectories with executable validation code.
Using TerminalTraj, we curate 32K Docker images and generate 50,733 verified terminal trajectories across eight domains.
Models trained on this data with the Qwen2.5-Coder backbone achieve consistent performance improvements on TerminalBench (TB), with gains of up to 20\% on TB 1.0 and 10\% on TB 2.0 over their respective backbones.
Notably, \textbf{TerminalTraj-32B} achieves strong performance among models with fewer than 100B parameters, reaching 35.30\% on TB 1.0 and 22.00\% on TB 2.0, and demonstrates improved test-time scaling behavior.

Forum: https://openreview.net/forum?id=PeFSCRulgy

---

## 379. Keeping a Secret Requires a Good Memory: Unconditional Streaming Lower-Bounds for Differentially Private Algorithms

**Abstract:** We study the computational cost of differential privacy in terms of memory efficiency. Specifically, we establish for the first time an unconditional space lower bound for user-level differential privacy by introducing a novel proof technique based on a multi-player communication game. We apply our framework, as an example, to the fundamental problem of estimating the number of distinct elements in a stream: we prove that any private algorithm requires almost $\widetilde{\Omega}(T^{1/3})$ space (where $T$ denotes the length of the stream) to achieve certain error rates in a promise variant of the problem, resolving an open problem in the literature (by Jain et al. 2023 and Cummings et al. 2025) and establishes the first exponential separation between the space complexity of private algorithms and their non-private $\widetilde{O}(1)$ counterparts for a natural statistical estimation task. Furthermore, we show that this communication-theoretic technique generalizes to broad classes of problems, yielding lower bounds for private medians, quantiles, and max-select.

Forum: https://openreview.net/forum?id=JiB2eXY8fT

---

## 380. Minimax Optimal Strategy for Delayed Observations in Online Reinforcement Learning

**Abstract:** We study reinforcement learning with delayed state observation, where the agent observes the current state after some random number of time steps.
We propose an algorithm that combines the augmentation method and the upper confidence bound approach.
For tabular Markov decision processes (MDPs), we derive a regret bound of $\tilde{\mathcal{O}}(H \sqrt{D_{\max} SAK})$, where $S$ and $A$ are the cardinalities of the state and action spaces, $H$ is the time horizon, $K$ is the number of episodes, and $D_{\max}$ is the maximum length of the delay.
We also provide a matching lower bound up to logarithmic factors, showing the optimality of our approach.
Our analytical framework formulates this problem as a special case of a broader class of MDPs, where their transition dynamics decompose into a known component and an unknown but structured component.
We establish general results for this abstract setting, which may be of independent interest.

Forum: https://openreview.net/forum?id=fFupHW7Jqx

---

## 381. Rational Transductors

**Abstract:** Standard Transformers excel at semantic modeling but struggle with rigid
  sequential logic and state tracking.  Theoretical work establishes that
  self-attention is limited to $\mathsf{AC}^0$ (under hard attention) or $\mathsf{TC}^0$ (under
  soft attention), complexity classes that often fail to support robust length
  generalization on sequential problems without intermediate chain-of-thought
  (see hahn2020theoretical and merrill2022saturated).  In this work, we introduce
  \emph{Rational Transductors}, a dual-stream architecture that augments the
  Transformer with a matrix-valued recurrence derived from Weighted Finite
  Automata (WFA).  By injecting rational state information into the attention
  mechanism via a *Deep Rational Injection* scheme, our framework strictly
  generalizes Transformers to capture all Regular Languages, $\mathsf{NC}^1$-complete
  problems (such as Boolean Formula Evaluation), and fundamental separations
  like Parity and Modular Counting, while preserving $O(\log T)$ parallel
  training efficiency.  Theoretical analysis and empirical results demonstrate
  that Rational Transductors solve the "Regular Gap," enabling robust length
  generalization on algorithmic tasks where standard Transformers fail, without
  the sequential computational bottlenecks of traditional RNNs.

Forum: https://openreview.net/forum?id=uEZpyELNuB

---

## 382. On the Convergence Rate of LoRA Gradient Descent

**Abstract:** The low-rank adaptation (LoRA) algorithm for fine-tuning large models has grown popular in recent years due to its remarkable performance and low computational requirements. LoRA trains two "adapter" matrices that form a low-rank representation of the model parameters, thereby massively reducing the number of parameters that need to be updated at every step. Although LoRA is simple, its convergence is poorly understood due to the lack of Lipschitz smoothness, a key condition for classic convergence analyses. As a result, current theoretical results only consider asymptotic behavior or assume strong boundedness conditions which artificially enforce Lipschitz smoothness. In this work, we provide for the first time a non-asymptotic convergence analysis of the *original LoRA gradient descent* algorithm, which reflects widespread practice, without such assumptions. Our work relies on three key steps: i) reformulating the problem in terms of the outer product of the stacked adapter matrices, ii) a modified descent lemma for the "Lipschitz-like" reparametrized function, and iii) controlling the step size. With this approach, we prove that LoRA gradient descent converges to a stationary point at rate $O(\frac{1}{\log T})$, where $T$ is the number of iterations.

Forum: https://openreview.net/forum?id=9GRlBVAXq8

---

## 383. Beyond Language Modeling: An Exploration of Multimodal Pretraining

**Abstract:** The visual world offers a critical axis for advancing foundation models beyond language. Despite growing interest in this direction, the design space for native multimodal models remains opaque. We provide empirical clarity through controlled, from-scratch pretraining experiments, isolating the factors that govern multimodal pretraining without interference from language pretraining. We adopt the Transfusion framework, using next-token prediction for language and diffusion for vision, to train on diverse data including text, video, image-text pairs, and even action-conditioned video. Our experiments yield four key insights: (i) Representation Autoencoder (RAE) provides an optimal unified visual representation by excelling at both visual understanding and generation; (ii) visual and language data are complementary and yield synergy for downstream capabilities; (iii) unified multimodal pretraining leads naturally to world modeling, with capabilities emerging from general training; and (iv) Mixture-of-Experts (MoE) enables efficient and effective multimodal scaling while naturally inducing modality specialization. Through IsoFLOP analysis, we compute scaling laws for both modalities and uncover a scaling asymmetry: vision is significantly more data-hungry than language. We demonstrate that the MoE architecture harmonizes this scaling asymmetry by providing the high model capacity required by language while accommodating the data-intensive nature of vision, paving the way for truly unified multimodal models.

Forum: https://openreview.net/forum?id=xAKrRYm6pc

---

## 384. Certifying Capabilities from Finite Tests: When Is It Possible?

**Abstract:** Modern foundation models are evaluated through broad capabilities such as arithmetic, reasoning, safety, and robustness, yet it remains unclear in a principled sense when *finite tests* can meaningfully certify such claims. We develop a rigorous theory of capability evaluation by formalizing evaluation as inference over a task family and asking when guarantees over the full family can be inferred from a strict subset of tests. We analyze two canonical regimes. In stochastic multi-environment evaluation, we characterize when uniform certification is possible across multiple environments and show that the sample complexity is governed by a $\chi^2$-radius of the environment family, yielding near-optimal evaluation protocols with matching lower bounds under a natural overlap condition. In contrast, for worst-case, rule-like capabilities, we establish fundamental impossibility results. Even for structured model classes such as Boolean circuits of bounded size, black-box evaluation cannot, in general, certify global properties. Together, these results provide a principled framework for understanding when finite evaluation can and cannot certify capabilities.

Forum: https://openreview.net/forum?id=cNCWYrSV5R

---

## 385. High-Accuracy Sampling for Diffusion Models and Log-Concave Distributions

**Abstract:** We present algorithms for diffusion model sampling which obtain $\delta$-error in $\mathrm{polylog}(1/\delta)$ steps, given access to $\widetilde O(\delta)$-accurate score estimates in $L^2$. This is an exponential improvement over all previous results. Specifically, under minimal data assumptions, the complexity is $\widetilde O(d\mathrm{polylog}(1/\delta))$ where $d$ is the dimension of the data; under a non-uniform $L$-Lipschitz condition, the complexity is $\widetilde O(\sqrt{dL}\mathrm{polylog}(1/\delta))$; and if the data distribution has intrinsic dimension $d_\star$, then the complexity reduces to $\widetilde O(d_\star\mathrm{polylog}(1/\delta))$. Our approach also yields the first $\mathrm{polylog}(1/\delta)$ complexity sampler for general log-concave distributions using only gradient evaluations.

Forum: https://openreview.net/forum?id=GW3umRqsZZ

---

## 386. A Fully First-Order Layer for Differentiable Optimization

**Abstract:** Differentiable optimization studies how to embed a mathematical program as a differentiable layer in machine learning pipelines. However, existing approaches typically rely on implicit differentiation, involving expensive Hessian computation while differentiating through optimality conditions. To address this challenge, we formulate the differentiable optimization problem as a bilevel optimization instance.
We construct a new active-set Lagrangian as a proxy to compute an $\epsilon$-approximate hypergradient using only near-constant $O(\log (1/\epsilon))$ first-order information. We also show that applying this efficient hypergradient oracle to constrained bilevel optimization improves the overall gradient complexity to $\tilde{O}(\delta^{-1}\epsilon^{-3})$ to reach a $(\delta, \epsilon)$-Goldstein stationary point. We implement our method `FFOLayer`, as a drop-in Python library compatible with existing differentiable optimization solvers. Our algorithm shows significantly faster computation with similar convergence compared to other existing solvers. The source code is available at [https://github.com/GT-KOALA/FFOLayer](https://github.com/GT-KOALA/FFOLayer).

Forum: https://openreview.net/forum?id=jJur8Fq7IK

---

## 387. Agent0-VL: Exploring Self-Evolving Agent for Tool-Integrated Vision-Language Reasoning

**Abstract:** Large Vision-Language Models (LVLMs) have achieved remarkable progress in multimodal reasoning tasks; however, their learning remains constrained by the limitations of human-annotated supervision. Recent self-rewarding approaches attempt to overcome this constraint by allowing models to act as their own critics or reward providers. Yet, purely text-based self-evaluation struggles to verify complex visual reasoning steps and often suffers from evaluation hallucinations. To address these challenges, inspired by recent advances in tool-integrated reasoning, we propose Agent0-VL, a self-evolving vision-language agent that achieves continual improvement with tool-integrated reasoning. Agent0-VL incorporates tool usage not only into reasoning but also into self-evaluation and self-repair, enabling the model to introspect, verify, and refine its reasoning through evidence-grounded analysis. It unifies two synergistic roles within a single LVLM: a Solver that performs multi-turn tool-integrated reasoning, and a Verifier that generates structured feedback and fine-grained self-rewards through tool-grounded critique. These roles interact through a Self-Evolving Reasoning Cycle, where tool-based verification and reinforcement learning jointly align the reasoning and evaluation distributions for stable self-improvement. Through this zero-external-reward evolution, Agent0-VL aligns its reasoning and verification behaviors without any human annotation or external reward models, achieving continual self-improvement. Experiments on chart reasoning, geometric problem solving, and visual scientific analysis show that Agent0-VL achieves an 12.5% improvement over the Qwen-VL base model.

Forum: https://openreview.net/forum?id=aA02f0y0LY

---

## 388. DreamDojo: A Generalist Robot World Model from Large-Scale Human Videos

**Abstract:** Being able to simulate the outcomes of actions in varied environments will revolutionize the development of generalist agents at scale. However, modeling these world dynamics, especially for dexterous robotics tasks, poses significant challenges due to limited data coverage and scarce action labels. As an endeavor towards this end, we introduce DreamDojo, a foundation world model that learns diverse interactions and dexterous controls from 44k hours of egocentric human videos. Our data mixture represents the largest video dataset to date for world model pretraining, spanning a wide range of daily scenarios with diverse objects and skills. To address the scarcity of action labels, we introduce continuous latent actions as unified proxy actions, enhancing interaction knowledge transfer from unlabeled videos. After post-training on small-scale target robot data, DreamDojo demonstrates a strong understanding of physics and precise action controllability. We also devise a distillation pipeline that accelerates DreamDojo to a real-time speed of 10.93 FPS and further improves consistency to the context. Our work enables several important applications based on generative world models, including live teleoperation, policy evaluation, and model-based planning. Systematic evaluation on multiple challenging out-of-distribution (OOD) benchmarks verifies the significance of our method for simulating open-world, contact-rich tasks, paving the way for general-purpose robot world models.

Forum: https://openreview.net/forum?id=FuvU7PTyED

---

## 389. PhenoBrain: Phenotype-Conditioned Long-Range Communication for Multi-Modal Brain Network Analysis

**Abstract:** Multi-modal brain network analysis aims to predict neuropsychiatric status from functional connectomes with heterogeneous phenotypes. However, most existing methods treat phenotypes as auxiliary features and perform late fusion, implicitly assuming that the connectome representation should be learned in the same way regardless of phenotype. However, in clinical neuroscience the same functional connectivity pattern may support different conclusions under different phenotype contexts. To bridge this gap, we propose PhenoBrain, a novel framework for multi-modal brain network analysis that injects phenotype information at the mechanism level rather than only at the classifier level. Specifically, we propose a phenotype-conditioned long-range routing mechanism, which learns a subject-specific multi-hop communication kernel to model long-range connectome interactions. Furthermore, we propose a phenotypic-guided attention mechanism regulation method, which uses phenotypic information as a conditional prior to regulate the learning process of attention in brain networks. To verify the effectiveness of our method, we constructed two multi-modal brain network analysis datasets based on open-source image data. Extensive experiments demonstrate that PhenoBrain achieves state-of-the-art performance.

Forum: https://openreview.net/forum?id=9NqKL9QQ4a

---

## 390. From Pixels to Tokens: A Systematic Study of Latent Action Supervision for Vision-Language-Action Models

**Abstract:** Latent actions serve as an intermediate representation that enables consistent modeling of vision-language-action (VLA) models across heterogeneous datasets. However, approaches to supervising VLAs with latent actions are fragmented and lack a systematic comparison. This work structures the study of latent action supervision from two perspectives: (i) regularizing the trajectory via image-based latent actions, and (ii) unifying the target space with action-based latent actions. Under a unified VLA baseline, we instantiate and compare four representative integration strategies. Our results reveal a formulation-task correspondence: image-based latent actions benefit long-horizon reasoning, whereas action-based latent actions excel at complex motor coordination. Furthermore, we find that directly supervising the VLM with discrete latent action tokens yields the most effective performance. Finally, our experiments offer initial insights into the benefits of latent action supervision in mixed-data, suggesting a promising direction for VLA training.

Forum: https://openreview.net/forum?id=VMsumctGvg

---

## 391. Beyond First-order Asymptotics in Sequential Mean Testing

**Abstract:** We revisit the problem of sequentially testing the mean of bounded distributions in a level-$\alpha$ power-one framework. We study a $\mathrm{KL_{inf}}$-based sequential test that is known to attain the information-theoretic lower bound on the expected stopping time with exact constants as $\alpha \to 0$. Going beyond first-order asymptotics, we establish a central limit theorem (CLT) for the stopping time of this test. Our analysis proceeds in two steps. First, we prove a  novel CLT for the $\mathrm{KL_{inf}}$ statistic itself, characterizing its fluctuations around its deterministic limit. We then leverage this result to show that the stopping time, centered appropriately and scaled by $\sqrt{\log(1/\alpha)}$, converges in distribution to a Gaussian limit with an explicit variance. This yields a second-order characterization of an asymptotically optimal sequential test for bounded distributions. Finally, we present numerical experiments that corroborate our theoretical findings.

Forum: https://openreview.net/forum?id=HMyCBL2yMV

---

## 392. Bulk-Calibrated Credal Ambiguity Sets: Fast, Tractable Decision Making under Out-of-Sample Contamination

**Abstract:** Distributionally robust optimisation (DRO) minimises the worst-case expected loss over an ambiguity set that can capture distributional shifts in out-of-sample environments. While Huber (linear-vacuous) contamination is a classical minimal-assumption model for an $\varepsilon$-fraction of arbitrary perturbations, including it in an ambiguity set can make the worst-case risk infinite and the DRO objective vacuous unless one imposes strong boundedness or support assumptions. We address these challenges by introducing bulk-calibrated credal ambiguity sets: we learn a high-mass bulk set from data while considering contamination inside the bulk and bounding the remaining tail contribution separately. This leads to a closed-form, finite $\mathrm{mean}+\sup$ robust objective and tractable linear or second-order cone programs for common losses and bulk geometries. Through this framework, we highlight and exploit the equivalence between the imprecise probability (IP) notion of upper expectation and the worst-case risk, demonstrating how IP credal sets translate into DRO objectives with interpretable tolerance levels. Experiments on heavy-tailed inventory control, geographically shifted house-price regression, and demographically shifted text classification show competitive robustness-accuracy trade-offs and efficient optimisation times, using Bayesian, frequentist, or empirical reference distributions.

Forum: https://openreview.net/forum?id=QO82qIzEsP

---

## 393. End-to-End Compression for Tabular Foundation Models

**Abstract:** The long-standing dominance of gradient-boosted decision trees for tabular data has recently been challenged by in-context learning tabular foundation models. In-context learning methods fit and predict in one forward pass without parameter updates by leveraging the training data as context for predicting on query test points. 
While recent tabular foundation models achieve state-of-the-art performance, their transformer architecture based on the attention mechanism has quadratic complexity regarding dataset size, which in turn increases the overhead on training and inference time, and limits the capacity of the models to handle large-scale datasets. In this work, we propose TACO, an end-to-end tabular compression model that compresses the training dataset in a latent space. We test our method on the TabArena benchmark, where our proposed method is up to 94x faster in inference time, while consuming up to 97% less memory compared to the state-of-the-art tabular Transformer architecture, all while retaining performance without significant degradation. Lastly, our method not only scales better with increased dataset sizes, but it also achieves better performance compared to other baselines.

Forum: https://openreview.net/forum?id=84mfkGDxYh

---

## 394. Perceptrons and Localization of Attention’s Mean-Field Landscape

**Abstract:** The forward pass of a Transformer can be seen as an interacting particle system on the unit sphere: time plays the role of layers, particles that of token embeddings, and the unit sphere idealizes layer normalization. In some weight settings the system can even be seen as a gradient flow for an explicit energy, and one can  make sense of the infinite context length *mean-field* limit thanks to Wasserstein gradient flows. In this paper we study the effect of the perceptron block in this setting, and show that critical points are generically atomic and localized on subsets of the sphere.

Forum: https://openreview.net/forum?id=rO2yyZiy4v

---

## 395. Diffusion Flow Matching: Dimension-Improved KL Bounds and Wasserstein Guarantees

**Abstract:** Diffusion Flow Matching (DFM) has recently emerged as a versatile framework for generative modeling, yet its theoretical convergence properties remain only partially understood. In this work, we provide refined and novel convergence guarantees for Brownian motion based DFMs, focusing on the  discretization error. Our analysis is conducted under the Kullback–Leibler (KL) divergence and the 2-Wasserstein distance. Under finite-moment conditions and a mild score integrability assumption, we derive KL convergence bounds with improved dimensional dependence compared to prior work, achieving, up to our knowledge, state-of-the-art scaling under minimal conditions. We further extend the analysis to the 2-Wasserstein distance: under an additional first-order score integrability assumption and a weak log-concavity condition, we obtain convergence guarantees with dimensional dependence consistent with the KL case.

Forum: https://openreview.net/forum?id=zl3akehFBq

---

## 396. When the Prompt Becomes Visual: Vision-Centric Jailbreak Attacks for Large Image Editing Models

**Abstract:** Recent advances in large image editing models have shifted the paradigm from text-driven instructions to vision-prompt editing, where user intent is inferred directly from visual inputs such as marks, arrows, and visual–text prompts. While this paradigm greatly expands usability, it also introduces a critical and underexplored safety risk: the attack surface itself becomes visual. In this work, we propose Vision-Centric Jailbreak Attack (VJA), the first visual-to-visual jailbreak attack that conveys malicious instructions purely through visual inputs. To systematically study this emerging threat, we introduce IESBench, a safety-oriented benchmark for image editing models. Extensive experiments on IESBench demonstrate that VJA effectively compromises state-of-the-art commercial models, achieving attack success rates of up to 80.9% on Nano Banana Pro and 70.1% on GPT-Image-1.5. To mitigate this vulnerability, we propose a training-free defense based on introspective multimodal reasoning, which substantially improves the safety of poorly aligned models to a level comparable with commercial systems, without auxiliary guard models and with negligible computational overhead. Our findings expose new vulnerabilities, provide both a benchmark and practical defense to advance safe and trustworthy modern image editing systems.

Forum: https://openreview.net/forum?id=wQxRphkfxn

---

## 397. Toward Stable Value Alignment: Introducing Independent Modules for Consistent Value Guidance

**Abstract:** Aligning large language models (LLMs) with human values typically relies on post-training or inference-time steering that directly manipulates the backbone’s parameters or representation space. However, a critical gap exists: the model’s residual stream is highly dynamic, in which values exist as fragile, low-dimensional properties, inherently incompatible with  the stability required for consistent value expression. In this paper, we propose the Stable Value Guidance Transformer (SVGT), which addresses this gap through an independent value module incorporating two key designs: (1) *independent value modeling*, maintaining normative representations in a dedicated value space isolated from the backbone, and (2) *explicit behavioral guidance*, transducing these stable signals into learnable latent Bridge Tokens. These tokens serve as dynamic value anchors to explicitly steer the generative trajectory, ensuring robust adherence across diverse contexts without disrupting the backbone’s internal representations. Experiments across multiple backbones and safety benchmarks show that SVGT generally reduces harmful scores by over 70\% while maintaining generation fluency, demonstrating the efficacy of architecturally grounded value modeling.

Forum: https://openreview.net/forum?id=R80JLmXXPJ

---

## 398. Ratio-Variance Regularized Policy Optimization

**Abstract:** Standard on-policy reinforcement learning relies on heuristic clipping to enforce trust regions, but this mechanism imposes a severe cost by indiscriminately truncating high-return yet high-divergence updates. We demonstrate that explicitly constraining the *policy ratio **variance*** provides a principled local approximation to trust-region constraints, eliminating the need for binary hard clipping. By acting as a distributional ''soft brake'', this approach preserves critical gradient signals from novel discoveries while naturally down-weighting and enabling the reuse of stale, off-policy data. We introduce **R$^2$VPO** (Ratio-Variance Regularized Policy Optimization), which implements this constraint via a primal–dual optimization framework. Extensive evaluations across $7$ LLM scales, spanning both fast and slow reasoning paradigms, and $10$ robotic control tasks demonstrate the generality of the proposed approach. R$^2$VPO achieves substantial performance gains on mathematical reasoning benchmarks, with particularly pronounced improvements on smaller models, while significantly improving sample efficiency. Furthermore, it consistently outperforms PPO baselines in continuous control domains, particularly in sparse-reward and dynamic environments. Together, these findings establish ratio-variance regularization as a principled foundation for stable and data-efficient policy optimization.

Forum: https://openreview.net/forum?id=NT4Cz09S4w

---

## 399. Copyright-Bench: Agentic Evaluation of Copyright Law Compliance

**Abstract:** Large language model (LLM) agents increasingly perform commercial tasks that involve retrieving external content such as images and, where appropriate, reproducing that content. LLM agents should comply with the law, including copyright law. Presently, however, we lack adequate frameworks to assess whether they do so in practice. To that end, we introduce **Copyright-Bench**, a benchmark designed to evaluate *LLM agents' compliance with* *copyright law*. Copyright-Bench is comprised of realistic commercial tasks---website development, merchandise design, and pitch deck production---that involve agents selecting between public-domain content (the use of which is *legal*) and copyrighted content (the use of which is *infringing* in this setting). The evaluation introduces prompt variations that simulate different user preferences, as well as time pressure. Comparing state-of-the-art LLM agents against a human baseline, we find that: (1) agents select copyrighted works despite the availability of public-domain alternatives; and (2) for open-weights models, violation rates increase in response to certain user preferences and simulated time pressure.

Forum: https://openreview.net/forum?id=7y6a5ouziq

---

## 400. GoodDiffusion: Proactive Copyright Protection for Diffusion Bridge Models via Learnable Sample-specific Signatures

**Abstract:** This paper tackles the challenging problem of developing a proactive copyright protection mechanism that cuts off unauthorized use of diffusion bridge models. Existing studies largely fall into post-hoc attribution (e.g., watermarking and fingerprinting) or degradation-only defenses, which offer only indirect and limited preventive effect. We therefore propose GoodDiffusion, inspired by backdoor mechanisms, to enforce model-level use-time control by internalizing authorization into the generative process through a selectively permissive, otherwise closed behavior. Specifically, GoodDiffusion preserves high-quality generation for authorized queries carrying valid signatures, yet refuses to generate for unauthorized inputs. We further empirically show that naive static-signature designs (like conventional backdoor injection) are fundamentally fragile, since a surrogate signature can be efficiently recovered via gradient-based optimization. To strengthen security, we introduce a Learnable Signature Network (LSN) that assigns sample-specific signatures conditioned on each input. This breaks the universality of signatures and prevents a surrogate from transferring across inputs.
Extensive experiments validate that GoodDiffusion effectively blocks unauthorized use while maintaining strong generation quality for authorized users.

Forum: https://openreview.net/forum?id=wkV9jSqwa9

---

## 401. mHC: Manifold-Constrained Hyper-Connections

**Abstract:** Recently, studies exemplified by Hyper-Connections (HC) have extended the ubiquitous residual connection paradigm established over the past decade by expanding the residual stream width and diversifying connectivity patterns. While yielding substantial performance gains, this diversification fundamentally compromises the identity mapping property intrinsic to the residual connection, which causes severe training instability and restricted scalability, and additionally incurs notable memory access overhead. To address these challenges, we propose Manifold-Constrained Hyper-Connections (mHC), a general framework that projects the residual connection space of HC onto a specific manifold to restore the identity mapping property, while incorporating rigorous infrastructure optimization to ensure efficiency. Empirical experiments demonstrate that mHC is effective for training at scale, offering tangible performance improvements and superior scalability. We anticipate that mHC, as a flexible and practical extension of HC, will contribute to a deeper understanding of topological architecture design and suggest promising directions for the evolution of foundational models.

Forum: https://openreview.net/forum?id=mDhyxu8WRb

---

## 402. On the Difficulty of Learning a Meta-network for Training Data Selection

**Abstract:** Synthetic data are increasingly used to train neural networks, yet distributional mismatch with real data limits their effectiveness when used indiscriminately. A common strategy is to learn data weights via bi-level optimization, which we refer to as Meta-learning for Training-data Selection (MTS). Interestingly, in practice, MTS often performs below expectation. We identify two obstacles in properly training MTS: a poor gradient signal-to-noise ratio (GSNR), which causes optimization difficulties, and lack of informative features that correlates with data quality. We present a mathematical analysis of MTS, which reveals the dynamics of normalized data weights and the relation between disparate data quality and poor GSNR. The analysis suggests a a simple yet effective solution: increasing the batch size. Further, we propose a set of informative features that capture the positions of training data in their distributions and training dynamics. Experiments across four benchmarks show consistent improvements, achieving average gains of 5.49\% over training without selection and 2.89\% over the strongest baseline.

Forum: https://openreview.net/forum?id=RzgNmKi5Hw

---

## 403. UniPercept: Towards Unified Perceptual-Level Image Understanding across Aesthetics, Quality, Structure, and Texture

**Abstract:** Multimodal large language models (MLLMs) have achieved remarkable progress in visual understanding tasks such as visual grounding, segmentation, and captioning. However, their ability to perceive perceptual-level image features remains limited. In this work, we present UniPercept-Bench, a unified framework for perceptual-level image understanding across three key domains: Aesthetics, Quality, Structure and Texture. We establish a hierarchical definition system and construct large-scale datasets to evaluate perceptual-level image understanding. Based on this foundation, we develop a strong baseline UniPercept trained via Domain-Adaptive Pre-Training and Task-Aligned RL, enabling robust generalization across both Visual Rating (VR) and Visual Question Answering (VQA) tasks. UniPercept outperforms existing MLLMs on perceptual-level image understanding and can serve as a plug-and-play reward model for text-to-image generation. This work defines perceptual-level image understanding in the era of MLLMs and, through the introduction of a comprehensive benchmark together with a strong baseline, provides a solid foundation for advancing perceptual-level multimodal image understanding.

Forum: https://openreview.net/forum?id=8VVbElV06v

---

## 404. Nash Equilibria in Games with Playerwise Concave Coupling Constraints: Existence and Computation

**Abstract:** We study the existence and computation of Nash equilibria in concave games where the players' admissible strategies are subject to shared coupling constraints. Under playerwise concavity of constraints, we prove existence of Nash equilibria. Our proof leverages topological fixed point theory and novel structural insights into the contractibility of feasible sets, and relaxes strong assumptions for existence in prior work. Having established existence, we address the question of whether in the presence of coupling constraints, playerwise independent learning dynamics have convergence guarantees. We address this positively for the class of potential games by designing a convergent algorithm. To account for the possibly nonconvex feasible region, we employ a log barrier regularized gradient ascent with adaptive stepsizes.  Starting from an initial feasible strategy profile and under exact gradient feedback, the proposed method converges to an $\epsilon$-approximate constrained Nash equilibrium within $\mathcal{O}(\epsilon^{-3})$ iterations.

Forum: https://openreview.net/forum?id=BGmc3O41ZH

---

## 405. Fair Classification with Efficient and Post-hoc Controllable Fairness-Accuracy Trade-off

**Abstract:** Post-hoc controllability of fair machine learning models, the ability to control the trade-off between fairness and accuracy after training, is valuable for practical deployment. Existing post-processing methods provide such post-hoc controllability but often suffer from significant accuracy degradation, whereas in-processing methods achieve efficient trade-offs but require computationally expensive retraining for each change in trade-off ratio. To achieve both post-hoc controllability and efficient trade-offs, we propose a novel fair classification algorithm that learns effective feature representations to improve the trade-off efficiency of post-processing fair classifiers, by a gradient-based optimization approach. Experimental results on real-world datasets demonstrate that our method achieves trade-off efficiency comparable to, or even surpassing, in-processing methods, without requiring any retraining.

Forum: https://openreview.net/forum?id=lyN8OMVyMt

---

## 406. OSM+: Billion-Level Open Street Map  Dataset for City-wide Experiments

**Abstract:** Road network data provides rich information about cities, but processing worldwide OpenStreetMap (OSM) data is computationally intensive, and the resulting graphs are often difficult to unify for benchmarking downstream tasks. Existing graph learning benchmarks fail to capture the billion-scale and unique topological properties of real-world road networks, leaving model scalability underexplored. To close this gap, we process OSM data with distributed cloud computing using 5,000 cores and release **OSM+**, a structured worldwide 1-billion-vertex road network graph dataset designed for high accessibility and usability. OSM+ is open source and globally downloadable, providing an open-box graph structure together with an easy spatial query interface that allows users to retrieve, inspect, and integrate road network topology, geometry, and attributes without repeatedly preprocessing raw OSM files. We demonstrate the utility of OSM+ through four illustrative use cases: basic query, city boundary detection, traffic prediction, and traffic policy control. For traffic prediction, we construct a new 31-city benchmark by processing traffic data and combining it with OSM+, enabling broader spatial coverage and more comprehensive evaluation than commonly used datasets, while scaling from hundreds of road network intersections to thousands. For traffic policy control, we release a new six-city dataset at a much larger scale, introducing challenges for thousand-scale multi-agent coordination and controller scalability. We also provide data processing tools for integrating multimodal spatial-temporal data with OSM+ for geospatial foundation model training, thereby expediting the discovery of compelling scientific insights.

Forum: https://openreview.net/forum?id=CMeeyJzWZ5

---

## 407. PhotoAgent: Exploratory Visual Aesthetic Planning with Large Vision Models

**Abstract:** With the recent fast development of generative models, instruction-based image editing has shown great potential in generating high-quality images. However, the quality of editing highly depends on carefully designed instructions, placing the burden of task decomposition and sequencing entirely on the user. To achieve autonomous image editing, we present PhotoAgent, a system that advances image editing through explicit aesthetic planning. Specifically, PhotoAgent formulates autonomous image editing as a long-horizon decision-making problem. It reasons over user aesthetic intent, plans multi-step editing actions via tree search, and iteratively refines results through closed-loop execution with memory and visual feedback, without requiring step-by-step user prompts. To support reliable evaluation in real-world scenarios, we introduce UGC-Edit, an aesthetic evaluation benchmark consisting of 7,000 photos and a learned aesthetic reward model. We also construct a test set containing 1,017 photos to systematically assess autonomous photo editing performance. Extensive experiments demonstrate that PhotoAgent significantly outperforms existing methods in both instruction faithfulness and visual quality across a diverse range of editing scenarios.

Forum: https://openreview.net/forum?id=Ws8swqL5ob

---

## 408. $\tau^2$-Bench: Evaluating Conversational Agents in a Dual-Control Environment

**Abstract:** Existing benchmarks for conversational AI agents simulate _single-control_ environments, where only the AI agent can use tools to interact with the world, while the user remains a passive information provider. This differs from real-world scenarios like technical support, where users need to actively participate in modifying the state of the (shared) world. In order to address this gap, we introduce $\tau^2$-bench, with four key contributions:

(1) A novel Telecom dual-control domain modeled as a Dec-POMDP, where both agent and user make use of tools to act in a shared, dynamic environment that tests both agent coordination and communication;
(2) A compositional task generator that programmatically creates diverse, verifiable tasks from atomic components, ensuring domain coverage and controlled complexity;
(3) A reliable user simulator tightly coupled with the environment, whose behavior is constrained by tools and observable states, improving simulation fidelity;
(4) Fine-grained analysis of agent performance through multiple ablations including separating errors arising from reasoning vs communication/coordination.

In particular, our experiments show significant performance drops when agents shift from no-user to dual-control, highlighting the challenges of guiding users. Overall, $\tau^2$-bench provides a controlled testbed for agents that must both reason effectively and guide user actions. Code, data, and leaderboard are available at https://taubench.com/.

Forum: https://openreview.net/forum?id=OC2z7iSQKa

---

## 409. Error Propagation Mechanisms and Compensation Strategies for Quantized Diffusion Models

**Abstract:** Diffusion models have transformed image synthesis by establishing unprecedented quality and creativity benchmarks. Nevertheless, their large-scale deployment faces challenges due to computationally intensive iterative denoising processes. Although post-training quantization (PTQ) provides an effective pathway for accelerating sampling, the iterative nature of diffusion models causes stepwise quantization errors to accumulate progressively during generation, inevitably compromising output fidelity. To address this challenge, we develop a theoretical framework that mathematically formulates error propagation in Diffusion Models (DMs), deriving per-step quantization error propagation equations and establishing the first closed-form solution for cumulative error. Building on this theoretical foundation, we propose a timestep-aware cumulative error compensation scheme. Extensive experiments on multiple image datasets demonstrate that our compensation strategy effectively mitigates error propagation, significantly enhancing existing PTQ methods. Specifically, it achieves a 1.2 PSNR improvement over SVDQuant on SDXL W4A4, while incurring only an additional $<$ 0.5\% time overhead.

Forum: https://openreview.net/forum?id=4YIY66tpvQ

---

## 410. WestWorld: A Knowledge-Encoded Scalable Trajectory World Model for Diverse Robotic Systems

**Abstract:** Trajectory world models play a crucial role in robotic dynamics learning, planning, and control. While recent works have explored trajectory world models for diverse robotic systems, they struggle to scale to a large number of distinct system dynamics and overlook domain knowledge of physical structures. To address these limitations, we introduce *WestWorld*, a kno**W**ledge-**E**ncoded **S**calable **T**rajectory **World** model for diverse robotic systems. To tackle the scalability challenge, we propose a novel system-aware Mixture-of-Experts (Sys-MoE) that dynamically combines and routes specialized experts for different robotic systems via a learnable system embedding. To further enhance zero-shot generalization, we incorporate domain knowledge of robot physical structures by introducing a structural embedding that aligns trajectory representations with morphological information. After pretraining on 89 complex environments spanning diverse morphologies across both simulation and real-world settings, *WestWorld* achieves significant improvements over competitive baselines in zero- and few-shot trajectory prediction. Additionally, it shows strong scalability across a wide range of robotic environments and significantly improves performance on downstream model-based control for different robots. Finally, we deploy our model on a real-world Unitree Go1, where it demonstrates stable locomotion performance. The code is available at https://github.com/511205787/WestWorld.

Forum: https://openreview.net/forum?id=ncRRCG4BfP

---

## 411. Transformer Circuits Can Realize Clustering Algorithms

**Abstract:** Although transformers are most commonly optimized as statistical sequence models, it is unclear to what extent they can implement and learn exact algorithmic computations. Here, we specify a transformer implementation from first principles that executes a fundamental and widely used method for $k$-means clustering: Lloyd's algorithm. We theoretically prove and empirically demonstrate that this implementation of a transformer architecture, which we term the _$k$-means transformer_, exactly implements Lloyd's algorithm for $k$-means clustering using the standard circuit mechanisms of modern transformers: attention block, residual connections, and feed-forward block. In learning experiments, we find that training this base architecture on $k$-means clustering yields a generalizable clustering algorithm that surpasses Lloyd's algorithm in terms of clustering quality. Finally, we demonstrate that interpretable alterations (e.g., inclusion of layer normalizations) to this architecture yields diverse and novel variants of clustering algorithms, including soft $k$-means, spherical $k$-means, trimmed $k$-means. Overall, our results show that transformer circuit mechanisms can instantiate exact algorithmic routines for clustering, while simultaneously providing an effective learnable model.

Forum: https://openreview.net/forum?id=2jw5U060C4

---

## 412. Discretized Density-Guided Source-Free Domain Adaptation for Regression

**Abstract:** Source-Free Domain Adaptation (SFDA) enables model adaptation under distribution shifts without access to source data, providing a practical solution for privacy-sensitive applications and having shown substantial progress in classification. 
In contrast, regression involves ordered and continuous target variables, posing unique challenges for representation adaptation and pseudo-label refinement in the SFDA setting. 
To address this gap, we propose a novel algorithm for continuous label prediction in SFDA that leverages instance-dependent, discretized density–informed supervisory signals to refine pseudo-labels within an uncertainty-aware paradigm. 
By incorporating auxiliary discretized distribution learning, our method also promotes more compact and structured feature representations, mitigating the inherent difficulties of adapting regression models under distribution shift. 
We theoretically demonstrate that the resulting density structure is robust to potential perturbations, supporting reliable SFDA for regression. 
Extensive experiments across multiple benchmarks validate the effectiveness of the proposed approach.

Forum: https://openreview.net/forum?id=UQLGAjV1cf

---

## 413. MV-FGAD: Towards Efficient and Effective Federated Graph Anomaly Detection via Multi-view Learning

**Abstract:** Federated graph anomaly detection (GAD) aims to identify abnormal nodes in distributed subgraphs through federated learning. However, existing methods suffer from two limitations. 1) Their reliance on neighborhood aggregation assumes that anomalous information can be sufficiently captured, which often fails in federated learning with partitioned client subgraphs. 2) They overlook the detection bottleneck caused by weak attribute or structural anomalies. To tackle these challenges, we revisit federated GAD and reveal that weak anomalies exhibit harder-to-detect signals compared to strong anomalies. Specifically, we propose MV-FGAD, an efficient and effective federated GAD framework for mining anomalies of varying strengths. MV-FGAD introduces a federated knowledge learning module to aggregate and broadcast shared knowledge, which is further exploited to optimize local topological structures. Moreover, it designs a multi-view learning mechanism to capture diverse anomaly patterns, and adopts Mahalanobis distance–based scoring strategy to quantify node abnormality across views. Extensive experiments on real-world datasets of varying types and scales demonstrate MV-FGAD's efficiency and effectiveness. Our code is publicly available at https://github.com/Junyi-Yan/MV-FGAD.

Forum: https://openreview.net/forum?id=yBcY0bY45t

---

## 414. DR Tulu: Reinforcement Learning with Evolving Rubrics for Deep Research

**Abstract:** Deep research agents perform multi-step research to produce long-form, well-attributed answers. However, most open deep research agents are trained on easily verifiable short-form QA tasks via reinforcement learning with verifiable rewards, which does not extend to realistic long-form tasks. We address this with Reinforcement Learning with Evolving Rubrics (RLER), where rubrics are constructed and maintained to co-evolve with the policy model during training. This allows the rubrics to incorporate newly explored information from search and contrasting model responses, enabling better fact checking and more discriminative on-policy feedback. Using RLER, we develop Deep Research Tulu (DR Tulu-8B), the first fully open model that is directly trained for open-ended, long-form deep research. Across four long-form deep research benchmarks in science, healthcare, and general domains, DR Tulu-8B substantially outperforms existing open deep research agents (by 15.6% over Tongyi DR on average) and matches or exceeds proprietary deep research agents (by 0.7% over OpenAI DR on average), while being significantly smaller and cheaper per query (1000x cheaper than OpenAI DR per query).

Forum: https://openreview.net/forum?id=97NEP1pyS3

---

## 415. INDUCTION: Finite-Structure Concept Synthesis in First-Order Logic

**Abstract:** Induction is the search for a general rule that explains observations. We study logical induction in finite relational worlds: each problem gives small structures over a fixed vocabulary, labels objects belonging to an unknown unary concept, and asks for one first-order formula φ(x) that accounts for those labels across worlds. Finite domains make formulas mechanically checkable by exact evaluation and SMT. We introduce INDUCTION, a benchmark for finite-structure concept synthesis with three regimes: FULLOBS (full observation), where all facts are observed; CI (contrastive induction), where YES/NO worlds require discriminative hypotheses; and EC (existential completion), where validity is defined by world-local completion of unknown facts. We evaluate frontier language models, include symbolic synthesis baselines, and score both validity and formula size. Prompted models show real but incomplete capability, with sharp difficulty gradients and hard structural families. Held-out evaluation shows that compact formulas generalize far better than bloated ones; parsimony separates concept recovery from finite-world fit.

Forum: https://openreview.net/forum?id=kny606LUdl

---

## 416. dnaHNet: A Scalable and Hierarchical Foundation Model for Genomic Sequence Learning

**Abstract:** Genomic foundation models have the potential to decode DNA syntax, yet face a fundamental tradeoff. Standard subword tokenizers fragment biologically meaningful motifs such as codons and regulatory elements, while nucleotide-level models preserve biological coherence but incur prohibitive computational costs for long contexts. We introduce dnaHNet, a state-of-the-art tokenizer-free autoregressive model that segments and models genomic sequences end to end. Using a differentiable dynamic chunking mechanism, dnaHNet compresses raw nucleotides into latent tokens adaptively, balancing compression with predictive accuracy. Pretrained on prokaryotic genomes, dnaHNet outperforms leading architectures including StripedHyena2 in scaling and efficiency. This recursive chunking yields quadratic FLOP reductions, enabling $>3 \times$ inference speedup over Transformers. On zero-shot tasks, dnaHNet achieves superior performance in predicting protein variant fitness and gene essentiality, while automatically discovering hierarchical biological structures without supervision. These results establish dnaHNet as a scalable, interpretable framework for next-generation genomic modeling.

Forum: https://openreview.net/forum?id=6pN2KNCspk

---

## 417. Welfare-Optimal Classification with Accuracy Auctions

**Abstract:** Prediction algorithms are increasingly used to inform decisions about humans, but maximizing accuracy—the standard learning objective—is not necessarily optimal for this purpose. Instead, we propose optimizing social welfare, defined as the average gain users receive from correct predictions. Welfare enables to express, and therefore account for, heterogeneity in how much users benefit from accuracy. But since these valuations are private and users can benefit from overreporting them, learning must simultaneously elicit truthful values and optimize welfare with respect to them. To this end, we propose a novel learning algorithm that incorporates a truthful auction. We show how to compute allocations and prices efficiently, and bound the number of paying users—which surprisingly is independent of the sample size. We conclude with experiments on real and synthetic data that demonstrate our algorithm and explore the connections between welfare and accuracy.

Forum: https://openreview.net/forum?id=MA1LUDNA3s

---

## 418. Iterative Refinement Neural Operators are Learned Fixed-Point Solvers: A Principled Approach to Spectral Bias Mitigation

**Abstract:** Neural operators serve as fast, data-driven surrogates for scientific modeling but typically rely on a monolithic, single-pass inference procedure that struggles to resolve high-frequency details, a limitation known as spectral bias. We introduce the Iterative Refinement Neural Operator (IRNO), which augments pre-trained operators with a learned refinement module iteratively applied via fixed-point iteration. IRNO decomposes the prediction into a coarse initialization followed by successive residual corrections, paralleling classical numerical solvers. Under mild assumptions, we establish contraction of the induced operator, ensuring convergence to a unique fixed point. To explicitly target high-frequency errors, we propose a progressive spectral loss that adaptively increases penalty on high-frequency components over refinement steps during training. 
Across physical systems, IRNO consistently lowers error, with up to 56.05\% improvement on turbulent flow. On Active Matter, spectral analysis reveals that, relative to base operator, the normalized error ratios decrease to 27.72–36.10\% in low-, 5.07–6.68\% in mid-, and 1.48–2.04\% in high-frequencies, remaining stable beyond the trained iteration count.

Forum: https://openreview.net/forum?id=4lAS8tI23u

---

## 419. Estimating Tail Risks in Language Model Output Distributions

**Abstract:** Language models are increasingly capable and are being rapidly deployed on a population-level scale. As a result, the safety of these models is increasingly high-stakes. Fortunately, advances in alignment have significantly reduced the likelihood of harmful model outputs. However, when models are queried billions of times in a day, even rare worst-case behaviors will occur. Current safety evaluations focus on capturing the distribution of inputs that yield harmful outputs. These evaluations disregard the probabilistic nature of models and their tail output behavior. To measure this tail risk, we propose a method to efficiently estimate the probability of harmful outputs for any input query. Instead of naive brute-force sampling from the target model, where harmful outputs could be rare, we operationalize importance sampling by creating unsafe versions of the target model. These unsafe versions enable sample-efficient estimation by making harmful outputs more probable. On benchmarks measuring misuse and misalignment, these estimates match brute-force Monte Carlo estimates using 10–20× fewer samples. For example, we can estimate probability of harmful outputs on the order of $10^{−4}$ with just 500 samples. Additionally, we find that these harmfulness estimates can reveal the sensitivity of models to perturbations in model input and predict deployment risks. Our work demonstrates that rare-event estimation is both critical and feasible for safety evaluations.

Forum: https://openreview.net/forum?id=Joka19sTny

---

## 420. VALUEFLOW: Toward Pluralistic and Steerable Value-based Alignment in Large Language Models

**Abstract:** Aligning Large Language Models (LLMs) with the diverse spectrum of human values remains a central challenge: preference-based methods often fail to capture deeper motivational principles. Value-based approaches offer a more principled path, yet three gaps persist: extraction often ignores hierarchical structure, evaluation detects presence but not calibrated intensity, and steerability of LLMs at controlled intensities remains insufficiently understood. To address these limitations, we introduce VALUEFLOW, a unified framework that spans extraction, evaluation, and steering with calibrated intensity control. The framework integrates three components: (i) HiVES, a hierarchical value embedding space that captures intra- and cross-theory value structure; (ii) the Value Intensity DataBase (VIDB), a large-scale resource of value-labeled texts with intensity estimates derived from ranking-based aggregation; and (iii) an anchor-based evaluator that produces consistent intensity scores for model outputs by ranking them against VIDB panels. Using VALUEFLOW, we conduct a comprehensive large-scale study across ten models and four value theories, identifying asymmetries in steerability and composition laws for multi-value control. This paper establishes a scalable infrastructure for evaluating and controlling value intensity, advancing pluralistic alignment of LLMs.

Forum: https://openreview.net/forum?id=6zVV84vnCJ

---

## 421. Understanding Self-Supervised Learning via Latent Distribution Matching

**Abstract:** Self-supervised learning (SSL) excels at finding general-purpose latent representations from complex data, yet lacks a unifying theoretical framework that explains the diverse existing methods and guides the design of new ones. We cast SSL as latent distribution matching (LDM): learning representations that maximize their log-probability under an assumed latent model (alignment), while maximizing latent entropy to prevent collapse (uniformity). This view unifies independent component analysis with contrastive, non-contrastive, and predictive SSL methods, including stop gradient approaches. Leveraging LDM, we derive a nonlinear, sampling-free Bayesian filtering model with a Kalman-based predictor for high-dimensional timeseries. We further prove that predictive LDM yields identifiable latent representations under mild assumptions, even with nonlinear predictors. Overall, LDM clarifies the assumptions behind established SSL methods and provides principled guidance for developing new approaches.

Forum: https://openreview.net/forum?id=schXkGSTus

---

## 422. Near-Optimal Private Linear Regression via Iterative Hessian Mixing

**Abstract:** We study differentially private ordinary least squares (DP-OLS) with bounded data $(X,Y)$ via sketching-based mechanisms. While Gaussian sketching approaches have been explored for DP-OLS (Sheffet, 2017), they are typically viewed as less competitive than the Adaptive Sufficient Statistics Perturbation (AdaSSP) method (Wang, 2018), which directly perturbs the sufficient statistics $(X^{\top}X, X^{\top}Y)$. This method was shown to be close to information-theoretically optimal, while also exhibiting strong empirical performance. In this work, we propose \emph{Iterative Hessian Mixing} (IHM), an algorithm that builds on Gaussian-sketching approaches to DP-OLS and is inspired by the Iterative Hessian Sketch (Pilanci & Wainwright, 2016). We prove that IHM is differentially private and provide utility guarantees in the form of excess empirical risk bounds. These bounds improve upon those of AdaSSP by removing a multiplicative factor that can be as large as the square root of the data dimension. The design of the IHM is based on new accuracy guarantees that we present for prior Gaussian sketching approaches for DP-OLS, which clarify when these methods are expected to perform well and how IHM circumvents their inherent limitations.
We also conduct a rigorous empirical evaluation on a large suite of datasets, demonstrating that IHM consistently outperforms prior baselines, including AdaSSP.

Forum: https://openreview.net/forum?id=IwP8LZzc97

---

## 423. BFTS: Thompson Sampling with Bayesian Additive Regression Trees

**Abstract:** We propose Bayesian Forest Thompson Sampling (BFTS), which performs Thompson sampling using arm-wise Bayesian Additive Regression Trees (BART) to model each action's mean reward and generate MCMC-based posterior draws for decision-making. We derive an information-theoretic Bayesian regret bound of order $\widetilde{\mathcal O}(K\sqrt{T})$ for ideal posterior sampling under a correctly specified Bayesian design. Empirically, BFTS achieves competitive regret on nonlinear synthetic benchmarks with near-nominal uncertainty calibration, attains the best average rank across nine OpenML contextual bandit benchmarks, and yields higher estimated policy values than linear, neural, and tree-ensemble baselines in a Drink Less micro-randomized trial case study. Across OpenML benchmarks, BFTS is robust to hyperparameter choices.

Forum: https://openreview.net/forum?id=Z1nbtKcLQk

---

## 424. Gradient Flow Through Diagram Expansions: Learning Regimes and Explicit Solutions

**Abstract:** We develop a general mathematical framework to analyze scaling regimes and derive explicit analytic solutions for gradient flow (GF) in large learning problems. Our key innovation is a formal power series expansion of the loss evolution, with coefficients encoded by diagrams akin to Feynman diagrams. We show that this expansion has a well-defined large-size limit that can be used to reveal different learning phases and, in some cases, to obtain explicit solutions of the nonlinear GF. We focus on learning Canonical Polyadic (CP) decompositions of high-order tensors, and show that this model has several distinct extreme lazy and rich GF regimes such as free evolution, NTK and under- and over-parameterized mean-field. We show that these regimes depend  on the parameter scaling, tensor order, and symmetry of the model in a specific and subtle way. Moreover, we propose a general approach to summing the formal loss expansion by reducing it to a PDE; in a wide range of scenarios, it turns out to be first-order and solvable by the method of characteristics. We observe a very good agreement of our theoretical predictions with experimental results.

Forum: https://openreview.net/forum?id=BXE3Z0EHCs

---

## 425. ClinTutor-R1: Advancing Scalable and Robust One-to-Many Alignment in Clinical Socratic Education

**Abstract:** While Large Language Models (LLMs) have achieved remarkable success in dyadic (one-on-one) instruction, they face significant challenges in One-to-Many alignment, such as clinical ward rounds, where an instructor must simultaneously guide a diverse group of trainees. Current models often suffer from context dilution and goal misalignment, failing to balance individual scaffolding with collective learning progress. To address this, we introduce ClinEdu, a multi-agent pedagogical simulator that model the complexity of group dynamics. Leveraging this platform, we construct ClinTeach, a large-scale dataset of Socratic teaching dialogues, and propose ClinTutor-R1, the first multimodal agent explicitly architected to achieve one-to-many alignment in clinical education, employing an explicit internal thinking mechanism to model both individual belief states and group consensus. We validate our framework through a comprehensive protocol covering both standard static benchmarks and rigorous in-situ interactive evaluation within ClinEdu. Experimental results demonstrate that ClinTutor-R1 outperforms base models by over 20% and achieves parity with proprietary reasoning models , while exhibiting exceptional scalability in maintaining instructional quality across expanding student cohorts.

Forum: https://openreview.net/forum?id=15Gs65kvHS

---

## 426. Dynamics Reveals Structure: Challenging the Linear Propagation Assumption

**Abstract:** Neural networks adapt through first-order parameter updates, yet it remains unclear whether such updates preserve logical coherence.
We investigate the geometric limits of the Linear Propagation Assumption (LPA), the premise that local updates coherently propagate to logical consequences.
To formalize this, we adopt relation algebra and study three core operations on relations: negation flips truth values, converse swaps argument order, and composition chains relations.
For negation and converse, we prove that guaranteeing direction-agnostic first-order propagation necessitates a tensor factorization separating entity-pair context from relation content.
However, for composition, we identify a fundamental obstruction.
We show that composition reduces to conjunction, and prove that any conjunction well-defined on linear features must be bilinear.
Since bilinearity is incompatible with negation, this forces the feature map to collapse.
These results suggest that failures in knowledge editing, the reversal curse, and multi-hop reasoning may stem from common structural limitations inherent to the LPA.

Forum: https://openreview.net/forum?id=NTd7TQYQkX

---

## 427. Unifying Masked Diffusion Models with Various Generation Orders and Beyond

**Abstract:** Masked diffusion models (MDMs) are a potential alternative to autoregressive models (ARMs) for language generation, but generation quality depends critically on the generation order. Prior work either hard-codes an ordering (e.g., blockwise left-to-right) or learns an ordering policy for a pretrained MDM, which incurs extra cost and can yield suboptimal solutions due to the two-stage optimization. Motivated by this, we propose order-expressive masked diffusion model (OeMDM) for a broad class of diffusion generative processes with various generation orders, enabling the interpretation of MDM, ARM, and block diffusion in a single framework. Furthermore, building on OeMDM, we introduce learnable-order masked diffusion model (LoMDM), which jointly learns the generation ordering and diffusion backbone through a single objective from scratch, enabling the diffusion model to generate text in context-dependent ordering. Empirically, we confirm that LoMDM outperforms various discrete diffusion models across multiple language modeling benchmarks.

Forum: https://openreview.net/forum?id=ATpOQt9VVd

---

## 428. Linear Causal Representation Learning by Topological Ordering, Pruning, and Disentanglement

**Abstract:** Causal representation learning (CRL) has garnered increasing interest from the causal inference and artificial intelligence communities due to its potential to disentangle complex data-generating mechanism into causally interpretable latent features by leveraging the heterogeneity of modern datasets. In this paper, we further contribute to the CRL literature, by focusing on the stylized linear structural causal model over latent features and assuming a linear mixing function that maps latent features to the observed data or measurements. Existing linear CRL methods often rely on stringent assumptions, such as access to single-node interventional data or restrictive distributional constraints on latent features and/or exogenous measurement noise. However, these prerequisites can be easy to violate in practice. In this work, we propose a novel linear CRL algorithm that, unlike existing methods, operates under weaker assumptions on environment heterogeneity and data-generating distributions while still recovering latent causal features up to an equivalence class. We further validate our new algorithm via synthetic experiments and an interpretability analysis of large language models, demonstrating both its superiority over competing methods in finite samples and its potential in integrating causality into understanding artificial intelligence. The source code is available at [the accompanying GitHub link](https://github.com/utulie/code_for_linear_crl_paper_creator).

Forum: https://openreview.net/forum?id=4COS6SMf8r

---

## 429. Learning Unmasking Policies for Diffusion Language Models

**Abstract:** Diffusion (Large) Language Models (dLLMs) now match the downstream performance of their autoregressive counterparts on many tasks, while holding the promise of being more efficient during inference. One critical design aspect of dLLMs is the \textit{sampling procedure} that selects which tokens to unmask at each diffusion step. Indeed, recent work has found that heuristic strategies such as confidence thresholding improve both sample quality and token throughput compared to random unmasking. However, such heuristics have downsides: they require manual tuning, and we observe that their performance degrades with larger block sizes. In this work, we instead propose to train sampling procedures using reinforcement learning. Specifically, we formalize masked diffusion sampling as a Markov decision process in which the dLLM serves as the environment, and propose a lightweight policy based on a single-layer transformer that maps dLLM token confidences to unmasking decisions. Our experiments show that these trained policies match the performance of state-of-the-art heuristics when combined with semi-autoregressive (block) generation, while outperforming them in the full-diffusion setting. Our code is available at [https://github.com/apple/ml-rl-dllm](https://github.com/apple/ml-rl-dllm).

Forum: https://openreview.net/forum?id=F9NDKf5oPy

---

## 430. OmniFit: Bridging Modalities via Layer-Adaptive Token Compression for Omnimodal Large Language Models

**Abstract:** Emerging Omni-modal Large Language Models (OmniLLMs) enable real-time interaction across video, audio, and text but suffer from prohibitive computational costs due to the quadratic complexity of processing continuous streaming inputs. Existing token compression strategies remain suboptimal as they typically rely on biased modality-centric priors or enforce uniform retention policies, neglecting the heterogeneity across layers and the critical role of cross-modality alignment. To address these challenges, we propose OmniFit, a training-free framework that decouples interaction profiling from inference execution. OmniFit incorporates Layer-Adaptive Heterogeneity Profiling (LAHP) to dynamically allocate computational budgets based on layer-wise redundancy and modality preferences, preserving tokens according to the characteristics of each layer. Furthermore, we introduce Alignment-Rectified Token Selection (ARTS), a lightweight mechanism that efficiently identifies tokens semantically aligned with cross-modal cues. Extensive experiments on 3 model series across 10 benchmarks demonstrate that OmniFit establishes a new Pareto frontier, retaining 98\% of model performance with only 20\% token usage and achieves up to 2.31$\times$ end-to-end inference speedup and 2.5$\times$ VRAM saving, significantly outperforming state-of-the-art methods.

Forum: https://openreview.net/forum?id=8RY20mLzup

---

## 431. Pretrained Vision-Language-Action Models are Surprisingly Resistant to Forgetting in Continual Learning

**Abstract:** Continual learning is a long-standing challenge in robot policy learning, where a policy must acquire new skills over time without catastrophically forgetting previously learned ones. While prior work has extensively studied continual learning in relatively small behavior cloning (BC) policy models trained from scratch, its behavior in modern large-scale pretrained Vision-Language-Action (VLA) models remains underexplored. In this work, we find that pretrained VLAs are remarkably resistant to forgetting compared with smaller policy models trained from scratch. Simple Experience Replay (ER) works surprisingly well on VLAs, sometimes achieving zero forgetting even with a small replay data size. Our analysis reveals that pretraining plays a critical role in downstream continual learning performance: large pretrained models mitigate forgetting with a small replay buffer size while maintaining strong forward learning capabilities. Furthermore, we find that VLAs can retain relevant knowledge from prior tasks despite performance degradation during learning new tasks. This knowledge retention enables rapid recovery of seemingly forgotten skills through finetuning. Together, these insights imply that large-scale pretraining fundamentally changes the dynamics of continual learning, enabling models to continually acquire new skills over time with simple replay.

Forum: https://openreview.net/forum?id=VzdSHEab4G

---

## 432. Dissecting Multimodal In-Context Learning: Modality Asymmetries and Circuit Dynamics in modern Transformers

**Abstract:** Transformer-based multimodal large language models often exhibit in-context learning (ICL) capabilities. Motivated by this phenomenon, we ask: how do transformers learn to associate information across modalities from in-context examples? We investigate this through controlled experiments on small transformers trained on synthetic classification tasks, enabling precise manipulation of data statistics and model architecture. We begin by revisiting core principles of unimodal ICL in modern transformers. While several prior findings replicate, we find that Rotary Position Embeddings (RoPE) can delay the onset of ICL circuits. Extending to the multimodal setting reveals a fundamental learning asymmetry: when pretrained on high-diversity data from a primary modality, surprisingly low data complexity in the secondary modality suffices for multimodal ICL to emerge. Mechanistic analysis shows that both settings rely on an induction-style mechanism that copies labels from matching in-context exemplars; multimodal training refines and extends these circuits across modalities. Our findings provide a mechanistic foundation for understanding multimodal ICL in modern transformers and introduce a controlled testbed for future investigation. Code is available at: https://github.com/YiranHuangIrene/multimodal-icl

Forum: https://openreview.net/forum?id=fhPu6dCiwt

---

## 433. Latent Laplace Diffusion for Irregular Multivariate Time Series

**Abstract:** Irregular multivariate time series impose a trade-off for long-horizon forecasting: discrete methods can distort temporal structure via re-gridding, while continuous-time models often require sequential solvers prone to drift. To bridge this gap, we present Latent Laplace Diffusion (LLapDiff), a generative framework that models the target as a low-dimensional latent trajectory, enabling horizon-wide generation without step-by-step integration over physical time. We guide the reverse process utilizing a stable modal parameterization motivated by stochastic port-Hamiltonian dynamics, and parameterize its mean evolution in the Laplace domain via learnable complex-conjugate poles, enabling direct evaluation over irregular timestamps. We also link continuous dynamics to irregular observations through renewal-averaging analysis, which maps sampling gaps to effective event-domain poles and motivates a gap-aware history summarizer. Extensive experiments show that LLapDiff improves over baselines in long-horizon forecasting, and its continuous-time generative nature supports missing-value imputation by querying the same model at historical timestamps. Code is available at \url{https://github.com/pixelhero98/LLapDiffusion}.

Forum: https://openreview.net/forum?id=t73XUJvyQr

---

## 434. The Double-Edged Nature of the Rashomon Set for Trustworthy Machine Learning

**Abstract:** Real-world machine learning (ML) pipelines rarely produce a single model; instead, they produce a Rashomon set of many near-optimal ones. We show that this multiplicity reshapes key aspects of trustworthiness. At the individual-model level, sparse interpretable models tend to preserve privacy but are fragile to adversarial attacks. In contrast, the diversity within a large Rashomon set enables reactive robustness: even when an attack compromises one model, a practitioner can switch to a different near-optimal model that remains accurate, without retraining.  However, the same diversity increases information leakage, as disclosing more near-optimal models provides an attacker with progressively richer views of the training data. This produces a robustness–privacy trade-off governed by diversity, which we analyze theoretically and empirically. Beyond this trade-off, Rashomon sets are stable under small distribution shifts, so a set computed once remains valid under such shifts without re-computation. Our results highlight the dual role of Rashomon sets as both a resource and a risk for trustworthy ML.

Forum: https://openreview.net/forum?id=Z6OZh3CgkU

---

## 435. T$^2$PO: Uncertainty-Guided Exploration Control for Stable Multi-Turn Agentic Reinforcement Learning

**Abstract:** Recent progress in multi-turn reinforcement learning (RL) has significantly improved reasoning LLMs' performances on complex interactive tasks. 
Despite advances in stabilization techniques such as fine-grained credit assignment and trajectory filtering, instability remains pervasive and often leads to training collapse.
We argue that this instability stems from inefficient exploration in multi-turn settings, where policies continue to generate low-information actions that neither reduce uncertainty nor advance task progress.
To address this issue, we propose Token- and Turn-level Policy Optimization (T$^2$PO), an uncertainty-aware framework that explicitly controls exploration at fine-grained levels. At the token level, T$^2$PO monitors uncertainty dynamics and triggers a thinking intervention once  the marginal uncertainty change falls below a threshold. 
At the turn level, T$^2$PO identifies interactions with negligible exploration progress and dynamically resamples such turns to avoid wasted rollouts. 
We evaluate T$^2$PO in diverse environments, including WebShop, ALFWorld, and Search QA, demonstrating substantial gains in training stability and performance improvements with better exploration efficiency. Code is available at https://github.com/WillDreamer/T2PO.

Forum: https://openreview.net/forum?id=aD1zjvdJN4

---

## 436. Beyond Theorem Proving: Formulation, Framework and Benchmark for Formal Problem-Solving

**Abstract:** Large language models (LLMs) have achieved remarkable progress in mathematical reasoning, yet persistently suffer from hallucinations and erroneous logic. While formal theorem proving (FTP) shows promise in process-level reliability, it is limited to _verification_ (checking known propositions). This leaves constructive problem-solving (finding unknown terms that satisfy specific conditions) underexplored and disconnected from process-level verifiability.
To bridge this gap, we introduce **FPS** (_**F**ormal **P**roblem-**S**olving_), a principled framework to encompass the end-to-end problem-solving process in Lean 4. In FPS, the answer is an unknown metavariable coupled with a proof obligation, forcing it to be mathematically derived and verified. We further present **D-FPS** (_**D**eductive **FPS**_), which structures solving into forward derivation and backward verification, aligning more closely with human reasoning steps.
Three benchmarks of over 1,000 problems are constructed for evaluation: **FormalMath500**, **MiniF2F-Solving**, and **PutnamBench-Solving**. We further propose **RPE** (_**R**estricted **P**ropositional **E**quivalence_), a symbolic metric that evaluates semantic correctness beyond brittle string matching. Extensive experiments with state-of-the-art provers reveal that solving is significantly harder than proving, highlighting the ``alignment tax'' required to transition from loose validity checking to constructive, human-aligned reasoning.
Code and data are available at [https://github.com/Purewhite2019/formal_problem_solving_main](https://github.com/Purewhite2019/formal_problem_solving_main).

Forum: https://openreview.net/forum?id=hgMZraPlSv

---

## 437. Stable-GFlowNet: Toward Diverse and Robust LLM Red-Teaming via Contrastive Trajectory Balance

**Abstract:** Large Language Model Red-Teaming, which proactively identifies vulnerabilities of large language models, is an essential process for ensuring safety.
Finding effective and diverse attacks in red team activities is important, but achieving both is challenging.
Generative Flow Networks (GFN) that perform distribution matching are a promising method, but they are notorious for training instability and mode collapse.
In particular, unstable reward functions in red team activities accelerate mode collapse.
We propose Stable-GFN (S-GFN), which eliminates Z estimation in GFN and reduces training instability.
S-GFN avoids Z-estimation through pairwise comparisons and employs a robust masking methodology against noisy rewards.
Additionally, we propose a fluency stabilizer to prevent the model from getting stuck in local optima that produce gibberish.
S-GFN provides more stable training while maintaining the optimal policy of GFN.
We demonstrate the overwhelming attack performance and diversity of S-GFN across various settings.

Forum: https://openreview.net/forum?id=OyPE1ganBR

---

## 438. CIRBench: Evaluating Large Language Models as LLVM IR Optimizers

**Abstract:** Large language models are beginning to introduce a new paradigm for compilation: instead of only assisting at the source level, they can operate directly on **intermediate representations (IRs)**, the compiler’s internal code representation, Early studies suggest that LLM-guided optimization can sometimes rival traditional compiler optimizations on selected programs, but evidence remains fragmented.
Yet the community still lacks a rigorous IR-level benchmark that tests whether a model not only understands IR but can rewrite it under compiler-grade semantic constraints with meaningful performance impact.
We present **CIRBench**, a benchmark of 800 curated IR instances spanning four compiler-oriented tracks: Analysis infers IR properties, Repair fixes invalid IR, Refactor applies a single semantics-preserving compiler optimization, and Transform performs performance-oriented rewrites, together mirroring core optimization responsibilities in modern compilers.
CIRBench combines verifier, equivalence checking, and end-to-end performance measurement into a unified, layered correctness-aware evaluation of LLMs on IR.
On six mainstream LLMs, CIRBench shows that current models fail on many IR analysis and rewriting instances and on median underperform the compiler baseline, but we also observe a maximum speedup of $4.96\times$ over -O3.
These findings highlight both the opportunities and the remaining challenges of using LLMs inside optimizing compilers.

Forum: https://openreview.net/forum?id=D1ahrfGJ5e

---

## 439. EntroKV: Entropy-Guided Dynamic Budget Allocation for KV-Cache Compression

**Abstract:** The prohibitive memory footprint of the Key-Value (KV) cache imposes a critical bottleneck for efficient long-context LLM serving. 
Current compression techniques typically rely on static or uniform budget allocation, overlooking the significant heterogeneity in information density across attention heads. 
To address this, we introduce \textsc{EntroKV}, an entropy-driven dynamic budget allocation framework. 
Our method enables dynamic and rational allocation across layers, attention heads, and different tasks.
We demonstrate that attention entropy serves as a robust proxy for compression sensitivity: heads with high entropy require larger retention budgets, whereas low-entropy heads can be aggressively compressed without accuracy degradation. 
Functioning as a lightweight, plug-and-play module, \textsc{EntroKV} optimizes budget scheduling in real-time and is compatible with diverse compression operators. 
Extensive experiments demonstrate that \textsc{EntroKV} consistently outperforms baselines, retaining $\sim$98\% of full-cache performance at a 30\% budget ratio with negligible computational overhead. 
Our code is available at \url{https://anonymous.4open.science/r/EntroKV-D0C8/}.

Forum: https://openreview.net/forum?id=xhAMjsnWUe

---

## 440. From Text to Forecasts: Bridging Modality Gap with Temporal Evolution Semantic Space

**Abstract:** Incorporating textual information into time-series forecasting holds promise for addressing event- driven non-stationarity; however, a fundamental modality gap hinders effective fusion: textual descriptions express temporal impacts implicitly and qualitatively, whereas forecasting models rely on explicit and quantitative signals. Through controlled semi-synthetic experiments, we show that existing methods over-attend to redundant tokens and struggle to reliably translate textual semantics into usable numerical cues. To bridge this gap, we propose TESS, which introduces a Temporal Evolution Semantic Space as an intermediate bottleneck between modalities. This space consists of interpretable, numerically grounded temporal primitives—distribution shift, volatility, shape, and lag—extracted from text by an LLM via structured prompting and filtered through confidence-aware gating. Experiments on four real-world datasets demonstrate up to a 29% reduction in forecasting error compared to state-of-the-art unimodal and multimodal baselines. Code is available at: https://github.com/olivia3395/TESS.

Forum: https://openreview.net/forum?id=S2Fd1GEyv6

---

## 441. Chamaileon: Cross-Context Binder Design with Contextualized Modeling and Mixed Sampling

**Abstract:** The rapid evolution of generative models has unlocked new potentials in protein binder design, a pivotal task in structural biology, by facilitating end-to-end generation via joint sequence-structure modeling or hallucination. However, existing approaches are predominantly implemented under a single-target, single-state assumption, limiting their ability to model multi-target or multi-state interactions required for advanced function-oriented protein design. Here, we introduce Chamaileon, which unifies multi-target and multi-state binder design by formulating the problem as cross-context binding landscape modeling. The framework is underpinned by a training paradigm termed \textit{In-Context Complex Co-Design (I3CD)} for context-aware sequence-structure co-modeling. During inference, we employ \textit{Mixture-of-Paths Sampling (MoPS)}, a scalable strategy that optimizes a single sequence across contexts while alleviating the scarcity of high-quality multi-conformational paired data. Extensive evaluation on our newly constructed benchmark, \textit{CROSS}, demonstrates that Chamaileon effectively generates sequences adaptable to diverse conformational landscapes and multi-target requirements.

Forum: https://openreview.net/forum?id=JAQ9bm0Rp4

---

## 442. Dynamic Stratified Contrastive Learning with Upstream Augmentation for MILP Branching

**Abstract:** Mixed Integer Linear Programming (MILP) is a fundamental NP-hard problem that has garnered significant attention from both academia and industry.
The Branch-and-Bound (B&B) algorithm is the dominant approach for solving MILPs, where branching decisions play a critical role and have recently been enhanced by neural methods.
However, these methods still struggle with semantic variation across depths, the scarcity of upstream nodes, and the costly collection of strong branching samples. 
To address these issues, we propose SC-MILP, a Dynamic Stratified  Contrastive Training Framework for MILP Branching.
Our method groups B&B nodes based on their feature distributions and learns depth-aware, fine-grained node representations through dynamic stratified contrastive training.
To address data scarcity and imbalance at upstream nodes, we introduce an upstream-augmented MILP derivation procedure that generates both theoretically equivalent and perturbed instances.
Experiments on both synthetic and real-world MILP benchmarks, including large-scale instances, show that SC-MILP significantly improves branching accuracy, reduces solving time, with particularly strong gains at upstream nodes.

Forum: https://openreview.net/forum?id=LDNH3sQtMW

---

## 443. A Recursive Decomposition Framework for Causal Structure Learning in the Presence of Latent Variables

**Abstract:** Constraint-based causal discovery is widely used for learning causal structures, but heavy reliance on conditional independence (CI) testing makes it computationally expensive in high-dimensional settings.
To mitigate this limitation, many divide-and-conquer frameworks have been proposed, but most assume causal sufficiency, i.e., no latent variables.
In this paper, we show that divide-and-conquer strategies can be theoretically generalized beyond causal sufficiency to settings with latent variables. 
Specifically, we propose a recursive decomposition framework, termed DiCoLa, that enables divide-and-conquer causal discovery in the presence of latent variables. It recursively decomposes the global learning task into smaller subproblems and integrates their solutions through a principled reconstruction step to recover the global structure.
We theoretically establish the soundness and completeness of the proposed framework. Extensive experiments on synthetic data demonstrate that our approach significantly improves computational efficiency across a range of causal discovery algorithms, while experiments on a real-world dataset further illustrate its practical effectiveness.

Forum: https://openreview.net/forum?id=jdbQzlnYll

---

## 444. CLEAR: Context-Aware Learning with End-to-End Mask-Free Inference for Adaptive Video Subtitle Removal

**Abstract:** Video subtitle removal is essential for content localization and media re-editing, yet existing mask-guided diffusion methods face critical limitations: training inefficiency requiring extensive annotations and full model fine-tuning, inference complexity demanding explicit mask sequences, and static prior utilization unable to adapt to quality variations. We present CLEAR (Context-aware Learning for End-to-end Adaptive subtitle Removal), a lightweight adapter-based framework addressing these challenges through three technical innovations. First, self-supervised prior learning (Stage I) extracts occlusion guidance from video pairs using pixel differences as weak supervision, eliminating annotation dependency while learning generalizable subtitle features across languages. Second, LoRA-based adaptive refinement (Stage II) enables parameter-efficient training that preserves pre-trained visual priors while achieving true mask-free end-to-end inference without external detection modules. Third, adaptive focal weighting dynamically adjusts prior influence based on local quality assessment, effectively handling diverse subtitle styles and noisy guidance signals. Extensive experiments demonstrate CLEAR's superior performance in multilingual subtitle removal while requiring only 0.77% trainable parameters, establishing a new paradigm for efficient video text removal without inference-time mask dependencies.

Forum: https://openreview.net/forum?id=HszNb2xE7P

---

## 445. Mind-Omni: A Unified Multi-Task Framework for Brain-Vision-Language Modeling via Discrete Diffusion

**Abstract:** Modeling the interplay between external stimuli and internal neural representations is a pivotal research area for Brain-Computer Interfaces (BCIs). A major limitation of prior work is the prevailing paradigm of specialized, single-task models, which curtails versatility and neglects inter-task synergies. To address this, we propose Mind-Omni, the first versatile framework that unifies seven distinct encoding and decoding tasks through a discrete diffusion paradigm. At its core is a novel Brain Tokenizer that transforms heterogeneous, continuous brain signals into standardized, discrete tokens. This enables direct, token-level interactions for mutual understanding and generation between any two or more modalities within a shared semantic space. To unlock advanced reasoning capabilities, we further curate a specialized Brain Question Answering (BQA) instruction-tuning dataset. Our model not only establishes a new state-of-the-art among multi-task unified frameworks but also provides strong evidence for multi-task synergy. By demonstrating performance competitive with, and at times superior to, larger specialized models, our work offers a powerful new paradigm for neural modeling and paves the way for foundation models of neural activity.

Forum: https://openreview.net/forum?id=3gCdh3u2GK

---

## 446. FedPissa: Towards Federated Personalized Adaptation of Foundation Models via LoRA Subspace Mapping

**Abstract:** LoRA efficiently adapts large pre-trained models via low-rank updates, making it a strong parameter-efficient fine-tuning (PEFT) method. When integrated with Federated Learning (FL), it enables collaborative fine-tuning across distributed clients, leveraging rich downstream data without exposing private information. However, this strategy is hindered by data heterogeneity and limits personalization performance. To address this, personalized FedLoRA approaches have been proposed and employ a dual-LoRA architecture, e.g., one branch for global knowledge and another for client-specific adaptation. Nevertheless, this dual-LoRA design introduces additional computational overhead and structural redundancy. To address this limitation, we propose FedPissa, the first framework that rethinks single-LoRA via selective aggregation and subspace decorrelation. We selectively aggregate LoRA components based on their aggregation dynamics, and further apply a decorrelated subspace projection to mitigate heterogeneous update conflicts, reducing cross-client interference and improving personalized adaptation. Experiments on texual and visual scenario show that FedPissa not only achieves up to 35\% lower communication and computation cost, but also improves superior compared to counterparts.

Forum: https://openreview.net/forum?id=ZEWN40uNEh

---

## 447. TokSuite: Measuring the Impact of Tokenizer Choice on Language Model Behavior

**Abstract:** Tokenizers provide the fundamental basis through which text is represented and processed by language models (LMs).
Despite the importance of tokenization, its role in LM performance and behavior is poorly understood due to the challenge of measuring the impact of tokenization in isolation.
To address this need, we present TokSuite, a collection of models and a benchmark that supports research into tokenization's influence on LMs. 
Specifically, we release fourteen pre-trained models that use different off-the-shelf tokenizers but are otherwise identical, using the same architecture, dataset, training budget, and initialization. 
We also release a multilingual robustness benchmark that measures model performance under real-world perturbations in English, Chinese, Farsi, Italian, and Turkish, curated by native annotators. 
Together, TokSuite allows robust decoupling of the influence of a model's tokenizer, supporting a series of novel findings that elucidate the respective benefits and shortcomings of a wide range of popular tokenizers.

Forum: https://openreview.net/forum?id=vIZz7LvObC

---

## 448. Causal Modeling of Selection in Evolution

**Abstract:** Understanding potential selection in data is crucial for causal discovery; we argue that "selection" in common narratives takes two forms, which we term _static_ and _evolutionary_ selection, respectively. Static selection refers to a one-shot filtering process where observed data consist of a _subset_ of the population of interest, as in survey volunteer bias. Evolutionary selection, in contrast, operates through repeated rounds of differential fitness in reproduction, where observed data constitute the latest _generation_ shaped by a historical trajectory, as in immune adaptation, antibiotic resistance, and social norm emergence. Existing methods largely conflate these two forms and rely on an identical graphical model of selection. We show that this model is valid for static settings but fails to characterize data under evolution, yielding false discovery results. To address this, we introduce a new model that specifically characterizes evolutionary selection, and develop a sound and complete procedure for identifying such models from data across one or multiple environments or generations. Experimental results validate the method's ability to uncover the relevant mechanisms underlying evolution from data.

Forum: https://openreview.net/forum?id=mOcTXKawFY

---

## 449. Overcoming PINNs Failure Modes In High Dimension With Low-Rank Fourier Sum

**Abstract:** Physics-informed neural networks (PINNs) can be unreliable on PDEs with oscillatory, multiscale, stiff, or long-time solutions, and these difficulties worsen in high dimensions where collocation-based training yields large numerical integration error and high-variance gradients. We propose Low-Rank Fourier Sums (LoRFS), representing the solution as a low-rank sum of separable Fourier expansions (products of one-dimensional Fourier series across coordinates). This makes high-frequency structure explicit and enables closed-form evaluation of common physics-based objectives and their gradients (e.g., $L^2$ residual and variational losses), replacing sampling-based collocation estimates with analytic loss evaluation and eliminating sampling noise. We further provide theoretical results that clarify why LoRFS is particularly well suited to high-dimensional regimes. Across canonical PINN failure-mode benchmarks and their high-dimensional extensions, LoRFS consistently outperforms strong PINN baselines and remains stable in regimes where competing methods degrade.

Forum: https://openreview.net/forum?id=fq3yjso8Tn

---

## 450. GEM: Geometric Erasure by Contrastive Velocity Matching in Rectified Flows

**Abstract:** While the rapid adoption of multimodal generative models offers immense potential, it has also increased the risks of harmful content synthesis, deepfakes, and copyright infringements. To address these challenges, concept erasure has emerged as a prospective safeguard. However, as the field gradually transitions from U-Net-based diffusion models to Rectified Flow Transformers, erasure research has struggled to keep pace. In this work, we introduce GEM, a simple but highly effective erasure framework for Rectified Flow models. As part of our contribution, we establish a principled bridge between trajectory-based unlearning grounded in Generative Flow Networks and classic teacher-guided erasure: we translate trajectory-based signals into a teacher-guided flow-matching setup that unifies the strengths of both paradigms. Concretely, a teacher provides complementary attraction and repulsion signals that we combine into a single geometric guidance objective, yielding targeted suppression of unwanted concepts while preserving benign generation.

Forum: https://openreview.net/forum?id=NBMCwxTRSA

---

## 451. Dynamic Programming for Epistemic Uncertainty in Markov Decision Processes

**Abstract:** In this paper, we propose a general theory of ambiguity-averse MDPs, which treats the uncertain transition probabilities as random variables and evaluates a policy via a risk measure applied to its random return. This ambiguity-averse MDP framework unifies several models of MDPs with epistemic uncertainty for specific choices of risk measures.  We extend the concepts of value functions and Bellman operators to our setting. Based on these objects, we establish the consequences of dynamic programming principles in this framework (existence of stationary policies, value and policy iteration algorithms), and we completely characterize law-invariant risk measures compatible with dynamic programming. Our work draws connections among several variants of MDP models and fully delineates what is possible under the dynamic programming paradigm and which risk measures require leaving it.

Forum: https://openreview.net/forum?id=oUv02QKUxG

---

## 452. SVD as a Fast Interpretability Method for Transformers

**Abstract:** Mechanistic interpretability of Transformer models commonly relies on training auxiliary proxy models, such as Sparse Autoencoders or Cross-Layer Transcoders. While effective, these post-hoc approaches introduce approximation bias and incur substantial computational overhead. We propose an alternative, training-free interpretability framework that directly exploits the Singular Value Decomposition (SVD) of weight matrices in Transformer MLP sublayers. By operating natively on model parameters, our method improves scalability while preserving fidelity to the original weights. We show that the projection matrices of MLP sublayers admit a natural decomposition into orthogonal, interpretable rank-1 subspaces, which we term **Detector-Effector Units** (DEUs). Within each unit, a singular vector functions as a detector of input patterns and modulates a coupled effector vector that encodes output semantics. Building on this structure, we introduce **Subspace Contribution Analysis** (SCA), a diagnostic method that quantifies the direct causal contribution of individual native subspaces to model predictions. Experiments across the GPT-2 family demonstrate that our framework, **Native Network Anatomy** (NaNA), identifies dominant functional pathways with orders-of-magnitude efficiency gains over training-based interpretability baselines, while maintaining weight fidelity. Our results suggest that SVD-based analyses provide a scalable and faithful alternative to learned proxy approaches for mechanistic interpretability.

Forum: https://openreview.net/forum?id=7tt8TwMjdJ

---

## 453. Optimal Transport under Group Fairness Constraints

**Abstract:** Ensuring fairness in matching algorithms is a key challenge in allocating scarce resources and positions. Focusing on Optimal Transport (OT), we introduce a novel notion of group fairness requiring that the probability of matching two individuals from any two given groups in the OT plan satisfies a predefined target. We first propose a modified Sinkhorn algorithm to compute perfectly fair transport plans efficiently. Since exact fairness can significantly degrade matching quality in practice, we then develop two relaxation strategies. The first one involves solving a penalized OT problem, for which we derive novel finite-sample complexity guarantees. Our second strategy leverages bilevel optimization to learn a ground cost that induces a fair OT solution, and we establish a bound on the deviation of fairness when matching unseen data. Finally, we present empirical results illustrating the performance of our approaches and the trade-off between fairness and transport cost.

Forum: https://openreview.net/forum?id=tibRKqUHcv

---

## 454. From Distribution to Geometry: Stable Graph Generalization via Invariant Barycenters

**Abstract:** Graph neural networks (GNNs) excel in graph analyzing tasks but often suffer from poor generalization under Out-of-Distribution (OOD) scenarios. Although this problem has attracted increasing attention, most solutions primarily rely on empirical designs, lacking effective mechanisms to characterize and quantify invariance for graph representation learning. To address these limitations, we propose DIGL, a novel graph learning method that improves the OOD generalization of GNNs. Our work makes an initial attempt to geometrize invariance for graphs by introducing computational optimal transport (OT) theory to characterize invariance principle. Specifically, we formulate the underlying invariant prototype shared by graphs across different environments as a distribution barycenter, and consider graph representations in each specific environment as distortions of the prototype. Building on this idea, we establish an invariant learning framework to promote the model to learn purely invariant graph representations for downstream tasks. Moreover, we derive a unified optimization objective for model implementation and provide theoretical analysis to justify our method. Extensive experiments on a broad range of benchmark datasets demonstrate the superior generalization ability of our method compared with baseline methods under various OOD settings.

Forum: https://openreview.net/forum?id=YI3IpasVw5

---

## 455. Interpretable Functional Koopman Learning with Non-Markovian Closure for Spatiotemporal Systems

**Abstract:** Precise prediction of spatiotemporal dynamics over predictive horizons is constrained by the computational cost of high-fidelity solvers and the sparsity, noise, and irregularity of data. We introduce MERLIN, a Koopman-based framework that lifts dynamics to the evolution of learned *observation functionals* with near-linear progression, enabling full-field reconstruction at arbitrary resolutions. Theoretically, we develop a functional Koopman theory for PDEs and compensate for the loss of finite-dimensional linear invariance via the Mori–Zwanzig formalism, which augments the linear backbone with non-Markovian memory terms to improve predictive accuracy. Practically, MERLIN employs discretization-invariant *function encoders* that map partial, irregular observations to observables, and resolution-free *function decoders* that reconstruct states at arbitrary query points. Training under linear constraints yields an interpretable, low-dimensional model
that captures principal modes and supports reduced-order modeling, while memory
correction further enables stable long-horizon rollouts even in ultra-low-dimensional
latent spaces. Our code is available at: https://github.com/RobinLufdu/MERLIN.

Forum: https://openreview.net/forum?id=lquDiBCgNJ

---

## 456. Mind Your Margin and Boundary: Are Your Distilled Datasets Truly Robust?

**Abstract:** Dataset distillation (DD) compresses a large training set into a small synthetic set for efficient training, but most DD methods optimize only clean accuracy and leave robustness uncontrolled. Recent robust DD methods improve robustness, yet they often suffer from a poor accuracy–robustness trade-off because they (i) treat all adversarially perturbed examples uniformly, despite robust risk being dominated by near-zero robust margins, and (ii) do not explicitly increase inter-class separation in the decision boundary where attacks concentrate. We present Contrastive Curriculum for Robust Dataset Distillation (C$^2$R), a framework that couples an attack-aware curriculum with a contrastive robustness objective. From a robust-margin perspective, we derive a \emph{perturbation score} that approximates each sample’s robust hinge, enabling a curriculum that prioritizes the smallest-margin adversaries that most directly drive robust error. In parallel, a class-balanced contrastive robustness loss enforces adversarial invariance while explicitly widening boundary separation across classes. Experiments on CIFAR-10/100, Tiny-ImageNet, and multiple ImageNet-1K subsets under six attacks show that C$^2$R achieves the best robust accuracy, outperforming prior robust DD by 2.8% on average.

Forum: https://openreview.net/forum?id=Y3K0pPyVcs

---

## 457. RelaxFlow: Text-Driven Amodal 3D Generation

**Abstract:** Image-to-3D generation faces inherent semantic ambiguity under occlusion, where partial observation alone is often insufficient to determine object category. 
In this work, we formalize *text-driven amodal 3D generation*, where text prompts steer the completion of unseen regions while strictly preserving input observation. Crucially, we identify that these objectives demand distinct control granularities: rigid control for the observation versus relaxed structural control for the prompt. To this end, we propose **RelaxFlow**, a training-free dual-branch framework that decouples control granularity via a Multi-Prior Consensus Module and a Relaxation Mechanism. Theoretically, we prove that our relaxation is equivalent to applying a low-pass filter on the generative vector field, which suppresses high-frequency instance details to isolate geometric structure that accommodates the observation. To facilitate evaluation, we introduce two diagnostic benchmarks, **ExtremeOcc-3D** and **AmbiSem-3D**. Extensive experiments demonstrate that RelaxFlow successfully steers the generation of unseen regions to match the prompt intent without compromising visual fidelity. Code and datasets will be released.

Forum: https://openreview.net/forum?id=UamxHbDR3p

---

## 458. ECHO: Elastic Speculative Decoding with Sparse Gating for High-Concurrency Scenarios

**Abstract:** Speculative Decodin promises to accelerate Large Language Model inference, yet its efficacy often degrades in production-grade scenarios. Existing evaluations typically overlook the compute-bound nature of high-concurrency regimes, where verification compute becomes the dominant bottleneck. Consequently, prior methods face a dilemma: static trees incur massive verification waste, while dynamic trees suffer from cumulative misjudgments and kernel incompatibility.
To bridge this gap, we introduce ECHO, a high concurrency-oriented framework integrated into SGLang that reformulates speculative execution as a budgeted scheduling problem.
Crucially, ECHO employs sparse confidence gating to manage the batch as a unified super-tree, elastically pivoting budget between depth and width to co-optimize the trade-off between reducing global verification steps and maximizing per-step efficiency.
Extensive evaluations across diverse model scales—particularly the industrial-grade Qwen3-235B—demonstrate that ECHO consistently outperforms state-of-the-art baselines in both low-load and high-load scenarios, achieving up to 5.35$\times$ walltime speedup and delivering over 20\% relative speedup gain against the strongest baselines.

Forum: https://openreview.net/forum?id=L31hKCWRsN

---

## 459. Information Flow Reveals When to Trust Language Models

**Abstract:** In retrieval-augmented generation, language models can generate incorrect responses if they fail to utilize query-relevant content from the retrieved evidence. This shifts the focus of uncertainty quantification (UQ) toward assessing contextual grounding, i.e., whether predictions are supported by query-relevant tokens. Recent UQ methods unpack language models to characterize how inputs are processed. Nevertheless, these methods focus on a few layers and overlook the whole progressive propagation within the model, thereby failing to fully capture the grounding dynamics essential for reliable uncertainty estimation. We use information flow to build a layer-wise trace that reveals each context token’s contribution to the output, providing an interpretable basis for assessing reliability. From this analysis, we introduce two measures to calibrate prediction confidence. The first, \textit{simulatability}, posits that a prediction is more likely to be correct when context token contributions align closely with their true relevance. The second, \textit{concentration}, asserts that a response is more likely to be correct when it is derived from a narrow, focused subset of tokens. Experiments show that our method achieves an average AUROC of 0.709, exceeding the runner-up performance of 0.676, while maintaining moderate computational cost.

Forum: https://openreview.net/forum?id=vd8HzoFZ7v

---

## 460. Incremental BPE Tokenization

**Abstract:** We propose a novel algorithm for incremental Byte Pair Encoding (BPE) tokenization.
The algorithm processes each input byte in **worst-case** $\mathcal{O}(\log^2 t)$ time,
leading to an overall complexity of $\mathcal{O}(n \log^2 t)$,
where $n$ is the input length and $t$ is the maximum token length.
The algorithm incrementally maintains BPE tokenization results for every prefix of the input text,
implementing the standard BPE merge procedure defined by a fixed set of merge rules.
This enables efficient partial tokenization in streaming settings.
Functioning as a drop-in replacement for standard BPE,
our approach achieves a speedup of up to ${\sim}3\times$ over Hugging Face's tokenizers,
and demonstrates significant latency reductions over OpenAI's tiktoken on pathological inputs.
We further introduce an eager output algorithm that enables streaming output,
emitting tokens as soon as token boundaries are determined during incremental tokenization.
Overall, our results demonstrate that BPE tokenization can be performed incrementally
with strong worst-case guarantees,
while providing practical latency benefits in modern large language model pipelines.
The source code is available at https://github.com/ModelTC/mtc-inc-bpe.

Forum: https://openreview.net/forum?id=ZbWgrDzCQo

---

## 461. Attribution-Guided and Coverage-Maximized Pruning for Structural MoE Compression

**Abstract:** Mixture-of-Experts (MoE) models scale compute efficiently, yet they remain expensive to deploy due to substantial memory footprint and inference overhead. Prior methods mainly operate at the expert level, either removing whole experts or ranking experts by importance. However, such expert-wise decisions are too coarse to identify redundancy, and often misallocate pruning budgets and limits compression. This issue worsens in large MoEs with dynamic routing and heterogeneous experts. To alleviate this dilemma, we for the first time observe that information in MoE experts is highly concentrated in a few channels, leaving substantial redundancy even in "high importance" experts. Accordingly, we propose a structural pruning framework tailored for MoEs, reforming the prune-ratio objective to maximizing channel-score coverage via an efficient attribution-based approximation. Experiments on DeepSeek and Qwen MoEs retain accuracy under 50\% or 25\% pruning joinly with 4-bit quantization, reducing the memory footprint of Qwen3-30B-A3B by 5.27$\times$, and outperforming state-of-the-art baselines under diverse benchmarks.

Forum: https://openreview.net/forum?id=oreET6Wz52

---

## 462. Less is Enough: Synthesizing Diverse Data in LLM Feature Space with Sparse Autoencoders

**Abstract:** The diversity of post-training data is critical for effective downstream performance in large language models (LLMs). 
Many existing approaches to constructing post-training data quantify diversity using text-based metrics that capture linguistic variation, but such metrics provide only weak signals for the task-relevant features that determine downstream performance.
In this work, we introduce ***Feature Activation Coverage* (FAC)** which measures data diversity in an interpretable feature space. 
Building upon this metric, we further propose a diversity-driven data synthesis framework, named **FAC Synthesis**, that first uses a sparse autoencoder to identify missing features from a seed dataset, and then generates synthetic samples that explicitly reflect these features.
Experiments show that our approach consistently improves both data diversity and downstream performance on various tasks, including instruction following, toxicity detection, reward modeling, and behavior steering. 
Interestingly, we identify a shared, interpretable feature space across model families (i.e., LLaMA, Mistral, and Qwen), enabling cross-model knowledge transfer.
Our work provides a solid and practical methodology for exploring data-centric optimization of LLMs.

Forum: https://openreview.net/forum?id=GVupwQ0578

---

## 463. RED-HDP-HMM: Observation-Dependent Durations for Bayesian Nonparametric Sequential Models

**Abstract:** The Hierarchical Dirichlet Process Hidden Markov Model (HDP-HMM) is a Bayesian nonparametric extension of the classical Hidden Markov Model, well-suited for learning from (spatio-)temporal data. To relax the restrictive geometric assumption on state durations, the HDP Hidden Semi-Markov Model was introduced. However, both models assume stationary state durations, which limits their expressive power. In this work, we extend the HDP-HMM framework by incorporating recurrent explicit duration modeling, resulting in a more general and flexible model: the Recurrent Explicit Duration HDP-HMM (RED-HDP-HMM). We propose a Gibbs sampling method for efficient inference in this model. Empirical results on both synthetic and real-world segmentation tasks demonstrate that RED-HDP-HMM consistently outperforms the disentangled sticky HDP-HMM and the standard sticky HDP-HMM.

Forum: https://openreview.net/forum?id=MqhcqHVc8l

---

## 464. A Causal Decomposition Approach for Fair Contextual Multi-Armed Bandits

**Abstract:** Counterfactual reasoning is one of the fundamental facets of human cognition, involved in various tasks such as explanation, credit assignment, blame, and responsibility. It describes the queries what would have happened had some intervention been performed given that something else, corresponding to Layer 3 of the Pearl Causal Hierarchy.  In this project, we examine a specific type of counterfactual quantities, called counterfactual direct (Ctf-DE), indirect (Ctf-IE), and spurious (Ctf-SE) effects for quantifying fairness in a sequential decision-making framework. Building on these measures, we formulate an online causally-fair learning problem with multiple long-term constraints and study it in both non-parametric contextual bandits and parametric logistic bandits settings. We achieve sublinear regret and violations bounds for both bandits settings with roundwise counterfactual fairness constraints (that are a priori unknown) without Slater's condition. For logistic bandits, our method achieves $\mathcal{O}(1)$ per-round time complexity using an online mirror descent estimator, yielding an efficient algorithm.

Forum: https://openreview.net/forum?id=kHEL0GD175

---

## 465. Generalizable and Composable Multi-Model Embedding Translation

**Abstract:** Embedding translation enables interoperability across embedding models, allowing embedding vectors to be reused without costly re-embedding.  However, existing methods are typically evaluated under simplified pairwise and i.i.d. settings and behave as black boxes at inference time, leading to unreliable performance under out-of-distribution (OOD) inputs, multi-model mixing, and composed translations.  We analyze embedding translation from a geometric perspective and derive an interpretable error bound that explains systematic error amplification under OOD inputs, mixing and chaining.  Building on this, we propose a geometry-aware confidence metric and a Hierarchical Mixture of Experts (HMoE) framework with localized, parameter-efficient adaptation.  Following the MTEB leaderboard, we conduct large-scale experiments over 10 embedding models and 6 datasets across 90 pairwise translation settings.  HMoE outperforms every baseline for every model pair over every dataset under OOD scenarios. Furthermore, multi-model mixing and chaining only degrade our performance in Recall@100 by $0.5\% -- 2.6\%$, compared to $7.2\% -- 92.3\%$ recall drop by existing methods.

Forum: https://openreview.net/forum?id=qmfp2eqYD1

---

## 466. DAVE: Distribution-Aware Attribution via ViT Gradient Decomposition

**Abstract:** Vision Transformers (ViTs) have become a dominant architecture in computer vision, yet producing stable and high-resolution attribution maps remains challenging. Architectural components such as patch embeddings and attention routing often introduce structured artifacts in pixel-level explanations, leading many existing methods to rely on coarse patch-level attributions. We introduce DAVE (Distribution-Aware Attribution via ViT Gradient Decomposition), a mathematically grounded attribution method for ViTs based on a structured decomposition of the input gradient. By exploiting architectural properties of ViTs, DAVE isolates locally equivariant and stable components of the effective input-output mapping while suppressing architecture-induced artifacts and instability. Consequently, DAVE produces robust, precise, and class-consistent attribution maps that highlight model-relevant visual features. Experimental results show that across supervised, self-supervised, and inherently interpretable ViTs, DAVE outperforms prior methods on localization, faithfulness, and user studies.

Forum: https://openreview.net/forum?id=ykTMNA6Mbh

---

## 467. Don't Reinvent the Wheel, Just Realign the Spokes: Resource-Efficient Federated Fine-Tuning via Rank-Wise Expert Assembly

**Abstract:** Federated fine-tuning presents a promising avenue for adapting Large Language Models (LLMs) to downstream tasks while preserving data privacy. However, the prohibitive computational and communication overhead of LLM adaptation inhibits its deployment on resource-constrained edge devices. In this paper, we propose SmartFed, a resource-efficient framework that circumvents expensive training from scratch by intelligently reusing knowledge embedded in existing LoRA modules. To fully exploit this potential and ensure scalability, we introduce the Mixture of Rank-Wise Experts (MoRE). MoRE decomposes LoRA modules into fine-grained rank-level experts, which are selectively activated based on input semantics and resource budgets. Furthermore, to optimize resource utilization, we propose Elastic Expert Quota Allocation (EEQA), a strategy that adaptively distributes expert capacity across parameter matrices based on their contribution to model performance. Extensive evaluations across multiple benchmarks demonstrate that SmartFed significantly outperforms state-of-the-art methods in both model performance and training efficiency. Our code is publicly available at https://github.com/benmagnifico/SmartFed.

Forum: https://openreview.net/forum?id=HdsEUNo4C4

---

## 468. Prototype-guided Bilateral Alignment Multimodal Federated Learning

**Abstract:** Multimodal federated learning (MFL) has emerged as a pivotal paradigm for leveraging distributed data to enhance model performance. However, existing methods  predominantly rely on idealized assumptions of model homogeneity and balanced modality distributions, rendering them ill-suited for practical scenarios characterized by heterogeneous client architectures and severe modality imbalance. To address these challenges, we propose a \textbf{M}ultimodal \textbf{Fed}erated learning Prototype-guided Bilateral Alignment (MFedPBA) framework. MFedPBA facilitates robust knowledge synergy through a dual alignment mechanism: (i) at the feature level, it aligns heterogeneous feature spaces via a projection encoder optimized by contrastive learning and the Gromov-Wasserstein distance; (ii) at the decision level, it employs an entropy-weighted aggregation of naturally aligned logit prototypes. This novel design achieves robust MFL by jointly tackling heterogeneous feature spaces and collectively aggregating decisions.
Extensive experiments demonstrate that our method significantly outperforms state-of-the-art baselines under conditions of model heterogeneity and modality imbalance.

Forum: https://openreview.net/forum?id=aNgkaEagBG

---

## 469. Vision2Web: A Hierarchical Benchmark for Visual Website Development with Agent Verification

**Abstract:** Recent advances in large language models have improved the capabilities of coding agents, yet systematic evaluation of complex, end-to-end website development remains limited. To address this gap, we introduce Vision2Web, a hierarchical benchmark for visual website development, spanning from static UI-to-code generation, interactive multi-page frontend reproduction, to long-horizon full-stack website development. The benchmark is constructed from real-world websites and comprises a total of 193 tasks across 16 categories, with 918 prototype images and 1,255 test cases. To support flexible, thorough and reliable evaluation, we propose workflow-based agent verification paradigm based on two complementary components: a GUI agent verifier and a VLM-based judge. We evaluate multiple visual language models instantiated under different coding-agent frameworks, revealing substantial performance gaps at all task levels, with state-of-the-art models still struggling on full-stack development.

Forum: https://openreview.net/forum?id=lJpXXwhRRF

---

## 470. Detecting the Semantic Fixed Point: A Geometric Framework for Efficient Inference

**Abstract:** Each layer of a Transformer refines the hidden state toward a prediction, an iterative process resembling fixed-point iteration. Yet when should this iteration terminate? Existing early exit methods rely on output confidence as a proxy for internal convergence. We take a more direct approach by examining the geometry of the hidden state trajectory. We find that layer-wise updates exhibit a two-phase structure: large, volatile updates in early layers, followed by small, aligned updates as the model propagates an already-formed representation. The transition is remarkably sharp. This yields a simple criterion: exit when step size vanishes and direction stabilizes. We track the normalized update norm and cosine similarity between consecutive updates, exiting when both indicate convergence. The overhead is $O(d)$ per layer, independent of vocabulary size, requiring no learned components or architectural modifications. On LLaMA-2-7B and LLaMA-2-13B across question answering and commonsense reasoning tasks, this geometric criterion reduces FLOPs by 30--35\% while retaining over 98\% of full-depth accuracy.

Forum: https://openreview.net/forum?id=DACN5xM4h7

---

## 471. RAGEN-2: Reasoning Collapse in Agentic RL

**Abstract:** RL training of multi-turn LLM agents is unstable, and reasoning quality drives task performance. Entropy, the standard reasoning-stability monitor, only measures within-input diversity and misses whether reasoning depends on the input. We identify **template collapse**: stable entropy alongside input-agnostic boilerplate, invisible to entropy and existing metrics. We diagnose it via a **mutual-information (MI) proxy** that scores cross-input distinguishability online; across tasks, MI correlates with final performance far more strongly than entropy. We then explain collapse via a **signal-to-noise ratio (SNR)** mechanism: low within-input reward variance weakens task gradients, letting input-agnostic regularization dominate and erase cross-input differences. We mitigate this with **SNR-Aware Filtering**, prioritizing high-variance prompts each iteration. Across planning, math reasoning, web navigation, and code execution, the method consistently improves input dependence and task performance.

Forum: https://openreview.net/forum?id=01caH9oj7C

---

## 472. Latent Collaboration in Multi-Agent Systems

**Abstract:** Multi-agent systems (MAS) extend large language models (LLMs) from independent single-model reasoning to coordinative system-level intelligence. While existing LLM agents depend on text-based mediation for reasoning and communication, we take a step forward by enabling models to collaborate directly within the continuous latent space. We introduce LatentMAS, an end-to-end training-free framework that enables pure latent collaboration among LLM agents. In LatentMAS, each agent first performs auto-regressive latent thoughts generation through last-layer hidden embeddings instead of text. Then, a shared latent working memory preserves and transfers each agent's internal representations and latent thoughts, ensuring lossless information exchange without re-encoding. We provide detailed theoretical analyses showing that LatentMAS achieves higher expressiveness and lossless information preservation with lower overall complexity than standard text-based MAS. In addition, empirical evaluations across 9 comprehensive benchmarks spanning math and science reasoning, commonsense understanding, and code generation show that LatentMAS outperforms advanced single agents and text-based MAS baselines, achieving up to 14.6\% higher accuracy, reducing output token usage by 70.8\%-83.7\%, and providing 4$\times$-4.3$\times$ faster end-to-end inference.

Forum: https://openreview.net/forum?id=syG9I9ofd8

---

## 473. Learning in Structured Stackelberg Games

**Abstract:** We initiate the study of *structured Stackelberg games*, a novel form of strategic interaction between a leader and a follower where contextual information can be predictive of the follower's (unknown) type. Motivated by applications such as security games and AI safety, we show how this additional structure can help the leader learn a utility-maximizing policy in both the online and distributional settings. In the online setting, we first prove that standard learning-theoretic measures of complexity do not characterize the difficulty of the leader's learning task. We find that there exists a learning-theoretic measure of complexity, analogous to the Littlestone dimension in online classification, that *tightly* characterizes the leader's instance-optimal regret. We term this the *Stackelberg-Littlestone dimension*, and leverage it to provide a provably optimal online learning algorithm. In the distributional setting, we provide analogous results by showing that two new dimensions control the sample complexity upper- and lower-bound.

Forum: https://openreview.net/forum?id=XrKzHGg2jB

---

## 474. Even Faster Kernel Matrix Linear Algebra via Density Estimation

**Abstract:** This paper studies the use of *kernel density estimation* (KDE) for linear algebraic tasks involving the *kernel matrix* of a collection of $n$ data points in $\mathbb{R}^d$.
In particular, we improve upon the best existing algorithms for computing the following up to $(1+\varepsilon)$ relative error for a Gaussian kernel matrix and other kernels: matrix-vector products, matrix-matrix products, the spectral norm, and sum of all entries. The runtimes of our algorithms depend linearly on the dimension $d$, sub-quadratically in the number of points $n$, and polynomially on the target error $\varepsilon$. Importantly, the dependence on $n$ in each case is far lower when accessing the kernel matrix through KDE queries as opposed to reading individual entries. Our improvements over existing best algorithms (particularly those of [Backurs et al. ICML `21]) for these tasks reduce the polynomial dependence on $\varepsilon$, and additionally decrease the dependence on $n$ in the case of computing the sum of all entries of the kernel matrix. For example, we reduce the power of $1/\epsilon$ from $\approx 7.7$ to $\approx 3.2$ for a $1-\varepsilon$ relative error estimation of the spectral norm of a Gaussian kernel matrix. We complement our upper bounds with several lower bounds for related problems, which provide (conditional) quadratic time hardness results and additionally hint at the limits of KDE based approaches for the problems we study.

Forum: https://openreview.net/forum?id=ueNIrBXz7R

---

## 475. On the Expressive Power of Permutation-Equivariant Weight-Space Networks

**Abstract:** Weight-space learning studies neural architectures that operate directly on the parameters of other neural networks. Motivated by the growing availability of pretrained models, recent work has demonstrated the effectiveness of weight-space networks across a wide range of tasks.
SOTA weight-space networks rely on permutation-equivariant designs to improve generalization. However, this may negatively affect expressive power, warranting theoretical investigation.
Importantly, unlike other structured domains, weight-space learning targets maps operating on both weight and function spaces, making expressivity analysis particularly subtle.
While a few prior works provide partial expressivity results, a comprehensive characterization is still missing. In this work, we address this gap by developing a systematic theory for expressivity of weight-space networks.
We first prove that all prominent permutation-equivariant networks are equivalent in expressive power. We then establish universality in both weight- and function-space settings under mild, natural assumptions on the input weights, and characterize the edge-case regimes where universality no longer holds. 
Guided by our theoretical results, we show that slight modifications to existing weight-space models yield a 34\% improvement over prior SOTA, demonstrating the practical relevance of our framework.

Forum: https://openreview.net/forum?id=gQXkbQSZWS

---

## 476. Walrus: A Cross-domain Foundation Model for Continuum Dynamics

**Abstract:** Foundation models have transformed machine learning for language and vision, but achieving comparable impact in physical simulation remains a challenge. Data heterogeneity and unstable long-term dynamics inhibit learning from sufficiently diverse dynamics, while varying resolutions and dimensionalities challenge efficient training on modern hardware. Through empirical and theoretical analysis, we incorporate new approaches to mitigate these obstacles, including a harmonic-analysis–based stabilization method, load-balanced distributed 2D-3D training strategies, and compute-adaptive tokenization. Using these tools, we develop Walrus, a transformer-based foundation model developed primarily for fluid-like continuum dynamics. Walrus is pretrained on nineteen diverse scenarios spanning astrophysics, geoscience, rheology, plasma physics, acoustics, and classical fluids. Experiments show that Walrus outperforms prior foundation models on both short- and long-term prediction horizons on downstream tasks and across the breadth of pretraining data, while ablation studies confirm the value of our contributions to forecast stability, training throughput, and transfer performance over conventional approaches.

Forum: https://openreview.net/forum?id=kcyOjXoUZu

---

## 477. Neuro-evolutionary Continual Reinforcement Learning

**Abstract:** Deploying robots in open-ended real-world environments demands continual learning capabilities to adapt to an ever-expanding range of tasks. This requires retaining previously acquired skills without forgetting while effectively leveraging prior knowledge to learn new ones. Inspired by neuroscience, we propose **N**euro-**e**volutionary **C**ontinual **R**einforcement **L**earning (**Nevo-CRL**). Nevo-CRL maintains a fixed-capacity monolithic policy network, solving tasks by optimizing inter-layer connectivity and neuron parameters.
For each new task, Nevo-CRL constructs a mask population to selectively activate the outputs of each hidden layer, thereby forming a task-specific policy population. Upon completing each task, the best-performing mask is stored, and its activated neurons are frozen to prevent catastrophic forgetting. To facilitate knowledge transfer, Nevo-CRL reuses neurons from acquired skills based on semantic similarity between tasks, while dynamically allocating additional neurons for task-specific adaptation.
In the learning process, Nevo-CRL iteratively adjusts masks via importance-guided crossover to optimize the policy network connectivity. To improve neuron utilization, we prune low-activity connections to recycle neurons. Experiments demonstrate that Nevo-CRL achieves state-of-the-art performance among continual RL methods.
The code is available at [https://github.com/yeshenpy/Nevo-CRL](https://github.com/yeshenpy/Nevo-CRL).

Forum: https://openreview.net/forum?id=Hv0jK8xYcT

---

## 478. EgoTactile: Learning Grasp Pressure for Everyday Objects from Egocentric Video

**Abstract:** Estimating full-hand grasp pressure from egocentric video is critical for immersive VR and robotic manipulation, yet dense tactile sensing often relies on intrusive hardware. 
Existing vision-based methods predominantly rely on planar surfaces or fingertip contacts, failing to generalize to complex 3D object interactions. 
Therefore, we introduce EgoTactile, a benchmark pairing egocentric video with full-hand pressure supervision for diverse everyday objects, incorporating a bare-hand transfer subset to enable generalization to natural scenarios. 
Leveraging this benchmark, we first establish EgoPressureFormer as a discriminative baseline. Beyond this, to explicitly address the uncertainty in partial observations, we propose EgoPressureDiff, a conditional diffusion framework that adapts a large-scale pre-trained video diffusion backbone. By combining rich world knowledge priors with a Physically-Informed Feature Rectification layer to inject semantic constraints, our approach effectively hallucinates plausible contact patterns and resolves visual-physical ambiguities. 
Extensive experiments demonstrate that our method achieves superior performance on the benchmark and robust transferability to in-the-wild scenarios. Our project page is at https://egotactile.github.io/.

Forum: https://openreview.net/forum?id=xBkLTpOu2V

---

## 479. Listening Through the Noise: Cauchy-Driven Diffusion Bridges for Robust Gastrointestinal Auscultation and Clinical Benchmarking

**Abstract:** Gastrointestinal (GI) motility assessment via bowel sounds (BS) offers a non-invasive alternative to resource-intensive clinical standards. However, the diagnostic utility of BS is often compromised by its spectral overlap with non-stationary speech interference. While generative models have advanced signal restoration, traditional Gaussian-based diffusion frameworks struggle with the impulsive, heavy-tailed nature of real-world clinical noise. In this paper, we propose a novel Cauchy-driven Diffusion Bridge framework to isolate high-fidelity bowel sounds from complex interference. Our contributions are three-fold: (1) We introduce ClinBS, a large-scale clinical dataset (over 25 hours) containing rare pathological transients verified by experts; (2) We mathematically formulate a Cauchy bridge driver, deriving closed-form expressions for the score and density to better model heavy-tailed perturbations; and (3) We implement an efficient sampling procedure via Gaussian scale-mixture reparameterization. Extensive experiments show our framework achieves state-of-the-art performance, outperforming baselines by 13.4%–49.8% across core metrics and elevating abnormal BS recognition accuracy to 88.01%. These results demonstrate the system's potential for robust clinical GI monitoring and diagnosis.

Forum: https://openreview.net/forum?id=EYAfw6czcC

---

## 480. Multimodal Latent Language Modeling with Next-Token Diffusion

**Abstract:** Multimodal generative models require a unified approach to handle both discrete data (e.g., text and code) and continuous data (e.g., image, audio, video). In this work, we propose Latent Language Modeling (LatentLM), which seamlessly integrates continuous and discrete data using causal Transformers. Specifically, we employ a variational autoencoder (VAE) to represent continuous data as latent vectors and introduce next-token diffusion for autoregressive generation of these vectors. Additionally, we develop $\sigma$-VAE to address the challenges of variance collapse, which is crucial for autoregressive modeling. Extensive experiments demonstrate the effectiveness of LatentLM across various modalities. In image generation, LatentLM sis competitive with or outperforms DiT-style baselines under matched unified settings. When integrated into multimodal large language models, LatentLM provides a general-purpose interface that unifies multimodal generation and understanding. Experimental results show that LatentLM achieves favorable performance compared to Transfusion and vector quantized models in the setting of scaling up training tokens. In text-to-speech synthesis, LatentLM outperforms the state-of-the-art VALL-E 2 model in speaker similarity and robustness, while requiring 10 fewer decoding steps. The results establish LatentLM as a highly effective and scalable approach to advance large multimodal models.

Forum: https://openreview.net/forum?id=PnTXyTR2VG

---

## 481. LASER: Learning Active Sensing for Continuum Field Reconstruction

**Abstract:** High-fidelity measurements of continuum physical fields are essential for scientific discovery and engineering design but remain challenging under sparse and constrained sensing. Conventional reconstruction methods typically rely on fixed sensor layouts, which cannot adapt to evolving physical states. We propose LASER, a unified, closed-loop framework that formulates active sensing as a Partially Observable Markov Decision Process (POMDP). At its core, LASER employs a continuum field latent world model that captures the underlying physical dynamics and provides intrinsic reward feedback. This enables a reinforcement learning policy to simulate ''what-if'' sensing scenarios within a latent imagination space. By conditioning sensor movements on predicted latent states, LASER navigates toward potentially high-information regions beyond current observations. Our experiments demonstrate that LASER consistently outperforms static and offline-optimized strategies, achieving high-fidelity reconstruction under sparsity across diverse continuum fields.

Forum: https://openreview.net/forum?id=hnoSRifw1F

---

## 482. LaST$_{0}$: Latent Spatio-Temporal Chain-of-Thought for Robotic Vision-Language-Action Model

**Abstract:** Vision-Language-Action (VLA) models have recently shown strong generalization, with some approaches seeking to explicitly generate linguistic reasoning traces or predict future observations prior to execution. However, explicit reasoning typically incurs non-negligible inference latency, which constrains the temporal resolution required for robotic manipulation. Moreover, such reasoning is confined to the linguistic space, imposing a representational bottleneck that struggles to faithfully capture ineffable physical attributes. To mitigate these limitations, we propose LaST$_0$, a framework that enables efficient reasoning before acting through a Latent Spatio-Temporal Chain-of-Thought (CoT), capturing fine-grained physical and robotic dynamics that are often difficult to verbalize. Specifically, we introduce a token-efficient latent CoT space that models future visual dynamics, 3D structural information, and robot proprioceptive states, and further extends these representations across time to enable temporally consistent implicit reasoning trajectories. Furthermore, LaST$_0$ adopts a dual-system architecture implemented via a Mixture-of-Transformers design, where a reasoning expert conducts low-frequency latent inference and an acting expert generates high-frequency actions conditioned on robotics-oriented latent representations. To facilitate coordination, LaST$_0$ is trained with heterogeneous operation frequencies, enabling adaptive switching during deployment. Across 10 real-world tasks spanning tabletop, mobile, and dexterous hand manipulation, LaST$_0$ improves mean success rates by 13%, 14% and 14% over prior SOTA VLA methods, respectively.

Forum: https://openreview.net/forum?id=lwOoBzJykL

---

## 483. Beyond Global Alignment: Fine-Grained Motion-Language Retrieval via Pyramidal Shapley-Taylor Learning

**Abstract:** As a foundational task in human-centric cross-modal intelligence, motion-language retrieval aims to bridge the semantic gap between natural language and human motion, enabling intuitive motion analysis, yet existing approaches predominantly focus on aligning entire motion sequences with global textual representations. This global-centric paradigm overlooks fine-grained interactions between local motion segments and individual body joints and text tokens, inevitably leading to suboptimal retrieval performance. To address this limitation, we draw inspiration from the pyramidal process of human motion perception (from joint dynamics to segment coherence, and finally to holistic comprehension) and propose a novel Pyramidal Shapley-Taylor (PST) learning framework for fine-grained motion-language retrieval. Specifically, the framework decomposes human motion into temporal segments and spatial body joints, and learns cross-modal correspondences through progressive joint-wise and segment-wise alignment in a pyramidal fashion, effectively capturing both local semantic details and hierarchical structural relationships. Extensive experiments on multiple public benchmark datasets demonstrate that our approach significantly outperforms state-of-the-art methods, achieving precise alignment between motion segments and body joints and their corresponding text tokens.

Forum: https://openreview.net/forum?id=RuTNWE1wr7

---

## 484. DRPBench: Evaluating LLMs in Concurrent Code Comprehension via Fine-Grained Data Race Prediction

**Abstract:** Large Language Models (LLMs) have demonstrated sophisticated comprehension of sequential code, yet their capacity for reasoning about concurrent programs remains largely unquantified. We introduce DRPBench, a benchmark designed to evaluate the concurrent code comprehension of LLMs by measuring their data race prediction performance. To address the challenge of runtime non-determinism for evaluation on concurrent programs, we frame the evaluation as a fine-grained static prediction task using 1,003 programs from the SV-COMP suite, featuring 557 manually annotated data races with precise variable- and line-level granularity. Our evaluation of 15 state-of-the-art LLMs—spanning standard, reasoning, and agentic variants—reveals that DRPBench effectively differentiates concurrent code comprehension capabilities of LLMs. While the top-performing model (Gemini 3 with test-time reasoning) achieves an F1 score of 74.89%, most models struggle significantly (scoring less than 60%), with Llama 3 70B achieving only 8.80%. Beyond benchmarking, we characterize two primary failure modes: (1) shared-variable distraction, where multiple variable appearances degrade comprehension accuracy, and (2) synchronization-logic myopia, the inability to interpret non-standard synchronization implementations. Our findings provide a diagnostic roadmap for enhancing concurrent code comprehension of LLMs in future development.

Forum: https://openreview.net/forum?id=6249M0mKR2

---

## 485. FlexRank: Nested Low-Rank Knowledge Decomposition for Adaptive Model Deployment

**Abstract:** The growing scale of deep neural networks, encompassing large language models (LLMs) and vision transformers (ViTs), has made training from scratch prohibitively expensive and deployment increasingly costly.
These models are often used as computational monoliths with fixed cost, hindering adaptive deployment across different cost budgets.
We argue that nested components, ordered by importance, can be extracted from pretrained models and selectively activated within the available computational budget. To this end, our proposed FlexRank method leverages low-rank weight decomposition with nested, importance-based consolidation to extract submodels of increasing capabilities. Our approach enables a _``train-once, deploy-everywhere''_ paradigm offering a graceful trade-off between cost and performance without training from scratch for each budget - advancing practical deployment of large models.

Forum: https://openreview.net/forum?id=DK0kvnNelx

---

## 486. POET-X: Memory-efficient LLM Training by Scaling Orthogonal Transformation

**Abstract:** Efficient and stable training of large language models (LLMs) remains a core challenge in modern machine learning systems. To address this challenge, Reparameterized Orthogonal Equivalence Training (POET), a spectrum-preserving framework that optimizes each weight matrix through orthogonal equivalence transformation, has been proposed. Although POET provides strong training stability, its original implementation incurs high memory consumption and computational overhead due to intensive matrix multiplications. To overcome these limitations, we introduce POET-X, a scalable and memory-efficient variant that performs orthogonal equivalence transformations with significantly reduced computational cost. POET-X maintains the generalization and stability benefits of POET while achieving substantial improvements in throughput and memory efficiency. In our experiments, POET-X enables the pretraining of billion-parameter LLMs on a single Nvidia H100 GPU, and in contrast, standard optimizers such as AdamW run out of memory under the same settings.

Forum: https://openreview.net/forum?id=et8jpWLUuD

---

## 487. Advancing LLM Reasoning with Natural Language and Numerical Feedback

**Abstract:** Recent advances in reinforcement learning (RL) using numerical rewards have significantly enhanced the complex reasoning capabilities of large language models (LLMs). However, we identify three fundamental limitations of purely numerical feedback: performance plateaus, ineffective spontaneous self-reflection, and persistent failures. We show that plateaued RL models can successfully refine failed solutions when given natural language critiques. Motivated by this, we propose Critique-GRPO, an online RL framework that integrates both natural language and numerical feedback for policy optimization. This approach enables LLMs to learn simultaneously from initial responses and critique-guided refinements, effectively internalizing the exploration benefits of both stages. Extensive experiments show that Critique-GRPO outperforms all compared supervised and RL-based fine-tuning methods, achieving average Pass@1 improvements of approximately +15.0-21.6% on various Qwen models and +7.3% on Llama-3.2-3B-Instruct across eight challenging reasoning tasks. Notably, Critique-GRPO facilitates effective self-improvement through self-critiquing, achieving substantial gains over GRPO, e.g., a +16.7% Pass@1 improvement on AIME 2024. The code and models are released at: https://github.com/zhangxy-2019/critique-GRPO

Forum: https://openreview.net/forum?id=gz7hVnrRWq

---

## 488. Unsupervised Partner Design Enables Robust Ad-hoc Teamwork

**Abstract:** We introduce Unsupervised Partner Design (UPD), a population-free multi-agent reinforcement learning method for robust ad-hoc teamwork. 
UPD generates training partners on-the-fly and selects them adaptively based on a learnability criterion, removing the need for pre-trained partner populations or manual parameter tuning.
We show that this simple mechanism enables effective partner diversity and can be extended to joint partner-environment selection when a procedural level generator is available. 
Across Level-Based Foraging, Overcooked-AI, and the Overcooked Generalisation Challenge, UPD consistently achieves strong performance compared to both population-based and population-free baselines. 
In a human-AI user study, agents trained with UPD achieve higher returns and are rated as more adaptive, more human-like, and less frustrating than all evaluated baseline methods.

Forum: https://openreview.net/forum?id=0xtMUL0eiF

---

## 489. GR-LoRA: Gradient-Recycling Low-Rank Adaptation for Class-Incremental Learning

**Abstract:** Pre-trained models with parameter-efficient fine-tuning have shown strong effectiveness in Class-Incremental Learning (CIL), which seeks to balance model plasticity and stability. In this context, orthogonality constraints can significantly enhance model stability, yet their reliance on subspace inevitably compromises model plasticity over long tasks. To address this, we propose Gradient-Recycling Low-Rank Adaptation (GR-LoRA), which reconciles stability and plasticity by recycling the gradients discarded in orthogonal projection. Specifically, GR-LoRA recycles post-decomposition non-orthogonal gradient components into task-specific lightweight modules and selects optimal module via entropy to improve plasticity, while incorporating local and global mismatch suppression to preserve stability by synthesizing out-of-distribution representations across all tasks. Theoretical analysis confirms that this recycling strategy preserves stability and improves plasticity. Experimental results from multiple CIL benchmarks verify the effectiveness and general applicability of GR-LoRA.

Forum: https://openreview.net/forum?id=MhMoUuoA1g

---

## 490. Mitigating Reward Hacking in RLHF via Bayesian Non-negative Reward Modeling

**Abstract:** Reward models learned from human preferences are central to aligning large language models (LLMs) via reinforcement learning from human feedback, yet they are often vulnerable to reward hacking due to noisy annotations and systematic biases such as response length or style. We propose Bayesian Non-Negative Reward Model (BNRM), a principled reward modeling framework that integrates non-negative factor analysis into Bradley–Terry (BT) preference model.BNRM represents rewards through a sparse, non-negative latent factor generative process that operates at two complementary levels: instance-specific latent variables induce disentangled reward representations, while sparsity over global latent factors acts as an implicit debiasing mechanism that suppresses spurious correlations. Together, this disentanglement-then-debiasing structure enables robust uncertainty-aware reward learning. To scale BNRM to modern LLMs, we develop an amortized variational inference network conditioned on deep model representations, allowing efficient end-to-end training. Extensive empirical results demonstrate that BNRM substantially mitigates reward over-optimization, improves robustness under distribution shifts, and yields more interpretable reward decompositions than strong baselines.

Forum: https://openreview.net/forum?id=DfhMMHXDuu

---

## 491. Just Noticeable Difference Modeling for Deep Visual Features

**Abstract:** Deep visual features are increasingly used as the interface in vision systems, motivating the need to describe feature characteristics and control feature quality for machine perception. Just noticeable difference (JND) characterizes the maximum imperceptible distortion for images under human or machine vision. Extending it to deep visual features naturally meets the above demand by providing a task-aligned tolerance boundary in feature space, offering a practical reference for controlling feature quality under constrained resources. We propose FeatJND, a task-aligned JND formulation that predicts the maximum tolerable per-feature perturbation map while preserving downstream task performance. We propose a FeatJND estimator at standardized split points and validate it across image classification, detection, and instance segmentation. Under matched distortion strength, FeatJND-based distortions consistently preserve higher task performance than unstructured Gaussian perturbations, and attribution visualizations suggest FeatJND can suppress non-critical feature regions. As an application, we further apply FeatJND to token-wise dynamic quantization and show that FeatJND-guided step-size allocation yields clear gains over random step-size permutation and global uniform step size under the same noise budget. The source code is available at https://github.com/ruizhao26/FeatJND.

Forum: https://openreview.net/forum?id=RxjGbyLFT1

---

## 492. TideGS: Scalable Training of Over One Billion 3D Gaussian Splatting Primitives via Out-of-Core Optimization

**Abstract:** Training 3D Gaussian Splatting (3DGS) at billion-primitive scale is fundamentally memory-bound: each Gaussian primitive carries a large attribute vector, and the aggregate parameter table quickly exceeds GPU capacity, limiting prior systems to tens of millions of Gaussians on commodity single-GPU hardware. We observe that 3DGS training is inherently sparse and trajectory-conditioned: each iteration activates only the Gaussians visible from the current camera batch, so GPU memory can serve as a working-set cache rather than a persistent parameter store. Building on this insight, we introduce TideGS, an out-of-core training framework that manages parameters across an SSD-CPU-GPU hierarchy via three synergistic techniques: block-virtualized geometry for SSD-aligned spatial locality, a hierarchical asynchronous pipeline to overlap I/O with computation, and trajectory-adaptive differential streaming that transfers only incremental working-set deltas between iterations. Experiments show that TideGS enables training with over one billion Gaussians on a single 24-GB GPU while achieving the best reconstruction quality among evaluated single-GPU baselines on large-scale scenes, scaling beyond prior out-of-core baselines (e.g., ~100M Gaussians) and standard in-memory training (e.g., ~11M Gaussians). Project page: https://sponge-lab.github.io/TideGS/

Forum: https://openreview.net/forum?id=K8UWWJUNYf

---

## 493. Asymmetric Multi-View Clustering with Hyperbolic Uncertainty Modeling

**Abstract:** Deep Multi-View Clustering (MVC) aims to learn a unified semantic representation from diverse data sources without supervision. However, current approaches relying on flat Euclidean embeddings often fail to model data uncertainty, resulting in rigid alignment where high-quality views are forced to drift toward corrupted ones. To address these challenges, we propose the Hyperbolic Asymmetric Multi-view Clustering (HAMC) framework. HAMC maps view-specific features into the Poincaré ball and uses radial geometry as a confidence proxy, encouraging confident representations to occupy larger radial distances while allowing ambiguous or noisy samples to remain closer to the origin. To mitigate noise, we introduce an asymmetric view alignment mechanism, enabling reliable views to unidirectionally guide unreliable ones. Furthermore, a consensus-aware cluster learning strategy is designed to construct robust global pseudo-labels via a confidence-based screening scheme, refining the cluster structure. Extensive experiments against 13 baselines demonstrate that HAMC achieves state-of-the-art performance.

Forum: https://openreview.net/forum?id=OTKdKRLXf7

---

## 494. Neural Thickets: Diverse Task Experts Are Dense Around Pretrained Weights

**Abstract:** Pretraining produces a learned parameter vector that is typically treated as a starting point for further iterative adaptation. In this work, we instead view the outcome of pretraining as a distribution over parameter vectors, whose support already contains task-specific experts. We show that in smaller or insufficiently trained models such expert solutions occupy a negligible fraction of the volume of this distribution, making their discovery reliant on structured optimization methods such as gradient descent. In contrast, in large, well-pretrained models the density of task-experts increases dramatically, so that diverse specialists populate a substantial fraction of the neighborhood around the pretrained weights. Motivated by this perspective, we explore a simple, fully parallel post-training method that samples $N$ parameter vectors at random, selects the top $K$, and ensembles them via majority vote to combine complementary expertise. Despite its simplicity, this approach is competitive with standard post-training methods such as PPO, GRPO, and ES for contemporary large-scale models.

Forum: https://openreview.net/forum?id=92oF5bU4cU

---

## 495. FLIP2: Expanding Protein Fitness Landscape Benchmarks for Real-World Machine Learning Applications

**Abstract:** Machine learning methods that predict protein fitness from sequence remain sensitive to changes in data distributions, limiting generalization across common conditions encountered in protein engineering. Practically, protein engineers are thus left wondering about the effective utility of ML tools. 
The FLIP benchmark established protocols for testing generalization under some domain shifts, but it was limited to measurements of stability, binding, and viral capsid viability.
We introduce FLIP2, a protein fitness benchmark spanning seven new datasets, including enzymes, protein-protein interactions, and light-sensitive proteins, as well as splits that measure generalization relevant to real-world protein engineering campaigns. 
Evaluating a suite of benchmark models across these datasets and suites reveals that
simpler models often matched or outperformed fine-tuned protein language models on FLIP2, challenging the utility of existing transfer learning techniques. Provenance for all datasets has been recorded and we redistribute all data CC-BY 4.0 to facilitate continued progress.

Forum: https://openreview.net/forum?id=Ue8aOgz9T9

---

## 496. OMAC: A Holistic Optimization Framework for LLM-Based Multi-Agent Collaboration

**Abstract:** Agents powered by advanced large language models (LLMs) have demonstrated impressive capabilities across diverse complex applications.  Recently, Multi-Agent Systems (MAS), wherein multiple agents collaborate and communicate with each other, have exhibited enhanced capabilities in complex tasks, such as high-quality code generation and arithmetic reasoning. However, the development of such systems often relies on handcrafted methods, and the literature on systematic design and optimization of LLM-based MAS remains limited.
In this work, we introduce OMAC, a general framework designed for holistic optimization of LLM-based MAS. Specifically, we identify five key optimization dimensions for MAS, encompassing both agent functionality and collaboration structure. Building upon these dimensions, we first propose a general algorithm, utilizing two actors termed the Semantic Initializer and the Contrastive Comparator, to optimize any single dimension. Then, we present an algorithm for joint optimization across multiple dimensions. Extensive experiments demonstrate the superior performance of OMAC on diverse tasks against recent approaches. Codes are available at: https://github.com/xiwenchao/OMAC.

Forum: https://openreview.net/forum?id=6C4YQcq8YX

---

## 497. Bad Seeing or Bad Thinking? Rewarding Perception for Multimodal Reasoning

**Abstract:** Achieving robust perception-reasoning synergy is a central goal for advanced Vision-Language Models (VLMs). Recent advancements have pursued this goal via architectural designs or agentic workflows. However, these approaches are often limited by static textual reasoning or complicated by the significant compute and engineering burden of external agentic complexity. Worse, this heavy investment does not yield proportional gains, often witnessing a "seesaw effect" on perception and reasoning. This motivates a fundamental rethinking of the true bottleneck. In this paper, we argue that the root cause of this trade-off is an ambiguity in modality credit assignment: when a VLM fails, is it due to flawed perception ("bad seeing") or flawed logic ("bad thinking")?
To resolve this, we introduce a reinforcement learning framework that improves perception-reasoning synergy by reliably rewarding the perception fidelity. We explicitly decompose the generation process into interleaved perception and reasoning steps. This decoupling enables targeted supervision on perception. Crucially, we introduce Perception Verification (PV), leveraging a "blindfolded reasoning" proxy to reward perceptual fidelity independently of reasoning outcomes. Furthermore, to scale training across free-form VL tasks, we propose Structured Verbal Verification, which replaces high-variance LLM judging with structured algorithmic execution. These techniques are integrated into a Modality-Aware Credit Assignment (MoCA) mechanism, which routes rewards to the specific source of error -- either bad seeing or bad thinking -- enabling a single VLM to achieve simultaneous performance gains across a wide task spectrum.

Forum: https://openreview.net/forum?id=duzdftKtUA

---

## 498. Rethinking LLM Ensembling from the Perspective of Mixture Models

**Abstract:** Model ensembling is a well-established technique for improving the performance of machine learning models. Conventionally, this involves averaging the output distributions of multiple models and selecting the most probable label. This idea has been naturally extended to large language models (LLMs), yielding improved performance but incurring substantial computational cost. This inefficiency stems from directly applying conventional ensemble implementation to LLMs, which require a separate forward pass for each model to explicitly compute the ensemble distribution. In this paper, we propose the Mixture-model-like Ensemble (ME). By reinterpreting the ensemble as a mixture model, ME stochastically selects a single model at each step to generate the next token, thereby avoiding the need to explicitly compute the full ensemble distribution. ME is mathematically equivalent to sampling from the ensemble distribution, but requires invoking only one model, making it 1.78×-2.68× faster than conventional ensembling. Furthermore, this perspective connects LLM ensembling and token-level routing methods, suggesting that LLM ensembling is a special case of routing methods. Our findings open new avenues for efficient LLM ensembling and motivate further exploration of token-level routing strategies for LLMs. Our code is available at https://github.com/Kamichanw/Mixture-model-like-Ensemble.

Forum: https://openreview.net/forum?id=YVk8EDxBWx

---

## 499. MiniAppBench: Evaluating the Shift from Text to Interactive HTML Responses in LLM-Powered Assistants

**Abstract:** With the rapid advancement of Large Language Models (LLMs) in code generation, human-AI interaction is evolving from static text responses to dynamic, interactive HTML-based applications, which we term **MiniApps**. These applications require models to not only render visual interfaces but also construct customized interaction logic that adheres to real-world principles. However, existing benchmarks primarily focus on algorithmic correctness or static layout reconstruction, failing to capture the capabilities required for this new paradigm. To address this gap, we introduce **MiniAppBench**, the first comprehensive benchmark designed to evaluate principle-driven, interactive application generation. Sourced from a real-world application with **10M+** generations, MiniAppBench distills 500 tasks across six domains (e.g., Games, Science, and Tools). Furthermore, to tackle the challenge of evaluating open-ended interactions where no single ground truth exists, we propose **MiniAppEval**, an agentic evaluation framework. Leveraging browser automation, it performs human-like exploratory testing to systematically assess applications across three dimensions: Intention, Static, and Dynamic. Our experiments reveal that current LLMs still face significant challenges in generating high-quality MiniApps, while MiniAppEval demonstrates high alignment with human judgment, establishing a reliable standard for future research. Our homepage is available in https://miniappbench.github.io.

Forum: https://openreview.net/forum?id=pwbLmew1aq

---

## 500. GFD-EMVC: Evolutionary Multi-View Classification with Label Noise via Gradient and Feature Dual-Perception

**Abstract:** This paper studies a fundamental yet often overlooked premise in evolutionary multi-view classification (EMVC): the impact of label noise on EMVC, such as distorting fitness landscapes shaped by individual fitness values (e.g., test accuracy).
Traditional EMVC assumes training labels are noise-free, yet this often fails in practice.
As a result, label noise introduces harmful supervision during the training phase, resulting in distorted fitness landscapes and the emergence of fitness evaluation bias (FEB). This bias misguides the evolutionary trajectory, causing the search process to stagnate in local optima. 
Given that label noise largely stems from the mislabeling of samples near their decision boundaries by human annotators, we thus compared the decision boundaries of human annotators and models, and found discrepancies between the two. Based on this observation, we propose a simple yet effective ``detect-then-calibrate" data purification framework that leverages outlier analysis in the gradient space (i.e., treating outliers as noisy samples) and prototype calibration in the feature space (i.e., utilizing feature prototypes of noise-free samples to correct the labels of noisy samples).
Experimental results demonstrate that this strategy can effectively purify the data and alleviate FEB; moreover, it can improve the performance of various multi-view learning paradigms in label noise scenarios.
(https://github.com/LiShuailzn/ICML-2026-GFD-EMVC)

Forum: https://openreview.net/forum?id=aWUlNpaZs1

---

## 501. Geometric Flow Grounding: A Unified Manifold Decoupling Framework for Dynamics Discovery and Verification

**Abstract:** Modeling complex dynamics from observational data is fundamental to scientific discovery and artificial intelligence. However, existing approaches are often plagued by the entanglement of static state representations and instantaneous motion, leading to accumulated errors and off-manifold hallucinations where predicted trajectories violate intrinsic geometric constraints. To address this, we propose Geometric Flow Grounding, a unified framework that enforces dynamic evolution strictly along the tangent bundle of the learned data manifold via a differentiable Neural Tangent Projection Layer. By geometrically decoupling state representation from tangential dynamics, our method generalizes across diverse data regimes. In scientific discovery, GFG reduces numerical aliasing and improves long-horizon stability in sparse dynamical systems, while recovering interpretable gene regulatory motifs from single-cell data. For trustworthy AI, the projection residual provides a zero-shot metric for deepfake video detection by revealing inconsistencies with the implicit flow of pre-trained world models. Our results establish manifold-constrained projection as a universal operator for both discovering natural laws and verifying synthetic content. Code will be available at \url{https://github.com/yuchang97/GFG-public}

Forum: https://openreview.net/forum?id=IYSBJvVRTx

---

## 502. ScaleMoE: Mixture-of-Experts for Scalable Continuous Control in Actor-Critic Reinforcement Learning

**Abstract:** Scaling network remains a bottleneck in deep reinforcement learning (RL): simply enlarging actor–critic networks destabilizes training and soon saturates performance. Although recent monolithic architectures such as SimBa and BRC have shown that carefully designed inductive biases can enable positive scaling up to a certain size, their improvements plateau soon as model parameters grow further. This work introduces ScaleMoE, a scalable RL architecture that integrates Mixture-of-Experts (MoE) modules into both the actor and critic of modern continuous control algorithms.  Two complementary gating schemes are studied: output-level aggregation of per-expert policies and Q-functions, and feature-level fusion of expert representations before a shared head. We instantiate ScaleMoE on two representative monolithic RL baselines: the single-task method SimBa and the multi-task method BRC. Experiments across the DeepMind Control Suite, MetaWorld, and HumanoidBench show that progressively increasing the number of experts (up to 64) yields substantial improvements in returns, significantly outperforming monolithic networks of comparable or even greater parameter counts. Results demonstrate that ScaleMoE provides an   efficient and effective scaling axis for deep RL in continuous control.

Forum: https://openreview.net/forum?id=oVS7o5jKJF

---

## 503. PACT: Self-Evolving Physical Safety Alignment for Diffusion Policies in Embodied Manipulation

**Abstract:** Diffusion policies have achieved remarkable success in robotic manipulation, yet they often fail to satisfy strict physical constraints required for safe deployment.  Existing approaches impose safety either prematurely during training or reactively via external guardrails at test time, limiting policy expressivity and overall scalability. We propose Physical safety Alignment for Constrained Trajectories (PACT), a self-evolving post-training framework that projects pretrained diffusion policies onto constraint-feasible regions without accessing demonstration data or task rewards. PACT distills constraint gradients into the diffusion model through a reverse-KL objective with dense supervision across timesteps. It incorporates a curriculum that progressively tightens constraints while maintaining theoretically bounded policy shift and monotone improvement, mitigating the safety-performance trade-off from catastrophic forgetting. On simulated and real-world embodied manipulation benchmarks, PACT significantly reduces safety violations by 31.0% on average while improving task success by 30.7%.

Forum: https://openreview.net/forum?id=ePFvXPdvhM

---

## 504. ReViT: Rotational-equivariant Vision Transformers for Neural PDE Solvers

**Abstract:** Physics obeys strict symmetries like rotational equivariance. However, the standard Transformer architectures widely used in physics foundation models do not enforce these constraints by construction. We introduce ReViT, a rotationally equivariant Vision Transformer framework for neural PDE solvers operating on grid-based physical fields that achieves exact equivariance for the discrete groups $C_4$ (2D) and the chiral octahedral group $O$ (3D), with bounded approximate $\mathrm{SO}(d)$ equivariance for continuous rotations. ReViT maps scalar and vector inputs into locally invariant representations derived from physics-based canonical bases, enabling the use of standard self-attention without symmetry violations. Built on a hierarchical Swin-style backbone with a precomputed reference basis pyramid, ReViT preserves equivariance across multi-scale operations. 
We evaluate ReViT on a wide range of 2D and 3D PDE benchmarks, such as Magnetohydrodynamics and Turbulent Channel Flows, demonstrating significant gains over state-of-the-art baselines. ReViT exhibits strong generalization, and reduces MSE by up to 65\% compared with the best-performing alternatives.

Forum: https://openreview.net/forum?id=fy2IXBBThK

---

## 505. Spectral-Informed Neural Networks Outperform Spectral methods in High-dimensional PDEs

**Abstract:** For low-dimensional problems ($d\leq3$), spectral methods can achieve exceptionally high accuracy. For middle-dimensional problems ($4 \leq d \lesssim 10$), spectral methods remain feasible through specific techniques such as sparse grids or hyperbolic cross. However, for high-dimensional problems ($d\gg 10$), spectral methods suffer frome the curse of dimensionality. Physics-informed neural networks (PINNs) have emerged as a promising approach to overcome this challenge, offering scalability to high dimensions, but often suffer from limited accuracy and efficiency. Recently proposed spectral-informed neural networks (SINNs) combine spectral methods with PINNs, operating directly in the spectral domain to avoid spatial derivative computations and to reduce memory consumption. In this work, we introduce Modified SINNs, which integrate coefficient decay scaling and basis embeddings motivated by harmonic analysis to enhance accuracy in high-dimensional problems and enable accurate approximation of unknown spectral coefficients. Numerical experiments on steady and time-dependent partial differential equations demonstrate that Modified SINNs outperform sparse grid spectral methods on middle-dimensional problems with incomplete spectral information and achieve superior accuracy compared to PINNs on high-dimensional problems.

Forum: https://openreview.net/forum?id=KAHCMPsPeI

---

## 506. Thinking in Flow: A Dissipative Stabilization Operator for Robust Autoregressive Reasoning

**Abstract:** Chain-of-Thought (CoT) prompting enables multi-step reasoning in large language models, yet long-horizon generation remains brittle under distribution shift and context interference: irrelevant cues persist, small deviations compound into inference drift, and late-stage corrections can destabilize the trajectory. We recast autoregressive decoding as a perturbed long-horizon dynamical system and introduce an *inference-time stabilization operator* that targets *trajectory-level* reliability rather than token-level fluency. Specifically, we propose *ODE-guided language models*, which augment a base Transformer with a persistent continuous-time *thought state* whose dynamics are explicitly designed to be dissipative, enabling stable evidence accumulation with controlled forgetting. Instantiating this framework, *Thinking in Flow* (TiF) equips the model with a lightweight Neural ODE controller and injects its output through post-norm residual updates to achieve numerically stable, low-intrusion steering. A demand--supply (uncertainty--capacity) gate determines *when* intervention is warranted, while a direction gate determines *how* to steer in representation space, yielding selective, do-no-harm corrections instead of persistent bias. We establish well-posedness, dissipativity, and incremental stability of the controlled thought dynamics, implying bounded interventions over arbitrarily long contexts, and empirically demonstrate improved robustness to distractions and semantic perturbations, while matching or improving accuracy on mathematical reasoning benchmarks across both the Llama and Qwen model families; we further observe gains on non-mathematical BBH reasoning tasks when training TiF on Llama.

Forum: https://openreview.net/forum?id=9IQpUEKOGM

---

## 507. Controlled LLM Training on Spectral Sphere

**Abstract:** Scaling large models requires optimization strategies that ensure rapid convergence grounded in stability. Maximal Update Parametrization ($\boldsymbol{\mu}$P) provides a theoretical safeguard for width-invariant $\Theta(1)$ activation control, whereas emerging optimizers like Muon are only "half-aligned" with these constraints: they control updates but allow weights to drift. To address this limitation, we introduce the **Spectral Sphere Optimizer (SSO)**, which enforces strict module-wise spectral constraints on both weights and their updates. By deriving the steepest descent direction on the spectral sphere, SSO realizes a fully $\boldsymbol{\mu}$P-aligned optimization process. To enable large‑scale training, we implement SSO as an efficient parallel algorithm within Megatron. Through extensive pretraining on diverse architectures, including Dense 1.7B, MoE 8B-A1B, and 200-layer DeepNet models, SSO consistently outperforms AdamW and Muon. Furthermore, we observe significant practical stability benefits, including improved MoE router load balancing, suppressed outliers, and strictly bounded activations.

Forum: https://openreview.net/forum?id=5kTn1c3vtt

---

## 508. Multimodal Nested Learning for Decoupled and Coordinated Optimization

**Abstract:** Multimodal learning aims to integrate multi-sensor data to exploit their complementary information, embracing a more comprehensive real-world perception and understanding. However, heterogeneous discrepancies across modalities consistently trigger imbalanced multimodal optimization, restricting the joint learning performance. Although existing methods mitigate this issue through optimization modulation and conflict alleviation, they still suffer from entangled optimization and uniform learning pace in conventional monolithic frameworks, limiting the effectiveness of multimodal learning. To address this issue, we propose a novel Multimodal Nested Learning Framework (MoNet), which reformulates the monolithic framework into nested sub-processes, decoupling and coordinating multimodal learning. To achieve this, we present a Decoupled Multimodal Stable Memory block (DMSM) as the outermost nested level, which decouples multimodal learning into independent optimization streams for semantic exploitation across modalities. Additionally, we develop an Adaptive Multimodal Coordinated Fusion block (AMCF), which constitutes the inner nested level. It attempts to coordinate multimodal information integration across multi-timescale nested memories, balancing multimodal fusion. Extensive experimental results on eight datasets across three tasks demonstrate the superiority of MoNet. Code is available at https://github.com/Yangl1nFeng/MoNet.

Forum: https://openreview.net/forum?id=8XOncDWc5j

---

## 509. Towards Long-Horizon Interpretability: Efficient and Faithful Multi-Token Attribution for Reasoning LLMs

**Abstract:** Token attribution methods provide intuitive explanations for language model outputs by identifying causally important input tokens. However, as modern LLMs increasingly rely on extended reasoning chains, existing schemes face two critical challenges: (1) efficiency bottleneck, where attributing a target span of $M$ tokens within a context of length $N$ requires $\mathcal{O}(M \cdot N)$ operations, making long-context attribution prohibitively slow; and (2) faithfulness drop, where intermediate reasoning tokens absorb attribution mass, preventing importance from propagating back to the original input. To address these, we introduce FlashTrace, an efficient multi-token attribution method that employs span-wise aggregation to compute attribution over multi-token targets in a single pass, while maintaining faithfulness. Moreover, we design a recursive attribution mechanism that traces importance through intermediate reasoning chains back to source inputs. Extensive experiments on long-context retrieval (RULER) and multi-step reasoning (MATH, MorehopQA) tasks demonstrate that FlashTrace achieves over $130\times$ speedup over existing baselines while maintaining superior faithfulness. We further analyze the dynamics of recursive attribution, showing that even a single recursive hop improves faithfulness by tracing importance through the reasoning chain.

Forum: https://openreview.net/forum?id=KY5Q9V9F5C

---

## 510. Efficiently Training Time-to-First-Spike Spiking Neural Networks from Scratch

**Abstract:** Spiking Neural Networks (SNNs), with their event-driven and biologically inspired mechanisms, are well-suited for energy-efficient neuromorphic hardware. Neural coding, which is critical to SNNs, determines how information is represented via spikes. While Time-to-First-Spike (TTFS) coding uses a single spike per neuron to offer extreme sparsity and energy efficiency, it often suffers from unstable training and low accuracy due to its sparse firing.
To address these challenges, we propose a training framework that incorporates parameter initialization, training normalization, a temporal output decoder, and a re-evaluation of the pooling layer. 
The proposed parameter initialization and training normalization mitigate signal diminishing and gradient vanishing, which helps stabilize training. Our output decoder aggregates temporal spikes to encourage earlier firing, thereby reducing latency.
The re-evaluation of the pooling layer demonstrates that max-pooling violates single-spike constraints, which should be avoided, whereas average-pooling preserves them.
Experiments show that our framework stabilizes and accelerates training, reduces latency, and achieves state-of-the-art accuracy for step-by-step TTFS SNNs on MNIST ($99.48\%$), Fashion-MNIST ($92.90\%$), CIFAR10 ($90.56\%$), CIFAR100 ($70.27\%$) and DVS Gesture ($95.83\%$).

Forum: https://openreview.net/forum?id=3EcT46wsdc

---

## 511. Markov Chain Monte Carlo without Evaluating the Target: an Auxiliary Variable Approach

**Abstract:** In sampling tasks, it is common for target distributions to be known up to a normalizing constant. However, in many situations, even evaluating the unnormalized distribution can be costly or infeasible. This issue arises in scenarios such as sampling from the Bayesian posterior for tall datasets and the `doubly-intractable' distributions. In this paper, we begin by observing that seemingly different Markov chain Monte Carlo (MCMC) algorithms, such as the exchange algorithm, PoissonMH, and TunaMH, can be unified under a simple common procedure. We then extend this procedure into a novel framework that allows the use of auxiliary variables in both the proposal and the acceptance--rejection step. Several new MCMC algorithms emerge from this framework that uses estimated gradients to guide the proposal moves. They have demonstrated significantly better performance than existing methods on both synthetic and real datasets. We also develop theory for the new framework and use it to simplify and extend results for existing algorithms. The code to reproduce the experimental results can be found at https://github.com/ywwes26/Auxiliary-MCMC.

Forum: https://openreview.net/forum?id=dDkl5ZcyTl

---

## 512. Alignment-Sensitive Minimax Rates for Spectral Algorithms with Learned Kernels

**Abstract:** We study spectral algorithms in settings where kernels may arise from a learning procedure, and ask how the resulting spectral order affects risk. We introduce the effective span dimension (ESD), an alignment-sensitive complexity measure that depends jointly on the target signal, the spectral order induced by the kernel, and the noise level $\sigma^2$. The ESD is defined without requiring eigen-decay conditions or source conditions, and it captures how much of the target signal lies in the leading spectral span. We prove that, for sequence models whose ESD is at most $K$, the minimax excess risk scales as $\sigma^2 K$, and we extend the framework to linear models and RKHS regression. Furthermore, we analyze over-parameterized gradient flow in a fixed-eigenbasis spectral learning model and prove that it can reduce the ESD under certain conditions. Together with numerical experiments, these results connect adaptive feature learning with reductions in ESD and offer a novel perspective on generalization beyond traditional fixed-kernel theories.

Forum: https://openreview.net/forum?id=4HrWo5x7YF

---

## 513. A Random Matrix Perspective on the Consistency of Diffusion Models

**Abstract:** Diffusion models trained on different, non-overlapping subsets of a dataset often produce strikingly similar outputs when given the same noise seed. We trace this consistency to a simple linear effect: the shared Gaussian statistics across splits already predict much of the generated images. To formalize this, we develop a random matrix theory (RMT) framework that quantifies how finite datasets shape the expectation and variance of the learned denoiser and sampling map in the linear setting. For expectations, sampling variability acts as a renormalization of the noise level through a self-consistent relation $\sigma^2\to\kappa(\sigma^2)$, explaining why limited data overshrink low-variance directions and pull samples toward the dataset mean. For fluctuations, our variance formulas reveal three key factors behind cross-split disagreement: \textit{anisotropy} across eigenmodes, \textit{inhomogeneity} across inputs, and overall scaling with dataset size. Extending deterministic-equivalence tools to fractional matrix powers further allows us to analyze entire sampling trajectories. The theory sharply predicts the behavior of linear diffusion models, and we validate its predictions on UNet and DiT architectures in their non-memorization regime, identifying where and how samples deviates across training data split. This provides a principled baseline for reproducibility in diffusion training, linking spectral properties of data to the stability of generative outputs.

Forum: https://openreview.net/forum?id=iPjuUQbkfl

---

## 514. FlashSinkhorn: IO-Aware Entropic Optimal Transport on GPU

**Abstract:** Entropic optimal transport (EOT) via Sinkhorn iterations is widely used in modern machine learning, yet GPU solvers remain inefficient at scale. Tensorized implementations suffer quadratic HBM traffic from dense $n\times m$ interactions, while existing online backends avoid storing dense matrices but still rely on generic tiled map-reduce reduction kernels with limited fusion. We present **FlashSinkhorn**, an IO-aware EOT solver for squared Euclidean cost that rewrites stabilized log-domain Sinkhorn updates as row-wise LogSumExp reductions of biased dot-product scores, the same normalization as transformer attention. This enables FlashAttention-style fusion and tiling: fused Triton kernels stream tiles through on-chip SRAM and update dual potentials in a single pass, substantially reducing HBM IO per iteration while retaining linear-memory operations. We further provide streaming kernels for transport application, enabling scalable first- and second-order optimization. On A100 GPUs, FlashSinkhorn achieves up to $32\times$ forward-pass and $161\times$ end-to-end speedups over state-of-the-art online baselines on point-cloud OT, improves scalability on OT-based downstream tasks. For reproducibility, we release an open-source implementation at https://github.com/ot-triton-lab/flash-sinkhorn.

Forum: https://openreview.net/forum?id=VzIA4MASxK

---

## 515. The Flexibility Trap: Rethinking the Value of Arbitrary Order in Diffusion Language Models

**Abstract:** Diffusion Large Language Models (dLLMs) break the rigid left-to-right constraint of traditional LLMs, enabling token generation in arbitrary orders. Intuitively, this flexibility implies a solution space that strictly supersets the fixed autoregressive trajectory, theoretically unlocking superior reasoning potential. However, in this paper, we find that for general reasoning tasks (e.g., mathematics and coding), arbitrary order generation may in fact limit the reasoning potential of dLLMs.
We observe that dLLMs tend to exploit this order flexibility to bypass high-uncertainty tokens that are crucial for exploration, which can lead to a premature collapse of solution coverage.
This observation motivates a rethink of RL approaches for dLLMs, where considerable complexities, such as handling combinatorial trajectories and intractable likelihoods, are often devoted to preserving this flexibility.
We show that effective reasoning can be elicited by simply forgoing arbitrary order and applying standard Group Relative Policy Optimization (GRPO) instead.
Our approach, JustGRPO, is minimalist yet surprisingly effective (e.g., 89.1\% accuracy on GSM8K) while fully retaining the parallel decoding ability of dLLMs. Code: https://github.com/LeapLabTHU/JustGRPO.

Forum: https://openreview.net/forum?id=kpgURPRMGf

---

## 516. SplAttN: Bridging 2D and 3D with Gaussian Soft Splatting and Attention for Point Cloud Completion

**Abstract:** Although multi-modal learning has advanced point cloud completion, the theoretical mechanisms remain unclear. Recent works attribute success to the connection between modalities, yet we identify that standard hard projection severs this connection: projecting a sparse point cloud onto the image plane yields an extremely sparse support, which hinders visual prior propagation, a failure mode we term Cross-Modal Entropy Collapse. To address this practical limitation, we propose SplAttN, which replaces hard projection with Differentiable Gaussian Splatting to produce a dense, continuous image-plane representation. By reformulating projection as continuous density estimation, SplAttN avoids collapsed sparse support, facilitates gradient flow, and improves cross-modal connection learnability. Extensive experiments show that SplAttN achieves state-of-the-art performance on PCN and ShapeNet-55/34. Crucially, we utilize the real-world KITTI benchmark as a stress test for multi-modal reliance. Counter-factual evaluation reveals that while baselines degenerate into unimodal template retrievers insensitive to visual removal, SplAttN maintains a robust dependency on visual cues, validating that our method establishes an effective cross-modal connection. Code is available at https://github.com/zay002/SplAttN.

Forum: https://openreview.net/forum?id=vTp9JToZl9

---

## 517. A Kinetic Energy Perspective of Flow Matching

**Abstract:** Flow-based generative models can be viewed through a physics lens: sampling transports a particle from noise to data by integrating a learned velocity field, and each sample corresponds to a trajectory with its own dynamical effort. Motivated by classical mechanics, we introduce Kinetic Path Energy (KPE), an action-like, per-sample diagnostic that measures the accumulated kinetic effort along an ordinary differential equation (ODE) trajectory. Empirically, KPE exhibits two robust correspondences: {i} higher KPE predicts stronger semantic fidelity; {ii} high-KPE trajectories land in sparse representation regions. We further provide theoretical guarantees linking trajectory energy to data sparsity. Paradoxically, this correlation is non-monotonic. At sufficiently high energy, generation can degenerate into memorization. Leveraging the closed-form formula of empirical flow matching, we show that extreme energies drive trajectories toward near-copies of training examples. This yields a Goldilocks principle and motivates Kinetic Trajectory Shaping (KTS), a training-free two-phase inference strategy that boosts early motion and enforces a late-time soft landing, reducing memorization and improving generation quality across benchmark tasks.

Forum: https://openreview.net/forum?id=d6IyHrX48y

---

## 518. The Tell-Tale Norm: $\ell_2$ Magnitude as a Signal for Reasoning Dynamics in Large Language Models

**Abstract:** Recent work has sought to understand Large Language Models (LLMs) reasoning, yet a principled, model-intrinsic signal that captures its *layer-wise reasoning dynamics* remains underexplored. We bridge this gap by demonstrating that **the $\ell_2$ norm of hidden states serves as an endogenous signal of the model's reasoning intensity**. Using Sparse Autoencoders (SAEs) as a diagnostic probe, we observe that LLMs' internal reasoning is marked by a sharp increase in reasoning feature activations concentrated in late layers. Motivated by this pattern, we establish a formal link between reasoning intensity and the model's latent geometry and theoretically prove that the $\ell_2$ norm of hidden states bounds the activation strength of SAE reasoning features. Empirical correlation analysis and causal interventions further prove $\ell_2$ norm as a faithful indicator, where heightened norms consistently correspond to critical reasoning steps. We then introduce three test-time scaling techniques guided by $\ell_2$ norms: Adaptive Layer-wise Reasoning Recursion, (ii) Endogenous Reasoning State Steering, and (iii) $\ell_2$-guided Response Selection, which requires no additional training or data and is compatible with advanced inference engines. Experiments across model architectures and benchmarks show that $\ell_2$-norm-based techniques significantly improve reasoning performance, offering a principled yet simple lens to perceive and control LLM latent reasoning dynamics. Our codes are available at https://github.com/zjy1298/The-Tell-Tale-Norm.

Forum: https://openreview.net/forum?id=03ZTlJuX0y

---

## 519. Manifold-Optimal Guidance: A Unified Riemannian Control View of Diffusion Guidance

**Abstract:** Classifier-Free Guidance (CFG) serves as the de facto control mechanism for conditional diffusion, yet high guidance scales notoriously induce oversaturation, texture artifacts, and structural collapse. We attribute this failure to a geometric mismatch: standard CFG performs Euclidean extrapolation in ambient space, inadvertently driving sampling trajectories off the high-density data manifold. To resolve this, we present Manifold-Optimal Guidance (MOG), a framework that reformulates guidance as a local optimal control problem. MOG yields a closed-form, geometry-aware Riemannian update that corrects off-manifold drift without requiring retraining. Leveraging this perspective, we further introduce Auto-MOG, a dynamic energy-balancing schedule that adaptively calibrates guidance strength, effectively eliminating the need for manual hyperparameter tuning. Extensive validation demonstrates that MOG yields superior fidelity and alignment compared to baselines, with virtually no added computational overhead.

Forum: https://openreview.net/forum?id=eSxGdQ0zwG

---

## 520. NeuronCtrl: Geometry-Aware Safe Closed-Loop Generative Control for Neuronal Microenvironment Dynamics

**Abstract:** Neuromodulation can be viewed as closed-loop control of high-dimensional spatiotemporal fields on irregular 3D morphologies, coupling membrane electrophysiology with ionic reaction--diffusion. This view supports high-rate feedback and systematic in-silico evaluation, yet is difficult in practice. Unlike classical PDE control with known equations on regular domains, neuronal microenvironments exhibit complex, often unknown biophysics on irregular shapes. High-fidelity simulators are too costly for real-time control with repeated planning. The discretized field is sparsely observed and must satisfy hard full-field safety constraints. We introduce NeuronCtrl, a modular operator-level framework for safe, closed-loop generative control of neuronal microenvironment dynamics. Given measurements, actions, and morphology, a history-conditioned observer infers the latent field, a morphology-aware neural operator predicts one-step dynamics, and a flow-matching conditional flow proposes actions conditioned on user preferences. Safety is enforced via complementary barrier-based mechanisms at both the action and field levels, with minimal intervention. When latency is critical, the multi-step generator is distilled into a single-step policy while retaining the same safety filter. Experiments across three high-fidelity 3D neuromodulation benchmarks spanning deep brain stimulation, extracellular reaction--diffusion control, and astrocytic potassium regulation demonstrate improved trade-offs among cost, safety, and latency. Code is available at https://github.com/HowieHsu0126/NeuronControl.

Forum: https://openreview.net/forum?id=ZK3h2ENA67

---

## 521. On the Interplay of Pre-Training, Mid-Training, and RL on Reasoning Language Models

**Abstract:** Recent reinforcement learning (RL) techniques have yielded impressive reasoning improvements in language models, yet it remains unclear whether RL truly extends a model's reasoning ability beyond pre-training. A central challenge is the lack of control in modern training pipelines, where opaque pre-training data, underexplored mid-training, and complex RL interactions obscure causal effects. To resolve this ambiguity, we develop a controlled experimental framework that isolates the causal contributions of pre-training, mid-training, and RL-based post-training. Our approach employs synthetic reasoning tasks with explicit atomic operations, parseable step-by-step reasoning traces, and systematic manipulation of training distributions. We evaluate along: *extrapolative generalization* to more complex compositions and *contextual generalization* across surface contexts. Using this framework, we reconcile competing views on RL’s effectiveness: 1) RL produces true capability gains (pass@128) only when pre-training leaves sufficient headroom and RL data target the model’s *edge of competence*, tasks that are difficult but not yet out of reach. 2) Contextual generalization requires minimal yet sufficient pre-training exposure, after which RL reliably transfers. 3) Mid-training significantly enhances performance under fixed compute compared with RL alone, demonstrating its central but underexplored role. 4) Process-level rewards reduce reward hacking and improve reasoning fidelity. Together, these results clarify the interplay between pre-training, mid-training, and RL, offering a foundation for improving reasoning language models training strategies. Codes and data are avaialble at [Github](https://github.com/Interplay-LM-Reasoning/Interplay-LM-Reasoning) and [HuggingFace](https://huggingface.co/Interplay-LM-Reasoning)

Forum: https://openreview.net/forum?id=TBaUfO9znF

---

## 522. Investigating Memory in Model-Free RL with POPGym Arcade

**Abstract:** How should we analyze memory in deep RL? We introduce tools for analyzing policies under partial observability and revealing how agents use memory to make decisions. To utilize these tools, we present POPGym Arcade, a collection of Atari-inspired, hardware-accelerated environments sharing a single observation and action space. Each environment provides fully and partially observable variants, enabling counterfactual studies on observability. We find that controlled studies are necessary for fair comparisons and identify a pathology where value functions smear credit over irrelevant history. Using this pathology, we demonstrate how out-of-distribution scenarios can contaminate memory, perturbing the policy far into the future.

Forum: https://openreview.net/forum?id=vBMdonzFEV

---

## 523. Skip a Layer or Loop It? Learning Program-of-Layers in LLMs

**Abstract:** Large language models (LLMs) perform inference by following a fixed depth and order, non-recurrent execution of all layers. We reveal the wide existence of training-free, flexible, dynamic *program-of-layers (PoLar)*, where pretrained layers can be packed as modules and then skipped or looped to form a customized program for each input. For most inputs, substantially shorter program executions can achieve the same or better accuracy, while incorrect predictions of the original LLM can be corrected by alternative programs with fewer layers. These observations indicate that inference admits multiple valid latent computations beyond the standard forward pass. To efficiently achieve PoLar in practice, we propose a lightweight PoLar prediction network, which learns to generate execution programs that dynamically skip or repeat pretrained layers for each input. Experiments on mathematical reasoning benchmarks demonstrate that PoLar consistently improves accuracy over standard inference and prior dynamic-depth methods, often while executing fewer layers, and that these gains persist under out-of-distribution evaluation. Our results suggest that fixed-depth execution captures only a narrow subset of an LLM’s latent reasoning capacity.

Forum: https://openreview.net/forum?id=pl10b6EQAN

---

## 524. OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration

**Abstract:** As high-quality public text approaches exhaustion, a phenomenon known as the Data Wall—LLM pre-training is shifting from more tokens to better tokens. However, existing methods either rely on heuristic static filters that ignore training dynamics, or use dynamic yet optimizer-agnostic criteria based on raw gradients. We propose OPUS (Optimizer-induced Projected Utility Selection), a dynamic framework that defines utility in the optimizer-induced update space. OPUS scores candidates by projecting their effective updates, shaped by modern optimizers, onto a target direction derived from a stable, in-distribution proxy. To ensure scalability, we employ Ghost technique with CountSketch for computational efficiency, and Boltzmann sampling for data diversity, incurring only 4.7% additional compute overhead. OPUS achieves remarkable results across diverse corpora, quality tiers, optimizers, and model scales. It also outperforms previous data selection methods across different stages of training, including from-scratch pre-training and also mid-training. Beyond online selection, the OPUS utility score also demonstrates potential as a static filter for flagging and removing toxic documents from contaminated training corpora prior to training.

Forum: https://openreview.net/forum?id=lcLxGrk5N9

---

## 525. Enhancing Reasoning for Diffusion LLMs via Distribution Matching Policy Optimization

**Abstract:** Diffusion large language models (dLLMs) are promising alternatives to autoregressive large language models (AR-LLMs), as they potentially allow higher inference throughput. Reinforcement learning (RL) is crucial to enabling dLLMs to achieve performance comparable to that of AR-LLMs on important tasks, such as reasoning. However, RL algorithms well-suited to dLLMs' unique characteristics have yet to be developed. This paper proposes \textbf{Distribution Matching Policy Optimization (DMPO)}, a principled and theoretically grounded RL fine-tuning method specifically designed to enhance the reasoning capabilities of dLLMs by matching the dLLM policy distribution to the optimal, reward-tilted one through cross-entropy optimization. We identify a key implementation challenge with small training batch sizes and propose several effective solutions based on a novel weight baseline subtraction technique. DMPO exhibits superior performance on multiple reasoning benchmarks without supervised fine-tuning, achieving up to a $39.63$ percentage-point improvement in accuracy over prior non-DMPO RL baselines and $67.97$ percentage points over the base model, underscoring the effectiveness of the distribution-matching framework. Our code is available at https://github.com/yuchen-zhu-zyc/DMPO.

Forum: https://openreview.net/forum?id=09CSjVeDug

---

## 526. Just-In-Time Reinforcement Learning: Continual Learning in LLM Agents Without Gradient Updates

**Abstract:** While Large Language Model (LLM) agents excel at general tasks, they inherently struggle with continual adaptation due to the frozen weights after deployment. Conventional reinforcement learning (RL) offers a solution but incurs prohibitive computational costs and the risk of catastrophic forgetting. 
    We introduce Just-In-Time Reinforcement Learning (JitRL), a training-free framework that enables test-time policy optimization without any gradient updates. 
    JitRL maintains a dynamic, non-parametric memory of experiences and retrieves relevant trajectories to estimate action advantages on-the-fly. 
    These estimates are then used to directly modulate the LLM's output logits. 
    We theoretically prove that this additive update rule is the exact closed-form solution to the KL-constrained policy optimization objective. 
    Extensive experiments on WebArena and Jericho demonstrate that JitRL establishes a new state-of-the-art among training-free methods. 
    Crucially, JitRL outperforms the performance of computationally expensive fine-tuning methods (e.g., WebRL) while reducing monetary costs by over 30 times, offering a scalable path for continual learning agents. The code is available at https://github.com/liushiliushi/JitRL.

Forum: https://openreview.net/forum?id=pLvye0zHUC

---

## 527. LIMSSR: LLM-Driven Sequence-to-Score Reasoning under Training-Time Incomplete Multimodal Observations

**Abstract:** Real-world multimodal learning is often hindered by missing modalities. While Incomplete Multimodal Learning (IML) has gained traction, existing methods typically rely on the unrealistic assumption of full-modal availability during training to provide reconstruction supervision or cross-modal priors. This paper tackles the more challenging setting of IML under training-time incomplete observations, which precludes reliance on a "God's eye view" of complete data. We propose LIMSSR (LLM-Driven Incomplete Multimodal Sequence-to-Score Reasoning), a framework that reformulates this challenge as a conditional sequence reasoning task. LIMSSR leverages the semantic reasoning capabilities of Large Language Models via Prompt-Guided Context-Aware Modality Imputation and Multidimensional Representation Fusion to infer latent semantics from available contexts without direct reconstruction. To mitigate hallucinations, we introduce a Mask-Aware Dual-Path Aggregation to dynamically calibrate inference uncertainty. Extensive experiments on three Action Quality Assessment datasets demonstrate that LIMSSR significantly outperforms state-of-the-art baselines without relying on complete training data, establishing a new paradigm for data-efficient multimodal learning. Code is available at https://github.com/XuHuangbiao/LIMSSR.

Forum: https://openreview.net/forum?id=0RLF7Sj2Fg

---

## 528. Failure-Driven Workflow Refinement

**Abstract:** Workflow optimization for tool-using LLM agents is often cast as global search over candidate graphs, scored by a scalar metric. This collapses rich, multi-step failure traces into binary outcomes, obscuring recurring failure structure and making refinement inefficient. We reframe optimization as \emph{distributional refinement}: each workflow induces a density over a \textbf{Failure Signature Space} $\mathcal{F}$, and the goal is to minimize its \textbf{Expected Failure Mass}. We propose \textbf{CE-Graph}, which maintains a counterexample pool, estimates dense failure modes, and applies operator-constrained graph edits via a \textbf{Propose-and-Verify} loop with a convergence-aware stopping rule. Across math, code, and QA benchmarks, CE-Graph improves robustness while reducing optimization cost compared to strong workflow-search baselines, suggesting reliability emerges from learning and reshaping failure landscapes rather than merely maximizing aggregate success rates.

Forum: https://openreview.net/forum?id=GbYHY1RVUa

---

## 529. Required Spine Optional Limbs: Heterogeneous Federated Learning via Backbone-sharing and Activation-guided Selection

**Abstract:** Although Federated Learning (FL) offers advantages in privacy-preserving for cross-device collaborative learning, its practical deployment remains severely constrained by heterogeneous hardware resources and non-IID (non-independent and identically distributed) data across devices. Sub-model extraction has emerged as a widely adopted strategy for enabling collaborative training among devices with heterogeneous models. However, existing sub-model extraction methods in FL typically rely on coarse-grained stochastic selection or rigid rule-based neuron selection, which severely limits training performance. Specifically, stochastic strategies lead to severe parameter conflicts under non-IID data distributions, while rule-based approaches lack diversity in neuron selection per device, preventing comprehensive parameter optimization. To address this problem, this paper presents a novel sub-model extraction-based FL framework, named SpineFL, which adopts a backbone-sharing mechanism and an activation-guided pruning strategy for sub-model extraction. Specifically, SpineFL decomposes each global model layer into two portions: i) a mandatory backbone shared by all the sub-models to maintain model generalization, and ii) a dynamic portion for sub-model extraction. SpineFL adopts the activation-guided selection strategy to probabilistically select neurons according to their activation frequency from the dynamic portion to generate sub-model, where neurons exhibiting higher historical activation are more likely to be included, thereby simultaneously addressing parameter conflicts while preserving selection diversity. Experimental results demonstrate that compared with state-of-the-art heterogeneous FL methods, SpineFL can achieve up to 3.28% accuracy improvement.

Forum: https://openreview.net/forum?id=8LZfyxIQdO

---

## 530. Compositional Generalization Requires Linear, Orthogonal Representations in Vision Embedding Models

**Abstract:** Compositional generalization, the ability to recognize familiar parts in novel contexts, is a defining property of intelligent systems, yet modern models, despite massive training sets, see only a tiny fraction of the combinatorial input space. We ask what structure representations {must} have to support generalization to unseen combinations. We formalize three desiderata (divisibility, transferability, stability) and show they impose necessary geometric constraints under standard training: representations must decompose linearly into per-concept components, orthogonal across concepts. This grounds the Linear Representation Hypothesis as a necessary consequence of compositional generalization, and yields dimension bounds linking the number of composable concepts to embedding geometry. Empirically, across CLIP, SigLIP, and DINO, we find partial linear factorization with low-rank near-orthogonal per-concept factors, and the degree of this structure correlates with compositional generalization on unseen combinations. As models continue to scale, these conditions predict the geometry they may converge to. Code: https://github.com/oshapio/necessary-compositionality

Forum: https://openreview.net/forum?id=AQZZWVp6XA

---

## 531. Holi-Spatial: Evolving Video Streams into Holistic 3D Spatial Intelligence

**Abstract:** The pursuit of spatial intelligence fundamentally relies on access to large-scale, fine-grained 3D data. However, existing approaches predominantly construct spatial understanding benchmarks by generating question–answer (QA) pairs from a limited number of manually annotated datasets, rather than systematically annotating new large-scale 3D scenes from raw web data. As a result, their scalability is severely constrained, and model performance is further hindered by domain gaps inherent in these narrowly curated datasets.
In this work, we propose \textbf{Holi-Spatial}, the first fully automated, large-scale, spatially-aware multimodal dataset, constructed from raw video inputs without human intervention, using the proposed data curation pipeline. Holi-Spatial supports multi-level spatial supervision, ranging from geometrically accurate 3D Gaussian Splatting (3DGS) reconstructions with rendered depth maps to object-level and relational semantic annotations, together with corresponding spatial Question–Answer (QA) pairs.
Following a principled and systematic pipeline, we further construct \textbf{Holi-Spatial-4M}, the first large-scale, high-quality 3D semantic dataset, containing 12K optimized 3DGS scenes, 1.3M 2D masks, 320K 3D bounding boxes, 320K instance captions, 1.2M 3D grounding instances, and 1.2M spatial QA pairs spanning diverse geometric, relational, and semantic reasoning tasks.
Holi-Spatial demonstrates exceptional performance in data curation quality, significantly outperforming existing feed-forward and per-scene optimized methods on datasets such as ScanNet, ScanNet++, and DL3DV. Furthermore, fine-tuning Vision-Language Models (VLMs) on spatial reasoning tasks using this dataset has also led to substantial improvements in model performance.

Forum: https://openreview.net/forum?id=UGAP2F6FfV

---

## 532. CONTINUUM: Restoring the Contiguous Tensor Abstraction Efficiently for Dynamic AI Workloads via Hardware Virtualization

**Abstract:** Emerging LLM workloads demand extreme memory agility. However, state-of-the-art inference systems such as vLLM rely on software-defined paging, which sacrifices the contiguous tensor abstraction. This rigid interface exposes fragmentation complexity to developers, imposing a severe engineering burden that stifles algorithmic innovation. We introduce CONTINUUM, a tensor memory virtualization subsystem implemented as a PyTorch extension. By bypassing serialized OS bottlenecks through a lightweight GPU driver extension, CONTINUUM significantly reduces mapping costs by orders of magnitude, from milliseconds to microseconds. Built atop this low-latency API, CONTINUUM provides Elastic Tensor, a set of flexible tensor operations that natively support complex memory dynamics and zero-copy topological aliasing. Evaluations demonstrate that CONTINUUM achieves significantly higher throughput across diverse dynamic scenarios, effectively lowering the barrier to implementing next-generation LLM applications.

Forum: https://openreview.net/forum?id=hROxrfMoXj

---

## 533. Automated Formal Proofs of Combinatorial Identities via Wilf–Zeilberger Guidance and LLMs

**Abstract:** Automating formal proofs of combinatorial identities is challenging for LLM-based provers, as long-horizon proof planning is required and unconstrained search quickly explodes. 
Symbolic methods such as the Wilf--Zeilberger (WZ) method can achieve a mechanized proof of combinatorial identities by constructing special auxiliary functions and demonstrating that they satisfy specific recurrence relations. 
We propose WZ-LLM, a neuro-symbolic framework that turns WZ proof plans into executable proof sketches in Lean~4 and uses an LLM-based prover 
to discharge the resulting machine-checkable subgoals.
We also train a dedicated WZ-Prover via a Lean-kernel-verified bootstrapping loop with expert-verified iteration, followed by DAPO-based refinement.
Experiments show that WZ-LLM achieves a 34\% proof success rate on LCI-Test  (100 classical combinatorial identities), outperforming strong baselines such as DeepSeek-V3 and Goedel-Prover-V2; 
moreover, on LCI-Test it proves 5 identities on which the symbolic-only baseline fails. 
WZ-LLM also improves performance on CombiBench and PutnamBench-Comb, suggesting the effectiveness of coupling symbolic proof sketches with learned formal reasoning.
Experiments show that WZ-LLM achieves a 34\% proof success rate on LCI-Test (100 classic combinatorial identities), outperforming strong baselines such as DeepSeek-V3 and Goedel-Prover-V2, and delivering consistent gains on CombiBench and PutnamBench-Comb. These results indicate that our framework provides two complementary strengths: improved direct proving for identities beyond the scope of WZ, and substantially higher end-to-end success when WZ sketches guide a specialized prover.

Forum: https://openreview.net/forum?id=Xxq7fcQUNR

---

## 534. SVL: Empowering Spiking Neural Networks for Efficient 3D Open-World Understanding

**Abstract:** Spiking Neural Networks (SNNs) offer an energy--efficient route to 3D spatio--temporal perception, yet they lag behind Artificial Neural Networks (ANNs) due to weak pretraining and heavy inference stacks, limiting generalization and multimodal reasoning (e.g., zero--shot 3D classification and open--world QA). We present a universal \textbf{S}pike--based \textbf{V}ision--\textbf{L}anguage pretraining framework (SVL) that equips SNNs with open--world 3D understanding while preserving end--to--end spike efficiency. SVL comprises two core components: (i) {Multi--scale Triple Alignment} (MTA), a label--free triplet contrastive objective aligning 3D, image, and text; and (ii) {Re--parameterizable Vision--Language Integration} (Rep--VLI), which converts offline text embeddings into lightweight weights for text--encoder--free inference. Moreover, we present the first fully spike--driven point Transformer, {Spike-driven PointFormer}, whose 3D spike--driven self--attention (3D-SDSA) reduces interactions to sparse additions, enabling faster, more efficient training. Extensive experiments show that SVL attains strong zero--shot 3D classification (85.4% top--1) and consistently outperforms prior SNNs on downstream tasks (e.g., +6.1% 3D cls, +2.1% DVS actions, +1.1% detection, +2.1% segmentation) while enabling open--world 3D question answering, sometimes outperforming ANNs. To the best of our knowledge, SVL represents the first scalable, generalizable, and hardware-friendly paradigm for 3D open-world understanding, effectively bridging the gap between SNNs and ANNs in complex open-world understanding tasks.

Forum: https://openreview.net/forum?id=Ai3a79cTvr

---

## 535. Conditional Equivalence of DPO and RLHF: Assumptions, Failure Modes, and Provable Alignment

**Abstract:** Direct Preference Optimization (DPO) has emerged as a popular alternative to Reinforcement Learning from Human Feedback (RLHF), offering theoretical equivalence with a simpler implementation. We prove this equivalence is _conditional_ rather than universal, depending on an implicit assumption frequently violated in practice: the RLHF-optimal policy must prefer human-preferred responses. When this assumption fails, DPO optimizes _relative advantage_ over the reference policy rather than _absolute alignment_ with human preferences, leading to pathological convergence where policies decrease DPO loss while preferring dispreferred responses. We characterize when this assumption is violated, show the existence of an undesirable solution space, and prove that DPO and RLHF optimize fundamentally different objectives in such cases. To address this, we introduce Constrained Preference Optimization (CPO), augmenting RLHF with constraints for provable alignment. We further provide a geometric interpretation through soft margin ranking, revealing that DPO implements margin ranking with potentially negative targets. Our theoretical analysis establishes when DPOs’ guarantees hold and provides solutions preserving simplicity with provable alignment. Comprehensive experiments on standard benchmarks demonstrate that CPO achieves state-of-the-art performance. Code is available at: _https://github.com/visitworld123/CPO_.

Forum: https://openreview.net/forum?id=7UEBX1KU1y

---

## 536. From Abstraction to Instantiation: Learning Behavioral Representation for Vision-Language-Action Model

**Abstract:** Vision-Language-Action (VLA) models often suffer from performance degradation under distribution shifts, as they struggle to learn generalized behavior representations across varying environments. While existing approaches attempt to construct behavior representations through action-centric latent variables, they are often limited by short-horizon temporal fragmentation and static execution-alignment, leading to inconsistent behaviors in complex scenarios. To address these limitations, we propose \textbf{BehaviorVLA}, a framework that facilitates robust manipulation through the learning of a temporally coherent behavioral representations. Our approach features two symmetric components: (1) the \textbf{Visuomotor Behavior Encoder (VBE)}, which utilizes a causal Mamba-based architecture to aggregate long-horizon trajectory information into a unified behavior representation; and (2) the \textbf{Phase-conditioned Behavior Decoder (PBD)}, which decodes this representation into precise actions by dynamically aligning task-level priors with real-time execution progress. Experiments on RoboTwin 2.0, LIBERO, and CALVIN demonstrate state-of-the-art success rates of 58\%, 98\%, and 4.36 (Avg.Len), respectively. Notably, in real-world sim-to-real transfer, BehaviorVLA matches the performance of OpenVLA-OFT using only 50\% of the demonstration data, showcasing its superior data efficiency and generalization.

Forum: https://openreview.net/forum?id=2NRrsz4nUB

---

## 537. SVRG and Beyond via Posterior Correction

**Abstract:** Stochastic Variance Reduced Gradient (SVRG) and its variants aim to speed-up training by using gradient corrections. Originally proposed over a decade ago, these methods have never been connected to any Bayesian method at a fundamental level. Here, we fill this gap and derive surprising new connections of SVRG to a recently proposed Bayesian method called `posterior correction'. Our main contribution is to show that SVRG can be recovered as a special case of posterior-correction over isotropic-Gaussian posteriors. Novel extensions of SVRG are automatically obtained by using more flexible exponential-family posteriors. We derive two new such extensions by using Gaussian families: a Newton-like variant with novel Hessian corrections, and an Adam-like extension that scales to large problems. Our work is the first to connect SVRG to Bayes and use it to speed-up training.

Forum: https://openreview.net/forum?id=3NQSeJOfkz

---

## 538. Reward-free Alignment for Conflicting Objectives

**Abstract:** Direct alignment methods are increasingly used to align large language models (LLMs) with human preferences. However, many real-world alignment problems involve multiple conflicting objectives, where naive aggregation of preferences can lead to unstable training and poor trade-offs. In particular, weighted loss methods may fail to identify update directions that simultaneously improve all objectives, and existing multi-objective approaches often rely on explicit reward models, introducing additional complexity and distorting user-specified preferences. The contributions of this paper are two-fold. First, we propose a **R**eward-free **A**lignment framework for **C**onflicted **O**bjectives (RACO) that directly leverages pairwise preference data and resolves gradient conflicts via a novel clipped variant of conflict-averse gradient descent. We provide convergence guarantees to Pareto-critical points that respect user-specified objective weights, and further show that clipping can strictly improve convergence rate in the two-objective setting. Second, we improve our method using some heuristics and conduct experiments to demonstrate the compatibility of the proposed framework for LLM alignment. Both qualitative and quantitative evaluations on multi-objective summarization and safety alignment tasks across multiple LLM families (Qwen 3, Llama 3, Gemma 3) show that our method consistently achieves better Pareto trade-offs compared to existing multi-objective alignment baselines.

Forum: https://openreview.net/forum?id=vSzRJyg6k0

---

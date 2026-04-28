Elucidating the Design Space of Diffusion-Based Generative Models
Tero Karras NVIDIA
Miika Aittala NVIDIA
Timo Aila NVIDIA
Samuli Laine NVIDIA
Abstract
We argue that the theory and practice of diffusion-based generative models are currently unnecessarily convoluted and seek to remedy the situation by presenting a design space that clearly separates the concrete design choices. This lets us identify several changes to both the sampling and training processes, as well as preconditioning of the score networks. Together, our improvements yield new state-of-the-art FID of 1.79 for CIFAR-10 in a class-conditional setting and 1.97 in an unconditional setting, with much faster sampling (35 network evaluations per image) than prior designs. To further demonstrate their modular nature, we show that our design changes dramatically improve both the efficiency and quality obtainable with pre-trained score networks from previous work, including improving the FID of a previously trained ImageNet-64 model from 2.07 to near-SOTA 1.55, and after re-training with our proposed improvements to a new SOTA of 1.36.
1 Introduction
Diffusion-based generative models [46] have emerged as a powerful new framework for neural image synthesis, in both unconditional [16, 37, 49] and conditional [17, 36, 37, 39, 40, 42, 43, 49] settings, even surpassing the quality of GANs [13] in certain situations [9]. They are also rapidly finding use in other domains such as audio [28, 38] and video [19] generation, image segmentation [4, 57] and language translation [35]. As such, there is great interest in applying these models and improving them further in terms of image/distribution quality, training cost, and generation speed.
The literature on these models is dense on theory, and derivations of sampling schedule, training dynamics, noise level parameterization, etc., tend to be based as directly as possible on theoretical frameworks, which ensures that the models are on a solid theoretical footing. However, this approach has a danger of obscuring the available design space — a proposed model may appear as a tightly coupled package where no individual component can be modified without breaking the entire system.
As our first contribution, we take a look at the theory behind these models from a practical standpoint, focusing more on the “tangible” objects and algorithms that appear in the training and sampling phases, and less on the statistical processes from which they might be derived. The goal is to obtain better insights into how these components are linked together and what degrees of freedom are available in the design of the overall system. We focus on the broad class of models where a neural network is used to model the score [22] of a noise level dependent marginal distribution of the training data corrupted by Gaussian noise. Thus, our work is in the context of denoising score matching [54].
Our second set of contributions concerns the sampling processes used to synthesize images using diffusion models. We identify the best-performing time discretization for sampling, apply a higherorder Runge–Kutta method for the sampling process, evaluate different sampler schedules, and analyze the usefulness of stochasticity in the sampling process. The result of these improvements is a significant drop in the number of sampling steps required during synthesis, and the improved sampler can be used as a drop-in replacement with several widely used diffusions models [37, 49].
36th Conference on Neural Information Processing Systems (NeurIPS 2022).
arXiv:2206.00364v2 [cs.CV] 11 Oct 2022


σ=0 0.2 0.5 1 2 3 5 7 10 20 50 σ=0 0.2 0.5 1 2 3 5 7 10 20 50
(a) Noisy images drawn from p(x; σ) (b) Ideal denoiser outputs D(x; σ)
Figure 1: Denoising score matching on CIFAR-10. (a) Images from the training set corrupted with varying levels of additive Gaussian noise. High levels of noise lead to oversaturated colors; we normalize the images for cleaner visualization. (b) Optimal denoising result from minimizing Eq. 2 analytically (see Appendix B.3). With increasing noise level, the result approaches dataset mean.
The third set of contributions focuses on the training of the score-modeling neural network. While we continue to rely on the commonly used network architectures (DDPM [16], NCSN [48]), we provide the first principled analysis of the preconditioning of the networks’ inputs, outputs, and loss functions in a diffusion model setting and derive best practices for improving the training dynamics. We also suggest an improved distribution of noise levels during training, and note that non-leaking augmentation [25] — typically used with GANs — is beneficial for diffusion models as well.
Taken together, our contributions enable significant improvements in result quality, e.g., leading to record FIDs of 1.79 for CIFAR-10 [29] and 1.36 for ImageNet [8] in 64×64 resolution. With all key ingredients of the design space explicitly tabulated, we believe that our approach will allow easier innovation on the individual components, and thus enable more extensive and targeted exploration of the design space of diffusion models. Our implementation and pre-trained models are available at https://github.com/NVlabs/edm
2 Expressing diffusion models in a common framework
Let us denote the data distribution by pdata(x), with standard deviation σdata, and consider the family of mollified distributions p(x; σ) obtained by adding i.i.d. Gaussian noise of standard deviation σ to the data. For σmax σdata, p(x; σmax) is practically indistinguishable from pure Gaussian noise. The
idea of diffusion models is to randomly sample a noise image x0 ∼ N (0, σ2
maxI), and sequentially denoise it into images xi with noise levels σ0 = σmax > σ1 > · · · > σN = 0 so that at each noise level xi ∼ p(xi; σi). The endpoint xN of this process is thus distributed according to the data.
Song et al. [49] present a stochastic differential equation (SDE) that maintains the desired distribution p as sample x evolves over time. This allows the above process to be implemented using a stochastic solver that both removes and adds noise at each iteration. They also give a corresponding “probability flow” ordinary differential equation (ODE) where the only source of randomness is the initial noise image x0. Contrary to the usual order of treatment, we begin by examining the ODE, as it offers a fruitful setting for analyzing sampling trajectories and their discretizations. The insights carry over to stochastic sampling, which we reintroduce as a generalization in Section 4.
ODE formulation. A probability flow ODE [49] continuously increases or reduces noise level of the image when moving forward or backward in time, respectively. To specify the ODE, we must first
choose a schedule σ(t) that defines the desired noise level at time t. For example, setting σ(t) ∝ √t is mathematically natural, as it corresponds to constant-speed heat diffusion [12]. However, we will show in Section 3 that the choice of schedule has major practical implications and should not be made on the basis of theoretical convenience.
The defining characteristic of the probability flow ODE is that evolving a sample xa ∼ p(xa; σ(ta))
from time ta to tb (either forward or backward in time) yields a sample xb ∼ p(xb; σ(tb)). Following previous work [49], this requirement is satisfied (see Appendix B.1 and B.2) by
dx = −σ ̇ (t) σ(t) ∇x log p(x; σ(t)) dt, (1)
where the dot denotes a time derivative. ∇x log p(x; σ) is the score function [22], a vector field that points towards higher density of data at a given noise level. Intuitively, an infinitesimal forward step of this ODE nudges the sample away from the data, at a rate that depends on the change in noise level. Equivalently, a backward step nudges the sample towards the data distribution.
Denoising score matching. The score function has the remarkable property that it does not depend on the generally intractable normalization constant of the underlying density function p(x; σ) [22],
2


Table 1: Specific design choices employed by different model families. N is the number of ODE solver iterations that we wish to execute during sampling. The corresponding sequence of time steps is {t0, t1, . . . , tN }, where tN = 0. If the model was originally trained for specific choices of N and {ti}, the originals are denoted by M and {uj}, respectively. The denoiser is defined as Dθ(x; σ) = cskip(σ)x + cout(σ)Fθ
(cin(σ)x; cnoise(σ)); Fθ represents the raw neural network layers.
VP [49] VE [49] iDDPM [37] + DDIM [47] Ours (“EDM”) Sampling (Section 3)
ODE solver Euler Euler Euler 2nd order Heun
Time steps ti<N 1 + i
N−1 ( s − 1) σ2
max
(σ2
min/σ2
max
)i
N−1 ubj0+ M−1−j0
N−1 i+ 1
2 c, where
uM = 0
uj−1=
√
u2
j +1
max(α ̄j−1/α ̄j ,C1) −1
(σmax
1
ρ+
i
N−1 (σmin
1
ρ −σmax
1
ρ ))ρ
Schedule σ(t)
√
e
1
2 βdt2+βmint −1 √t t t
Scaling s(t) 1/√
e
1
2 βdt2+βmint 1 1 1
Network and preconditioning (Section 5)
Architecture of Fθ DDPM++ NCSN++ DDPM (any) Skip scaling cskip(σ) 1 1 1 σ2
data/ (σ2 + σ2
data
)
Output scaling cout(σ) −σ σ −σ σ · σdata/√σ2
data + σ2
Input scaling cin(σ) 1/√σ2 + 1 1 1/√σ2 + 1 1/√σ2 + σ2
data
Noise cond. cnoise(σ) (M − 1) σ−1(σ) ln( 1
2 σ) M −1−arg minj |uj − σ| 1
4 ln(σ)
Training (Section 5)
Noise distribution σ−1(σ) ∼ U ( t, 1) ln(σ) ∼ U (ln(σmin), σ = uj, j ∼ U {0, M −1} ln(σ) ∼ N (Pmean, P 2
std)
ln(σmax))
Loss weighting λ(σ) 1/σ2 1/σ2 1/σ2 (note: ∗) (σ2 +σ2
data
) /(σ · σdata)2
Parameters βd = 19.9, βmin = 0.1 σmin = 0.02 α ̄j = sin2( π
2
j
M(C2+1) ) σmin = 0.002, σmax = 80
s = 10−3, t = 10−5 σmax = 100 C1 = 0.001, C2 = 0.008 σdata = 0.5, ρ = 7 M = 1000 M = 1000, j0 = 8† Pmean = −1.2, Pstd = 1.2 ∗ iDDPM also employs a second loss term Lvlb † In our tests, j0 = 8 yielded better FID than j0 = 0 used by iDDPM
and thus can be much easier to evaluate. Specifically, if D(x; σ) is a denoiser function that minimizes the expected L2 denoising error for samples drawn from pdata separately for every σ, i.e.,
Ey∼pdata En∼N (0,σ2I)‖D(y + n; σ) − y‖2
2, then ∇x log p(x; σ) = (D(x; σ) − x)/σ2, (2, 3)
where y is a training image and n is noise. In this light, the score function isolates the noise component from the signal in x, and Eq. 1 amplifies (or diminishes) it over time. Figure 1 illustrates the behavior of ideal D in practice. The key observation in diffusion models is that D(x; σ) can be implemented as a neural network Dθ(x; σ) trained according to Eq. 2. Note that Dθ may include additional pre- and post-processing steps, such as scaling x to an appropriate dynamic range; we will return to such preconditioning in Section 5.
Time-dependent signal scaling. Some methods (see Appendix C.1) introduce an additional scale schedule s(t) and consider x = s(t)ˆx to be a scaled version of the original, non-scaled variable ˆx. This changes the time-dependent probability density, and consequently also the ODE solution trajectories. The resulting ODE is a generalization of Eq. 1:
dx =
[ s ̇(t)
s(t) x − s(t)2 σ ̇ (t) σ(t) ∇x log p
(x
s(t) ; σ(t)
)]
dt. (4)
Note that we explicitly undo the scaling of x when evaluating the score function to keep the definition of p(x; σ) independent of s(t).
Solution by discretization. The ODE to be solved is obtained by substituting Eq. 3 into Eq. 4 to define the point-wise gradient, and the solution can be found by numerical integration, i.e., taking finite steps over discrete time intervals. This requires choosing both the integration scheme (e.g., Euler or a variant of Runge–Kutta), as well as the discrete sampling times {t0, t1, . . . , tN }. Many
prior works rely on Euler’s method, but we show in Section 3 that a 2nd order solver offers a better computational tradeoff. For brevity, we do not provide a separate pseudocode for Euler’s method applied to our ODE here, but it can be extracted from Algorithm 1 by omitting lines 6–8.
Putting it together. Table 1 presents formulas for reproducing deterministic variants of three earlier methods in our framework. These methods were chosen because they are widely used and
3


NFE=8 16 32 64 128 256 512 1024
23
5
10
20
50
100
200
FID
35 8 32 128 512 2048 8192
23
5
10
20
50
100
200
F5I0D0
27 8 16 32 64 128 256 512 1024
2
3
5
10
20
FID
79
Original sampler Our reimplementation + Heun & our {ti} + Our σ(t) & s(t) Black-box RK45
(a) Uncond. CIFAR-10, VP ODE (b) Uncond. CIFAR-10, VE ODE (c) Class-cond. ImageNet-64, DDIM
Figure 2: Comparison of deterministic sampling methods using three pre-trained models. For each curve, the dot indicates the lowest NFE whose FID is within 3% of the lowest observed FID.
achieve state-of-the-art performance, but also because they were derived from different theoretical foundations. Some of our formulas appear quite different from the original papers as indirection and recursion have been removed; see Appendix C for details. The main purpose of this reframing is to bring into light all the independent components that often appear tangled together in previous work. In our framework, there are no implicit dependencies between the components — any choices (within reason) for the individual formulas will, in principle, lead to a functioning model. In other words, changing one component does not necessitate changes elsewhere in order to, e.g., maintain the property that the model converges to the data in the limit. In practice, some choices and combinations will of course work better than others.
3 Improvements to deterministic sampling
Improving the output quality and/or decreasing the computational cost of sampling are common topics in diffusion model research (e.g., [10, 24, 31, 32, 33, 37, 44, 53, 55, 56, 59]). Our hypothesis is that the choices related to the sampling process are largely independent of the other components, such as network architecture and training details. In other words, the training procedure of Dθ should not dictate σ(t), s(t), and {ti}, nor vice versa; from the viewpoint of the sampler, Dθ is simply a black box [55, 56]. We test this by evaluating different samplers on three pre-trained models, each representing a different theoretical framework and model family. We first measure baseline results for these models using their original sampler implementations, and then bring these samplers into our unified framework using the formulas in Table 1, followed by our improvements. This allows us to evaluate different practical choices and propose general improvements to the sampling process that are applicable to all models.
We evaluate the “DDPM++ cont. (VP)” and “NCSN++ cont. (VE)” models by Song et al. [49] trained on unconditional CIFAR-10 [29] at 32×32, corresponding to the variance preserving (VP) and variance exploding (VE) formulations [49], originally inspired by DDPM [16] and SMLD [48]. We also evaluate the “ADM (dropout)” model by Dhariwal and Nichol [9] trained on class-conditional ImageNet [8] at 64×64, corresponding to the improved DDPM (iDDPM) formulation [37]. This model was trained using a discrete set of M = 1000 noise levels. Further details are given in Appendix C.
We evaluate the result quality in terms of Fréchet inception distance (FID) [15] computed between 50,000 generated images and all available real images. Figure 2 shows FID as a function of neural function evaluations (NFE), i.e., how many times Dθ is evaluated to produce a single image. Given that the sampling process is dominated entirely by the cost of Dθ, improvements in NFE translate directly to sampling speed. The original deterministic samplers are shown in blue, and the reimplementations of these methods in our unified framework (orange) yield similar but consistently better results. The differences are explained by certain oversights in the original implementations as well as our more careful treatment of discrete noise levels in the case of DDIM; see Appendix C. Note that our reimplementations are fully specified by Algorithm 1 and Table 1, even though the original codebases are structured very differently from each other.
Discretization and higher-order integrators. Solving an ODE numerically is necessarily an approximation of following the true solution trajectory. At each step, the solver introduces truncation error that accumulates over the course of N steps. The local error generally scales superlinearly with respect to step size, and thus increasing N improves the accuracy of the solution.
The commonly used Euler’s method is a first order ODE solver with O(h2) local error with respect to step size h. Higher-order Runge–Kutta methods [50] scale more favorably but require multiple
4


Algorithm 1 Deterministic sampling using Heun’s 2nd order method with arbitrary σ(t) and s(t).
1: procedure HEUNSAMPLER(Dθ(x; σ), σ(t), s(t), ti∈{0,...,N}) 2: sample x0 ∼ N (0, σ2(t0) s2(t0) I) . Generate initial sample at t0 3: for i ∈ {0, . . . , N − 1} do . Solve Eq. 4 over N time steps
4: di ←
( σ ̇ (ti)
σ(ti) + s ̇(ti)
s(ti)
)
xi − σ ̇ (ti)s(ti)
σ(ti) Dθ
( xi
s(ti) ; σ(ti)
)
. Evaluate dx/dt at ti
5: xi+1 ← xi + (ti+1 − ti)di . Take Euler step from ti to ti+1
6: if σ(ti+1) 6= 0 then . Apply 2nd order correction unless σ goes to zero
7: di′ ←
( σ ̇ (ti+1)
σ(ti+1) + s ̇(ti+1)
s(ti+1)
)
xi+1 − σ ̇ (ti+1)s(ti+1)
σ(ti+1) Dθ
( xi+1
s(ti+1) ; σ(ti+1)
)
. Eval. dx/dt at ti+1
8: xi+1 ← xi + (ti+1 − ti)( 1
2 di + 1
2 di′
) . Explicit trapezoidal rule at ti+1 9: return xN . Return noise-free sample at tN
evaluations of Dθ per step. Linear multistep methods have also been recently proposed for sampling
diffusion models [31, 59]. Through extensive tests, we have found Heun’s 2nd order method [2] (a.k.a. improved Euler, trapezoidal rule) — previously explored in the context of diffusion models by Jolicoeur-Martineau et al. [24] — to provide an excellent tradeoff between truncation error and NFE. As illustrated in Algorithm 1, it introduces an additional correction step for xi+1 to account for change
in dx/dt between ti and ti+1. This correction leads to O(h3) local error at the cost of one additional evaluation of Dθ per step. Note that stepping to σ = 0 would result in a division by zero, so we revert
to Euler’s method in this case. We discuss the general family of 2nd order solvers in Appendix D.2.
The time steps {ti} determine how the step sizes and thus truncation errors are distributed between different noise levels. We provide a detailed analysis in Appendix D.1, concluding that the step size should decrease monotonically with decreasing σ and it does not need to vary on a per-sample basis. We adopt a parameterized scheme where the time steps are defined according to a sequence of noise levels {σi}, i.e., ti = σ−1(σi). We set σi<N = (Ai + B)ρ and select the constants A and B so that σ0 = σmax and σN−1 = σmin, which gives
σi<N = (σmax
1
ρ+ i
N−1 (σmin
1
ρ − σmax
1
ρ ))ρ and σN = 0. (5)
Here ρ controls how much the steps near σmin are shortened at the expense of longer steps near σmax. Our analysis in Appendix D.1 shows that setting ρ = 3 nearly equalizes the truncation error at each step, but that ρ in range of 5 to 10 performs much better for sampling images. This suggests that errors near σmin have a large impact. We set ρ = 7 for the remainder of this paper.
Results for Heun’s method and Eq. 5 are shown as the green curves in Figure 2. We observe consistent improvement in all cases: Heun’s method reaches the same FID as Euler’s method with considerably lower NFE.
Trajectory curvature and noise schedule. The shape of the ODE solution trajectories is defined by functions σ(t) and s(t). The choice of these functions offers a way to reduce the truncation errors discussed above, as their magnitude can be expected to scale proportional to the curvature of dx/dt. We argue that the best choice for these functions is σ(t) = t and s(t) = 1, which is also the choice made in DDIM [47]. With this choice, the ODE of Eq. 4 simplifies to dx/dt = (x − D(x; t))/t and σ and t become interchangeable.
An immediate consequence is that at any x and t, a single Euler step to t = 0 yields the denoised image Dθ(x; t). The tangent of the solution trajectory therefore always points towards the denoiser output. This can be expected to change only slowly with the noise level, which corresponds to largely linear solution trajectories. The 1D ODE sketch of Figure 3c supports this intuition; the solution trajectories approach linear at both large and small noise levels, and have substantial curvature in only a small region in between. The same effect can be seen with real data in Figure 1b, where the change between different denoiser targets occurs in a relatively narrow σ range. With the advocated schedule, this corresponds to high ODE curvature being limited to this same range.
The effect of setting σ(t) = t and s(t) = 1 is shown as the red curves in Figure 2. As DDIM already employs these same choices, the red curve is identical to the green one for ImageNet-64. However, VP and VE benefit considerably from switching away from their original schedules.
5


0.0 0.2 0.4 0.6 0.8
2
1
0
1
2
x
0 200 400 600
40
20
0
20
40
x
0 5 10 15 20 25
40
20
0
20
40
x
t= t= t=
(a) Variance preserving ODE [49] (b) Variance exploding ODE [49] (c) DDIM [47] / Our ODE
Figure 3: A sketch of ODE curvature in 1D where pdata is two Dirac peaks at x = ±1. Horizontal t axis is chosen to show σ ∈ [0, 25] in each plot, with insets showing σ ∈ [0, 1] near the data. Example local gradients are shown with black arrows. (a) Variance preserving ODE of Song et al. [49] has solution trajectories that flatten out to horizontal lines at large σ. Local gradients start pointing towards data only at small σ. (b) Variance exploding variant has extreme curvature near data and the solution trajectories are curved everywhere. (c) With the schedule used by DDIM [47] and us, as σ increases the solution trajectories approach straight lines that point towards the mean of data. As σ → 0, the trajectories become linear and point towards the data manifold.
Discussion. The choices that we made in this section to improve deterministic sampling are summarized in the Sampling part of Table 1. Together, they reduce the NFE needed to reach highquality results by a large factor: 7.3× for VP, 300× for VE, and 3.2× for DDIM, corresponding to the highlighted NFE values in Figure 2. In practice, we can generate 26.3 high-quality CIFAR-10 images per second on a single NVIDIA V100. The consistency of improvements corroborates our hypothesis that the sampling process is orthogonal to how each model was originally trained. As further validation, we show results for the adaptive RK45 method [11] using our schedule as the dashed black curves in Figure 2; the cost of this sophisticated ODE solver outweighs its benefits.
4 Stochastic sampling
Deterministic sampling offers many benefits, e.g., the ability to turn real images into their corresponding latent representations by inverting the ODE. However, it tends to lead to worse output quality [47, 49] than stochastic sampling that injects fresh noise into the image in each step. Given that ODEs and SDEs recover the same distributions in theory, what exactly is the role of stochasticity?
Background. The SDEs of Song et al. [49] can be generalized [20, 58] as a sum of the probability flow ODE of Eq. 1 and a time-varying Langevin diffusion SDE [14] (see Appendix B.5):
dx± = −σ ̇ (t)σ(t)∇x log p(x; σ(t)) dt
} {{ }
probability flow ODE (Eq. 1)
± β(t)σ(t)2∇x log p(x; σ(t)) dt
} {{ }
deterministic noise decay
+
√2β(t)σ(t) dωt
} {{ }
noise injection
} {{ }
Langevin diffusion SDE
, (6)
where ωt is the standard Wiener process. dx+ and dx− are now separate SDEs for moving forward and backward in time, related by the time reversal formula of Anderson [1]. The Langevin term can further be seen as a combination of a deterministic score-based denoising term and a stochastic noise injection term, whose net noise level contributions cancel out. As such, β(t) effectively expresses the relative rate at which existing noise is replaced with new noise. The SDEs of Song et al. [49] are recovered with the choice β(t) = σ ̇ (t)/σ(t), whereby the score vanishes from the forward SDE.
This perspective reveals why stochasticity is helpful in practice: The implicit Langevin diffusion drives the sample towards the desired marginal distribution at a given time, actively correcting for any errors made in earlier sampling steps. On the other hand, approximating the Langevin term with discrete SDE solver steps introduces error in itself. Previous results [3, 24, 47, 49] suggest that non-zero β(t) is helpful, but as far as we can tell, the implicit choice for β(t) in Song et al. [49] enjoys no special properties. Hence, the optimal amount of stochasticity should be determined empirically.
Our stochastic sampler. We propose a stochastic sampler that combines our 2nd order deterministic ODE integrator with explicit Langevin-like “churn” of adding and removing noise. A pseudocode is given in Algorithm 2. At each step i, given the sample xi at noise level ti (= σ(ti)), we perform two
6


Algorithm 2 Our stochastic sampler with σ(t) = t and s(t) = 1.
1: procedure STOCHASTICSAMPLER(Dθ(x; σ), ti∈{0,...,N}, γi∈{0,...,N−1}, Snoise) 2: sample x0 ∼ N (0, t2
0 I)
3: for i ∈ {0, . . . , N − 1} do . γi =
{
min
( Schurn
N , √2−1
)
if ti∈[Stmin,Stmax]
0 otherwise
4: sample i ∼ N (0, S2
noise I)
5: tˆi ← ti + γiti . Select temporarily increased noise level tˆi 6: xˆi ← xi +
√tˆ2
i − t2
i i . Add new noise to move from ti to tˆi 7: di ← (xˆi − Dθ(xˆi; tˆi))/tˆi . Evaluate dx/dt at tˆi 8: xi+1 ← xˆi + (ti+1 − tˆi)di . Take Euler step from tˆi to ti+1 9: if ti+1 6= 0 then
10: di′ ← (xi+1 − Dθ(xi+1; ti+1))/ti+1 . Apply 2nd order correction 11: xi+1 ← xˆi + (ti+1 − tˆi)( 1
2 di + 1
2 di′
)
12: return xN
sub-steps. First, we add noise to the sample according to a factor γi ≥ 0 to reach a higher noise level
tˆi = ti + γiti. Second, from the increased-noise sample xˆi, we solve the ODE backward from tˆi to ti+1 with a single step. This yields a sample xi+1 with noise level ti+1, and the iteration continues. We stress that this is not a general-purpose SDE solver, but a sampling procedure tailored for the specific problem. Its correctness stems from the alternation of two sub-steps that each maintain the correct distribution (up to truncation error in the ODE step). The predictor-corrector sampler of Song et al. [49] has a conceptually similar structure to ours.
To analyze the main difference between our method and Euler–Maruyama, we first note a subtle discrepancy in the latter when discretizing Eq. 6. One can interpret Euler–Maruyama as first adding noise and then performing an ODE step, not from the intermediate state after noise injection, but assuming that x and σ remained at the initial state at the beginning of the iteration step. In our method, the parameters used to evaluate Dθ on line 7 of Algorithm 2 correspond to the state after
noise injection, whereas an Euler–Maruyama -like method would use xi; ti instead of xˆi; tˆi. In the limit of ∆t approaching zero there may be no difference between these choices, but the distinction appears to become significant when pursuing low NFE with large steps.
Practical considerations. Increasing the amount of stochasticity is effective in correcting errors made by earlier sampling steps, but it has its own drawbacks. We have observed (see Appendix E.1) that excessive Langevin-like addition and removal of noise results in gradual loss of detail in the generated images with all datasets and denoiser networks. There is also a drift toward oversaturated colors at very low and high noise levels. We suspect that practical denoisers induce a slightly nonconservative vector field in Eq. 3, violating the premises of Langevin diffusion and causing these detrimental effects. Notably, our experiments with analytical denoisers (such as the one in Figure 1b) have not shown such degradation.
If the degradation is caused by flaws in Dθ(x; σ), they can only be remedied using heuristic means during sampling. We address the drift toward oversaturated colors by only enabling stochasticity within a specific range of noise levels ti ∈ [Stmin, Stmax]. For these noise levels, we define γi = Schurn/N , where Schurn controls the overall amount of stochasticity. We further clamp γi to never introduce more new noise than what is already present in the image. Finally, we have found that the loss of detail can be partially counteracted by setting Snoise slightly above 1 to inflate the standard deviation for the newly added noise. This suggests that a major component of the hypothesized non-conservativity of Dθ(x; σ) is a tendency to remove slightly too much noise — most likely due to regression toward the mean that can be expected to happen with any L2-trained denoiser [30].
Evaluation. Figure 4 shows that our stochastic sampler outperforms previous samplers [24, 37, 49] by a significant margin, especially at low step counts. Jolicoeur-Martineau et al. [24] use a standard higher-order adaptive SDE solver [41] and its performance is a good baseline for such solvers in general. Our sampler has been tailored to the use case by, e.g., performing noise injection and ODE step sequentially, and it is not adaptive. It is an open question if adaptive solvers can be a net win over a well-tuned fixed schedule in sampling diffusion models.
Through sampler improvements alone, we are able to bring the ImageNet-64 model that originally achieved FID 2.07 [9] to 1.55 that is very close to the state-of-the-art; previously, FID 1.48 has been
7


NFE=16 32 64 128 256 512 1024 2048
1.8
2.0
2.2
2.4
2.6
2.8
3.0
3.2
FID
2.27
Deterministic Stmin,tmax = [0, ∞] Stmin,tmax + Snoise = 1 Optimal settings Snoise = 1 Original sampler Jolicoeur-Martineau et al. [24]
16 32 64 128 256 512 1024 2048
2.0
2.5
3.0
3.5
4.0
4.5
FID
2.23
16 32 64 128 256 512 1024 2048
1.4
1.6
1.8
2.0
2.2
2.4
2.6
2.8
3.0
FID
1.55
(a) Uncond. CIFAR-10, VP (b) Uncond. CIFAR-10, VE (c) Class-cond. ImageNet-64
Figure 4: Evaluation of our stochastic sampler (Algorithm 2). The purple curve corresponds to optimal choices for {Schurn, Stmin, Stmax, Snoise}; orange, blue, and green correspond to disabling the effects of Stmin,tmax and/or Snoise. The red curves show reference results for our deterministic sampler (Algorithm 1), equivalent to setting Schurn = 0. The dashed black curves correspond to the original stochastic samplers from previous work: Euler–Maruyama [49] for VP, predictor-corrector [49] for VE, and iDDPM [37] for ImageNet-64. The dots indicate lowest observed FID.
reported for cascaded diffusion [17], 1.55 for classifier-free guidance [18], and 1.52 for StyleGANXL [45]. While our results showcase the potential gains achievable through sampler improvements, they also highlight the main shortcoming of stochasticity: For best results, one must make several heuristic choices — either implicit or explicit — that depend on the specific model. Indeed, we had to find the optimal values of {Schurn, Stmin, Stmax, Snoise} on a case-by-case basis using grid search (Appendix E.2). This raises a general concern that using stochastic sampling as the primary means of evaluating model improvements may inadvertently end up influencing the design choices related to model architecture and training.
5 Preconditioning and training
There are various known good practices for training neural networks in a supervised fashion. For example, it is advisable to keep input and output signal magnitudes fixed to, e.g., unit variance, and to avoid large variation in gradient magnitudes on a per-sample basis [5, 21]. Training a neural network to model D directly would be far from ideal — for example, as the input x = y + n is a combination of clean signal y and noise n ∼ N (0, σ2I), its magnitude varies immensely depending on noise level σ. For this reason, the common practice is to not represent Dθ as a neural network directly, but instead train a different network Fθ from which Dθ is derived.
Previous methods [37, 47, 49] address the input scaling via a σ-dependent normalization factor and attempt to precondition the output by training Fθ to predict n scaled to unit variance, from which the signal is then reconstructed via Dθ(x; σ) = x − σFθ(·). This has the drawback that at large σ, the network needs to fine-tune its output carefully to cancel out the existing noise n exactly and give the output at the correct scale; note that any errors made by the network are amplified by a factor of σ. In this situation, it would seem much easier to predict the expected output D(x; σ) directly. In the same spirit as previous parameterizations that adaptively mix signal and noise (e.g., [10, 44, 53]), we propose to precondition the neural network with a σ-dependent skip connection that allows it to estimate either y or n, or something in between. We thus write Dθ in the following form:
Dθ(x; σ) = cskip(σ) x + cout(σ) Fθ
(cin(σ) x; cnoise(σ)), (7)
where Fθ is the neural network to be trained, cskip(σ) modulates the skip connection, cin(σ) and cout(σ) scale the input and output magnitudes, and cnoise(σ) maps noise level σ into a conditioning input for Fθ. Taking a weighted expectation of Eq. 2 over the noise levels gives the overall training loss
Eσ,y,n
[λ(σ) ‖D(y + n; σ) − y‖22
], where σ ∼ ptrain, y ∼ pdata, and n ∼ N (0, σ2I). The probability of sampling a given noise level σ is given by ptrain(σ) and the corresponding weight is given by λ(σ). We can equivalently express this loss with respect to the raw network output Fθ in Eq. 7:
Eσ,y,n
[
λ(σ) cout(σ)2
} {{ }
effective weight
∥
∥ Fθ
(cin(σ) · (y + n); cnoise(σ))
} {{ }
network output
−1
cout (σ )
(y − cskip(σ) · (y + n))
} {{ }
effective training target
∥ ∥
2 2
]
. (8)
This form reveals the effective training target of Fθ, allowing us to determine suitable choices for the preconditioning functions from first principles. As detailed in Appendix B.6, we derive our choices
8


Table 2: Evaluation of our training improvements. The starting point (config A) is VP & VE using our deterministic sampler. At the end (configs E,F), VP & VE only differ in the architecture of Fθ.
CIFAR-10 [29] at 32×32 FFHQ [27] 64×64 AFHQv2 [7] 64×64 Conditional Unconditional Unconditional Unconditional Training configuration VP VE VP VE VP VE VP VE A Baseline [49] (∗pre-trained) 2.48 3.11 3.01∗ 3.77∗ 3.39 25.95 2.58 18.52 B + Adjust hyperparameters 2.18 2.48 2.51 2.94 3.13 22.53 2.43 23.12 C + Redistribute capacity 2.08 2.52 2.31 2.83 2.78 41.62 2.54 15.04 D + Our preconditioning 2.09 2.64 2.29 3.10 2.94 3.39 2.79 3.81 E + Our loss function 1.88 1.86 2.05 1.99 2.60 2.81 2.29 2.28 F + Non-leaky augmentation 1.79 1.79 1.97 1.98 2.39 2.53 1.96 2.16 NFE 35 35 35 35 79 79 79 79
shown in Table 1 by requiring network inputs and training targets to have unit variance (cin, cout), and amplifying errors in Fθ as little as possible (cskip). The formula for cnoise is chosen empirically.
Table 2 shows FID for a series of training setups, evaluated using our deterministic sampler from Section 3. We start with the baseline training setup of Song et al. [49], which differs considerably between the VP and VE cases; we provide separate results for each (config A). To obtain a more meaningful point of comparison, we re-adjust the basic hyperparameters (config B) and improve the expressive power of the model (config C) by removing the lowest-resolution layers and doubling the capacity of the highest-resolution layers instead; see Appendix F.3 for further details. We then replace the original choices of {cin, cout, cnoise, cskip} with our preconditioning (config D), which keeps the results largely unchanged — except for VE that improves considerably at 64×64 resolution. Instead of improving FID per se, the main benefit of our preconditioning is that it makes the training more robust, enabling us to turn our focus on redesigning the loss function without adverse effects.
Loss weighting and sampling. Eq. 8 shows that training Fθ as preconditioned in Eq. 7 incurs
an effective per-sample loss weight of λ(σ)cout(σ)2. To balance the effective loss weights, we set
λ(σ) = 1/cout(σ)2, which also equalizes the initial training loss over the entire σ range as shown in Figure 5a (green curve). Finally, we need to select ptrain(σ), i.e., how to choose noise levels during training. Inspecting the per-σ loss after training (blue and orange curves) reveals that a significant reduction is possible only at intermediate noise levels; at very low levels, it is both difficult and irrelevant to discern the vanishingly small noise component, whereas at high levels the training targets are always dissimilar from the correct answer that approaches dataset average. Therefore, we target the training efforts to the relevant range using a simple log-normal distribution for ptrain(σ) as detailed in Table 1 and illustrated in Figure 5a (red curve).
Table 2 shows that our proposed ptrain and λ (config E) lead to a dramatic improvement in FID in all cases when used in conjunction with our preconditioning (config D). In concurrent work, Choi et al. [6] propose a similar scheme to prioritize noise levels that are most relevant w.r.t. forming the perceptually recognizable content of the image. However, they only consider the choice of λ in isolation, which results in a smaller overall improvement.
Augmentation regularization. To prevent potential overfitting that often plagues diffusion models with smaller datasets, we borrow an augmentation pipeline from the GAN literature [25]. The pipeline consists of various geometric transformations (see Appendix F.2) that we apply to a training image prior to adding noise. To prevent the augmentations from leaking to the generated images, we provide the augmentation parameters as a conditioning input to Fθ; during inference we set the them to zero to guarantee that only non-augmented images are generated. Table 2 shows that data augmentation provides a consistent improvement (config F) that yields new state-of-the-art FIDs of 1.79 and 1.97 for conditional and unconditional CIFAR-10, beating the previous records of 1.85 [45] and 2.10 [53].
Stochastic sampling revisited. Interestingly, the relevance of stochastic sampling appears to diminish as the model itself improves, as shown in Figure 5b,c. When using our training setup in CIFAR-10 (Figure 5b), the best results were obtained with deterministic sampling, and any amount of stochastic sampling was detrimental.
ImageNet-64. As a final experiment, we trained a class-conditional ImageNet-64 model from scratch using our proposed training improvements. This model achieved a new state-of-the-art FID of
9


σ=0.005 0.02 0.1 0.5 1 2 5 10 20 50
0.0
0.2
0.4
0.6
0.8
1.0
1.2
1.4
loss Loss after init CIFAR-10 Distribution of σ FFHQ-64
Schurn=0 10 20 30 40 50 60 70 80 90 100
2.0
2.5
3.0
3.5
4.0
FID VP, original VP, our model VE, original VE, our model
Schurn=0 10 20 30 40 50 60 70 80 90 100
1.2
1.4
1.6
1.8
2.0
2.2
2.4
2.6
FID
1.57
1.36
2.66
2.22
Original Our model
(a) Loss & noise distribution (b) Stochasticity on CIFAR-10 (c) Stochasticity on ImageNet-64
Figure 5: (a) Observed initial (green) and final loss per noise level, representative of the the 32×32 (blue) and 64×64 (orange) models considered in this paper. The shaded regions represent the standard deviation over 10k random samples. Our proposed training sample density is shown by the dashed red curve. (b) Effect of Schurn on unconditional CIFAR-10 with 256 steps (NFE = 511). For the original training setup of Song et al. [49], stochastic sampling is highly beneficial (blue, green), while deterministic sampling (Schurn = 0) leads to relatively poor FID. For our training setup, the situation is reversed (orange, red); stochastic sampling is not only unnecessary but harmful. (c) Effect of Schurn on class-conditional ImageNet-64 with 256 steps (NFE = 511). In this more challenging scenario, stochastic sampling turns out to be useful again. Our training setup improves the results for both deterministic and stochastic sampling.
1.36 compared to the previous record of 1.48 [17]. We used the ADM architecture [9] with no changes, and trained it using our config E with minimal tuning; see Appendix F.3 for details. We did not find overfitting to be a concern, and thus chose to not employ augmentation regularization. As shown in Figure 5c, the optimal amount of stochastic sampling was much lower than with the pre-trained model, but unlike with CIFAR-10, stochastic sampling was clearly better than deterministic sampling. This suggests that more diverse datasets continue to benefit from stochastic sampling.
6 Conclusions
Our approach of putting diffusion models to a common framework exposes a modular design. This allows a targeted investigation of individual components, potentially helping to better cover the viable design space. In our tests this let us simply replace the samplers in various earlier models, drastically improving the results. For example, in ImageNet-64 our sampler turned an average model (FID 2.07) to a challenger (1.55) for the previous SOTA model (1.48) [17], and with training improvements achieved SOTA FID of 1.36. We also obtained new state-of-the-art results on CIFAR-10 while using only 35 model evaluations, deterministic sampling, and a small network. The current high-resolution diffusion models rely either on separate super-resolution steps [17, 36, 40], subspace projection [23], very large networks [9, 49], or hybrid approaches [39, 42, 53] — we believe that our contributions are orthogonal to these extensions. That said, many of our parameter values may need to be re-adjusted for higher resolution datasets. Furthermore, we feel that the precise interaction between stochastic sampling and the training objective remains an interesting question for future work.
Societal impact. Our advances in sample quality can potentially amplify negative societal effects when used in a large-scale system like DALL·E 2, including types of disinformation or emphasizing sterotypes and harmful biases [34]. The training and sampling of diffusion models needs a lot of electricity; our project consumed ∼250MWh on an in-house cluster of NVIDIA V100s.
Acknowledgments. We thank Jaakko Lehtinen, Ming-Yu Liu, Tuomas Kynkäänniemi, Axel Sauer, Arash Vahdat, and Janne Hellsten for discussions and comments, and Tero Kuosmanen, Samuel Klenberg, and Janne Hellsten for maintaining our compute infrastructure.
References
[1] B. D. Anderson. Reverse-time diffusion equation models. Stochastic Processes and their Applications, 12(3):313–326, 1982.
[2] U. M. Ascher and L. R. Petzold. Computer Methods for Ordinary Differential Equations and DifferentialAlgebraic Equations. Society for Industrial and Applied Mathematics, 1998.
10


[3] F. Bao, C. Li, J. Zhu, and B. Zhang. Analytic-DPM: an analytic estimate of the optimal reverse variance in diffusion probabilistic models. In Proc. ICLR, 2022. [4] D. Baranchuk, A. Voynov, I. Rubachev, V. Khrulkov, and A. Babenko. Label-efficient semantic segmentation with diffusion models. In Proc. ICLR, 2022. [5] C. M. Bishop. Neural networks for pattern recognition. Oxford University Press, USA, 1995. [6] J. Choi, J. Lee, C. Shin, S. Kim, H. Kim, and S. Yoon. Perception prioritized training of diffusion models. In Proc. CVPR, 2022.
[7] Y. Choi, Y. Uh, J. Yoo, and J.-W. Ha. StarGAN v2: Diverse image synthesis for multiple domains. In Proc. CVPR, 2020. [8] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei. ImageNet: A large-scale hierarchical image database. In Proc. CVPR, 2009. [9] P. Dhariwal and A. Q. Nichol. Diffusion models beat GANs on image synthesis. In Proc. NeurIPS, 2021. [10] T. Dockhorn, A. Vahdat, and K. Kreis. Score-based generative modeling with critically-damped Langevin diffusion. In Proc. ICLR, 2022. [11] J. R. Dormand and P. J. Prince. A family of embedded Runge-Kutta formulae. Journal of computational and applied mathematics, 6(1):19–26, 1980.
[12] J. B. J. Fourier, G. Darboux, et al. Théorie analytique de la chaleur, volume 504. Didot Paris, 1822. [13] I. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, and Y. Bengio. Generative adversarial networks. In Proc. NIPS, 2014. [14] U. Grenander and M. I. Miller. Representations of knowledge in complex systems. Journal of the Royal Statistical Society: Series B (Methodological), 56(4):549–581, 1994.
[15] M. Heusel, H. Ramsauer, T. Unterthiner, B. Nessler, and S. Hochreiter. GANs trained by a two time-scale update rule converge to a local Nash equilibrium. In Proc. NIPS, 2017. [16] J. Ho, A. Jain, and P. Abbeel. Denoising diffusion probabilistic models. In Proc. NeurIPS, 2020. [17] J. Ho, C. Saharia, W. Chan, D. J. Fleet, M. Norouzi, and T. Salimans. Cascaded diffusion models for high fidelity image generation. Journal of Machine Learning Research, 23, 2022.
[18] J. Ho and T. Salimans. Classifier-free diffusion guidance. In NeurIPS 2021 Workshop on Deep Generative Models and Downstream Applications, 2021.
[19] J. Ho, T. Salimans, A. A. Gritsenko, W. Chan, M. Norouzi, and D. J. Fleet. Video diffusion models. In Proc. ICLR Workshop on Deep Generative Models for Highly Structured Data, 2022.
[20] C.-W. Huang, J. H. Lim, and A. C. Courville. A variational perspective on diffusion-based generative models and score matching. In Proc. NeurIPS, 2021. [21] L. Huang, J. Qin, Y. Zhou, F. Zhu, L. Liu, and L. Shao. Normalization techniques in training DNNs: Methodology, analysis and application. CoRR, abs/2009.12836, 2020. [22] A. Hyvärinen. Estimation of non-normalized statistical models by score matching. Journal of Machine Learning Research, 6(24):695–709, 2005. [23] B. Jing, G. Corso, R. Berlinghieri, and T. Jaakkola. Subspace diffusion generative models. In Proc. ECCV, 2022. [24] A. Jolicoeur-Martineau, K. Li, R. Piché-Taillefer, T. Kachman, and I. Mitliagkas. Gotta go fast when generating data with score-based models. CoRR, abs/2105.14080, 2021. [25] T. Karras, M. Aittala, J. Hellsten, S. Laine, J. Lehtinen, and T. Aila. Training generative adversarial networks with limited data. In Proc. NeurIPS, 2020. [26] T. Karras, M. Aittala, S. Laine, E. Härkönen, J. Hellsten, J. Lehtinen, and T. Aila. Alias-free generative adversarial networks. In Proc. NeurIPS, 2021. [27] T. Karras, S. Laine, and T. Aila. A style-based generator architecture for generative adversarial networks. In Proc. CVPR, 2018.
[28] Z. Kong, W. Ping, J. Huang, K. Zhao, and B. Catanzaro. DiffWave: A versatile diffusion model for audio synthesis. In Proc. ICLR, 2021. [29] A. Krizhevsky. Learning multiple layers of features from tiny images. Technical report, University of Toronto, 2009. [30] J. Lehtinen, J. Munkberg, J. Hasselgren, S. Laine, T. Karras, M. Aittala, and T. Aila. Noise2Noise: Learning image restoration without clean data. In Proc. ICML, 2018. [31] L. Liu, Y. Ren, Z. Lin, and Z. Zhao. Pseudo numerical methods for diffusion models on manifolds. In Proc. ICLR, 2022.
[32] C. Lu, Y. Zhou, F. Bao, J. Chen, C. Li, and J. Zhu. DPM-Solver: A fast ODE solver for diffusion probabilistic model sampling in around 10 steps. In Proc. NeurIPS, 2022. [33] E. Luhman and T. Luhman. Knowledge distillation in iterative generative models for improved sampling speed. CoRR, abs/2101.02388, 2021. [34] P. Mishkin, L. Ahmad, M. Brundage, G. Krueger, and G. Sastry. DALL·E 2 preview – risks and limitations. OpenAI, 2022.
[35] E. Nachmani and S. Dovrat. Zero-shot translation using diffusion models. CoRR, abs/2111.01471, 2021.
11


[36] A. Nichol, P. Dhariwal, A. Ramesh, P. Shyam, P. Mishkin, B. McGrew, I. Sutskever, and M. Chen. GLIDE: Towards photorealistic image generation and editing with text-guided diffusion models. In Proc. ICML, 2022. [37] A. Q. Nichol and P. Dhariwal. Improved denoising diffusion probabilistic models. In Proc. ICML, volume 139, pages 8162–8171, 2021. [38] V. Popov, I. Vovk, V. Gogoryan, T. Sadekova, and M. Kudinov. Grad-TTS: A diffusion probabilistic model for text-to-speech. In Proc. ICML, volume 139, pages 8599–8608, 2021. [39] K. Preechakul, N. Chatthee, S. Wizadwongsa, and S. Suwajanakorn. Diffusion autoencoders: Toward a meaningful and decodable representation. In Proc. CVPR, 2022. [40] A. Ramesh, P. Dhariwal, A. Nichol, C. Chu, and M. Chen. Hierarchical text-conditional image generation with CLIP latents. Technical report, OpenAI, 2022. [41] A. J. Roberts. Modify the improved Euler scheme to integrate stochastic differential equations. CoRR, abs/1210.0933, 2012. [42] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer. High-resolution image synthesis with latent diffusion models. In Proc. CVPR, 2022. [43] C. Saharia, W. Chan, H. Chang, C. A. Lee, J. Ho, T. Salimans, D. J. Fleet, and M. Norouzi. Palette: Image-to-image diffusion models. In Proc. SIGGRAPH, 2022. [44] T. Salimans and J. Ho. Progressive distillation for fast sampling of diffusion models. In Proc. ICLR, 2022. [45] A. Sauer, K. Schwarz, and A. Geiger. StyleGAN-XL: Scaling StyleGAN to large diverse datasets. In Proc. SIGGRAPH, 2022.
[46] J. Sohl-Dickstein, E. Weiss, N. Maheswaranathan, and S. Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In Proc. ICML, pages 2256–2265, 2015. [47] J. Song, C. Meng, and S. Ermon. Denoising diffusion implicit models. In Proc. ICLR, 2021. [48] Y. Song and S. Ermon. Generative modeling by estimating gradients of the data distribution. In Proc. NeurIPS, 2019.
[49] Y. Song, J. Sohl-Dickstein, D. P. Kingma, A. Kumar, S. Ermon, and B. Poole. Score-based generative modeling through stochastic differential equations. In Proc. ICLR, 2021. [50] E. Süli and D. F. Mayers. An Introduction to Numerical Analysis. Cambridge University Press, 2003. [51] C. Szegedy, V. Vanhoucke, S. Ioffe, J. Shlens, and Z. Wojna. Rethinking the Inception architecture for computer vision. In Proc. CVPR, 2016. [52] M. Tancik, P. P. Srinivasan, B. Mildenhall, S. Fridovich-Keil, N. Raghavan, U. Singhal, R. Ramamoorthi, J. T. Barron, and R. Ng. Fourier features let networks learn high frequency functions in low dimensional domains. In Proc. NeurIPS, 2020. [53] A. Vahdat, K. Kreis, and J. Kautz. Score-based generative modeling in latent space. In Proc. NeurIPS, 2021. [54] P. Vincent. A connection between score matching and denoising autoencoders. Neural Computation, 23(7):1661–1674, 2011. [55] D. Watson, W. Chan, J. Ho, and M. Norouzi. Learning fast samplers for diffusion models by differentiating through sample quality. In Proc. ICLR, 2022. [56] D. Watson, J. Ho, M. Norouzi, and W. Chan. Learning to efficiently sample from diffusion probabilistic models. CoRR, abs/2106.03802, 2021. [57] J. Wolleb, R. Sandkühler, F. Bieder, P. Valmaggia, and P. C. Cattin. Diffusion models for implicit image segmentation ensembles. In Medical Imaging with Deep Learning, 2022.
[58] Q. Zhang and Y. Chen. Diffusion normalizing flow. In Proc. NeurIPS, 2021. [59] Q. Zhang and Y. Chen. Fast sampling of diffusion models with exponential integrator. CoRR, abs/2204.13902, 2022.
12


Appendices
A Additional results
Figure 6 presents generated images for class-conditional ImageNet-64 [8] using the pre-trained ADM model by Dhariwal and Nichol [9]. The original DDIM [47] and iDDPM [37] samplers are compared to ours in both deterministic and stochastic settings (Sections 3 and 4). Figure 7 shows the corresponding results that we obtain by training the model from scratch using our improved training configuration (Section 5).
The original samplers and training configurations by Song et al. [49] are compared to ours in Figures 8 and 9 (unconditional CIFAR-10 [29]), Figure 10 (class-conditional CIFAR-10), and Figure 11 (FFHQ [27] and AFHQv2 [7]). For ease of comparison, the same latent codes x0 are used for each dataset/scenario across different training configurations and ODE choices. Figure 12 shows generated image quality with various NFE when using deterministic sampling.
Tables 3 and 4 summarize the numerical results on deterministic and stochastic sampling methods in various datasets, previously shown as functions of NFE in Figures 2 and 4.
B Derivation of formulas
B.1 Original ODE / SDE formulation from previous work
Song et al. [49] define their forward SDE (Eq. 5 in [49]) as
dx = f (x, t) dt + g(t) dωt, (9)
where ωt is the standard Wiener process and f (·, t) : Rd → Rd and g(·) : R → R are the drift and diffusion coefficients, respectively, where d is the dimensionality of the dataset. These coefficients are selected differently for the variance preserving (VP) and variance exploding (VE) formulations, and f (·) is always of the form f (x, t) = f (t) x, where f (·) : R → R. Thus, the SDE can be equivalently written as
dx = f (t) x dt + g(t) dωt. (10)
The perturbation kernels of this SDE (Eq. 29 in [49]) have the general form
p0t
(x(t) | x(0)) = N (x(t); s(t) x(0), s(t)2 σ(t)2 I), (11)
where N (x; μ, Σ) denotes the probability density function of N (μ, Σ) evaluated at x,
s(t) = exp
(∫ t
0
f (ξ) dξ
)
, and σ(t) =
√
∫t
0
g(ξ)2
s(ξ)2 dξ. (12)
The marginal distribution pt(x) is obtained by integrating the perturbation kernels over x(0):
pt(x) =
∫
Rd
p0t(x | x0) pdata(x0) dx0. (13)
Song et al. [49] define the probability flow ODE (Eq. 13 in [49]) so that it obeys this same pt(x):
dx = [f (t) x − 1
2 g(t)2 ∇x log pt(x)] dt. (14)
B.2 Our ODE formulation (Eq. 1 and Eq. 4)
The original ODE formulation (Eq. 14) is built around the functions f and g that correspond directly to specific terms that appear in the formula; the properties of the marginal distribution (Eq. 12) can only be derived indirectly based on these functions. However, f and g are of little practical interest in themselves, whereas the marginal distributions are of utmost importance in terms of training the model in the first place, bootstrapping the sampling process, and understanding how the ODE behaves in practice. Given that the idea of the probability flow ODE is to match a particular set of marginal
13


Deterministic, Original sampler (DDIM) Deterministic, Our sampler (Alg. 1)
Agaric Daisy Valley Pizza Balloon Beagle Ostrich
Agaric Daisy Valley Pizza Balloon Beagle Ostrich
FID 2.91 NFE 250 FID 2.66 NFE 79
Stochastic, Original sampler (iDDPM) Stochastic, Our sampler (Alg. 2)
Agaric Daisy Valley Pizza Balloon Beagle Ostrich
Agaric Daisy Valley Pizza Balloon Beagle Ostrich
FID 2.01 NFE 512 FID 1.55 NFE 1023
Figure 6: Results for different samplers on class-conditional ImageNet [8] at 64×64 resolution, using the pre-trained model by Dhariwal and Nichol [9]. The cases correspond to dots in Figures 2c and 4c.
14


Deterministic, Our sampler & training configuration
Agaric Daisy Valley Pizza Balloon Beagle Ostrich
FID 2.23 NFE 79
Stochastic, Our sampler & training configuration
Agaric Daisy Valley Pizza Balloon Beagle Ostrich
FID 1.36 NFE 511
Figure 7: Results for our training configuration on class-conditional ImageNet [8] at 64×64 resolution, using our deterministic and stochastic samplers.
15


Deterministic, Original sampler (p.flow), VP Deterministic, Original sampler (p.flow), VE
FID 2.94 NFE 256 FID 5.45 NFE 8192
Deterministic, Our sampler (Alg. 1), VP Deterministic, Our sampler (Alg. 1), VE
FID 3.01 NFE 35 FID 3.82 NFE 27
Stochastic, Original sampler (E–M), VP Stochastic, Original sampler (P–C), VE
FID 2.55 NFE 1024 FID 2.46 NFE 2048
Stochastic, Our sampler (Alg. 2), VP Stochastic, Our sampler (Alg. 2), VE
FID 2.27 NFE 511 FID 2.23 NFE 2047
Figure 8: Results for different samplers on unconditional CIFAR-10 [29] at 32×32 resolution, using the pre-trained models by Song et al. [49]. The cases correspond to dots in Figures 2a,b and 4a,b.
16


Original training (config A), VP Original training (config A), VE
FID 3.01 NFE 35 FID 3.77 NFE 35
Our training (config F), VP Our training (config F), VE
FID 1.97 NFE 35 FID 1.98 NFE 35
Figure 9: Results for different training configurations on unconditional CIFAR-10 [29] at 32×32 resolution, using our deterministic sampler with the same set of latent codes (x0) in each case.
17


Original training (config A), VP Original training (config A), VE
Truck Ship Horse Frog Dog Deer Cat Bird Car Plane
Truck Ship Horse Frog Dog Deer Cat Bird Car Plane
FID 2.48 NFE 35 FID 3.11 NFE 35
Our training (config F), VP Our training (config F), VE
Truck Ship Horse Frog Dog Deer Cat Bird Car Plane
Truck Ship Horse Frog Dog Deer Cat Bird Car Plane
FID 1.79 NFE 35 FID 1.79 NFE 35
Figure 10: Results for different training configurations on class-conditional CIFAR-10 [29] at 32×32 resolution, using our deterministic sampler with the same set of latent codes (x0) in each case.
18


FFHQ, Original training (config A), VP FFHQ, Original training (config A), VE
FID 3.39 NFE 79 FID 25.95 NFE 79
FFHQ, Our training (config F), VP FFHQ, Our training (config F), VE
FID 2.39 NFE 79 FID 2.53 NFE 79
AFHQv2, Original training (config A), VP AFHQv2, Original training (config A), VE
FID 2.58 NFE 79 FID 18.52 NFE 79
AFHQv2, Our training (config F), VP AFHQv2, Our training (config F), VE
FID 1.96 NFE 79 FID 2.16 NFE 79
Figure 11: Results for different training configurations on FFHQ [27] and AFHQv2 [7] at 64×64 resolution, using our deterministic sampler with the same set of latent codes (x0) in each case.
19


Class-conditional ImageNet-64, Pre-trained Class-conditional CIFAR-10, Our training, VP
FID 87.16 12.39 3.56 2.66 85.46 35.47 14.32 6.72 4.22 2.48 1.86 1.79 NFE 7 11 19 79 7 9 11 13 15 19 27 35
Unconditional FFHQ, Our training, VP Unconditional AFHQv2, Our Training, VP
FID 142.34 29.22 5.13 2.39 61.57 13.68 3.00 1.96 NFE 7 11 19 79 7 11 19 79
Figure 12: Image quality and FID as a function of NFE using our deterministic sampler. At 32×32 resolution, reasonable image quality is reached around NFE = 13, but FID keeps improving until NFE = 35. At 64×64 resolution, reasonable image quality is reached around NFE = 19, but FID keeps improving until NFE = 79.
20


Table 3: Evaluation of our improvements to deterministic sampling. The values correspond to the curves shown in Figure 2. We summarize each curve with two key values: the lowest observed FID for any NFE (“FID”), and the lowest NFE whose FID is within 3% of the lowest FID (“NFE”). The values marked with “–” are identical to the ones above them, because our sampler uses the same σ(t) and s(t) as DDIM.
Unconditional CIFAR-10 at 32×32 Class-conditional VP VE ImageNet-64 Sampling method FID ↓ NFE ↓ FID ↓ NFE ↓ FID ↓ NFE ↓ Original sampler [49, 9] 2.85 256 5.45 8192 2.85 250 Our Algorithm 1 2.79 512 4.78 8192 2.73 384 + Heun & our ti 2.88 255 4.23 191 2.64 79 + Our σ(t) & s(t) 2.93 35 3.73 27 – 
Black-box RK45 2.94 115 3.69 93 2.66 131
Table 4: Evaluation and ablations of our improvements to stochastic sampling. The values correspond to the curves shown in Figure 4.
Unconditional CIFAR-10 at 32×32 Class-conditional VP VE ImageNet-64 Sampling method FID ↓ NFE ↓ FID ↓ NFE ↓ FID ↓ NFE ↓ Deterministic baseline (Alg. 1) 2.93 35 3.73 27 2.64 79 Alg. 2, Stmin,tmax = [0, ∞], Snoise = 1 2.69 95 2.97 383 1.86 383
Alg. 2, Stmin,tmax = [0, ∞] 2.54 127 2.51 511 1.63 767 Alg. 2, Snoise = 1 2.52 95 2.84 191 1.84 255 Alg. 2, Optimal settings 2.27 383 2.23 767 1.55 511 Previous work [49, 9] 2.55 768 2.46 1024 2.01 384
distributions, it makes sense to treat the marginal distributions as first-class citizens and define the ODE directly based on σ(t) and s(t), eliminating the need for f (t) and g(t).
Let us start by expressing the marginal distribution of Eq. 13 in closed form:
pt(x) =
∫
Rd
p0t(x | x0) pdata(x0) dx0 (15)
=
∫
Rd
pdata(x0)
[
N
(x; s(t) x0, s(t)2 σ(t)2 I)
]
dx0 (16)
=
∫
Rd
pdata(x0)
[
s(t)−d N (x/s(t); x0, σ(t)2 I)
]
dx0 (17)
= s(t)−d
∫
Rd
pdata(x0) N (x/s(t); x0, σ(t)2 I) dx0 (18)
= s(t)−d [
pdata ∗ N (0, σ(t)2 I)
]
(x/s(t)), (19)
where pa ∗ pb denotes the convolution of probability density functions pa and pb. The expression inside the brackets corresponds to a mollified version of pdata obtained by adding i.i.d. Gaussian noise to the samples. Let us denote this distribution by p(x; σ):
p(x; σ) = pdata ∗ N (0, σ(t)2 I) and pt(x) = s(t)−d p(x/s(t); σ(t)). (20)
We can now express the probability flow ODE (Eq. 14) using p(x; σ) instead of pt(x):
dx = [f (t)x − 1
2 g(t)2 ∇x log [pt(x)]] dt (21)
=
[f (t)x − 1
2 g(t)2 ∇x log [s(t)−d p(x/s(t); σ(t))]] dt (22)
=
[f (t)x − 1
2 g(t)2 [∇x log s(t)−d + ∇x log p(x/s(t); σ(t))]] dt (23)
=
[f (t)x − 1
2 g(t)2 ∇x log p(x/s(t); σ(t))] dt. (24)
21


Next, let us rewrite f (t) in terms of s(t) based on Eq. 12:
exp
(∫ t
0
f (ξ) dξ
)
= s(t) (25)
∫t
0
f (ξ) dξ = log s(t) (26)
d
[∫ t
0
f (ξ) dξ
]
/dt = d[ log s(t)]/dt (27)
f (t) = s ̇(t)/s(t). (28)
Similarly, we can also rewrite g(t) in terms of σ(t):
√
∫t
0
g(ξ)2
s(ξ)2 dξ = σ(t) (29)
∫t
0
g(ξ)2
s(ξ)2 dξ = σ(t)2 (30)
d
[∫ t
0
g(ξ)2
s(ξ)2 dξ
]
/dt = d[σ(t)2]/dt (31)
g(t)2/s(t)2 = 2 σ ̇ (t) σ(t) (32)
g(t)/s(t) = √2 σ ̇ (t) σ(t) (33)
g(t) = s(t) √2 σ ̇ (t) σ(t). (34)
Finally, substitute f (Eq. 28) and g (Eq. 34) into the ODE of Eq. 24:
dx = [[f (t)] x − 1
2 [g(t)]2 ∇x log p(x/s(t); σ(t))] dt (35)
=
[
[s ̇(t)/s(t)] x − 1
2
[
s(t)√2 σ ̇ (t) σ(t)
]2
∇x log p(x/s(t); σ(t))
]
dt (36)
=
[
[s ̇(t)/s(t)] x − 1
2
[
2 s(t)2 σ ̇ (t) σ(t)
]
∇x log p(x/s(t); σ(t))
]
dt (37)
=
[ s ̇(t)
s(t) x − s(t)2 σ ̇ (t) σ(t) ∇x log p
(x
s(t) ; σ(t)
)]
dt. (38)
Thus we have obtained Eq. 4 in the main paper, and Eq. 1 is recovered by setting s(t) = 1:
dx = −σ ̇ (t) σ(t) ∇x log p(x; σ(t)) dt. (39)
Our formulation (Eq. 4) highlights the fact that every realization of the probability flow ODE is simply a reparameterization of the same canonical ODE; changing σ(t) corresponds to reparameterizing t, whereas changing s(t) corresponds to reparameterizing x.
B.3 Denoising score matching (Eq. 2 and Eq. 3)
For the sake of completeness, we derive the connection between score matching and denoising for a finite dataset. For a more general treatment and further background on the topic, see Hyvärinen [22] and Vincent [54].
Let us assume that our training set consists of a finite number of samples {y1, . . . , yY }. This implies pdata(x) is represented by a mixture of Dirac delta distributions:
pdata(x) = 1
Y
Y
∑
i=1
δ
(x − yi
), (40)
22


which allows us to also express p(x; σ) in closed form based on Eq. 20:
p(x; σ) = pdata ∗ N (0, σ(t)2 I) (41)
=
∫
Rd
pdata(x0) N (x; x0, σ2 I) dx0 (42)
=
∫
Rd
[
1 Y
Y
∑
i=1
δ
(x0 − yi
)
]
N
(x; x0, σ2 I) dx0 (43)
=1
Y
Y
∑
i=1
∫
Rd
N
(x; x0, σ2 I) δ(x0 − yi
) dx0 (44)
=1
Y
Y
∑
i=1
N
(x; yi, σ2 I). (45)
Let us now consider the denoising score matching loss of Eq. 2. By expanding the expectations, we can rewrite the formula as an integral over the noisy samples x:
L(D; σ) = Ey∼pdata En∼N (0,σ2I)
∥
∥D(y + n; σ) − y∥
∥
2
2 (46)
= Ey∼pdata Ex∼N (y,σ2I)
∥
∥D(x; σ) − y∥
∥
2
2 (47)
= Ey∼pdata
∫
Rd
N (x; y, σ2 I) ∥
∥D(x; σ) − y∥
∥
2
2 dx (48)
=1
Y
Y
∑
i=1
∫
Rd
N (x; yi, σ2 I) ∥
∥D(x; σ) − yi
∥ ∥
2
2 dx (49)
=
∫
Rd
1 Y
Y
∑
i=1
N (x; yi, σ2 I) ∥
∥D(x; σ) − yi
∥ ∥
2 2
} {{ }
=: L(D;x,σ)
dx. (50)
Eq. 50 means that we can minimize L(D; σ) by minimizing L(D; x, σ) independently for each x:
D(x; σ) = arg minD(x;σ) L(D; x, σ). (51)
This is a convex optimization problem; its solution is uniquely identified by setting the gradient w.r.t. D(x; σ) to zero:
0 = ∇D(x;σ)
[
L(D; x, σ)
]
(52)
0 = ∇D(x;σ)
[
1 Y
Y
∑
i=1
N (x; yi, σ2 I) ∥
∥D(x; σ) − yi
∥ ∥
2 2
]
(53)
0=
Y
∑
i=1
N (x; yi, σ2 I) ∇D(x;σ)
[∥
∥D(x; σ) − yi
∥ ∥
2 2
]
(54)
0=
Y
∑
i=1
N (x; yi, σ2 I)
[
2 D(x; σ) − 2 yi
]
(55)
0=
[Y ∑
i=1
N (x; yi, σ2 I)
]
D(x; σ) −
Y
∑
i=1
N (x; yi, σ2 I) yi (56)
D(x; σ) =
∑
i N (x; yi, σ2 I) yi
∑
i N (x; yi, σ2 I) , (57)
which gives a closed-form solution for the ideal denoiser D(x; σ). Note that Eq. 57 is feasible to compute in practice for small datasets — we show the results for CIFAR-10 in Figure 1b.
23


Next, let us consider the score of the distribution p(x; σ) defined in Eq. 45:
∇x log p(x; σ) = ∇xp(x; σ)
p(x; σ) (58)
=
∇x
[1 Y
∑
i N (x; yi, σ2 I)
]
[1 Y
∑
i N (x; yi, σ2 I)
] (59)
=
∑
i ∇xN (x; yi, σ2 I)
∑
i N (x; yi, σ2 I) . (60)
We can simplify the numerator of Eq. 60 further:
∇xN (x; yi, σ2 I) = ∇x
[
(2πσ2)− d
2 exp ‖x − yi‖22
−2 σ2
]
(61)
=
(2πσ2)− d
2 ∇x
[
exp ‖x − yi‖22
−2 σ2
]
(62)
=
[
(2πσ2)− d
2 exp ‖x − yi‖22
−2 σ2
]
∇x
[
‖x − yi‖22 −2 σ2
]
(63)
= N (x; yi, σ2 I) ∇x
[
‖x − yi‖22 −2 σ2
]
(64)
= N (x; yi, σ2 I)
[yi − x σ2
]
. (65)
Let us substitute the result back to Eq. 60:
∇x log p(x; σ) =
∑
i ∇xN (x; yi, σ2 I)
∑
i N (x; yi, σ2 I) (66)
=
∑
i N (x; yi, σ2 I)
[ yi−x σ2
]
∑
i N (x; yi, σ2 I) (67)
=
(∑
i N (x; yi, σ2 I)yi
∑
i N (x; yi, σ2 I) − x
)
/σ2. (68)
Notice that the fraction in Eq. 68 is identical to Eq. 57. We can thus equivalently write Eq. 68 as
∇x log p(x; σ) = (D(x; σ) − x)/σ2, (69)
which matches Eq. 3 in the main paper.
B.4 Evaluating our ODE in practice (Algorithm 1)
Let us consider x to be a scaled version of an original, non-scaled variable ˆx and substitute x = s(t) ˆx into the score term that appears in our scaled ODE (Eq. 4):
∇x log p(x/s(t); σ(t)) (70)
= ∇[s(t)xˆ] log p([s(t) ˆx]/s(t); σ(t)) (71)
= ∇s(t)ˆx log p(xˆ; σ(t)) (72)
=1
s(t) ∇ˆx log p(xˆ; σ(t)). (73)
We can further rewrite this with respect to D(·) using Eq. 3:
∇x log p(x/s(t); σ(t)) = 1
s(t)σ(t)2
(
D
(ˆx; σ(t)) − ˆx
)
. (74)
24


Let us now substitute Eq. 74 into Eq. 4, approximating the ideal denoiser D(·) with our trained model Dθ (·):
dx =
[
s ̇(t) x/s(t) − s(t)2 σ ̇ (t) σ(t)
[1
s(t)σ(t)2
(
Dθ
(ˆx; σ(t)) − ˆx
)]]
dt (75)
=
[ s ̇(t)
s(t) x − σ ̇ (t)s(t)
σ(t)
(
Dθ
(ˆx; σ(t)) − ˆx
)]
dt. (76)
Finally, backsubstitute ˆx = x/s(t):
dx =
[ s ̇(t)
s(t) x − σ ̇ (t)s(t)
σ(t)
(
Dθ
([ˆx]; σ(t)) − [ˆx]
)]
dt (77)
=
[ s ̇(t)
s(t) x − σ ̇ (t)s(t)
σ(t)
(
Dθ
([x/s(t)]; σ(t)) − [x/s(t)]
)]
dt (78)
=
[ s ̇(t)
s(t) x − σ ̇ (t)s(t)
σ(t) Dθ
(x/s(t); σ(t)) + σ ̇ (t)
σ(t) x
]
dt (79)
=
[( σ ̇ (t)
σ(t) + s ̇(t)
s(t)
)
x − σ ̇ (t)s(t)
σ(t) Dθ
(x/s(t); σ(t))
]
dt. (80)
We can equivalenty write Eq. 80 as
dx/dt =
( σ ̇ (t)
σ(t) + s ̇(t)
s(t)
)
x − σ ̇ (t)s(t)
σ(t) Dθ
(x
s(t) ; σ(t)
)
, (81)
matching lines 4 and 7 of Algorithm 1.
B.5 Our SDE formulation (Eq. 6)
We derive the SDE of Eq. 6 by the following strategy:
• The desired marginal densities p(x; σ(t)) are convolutions of the data density pdata and an isotropic Gaussian density with standard deviation σ(t) (see Eq. 20). Hence, considered as a function of the time t, the density evolves according to a heat diffusion PDE with time-varying diffusivity. As a first step, we find this PDE.
• We then use the Fokker–Planck equation to recover a family of SDEs for which the density evolves according to this PDE. Eq. 6 is obtained from a suitable parametrization of this family.
B.5.1 Generating the marginals by heat diffusion
We consider the time evolution of a probability density q(x, t). Our goal is to find a PDE whose solution with the initial value q(x, 0) := pdata(x) is q(x, t) = p(x, σ(t)). That is, the PDE should reproduce the marginals we postulate in Eq. 20.
The desired marginals are convolutions of pdata with isotropic normal distributions of time-varying standard deviation σ(t), and as such, can be generated by the heat equation with time-varying diffusivity κ(t). The situation is most conveniently analyzed in the Fourier domain, where the marginal densities are simply pointwise products of a Gaussian function and the transformed data density. To find the diffusivity that induces the correct standard deviations, we first write down the heat equation PDE:
∂q(x, t)
∂t = κ(t)∆xq(x, t). (82)
The Fourier transformed counterpart of Eq. 82, where the transform is taken along the x-dimension, is given by
∂qˆ(ν, t)
∂t = −κ(t)|ν|2qˆ(ν, t). (83)
The target solution q(x, t) and its Fourier transform qˆ(ν, t) are given by Eq. 20:
q(x, t) = p(x; σ(t)) = pdata(x) ∗ N (0, σ(t)2 I) (84)
qˆ(ν, t) = pˆdata(ν) exp
(
−1
2 |ν|2 σ(t)2)
. (85)
25


Differentiating the target solution along the time axis, we have
∂qˆ(ν, t)
∂t = −σ ̇ (t)σ(t) |ν|2 pˆdata(ν) exp
(
−1
2 |ν|2 σ(t)2)
(86)
= −σ ̇ (t)σ(t) |ν|2 qˆ(ν, t). (87)
Eqs. 83 and 87 share the same left hand side. Equating them allows us to solve for κ(t) that generates the desired evolution:
−κ(t)|ν|2qˆ(ν, t) = −σ ̇ (t)σ(t) |ν|2 qˆ(ν, t) (88)
κ(t) = σ ̇ (t)σ(t). (89)
To summarize, the desired marginal densities corresponding to noise levels σ(t) are generated by the PDE
∂q(x, t)
∂t = σ ̇ (t)σ(t)∆xq(x, t) (90)
from the initial density q(x, 0) = pdata(x).
B.5.2 Derivation of our SDE
Given an SDE
dx = f (x, t) dt + g(x, t) dωt, (91)
the Fokker–Planck PDE describes the time evolution of its solution probability density r(x, t) as
∂r(x, t)
∂t = −∇x · (f (x, t) r(x, t)) + 1
2 ∇x∇x : (D(x, t) r(x, t)), (92)
where Dij = ∑
k gikgjk is the diffusion tensor. We consider the special case g(x, t) = g(t) I of
x-independent white noise addition, whereby the equation simplifies to
∂r(x, t)
∂t = −∇x · (f (x, t) r(x, t)) + 1
2 g(t)2 ∆xr(x, t). (93)
We are seeking an SDE whose solution density is described by the PDE in Eq. 90. Setting r(x, t) = q(x, t) and equating Eqs. 93 and 90, we find the sufficient condition that the SDE must satisfy
−∇x · (f (x, t) q(x, t)) + 1
2 g(t)2 ∆xq(x, t) = σ ̇ (t) σ(t) ∆xq(x, t) (94)
∇x · (f (x, t) q(x, t)) =
(1
2 g(t)2 − σ ̇ (t) σ(t)
)
∆xq(x, t). (95)
Any choice of functions f (x, t) and g(t) satisfying this equation constitute a sought after SDE. Let us now find a specific family of such solutions. The key idea is given by the identity ∇x · ∇x = ∆x. Indeed, if we set f (x, t) q(x, t) = υ(t) ∇xq(x, t) for any choice of υ(t), the term ∆xq(x, t) appears on both sides and cancels out:
∇x · (υ(t) ∇xq(x, t)) =
(1
2 g(t)2 − σ ̇ (t) σ(t)
)
∆xq(x, t) (96)
υ(t) ∆xq(x, t) =
(1
2 g(t)2 − σ ̇ (t) σ(t)
)
∆xq(x, t) (97)
υ(t) = 1
2 g(t)2 − σ ̇ (t) σ(t). (98)
The stated f (x, t) is in fact proportional to the score function, as the formula matches the gradient of the logarithm of the density:
f (x, t) = υ(t) ∇xq(x, t)
q(x, t) (99)
= υ(t) ∇x log q(x, t) (100)
=
(1
2 g(t)2 − σ ̇ (t) σ(t)
)
∇x log q(x, t). (101)
26


Substituting this back into Eq. 91 and writing p(x; σ(t)) in place of q(x, t), we recover a family of SDEs whose solution densities have the desired marginals with noise levels σ(t) for any choice of g(t):
dx =
(1
2 g(t)2 − σ ̇ (t) σ(t)
)
∇x log p(x; σ(t)) dt + g(t) dωt. (102)
The free parameter g(t) effectively specifies the rate of noise replacement at any given time instance. The special case choice of g(t) = 0 corresponds to the probability flow ODE. The parametrization by g(t) is not particularly intuitive, however. To obtain a more interpretable parametrization, we set g(t) = √2 β(t) σ(t), which yields the (forward) SDE of Eq. 6 in the main paper:
dx+ = −σ ̇ (t)σ(t)∇x log p(x; σ(t)) dt + β(t)σ(t)2∇x log p(x; σ(t)) dt + √2β(t)σ(t) dωt.
(103)
The noise replacement is now proportional to the standard deviation σ(t) of the noise, with the proportionality factor β(t). Indeed, expanding the score function in the middle term according to Eq. 3 yields β(t) [D(x; σ(t)) − x] dt, which changes x proportionally to the negative noise component; the stochastic term injects new noise at the same rate. Intuitively, scaling the magnitude of Langevin exploration according to the current noise standard deviation is a reasonable baseline, as the data manifold is effectively “spread out” by this amount due to the blurring of the density.
The reverse SDE used in denoising diffusion is simply obtained by applying the time reversal formula of Anderson [1] (as stated in Eq. 6 of Song et al. [49]) on Eq. 103; the entire effect of the reversal is a change of sign in the middle term.
The scaled generalization of the SDE can be derived using a similar approach as with the ODE previously. As such, the derivation is omitted here.
B.6 Our preconditioning and training (Eq. 8)
Following Eq. 2, the denoising score matching loss for a given denoiser Dθ on a given noise level σ is given by
L(Dθ; σ) = Ey∼pdata En∼N (0,σ2I)
∥
∥Dθ(y + n; σ) − y∥
∥
2
2. (104)
We obtain overall training loss by taking a weighted expectation of L(Dθ; σ) over the noise levels:
L(Dθ) = Eσ∼ptrain
[λ(σ) L(Dθ; σ)] (105)
= Eσ∼ptrain
[
λ(σ) Ey∼pdata En∼N (0,σ2I)
∥
∥Dθ(y + n; σ) − y∥
∥
2 2
]
(106)
= Eσ∼ptrain Ey∼pdata En∼N (0,σ2I)
[
λ(σ) ∥
∥Dθ(y + n; σ) − y∥
∥
2 2
]
(107)
= Eσ,y,n
[
λ(σ) ∥
∥Dθ(y + n; σ) − y∥
∥
2 2
]
, (108)
where the noise levels are distributed according to σ ∼ ptrain and weighted by λ(σ).
Using our definition of Dθ(·) from Eq. 7, we can further rewrite L(Dθ) as
Eσ,y,n
[
λ(σ)∥
∥cskip(σ)(y+n) + cout(σ)Fθ
(cin(σ)(y+n); cnoise(σ)) − y∥
∥
2 2
]
(109)
= Eσ,y,n
[
λ(σ)∥
∥cout(σ)Fθ
(cin(σ)(y+n); cnoise(σ)) − (y − cskip(σ)(y + n))∥
∥
2 2
]
(110)
= Eσ,y,n
[
λ(σ)cout(σ)2∥
∥Fθ
(cin(σ)(y+n); cnoise(σ)) − 1
cout (σ )
(y − cskip(σ)(y+n))∥
∥
2 2
]
(111)
= Eσ,y,n
[
w(σ) ∥
∥Fθ
(cin(σ)(y+n); cnoise(σ)) − Ftarget(y, n; σ)∥
∥
2 2
]
, (112)
which matches Eq. 8 and corresponds to traditional supervised training of Fθ using standard L2 loss with effective weight w(·) and target Ftarget(·) given by
w(σ) = λ(σ) cout(σ)2 and Ftarget(y, n; σ) = 1
cout (σ )
(y − cskip(σ)(y + n)), (113)
We can now derive formulas for cin(σ), cout(σ), cskip(σ), and λ(σ) from first principles, shown in the “Ours” column of Table 1.
27


First, we require the training inputs of Fθ(·) to have unit variance:
Vary,n
[cin(σ)(y + n)] = 1 (114)
cin(σ)2 Vary,n
[y + n] = 1 (115)
cin(σ)2(σ2
data + σ2) = 1 (116)
cin(σ) = 1/
√
σ2 + σ2
data. (117)
Second, we require the effective training target Ftarget to have unit variance:
Vary,n
[Ftarget(y, n; σ)] = 1 (118)
Vary,n
[1 cout (σ )
(y − cskip(σ)(y + n))
]
= 1 (119)
1
cout(σ)2 Vary,n
[y − cskip(σ)(y + n)] = 1 (120)
cout(σ)2 = Vary,n
[y − cskip(σ)(y + n)] (121)
cout(σ)2 = Vary,n
[
(1 − cskip(σ)) y + cskip(σ) n
]
(122)
cout(σ)2 = (1 − cskip(σ))2 σ2
data + cskip(σ)2 σ2. (123)
Third, we select cskip(σ) to minimize cout(σ), so that the errors of Fθ are amplified as little as possible:
cskip(σ) = arg mincskip(σ) cout(σ). (124)
Since cout(σ) ≥ 0, we can equivalently write
cskip(σ) = arg mincskip(σ) cout(σ)2. (125)
This is a convex optimization problem; its solution is uniquely identified by setting the derivative w.r.t. cskip(σ) to zero:
0 = d[cout(σ)2]/dcskip(σ) (126)
0=d
[
(1 − cskip(σ))2 σ2
data + cskip(σ)2 σ2]
/dcskip(σ) (127)
0 = σ2
data d
[
(1 − cskip(σ))2]
/dcskip(σ) + σ2 d[cskip(σ)2]/dcskip(σ) (128)
0 = σ2
data
[2 cskip(σ) − 2] + σ2 [2 cskip(σ)] (129)
0 = (σ2 + σ2
data
) cskip(σ) − σ2
data (130)
cskip(σ) = σ2
data/(σ2 + σ2
data
). (131)
We can now substitute Eq. 131 into Eq. 123 to complete the formula for cout(σ):
cout(σ)2 = (1 − [cskip(σ)])2 σ2
data + [cskip(σ)]2 σ2 (132)
cout(σ)2 =
(
1−
[ σ2
data
σ2 + σ2
data
])2
σ2
data +
[ σ2
data
σ2 + σ2
data
]2
σ2 (133)
cout(σ)2 =
[ σ2 σdata σ2 + σ2
data
]2
+
[ σ2
data σ
σ2 + σ2
data
]2
(134)
cout(σ)2 =
(σ2 σdata
)2 + (σ2
data σ)2
(σ2 + σ2
data
)2 (135)
cout(σ)2 = (σ · σdata)2 (σ2 + σ2
data
)
(σ2 + σ2
data
)2 (136)
cout(σ)2 = (σ · σdata)2
σ2 + σ2
data
(137)
cout(σ) = σ · σdata
/
√
σ2 + σ2
data. (138)
28


Fourth, we require the effective weight w(σ) to be uniform across noise levels:
w(σ) = 1 (139)
λ(σ) cout(σ)2 = 1 (140)
λ(σ) = 1/cout(σ)2 (141)
λ(σ) = 1/
[ σ · σdata
√σ2 + σ2
data
]2
(142)
λ(σ) = 1/
[ (σ · σdata)2 σ2 + σ2
data
]
(143)
λ(σ) = (σ2 + σ2
data
)/(σ · σdata)2. (144)
We follow previous work and initialize the output layer weights to zero. Consequently, upon initialization Fθ(·) = 0 and the expected value of the loss at each noise level is 1. This can be seen by substituting the choices of λ(σ) and cskip(σ) into Eq. 109, considered at a fixed σ:
Ey,n
[
λ(σ)∥
∥cskip(σ)(y+n) + cout(σ)Fθ
(cin(σ)(y+n); cnoise(σ)) − y∥
∥
2 2
]
(145)
= Ey,n
[ σ2 + σ2
data
(σ · σdata)2
∥ ∥ ∥ ∥
σ2
data
σ2 + σ2
data
(y+n) − y
∥ ∥ ∥ ∥
2
2
]
(146)
= Ey,n
[ σ2 + σ2
data
(σ · σdata)2
∥ ∥ ∥ ∥
σ2
datan − σ2y σ2 + σ2
data
∥ ∥ ∥ ∥
2
2
]
(147)
= Ey,n
[1
σ2 + σ2
data
∥ ∥ ∥ ∥
σdata
σ n− σ
σdata
y
∥ ∥ ∥ ∥
2
2
]
(148)
=1
σ2 + σ2
data
Ey,n
[ σ2
data
σ2 〈n, n〉 + σ2
σ2
data
〈y, y〉 − 2〈y, n〉
]
(149)
=1
σ2 + σ2
data
[ σ2
data
σ2 Var(n)
} {{ }
=σ2
+ σ2
σ2
data
Var(y)
} {{ }
=σ2
data
−2 Cov(y, n)
} {{ }
=0
]
(150)
= 1 (151)
C Reframing previous methods in our framework
In this section, we derive the formulas shown in Table 1 for previous methods, discuss the corresponding original samplers and pre-trained models, and detail the practical considerations associated with using them in our framework.
In practice, the original implementations of these methods differ considerably in terms of the definitions of model inputs and outputs, dynamic range of image data, scaling of x, and interpretation of σ. We eliminate this variation by standardizing on a unified setup where the model always matches our definition of Fθ, image data is always represented in the continuous range [−1, 1], and the details of x and σ are always in agreement with Eq. 4.
We minimize the accumulation of floating point round-off errors by always executing Algorithms 1 and 2 at double precision (float64). However, we still execute the network Fθ(·) at single precision (float32) to minimize runtime and remain faithful to previous work in terms of network architecture.
C.1 Variance preserving formulation
C.1.1 VP sampling
Song et al. [49] define the VP SDE (Eq. 32 in [49]) as
dx = − 1
2
(
βmin + t (βmax − βmin
)
)
x dt +
√
βmin + t (βmax − βmin
) dωt, (152)
29


which matches Eq. 10 with the following choices for f and g:
f (t) = − 1
2 β(t), g(t) = √β(t), and β(t) = (βmax − βmin
) t + βmin. (153)
Let α(t) denote the integral of β(t):
α(t) =
∫t
0
β(ξ) dξ (154)
=
∫t
0
[
(βmax − βmin
) ξ + βmin
]
dξ (155)
=1
2
(βmax − βmin
) t2 + βmin t (156)
=1
2 βd t2 + βmin t, (157)
where βd = βmax − βmin. We can now obtain the formula for σ(t) by substituting Eq. 153 into Eq. 12:
σ(t) =
√ √ √ √
∫t
0
[g(ξ)]2
[s(ξ)]2 dξ (158)
=
√ √ √ √
∫t
0
[√β(ξ)]2
[1/
√
eα(ξ)]2 dξ (159)
=
√
∫t
0
β(ξ)
1/eα(ξ) dξ (160)
=
√
∫t
0
α ̇ (ξ) eα(ξ) dξ (161)
=
√
eα(t) − eα(0) (162)
=
√
e
1
2 βdt2+βmint − 1, (163)
which matches the “Schedule” row of Table 1. Similarly for s(t):
s(t) = exp
(∫ t
0
[f (ξ)] dξ
)
(164)
= exp
(∫ t
0
[− 1
2 β(ξ)] dξ
)
(165)
= exp
(
−1
2
[∫ t
0
β(ξ) dξ
])
(166)
= exp (− 1
2 α(t)) (167)
= 1/
√
eα(t) (168)
= 1/
√
e
1
2 βdt2+βmint, (169)
which matches the “Scaling” row of Table 1. We can equivalently write Eq. 169 in a slightly simpler form by utilizing Eq. 163:
s(t) = 1/√σ(t)2 + 1. (170)
Song et al. [49] choose to distribute the sampling time steps {t0, . . . , tN−1} at uniform intervals within [ s, 1]. This corresponds to setting
ti<N = 1 + i
N−1 ( s − 1), (171)
which matches the “Time steps” row of Table 1.
Finally, Song et al. [49] set βmin = 0.1, βmax = 20, and s = 10−3 (Appendix C in [49]), and choose to represent images in the range [−1, 1]. These choices are readily compatible with our formulation and are reflected by the “Parameters” section of Table 1.
30


C.1.2 VP preconditioning
In the VP case, Song et al. [49] approximate the score of pt(x) of Eq. 13 as1
∇x log pt(x) ≈ − 1
σ ̄(t) Fθ
(x; (M −1)t)
} {{ }
score(x;Fθ ,t)
, (172)
where M = 1000, Fθ denotes the network, and σ ̄(t) corresponds to the standard deviation of the perturbation kernel of Eq. 11.
Let us expand the definitions of pt(x) and σ ̄(t) from Eqs. 20 and 11, respectively, and substitute x = s(t)xˆ to obtain the corresponding formula with respect to the non-scaled variable ˆx:
∇x log [p(x/s(t); σ(t))] ≈ − 1
[s(t)σ(t)] Fθ
(x; (M −1)t) (173)
∇[s(t)xˆ] log p([s(t) ˆx]/s(t); σ(t)) ≈ − 1
s(t)σ(t) Fθ
([s(t) xˆ]; (M −1)t) (174)
1
s(t) ∇xˆ log p(ˆx; σ(t)) ≈ − 1
s(t)σ(t) Fθ
(s(t) ˆx; (M −1)t) (175)
∇xˆ log p(ˆx; σ(t)) ≈ − 1
σ(t) Fθ
(s(t) ˆx; (M −1)t). (176)
We can now replace the left-hand side with Eq. 3 and expand the definition of s(t) from Eq. 170:
[(
D
(ˆx; σ(t)) − ˆx
)
/σ(t)2]
≈ −1
σ(t) Fθ
(s(t) ˆx; (M −1)t) (177)
D
(ˆx; σ(t)) ≈ ˆx − σ(t) Fθ
(s(t) ˆx; (M −1)t) (178)
D
(ˆx; σ(t)) ≈ ˆx − σ(t) Fθ
([
1
√
σ (t)2 +1
]
ˆx; (M −1)t
)
, (179)
which can be further expressed in terms of σ by replacing σ(t) → σ and t → σ−1(σ):
D(ˆx; σ) ≈ ˆx − σ Fθ
(1
√σ2+1 ˆx; (M −1) σ−1(σ)
)
. (180)
We adopt the right-hand side of Eq. 180 as the definition of Dθ, obtaining
Dθ(ˆx; σ) = 1 ·
}{{} cskip
ˆx − σ
}{{} cout
· Fθ
(1
√σ 2 +1
} {{ } cin
· ˆx; (M −1) σ−1(σ)
} {{ } cnoise
)
, (181)
where cskip, cout, cin, and cnoise match the “Network and preconditioning” section of Table 1.
C.1.3 VP training
Song et al. [49] define their training loss as2
Et∼U ( t,1),y∼pdata,n ̄ ∼N (0,I)
[∥
∥σ ̄(t) score (s(t) y + σ ̄(t)  ̄n; Fθ, t) +  ̄n∥
∥
2 2
]
, (182)
where the definition of score(·) is the same as in Eq. 172. Let us simplify the formula by substituting σ ̄(t) = s(t)σ(t) and  ̄n = n/σ(t), where n ∼ N (0, σ(t)2I):
Et,y,n ̄
[∥
∥s(t)σ(t) score (s(t) y + [s(t)σ(t)]  ̄n; Fθ, t) +  ̄n∥
∥
2 2
]
(183)
= Et,y,n
[∥
∥s(t)σ(t) score (s(t) y + s(t)σ(t) [n/σ(t)]; Fθ, t) + [n/σ(t)]∥
∥
2 2
]
(184)
= Et,y,n
[∥
∥s(t)σ(t) score (s(t) (y + n); Fθ, t) + n/σ(t)∥
∥
2 2
]
. (185)
We can express score(·) in terms of Dθ(·) by combining Eqs. 172, 170, and 74:
score (s(t) x; Fθ, t) = 1
s(t)σ(t)2
(
Dθ
(x; σ(t)) − x
)
. (186)
1 https://github.com/yang- song/score_sde_pytorch/blob/1618ddea340f3e4a2ed7852a0694a809775cf8d0/models/utils.py#L144
2 https://github.com/yang- song/score_sde_pytorch/blob/1618ddea340f3e4a2ed7852a0694a809775cf8d0/losses.py#L73
31


Substituting this back into Eq. 185 gives
Et,y,n
[∥
∥s(t)σ(t)
[1
s(t)σ(t)2
(
Dθ
(y + n; σ(t)) − (y + n)
)]
+1
σ(t) n∥
∥
2 2
]
(187)
= Et,y,n
[∥
∥1
σ(t)
(
Dθ
(y + n; σ(t)) − (y + n)
)
+1
σ(t) n∥
∥
2 2
]
(188)
= Et,y,n
[1 σ(t)2
∥
∥Dθ
(y + n; σ(t)) − y∥
∥
2 2
]
. (189)
We can further express this in terms of σ by replacing σ(t) → σ and t → σ−1(σ):
Eσ−1(σ)∼U ( t,1)
} {{ } ptrain
Ey,n
[1 σ2
}{{}
λ
∥
∥Dθ
(y + n; σ) − y∥
∥
2 2
]
, (190)
which matches Eq. 108 with the choices for ptrain and λ shown in the “Training” section of Table 1.
C.1.4 VP practical considerations
The pre-trained VP model that we use on CIFAR-10 corresponds to the “DDPM++ cont. (VP)” checkpoint3 provided by Song et al. [49]. It contains a total of 62 million trainable parameters and supports a continuous range of noise levels σ ∈ [σ( t), σ(1)] ≈ [0.001, 152], i.e., wider than our preferred sampling range [0.002, 80]. We import the model directly as Fθ(·) and run Algorithms 1 and 2 using the definitions in Table 1.
In Figure 2a, the differences between the original sampler (blue) and our reimplementation (orange) are explained by oversights in the implementation of Song et al. [49], also noted by JolicoeurMartineau et al. [24] (Appendix D in [24]). First, the original sampler employs an incorrect multiplier4 in the Euler step: it multiplies dx/dt by −1/N instead of ( s − 1)/(N − 1). Second, it either overshoots or undershoots on the last step by going from tN−1 = s to tN = s − 1/N , where tN < 0 when N < 1000. In practice, this means that the generated images contain noticeable noise that becomes quite severe with, e.g., N = 128. Our formulation avoids these issues, because the step sizes in Algorithm 1 are computed consistently from {ti} and tN = 0.
C.2 Variance exploding formulation
C.2.1 VE sampling in theory
Song et al. [49] define the VE SDE (Eq. 30 in [49]) as
dx = σmin
( σmax
σmin
)t√
2 log σmax
σmin
dωt, (191)
which matches Eq. 10 with
f (t) = 0, g(t) = σmin
√2 log σd σt
d, and σd = σmax/σmin. (192)
The VE formulation does not employ scaling, which can be easily seen from Eq. 12:
s(t) = exp
(∫ t
0
[f (ξ)] dξ
)
= exp
(∫ t
0
[0] dξ
)
= exp(0) = 1. (193)
3vp/cifar10_ddpmpp_continuous/checkpoint_8.pth, https://drive.google.com/drive/folders/1xYjVMx10N9ivQQBIsEoXEeu9nvSGTBrC
4 https://github.com/yang- song/score_sde_pytorch/blob/1618ddea340f3e4a2ed7852a0694a809775cf8d0/sampling.py#L182
32


Substituting Eq. 192 into Eq. 12 suggests the following form for σ(t):
σ(t) =
√ √ √ √
∫t
0
[g(ξ)]2
[s(ξ)]2 dξ (194)
=
√ √ √ √
∫t
0
[σmin
√2 log σd σdξ
]2
[1]2 dξ (195)
=
√
∫t
0
σ2min
[2 log σd
] [σd2ξ
] dξ (196)
= σmin
√
∫t
0
[
log (σd2
)
][
(σd2
)ξ ]
dξ (197)
= σmin
√
(σd2
)t − (σd2
)0 (198)
= σmin
√σd2t − 1. (199)
Eq. 199 is consistent with the perturbation kernel reported by Song et al. (Eq. 29 in [49]). However, we note that this does not fulfill their intended definition of σ(t) = σmin
( σmax σmin
)t (Appendix C in [49]).
C.2.2 VE sampling in practice
The original implementation5 of Song et al. [49] uses reverse diffusion predictor6 to integrate discretized reverse probability flow7 of discretized VE SDE8. Put together, these yield the following update rule for xi+1:
xi+1 = xi + 1
2
(σ ̄2
i − σ ̄2
i+1
) ∇x log p ̄i(x), (200)
where
σ ̄i<N = σmin
( σmax
σmin
)1−i/(N −1)
and σ ̄N = 0. (201)
Interestingly, Eq. 200 is identical to the Euler iteration of our ODE with the following choices:
s(t) = 1, σ(t) = √t, and ti = σ ̄2
i . (202)
These formulas match the “Sampling” section of Table 1, and their correctness can be verified by substituting them into line 5 of Algorithm 1:
xi+1 = xi + (ti+1 − ti) di (203)
= xi + (ti+1 − ti)
[( σ ̇ (t)
σ(t) + s ̇(t)
s(t)
)
x − σ ̇ (t)s(t)
σ(t) D
(x
s(t) ; σ(t)
)]
(204)
= xi + (ti+1 − ti)
[ σ ̇ (t)
σ(t) x − σ ̇ (t)
σ(t) D(x; σ(t))
]
(205)
= xi − (ti+1 − ti) σ ̇ (t) σ(t)
[ (
D
(x; σ(t)) − x
)
/σ(t)2
]
(206)
= xi − (ti+1 − ti) σ ̇ (t) σ(t) ∇x log p(x; σ(t)) (207)
= xi − (ti+1 − ti)
[1
2√t
][√t
]
∇x log p(x; σ(t)) (208)
= xi + 1
2 (ti − ti+1) ∇x log p(x; σ(t)) (209)
= xi + 1
2
(σ ̄2
i − σ ̄2
i+1
) ∇x log p(x; σ(t)), (210)
5https://github.com/yang-song/score_sde_pytorch
6 https://github.com/yang- song/score_sde_pytorch/blob/1618ddea340f3e4a2ed7852a0694a809775cf8d0/sampling.py#L191 7 https://github.com/yang- song/score_sde_pytorch/blob/1618ddea340f3e4a2ed7852a0694a809775cf8d0/sde_lib.py#L102
8 https://github.com/yang- song/score_sde_pytorch/blob/1618ddea340f3e4a2ed7852a0694a809775cf8d0/sde_lib.py#L246
33


which is made identical to Eq. 200 by the choice p ̄i(x) = p(x; σ(ti)).
Finally, Song et al. [49] set σmin = 0.01 and σmax = 50 for CIFAR-10 (Appendix C in [49]), and choose to represent their images in the range [0, 1] to match previous SMLD models. Since our standardized range [−1, 1] is twice as large, we must multiply σmin and σmax by 2× to compensate. The “Parameters” section of Table 1 reflects these adjusted values.
C.2.3 VE preconditioning
In the VE case, Song et al. [49] approximate the score of pt(x) of Eq. 13 directly as9
∇x log pt(x) ≈ F ̄θ
(x; σ(t)), (211)
where the network F ̄θ is designed to include additional pre-10 and11 postprocessing12 steps:
F ̄θ
(x; σ) = 1
σ Fθ
(2x−1; log(σ)). (212)
For consistency, we handle the pre- and postprocessing using {cskip, cout, cin, cnoise} as opposed to baking them into the network itself.
We cannot use Eqs. 211 and 212 directly in our framework, however, because they assume that the images are represented in range [0, 1]. In order to use [−1, 1] instead, we replace pt(x) → pt(2x−1),
x→ 1
2x + 1
2 and σ → 1
2 σ:
∇[ 1
2 x+ 1
2 ] log pt
(2[ 1
2x + 1
2
]−1) ≈ 1
[1
2 σ] Fθ
(2[ 1
2x + 1
2
]−1; log [ 1
2 σ]) (213)
2 ∇x log pt(x) ≈ 2
σ Fθ
(
x; log ( 1
2 σ)
)
(214)
∇x log p(x; σ) ≈ 1
σ Fθ
(
x; log ( 1
2 σ)
)
. (215)
We can now express the model in terms of Dθ(·) by replacing the left-hand side of Eq. 215 with Eq. 3:
(
Dθ
(x; σ) − x
)
/σ2 = 1
σ Fθ
(
x; log ( 1
2 σ)
)
(216)
Dθ
(x; σ) = 1 ·
}{{} cskip
x+ σ·
}{{} cout
Fθ
(
1·
}{{} cin
x; log ( 1
2 σ)
} {{ } cnoise
)
, (217)
where cskip, cout, cin, and cnoise match the “Network and preconditioning” section of Table 1.
C.2.4 VE training
Song et al. [49] define their training loss similarly for VP and VE, so we can reuse Eq. 185 by borrowing the definition of score(·) from Eq. 216:
Et,y,n
[∥
∥s(t)σ(t) score (s(t) (y + n); Fθ, t) + n/σ(t)∥
∥
2 2
]
(218)
= Et,y,n
[∥
∥σ(t) score (y + n; Fθ, t) + n/σ(t)∥
∥
2 2
]
(219)
= Et,y,n
[∥
∥σ(t)
[(
Dθ
(y + n; σ(t)) − (y + n)
)
/σ(t)2]
+ n/σ(t)∥
∥
2 2
]
(220)
= Et,y,n
[1 σ(t)2
∥
∥Dθ
(y + n; σ(t)) − y∥
∥
2 2
]
. (221)
For VE training, the original implementation13 defines σ(t) = σmin
( σmax σmin
)t. We can thus rewrite Eq. 221 as
Eln(σ)∼U (ln(σmin),ln(σmax))
} {{ } ptrain
Ey,n
[1 σ2
}{{}
λ
∥
∥Dθ
(y + n; σ) − y∥
∥
2 2
]
, (222)
which matches Eq. 108 with the choices for ptrain and λ shown in the “Training” section of Table 1.
9 https://github.com/yang- song/score_sde_pytorch/blob/1618ddea340f3e4a2ed7852a0694a809775cf8d0/models/utils.py#L163
10 https://github.com/yang- song/score_sde_pytorch/blob/1618ddea340f3e4a2ed7852a0694a809775cf8d0/models/ncsnpp.py#L239
11 https://github.com/yang- song/score_sde_pytorch/blob/1618ddea340f3e4a2ed7852a0694a809775cf8d0/models/ncsnpp.py#L261
12 https://github.com/yang- song/score_sde_pytorch/blob/1618ddea340f3e4a2ed7852a0694a809775cf8d0/models/ncsnpp.py#L379
13 https://github.com/yang- song/score_sde_pytorch/blob/1618ddea340f3e4a2ed7852a0694a809775cf8d0/sde_lib.py#L234
34


C.2.5 VE practical considerations
The pre-trained VE model that we use on CIFAR-10 corresponds to the “NCSN++ cont. (VE)” checkpoint14 provided by Song et al. [49]. It contains a total of 63 million trainable parameters and supports a continuous range of noise levels σ ∈ [σ( t), σ(1)] ≈ [0.02, 100]. This is narrower than our preferred sampling range [0.002, 80], so we set σmin = 0.02 in all related experiments. Note that this limitation is lifted by our training improvements in config E, so we revert back to using σmin = 0.002 with configs E and F in Table 2. When importing the model, we remove the pre- and postprocessing steps shown in Eq. 212 to stay consistent with the definition of Fθ(·) in Eq. 217. With these changes, we can run Algorithms 1 and 2 using the definitions in Table 1.
In Figure 2b, the differences between the original sampler (blue) and our reimplementation (orange) are explained by floating point round-off errors that the original implementation suffers from at high step counts. Our results are more accurate in these cases because we represent xi at double precision in Algorithm 1.
C.3 Improved DDPM and DDIM
C.3.1 DDIM ODE formulation
Song et al. [47] make the observation that their deterministic DDIM sampler can be expressed as Euler integration of the following ODE (Eq. 14 in [47]):
dx(t) = (t)
θ
(
x(t)
√σ(t)2 + 1
)
dσ(t), (223)
where x(t) is a scaled version of the iterate that appears in their discrete update formula (Eq. 10 in
[47]) and θ is a model trained to predict the normalized noise vector, i.e., (t)
θ
(x(t)/√σ(t)2 + 1) ≈ n(t)/σ(t) for x(t) = y(t) + n(t). In our formulation, Dθ is trained to approximate the clean signal, i.e., Dθ
(x(t); σ(t)) ≈ y, so we can reinterpret θ in terms of Dθ as follows:
n(t) = x(t) − y(t) (224)
[n(t)/σ(t)] = (x(t) − [y(t)])/σ(t) (225)
(t) θ
(x(t)/√σ(t)2 + 1) = (x(t) − Dθ
(x(t); σ(t)))/σ(t). (226)
Assuming ideal (·) and D(·) in L2 sense, we can further simplify the above formula using Eq. 3:
(t)(x(t)/√σ(t)2 + 1) = (x(t) − D(x(t); σ(t)))/σ(t) (227)
= −σ(t)
[(
D
(x(t); σ(t)) − x(t)
)
/σ(t)2]
(228)
= −σ(t) ∇x(t) log p(x(t); σ(t)). (229)
Substituting Eq. 229 back into Eq. 223 gives
dx(t) = −σ(t) ∇x(t) log p(x(t); σ(t)) dσ(t), (230)
which we can further simplify by setting σ(t) = t:
dx = −t ∇x log p(x; σ(t)) dt. (231)
This matches our Eq. 4 with s(t) = 1 and σ(t) = t, reflected by the “Sampling” section of Table 1.
C.3.2 iDDPM time step discretization
The original DDPM formulation of Ho et al. [16] defines the forward process (Eq. 2 in [16]) as a Markov chain that gradually adds Gaussian noise to  ̄x0 ∼ pdata according to a discrete variance schedule {β1, . . . , βT }:
q( ̄xt |  ̄xt−1) = N ( ̄xt; √1 − βt  ̄xt−1, βt I). (232)
14ve/cifar10_ncsnpp_continuous/checkpoint_24.pth, https://drive.google.com/drive/folders/1b0gy_LLgO_DaQBgoWXwlVnL_rcAUgREh
35


The corresponding transition probability from  ̄x0 to  ̄xt (Eq. 4 in [16]) is given by
q( ̄xt |  ̄x0) = N (x ̄t; √α ̄t  ̄x0, (1 − α ̄t) I), where α ̄t =
t
∏
s=1
(1 − βs). (233)
Ho et al. [16] define {βt} based on a linear schedule and then calculate the corresponding {α ̄t} from Eq. 233. Alternatively, one can also define {α ̄t} first and then solve for {βt}:
α ̄t =
t
∏
s=1
(1 − βs) (234)
α ̄t = α ̄t−1 (1 − βt) (235)
βt = 1 − α ̄t
α ̄t−1
. (236)
The improved DDPM formulation of Nichol and Dhariwal [37] employs a cosine schedule for α ̄t (Eq. 17 in [37]), defined as
α ̄t = f (t)
f (0) , where f (t) = cos2
( t/T + s
1+s · π
2
)
, (237)
where s = 0.008. In their implementation15, however, Nichol et al. leave out the division by f (0) and simply define16
α ̄t = cos2
( t/T + s
1+s · π
2
)
. (238)
To prevent singularities near t = T , they also clamp βt to 0.999. We can express the clamping in terms of α ̄t by utilizing Eq. 233 and Eq. 234:
α ̄′
t=
t
∏
s=1
(1 − [β′
s]) (239)
=
t
∏
s=1
(
1 − min ([βs], 0.999)
)
(240)
=
t
∏
s=1
(
1 − min
(
1 − α ̄s
α ̄s−1
, 0.999
))
(241)
=
t
∏
s=1
max
( α ̄s
α ̄s−1
, 0.001
)
. (242)
Let us now reinterpret the above formulas in our unified framework. Recall from Table 1 that we denote the original iDDPM sampling steps by {uj} in the order of descending noise level σ(uj), where j ∈ {0, . . . , M }. To harmonize the notation of Eq. 233, Eq. 238, and Eq. 239, we thus have to replace T −→ M and t −→ M − j:
q(x ̄j |  ̄xM ) = N (x ̄j;
√
α ̄′
j  ̄xM , (1 − α ̄′
j) I), (243)
α ̄j = cos2
( (M − j)/M + C2 1 + C2
·π
2
)
, and (244)
α ̄′
j=
j
∏
s=M −1
max
( α ̄j
α ̄j+1
, C1
)
= α ̄′
j+1 max
( α ̄j
α ̄j+1
, C1
)
, (245)
where the constants are C1 = 0.001 and C2 = 0.008.
15https://github.com/openai/improved-diffusion
16 https://github.com/openai/improved- diffusion/blob/783b6740edb79fdb7d063250db2c51cc9545dcd1/improved_diffusion/gaussian_ diffusion.py#L39
36


We can further simplify Eq. 244:
α ̄j = cos2
( (M − j)/M + C2 1 + C2
·π
2
)
(246)
= cos2
(π
2
(1 + C2) − j/M 1 + C2
)
(247)
= cos2
(π
2−π
2
j
M (1 + C2)
)
(248)
= sin2
(π
2
j
M (1 + C2)
)
, (249)
giving the formula shown in the “Parameters” section of Table 1.
To harmonize the definitions of x and  ̄x, we must match the perturbation kernel of Eq. 11 with the transition probability of Eq. 243 for each time step t = uj:
p0t
(x(uj) | x(0)) = q( ̄xj |  ̄xM ) (250)
N
(x(uj); s(t) x(0), s(uj)2 σ(uj)2 I) = N
(
 ̄xj ;
√
α ̄′
j  ̄xM , (1 − α ̄′
j
)I
)
. (251)
Substituting s(t) = 1 and σ(t) = t from Appendix C.3.1, as well as  ̄xM = x(0):
N
(x(uj); x(0), u2
j I) = N
(
x ̄ j ;
√
α ̄′
j x(0), (1 − α ̄′
j
)I
)
. (252)
We can match the means of these two distributions by defining  ̄xj =
√
α ̄′
j x(uj):
N
(x(uj); x(0), u2
j I) = N
(√
α ̄′
j x(uj);
√
α ̄′
j x(0), (1 − α ̄′
j
)I
)
(253)
=N
(
x(uj); x(0), 1 − α ̄′
j
α ̄′
j
I
)
. (254)
Matching the variances and solving for α ̄′
j gives
u2
j = (1 − α ̄′
j ) / α ̄′
j (255)
u2
j α ̄′
j = 1 − α ̄′
j (256)
u2
j α ̄′
j + α ̄′
j = 1 (257)
(u2
j + 1) α ̄′
j = 1 (258)
α ̄′
j = 1 / (u2
j + 1). (259)
Finally, we can expand the left-hand side using Eq. 245 and solve for uj−1:
α ̄′
j+1 max(α ̄j /α ̄j+1, C1) = 1 / (u2
j + 1) (260)
α ̄′
j max(α ̄j−1/α ̄j, C1) = 1 / (u2
j−1 + 1) (261)
[1 / (u2
j + 1)] max(α ̄j−1/α ̄j, C1) = 1 / (u2
j−1 + 1) (262)
max(α ̄j−1/α ̄j, C1) (u2
j−1 + 1) = u2
j + 1 (263)
u2
j−1 + 1 = (u2
j + 1) / max(α ̄j−1/α ̄j, C1) (264)
uj−1 =
√
u2
j +1
max(α ̄j−1/α ̄j, C1) − 1, (265)
giving a recurrence formula for {uj}, bootstrapped by uM = 0, that matches the “Time steps” row of Table 1.
37


C.3.3 iDDPM preconditioning and training
We can solve Dθ(·) from Eq. 227 by substituting σ(t) = t from Appendix C.3.1:
(j) θ
(
x/
√
σ2 + 1
)
=
(x − Dθ(x; σ))/σ (266)
Dθ(x; σ) = x − σ (j)
θ
(
x/
√
σ2 + 1
)
. (267)
We choose to define Fθ(·; j) = (j)
θ (·) and solve j from σ by finding the nearest uj:
Dθ(x; σ) = 1 ·
}{{} cskip
x −σ
}{{} cout
· Fθ
(1
√σ2 +1
} {{ } cin
· x; arg minj |uj − σ|
} {{ } cnoise
)
, (268)
where cskip, cout, cin, and cnoise match the “Network and preconditioning” section of Table 1.
Note that Eq. 268 is identical to the VP preconditioning formula in Eq. 181. Furthermore, Nichol and Dhariwal [37] define their main training loss Lsimple (Eq. 14 in [37]) the same way as Song et al. [49], with σ drawn uniformly from {uj}. Thus, we can reuse Eq. 190 with σ = uj, j ∼ U(0, M − 1), and λ(σ) = 1/σ2, matching the “Training” section of Table 1. In addition to Lsimple, Nichol and Dhariwal [37] also employ a secondary loss term Lvlb; we refer the reader to Section 3.1 in [37] for details.
C.3.4 iDDPM practical considerations
The pre-trained iDDPM model that we use on ImageNet-64 corresponds to the “ADM (dropout)” checkpoint17 provided by Dhariwal and Nichol [9]. It contains 296 million trainable parameters and supports a discrete set of M = 1000 noise levels σ ∈ {uj} ≈ {20291, 642, 321, 214, 160, 128, 106, 92, 80, 71, . . . , 0.0064}. The fact that we can only evaluate Fθ these specific choices of σ presents three practical challenges:
1. In the context of DDIM, we must choose how to resample {uj} to yield {ti} for N 6= M . Song et al. [47] employ a simple resampling scheme where ti = uk·i for resampling factor
k ∈ Z+. This scheme, however, requires that 1000 ≡ 0 (mod N ), which limits the possible choices for N considerably. Nichol and Dhariwal [37], on the other hand, employ a more flexible scheme where ti = uj with j = b(M − 1)/(N − 1) · ic. We note, however, that in practice the values of uj<8 are considerably larger than our preferred σmax = 80. We choose to skip these values by defining j = bj0 + (M − 1 − j0)/(N − 1) · ic with j0 = 8, matching the “Time steps” row in Table 1. In Figure 2c, the differences between the original sampler (blue) and our reimplementation (orange) are explained by this choice.
2. In the context of our time step discretization (Eq. 5), we must ensure that σi ∈ {uj}. We accomplish this by rounding each σi to its nearest supported counterpart, i.e., σi ← uarg minj |uj−σi|, and setting σmin = 0.0064 ≈ uN−1. This is sufficient, because Algorithm 1 only evaluates Dθ(·; σ) with σ ∈ {σi<N }.
3. In the context of our stochastic sampler, we must ensure that tˆi ∈ {uj}. We accomplish this by replacing line 5 of Algorithm 2 with tˆi ← uarg minj |uj−(ti+γiti)|.
With these changes, we are able to import the pre-trained model directly as Fθ(·) and run Algorithms 1 and 2 using the definitions in Table 1. Note that the model outputs both θ(·) and Σθ(·), as described in Section 3.1 of [37]; we use only the former and ignore the latter.
D Further analysis of deterministic sampling
D.1 Truncation error analysis and choice of discretization parameters
As discussed in Section 3, the fundamental reason why diffusion models tend to require a large number of sampling steps is that any numerical ODE solver is necessarily an approximation; the
17https://openaipublic.blob.core.windows.net/diffusion/jul-2021/64x64_diffusion.pt
38


σ=0.02 0.1 0.5 1 2 5 10 20 50
10-4
10-3
10-2
10-1
100
101
102
‖τ ‖
ρ = 1.0 1.5 2.0 3.0 7.0
σ=0.02 0.1 0.5 1 2 5 10 20 50
10-4
10-3
10-2
10-1
100
101
102
‖τ ‖
ρ = 1.0 1.5 2.0 3.0 7.0
ρ=1 2 3 4 5 6 7 8 9 10
3
4
5
6
7
8
9
10
FID CIFAR-10, VP, N = 32 CIFAR-10, VE, N = 64 ImageNet-64, N = 12
(a) Truncation error, VE + Euler (b) Truncation error, VE + Heun (c) FID as a function of ρ
Figure 13: (a) Local truncation error (y-axis) at different noise levels (x-axis) using Euler’s method with the VE-based CIFAR-10 model. Each curve corresponds to a different time step discretization, defined for N = 64 and a specific choice for the polynomial exponent ρ. The values represent the root mean square error (RMSE) between one Euler iteration and a sequence of multiple smaller Euler iterations, representing the ground truth. The shaded regions, barely visible at low σ, represent standard deviation over different latents x0. (b) Corresponding error curves for Heun’s 2nd order method (Algorithm 1). (c) FID (y-axis) as a function of the polynomial exponent (x-axis) for different models, measured using Heun’s 2nd order method. The shaded regions indicate the range of variation between the lowest and highest observed FID, and the dots indicate the value of ρ that we use in all other experiments.
larger the steps, the farther away we drift from the true solution at each step. Specifically, given the value of xi−1 at time step i − 1, the solver approximates the true x∗
i as xi, resulting in local
truncation error τ i = x∗
i − xi. The local errors get accumulated over the N steps, ultimately leading to global truncation error eN .
Euler’s method is a first order ODE solver, meaning that τ i = O (h2
i
) for any sufficiently smooth x(t), where hi = |ti − ti−1| is the local step size [50]. In other words, there exist some C and H
such that ||τ i|| < Ch2
i for every hi < H, i.e., halving hi reduces τ i by 4×. Furthermore, if we assume that Dθ is Lipschitz continuous — which is true for all network architectures considered in this paper — the global truncation error is bounded by ||eN || ≤ E maxi ||τ i||, where the value of E depends on N , t0, tN , and the Lipschitz constant [50]. Thus, reducing the global error for given N , which in turn enables reducing N itself, boils down to choosing the solver and {ti} so that maxi ||τ i|| is minimized.
To gain insight on how the local truncation error behaves in practice, we measure the values of τ i over different noise levels using the VE-based CIFAR-10 model. For a given noise level, we
set ti = σ−1(σi) and choose some ti−1 > ti depending on the case. We then sample xi−1
from p(x; σi−1) and estimate the true x∗
i by performing 200 Euler steps over uniformly selected subintervals between ti−1 and t. Finally, we plot the mean and standard deviation of the root mean
square error (RMSE), i.e., ||τ i||/√dim τ, as a function of σi, averaged over 200 random samples of xi−1. Results for Euler’s method are shown in Figure 13a, where the blue curve corresponds to
uniform step size hσ = 1.25 with respect to σ, i.e., σi−1 = σi + hσ and ti−1 = σ−1(σi−1). We see that the error is very large (RMSE ≈ 0.56) for low noise levels (σi ≤ 0.5) and considerably smaller for high noise levels. This is in line with the common intuition that, in order to reduce eN , the step size should be decreased monotonically with decreasing σ. Each curve is surrounded by a shaded region that indicates standard deviation, barely visible at low values of σ. This indicates that τ i is nearly constant with respect to xi−1, and thus there would be no benefit in varying {ti} schedule on a per-sample basis.
A convenient way to vary the local step size depending on the noise level is to define {σi} as a linear resampling of some monotonically increasing, unbounded warp function w(z). In other words, σi<N = w(Ai + B) and σN = 0, where constants A and B are selected so that σ0 = σmax and σN−1 = σmin. In practice, we set σmin = max(σlo, 0.002) and σmax = min(σhi, 80), where σlo and σhi are the lowest and highest noise levels supported by a given model, respectively; we have found these choices to perform reasonably well in practice. Now, to balance τ i between low and high
noise levels, we can, for example, use a polynomial warp function w(z) = zρ parameterized by the
39


exponent ρ. This choice leads to the following formula for {σi}:
σi<N =
(
σmax
1
ρ+ i
N −1
(
σmin
1
ρ − σmax
1 ρ
)
)ρ
, σN = 0, (269)
which reduces to uniform discretization when ρ = 1 and gives more and more emphasis to low noise levels as ρ increases.18
Based on the value of σi, we can now compute σi−1 = (σ1/ρ
i − A)ρ, which enables us to visualize
τ i for different choices of ρ in Figure 13a. We see that increasing ρ reduces the error for low noise levels (σ < 10) while increasing it for high noise levels (σ > 10). Approximate balance is achieved at ρ = 2, but RMSE remains relatively high (∼ 0.03), meaning that Euler’s method drifts away from the correct result by several ULPs at each step. While the error could be reduced by increasing N , we would ideally like the RMSE to be well below 0.01 even with low step counts.
Heun’s method introduces an additional correction step for xi+1 to account for the fact that dx/dt may change between ti and ti+1; Euler’s method assumes it to be constant. The correction leads
to cubic convergence of the local truncation error, i.e., τ i = O (h3
i
), at the cost of one additional evaluation of Dθ per step. We discuss the general family of Heun-like schemes later in Appendix D.2. Figure 13b shows local truncation error for Heun’s method using the same setup as Figure 13a. We see that the differences in ||τ i|| are generally more pronounced, which is to be expected given the quadratic vs. cubic convergence of the two methods. Cases where Euler’s method has low RMSE tend to have even lower RMSE with Heun’s method, and vice versa for cases with high RMSE. Most remarkably, the red curve shows almost constant RMSE ∈ [0.0030, 0.0045]. This means that the combination of Eq. 269 and Heun’s method is, in fact, very close to optimal with ρ = 3.
Thus far, we have only considered the raw numerical error, i.e., component-wise deviation from the true result in RGB space. The raw numerical error is relevant for certain use cases, e.g., image manipulation where the ODE is first evaluated in the direction of increasing t and then back to t = 0 again — in this case, ||eN || directly tells us how much the original image degrades in the process and we can use ρ = 3 to minimize it. Considering the generation of novel images from scratch, however, it is reasonable to expect different noise levels to introduce different kinds of errors that may not necessarily be on equal footing considering their perceptual importance. We investigate this in Figure 13c, where we plot FID as a function of ρ for different models and different choices of N . Note that the ImageNet-64 model was only trained for a discrete set of noise levels; in order to use it with Eq. 269, we round each ti to its nearest supported counterpart, i.e., t′
i = uarg minj |uj −ti|.
From the plot, we can see that even though ρ = 3 leads to relatively good FID, it can be reduced further by choosing ρ > 3. This corresponds to intentionally introducing error at high noise levels to reduce it at low noise levels, which makes intuitive sense because the value of σmax is somewhat arbitrary to begin with — increasing σmax can have a large impact on ||eN ||, but it does not affect the resulting image distribution nearly as much. In general, we have found ρ = 7 to perform reasonably well in all cases, and use this value in all other experiments.
D.2 General family of 2nd order Runge–Kutta variants
Heun’s method illustrated in Algorithm 1 belongs to a family of explicit two-stage 2nd order RungeKutta methods, each having the same computational cost. A common parameterization [50] of this family is,
di = f (xi; ti) ; xi+1 = xi + h
[(
1− 1
2α
)
di + 1
2α f (xi + αhdi; ti + αh)
]
, (270)
where h = ti+1 − ti and α is a parameter that controls where the additional gradient is evaluated and
how much it influences the step taken. Setting α = 1 corresponds to Heun’s method, and α = 1
2 and
α= 2
3 yield so-called midpoint and Ralston methods, respectively. All these variants differ in the
kind of approximation error they incur due to the geometry of the underlying function f .
To establish the optimal α in our use case, we ran a separate series of experiments. According to the results, it appears that α = 1 is very close to being optimal. Nonetheless, the experimentally
18In the limit, Eq. 269 reduces to the same geometric sequence employed by original VE ODE when ρ → ∞. Thus, our discretization can be seen as a parametric generalization of the one proposed by Song et al. [49].
40


Algorithm 3 Deterministic sampling using general 2nd order Runge–Kutta, σ(t) = t and s(t) = 1.
1: procedure ALPHASAMPLER(Dθ(x; σ), ti∈{0,...,N}, α) 2: sample x0 ∼ N (0, t2
0 I) 3: for i ∈ {0, . . . , N − 1} do
4: hi ← ti+1 − ti . Step length 5: di ← (xi − Dθ(xi; ti))/ti . Evaluate dx/dt at (x, ti) 6: (x′
i, t′
i) ← (xi + αhdi, ti + αh) . Additional evaluation point 7: if t′
i 6= 0 then 8: di′ ← (x′
i − Dθ(x′
i; t′
i))/t′
i . Evaluate dx/dt at (x′
i, t′
i)
9: xi+1 ← xi + h
[(
1− 1
2α
)
di + 1
2α d′
i
]
. Second order step from ti to ti+1
10: else 11: xi+1 ← xi + hdi . Euler step from ti to ti+1 12: return xN
best choice was α = 1.1 that performed slightly better, even though values greater than one are theoretically hard to justify as they overshoot the target ti+1. As we have no good explanation for this observation and cannot tell if it holds in general, we chose not to make α a new hyperparameter and instead fixed it to 1, corresponding exactly to Heun’s method. Further analysis is left as future work, including the possibility of having α vary during sampling.
An additional benefit of setting α = 1 is that it makes it possible to use pre-trained neural networks Dθ(x; σ) that have been trained only for specific values of σ. This is because a Heun step evaluates
the additional gradient at exactly ti+1 unlike the other 2nd order variants. Hence it is sufficient to ensure that each ti corresponds to a value of σ that the network was trained for.
Algorithm 3 shows the pseudocode for a general 2nd order solver parameterized by α. For clarity, the pseudocode assumes the specific choices of σ(t) = t and s(t) = 1 that we advocate in Section 3. Note that the fallback to Euler step (line 11) can occur only when α ≥ 1.
E Further results with stochastic sampling
E.1 Image degradation due to excessive stochastic iteration
Figure 14 illustrates the image degradation caused by excessive Langevin iteration (Section 4, “Practical considerations”). These images are generated by doing a specified number of iterations at a fixed noise level σ so that at each iteration an equal amount of noise is added and removed. In theory, Langevin dynamics should bring the distribution towards the ideal distribution p(x; σ) but as noted in Section 4, this holds only if the denoiser Dθ(x; σ) induces a conservative vector field in Eq. 3.
As seen in the figure, it is clear that the image distribution suffers from repeated iteration in all cases, although the exact failure mode depends on dataset and noise level. For low noise levels (below 0.2 or so), the images tend to oversaturate starting at 2k iterations and become fully corrupted after that. Our heuristic of setting Stmin > 0 is designed to prevent stochastic sampling altogether at very low noise levels to avoid this effect.
For high noise levels, we can see that iterating without the standard deviation correction, i.e., when Snoise = 1.000, the images tend to become more abstract and devoid of color at high iteration counts; this is especially visible in the 10k column of CIFAR-10 where the images become mostly black and white with no discernible backgrounds. Our heuristic inflation of standard deviation by setting Snoise > 1 counteracts this tendency efficiently, as seen in the corresponding images on the right hand side of the figure. Notably, this still does not fix the oversaturation and corruption at low noise levels, suggesting multiple sources for the detrimental effects of excessive iteration. Further research will be required to better understand the root causes of these observed effects.
Figure 15 presents the output quality of our stochastic sampler in terms of FID as a function of Schurn at fixed NFE, using pre-trained networks of Song et al. [49] and Dhariwal and Nichol [9]. Generally, for each case and combination of our heuristic corrections, there is an optimal amount of stochasticity after which the results start to degrade. It can also be seen that regardless of the value of Schurn, the
41


Uncond. CIFAR-10, Pre-trained, VP, Snoise = 1.000 Uncond. CIFAR-10, Pre-trained, VP, Snoise = 1.007
0.02
0.05
0.10
0.20
0.30
0.40
0.50
0.60
0.70
0.80
Step 0 100 200 500 1,000 2,000 5,000 10,000 σ Step 0 100 200 500 1,000 2,000 5,000 10,000
Cond. ImageNet-64, Pre-trained, Snoise = 1.000 Cond. ImageNet-64, Pre-trained, Snoise = 1.003
0.05
0.10
0.20
0.30
0.40
0.50
0.60
0.70
Step 0 500 1,000 2,000 5,000 10,000 σ Step 0 500 1,000 2,000 5,000 10,000
Figure 14: Gradual image degradation with repeated addition and removal of noise. We start with a random image drawn from p(x; σ) (first column) and run Algorithm 2 for a certain number of
steps (remaining columns) with fixed γi = √2 − 1. Each row corresponds to a specific choice of σ (indicated in the middle) that we keep fixed throughout the entire process. We visualize the results after running them through the denoiser, i.e., Dθ(xi; σ).
42


Schurn=0 10 20 30 40 50 60 70 80 90 100
2.0
2.5
3.0
3.5
4.0
4.5
5.0
FID Deterministic Stmin,tmax + Snoise = 1 Stmin,tmax = [0, ∞] Snoise = 1 Optimal settings
0 10 20 30 40 50 60 70 80 90 100
2.0
2.5
3.0
3.5
4.0
FID
0 10 20 30 40 50 60 70 80 90 100
1.4
1.6
1.8
2.0
2.2
2.4
2.6
2.8
FID
(a) Uncond. CIFAR-10, VP (b) Uncond. CIFAR-10, VE (c) Class-cond. ImageNet-64
Figure 15: Ablations of our stochastic sampler (Algorithm 2) parameters using pre-trained networks of Song et al. [49] and Dhariwal and Nichol [9]. Each curve shows FID (y-axis) as a function of Schurn (x-axis) for N = 256 steps (NFE = 511). The dashed red lines correspond to our deterministic sampler (Algorithm 1), equivalent to setting Schurn = 0. The purple curves correspond to optimal choices for {Stmin, Stmax, Snoise}, found separately for each case using grid search. Orange, blue, and green correspond to disabling the effects of Stmin,tmax and/or Snoise. The shaded regions indicate the range of variation between the lowest and highest observed FID.
Table 5: Parameters used for the stochastic sampling experiments in Section 4.
Parameter CIFAR-10 ImageNet Grid search
VP VE Pre-trained Our model Schurn 30 80 80 40 0, 10, 20, 30, . . . , 70, 80, 90, 100 Stmin 0.01 0.05 0.05 0.05 0, 0.005, 0.01, 0.02, . . . , 1, 2, 5, 10 Stmax 1 1 50 50 0.2, 0.5, 1, 2, . . . , 10, 20, 50, 80 Snoise 1.007 1.007 1.003 1.003 1.000, 1.001, . . . , 1.009, 1.010
best results are obtained by enabling all corrections, although whether Snoise or Stmin,tmax is more important depends on the case.
E.2 Stochastic sampling parameters
Table 5 lists the values for Schurn, Stmin, Stmax, and Snoise that we used in our stochastic sampling experiments. These were determined with a grid search over the combinations listed in the rightmost column. It can be seen that the optimal parameters depend on the case; better understanding of the degradation phenomena will hopefully give rise to more direct ways of handling the problem in the future.
F Implementation details
We implemented our techniques in a newly written codebase, based loosely on the original implementations by Song et al.19 [49], Dhariwal and Nichol20 [9], and Karras et al.21 [26]. We performed extensive testing to verify that our implementation produced exactly the same results as previous work, including samplers, pre-trained models, network architectures, training configurations, and evaluation. We ran all experiments using PyTorch 1.10.0, CUDA 11.4, and CuDNN 8.2.0 on NVIDIA DGX-1’s with 8 Tesla V100 GPUs each.
Our implementation and pre-trained models are available at https://github.com/NVlabs/edm
19https://github.com/yang-song/score_sde_pytorch 20https://github.com/openai/guided-diffusion 21https://github.com/NVlabs/stylegan3
43


Table 6: Our augmentation pipeline. Each training image undergoes a combined geometric transformation based on 8 random parameters that receive non-zero values with a certain probability. The model is conditioned with an additional 9-dimensional input vector derived from these parameters.
Augmentation Transformation Parameters Prob. Conditioning Constants
x-flip SCALE2D
(1 − 2a0, 1) a0 ∼ U {0, 1} 100% a0 Aprob = 12%
y-flip SCALE2D
(1, 1 − 2a1
) a1 ∼ U {0, 1} Aprob a1 or 15%
Scaling SCALE2D
((Ascale)a2 , a2 ∼ N (0, 1) Aprob a2 Ascale = 20.2
(Ascale)a2 )
Rotation ROTATE2D
(−a3
) a3 ∼ U (−π, π) Aprob cos a3 − 1
sin a3
Anisotropy ROTATE2D
(a4
) a4 ∼ U (−π, π) Aprob a5 cos a4 Aaniso = 20.2
SCALE2D
((Aaniso)a5 , a5 ∼ N (0, 1) a5 sin a4
1/(Aaniso)a5 )
ROTATE2D
(−a4
)
Translation TRANSLATE2D
((Atrans)a6, a6 ∼ N (0, 1) Aprob a6 Atrans = 1/8
(Atrans)a7
) a7 ∼ N (0, 1) a7
F.1 FID calculation
We calculate FID [15] between 50,000 generated images and all available real images, without any augmentation such as x-flips. We use the pre-trained Inception-v3 model provided with StyleGAN322 [26] that is, in turn, a direct PyTorch translation of the original TensorFlow-based model23. We have verified that our FID implementation produces identical results compared to Dhariwal and Nichol [9] and Karras et al. [26]. To reduce the impact of random variation, typically in the order of ±2%, we compute FID three times in each experiment and report the minimum. We also highlight the difference between the highest and lowest achieved FID in Figures 4, 5b, 13c, and 15.
F.2 Augmentation regularization
In Section 5, we propose to combat overfitting of Dθ using conditional augmentation. We build our augmentation pipeline around the same concepts that were originally proposed by Karras et al. [25] in the context of GANs. In practice, we employ a set of 6 geometric transformations; we have found other types of augmentations, such as color corruption and image-space filtering, to be consistently harmful for diffusion-based models.
The details of our augmentation pipeline are shown in Table 6. We apply the augmentations independently to each training image y ∼ pdata prior to adding the noise n ∼ N (0, σ2I). First, we determine whether to enable or disable each augmentation based on a weighted coin toss. The probability of enabling a given augmentation (“Prob.” column) is fixed to 12% for CIFAR-10 and 15% for FFHQ and AFHQv2, except for x-flips that are always enabled. We then draw 8 random parameters from their corresponding distributions (“Parameters” column); if a given augmentation is disabled, we override the associated parameters with zero. Based on these, we construct a homogeneous 2D transformation matrix based on the parameters (“Transformation” column). This transformation is applied to the image using the implementation of [25] that employs 2× supersampled high-quality Wavelet filters. Finally, we construct a 9-dimensional conditioning input vector (“Conditioning” column) and feed it to the denoiser network, in addition to the image and noise level inputs.
The role of the conditioning input is to present the network with a set of auxiliary tasks; in addition to the main task of modeling p(x; σ), we effectively ask the network to also model an infinite set of distributions p(x; σ, a) for each possible choice of the augmentation parameters a. These auxiliary tasks provide the network with a large variety of unique training samples, preventing it from
22 https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl
23 http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
44


Table 7: Hyperparameters used for the training runs in Section 5.
Hyperparameter CIFAR-10 FFHQ & AFHQv2 ImagetNet
Baseline Ours Baseline Ours Ours Number of GPUs 4 8 4 8 32 Duration (Mimg) 200 200 200 200 2500 Minibatch size 128 512 128 256 4096 Gradient clipping X – X – Mixed-precision (FP16) – – – – X
Learning rate ×104 2 10 2 2 1 LR ramp-up (Mimg) 0.64 10 0.64 10 10 EMA half-life (Mimg) 0.89 / 0.9 0.5 0.89 / 0.9 0.5 50 (VP / VE) (VP / VE) Dropout probability 10% 13% 10% 5% / 25% 10% (FFHQ / AFHQ) Channel multiplier 128 128 128 128 192 Channels per resolution 1-2-2-2 2-2-2 1-1-2-2-2 1-2-2-2 1-2-3-4 Dataset x-flips X – X – Augment probability – 12% – 15% 
overfitting to any individual sample. Still, the auxiliary tasks appear to be beneficial for the main task; we speculate that this is because the denoising operation itself is similar for every choice of a.
We have designed the conditioning input so that zero corresponds to the case where no augmentations were applied. During sampling, we simply set a = 0 to obtain results consistent with the main task. We have not observed any leakage between the auxiliary tasks and the main task; the generated images exhibit no traces of out-of-domain geometric transformations even with Aprob = 100%. In practice, this means that we are free to choose the constants {Aprob, Ascale, Aaniso, Atrans} any way we like as long as the results improve. Horizontal flips serve as an interesting example. Most of the prior work augments the training set with random x-flips, which is beneficial for most datasets but has the downside that any text or logos may appear mirrored in the generated images. With our non-leaky augmentations, we get the same benefits without the downsides by executing the x-flip augmentation with 100% probability. Thus, we rely exclusively on our augmentation scheme and disable dataset x-flips to ensure that the generated images stay true to the original distribution.
F.3 Training configurations
Table 7 shows the exact set of hyperparameters that we used in our training experiments reported in Section 5. We will first detail the configurations used with CIFAR-10, FFHQ, and AFHQv2, and then discuss the training of our improved ImageNet model.
Config A of Table 2 (“Baseline”) corresponds to the original setup of Song et al. [49] for the two cases (VP and VE), and config F (“Ours”) corresponds to our improved setup. We trained each model until a total of 200 million images had been drawn from the training set, abbreviated as “200 Mimg” in Table 7; this corresponds to a total of ∼400,000 training iterations using a batch size of 512. We saved a snapshot of the model every 2.5 million images and reported results for the snapshot that achieved the lowest FID according to our deterministic sampler with NFE = 35 or NFE = 79, depending on the resolution.
In config B, we re-adjust the basic hyperparameters to enable faster training and obtain a more meaningful point of comparison. Specifically, we increase the parallelism from 4 to 8 GPUs and batch size from 128 to 512 or 256, depending on the resolution. We also disable gradient clipping, i.e., forcing ‖dL(Dθ)/dθ‖2 ≤ 1, that we found to provide no benefit in practice. Furthermore, we increase the learning rate from 0.0002 to 0.001 for CIFAR-10, ramping it up during the first 10 million images, and standardize the half-life of the exponential moving average of θ to 0.5 million images. Finally, we adjust the dropout probability for each dataset as shown in Table 7 via a full grid search at 1% increments. Our total training time is approximately 2 days for CIFAR-10 at 32×32 resolution and 4 days for FFHQ and AFHQv2 at 64×64 resolution.
45


Table 8: Details of the network architectures used in this paper.
Parameter DDPM++ NCSN++ ADM
(VP) (VE) (ImageNet) Resampling filter Box Bilinear Box Noise embedding Positional Fourier Positional Skip connections in encoder – Residual Skip connections in decoder – – Residual blocks per resolution 4 4 3 Attention resolutions {16} {16} {32, 16, 8} Attention heads 1 1 6-9-12 Attention blocks in encoder 4 4 9 Attention blocks in decoder 2 2 13
In config C, we improve the expressive power of the model by removing the 4×4 layers and doubling the capacity of the 16×16 layers instead; we found the former to mainly contribute to overfitting, whereas the latter were critical for obtaining high-quality results. The original models of Song et al. [49] employ 128 channels at 64×64 (where applicable) and 32×32, and 256 channels at 16×16, 8×8, and 4×4. We change these numbers to 128 channels at 64×64 (where applicable), and 256 channels at 32×32, 16×16, and 8×8. We abbreviate these counts in Table 7 as multiples of 128, listed from the highest resolution to the lowest. In practice, this rebalancing reduces the total number of trainable parameters slightly, resulting in ∼56 million parameters for each model at 32×32 resolution and ∼62 million parameters at 64×64 resolution.
In config D, we replace the original preconditioning with our improved formulas (“Network and preconditioning” section in Table 1). In config E, we do the same for the noise distribution and loss weighting (“Training” section in Table 1). Finally, in config F, we enable augmentation regularization as discussed in Appendix F.2. The other hyperparameters remain the same as in config C.
With ImageNet-64, it is necessary to train considerably longer compared to the other datasets in order to reach state-of-the-art results. To reduce the training time, we employed 32 NVIDIA Ampere GPUs (4 nodes) with a batch size of 4096 (128 per GPU) and utilized the high-performance Tensor Cores via mixed-precision FP16/FP32 training. In practice, we store the trainable parameters as FP32 but cast them to FP16 when evaluating Fθ, except for the embedding and self-attention layers, where we found the limited exponent range of FP16 to occasionally lead to stability issues. We trained the model for two weeks, corresponding to ∼2500 million images drawn from the training set and ∼600,000 training iterations, using learning rate 0.0001, exponential moving average of 50 million images, and the same model architecture and dropout probability as Dhariwal and Nichol [9]. We did not find overfitting to be a concern, and thus chose to not employ augmentation regularization.
F.4 Network architectures
As a result of our training improvements, the VP and VE cases become otherwise identical in config F except for the network architecture; VP employs the DDPM++ architecture while VE employs NCSN++, both of which were originally proposed by Song et al. [49]. These architectures correspond to relatively straightforward variations of the same U-net backbone with three differences, as illustrated in Table 8. First, DDPM++ employs box filter [1, 1] for the upsampling and downsampling layers whereas NCSN++ employs bilinear filter [1, 3, 3, 1]. Second, DDPM++ inherits its positional encoding scheme for the noise level directly from DDPM [16] whereas NCSN++ replaces it with random Fourier features [52]. Third, NCSN++ incorporates additional residual skip connections from the input image to each block in the encoder, as explained in Appendix H of [49] (“progressive growing architectures”).
For class conditioning and augmentation regularization, we extend the original DDPM++ and NCSN++ arhictectures by introducing two optional conditioning inputs alongside the noise level input. We represent class labels as one-hot encoded vectors that we first scale by
√C, where C is the total number of classes, and then feed through a fully-connected layer. For the augmentation parameters, we feed the conditioning inputs of Appendix F.2 through a fully-connected layer as-is.
46


We then combine the resulting feature vectors with the original noise level conditioning vector through elementwise addition.
For class-conditional ImageNet-64, we use the ADM architecture of Dhariwal and Nichol [9] with no changes. The model has a total of ∼296 million trainable parameters. As detailed in Tables 7 and 8, the most notable differences to DDPM++ include the use of a slightly shallower model (3 residual blocks per resolution instead of 4) with considerably more channels (e.g., 768 in the lowest resolution instead of 256), more self-attention layers interspersed throughout the network (22 instead of 6), and the use of multi-head attention (e.g., 12 heads in the lowest resolution). We feel that the precise impact of architectural choices remains an interesting question for future work.
F.5 Licenses
Datasets:
• CIFAR-10 [29]: MIT license
• FFHQ [27]: Creative Commons BY-NC-SA 4.0 license
• AFHQv2 [7]: Creative Commons BY-NC 4.0 license
• ImageNet [8]: The license status is unclear
Pre-trained models:
• CIFAR-10 models by Song et al. [49]: Apache V2.0 license
• ImageNet-64 model by Dhariwal and Nichol [9]: MIT license
• Inception-v3 model by Szegedy et al. [51]: Apache V2.0 license
47
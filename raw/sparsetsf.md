170

IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE, VOL. 48, NO. 1, JANUARY 2026

SparseTSF: Lightweight and Robust Time Series
Forecasting via Sparse Modeling
Shengsheng Lin , Weiwei Lin , Senior Member, IEEE, Wentai Wu , Member, IEEE, Haojun Chen,
and C. L. Philip Chen , Life Fellow, IEEE

Abstract—This paper introduces SparseTSF, a novel and extremely lightweight method for Long-term Time Series Forecasting
(LTSF), designed to address the challenges of modeling complex
temporal dependencies over extended horizons with minimal computational resources. At the heart of SparseTSF lies the CrossPeriod Sparse Forecasting technique, which simplifies the forecasting task by downsampling the original sequences to focus on
cross-period trend prediction. This technique not only significantly
reduces model complexity and the number of parameters but also
serves as an implicit regularization mechanism that enhances the
model’s robustness, achieving an optimal balance between performance and efficiency. Based on this technique, SparseTSF uses
fewer than 1,000 parameters to achieve competitive performance
compared to state-of-the-art methods, with evident advantages
under longer look-back windows (e.g., 720) that allow the model to
better exploit inherent periodicity and trend information. Furthermore, SparseTSF showcases remarkable generalization capabilities, making it well-suited for scenarios with limited computational
resources, small samples, or low-quality data.
Index Terms—Time series forecasting, long-term forecasting,
lightweight models, sparse modeling, machine learning.

I. INTRODUCTION
IME series forecasting holds significant value in domains
such as traffic flow, product sales, and energy consumption,
as accurate predictions enable decision-makers to plan proactively [1], [2], [3], [4]. Achieving precise forecasts typically
relies on powerful yet complex deep learning models, such as
Recurrent Neural Networks (RNNs) [5], [6], [7], Convolutional

T

Received 21 November 2024; revised 28 June 2025; accepted 21 August 2025.
Date of publication 25 August 2025; date of current version 3 December 2025.
This work was supported in part by Guangdong Major Project of Basic and
Applied Basic Research under Grant 2019B030302002, in part by Guangdong
Provincial Natural Science Foundation Project under Grant 2025A1515010113,
in part by Guangxi Key Research and Development Project under Grant
2024AB02018, in part by the Innovative Development Joint Fund of Natural
Science foundation in Shandong Province under Grant ZR2024LZH012, and
in part by the Major Key Project of PCL, China, under Grant PCL2025AS11.
Recommended for acceptance by A. Sharma. (Corresponding author: Weiwei
Lin.)
Shengsheng Lin, Haojun Chen, and C. L. Philip Chen are with the School
of Computer Science and Engineering, South China University of Technology,
Guangzhou 510641, China.
Weiwei Lin is with the School of Computer Science and Engineering, South
China University of Technology, Guangzhou 510641, China, and also with
Pengcheng Laboratory, Shenzhen 518055, China (e-mail: linww@scut.edu.cn).
Wentai Wu is with the Department of Computer Science, College of Information Science and Technology, Jinan University, Guangzhou 510632, China.
The code is publicly available at this repository: https://github.com/lss-1138/
SparseTSF.
Digital Object Identifier 10.1109/TPAMI.2025.3602445

Fig. 1. Resampling the electricity dataset with a daily interval results in
subsequences with similar patterns.

Neural Networks (CNNs) [8], [9], [10], [11], Graph Neural
Networks (GNNs) [12], [13], [14], and Transformers [15], [16],
[17], [18], [19]. In recent years, there has been a growing interest
in Long-term Time Series Forecasting (LTSF), which demands
models to provide an extended predictive view for advanced
planning [20], [21], [22], [23], [24].
Although a longer predictive horizon offers convenience, it
also introduces greater uncertainty [25]. This demands models
capable of extracting more extensive temporal dependencies
from longer historical windows. Consequently, modeling becomes more complex to capture these long-term temporal dependencies. For instance, Transformer-based models often have
millions or tens of millions of parameters, limiting their practical
usability, especially in scenarios with restricted computational
resources [26], [27].
In fact, the basis for accurate long-term time series forecasting
lies in the inherent periodicity and trend of the data [28], [29],
[30], [31], [32]. For example, long-term forecasts of household
electricity consumption are feasible due to the clear daily
and weekly patterns in such data. Particularly for daily
patterns, resampling electricity consumption at a specific time
each day into daily sequences results in subsequences that
exhibit similar or consistent trends, as illustrated in Fig. 1.
In this case, the original sequence’s periodicity and trend are
decomposed and transformed. That is, daily periodic patterns
are transformed into inter-subsequence dynamics, while trend
patterns are reinterpreted as intra-subsequence characteristics.
This decomposition offers a novel perspective for designing
lightweight LTSF models.
In this paper, we pioneer the exploration of how to utilize
this inherent periodicity and decomposition in data to construct specialized lightweight time series forecasting models.

0162-8828 © 2025 IEEE. All rights reserved, including rights for text and data mining, and training of artificial intelligence and similar technologies.
Personal use is permitted, but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.

Authorized licensed use limited to: Beijing Jiaotong University. Downloaded on April 28,2026 at 04:57:04 UTC from IEEE Xplore. Restrictions apply.

LIN et al.: SPARSETSF: LIGHTWEIGHT AND ROBUST TIME SERIES FORECASTING VIA SPARSE MODELING

Fig. 2. Comparison of MSE and parameters between SparseTSF and other
mainstream models on the Traffic dataset with a forecast horizon of 720.

171

on cross-period trend prediction, effectively extracting periodic features while minimizing the model’s complexity
and parameter count.
r The Sparse technique serves as a form of structural implicit
regularization, enabling the model to focus more on significant items within historical periods, thereby significantly
enhancing the model’s robustness.
r Based on the Sparse technique, we present the SparseTSF
model, which requires fewer than 1,000 parameters, significantly reducing the computational resource demand of
forecasting models.
r The proposed SparseTSF model not only attains competitive or surpasses state-of-the-art predictive accuracy with a
remarkably minimal parameter scale but also demonstrates
robust generalization capabilities.
II. RELATED WORK

Specifically, we introduce SparseTSF, an extremely lightweight
LTSF model built on a backbone of either a single-layer linear
architecture or shallow Multi-Layer Perceptrons (MLP). Technically, we propose the Cross-Period Sparse Forecasting technique
(hereinafter referred to as Sparse technique). It first downsamples the original sequences with constant periodicity into
subsequences, then performs predictions on each downsampled
subsequence, simplifying the original time series forecasting
task into a cross-period trend prediction task. This approach
yields two benefits: (i) effective decoupling of data periodicity
and trend, enabling the model to stably identify and extract
periodic features while focusing on predicting trend changes,
and (ii) extreme compression of the model’s parameter size,
significantly reducing the demand for computational resources.
As shown in Fig. 2, SparseTSF achieves near state-of-the-art
prediction performance with less than 1,000 trainable parameters under the long look-back window setting (e.g., 720), which
makes it 1∼4 orders of magnitude smaller than its counterparts.
Based on our previous conference version [33], this journal
paper introduces several substantial extensions, including:
r The introduction of the SparseTSF/MLP variant, which
significantly enhances the performance of the SparseTSF
model in high-dimensional multivariate forecasting scenarios.
r A more in-depth theoretical analysis demonstrating that
the proposed Sparse technique acts as implicit regularization within the model architecture, which substantially
improves the model’s robustness and generalization capabilities.
r The experiments have been expanded to provide more thorough validation of the SparseTSF model, encompassing
a wider array of datasets, incorporating comparisons for
univariate forecasting scenarios, and including more comprehensive generalization assessments along with further
evaluations.
Finally, the contributions of this paper can be summarized as
follows:
r We propose a novel Cross-Period Sparse Forecasting technique, which downsamples the original sequences to focus

A. Development of the LTSF Field
The LTSF tasks, which aim at predicting over an extended horizon, are inherently more challenging. Initially, the
Transformer architecture [34], known for its robust long-term
dependency modeling capabilities, gained widespread attention in the LTSF domain. Models such as LogTrans [35],
TFT [36], Informer [21], Autoformer [22], Pyraformer [37],
NS-Transformer [38], and FEDformer [39] have modified the
native structure of Transformer to suit time series forecasting
tasks. More recent advancements, like PatchTST [40], Crossformer [41] and PETformer [42], demonstrate that the original
Transformer architecture can achieve impressive results with
an appropriate patch strategy, a technique that is prevalently
employed in the realm of computer vision [43], [44], [45].
Besides Transformer architectures, CNN-based and MLPbased methods are also mainstream approaches in time series
forecasting. Notable examples include SCINet [46], TimesNet [10], MICN [47], ModernTCN [48], NHITS [49], HDMixer [50], TimeMixer [51] and SOFTS [52]. Recent studies
have shown that transferring pretrained Large Language Models
(LLMs) to the time series domain can also yield commendable
results [53], [54], [55], [56], [57], [58], [59], [60]. Moreover, recent works have revealed that RNN and GNN networks can also
perform well in LTSF tasks, as exemplified by SegRNN [25],
WITRAN [61], SutraNets [62], FourierGNN [63], and CrossGNN [13]. However, despite their strong performance, these
models often require a large number of parameters, which limits
their applicability in real-world scenarios.
B. Progress in Lightweight Forecasting Models
Since DLinear [64] demonstrated that simple models could already extract strong temporal periodic dependencies, numerous
studies have been pushing LTSF models towards lightweight
designs, including LightTS [65], TiDE [66], TSMixer [67],
SSCNN [27], CycleNet [29], and TimeBase [68]. Recently, FITS
emerged as a milestone in the lightweight LTSF process, being
the first to reduce the LTSF model scale to the 10k parameter
level while maintaining excellent predictive performance [26].

Authorized licensed use limited to: Beijing Jiaotong University. Downloaded on April 28,2026 at 04:57:04 UTC from IEEE Xplore. Restrictions apply.

172

IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE, VOL. 48, NO. 1, JANUARY 2026

FITS achieved this by transforming time-domain forecasting
tasks into frequency-domain ones and using low-pass filters
to reduce the required number of parameters. In this paper,
our proposed SparseTSF model takes lightweight model design
to the extreme. Utilizing the Cross-Period Sparse Forecasting
technique, it’s the first to reduce model parameters to below 1k.
III. METHODOLOGY
A. Preliminaries
1) Long-Term Time Series Forecasting: The task of LTSF
involves predicting future values over an extended horizon using
previously observed multivariate time series (MTS) data. It
is formalized as ȳt+1:t+H = fθ (xt−L+1:t ), where xt−L+1:t ∈
RL×C and ȳt+1:t+H ∈ RH×C . In this formulation, L represents
the length of the historical look-back window, C is the number of
distinct features or channels, and H is the length of the forecast
horizon. The main goal of LTSF is to extend the forecast horizon
H as it provides richer and more advanced guidance in practical
applications. However, an extended forecast horizon H also
increases the complexity of the model, leading to a significant
increase in parameters in mainstream models. To address this
challenge, our research focuses on developing models that are
not only extremely lightweight but also robust and effective.
2) Channel-Independent Strategy: Recent advancements in
the field of LTSF have seen a shift towards a channelindependent (CI) approach, especially when dealing with multivariate time series data [40], [69]. This strategy simplifies the
forecasting process by focusing on individual univariate time
series within the dataset. Instead of the traditional approach,
which utilizes the entire multivariate historical data to predict
future outcomes, the CI method finds a shared function fθ :
(i)
(i)
xt−L+1:t ∈ RL → ȳt+1:t+H ∈ RH for each univariate series.
This approach provides a more targeted and simplified prediction
model for each channel, reducing the complexity of accounting
for inter-channel relationships.
As a result, the main goal of mainstream state-of-the-art
models in recent years has shifted towards effectively predict by modeling long-term dependencies, including periodicity
and trends, in univariate sequences. For instance, models like
DLinear achieve this by extracting dominant periodicity from
univariate sequences using a single linear layer [64]. More
advanced models, such as PatchTST [40] and TiDE [66], employ
more complex structures on single channels to extract temporal
dependencies, aiming for superior predictive performance.
In this paper, we also adopt the CI strategy and aim to explore
how to design an even more lightweight yet effective approach
for modeling long-term dependencies in single-channel time
series. In addition, we provide a detailed discussion of the
advantages and limitations of both CI and channel-dependent
(CD) strategies in Section V-D6. For simplicity, we omit the
channel dimension in the following formulations, assuming that
the model input is x ∈ RL and the model output is ȳ ∈ RH .
B. SparseTSF
Given that the data to be forecasted often exhibits constant,
periodicity a priori (e.g., electricity consumption and traffic

Algorithm 1: Overall Pseudocode of SparseTSF.
Require: Historical look-back window x ∈ RL
Ensure: Forecasting
result ȳ ∈ RH
L
1
1: μ ← L i=1 xi
2: x ← x − μ /* Normalize the sequence */
3: x ← Conv1d(x, 2 ×  w2  + 1) + x /* Sliding
aggregation */
4: X ← Reshape(x , (n, w)) /* Downsampling into
n × w matrix */
5: Ȳ ← Backbone(X) /* Forecasting each subsequence
via shared Linear or MLP */
6: ȳ ← Reshape(Ȳ , (H)) /* Upsampling to the final
forecast sequence */
7: ȳ ← ȳ + μ /* De-normalize the forecast result */

flow typically have fixed daily cycles), we propose the CrossPeriod Sparse Forecasting technique to enhance the extraction of
long-term sequential dependencies while reducing the model’s
parameter scale. Utilizing a single linear layer or a dual-layer
MLP to model the LTSF task within this framework leads to our
SparseTSF model, as illustrated in Fig. 3 and Algorithm 1.
1) Cross-Period Sparse Forecasting: Assuming that the time
series x has a known periodicity w, the first step is to downsample the original series into w subsequences, each of length
L
. A backbone model with shared parameters is then
n = w
applied to these subsequences for prediction. After prediction,
the w subsequences, each of length m =  H
w , are upsampled
back into a complete forecast sequence of length H.
Mathematically, the downsampling process can be represented as:
Xi,k = xi+k×w ,

(1)

where i = 1, 2, . . . , w and k = 1, 2, . . . , n. The sparse forecasting is equivalent to using the downsampled historical sequences
directly to predict the future subsequences:
Ȳi = fθ (Xi ),

(2)

where i = 1, 2, . . . , w, and fθ is the shared backbone model
for prediction. In this paper, the backbone is either a simple
single-layer Linear model or a dual-layer MLP. After obtaining
the predicted future subsequences, the upsampling process can
be represented as:
ȳi+l×w = Ȳi,l ,

(3)

where i = 1, 2, . . . , w and l = 1, 2, . . . , m.
Technically, both the downsampling and upsampling processes can be efficiently implemented via matrix reshaping and
transposition (assuming L and H are divisible by w). Specifically, the downsampling process corresponds to reshaping x of
length L into an n × w matrix, which is then transposed into
a w × n matrix X. The upsampling step transposes the w × m
matrix Ȳ and reshapes it back into a complete forecast sequence
ȳ of length H.
At this stage, the basic sparse forecasting process has been
implemented. However, this approach still faces two issues: (i)
potential loss of information, as only one data point per period

Authorized licensed use limited to: Beijing Jiaotong University. Downloaded on April 28,2026 at 04:57:04 UTC from IEEE Xplore. Restrictions apply.

LIN et al.: SPARSETSF: LIGHTWEIGHT AND ROBUST TIME SERIES FORECASTING VIA SPARSE MODELING

173

Fig. 3. SparseTSF architecture. SparseTSF/Linear and SparseTSF/MLP represent using a single-layer Linear model and a dual-layer MLP model, respectively,
as the backbone of SparseTSF.

Fig. 4.

Schematic illustration of SparseTSF on the time axis.

is utilized for prediction while the others are ignored, and (ii)
amplification of the impact of outliers, as extreme values within
the downsampled subsequences can directly affect the prediction
of specific subsequences.
To address these issues, we additionally perform a sliding
aggregation on the original sequence before executing sparse
forecasting, as illustrated in Fig. 3. Each aggregated data point
incorporates information from neighboring points within its
period, addressing issue (i). Moreover, as the aggregated value
represents a weighted average of surrounding points, it mitigates
the impact of outliers, thus resolving issue (ii). Technically, this
sliding aggregation can be implemented using a 1D convolution
with zero-padding and a kernel size of 2 ×  w2  + 1. The process
can be formulated as follows:
x = x + Conv1D(x).

(4)

Intuitively, the overall forecasting process of SparseTSF appears as a sliding forecast with a sparse interval of w, performed
by a unified backbone with parameter sharing within a constant
period w. This can be viewed as a model performing sparse
sliding prediction across periods, as illustrated in Fig. 4.
2) Backbone: The goal of this paper is to explore the extreme limits of model lightweighting for time series forecasting. Therefore, we use a simple single-layer linear model or a
dual-layer MLP model (with ReLU [70] as the activation function) as the backbone fθ , constructing SparseTSF/Linear and
SparseTSF/MLP. The former offers the utmost simplicity for the
model, while the latter possesses a stronger nonlinear learning
capability, making it more suitable for more complex scenarios,
such as high-dimensional multivariate forecasting tasks. This

is because nonlinearity allows the model to capture and distinguish different patterns across channels in high-dimensional
data, whereas pure linear models struggle to differentiate effectively [71].
3) Instance Normalization: Time series data often exhibit
distributional shifts between training and testing datasets. Recent studies have shown that employing simple instance normalization strategies between the input and output of models can
help mitigate this issue [25], [71], [72], [73]. In our work, we also
utilize a straightforward normalization strategy. Specifically, we
subtract the mean of the sequence (i.e., μ) from itself before it
enters the model and add it back after the model’s output. This
process is formulated as follows:
x = x − μ,

(5)

ȳ = ȳ + μ.

(6)

4) Loss Function: In alignment with current mainstream
practices in the field, we adopt the classic Mean Squared Error (MSE) as the loss function for SparseTSF. This function
measures the discrepancy between the predicted values ȳ and
the actual ground truth y. It is formulated as:
Loss = y − ȳ 22 .

(7)

IV. THEORETICAL ANALYSIS
In this section, we conduct a theoretical analysis using
SparseTSF/Linear as an example, focusing on its parameter
efficiency and the effectiveness of the Sparse technique. We
demonstrate that the Sparse technique acts as an implicit 1
regularization, enhancing the model’s robustness to noise and
improving generalization performance.
A. Parameter Efficiency
Theorem IV.1 (Parameter Count): Given a historical lookback window length L, a forecast horizon H, and a constant
periodicity w, the total number of parameters required for the
L
w
 × H
SparseTSF model is  w
w  + 2 ×  2  + 1.
Proof: The SparseTSF model consists of two main components: a 1D convolutional layer for sliding aggregation and a linear layer for sparse sliding prediction. The number of parameters

Authorized licensed use limited to: Beijing Jiaotong University. Downloaded on April 28,2026 at 04:57:04 UTC from IEEE Xplore. Restrictions apply.

174

IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE, VOL. 48, NO. 1, JANUARY 2026

in the 1D convolutional layer (without bias) is determined by the
kernel size, which is 2 ×  w2  + 1. For the linear layer (without
L
bias), the number of parameters is n × m, where n =  w
 and
H
m =  w , respectively.
By combining the parameters from both layers, the total count
L
w
 × H
is: n × m + 2 ×  w2  + 1 =  w
w  + 2 ×  2  + 1. 
In LTSF tasks, the look-back window length L and forecast
horizon H are usually quite large, for instance, up to 720, while
the intrinsic periodicity w of the data is also typically large, such
L
w
 × H
as 24. In this scenario,  w
w  + 2 ×  2  + 1 = 925
L × H = 518, 400. This means that the parameter scale of the
SparseTSF model is much lighter than even the simplest standard linear model. This demonstrates the extremely lightweight
nature of the SparseTSF model.

to the original output sequence ȳ ∈ RH :

B. Implicit Regularization
In this subsection, we demonstrate that the sparsification
introduced by the Sparse technique in SparseTSF acts as an
implicit regularization on the model’s weights, which enhances
the model’s robustness to noise and improves generalization
performance.
Theorem IV.2 (Optimization Objective for SparseTSF/Linear
Model): The optimization objective for a Linear model combined with Sparse technique is given by:


min y − Wf x 22 + α Wforbid 1 + β σ 2 (Wshared )1 , (8)
Wf

where α, β → ∞, and:
Wforbid 1 =

m 
n 


|(Wf )i+kw,j+lw |,

(9)

m
 n


 2
σ (Wshared ) =
|σ 2 ((Wf )i+kw,j+lw )|,
1



k=1 l=1

(10)

k=1 l=1 i=j

i=j

L
where m =  H
w , n =  w , i, j = 1, 2, . . . , w.
Proof: In the Sparse technique, we reformulate the prediction
problem by utilizing the downsampled historical sequence X ∈
Rw×n to predict the downsampled future sequence Y ∈ Rw×m ,
L
 and m =  H
where n =  w
w  represent the lengths of the
downsampled historical and future subsequences, respectively.
Using a weight-shared fully connected linear layer (without
bias) to model all subsequence prediction tasks, we express this
as:

Ȳi = Ws Xi , i = 1, 2, . . . , w,

(11)

where Ȳi ∈ Rm , Xi ∈ Rn , and Ws ∈ Rm×n . The optimization
objective for the subsequence prediction task is given by:
min
Ws

w

i=1

Fig. 5. The complete equivalent parameter matrix Wf obtained by appropriately copying and zero-padding the sparse prediction parameter matrix Ws .

Yi − Ws Xi 22 .

(12)

Here, by appropriately copying and zero-padding the elements of Ws ∈ Rm×n , we can obtain an equivalent weight matrix Wf ∈ RH×L that maps the original input sequence x ∈ RL

(Wf )i+kw,j+lw =

(Ws )k,l
0

if i = j,
if i = j,

(13)

where k = 1, 2, . . . , m, l = 1, 2, . . . , n and i, j = 1, 2, . . . , w.
Fig. 5 illustrates the generation of the equivalently transformed parameter matrix Wf . The matrix Wf is divided into
m × n smaller matrices, each shaped w × w. Within each small
matrix, each element of Ws is copied (shared) w times along
the diagonal, corresponding to the condition i = j, while offdiagonal elements are filled with zeros, corresponding to the
condition i = j.
Importantly, the equivalent parameter matrix Wf in Fig. 5 can
also be derived directly from the full weight matrix Wf „ which
is used in the original sequence prediction task, by introducing two distinct regularization terms. These two regularization
terms enforce the sparse and weight-sharing properties in Wf ,
respectively.
Specifically, the first regularization term, α Wforbid 1 , enforces sparsity by penalizing the absolute sum of off-diagonal
elements in the smaller w × w matrices of Wf . This term can
be expressed as:
α Wforbid 1 = α

m
 n


|(Wf )i+kw,j+lw |.

(14)

k=1 l=1 i=j

As α → ∞, this term ensures that the off-diagonal elements are
driven to zero, creating the block-sparse structure observed in
Wf .
The second regularization term, β σ 2 (Wshared ) 1 , enforces
weight-sharing by constraining the variance of diagonal elements within each w × w submatrix of Wf . This term is expressed as:
m 
n



|σ 2 ((Wf )i+kw,j+lw )|.
β σ 2 (Wshared )1 = β



k=1 l=1

(15)

i=j

As β → ∞, this term forces the diagonal elements to be equal,
aligning with the repeated diagonal items observed in Wf .
By combining these two regularization terms with the original
sequence prediction loss, we obtain the complete optimization

Authorized licensed use limited to: Beijing Jiaotong University. Downloaded on April 28,2026 at 04:57:04 UTC from IEEE Xplore. Restrictions apply.

LIN et al.: SPARSETSF: LIGHTWEIGHT AND ROBUST TIME SERIES FORECASTING VIA SPARSE MODELING

175

TABLE I
SUMMARY OF DATASETS

objective:



min y − Wf x 22 + α Wforbid 1 + β σ 2 (Wshared )1 , (16)
Wf

where α, β → ∞.
After sufficient training, the learned parameter matrix Wf will
become equivalent to the augmented matrix Wf of the sparse
prediction parameter matrix Ws . Therefore, the optimization objective (12) for a Linear model combined with sparse forecasting
is equivalent to (16).

Definition IV.3 (Optimization Objective of Linear Model With
1 Regularization): The optimization objective for a Linear
model combined with 1 regularization on the original sequence
prediction task is given by:
min y − Wf x 22 + λ Wf 1 ,
Wf

(17)

where y ∈ RH , x ∈ RL , Wf ∈ RH×L , and λ > 0 is the regularization parameter.
Introducing 1 regularization in neural networks facilitates
feature selection by promoting sparsity, driving the weights
associated with less significant features toward zero, and consequently enhancing the model’s robustness and generalization [74]. Comparing Definition IV.3 and Theorem IV.2, it can be
observed that both optimization objectives share the same form.
Specifically, the first term (empirical loss y − Wf x 22 ) quantifies the discrepancy between the model’s predictions and the
actual values, while the additional terms act as a regularization
term that constrains model complexity.
In particular, in the optimization objective of the Sparse
technique (8), the second term α Wforbid 1 imposes a structural
constraint by forcing the weights of weakly correlated features
to zero, and the third term, β σ 2 (Wshared ) 1 , enforces parameter
sharing across structures, thereby reducing the model’s degrees
of freedom. This form of regularization plays a role similar
to that of 1 regularization in encouraging sparsity, but it is
not equivalent. Unlike standard 1 , our formulation introduces
domain-informed structural constraints (i.e., zeroing out offdiagonal blocks and enforcing intra-block weight sharing) that
cannot be captured by unstructured 1 or p norms alone.
In conclusion, the sparse forecasting technique introduced by
SparseTSF functions as an implicit regularization mechanism
(analogous to 1 regularization), improving the model’s robustness and generalization capabilities. This structural sparsity
design allows it to target specific patterns within the parameter
space, enabling SparseTSF to more effectively extract meaningful features when handling noisy and complex data, ultimately
achieving extreme model compression while improving forecasting performance.
V. EXPERIMENTS
In this section, we present the experimental results of
SparseTSF on mainstream LTSF benchmarks. Additionally, we
discuss the efficiency advantages brought by the lightweight
architecture of SparseTSF. Furthermore, we conduct ablation
studies and analysis to further reveal the effectiveness of the
Sparse technique.

A. Experimental Setup
Datasets: We conducted experiments on the mainstream
LTSF datasets characterized by varied periodicity: ETTh1,
ETTh2, Electricity, Traffic, Solar-Energy, and Weather. These
datasets cover various domains, sampling frequencies, and
scales. The hyperparameter w of SparseTSF is set to correspond
to the inherent daily cycle of the data (e.g., w = 24 for ETTh1)
or adjusted to a smaller value for datasets with an exceptionally
long daily cycle (e.g., w = 4 for Weather). The criteria for
selecting this hyperparameter are explained in Section V-D3.
Detailed information about these datasets is presented in Table I.
Baselines: We compared our approach against state-of-theart and representative methods in the field, including Autoformer [22], FEDformer [39], FiLM [75], DLinear [64],
PatchTST [40], iTransformer [19] and FITS [26]. Following
FITS, SparseTSF adopts a default look-back length of 720, aiming to capture useful temporal dependencies from a longer historical window while maintaining an extremely compact model
design. Additionally, it is important to note that a long-standing
bug existed among these baselines, which caused the last batch
of test data to be discarded during the testing phase, leading to
inaccurate evaluations [76]. To ensure a fair comparison, we
reran these baselines after fixing this issue, using a uniform
look-back length of 720.
Environment: We implemented SparseTSF using PyTorch [77] and executed all experiments with the Adam optimizer [78] for 30 epochs. A learning rate decay of 0.8 was applied after the initial 3 epochs, and early stopping was employed
with a patience of 5 epochs. All experiments were conducted on
a single NVIDIA RTX 4090 GPU with 24 GB of memory.

B. Main Results
Table II presents a performance comparison between
SparseTSF and other baseline models in the multivariate time
series forecasting task. It is evident that SparseTSF (both
the Linear and MLP variants) ranks within the top three in
most scenarios, achieving or closely approaching state-of-theart levels with a significantly smaller parameter scale. In this
multivariate forecasting context, SparseTSF/MLP outperforms
SparseTSF/Linear, securing 39 top-three placements compared
to 24 for SparseTSF/Linear. This is attributed to SparseTSF’s
adoption of the mainstream channel-independent strategy, where
a shared model is used to model multichannel sequences. In this
case, the nonlinear capabilities of the MLP allow it to capture
distinct temporal patterns across multiple channels, leading to
enhanced forecasting performance [71].

Authorized licensed use limited to: Beijing Jiaotong University. Downloaded on April 28,2026 at 04:57:04 UTC from IEEE Xplore. Restrictions apply.

176

IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE, VOL. 48, NO. 1, JANUARY 2026

TABLE II
COMPARISON OF MULTIVARIATE LTSF RESULTS BETWEEN SPARSETSF AND OTHER MAINSTREAM MODELS

TABLE III
COMPARISON OF UNIVARIATE LTSF RESULTS BETWEEN SPARSETSF AND OTHER MAINSTREAM MODELS

Furthermore, Table III displays the comparative results of
SparseTSF and other baseline models in the univariate time
series forecasting task. SparseTSF continues to demonstrate exceptional performance, with SparseTSF/Linear achieving stateof-the-art results overall. In contrast to the multivariate forecasting task, SparseTSF/Linear outperforms SparseTSF/MLP here.
Additionally, linear-based models such as FITS and DLinear

also show better performance compared to deeper nonlinear
models like PatchTST. This is because, in unichannel forecasting tasks, linear-based methods possess sufficient predictive
capability and robustness without requiring additional nonlinear
capacity to fit multiple channel patterns.
In summary, the findings above emphatically demonstrate
the superiority of the Sparse technique proposed in this paper.

Authorized licensed use limited to: Beijing Jiaotong University. Downloaded on April 28,2026 at 04:57:04 UTC from IEEE Xplore. Restrictions apply.

LIN et al.: SPARSETSF: LIGHTWEIGHT AND ROBUST TIME SERIES FORECASTING VIA SPARSE MODELING

TABLE IV
STATIC AND RUNTIME METRICS OF SPARSETSF AND OTHER MAINSTREAM
MODELS ON THE TRAFFIC DATASET

177

TABLE V
COMPARISON OF THE SCALE OF PARAMETERS BETWEEN SPARSETSF/LINEAR
AND FITS MODELS UNDER DIFFERENT CONFIGURATIONS OF LOOK-BACK
LENGTH AND FORECAST HORIZON, WHERE SPARSETSF OPERATES WITH
w = 24 AND FITS EMPLOYS COF AT THE 2TH HARMONIC

TABLE VI
ABLATION RESULTS OF THE SPARSE TECHNIQUE

Specifically, as stated in Theorem IV.2, the Sparse technique
acts as an implicit regularization, allowing the model to focus
more on the important elements of historical periodicity while
reducing attention to irrelevant items. This process more effectively decouples the periodicity and trends from the data, thereby
enabling exceptional predictive performance in long-horizon
scenarios.
C. Efficiency Advantages of SparseTSF
Beyond its powerful predictive performance, another significant benefit of the SparseTSF model is its extreme lightweight
nature. Previously, Fig. 2 visualized the parameter-performance
comparison of SparseTSF with other mainstream models.
Here, we further present a comprehensive comparison between
SparseTSF and these baseline models in terms of both static and
runtime metrics, including:
1) Parameters: The total number of trainable parameters in
the model, representing the model’s size.
2) MACs (Multiply-Accumulate Operations): A common
measure of computational complexity in neural networks,
indicating the number of multiply-accumulate operations
required by the model.
3) Max Memory: The maximum memory usage during the
model training process.
4) Epoch Time: The training duration for a single epoch. This
metric was averaged over 3 runs.
Table IV displays the comparative results. It is evident that
SparseTSF significantly outperforms other models in terms
of both static metrics, such as the number of parameters,
and runtime metrics like maximum memory usage. Notably,
SparseTSF/Linear reduces the model size required for LTSF
tasks to below one thousand parameters for the first time.
Furthermore, the SparseTSF/MLP variant, which possesses
stronger multivariate forecasting capabilities, also maintains a
parameter count of under eight thousands, which is significantly
lower than that of other baselines. This characteristic allows
SparseTSF to be deployed on devices with very limited computational resources.
Additionally, we conducted a comprehensive comparison
with FITS, a recent milestone work in the field of LTSF model
lightweight progression. The results in Table V reveal that
SparseTSF significantly surpasses FITS in terms of parameter

scale under any input-output length configuration. Therefore,
SparseTSF marks another significant advancement in the journey towards lightweight LTSF models.
D. Ablation Studies and Analysis
Beyond its ultra-lightweight characteristics, the Sparse technique also possesses a robust capability to extract periodic
features, which we will delve further into in this section.
1) Effectiveness of the Sparse Technique: To validate the
effectiveness of the Sparse technique, we conducted ablation
experiments on three different datasets: (i) ETTh1, a dataset with
daily periodicity sampled hourly; (ii) Traffic, a dataset with both
daily and weekly periodicity sampled hourly; and (iii) Weather, a
dataset with daily periodicity sampled every 10 minutes. For the
first two datasets, we set the hyperparameter w to reflect the daily
periodic length, specifically w = 24. However, for the Weather
dataset, the daily periodic length is relatively long (i.e., 144),
and directly setting w to this length would result in excessively
short downsampled subsequences, leading to underutilization
of information. To strike a balance, we set w to 4 for this
dataset. The criteria for selecting the hyperparameter w are
further discussed in the subsequent section.
From the ablation results in Table VI, it is evident that the
Sparse technique improves model performance in most cases.
Specifically, in the ETTh1 dataset, both the Linear and MLP
backbone models benefit significantly from SparseTSF. In the
Traffic and Weather datasets, although there are a few instances

Authorized licensed use limited to: Beijing Jiaotong University. Downloaded on April 28,2026 at 04:57:04 UTC from IEEE Xplore. Restrictions apply.

178

Fig. 6.

IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE, VOL. 48, NO. 1, JANUARY 2026

Visualization of normalized weights of the model trained on the Traffic dataset with both look-back length (X-axis) and forecast horizon (Y -axis) of 336.

where the mean squared error (MSE) slightly increases, the
Sparse technique generally leads to a decrease in error. Overall,
these results emphatically illustrate the efficacy of the Sparse
technique.
2) Representation Learning of the Sparse Technique: In
Section IV, we theoretically analyzed the reasons why the Sparse
technique can enhance the performance of forecasting tasks.
Here, we further reveal the role of the Sparse technique from a
representation learning perspective. Fig. 6 shows the distribution
of normalized weights for both the trained Linear model and the
SparseTSF/Linear model.
In this context, the weight of the Linear model is an L × H
matrix, which can be directly obtained. However, since the
SparseTSF model is a sparse model, we need to derive its equivalent weights. As shown in (13), we can obtain an equivalent
weight matrix of shape L × H through weight copying and zero
padding. For ease of operation, we can also implement this using
a simpler approach: (i) first, input H one-hot encoded vectors
of length L into the SparseTSF model (when L equals H, this
simplifies to an identity matrix, where diagonal elements are 1
and all other elements are 0); (ii) then, obtain and transpose the
corresponding output to get the equivalent L × H weight matrix
of SparseTSF. When L equals H, this process is formulated as:
⎛⎡

1 0
⎜⎢
0 1
⎢
⎜
⎢
weight = SparseTSF ⎜
⎜⎢ .. ..
⎝⎣ . .
0 0

···
···
..
.
···

⎤⎞
0
⎥⎟
0⎥ ⎟
⎟
.. ⎥
⎥⎟ .
. ⎦⎠
1

(18)

From the visualization in Fig. 6, two observations can be
made: (i) The Linear model can learn evenly spaced weight
distribution stripes (i.e., periodic features) from the data, indicating that single linear layer can already extract the primary
periodic characteristics from a univariate series with the CI
strategy. These findings are consistent with previous research
conclusions [64]. (ii) Compared to the Linear model, SparseTSF
learns more distinct evenly spaced weight distribution stripes,
indicating that SparseTSF has a stronger capability in extracting

periodic features. This is because the Sparse technique provides
the model with a structural prior bias, enabling it to focus more
on historically relevant items while reducing attention to weakly
correlated items. This phenomenon aligns with the findings in
Section IV.
In conclusion, the Sparse technique can enhance the model’s
performance in LTSF tasks by strengthening its ability to extract
periodic features from data.
3) Impact of the Hyperparameter w: Through the theoretical
analysis in Section IV, we observe that the Sparse technique
serves as implicit regularization, which constrains the model
to ignore weakly related elements and enforces core weight
sharing. An appropriate level of regularization can enhance
the model’s robustness against noisy data, thereby improving
predictive performance. Here, the hyperparameter w in Sparse
is a key factor determining the strength of regularization; specifically, a larger w results in stronger regularization.
Fig. 7 illustrates the performance of the SparseTSF/Linear
model under different hyperparameter w values. For the Traffic
dataset, the best performance is achieved at w = 24 (which
aligns with its daily periodicity). In this case, the Sparse technique enables cross-period trend forecasting, allowing the model
to focus on the corresponding historical data, effectively decoupling the periodicity and trends. When w < 24, even without
precise alignment to the periodic length, the Sparse technique
maintains decent performance due to its regularization effect,
which enhances the model’s focus on important features. However, when w > 24 (exceeding the daily period), overly sparse
parameter connections lead to a loss of critical information
regarding periodic features, resulting in significant performance
degradation (i.e., increased error). As shown in Fig. 7, at w = 24,
the number of parameters in SparseTSF is reduced by a factor
of 500 compared to the original direct forecasting approach,
achieving extreme parameter compression. Therefore, for this
hourly-sampled daily periodic dataset (e.g., ETTh1 and Electricity), we recommend setting w = 24 to achieve the best
performance-efficiency balance.
In contrast, for the Weather dataset (with data collected
every ten minutes), theoretically, w = 144 matches its daily

Authorized licensed use limited to: Beijing Jiaotong University. Downloaded on April 28,2026 at 04:57:04 UTC from IEEE Xplore. Restrictions apply.

LIN et al.: SPARSETSF: LIGHTWEIGHT AND ROBUST TIME SERIES FORECASTING VIA SPARSE MODELING

179

Fig. 7. Performance and parameter size comparison of the SparseTSF/Linear model under different hyperparameters w. Both the look-back length and forecast
horizon are set to 720, and the results are averaged over 10 runs.

periodicity for cross-period trend forecasting. However, an excessively large w can lead to overly strong regularization (as
the downsampled subsequences become very short), resulting
in insufficient information for the model during predictions. In
this case, directly setting w to 144 would lead to significant
performance declines. Nonetheless, setting w to a smaller value
for appropriate regularization can still enhance the model’s
robustness and yield satisfactory performance. For instance,
when w = 4, the model’s predictive accuracy improves while the
parameter size decreases by more than 10 times. Therefore, for
this densely sampled daily periodic dataset (e.g., Solar-Energy),
we recommend a smaller w (e.g., w = 4) to achieve the optimal
balance of performance and efficiency.
4) Generalization Ability of the SparseTSF: The Sparse technique enhances the model’s robustness, particularly by improving its utilization of historical periodic characteristics. Therefore, the generalization capability of a trained SparseTSF model
on different datasets with the same principal periodicity is
promising. To investigate this, we further studied the crossdomain generalization performance of the SparseTSF model
(i.e., training on a dataset from one domain and testing on a
dataset from another). Specifically, we evaluated the generalization ability of SparseTSF under four different transfer scenarios:
1) ETTh2 → ETTh1: Same period length (w = 24) and same
number of variables (C = 7). This can be considered a
transfer between very similar domains, as the two datasets
are of the same type but collected from different machines,
and forecasting models are expected to perform best in this
setting.
2) Electricity → ETTh1: Partially overlapping period lengths
(both have a daily cycle length of 24; in addition, Electricity exhibits a strong weekly cycle length of 168), and
different numbers of variables (Electricity has 321 variables). In this case, CI-based methods (such as SparseTSF
and PatchTST) can work properly, while traditional multivariate models (e.g., FEDformer) fail because their architecture treats the number of variables as a crucial structural
parameter. Although iTransformer is a multivariate model,
its actual implementation is decoupled from the number of
variables by transposing the Transformer and using selfattention to aggregate inter-channel dependencies. Thus,
it can work in this scenario.

3) Solar-Energy → ETTh1: The period lengths are mismatched (Solar-Energy has a period length of 144 and
the hyperparameter w is set to 4) and different numbers
of variables (Solar-Energy contains 137 variables). This
setting introduces the largest domain gap, testing how well
the model can generalize when its learned temporal and
structural priors are no longer aligned with the target.
4) Traffic → Electricity: Same period length (both have a
daily cycle of w=24 and a weekly cycle of w=168) but different domains and different numbers of variables (Traffic
has 862 variables, while Electricity has 321). These two
datasets are much larger in scale, so transfer learning
between them poses a more rigorous test on whether
models have captured truly robust feature dependencies.
Experimental results, presented in Table VII, show that
SparseTSF consistently exhibits strong generalization across all
scenarios. In settings with the same period length, such as ETTh2
→ ETTh1 and Traffic → Electricity, both SparseTSF/Linear and
SparseTSF/MLP attain leading performance. In settings with
different period lengths (e.g., Solar-Energy → ETTh1), performance understandably drops due to the misalignment in period
lengths. Nonetheless, SparseTSF/Linear still ranks among the
top two models, indicating a level of resilience even in the
absence of strong periodic priors.
Overall, SparseTSF/Linear performs better when generalizing to ETTh1, likely due to its lower model capacity and stronger
robustness against overfitting. In contrast, SparseTSF/MLP
achieves superior performance on large-scale datasets like Traffic and Electricity, suggesting that such datasets may demand
greater nonlinear modeling capacity to capture more diverse
channel patterns.
Furthermore, compared to other nonlinear models such as
PatchTST, SparseTSF/MLP demonstrates superior robustness,
which we attribute to the structural regularization effects introduced by the Sparse technique. By enforcing cross-period
sparsity, the model avoids overfitting to noise or spurious correlations across variables, which often hinder generalization.
Additionally, among advanced nonlinear models, as a ChannelDependent (CD)-based model, iTransformer performs the worst,
as the inter-channel dependencies learned by Cd-based models
are often domain-specific and thus difficult to transfer across
datasets with different variable structures.

Authorized licensed use limited to: Beijing Jiaotong University. Downloaded on April 28,2026 at 04:57:04 UTC from IEEE Xplore. Restrictions apply.

180

IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE, VOL. 48, NO. 1, JANUARY 2026

TABLE VII
COMPARISON OF GENERALIZATION CAPABILITIES BETWEEN SPARSETSF AND OTHER MAINSTREAM MODELS

Fig. 8. Performance of SparseTSF/Linear with varying look-back
lengths L. The result is averaged over different forecast horizons
H ∈ {96, 192, 336, 720}.

In summary, SparseTSF exhibits excellent generalization capabilities, particularly in transfer scenarios where the dominant
periodicity is preserved. Its strong performance underscores
the effectiveness of the Sparse technique in extracting stable,
transferable temporal representations. This makes SparseTSF
a highly practical choice for real-world applications involving
limited samples, low-quality data, or deployment in resourceconstrained environments.
5) Impact of Look-Back Length: The look-back length determines the richness of historical information that the model
can utilize. Generally, models are expected to perform better
with longer input lengths if they possess robust long-term dependency modeling capabilities. Fig. 8 presents the performance
of SparseTSF at different look-back lengths.
Two phenomena can be observed: (i) longer look-back windows yield better performance, demonstrating SparseTSF’s capacity for modeling long-term dependencies, and (ii) the performance of the ETTh1 and ETTh2 datasets remains relatively

stable across different look-back windows, while the performance of the Traffic and Electricity datasets varies significantly,
particularly between look-backs of 96 to 192, where accuracy
experiences notable fluctuations.
We can further discuss the reasons behind the second point.
The ETTh1 and ETTh2 datasets exhibit a significant daily periodic pattern (w = 24). In this case, look-back lengths of 96
can achieve good results because they fully encompass the daily
periodic pattern. However, the Traffic and Electricity datasets
feature not only a significant daily periodic pattern (w = 24)
but also a noticeable weekly periodic pattern (w = 168). Here, a
look-back of 96 cannot cover the entire weekly periodic pattern,
leading to a significant drop in performance. This underscores
the necessity of sufficiently long look-back lengths (at least covering the entire cycle length) for accurate predictions. Given the
extremely lightweight nature of SparseTSF, we strongly recommend providing adequately long look-back windows whenever
feasible.
6) Impact of Channel Independent Strategy: SparseTSF is
built upon a channel-independent (CI) strategy and achieves
excellent performance under long look-back window settings
(i.e., L = 720) in terms of both forecasting accuracy and computational efficiency. However, many recent state-of-the-art models adopt a channel-dependent (CD) strategy (such as iTransformer [19], Leddam [24], and TQNet [79]) and exhibit strong
performance under short look-back window settings (i.e., L =
96). This is mainly attributed to their well-crafted multi-channel
interaction mechanisms, which are capable of extracting additional cross-variable dependencies to compensate for the limited
temporal context, thereby enhancing prediction accuracy.
However, as the look-back window becomes longer, the advantage of such multivariate deep models starts to diminish.
As shown in Fig. 9, CI-based methods like SparseTSF and
PatchTST gradually match or even outperform CD-based methods (e.g., iTransformer) when more historical data is available.

Authorized licensed use limited to: Beijing Jiaotong University. Downloaded on April 28,2026 at 04:57:04 UTC from IEEE Xplore. Restrictions apply.

LIN et al.: SPARSETSF: LIGHTWEIGHT AND ROBUST TIME SERIES FORECASTING VIA SPARSE MODELING

181

more robust information from longer look-back windows.
In scenarios with insufficient look-back, the lack of sufficient valid information can lead to suboptimal performance from SparseTSF.
Therefore, one of our key future research directions is to
address these potential limitations by designing additional information extraction modules to enhance SparseTSF’s capabilities,
thus achieving a balance between performance and parameter
size.

VII. CONCLUSION
Fig. 9. Performance comparison between CI- and CD-based models under
varying look-back lengths on the Traffic dataset with forecast horizon H = 96.

This phenomenon can be primarily explained by three factors:
(i) a sufficiently long historical window inherently provides rich
temporal information, which can mitigate the lack of explicit
inter-channel modeling; (ii) longer input sequences introduce
more noise, and attempting to model complex variable interactions in such settings may lead to a higher risk of overfitting; and
(iii) as demonstrated in recent work such as RLinear [71], CIbased models equipped with non-linear function approximators
are still capable of effectively modeling multivariate time series.
Specifically, while linear CI models can only capture shared
patterns across all channels, non-linear models can learn diverse
patterns for different variables, making them more expressive
and suitable for complex multivariate forecasting tasks.
This is exactly the motivation behind our introduction of the
SparseTSF/MLP variant in this journal extension. By equipping
SparseTSF with shallow MLP structures, the model gains enhanced capability to capture multi-channel dynamics, thereby
improving its forecasting accuracy on high-dimensional datasets
such as Traffic and Electricity.
VI. DISCUSSION
The SparseTSF model proposed in this paper excels at handling data with a stable primary period, demonstrating enhanced periodic feature extraction capabilities and an extremely
lightweight architecture. However, there are several scenarios
where SparseTSF may face limitations:
1) Ultra-Long Periods: In cases involving ultra-long periods
(e.g., periods exceeding 100), the Sparse technique leads
to overly sparse parameter connections. In such cases,
denser connections (i.e., smaller hyperparameter w) may
be required to maintain model performance.
2) Multiple Periods: SparseTSF may struggle with data that
contains multiple intertwined periods, as the Sparse technique can only downsample and decompose one primary
period. In such cases, SparseTSF can only focus on the
smallest period to ensure effective modeling (e.g., in scenarios where daily and weekly periods overlap, only the
daily period is considered).
3) Insufficient Look-Back Windows: As a sparse and
lightweight forecasting model, SparseTSF tends to extract

In this paper, we introduce the Cross-Period Sparse Forecasting technique and the corresponding SparseTSF model. Through
detailed theoretical analysis and experimental validation, we
have demonstrated the lightweight nature of the SparseTSF
model and its ability to effectively extract periodic features. In
particular, we theoretically prove that the proposed Sparse technique serves as implicit regularization, significantly enhancing
the model’s robustness. Achieving competitive or even surpassing the performance of current state-of-the-art models with a
minimal parameter scale, SparseTSF emerges as a strong candidate for deployment in computational resource-constrained
environments. Additionally, SparseTSF exhibits strong generalization capabilities, opening new possibilities for applications in
small sample and low-quality data scenarios. SparseTSF represents a significant milestone in the development of lightweight
models for long-term time series forecasting. Finally, we aim to
further address the challenges associated with extracting features
from ultra-long-periodic and multi-periodic data, striving to
achieve an optimal balance between model performance and
parameter size in future work.

REFERENCES
[1] Y. Wang, H. Wu, J. Dong, Y. Liu, M. Long, and J. Wang,
“Deep time series models: A comprehensive survey and benchmark,”
2024, arXiv:2407.13278.
[2] Z. Shao et al., “Exploring progress in multivariate time series forecasting:
Comprehensive benchmarking and heterogeneity analysis,” IEEE Trans.
Knowl. Data Eng., vol. 37, no. 1, pp. 291–305, Jan. 2025.
[3] B. N. Oreshkin, D. Carpov, N. Chapados, and Y. Bengio, “N-BEATS:
Neural basis expansion analysis for interpretable time series forecasting,”
in Proc. Int. Conf. Learn. Representations, 2020.
[4] K. G. Olivares, C. Challu, G. Marcjasz, R. Weron, and A. Dubrawski,
“Neural basis expansion analysis with exogenous variables: Forecasting
electricity prices with NBEATSx,” Int. J. Forecasting, vol. 39, no. 2,
pp. 884–900, 2023.
[5] G. Lai, W.-C. Chang, Y. Yang, and H. Liu, “Modeling long-and short-term
temporal patterns with deep neural networks,” in Proc. 41st Int. ACM
SIGIR Conf. Res. Develop. Inf. Retrieval, 2018, pp. 95–104.
[6] S. Smyl, “A hybrid method of exponential smoothing and recurrent neural
networks for time series forecasting,” Int. J. Forecasting, vol. 36, no. 1,
pp. 75–85, 2020.
[7] D. Salinas, V. Flunkert, J. Gasthaus, and T. Januschowski, “DeepAR:
Probabilistic forecasting with autoregressive recurrent networks,” Int. J.
Forecasting, vol. 36, no. 3, pp. 1181–1191, 2020.
[8] S. Bai, J. Z. Kolter, and V. Koltun, “An empirical evaluation of generic convolutional and recurrent networks for sequence modeling,” 2018, arXiv:
1803.01271.
[9] J.-Y. Franceschi, A. Dieuleveut, and M. Jaggi, “Unsupervised scalable
representation learning for multivariate time series,” in Proc. Int. Conf.
Neural Inf. Process. Syst., 2019, Art. no. 418.

Authorized licensed use limited to: Beijing Jiaotong University. Downloaded on April 28,2026 at 04:57:04 UTC from IEEE Xplore. Restrictions apply.

182

IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE, VOL. 48, NO. 1, JANUARY 2026

[10] H. Wu, T. Hu, Y. Liu, H. Zhou, J. Wang, and M. Long, “TimesNet:
Temporal 2D-variation modeling for general time series analysis,” in Proc.
Int. Conf. Learn. Representations, 2023.
[11] A. Q. B. Silva, W. N. Gonçalves, and E. T. Matsubara, “DESCINet:
A hierarchical deep convolutional neural network with skip connection
for long time series forecasting,” Expert Syst. Appl., vol. 228, 2023,
Art. no. 120246.
[12] M. Jin et al., “A survey on graph neural networks for time series: Forecasting, classification, imputation, and anomaly detection,” IEEE Trans.
Pattern Anal. Mach. Intell., vol. 46, no. 12, pp. 10466–10485, Dec. 2024.
[13] Q. Huang et al., “CrossGNN: Confronting noisy multivariate time series
via cross interaction refinement,” in Proc. Int. Conf. Neural Inf. Process.
Syst., 2024, Art. no. 2031.
[14] M. Jin, Y. Zheng, Y.-F. Li, S. Chen, B. Yang, and S. Pan, “Multivariate
time series forecasting with dynamic graph neural ODEs,” IEEE Trans.
Knowl. Data Eng., vol. 35, no. 9, pp. 9168–9180, Sep. 2023.
[15] Q. Wen et al., “Transformers in time series: A survey,”
2022, arXiv:2202.07125.
[16] B. Li et al., “DifFormer: Multi-resolutional differencing transformer with
dynamic ranging for time series analysis,” IEEE Trans. Pattern Anal.
Mach. Intell., vol. 45, no. 11, pp. 13586–13598, Nov. 2023.
[17] Z. Zhang, L. Meng, and Y. Gu, “SageFormer: Series-aware framework for
long-term multivariate time series forecasting,” IEEE Internet Things J.,
vol. 11, no. 10, pp. 18435–18448, May 2024.
[18] R. Ilbert et al., “SAMformer: Unlocking the potential of transformers in
time series forecasting with sharpness-aware minimization and channelwise attention,” in Proc. 41st Int. Conf. Mach. Learn., 2024, Art. no. 841.
[19] Y. Liu et al., “iTransformer: Inverted transformers are effective for time
series forecasting,” in Proc. 12th Int. Conf. Learn. Representations, 2024.
[20] Z. Chen, M. Ma, T. Li, H. Wang, and C. Li, “Long sequence time-series
forecasting with deep learning: A survey,” Inf. Fusion, vol. 97, 2023,
Art. no. 101819.
[21] H. Zhou et al., “Informer: Beyond efficient transformer for long sequence time-series forecasting,” in Proc. AAAI Conf. Artif. Intell., 2021,
pp. 11106–11115.
[22] H. Wu, J. Xu, J. Wang, and M. Long, “Autoformer: Decomposition
transformers with auto-correlation for long-term series forecasting,” in
Proc. Int. Conf. Neural Inf. Process. Syst., 2021, pp. 22419–22430.
[23] X. Qiu, X. Wu, Y. Lin, C. Guo, J. Hu, and B. Yang, “DUET: Dual
clustering enhanced multivariate time series forecasting,” in Proc. 31st
ACM SIGKDD Conf. Knowl. Discov. Data Mining, 2025, pp. 1185–1196.
[24] G. Yu, J. Zou, X. Hu, A. I. Aviles-Rivero, J. Qin, and S. Wang, “Revitalizing
multivariate time series forecasting: Learnable decomposition with interseries dependencies and intra-series variations modeling,” in Proc. Int.
Conf. Mach. Learn., 2024, pp. 57818–57841.
[25] S. Lin, W. Lin, W. Wu, F. Zhao, R. Mo, and H. Zhang, “SegRNN:
Segment recurrent neural network for long-term time series forecasting,”
2023, arXiv:2308.11200.
[26] Z. Xu, A. Zeng, and Q. Xu, “FITS: Modeling time series with 10k
parameters,” in Proc. 12th Int. Conf. Learn. Representations, 2024.
[27] J. Deng, F. Ye, D. Yin, X. Song, I. Tsang, and H. Xiong, “Parsimony or
capability? Decomposition delivers both in long-term time series forecasting,” in Proc. 38th Conf. Neural Inf. Process. Syst., 2024, pp. 66687–66712.
[28] T. Dai et al., “Periodicity decoupling framework for long-term series
forecasting,” in Proc. Int. Conf. Learn. Representations, 2024.
[29] S. Lin, W. Lin, X. Hu, W. Wu, R. Mo, and H. Zhong, “CycleNet: Enhancing
time series forecasting through modeling periodic patterns,” in Proc. 38th
Conf. Neural Inf. Process. Syst., 2024, Art. no. 3373.
[30] W. Fan et al., “DEPTS: Deep expansion learning for periodic time series
forecasting,” 2022, arXiv:2203.07681.
[31] G. Yu, J. Zou, X. Hu, A. I. Aviles-Rivero, J. Qin, and S. Wang, “Revitalizing
multivariate time series forecasting: Learnable decomposition with interseries dependencies and intra-series variations modeling,” in Proc. 41st
Int. Conf. Mach. Learn., 2024, Art. no. 2385.
[32] X. He, Y. Li, J. Tan, B. Wu, and F. Li, “OneShotSTL: One-shot seasonaltrend decomposition for online time series anomaly detection and forecasting,” 2023, arXiv:2304.01506.
[33] S. Lin, W. Lin, W. Wu, H. Chen, and J. Yang, “SparseTSF: Modeling
long-term time series forecasting with 1k parameters,” in Proc. 41st Int.
Conf. Mach. Learn., 2024, Art. no. 1216.
[34] A. Vaswani et al., “Attention is all you need,” in Proc. Int. Conf. Neural
Inf. Process. Syst., 2017, pp. 5998–6008.
[35] S. Li et al., “Enhancing the locality and breaking the memory bottleneck
of transformer on time series forecasting,” in Proc. Int. Conf. Neural Inf.
Process. Syst., 2019, pp. 5244–5254.

[36] B. Lim, S. Ö. Arık, N. Loeff, and T. Pfister, “Temporal fusion transformers
for interpretable multi-horizon time series forecasting,” Int. J. Forecasting,
vol. 37, no. 4, pp. 1748–1764, 2021.
[37] S. Liu et al., “Pyraformer: Low-complexity pyramidal attention for longrange time series modeling and forecasting,” in Proc. Int. Conf. Learn.
Representations, 2022.
[38] Y. Liu, H. Wu, J. Wang, and M. Long, “Non-stationary transformers:
Exploring the stationarity in time series forecasting,” in Proc. Int. Conf.
Neural Inf. Process. Syst., 2022, pp. 9881–9893.
[39] T. Zhou, Z. Ma, Q. Wen, X. Wang, L. Sun, and R. Jin, “FEDformer:
Frequency enhanced decomposed transformer for long-term series forecasting,” in Proc. Int. Conf. Mach. Learn., 2022, pp. 27268–27286.
[40] Y. Nie, N. H. Nguyen, P. Sinthong, and J. Kalagnanam, “A time series is
worth 64 words: Long-term forecasting with transformers,” in Proc. Int.
Conf. Learn. Representations, 2023.
[41] Y. Zhang and J. Yan, “Crossformer: Transformer utilizing cross-dimension
dependency for multivariate time series forecasting,” in Proc. 11th Int.
Conf. Learn. Representations, 2023.
[42] S. Lin, W. Lin, W. Wu, S. Wang, and Y. Wang, “PETformer: Longterm time series forecasting via placeholder-enhanced transformer,” IEEE
Trans. Emerg. Topics Comput. Intell., vol. 9, no. 2, pp. 1189–1201,
Apr. 2025.
[43] A. Dosovitskiy et al., “An image is worth 16 x 16 words: Transformers for
image recognition at scale,” 2020, arXiv: 2010.11929.
[44] K. He, X. Chen, S. Xie, Y. Li, P. Dollár, and R. Girshick, “Masked
autoencoders are scalable vision learners,” in Proc. IEEE/CVF Conf.
Comput. Vis. Pattern Recognit., 2022, pp. 16000–16009.
[45] Z. Liu et al., “Swin transformer: Hierarchical vision transformer using
shifted windows,” in Proc. IEEE/CVF Int. Conf. Comput. Vis., 2021,
pp. 10012–10022.
[46] M. Liu et al., “SCINet: Time series modeling and forecasting with sample
convolution and interaction,” in Proc. Int. Conf. Neural Inf. Process. Syst.,
2022, pp. 5816–5828.
[47] H. Wang, J. Peng, F. Huang, J. Wang, J. Chen, and Y. Xiao, “MICN: Multiscale local and global context modeling for long-term series forecasting,”
in Proc. 11th Int. Conf. Learn. Representations, 2022.
[48] D. Luo and X. Wang, “ModernTCN: A modern pure convolution structure
for general time series analysis,” in Proc. 12th Int. Conf. Learn. Representations, 2024.
[49] C. Challu, K. G. Olivares, B. N. Oreshkin, F. G. Ramirez, M. M. Canseco,
and A. Dubrawski, “NHiTS: Neural hierarchical interpolation for time
series forecasting,” in Proc. AAAI Conf. Artif. Intell., 2023, pp. 6989–6997.
[50] Q. Huang et al., “HDMixer: Hierarchical dependency with extendable
patch for multivariate time series forecasting,” in Proc. AAAI Conf. Artif.
Intell., 2024, pp. 12608–12616.
[51] S. Wang et al., “TimeMixer: Decomposable multiscale mixing for time
series forecasting,” in Proc. 12th Int. Conf. Learn. Representations, 2024.
[52] L. Han, X.-Y. Chen, H.-J. Ye, and D.-C. Zhan, “SOFTS: Efficient multivariate time series forecasting with series-core fusion,” in Proc. 38th Conf.
Neural Inf. Process. Syst., 2024, Art. no. 2046.
[53] T. Zhou et al., “One fits all: Power general time series analysis by pretrained
LM,” in Proc. Int. Conf. Neural Inf. Process. Syst., 2023, pp. 43322–43355.
[54] C. Chang, W.-Y. Wang, W.-C. Peng, and T.-F. Chen, “LLM4TS: Aligning
pre-trained LLMs as data-efficient time-series forecasters,” ACM Trans.
Intell. Syst. Technol., vol. 16, 2025, Art. no. 60.
[55] M. Jin et al., “Time-LLM: Time series forecasting by reprogramming large
language models,” in Proc. 12th Int. Conf. Learn. Representations, 2024.
[56] H. Xue and F. D. Salim, “PromptCast: A new prompt-based learning
paradigm for time series forecasting,” IEEE Trans. Knowl. Data Eng.,
vol. 36, no. 11, pp. 6851–6864, Nov. 2024.
[57] Y. Liu, H. Zhang, C. Li, X. Huang, J. Wang, and M. Long, “Timer:
Generative pre-trained transformers are large time series models,” in Proc.
41st Int. Conf. Mach. Learn., 2024, Art. no. 1313.
[58] Y. Liu, G. Qin, X. Huang, J. Wang, and M. Long, “AutoTimes: Autoregressive time series forecasters via large language models,” in Proc. 38th
Annu. Conf. Neural Inf. Process. Syst., 2024, Art. no. 3882.
[59] M. Jin et al., “Position: What can large language models tell us about time
series analysis,” in Proc. 41st Int. Conf. Mach. Learn., 2024, Art. no. 895.
[60] G. Woo, C. Liu, A. Kumar, C. Xiong, S. Savarese, and D. Sahoo,
“Unified training of universal time series forecasting transformers,”
2024, arXiv:2402.02592.
[61] Y. Jia, Y. Lin, X. Hao, Y. Lin, S. Guo, and H. Wan, “WITRAN: Water-wave
information transmission and recurrent acceleration network for longrange time series forecasting,” in Proc. Int. Conf. Neural Inf. Process.
Syst., 2024, Art. no. 544.

Authorized licensed use limited to: Beijing Jiaotong University. Downloaded on April 28,2026 at 04:57:04 UTC from IEEE Xplore. Restrictions apply.

LIN et al.: SPARSETSF: LIGHTWEIGHT AND ROBUST TIME SERIES FORECASTING VIA SPARSE MODELING

[62] S. Bergsma, T. Zeyl, and L. Guo, “SutraNets: Sub-series autoregressive
networks for long-sequence, probabilistic forecasting,” in Proc. Int. Conf.
Neural Inf. Process. Syst., 2023, pp. 30518–30533.
[63] K. Yi et al., “FourierGNN: Rethinking multivariate time series forecasting
from a pure graph perspective,” in Proc. Int. Conf. Neural Inf. Process.
Syst., 2024, Art. no. 3050.
[64] A. Zeng, M. Chen, L. Zhang, and Q. Xu, “Are transformers effective
for time series forecasting?,” in Proc. AAAI Conf. Artif. Intell., 2023,
pp. 11121–11128.
[65] T. Zhang et al., “Less is more: Fast multivariate time series forecasting
with light sampling-oriented MLP structures,” 2022, arXiv:2207.01186.
[66] W. Das, Abhi, A. Leach, S. Mathur, R. Sen, and R. Yu, “Long-term forecasting with tide: Time-series dense encoder,” 2023, arXiv:2304.08424.
[67] V. Ekambaram, A. Jati, N. Nguyen, P. Sinthong, and J. Kalagnanam,
“TSMixer: Lightweight MLP-mixer model for multivariate time series
forecasting,” in Proc. 29th ACM SIGKDD Conf. Knowl. Discov. Data
Mining, 2023, pp. 459–469.
[68] Q. Huang, Z. Zhou, K. Yang, Z. Yi, X. Wang, and Y. Wang, “TimeBase:
The power of minimalism in efficient long-term time series forecasting,”
in Proc. 42nd Int. Conf. Mach. Learn., 2025.
[69] L. Han, H.-J. Ye, and D.-C. Zhan, “The capacity and robustness trade-off:
Revisiting the channel independent strategy for multivariate time series
forecasting,” IEEE Trans. Knowl. Data Eng., vol. 36, no. 11, pp. 7129–
7142, Nov. 2024.
[70] V. Nair and G. E. Hinton, “Rectified linear units improve restricted Boltzmann machines,” in Proc. 27th Int. Conf. Mach. Learn., 2010, pp. 807–814.
[71] Z. Li, S. Qi, Y. Li, and Z. Xu, “Revisiting long-term time series forecasting:
An investigation on linear mapping,” 2023, arXiv:2305.10721.
[72] T. Kim, J. Kim, Y. Tae, C. Park, J.-H. Choi, and J. Choo, “Reversible
instance normalization for accurate time-series forecasting against distribution shift,” in Proc. Int. Conf. Learn. Representations, 2021.
[73] Y. Liu, C. Li, J. Wang, and M. Long, “Koopa: Learning non-stationary
time series dynamics with koopman predictors,” in Proc. Int. Conf. Neural
Inf. Process. Syst., 2023, Art. no. 538.
[74] R. Tibshirani, “Regression shrinkage and selection via the lasso,” J. Roy.
Statist. Soc. Ser. Statist. Methodol., vol. 58, no. 1, pp. 267–288, 1996.
[75] T. Zhou et al., “FiLM: Frequency improved legendre memory model for
long-term time series forecasting,” in Proc. Int. Conf. Neural Inf. Process.
Syst., 2022, pp. 12677–12690.
[76] X. Qiu et al., “TFB: Towards comprehensive and fair benchmarking of
time series forecasting methods,” 2024, arXiv:2403.20150.
[77] A. Paszke et al., “PyTorch: An imperative style, high-performance deep
learning library,” in Proc. Int. Conf. Neural Inf. Process. Syst., 2019,
Art. no. 721.
[78] D. P. Kingma and J. Ba, “Adam: A method for stochastic optimization,”
2014, arXiv:1412.6980.
[79] S. Lin, H. Chen, H. Wu, C. Qiu, and W. Lin, “Temporal query network
for efficient multivariate time series forecasting,” in Proc. 42nd Int. Conf.
Mach. Learn., 2025.

Shengsheng Lin received the bachelor’s degree from
the South China University of Technology, in 2022.
He is currently working toward the doctoral degree
in computer technology with the School of Computer
Science and Engineering, South China University of
Technology, Guangdong, China. His research interests include machine learning and time series forecasting. He has published more than five research
papers as the first author and serves as a reviewer
for high-impact journals and conferences such as the
IEEE Transactions on Knowledge and Data Engineering, Artificial Intelligence Journal, KDD, ICML, and NeurIPS.

183

Weiwei Lin (Senior Member, IEEE) received the BS
and MS degrees from Nanchang University, in 2001
and 2004, respectively, and the PhD degree in computer application from the South China University of
Technology, in 2007. He has been a visiting scholar
with Clemson University from 2016 to 2017. Currently, he is a professor with the School of Computer
Science and Engineering, South China University of
Technology. His research interests include distributed
systems, cloud computing, and AI application technologies. He has published more than 200 papers
in refereed journals and conference proceedings. He has been a reviewer for
many international journals, including ICML, IEEE Transactions on Parallel
and Distributed Systems, IEEE Transactions on Services Computing, IEEE
Transactions on Cloud Computing, IEEE Transactions on Computers, IEEE
Transactions on Cybernetics, etc. He is a distinguished member of CCF.
Wentai Wu (Member, IEEE) received the bachelor’s
and master’s degrees from the South China University
of Technology, in 2015 and 2018, respectively, and the
PhD degree in computer science from the University
of Warwick, United Kingdom, in 2022, sponsored by
CSC. He is currently an associate professor with the
Department of Computer Science, College of Information Science and Technology, Jinan University. His
research interests mainly include distributed systems,
edge intelligence, sustainable computing, and collaborative machine learning. He has published more
than 20 research papers and serves as reviewer for high-impact journals and
conferences, such as IEEE Transactions on Parallel and Distributed Systems,
IEEE Transactions on Mobile Computing, IEEE Transactions on Big Data,
ICML, and NeurIPS. He was listed among the top 2% scientists in distributed
computing subfield in 2023 per composite citation indicator.

Haojun Chen received the BS degree in automation
from the South China University of Technology, in
2023. He is currently working toward the MS degree
with the School of Computer Science and Engineering, South China University of Technology, China.
His research interests include cloud computing,
machine learning, and time series forecasting.

C. L. Philip Chen (Life Fellow, IEEE) received the
MS degree in electrical engineering from the University of Michigan at Ann Arbor, Ann Arbor, MI, USA,
in 1985, and the PhD degree in electrical engineering
from Purdue University, West Lafayette, IN, USA,
in 1988. He was a tenured professor with the Department head and an associate dean of two different
universities in the U.S. for 23 years. He is currently
the head with the School of Computer Science and
Engineering, South China University of Technology,
Guangzhou, Guangdong, China. His current research
interests include systems, cybernetics, and computational intelligence. He is a
fellow of AAAS, IAPR, CAA, and HKIE. He received the 2016 Outstanding
Electrical and Computer Engineers Award from his alma mater, Purdue University. He has been the editor-in-chief of IEEE Transaction on Systems, Man,
and Cybernetics: Systems since 2014 and an associate editor of several IEEE
Transactions. He was the chair of TC 9.1 Economic and Business Systems of
the International Federation of Automatic Control from 2015 to 2017 and also
a program evaluator of the Accreditation Board of Engineering and Technology
Education of the U.S. for computer engineering, electrical engineering, and
software engineering programs. He was the IEEE SMC Society president from
2012 to 2013 and a vice president of the Chinese Association of Automation
(CAA).

Authorized licensed use limited to: Beijing Jiaotong University. Downloaded on April 28,2026 at 04:57:04 UTC from IEEE Xplore. Restrictions apply.


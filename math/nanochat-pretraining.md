# Pretraining nanochat: from likelihood to a complete training run

This note explains the complete execution of [scripts/base_train.py](../nanochat/scripts/base_train.py#L1): what is optimized, how the model and training budget are chosen, how a global batch produces an update, how hyperparameters evolve, and what evaluation and checkpoints mean.

The source snapshot is nanochat commit `92d63d4e8bb4df75c3b71618f31ddde2378b2bcd`. Exact line links refer to that checkout. Mathematical identities, approximate scaling arguments, empirical settings, and implementation discrepancies are identified separately.

You can treat the [architecture note](nanochat-model-architecture.md) as defining the differentiable model and the [tokenization/data note](nanochat-tokenization-and-data.md) as defining its input distribution. This note connects them into a training process. The deeper matrix-optimizer derivation and communication implementation will belong in the optimization/distributed-training companion; precision kernels and hardware analysis will belong in the efficiency companion. Neither companion is required to follow the execution here.

## Contents

1. [What pretraining accomplishes](#1-what-pretraining-accomplishes)
2. [Notation and the execution map](#2-notation-and-the-execution-map)
3. [The exact loss and its gradient](#3-the-exact-loss-and-its-gradient)
4. [Constructing a model from depth](#4-constructing-a-model-from-depth)
5. [Three parameter counts and the FLOP estimate](#5-three-parameter-counts-and-the-flop-estimate)
6. [Choosing how long to train](#6-choosing-how-long-to-train)
7. [Global batches and gradient accumulation](#7-global-batches-and-gradient-accumulation)
8. [Automatic batch and hyperparameter scaling](#8-automatic-batch-and-hyperparameter-scaling)
9. [Which parameters use which optimizer settings](#9-which-parameters-use-which-optimizer-settings)
10. [Learning-rate, momentum, and decay schedules](#10-learning-rate-momentum-and-decay-schedules)
11. [One complete iteration in code order](#11-one-complete-iteration-in-code-order)
12. [Where precision and compilation enter](#12-where-precision-and-compilation-enter)
13. [Evaluation during pretraining](#13-evaluation-during-pretraining)
14. [Interpreting the logs and timing](#14-interpreting-the-logs-and-timing)
15. [Checkpoints and resuming](#15-checkpoints-and-resuming)
16. [A worked reference-scale configuration](#16-a-worked-reference-scale-configuration)
17. [Comparison with your training loop](#17-comparison-with-your-training-loop)
18. [Small checks for understanding](#18-small-checks-for-understanding)
19. [Source discrepancies and verification limits](#19-source-discrepancies-and-verification-limits)

## 1. What pretraining accomplishes

Pretraining starts from newly initialized model weights and adjusts them to predict the next token in a large text corpus. For a token sequence beginning with BOS,

$$
p_\theta(s_1,\ldots,s_T\mid\operatorname{BOS})
=\prod_{t=0}^{T-1}
p_\theta(s_{t+1}\mid\operatorname{BOS},s_1,\ldots,s_t).
$$

This factorization is the probability chain rule with a causal model supplying the conditional distributions. In practice, context is bounded by sequence construction and the model's attention pattern.

The supervision comes from the corpus itself: the observed next token is the label. There is no separate human answer attached to every prediction. This is why next-token training is called self-supervised even though it uses an ordinary supervised classification loss internally.

Predicting diverse text can reward representations of syntax, facts, programs, mathematical relationships, and other recurring structure. The objective remains predictive likelihood. It does not explicitly reward factual truth, obedience to user requests, or conversational helpfulness. A pretrained base model is consequently a text continuation model; later conversational training changes its behavior.

### 1.1 The training distribution is already a choice

The relevant distribution is the text retained by curation, tokenization, document packing, and cropping. The model does not train on an abstract uniform sample of all human knowledge. For this checkout, the input pipeline uses the selected local ClimbMix shards described in the previous note.

The meaning of “from scratch” concerns the model's initial weights. It does not require collecting a raw web corpus yourself or consuming the entire available corpus.

## 2. Notation and the execution map

| Symbol | Meaning | Code name |
| --- | --- | --- |
| $L$ | Transformer depth | `args.depth` |
| $d$ | Hidden width | `model.config.n_embd` |
| $d_h$ | Attention head dimension | `args.head_dim` |
| $V$ | Actual tokenizer vocabulary size | `vocab_size` |
| $T$ | Input positions per row | `args.max_seq_len` |
| $b$ | Rows per local microbatch | `args.device_batch_size` |
| $W$ | Number of distributed ranks | `ddp_world_size` |
| $A$ | Accumulation microsteps per iteration | `grad_accum_steps` |
| $G$ | Global target tokens per iteration | `total_batch_size` |
| $s$ | Completed iterations at loop entry | `step` |
| $K$ | Planned total iterations | `num_iterations` |
| $N_s$ | Parameters used for the scaling heuristic | `num_scaling_params` |
| $N_{mat}$ | Parameters counted in token-stream matrix products | `num_matmul_params()` |
| $P$ | Total learned parameters | `num_params` |
| $D_*$ | Token horizon used by automatic scaling | `target_tokens` |
| $D_{run}=KG$ | Planned processed target positions | `total_tokens` |
| $\Phi$ | Estimated forward-plus-backward FLOPs per token | `num_flops_per_token` |

Here capital $D$ refers to training tokens, not the head dimension called $D$ in the architecture note. We use $d_h$ for the latter. A “step” normally means one global update attempt, containing $A$ forward/backward microsteps. An epoch is a separate traversal of the data source and is not the same unit.

```mermaid
flowchart TD
    setup[Initialize runtime, seeds, and process group]
    setup --> tokenizer[Load tokenizer and byte-count artifact]
    tokenizer --> model[Construct and initialize model; optionally load checkpoint]
    model --> precision[Optional FP8 conversion, then compile]
    precision --> budget[Compute counts, horizon proxy, batch, and base hyperparameters]
    budget --> optimizer[Build optimizer groups; optionally restore their state]
    optimizer --> data[Build loaders and fetch first batch]
    data --> schedules[Resolve iteration count and schedules]
    schedules --> top[Loop entry: step s means s iterations completed]
    top --> evaluate[Scheduled evaluation, sampling, and checkpointing]
    evaluate --> end{Is s equal to K?}
    end -- Yes --> finish[Finish without another update]
    end -- No --> backward[A local forward/backward microsteps]
    backward --> update[Set schedules; synchronize gradients and update parameters]
    update --> clear[Clear gradients, log, increment s]
    clear --> top
```

The model is replicated across ranks. The optimizer handles distributed gradient reduction and state ownership; the script does not wrap GPT in `DistributedDataParallel`, despite variable names and comments using “DDP.”

## 3. The exact loss and its gradient

**Code:** [GPT's output/loss path](../nanochat/nanochat/gpt.py#L510) and [the microstep forward/backward](../nanochat/scripts/base_train.py#L510).

### 3.1 Inputs, targets, and teacher forcing

For each packed row $r_0,\ldots,r_T$, the loader creates

$$
X_{j,t}=r_t,\qquad Y_{j,t}=r_{t+1},
\qquad j=0,\ldots,b-1,\quad t=0,\ldots,T-1.
$$

For example:

```text
packed row: [BOS, A, B, C]
inputs:     [BOS, A, B]
targets:    [A,   B, C]
```

The model reads the true prefix when predicting each target. This is teacher forcing: training does not sample an intermediate token and substitute it into the next input position.

All $T$ positions can be computed in one forward pass. Although the full input array exists in memory, position $t$ cannot use later input positions because the model's attention is causal; its other mixing operations also respect causality. Parallel computation does not change the conditional prediction task.

### 3.2 Cross-entropy is negative conditional log likelihood

Let $z_{j,t,v}$ be the raw linear-classifier output. Nanochat first computes

$$
\bar z_{j,t,v}=15\tanh(z_{j,t,v}/15),\qquad
p_{j,t,v}=\frac{e^{\bar z_{j,t,v}}}{\sum_{u=0}^{V-1}e^{\bar z_{j,t,u}}}.
$$

Its per-position loss is

$$
\ell_{j,t}=-\ln p_{j,t,Y_{j,t}}
=\log\sum_{v=0}^{V-1}e^{\bar z_{j,t,v}}
-\bar z_{j,t,Y_{j,t}}.
$$

`F.cross_entropy` takes logits directly and combines log-softmax with target selection. Do not apply a separate softmax before passing logits to that function.

The ordinary pretraining microbatch loss is

$$
\mathcal L_{micro}=\frac1{bT}\sum_{j,t}\ell_{j,t}.
$$

The code flattens logits to $(bT,V)$ and targets to $(bT,)$. In the normal pretraining loader, all positions have real targets: there is no assistant-only mask or padding. A BOS that appears as a target inside a packed row contributes to this loss. The loss function supports `ignore_index=-1`, but this loader does not normally insert it. With ignored positions, mean cross-entropy divides by the number of nonignored targets instead.

There is no MoE balancing term in nanochat's base objective. Weight decay is applied by the optimizer rather than added explicitly to this loss.

### 3.3 Why this objective learns the data distribution

For a context $c$, let $q(v\mid c)$ denote the conditional target distribution induced by the training data. Then

$$
\mathbb E_{v\sim q(\cdot\mid c)}[-\ln p_\theta(v\mid c)]
=H(q(\cdot\mid c))+
\operatorname{KL}\!\left(q(\cdot\mid c)\,\|\,p_\theta(\cdot\mid c)\right).
$$

The entropy term is independent of the model. Minimizing expected cross-entropy therefore minimizes this conditional KL divergence, subject to model capacity, available data, and optimization. This identity does not say that finite training finds the optimum, or that matching a text distribution guarantees truthfulness.

### 3.4 The signal passed into backpropagation

For one microbatch,

$$
\frac{\partial\mathcal L_{micro}}{\partial\bar z_{j,t,v}}
=\frac{p_{j,t,v}-\mathbf1[v=Y_{j,t}]}{bT}.
$$

After the softcap chain rule,

$$
\frac{\partial\mathcal L_{micro}}{\partial z_{j,t,v}}
=\frac{p_{j,t,v}-\mathbf1[v=Y_{j,t}]}{bT}
\operatorname{sech}^2(z_{j,t,v}/15).
$$

The correct class has a negative logit derivative unless its probability is already one; other classes have positive derivatives. Gradient descent acts through the entire network to improve the target's relative score. Saturated raw logits receive an attenuated derivative through the softcap.

Autograd then applies the chain rule through unembedding, normalization, the transformer blocks, and embedding lookups. For example, with flattened hidden vectors $U$ and raw-logit derivatives $\Delta$, the unembedding derivative is $\Delta^\top U$ on the valid vocabulary rows. Embedding-table derivatives sum the lookup contributions wherever an ID appears. The integer IDs and tokenizer merge rules are not differentiated.

### 3.5 Units and a starting-loss check

Natural logarithms give nats per target token. For an exactly uniform output distribution,

$$
\mathcal L=\ln V.
$$

At $V=32768$, this is approximately 10.3972 nats/token. The small initial readout weights make a near-uniform distribution a useful starting-scale check, rather than an exact prediction of the first measured batch.

Token-level perplexity is $e^{\mathcal L}$. Equivalently, $e^{-\mathcal L}$ is the geometric mean of target probabilities; it is not their arithmetic mean or prediction accuracy. Comparisons across tokenizers need the byte normalization discussed later.

## 4. Constructing a model from depth

**Code:** [training arguments](../nanochat/scripts/base_train.py#L50), [build_model_meta](../nanochat/scripts/base_train.py#L129), [allocation and initialization](../nanochat/scripts/base_train.py#L146), and [compute_init](../nanochat/nanochat/common.py#L174).

### 4.1 The depth dial

With aspect ratio $a$ and requested head dimension $d_h$, the builder sets

$$
d=d_h\left\lceil\frac{La}{d_h}\right\rceil,\qquad
H=d/d_h,\qquad H_{kv}=H.
$$

Defaults are $a=64$, $d_h=128$, and $T=2048$. Width is rounded upward so every head has exactly the requested dimension. For instance:

| Depth | Width | Heads | Head dimension |
| ---: | ---: | ---: | ---: |
| 12 | 768 | 6 | 128 |
| 20 | 1280 | 10 | 128 |
| 24 | 1536 | 12 | 128 |

The training CLI defaults to depth 20, whereas `GPTConfig()` by itself defaults to 12. Training also passes explicit optimizer arguments that can differ from `GPT.setup_optimizer`'s function defaults. Follow the call path rather than treating every default in the repository as the same configuration.

Because width grows approximately linearly with depth, the dominant block matrices grow roughly as $Ld^2\propto L^3$. The embedding and value-table terms scale differently. Doubling depth is consequently much more than doubling this model's parameter count.

### 4.2 Why build on a meta device?

The sequence is:

```python
model = build_model_meta(args.depth)
model.to_empty(device=device)
model.init_weights()
```

The meta model establishes shapes and module structure without allocating ordinary parameter storage. `to_empty` allocates storage on the destination device; it does not supply meaningful values. `init_weights` initializes all parameters and the deterministic rotary buffers. This separates structural construction from data allocation and avoids an unnecessary full initialized CPU copy.

On resume, saved weights replace the initialized values. Initialization still creates buffers that are not present in the parameter checkpoint.

### 4.3 Seeds and rank consistency

`compute_init` sets the PyTorch global seed to 42 and the CUDA seed when applicable. Each rank starts with the same initialization. A CUDA distributed launch establishes an NCCL process group and selects the local GPU. Same weights and different local data are the intended data-parallel setup.

The code does not enable fully deterministic algorithms. Matching a seed is therefore one ingredient of reproducibility, not a guarantee of bitwise equality across devices, kernels, or versions.

## 5. Three parameter counts and the FLOP estimate

**Code:** [GPT.num_scaling_params](../nanochat/nanochat/gpt.py#L390), [get_scaling_params](../nanochat/scripts/base_train.py#L263), [GPT.num_matmul_params](../nanochat/nanochat/gpt.py#L341), and [GPT.estimate_flops](../nanochat/nanochat/gpt.py#L319).

### 5.1 Total capacity versus the count used for scaling

$P$ counts every learned parameter: the main embedding, unembedding, transformer matrices, value tables, and scalars.

The horizon heuristic instead uses

$$
N_s=P_{transformer\ matrices}+P_{lm\ head}.
$$

It excludes the main token table, value-embedding tables, and global scalars/smear parameters. The transformer category includes the small per-head value-gate matrices. Nanochat's [scaling experiments](../nanochat/dev/LOG.md#L502) explain its empirical preference for this count.

For a d12 model with the reference vocabulary:

$$
P=286,261,730,\qquad N_s=110,100,912.
$$

A run labelled “12 tokens per parameter” uses $12N_s$, not $12P$. Much of the difference comes from the large value tables, which add capacity through lookups with comparatively little token-stream matrix multiplication.

### 5.2 The matrix-product count

Before FP8 conversion, `num_matmul_params` sums weights of the custom `Linear` modules. That includes the global smear gate but excludes embedding lookups. For this architecture,

$$
N_{mat}=N_s+24
$$

in the ordinary unconverted model. The additional 24 are the smear gate's input weights; its scalar and backout coefficient are not matrix products. Counts must be rechecked when changing architecture or module types.

### 5.3 Where the factor six comes from

For a matrix product $Y=XW^\top$, let $X$ have $n$ rows, input width $d_{in}$, and output width $d_{out}$. Its approximate costs are:

- Forward: $2nd_{in}d_{out}$ FLOPs.
- Input derivative: another $2nd_{in}d_{out}$.
- Weight derivative: another $2nd_{in}d_{out}$.

Thus a trainable matrix contributes approximately six FLOPs per weight per token for forward plus backward. This is an accounting approximation that omits elementwise operations and some boundary cases, not an elapsed-time formula.

The script's per-token estimate is

$$
\Phi_{code}=6N_{mat}+
12Hd_h\sum_{\ell=0}^{L-1}\min(T,w_\ell),
$$

where a negative window is treated as full $T$. The second term estimates query/key and probability/value products, including their backward passes. It does not materialize an attention matrix merely to count it.

This attention term approximates the causal/window geometry; it does not exactly count the triangular ramp or the inclusive current-token boundary. The estimate also omits optimizer work, communication, tokenization, and evaluation. Floating-point precision and kernel efficiency affect time even when nominal matrix dimensions are unchanged.

For a token horizon $D_{run}$,

$$
C_{train,estimate}=\Phi_{code}D_{run}.
$$

**Snapshot-specific FP8 qualification:** conversion changes the `Linear` subclass, and this checkout's count does not include the converted modules. Section 19 verifies the resulting undercount. The formulas and numerical examples using the ordinary model's count should not be confused with the value printed by that broken counting path after conversion.

## 6. Choosing how long to train

**Code:** [target token calculation](../nanochat/scripts/base_train.py#L269) and [horizon precedence](../nanochat/scripts/base_train.py#L338).

### 6.1 Compute-optimal does not mean train indefinitely

With a fixed compute budget, model size and exposure compete. Increasing the model size leaves fewer tokens affordable; using a very small model allows more exposure but can limit capacity.

A common empirical fit, used in compute-optimal scaling research such as [Chinchilla](https://arxiv.org/abs/2203.15556), is

$$
\mathcal L(N,D)\approx\mathcal L_\infty+aN^{-\alpha}+bD^{-\beta}.
$$

Under the simplifying cost model $C=\kappa ND$, substitute $D=C/(\kappa N)$ and differentiate. At a stationary point,

$$
\alpha aN^{-\alpha}=\beta bD^{-\beta}.
$$

Consequently,

$$
N_{opt}\propto C^{\beta/(\alpha+\beta)},\qquad
D_{opt}\propto C^{\alpha/(\alpha+\beta)}.
$$

If the exponents are equal, both grow as $\sqrt C$, giving a constant optimal $D/N$ within that fitted family. The exponents, coefficients, and parameter-count convention are empirical. This calculation does not establish a universal token/parameter constant for all architectures or corpora.

### 6.2 Nanochat's practical horizon rule

For the usual positive ratio setting $\rho$, the script sets

$$
D_*=\lfloor\rho N_s\rfloor.
$$

The CLI default is $\rho=12$; the current reference speedrun overrides it to 8. The latter is chosen for reaching the benchmark capability quickly, not as a universal optimum for base-model loss.

`D_*` counts runtime target positions, not unique documents, downloaded gigabytes, or an obligation to read all ClimbMix. Repeated corpus traversals can still count toward it.

### 6.3 Three ways to set the iteration count

After determining $G$ and $\Phi$, the script chooses the first enabled option:

| Priority | Argument | Iteration count |
| --- | --- | --- |
| 1 | `--num-iterations=K` | Explicit positive $K$ |
| 2 | `--target-flops=C` | $K=\operatorname{round}(C/(\Phi G))$ |
| 3 | `--target-param-data-ratio=rho` | $K=\lfloor D_*/G\rfloor$ |

The actual planned token count is always

$$
D_{run}=KG.
$$

The ratio route rounds down to a whole global batch; the FLOP route rounds the number of iterations and may overshoot or undershoot the requested estimated budget. At least one full iteration is needed for an actual training update.

An explicit iteration or FLOP horizon overrides $K$, but it does not cause the earlier automatic batch/decay heuristics to be recomputed from the resulting $D_{run}$. Those heuristics still use their ratio-based horizon proxy. Keep the proxy and actual budget distinct when changing flags.

### 6.4 Compute budget and wall-clock objective differ

For fixed $C$, two model/data combinations can have similar nominal arithmetic but different utilization, memory demands, and evaluation costs. A speedrun optimizes time to a particular capability threshold; an inference-focused deployment might favor a smaller model trained for longer. These objectives need not select the same $(N,D)$.

The automatic rules provide a tuned starting configuration. They do not remove the need to specify which resource and quality objective an experiment is optimizing.

## 7. Global batches and gradient accumulation

**Code:** [batch accounting](../nanochat/scripts/base_train.py#L406), [loss division](../nanochat/scripts/base_train.py#L513), and the optimizer's [AdamW](../nanochat/nanochat/optim.py#L274) / [Muon](../nanochat/nanochat/optim.py#L295) reductions.

### 7.1 Count tokens, not just rows

One rank's microbatch contains $bT$ prediction targets. One simultaneous forward/backward across $W$ ranks contains $WbT$. With $A$ accumulation rounds,

$$
\boxed{G=WAbT},\qquad
A=\frac{G}{WbT}.
$$

The script asserts divisibility. It cannot simulate a fractional microstep.

For $G=1,048,576$, $W=8$, $b=16$, and $T=2048$:

$$
bT=32768,\qquad WbT=262144,\qquad A=4.
$$

Every rank processes four local microbatches, followed by one optimizer invocation. There are $WAb=512$ rows in the resulting global batch.

Keeping $G,b,T$ fixed while changing $W$ changes $A$. One GPU would use $A=32$ for this example, four GPUs $A=8$, and sixteen GPUs $A=2$. More GPUs distribute the global batch rather than requiring it to grow.

### 7.2 Why divide each loss by accumulation steps?

Let $\mathcal L_{r,a}$ be the mean loss on microbatch $a$ of rank $r$, each containing $bT$ targets. The desired global objective is

$$
\mathcal L_{global}=\frac1{WA}\sum_{r=0}^{W-1}\sum_{a=0}^{A-1}
\mathcal L_{r,a}.
$$

At fixed parameters, linearity of differentiation gives

$$
\nabla_\theta\mathcal L_{global}
=\frac1W\sum_r\left[
\sum_a\nabla_\theta\left(\mathcal L_{r,a}/A\right)
\right].
$$

This is exactly the intended composition:

1. `model(x, y)` returns the mean over that local microbatch's $bT$ targets.
2. Divide by $A$ before each backward.
3. Repeated backward calls add into each parameter's `.grad`.
4. The optimizer's reductions average these accumulated gradients across $W$ ranks.

There is no additional division by $W$ in the loss because the optimizer already uses average reductions. Adding another would change the gradient scale.

### 7.3 Accumulation is not several smaller optimizer steps

All $A$ microsteps use the same parameter values. Only after accumulation and cross-rank averaging does the optimizer update its moment state and weights.

Performing $A$ optimizer steps instead would evaluate later gradients at changed parameters and update momentum $A$ times. For an adaptive/nonlinear optimizer, averaging separate parameter updates also does not reproduce an update computed from the averaged gradient.

Nanochat postpones distributed gradient reduction until the optimizer step, reducing communication frequency. There is no DDP wrapper performing another gradient synchronization during each backward.

### 7.4 Conditions for equivalence

The identity above assumes the same examples, same parameters, equal target counts, and a batch-separable forward computation. Nanochat's dense model has no batch normalization or dropout, so concatenating a fixed set of independent rows versus accumulating them has the expected real-arithmetic equivalence. Floating-point summation order and kernel choices can still differ.

Changing rank count or local batch shape can also change which documents the loader packs together. Equal gradient formulas do not guarantee that two whole runs consume identical batches.

With variable numbers of valid labels, an unweighted mean of microbatch means is not generally the same as a mean over all valid tokens. This distinction becomes important for later masked conversation objectives.

Your model has an additional complication: MoE capacity decisions and its batch-level load-balancing term depend on the contents of a microbatch. Changing the batch partition can change routing/drop decisions and the auxiliary objective itself. The usual accumulation proof applies to additive losses on identical per-example computations; it does not automatically establish full-batch equivalence for that MoE implementation.

## 8. Automatic batch and hyperparameter scaling

**Code:** [reference and batch scaling](../nanochat/scripts/base_train.py#L271), [batch LR correction](../nanochat/scripts/base_train.py#L286), [weight-decay scaling](../nanochat/scripts/base_train.py#L296), and [width scaling](../nanochat/nanochat/gpt.py#L432).

### 8.1 Reference configuration

The script builds a shapes-only d12 reference using the current aspect ratio, vocabulary, head dimension, context length, and architecture. It uses

$$
G_{ref}=2^{19}=524288,\qquad
D_{ref}=\rho N_{s,ref}.
$$

The reference batch is an empirical anchor. Building a new reference model does not automatically retune that anchor after arbitrary architecture or data changes.

### 8.2 Batch size grows with the horizon

The predicted global token batch is

$$
G_{pred}=G_{ref}\left(\frac{D_*}{D_{ref}}\right)^{0.383},
$$

then rounded to a nearby power of two:

$$
G=2^{\operatorname{round}(\log_2G_{pred})}.
$$

An explicit `--total-batch-size` bypasses this prediction. The exponent comes from the batch-scaling work cited by the code: [Power Lines](https://arxiv.org/abs/2505.13738), which reports horizon-dependent empirical fits, including the 0.383 exponent for a tuned setting. It is not a theorem that every optimizer or corpus has that exponent.

With the same positive $\rho$ for target and reference, $D_*/D_{ref}$ is approximately $N_s/N_{s,ref}$. The ratio cancels; changing $\rho$ alone therefore does not generally increase this script's automatic batch. Rounding can introduce small integer effects.

For a fixed horizon, a larger $G$ means fewer updates. Smaller batches can offer more updates but more overhead; larger batches can improve parallel utilization while offering diminishing statistical benefit. Correlations between token gradients complicate simple independent-sample noise arguments. The best batch balances these effects for the measured objective.

### 8.3 Batch-dependent learning-rate correction

Define

$$
c_G=\sqrt{G/G_{ref}}.
$$

Before constructing optimizer groups, the script multiplies the embedding, unembedding, scalar, and matrix LR arguments by $c_G$. Doubling the global batch consequently multiplies those inputs by $\sqrt2$.

This square-root rule is a chosen transfer heuristic. In particular, the source explicitly says applying the same correction to Muon has not been studied carefully. It is not derived merely from the algebraic equivalence of gradient accumulation.

### 8.4 Width-dependent learning-rate correction

Inside `GPT.setup_optimizer`, another factor is

$$
c_d=(d/768)^{-1/2}.
$$

This factor applies to the unembedding, main embedding, and value-table learning rates. Other groups have their own settings, shown in section 9.

The code calls its reference transfer “muP style.” The broader idea of transferring tuned hyperparameters between widths has a formal literature, including [Tensor Programs V](https://arxiv.org/abs/2203.03466). Nanochat's particular collection of exponents and group settings should still be treated as its empirically tuned parameterization, not as a proof that it implements every requirement of formal maximal-update parameterization.

### 8.5 Deriving the base weight-decay scaling

For simple decoupled decay with a generic update direction $u_s$,

$$
\theta_{s+1}=(1-\eta\lambda)\theta_s-\eta u_s.
$$

Unrolling this recurrence gives a contribution from each earlier update multiplied by powers of $(1-\eta\lambda)$. For small $\eta\lambda$,

$$
(1-\eta\lambda)^n\approx e^{-n\eta\lambda},
\qquad \tau_{steps}\approx\frac1{\eta\lambda}.
$$

Relative to a horizon of $D/G$ steps, this timescale is

$$
\tau_{horizon}\approx\frac{G}{\eta\lambda D}.
$$

This is the timescale perspective behind the [AdamW scaling work](https://arxiv.org/abs/2405.13698) referenced by the source. Holding this normalized timescale constant between reference and target yields

$$
\frac{G}{\eta\lambda D}
=\frac{G_{ref}}{\eta_{ref}\lambda_{ref}D_{ref}},
$$

$$
\lambda=\lambda_{ref}
\frac{G}{G_{ref}}
\frac{\eta_{ref}}\eta
\frac{D_{ref}}D.
$$

Using $\eta/\eta_{ref}=\sqrt{G/G_{ref}}$ gives the code's base rule:

$$
\boxed{\lambda_0=\lambda_{arg}\sqrt{G/G_{ref}}
\frac{D_{ref}}{D_*}}.
$$

The script applies this to Muon's matrix decay. Its cautious, coordinate-dependent decay and varying LR/decay schedules do not exactly follow the simple scalar recurrence above. The transfer is therefore a motivation and heuristic, not an exact Muon invariance result. It also uses the ratio-based horizon proxy, not necessarily the explicitly overridden actual run horizon.

Here “horizon” means the planned exposure used in the heuristic. It is not automatically one traversal of the entire downloaded corpus, even though the source comments call the normalized quantity an epoch timescale.

## 9. Which parameters use which optimizer settings

**Code:** [GPT.setup_optimizer](../nanochat/nanochat/gpt.py#L419), [base_train's arguments to it](../nanochat/scripts/base_train.py#L308), and [MuonAdamW.step](../nanochat/nanochat/optim.py#L429).

The optimizer is one object combining different update rules for different parameter groups. “Use Muon for matrices” does not mean every two-dimensional parameter uses Muon: embedding tables and unembedding are also matrices but are assigned AdamW.

Let $\eta_e,\eta_u,\eta_m,\eta_s$ denote the CLI embedding, unembedding, matrix, and scalar LR arguments. The group base learning rates before the temporal schedule are:

| Parameters | Rule | Base LR $\eta_g^0$ | Adam betas | Decay setting |
| --- | --- | --- | --- | --- |
| Main token embedding | AdamW | $\eta_e c_Gc_d$ | $(0.8,0.995)$ | 0.001 |
| Unembedding | AdamW | $\eta_u c_Gc_d$ | $(0.8,0.96)$ | 0.01 |
| Value-embedding tables | AdamW | $0.5\eta_e c_Gc_d$ | $(0.8,0.995)$ | 0.01 |
| Residual coefficients $\alpha_\ell$ | AdamW | $0.01\eta_s c_G$ | $(0.8,0.95)$ | 0.05 |
| Input-injection coefficients $\beta_\ell$ | AdamW | $\eta_s c_G$ | $(0.96,0.95)$ | 0 |
| Smear gate, smear coefficient, backout coefficient | AdamW | 0.2 | $(0.8,0.95)$ | 0 |
| All parameters inside transformer blocks, including value gates | Muon | $\eta_m c_G$ | Separate Muon moment settings | Scheduled from $\lambda_0$ |

The script defaults are $\eta_e=0.3$, $\eta_u=0.008$, $\eta_m=0.02$, $\eta_s=0.5$, and $\lambda_{arg}=0.28$. AdamW's epsilon is $10^{-10}$ in these groups. The value-gate projection inside each attention block is a Muon parameter, whereas the independently indexed value table is AdamW.

Several consequences are easy to miss:

- `--weight-decay` sets the scaled/scheduled Muon coefficient; it does not replace the hardcoded AdamW decays in this table.
- The smear/backout group's base LR is fixed at 0.2, rather than scaled by $c_G$ or $c_d$.
- Every group's LR, including that fixed-base group, subsequently receives the common temporal multiplier.
- Equal-shaped Muon parameters are grouped for stacking and efficient updates. Inside [_compute_muon](../nanochat/nanochat/optim.py#L362), a matrix of shape $(n_{out},n_{in})$ receives an additional LR multiplier $\sqrt{\max(1,n_{out}/n_{in})}$. The MLP up projection $(4d,d)$ consequently gets a factor two at the kernel level.

### 9.1 What the update rule contributes

AdamW tracks first and second moments of gradients, constructs a coordinatewise adaptive direction, and applies decoupled weight decay. Muon starts from momentum and transforms matrix updates using approximate orthogonalization and further scale adaptation. Their full derivations are deferred, but both operate on the already accumulated, cross-rank-averaged gradients.

The distinction between loss and optimizer matters. Adding $\lambda\|\theta\|^2/2$ to the loss would insert $\lambda\theta$ into the gradients and their adaptive moments. Decoupled decay instead directly changes the parameter update; these are generally different with adaptive optimizers.

For Muon, the current final kernel step has the form

$$
\theta_i\leftarrow\theta_i-
\eta_{eff}\left[u_i+\lambda\theta_i\mathbf1(u_i\theta_i\ge0)\right],
$$

where $u$ is its transformed update. This sign-dependent shrinkage is the “cautious” part. The derivation and matrix geometry will be examined in the optimizer companion.

### 9.2 Why use different groups?

Embedding lookups, the output classifier, residual coefficients, and hidden projections play different roles and have different scale sensitivities. The partition allows their dynamics to be tuned separately. The table contains empirical settings; they do not follow uniquely from the next-token objective.

## 10. Learning-rate, momentum, and decay schedules

**Code:** [get_lr_multiplier](../nanochat/scripts/base_train.py#L360), [get_muon_momentum](../nanochat/scripts/base_train.py#L372), [get_weight_decay](../nanochat/scripts/base_train.py#L385), and [assignment before each update](../nanochat/scripts/base_train.py#L520).

### 10.1 Common LR schedule

Let $U$ be `warmup_steps`, $Q=\operatorname{round}(rK)$ the warmdown length, $r$ the configured warmdown fraction, and $f_{min}$ the final LR fraction. For ordinary nonoverlapping phases,

$$
f(s)=
\begin{cases}
(s+1)/U,&s<U,\\
1,&s\ge U\text{ and }s\le K-Q,\\
f_{min}+(1-f_{min})(K-s)/Q,&\text{otherwise}.
\end{cases}
$$

Each parameter group uses

$$
\eta_g(s)=\eta_g^0f(s),
$$

before any Muon matrix-shape multiplier. The defaults are $U=40$, $r=0.65$, and $f_{min}=0.05$.

Warmup gradually activates the chosen update scale while optimizer state forms. The constant segment maintains that scale, and warmdown permits smaller final adjustments. Their best lengths are empirical. Unlike your cosine LR decay, nanochat uses a linear warmdown for the learning rate.

For an illustrative $K=1000$, $U=40$, $Q=650$:

| Update index $s$ | LR multiplier |
| ---: | ---: |
| 0 | 0.025 |
| 39 | 1 |
| 350 | 1 |
| 675 | 0.525 |
| 999 | Approximately 0.05146 |
| 1000 | 0.05, if evaluated as a function |

The optimizer actually runs at indices $0,\ldots,K-1$. At $s=K$ the loop evaluates/saves and exits, so the exact endpoint value is normally not applied to an update.

### 10.2 Muon momentum schedule

Let $S_{down}=K-Q$. The source's actual branch order gives

$$
\mu(s)=
\begin{cases}
0.85+(0.97-0.85)s/400,&s<400,\\
0.97+(0.90-0.97)(s-S_{down})/Q,
&s\ge400\text{ and }s\ge S_{down},\\
0.97,&\text{otherwise}.
\end{cases}
$$

Momentum initially rises over a fixed 400 update indices; later it falls during LR warmdown. This schedule affects Muon groups, not the AdamW betas listed above.

The order of conditions matters for short runs. If $S_{down}<400$, the first branch takes precedence until index 400, potentially producing a jump when the warmdown branch first becomes active. A run shorter than 400 iterations never leaves the first momentum branch.

### 10.3 Muon weight-decay schedule

The coefficient is

$$
\lambda(s)=\frac{\lambda_0}{2}
\left[1+\cos\left(\pi s/K\right)\right].
$$

It starts at $\lambda_0$ and approaches zero. Again, the exactly zero endpoint at $K$ is not normally used for an update. AdamW decay coefficients remain fixed, although their per-update shrinkage changes because their LRs follow $f(s)$.

For ordinary decoupled decay, shrinkage is controlled by the product $\eta(s)\lambda(s)$, not by $\lambda(s)$ alone. Muon's cautious mask adds coordinate dependence. Calling this “cosine decay” without specifying the variable would be ambiguous: here the decay coefficient is cosine-scheduled, while the LR has linear warmdown.

### 10.4 Small runs can exercise different phases

The clean three-phase description assumes a suitable horizon. The actual functions prioritize warmup branches when phases overlap. An educational 20-step run with a 40-step warmup does not reach the intended plateau or warmdown. Such a run can verify execution but does not reproduce the schedule proportions of a full pretraining experiment.

## 11. One complete iteration in code order

**Code:** [loop entry](../nanochat/scripts/base_train.py#L416), [training microsteps](../nanochat/scripts/base_train.py#L510), [optimizer invocation](../nanochat/scripts/base_train.py#L528), and [step increment](../nanochat/scripts/base_train.py#L584).

Assume for now that FP16 loss scaling is unnecessary and no update is skipped. Let $\theta_s$ be the parameters after $s$ completed iterations.

### 11.1 Evaluation and saving happen before the next update

At loop entry, `step=s` and the current weights are $\theta_s$. The code checks whether this is the terminal pass, runs enabled evaluation/sampling/save operations when due, then exits if $s=K$.

Thus a checkpoint at step 250 contains weights after 250 iterations, before update index 250. The loop visits step values $0,\ldots,K$ but performs only $K$ training iterations. The last visit exists to evaluate and save the final model, not to perform an extra update.

### 11.2 Accumulate gradients

For each local microstep:

```python
loss = model(x, y)
train_loss = loss.detach()
(loss / grad_accum_steps).backward()
x, y, dataloader_state_dict = next(train_loader)
```

The forward builds a differentiable computation for the current microbatch. Backward adds its scaled parameter derivatives into `.grad`; it does not update the parameters. There is normally no need to retain this microbatch's graph after backward, so accumulation saves activation memory compared with processing the entire global batch at once.

`detach()` creates the logging view without retaining a derivative path. The actual loss used for backward remains differentiable. No call to `.item()` is needed inside the microstep loop, avoiding a forced CPU read of the loss each time.

The source then requests the next batch, potentially overlapping CPU-side preparation with queued GPU work. After the last microstep, `x,y` already hold the next iteration's first batch. Their prefetched state is relevant to checkpoint fidelity.

Gradients start absent on a new model and are cleared after each iteration. They intentionally remain accumulated between the $A$ microsteps.

### 11.3 Set update hyperparameters

After the backward calls, the script evaluates $f(s)$, $\mu(s)$, and $\lambda(s)$ and assigns them to the optimizer groups. Setting LR after backward is valid: LR affects the optimizer update, not the derivative of this explicitly defined loss.

Every group receives its scheduled LR. Only Muon groups receive the scheduled momentum and weight-decay coefficient.

### 11.4 Synchronize and update

`optimizer.step()` performs the following logical sequence:

1. Average accumulated gradients across ranks, using the group's communication path.
2. Update the owned optimizer moment state.
3. Transform the gradient into the AdamW or Muon update direction.
4. Update the corresponding parameters and apply that group's decay.
5. Gather updated parameter portions so ranks retain matching model replicas.

The actual implementation overlaps asynchronous operations, but its intended input is the global mean gradient derived in section 7. On one rank, communication is skipped.

The pretraining script does not call gradient-norm clipping. This differs from your explicit `clip_grad_norm_(..., 1.0)`.

### 11.5 Clear gradients, log, and advance

The script calls

```python
model.zero_grad(set_to_none=True)
```

This clears gradient buffers; it does not clear optimizer momentum or reset learned parameters. Then it records timing and loss statistics and increments `step`.

A simplified lifecycle is:

```text
theta_s, optimizer_state_s
    -> A scaled backward passes per rank
    -> global gradient average
    -> scheduled optimizer update
    -> theta_(s+1), optimizer_state_(s+1)
    -> gradients cleared
```

### 11.6 What one step does not mean

One step is not one token, one local forward, one complete dataset epoch, or one generated response. It is one iteration consuming $G$ target positions and ordinarily applying one optimizer update. In FP16, an overflow can cause the update to be skipped while the iteration still advances; token exposure and successful-update counts can then differ.

## 12. Where precision and compilation enter

**Code:** [FP8 setup](../nanochat/scripts/base_train.py#L167), [compile call](../nanochat/scripts/base_train.py#L246), [GradScaler creation](../nanochat/scripts/base_train.py#L323), and [scaled-step branch](../nanochat/scripts/base_train.py#L528).

### 12.1 Numerical representation versus the objective

The likelihood equations specify a real-arithmetic function. Practical execution approximates it using floating-point representations and particular kernels.

Nanochat manages its ordinary compute dtype explicitly through `COMPUTE_DTYPE`, rather than putting the whole forward under autocast as your training loop does. Its custom linear layers cast weights for the forward matrix products. Most linear master weights are fp32; embedding tables can be stored in the compute dtype. Logits and loss computation are promoted to fp32 along the model's readout path.

The detailed rounding, accumulation, and memory tradeoffs belong in the efficiency companion. For following the loop, keep separate the stored parameter dtype, forward activation/matmul dtype, and optimizer arithmetic.

### 12.2 Optional FP8 conversion

Before compilation, a CUDA run with `--fp8` replaces sufficiently large, aligned linear modules with `Float8Linear` while sharing their original weight parameters. The filter requires input/output dimensions divisible by 16 and at least 128. The small smear and value-gate projections do not pass that filter at the reference scale.

The local FP8 implementation supports the tensorwise recipe. The CLI also advertises `rowwise`, but [Float8LinearConfig.from_recipe_name](../nanochat/nanochat/fp8.py#L234) rejects it in this snapshot. A non-CUDA invocation ignores the FP8 flag.

Conversion preserves the parameter identities and matrix dimensions, while changing numerical execution. It does not imply that checkpoints contain only 8-bit parameters or that the model is eight times smaller. The FLOP-counting subclass issue described in section 19 is separate from that intended numerical substitution.

### 12.3 Why compile and retain the original model?

`torch.compile(model, dynamic=False)` targets the repeatedly used fixed training shapes. It can fuse operations and reduce interpreter/kernel-launch overhead; the first executions may incur compilation cost.

`orig_model` retains access to the underlying uncompiled module and shared weights. Saving through it avoids wrapping the state dictionary in compile-specific names. Variable-shape CORE evaluation and sampling also use it to avoid repeatedly compiling new input shapes.

`dynamic=False` is a shape specialization choice, not a guarantee that the whole program becomes one kernel or one graph. Mode and module changes can still lead to separate compiled paths.

### 12.4 FP16 loss scaling

When `COMPUTE_DTYPE` is fp16, the script uses a scale $S_{loss}$:

$$
\nabla_\theta(S_{loss}\mathcal L)
=S_{loss}\nabla_\theta\mathcal L.
$$

It scales each accumulated loss before backward, unscales gradients before the optimizer, and adapts the scale according to finite-gradient checks. The mathematical target gradient is unchanged when scaling/unscaling are exact and no overflow occurs.

All ranks must agree on skipping an update. The code combines nonfinite flags using an all-reduce maximum: if any rank reports overflow, every rank skips the optimizer step. This keeps the replicas and collective-call sequence consistent. It then updates the scale and clears gradients.

BF16/fp32 runs do not use this GradScaler branch. FP16 scaling details and numerical limits will receive a fuller treatment in the efficiency note.

## 13. Evaluation during pretraining

**Code:** [validation BPB](../nanochat/scripts/base_train.py#L420), [CORE evaluation](../nanochat/scripts/base_train.py#L438), [sampling](../nanochat/scripts/base_train.py#L455), and [disable_fp8](../nanochat/scripts/base_train.py#L196).

### 13.1 What the three checks measure

| Check | Question | Participating ranks |
| --- | --- | --- |
| Held-out BPB | How well does the model predict retained validation text per represented byte? | All ranks |
| CORE | How well does the base model perform on the selected benchmark tasks? | All ranks |
| Greedy prompt completions | What continuations does the model visibly produce? | Rank 0 |

These measure different properties. Lower validation loss is useful evidence of better predictive fit, while downstream task performance tests more specific capabilities. A few attractive completions are a qualitative diagnostic, not a reliable capability estimate.

Evaluation functions disable gradient recording where appropriate. Calling `.eval()` itself only changes module evaluation behavior; it is not the operation that turns off autograd. This architecture has no dropout, but explicit train/eval phases still organize execution and compilation.

### 13.2 Validation token accounting

Each validation microbatch spans $WbT$ target positions globally. The number of validation iterations is

$$
K_{eval}=\left\lfloor
\frac{E_{requested}}{WbT}
\right\rfloor.
$$

The loader is rebuilt for each evaluation, providing a consistent held-out starting region for a fixed setup. The actual target-position budget is $K_{eval}WbT$; the byte metric subsequently excludes special or ignored targets. Choosing a requested budget below one global validation batch yields zero iterations rather than a useful evaluation.

BPB is

$$
\operatorname{BPB}=
\frac{\text{sum ordinary-target negative log probabilities}}
{\ln2\times\text{sum represented target bytes}}.
$$

The evaluator sums numerator and denominator across ranks before dividing. This weights the result by represented bytes rather than averaging rank-local ratios. Raw training loss and validation BPB have different units and target inclusion rules; their numerical magnitudes should not be compared directly.

### 13.3 CORE and baseline centering

[evaluate_core](../nanochat/scripts/base_eval.py#L57) loads its evaluation bundle if needed, evaluates tasks, and aggregates their baseline-centered accuracies. For task accuracy $a_j$ and baseline probability $r_j$,

$$
c_j=\frac{a_j-r_j}{1-r_j},\qquad
\operatorname{CORE}=\frac1J\sum_{j=1}^Jc_j.
$$

The metadata expresses baseline percentages, so the source multiplies those values by 0.01 before using this formula. A score of zero corresponds to baseline accuracy, one to perfect accuracy; below-baseline performance can produce negative values. CORE is a mean across tasks, not an overall accuracy weighted by every task's example count.

The task set comes from the bundle's `core.yaml`, and `max_per_task` controls subsampling. The task evaluator distributes examples across ranks. Detailed prompt construction and scoring belong in a later evaluation note.

### 13.4 Scheduling and precision of evaluation

For an enabled positive interval:

- BPB runs at step 0, every configured interval, and the final step.
- CORE and sampling run at positive periodic steps and the final step, omitting the ordinary initial pass.
- Setting the interval to a nonpositive value disables that check, including its final check.

`disable_fp8` temporarily replaces FP8 linear modules with the ordinary custom `Linear` class sharing the same weights, then restores them. This evaluates through the normal compute-dtype path rather than FP8 matmuls. Comments call it BF16 evaluation because that is the intended hardware setup; the actual ordinary dtype still comes from the runtime configuration.

The loop tracks the minimum observed validation BPB for reporting. It does not automatically stop when validation worsens or save a separate best-validation checkpoint. Its planned termination is the iteration budget.

## 14. Interpreting the logs and timing

**Code:** [loss capture](../nanochat/scripts/base_train.py#L512), [EMA and throughput](../nanochat/scripts/base_train.py#L547), and [cumulative timing](../nanochat/scripts/base_train.py#L555).

### 14.1 The printed training loss is not the global-batch mean

`train_loss = loss.detach()` is overwritten on every microstep, before the division by $A$. After the loop, only the final microbatch's unscaled mean loss remains. Each rank holds its own value, and the printed/master-logged statistic comes from rank 0. The source does not average these loss values across microsteps or ranks for reporting.

For local losses $[3,2]$, the accumulation contributes the mean gradient of both microbatches, while the raw loss used for the log is 2. This distinction does not change the gradients; it changes the interpretation of a noisy monitoring statistic. Your loop explicitly accumulates the losses for reporting instead.

### 14.2 EMA and its initial bias correction

With $\beta=0.9$, the displayed statistic is based on

$$
m_s=\beta m_{s-1}+(1-\beta)\ell_s,\qquad m_{-1}=0,
$$

$$
\widehat m_s=\frac{m_s}{1-\beta^{s+1}}.
$$

Unrolling gives $(1-\beta)\sum_{i=0}^{s}\beta^{s-i}\ell_i$. For a constant loss, its weights sum to $1-\beta^{s+1}$, explaining the correction. This smooths the logged final-microbatch losses, not the model's weights and not an independently measured full-batch average.

### 14.3 Throughput and MFU

For timed iteration duration $\Delta t$,

$$
\operatorname{tokens/sec}=G/\Delta t,\qquad
\operatorname{estimated\ FLOPs/sec}=\Phi G/\Delta t,
$$

$$
\operatorname{MFU}(\%)=
100\frac{\Phi G/\Delta t}
{W\times\operatorname{GPU\ peak\ BF16\ FLOPs/sec}}.
$$

This is an estimate of model arithmetic throughput relative to a selected peak. It is not a direct measurement of active GPU cycles, memory-bandwidth utilization, or every operation performed by the program. It depends on the FLOP count being meaningful. An FP8 run also uses the BF16 peak as its reference denominator, so interpretation differs from an FP8-peak hardware utilization ratio.

For CPU/MPS, the script substitutes an infinite peak and MFU is not meaningful. CUDA timing explicitly synchronizes; its CPU/MPS synchronization helper is a no-op, so asynchronous device timing needs separate care when interpreting those paths.

### 14.4 What is included in training time?

The timed region includes local forward/backward work, requests for upcoming batches, optimizer communication/update, gradient clearing, and synchronization. Periodic BPB, CORE, samples, checkpoint writes, and setup lie outside that region.

The cumulative `total_training_time` additionally includes only iterations with `step > 10`: indices 0 through 10 are omitted. Its comments informally refer to the first ten steps, but the condition excludes eleven update indices. This attempts to suppress initialization/compilation transients; it is not end-to-end process wall time.

The logged `step` uses the current update index before increment. At loop entry $s$, evaluation/checkpointing describe $\theta_s$; the subsequent training loss was evaluated using $\theta_s$, while the optimizer then creates $\theta_{s+1}$. The `flops_so_far` value was computed as $s\Phi G$ at loop entry. Read those labels consistently when reconstructing cumulative progress.

ETA extrapolates recent accumulated step timing and is approximate. The reported data-loader epoch and row-group state can also be ahead of consumed data because the loader buffers and prefetches.

## 15. Checkpoints and resuming

**Code:** [save condition and metadata](../nanochat/scripts/base_train.py#L476), [save_checkpoint](../nanochat/nanochat/checkpoint_manager.py#L41), [load_checkpoint](../nanochat/nanochat/checkpoint_manager.py#L60), [weight restoration](../nanochat/scripts/base_train.py#L153), and [optimizer restoration](../nanochat/scripts/base_train.py#L318).

### 15.1 What a resumable training state contains

A useful conceptual state is

$$
\mathcal S_s=
(\theta_s,\ \text{optimizer moments},\ s,\ \text{schedule configuration},
\ \text{data position},\ \text{numerical/RNG state},\ \text{logging state}).
$$

Parameters alone are sufficient to evaluate that trained model, but do not specify how its next training update should behave. Momentum and adaptive moments summarize earlier gradients; resetting them changes continuation even when weights are identical.

### 15.2 Files written by nanochat

The default location is `NANOCHAT_BASE_DIR/base_checkpoints/d{depth}/`, with an optional model-tag override. For a checkpoint at step 5568:

```text
base_checkpoints/d24/
├── model_005568.pt
├── meta_005568.json
├── optim_005568_rank0.pt
├── optim_005568_rank1.pt
└── ...
```

Rank 0 saves the model state and JSON metadata. Every rank saves its own optimizer state because ownership is sharded. The metadata includes model configuration, user arguments, resolved batch/context settings, step, latest validation BPB, data-loader state, and running log statistics.

Saving `orig_model.state_dict()` avoids compile-wrapper prefixes. Deterministic rotary buffers marked nonpersistent are reconstructed rather than saved. The tokenizer and its byte-count table remain separate files and must stay compatible with the checkpoint's ID semantics.

The default `save_every=-1` saves at the end. Positive intervals additionally save at eligible intermediate steps, excluding the initial and immediate resume passes. Tracking `min_val_bpb` does not by itself cause a best-model save.

### 15.3 What happens on resume?

The script builds and initializes a model using the invocation's current configuration, loads saved weights, optionally converts/compiles it, constructs optimizer groups, then loads that rank's saved optimizer state. It restores the saved loop step and logging state and supplies saved source progress to the data loader.

Loading optimizer state also restores group metadata such as initial LRs and moment settings. Temporal schedules are then evaluated using the current script/arguments and restored step. A new command line does not automatically replay every original flag from `user_config` merely because that dictionary is present in the checkpoint.

The intended continuation therefore uses compatible model/tokenizer settings, optimizer group ordering, and distributed ownership. This script has no general optimizer-state resharding path for an arbitrary new world size.

### 15.4 Why continuation is not promised to be bitwise exact

The saved state does not include every component of the conceptual $\mathcal S_s$:

- The packing buffer, within-row-group cursor, and already prefetched next `x,y` are not serialized.
- PyTorch CPU/CUDA RNG states are not stored in these checkpoint calls.
- FP16 GradScaler state is not stored.
- Recomputed runtime configuration and changed kernels can alter execution.

The model has no dropout and the data ordering is largely deterministic, so some omitted randomness is less active here than in other systems. The data-loader omission still makes its progress restoration approximate. “Resume training” and “reproduce the exact uninterrupted trajectory” are different guarantees.

## 16. A worked reference-scale configuration

**Code:** [the reference pretraining invocation](../nanochat/runs/speedrun.sh#L67) and the builder/scaling functions linked above.

Consider the architecture and token horizon selected by the current speedrun:

$$
L=24,\quad d=1536,\quad H=H_{kv}=12,\quad d_h=128,
$$

$$
V=32768,\quad T=2048,\quad W=8,\quad b=16,\quad\rho=8.
$$

The real speedrun enables FP8. First calculate the ordinary model's parameter and nominal FLOP accounting; then distinguish the snapshot's post-conversion counter.

### 16.1 Parameters, batch, and tokens

| Quantity | Value |
| --- | ---: |
| Total parameters $P$ | 1,384,122,122 |
| Scaling parameters $N_s$ | 729,810,624 |
| Unconverted matrix-product count $N_{mat}$ | 729,810,648 |
| d12 reference scaling parameters $N_{s,ref}$ | 110,100,912 |
| Reference horizon at the same $\rho$ | 880,807,296 tokens |
| Target horizon $D_*=8N_s$ | 5,838,484,992 tokens |
| Rounded automatic global batch $G$ | 1,048,576 tokens |
| Iterations $K=\lfloor D_*/G\rfloor$ | 5,568 |
| Planned processed tokens $D_{run}=KG$ | 5,838,471,168 |
| Local microbatch targets $bT$ | 32,768 |
| Global targets per accumulation round $WbT$ | 262,144 |
| Accumulation rounds $A$ | 4 |

The rounding loses only 13,824 target positions relative to $D_*$ in this example. The “8 tokens per parameter” label refers to $N_s$, whereas dividing by all 1.384B parameters gives a different ratio. Neither ratio is the corpus's total available token count.

### 16.2 Hyperparameters

Here

$$
c_G=\sqrt2,\qquad c_d=1/\sqrt2,
$$

so their product is one for the three width-scaled AdamW groups. Before the temporal multiplier:

| Group | Base LR |
| --- | ---: |
| Main embedding | 0.3 |
| Unembedding | 0.008 |
| Value tables | 0.15 |
| Residual coefficients | Approximately 0.007071 |
| Input-injection coefficients | Approximately 0.707107 |
| Smear/backout group | 0.2 |
| Muon matrix groups before shape multiplier | Approximately 0.028284 |

The base Muon decay is

$$
\lambda_0=0.28\sqrt2\,
\frac{110,100,912}{729,810,624}
\approx0.0597384.
$$

At the first update all group LRs receive the warmup multiplier $1/40$. The warmdown length is $\operatorname{round}(0.65\cdot5568)=3619$, starting from the plateau boundary at index 1949.

These are different settings for different parameter roles. A single printed scalar such as `lrm` is the common multiplier, not the complete effective LR of every parameter.

### 16.3 FLOPs and the FP8 counting discrepancy

Before conversion, the source's nominal count is

$$
\Phi_{ordinary}=4,775,225,616\quad\text{FLOPs/token}.
$$

For the planned token count this gives

$$
C_{ordinary,estimate}\approx2.7880\times10^{19}\quad\text{FLOPs}.
$$

These numbers describe the nominal estimator, not a measured hardware trace or an elapsed-time prediction.

With the exact source's FP8 filter applied, 145 linear modules are converted, and the counter subsequently sees only 1752 remaining custom-`Linear` weights. It reports

$$
\Phi_{after\ conversion}=396,372,240,
$$

even though the large converted matrix products still execute. The drop is a counting inconsistency, not an approximately twelve-fold elimination of model arithmetic. Parameter-based token horizons remain based on the structural parameter sums, but reported FLOPs/MFU and a `--target-flops`-derived horizon are affected.

### 16.4 What the numerical example does and does not specify

It specifies a model, token budget, global batch, accumulation count, and scheduled update scales. It does not predict final quality, actual seconds per step, or the exact number of distinct documents retained by packing.

The reference script's selected shards supply the data; its roughly 17 GB download estimate is not a conversion rule from gigabytes to training tokens. Segmentation, document lengths, cropping, and repeated traversal affect exposure. Hardware and kernels determine how quickly the configured arithmetic is performed.

## 17. Comparison with your training loop

**Code:** your [train.py](../transformer-from-scratch/train.py#L19), particularly [accumulation](../transformer-from-scratch/train.py#L76), [clipping](../transformer-from-scratch/train.py#L98), and [update](../transformer-from-scratch/train.py#L107).

| Aspect | Your implementation | Nanochat pretraining |
| --- | --- | --- |
| Objective | CE plus MoE balancing | Next-token CE |
| Model/data selection | Explicit small configuration and Tiny Shakespeare | Depth-derived configuration and prepared text shards |
| Step naming | Loop variable called `epoch`, but each iteration is a sampled-batch update | Separate update counter and data-loader traversal epoch |
| Accumulation | Local microbatches, loss divided by accumulation count | Same principle plus optimizer-side mean reduction across ranks |
| Reporting loss | Accumulated microbatch loss | Rank 0's last microbatch loss, then EMA |
| Optimizer | AdamW for the model | AdamW and Muon parameter groups |
| LR schedule | Linear warmup then cosine LR decay | Linear warmup, plateau, linear warmdown |
| Other schedules | Fixed optimizer settings in the inspected loop | Muon momentum and cosine coefficient decay |
| Clipping | Global norm clipped to 1 | No gradient clipping call |
| Numerical path | Autocast context | Explicit dtype policy; optional FP8; FP16 GradScaler branch |
| Evaluation | Final train/validation CE estimate | Periodic held-out BPB, CORE, and completions |
| Saving | Model state only | Model, metadata, and per-rank optimizer state |

You already implemented the central backward/accumulation/update pattern. The new conceptual work is accounting for the complete global objective, determining budgets and scale transfers, understanding heterogeneous update rules, and distinguishing measurement/checkpoint behavior from informal labels.

## 18. Small checks for understanding

These are optional mathematical or scratch-code exercises, not commands to launch a production run.

1. **Loss and causality.** For `[BOS, A, B, C]`, write each input prefix and target. Explain why parallel teacher forcing does not expose target B to the position predicting B.
2. **Cross-entropy derivative.** Starting from log-sum-exp, derive $p-e_y$ and then include the tanh-softcap derivative and mean-loss denominator.
3. **Accumulation denominator.** With $W=4$, $b=8$, $T=1024$, and $G=262144$, derive $A=8$. Identify every averaging factor and where it occurs in the source.
4. **Hardware change.** Halve $b$ while keeping $G,W,T$ fixed. What happens to $A$, the ideal gradient, and activation memory? Which assumptions are needed for equivalence?
5. **Different meanings of loss.** For microbatch losses `[4, 3, 2, 1]`, identify the mean objective represented by accumulation and the raw statistic that nanochat logs on that rank.
6. **Parameter convention.** For d12, calculate the token horizon using $12N_s$ and using $12P$. Explain which one the script actually uses.
7. **Budget scaling.** With a constant optimal data/parameter ratio and approximate cost $C\propto ND$, derive the $\sqrt C$ scaling of both quantities. State the assumptions.
8. **Schedule boundaries.** Evaluate LR, momentum, and Muon decay at indices 0, 39, 399, 400, and $K-1$ for a long horizon. Then repeat with $K=1000$ and identify the momentum-branch transition.
9. **Decay timescale.** Derive $\lambda\propto\sqrt G/D$ under the stated LR scaling and normalized-timescale assumption. Explain why cautious Muon decay does not exactly satisfy the underlying scalar recurrence.
10. **Loop endpoint.** If $K=3$, list weights/checkpoints visible at loop-entry steps 0,1,2,3. Count forward/backward iterations and optimizer invocations.
11. **Metric units.** Explain why a training loss of 2.0 nats/token and validation BPB of 0.8 cannot be ranked by simply comparing 2.0 with 0.8.
12. **Resume state.** Identify what is lost if only model weights are restored. Then identify which state remains absent even when nanochat's optimizer and JSON metadata are also restored.
13. **Counter audit.** Inspect the inheritance of `Linear` and `Float8Linear`. Explain why sharing the same weights is insufficient to make `isinstance(module, Linear)` count both.

## 19. Source discrepancies and verification limits

### 19.1 The FP8 counting issue in this snapshot

The relevant chain is:

1. [base_train converts modules](../nanochat/scripts/base_train.py#L189) before counting FLOPs.
2. [Float8Linear](../nanochat/nanochat/fp8.py#L195) subclasses `nn.Linear`, rather than nanochat's custom `Linear`.
3. [num_matmul_params](../nanochat/nanochat/gpt.py#L348) includes only instances of the custom `Linear`.
4. [estimate_flops](../nanochat/nanochat/gpt.py#L338) uses that incomplete sum.
5. The result feeds both [FLOP-budget iterations](../nanochat/scripts/base_train.py#L346) and [MFU](../nanochat/scripts/base_train.py#L554).

This was reproduced by applying the source conversion to a meta-device model: no FP8 kernel or GPU training was needed to demonstrate the structural mismatch. Changing numerical precision should not remove the converted matrix dimensions from nominal arithmetic accounting. Neither repository was modified to correct it.

### 19.2 Other distinctions worth retaining

| Informal description | Executed behavior |
| --- | --- |
| “Tokens per parameter” | Uses transformer matrices plus unembedding, not all parameters |
| “Automatic horizon” | Iteration/FLOP flags take precedence for $K$; earlier heuristic proxies still exist |
| “All LRs scale with width and batch” | Group-specific formulas and a fixed smear/backout base LR apply |
| “Weight decay is the CLI value” | Muon is scaled/scheduled; AdamW has separate hardcoded coefficients |
| “The last LR is exactly its final fraction” | The exact endpoint lies at the terminal no-update pass |
| “Momentum warms down during the final fraction” | The first 400-index branch has precedence in short runs |
| “Training loss is the global mean” | The displayed EMA uses rank 0's final microbatch |
| “The script uses DDP” | A process group is used; synchronization lives in the custom optimizer |
| “Evaluation disables autograd through eval()” | No-gradient contexts/decorators do that; eval() changes module mode |
| “BF16 evaluation” | The ordinary compute dtype is used; it need not be BF16 on every runtime |
| “Total training time is wall time” | Evaluation/setup/save time and indices 0 through 10 are excluded |
| “The minimum BPB model is saved” | Minimum is tracked; saving follows the save/final-step condition |
| “A checkpoint restores the exact next batch” | Outstanding buffers and prefetched batches are omitted |

The trainer also prints an old warning that SDPA has no sliding-window support. The current [fallback implementation](../nanochat/nanochat/flash_attention.py#L99) does construct a sliding-window mask. Functional support and efficient kernel execution are different questions; the warning should not be read as the current mathematical mask definition.

### 19.3 What was checked for this note

Parameter totals, model dimensions, global-batch/iteration arithmetic, the worked reference configuration, and the FP8 subclass-count discrepancy were checked against source definitions using small or meta-device computations. The softcap/cross-entropy derivative was compared with autograd. A tiny instance of the actual GPT, using the source's SDPA fallback, verified accumulation and simulated rank averaging against a concatenated batch; the largest absolute parameter-gradient difference was approximately $3.35\times10^{-8}$. Source-defined LR, momentum, and decay functions were checked at phase boundaries and on a short overlapping-phase horizon. Local file targets and line anchors were also checked.

No production pretraining run, corpus download, GPU timing benchmark, or distributed communication job was launched. Local numerical checks used CPU PyTorch 2.6.0 rather than reproducing the repository's pinned CUDA environment. Simulated rank averaging checks the gradient algebra, not actual collective communication. No final quality or speed result is inferred from those checks.

The next companion can now focus on the optimizer itself: why a matrix polar-factor direction is useful, what the finite iteration computes, how the additional scale adaptation works, and how the distributed optimizer assigns and synchronizes state.

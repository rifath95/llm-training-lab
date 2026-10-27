# Pretraining nanochat: from likelihood to a complete training run

This note explains the complete execution of [scripts/base_train.py](../nanochat/scripts/base_train.py#L1): what is optimized, how the model and training budget are chosen, how a global batch produces an update, how hyperparameters evolve, and what evaluation and checkpoints mean.

Start with the training loop you already implemented: take some text, predict the next tokens, compute a loss, call backward, and update the weights. Nanochat keeps that basic structure. The additional work is deciding how much text to process, splitting an update across GPUs and microbatches, choosing settings for different parameter groups, and measuring/saving the resulting model.

Three questions organize the note:

1. **What are we minimizing?** Sections 3–4 connect the model to the prediction task.
2. **How much training should we do, and with what settings?** Sections 5–10 explain counts, budgets, batches, and hyperparameters.
3. **What actually happens during a run?** Sections 11–16 follow updates, numerical execution, evaluation, logs, and checkpoints.

The notation table below is a reference, not a prerequisite to memorize. Each calculation introduces the quantities it needs. In particular, **tokens per update**, **number of updates**, and **total training tokens** are separate quantities; many later formulas are just bookkeeping between them.

The source snapshot is nanochat commit `92d63d4e8bb4df75c3b71618f31ddde2378b2bcd`. Exact line links refer to that checkout. Mathematical identities, approximate scaling arguments, empirical settings, and implementation discrepancies are identified separately.

You can treat the [architecture note](01-nanochat-model-architecture.md) as defining the differentiable model and the [tokenization/data note](02-nanochat-tokenization-and-data.md) as defining its input distribution. This note connects them into a training process. The deeper matrix-optimizer derivation and communication implementation will belong in the optimization/distributed-training companion; precision kernels and hardware analysis will belong in the efficiency companion. Neither companion is required to follow the execution here.

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

Pretraining starts from newly initialized model weights and adjusts them to predict the next token in a large text corpus. Write $\theta$ for the collection of learned model parameters, and $p_\theta$ for the probabilities the model assigns. For a token sequence beginning with BOS,

$$
p_\theta(s_1,\ldots,s_T\mid\mathrm{BOS})
=\prod_{t=0}^{T-1}
p_\theta(s_{t+1}\mid\mathrm{BOS},s_1,\ldots,s_t).
$$

This factorization is the probability chain rule with a causal model supplying the conditional distributions. In practice, context is bounded by sequence construction and the model's attention pattern.

The supervision comes from the corpus itself: the observed next token is the label. There is no separate human answer attached to every prediction. This is why next-token training is called self-supervised even though it uses an ordinary supervised classification loss internally.

Predicting diverse text can reward representations of syntax, facts, programs, mathematical relationships, and other recurring structure. The objective remains predictive likelihood. It does not explicitly reward factual truth, obedience to user requests, or conversational helpfulness. A pretrained base model is consequently a text continuation model; later conversational training changes its behavior.

### 1.1 The training distribution is already a choice

The relevant distribution is the text retained by curation, tokenization, document packing, and cropping. The model does not train on an abstract uniform sample of all human knowledge. For this checkout, the input pipeline uses the selected local ClimbMix shards described in the previous note.

The meaning of “from scratch” concerns the model's initial weights. It does not require collecting a raw web corpus yourself or consuming the entire available corpus.

## 2. Notation and the execution map

First distinguish three units of work. A **microbatch** is the group of sequences processed in one forward/backward pass on one GPU. A **training step** combines one or more microbatches, possibly across several GPUs, before one optimizer update. A **training run** performs many such steps.

If one step processes $G$ tokens and the run performs $K$ steps, it processes $KG$ tokens in total. Section 7 explains how the microbatches add up to $G$.

| Symbol | Meaning | Code name |
| --- | --- | --- |
| $L$ | Transformer depth | `args.depth` |
| $d$ | Hidden width | `model.config.n_embd` |
| $d_h$ | Attention head dimension | `args.head_dim` |
| $V$ | Actual tokenizer vocabulary size | `vocab_size` |
| $T$ | Input positions per row | `args.max_seq_len` |
| $b$ | Sequences in one local microbatch | `args.device_batch_size` |
| $W$ | Number of training processes, usually one per GPU | `ddp_world_size` |
| $A$ | Local microbatches accumulated before an optimizer update | `grad_accum_steps` |
| $G$ | Tokens per optimizer step, across all processes and microbatches | `total_batch_size` |
| $s$ | Completed iterations at loop entry | `step` |
| $K$ | Number of training steps planned for the run | `num_iterations` |
| $N_s$ | Parameters used for the scaling heuristic | `num_scaling_params` |
| $N_{mat}$ | Parameters counted in token-stream matrix products | `num_matmul_params()` |
| $P$ | Total learned parameters | `num_params` |
| $D_*$ | Target training-token budget used by automatic scaling | `target_tokens` |
| $D_{run}=KG$ | Planned processed target positions | `total_tokens` |
| $\Phi$ | Estimated forward-plus-backward FLOPs per token | `num_flops_per_token` |

Here capital $D$ refers to training tokens, not the head dimension called $D$ in the architecture note. We use $d_h$ for the latter. A “step” normally means one global update attempt, containing $A$ forward/backward microsteps. An epoch is a separate traversal of the data source and is not the same unit.

The code calls each training process a **rank**. Its rank is an identifier such as 0, 1, or 2; $W$ counts how many processes there are. This distinction is explained with an example in section 7.

The diagram is an execution map to consult while reading; its setup stages will be explained individually below.

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

The training example is ordinary text shifted by one token: predict the token immediately after each input position. A row contains $T+1$ tokens so that it can supply $T$ input/target pairs. For each packed row $r_0,\ldots,r_T$, the loader creates

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

For each position, we want a scalar penalty that is large when the model assigns little probability to the observed next token. Negative log probability does exactly that: assigning probability 0.5 costs about 0.693 nats, whereas assigning 0.01 costs about 4.605 nats.

Let $z_{j,t,v}$ be the raw linear-classifier output for sequence $j$, position $t$, and candidate vocabulary token $v$. Nanochat first computes

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

Why does predicting the observed token encourage learning a distribution rather than just increasing arbitrary scores? Consider a fixed context $c$. Let $q(v\mid c)$ be the data's distribution of continuations and $p_\theta(v\mid c)$ the model's prediction. The expected penalty separates into a fixed part and a distribution-mismatch part:

$$
\mathbb E_{v\sim q(\cdot\mid c)}[-\ln p_\theta(v\mid c)]
=H(q(\cdot\mid c))+
\mathrm{KL}\!\left(q(\cdot\mid c)\,\|\,p_\theta(\cdot\mid c)\right).
$$

Here $H(q)$ measures the uncertainty of the data's next token, and KL divergence measures how much the model distribution disagrees with it. The entropy term is independent of the model. Minimizing expected cross-entropy therefore minimizes this conditional KL divergence, subject to model capacity, available data, and optimization. This identity does not say that finite training finds the optimum, or that matching a text distribution guarantees truthfulness.

### 3.4 The signal passed into backpropagation

The loss is useful because it gives a derivative for every candidate token's score. For one microbatch containing $b$ sequences of $T$ targets each,

$$
\frac{\partial\mathcal L_{micro}}{\partial\bar z_{j,t,v}}
=\frac{p_{j,t,v}-\mathbf1[v=Y_{j,t}]}{bT}.
$$

After the softcap chain rule,

$$
\frac{\partial\mathcal L_{micro}}{\partial z_{j,t,v}}
=\frac{p_{j,t,v}-\mathbf1[v=Y_{j,t}]}{bT}
\mathrm{sech}^2(z_{j,t,v}/15).
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

The CLI lets you choose model size mainly through depth. The builder then chooses a matching width and number of heads, rather than requiring you to tune all three separately. With depth $L$, aspect ratio $a$ (the approximate width/depth ratio), and requested head dimension $d_h$, it sets

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

Before training, we need both a description of the model's tensor shapes and actual storage for their numerical values. PyTorch's **meta device** provides just the first: tensors have shapes and dtypes but no stored elements. The sequence is:

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

Before deciding how long to train, we need a measure of model size. There are three counts because we are asking three different questions:

| Question | Count |
| --- | --- |
| How many learnable numbers does the model contain? | Total parameters $P$ |
| Which measure of model size does nanochat use to choose a token budget? | Scaling count $N_s$ |
| How many weights participate in the learned dense matrix products used to estimate compute? | Matrix-product count $N_{mat}$ |

These are bookkeeping conventions for different purposes. They do not describe three different models.

### 5.1 Actual parameter count versus the count used to choose a training budget

**The actual count, $P$, includes every learned parameter.** It answers the usual question “How many parameters does this model have?” Here,

$$
P=P_{token\ embeddings}+P_{value\ embeddings}
+P_{transformer\ blocks}+P_{output\ head}+P_{other}.
$$

**The scaling count, $N_s$, is the number inserted into an empirical training-budget rule.** Section 6 will explain that rule: choose a target of approximately $\rho N_s$ training tokens, where $\rho$ is a tuned tokens-per-parameter ratio. Nanochat uses

$$
N_s=P_{transformer\ matrices}+P_{lm\ head}.
$$

Thus $N_s$ leaves out the main token table, value-embedding tables, and global scalars/smear parameters. The source's `transformer_matrices` category includes all parameters inside the blocks, including the small per-head value-gate matrices.

Why might that be useful? An embedding lookup selects one row of a table; a dense projection uses every weight of its matrix for each token. Adding value-embedding tables can therefore add many parameters with comparatively little extra matrix multiplication per token. Counting these parameters identically need not give the best predictor of an appropriate training budget.

That is a motivation, not a proof of which count to use. Karpathy compared conventions experimentally and found that transformer-block parameters plus the output head gave cleaner scaling relationships in these [scaling experiments](../nanochat/dev/LOG.md#L502). **Every excluded parameter still participates in training.** Only its inclusion in the budget calculation changes. Also, total parameter count is not a precise mathematical measure of model capacity.

For a d12 model with the reference vocabulary:

$$
P=286,261,730,\qquad N_s=110,100,912.
$$

A run labelled “12 tokens per parameter” therefore targets about $12\times110\text{M}=1.32$ billion tokens. Using $12P$ would instead give about 3.44 billion tokens. Much of the count difference comes from the large value tables. A quoted token/parameter ratio is meaningful only after specifying which parameter count is in the denominator.

### 5.2 The matrix-product count

Next we want to estimate computation. Count the weights actually used in the learned dense projections, because each is repeatedly involved in multiply/add operations. Before FP8 conversion, `num_matmul_params` implements this by summing weights of the custom `Linear` modules. That includes the global smear gate but excludes embedding lookups. For this architecture,

$$
N_{mat}=N_s+24
$$

in the ordinary unconverted model. The additional 24 are the smear gate's input weights; its scalar and backout coefficient are not matrix products. Counts must be rechecked when changing architecture or module types.

### 5.3 Where the factor six comes from

The quantity we want is **total training arithmetic**. A common approximation for dense language models is

$$
\boxed{C\approx6ND},
$$

where $C$ is floating-point operations (FLOPs), $N$ is an appropriate dense-weight parameter count, and $D$ is processed training tokens. The factor six means approximately **six operations per matrix weight per token, including forward and backward**. Here is its origin.

Consider $Y=XM^\top$, where $X$ contains $n$ token vectors, the weight matrix $M$ has shape $(d_{out},d_{in})$, and $N_{matrix}=d_{in}d_{out}$. Given the output derivative $\Delta=\partial\mathcal L/\partial Y$:

| Operation | Matrix expression | Approximate FLOPs |
| --- | --- | --- |
| Forward | $Y=XM^\top$ | $2nN_{matrix}$ |
| Backward to inputs | $\partial\mathcal L/\partial X=\Delta M$ | $2nN_{matrix}$ |
| Backward to weights | $\partial\mathcal L/\partial M=\Delta^\top X$ | $2nN_{matrix}$ |
| Total | All three products | $6nN_{matrix}$ |

Each factor of two counts one multiplication and one addition per dot-product contribution; exact operation counts differ slightly at boundaries. Summing over all learned projections and all training tokens gives $6N_{mat}D$. This is the connection to $C\approx6ND$; it is an arithmetic estimate, not a formula for elapsed seconds.

Attention also performs products such as $QK^\top$ and $\mathrm{softmax}(QK^\top)V$. These multiply activations by activations, so their cost is missing from a count of learned projection weights. Nanochat adds an attention term to its **per-token** estimate:

$$
\Phi_{code}=6N_{mat}+
12Hd_h\sum_{\ell=0}^{L-1}\min(T,w_\ell),
$$

Here $\Phi_{code}$ means FLOPs per training token, $H$ is the number of query heads, $d_h$ is head dimension, and $w_\ell$ is layer $\ell$'s left attention window. A negative window is treated as full $T$, the sequence length. The second term estimates those attention products and their backward passes. Its factor 12 comes from two activation matrix products, each with a forward and two backward products at roughly two FLOPs per multiply/add contribution. The code does not materialize an attention matrix merely to count it.

This attention term approximates the causal/window geometry; it does not exactly count the triangular ramp or the inclusive current-token boundary. The estimate also omits optimizer work, communication, tokenization, and evaluation. Floating-point precision and kernel efficiency affect time even when nominal matrix dimensions are unchanged.

Finally multiply FLOPs/token by the number of processed tokens $D_{run}$:

$$
C_{train,estimate}=\Phi_{code}D_{run}.
$$

**Snapshot-specific FP8 qualification:** conversion changes the `Linear` subclass, and this checkout's count does not include the converted modules. Section 19 verifies the resulting undercount. The formulas and numerical examples using the ordinary model's count should not be confused with the value printed by that broken counting path after conversion.

## 6. Choosing how long to train

**Code:** [target token calculation](../nanochat/scripts/base_train.py#L269) and [horizon precedence](../nanochat/scripts/base_train.py#L338).

### 6.1 Compute-optimal does not mean train indefinitely

Suppose you have a fixed compute budget. Should you train a large model on relatively little text, or a smaller model on more text? That is the allocation problem behind compute-optimal scaling. Increasing the model size leaves fewer tokens affordable; using a very small model allows more exposure but can limit capacity.

To express this mathematically, let $N$ be a chosen model-size measure and $D$ the number of training tokens. A common empirical fit, used in compute-optimal scaling research such as [Chinchilla](https://arxiv.org/abs/2203.15556), is

$$
\mathcal L(N,D)\approx\mathcal L_\infty+aN^{-\alpha}+bD^{-\beta}.
$$

The two decreasing terms model improvements from more parameters and more data; $\mathcal L_\infty$ is a fitted limiting loss. The positive coefficients $a,b$ and exponents $\alpha,\beta$ come from experiments. They are unrelated to microbatch size $b$ or the model's residual coefficients.

Under the simplifying cost model $C=\kappa ND$, with constant $\kappa$ (approximately 6 in the dense approximation), substitute $D=C/(\kappa N)$:

$$
\mathcal L(N\mid C)\approx\mathcal L_\infty
+aN^{-\alpha}+b(\kappa N/C)^\beta.
$$

The parameter-dependent term improves as $N$ increases, but the data-dependent term worsens because fewer tokens remain affordable. Differentiate with respect to $N$ and multiply the stationary equation by $N$. At a stationary point,

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

Rather than fitting that loss surface for every run, nanochat uses a simple rule: **choose a tuned number of training tokens per scaling parameter**. Write that ratio as $\rho$ (“rho”). For scaling parameter count $N_s$, it sets

$$
D_*=\lfloor\rho N_s\rfloor.
$$

Here $D_*$ is the desired number of processed training tokens; the floor just makes it an integer. At $N_s\approx110$ million and $\rho=12$, the target is approximately 1.32 billion tokens.

**How is $\rho$ chosen?** Run several model sizes at a fixed compute budget, allocating each its affordable number of tokens. Fit or compare their validation losses, identify the best model/token allocation, and calculate $D/N_s$ there. Repeat across budgets. Karpathy's recorded experiments using this count gave ratios around 10–11 over the tested budgets. The CLI defaults to 12; these experiments motivate the range, but do not mathematically single out exactly 12. The current reference speedrun uses 8 to reach its benchmark goal quickly. These are empirical settings for particular objectives, not a limit beyond which more data stops helping.

`D_*` counts runtime target positions, not unique documents, downloaded gigabytes, or an obligation to read all ClimbMix. Repeated corpus traversals can still count toward it.

### 6.3 Three ways to set the iteration count

We now need to turn a total budget into a number of optimizer steps. Keep these three quantities separate:

| Quantity | Meaning | Units |
| --- | --- | --- |
| $G$ | Global batch: tokens processed for one optimizer step | Tokens/step |
| $K$ | Number of training steps planned for the run | Steps |
| $D_{run}=KG$ | Total processed tokens in that run | Tokens |

“Global” means counting every GPU and every accumulated microbatch. Section 7 explains how those pieces add up. For now, if $G=1{,}000{,}000$ tokens/step and $K=1000$ steps, the run processes one billion tokens.

The script supports three ways to choose $K$. It uses the first enabled option:

| Priority | Argument | Iteration count |
| --- | --- | --- |
| 1 | `--num-iterations=K` | Explicit positive $K$ |
| 2 | `--target-flops=C` | $K=\mathrm{round}(C/(\Phi G))$ |
| 3 | `--target-param-data-ratio=rho` | $K=\lfloor D_*/G\rfloor$ |

The FLOP-budget formula divides total available FLOPs $C$ by FLOPs per step $\Phi G$. The token-budget formula divides desired tokens $D_*$ by tokens per step $G$. Both are unit conversions from a budget into a step count.

The actual planned token count is always

$$
D_{run}=KG.
$$

The ratio route rounds down to a whole global batch; the FLOP route rounds the number of iterations and may overshoot or undershoot the requested estimated budget. At least one full iteration is needed for an actual training update.

For a concrete rounding example, a target of 10.5 million tokens with $G=1$ million gives $K=10$, hence $D_{run}=10$ million. The code cannot process half of its configured global batch as a final partial step.

An explicit iteration or FLOP budget overrides $K$, but it does not cause the earlier automatic batch/decay heuristics to be recomputed from the resulting $D_{run}$. Those heuristics still use the ratio-based estimate $D_*$. Thus $D_*$ is the budget used to suggest settings, while $D_{run}$ is what the resolved step count actually schedules.

### 6.4 Compute budget and wall-clock objective differ

For fixed $C$, two model/data combinations can have similar nominal arithmetic but different utilization, memory demands, and evaluation costs. A speedrun optimizes time to a particular capability threshold; an inference-focused deployment might favor a smaller model trained for longer. These objectives need not select the same $(N,D)$.

The automatic rules provide a tuned starting configuration. They do not remove the need to specify which resource and quality objective an experiment is optimizing.

## 7. Global batches and gradient accumulation

**Code:** [batch accounting](../nanochat/scripts/base_train.py#L406), [loss division](../nanochat/scripts/base_train.py#L513), and the optimizer's [AdamW](../nanochat/nanochat/optim.py#L274) / [Muon](../nanochat/nanochat/optim.py#L295) reductions.

### 7.1 Count tokens, not just rows

The question here is: **how do several small forward/backward passes add up to one global optimizer step?** We split the work because the whole batch may not fit on one GPU at once.

First define the process terminology. A **rank** is the identifier of one distributed training process, usually attached to one GPU. Four GPUs normally mean four processes with ranks 0, 1, 2, and 3. **$W=4$ is their count**, called the world size; $W$ is not itself a rank identifier. One-GPU training has rank 0 and $W=1$. Each rank holds a model copy and processes different local data.

Next define the batch sizes:

- $b$ is the number of sequences in **one local microbatch**.
- $T$ is the number of prediction targets per sequence, so that microbatch has $bT$ targets.
- $A$ is how many such microbatches **each rank** processes before the optimizer updates the weights.

One accumulation round is one local microbatch per rank, giving $WbT$ targets across all ranks. After $A$ rounds, the global batch is

$$
\boxed{G=WAbT},\qquad
A=\frac{G}{WbT}.
$$

For example, with $W=2$ processes, $b=2$ sequences per microbatch, $T=4$ targets per sequence, and $A=3$ rounds:

| Accumulation round | Rank 0 | Rank 1 | Targets across both ranks |
| --- | --- | --- | ---: |
| First | 2 sequences × 4 targets | 2 different sequences × 4 targets | 16 |
| Second | 2 more sequences × 4 targets | 2 more sequences × 4 targets | 16 |
| Third | 2 more sequences × 4 targets | 2 more sequences × 4 targets | 16 |
| Then update | Accumulated gradients are averaged across ranks | Same resulting model update | 48 total |

Thus $G=2\times3\times2\times4=48$ targets per optimizer step. All those forward/backward passes use the same weights; the update comes after them.

Given $G,W,b,T$, the script computes $A=G/(WbT)$ and asserts divisibility. It cannot simulate a fractional microstep.

For a configuration closer to the reference run, take $G=1,048,576$, $W=8$, $b=16$, and $T=2048$:

$$
bT=32768,\qquad WbT=262144,\qquad A=4.
$$

Every rank processes four local microbatches, followed by one optimizer invocation. There are $WAb=512$ rows in the resulting global batch.

Keeping $G,b,T$ fixed while changing $W$ changes $A$. One GPU would use $A=32$ for this example, four GPUs $A=8$, and sixteen GPUs $A=2$. More GPUs distribute the global batch rather than requiring it to grow.

### 7.2 Why divide each loss by accumulation steps?

We want the update to use the **mean gradient over all $G$ targets**, despite computing them in pieces. Distinguish sizes from indices:

| Symbol | Role | Small example: $b=2,A=3,W=2$ |
| --- | --- | --- |
| $b$ | Size of each local microbatch, in sequences | $b=2$ |
| $A$ | Number of local microbatches before an update | $A=3$ |
| $a$ | Index identifying one of those microbatches | $a=0,1,2$ |
| $W$ | Number of processes | $W=2$ |
| $r$ | Index identifying one process | $r=0,1$ |

Thus $\mathcal L_{1,2}$ means “the mean loss of the third microbatch on rank 1.” It already averages over that microbatch's $bT$ targets. Because every microbatch has the same number of targets, the desired global loss is the mean of the $WA$ microbatch means:

$$
\mathcal L_{global}=\frac1{WA}\sum_{r=0}^{W-1}\sum_{a=0}^{A-1}
\mathcal L_{r,a}.
$$

For the small example, that is

$$
\mathcal L_{global}=\frac16(
\mathcal L_{0,0}+\mathcal L_{0,1}+\mathcal L_{0,2}
+\mathcal L_{1,0}+\mathcal L_{1,1}+\mathcal L_{1,2}).
$$

Every target receives weight $1/(WAbT)=1/G$. At fixed parameters, linearity of differentiation gives

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

Suppose we have settings that work reasonably well for a small model, and now want to train a larger one. **How should we change the batch size, learning rates, and weight decay without retuning everything from the beginning?** That is the problem this section addresses.

These quantities are **hyperparameters**: choices controlling training, rather than weights learned by backward. The code starts from a reference configuration and applies scaling rules to suggest settings for the target run. Most of these rules are empirical; the decay rule also has an approximate mathematical motivation.

The order is:

```text
reference model and target model
    -> compare their token budgets
    -> choose target global batch G
    -> adjust learning-rate inputs for G
    -> adjust selected learning rates for width d
    -> adjust Muon's base weight-decay coefficient
```

This section chooses base settings. Section 9 assigns them to parameter groups; section 10 varies them over the course of training.

### 8.1 What is the reference, and what does “horizon” mean?

A **reference** is a baseline against which we express changes. The script builds a shapes-only depth-12 model using the current aspect ratio, vocabulary, head dimension, context length, and architecture. It does not train that model as part of setup; it only needs its parameter count $N_{s,ref}$.

The reference has global batch and token budget

$$
G_{ref}=2^{19}=524288,\qquad
D_{ref}=\rho N_{s,ref}.
$$

The target model has scaling parameter count $N_s$ and target budget $D_*=\rho N_s$. A subscript `ref` means “the reference value”; a ratio such as $D_*/D_{ref}$ tells us how many times larger the target budget is.

Here **training horizon** means the planned number of processed training tokens. It does not mean sequence length, model context, or one traversal of the dataset. We will use “token budget” below to keep that meaning explicit.

For the running arithmetic example, suppose the target has **four times the reference token budget and twice the reference width**. This is an illustrative choice of ratios, not a claim that every doubling of width produces exactly four times the budget.

The reference batch of 524,288 tokens is an empirical anchor. Building a new shapes-only reference does not automatically retune that anchor after arbitrary architecture or data changes.

### 8.2 Choose the global batch from the token-budget ratio

A larger batch lets you process more tokens before synchronizing/updating, which can improve hardware utilization. But at a fixed token budget it also means fewer optimizer updates. The best batch balances useful gradient information, number of updates, and execution cost.

Nanochat's rule is: **as the token budget grows, increase the global batch, but more slowly than the budget**. It predicts

$$
G_{pred}=G_{ref}\left(\frac{D_*}{D_{ref}}\right)^{0.383},
$$

then rounds to a nearby power of two:

$$
G=2^{\mathrm{round}(\log_2G_{pred})}.
$$

In our example,

$$
G_{pred}=G_{ref}\,4^{0.383}\approx1.70G_{ref},
$$

which rounds to $G=2G_{ref}=1{,}048{,}576$ tokens per step. Four times the token budget and twice the batch means approximately **twice as many optimizer steps**, since steps equal tokens divided by tokens/step.

Why 0.383? It is a fitted exponent from the batch-scaling work cited by the source, [Power Lines](https://arxiv.org/abs/2505.13738), which reports this exponent for a tuned setting. It is not derived from the loss or a universal theorem about all corpora and optimizers. An explicit `--total-batch-size` bypasses the prediction.

With the same positive $\rho$ for target and reference, $D_*/D_{ref}$ is approximately $N_s/N_{s,ref}$. The ratio cancels; changing $\rho$ alone therefore does not generally increase this script's automatic batch. Rounding can introduce small integer effects.

### 8.3 Adjust learning-rate inputs for the chosen batch

The learning rate $\eta$ controls the scale of an optimizer's parameter update. Averaging more examples usually reduces gradient noise, which can permit larger updates. That motivates adjusting the learning rate when the global batch changes, although it does not establish a unique adjustment formula.

Nanochat uses a square-root multiplier:

$$
c_G=\sqrt{G/G_{ref}}.
$$

Here $c_G$ is a dimensionless correction factor, not a new learning rate. Before constructing optimizer groups, the script multiplies the embedding, unembedding, scalar, and matrix LR arguments by $c_G$.

For our doubled global batch, $c_G=\sqrt2$. For example, a matrix LR argument of 0.02 becomes $0.02\sqrt2\approx0.0283$ before later scheduling and matrix-shape adjustments. The fixed smear/backout group has its own exception, shown in section 9.

This square-root rule is a chosen transfer heuristic. In particular, the source explicitly says applying the same correction to Muon has not been studied carefully. It is not derived merely from the algebraic equivalence of gradient accumulation.

### 8.4 Separately adjust selected learning rates for model width

Changing width also changes the scale relationships inside the model; a learning rate tuned at width 768 need not transfer unchanged to width 1536. Inside `GPT.setup_optimizer`, nanochat applies another empirical correction:

$$
c_d=(d/768)^{-1/2}.
$$

Here $d$ is hidden width and 768 is the default reference width. Wider models get a smaller multiplier. This factor applies to the unembedding, main embedding, and value-table learning rates. It is not applied to every parameter group.

For our twice-as-wide target, $d=1536$ and $c_d=1/\sqrt2$. The batch and width factors therefore cancel for these groups:

$$
c_Gc_d=\sqrt2\,\frac1{\sqrt2}=1.
$$

For example, the main embedding's LR input of 0.3 becomes $0.3c_Gc_d=0.3$. The matrix LR keeps the batch correction without this width correction. Section 9 lists the complete group-specific formulas.

The code calls its reference transfer “muP style.” The broader idea of transferring tuned hyperparameters between widths has a formal literature, including [Tensor Programs V](https://arxiv.org/abs/2203.03466). Nanochat's particular collection of exponents and group settings should still be treated as its empirically tuned parameterization, not as a proof that it implements every requirement of formal maximal-update parameterization.

### 8.5 Adjust decay to account for its cumulative effect over the run

Weight decay shrinks parameters during optimizer updates. Its effect depends on **how strongly we shrink at each step and how many steps we take**. Keeping the same coefficient while training for longer need not preserve the same total shrinkage.

Let $\eta$ be learning rate and $\lambda$ the decay coefficient. First ignore the gradient update and hold both constant. Ordinary decoupled decay then gives

$$
\theta_{s+1}=(1-\eta\lambda)\theta_s.
$$

After $K$ steps,

$$
\theta_K=(1-\eta\lambda)^K\theta_0
\approx e^{-\eta\lambda K}\theta_0,
$$

where the approximation assumes small $\eta\lambda$. Because $K=D/G$, the accumulated shrinkage is controlled by

$$
\boxed{\eta\lambda\frac DG}.
$$

This gives an approximate transfer principle: **preserve that quantity when moving from reference to target**. Using a generic token budget $D$ for the target,

$$
\eta\lambda\frac DG
=\eta_{ref}\lambda_{ref}\frac{D_{ref}}{G_{ref}}.
$$

Solve for the new decay coefficient:

$$
\lambda=\lambda_{ref}
\frac G{G_{ref}}
\frac{\eta_{ref}}\eta
\frac{D_{ref}}D.
$$

Insert the square-root batch LR rule, $\eta/\eta_{ref}=\sqrt{G/G_{ref}}$, and use nanochat's budget estimate $D=D_*$:

$$
\boxed{\lambda_0=\lambda_{arg}\sqrt{G/G_{ref}}
\frac{D_{ref}}{D_*}}.
$$

Here $\lambda_{arg}$ is the CLI's reference decay input (default 0.28), while $\lambda_0$ is the resulting base coefficient for the target's Muon groups. In our example,

$$
\lambda_0=\lambda_{arg}\frac{\sqrt2}{4}
\approx0.354\lambda_{arg}.
$$

Why smaller? There are about twice as many updates, and their base LR is $\sqrt2$ times larger. Reducing the decay coefficient compensates for both in this simplified accumulated-shrinkage calculation.

To connect this with the timescale language in the [AdamW scaling work](https://arxiv.org/abs/2405.13698) cited by the source, now restore a generic optimizer update direction $u_s$:

$$
\theta_{s+1}=(1-\eta\lambda)\theta_s-\eta u_s.
$$

Unrolling this recurrence multiplies the remaining contribution of each earlier update by powers of $(1-\eta\lambda)$. Its decay timescale, measured in optimizer steps, is

$$
(1-\eta\lambda)^n\approx e^{-n\eta\lambda},
\qquad \tau_{steps}\approx\frac1{\eta\lambda}.
$$

That is approximately how many steps it takes for a contribution to shrink by a factor $e^{-1}$. As a fraction of the run's $D/G$ steps,

$$
\tau_{horizon}\approx\frac{G}{\eta\lambda D}.
$$

Keeping this fraction constant is equivalent to keeping the accumulated-shrinkage quantity constant. The source calls it an epoch timescale, but the normalizing token budget here need not equal one traversal of the downloaded corpus.

The actual script applies the transfer to Muon's matrix decay. Muon's cautious, coordinate-dependent decay and varying LR/decay schedules do not exactly follow the simple recurrence above. The derivation motivates the setting; it is not an exact invariance theorem for the real optimizer. It also uses $D_*$, not necessarily the actual budget $D_{run}$ after an explicit iteration/FLOP override.

### 8.6 Put the adjustments together

For the illustrative four-times-budget, twice-width example:

| Setting | Change relative to reference | Reason for the rule |
| --- | --- | --- |
| Global token batch $G$ | Approximately $2\times$ after rounding | Empirical budget-to-batch fit |
| LR inputs | $\sqrt2\times$ | Chosen batch-size transfer |
| Embedding/value-table/output-head LR width factor | $1/\sqrt2$ | Chosen width transfer |
| Muon base decay | Approximately $0.354\times$ | Approximate preservation of cumulative shrinkage |

These are the base values we will now assign to parameter groups. A further schedule will change their effective values as training proceeds.

## 9. Which parameters use which optimizer settings

**Code:** [GPT.setup_optimizer](../nanochat/nanochat/gpt.py#L419), [base_train's arguments to it](../nanochat/scripts/base_train.py#L308), and [MuonAdamW.step](../nanochat/nanochat/optim.py#L429).

The optimizer is one object combining different update rules for different parameter groups. “Use Muon for matrices” does not mean every two-dimensional parameter uses Muon: embedding tables and unembedding are also matrices but are assigned AdamW.

You can think of a **parameter group** as a list of tensors sharing an update rule and its settings. Your [configure_optimizers](../transformer-from-scratch/model.py#L300) already uses groups for decay choices; nanochat also uses them to assign different algorithms and learning rates to different parts of the model.

To read the table, keep three stages separate. Start with a CLI learning-rate argument. Multiply by the batch and, where applicable, width corrections from section 8 to obtain a **group base LR**. Finally, section 10 multiplies that base LR by a function of training step. For example, the main embedding follows

$$
\underbrace{\eta_e}_{\text{CLI input}}
\ \longrightarrow\
\underbrace{\eta_e c_Gc_d}_{\text{group base LR}}
\ \longrightarrow\
\underbrace{\eta_e c_Gc_d f(s)}_{\text{LR at step }s}.
$$

Here $f(s)$ will be the common schedule multiplier. Let $\eta_e,\eta_u,\eta_m,\eta_s$ denote the CLI embedding, unembedding, matrix, and scalar LR arguments. The group base learning rates before that temporal schedule are:

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

The Adam betas specify how strongly its running averages retain earlier gradients and squared gradients; a beta closer to one gives a longer memory. Epsilon is a small denominator-stabilizing constant. The companion optimizer note will derive those recurrences. Here their role is to identify exactly which configuration each group receives. The symbols $\alpha_\ell,\beta_\ell$ in the parameter column are learned model coefficients, distinct from optimizer betas.

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

Here $u_i$ is coordinate $i$ of the transformed update, $\eta_{eff}$ is its effective LR including the shape multiplier, and the indicator is 1 when $u_i\theta_i\ge0$, otherwise 0. When the update points toward zero, decay adds shrinkage; when it points away from zero, this mask suppresses decay. This sign-dependent shrinkage is the “cautious” part. The derivation and matrix geometry will be examined in the optimizer companion.

### 9.2 Why use different groups?

Embedding lookups, the output classifier, residual coefficients, and hidden projections play different roles and have different scale sensitivities. The partition allows their dynamics to be tuned separately. The table contains empirical settings; they do not follow uniquely from the next-token objective.

## 10. Learning-rate, momentum, and decay schedules

**Code:** [get_lr_multiplier](../nanochat/scripts/base_train.py#L360), [get_muon_momentum](../nanochat/scripts/base_train.py#L372), [get_weight_decay](../nanochat/scripts/base_train.py#L385), and [assignment before each update](../nanochat/scripts/base_train.py#L520).

A base LR is a scale for the run, not a promise to take equally large updates throughout it. A **schedule** changes an optimizer setting according to training progress. Here progress is the step index $s=0,1,\ldots,K-1$, where $K$ is the planned number of training steps.

Nanochat schedules three distinct quantities: the LR of every group, Muon's momentum, and Muon's decay coefficient. They have different curves and should not be conflated.

### 10.1 Common LR schedule

In words, the LR **rises linearly during warmup, stays constant, then falls linearly during warmdown**. It is convenient to express that curve as a dimensionless multiplier $f(s)$ applied to every group's base LR.

Let $U$ be the number of warmup steps (`warmup_steps`), $r$ the fraction of the run reserved for warmdown, $Q=\mathrm{round}(rK)$ that warmdown length in steps, and $f_{min}$ the final multiplier. For ordinary nonoverlapping phases,

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

For example, a group with base LR 0.02 uses $0.02/40=0.0005$ on the first update. On the plateau it uses 0.02; near the end it approaches $0.02\times0.05=0.001$. The same multiplier gives a different numerical LR for a group whose base is 0.3.

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

Momentum retains information about previous gradients when constructing the current update. The coefficient $\mu$ controls that retention; larger values give more persistence. Nanochat starts Muon's coefficient at 0.85, raises it toward 0.97, then lowers it toward 0.90 late in training.

Let $S_{down}=K-Q$ be the step where warmdown begins. The source's actual branch order gives

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

Section 8 chose a base decay coefficient $\lambda_0$ to account for the overall scale of the run. The schedule now gradually reduces it from that base toward zero. The coefficient at step $s$ is

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

The previous sections chose the settings; this section follows one actual update using them. Keep one invariant in mind: weights remain at $\theta_s$ throughout all microbatch backward calls, and change to $\theta_{s+1}$ only at the optimizer step.

### 11.1 Evaluation and saving happen before the next update

At loop entry, `step=s` and the current weights are $\theta_s$. The code checks whether this is the terminal pass, runs enabled evaluation/sampling/save operations when due, then exits if $s=K$.

Thus a checkpoint at step 250 contains weights after 250 iterations, before update index 250. The loop visits step values $0,\ldots,K$ but performs only $K$ training iterations. The last visit exists to evaluate and save the final model, not to perform an extra update.

For $K=3$, the ordering is:

```text
step 0: inspect theta_0 -> train/update -> theta_1
step 1: inspect theta_1 -> train/update -> theta_2
step 2: inspect theta_2 -> train/update -> theta_3
step 3: final evaluation/save of theta_3 -> exit
```

“Inspect” means checking the evaluation/save conditions; it does not mean every check runs on every step.

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

The likelihood equations specify a real-arithmetic function. A GPU must approximate it using finite-bit numbers. Lower-precision formats can save memory and make matrix multiplication faster, but also change rounding and the range of representable values. A **kernel** is the concrete device operation that carries out such a computation.

Do not assume a single precision label applies to the whole training process: the stored weights, the numbers used in a forward matmul, and the optimizer's running state can use different formats. “Master weights” means the stored parameter values that the optimizer actually updates; a lower-precision copy or cast can be used for a matrix product.

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

Some gradients may be too small to represent usefully in FP16 and can round to zero. Loss scaling tries to keep backward values in a usable range: multiply the loss by a positive scale, do backward, then divide the resulting gradients by the same scale before the optimizer. When `COMPUTE_DTYPE` is fp16, the script uses a scale $S_{loss}$:

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

Training loss tells us how well we predict the batches currently being optimized. We also need measurements on text held out from training, and tests of capabilities that matter to the eventual application. During these checks we measure the current weights without training on the evaluation examples.

| Check | Question | Participating ranks |
| --- | --- | --- |
| Held-out BPB | How well does the model predict retained validation text per represented byte? | All ranks |
| CORE | How well does the base model perform on the selected benchmark tasks? | All ranks |
| Greedy prompt completions | What continuations does the model visibly produce? | Rank 0 |

These measure different properties. Lower validation loss is useful evidence of better predictive fit, while downstream task performance tests more specific capabilities. A few attractive completions are a qualitative diagnostic, not a reliable capability estimate.

Evaluation functions disable gradient recording where appropriate. Calling `.eval()` itself only changes module evaluation behavior; it is not the operation that turns off autograd. This architecture has no dropout, but explicit train/eval phases still organize execution and compilation.

### 13.2 Validation token accounting

Let $E_{requested}$ be the requested number of validation target tokens. Evaluation has no gradient-accumulation update: each forward across the ranks processes $WbT$ target positions. The number of validation iterations is therefore

$$
K_{eval}=\left\lfloor
\frac{E_{requested}}{WbT}
\right\rfloor.
$$

The loader is rebuilt for each evaluation, providing a consistent held-out starting region for a fixed setup. The actual target-position budget is $K_{eval}WbT$; the byte metric subsequently excludes special or ignored targets. Choosing a requested budget below one global validation batch yields zero iterations rather than a useful evaluation.

**Bits per byte (BPB)** expresses prediction loss per represented text byte, rather than per tokenizer token. Divide total negative log probability by $\ln2$ to convert nats to bits, then divide by the number of bytes those ordinary target tokens represent:

$$
\mathrm{BPB}=
\frac{\text{sum ordinary-target negative log probabilities}}
{\ln2\times\text{sum represented target bytes}}.
$$

The evaluator sums numerator and denominator across ranks before dividing. This weights the result by represented bytes rather than averaging rank-local ratios. Raw training loss and validation BPB have different units and target inclusion rules; their numerical magnitudes should not be compared directly.

For example, a summed loss of $100\ln2$ nats over targets representing 200 bytes gives 100 bits/200 bytes, or 0.5 BPB. This is a normalization of prediction loss, not a measurement of how much GPU memory those tokens occupy.

### 13.3 CORE and baseline centering

[evaluate_core](../nanochat/scripts/base_eval.py#L57) loads its evaluation bundle if needed, evaluates tasks, and aggregates their baseline-centered accuracies. Baseline centering expresses progress from a task's reference accuracy toward perfect accuracy. For task $j$, let $a_j$ be measured accuracy and $r_j$ its baseline accuracy, both as fractions. Then

$$
c_j=\frac{a_j-r_j}{1-r_j},\qquad
\mathrm{CORE}=\frac1J\sum_{j=1}^Jc_j.
$$

The metadata expresses baseline percentages, so the source multiplies those values by 0.01 before using this formula. A score of zero corresponds to baseline accuracy, one to perfect accuracy; below-baseline performance can produce negative values. CORE is a mean across tasks, not an overall accuracy weighted by every task's example count.

For a task with baseline $r_j=0.25$ and measured accuracy $a_j=0.625$, the normalized score is $(0.625-0.25)/(1-0.25)=0.5$: halfway from baseline to perfect. $J$ is the number of tasks included in the final mean.

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

A single microbatch loss is noisy, so the log smooths it with an **exponential moving average (EMA)**. Each new value receives some weight while older values retain geometrically decreasing weights. Let $\ell_s$ be the final-microbatch loss logged at update index $s$ and $m_s$ its running average. With $\beta=0.9$,

$$
m_s=\beta m_{s-1}+(1-\beta)\ell_s,\qquad m_{-1}=0,
$$

$$
\widehat m_s=\frac{m_s}{1-\beta^{s+1}}.
$$

Unrolling gives $(1-\beta)\sum_{i=0}^{s}\beta^{s-i}\ell_i$. For a constant loss, its weights sum to $1-\beta^{s+1}$, explaining the correction. This smooths the logged final-microbatch losses, not the model's weights and not an independently measured full-batch average.

For example, if the first logged loss is 4, the zero-initialized EMA becomes $m_0=0.1\times4=0.4$. Dividing by $1-0.9=0.1$ recovers 4 instead of misleadingly displaying 0.4. The same correction accounts for the missing initial history at later steps.

### 14.3 Throughput and MFU

Throughput measures how much work completes per second. **Model FLOPs utilization (MFU)** compares estimated model arithmetic per second with the chosen theoretical peak across the GPUs. For timed iteration duration $\Delta t$,

$$
\mathrm{tokens/sec}=G/\Delta t,\qquad
\mathrm{estimated\ FLOPs/sec}=\Phi G/\Delta t,
$$

$$
\mathrm{MFU}(\%)=
100\frac{\Phi G/\Delta t}
{W\times\mathrm{GPU\ peak\ BF16\ FLOPs/sec}}.
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

Suppose training stops halfway through. Saving weights lets us run the current model, but continuing training also needs the optimizer's memory, the step index for schedules, and a position in the data stream. A useful conceptual state is

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

“Sharded” means each rank owns only part of the full optimizer state. Model weights are replicated across ranks after an update, but all the optimizer's moment tensors need not be. That is why one rank's model file is sufficient for the weights, while resuming the distributed optimizer needs the per-rank optimizer files.

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

We can now assemble the earlier calculations into one real configuration. Read this section as a chain: architecture determines parameter counts; those counts give a target token budget; the budget suggests a batch; the batch gives an update count and accumulation count; the chosen scales give group hyperparameters.

Consider the architecture and token budget selected by the current speedrun:

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

To see where the table comes from, the two central calculations are

$$
K=\left\lfloor\frac{5{,}838{,}484{,}992}{1{,}048{,}576}\right\rfloor
=5568\text{ steps},
\qquad
A=\frac{1{,}048{,}576}{8\times16\times2048}=4.
$$

Each of the eight GPUs processes four local microbatches, then the optimizer updates once; this repeats for 5568 steps. The rounding loses only 13,824 target positions relative to $D_*$. The “8 tokens per parameter” label refers to $N_s$, whereas dividing by all 1.384B parameters gives a different ratio. Neither ratio is the corpus's total available token count.

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

At the first update all group LRs receive the warmup multiplier $1/40$. The warmdown length is $\mathrm{round}(0.65\cdot5568)=3619$, starting from the plateau boundary at index 1949.

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

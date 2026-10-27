# Efficient nanochat training: precision, attention kernels, compilation, and memory

The architecture note defined the model, the pretraining note defined the training run, and the optimization note explained how gradients become updates. This note asks: **how do we execute those computations with a practical amount of GPU memory and time, and what numerical changes does that introduce?**

The reference checkout is nanochat commit `92d63d4e8bb4df75c3b71618f31ddde2378b2bcd`; comparisons use your `transformer-from-scratch` at `56092a1818027c800f08167ec0c1c65b983711e1`. Source links point to exact lines in those snapshots. Hardware compatibility statements describe this code's decisions, not a continuously updated catalogue of every GPU or kernel release.

**How to skim:** every numbered section and subsection begins with a **Gist** paragraph. Those paragraphs give the purpose and result before the details. Read sections 1–4 for ordinary mixed precision, 5–7 for FP8 linear layers, 8–11 for efficient attention, and 12–15 for compilation, memory, and measurement. The final sections connect to your code and give optional small checks.

Two distinctions recur throughout. **A tensor's stored dtype does not completely specify the arithmetic used by a kernel.** And **an implementation that is algebraically equivalent in real arithmetic need not produce bitwise-identical floating-point results**. FlashAttention reorganizes the attention calculation; FP8 also changes the numerical representation of its operands. We will keep those mechanisms separate.

## Contents

1. [What makes a training computation expensive?](#1-what-makes-a-training-computation-expensive)
2. [Floating-point formats: range and resolution](#2-floating-point-formats-range-and-resolution)
3. [Nanochat's ordinary mixed-precision path](#3-nanochats-ordinary-mixed-precision-path)
4. [FP16 loss scaling and synchronized skipped updates](#4-fp16-loss-scaling-and-synchronized-skipped-updates)
5. [FP8 quantization: scale, round, and reconstruct](#5-fp8-quantization-scale-round-and-reconstruct)
6. [The three FP8 matrix products of a linear layer](#6-the-three-fp8-matrix-products-of-a-linear-layer)
7. [Shapes, strides, and the cost of arranging operands](#7-shapes-strides-and-the-cost-of-arranging-operands)
8. [Why attention needs a different memory strategy](#8-why-attention-needs-a-different-memory-strategy)
9. [Deriving attention with online softmax](#9-deriving-attention-with-online-softmax)
10. [Attention backward without saving the full probability matrix](#10-attention-backward-without-saving-the-full-probability-matrix)
11. [Which attention path this checkout actually executes](#11-which-attention-path-this-checkout-actually-executes)
12. [Compilation and fusion](#12-compilation-and-fusion)
13. [Accounting for training memory](#13-accounting-for-training-memory)
14. [A worked d24 memory example](#14-a-worked-d24-memory-example)
15. [Timing, data movement, and meaningful throughput](#15-timing-data-movement-and-meaningful-throughput)
16. [Connections to your code and small experiments](#16-connections-to-your-code-and-small-experiments)
17. [Verification limits, source caveats, and next steps](#17-verification-limits-source-caveats-and-next-steps)

## 1. What makes a training computation expensive?

**Gist:** training can be limited by arithmetic, data movement, memory capacity, or execution overhead. Reducing one cost does not necessarily reduce the others. The point of an efficiency technique is to remove a specific bottleneck while retaining acceptable training behavior.

### 1.1 Four resources to keep separate

**Gist:** “fewer FLOPs,” “less memory,” and “faster” are different claims. We need to identify which resource an implementation changes before predicting its effect.

| Resource | Example cost | Relevant technique |
| --- | --- | --- |
| Arithmetic | Large dense matrix products | Tensor-core-compatible dtypes and shapes |
| Memory traffic | Repeatedly reading/writing attention probabilities | Tiling, fusion, recomputation |
| Memory capacity | Weights, gradients, histories, saved activations | Lower storage precision, state sharding, smaller microbatches |
| Execution overhead | Launching many small operations; Python dispatch | Compilation, fused kernels, batched operations |

A **kernel** is a device implementation of an operation or fused group of operations. A **tensor core** is specialized GPU hardware for matrix operations with supported formats and shapes. A **tile** is a small block of a larger matrix processed together. These are execution concepts; they do not by themselves define a new learning objective.

### 1.2 A simple arithmetic-intensity model

**Gist:** matrix multiplication becomes easier to run efficiently when each value loaded from memory is reused many times. Large training batches create that reuse; single-token decoding often has much less of it.

For $Y=XP^\top$ with shapes $(M,K)(K,N)$, arithmetic is approximately $2MKN$ FLOPs. If each entry uses $p$ bytes, an idealized read-once/write-once traffic count is

$$
B_{traffic}\approx p(MK+NK+MN),\qquad
I=\frac{2MKN}{p(MK+NK+MN)}.
$$

Here $I$ is arithmetic intensity, in FLOPs/byte. This is a simplified lower-traffic model, not a description of every cache access. If a device has arithmetic capacity $F_{peak}$ and memory bandwidth $B_{mem}$, an idealized lower bound is

$$
\Delta t\gtrsim\max\left(\frac{\text{FLOPs}}{F_{peak}},
\frac{\text{bytes moved}}{B_{mem}}\right).
$$

Launch overhead, communication, imperfect reuse, and dependencies can make execution slower. Merely increasing peak arithmetic speed helps little when moving data dominates.

For training, $M=bT$ can be large: one weight matrix is reused across many token vectors. For decoding with one new token per sequence, $M$ may be only the active sequence batch size. When weight traffic dominates, the intensity approaches $2M/p$. This is why efficient training and efficient decoding are related but distinct problems; the inference note will revisit the latter.

## 2. Floating-point formats: range and resolution

**Gist:** floating point distributes a finite number of representable values across a range of magnitudes. More exponent bits give range; more fraction bits give finer resolution. BF16 trades precision for much wider range than FP16, despite both occupying two bytes.

### 2.1 The exponent and significand do different jobs

**Gist:** the exponent controls the scale of a number, and the fraction determines detail within that scale. Rounding is inevitable, and the spacing between representable values grows with magnitude.

For an ordinary normal binary number, write

$$
x=(-1)^s\,2^{e-\mathrm{bias}}\left(1+\frac f{2^r}\right),
$$

where $s$ is a sign bit, $e$ an encoded exponent, and $r$ the number of explicitly stored fraction bits. Special exponent patterns handle zero, subnormals, and format-specific exceptional values.

In the interval $[1,2)$, spacing is $2^{-r}$. Rounding to nearest has error at most half a spacing there; the usual unit roundoff is $u=2^{-(r+1)}$ for normal values under that rounding rule. Relative error descriptions require care near zero, overflow, and subnormal values.

| Format | Bytes | Exponent bits | Fraction bits | Spacing above 1 | Largest finite magnitude |
| --- | ---: | ---: | ---: | ---: | ---: |
| FP32 | 4 | 8 | 23 | $2^{-23}$ | About $3.40\times10^{38}$ |
| BF16 | 2 | 8 | 7 | $2^{-7}=0.0078125$ | About $3.39\times10^{38}$ |
| FP16 | 2 | 5 | 10 | $2^{-10}=0.0009765625$ | 65,504 |
| FP8 E4M3FN | 1 | 4 | 3 | $2^{-3}=0.125$ | 448 |
| FP8 E5M2 | 1 | 5 | 2 | $2^{-2}=0.25$ | 57,344 |

These are the particular FP8 encodings used here, not every format called FP8. E4M3FN has special finite-range conventions, so do not infer its maximum by blindly applying an IEEE FP16 exponent rule. The [FP8 format paper](https://arxiv.org/abs/2209.05433) describes this range/precision tradeoff.

### 2.2 Underflow, overflow, and lost small updates

**Gist:** a calculation can fail because a value is too large, too small, or too close to another representable value. Keeping an fp32 master parameter helps accumulate small changes, even when the next forward uses a coarser version.

FP16's smallest positive normal value is $2^{-14}$, about $6.10\times10^{-5}$; its smallest positive subnormal is $2^{-24}$, about $5.96\times10^{-8}$. BF16's corresponding thresholds are $2^{-126}$ and $2^{-133}$. Actual kernels may flush some subnormal arithmetic to zero. The exponent range is one reason BF16 is often easier to train with than FP16.

Range does not solve coarse resolution. Subtracting $10^{-4}$ from a parameter near 1 and rounding immediately back to BF16 can leave its stored value unchanged. Repeating that same rounded-away subtraction does not accumulate a hidden remainder.

An fp32 master parameter can accumulate those changes. Each forward may still see the same BF16 cast for several updates until a rounding boundary is crossed. The master is more accurate storage, not a way to make the forward operands retain fp32 information after casting.

### 2.3 Input precision, accumulation, and output dtype

**Gist:** a dot product has three potentially different precisions: its operands, its running accumulation, and its stored output. An fp32 output cannot recover information already lost by rounding its inputs.

For $y=\sum_i a_ib_i$, error can arise when representing $a_i,b_i$, multiplying them, accumulating partial sums, and storing $y$. Hardware and kernel settings determine those stages. A BF16 matrix product can use wider internal accumulators while returning a BF16 tensor; its `.dtype` alone does not reveal all intermediate arithmetic.

Nanochat also [sets float32 matmul precision to `high`](../nanochat/nanochat/common.py#L194) on CUDA. According to [PyTorch's precision documentation](https://docs.pytorch.org/docs/2.9/generated/torch.set_float32_matmul_precision.html), this permits faster reduced-precision internal algorithms for float32 matrix products when available, without changing the output dtype. Thus “stored and returned as FP32” is not a promise that every multiply used the highest-precision FP32 path. This setting is separate from BF16 and FP8 operand conversion.

## 3. Nanochat's ordinary mixed-precision path

**Gist:** nanochat explicitly manages important casts instead of placing its entire forward under autocast. Most projection weights are stored in fp32, while activations and matrix products follow the selected compute dtype. Embedding tables and optimizer histories have important exceptions.

### 3.1 Which dtype is selected?

**Gist:** this checkout defaults to BF16 on sufficiently recent CUDA devices and FP32 otherwise. An environment override is read at import time; it is a request, not a universal hardware-compatibility guarantee.

**Code:** [_detect_compute_dtype](../nanochat/nanochat/common.py#L17).

The selection order is:

1. If `NANOCHAT_DTYPE` is set, map `bfloat16`, `float16`, or `float32` to the requested dtype.
2. Otherwise, CUDA capability at least `(8,0)` selects BF16.
3. Older CUDA devices, CPU, and MPS select FP32 by default in this code.

`COMPUTE_DTYPE` is assigned once when `common.py` is imported. Device autodetection and this dtype choice are related but separate. For example, opting into BF16 on MPS changes dtype selection without changing the CUDA-only FP8 branch.

An old comment says FP16 GradScaler is not implemented; the current `base_train.py` does contain that branch. Use executable paths rather than that comment to understand current pretraining behavior.

### 3.2 Follow the casts through one ordinary forward

**Gist:** the casts are local and visible: embed in the chosen dtype, cast linear weights to their input's dtype, then promote logits before the softcap and loss. Promoting the logits improves subsequent arithmetic but does not redo their preceding matmul in fp32.

**Code:** [embedding storage](../nanochat/nanochat/gpt.py#L265), [Linear.forward](../nanochat/nanochat/gpt.py#L49), [forward input cast](../nanochat/nanochat/gpt.py#L472), [readout promotion](../nanochat/nanochat/gpt.py#L514).

| Object or operation | Ordinary BF16 configuration |
| --- | --- |
| Hidden projection and output-head parameter storage | FP32 |
| Main/value embedding parameter storage | BF16 |
| Embedding output / main activation path | BF16 |
| Ordinary custom linear operands | Activation dtype; weights cast for the product |
| Rotary cosine/sine buffers | Compute dtype |
| Logits after explicit `.float()` | FP32, before softcap and CE |
| AdamW moment storage | Same dtype as the owned parameter slice |
| AdamW update arithmetic | Explicit fp32 casts inside its kernel |
| Muon momentum storage | Parameter dtype; polar work may use BF16 |

In the explicit FP16 configuration, embeddings stay fp32 because the scaler path cannot unscale FP16 gradient storage. The embedding output is nevertheless cast to FP16 for the model computation. Parameter gradients generally follow the leaf parameter's dtype; casting a weight for the forward does not change that parameter object's persistent dtype.

`norm(x)` calls `F.rms_norm` without forcing an explicit Python-level fp32 cast. Do not deduce an internal reduction dtype solely from its BF16 input/output; reduction details belong to the chosen kernel. Likewise, scalar promotion follows PyTorch's rules, not a claim that every tensor in the whole graph has one dtype.

### 3.3 Explicit casting versus your autocast context

**Gist:** autocast chooses execution dtypes for eligible operations; explicit casting chooses them in the model code. Neither approach automatically makes all parameters or all arithmetic low precision.

Your [training loop](../transformer-from-scratch/train.py#L84) uses `torch.autocast(..., dtype=torch.bfloat16)`. Eligible matrix operations can run in lower precision while other operations follow their own policy. It does not permanently convert every model weight to BF16. The [AMP documentation](https://docs.pytorch.org/docs/2.9/amp.html) describes these per-operation policies.

Nanochat's custom `Linear` instead calls `F.linear(x, self.weight.to(dtype=x.dtype))`. This makes a major dtype boundary explicit and avoids a broad autocast context in base training. Autograd propagates gradients through the cast, including conversion back to the master parameter's gradient dtype.

This does not mean explicit casting is always faster or more accurate. Repeated casts can themselves create work or temporary storage; compilation and operation reuse influence their practical cost. The distinction is control of the policy, not a universal ranking of the two approaches.

## 4. FP16 loss scaling and synchronized skipped updates

**Gist:** small backward values can disappear in FP16. Loss scaling enlarges them during backward, then removes that factor before the optimizer. All ranks must agree if nonfinite gradients require skipping an update.

### 4.1 Why scaling the loss scales every gradient

**Gist:** a constant multiplier on the scalar loss propagates through the chain rule. The optimizer should receive the unscaled gradient, so scaling is not intended to change the learning rate.

For a scale $S>0$,

$$
\nabla_\theta(S\mathcal L)=S\nabla_\theta\mathcal L.
$$

If a tiny derivative would otherwise round to zero, multiplying upstream derivatives by $S$ can keep it representable during backward. Dividing accumulated parameter gradients by $S$ before updating aims to recover the original gradient. It cannot recover information already lost in the forward or fix every possible numerical failure.

Too large a scale can instead cause overflow. `GradScaler` adapts it using nonfinite-gradient checks. BF16's wider exponent range usually removes the need for this particular technique; this trainer creates a scaler only for the FP16 compute path.

### 4.2 Scaling and accumulation must use a consistent scale

**Gist:** each microbatch loss is first normalized for accumulation, then scaled. Unscale only after all microbatches of the update have contributed, keeping one scale for that accumulation window.

**Code:** [GradScaler creation](../nanochat/scripts/base_train.py#L324), [scaled backward](../nanochat/scripts/base_train.py#L513), [unscale and update](../nanochat/scripts/base_train.py#L529).

The sequence is

```text
for each of A microbatches:
    backward on S * (microbatch_mean_loss / A)
unscale all accumulated parameter gradients by S
perform the optimizer step if gradients are finite
adjust S for future iterations
clear gradients
```

Updating $S$ halfway through accumulation would mix differently scaled contributions in one `.grad` buffer. Nanochat's order avoids that. Its scale is not serialized in the checkpoint path discussed earlier, so resuming need not reproduce an uninterrupted FP16 scaling trajectory exactly.

### 4.3 All ranks must make the same skip decision

**Gist:** if one rank skips while others enter optimizer collectives or update their weights, the distributed algorithm breaks. Nanochat combines nonfinite flags before letting the scaler decide whether to call the optimizer.

After unscaling, the trainer [all-reduces the scaler's found-inf flags using MAX](../nanochat/scripts/base_train.py#L533). Any rank's nonfinite flag therefore makes every participating rank skip that optimizer invocation. The loss scale is then updated and gradients cleared.

This code accesses a private scaler helper, `_found_inf_per_device`, so compatibility follows the actual PyTorch implementation in the environment. The training-loop counter still advances on a skipped update. Tokens processed, iterations attempted, and successful weight updates can consequently diverge. The CUDA-oriented branch should not be assumed to become a tested CPU/MPS FP16 trainer merely by changing one environment variable.

## 5. FP8 quantization: scale, round, and reconstruct

**Gist:** FP8 has too little unscaled range/resolution to blindly cast every tensor. Nanochat dynamically rescales each operand to use the available range, stores an FP8 approximation, and supplies the inverse scale to the matrix-product kernel. Scaling reduces avoidable range loss but does not eliminate quantization error.

### 5.1 The exact tensorwise scaling rule

**Gist:** one scalar maps a tensor's largest absolute value to the format's largest finite value. Every entry shares that scale, so tensor outliers influence the resolution available to smaller entries.

**Code:** [_to_fp8](../nanochat/nanochat/fp8.py#L82).

For a finite input tensor $X$, let $F_{max}$ be 448 for E4M3FN or 57,344 for E5M2. The code computes

$$
a=\max_{i,j}|X_{ij}|,\qquad
\alpha=\frac{F_{max}}{\max(a,10^{-12})},
$$

$$
Q_X=\mathrm{cast}_{FP8}
\left(\mathrm{clamp}(\alpha X,-F_{max},F_{max})\right),
\qquad d_X=1/\alpha.
$$

The represented approximation is $\widehat X=d_XQ_X$, with multiplication understood after promotion to a suitable arithmetic format. The stored FP8 value is not directly in the original tensor's units: the inverse scale restores those units.

The implementation computes the maximum from `x.float()`, computes the scale division in float64, then stores that scale as float32. This is a deliberate numerical choice in the quantizer. The epsilon keeps an all-zero input's scale finite; its quantized data remains zero. There is no learned quantization scale here and no moving history of past maxima.

### 5.2 Why scaling does not remove rounding error

**Gist:** fitting the maximum into range cannot create more fraction bits. Tensorwise scaling also cannot simultaneously give every differently sized row its own best range.

E4M3FN's smallest positive subnormal is $2^{-9}$. With rounding to nearest, sufficiently small scaled values can become zero. For example, an input tensor containing both 1000 and 0.001 has $\alpha=448/1000=0.448$; the smaller entry becomes 0.000448 before the FP8 cast, below half the smallest subnormal, and rounds to zero.

In the ordinary normal range, write $\widehat X=X+E_X$. A relative-error model such as $|E_{X,ij}|\lesssim u|X_{ij}|$ is only a local approximation away from exceptional values and underflow. For a matrix product,

$$
\widehat X\widehat P^\top-XP^\top
=E_XP^\top+XE_P^\top+E_XE_P^\top,
$$

even before accounting for multiplication/accumulation/output rounding. This shows why operand quantization errors propagate through the product and why later promoting the output cannot undo them.

Rowwise or blockwise scaling can adapt to local magnitude differences, at the cost of additional scale handling and different kernel requirements. This checkout implements only the tensorwise recipe. Do not infer a general superiority claim from the module comment about which scaling method is faster.

### 5.3 This is floating-point scaling, not integer wraparound

**Gist:** explicit clamping makes the intended saturation behavior clear. An out-of-range floating-point cast should not be interpreted as modular integer arithmetic.

The source comment says the default cast “wraps”; that is not a reliable mathematical description of FP8 conversion. Depending on the format and conversion, unrepresentable values can become nonfinite or otherwise fail to provide the intended saturated value. The explicit clamp specifies finite bounds before the cast. The derivation above assumes finite inputs; taking an absolute maximum does not repair a tensor already containing NaNs or infinities.

### 5.4 What the FP8 switch actually converts

**Gist:** `--fp8` replaces selected linear modules while retaining their original parameter objects. It does not quantize the whole model, optimizer, attention, or checkpoint into eight-bit storage.

**Code:** [conversion filter](../nanochat/scripts/base_train.py#L178), [Float8Linear.from_float](../nanochat/nanochat/fp8.py#L216), [recipe selection](../nanochat/nanochat/fp8.py#L234).

In a CUDA run, selected `nn.Linear` modules must have input and output sizes divisible by 16 and both at least 128. This is the local recipe's alignment/size policy, not a universal restriction on every future FP8 implementation. Small smear and value-gate projections remain ordinary linear layers. Large hidden projections and the head can pass the filter.

The replacement shares the existing weight and bias parameters. Their stored dtypes and optimizer histories do not become FP8 merely because the matmul operands do. `Float8Linear.forward` explicitly casts the incoming activation to `COMPUTE_DTYPE`, flattens leading dimensions, and calls the custom autograd function.

The CLI advertises `rowwise`, but this local `Float8LinearConfig` accepts only `tensorwise`. Non-CUDA runs ignore the FP8 switch in this trainer. Actual `_scaled_mm` support still depends on hardware, build, dtype combinations, and layout; the Python filter alone does not establish compatibility on every CUDA device.

## 6. The three FP8 matrix products of a linear layer

**Gist:** one learned linear layer has one forward and two backward matrix products. Nanochat quantizes operands for all three and explicitly defines the backward. Its gradient is the implemented low-precision approximation, not the exact derivative of a rounding function.

**Code:** [_Float8Matmul.forward](../nanochat/nanochat/fp8.py#L133), [backward](../nanochat/nanochat/fp8.py#L157).

### 6.1 Start with the ordinary linear derivatives

**Gist:** use the same three products that produced the factor six in the pretraining compute estimate. We now inspect their operand formats rather than their operation count.

Flatten the local activation to $X\in\mathbb R^{M\times K}$, with $M=bT$, and let $P\in\mathbb R^{N\times K}$. With output derivative $D=\partial\mathcal L/\partial Y$,

$$
Y=XP^\top,\qquad
\frac{\partial\mathcal L}{\partial X}=DP,\qquad
\frac{\partial\mathcal L}{\partial P}=D^\top X.
$$

The source's chosen formats are:

| Product | Left operand | Right operand | Shape of result |
| --- | --- | --- | --- |
| Forward | $Q_X$: E4M3FN | $Q_P^\top$: E4M3FN | $(M,N)$ |
| Input derivative | $Q_D$: E5M2 | $Q_P$: E4M3FN | $(M,K)$ |
| Weight derivative | $Q_D^\top$: E5M2 | $Q_X$: E4M3FN | $(N,K)$ |

E4M3FN allocates more bits to fraction precision; E5M2 allocates more to exponent range. This is the recipe's selected tradeoff for activations/weights versus backward operands, not a theorem about every tensor's optimal representation.

### 6.2 Inverse scales restore the product's units

**Gist:** each product needs both operands' inverse scales. `_scaled_mm` incorporates them so that its result approximates the original-units product, rather than the raw product of enlarged FP8 values.

For the forward,

$$
Y\approx d_Xd_P\,(Q_XQ_P^\top).
$$

Similarly, the input derivative uses $d_Dd_P$, and the weight derivative uses $d_Dd_X$. The forward saves the quantized activation and weight plus their inverse scales for backward; backward quantizes the incoming $D$ once and reuses those saved operands.

The requested forward output dtype is `input_2d.dtype`; the two requested backward output dtypes are `grad_output.dtype`. These are normally BF16 in the intended path. If the weight parameter is fp32, autograd's parameter-gradient path converts the returned weight gradient to the leaf's fp32 dtype; that conversion cannot recover precision lost by producing it in BF16.

The forward sets `use_fast_accum=True`, while both backward products set it to `False`. These flags select different accumulation behavior in the underlying kernel. They do not mean that all arithmetic is FP8, nor do they by themselves establish an exact internal accumulation model on every backend.

### 6.3 What custom backward does and does not differentiate

**Gist:** dynamic maxima, rounding, and saturation are not differentiated as ordinary model operations here. The custom backward supplies linear-layer-shaped approximate derivatives using quantized operands.

`_to_fp8` is under `@torch.no_grad()`, and `_Float8Matmul` provides its own backward. It does not apply the mathematical derivative of a piecewise-constant cast, which would be zero almost everywhere and unsuitable for the intended training procedure. Nor does it differentiate the maximum-based scale.

Instead, it substitutes low-precision matrix products into the usual derivative formulas. This is an explicit surrogate/approximate backward rule. It should be assessed by numerical error and training behavior, rather than called the exact gradient of a smooth quantized forward. Changing FP8 recipes can therefore change both forward predictions and the update trajectory.

## 7. Shapes, strides, and the cost of arranging operands

**Gist:** matrix shape specifies which entries mathematically interact; strides specify where those entries live in memory. A fast kernel may require a particular layout, and changing layout can require copies that consume time and temporary memory.

### 7.1 A transpose can be a view rather than a copy

**Gist:** transposing a contiguous matrix usually changes shape/strides without rearranging its data. This can produce exactly the column-major layout the FP8 kernel's second operand needs.

For a contiguous $(M,K)$ matrix, the strides are normally $(K,1)$: moving one row advances $K$ entries, moving one column advances one. Its transpose has strides $(1,K)$ and shares storage.

The FP8 code supplies the first operand in row-major form and the second in column-major form. A contiguous weight $P$ of shape $(N,K)$ already gives a suitable $P^\top$ view for the forward. No weight-transpose copy is needed there.

For an operand that is not already in the required orientation, [_to_col_major](../nanochat/nanochat/fp8.py#L110) uses

```python
x.t().contiguous().t()
```

The logical values and final shape match `x`, but an ordinary nondegenerate contiguous input ends with column-major strides. The middle `.contiguous()` is the operation that rearranges data; a plain final transpose does not undo that physical arrangement.

### 7.2 Backward has real layout work

**Gist:** the saved weight/input layouts that were convenient for forward are not automatically convenient for both derivative products. Those copies are part of FP8's overhead.

Input backward converts the saved weight to column-major layout. Weight backward makes `grad_output.T` contiguous for the first operand, and converts the saved input to column-major for the second. `_scaled_mm` also requires 2D operands, so `Float8Linear` reshapes the original activation and restores its leading dimensions afterward.

`reshape` may return a view or a copy depending on strides. Likewise, a `.contiguous()` call can be free if the required layout already holds, or allocate/copy if it does not. The relevant measurement is the executed path, not the presence of the method name alone.

FP8 can speed up large GEMMs while paying extra reductions, scaling, casts, and layout costs. A twofold peak hardware arithmetic ratio does not imply a twofold end-to-end training speedup.

## 8. Why attention needs a different memory strategy

**Gist:** attention's problem is not just large projection matrices. A direct implementation also constructs a score/probability matrix for every head, with size quadratic in sequence length. FlashAttention avoids storing that full matrix in device main memory by changing the evaluation order.

### 8.1 The matrix we would rather not save

**Gist:** an attention output is modest in size, but its naive intermediate has one entry for every query/key pair. Computing that intermediate, writing it out, and reading it back can dominate memory traffic and capacity.

For one head, with $Q,K\in\mathbb R^{T\times d_h}$ and values $V\in\mathbb R^{T\times d_v}$,

$$
S=QK^\top/\sqrt{d_h}+\mathrm{mask},\qquad
A=\mathrm{softmax}(S),\qquad O=AV.
$$

Softmax acts over allowed key positions separately for each query. With local batch $b$ and $H$ query heads, a naive score or probability tensor has $bHT^2$ entries. At two bytes each, doubling $T$ quadruples that tensor's storage. The output has only $bHTd_v$ entries.

This is a property of a materialized implementation, not a proof that exact attention inherently requires storing $T^2$ values. The output only needs a weighted sum for each row. We can accumulate that sum while streaming blocks of keys and values.

### 8.2 What FlashAttention changes

**Gist:** FlashAttention tiles exact attention to exploit fast on-chip storage and avoid repeated transfers of large intermediates to/from high-bandwidth device memory. It saves memory through tiling and backward recomputation, not by replacing softmax with a different model.

GPU **HBM** is large device main memory; on-chip shared memory/registers are much smaller but faster. A tiled kernel loads manageable blocks, computes scores, updates row statistics/output accumulators, and writes compact results. The [FlashAttention paper](https://arxiv.org/abs/2205.14135) develops this IO-aware exact-attention strategy.

For full attention, arithmetic still grows quadratically with context length; the avoided object is the large HBM-resident score/probability matrix. A tiled implementation still has finite temporary tile storage. “Exact” means the intended attention equation is unchanged in real arithmetic, not that every implementation has identical rounding or bitwise output.

The following derivation explains how this is possible without reading external kernel source. Nanochat delegates the optimized kernel itself to FA3 or PyTorch's selected SDPA implementation.

## 9. Deriving attention with online softmax

**Gist:** softmax needs a global denominator, but we do not have to know it before accumulating information from earlier blocks. Keep a running maximum, exponential sum, and weighted-value sum; rescale the old accumulators whenever a new maximum appears.

### 9.1 Stable softmax as three sufficient quantities

**Gist:** subtracting a common maximum prevents unnecessarily large exponentials and cancels in the ratio. For each query row, the entire attention result can be summarized by a scalar maximum, a scalar denominator, and a value-sized vector.

Fix one query and write its allowed scores as $s_j=q^\top k_j/\sqrt{d_h}$. Then

$$
o=\frac{\sum_j e^{s_j}v_j}{\sum_j e^{s_j}}
=\frac{\sum_j e^{s_j-m}v_j}{\sum_j e^{s_j-m}},
\qquad m=\max_j s_j.
$$

Define $\ell=\sum_j e^{s_j-m}$ and $u=\sum_j e^{s_j-m}v_j$. The answer is $u/\ell$. We need the accumulated weighted sum, not a stored copy of each probability.

### 9.2 Merge a new block with the accumulated result

**Gist:** when a new block raises the running maximum, the old numerator and denominator are expressed in the old units. Multiply both by the same correction before adding the new block.

Suppose $(m,\ell,u)$ summarizes already processed keys. For a new block $J$, set

$$
m'=\max\left(m,\max_{j\in J}s_j\right),\qquad
a=e^{m-m'},\qquad p_j=e^{s_j-m'}.
$$

Then

$$
\boxed{\ell'=a\ell+\sum_{j\in J}p_j},\qquad
\boxed{u'=au+\sum_{j\in J}p_jv_j}.
$$

Why multiply the old sum by $a$? Each old term obeys $e^{s_j-m'}=e^{m-m'}e^{s_j-m}$. The same identity applies to its weighted-value contribution. Repeating the merge therefore reproduces the full softmax numerator/denominator without materializing them across all keys at once.

Initialize with $m=-\infty$, $\ell=0$, and $u=0$, then divide $u/\ell$ after the allowed blocks have been processed. Fully masked blocks contribute nothing and need appropriate handling: evaluating $-\infty-(-\infty)$ blindly would create NaNs. Each row must have at least one valid key to define this ordinary softmax result.

### 9.3 A two-key example

**Gist:** the running denominator changes units when the larger score arrives, but the final answer is exactly the same weighted average. This is the core idea behind the more elaborate GPU implementation.

Take scalar values $v_1=2$, $v_2=10$ and scores $s_1=0$, $s_2=\ln3$. After the first key, $m=0$, $\ell=1$, $u=2$. Processing the second gives

$$
m'=\ln3,\quad a=1/3,\quad p_2=1,
$$

$$
\ell'=1/3+1=4/3,\qquad
u'=2/3+10=32/3,\qquad
o=u'/\ell'=8.
$$

The ordinary softmax weights are $1/4$ and $3/4$, giving the same $2/4+30/4=8$. A real kernel processes query/key tiles in parallel, but each row follows this kind of normalization bookkeeping.

## 10. Attention backward without saving the full probability matrix

**Gist:** backward can recompute probability tiles from queries, keys, and saved row statistics. It does more arithmetic to avoid saving/reloading a huge intermediate. This trade is worthwhile when memory movement/capacity is the limiting cost.

### 10.1 Reconstruct probabilities from compact statistics

**Gist:** saving the row log-sum-exp is enough to reconstruct any probability tile later. This is specialized attention recomputation, distinct from checkpointing whole transformer blocks.

Let

$$
L_i=m_i+\ln\ell_i
=\ln\sum_j e^{S_{ij}}.
$$

For an allowed score, $A_{ij}=e^{S_{ij}-L_i}$; masked entries have probability zero. A backward kernel can recompute a block of $S$ from $Q$ and $K$ and immediately form its probability block using $L_i$. It need not retain the full forward $A$ tensor in HBM.

Queries, keys, values, outputs, compact statistics, and other necessary saved tensors still consume memory. The claim is about avoiding quadratic score/probability storage, not zero-memory backward. Nanochat's base trainer does not call general transformer activation checkpointing; attention's internal recomputation does not imply all model activations are checkpointed.

### 10.2 Differentiate the attention equation

**Gist:** the softmax derivative needs one row reduction and local probability values. Both can be obtained without storing a full Jacobian or full attention matrix.

Let $D=\partial\mathcal L/\partial O$. Direct differentiation gives

$$
\frac{\partial\mathcal L}{\partial V}=A^\top D,\qquad
E=\frac{\partial\mathcal L}{\partial A}=DV^\top.
$$

For each score row,

$$
\frac{\partial\mathcal L}{\partial S_{ij}}
=A_{ij}\left(E_{ij}-\sum_k A_{ik}E_{ik}\right).
$$

The row reduction can also be written

$$
\sum_k A_{ik}E_{ik}=D_i\cdot O_i,
$$

because $O_i=\sum_k A_{ik}V_k$. This is one scalar per query row. Finally,

$$
\frac{\partial\mathcal L}{\partial Q}
=\frac1{\sqrt{d_h}}\frac{\partial\mathcal L}{\partial S}K,
\qquad
\frac{\partial\mathcal L}{\partial K}
=\frac1{\sqrt{d_h}}\left(\frac{\partial\mathcal L}{\partial S}\right)^\top Q.
$$

A tiled backward accumulates these contributions blockwise. These equations explain the possibility; the external kernel additionally handles parallel work partitioning, numerical choices, and reduction scheduling.

## 11. Which attention path this checkout actually executes

**Gist:** nanochat presents one attention interface with an external FA3 branch and a PyTorch SDPA fallback. Falling back from its FA3 branch does not necessarily mean falling back to naive materialized attention. Local masks and kernel eligibility still matter.

### 11.1 FA3 selection and the meaning of `--fp8`

**Gist:** the wrapper selects FA3 only when its kernel loads and the compute dtype is BF16. Enabling FP8 linear layers does not independently switch attention to an FP8 attention algorithm.

**Code:** [kernel loading](../nanochat/nanochat/flash_attention.py#L23), [dtype gate](../nanochat/nanochat/flash_attention.py#L56), [dispatch](../nanochat/nanochat/flash_attention.py#L127).

The loader tries a Hopper-specific Hugging Face kernel when CUDA capability's major version is 9, and a community kernel otherwise; exceptions give a fallback. Availability and the actual loaded build determine support. Comments listing GPU generations should not be treated as an exhaustive current compatibility specification.

`_resolve_use_fa3` enables the automatic FA3 branch only for `COMPUTE_DTYPE == torch.bfloat16`. The ordinary model passes its BF16 Q/K/V tensors to that API. There is no FP8 attention scale/quantization path added by `--fp8` in this wrapper.

[FlashAttention-2](https://arxiv.org/abs/2307.08691) improves parallel work partitioning, while [FlashAttention-3](https://arxiv.org/abs/2407.08608) develops Hopper-oriented asynchronous execution and low-precision techniques. Those papers' capabilities are broader than the precise path nanochat invokes; do not infer that all published FA3 variants run here.

### 11.2 SDPA is an interface with backend selection

**Gist:** PyTorch can choose a fused attention implementation behind SDPA. The wrapper's transpose changes head layout; it does not itself force naive attention or physically copy every tensor.

**Code:** [SDPA wrapper](../nanochat/nanochat/flash_attention.py#L130), [_sdpa_attention](../nanochat/nanochat/flash_attention.py#L77).

FA3-facing tensors have shape $(b,T,H,d_h)$. The fallback transposes them to PyTorch SDPA's $(b,H,T,d_h)$ convention, enables GQA if query and KV head counts differ, and transposes the result back.

[PyTorch's SDPA documentation](https://docs.pytorch.org/docs/2.9/generated/torch.nn.functional.scaled_dot_product_attention.html) describes input-dependent backend selection, including fused and math implementations. Dtype, device, head shapes, masks, strides, and build affect eligibility. Therefore `USE_FA3=False` does not by itself identify the final kernel or its memory behavior.

The fallback assumes causal attention in its implementation; this is the model's intended use. The public argument `causal=False` is not a general switch to noncausal full attention in every fallback branch. The single-query cached case intentionally uses noncausal SDPA over already valid past/current keys.

### 11.3 Window masks change both the model and the available optimizations

**Gist:** local attention limits which keys a query may use. An efficient kernel can exploit that restricted structure, but expressing it as a dense boolean mask does not guarantee less dense arithmetic.

The model [constructs its per-layer windows](../nanochat/nanochat/gpt.py#L298). For $T=2048$, its short window is 512 preceding positions; the current position also counts, so a mature row can attend to 513 keys. The expression rounds the nominal quarter-context window upward to a multiple of 128. A nearby comment saying `2048 -> 768` is stale; the executed expression produces 512.

For a left window $w$ with $T\ge w+1$, the exact number of causal query/key pairs in one head is

$$
\sum_{t=0}^{T-1}\min(t+1,w+1)
=T(w+1)-\frac{w(w+1)}2.
$$

Full causal attention has $T(T+1)/2$ pairs. A kernel that skips disallowed tiles can exploit the difference. The SDPA fallback [constructs an explicit mask](../nanochat/nanochat/flash_attention.py#L102) for general local/chunk cases, with a logical $T_q\times T_k$ boolean array broadcast across batch/heads. That allocation and backend's mask support affect realized savings.

The training warning “SDPA has no support for sliding window attention” is stale as a mathematical claim: this helper does implement the mask. Functional support does not establish specialized sparse/windowed execution. The final layer still uses full context in nanochat's pattern, so local attention is also an architectural choice, not merely an equivalent rearrangement like full FlashAttention.

## 12. Compilation and fusion

**Gist:** eager execution can launch many small operations with substantial overhead and intermediate memory traffic. Compilation captures reusable computation and can fuse suitable operations. It does not automatically combine an entire training run into one kernel or prove its numerical equivalence bit for bit.

### 12.1 What fusion can save

**Gist:** fusing compatible operations can avoid writing each intermediate to device main memory and reduce kernel launches. Large library matrix products may remain separate calls while surrounding elementwise work is fused.

For example, an eager chain like `scale -> add -> square -> reduce -> normalize` may launch several kernels and create intermediates. A compiler can sometimes generate fewer kernels with reused intermediate values. This is especially useful for normalization, elementwise residual operations, and the many small operations in an optimizer update.

Nanochat [compiles its training model](../nanochat/scripts/base_train.py#L246) with `dynamic=False`. Its AdamW and Muon update functions are separately decorated with `fullgraph=True`, requesting a complete capture of those kernel functions rather than silently accepting graph breaks inside them. This is not a `fullgraph=True` promise for the entire Python training loop.

### 12.2 Why fixed shapes help, and why warmup timing is misleading

**Gist:** repeated shapes let compiled artifacts be reused. Changing shapes, module structure, or guarded assumptions may require another compilation. Initial timings can include compilation rather than steady training work.

The local microbatch and sequence length are fixed in the repeated training forward. `dynamic=False` asks for shape-specialized compilation rather than general dynamic-shape handling. It does not mean changing a shape is impossible, but such a change can cause recompilation or failure under the relevant settings. See [torch.compile](https://docs.pytorch.org/docs/2.9/generated/torch.compile.html) for the API contract.

The trainer retains `orig_model` with shared underlying weights for saving and variable-shaped CORE evaluation/sampling. Other mode/module changes, including temporarily disabling FP8, can also affect compiled guards. A short benchmark that includes first-use compilation can rank implementations differently from a steady-state benchmark; both startup cost and steady time matter for short runs.

### 12.3 `allow_in_graph` is not a universal opaque boundary

**Gist:** nanochat marks its FP8 autograd function for Dynamo to accept as a graph callable. This does not by itself prevent later compiler stages from tracing through it, despite the module's stronger “opaque node” description.

**Code:** [FP8 decorator](../nanochat/nanochat/fp8.py#L124).

The [PyTorch fine-grained tracing documentation](https://docs.pytorch.org/docs/2.9/torch.compiler_fine_grain_apis.html) distinguishes Dynamo capture from later AOTAutograd tracing. `allow_in_graph` affects the former; AOTAutograd can trace through the callable. It is therefore unjustified to infer a permanent fusion barrier across every FP8 operation solely from that decorator.

The source's performance comparison against torchao is commentary about an implementation/measurement context, not a benchmark established by these lines. We have not measured either path here. Determine actual fusion and overhead from generated graphs/profiling for the exact build and shapes, rather than assuming “custom autograd” implies one particular compiler outcome.

## 13. Accounting for training memory

**Gist:** a model's parameter file is only one part of training memory. We must separately count gradients, optimizer histories, saved activations, kernel/communication workspaces, and allocator overhead. FP8 and FlashAttention improve selected terms, not the entire total by one universal factor.

### 13.1 Persistent weights, gradients, and histories

**Gist:** weights remain throughout training; gradients and optimizer histories have different lifetimes and ownership. Mixed dtypes and Muon/AdamW grouping make a single “bytes per parameter” shortcut unreliable here.

For parameters $p$, stored-weight bytes are

$$
M_{weights}=\sum_p \mathrm{numel}(p)\,\mathrm{bytes}(\mathrm{dtype}(p)).
$$

When present, ordinary parameter gradients usually have the same shape/dtype as their leaf parameters. The trainer clears them with `set_to_none=True` after the update, so their allocation and lifetime need not be constant across the step.

AdamW stores two moment tensors per owned coordinate. This Muon variant stores one full momentum matrix plus one row/column second-moment vector per owned matrix. Large histories are sharded as explained in the optimization note, while weights remain replicated. Histories are initialized lazily, so a measurement before the first optimizer update misses them.

### 13.2 Activations, logits, and workspaces

**Gist:** backward needs forward information, which can outweigh parameter storage for large batches or contexts. Attention fusion removes one major quadratic intermediate, but residual/MLP activations and vocabulary logits remain substantial.

Useful individual tensor scales include:

| Object | Number of entries |
| --- | ---: |
| One residual-stream tensor | $bTd$ |
| One dense expanded MLP tensor | $4bTd$ |
| Naive attention score/probability tensor | $bHT^2$ |
| Full training logits | $bTV$ |
| One Muon Gram intermediate | Smaller-dimension squared per owned matrix |

Multiply by that tensor's actual dtype size. These counts are not a complete autograd memory formula: multiple tensors, aliases, saved inputs, temporary casts, and kernel choices determine the peak. FP8 saves quantized linear inputs/weights for its backward, but master weights and other graph activations still exist.

The readout explicitly constructs full-sequence logits, promotes them to fp32, and applies a softcap before cross-entropy. A fused/chunked linear-cross-entropy implementation could reduce some of that footprint, but the inspected Python source does not implement such an algorithm. Compilation may alter intermediate lifetimes; do not count every syntactic expression as an independently live permanent allocation.

### 13.3 Microbatching versus activation checkpointing

**Gist:** accumulation reduces the number of examples whose activation graphs must live together. Activation checkpointing instead trades recomputation for fewer saved intermediates within a forward graph. They solve related but different memory problems.

Nanochat backpropagates each microbatch before constructing the next one's graph. Increasing accumulation count $A$ while shrinking local microbatch $b$ can preserve the global token batch and reduce per-forward activation memory. It does not require retaining all $A$ activation graphs simultaneously. Parameters and accumulated gradients still remain.

**Activation checkpointing** would save selected boundary tensors and recompute some forward operations during backward. The base trainer does not explicitly apply general block checkpointing. This differs from attention's specialized internal recomputation and also from a **training checkpoint file**, which saves state to disk for resume.

### 13.4 Training activations are not an inference KV cache

**Gist:** teacher-forced training computes a sequence in parallel and needs gradients through it. Inference caching stores past keys/values to avoid recomputing them during future token generation. Their memory purposes and lifetimes differ.

In this model's training path, `kv_cache=None`; there is no persistent generation cache growing across optimizer steps. For inference, [KVCache allocates K and V](../nanochat/nanochat/engine.py#L99) with shape $(L,B,C,H_{kv},d_h)$ each, where $C$ is allocated cache capacity, not necessarily current occupied length.

Its principal bytes are

$$
M_{KV}=2LBCH_{kv}d_h\,p,
$$

where $p$ is bytes per cached entry. Metadata and the previous embedding for smear add small terms. GQA reduces $H_{kv}$; this checkout's base-training builder uses $H_{kv}=H$ by default. Local attention does not automatically shrink these cache allocations to a rolling window, particularly when full-context layers also exist.

## 14. A worked d24 memory example

**Gist:** the source's reference configuration makes it clear why one optimization cannot solve every memory cost. The weights are several GiB, but even one fp32 logits tensor is 4 GiB at the reference local batch. FlashAttention avoids another large tensor per layer.

### 14.1 Count specific tensors before estimating a peak

**Gist:** these numbers are source-based sizes for clearly stated tensors and persistent histories. They are not measured peak GPU memory or a claim that all listed tensors coexist independently.

Use the same configuration as the pretraining worked example:

$$
L=24,\quad d=1536,\quad H=H_{kv}=12,\quad d_h=128,
$$

$$
b=16,\quad T=2048,\quad V=32768,\quad W=8,
$$

with BF16 compute/embedding storage, fp32 ordinary projection storage, and the current mixed optimizer. Here $b$ is sequences per local microbatch and $W$ is world size. We use GiB/MiB as binary units: $1\,\mathrm{GiB}=2^{30}$ bytes and $1\,\mathrm{MiB}=2^{20}$ bytes.

| Quantity | Size | Scope |
| --- | ---: | --- |
| Stored parameters with the source's mixed dtypes | About 3.938 GiB | Replicated on every rank |
| Full parameter gradients, when all are allocated | About 3.938 GiB | Per rank |
| Persistent optimizer histories under eight-rank ownership | About 0.668 GiB | Per rank, including small replicated state and Muon padding |
| One BF16 residual tensor $(b,T,d)$ | 96 MiB | One tensor |
| One BF16 expanded MLP tensor $(b,T,4d)$ | 384 MiB | One tensor |
| One naive BF16 attention matrix $(b,H,T,T)$ | 1.5 GiB | One layer's score or probability tensor |
| One fp32 logits tensor $(b,T,V)$ | 4 GiB | One tensor |

For example, the logits calculation is

$$
16\times2048\times32768\times4
=4{,}294{,}967{,}296\text{ bytes}=4\text{ GiB}.
$$

Weights plus full gradients plus the counted persistent histories total about 8.543 GiB per rank **before** activations, communication stacks, casts, matrix workspaces, runtime allocations, and allocator reservation. Even that subtotal is not necessarily simultaneously live at every point. The optimizer figure follows the actual same-shape groups, row slicing, and padded Muon chunks; it is not simply all optimizer storage divided by eight.

The parameter-byte count also persists after FP8 module replacement because the replacement shares the original parameter tensors. Quantized saved operands add their own memory/lifetime tradeoffs; they do not substitute an all-FP8 model file.

### 14.2 What changes when batch or context changes?

**Gist:** reducing the local microbatch scales many activation tensors linearly. Increasing sequence length makes naive attention matrices grow quadratically, while cache memory and many other tensors grow linearly.

Halving $b$ halves the individual residual, MLP, and logits sizes above, while increasing $A$ can preserve the global batch. Doubling $T$ doubles those three tensors but quadruples the naive attention matrix. FlashAttention removes the need to keep that entire matrix, but full-attention arithmetic still grows quadratically.

For inference cache capacity $C=2048$, BF16 storage, and $B=1$, the principal K/V allocation is

$$
2\times24\times1\times2048\times12\times128\times2
=301{,}989{,}888\text{ bytes}=288\text{ MiB}.
$$

At $B=16$ it is 4.5 GiB. This is an inference cache example, not another term to add automatically to the training table. Reducing KV heads from 12 to 3 would divide this principal cache term by four for a model configured that way.

## 15. Timing, data movement, and meaningful throughput

**Gist:** GPU operations are asynchronous, and a training process does more than matmuls. Timing must account for synchronization, warmup, input preparation, communication, and evaluation. The most useful outcome is time to the desired quality, with a clearly defined measurement window.

### 15.1 Why a Python timer may measure launch time

**Gist:** launching a CUDA operation can return before the GPU finishes it. Synchronize around the region or use appropriate device events when measuring completed work; also separate compilation warmup from steady-state execution.

The trainer [synchronizes before](../nanochat/scripts/base_train.py#L508) and [after](../nanochat/scripts/base_train.py#L542) its timed CUDA training region. Its final `train_loss.item()` also requests a CPU scalar and can synchronize device work. Avoiding `.item()` inside every microstep prevents repeatedly forcing such reads.

The synchronization helper is a no-op on CPU/MPS in this script; asynchronous MPS timing therefore needs separate care. Your own loop explicitly uses `torch.mps.synchronize()` when appropriate.

For an iteration of duration $\Delta t$ that consumes global batch $G$ tokens,

$$
\mathrm{tokens/sec}=G/\Delta t.
$$

Report context length, local/global batch, accumulation, dtype, optimizer, backend, and world size with a performance number. Otherwise an apparent speedup can simply reflect different work.

### 15.2 Input staging and opportunities for overlap

**Gist:** preparing the next batch while the GPU is busy can reduce idle time, but `non_blocking=True` is not itself proof of complete transfer/compute overlap. Buffer placement, stream ordering, and CPU work all matter.

**Code:** [loader buffers](../nanochat/nanochat/dataloader.py#L111), [host-to-device copy](../nanochat/nanochat/dataloader.py#L160), [next-batch request](../nanochat/scripts/base_train.py#L518).

The loader uses one persistent buffer containing contiguous input and target views. On its CUDA path, the CPU staging buffer is pinned, allowing asynchronous host-to-device transfer, and the copy requests `non_blocking=True`. A pinned buffer is host memory suitable for device transfers; it is not itself GPU memory.

The training script requests the next batch after queuing each backward. CPU tokenization/packing can then occur while queued GPU work executes. The inspected code does not establish a separate multi-stage prefetch engine with guaranteed complete hiding of every transfer. Data preparation or communication can still become the bottleneck after faster compute kernels are introduced.

### 15.3 Throughput, MFU, and time to quality answer different questions

**Gist:** more tokens per second is useful only relative to the task being trained, and MFU depends on a meaningful FLOP estimate and chosen hardware denominator. Quantization can change numerical learning behavior as well as speed.

The source's MFU compares estimated model FLOPs/sec with a tabulated peak **BF16** throughput across ranks, including when FP8 linears are enabled. It excludes optimizer arithmetic from the model-FLOP estimate. It is not a direct hardware profiler or an FP8-peak utilization percentage.

This snapshot also has the [FP8 counting discrepancy](03-nanochat-pretraining.md#191-the-fp8-counting-issue-in-this-snapshot): converted linears are omitted by the custom-`Linear` type-based count. That makes its post-conversion FLOP/MFU reporting, and FLOP-derived horizons, unreliable without correcting the count. Nominal matrix dimensions have not disappeared.

Evaluation, sampling, checkpoint writes, and setup are outside the per-update timed region. The cumulative reported training time also excludes update indices 0 through 10. End-to-end wall time is a different measurement. Allocated-memory peaks and allocator-reserved memory likewise answer different questions; the script reports CUDA `max_memory_allocated`, not every byte reserved by the process or driver.

A useful comparison fixes the intended model/data/target metric, measures numerical error or learning impact, and reports time to that target. A faster step that requires substantially more steps to reach the target may not be a faster training method.

## 16. Connections to your code and small experiments

**Gist:** your implementation already uses several of these efficiency ideas through PyTorch: autocast, SDPA, gradient accumulation, and device synchronization. Nanochat makes some policies more explicit and adds FP8 orchestration and a different attention dispatch. Small examples can isolate the numerical effects before any large benchmark.

### 16.1 Compare the actual implementations, not their names

**Gist:** using PyTorch SDPA means you may already benefit from a fused attention backend. The new material is understanding the algorithm behind that interface and what this checkout adds around it.

| Topic | Your implementation | Nanochat |
| --- | --- | --- |
| Forward precision | Autocast context | Explicit activation/weight casts |
| FP32 matmul policy | Calls `set_float32_matmul_precision('high')` on CUDA | Same high-level policy on CUDA |
| Attention API | [SDPA in latent attention](../transformer-from-scratch/model.py#L135) | External FA3 or SDPA wrapper |
| KV representation | Compressed latent/cache components | Full per-layer K/V tensors, GQA supported |
| Linear FP8 | No corresponding path in inspected code | Tensorwise FP8 custom forward/backward |
| Device timing | [CUDA/MPS synchronization](../transformer-from-scratch/train.py#L63) | CUDA synchronization; non-CUDA helper is a no-op |
| Activation memory control | Local microbatching and accumulation | Same principle, with different model/data dimensions |
| Compilation | Commented-out model compile in your loop | Compiled training model and optimizer functions |

Your active attention implementation reconstructs keys/values from latent components; that changes representation and cache costs. FlashAttention changes how a supplied attention problem is evaluated. They are different axes of design and can coexist when the resulting tensors/API are compatible. We will compare their actual decoding costs in the inference note.

### 16.2 Observe rounding and layout directly

**Gist:** a few scalar/tensor operations demonstrate why master storage and layout matter without involving a model or GPU.

In a temporary Python session with PyTorch, try:

```python
import torch

low = torch.tensor(1.0, dtype=torch.bfloat16)
master = torch.tensor(1.0, dtype=torch.float32)
print(low - 1e-4)       # small change can disappear on BF16 storage
print(master - 1e-4)    # fp32 retains it

x = torch.arange(15.0).reshape(3, 5)
column_major = x.t().contiguous().t()
print(x.stride(), x.t().stride(), column_major.stride())
torch.testing.assert_close(x, column_major)
```

The final tensors have equal values but different layouts. Then inspect `torch.finfo` for FP16, BF16, FP32, and the two FP8 formats. Which differences concern range, which concern resolution, and which concern bytes?

For an FP8 exercise, use the source's `_to_fp8` equation on a tensor containing `[1000, 1, 0.001]`. Compare the reconstruction $d_XQ_X$ with the original and identify which entry disappears. This is a cast/scale check, not evidence that the CUDA GEMM is available on that machine.

### 16.3 Reproduce online softmax before studying the GPU implementation

**Gist:** the mathematical streaming rule is short enough to implement directly. Match it against ordinary softmax first, then reason about masking and backward separately.

This CPU example uses finite unmasked scores, so every block contributes valid entries:

```python
import torch

def online_attention_row(scores, values, block_size=2):
    maximum = scores.new_tensor(-float("inf"))
    denominator = scores.new_zeros(())
    numerator = values.new_zeros(values.shape[-1])
    for start in range(0, len(scores), block_size):
        s = scores[start:start + block_size]
        v = values[start:start + block_size]
        next_maximum = torch.maximum(maximum, s.max())
        correction = (maximum - next_maximum).exp()
        unnormalized = (s - next_maximum).exp()
        denominator = correction * denominator + unnormalized.sum()
        numerator = correction * numerator + unnormalized @ v
        maximum = next_maximum
    return numerator / denominator

scores = torch.tensor([0.0, 3.0, -2.0, 5.0], dtype=torch.float64)
values = torch.tensor([[2., 1.], [4., 3.], [6., 5.], [8., 7.]],
                      dtype=torch.float64)
reference = scores.softmax(dim=0) @ values
for block_size in [1, 2, 3, 4]:
    torch.testing.assert_close(
        online_attention_row(scores, values, block_size), reference
    )
```

Questions worth checking before moving on:

1. Why must both the numerator and denominator be rescaled when the maximum increases?
2. If you add 1000 to every score, should the result change? Which naive exponentiation would overflow?
3. How would you skip a fully masked block without evaluating undefined infinity differences?
4. Differentiate dense and online versions with autograd on the same small input. Which exact identity in section 10 explains their matching gradients?
5. Halve the worked example's local microbatch but preserve global batch with accumulation. Which memory terms shrink, and which stay approximately unchanged?
6. Replace the short window with a full window. Is that purely a numerical implementation change, or a different model computation?

## 17. Verification limits, source caveats, and next steps

**Gist:** the equations and memory counts can be checked on small CPU/meta tensors. GPU speed, real FP8 GEMM behavior, BF16 accumulation details, and compiled kernel choices require the actual execution environment. The source contains comments that should not be mistaken for verified measurements or API guarantees.

### 17.1 Implementation claims easy to overread

**Gist:** the table records the distinctions that matter when interpreting flags, comments, and performance logs. The executable path and measured behavior take priority over an informal label.

| Label or comment | Precise interpretation here |
| --- | --- |
| “BF16 training” | Mixed storage/compute paths, with explicit fp32 logits and optimizer arithmetic |
| “FP32 output” | Does not identify all operand/accumulator precisions |
| “FP8 model” | Selected linear operands are quantized; master parameters remain in original storage |
| “FP8 wraps” | Do not model float conversion as modular integer wraparound; explicit clamp defines bounds |
| “FA3 supports FP8” | The wrapper's ordinary selected path is BF16; linear FP8 is separate |
| “No FA3” | Does not uniquely identify SDPA's selected backend |
| “SDPA cannot do local attention” | This wrapper builds a local mask; specialized efficient execution is a separate question |
| “Quarter context: 2048 → 768” | The current expression gives 512 preceding positions |
| “FP8 function is opaque to compilation” | `allow_in_graph` is a Dynamo-level instruction, not an all-stage opacity guarantee |
| “FlashAttention is linear” | Avoids full quadratic HBM intermediate storage; full attention arithmetic remains quadratic |
| “Sharding divides all memory by GPU count” | Parameters/full local gradients remain replicated; temporary allocations also matter |
| “Reported MFU is hardware utilization” | It uses an estimator and BF16 peak; this snapshot's FP8 count also undercounts |

The private `_scaled_mm` and scaler helper interfaces are particularly tied to the installed PyTorch build. This note describes the checked source rather than silently replacing it with a newer library's API.

### 17.2 Checks performed for this note

**Gist:** the checks verify the derivations, source dtype flow, and accounting. They deliberately distinguish simulated FP8 matmul arithmetic from executing a GPU FP8 kernel.

- Format constants and the rounded-away BF16 parameter change were checked with local PyTorch.
- The source's `_to_fp8` was exercised on CPU for both formats, including zeros and a tensor with a large magnitude disparity. `_to_col_major` preserved values with the expected changed strides.
- The actual custom autograd forward/backward was run with `_scaled_mm` replaced in an isolated scratch namespace by a CPU emulation of decoded-operand multiplication. Its three products, saved FP8 operands/scales, gradient dtypes, and shared master parameter matched the stated formulas. This does **not** test the real scaled CUDA kernel or its accumulation error.
- Online attention matched dense attention and the source's SDPA helper in float64 for full causal, local, cached-chunk, and single-query cases. Gradients matched autograd and the manually derived attention backward equations.
- Tiny actual GPT instances ran forward/backward on CPU with FP32, BF16, and FP16 compute selections, confirming embedding/master/gradient dtypes and fp32 loss. This does not exercise the CUDA GradScaler training loop.
- A shapes-only d24 model with the source's dtype conversion and optimizer grouping verified the weight bytes, per-rank history counts, reference window, and tensor-size arithmetic used above.
- The standalone examples, local links, gist paragraphs, and Markdown math delimiters were checked. Function labels use portable notation rather than the GitHub-rejected `operatorname` command discussed earlier.

The numerical environment was CPU PyTorch 2.6.0. No model training run, data download, real multi-GPU job, CUDA/FA3/FP8 timing, or compiled-kernel profiling was performed. Equality checks on tiny tensors establish neither final model quality nor a hardware speedup.

### 17.3 Primary sources and the next stage

**Gist:** these sources document the external numerical and kernel ideas used here. Next we can study evaluation and scaling experiments: how to decide whether changes in loss, capability, and time are meaningful.

- [PyTorch AMP](https://docs.pytorch.org/docs/2.9/amp.html): autocast and loss-scaling semantics.
- [PyTorch float32 matmul precision](https://docs.pytorch.org/docs/2.9/generated/torch.set_float32_matmul_precision.html): internal arithmetic versus output dtype.
- [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433): the two format choices and their tradeoffs.
- [FlashAttention](https://arxiv.org/abs/2205.14135): IO-aware attention with tiling and recomputation.
- [FlashAttention-2](https://arxiv.org/abs/2307.08691) and [FlashAttention-3](https://arxiv.org/abs/2407.08608): later parallelism and hardware-oriented developments.
- [PyTorch SDPA](https://docs.pytorch.org/docs/2.9/generated/torch.nn.functional.scaled_dot_product_attention.html): attention API and backend-selection behavior.
- [torch.compile](https://docs.pytorch.org/docs/2.9/generated/torch.compile.html) and [fine-grained tracing APIs](https://docs.pytorch.org/docs/2.9/torch.compiler_fine_grain_apis.html): graph capture, specialization, and the scope of `allow_in_graph`.

The connection to the earlier notes is now explicit: the model and objective determine a computation; the optimizer defines parameter updates; numerical formats and kernels determine how those computations are approximated and executed; measurements determine whether the combination achieves the intended quality efficiently.

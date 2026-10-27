# Inference in nanochat: from a prompt to cached, sampled, tool-using responses

The earlier notes explained the trained probability model and how its behavior is evaluated. This note follows **how a prompt becomes one or several responses**: prompt processing, KV-cache state, token sampling, stopping, and calculator-result insertion. It also compares that machinery with the latent-cache generation you implemented.

The source is nanochat commit `92d63d4e8bb4df75c3b71618f31ddde2378b2bcd`; comparisons use your `transformer-from-scratch` at `56092a1818027c800f08167ec0c1c65b983711e1`. Exact links refer to those checkouts. Code paths and observed limitations are described without modifying either repository.

**How to skim:** every numbered section and subsection begins with a **Gist**. Sections 1–7 explain cached model computation, 8–11 follow generation decisions and tools, 12–13 quantify memory/work, and 14–16 connect your implementation, the CLI, and the benchmark. The final sections provide small checks and the bridge to conversational training.

One distinction will prevent many indexing errors: **the logit at the last processed token predicts the next token**. Sampling that next token does not itself put its K/V entries into the cache. A subsequent model forward does that. Python generator suspension makes this distinction visible in nanochat's actual loop.

## Contents

1. [What belongs to the model, and what belongs to the engine?](#1-what-belongs-to-the-model-and-what-belongs-to-the-engine)
2. [Autoregressive generation: prefill and decode](#2-autoregressive-generation-prefill-and-decode)
3. [Why caching preserves the causal computation](#3-why-caching-preserves-the-causal-computation)
4. [The KVCache data structure](#4-the-kvcache-data-structure)
5. [Positions, RoPE, and cached attention masks](#5-positions-rope-and-cached-attention-masks)
6. [Smear needs history beyond keys and values](#6-smear-needs-history-beyond-keys-and-values)
7. [One prompt, several sampled responses](#7-one-prompt-several-sampled-responses)
8. [Turning logits into a sampling distribution](#8-turning-logits-into-a-sampling-distribution)
9. [Follow one generator iteration precisely](#9-follow-one-generator-iteration-precisely)
10. [Stopping and completed rows](#10-stopping-and-completed-rows)
11. [Calculator tools and sampled-versus-forced masks](#11-calculator-tools-and-sampled-versus-forced-masks)
12. [Cache memory and capacity](#12-cache-memory-and-capacity)
13. [Inference work and the latency/throughput tradeoff](#13-inference-work-and-the-latencythroughput-tradeoff)
14. [Comparison with your latent-cache implementation](#14-comparison-with-your-latent-cache-implementation)
15. [How the command-line conversation uses the engine](#15-how-the-command-line-conversation-uses-the-engine)
16. [What the inference benchmark actually measures](#16-what-the-inference-benchmark-actually-measures)
17. [Small experiments and checks performed](#17-small-experiments-and-checks-performed)
18. [The bridge to SFT and reinforcement learning](#18-the-bridge-to-sft-and-reinforcement-learning)

## 1. What belongs to the model, and what belongs to the engine?

**Gist:** GPT produces next-token logits from token IDs and optional cache state. The engine decides what token to take, how to manage several response rows, when to stop, and when to insert calculator outputs. Those decisions are part of generation, not learned parameter updates.

### 1.1 Model forward versus generation policy

**Gist:** a forward pass supplies scores; it does not inherently choose a response. Temperature, top-k, stopping, and tools define a policy around those scores and can change behavior without changing weights.

**Code:** [GPT.forward](../nanochat/nanochat/gpt.py#L459), [Engine.generate](../nanochat/nanochat/engine.py#L176).

The model returns logits with shape $(B,T,V)$, where $B$ is sequences in the call, $T$ is newly processed positions, and $V$ is the actual vocabulary size. Its output already includes nanochat's logit softcap. With no targets, no cross-entropy or optimizer update is involved.

The engine takes a nonempty Python list of prompt token IDs. Although the module's introduction calls it purely token-based, the implementation retains a tokenizer: tool-expression decoding, result encoding, and special-token recognition require it.

For this note, let $P$ be prompt length, $R$ the generated-column budget, $L$ transformer layers, $H$ query heads, $H_{kv}$ KV heads, $d_h$ head dimension, $d$ hidden width, and $C$ allocated cache capacity. Here $P$ is a length, not the weight-matrix symbol used in the optimizer note. Cache position $n$ means **how many token positions have already been processed into K/V**.

### 1.2 Evaluation mode is separate from disabling gradients

**Gist:** inference does not need backward graphs, but `.eval()` and inference-mode execution do different jobs. Loading an evaluation model chooses module behavior; the engine's inference-mode decorator removes autograd bookkeeping.

The checkpoint loader uses `phase="eval"` and calls `.eval()`. The engine's generator is decorated with `@torch.inference_mode()`. As [PyTorch documents](https://docs.pytorch.org/docs/2.9/generated/torch.autograd.grad_mode.inference_mode.html), inference mode disables additional autograd-related overhead; it does not itself switch dropout or other modules to evaluation behavior.

Nanochat's present GPT has no dropout, but this distinction matters when comparing against your implementation, whose `generate` explicitly remembers the training mode, switches to eval, and restores it afterward. Neither kind of inference call updates optimizer histories or learned weights.

## 2. Autoregressive generation: prefill and decode

**Gist:** prefill processes the prompt in parallel and prepares the prediction of the first response token. Decode then processes one newly selected token at a time to predict the following token. The serial dependency is between generated tokens, not between every prompt position's matrix multiplication.

### 2.1 What the autoregressive factorization means operationally

**Gist:** each next-token distribution depends on the prompt and response prefix already chosen. Training supplied true continuations; generation feeds back the engine's own chosen or externally forced tokens.

For prompt $x_{0:P-1}$ and response $y_{0:R-1}$, ordinary model continuation probabilities factor as

$$
p_\theta(y_{0:R-1}\mid x)
=\prod_{j=0}^{R-1}p_\theta(y_j\mid x,y_{\lt j}).
$$

The engine may sample from a temperature/top-k-modified distribution instead of exactly $p_\theta$, and forced tool outputs are not sampled from it. We will formalize that policy distinction in sections 8 and 11.

### 2.2 The first response token comes from the prompt forward

**Gist:** processing a prompt of length $P$ produces logits at all prompt positions. The final position predicts response token $y_0$; no extra one-token forward is needed before selecting it.

**Code:** [batch-one prefill](../nanochat/nanochat/engine.py#L194).

The model reads prompt positions $0,\ldots,P-1$, stores their K/V at every layer, and returns its logits. The engine selects `logits[:, -1, :]`. After this prefill, $n=P$.

The source's GPT still computes the output projection for every prompt position before the engine takes the final one. A slice of the returned tensor does not mean the earlier vocabulary projections were skipped. This contributes to prefill computation and temporary logits memory.

### 2.3 Decode keeps a one-token processing/prediction offset

**Gist:** once $y_0$ is selected, forwarding $y_0$ inserts it at cache position $P$ and predicts $y_1$. The cache stores processed inputs, not future sampled tokens automatically.

```text
forward prompt x[0:P] -> cache length P     -> logits for y0
choose y0
forward y0          -> cache length P+1   -> logits for y1
choose y1
forward y1          -> cache length P+2   -> logits for y2
...
```

The first selection therefore requires prompt prefill; each further selection ordinarily requires one new-token forward. The actual Python generator sometimes performs an otherwise unnecessary final forward when resumed for exhaustion; section 9 traces that detail rather than hiding it in this idealized picture.

## 3. Why caching preserves the causal computation

**Gist:** appending future tokens does not change earlier tokens' representations in a causal transformer. We can therefore reuse the keys/values already computed for those positions instead of recomputing their entire forward histories at each generation step.

### 3.1 What exactly is stored?

**Gist:** nanochat caches the key after its position rotation, normalization, and sharpening scale, and the value after its value-embedding mixture. It does not store just raw token embeddings or final hidden states.

**Code:** [attention transformations and cache call](../nanochat/nanochat/gpt.py#L89).

At a layer, write its normalized input at position $t$ as $u_t$. The stored key has the schematic form

$$
k_t=1.2\,N\!\left(\mathcal R_t(W_Ku_t)\right),
$$

where $\mathcal R_t$ is the implemented RoPE rotation and $N$ is per-head RMS normalization. The query is transformed similarly but only the current query is needed for current attention. The stored value is

$$
v_t=W_Vu_t+g_t\odot E^{VE}[x_t]
$$

on value-embedding layers, with the same headwise gate and reshaping defined in the architecture note. Layers without that table use the projection alone.

For current query head $h$, a key position $s$ has score

$$
a_{h,t,s}=q_{h,t}^\top k_{g(h),s}/\sqrt{d_h},
$$

with the causal/window mask applied before softmax. In GQA, $g(h)$ maps each group of query heads to its shared KV head; for equal head counts it is simply $h$.

### 3.2 An induction through layers and positions

**Gist:** if earlier layer inputs are unchanged, their cached projections are unchanged. Causal attention and tokenwise operations propagate that property through every layer, provided positions and extra state are consistent.

At the embedding level, old token IDs and positions do not change. Smear uses only the current and previous token's embedding, so appending a future token does not alter the old mixed embeddings either.

Suppose the old positions' inputs to a transformer layer match the full-prefix computation. Their queries/keys/values and attended sets do not depend on future positions, so their outputs match as well. The tokenwise MLP, normalizations, residual/input mixing, and same-position backout operation preserve this causal property. By induction, old K/V remain valid at all layers.

For the newly appended token, attention over those cached K/V plus its own K/V then reproduces its full-prefix attention. This is an algebraic equivalence, subject to correct masking, positions, and smear state. Different batch shapes/kernel reduction orders can still produce numerical differences.

### 3.3 What the cache saves and what it does not

**Gist:** caching avoids recomputing old positions' projections and MLPs. A new query must still read and attend to the allowed past keys/values; it has not reduced exact full-context attention to a context-independent lookup.

For each new token, the model still computes new embeddings, all layer projections/MLPs, attention against history, and an output-head vector. No previous queries are cached because future attention does not need them.

Caching is also not an optimizer state or saved training activation graph. In inference there is no need to retain the backward computation for these tokens. Cache validity depends on the particular weights, token prefix, positions, and model state that produced it; changing the weights or editing the prefix generally requires rebuilding affected cache contents.

## 4. The KVCache data structure

**Gist:** the cache is a fixed-capacity collection of dense K/V buffers plus position counters and one previous embedding. Processing appends into unused slices. Logical length and allocated capacity are different quantities.

### 4.1 Buffer shapes and views

**Gist:** each layer's keys and values have shape $(B,C,H_{kv},d_h)$. The full cache prepends a layer dimension, and `get_layer_cache` returns views into those existing buffers.

**Code:** [KVCache.__init__](../nanochat/nanochat/engine.py#L92), [get_layer_cache](../nanochat/nanochat/engine.py#L115).

The allocations are

$$
K_{cache},V_{cache}\in\mathbb R^{L\times B\times C\times H_{kv}\times d_h}.
$$

They are zero-initialized in the chosen cache dtype, which the engine sets to `COMPUTE_DTYPE`. Slices for individual layers share that allocation. There is no `torch.cat` to grow these full cache buffers on each token.

`cache_seqlens` is a length-$B$ int32 device tensor. `prev_embedding` starts as `None` and later holds a $(B,1,d)$ normalized pre-smear embedding. `C` is a storage reservation; valid attention reads only the occupied prefix or its permitted window.

### 4.2 Advance the shared position once per model forward

**Gist:** every layer must write a new token at the same absolute position. The cache advances after the final attention layer, not once for each layer.

**Code:** [final-layer advancement](../nanochat/nanochat/gpt.py#L121), [KVCache.advance](../nanochat/nanochat/engine.py#L119).

If $n$ tokens are cached and a call processes $T_{new}$ positions, layer $\ell$ writes

$$
K_{cache}[\ell,:,n:n+T_{new}]=K_{new}^{(\ell)},
\quad
V_{cache}[\ell,:,n:n+T_{new}]=V_{new}^{(\ell)}.
$$

All layers use that same initial $n$. After the final layer's attention, `advance(T_new)` adds to all row counters. Advancing in every layer would multiply the logical token count by depth and misalign the layer histories.

Although the counter is per row, `get_pos()` and the fallback attention read its first entry and assume equal positions across rows. This engine generates several continuations of one prompt in lockstep; it is not a general ragged-position request batcher.

### 4.3 Reset does not have to erase old bytes

**Gist:** resetting logical length to zero and clearing smear state makes old entries invalid. The next prompt overwrites the used prefix, while untouched stale storage is excluded from attention.

**Code:** [KVCache.reset](../nanochat/nanochat/engine.py#L106).

`reset` zeroes the counters and sets `prev_embedding=None`; it does not zero every K/V entry. Correctness comes from the valid-length invariant, not from assuming every unused entry contains zeros. Reading the entire capacity instead of the occupied prefix would therefore be an error even if the initial allocation happened to be zero-filled.

## 5. Positions, RoPE, and cached attention masks

**Gist:** cached inference requires absolute token positions and a mask aligned to the cached prefix. A one-query decode call is particularly easy to get wrong: its only query belongs at the end of the key history, not at the upper-left corner of an ordinary triangular mask.

### 5.1 New tokens use the old cache length as their position offset

**Gist:** RoPE position numbering continues from the prompt. Past keys are already rotated and should not be rerotated at every generation step.

**Code:** [GPT's RoPE slice](../nanochat/nanochat/gpt.py#L467).

The forward obtains $n=\mathrm{kv\_cache.get\_pos()}$ and uses cosine/sine entries $n:n+T_{new}$. Thus the first response token, when forwarded, uses position $P$.

For a consistent rotation convention, pairwise query/key dot products depend on relative position through $\mathcal R_t^\top\mathcal R_s$. Resetting each new query's RoPE position to zero while keeping old rotated keys would destroy those intended relative offsets. The cache stores each key in its already position-dependent form.

### 5.2 Why single-token decode uses noncausal SDPA over valid keys

**Gist:** the cache view contains only past tokens plus the current token. That whole view is valid for its single query, so no additional causal triangle is needed; only any local window must be enforced.

**Code:** [single-query fallback branch](../nanochat/nanochat/flash_attention.py#L90).

[PyTorch SDPA](https://docs.pytorch.org/docs/2.9/generated/torch.nn.functional.scaled_dot_product_attention.html) documents upper-left causal alignment for non-square query/key lengths. With one query and four keys, blindly using `is_causal=True` can keep only the first key rather than all four cached keys.

Nanochat instead takes the valid keys through the current position, slices their last $w+1$ entries for a left window $w$ when needed, then calls SDPA with `is_causal=False`. Nothing in that view is a future key. This matches the distinction you already made in your latent-attention decode path.

### 5.3 Multi-query chunks need an offset mask

**Gist:** for a chunk appended to a cache, later queries may see more of the new chunk than earlier ones. The attention helper handles this with absolute row indices; that does not guarantee every other model operation supports arbitrary nonempty-cache chunking.

**Code:** [explicit offset mask](../nanochat/nanochat/flash_attention.py#L102), [cache insertion in fallback](../nanochat/nanochat/flash_attention.py#L165).

With $n$ old positions, query index $i$ in the new chunk corresponds to absolute position $n+i$. A valid key index $j$ satisfies

$$
j\le n+i,
\qquad
(n+i)-j\le w\quad\text{when a local left window applies}.
$$

For $n=3$ and a two-token chunk, the first new query sees keys 0–3; the second sees 0–4, before any local-window restriction. The helper constructs this mask rather than using a triangle starting at key zero for each chunk.

On the FA3 branch, the external cache API performs insertion/attention; on the fallback branch, Python writes the K/V slices and selects the occupied prefix before calling SDPA. Both are intended to implement the same cached attention. The next section identifies an additional full-model chunk boundary that attention alone does not fix.

## 6. Smear needs history beyond keys and values

**Gist:** nanochat mixes the previous token's normalized embedding into the current one before the transformer. The cache therefore also needs one pre-smear embedding. This state is essential for matching full-prefix logits once the learned smear coefficient is nonzero.

### 6.1 Store the previous normalized embedding, not the previous smeared result

**Gist:** smear is a one-step embedding mixture. Storing the already mixed result would recursively propagate older embeddings and implement a different operation.

**Code:** [cached smear path](../nanochat/nanochat/gpt.py#L482).

Write $e_t=N(E[x_t])$ for the normalized embedding. For $t>0$, the architecture uses

$$
\widetilde e_t=e_t+
\lambda_s\,\sigma(W_s e_t[0:24])\,e_{t-1}.
$$

In single-token decode, `x_pre_smear` retrieves the previous $e$, then `prev_embedding` is updated to the new $e_t$ before the mixture. The transformer receives $\widetilde e_t$. During initial multi-token prefill, ordinary neighboring slices compute the mixture and the last unmixed $e$ is retained for the next call.

### 6.2 A full-model limitation of nonempty-cache multi-token chunks

**Gist:** the intended engine path—empty-cache prefill, then one-token calls—handles smear history. In this checkout, a multi-token call into a nonempty cache omits the old previous embedding for the first token of the new chunk. With active smear, that need not match one full-prefix forward.

The `T > 1` cached branch mixes positions 1 onward using neighbors inside the new chunk, but leaves its first token untouched. It does not use `x_pre_smear` for that boundary. The attention helper's correct chunk mask cannot restore the missing embedding mixture.

This discrepancy was reproduced with a tiny actual GPT, nonzero smear, and nontrivial transformer projections. Initial prefill followed by one-token decode matched the full-prefix logits within float32 tolerance; an additional multi-token chunk did not. At initialization the smear coefficient is zero, so a test that leaves it zero can miss this issue. This note records the limitation without changing the code.

## 7. One prompt, several sampled responses

**Gist:** the engine performs prompt prefill once at batch size one, then copies its cache into several response rows. All rows start with the same next-token distribution but can sample different tokens. This saves duplicate prompt computation, while still duplicating prefix-cache storage.

### 7.1 Expand logits, copy K/V, and preserve smear state

**Gist:** identical first-step logits are shared as an expanded view; cached tensor values are copied into separate response rows. The previous embedding is also replicated so each continuation starts from the correct smear state.

**Code:** [logit expansion](../nanochat/nanochat/engine.py#L206), [decode-cache creation](../nanochat/nanochat/engine.py#L209), [KVCache.prefill](../nanochat/nanochat/engine.py#L123).

After prefill, the $(1,V)$ final logits are expanded to $(B,V)$. Sampling runs on all $B$ rows; it does not sample one first token and broadcast that sampled ID. At positive temperature, independent row draws can differ even though their distributions match. At temperature zero, matching logits choose the same greedy token.

The decode cache allocates $B$ rows and copies the prompt's occupied K/V prefix into each using batch broadcasting. `prev_embedding` is expanded and cloned to $(B,1,d)$. That clone is actual independent row storage, not a writable shared broadcast view. The temporary prefill cache is then deleted.

### 7.2 Shared computation is not shared physical prefix storage

**Gist:** this implementation avoids repeating the prompt forward, but each response owns a dense copy of its prompt cache. Prefix sharing through reference-counted blocks would be an additional serving-system feature.

For a finite generated-column budget $R$, decode capacity is $C=P+R$. With `max_tokens=None`, the code instead reserves `model.config.sequence_len`; that is not automatic unbounded growth. Prompt/cache/rotary limits still have to accommodate the actual calls.

During replication, the batch-one prefill cache and the new batch-$B$ allocation coexist until the former is released. Cache allocation/zeroing and copying therefore contribute to first-token latency and peak memory, even when prompt computation itself ran once.

Dense preallocation is simpler than paged cache management. The [PagedAttention work](https://arxiv.org/abs/2309.06180) studies block-based storage and sharing for serving, but nanochat's `KVCache` is not an implementation of that system. It also does not admit unrelated arriving requests into the batch while other rows decode.

## 8. Turning logits into a sampling distribution

**Gist:** sampling is a policy on the model's logits. Temperature changes relative probabilities, top-k restricts support, and the zero-temperature branch chooses an argmax. These policies change response statistics without altering the model parameters.

### 8.1 Temperature is a log-probability scale

**Gist:** lower positive temperature concentrates probability on larger logits; higher temperature spreads it out. It is the Boltzmann form with energies equal to negative logits, not an adjustment to the trained weights.

**Code:** [sample_next_token](../nanochat/nanochat/engine.py#L141).

For $\tau>0$ and logits $z_v$,

$$
q_\tau(v)=\frac{e^{z_v/\tau}}{\sum_u e^{z_u/\tau}},
\qquad
\frac{q_\tau(i)}{q_\tau(j)}=e^{(z_i-z_j)/\tau}.
$$

A logit gap of 2 gives odds $e^2\approx7.39$ at $\tau=1$, but $e^4\approx54.6$ at $\tau=0.5$. For fixed support, its entropy obeys

$$
\frac{dH(q_\tau)}{d\tau}
=\frac{\mathrm{Var}_{q_\tau}(z)}{\tau^3}\ge0.
$$

This follows by differentiating $H=\ln Z-\mathbb E[z]/\tau$, using $d\mathbb E[z]/d\tau=-\mathrm{Var}(z)/\tau^2$. Greater entropy means less concentration, not guaranteed creativity, truthfulness, or correctness.

At `temperature == 0`, the code bypasses softmax and directly returns `argmax`. This avoids dividing by zero. With tied maximum logits, this explicit tie decision need not equal the full limiting distribution of a positive-temperature softmax.

### 8.2 Top-k truncates the available choices

**Gist:** keep exactly the selected k highest-logit IDs, normalize probabilities within that set, and sample there. It is not beam search or nucleus/top-p sampling.

For the selected set $S_k$,

$$
q_{\tau,k}(v)=
\begin{cases}
e^{z_v/\tau}/\sum_{u\in S_k}e^{z_u/\tau},&v\in S_k,\\
0,&v\notin S_k.
\end{cases}
$$

The engine uses `torch.topk`, samples an index in that compact result, then gathers its original vocabulary ID. `top_k=None` or a nonpositive value disables truncation; an oversized value is capped at vocabulary size. At temperature zero the early argmax branch ignores top-k entirely.

The simpler [GPT.generate](../nanochat/nanochat/gpt.py#L544) instead threshold-masks logits below the kth-largest value. Ties at the boundary can retain more than k IDs there, whereas the engine's compact top-k returns k selected IDs. Even mathematically similar sampling implementations need not generate identical seeded streams when their support ordering or random-number consumption differs.

### 8.3 Seeds repeat a stream under a specified execution

**Gist:** `Engine.generate` creates a fresh generator seeded by its argument, default 42. Repeating a matching call can repeat samples, but changing batch size, forced-token behavior, numerical logits, or device can change the resulting stream.

The code invokes the sampler for every row on every column, including rows whose next token will be forced and rows already marked complete while others continue. Discarding a sampled token does not undo the random draw. The future generator state can therefore depend on bookkeeping outside the visibly sampled text.

A reproducible seed is useful for comparisons, but is not a guarantee that different implementations with the same abstract distribution choose identical samples. Compare probabilities/logits as well as token sequences when diagnosing a caching or sampling discrepancy.

## 9. Follow one generator iteration precisely

**Gist:** the generator yields a chosen token column before forwarding that column into the model. Resuming it performs the next-token forward. This ordering explains cache position, streaming latency, and an otherwise unnecessary final computation on exhaustion.

### 9.1 From logits to a yielded column

**Gist:** the current logits predict the next column; row state may replace sampled IDs with queued tool-result IDs before the column is exposed to the caller.

**Code:** [sampler and row loop](../nanochat/nanochat/engine.py#L234), [yield and subsequent forward](../nanochat/nanochat/engine.py#L269).

The loop:

1. Checks the column budget and whether all rows have completed.
2. Samples a candidate ID for each row.
3. Pops a queued forced ID if present, otherwise uses the sampled ID.
4. Records a sampled/forced mask, updates row history, stop state, and calculator state.
5. Yields `(token_column, token_masks)`.
6. On resumption, increments the generated-column counter and forwards the chosen column with shape $(B,1)$ to prepare new logits.

The conversion `next_ids[:,0].tolist()` brings IDs into Python; constructing the next input tensor sends the chosen column back to the device. This state-machine design has CPU/device coordination costs, unlike an entirely device-resident decode loop.

### 9.2 Cache state at the suspension point

**Gist:** immediately after the jth yielded response token, the cache may not yet contain that token. The token has been selected, but the generator's subsequent model call has not run.

For one row without forced tokens:

| Caller action | Yielded token | Cache length at that yield |
| --- | --- | ---: |
| First `next(gen)` | $y_0$ | $P$ |
| Second `next(gen)` | $y_1$ | $P+1$ |
| Third `next(gen)` | $y_2$ | $P+2$ |

This is not a missing-cache bug. The cache only needs $y_j$ when preparing the distribution of $y_{j+1}$. A consumer that stops reading a generator need not advance its cache further.

### 9.3 Exhaustion can perform one extra forward

**Gist:** in this source, the budget/all-complete checks are at the next loop entry. Resuming the final yield runs its following model forward before reaching those checks, even though no further output may be needed.

If a consumer fully exhausts a raw generator with budget $R$, the source can execute one prompt prefill plus $R$ one-token forwards while yielding only $R$ columns. Ideally only $R-1$ one-token forwards are needed to select those $R$ tokens. This was verified by recording calls to a scripted mock model.

`generate_batch` breaks immediately when its wrapper sees all rows terminate, so it can avoid resuming that terminal yield. When termination is only from the budget, it normally resumes to discover exhaustion and pays the extra forward. This detail matters for benchmark accounting later; it does not require an extra output token or change the returned maximum length.

## 10. Stopping and completed rows

**Gist:** the raw stream marks rows complete on two special tokens, but does not compact or freeze them while other rows continue. The convenience wrapper removes terminal tokens and ignores subsequent output from completed rows. These are distinct behaviors.

### 10.1 What stops a response?

**Gist:** this engine recognizes `<|assistant_end|>` and `<|bos|>` as row terminators, and `max_tokens` as a batch-column budget. Other special token names do not automatically stop generation.

**Code:** [row completion test](../nanochat/nanochat/engine.py#L248).

The raw generator includes the terminal token in its yielded column and sets that row's `completed=True`. Generation stops when all rows are complete or the budget is reached, subject to the suspension behavior just described.

This is not a string-level search for a stop substring. It operates on token IDs. The budget counts generated columns, including tool-result tokens and terminators when emitted, rather than promising R visible natural-language tokens per row.

### 10.2 Completed rows still occupy the batch

**Gist:** a finished row is not removed from the model batch. It can receive further sampled/forced IDs in the raw stream while another row remains active, consuming computation and random draws.

The per-row processing loop does not skip completed rows. Their cache positions continue advancing with the batch, and their row histories can keep growing internally. A raw-stream consumer must respect its own per-row stop state rather than displaying every subsequent ID as part of the answer.

This lockstep design keeps shapes and positions simple, but differs from a serving scheduler that removes finished requests and inserts new ones. The unused work can also make a nominal batch-token throughput larger than useful completed-response throughput.

### 10.3 What generate_batch returns

**Gist:** the wrapper returns full prompt-plus-continuation token lists and parallel masks, with each continuation cut before its first terminal token. Consumers must remove the prompt prefix themselves when they want answer-only text.

**Code:** [Engine.generate_batch](../nanochat/nanochat/engine.py#L277).

The wrapper starts each result as a copy of the prompt and each prompt mask as zeros. For active rows it appends nonterminal tokens and their masks; terminal IDs are not appended. Later raw columns for a completed row are ignored. It returns exactly one result list per requested sample.

This convention is why the chat evaluator slices away `len(encoded_prompt)` before decoding the completion. It also means returned masks contain no assistant-end terminal position, even if that terminal was sampled in the raw stream. Do not assume a returned rollout contains every policy decision made internally.

## 11. Calculator tools and sampled-versus-forced masks

**Gist:** the model learns to request calculations using token delimiters; the engine executes the supported expression and forces the result into the sequence. The mask distinguishes model-selected actions from external information, a distinction that will matter for RL credit assignment.

### 11.1 Recognize a call and queue its result

**Gist:** the engine collects expression tokens between Python-start/end markers. On a successful supported calculation, it queues output-start, encoded result text, and output-end for later columns.

**Code:** [tool state machine](../nanochat/nanochat/engine.py#L252), [use_calculator](../nanochat/nanochat/engine.py#L46).

At `<|python_start|>`, the row enters expression-collection mode. At `<|python_end|>`, it decodes the collected tokens and calls `use_calculator`. A non-`None` result produces the queue

```text
<|output_start|>  tokenizer.encode(str(result))  <|output_end|>
```

The next loop iterations pop one forced token at a time. Each is forwarded into the same cache, so subsequent model predictions are conditioned on the actual calculator output. On a failed/rejected calculation, no such result sequence is queued.

Despite the token names, this is a restricted calculator, not a general persistent Python REPL. The source allows selected arithmetic and string `.count()` expressions, removes commas, rejects `**` and listed patterns, and uses an `eval` helper with empty builtins and a signal-based timeout. It retains no user variables between calls. Its timeout mechanism also has the platform/thread assumptions of `SIGALRM`; the token convention alone does not establish a general tool-execution service.

### 11.2 Trace a request, result, and resumed answer

**Gist:** calculator results are inserted as context rather than chosen by the language model. They consume generation columns and update the cache just like other incoming tokens, but their masks are zero.

For a byte-tokenizer illustration where expression `2+3` has three tokens and result `5` one token:

| Stream portion | Origin | Mask per token |
| --- | --- | --- |
| `<|python_start|>` | Sampled | 1 |
| `2`, `+`, `3` | Sampled expression | 1,1,1 |
| `<|python_end|>` | Sampled | 1 |
| `<|output_start|>` | Forced | 0 |
| `5` | Calculator result, forced | 0 |
| `<|output_end|>` | Forced | 0 |
| Further answer tokens | Sampled | 1 |

Real BPE may segment the ordinary text differently; the special markers are individual designated IDs. The raw sampler still draws candidates during the forced columns and discards them, as section 8 noted. The forced result is not sampled merely because an unused candidate was drawn first.

### 11.3 Probability of a rollout is not the product of model probabilities everywhere

**Gist:** sampled positions follow the chosen decoding distribution, while external-result positions follow a deterministic insertion rule given tool output. Treating forced text as sampled model actions would assign the model credit for a computation supplied by the environment.

Let $m_t\in\{0,1\}$ mark sampled positions. The actual transition policy is schematically

$$
q(a_t\mid h_t)=
\begin{cases}
q_{\tau,k}(a_t\mid h_t),&m_t=1,\\
\mathbf1[a_t=f(h_t)],&m_t=0,
\end{cases}
$$

where $f$ is the queued external-token rule. For valid forced transitions, their log probability under this insertion rule is zero. The model's probability of that same token need not be one; that is a different probability.

The engine returns IDs and masks, not a complete table of sampling log probabilities. Later RL code will decide which positions to score and how its scoring policy relates to temperature/top-k generation. For now, preserve the distinction between response tokens, model actions, forced observations, and stripped terminal tokens.

## 12. Cache memory and capacity

**Gist:** cache storage grows with layers, response rows, reserved sequence capacity, KV heads, head dimension, and dtype bytes. Local attention can reduce reads without automatically shrinking this dense allocation. Capacity is also distinct from the range on which a checkpoint was trained.

### 12.1 Derive the cache-byte formula

**Gist:** count two arrays—keys and values—of the same shape. Query-head count affects attention work, but KV-head count controls the principal cache storage.

For $p$ bytes per cache entry,

$$
\boxed{M_{KV}=2LBCH_{kv}d_h p}.
$$

There are also $4B$ bytes of int32 counters and, once populated, about $Bdp$ bytes for `prev_embedding`, excluding object/allocator overhead. Prompt and decode caches coexist briefly during copying, so peak allocation can exceed the final decode-cache formula alone.

For d24 with $H_{kv}=12$, $d_h=128$, BF16 ($p=2$), and one row, each cached token costs

$$
2\times24\times12\times128\times2
=147{,}456\text{ bytes}=144\text{ KiB}.
$$

At capacity $C=2048$ that is 288 MiB per row, or 4.5 GiB for 16 rows. If $P=1024$ and $R=256$, engine capacity is 1280, giving 180 MiB per row; the temporary batch-one prefill cache adds another 144 MiB while it coexists. These are tensor-storage sizes, not measured device peaks.

GQA with three KV heads instead of twelve would divide the principal cache term by four for an appropriately configured model. Nanochat supports that architecture, but its base-training builder currently sets equal query/KV head counts. The [GQA paper](https://arxiv.org/abs/2305.13245) explains the shared-KV design; it does not make existing full-head checkpoints interchangeable without an architectural/training change.

### 12.2 Storage, attention window, and valid positions are separate limits

**Gist:** an allocated slot permits storing a token, a window determines which slots a query reads, and RoPE buffers determine available position data. None of these alone guarantees good behavior beyond the trained context.

The cache allocates its full capacity even for short-window layers. It is not a ring buffer that discards those layers' oldest slots. Full-context-pattern layers also retain their configured longer windows.

For a new call, $n+T_{new}$ must fit the cache and the rotary tables. GPT precomputes a rotary table longer than its training sequence length, but that is an implementation buffer, not a demonstration of reliable context extrapolation. Its forward checks the current chunk length in one assertion; it is not a complete central policy for growing or truncating arbitrary conversations.

The engine does not automatically crop long prompts or allocate an unbounded stream. With finite $R$ it reserves $P+R$; with no specified budget it uses the model configuration's capacity. Reusing a larger allocation does not alter the attention windows or retrain the model for longer dependencies.

## 13. Inference work and the latency/throughput tradeoff

**Gist:** prefill has many token vectors to process in parallel; decode typically has only one new vector per response row. KV caching saves repeated history computation but still requires reading allowed past K/V. Batching improves weight reuse at the cost of cache memory and potentially latency.

### 13.1 Compare full-prefix recomputation with cached work

**Gist:** without caching, every generation iteration repeats old projections/MLPs. With caching, those dense layers run on the prompt once and then only newly processed tokens. Attention against history still grows with the relevant context/window.

Let $N_{mat}$ count the learned projection weights, as in the pretraining note. Ignoring elementwise work, one token's forward projections cost about $2N_{mat}$ FLOPs. Selecting $R$ tokens by full-prefix recomputation processes

$$
\sum_{j=0}^{R-1}(P+j)=RP+\frac{R(R-1)}2
$$

token rows through those layers. The ideal cached path processes $P+R-1$ such rows; source-level generator exhaustion can add one extra row as already explained.

For a decode call whose occupied key count after insertion is $c$, let $a_\ell(c)$ be keys attended per query at layer $\ell$. Attention's two products cost approximately

$$
C_{attn,decode}\approx4Hd_h\sum_{\ell=0}^{L-1}a_\ell(c),
\qquad a_\ell(c)=\min(c,w_\ell+1)
$$

for a finite left window $w_\ell$. An unlimited layer would use $a_\ell(c)=c$. Thus one-token decode is not constant work in history length for full attention.

For a prompt prefill, count causal pairs instead:

$$
C_{prefill}\approx2PN_{mat}
+4Hd_h\sum_\ell\sum_{t=0}^{P-1}\min(t+1,w_\ell+1).
$$

These are arithmetic estimates, not runtime predictions. Softmax, normalization, casts, cache writes, sampling, and the state machine add costs.

### 13.2 Why small-batch decode is often bandwidth-sensitive

**Gist:** a single new token reuses a dense weight matrix far fewer times than a large training/prefill batch. More response rows share the same matrix reads within larger products, increasing arithmetic intensity until other limits dominate.

For $B$ current token vectors and a large dense matrix with $N_w$ weights, arithmetic is about $2BN_w$. If its dominant traffic is reading those weights once at $p_w$ bytes each, the idealized intensity is approximately $2B/p_w$ FLOPs/byte. This is a model of one regime, not a guarantee about every kernel/cache level.

K/V traffic grows with active context and rows, while weights can be reused across rows. Increasing $B$ can improve aggregate tokens/sec without proportionally improving a single user's inter-token latency. Larger batches also reserve more cache memory and do more work after some rows have completed in this particular engine.

As in the efficiency note, compare a compute-bound and bandwidth-bound estimate rather than assuming one applies at every batch size. At sufficient batch size, arithmetic, cache traffic, sampling, or host coordination can become limiting instead.

### 13.3 Cost helpers are approximate accounting functions

**Gist:** the model exposes convenient FLOP and cache-byte estimates for the benchmark. They are structural estimates with simplified window counting, not device-traffic measurements or exact latency bounds.

**Code:** [estimate_decode_flops](../nanochat/nanochat/gpt.py#L351), [estimate_prefill_flops](../nanochat/nanochat/gpt.py#L362), [kv_bytes_per_token](../nanochat/nanochat/gpt.py#L374), [kv_read_bytes](../nanochat/nanochat/gpt.py#L380).

The source uses `min(context_len, window)` for attention reads/work, whereas a left window includes the current key as well. Its prefill ramp likewise uses the simplified window count. These are small boundary differences for large windows, but the exact mask equations remain the model definition.

The helper formulas assume the builder's positive windows; directly experimenting with a negative unlimited-window sentinel would need corresponding estimator handling. Precision or module-type changes also require auditing the structural projection count, including the FP8 counting issue described in the pretraining note.

## 14. Comparison with your latent-cache implementation

**Gist:** you already implemented the main prefill/one-token-decode pattern. Your cache compresses history into normalized latent vectors plus shared rotary keys; nanochat stores the full transformed K/V for its chosen KV heads. This trades reconstruction work against stored history size.

### 14.1 What your cache actually contains

**Gist:** your attention stores $(d_{latent}+d_{rope})$ entries per position per layer, then reconstructs ordinary per-head keys/values from the cached latents during each forward. Nanochat stores $2H_{kv}d_h$ entries and directly attends to them.

**Code:** your [latent/rotary cache updates](../transformer-from-scratch/model.py#L96), [value and key reconstruction](../transformer-from-scratch/model.py#L115), and [generate](../transformer-from-scratch/model.py#L325).

In your active configuration, $d=512$, $H=16$, $d_h=32$, $d_{latent}=128$, and $d_{rope}=8$. Your principal cache stores $128+8=136$ entries per position/layer. A hypothetical ordinary full-K/V MHA cache with the same $H,d_h$ would store $2\times16\times32=1024$ entries, about 7.53 times as many. This compares storage representations at those dimensions, not the quality or total runtime of the two repositories' differently sized models.

Your latent cache grows with `torch.cat` and is cropped to `block_size`; nanochat preallocates fixed-capacity dense buffers and writes slices. Smaller stored representations do not by themselves guarantee faster inference, because reconstruction, layout changes, and allocations also matter.

### 14.2 Where reconstruction can be moved algebraically

**Gist:** low-rank key/value reconstruction admits useful linear-algebra identities. Your implementation explicitly reconstructs history; absorbing those maps into current-query/output computation would be an additional optimization, not something the present code already does.

For one head, let cached latent be $c_s$, non-rotary key $k_s=A_hc_s$, and value $v_s=B_hc_s$. Then

$$
q_h^\top k_s=q_h^\top A_hc_s
=(A_h^\top q_h)^\top c_s,
$$

and, after attention weights $\alpha_s$ are known,

$$
\sum_s\alpha_sv_s
=B_h\left(\sum_s\alpha_sc_s\right).
$$

This can avoid rebuilding every historical non-rotary key/value in the same way. The separate rotary-key term must still be handled, and changing the internal dot-product dimension must preserve the original intended attention scale rather than blindly using a new SDPA default.

The [DeepSeek-V2 report](https://arxiv.org/abs/2405.04434) describes multi-head latent attention as a cache-compression design. Your source already implements latent storage, but its explicit `self.v(latent)` and `self.k_nope(latent)` over the cached sequence show that these absorption identities are not yet its executed decode path.

### 14.3 Context cropping and whole-model equivalence need care

**Gist:** a cache of historical representations is not generally identical to recomputing a cropped text window from scratch. Your MoE routing can add another batch-partition dependence, so test attention caching separately from broader model changes.

Old cached states may already encode information from tokens later removed by cropping. Recomputing a suffix from scratch removes that upstream history at every layer. Your no-cache generation also crops/reset-recomputes prompt positions, whereas cached decode keeps its position counter advancing. Even where RoPE relative shifts can be handled consistently, the retained deeper-layer states make the two procedures different beyond a cropping boundary.

Within a supported context, compare cached and full-prefix logits using identical inputs and eval behavior. Your MoE capacity/drop decisions can depend on whether a full sequence or one token is processed at a time; that can change the computation independently of the KV cache. A failed whole-model equivalence check should therefore be localized before blaming RoPE or cache indexing.

Your sampler also directly divides by temperature and samples from softmax, without nanochat's dedicated temperature-zero branch or per-call seeded generator. Its top-k thresholding handles ties differently from the compact engine sampler. These are additional reasons not to expect bitwise-identical generated text solely from sharing a cache concept.

## 15. How the command-line conversation uses the engine

**Gist:** the CLI preserves conversation history as token IDs, wraps each new user turn in role markers, and invokes a fresh engine generation. It does not keep one growing KV-cache object across all turns of the interactive session.

### 15.1 Construct the next assistant prefix

**Gist:** conversation formatting tells the model which speaker should continue. Those marker IDs are learned parts of the input, not a separate optimization or reasoning algorithm.

**Code:** [conversation state](../nanochat/scripts/chat_cli.py#L43), [user/assistant markers](../nanochat/scripts/chat_cli.py#L71).

The token history starts with BOS. Each new turn appends

```text
<|user_start|> encode(user_text) <|user_end|> <|assistant_start|>
```

The engine then prefills the entire history and generates with one sample, a 256-column budget, and the CLI's temperature/top-k settings. Default temperature is 0.6 and top-k 50 in this script, unlike the greedy default of the benchmark.

The CLI consumes the raw generator, prints decoded tokens, and stores the emitted response. If it did not end in assistant-end, it appends that marker before adding the response to the conversation history. `clear` resets the token list to BOS. None of that changes learned parameters.

### 15.2 Persistent text history is different from persistent cache reuse

**Gist:** each `Engine.generate` allocates and prefills anew, even when the CLI already has the previous conversation's token prefix. Reusing a session cache would need a longer-lived state/API and careful agreement about which tokens were actually processed.

That distinction follows directly from cache allocation inside `generate`, rather than from a method returning a reusable session cache. It is particularly relevant because the final selected/emitted token can be one position ahead of the cached inputs at a generator suspension point.

The CLI's per-token display also decodes individual IDs. A byte-level token may contain only part of a UTF-8 character, so robust display streaming can need byte buffering even when token-ID generation itself is correct. That is a text presentation concern, separate from the attention cache and probability policy.

## 16. What the inference benchmark actually measures

**Gist:** the benchmark times first-token and subsequent-column latency while sweeping response batch size. Its outputs include useful structural estimates, but also account for this particular generator's allocations, copying, host coordination, stopping, and approximations. They are not measurements of a production request scheduler.

### 16.1 First-token latency and later decode time

**Gist:** the first `next(generator)` includes batch-one prefill, cache replication, and selecting the first column. Later successful `next()` calls include the previous column's forward plus selecting the next one.

**Code:** [bench_generate](../nanochat/scripts/infer_bench.py#L55), [batch sweep metrics](../nanochat/scripts/infer_bench.py#L209).

The benchmark is CUDA-only and asserts one process. It synchronizes around timed completed work, performs warmup calls, and records peak allocated CUDA memory. TTFT is time to first yielded column; it excludes loading/tokenizing the model prompt outside the measured generator, but includes internal allocations/copies and sampling.

For $n_{steps}$ successful later columns with durations $\Delta t_j$, it reports

$$
\mathrm{TPOT}=\text{the middle sorted step duration},\qquad
\mathrm{tokens/sec}=\frac{B\,n_{steps}}{\sum_j\Delta t_j}.
$$

The source uses the upper middle element for an even count, rather than averaging both middle values. With R columns and no early stop, normally $n_{steps}=R-1$ because the first column belongs to TTFT.

### 16.2 Bandwidth and compute utilization use estimated work

**Gist:** MBU and MFU divide estimated bytes or FLOPs per decode step by measured time and a tabulated peak. The estimates are useful comparisons, not direct counters of device traffic or universally valid physical ceilings.

The script's bandwidth estimate is

$$
B_{step,code}=M_{all\ stored\ weights}
+B\,M_{KV\ reads}(c_{mid}),
$$

with representative context $c_{mid}=P+\lfloor R/2\rfloor$. It reports

$$
\mathrm{MBU}(\%)=100\frac{B_{step,code}/\mathrm{TPOT}}{\mathrm{peak\ bandwidth}},
$$

and a similarly estimated decode MFU using $B$ times per-token forward FLOPs and peak BF16 compute.

The all-weight byte sum includes entire token/value embedding tables, but decoding performs indexed row lookups rather than reading every table entry each step. Hardware caches, repeated rows, casts, intermediates, and actual kernel traffic further complicate the model. Thus its “theoretical ceiling” label is a heuristic estimate for this accounting model, not a rigorous upper bound applying to every implementation.

Likewise, its estimated maximum rows fitting in memory subtracts weights and divides the rest by full-context cache bytes. It omits other live allocations and transient replication, so it is not a guarantee that that many engine rows will fit.

### 16.3 Stopping and coroutine details affect the interpretation

**Gist:** raw engine steps can include completed rows, and the final call that discovers exhaustion can perform extra work outside the list of successful step timings. Read latency/throughput results with that source behavior in mind.

`bench_generate` times successful yielded columns and breaks on `StopIteration`. The source's final-resume forward can run on that failing call before exhaustion is detected, while its duration is not appended to `step_times`. The GPU timing/queue interaction therefore need not match timing a minimal R-token decoder end to end.

It also computes nominal throughput as $B$ tokens per successful column even after some rows have already terminated internally. That can differ from useful nonterminal output throughput. Early-stop lengths and the distribution of response lengths should accompany comparisons.

The benchmark clamps prompt length so prompt plus requested decode fits `config.sequence_len`; that is a benchmark policy, not an engine-wide truncation guarantee. It loads the ordinary uncompiled model path in this script. A stale source help string mentions `mid`, but the inspected checkpoint loader maps `base`, `sft`, and `rl` sources. Benchmarking a different compiled or serving path would be a separate measurement.

## 17. Small experiments and checks performed

**Gist:** cache correctness is best tested before comparing sampled sentences: compare logits with identical weights/prefixes and check state at every step. Small CPU models can exercise the indexing, masks, and generator protocol without a trained checkpoint or GPU benchmark.

### 17.1 Isolate the one-query attention mask

**Gist:** fixed Q/K/V tensors let you verify the cached attention equation without involving embedding, RoPE, or model weights. This is a useful first check when a full-model cache result differs.

This standalone CPU example selects the fifth query from a full causal attention calculation and recomputes it against just its valid key prefix:

```python
import torch
import torch.nn.functional as F

gen = torch.Generator().manual_seed(7)
q = torch.randn(1, 2, 7, 4, generator=gen, dtype=torch.float64)
k = torch.randn(1, 2, 7, 4, generator=gen, dtype=torch.float64)
v = torch.randn(1, 2, 7, 4, generator=gen, dtype=torch.float64)
full = F.scaled_dot_product_attention(q, k, v, is_causal=True)
position = 4
cached_row = F.scaled_dot_product_attention(
    q[:, :, position:position+1],
    k[:, :, :position+1],
    v[:, :, :position+1],
    is_causal=False,
)
torch.testing.assert_close(cached_row, full[:, :, position:position+1])
```

Then deliberately set the last flag to `True` and inspect which keys remain allowed. Add a left-window restriction by slicing the key prefix to its last $w+1$ entries. This tests attention's row alignment, not the full model's cache-validity induction by itself.

### 17.2 Questions that expose the important state boundaries

**Gist:** if you can predict cache length, token origin, and stop behavior at each boundary, the core engine is understandable without memorizing its implementation.

1. For $P=4$ and $R=3$, list the inputs to the prefill and each one-token forward. At each yield, is its newly selected token already cached? What extra call can occur on full exhaustion?
2. For $B=8$ continuations, which prompt work is performed once, and which memory is copied eight times? What changes if the first token is greedy rather than sampled?
3. If a row emits assistant-end before another row, compare its subsequent raw stream, returned `generate_batch` list, and device computation.
4. For a successful calculator call whose encoded result has five tokens, count forced output-marker/result columns and identify their masks. Does drawing unused candidates on those columns affect future RNG state?
5. Explain why cache reset can be correct without overwriting all the old keys and values.
6. Derive the cache-byte formula for a GQA configuration with $H=12$ and $H_{kv}=3$. Which attention operations still scale with the number of query heads?
7. Use a nonzero smear coefficient when comparing full-prefix logits with cached decode. Then try a two-token chunk into a nonempty cache and identify the missing boundary term.

The repository's [Engine tests](../nanochat/tests/test_engine.py#L1) and [attention fallback tests](../nanochat/tests/test_attention_fallback.py#L1) provide existing cases to read. Their mock/dtype/device scope matters: a mock state-machine test and a real-model numerical test establish different things.

### 17.3 What was checked while preparing this note

**Gist:** the checks cover real small-model cache numerics, the source's state machine, sampling formulas, and storage arithmetic. They do not measure GPU speed, learned response quality, or external FA3 correctness.

- A tiny actual GPT with GQA, explicit local/full windows, nonzero smear, and nonzero attention/MLP output projections matched full-prefix logits under initial prefill plus one-token cached decoding. The largest absolute difference in the checked float32 example was approximately $2\times10^{-9}$.
- The same model reproduced the nonempty-cache multi-token smear-boundary discrepancy. This was treated as an observed limitation, not silently repaired.
- Cache replication copied K/V and independent smear state correctly. Reset left backing bytes unchanged but excluded stale tails from new-prompt attention.
- The source SDPA helper matched explicit offset/window masks, and a one-query example demonstrated the incorrect upper-left causal-mask alternative.
- A scripted model recorded generator call order and cache position at suspension, the otherwise-unused final forward, completed-row raw output, wrapper trimming, and exclusion of terminal tokens.
- A controlled `2+3` call exercised the actual calculator helper, forced queue, masks, discarded draws, and resumed answer. No arbitrary user/generated program was executed.
- Greedy/top-k support, same-seed repeatability in a fixed setup, expanded first-token sampling, and the temperature-entropy identity were checked. All eight existing Engine mock-test functions were also executed against isolated source definitions, rather than represented as a full package/pytest/GPU test run.
- Your actual latent-attention component matched its full-prefix result under cached decoding within its supported context, separately from the MoE model. The latent projection identities and memory examples were verified numerically.
- The standalone snippet, local code links, gist paragraphs, and Markdown math were checked with the portable notation used in the earlier notes.

These checks used CPU PyTorch 2.6.0 with the SDPA path and small initialized/synthetic models. No trained checkpoint, dataset download, CUDA/FA3 run, real serving scheduler, or inference throughput benchmark was used. Tiny equivalence checks do not establish exact equality across devices/kernels or long-context model quality.

## 18. The bridge to SFT and reinforcement learning

**Gist:** inference now explains how a checkpoint produces actions and observations. SFT teaches which conversational continuations to imitate; RL later generates rollouts with this machinery and assigns learning signals to selected positions.

### 18.1 What generation machinery does not teach by itself

**Gist:** adding an engine, role markers, or a calculator does not by itself train a base model to use them well. The model must have learned suitable conversational and tool-call behavior from its data/objectives.

The engine can recognize a valid tool token sequence and supply a result, but choosing a useful calculation, using that result, and ending the response are model behaviors. The next note will follow conversation rendering, assistant-only supervision, mixtures, masks, optimizer initialization, and the actual SFT training loop.

For RL, the distinctions already established become essential: sampled versus forced tokens, termination tokens omitted by the wrapper, temperature/top-k-modified behavior, verifier reward, and the log probabilities used by the training loss. We will derive the actual implemented policy-gradient objective before equating it with any named algorithm.

### 18.2 Primary sources and extensions beyond this engine

**Gist:** the references describe related cache and execution ideas; they are not all features present in nanochat. Keep a clear boundary between understanding this compact engine and studying a larger serving system.

- [PyTorch inference mode](https://docs.pytorch.org/docs/2.9/generated/torch.autograd.grad_mode.inference_mode.html): autograd execution context versus module evaluation mode.
- [PyTorch SDPA](https://docs.pytorch.org/docs/2.9/generated/torch.nn.functional.scaled_dot_product_attention.html): non-square causal alignment and attention API semantics.
- [GQA](https://arxiv.org/abs/2305.13245): shared KV heads and their inference motivation.
- [DeepSeek-V2](https://arxiv.org/abs/2405.04434): multi-head latent attention and cache compression, relevant to your representation.
- [PagedAttention](https://arxiv.org/abs/2309.06180): an extension to block-managed/shared cache storage for serving; this note's dense-cache engine does not implement it.

The complete chain is now concrete: **format a prompt → prefill its causal state → select or force a token → process it at the correct position → update cache and row state → continue until the protocol stops**. The next stage explains how training changes the probabilities along that chain to make the model behave like an assistant.

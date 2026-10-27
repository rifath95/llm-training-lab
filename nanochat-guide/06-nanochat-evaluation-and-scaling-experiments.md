# Evaluating nanochat: prediction loss, task scores, and scaling experiments

The previous notes explained how to build, train, optimize, and execute the model. This note asks **how we determine what the trained model learned, whether a change helped, and how to allocate a training budget**. We will follow both the evaluation code and the experiments that motivate nanochat's scaling rules.

The source snapshot is nanochat commit `92d63d4e8bb4df75c3b71618f31ddde2378b2bcd`, compared with your `transformer-from-scratch` at `56092a1818027c800f08167ec0c1c65b983711e1`. Exact source links refer to those checkouts. Mathematical examples labelled illustrative are constructed examples, not measured nanochat results.

**How to skim:** each numbered section and subsection starts with a **Gist** paragraph. Sections 1–6 explain base-model evaluation; sections 7–10 explain conversational evaluation; sections 11–12 explain uncertainty and generalization; sections 13–16 explain experimental comparisons and scaling analysis. The final sections connect to your implementation and offer small checks.

Evaluation is a **protocol**, not just a formula: checkpoint, tokenizer, data, prompt, scoring rule, sampling budget, and verifier jointly define the result. Two numbers with the same dataset name can measure different things when those choices differ.

## Contents

1. [What question is each evaluation answering?](#1-what-question-is-each-evaluation-answering)
2. [Bits per byte: the exact held-out prediction metric](#2-bits-per-byte-the-exact-held-out-prediction-metric)
3. [Where evaluation data and checkpoints come from](#3-where-evaluation-data-and-checkpoints-come-from)
4. [CORE prompts: three kinds of prediction problem](#4-core-prompts-three-kinds-of-prediction-problem)
5. [CORE scoring: token boundaries and next-token alignment](#5-core-scoring-token-boundaries-and-next-token-alignment)
6. [From task accuracy to the CORE aggregate](#6-from-task-accuracy-to-the-core-aggregate)
7. [Conversational multiple-choice evaluation](#7-conversational-multiple-choice-evaluation)
8. [Generative evaluation and multiple attempts](#8-generative-evaluation-and-multiple-attempts)
9. [What the GSM8K and HumanEval verifiers accept](#9-what-the-gsm8k-and-humaneval-verifiers-accept)
10. [What ChatCORE combines](#10-what-chatcore-combines)
11. [Uncertainty: how much should a score difference convince us?](#11-uncertainty-how-much-should-a-score-difference-convince-us)
12. [Generalization, contamination, and repeated selection](#12-generalization-contamination-and-repeated-selection)
13. [Designing a meaningful comparison](#13-designing-a-meaningful-comparison)
14. [IsoFLOP experiments and the model/data tradeoff](#14-isoflop-experiments-and-the-modeldata-tradeoff)
15. [Deriving what the scaling notebook fits](#15-deriving-what-the-scaling-notebook-fits)
16. [Auditing the experiment scripts before interpreting their output](#16-auditing-the-experiment-scripts-before-interpreting-their-output)
17. [Connections to your implementation and small exercises](#17-connections-to-your-implementation-and-small-exercises)
18. [Checks performed and what comes next](#18-checks-performed-and-what-comes-next)

## 1. What question is each evaluation answering?

**Gist:** prediction loss, choosing an answer, generating a solution, and looking at a few samples measure different behavior. No single scalar completely characterizes a language model. Understanding the question comes before interpreting the score.

### 1.1 The four measurements in this repository

**Gist:** BPB measures predictive fit; CORE measures selected base-model task performance; chat evaluations test constrained answers or generated solutions; samples offer qualitative inspection.

| Measurement | Actual question | Preferred direction |
| --- | --- | --- |
| Held-out BPB | How much negative log probability is assigned per represented text byte? | Lower |
| Base CORE | How often does this prompt/scoring protocol solve each configured task, relative to its baseline? | Higher |
| Chat categorical accuracy | Which valid answer-letter token has the highest next-token score? | Higher |
| Chat generative success | Does at least one returned completion pass the task verifier? | Higher |
| Printed samples | What continuations does the model visibly produce for these prompts? | Qualitative |

**Code map:** [evaluate_bpb](../nanochat/nanochat/loss_eval.py#L9), [evaluate_core](../nanochat/scripts/base_eval.py#L57), [run_categorical_eval](../nanochat/scripts/chat_eval.py#L87), and [run_generative_eval](../nanochat/scripts/chat_eval.py#L28).

### 1.2 Same model, different measurement

**Gist:** a model can improve average likelihood without changing which answer wins an argmax. It can also become better at a particular task while changing unrelated text prediction very little.

For a two-choice problem whose correct answer is A, changing probabilities from $(0.49,0.51)$ to $(0.51,0.49)$ flips accuracy from 0 to 1 with only a small loss change. Changing from $(0.60,0.40)$ to $(0.90,0.10)$ substantially improves the correct-answer log loss while accuracy remains 1.

An argmax score discards confidence information; likelihood retains it. Conversely, generic text likelihood averages over a distribution that may place little weight on the capability you care about. A good experiment reports relevant views of the same model rather than assuming they must move identically.

## 2. Bits per byte: the exact held-out prediction metric

**Gist:** BPB sums negative log probabilities for ordinary, valid target tokens, converts nats to bits, and divides by their represented bytes. The denominator is text bytes, not token count or tensor storage. It provides common units across tokenizers, provided the underlying evaluation conditions are also comparable.

### 2.1 Derive the numerator and denominator

**Gist:** treat the predicted text as an idealized coding problem: assigning probability $p$ costs $-\log_2p$ bits. Divide total coding cost by the amount of text represented, rather than by how the tokenizer happened to split it.

For evaluated target positions $i$, define

$$
\ell_i=-\ln p_\theta(y_i\mid x_{\le i}),\qquad
b_i=\text{number of raw bytes represented by target }y_i.
$$

Let $\mathcal I$ include valid targets with $b_i>0$. Then

$$
\boxed{\mathrm{BPB}=
\frac{\sum_{i\in\mathcal I}\ell_i}{\ln2\sum_{i\in\mathcal I}b_i}}.
$$

For 100 bits of prediction cost over 200 represented bytes, BPB is 0.5. A token spanning four bytes contributes its full loss once and four bytes to the denominator; it is not counted as four independent predictions.

When mean loss $\bar\ell$ and mean bytes/token $R$ are computed over the same included targets,

$$
\mathrm{BPB}=\frac{\bar\ell}{R\ln2}.
$$

For an illustrative tokenizer A, $\bar\ell=1.2$ and $R=2$ gives about 0.866 BPB. For tokenizer B, $\bar\ell=1.8$ and $R=4$ gives about 0.649 BPB. B has worse nats/token but better bits/byte. Comparing raw token losses across tokenizers would reverse the conclusion.

### 2.2 Which positions count in nanochat?

**Gist:** ordinary token bytes count, special-token targets do not, and ignored labels do not. Special tokens can still provide input context even when their prediction loss is excluded from this metric.

**Code:** [byte-table construction](../nanochat/scripts/tok_train.py#L78), [BPB accumulation](../nanochat/nanochat/loss_eval.py#L31).

`tok_train.py` stores each ordinary token's raw byte length from `decode_single_token_bytes`; special tokens get length zero. Raw bytes matter because an individual byte-level token need not form a complete Unicode character on its own. Counting decoded string characters would measure something else.

The evaluator requests `loss_reduction='none'`, then sums losses only where the target byte length is positive. For `ignore_index=-1`, it uses a safe indexing path so negative labels do not accidentally index the byte table from its end. That indexing guard does not make arbitrary negative labels legal inputs to the model's cross-entropy; the model's configured ignore label still matters.

The training CE includes targets such as BOS when present. BPB omits those zero-byte targets. Consequently, even on the same batch, training mean CE and BPB differ in both units and included positions.

### 2.3 Aggregate totals, not batch or rank averages

**Gist:** the correct aggregate weights each observation by its represented bytes. Averaging already-normalized BPB values equally can give the wrong result when evaluated text lengths differ.

The evaluator [sums the two totals across ranks](../nanochat/nanochat/loss_eval.py#L57) before dividing. If rank 0 has 100 bits over 100 bytes and rank 1 has 100 bits over 400 bytes, then

$$
\mathrm{BPB}_{global}=\frac{100+100}{100+400}=0.4,
$$

whereas averaging their BPBs gives $(1+0.25)/2=0.625$. The same issue applies to averaging unequal-sized documents or batches. If total included bytes are zero, this evaluator returns infinity rather than a useful finite score.

### 2.4 Byte normalization does not standardize the whole experiment

**Gist:** common units help, but tokenizer changes can also change context in bytes, packing, and which documents fit. Compare the same text and clearly state the context/selection protocol before attributing a BPB difference solely to model quality.

A fixed 2048-token context can represent different amounts of text with different tokenizers. Packing and cropping can then change the evaluated regions and conditioning. Even with identical raw text, different segmentation defines different prediction events and canonical token paths; byte normalization does not magically make every protocol identical.

In standalone [base_eval](../nanochat/scripts/base_eval.py#L204), the requested split-token budget is rounded down to a multiple of $WbT$, where $W$ is ranks, $b$ local sequence batch, and $T$ context length. It evaluates both train and validation loaders. The token budget is not a byte budget, so included byte totals need not match across tokenizers. A budget smaller than one global batch yields no useful evaluation.

## 3. Where evaluation data and checkpoints come from

**Gist:** the base evaluator loads a checkpoint plus tokenizer, reuses the pretraining text loader for BPB, and fetches a separate task bundle for CORE if needed. Chat tasks use their own dataset adapters. These are different evaluation data sources.

### 3.1 Checkpoint, tokenizer, and numerical path

**Gist:** selecting the weights is part of defining the experiment. The default loader can guess a model tag and latest step; an explicit comparison should identify the intended tag, step, tokenizer, and dtype.

**Code:** [load_model_from_dir](../nanochat/nanochat/checkpoint_manager.py#L148), [build_model](../nanochat/nanochat/checkpoint_manager.py#L76).

`base_eval` loads source `base`; `chat_eval` loads its requested `sft` or `rl` source. Without a model tag, the loader guesses the largest depth-tagged directory, falling back to directory modification time. Without a step, it chooses the highest available checkpoint step. “Latest” and “best on this metric” are different selections.

The loader rebuilds ordinary GPT modules and loads parameters; it does not reconstruct FP8 linear wrappers as a permanent model architecture. During pretraining, the trainer also temporarily disables FP8 for evaluation. The ordinary dtype path still follows runtime configuration, and CPU/MPS checkpoint loading promotes stored BF16 tensors to fp32. Compare numerical paths deliberately when measuring tiny differences.

The tokenizer vocabulary size is checked against the checkpoint configuration. Equal vocabulary size alone does not prove that token IDs have identical semantics; retain the actual tokenizer and byte table used with the model.

### 3.2 The CORE bundle is a separate artifact

**Gist:** CORE task membership, few-shot counts, data paths, and baselines are loaded from a downloaded bundle, not hardcoded as one list in `core_eval.py`. The Python evaluator specifies how to interpret those files.

**Code:** [bundle URL and installation](../nanochat/scripts/base_eval.py#L42), [configuration loading](../nanochat/scripts/base_eval.py#L68).

The source points to [Karpathy's eval bundle ZIP](https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip). It is unpacked under `NANOCHAT_BASE_DIR/eval_bundle/` (normally the nanochat cache directory), containing:

```text
eval_bundle/
    core.yaml                 # task entries under icl_tasks
    eval_meta_data.csv        # task labels and random-baseline percentages
    eval_data/...             # task JSONL files referenced by the YAML
```

For each entry, the code reads its label, task type, dataset URI, first `num_fewshot` value, and optional continuation delimiter. It does not average all few-shot settings in the list. The evaluator credits the [DataComp-LM evaluation work](https://arxiv.org/abs/2406.11794), but this bundle plus local scoring code define the exact executed protocol.

No bundle was cached at the default local location during this note's preparation, and none was downloaded. Therefore this note does not invent a verified list of its current task names or assert that every paper reference task is present. `core.yaml` is the authoritative membership list for a particular installed bundle. Record that artifact alongside the code revision for reproducibility; the URL alone is not an immutable dataset hash.

### 3.3 Chat-task data sources

**Gist:** conversational evaluations use explicit task adapters with their own rendering and success checks. Their standard test splits are loaded from Hugging Face parquet exports and deterministically shuffled in this code.

| Task | Adapter's dataset identifier | Evaluation split/type |
| --- | --- | --- |
| ARC-Easy / ARC-Challenge | `allenai/ai2_arc` | Test, categorical |
| MMLU | `cais/mmlu`, subset `all` | Test, categorical |
| GSM8K | `openai/gsm8k`, subset `main` | Test, generative |
| HumanEval | `openai/openai_humaneval`, config `openai_humaneval` | Test, generative |

See [run_chat_eval's task mapping](../nanochat/scripts/chat_eval.py#L160) and [load_hub_dataset](../nanochat/tasks/common.py#L45). The loader caches parquet files and a manifest. Those paths do not themselves pin a Hugging Face revision or prove that data never appeared in training. The adapters shuffle with seed 42; `max_problems` selects a prefix of that resulting task view.

## 4. CORE prompts: three kinds of prediction problem

**Gist:** base models are evaluated through text continuation prompts rather than conversation role tokens. The evaluator supports multiple-choice, schema, and language-modeling tasks. They differ in which text varies and which portion is scored.

### 4.1 Multiple choice: same question, different continuations

**Gist:** one candidate prompt is formed per answer option. The evaluator scores the option's inferred token continuation, rather than asking the base model to generate an answer letter.

**Code:** [render_prompts_mc](../nanochat/nanochat/core_eval.py#L17).

Ignoring few-shot demonstrations for the moment:

```text
candidate 0: query + delimiter + choice 0
candidate 1: query + delimiter + choice 1
candidate 2: query + delimiter + choice 2
```

Every candidate can be forwarded together as a separate batch row. Its answer text is present as an input suffix for teacher-forced scoring, but causality prevents a position from looking ahead at the next token it is predicting. This is the same next-token alignment used during pretraining.

### 4.2 Schema: different contexts, same ending

**Gist:** schema scoring asks which candidate context makes one shared continuation most plausible. Unlike ordinary multiple choice, the varying text is before the scored suffix.

**Code:** [render_prompts_schema](../nanochat/nanochat/core_eval.py#L36).

The candidate strings have the form

```text
candidate 0: context option 0 + delimiter + common continuation
candidate 1: context option 1 + delimiter + common continuation
```

The tokenized common suffix is scored under each candidate prefix. Thus the task compares conditional predictions of an ending, not the unconditional likelihood of the whole context-plus-ending string.

### 4.3 Language modeling and few-shot context

**Gist:** the language-modeling task type checks whether every continuation token is the model's top next-token choice under the true preceding tokens. Few-shot demonstrations are prompt examples, not gradient updates.

**Code:** [render_prompts_lm](../nanochat/nanochat/core_eval.py#L56), [demonstration selection](../nanochat/nanochat/core_eval.py#L176).

For the LM type, the renderer makes a context-only string and a context-plus-continuation string so the continuation span can be identified. The context is trimmed to reduce token-prefix ambiguities from trailing whitespace.

If a task requests $f$ demonstrations, the evaluator samples $f$ other examples using `random.Random(1234 + idx)` and prepends their correct examples to the current prompt. The current example is excluded. Its model weights do not change; this is in-context conditioning.

Before evaluation, the task data is shuffled with seed 1337 and optionally truncated by `max_per_task`. Demonstrations are sampled from that already-truncated data. Changing the cap can therefore change both the evaluation questions and the demonstration pool. A cap too small to supply $f$ other examples causes a sampling error rather than a meaningful reduced test.

## 5. CORE scoring: token boundaries and next-token alignment

**Gist:** the exact token span is as important as the probability formula. The code infers boundaries after tokenization, shifts loss positions by one, and uses different correctness rules for candidate ranking versus LM exact-match prediction.

### 5.1 How the continuation boundaries are inferred

**Gist:** a character-level prompt boundary need not survive as a token boundary. Nanochat therefore compares token sequences: a common prefix for MC, a common suffix for schema, and an asserted prefix for LM.

**Code:** [batch_sequences_mc](../nanochat/nanochat/core_eval.py#L113), [schema batching](../nanochat/nanochat/core_eval.py#L123), [LM batching](../nanochat/nanochat/core_eval.py#L133).

- MC: find the longest common token prefix across all candidate prompts. Score each candidate after that prefix.
- Schema: find the longest common token suffix across all candidates. Score that suffix under each differing context.
- LM: encode without/with continuation and assert the first token sequence is an exact prefix of the second. Score the appended tokens.

BOS is prepended during tokenization. These procedures handle token merging at boundaries better than assuming that tokenizing a standalone answer yields the same suffix IDs as tokenizing the complete prompt.

They also define a particular protocol. If choices begin with shared answer text, MC's common prefix can extend inside the answer; those shared tokens are then excluded. Schema's shared suffix can similarly extend beyond a semantically annotated boundary. The code scores its inferred spans, not an independently supplied answer-token annotation.

### 5.2 Why the score slice starts at `start - 1`

**Gist:** output position $t$ predicts input token $t+1$. To score target tokens at positions $s$ through $e-1$, use loss positions $s-1$ through $e-2$.

**Code:** [forward_model](../nanochat/nanochat/core_eval.py#L145), [scored slices](../nanochat/nanochat/core_eval.py#L229).

For a row with zero-based positions,

```text
input index:    0      1      2      3
input token:  BOS   context   A      B
output index:  0      1      2      3
predicts:   context  A      B     no next target
```

If the continuation is `[A, B]`, then $s=2$, $e=4$, and its losses come from `[1:3]`, expressed in Python as `[s-1:e-1]`.

`forward_model` rolls input IDs left to construct targets and sets its final loss column to NaN because no genuine next token exists there. Valid continuation slices exclude that column. It also returns each position's vocabulary argmax.

Rows are right-padded with BOS to form a batch. With causal attention, later padding cannot influence earlier valid positions; the explicit span bounds keep padding targets out of the score. The code assumes usable, nonempty scored spans. Degenerate identical candidates or malformed boundaries deserve inspection rather than treating a NaN mean as a valid measurement.

### 5.3 MC/schema use mean token loss, not sequence probability

**Gist:** nanochat chooses the candidate with the smallest average continuation NLL. This is length-normalized token likelihood; it can rank candidates differently from their full conditional sequence probabilities.

For candidate $j$ with scored span length $L_j=e_j-s_j$,

$$
S_j=\frac1{L_j}\sum_{t=s_j}^{e_j-1}
-\ln p_\theta(x_t^{(j)}\mid x_{\lt t}^{(j)}),
\qquad \widehat j=\arg\min_j S_j.
$$

Minimizing the sum alone maximizes the product of token probabilities. Minimizing the mean maximizes their geometric mean. For candidate A with one token of probability 0.4, versus B with two tokens each of probability 0.6:

| Candidate | Sequence probability | Mean NLL |
| --- | ---: | ---: |
| A | 0.4 | About 0.916 |
| B | $0.6^2=0.36$ | About 0.511 |

Full sequence probability prefers A; this evaluator prefers B. Length normalization removes a simple product-length effect but defines a different decision rule. It does not universally eliminate all length/tokenization biases.

Schema candidates share the same inferred token suffix length, so mean versus summed loss does not change their ranking within that example. MC candidate lengths can differ. Exact ties use the first minimum in list order.

### 5.4 LM correctness is an all-token indicator

**Gist:** the LM task type returns one success bit only if every scored token matches the corresponding next-token argmax. It does not report the mean continuation likelihood or require generated reasoning text.

The condition is

$$
C=\prod_{t=s}^{e-1}
\mathbf1\!\left[
x_t=\arg\max_v p_\theta(v\mid x_{\lt t})\right].
$$

It is computed under teacher forcing in one forward. If all predictions match, a greedy autoregressive rollout from the same prefix also follows this continuation for its specified length, by induction. The check does not separately assess stopping behavior, alternative correct wordings, or stochastic generation.

The optional left-truncation branch applies only when a model exposes `max_seq_len`. Nanochat's GPT does not define that attribute in this checkout; do not assume the evaluator automatically crops its prompts to `config.sequence_len`. Its actual window and rotary-cache constraints still apply.

## 6. From task accuracy to the CORE aggregate

**Gist:** each example becomes a success bit, each task becomes an accuracy, and CORE averages baseline-centered task accuracies. Tasks get equal aggregate weight even if they contain different numbers of examples.

### 6.1 Distributed accuracy sums example outcomes

**Gist:** ranks evaluate disjoint question indices and sum the outcome array, then average over all questions. It avoids giving a rank with fewer examples the same weight as a rank with more examples.

**Code:** [evaluate_task](../nanochat/nanochat/core_eval.py#L244).

With $W$ ranks, rank $r$ evaluates indices $r,r+W,r+2W,\ldots$. Each writes its assigned outcomes into a zero-initialized vector of length $n$. Summing those vectors reconstructs all outcomes, so task accuracy is

$$
a=\frac1n\sum_{i=1}^n C_i.
$$

Replicating the data list and seeded prompt logic keeps the partition consistent. This is distribution of evaluation examples, not optimizer-state ownership or training on the benchmark.

### 6.2 Center at baseline, then average tasks

**Gist:** centering maps a task's nominal baseline to 0 and perfection to 1. The resulting number can be negative; it is not necessarily a probability or a percentage of correctly answered questions.

**Code:** [baseline centering and CORE mean](../nanochat/scripts/base_eval.py#L112).

For task $j$ with accuracy $a_j$ and baseline fraction $r_j$,

$$
c_j=\frac{a_j-r_j}{1-r_j},\qquad
\boxed{\mathrm{CORE}=\frac1J\sum_{j=1}^J c_j}.
$$

The CSV stores baseline percentages, so the source first multiplies them by 0.01. At $r=0.25$, an accuracy of 0.625 maps to 0.5. An accuracy of 0 maps to $-1/3$, demonstrating why the aggregate is not guaranteed to lie in $[0,1]$.

Equal task weighting differs from pooled accuracy. A task with 100 examples and one with 10,000 both contribute one centered number. Inspect the per-task vector as well as its mean; improvements concentrated in one task family may be hidden by an aggregate. Baseline scaling also multiplies accuracy changes by $1/(1-r_j)$, changing their relative influence.

## 7. Conversational multiple-choice evaluation

**Gist:** a chat model receives a conversation-formatted problem and predicts a valid answer letter. This differs from CORE's likelihood scoring of answer continuations. Formatting, token IDs, and the position at which logits are read are part of the evaluation.

### 7.1 Remove the reference answer before prompting

**Gist:** task objects contain an assistant reference so a verifier knows what is correct. The completion renderer removes that reference before building the model prompt, then appends the assistant-start token.

**Code:** [render_for_completion](../nanochat/nanochat/tokenizer.py#L241), [multiple-choice rendering](../nanochat/tasks/common.py#L187).

The renderer deep-copies the conversation, pops its last assistant message, renders the remaining messages, and appends `<|assistant_start|>`. The original reference remains available to the task's `evaluate` method. Merely seeing a reference in the task dictionary does not imply that the model receives it.

`render_mc` writes options with a letter after each choice, such as `Paris=A`, and asks for only the letter. The no-space convention is deliberate: a standalone `A` token and a space-prefixed ` A` can have different token IDs. The categorical evaluator separately encodes each valid letter and asserts it is one token.

### 7.2 Restrict the decision to allowed letter tokens

**Gist:** the scorer asks which valid letter is most likely next, not whether unconstrained generation would actually produce that letter. It discards the logits of all non-answer tokens.

For prompt $c_i$ with valid letter-token set $\mathcal A_i$ and final prompt-position logits $z_i$,

$$
\widehat a_i=\arg\max_{v\in\mathcal A_i}z_{i,v}.
$$

This is equivalent to maximizing the softmax renormalized over only the valid letters:

$$
\widetilde p(v\mid c_i,\mathcal A_i)
=\frac{e^{z_{i,v}}}{\sum_{u\in\mathcal A_i}e^{z_{i,u}}}.
$$

It is not the full-vocabulary probability of following the requested response format. For instance, an ordinary word could have a larger logit than every letter; the scorer still selects the best allowed letter. This makes the score useful for answer discrimination but different from free-response instruction following.

### 7.3 Batch variable-length questions without reading the padding

**Gist:** prompts are right-padded for one parallel forward, but each question is scored at its own last genuine prompt token. Distributed ranks sum passed/total counts, rather than equally averaging rank-local accuracies.

**Code:** [prompt lengths and right padding](../nanochat/scripts/chat_eval.py#L104), [focused logits](../nanochat/scripts/chat_eval.py#L131).

For a prompt with $n_i$ tokens, the answer logits are at position $n_i-1$, even if another prompt makes the padded batch longer. Using the final padded column for every example would ask a different question for shorter prompts. Causality keeps later BOS padding from changing earlier valid logits in the intended real-arithmetic computation.

Ranks receive disjoint batches of problems, including a potentially short final batch. The final accuracy is total successes divided by total evaluated problems. For MMLU, the adapter uses all test questions and retains subject metadata, but this loop does not separately macro-average the 57 subjects; MMLU contributes one pooled question accuracy to ChatCORE.

## 8. Generative evaluation and multiple attempts

**Gist:** generative tasks require actual completions, not just a letter logit. The model/engine produces one or more attempts and a verifier judges them. The attempt count and generation settings directly affect the reported success rate.

### 8.1 The actual loop counts problems, not successful samples

**Gist:** a problem counts as solved if any returned completion passes. Two successes among six completions can therefore produce a different result from two solved problems among three questions.

**Code:** [run_generative_eval](../nanochat/scripts/chat_eval.py#L28).

For each problem $i$, the code renders a completion prompt, asks `Engine.generate_batch` for $k$ samples, removes the prompt prefix from each returned token list, decodes the continuations, and obtains verifier outcomes $R_{i,1},\ldots,R_{i,k}\in\{0,1\}$. It records

$$
Y_i=\mathbf1\!\left[\sum_{j=1}^kR_{i,j}>0\right],\qquad
\widehat A_k=\frac1n\sum_{i=1}^nY_i.
$$

For outcomes `[wrong, right]`, `[wrong, wrong]`, `[right, wrong]`, two of three problems pass: $\widehat A_2=2/3$. The fraction of passing individual samples is only $2/6=1/3$. The script returns the first statistic.

At $k=1$, this is ordinary single-completion success. At $k>1$, report the attempt budget explicitly; an unlabeled “accuracy” no longer communicates the whole protocol.

### 8.2 Why more attempts can raise the score without changing weights

**Gist:** multiple attempts give the model more chances to produce a verifier-accepted answer. Under independent attempts, a problem with single-attempt success probability $q_i$ has success probability $1-(1-q_i)^k$ after $k$ tries.

The derivation is simply the complement of all attempts failing:

$$
\Pr(Y_i=1)=1-\Pr(R_{i,1}=\cdots=R_{i,k}=0)
=1-(1-q_i)^k.
$$

At $q_i=0.2$, five independent attempts succeed with probability $1-0.8^5\approx0.672$. That is a change in inference budget, not an improvement in the model's parameters.

Across heterogeneous problems, the expected score is $\frac1n\sum_i[1-(1-q_i)^k]$, generally not $1-(1-\bar q)^k$. The model can be reliable on some questions and almost incapable on others. Independence is also an assumption: repeating deterministic greedy generation does not create independent new chances.

For comparison, the [HumanEval reference estimator](https://github.com/openai/human-eval/blob/master/human_eval/evaluation.py) can estimate pass@$k$ from $n_s\ge k$ sampled attempts with $c$ successes:

$$
\widehat{\mathrm{pass@}k}
=1-\frac{\binom{n_s-c}{k}}{\binom{n_s}{k}}.
$$

The ratio is the fraction of size-$k$ subsets containing no success. Nanochat's loop does not estimate several $k$ values from a larger saved sample set; it directly takes `any` across the requested attempts. When $k=n_s$, the combinatorial estimator reduces to that same any-success indicator. The [HumanEval paper](https://arxiv.org/abs/2107.03374) explains evaluation through repeated sampling and functional tests.

### 8.3 Temperature, top-k, token budget, and tools define the experiment

**Gist:** the evaluation policy includes how completions are sampled and what external execution is available. “Same checkpoint” is insufficient to reproduce a generative score without those settings.

The chat CLI defaults are temperature 0, top-k 50, one sample, and 512 new-token steps. At temperature 0, [sample_next_token](../nanochat/nanochat/engine.py#L141) returns the vocabulary argmax directly; top-k sampling is bypassed. Multiple greedy rows ordinarily repeat the same path rather than discovering diverse solutions.

At positive temperature, top-k restricts the candidate set before probabilities are normalized. `Engine.generate` creates a generator with default seed 42 for each call. This improves repeatability, but batching, attempt count, device, and numerical changes can still change sampled paths. Report the seed and settings rather than assuming “deterministic seed” eliminates sampling uncertainty.

The engine can recognize calculator-call tokens and inject tool-result tokens. Its per-column generation loop counts forced tool tokens toward its token-step budget as well as sampled tokens. A result can therefore include external arithmetic, and a long tool result can use part of the allowed generation budget. Tool-enabled and tool-free scores answer different questions; detailed engine mechanics belong in the inference note.

## 9. What the GSM8K and HumanEval verifiers accept

**Gist:** a verifier defines operational correctness for a benchmark. GSM8K uses an answer-string extraction rule; HumanEval assembles and executes code tests. These checks do not independently certify the quality of every reasoning step or every behavior outside the tests.

### 9.1 GSM8K: extract a marked number and compare strings

**Gist:** this adapter searches for `#### number`, strips commas, and compares the extracted text to the reference. It does not symbolically verify the derivation or normalize every numerically equivalent expression.

**Code:** [extract_answer](../nanochat/tasks/gsm8k.py#L22), [GSM8K.evaluate](../nanochat/tasks/gsm8k.py#L86).

The regex accepts an optional minus sign and digits, dots, and commas after `#### `. It uses the first matching marker. Thus `#### 1,000` becomes `1000`, but `#### 1000.0` remains `1000.0` and does not equal the string `1000`.

A correct-looking number without the required marker is not accepted by this rule when the reference parses normally. A wrong derivation ending in the correct marked number can pass. A valid reference parse is an assumption worth verifying for task data; the checker does not introduce a separate proof of reference correctness.

The task's `reward` method reuses this evaluation outcome as a float. This is a preview of why verifier definitions matter again in reinforcement learning: the optimized reward is the implemented criterion, with its strengths and limitations.

### 9.2 HumanEval: test functional behavior of the extracted program

**Gist:** the checker extracts a code block or plain completion, adds selected imports and task tests, and asks the execution helper whether the tests passed. Formatting, execution behavior, and test coverage all affect the score.

**Code:** [extract_program](../nanochat/tasks/humaneval.py#L23), [HumanEval.evaluate](../nanochat/tasks/humaneval.py#L79).

The task stores a user prompt, reference full solution, entry point, and tests. The completion renderer removes the reference from the prompt before generation. Evaluation takes imports from the beginning of the original prompt, the first fenced code block if one is found (otherwise the full completion), the tests, and a call to `check(entry_point)`.

This is not automatically the same program-construction protocol as evaluating a raw code continuation appended to a function prefix. A different extraction or prompt convention can change results even for the same model.

Passing establishes that the assembled program passed the executed finite tests under that execution environment. It does not establish correctness for every possible input. Execution failures and incomplete/truncated solutions affect the measured success too. No generated programs or benchmark execution jobs were launched to prepare this note.

## 10. What ChatCORE combines

**Gist:** ChatCORE is a mean of five centered task scores, mixing three restricted-letter evaluations with two generative ones. It is a convenient summary of this protocol, not the same quantity as base CORE or a universal conversational-ability scale.

### 10.1 The exact five-task mean

**Gist:** the baseline is 0.25 for ARC-Easy, ARC-Challenge, and MMLU, and zero for GSM8K and HumanEval. The baseline constants are fixed in the script, rather than inferred anew from each problem's option count.

**Code:** [task set and baselines](../nanochat/scripts/chat_eval.py#L202), [ChatCORE calculation](../nanochat/scripts/chat_eval.py#L228).

Using $a_{AE},a_{AC},a_M,a_G,a_H$ for those five returned scores,

$$
\boxed{\mathrm{ChatCORE}=\frac15\left[
\frac{a_{AE}-0.25}{0.75}
+\frac{a_{AC}-0.25}{0.75}
+\frac{a_M-0.25}{0.75}
+a_G+a_H\right]}.
$$

It is printed only when all five required task names were evaluated. Like CORE, it can be negative when categorical results fall sufficiently below baseline, despite comments informally describing a 0-to-1 range.

For illustrative raw scores $(0.55,0.40,0.325,0.20,0.10)$, centered scores are $(0.40,0.20,0.10,0.20,0.10)$ and ChatCORE is 0.20. That does not mean 20% of all pooled benchmark questions were solved.

### 10.2 Why scores need their component protocols

**Gist:** increasing generative attempts can raise two components while the categorical components stay unchanged. Changes in aggregation, formatting, tools, or task caps can also change the aggregate without changing the model.

`num_samples` affects the generative loop, not categorical letter selection. A ChatCORE result using eight GSM8K/HumanEval attempts is not directly comparable with a one-attempt result just because the script prints the same metric name.

Likewise, reducing `max_problems` changes the task prefix evaluated. MMLU's pooled question weighting and the outer equal weighting of the five tasks are two separate aggregation choices. Report the five raw scores, centered scores, and generation settings alongside the aggregate.

## 11. Uncertainty: how much should a score difference convince us?

**Gist:** a measured difference can reflect finite evaluation data, stochastic generation, or variation between training runs. These are different uncertainty sources. A fixed seed helps repeat one experiment but does not measure all three.

### 11.1 A first approximation for an accuracy

**Gist:** if evaluated examples behave like independent draws, binomial sampling gives a useful first estimate of accuracy uncertainty. Correlated questions and shared prompts can make that estimate optimistic.

For $n$ binary outcomes with success probability $p$,

$$
\widehat p=\frac1n\sum_iY_i,\qquad
\mathrm{SE}(\widehat p)\approx
\sqrt{\frac{\widehat p(1-\widehat p)}n}.
$$

At accuracy 0.6 on 100 examples, this is about 0.049, or 4.9 percentage points. A rough normal 95% interval is $\widehat p\pm1.96\,\mathrm{SE}$, about $0.6\pm0.096$. At 1000 examples the analogous half-width is about 0.030.

These are sampling approximations, not confidence statements automatically emitted by nanochat. The normal approximation is especially poor for tiny samples or near-zero/one accuracies; alternative intervals or resampling may be more appropriate. If the benchmark is treated as a fixed complete set, the uncertainty question is generalization beyond that set, not uncertainty about the arithmetic of its known outcomes.

### 11.2 Compare the same examples as paired observations

**Gist:** when two models answer the same questions, look at which questions changed. Pairing can estimate uncertainty more directly than treating the two accuracy estimates as independent.

Let $Y_i^A,Y_i^B\in\{0,1\}$ and $d_i=Y_i^B-Y_i^A$. Then the accuracy difference is

$$
\widehat\Delta=\frac1n\sum_i d_i,
\qquad
\mathrm{SE}(\widehat\Delta)
\approx\sqrt{\frac{\sum_i(d_i-\widehat\Delta)^2}{n(n-1)}}.
$$

For 1000 questions, if B fixes 40 of A's failures but breaks 20 of A's successes, its improvement is $(40-20)/1000=0.02$. The paired SE is about 0.0077 under the independent-question approximation. The shared correct/incorrect outcomes affect uncertainty differently from the disagreements.

For BPB, compare per-document or block-level loss/byte contributions with the same evaluation text. Adjacent token losses share context and are correlated; treating millions of tokens as millions of independent experiments would overstate confidence. A document/block bootstrap can resample those units and recompute the ratio of totals, preserving the metric's byte weighting.

### 11.3 Aggregates and training variability

**Gist:** an aggregate inherits uncertainty from its components, and a promising single training seed does not establish a reliably better training method. Keep evaluation noise separate from variability in the learned checkpoint.

If task accuracies were independent, the centered mean would have approximate variance

$$
\mathrm{Var}(\mathrm{CORE})
=\frac1{J^2}\sum_j\frac{\mathrm{Var}(a_j)}{(1-r_j)^2}.
$$

Shared data/prompt effects can add covariance terms. Resampling should respect the intended task weighting and relevant clusters rather than pooling all question outcomes as if the score were one simple binomial mean.

Different training seeds or shard orders can produce different checkpoints even with the same recipe. Repeated independent training runs address that variation; repeated stochastic completions from one checkpoint address sampling variation. The repository's standard evaluators mainly print point estimates and do not automatically perform these uncertainty analyses.

## 12. Generalization, contamination, and repeated selection

**Gist:** held-out performance is evidence about the evaluated distribution under its protocol. It does not alone establish broad generalization, absence of training overlap, or an unbiased estimate after extensive tuning on the same benchmark.

### 12.1 A validation split and a benchmark test set have different roles

**Gist:** use validation feedback to choose settings, then reserve a genuinely separate assessment for the final claim where possible. Repeatedly selecting winners on a test metric makes that metric part of the development loop.

BPB validation uses text separated by the pretraining loader's split rule. CORE and chat benchmarks ask additional task questions. Neither name alone proves independence from every training source. Few-shot demonstrations intentionally include other examples' answers in the prompt; that is allowed conditioning under the specified few-shot protocol, not a gradient update or disclosure of the current answer.

When many variants/checkpoints are tried, the best observed validation score tends to include favorable noise. Selecting the minimum from many noisy estimates makes the selected estimate optimistic. Evaluating the selected recipe on fresh data or independent runs helps test whether its advantage persists.

### 12.2 What contamination would mean here

**Gist:** overlap can occur through pretraining text, supervised examples, answer explanations, or near-duplicates. A test split label in a dataset adapter does not certify that none of those appeared elsewhere in the pipeline.

An exact question/answer copy, a paraphrase with the same solution, or an online discussion of a benchmark can all create different forms of overlap. Decontamination needs explicit data matching/provenance work; this evaluator does not infer it from a high score.

Similarly, intentionally training on benchmark training splits is different from inadvertently including test items in a large web corpus. The later SFT note will identify actual training mixtures and splits. Do not read the evaluation task mapping as a complete audit of what the model previously saw.

### 12.3 A reproducible result is more than a scalar

**Gist:** retain enough information to reconstruct the measurement and diagnose differences. This is especially useful when the source, dataset bundle, tokenizer, or sampling settings evolve.

For a concrete result, record checkpoint/source/tag/step, code revision, tokenizer/byte-table identity, dataset or bundle version/hashes, question subset, few-shot setup, prompt templates, scoring normalization, dtype/backend, and sampling/tool/token budgets where relevant.

Retaining per-task and preferably per-example outcomes permits paired comparisons and error analysis later. The standard standalone base evaluator writes a per-task CORE CSV; chat evaluation primarily prints results. Neither automatically archives every artifact or per-example outcome required for a full reproducibility record.

## 13. Designing a meaningful comparison

**Gist:** define what stays fixed before saying one model or recipe is better. Matching update count, tokens, FLOPs, wall time, or inference budget creates different experiments. A useful comparison names the resource constraint and the quality objective.

### 13.1 What do you hold fixed?

**Gist:** there is no single universal fair budget. Pick the one corresponding to the scientific or practical question, and report the consequences for the others.

| Fixed quantity | Question it helps answer | What can still differ? |
| --- | --- | --- |
| Number of optimizer steps | What happens under this update-count schedule? | Tokens, FLOPs, time |
| Training tokens | Which recipe learns more from this amount of exposure? | Compute and runtime |
| Estimated training FLOPs | Which model/data allocation uses this arithmetic budget best? | Hardware utilization and overhead |
| Wall time on stated hardware | Which method reaches better quality in this time? | Tokens and arithmetic |
| Generative attempts/tokens/tools | Which checkpoint performs better under this inference protocol? | Training cost |

For instance, an optimizer with more work per update can win at fixed tokens but lose at fixed wall time. An FP8 path can process tokens faster but change their learning effect. A smaller model can afford more tokens under a fixed compute budget. These are measurable tradeoffs, not contradictory evaluations.

### 13.2 Ablate one mechanism and identify the tuning policy

**Gist:** a controlled ablation changes one intended mechanism while preserving the rest as far as possible. If hyperparameters are retuned or automatically rescaled, that tuning policy becomes part of the treatment being compared.

For an illustrative smear ablation, one could compare the current architecture with the smear path disabled, using the same tokenizer, data order, model size convention, total budget, evaluation protocol, and specified hyperparameter policy. Changing data mixture and optimizer simultaneously would make attributing the result to smear difficult.

There are two legitimate but different comparisons: “swap this mechanism using the same settings” and “compare both mechanisms after comparable tuning.” The first measures a drop-in effect; the second measures performance under the chosen tuning effort. Include tuning cost if the practical claim concerns total experimentation cost.

### 13.3 Read curves and error cases, not only final winners

**Gist:** learning curves show whether a gain is early, persistent, or an artifact of stopping. Per-task/error analysis shows where an aggregate improvement comes from. Final scalar scores hide both kinds of structure.

Plot the metric against the relevant resource axis: tokens, compute, or measured time. A warmdown or momentum schedule tied to the planned endpoint can make an intermediate checkpoint differ from a separately trained shorter run, even at the same consumed tokens.

A useful error analysis asks whether failures concern factual discrimination, response format, length truncation, verifier extraction, numerical answer matching, or program behavior. That diagnosis can guide a targeted experiment more effectively than assuming every low aggregate score indicates insufficient model capacity.

## 14. IsoFLOP experiments and the model/data tradeoff

**Gist:** an isoFLOP experiment trains several model sizes with approximately equal arithmetic budgets, giving larger models fewer tokens. The best point estimates a compute allocation for that recipe and data distribution. Repeat across budgets to infer how the preferred allocation scales.

### 14.1 Construct the equal-compute slice

**Gist:** choose a compute budget $C$ and model sizes, estimate each model's cost per token, and adjust its token count accordingly. Comparing models all trained on the same tokens would answer a different question.

Using the trainer's per-token estimate $\Phi(N)$ and global token batch $G$,

$$
K=\mathrm{round}\!\left(\frac{C}{\Phi(N)G}\right),\qquad
D=KG.
$$

Here $K$ is the number of training steps, $G$ is tokens per step, $D$ is total processed tokens, and $N$ is a stated model-size convention. Rounding makes the achieved estimated budget approximate. The familiar simplified cost $C\approx6ND$ gives $D\approx C/(6N)$, but the actual trainer also estimates attention cost.

For a **constructed illustration**, take $C=6\times10^{18}$ and use the simplified cost:

| Parameters $N$ | Affordable tokens $D$ | Tokens/parameter $D/N$ | Illustrative validation BPB |
| --- | ---: | ---: | ---: |
| 100 million | 10 billion | 100 | 0.85 |
| 200 million | 5 billion | 25 | 0.80 |
| 400 million | 2.5 billion | 6.25 | 0.85 |

The losses here are invented to illustrate a valley, not results from nanochat. The small model receives more exposure but has less capacity; the largest model receives less exposure. The middle model wins in this example. This is the tradeoff studied by [compute-optimal scaling research](https://arxiv.org/abs/2203.15556).

### 14.2 What “optimal” is conditional on

**Gist:** the minimum depends on data, tokenizer, model family, optimizer, settings, and target metric. An optimum for validation BPB need not minimize time to a downstream capability or lifetime training-plus-inference cost.

Nanochat's size dial changes depth and width together; the pretraining note showed that dominant block parameters grow roughly cubically with depth under its aspect-ratio rule. Its batch/LR/decay transfer rules also vary with size. The experiment therefore estimates the preferred model/data allocation for this family and training policy, not for arbitrary neural architectures under perfect optimization.

Keeping compute fixed while increasing size should generally move data allocation in the opposite direction. If a curve shows no interior valley in the tested sizes, it does not establish a measured optimum beyond them. A broader experiment or an explicitly justified extrapolation would be needed.

### 14.3 From several budgets to a transfer rule

**Gist:** locate an estimated best size and token count for each compute budget, then fit how those optima change. A nearly constant token/parameter ratio is an empirical finding within the tested range, not a universal constant.

Suppose the estimated optima follow

$$
N_{\star}(C)=A C^{\gamma_N},\qquad
D_{\star}(C)=B C^{\gamma_D}.
$$

Then

$$
\frac{D_{\star}}{N_{\star}}=\frac BA C^{\gamma_D-\gamma_N}.
$$

If the exponents are approximately equal, the ratio varies slowly with compute. This connects to the pretraining rule $D\approx\rho N_s$. Changing the parameter-count convention changes the numerical ratio and possibly the apparent fitted relationship; it does not change the loss achieved by an already completed run.

The earlier Chinchilla-style derivation used an empirical loss surface and a simplified compute constraint. The notebook below takes a more local route: fit a valley at each budget, then fit power laws through its estimated minima.

## 15. Deriving what the scaling notebook fits

**Gist:** the analysis notebook turns an experiment CSV into estimated isoFLOP minima using quadratics in log parameter count. It then fits straight lines in log compute to those minima. These are fitting choices with finite-data uncertainty, not exact identities imposed by the model.

**Code:** [scaling_analysis.ipynb](../nanochat/dev/scaling_analysis.ipynb), especially the cells beginning `Define which parameters to count`, `IsoFLOP`, and `Optimal Ratio Summary`.

### 15.1 Choose the horizontal-axis parameter convention

**Gist:** the notebook keeps a granular parameter breakdown and currently defines effective parameters as transformer parameters plus the output head. This matches the pretraining scaling count; it is not the total stored parameter count.

The active `compute_effective_params` returns `params_transformer + params_lm_head`. It computes each run's `param_data_ratio = tokens_trained / effective_params`. Alternatives for total or transformer-only counts are commented for experiments.

The input filtering keeps positive, nonmissing validation BPB values. That catches some incomplete rows, but it does not verify every token count, cost estimate, protocol, or failure mode. A well-formed number is not automatically a well-defined experiment.

### 15.2 Fit the valley in log-size coordinates

**Gist:** a quadratic gives a simple smooth estimate between tested sizes. Its vertex can be derived in one line, but accepting that vertex requires checking curvature and whether it lies inside the measured range.

For one budget, set $x=\log_{10}N$ and fit

$$
\widehat L(x)=ax^2+bx+c.
$$

If $a>0$, the derivative vanishes at

$$
x_{\star}=-\frac b{2a},\qquad N_{\star}=10^{x_{\star}},\qquad
\widehat L_{\star}=c-\frac{b^2}{4a}.
$$

If $a\le0$, the notebook falls back to the best observed row. If $a>0$, it accepts the vertex even when outside the sampled range; a convex extrapolated minimum is not the same evidence as an observed interior valley. A nearly flat curvature also makes the estimated location sensitive to small loss perturbations.

The notebook then uses `np.interp` in log-size to obtain tokens and token/parameter ratios from the measured runs. Outside the observed range, that interpolation clamps to endpoint values, even though the quadratic vertex may be outside. Moreover, independently interpolating $D$ and $D/N$ does not guarantee that the resulting ratio equals the interpolated $D$ divided by fitted $N_{\star}$. Check that consistency before interpreting a precise preferred ratio.

### 15.3 Fit the scaling exponents and retain their uncertainty

**Gist:** a straight line in log coordinates corresponds to a power law. With only a few budgets and estimated minima, the exponent is a summary of those experiments, not a high-precision law of nature.

The second fit is

$$
\log_{10}N_{\star} = u_N+\gamma_N\log_{10}C,\qquad
\log_{10}D_{\star} = u_D+\gamma_D\log_{10}C.
$$

Exponentiating gives the power laws in section 14. The notebook reports their difference, evaluates the fitted ratio at the geometric mean of the smallest/largest compute budgets, and also prints the mean/spread of interpolated per-budget ratios.

Uncertainty enters at several levels: noisy trained losses, sparse model sizes, the quadratic approximation, interpolation, and the final power-law regression. A standard deviation across budget ratios is not automatically a confidence interval on the underlying law. Repeating seeds or resampling the experimental measurements would probe different parts of that uncertainty.

## 16. Auditing the experiment scripts before interpreting their output

**Gist:** these shell scripts encode useful experimental designs but contain assumptions tied to older trainer/log versions. Read their current calls and extraction logic before treating their CSV output as ground truth. This note documents observed mismatches without modifying either repository.

### 16.1 What scaling_laws.sh intends to measure

**Gist:** the script crosses four compute budgets with six depths, runs a final full-task CORE evaluation, and records validation loss, parameter breakdown, actual tokens, and elapsed wall time. That is an intended grid of 24 training runs, not a job launched by this note.

**Code:** [budget/depth grid](../nanochat/runs/scaling_laws.sh#L5), [training invocation](../nanochat/runs/scaling_laws.sh#L78), [CSV extraction](../nanochat/runs/scaling_laws.sh#L101).

The grid uses budgets $10^{18},2.15\times10^{18},4.64\times10^{18},10^{19}$ and depths 10 through 20 in steps of two. It passes `--target-flops` and `--target-param-data-ratio=-1`. The explicit FLOP horizon determines $K$.

However, `-1` does not disable earlier batch/decay heuristics. They still form $D_{proxy}=-N_s$ and $D_{ref}=-N_{s,ref}$, whose ratio is positive $N_s/N_{s,ref}$. Thus automatic batch and decay retain parameter-ratio-based scaling instead of being recomputed from the requested FLOP budget's eventual actual tokens. This affects what training policy the grid compares.

The script requests `100 * 524288 = 52,428,800` validation target tokens, despite a comment saying about 100M. Periodic BPB retains the trainer's ordinary interval, while the very large CORE interval is intended to leave CORE to the final step for these horizons. Its elapsed `TRAIN_TIME` includes the launched training process, evaluations, saving, and startup; it differs from the trainer's narrowly accumulated update time.

### 16.2 Current extraction and failure assumptions

**Gist:** a logged CSV can misrepresent a run if its parsers assume old labels, fixed batch sizes, or successful completion. Missing results should remain distinguishable from real zeros.

[miniseries.sh](../nanochat/runs/miniseries.sh#L85) still searches for a combined `Number of parameters:` line, while the current trainer prints [separate parameter-count rows](../nanochat/scripts/base_train.py#L252). It also calculates `tokens_trained = num_iterations * 524288` instead of reading the resolved batch. Automatic global batch changes with model size, so that token count can be wrong even when training succeeds.

`scaling_laws.sh` reads the detailed counts and resolved global batch, making its token accounting more appropriate for this trainer. Both helpers substitute 0 for some missing CORE extraction results, which can hide a failed/missing measurement as a genuine score.

The scripts pipe training output through `tee` without enabling `pipefail`. They therefore do not reliably propagate every training-process failure through the pipeline status before parsing/appending results. The scaling sweep also skips runs already represented in its CSV. Check actual completion and recorded quantities before fitting a curve; a stale or incomplete row can otherwise persist.

The default scaling script does not enable FP8. If adapting it to do so, also address the [FP8 FLOP-count mismatch](03-nanochat-pretraining.md#191-the-fp8-counting-issue-in-this-snapshot); a broken cost estimator invalidates an intended equal-compute comparison.

### 16.3 Miniseries is not an isoFLOP sweep

**Gist:** the miniseries trains several sizes using the ordinary ratio-based horizon rule, rather than holding compute constant. It helps characterize a family of scaled runs, but larger models also receive more tokens and more compute.

The script varies depth without passing an explicit FLOP budget. With $D\approx\rho N_s$ and the simplified cost $C\propto ND$, compute grows roughly quadratically in the relevant parameter size under a fixed ratio, with attention and other architectural terms adding qualifications.

Thus a better final score for the larger member does not isolate capacity from extra training resources. Its relationship to the scaling sweep is complementary: one follows a proposed transfer recipe across sizes; the other probes model/data allocation at each fixed compute level.

## 17. Connections to your implementation and small exercises

**Gist:** your held-out CE estimator already measures predictive fit. Nanochat extends it with byte-normalized text loss, explicit task protocols, conversational verifiers, and experiments across resource budgets. Tiny examples can check most of the scoring logic without loading benchmark data.

### 17.1 Your current evaluation as a starting point

**Gist:** the central loss-estimation pattern is familiar: disable gradients, switch to eval mode, sample held-out batches, and aggregate CE. The new work is defining comparable units and additional task-specific measurements.

Your [estimate_loss](../transformer-from-scratch/data.py#L35) averages cross-entropy over random fixed-shaped batches for train and validation. It excludes the returned MoE balancing term from that reported CE. Equal batch shapes make an unweighted mean of those batch means appropriate for the sampled token positions.

Its character tokenizer, Tiny Shakespeare distribution, random-window selection, and context differ from nanochat's BPE/text mixture. A raw loss comparison between the two repositories is therefore not a quality ranking. Even byte-normalizing would only fix the units; matching data and conditioning would remain necessary.

### 17.2 Small arithmetic checks

**Gist:** these examples isolate normalization and decision rules. They do not require a trained model, GPU, dataset download, or code execution from generated completions.

```python
import math

def bpb(total_nats, total_bytes):
    return total_nats / (math.log(2) * total_bytes)

def centered(accuracy, baseline):
    return (accuracy - baseline) / (1 - baseline)

assert math.isclose(bpb(200 * math.log(2), 500), 0.4)
assert math.isclose(centered(0.625, 0.25), 0.5)
assert math.isclose(centered(0.0, 0.25), -1/3)
assert math.isclose(1 - (1 - 0.2)**5, 0.67232)

sequence_probabilities = [0.4, 0.6 * 0.6]
mean_nlls = [-math.log(0.4), -math.log(0.6)]
assert sequence_probabilities[0] > sequence_probabilities[1]
assert mean_nlls[0] > mean_nlls[1]
```

Then explain which candidate nanochat's MC rule chooses and why that differs from the product ranking. Write the exact `[s-1:e-1]` loss slice for a continuation at input indices 4,5,6. Compare a task-averaged CORE with pooled accuracy for one small and one large task.

### 17.3 A compact experimental-design exercise

**Gist:** planning one controlled experiment is a useful test of whether the earlier training, optimizer, and efficiency notes now fit together.

Suppose you want to test whether FP8 reaches the same validation BPB faster. Specify the checkpoint initialization, data/tokenizer, model configuration, global batch, optimizer/schedules, evaluation subset, ordinary evaluation dtype, and primary time-to-target metric. Identify the source's FLOP-count issue before using FLOPs as an alternative axis.

For a scaling exercise, fit the three invented points in section 14 with a quadratic in $\log_{10}N$. The symmetric valley should give $N_{\star}=200$ million. Why do three points with a perfect quadratic fit not establish that the quadratic is the right loss surface? What extra observation would test the inferred optimum rather than just restating the fit?

## 18. Checks performed and what comes next

**Gist:** this note's scoring equations were checked against source functions on synthetic inputs; benchmark results and full scaling experiments were not run. The next stage is inference: how prompt processing, KV caching, sampling, and tools produce the completions we have just evaluated.

### 18.1 What was verified

**Gist:** the checks focus on exact scoring behavior and indexing, where plausible informal descriptions can hide important differences.

- Source BPB matched byte-weighted totals with ordinary, special, and ignored targets, including both indexing branches and zero-byte behavior.
- Actual CORE rendering/batching/forward/evaluation functions were exercised with a small synthetic causal probability model and tokenizer. Checks covered next-token loss alignment, padding, MC mean-versus-product ranking, schema scoring, and LM all-token correctness.
- CORE's rank-strided partition and global outcome sum were checked with simulated collectives; no real distributed transport was used.
- Actual chat categorical scoring selected valid letters at each prompt's genuine end even when a non-letter logit was larger. The generative loop returned two solved problems out of three for the constructed two-attempt example.
- The actual completion renderer removed the reference without mutating the original task conversation. GSM8K extraction/string comparison and HumanEval extraction/program assembly were checked; execution was stubbed rather than running generated programs.
- Centering, ChatCORE arithmetic, multiple-attempt formulas, paired standard error, and the illustrative isoFLOP quadratic were checked numerically. Notebook cells and shell assumptions were inspected without launching them.
- The standalone example, gist paragraphs, source links, and math delimiters were checked using the portable notation from the earlier GitHub rendering fix.

The numerical environment was CPU PyTorch 2.6.0. No evaluation bundle or benchmark dataset was downloaded, no trained checkpoint was scored, and no language-model training/scaling sweep was launched. The synthetic checks establish code behavior and arithmetic, not real benchmark performance or empirical scaling exponents.

### 18.2 Primary references and the next connection

**Gist:** the external references explain the benchmark/scaling ideas; the local implementation defines which versions are actually executed. Inference is the remaining link between a trained probability model and a conversational rollout.

- [DataComp-LM](https://arxiv.org/abs/2406.11794): controlled data experiments and downstream evaluation context for CORE.
- [Evaluating Large Language Models Trained on Code](https://arxiv.org/abs/2107.03374): HumanEval and repeated-sampling evaluation.
- [HumanEval reference evaluator](https://github.com/openai/human-eval/blob/master/human_eval/evaluation.py): the combinatorial pass@$k$ estimator.
- [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556): the model-size/data allocation question under a compute constraint.

The inference note can now follow one concrete question: given a prompt and a trained checkpoint, how do we produce one or several responses efficiently while preserving cache positions, sampling semantics, stopping behavior, and tool-result insertion? That machinery will also be reused in the later SFT/RL pipeline discussions.

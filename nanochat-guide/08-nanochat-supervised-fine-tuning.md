# Supervised fine-tuning in nanochat: from a base model to an assistant

Pretraining taught the model to continue text. This note explains how **supervised fine-tuning (SFT)** teaches it to continue a conversation in the assistant's role: answer a user, follow a response format, request a calculator operation, use the returned result, and stop.

The architecture and next-token cross-entropy are familiar. The new ideas are the **distribution of examples**, their **conversation format**, and **which token predictions receive a loss**. Most of the conceptual work is understanding those choices before following the training loop.

The source is nanochat commit `92d63d4e8bb4df75c3b71618f31ddde2378b2bcd`; comparisons use your `transformer-from-scratch` at `56092a1818027c800f08167ec0c1c65b983711e1`. Links point to exact lines in those local checkouts. External dataset pages describe their published contents; the task loader does not pin a dataset revision.

The main prerequisites are the [pretraining](03-nanochat-pretraining.md), [optimization](04-nanochat-optimization-and-distributed-training.md), and [inference](07-nanochat-inference-and-generation.md) notes. We reuse their model and optimizer mathematics rather than deriving the transformer or Muon again.

**How to skim:** every numbered section and subsection starts with a **Gist**. Sections 1–7 explain the learning problem, data, masks, and tools. Sections 8–12 follow packing, initialization, updates, evaluation, and saved artifacts. Section 13 compares your implementation. Section 14 groups implementation caveats, so they do not obscure the central SFT ideas. Sections 15–16 offer small checks and connect this stage to RL.

This is an explanation of the inspected implementation. Neither source repository has been modified, and no training run or dataset download was performed to prepare this note.

## Contents

1. [What SFT changes](#1-what-sft-changes)
2. [The implementation map](#2-the-implementation-map)
3. [The actual training mixture](#3-the-actual-training-mixture)
4. [Rendering a conversation into tokens](#4-rendering-a-conversation-into-tokens)
5. [The loss mask and the one-token shift](#5-the-loss-mask-and-the-one-token-shift)
6. [The objective and its normalization](#6-the-objective-and-its-normalization)
7. [How calculator use becomes a supervised example](#7-how-calculator-use-becomes-a-supervised-example)
8. [Packing conversations into fixed-size rows](#8-packing-conversations-into-fixed-size-rows)
9. [Loading the base model and optimizer](#9-loading-the-base-model-and-optimizer)
10. [Learning rates and one optimizer update](#10-learning-rates-and-one-optimizer-update)
11. [Evaluating the assistant during SFT](#11-evaluating-the-assistant-during-sft)
12. [What the checkpoint contains](#12-what-the-checkpoint-contains)
13. [Comparison with your implementation](#13-comparison-with-your-implementation)
14. [Implementation caveats in this checkout](#14-implementation-caveats-in-this-checkout)
15. [Small experiments and checks performed](#15-small-experiments-and-checks-performed)
16. [What changes when we move to RL](#16-what-changes-when-we-move-to-rl)

## 1. What SFT changes

**Gist:** SFT starts with pretrained weights and increases the probability of demonstrated assistant responses given their conversation context. It uses supervised next-token prediction, with a mask selecting the assistant's contributions.

### 1.1 A base model already has the machinery to converse

**Gist:** no new attention mechanism or conversational network is introduced. The same GPT is trained on examples that identify who is speaking and what a useful assistant continuation looks like.

A base language model defines a distribution over token sequences:

$$
p_\theta(z_0,\ldots,z_{M-1})
=\prod_{i=0}^{M-1}p_\theta(z_i\mid z_{<i}).
$$

It can continue many kinds of text: an article, dialogue, code, a question, or an answer. Pretraining alone does not specify that, after a particular user message, it should give a useful answer in a designated assistant role and then stop.

SFT supplies examples of that behavior. For a simple prompt $x$ and demonstrated answer $y$, including its end marker, its core objective is

$$
-\log p_\theta(y\mid x)
=-\sum_{j=0}^{|y|-1}\log p_\theta(y_j\mid x,y_{<j}).
$$

The prompt is part of the conditioning information. Its own likelihood need not be optimized by this objective.

All of GPT's existing parameters remain eligible for updates: embeddings, attention and MLP matrices, output head, value embeddings, and scalar mixing parameters. This script performs full-parameter fine-tuning. It does not introduce adapters, LoRA matrices, a frozen backbone, or a classifier head. See [model loading](../nanochat/scripts/chat_sft.py#L94) and [the complete parameter grouping](../nanochat/nanochat/gpt.py#L419).

### 1.2 Why demonstrations can change behavior

**Gist:** the training examples define the behavior being imitated. Cross-entropy pushes the model toward their conditional token distributions; its usefulness depends on the quality, coverage, and weighting of those demonstrations.

For a fixed context $h$, let $q(v\mid h)$ be the demonstration distribution over next tokens. The expected token loss is

$$
H(q,p_\theta)
=-\sum_v q(v\mid h)\log p_\theta(v\mid h)
=H(q)+D_{\mathrm{KL}}(q\Vert p_\theta).
$$

Since $H(q)$ does not depend on $\theta$, minimizing this cross-entropy minimizes a forward KL divergence at the contexts supplied by the data. In a finite dataset, each recorded continuation acts as a target sample; there need not be several answers for the same prompt.

This explains both the attraction and limitation of SFT. Good demonstrations give a direct signal for desired behavior. Incorrect answers, unwanted styles, and narrow coverage also become targets. The equation contains no independent test that the demonstration is true or optimal.

The [InstructGPT paper](https://arxiv.org/abs/2203.02155) is a useful historical reference: it distinguishes supervised training on demonstrations from later training using ranked outputs and human feedback. Nanochat's SFT stage uses its own dataset mixture, described below; the presence of SFT does not imply that a full RLHF pipeline has been implemented.

### 1.3 Teacher forcing is still the training method

**Gist:** training predicts each demonstrated answer token while seeing the true preceding answer tokens. It does not generate an answer and then compare the whole generated string with the reference.

Suppose the reference answer is “The answer is 5.” During training, the model predicts “answer” after seeing the reference prefix “The”; it does not first sample an alternative word and then continue from that sampled word. Causal masking lets the predictions for all positions be computed in parallel.

At inference, the next context contains the model's own earlier choices. Therefore an error can move generation into contexts that are poorly represented by the teacher-forced examples. SFT trains behavior on demonstrated histories; it does not directly optimize the success rate of complete sampled responses. That distinction will motivate RL.

## 2. The implementation map

**Gist:** the trainer composes a small number of familiar pieces: load a base checkpoint, construct conversation tasks, render tokens and masks, pack rows, compute masked cross-entropy, and update with Muon/AdamW.

The most useful entry points are:

| Responsibility | Exact entry point |
| --- | --- |
| Load base model and inherit settings | [chat_sft.py, line 94](../nanochat/scripts/chat_sft.py#L94) |
| Construct/warm-start optimizer | [chat_sft.py, line 132](../nanochat/scripts/chat_sft.py#L132) |
| Construct training and validation mixtures | [chat_sft.py, line 161](../nanochat/scripts/chat_sft.py#L161) |
| Mix task rows | [TaskMixture](../nanochat/tasks/common.py#L129) |
| Serialize messages and produce loss mask | [render_conversation](../nanochat/nanochat/tokenizer.py#L140) |
| Pack and shift examples | [sft_data_generator_bos_bestfit](../nanochat/scripts/chat_sft.py#L180) |
| Compute masked cross-entropy | [GPT.forward, loss branch](../nanochat/nanochat/gpt.py#L517) |
| Accumulate gradients and update | [chat_sft.py, line 425](../nanochat/scripts/chat_sft.py#L425) |
| Evaluate and save | [evaluation](../nanochat/scripts/chat_sft.py#L339), [checkpoint saving](../nanochat/scripts/chat_sft.py#L391) |

The data path is:

```text
Hub dataset rows
  -> task-specific conversation dictionaries
  -> one shuffled TaskMixture index
  -> rendered token IDs + per-token supervision mask
  -> best-fit packing + BOS padding
  -> shifted inputs and targets, ignored targets = -1
  -> ordinary GPT forward and masked cross-entropy
  -> accumulated gradients -> MuonAdamW update
```

The inference engine is used for sampled evaluation. It is not needed to produce the teacher-forced training targets.

## 3. The actual training mixture

**Gist:** this checkout combines general conversations, multiple-choice answers, and grade-school math solutions with calculator annotations. It increases MMLU and GSM8K exposure by putting repeated copies of their task views into a finite shuffled mixture.

### 3.1 Where the data comes from

**Gist:** SFT uses different datasets from the ClimbMix pretraining corpus. Each task converts a public dataset's records into a common conversation representation.

**Code:** [training mixture](../nanochat/scripts/chat_sft.py#L162), [SmolTalk adapter](../nanochat/tasks/smoltalk.py#L12), [MMLU adapter](../nanochat/tasks/mmlu.py#L13), [GSM8K adapter](../nanochat/tasks/gsm8k.py#L38).

| Task | Published source | Training split | What becomes the assistant target |
| --- | --- | --- | --- |
| SmolTalk | [HuggingFaceTB/smol-smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk) | `train` | Existing assistant messages in general conversations |
| MMLU | [cais/mmlu](https://huggingface.co/datasets/cais/mmlu) | `all / auxiliary_train` | The correct letter, `A`, `B`, `C`, or `D` |
| GSM8K | [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) | `main / train` | Worked solution, tool requests, and final numerical answer |

The SmolTalk dataset card identifies the smaller mixture as roughly 460K training conversations and includes synthetic data. “Supervised” here means there are supplied targets; it does not mean every answer was written by a human. Its examples include ordinary assistance, rewriting, constraints, and code-related requests. [Published dataset card](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk).

The task loader [load_hub_dataset](../nanochat/tasks/common.py#L45) queries Hugging Face's Parquet export API, downloads the shards, writes a manifest, and reads them through PyArrow. It caches them below

```text
<nanochat base directory>/task_data/<dataset-name-with-slashes-replaced>/<subset>/<split>/
```

A file lock coordinates first-time access by several ranks. This is a separate cache from pretraining's ClimbMix shards. The dataset identifiers and splits are explicit, but a commit/revision hash is not passed to the Hub API; fully reproducible data provenance would also record the actual dataset revision or downloaded artifacts.

### 3.2 Repetition implements oversampling

**Gist:** the code makes three logical copies of MMLU and four of GSM8K by default, then shuffles the combined row index. It is a finite mixture containing duplicates, rather than a fresh weighted source draw for every batch.

The defaults are `mmlu_epochs=3` and `gsm8k_epochs=4`. Let the numbers of original training rows be $n_S,n_M,n_G$. The mixture contains

$$
N_{\mathrm{mix}}=n_S+3n_M+4n_G
$$

logical row occurrences. [TaskMixture.__init__](../nanochat/tasks/common.py#L135) creates every `(task_idx, local_idx)` pair and shuffles that list using `random.Random(42)`.

If we chose an index uniformly from this finite list, its source probabilities would be

$$
P(S)=\frac{n_S}{N_{\mathrm{mix}}},\qquad
P(M)=\frac{3n_M}{N_{\mathrm{mix}}},\qquad
P(G)=\frac{4n_G}{N_{\mathrm{mix}}}.
$$

Using the source's rounded counts, about 460K, 100K, and 7.5K rows respectively, these are roughly 58%, 38%, and 4%. These are **row-occurrence proportions**, not proportions of computed tokens or supervised answer tokens.

The repeated adapters use the same seed for their dataset shuffles. They do not generate three new correct answers to an MMLU question or four new solutions to a GSM8K problem. The same examples are exposed repeatedly.

This is the concrete version of the mixture discussion from the data note: repetition changes a finite dataset's proportions before traversal. The actual packer then reorders buffered examples by length, so batch order is not a sequence of independent uniform random draws.

### 3.3 What each task teaches

**Gist:** the sources supply different kinds of behavioral supervision. A long multi-turn conversation, a one-letter answer, and a worked solution have very different amounts and structures of supervision.

SmolTalk supplies ordinary alternating user/assistant turns. Earlier assistant responses remain in the context, and every retained assistant message is supervised. The renderer does not restrict training to the last answer of a multi-turn conversation.

MMLU [builds a multiple-choice user message](../nanochat/tasks/mmlu.py#L28) and supplies just the correct letter as the assistant message. [render_mc](../nanochat/tasks/common.py#L187) places each choice before its letter, such as `- choice text=A`, and asks for only a letter. This formatting also avoids changing the letter's tokenization through leading whitespace. The source's claims about what smaller models prefer are empirical formatting choices, not mathematical necessities.

GSM8K [splits its worked answer into parts](../nanochat/tasks/gsm8k.py#L51). Text is ordinary assistant content; annotations such as `<<2+3=5>>` become a tool request `2+3` and an externally supplied result `5`. Section 7 follows those parts through the mask.

Repeating a short-answer task changes how often its questions appear, but it does not automatically give that task a chosen fraction of the total training gradient. Answer lengths, truncation, packing, and the loss normalization in section 6 also matter.

## 4. Rendering a conversation into tokens

**Gist:** a conversation is serialized into a single token stream with role and tool delimiters. A parallel mask records which stream elements are predictions the assistant should learn to make.

### 4.1 The basic role format

**Gist:** special tokens mark message boundaries; they are part of the model's context and output vocabulary. A simple user/assistant exchange becomes one causal sequence.

**Code:** [special-token definitions](../nanochat/nanochat/tokenizer.py#L9), [role serialization](../nanochat/nanochat/tokenizer.py#L176).

For a user asking “Hi” and an assistant answering “Hello”, the schematic stream is

```text
<|bos|>
<|user_start|>Hi<|user_end|>
<|assistant_start|>Hello<|assistant_end|>
```

The line breaks above are for readability; the renderer concatenates the components without inserting those line breaks. The words may occupy several BPE tokens. Each special marker has its own reserved token ID.

These special tokens were reserved when nanochat's tokenizer was created. SFT uses that existing vocabulary and output head; it does not resize the model to add roles. Reserving an ID does not mean that its useful conversational meaning was learned during ordinary document pretraining. Conversely, an output row can receive softmax gradients even when its token is never the target, so “previously unused role token” should not be interpreted as “its parameters could never have changed.”

### 4.2 The renderer enforces a particular conversation schema

**Gist:** this is a small, explicit chat format. It expects alternating user and assistant messages, with optional tool parts inside an assistant message.

After handling an optional initial system message, [the role assertions](../nanochat/nanochat/tokenizer.py#L180) require

```text
user, assistant, user, assistant, ...
```

User content must be a string. Assistant content can be a string or a list of `text`, `python`, and `python_output` parts. Arbitrary role sequences or a separate generic `tool` role are not supported by this renderer.

An initial system message is [merged into the first user message](../nanochat/nanochat/tokenizer.py#L157):

```text
system text + "\n\n" + first user text
```

The renderer deep-copies before that merge. There is no separate system-role token in this scheme. Consequently, system content is context under the same zero-loss mask as the first user message; the serialization does not encode a separate privileged role boundary for it.

### 4.3 Training and inference share the assistant boundary

**Gist:** during SFT the complete reference answer follows `<|assistant_start|>`. During generation the prompt ends at that marker, and the model supplies the continuation.

[render_for_completion](../nanochat/nanochat/tokenizer.py#L241) copies the conversation, removes its final reference assistant message, renders the remaining context, and appends `<|assistant_start|>`.

This makes the first generated answer token correspond to the first supervised answer token in SFT. Keeping this boundary identical matters: a different inference template can condition the model on a prefix it was not trained to interpret in the same way.

## 5. The loss mask and the one-token shift

**Gist:** the renderer's mask belongs to tokens being predicted. Because an input position predicts the following token, the training code must shift the mask together with the targets.

### 5.1 Which tokens are supervised?

**Gist:** assistant text, calculator requests, and the assistant end marker receive loss. User content, role-opening markers, and calculator results provide context without receiving their own next-token prediction loss.

The exact mask assignments in [render_conversation](../nanochat/nanochat/tokenizer.py#L189) are:

| Stream component | Mask | Why |
| --- | --- | --- |
| `<|bos|>` | 0 | Start/context boundary supplied by the data path |
| `<|user_start|>`, user text, `<|user_end|>` | 0 | User input is supplied to the assistant |
| `<|assistant_start|>` | 0 | The application supplies the assistant-role opening |
| Assistant text | 1 | The model should produce the demonstrated response |
| `<|python_start|>`, expression, `<|python_end|>` | 1 | The model must choose to request the calculation and specify it |
| `<|output_start|>`, result, `<|output_end|>` | 0 | The external tool supplies these observations |
| `<|assistant_end|>` | 1 | The model should learn when to finish its response |
| BOS padding added by the packer | 0 | Padding is not an answer |

The distinction is **who supplies the token during use**, rather than whether it looks like ordinary prose or a special token. Some special tokens are supervised. The broad comment near the loader's mask application should be read in light of these actual assignments.

### 5.2 A complete seven-token example

**Gist:** predicting the first assistant content token is supervised at the position containing `<|assistant_start|>`. Masking that input position merely because the marker itself has mask zero would lose the first answer prediction.

For this example only, assume `u` and `a` each encode to one token. Let the renderer return tokens $z_i$ and masks $m_i$:

| Rendered index $i$ | $z_i$ | $m_i$ |
| --- | --- | --- |
| 0 | BOS | 0 |
| 1 | user-start | 0 |
| 2 | `u` | 0 |
| 3 | user-end | 0 |
| 4 | assistant-start | 0 |
| 5 | `a` | 1 |
| 6 | assistant-end | 1 |

The shifted training tensors are

```text
input:   [BOS, user-start, u, user-end, assistant-start, a]
target:  [ -1,         -1, -1,       -1,               a, assistant-end]
```

At input position 4, the model has seen assistant-start and should predict `a`. At position 5, it has seen `a` and should predict assistant-end. There is no loss for predicting assistant-start itself.

In general, for a row of $T+1$ tokens,

$$
X_t=z_t,\qquad
Y_t=\begin{cases}
z_{t+1},&m_{t+1}=1,\\
-1,&m_{t+1}=0,
\end{cases}
\qquad 0\leq t<T.
$$

[The loader](../nanochat/scripts/chat_sft.py#L280) implements this with `batch[:, :-1]`, `batch[:, 1:]`, and `mask[:, 1:]`. GPT's cross-entropy uses `ignore_index=-1`.

### 5.3 Zero target loss does not freeze the context

**Gist:** the assistant's losses still backpropagate through the representations of preceding user and tool tokens. The mask removes selected local output losses; it does not remove those tokens from the computation graph.

If an answer representation $h_t$ attends to an earlier user representation $h_s$, an answer loss can contribute

$$
\frac{\partial\ell_t}{\partial\theta}
=\frac{\partial\ell_t}{\partial h_t}
\frac{\partial h_t}{\partial h_s}
\frac{\partial h_s}{\partial\theta}
+\text{other paths}.
$$

The user token's embedding and the weights processing that context can therefore be trained even though predicting the user token is not an SFT target. The same reasoning applies to a calculator result that informs the final answer.

There are three distinct mechanisms:

- A **loss mask** chooses which output predictions contribute cross-entropy.
- An **attention mask** chooses which context positions can influence each representation.
- **Parameter freezing** chooses which weights are eligible for updates.

Nanochat's SFT mask implements the first. Its causal/local attention still controls information flow, and the trainer does not freeze parameters.

## 6. The objective and its normalization

**Gist:** each microbatch loss is averaged over its nonignored target tokens. The trainer then averages these microbatch means through gradient accumulation and distributed gradient reduction. When valid-label counts differ, this differs from one mean over all valid tokens in the global batch.

### 6.1 Masked conditional cross-entropy

**Gist:** compute the familiar next-token cross-entropy, discard prompt/tool-result/padding targets, and average what remains. The forward model and logit softcap are unchanged.

For row $k$, prediction position $t$, and vocabulary item $v$, write the raw output-head logit as $u_{k,t,v}$. Nanochat uses

$$
s_{k,t,v}=15\tanh(u_{k,t,v}/15),\qquad
p_{k,t,v}=\frac{e^{s_{k,t,v}}}{\sum_j e^{s_{k,t,j}}}.
$$

Let $q_{k,t}=\mathbf{1}[Y_{k,t}\neq-1]$ and let $Z=\sum_{k,t}q_{k,t}$ be the microbatch's number of valid labels. For $Z>0$,

$$
\mathcal L_{\mathrm{micro}}
=\frac{1}{Z}\sum_{k,t:q_{k,t}=1}-\log p_{k,t,Y_{k,t}}.
$$

[GPT.forward](../nanochat/nanochat/gpt.py#L510) returns exactly this mean through PyTorch's class-index cross-entropy with ignored targets. The denominator counts valid labels; it is not the full rectangular tensor size $bT$.

For a valid label, the raw-logit derivative is

$$
\frac{\partial\mathcal L_{\mathrm{micro}}}{\partial u_{k,t,v}}
=\frac{1}{Z}
\left(p_{k,t,v}-\mathbf{1}[v=Y_{k,t}]\right)
\left(1-\tanh^2(u_{k,t,v}/15)\right).
$$

For an ignored position, this direct output-logit derivative is zero. Section 5.3 explains why that does not imply zero gradient through its contextual representations.

### 6.2 Token counts measure several different things

**Gist:** the global token batch counts all computed positions, including prompts and padding. The number of supervised labels is smaller and variable, so a fixed token budget is not a fixed amount of answer supervision.

Use the same notation as the pretraining note:

| Symbol | Meaning |
| --- | --- |
| $b$ | Number of packed rows in one microbatch on one rank |
| $T$ | Number of input positions per row |
| $W$ | Number of distributed processes/ranks |
| $A$ | Number of microbatches accumulated per optimizer update |
| $G=WAbT$ | Global computed token positions per optimizer update |
| $Z_{r,a}$ | Valid assistant target tokens in microbatch $a$ on rank $r$ |

Here $a$ indexes microbatches; $b$ is the number of rows inside each microbatch. [The trainer computes](../nanochat/scripts/chat_sft.py#L121)

$$
A=\frac{G}{WbT},
$$

requiring an integer result. For example, $G=524288$, $W=8$, $b=16$, and $T=2048$ give $A=2$.

If that update contains 100K valid assistant targets, it still computes 524288 positions. Prompt representations are useful computation; padded positions are overhead. Neither should be confused with additional labeled answers.

### 6.3 A mean of means is not always a global token mean

**Gist:** averaging equally sized pretraining microbatches also averages their tokens equally. In masked SFT, equally sized tensors can contain different numbers of valid labels, so averaging their means gives each microbatch equal weight instead.

[The microbatch loop](../nanochat/scripts/chat_sft.py#L425) divides each mean loss by $A$. The optimizer's distributed reductions average gradients over $W$ ranks. Assuming identical model replicas when gradients are evaluated, the resulting gradient corresponds to

$$
\mathcal L_{\mathrm{script}}
=\frac{1}{WA}\sum_{r=0}^{W-1}\sum_{a=0}^{A-1}
\frac{1}{Z_{r,a}}\sum_{j\in\mathcal S_{r,a}}\ell_j,
$$

where $\mathcal S_{r,a}$ is that microbatch's set of valid labels. By contrast, a single mean over all valid labels would be

$$
\mathcal L_{\mathrm{token}}
=\frac{\sum_{r,a}\sum_{j\in\mathcal S_{r,a}}\ell_j}
{\sum_{r,a}Z_{r,a}}.
$$

These coincide when all $Z_{r,a}$ are equal. For a simple counterexample, suppose two microbatches have these losses:

```text
microbatch 1: one valid token, loss 4
microbatch 2: three valid tokens, losses 1, 1, 1
```

Then

$$
\mathcal L_{\mathrm{script}}=\frac{4+1}{2}=2.5,
\qquad
\mathcal L_{\mathrm{token}}=\frac{4+1+1+1}{4}=1.75.
$$

Under the script's normalization, each label in a sparse microbatch receives a larger coefficient than each label in a dense microbatch. This is a property of the implemented objective, rather than an algebraic error in accumulation. It also means changing the grouping into microbatches can change the gradient even with the same set of examples.

If one wanted a global valid-token mean, the accumulation/reduction weights would have to account for the $Z_{r,a}$ counts. That is a different implementation choice; this note does not change the trainer.

### 6.4 Row weighting, answer-length weighting, and task weighting

**Gist:** repeating a dataset controls its row multiplicity. The amount of supervision also depends on how many answer tokens survive rendering and how the microbatch means are formed.

With a global valid-token mean, a conversation with 200 supervised tokens contributes 100 times as many token-loss terms as one with two. A conversation-level mean would instead normalize each conversation first. Neither convention follows automatically from the phrase “SFT.”

Nanochat normalizes after several conversations have been packed into a microbatch. Within that microbatch, longer retained assistant responses supply more terms; across microbatches, section 6.3's equal-mean weighting applies.

For a rough intuition under token weighting, a task's label exposure scales like

$$
\text{multiplicity}\times\text{row count}\times
\mathbb E[\text{retained supervised tokens per row}].
$$

For this script, that expression is only an exposure estimate, not an exact formula for the task's share of the gradient. Length-dependent packing, truncation, and the per-microbatch denominator all intervene. This is why “MMLU is 38% of mixture rows” does not establish “MMLU is 38% of the optimization signal.”

## 7. How calculator use becomes a supervised example

**Gist:** the model learns to request a calculation and continue from its result. It is not required to predict the externally computed result as an action of its own.

### 7.1 Turning a GSM8K annotation into actions and observations

**Gist:** the adapter separates the expression from its supplied result before tokenization. The SFT data path reads these annotations; it does not run the calculator to create the training answer.

Take this small GSM8K-style solution:

```text
Compute <<2+3=5>>5.
#### 5
```

[GSM8K.get_example](../nanochat/tasks/gsm8k.py#L51) splits `<<...>>` spans, removes the angle delimiters, and splits the interior at its last `=`. It produces assistant parts equivalent to

```python
[
    {"type": "text",          "text": "Compute "},
    {"type": "python",        "text": "2+3"},
    {"type": "python_output", "text": "5"},
    {"type": "text",          "text": "5.\n#### 5"},
]
```

The renderer then produces the schematic assistant stream:

```text
assistant-start                         mask 0
Compute                                 mask 1
python-start  2+3  python-end           mask 1 throughout
output-start  5    output-end          mask 0 throughout
5.\n#### 5                              mask 1
assistant-end                           mask 1
```

The `5` inside the tool result is masked. The `5` in the final assistant answer is supervised. Their textual equality does not imply equal roles in learning.

### 7.2 Why tool outputs have zero loss

**Gist:** tool results are observations supplied by the environment. Supervision is directed at the assistant's choices before and after those observations.

If the trajectory contains assistant action tokens $a_j$ interleaved with external observations $o_j$, the supervised likelihood uses the factors

$$
\prod_{j\in\text{assistant positions}}
p_\theta(a_j\mid\text{preceding actions and observations}).
$$

It does not include a model likelihood term for every observation merely because observations occupy positions in the token stream. This is the same distinction we saw in the inference engine between sampled tokens and forced tokens.

[Engine.generate](../nanochat/nanochat/engine.py#L252) detects the closing Python marker, evaluates the expression, and queues the output markers/result as forced tokens. SFT teaches a compatible pattern: request an expression, read an output, and use it in further text.

The training example supplies the reference expression and reference result. Consequently SFT by itself does not measure whether the model would choose a useful expression, avoid malformed requests, or recover from a different tool result when sampling. Those are rollout-level questions for evaluation or later RL.

### 7.3 Learning to stop is part of the objective

**Gist:** the final assistant marker is a supervised target. It teaches a termination decision that ordinary answer-text imitation would otherwise omit.

For answer tokens $y_0,\ldots,y_{n-1}$ and end marker $e$, the response likelihood includes

$$
\left[\prod_{j=0}^{n-1}p_\theta(y_j\mid x,y_{<j})\right]
p_\theta(e\mid x,y_{0:n-1}).
$$

Dropping the final factor would train the answer's content without explicitly rewarding termination at its demonstrated boundary. Including it gives a direct stop signal. In a one-letter MMLU answer, the letter and end marker together form a very short supervised continuation.

At inference, [the engine stops on assistant-end or BOS](../nanochat/nanochat/engine.py#L248). Those two inference stop conditions are not equivalent SFT targets: assistant-end is supervised, while BOS remains masked by this renderer.

## 8. Packing conversations into fixed-size rows

**Gist:** variable-length conversations are combined into rectangular tensors so the GPU can process them efficiently. The packer preserves each rendered conversation it selects and pads unused space; that does not eliminate truncation earlier in rendering.

### 8.1 The best-fit procedure

**Gist:** keep a buffer of rendered conversations, repeatedly take the longest one that fits the remaining row space, and pad when none fits. A row stores $T+1$ tokens so it can yield $T$ input/target pairs.

**Code:** [generator setup](../nanochat/scripts/chat_sft.py#L180), [buffer refill](../nanochat/scripts/chat_sft.py#L204), [best-fit selection](../nanochat/scripts/chat_sft.py#L229).

Each rank has its own buffer, with default capacity 100 conversations. Its fetch cursor starts at its rank number and advances by world size $W$. For each of the $b$ output rows:

1. Start an empty row with capacity $T+1$.
2. Fill the conversation buffer as needed.
3. Find the longest buffered conversation whose full rendered length fits the remaining space.
4. Append its IDs and mask, remove it from the buffer, and repeat.
5. If none fits, fill the remaining positions with BOS and mask zero.

For capacity 16 and two conversations of length 7, the result is

```text
[ conversation A: 7 tokens ][ conversation B: 7 tokens ][ BOS ][ BOS ]
```

Shifting yields inputs and targets of length 15. The padding targets are ignored, as are the masked user/role tokens inside both conversations. Each selected conversation already begins with BOS, so each output row also begins with BOS when it contains data.

### 8.2 A BOS boundary does not isolate attention

**Gist:** concatenated conversations share the row's causal context. Neither BOS nor the loss mask creates a block-diagonal attention mask or resets positions between conversations.

If conversation B follows A in one row, B's assistant predictions may depend on A's earlier tokens through causal attention, subject to each layer's local/full window. The model also applies its embedding smear across adjacent row positions; inserting BOS does not reset every contextual computation.

Thus the exact training conditional is closer to

$$
p_\theta(\text{B answer token}\mid
\text{earlier packed context},\text{B conversation prefix})
$$

than to a guarantee of conditioning only on B in isolation. BOS marks a boundary the model can learn to interpret. Enforcing independent conversations would require additional attention/position handling that this packer does not supply.

The zero loss on B's BOS means the model is not penalized for failing to predict the transition into that new conversation. It does not mean the earlier conversation becomes invisible afterward.

### 8.3 Padding and truncation are different operations

**Gist:** padding adds ignored positions to a short row; truncation removes part of a long example. This code pads during packing but also truncates every conversation in the renderer.

The packer's docstring says it pads instead of cropping to avoid discarding tokens. That describes the packing step's handling of the already rendered sequences.

However, [render_conversation](../nanochat/nanochat/tokenizer.py#L140) defaults to `max_tokens=2048` and [slices both lists](../nanochat/nanochat/tokenizer.py#L221) to that length. [The SFT loader calls it without an override](../nanochat/scripts/chat_sft.py#L208).

Consequently:

- A conversation longer than 2048 rendered tokens loses its tail before packing, even if the row capacity is larger.
- Truncation can cut through an assistant response or a tool block, or remove the final end marker.
- A sufficiently long user message can consume the whole retained prefix, leaving no supervised assistant targets.
- Setting `--max-seq-len` changes row capacity, not this renderer's default limit.

With the inherited default $T=2048$, a rendered conversation is at most 2048 tokens and fits into a 2049-token row. With a smaller $T$, some rendered conversations can be too large to fit any row. Section 14 treats the consequences for the current buffer logic.

## 9. Loading the base model and optimizer

**Gist:** SFT starts from a pretrained model and usually its optimizer histories. It builds fresh group learning rates, loads optimizer state if available, restores those fresh rates, and then scales them for SFT.

### 9.1 Which settings are inherited?

**Gist:** the default SFT batch and length settings come from checkpoint metadata; the principal learning-rate inputs come from the pretraining script's saved user configuration. Explicit SFT arguments override these inherited values.

[The model is loaded first](../nanochat/scripts/chat_sft.py#L94) from source `"base"`. An explicit `--model-tag` selects a model directory; `--model-step` selects a base checkpoint step. Without them, the loader [selects a model tag and latest saved step](../nanochat/nanochat/checkpoint_manager.py#L148).

[The inheritance table](../nanochat/scripts/chat_sft.py#L98) resolves:

| Setting | Inheritance source | Fallback if absent |
| --- | --- | --- |
| `max_seq_len` | Top-level base metadata | 2048 |
| `device_batch_size` | Top-level base metadata | 32 |
| `total_batch_size` | Top-level base metadata | 524288 |
| `embedding_lr` | Base `user_config` | 0.3 |
| `unembedding_lr` | Base `user_config` | 0.004 |
| `matrix_lr` | Base `user_config` | 0.02 |

Fallback is not necessarily the normal observed value. For example, this checkout's [base pretraining parser](../nanochat/scripts/base_train.py#L63) defaults to `unembedding_lr=0.008`, so a checkpoint recording that input passes 0.008 to SFT rather than the 0.004 fallback.

These inherited LR inputs are not the nearly zero instantaneous rates at the end of pretraining. They are used to construct fresh parameter-group rates. Pretraining's separate batch-size LR multiplier is [applied in base_train](../nanochat/scripts/base_train.py#L287); the SFT setup does not reapply that calculation to its inherited raw inputs.

### 9.2 Fresh parameter-group learning rates

**Gist:** the optimizer retains nanochat's different learning-rate scales for embeddings, output weights, matrices, and scalar mechanisms. The SFT factor multiplies those group rates rather than replacing them with one common LR.

Let $c_d=(d/768)^{-1/2}$ and let $\alpha$ be `init_lr_frac`, default 0.8. Ignoring the later progress multiplier, [GPT.setup_optimizer](../nanochat/nanochat/gpt.py#L419) and [the SFT scaling loop](../nanochat/scripts/chat_sft.py#L156) give:

| Parameter group | SFT initial group LR |
| --- | --- |
| Output head | $\alpha\,\eta_{\mathrm{unembed}}c_d$ |
| Main token embedding | $\alpha\,\eta_{\mathrm{embed}}c_d$ |
| Value embeddings | $\alpha\,\eta_{\mathrm{embed}}c_d/2$ |
| Residual mixing scalars | $\alpha\,(0.5)(0.01)$ |
| Input-residual mixing scalars | $\alpha\,(0.5)$ |
| Smear/backout group | $\alpha\,(0.2)$ |
| Muon matrix groups | $\alpha\,\eta_{\mathrm{matrix}}$ |

Here 0.5 is `setup_optimizer`'s default `scalar_lr`; the SFT call does not pass a separate scalar-LR input. The Muon kernel still applies its shape-dependent update scaling, explained in the optimization note; that factor is not included in this table of stored group rates.

For $d=1536$, $\eta_{\mathrm{embed}}=0.3$, $\eta_{\mathrm{unembed}}=0.008$, and $\eta_{\mathrm{matrix}}=0.02$, the main embedding, head, and matrix rates are approximately 0.1697, 0.00453, and 0.016 before the progress multiplier.

These are empirically chosen scales for different parameter geometries. The factor 0.8 is a recipe choice, not a theorem that SFT should always use 80% of pretraining's rate.

### 9.3 What optimizer warm-start actually restores

**Gist:** loading the optimizer preserves more than momentum. It also restores counters, other state buffers, and parameter-group metadata; the script then explicitly restores only learning rates before resetting their SFT baselines.

The default `load_optimizer=1` loads [the rank's base optimizer shard](../nanochat/scripts/chat_sft.py#L139). The sequence is:

```text
construct fresh SFT optimizer
save its group learning rates
load pretrained optimizer state_dict
restore the saved fresh group learning rates
multiply rates by init_lr_frac
set initial_lr to those scaled rates
```

For AdamW, restored state includes the first moment, second moment, and step counter. For Muon, it includes momentum and its factored variance-related history. Group metadata such as betas and weight decay also comes from the loaded state dictionary. The printed phrase “momentum buffers only” is therefore narrower than the actual operation.

Warm-start can avoid discarding useful running statistics, but it also carries information from a different data distribution. For example, with fixed Adam first-moment coefficient $\beta_1$, after $k$ new gradients,

$$
M_{n+k}
=\beta_1^kM_n
+(1-\beta_1)\sum_{j=1}^k\beta_1^{k-j}G_j^{\mathrm{SFT}}.
$$

Old history decays over updates; it is not instantly converted into an SFT estimate. The restored step counter also affects Adam's bias corrections. Resetting the LR is distinct from resetting these histories.

If the optimizer shard is absent, [the loader returns `None`](../nanochat/nanochat/checkpoint_manager.py#L187), and SFT keeps a fresh optimizer. The expected parameter-group structure and distributed state layout must be compatible; the script does not implement arbitrary optimizer-state resharding across changed world sizes.

### 9.4 “Zero weight decay” needs qualification

**Gist:** passing `weight_decay=0.0` to setup controls fresh Muon groups. AdamW groups have their own nonzero decay constants, and loading old optimizer metadata can restore a nonzero Muon decay too.

[The SFT call](../nanochat/scripts/chat_sft.py#L132) passes `weight_decay=0.0`. But [GPT.setup_optimizer](../nanochat/nanochat/gpt.py#L439) hardcodes AdamW decay values for the head, embeddings, value embeddings, and residual scalars. Other scalar groups have zero decay.

Moreover, `optimizer.load_state_dict` overwrites group metadata. After loading, the script restores rates but does not explicitly reset weight decay. A base checkpoint's Muon decay can therefore survive into SFT.

The pretraining schedule makes Muon decay very small near the end; the last executed update generally uses a value just above zero rather than evaluating the cosine exactly at its terminal boundary. Loading an earlier base checkpoint can carry a larger value. The SFT loop does not run pretraining's decay schedule afterward.

Thus the comment that SFT “continues with zero” should not be interpreted as a guarantee that every group's weight decay is zero in the resulting optimizer.

## 10. Learning rates and one optimizer update

**Gist:** the update mechanics remain accumulated backward passes followed by the mixed optimizer. SFT schedules its rate using approximate data progress because packing makes the number of batches in a traversal hard to predict.

### 10.1 Progress-based warmup, plateau, and warmdown

**Gist:** with defaults, group rates stay at their SFT initial values for the first half of progress, then fall linearly toward zero. Progress is estimated by the loader, so its definition matters.

For progress $p$, warmup fraction $w$, warmdown fraction $h$, and final LR fraction $f$, [get_lr_multiplier](../nanochat/scripts/chat_sft.py#L307) implements, apart from its tiny warmup offset,

$$
\lambda(p)=
\begin{cases}
p/w,&p<w,\\
1,&w\leq p\leq1-h,\\
1-(1-f)\dfrac{p-(1-h)}{h},&p>1-h.
\end{cases}
$$

The group rate is $\eta_g(p)=\eta_{g,\mathrm{initial}}\lambda(p)$. With defaults $w=0,h=0.5,f=0$,

$$
\lambda(p)=\begin{cases}
1,&p\leq0.5,\\
2(1-p),&p>0.5.
\end{cases}
$$

For example, progress 0.75 gives half the SFT initial rates. The purpose is to make smaller late adjustments after broad exposure to the mixture, rather than maintaining equally large steps to the final examples. Its exact shape is an empirical schedule.

Muon momentum uses [a separate update-count schedule](../nanochat/scripts/chat_sft.py#L317):

$$
\mu(k)=0.85+0.10\min(k/300,1),
$$

where $k$ is the optimizer-loop step before the update. Loaded momentum buffers can be retained while the coefficient used to update them is changed. Neither schedule is a new loss term.

### 10.2 The actual update sequence

**Gist:** every microbatch computes masked CE and contributes $1/A$ of its mean gradient. After accumulation, group rates and Muon momentum are set, the optimizer performs its reductions and updates, and gradients are cleared.

**Code:** [initial prefetch](../nanochat/scripts/chat_sft.py#L324), [microbatch loop](../nanochat/scripts/chat_sft.py#L425), [optimizer step](../nanochat/scripts/chat_sft.py#L436).

The loop starts with one prefetched `(x,y)` pair. For each update:

1. Run `loss = model(x, y)`, obtaining masked mean cross-entropy.
2. Divide by $A$ and backpropagate. If using FP16, apply the gradient scaler.
3. Fetch the next microbatch and update the monotonic progress estimate.
4. Repeat until $A$ microbatches have contributed gradients.
5. Use the latest progress to set group LRs, and the update count to set Muon momentum.
6. Call the optimizer, whose internal communication averages gradients and gathers updated parameter shards.
7. Clear gradients, increment the update counter, and log.

The model is compiled with `dynamic=False`. There is no separate FP8 conversion path in this SFT script. It uses the same model precision machinery discussed in the efficiency note and enables a GradScaler for FP16. The loop does not introduce a new gradient-clipping operation.

An ignored target removes its own CE term, but the implementation still computes transformer representations and vocabulary logits for the rectangular batch. The mask is not an optimization that skips prompt forward computation or selectively evaluates the output head.

### 10.3 What progress and epoch mean here

**Gist:** progress estimates packed conversation consumption, while the logged epoch tracks the fetch cursor wrapping. Buffering and prefetching separate both of these from an exact count of completed optimizer updates.

Without a positive `num_iterations`, [the generator](../nanochat/scripts/chat_sft.py#L269) sets approximately

$$
p=\frac{\text{rank-local consumed counter}}{N_{\mathrm{mix}}}.
$$

The counter starts at the rank number and increases by $W$ whenever that rank packs a conversation. This estimates global progress if ranks consume similar numbers of conversations. With unequal lengths and packing, the estimates can differ.

The fetch cursor can wrap while examples remain in the buffer. Therefore the logged `epoch` is not a certificate that every example has already contributed a training loss. Nor is dataset exhaustion a precise, synchronized one-pass traversal across all ranks. Section 14 explains the important consequences, including the positive-iteration flag's actual units.

## 11. Evaluating the assistant during SFT

**Gist:** the script checks both fit to held-out assistant continuations and task-level performance. Those answer different questions, and the selected validation splits overlap some of the benchmark test data used in ChatCORE.

### 11.1 Validation BPB measures supplied answers

**Gist:** validation runs the same masked, teacher-forced data path. Its BPB includes ordinary supervised target tokens and excludes special tokens, so it is not exactly the training CE expressed in another unit.

[The validation mixture](../nanochat/scripts/chat_sft.py#L169) contains:

- SmolTalk `test`;
- the first 5200 rows of the deterministically shuffled MMLU `test` adapter;
- the first 420 rows of the deterministically shuffled GSM8K `test` adapter.

[evaluate_bpb](../nanochat/nanochat/loss_eval.py#L9) sums negative log probabilities for nonignored targets whose token-byte count is positive, and divides by their total byte count:

$$
\mathrm{BPB}_{\mathrm{SFT}}
=\frac{\sum_{j\in\mathcal V}\ell_j}
{\ln2\sum_{j\in\mathcal V}\mathrm{bytes}(Y_j)}.
$$

Here $\mathcal V$ includes valid ordinary target tokens. Special tokens have byte count zero and are excluded from both numerator and denominator. For example, assistant-end and Python delimiters are SFT training targets, but do not contribute to this BPB metric.

Prompt tokens and tool outputs also do not contribute to the target metric because they are ignored by the loss mask. They remain conditioning context. Lower validation BPB means better prediction of supplied ordinary assistant continuations, not necessarily higher success under free generation.

[The evaluation budget](../nanochat/scripts/chat_sft.py#L343) is converted to a number of rectangular batches using $bTW$, so `eval_tokens` is a computed-position budget, not a promised count of assistant target tokens. The validation generator cycles if necessary. Final BPB evaluation is forced by `last_step` even when periodic evaluation is disabled.

### 11.2 ChatCORE measures task behavior

**Gist:** categorical tasks check answer-letter predictions; generative tasks check completed responses. The trainer aggregates baseline-centered accuracies, with small default generative evaluation subsets.

[The SFT evaluation block](../nanochat/scripts/chat_sft.py#L359) uses ARC-Easy, ARC-Challenge, MMLU, GSM8K, and HumanEval. With accuracy $a_i$ and baseline $b_i$,

$$
\mathrm{ChatCORE}
=\frac15\sum_{i=1}^{5}\frac{a_i-b_i}{1-b_i}.
$$

The categorical baselines are 0.25; the generative baselines are zero. `ChatCORE_cat` averages only the three categorical tasks. A centered score can be negative when accuracy is below baseline; the comment describing a strict 0-to-1 range is not a mathematical bound.

The trainer calls [run_chat_eval](../nanochat/scripts/chat_eval.py#L156) with its default one sample per problem, greedy temperature zero, top-k 50, and a 512-new-token budget. Categorical tasks default to all examples; GSM8K and HumanEval default to at most 24 problems each. Those generative estimates are consequently coarse and sensitive to a few successes or failures.

Periodic ChatCORE is skipped at step zero and runs at configured positive intervals and at the end when enabled. Unlike final BPB, it can be disabled by a nonpositive `chatcore_every`.

### 11.3 Validation and final-test independence

**Gist:** the SFT validation answers come from standard MMLU/GSM8K test splits also used by ChatCORE. They are not training-gradient targets in this script, but repeated selection using them would make later scores on those same examples development results.

The training mixture uses MMLU auxiliary training and GSM8K training, not their test splits. However, the validation prefixes above are subsets of the test adapters used by [run_chat_eval](../nanochat/scripts/chat_eval.py#L161). With the default ordering, the short GSM8K generation subset also lies inside the validation prefix.

There is no backward pass through validation. Nevertheless, if one chooses mixtures, learning rates, or stopping rules by repeatedly inspecting these scores, that information influences the experiment. It would then be inaccurate to describe a score on the overlapping examples as an untouched final test.

The script tracks the minimum observed BPB for logging. It does not automatically restore or save a best-BPB model. To assess retained general language modeling as well as assistant behavior, one would also compare suitable pretraining-domain validation metrics; the SFT loop's conversation BPB alone cannot establish that nothing was forgotten.

## 12. What the checkpoint contains

**Gist:** the final SFT artifact is another ordinary GPT checkpoint plus optimizer shards and metadata. The same inference engine can load it; conversation ability resides in the weights and the agreed template, rather than in a different model class.

### 12.1 The saved artifact and later loading

**Gist:** rank zero saves model weights and JSON metadata; every rank saves its optimizer shard. The source name `sft` selects the SFT checkpoint directory for evaluation or generation.

[The end-of-run save](../nanochat/scripts/chat_sft.py#L392) writes beneath

```text
<nanochat base directory>/chatsft_checkpoints/<model-tag-or-dDEPTH>/
```

[save_checkpoint](../nanochat/nanochat/checkpoint_manager.py#L41) produces files of the form

```text
model_000123.pt
meta_000123.json
optim_000123_rank0.pt
optim_000123_rank1.pt
...
```

The number is the SFT loop's completed optimizer-step counter. The metadata contains final validation BPB, model configuration, and a saved user configuration. [load_model("sft", ...)](../nanochat/nanochat/checkpoint_manager.py#L163) routes later consumers to these files.

This checkpoint can be evaluated or passed to the CLI/engine discussed in the inference note. The [speedrun](../nanochat/runs/speedrun.sh#L75) launches SFT after base training and then evaluates the SFT checkpoint. RL is a separate continuation, not launched by those SFT commands.

### 12.2 Saving optimizer state is not a complete SFT resume protocol

**Gist:** this script always initializes from source `base`; it does not expose a path that restores the entire SFT loop and loader state from a saved SFT checkpoint.

The metadata does not contain the current packing buffer, cursor/consumption counters, prefetched batch, RNG states, scaler state, or a complete loop/scheduler continuation state. Also, `user_config` [was copied immediately after argument parsing](../nanochat/scripts/chat_sft.py#L67), before inherited `None` values were resolved.

Therefore “an SFT checkpoint contains optimizer state” and “this script can exactly resume interrupted SFT” are different claims. The former is true. The latter would require additional restore logic and saved state. Changing world size also requires attention to sharded optimizer compatibility, rather than merely loading a different number of shard files.

## 13. Comparison with your implementation

**Gist:** you already implemented the model/gradient machinery needed for SFT. The biggest additions are a conversation-aware data path and exact control of target supervision, followed by post-training evaluation and checkpoint handling.

Your relevant anchors are [get_batch](../transformer-from-scratch/data.py#L26), [the accumulated training loop](../transformer-from-scratch/train.py#L80), [MyGPT.forward](../transformer-from-scratch/model.py#L280), and [configure_optimizers](../transformer-from-scratch/model.py#L300).

| Part | Your current code | Nanochat SFT |
| --- | --- | --- |
| Initialization | Construct a new `MyGPT` | Load base GPT weights |
| Data | Random windows of a character-encoded Shakespeare stream | Shuffled task mixture of formatted conversations |
| Boundaries | Text-window boundaries | Explicit user/assistant/tool markers |
| Targets | Every next character in each window | Assistant action/content/end targets; other positions ignored |
| Loss denominator | Every target position | Valid targets inside each microbatch |
| Packing | Fixed windows from one stream | Full rendered conversations fitted into rows, plus padding |
| Optimizer | AdamW groups | Mixed Muon/AdamW, optionally warm-started |
| Other objective terms | MoE load-balancing term | No MoE or router auxiliary loss in this GPT |
| Evaluation | Next-character validation loss | Masked conversation BPB and ChatCORE tasks |

Your cross-entropy call currently uses its default ignore index rather than explicitly accepting nanochat's `-1`. Passing nanochat-style targets unchanged to it would not implement the intended mask. A future SFT adaptation would need an agreed ignore value and matching loss behavior.

Your MoE auxiliary objective is a separate issue. Masking cross-entropy does not automatically mask router statistics or a load-balancing penalty; one would need to decide and describe their treatment separately. The prompt still passes through the experts and affects contextual computation.

Also, your existing character vocabulary does not already contain nanochat's reserved BPE role IDs. A chat format would have to be defined consistently with your tokenizer and model dimensions. These are comparisons and possible design choices; neither repository is changed here.

## 14. Implementation caveats in this checkout

**Gist:** the central SFT method is straightforward, but several details of this particular trainer do not match its comments or CLI descriptions. These observations come from source inspection and small isolated checks; they are not claims that a full training run was benchmarked or repaired.

### 14.1 Oversized or unsupervised examples can produce empty losses

**Gist:** the buffer does not remove conversations that cannot fit any row. If it fills with oversized examples, the loader can emit only padding without advancing consumption; even otherwise fitting examples can have no answer labels after truncation.

[The best-fit loop](../nanochat/scripts/chat_sft.py#L231) only pops an example when it fits. If all buffered lengths exceed $T+1$, none is removed, no new examples enter that full buffer, and each row becomes BOS padding. The default consumption-based stopping counter does not advance.

The resulting targets are all `-1`. Mean cross-entropy over an empty set of valid labels is not a useful zero-loss training example; the checked PyTorch execution returns `NaN`. The trainer has no explicit valid-label-count guard.

This failure is particularly relevant after reducing `max_seq_len` without changing the renderer's 2048-token default. Separately, a long user-only retained prefix can create an all-ignored example even if its length fits. A robust data path would explicitly handle both maximum length and presence of usable supervised labels.

### 14.2 The positive iteration flag counts loader yields

**Gist:** `--num-iterations` is described as optimizer steps, but its stopping counter is inside the data generator and advances on every yielded microbatch. Prefetching introduces an additional offset.

[The generator increments `it`](../nanochat/scripts/chat_sft.py#L263) once per batch, while [the training loop](../nanochat/scripts/chat_sft.py#L425) consumes $A$ batches per update and fetches the next one after every backward. It does not interrupt accumulation immediately when `last_step` becomes true.

Assuming the dataset-consumption condition does not trigger earlier, for a positive limit $K$ the number of completed optimizer updates is

$$
U=\left\lceil\frac{K-1}{A}\right\rceil.
$$

For $K=1$, the initial prefetch already requests termination, so zero updates occur. For $K=4,A=2$, two updates occur, involving batches 1–4 and leaving batch 5 prefetched.

Moreover, [the consumed-dataset stopping check](../nanochat/scripts/chat_sft.py#L275) still applies with a positive $K$. Supplying a very large iteration count does not bypass the dataset-consumption stop to guarantee several complete mixture traversals.

### 14.3 Prefetching can push the LR schedule beyond its range

**Gist:** the LR uses the progress of the newest prefetched batch, not just the batches already backpropagated. Progress can exceed one, and the schedule does not clamp it.

For the $K=4,A=2$ example, the progress used for updates is:

| Update | Backpropagated batches | Most recently prefetched batch | Progress used | Default LR multiplier |
| --- | --- | --- | --- | --- |
| 1 | 1, 2 | 3 | $3/4$ | 0.5 |
| 2 | 3, 4 | 5 | $5/4$ | -0.5 |

The second multiplier follows directly from $\lambda(p)=2(1-p)$. [The source assigns it to the group LR](../nanochat/scripts/chat_sft.py#L439); the custom optimizer consumes that value. It is therefore unsafe to infer the intended finite-horizon behavior from the CLI help or from a schedule sketch restricted to $0\leq p\leq1$.

Consumption-based progress can also overshoot a boundary when several examples are packed at once. The source has no general clamp to keep progress within one before computing the rate. This is an implementation boundary issue, not a property SFT mathematically requires.

### 14.4 Rank-local progress can produce different learning rates

**Gist:** the trainer synchronizes the stop flag across ranks, but does not synchronize their consumption-based progress. Unequal packing can therefore produce unequal group LRs during warmdown, with consequences for replicated parameters.

[The loop all-reduces `last_step`](../nanochat/scripts/chat_sft.py#L333). Each rank's `progress` remains local, and each computes its own LR multiplier. In a small source-generator check with short conversations on one rank and long conversations on another, the two progress estimates after three yields were 0.9 and 0.35. The default multipliers were consequently 0.2 and 1.0.

Gradient averaging alone does not correct differing learning rates. In particular, [small AdamW parameters](../nanochat/nanochat/optim.py#L274) have their gradients averaged and are updated as full replicated tensors on every rank. With identical weights, gradients, and histories but different $\eta_r$,

$$
\theta_r^+=(1-\eta_r\lambda)\theta-\eta_r H
$$

can differ between ranks. Those small parameters are not subsequently all-gathered by [the update path](../nanochat/nanochat/optim.py#L320). The residual/scalar groups include such parameters. Larger sharded parameters are gathered, but can still have different owner-specific update rates.

This source path creates a replica-consistency problem when local schedules differ; the scratch check established the control-flow condition and the unequal-update consequence, not the size of its effect in a real run. With positive $K$ and the same number of generator calls, the iteration-based progress is equal across ranks, though the horizon/prefetch issues above still apply.

### 14.5 A context-length override is not a full model reconfiguration

**Gist:** the model is built from the base checkpoint before SFT resolves its row length. The script does not then rebuild the model's context-dependent settings, yet it writes the SFT row length into saved model configuration.

Compare [model loading](../nanochat/scripts/chat_sft.py#L94), [argument inheritance](../nanochat/scripts/chat_sft.py#L98), and [saved `sequence_len`](../nanochat/scripts/chat_sft.py#L403). Changing `max_seq_len` changes the loader, but there is no corresponding reassignment/rebuild of the loaded GPT's precomputed rotary cache or attention-window sizes in this script.

The architecture note explains how windows depend on configured sequence length. If an override differs from the base setting, the configuration saved for a later reload can therefore describe context settings different from those used during SFT. A sufficiently large input can also exceed the actual rotary cache. Neither changing row length nor writing a new metadata field by itself establishes correct context extension.

### 14.6 The displayed loss has a small bookkeeping error

**Gist:** the logged training loss is an EMA of the final microbatch's loss, rather than the full update's mean, and its bias correction uses an exponent one ahead of the number of EMA observations.

[The loop increments `step`](../nanochat/scripts/chat_sft.py#L458) before updating and debiasing the EMA. After the first observation with $\beta=0.9$,

$$
M_1=(1-\beta)L_1,
$$

but the displayed value divides by $1-\beta^2$ rather than $1-\beta$. Thus the first displayed loss is $L_1/1.9$, not $L_1$. The discrepancy decays with time. This affects the log, not the differentiated training objective.

Similarly, reported token throughput uses all $G$ computed positions. It should not be read as supervised assistant tokens per second. Reading the loss and throughput labels with their exact definitions avoids drawing conclusions from an apparent early loss jump or from comparing differently masked workloads.

## 15. Small experiments and checks performed

**Gist:** you can verify the important ideas with tiny token sequences and synthetic tensors. No trained model or complete dataset is needed to understand target alignment, normalization, or the loader's control flow.

### 15.1 Hand-check one conversation and one tool call

**Gist:** first identify which tokens are the assistant's responsibility, then shift their masks with the targets. This checks the core SFT idea more directly than looking at a scalar loss curve.

Using sections 5 and 7, predict the targets for:

```text
BOS user-start "2+3?" user-end assistant-start
python-start "2+3" python-end
output-start "5" output-end
"5" assistant-end
```

The useful questions are: where is the first supervised prediction? Is the model supervised for opening the Python block? For generating the calculator result? For writing the final answer? For stopping?

If inspecting a real tokenizer later, [visualize_tokenization](../nanochat/nanochat/tokenizer.py#L226) colors IDs by the renderer mask. Remember that those colors describe target-token responsibility before the shift; the logit predicting a green token is at the preceding input position.

### 15.2 Check the mean-of-means effect in a standalone snippet

**Gist:** the unequal-label example can be reproduced with scalar losses. This exposes the weighting choice without any distributed hardware or transformer computation.

```python
import torch

# These are already token-level negative log probabilities.
first = torch.tensor([4.0])
second = torch.tensor([1.0, 1.0, 1.0])

script_mean = (first.mean() + second.mean()) / 2
token_mean = torch.cat([first, second]).mean()

print(script_mean.item())  # 2.5
print(token_mean.item())   # 1.75
```

For a next step, replace these constants with expressions depending on one scalar parameter and compare gradients under both reductions. Changing the denominator changes the optimizer's input gradient even when the same valid target tokens appear.

### 15.3 Suggested later ablations

**Gist:** change one supervision/data decision at a time and evaluate behavior that decision is supposed to teach. These are proposals for future runs, not experiments launched for this note.

| Change | What it tests | Useful measurements |
| --- | --- | --- |
| Remove assistant-end supervision | Whether a learned stop boundary matters | Correct stopping, truncation rate, generated lengths |
| Supervise tool outputs as well | How imitation changes when observations become targets | Tool-call success, behavior with forced outputs |
| Reset versus restore optimizer state | Value and cost of carrying pretrained histories | Early losses, validation/task results at equal update budgets |
| Change MMLU/GSM8K repetition | Exposure tradeoff across skills | Per-task scores and general conversational quality |
| Global valid-token mean versus microbatch means | Sensitivity to supervision density | Gradients on a fixed example set; then equal-budget outcomes |
| Independently mask packed conversations in attention | Effect of cross-conversation context | Prediction differences and held-out quality |

Before expensive comparisons, the horizon, length, and distributed-schedule caveats in section 14 would need explicit handling so that the intended experiment is the one actually run. No such changes are included here.

### 15.4 What was checked while preparing this note

**Gist:** isolated execution of the actual source definitions verified the rendering, packing, objective algebra, and several control-flow caveats. These checks do not establish trained assistant quality or GPU performance.

The checks used CPU PyTorch 2.11.0 and synthetic in-memory examples, without importing the training script's top-level execution:

- The actual conversation renderer produced the seven-token mask/shift example, the completion prefix, and a nonmutating system-message merge.
- The actual GSM8K adapter and renderer separated supervised calculator requests from ignored results. A long user message demonstrated default 2048-token truncation with no surviving assistant labels.
- The actual `TaskMixture` reproduced finite duplicated row multiplicities and its deterministic construction.
- The actual SFT generator packed two rendered conversations, ignored padding/boundaries as described, and reproduced the oversized-buffer all-ignored loss case.
- The source generator and LR function reproduced the positive-iteration/prefetch examples and the two-rank local progress discrepancy. No real distributed process group was launched for those generator checks.
- Autograd matched the masked CE plus softcap derivative, including zero direct logit gradients at ignored positions.
- The actual optimizer's eager CPU AdamW path retained moments, counters, and decay metadata when loading state and restoring fresh LRs. Identical replicated parameter/gradient inputs with unequal LRs produced unequal updates.
- The first EMA bias-correction discrepancy was checked arithmetically. Source links and Markdown structure were checked separately.

These are focused source/numerical checks, not a full package test run, a checkpoint-resume test, or an evaluation on downloaded benchmark data. Both source repositories remain untouched.

## 16. What changes when we move to RL

**Gist:** SFT learns from supplied continuations; RL samples the model's own continuations and assigns credit using rewards. The model, tokenizer, masks, optimizer, and inference engine now form the pieces of that next stage.

The pipeline we have covered is

```text
curated document corpus
  -> tokenizer + pretraining
  -> base checkpoint
  -> formatted demonstrations + masked SFT
  -> assistant checkpoint
  -> conversation/task evaluation and interactive generation
```

For SFT, the next-token learning signal is “increase probability of the demonstrated token.” For policy-gradient RL, it becomes “increase or decrease probability of sampled actions according to their reward relative to a baseline.”

GSM8K will make that concrete: the engine samples candidate solutions, a task checks final-answer correctness, and the trainer uses the resulting rewards. We will derive the implemented loss carefully, including which tokens are model actions, how group-centered rewards work, and why this checkout's simplified method differs from full PPO or GRPO.

The essential SFT chain to retain is:

$$
\boxed{
\text{conversation}
\longrightarrow (z_i,m_i)
\longrightarrow (X_t,Y_t)
\longrightarrow\text{masked conditional CE}
\longrightarrow\text{parameter update}.
}
$$

Once that chain is clear, most of the remaining SFT code is accounting for efficient batching, inherited optimizer state, and reliable measurement. We can discuss any of those parts before moving to RL.

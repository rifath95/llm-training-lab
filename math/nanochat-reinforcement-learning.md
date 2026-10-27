# Reinforcement learning in nanochat: from sampled solutions to policy gradients

The SFT note explained how the model learns from supplied assistant responses. This note explains the next change: **the model generates its own solutions, a checker scores them, and the trainer changes their probabilities according to those scores**.

The central idea is small. Sample several solutions to one question, subtract their mean reward, and use the resulting positive or negative weights on the sampled tokens' log probabilities. Understanding why that works requires separating the ideal policy-gradient derivation from this checkout's particular masks, normalization, and sampling settings.

The source is nanochat commit `92d63d4e8bb4df75c3b71618f31ddde2378b2bcd`; comparisons use your `transformer-from-scratch` at `56092a1818027c800f08167ec0c1c65b983711e1`. Code links point to exact lines in these checkouts. The [SFT](nanochat-supervised-fine-tuning.md), [inference](nanochat-inference-and-generation.md), and [optimization](nanochat-optimization-and-distributed-training.md) notes supply the prerequisites.

**How to skim:** every numbered section and subsection begins with a **Gist**. Sections 1–4 derive the learning signal and explain the reward. Sections 5–8 trace the actual rollouts, masks, loss, and gradients. Sections 9–11 explain sparse rewards, sampling-policy consistency, and the relationship to PPO/GRPO. Sections 12–15 cover updates, evaluation, checkpoints, and implementation boundaries. The final sections compare your code, offer small checks, and connect the full pipeline.

No source files in either repository were changed, and no training job, trained checkpoint, or dataset download was used to prepare this note.

## Contents

1. [The reinforcement-learning problem](#1-the-reinforcement-learning-problem)
2. [Deriving the policy gradient](#2-deriving-the-policy-gradient)
3. [Group-relative advantages](#3-group-relative-advantages)
4. [What the GSM8K reward actually checks](#4-what-the-gsm8k-reward-actually-checks)
5. [Sampling solutions with the inference engine](#5-sampling-solutions-with-the-inference-engine)
6. [Constructing the RL inputs and targets](#6-constructing-the-rl-inputs-and-targets)
7. [The exact loss implemented by the trainer](#7-the-exact-loss-implemented-by-the-trainer)
8. [What the gradient does to token probabilities](#8-what-the-gradient-does-to-token-probabilities)
9. [Sparse rewards and groups with no learning signal](#9-sparse-rewards-and-groups-with-no-learning-signal)
10. [On-policy sampling and the top-k mismatch](#10-on-policy-sampling-and-the-top-k-mismatch)
11. [How this differs from PPO and full GRPO](#11-how-this-differs-from-ppo-and-full-grpo)
12. [The optimizer and complete update loop](#12-the-optimizer-and-complete-update-loop)
13. [Evaluation and training diagnostics](#13-evaluation-and-training-diagnostics)
14. [What the RL checkpoint contains](#14-what-the-rl-checkpoint-contains)
15. [Implementation boundaries in this checkout](#15-implementation-boundaries-in-this-checkout)
16. [Comparison with your implementation](#16-comparison-with-your-implementation)
17. [Small experiments and checks performed](#17-small-experiments-and-checks-performed)
18. [Putting the complete pipeline together](#18-putting-the-complete-pipeline-together)

## 1. The reinforcement-learning problem

**Gist:** the assistant is a stochastic policy whose actions are tokens. A question starts an episode; generated text and calculator interactions form a trajectory; the final answer receives a correctness reward.

### 1.1 Questions, actions, observations, and rewards

**Gist:** this is ordinary sequential decision-making expressed through text. The model chooses tokens, while the environment supplies tool results and a final score.

For a question $x$, a history $h_t$ consists of the question, earlier assistant tokens, and any calculator outputs received so far. The model's next-token distribution defines a policy

$$
\pi_\theta(a_t\mid h_t).
$$

An action $a_t$ might be an ordinary word fragment, a digit, or a tool delimiter. Once a calculator call is complete, the engine can insert an observation: the calculated result. The assistant then acts again with that result in its context.

Write the full interaction as a trajectory $\omega$. A verifier gives $R(x,\omega)\in\{0,1\}$ according to whether the extracted final answer matches the reference. The natural goal is to increase expected reward:

$$
J(\theta)
=\mathbb E_{x\sim\mathcal D}
\mathbb E_{\omega\sim P_\theta(\cdot\mid x)}[R(x,\omega)].
$$

For this binary reward, $J$ is an average success probability under the specified generation policy and question distribution. It is not a universal measure of intelligence or of the validity of every reasoning step.

### 1.2 What changes relative to SFT?

**Gist:** SFT supplied the answer tokens and rewarded imitating them. RL supplies questions and a checker, then learns from the success or failure of its own sampled answers.

| Part | SFT | This RL script |
| --- | --- | --- |
| Initial weights | Base model | SFT model |
| Response used for training | Dataset demonstration | Current model's sampled response |
| Feedback | Correct next token at selected positions | One correctness reward per sampled response |
| Token weights | Positive supervised CE weights | Positive, negative, or zero group-relative weights |
| Calculator results | Supplied by the demonstration | Produced by the rollout engine's calculator |
| Reference answer | Supplies targets | Supplies the verifier's comparison value |

[The RL script loads source `"sft"`](../nanochat/scripts/chat_rl.py#L74) and constructs GSM8K [training and test tasks](../nanochat/scripts/chat_rl.py#L80). It does not place the reference solution in the sampled prompt: [render_for_completion](../nanochat/nanochat/tokenizer.py#L241) removes the final reference assistant message first.

The reference solution remains available to the reward checker. Access to a reference answer for scoring is different from putting that answer into the model's generation context.

### 1.3 What kind of RL is implemented?

**Gist:** this is a compact policy-gradient method with verifiable rewards and group-mean baselines. It has no learned reward model, critic, value loss, reference-model KL, or PPO clipping.

[The script introduction](../nanochat/scripts/chat_rl.py#L1) explicitly puts “GRPO” in quotation marks and describes its simplifications. We will derive its actual loss before relating it to the larger algorithms.

There is no separate network estimating the expected return of a state. Several sampled answers to the same question supply an empirical comparison instead. There is also no human-preference labeling loop here. The reward is a deterministic answer checker, making this an example of reinforcement learning with verifiable rewards.

The transformer architecture does not change. The change is the data-generation process and the scalar weights applied to differentiable log probabilities.

## 2. Deriving the policy gradient

**Gist:** discrete sampling prevents ordinary backpropagation through chosen token IDs, but we can differentiate the probability of choosing those tokens. The resulting score-function identity turns a reward into a learning signal for the actions that produced it.

### 2.1 Why we do not differentiate the answer checker

**Gist:** correctness is not a smooth function of token IDs. Policy gradients differentiate the distribution over responses instead of requiring a derivative through the verifier or calculator.

For a fixed question, first suppress $x$ from the notation and assume a well-defined distribution over complete finite trajectories. Then

$$
J(\theta)=\sum_\omega P_\theta(\omega)R(\omega).
$$

The reward function evaluates the sampled trajectory; it has no explicit differentiable dependence on $\theta$. Therefore

$$
\begin{aligned}
\nabla_\theta J
&=\sum_\omega R(\omega)\nabla_\theta P_\theta(\omega)\\
&=\sum_\omega P_\theta(\omega)R(\omega)
\nabla_\theta\log P_\theta(\omega)\\
&=\mathbb E_{\omega\sim P_\theta}
\left[R(\omega)\nabla_\theta\log P_\theta(\omega)\right].
\end{aligned}
$$

This is the score-function, or likelihood-ratio, identity underlying REINFORCE. The classical reference is [Williams, 1992](https://link.springer.com/article/10.1007/BF00992696). The derivation here is self-contained; the practical question is which trajectory distribution and log-probability factors the implementation actually uses.

The nondifferentiability of picking a token or checking a string is no obstacle to this identity. The gradient acts on the probability of the sampled event, not on the event's discrete value.

### 2.2 A sequence log probability is a sum of action log probabilities

**Gist:** the trajectory probability factors into model decisions and environment transitions. If the environment has no trainable dependence on $\theta$, only the assistant's decision factors contribute score gradients.

Schematically, including external observations $o_t$,

$$
P_\theta(\omega\mid x)
=\prod_{t\in\mathcal A}\pi_\theta(a_t\mid h_t)
\prod_{t\in\mathcal O}P_{\mathrm{env}}(o_t\mid h_t).
$$

Here $\mathcal A$ indexes policy actions and $\mathcal O$ environment observations. A deterministic calculator is one possible environment transition. Since the second product has no explicit parameter dependence,

$$
\nabla_\theta\log P_\theta(\omega\mid x)
=\sum_{t\in\mathcal A}\nabla_\theta\log\pi_\theta(a_t\mid h_t).
$$

Thus an ideal Monte Carlo estimator is

$$
\widehat g
=\frac1S\sum_{i=1}^{S}R_i
\sum_{t\in\mathcal A_i}\nabla_\theta\log\pi_\theta(a_{i,t}\mid h_{i,t}).
$$

We have not differentiated through the calculator. We have assigned credit to the assistant decisions that caused a useful interaction with it.

For a fully faithful trajectory score, sampled stopping decisions also belong among the actions. The engine wrapper's treatment of these decisions will matter in section 5.

### 2.3 A surrogate loss supplies this gradient to autograd

**Gist:** after sampling, freeze the trajectories and rewards, recompute their log probabilities with gradients enabled, and minimize negative reward-weighted log probability.

At the weights that generated the samples, the differentiable surrogate

$$
\mathcal L_{\mathrm{PG}}(\theta)
=-\frac1S\sum_i R_i
\sum_{t\in\mathcal A_i}\log\pi_\theta(a_{i,t}\mid h_{i,t})
$$

has gradient $-\widehat g$. Gradient descent on this loss follows the estimated ascent direction for expected reward.

This surrogate's numerical value is not expected reward. Its usefulness lies in its derivative at the sampling weights. Repeatedly optimizing it on stale trajectories is a different procedure and needs additional analysis.

The sampled token IDs are constants in this backward pass. The fact that they were originally generated by the model does not mean an autograd graph survives through generation. Nanochat deliberately generates without gradient tracking, then performs an ordinary full-sequence forward to recover differentiable token log probabilities.

## 3. Group-relative advantages

**Gist:** rewards are compared among solutions to the same question. Subtracting the group mean makes successes relatively positive and failures relatively negative, while removing the need to learn a separate baseline network.

### 3.1 Why subtract a baseline?

**Gist:** a baseline independent of the sampled action can reduce estimator variance without changing the ideal expected policy gradient. It says whether a result was better or worse than an appropriate expectation.

For a baseline $b(x)$ fixed for the question,

$$
\mathbb E_{\omega\sim P_\theta(\cdot\mid x)}
\left[b(x)\nabla_\theta\log P_\theta(\omega\mid x)\right]
=b(x)\nabla_\theta\sum_\omega P_\theta(\omega\mid x)=0.
$$

We can therefore replace $R$ by $R-b(x)$ in the ideal estimator. A question-dependent baseline is useful because some questions are much easier than others. A reward of one is less surprising on an almost-solved question than on a question that nearly always fails.

“Advantage” here means a reward relative to a baseline. This script's advantage is not a learned state-action value minus a learned state value, nor a generalized-advantage-estimation computation over intermediate rewards.

### 3.2 The actual group mean

**Gist:** sample $S$ solutions for one question, compute their mean reward, and subtract it from every reward. Do not mix unrelated questions into this baseline.

[get_batch](../nanochat/scripts/chat_rl.py#L117) computes

$$
\overline R=\frac1S\sum_{i=1}^{S}R_i,\qquad
A_i=R_i-\overline R.
$$

For rewards $[1,0,1,0]$, the mean is 0.5 and the advantages are

$$
[0.5,-0.5,0.5,-0.5].
$$

For one success among sixteen samples, the successful response gets $15/16$ and every failed response gets $-1/16$. Within a question,

$$
\sum_i A_i=0.
$$

The trainer computes these advantages over the complete sampled group **before** slicing it into smaller forward/backward batches. Computing a separate mean inside each slice would define a different comparison and can erase useful differences between slices.

These tensors are produced without a gradient graph. During policy optimization, rewards and advantages are treated as constants; the trainer differentiates the log probabilities they multiply.

### 3.3 Including each sample in its own baseline has a finite-group effect

**Gist:** the group mean is correlated with each sample's reward, so the standard independent-baseline argument does not apply literally. Under ideal independent, complete-policy sampling, it scales the expected estimator by $(S-1)/S$.

Let $g_i=\nabla\log P_\theta(\omega_i\mid x)$ for independent samples from the same policy, with $\mathbb E[g_i]=0$. Consider the sequence-sum estimator without random length normalization:

$$
\widehat g_{\mathrm{group}}
=\frac1S\sum_i(R_i-\overline R)g_i.
$$

For $j\neq i$, independence gives $\mathbb E[R_jg_i]=0$. The self term remains:

$$
\mathbb E[\overline Rg_i]
=\frac1S\mathbb E[R_ig_i].
$$

Consequently,

$$
\mathbb E[\widehat g_{\mathrm{group}}]
=\left(1-\frac1S\right)\nabla J_x.
$$

For $S=16$, this factor is $15/16$. For $S=1$, it is zero: subtracting the only reward from itself eliminates every learning signal.

A leave-one-out baseline would use

$$
A_i^{\mathrm{LOO}}
=R_i-\frac{\sum_{j\neq i}R_j}{S-1}
=\frac{S}{S-1}(R_i-\overline R),\qquad S>1.
$$

This removes that particular scale factor under the stated assumptions. Nanochat does not apply this correction. The derivation does **not** prove that the entire implemented estimator is simply an unbiased gradient times $15/16$: random token-count denominators, omitted stop actions, and the sampling-policy mismatch introduce additional distinctions below.

### 3.4 Why there is no division by standard deviation

**Gist:** the script centers rewards but does not standardize them. That avoids a small or zero standard-deviation denominator and gives different weighting from standardized group advantages.

For binary rewards with empirical success fraction $\widehat p$, successful responses have advantage $1-\widehat p$ and failed responses have advantage $-\widehat p$.

A standardized version would additionally divide by a group standard deviation, approximately $\sqrt{\widehat p(1-\widehat p)}$ under the population-variance convention, plus whatever numerical stabilization the implementation chooses.

That division changes the relative scale of learning across questions with different success fractions. Omitting it is an algorithmic weighting choice. It is not merely deleting an unnecessary constant shared by every training example.

## 4. What the GSM8K reward actually checks

**Gist:** the reward checks an extracted answer string after `#### `. It does not inspect whether each reasoning step or calculator request was logically correct.

### 4.1 The extraction and comparison rule

**Gist:** find the first matching answer marker, remove commas from the extracted number string, and compare it with the reference's extracted string.

**Code:** [extract_answer](../nanochat/tasks/gsm8k.py#L22), [GSM8K.evaluate](../nanochat/tasks/gsm8k.py#L86), [GSM8K.reward](../nanochat/tasks/gsm8k.py#L109).

The regular expression is

```python
r"#### (\-?[0-9\.\,]+)"
```

The checker reads the reference from the final text part of the dataset's assistant solution and reads the prediction from decoded generated text. It uses the first regex match, strips surrounding whitespace from the captured value, removes commas, and tests string equality. `reward` returns the result as a float.

Examples for reference `#### 5`:

| Generated text | Reward | Reason |
| --- | --- | --- |
| `#### 5` | 1 | Extracted strings match |
| `The answer is 5` | 0 | Required answer marker is absent |
| `#### 5.0` | 0 | `5.0` and `5` are different strings |
| `#### 5` followed later by `#### 9` | 1 | The first match is used |

For a reference value `1000`, a generated `#### 1,000` matches after comma removal. This is a lightweight benchmark checker, not a general mathematical-equivalence engine. The expected dataset references contain usable answer markers; the implementation does not separately validate malformed references before comparing extracted values.

### 4.2 Outcome reward gives coarse credit

**Gist:** a successful final answer gives the same scalar advantage to all retained sampled tokens in that response. The checker cannot identify which intermediate step caused the success or error.

A correct answer reached through a dubious argument can still receive reward one. A nearly correct derivation with a wrong final number receives zero. The generated reasoning text is not checked step by step.

The learning signal can nevertheless favor useful patterns statistically: trajectories containing them may succeed more often across samples. But one positive advantage is not a certificate that every token in that trajectory was helpful.

There is no extra format reward, length penalty, process reward, or reward-model prediction in this script. Formatting matters indirectly because the verifier requires the answer marker. Calculator execution influences what information is available during generation, while the terminal reward still comes from the answer comparison.

## 5. Sampling solutions with the inference engine

**Gist:** the same GPT used for updates also generates the rollouts. Generation uses cached decoding without gradients, changes seeds between sampling chunks, executes calculator requests, and returns a provenance mask for sampled versus forced tokens.

### 5.1 One question, several response chunks

**Gist:** a question supplies one shared prompt and a group of alternative solutions. Device batch size limits how many solutions are generated together; it is not the number of distinct questions in that call.

[get_batch](../nanochat/scripts/chat_rl.py#L86) selects a conversation, [renders its completion prefix](../nanochat/scripts/chat_rl.py#L96), sets the model to eval mode, and calls [Engine.generate_batch](../nanochat/nanochat/engine.py#L277).

The defaults are:

| Setting | Default | Meaning |
| --- | --- | --- |
| `num_samples` | 16 | Requested candidate solutions for each question |
| `device_batch_size` | 8 | Candidate rows generated or differentiated together |
| `max_new_tokens` | 256 | Maximum emitted generation columns per rollout call |
| `temperature` | 1.0 | Training-rollout sampling temperature |
| `top_k` | 50 | Candidate token support at each sampled step |

With these settings, the trainer makes two calls of eight solutions. Within each call the engine prefills the prompt once and replicates its KV cache across the eight rows. The next call performs its own prefill; the trainer does not retain one shared prompt cache across both calls.

[The sampling seed](../nanochat/scripts/chat_rl.py#L105) depends on `(step, example_idx, sampling_step)`. This avoids deliberately reusing the same seed for both chunks. A seed is a reproducibility input, not a guarantee of identical numerical trajectories across different devices, kernels, or batching arrangements.

### 5.2 The mask records provenance, not token identity

**Gist:** prompt and forced-result tokens have mask zero. Tokens chosen by the model have mask one, including calculator request delimiters and expressions.

[Engine.generate_batch](../nanochat/nanochat/engine.py#L285) starts with zero masks for the prompt. [The streaming engine](../nanochat/nanochat/engine.py#L237) gives each subsequently emitted token mask one if sampled and zero if taken from the forced-token queue.

A normal calculator interaction therefore looks like:

```text
prompt and assistant-start                 0
assistant reasoning                        1
python-start, expression, python-end        1
output-start, computed result, output-end   0
assistant continuation and final answer     1
```

Tool calls are policy actions; successful tool outputs are environment observations. The source comment about ignoring “tool use tokens” means the forced output tokens, not every token associated with tools.

This mask is determined by how a token entered the trajectory. If the model itself samples a token that looks like an output delimiter, that sampled token is still an action. The mask does not classify responsibility by looking up whether the token ID belongs to a particular semantic category.

### 5.3 Stop tokens are removed before the RL trainer receives the sequence

**Gist:** the streaming engine samples stopping decisions, but `generate_batch` discards assistant-end and BOS from returned sequences. Their log probabilities consequently do not appear in this script's RL loss.

[The wrapper](../nanochat/nanochat/engine.py#L288) marks a row complete when it sees assistant-end or BOS and does not append that terminal token or its mask. This contrasts with SFT, which explicitly supervises assistant-end.

For a sampled response `a, b, assistant-end`, RL receives `a, b`, not the terminal decision. A complete trajectory score for that response would contain

$$
\log\pi_\theta(a\mid x)
+\log\pi_\theta(b\mid x,a)
+\log\pi_\theta(\mathrm{end}\mid x,a,b).
$$

The last term is missing from the returned-token loss. Its omission is an implementation simplification and prevents treating that loss as the exact score of the complete stopped trajectory. It does not make the network incapable of stopping: it starts from an SFT model, and other parameter updates can also affect stopping probabilities indirectly.

An immediately terminating sample can return only the prompt and have no valid action targets. A rollout that instead hits the fixed generation budget can retain its final nonterminal sampled token; that case does not require inventing a nonexistent sampled end token.

### 5.4 Generation and differentiation use different memory paths

**Gist:** cached inference creates the data efficiently, while a later full-sequence training forward constructs the backward graph. The trainer does not backpropagate through the KV-cache rollout process.

[get_batch](../nanochat/scripts/chat_rl.py#L85) is decorated with `torch.no_grad`, and the engine's generator runs under inference mode. These calls return integer token sequences and masks, not a retained chain of generation activations.

Later the trainer runs the sampled sequence through GPT again with targets and gradients enabled. This is teacher-forced evaluation of **self-generated data**: each action log probability is evaluated conditional on the actual sampled history, including calculator observations.

The cost includes both rollout inference and a differentiable forward/backward. Reusing token IDs is inexpensive; retaining every inference activation for backward would have very different memory costs. Model `.eval()` versus `.train()` controls module behavior, while gradient modes control autograd bookkeeping. The current GPT has no dropout, which helps these two evaluations represent the same network function up to numerical differences.

## 6. Constructing the RL inputs and targets

**Gist:** for one question, pad its candidate trajectories to a common length, shift by one token, and ignore prompt, forced-result, and padding targets. The resulting tensor is differentiated in smaller row batches.

### 6.1 From variable-length trajectories to a rectangle

**Gist:** the padding marker is assistant-end, but every added copy is masked out. Padding with an end marker does not restore the omitted sampled stop action.

[The batching code](../nanochat/scripts/chat_rl.py#L128) computes the longest returned sequence length, pads every sequence to it with assistant-end IDs, and pads the corresponding masks with zeros.

For candidate $i$, let its returned IDs be $z_{i,0:M_i-1}$ and its provenance mask be $m_{i,0:M_i-1}$. After padding to $M=\max_iM_i$, construct

$$
X_{i,t}=z_{i,t},\qquad
Y_{i,t}=\begin{cases}
z_{i,t+1},&m_{i,t+1}=1,\\
-1,&m_{i,t+1}=0,
\end{cases}
\quad 0\leq t<M-1.
$$

The mask is shifted with the targets, exactly as in SFT. Thus the assistant-start input predicts the first sampled response token. An externally forced result remains visible as context but has no local policy log-probability term.

The code clones `ids[:, 1:]` before writing `-1` into targets. Without that clone, modifying a view would corrupt the IDs underlying overlapping input/target slices.

### 6.2 Which dimensions count questions and which count solutions?

**Gist:** a global update contains several questions, with several solutions per question. Device batch size slices solutions, while `examples_per_step` counts questions across all ranks.

Use the following notation in this note:

| Symbol | Meaning | Default example |
| --- | --- | --- |
| $Q$ | Question slots per global update | 16 |
| $W$ | Number of distributed ranks/processes | 8 in an eight-GPU run |
| $E=Q/W$ | Questions handled per rank per update | 2 |
| $S$ | Candidate solutions per question | 16 |
| $b$ | Candidate rows per generation/training pass | 8 |
| $P=S/b$ | Passes per question, assuming divisibility | 2 |
| $T$ | Padded input length for a particular question's group | Variable |
| $Z_{r,e,p}$ | Valid retained action targets in a pass | Variable |

Here $e$ indexes questions on a rank, $i$ indexes a candidate solution, and $p$ indexes a pass through a slice of that candidate group. $P$ is a pass count, not prompt length. We use $\ell_x$ when a prompt length is needed.

[The trainer requires](../nanochat/scripts/chat_rl.py#L214) $Q$ to be divisible by $W$. With the table's settings, every rank accumulates gradients from two questions, each with two passes of eight solutions. The global update uses $QS=256$ solution trajectories and $EP=4$ backward calls per rank.

The input length $T$ includes prompt and generated context and varies between questions. There is no fixed global computed-token batch analogous to pretraining's $G=WAbT$ with a fixed $T$. Nor does every solution contain the same number of trainable action targets.

The implementation's actual sample count when $S$ is not divisible by $b$ is a caveat in section 15. Unless stated otherwise, the equations assume the default compatible settings.

## 7. The exact loss implemented by the trainer

**Gist:** each retained action token receives its response's advantage as a weight. Within each pass, the weighted log probabilities are divided by the number of valid tokens; pass, question, and rank averages supply additional factors.

### 7.1 Extracting log probabilities from cross-entropy

**Gist:** per-token CE is negative log probability. Asking GPT for unreduced CE, negating it, and reshaping yields the policy log probabilities while keeping ignored positions at zero.

[The source](../nanochat/scripts/chat_rl.py#L263) uses

```python
logp = -model(inputs, targets, loss_reduction='none').view_as(inputs)
pg_obj = (logp * advantages.unsqueeze(-1)).sum()
```

[GPT.forward](../nanochat/nanochat/gpt.py#L517) uses `ignore_index=-1`. With reduction `none`, ignored target positions contribute zero loss. The resulting `logp` tensor therefore contains selected target log probabilities at valid positions and zeros elsewhere.

This avoids a separate hand-written `log_softmax` plus gather. It still differentiates the full model distribution and includes nanochat's logit softcap. The targets select a vocabulary entry; they are not required to equal a dataset demonstration.

### 7.2 The local pass loss and global surrogate

**Gist:** the denominator is valid action tokens in that pass, multiplied by passes per question and questions per rank. Distributed gradient averaging then averages rank contributions.

Let $I_p$ be the candidate indices in pass $p$, and write $q_{i,t}=\mathbf1[Y_{i,t}\neq-1]$. Define

$$
Z_{r,e,p}=\sum_{i\in I_p}\sum_t q_{i,t}.
$$

[The local loss](../nanochat/scripts/chat_rl.py#L266) is

$$
\mathcal L_{r,e,p}
=-\frac{1}{EP\max(Z_{r,e,p},1)}
\sum_{i\in I_p}A_i
\sum_{t:q_{i,t}=1}\log\pi_\theta(Y_{i,t}\mid X_{i,\leq t}).
$$

The `max` corresponds to `.clamp(min=1)`. If a pass has no valid targets, its numerator is zero, so it contributes zero policy gradient without division by zero.

The optimizer averages gradients across ranks. Assuming model replicas agree when the gradients are evaluated, the complete accumulated gradient is the derivative of the fixed-data surrogate

$$
\boxed{
\mathcal L_{\mathrm{script}}
=-\frac1{WEP}\sum_{r,e,p}
\frac{\sum_{i\in I_p}A_i\sum_tq_{i,t}\log\pi_\theta(Y_{i,t}\mid X_{i,\leq t})}
{\max(Z_{r,e,p},1)}.
}
$$

The groups, rewards, advantages, token IDs, masks, and denominators are all held fixed while differentiating. This is the exact reduction structure in the inspected loop; it is more specific than simply saying “REINFORCE” or “GRPO.”

### 7.3 Token normalization changes trajectory weighting

**Gist:** a long response supplies more weighted action-token terms than a short one within the same pass. The code does not first normalize each response to equal weight.

For a response with $n_i$ valid actions, define its mean action log probability

$$
\overline\ell_i=\frac1{n_i}\sum_{t:q_{i,t}=1}\log\pi_\theta(Y_{i,t}\mid h_{i,t}),\qquad n_i>0.
$$

The pass numerator is $\sum_i A_i n_i\overline\ell_i$. A sequence-normalized alternative would use $\sum_i A_i\overline\ell_i$ and then average over responses. These give different length weights.

The source describes its reduction as DAPO-style token-level normalization. [DAPO's token-level objective](https://arxiv.org/html/2503.14476v1#S3.SS3) contrasts aggregation over all response tokens with first averaging each response separately. Nanochat uses that broad token-level idea, while retaining its own per-pass denominators and simpler policy-gradient objective. It does not implement the complete DAPO algorithm, which also includes other sampling and clipping mechanisms.

Even ordinary unnormalized REINFORCE contains a sum over a trajectory's action scores. Dividing by a sampled token count is an additional weighting choice. Since that count depends on generated lengths and may correlate with rewards and actions, it cannot generally be pulled out of the expectation as a universal constant. The exact expected-reward identity in section 2 should not silently be applied to this length-normalized estimator without those qualifications.

### 7.4 Averaging pass means differs from one group-wide token mean

**Gist:** each pass gets equal outer weight, even when valid-label counts differ. Changing the device batch size can therefore change more than memory usage.

Suppose one question has two candidates, advantages $+0.5$ and $-0.5$, and valid action lengths 1 and 3.

If both candidates occupy one pass, $Z=4$ and the per-token coefficients before the negative log-probability sign are

$$
+0.5/4=+0.125,\qquad -0.5/4=-0.125.
$$

If each candidate occupies its own pass, $P=2$. The coefficients become

$$
+0.5/(2\cdot1)=+0.25,\qquad
-0.5/(2\cdot3)=-1/12.
$$

Question/rank factors are common and omitted in this example. The two reductions are not equal. All advantages were computed over the same two-candidate group; only the division into passes changed.

This is the RL counterpart of the mean-of-microbatch-means distinction in the SFT note. A truly group-wide or globally valid-token-normalized implementation would require corresponding count-aware accumulation weights. This script does not perform that adjustment.

## 8. What the gradient does to token probabilities

**Gist:** positive advantage gives the familiar supervised CE direction; negative advantage reverses it. The final reward is applied to every retained sampled action, giving coarse credit across the whole response.

### 8.1 Positive and negative examples in one equation

**Gist:** for a fixed context, a positive advantage pushes up the sampled target's probability relative to alternatives. A negative advantage pushes in the opposite direction.

Let $s_v$ be the softcapped logits at one valid position, $\pi_v=\mathrm{softmax}(s)_v$, and $y$ its sampled target. With a positive normalization constant $C$, that position's loss is

$$
\ell=-\frac{A}{C}\log\pi_y.
$$

Its logit derivative is

$$
\frac{\partial\ell}{\partial s_v}
=\frac{A}{C}\left(\pi_v-\mathbf1[v=y]\right).
$$

For plain gradient descent on these logits, $A>0$ increases the sampled token's relative preference and $A<0$ decreases it. If $A=0$, there is no local policy-gradient contribution.

If $u_v$ is the raw head logit and $s_v=15\tanh(u_v/15)$, multiply this derivative by $1-\tanh^2(u_v/15)$ for the derivative with respect to $u_v$.

The actual network shares parameters across all contexts and the optimizer transforms the aggregate gradient, so this local logit explanation is not a guarantee that every individual sampled probability rises after the full mixed update. It identifies the learning signal supplied at each position.

### 8.2 Negative loss is not an error

**Gist:** reward-weighted log probabilities are a signed surrogate. Their numerical value can be positive, negative, or zero without directly measuring answer quality.

Ordinary CE is nonnegative. Here a negative advantage multiplies that CE by a negative number. Successful and unsuccessful trajectories contribute with opposite signs, so the overall loss can be negative.

It is therefore inappropriate to interpret this loss like pretraining validation loss: a lower value across different sampled groups is not a calibrated indication of improved model quality. Groups have different rewards, lengths, masks, and baselines.

Likewise, $\sum_iA_i=0$ does not imply that the loss or gradient vanishes. The advantages multiply different token log probabilities and different contextual score vectors. They cancel only particular shared contributions under matching weights, not the whole learning signal in general.

### 8.3 How the model receives credit for using a tool

**Gist:** the reward can reinforce the decision to call a useful calculator expression and the answer written after its result. The externally supplied result itself receives no policy loss.

Suppose a successful trajectory contains

```text
python-start 2+3 python-end
output-start 5 output-end
The answer is #### 5
```

The request delimiters and expression are sampled actions with positive advantage. The output block is masked out. The later answer tokens are again sampled actions with positive advantage.

The answer loss can still backpropagate through the model's representations of the tool result because it is part of the input context. Masking its local output score does not freeze its input embedding or remove its influence on attention. This is the same distinction between loss masking and attention masking established in the SFT note.

No derivative passes through the discrete decision to execute Python or through the external calculation. Credit arrives through the action probabilities and the conditional use of observed context.

## 9. Sparse rewards and groups with no learning signal

**Gist:** group centering needs a difference in rewards. If every sampled response succeeds or every one fails, all advantages are zero, so that question supplies no fresh policy-gradient signal.

### 9.1 Why both all-fail and all-success groups vanish

**Gist:** subtracting a common reward from itself produces zero. This encourages comparisons between alternatives rather than unconditional imitation of every success.

For rewards $[0,0,0,0]$, the mean is zero and all advantages are zero. For $[1,1,1,1]$, the mean is one and all advantages are again zero.

The all-fail case exposes reward sparsity: without a successful or partially rewarded alternative, this centered binary feedback cannot say which attempt was better. The all-success case means the current samples provide no relative distinction worth preferring under this estimator.

The script still performs sampling, scoring, forward passes, and backward calls for these groups. It does not dynamically replace a zero-advantage group with a new question.

Zero fresh policy gradient also does not guarantee unchanged parameters. Optimizer momentum from earlier updates can persist, and some parameter groups have weight decay. The distinction is between **this group's gradient contribution** and **the optimizer's eventual parameter displacement**.

### 9.2 How many samples help?

**Gist:** more samples increase the chance of observing both success and failure for a question, but also increase generation and training cost. They do not fix an effectively zero probability of discovering a useful answer.

If independent candidates have success probability $p$ for one question, a group of $S$ has a nonzero binary comparison exactly when it contains both outcomes:

$$
P(\text{mixed group})=1-p^S-(1-p)^S.
$$

For $S=16$:

| Per-sample success probability | Probability of a mixed group |
| --- | --- |
| $p=0.01$ | About 0.149 |
| $p=0.5$ | About 0.99997 |
| $p=0.99$ | About 0.149 |

The formula assumes independent draws conditional on the question. It illustrates why very hard and nearly solved questions often produce no centered signal. A larger group can reveal rare successes but costs more rollouts; it does not supply correctness information about unseen solutions.

Greedy decoding of a deterministic model/tool setup gives the same response repeatedly for a fixed question. Its identical rewards then yield zero advantages. This is one reason stochastic sampling is central to this training procedure, even though inference for an end user may use greedy decoding.

### 9.3 What SFT contributes to RL's starting point

**Gist:** SFT makes useful formatted responses and tool interactions plausible enough to sample. RL can then prefer successful variants instead of spending its budget discovering the conversation protocol from an unstructured base distribution.

A model that never writes a parseable `####` answer marker receives zero on almost every sample regardless of partial knowledge. A model that cannot complete a tool request cannot reliably exploit the calculator. The SFT stage provides demonstrations of both behaviors.

This is a practical dependency, not a theorem that all RL must start from SFT. The inspected pipeline does: [load_model("sft", ...)](../nanochat/scripts/chat_rl.py#L74). It then uses a small LR fraction and verifiable rewards to modify the resulting policy.

The outcome checker only selects among sampled trajectories; it cannot directly prescribe an unseen correct reasoning path. The initial policy, sampling distribution, reward design, and exploration budget jointly determine the learning opportunities.

## 10. On-policy sampling and the top-k mismatch

**Gist:** trajectories are generated by the current weights and consumed before the next optimizer update, which avoids stale-policy reuse. But the default top-k sampler and the full-softmax training loss define different action distributions, so the ideal on-policy identity does not apply exactly.

### 10.1 In what sense is the loop on policy?

**Gist:** generation and gradient accumulation use the same parameter version throughout one global update. The model weights change only after all local questions have been processed.

For update $k$, the same model at weights $\theta_k$ samples every local question's candidates. The trainer can compute backward gradients for one question and then sample the next: accumulating `.grad` does not itself change weights.

After all $E$ questions and $P$ passes, [optimizer.step](../nanochat/scripts/chat_rl.py#L296) updates the weights once. The next update samples fresh responses. There is no replay buffer and no repeated epoch of parameter updates over the same stored rollout group.

If the sampling distribution and loss policy coincide, this supports the simple one-update score-function estimator without an old-policy importance ratio. However, using the same weights is necessary but not sufficient: the sampling transformation must also agree with the probability model being differentiated.

### 10.2 Temperature and top-k define a different distribution

**Gist:** sampling renormalizes a filtered, temperature-scaled set of logits. The loss instead scores tokens under the unfiltered, temperature-one model softmax.

[sample_next_token](../nanochat/nanochat/engine.py#L141) uses the following distribution for temperature $\tau>0$:

$$
q_\theta(v\mid h)
=\frac{\mathbf1[v\in K_\theta(h)]e^{s_v(h)/\tau}}
{\sum_{j\in K_\theta(h)}e^{s_j(h)/\tau}},
$$

where $K_\theta(h)$ is the top-k support, or the full vocabulary when filtering is disabled. The policy-gradient loss's CE uses

$$
\pi_\theta(v\mid h)=\frac{e^{s_v(h)}}{\sum_je^{s_j(h)}}.
$$

At the defaults, $\tau=1$ but $k=50$, so these are generally not identical. The sampled token is in the top-k set, but that does not make its probability equal under both distributions.

For $\tau=1$, define $Z_K(h)=\sum_{j\in K_\theta(h)}\pi_\theta(j\mid h)$. For a sampled action in that set,

$$
\log q_\theta(a\mid h)
=\log\pi_\theta(a\mid h)-\log Z_K(h).
$$

The omitted normalization term generally has a nonzero derivative. At fixed top-k membership, differentiating the actual behavior distribution would subtract that derivative. Nonunit temperature introduces an additional score difference, and membership changes make hard filtering piecewise rather than globally smooth.

### 10.3 What this means for the mathematical interpretation

**Gist:** the script implements a reward-weighted full-softmax surrogate on top-k-generated data. That can be studied empirically, but it is not automatically an unbiased estimator of either unfiltered-model or top-k-policy expected reward.

The source uses no behavior-policy log probabilities, no top-k-normalizer correction, and no importance weights. Thus its header's “on policy” statement accurately describes parameter freshness while leaving this distributional mismatch unresolved.

There are special cases where a group baseline cancels particular context-common score differences. That does not establish general equality for a multi-token trajectory with different histories. An isolated two-token example with exact enumeration of two-sample groups produced different expected gradients for full-softmax scores and the actual top-k behavior scores.

Using temperature one and disabling top-k via a nonpositive `top_k` removes this particular sampler/loss mismatch. It would not by itself fix omitted stopping scores, group self-baseline effects, or length-dependent normalization. Conversely, simply adding an importance ratio is not a complete solution to hard support truncation: responses outside the behavior policy's support are never sampled.

These distinctions are useful when reading any “on-policy” LLM trainer. Ask both which weights generated the data and which token distribution was used after sampling transformations.

## 11. How this differs from PPO and full GRPO

**Gist:** nanochat retains group comparisons while removing several components of larger RL algorithms. Understanding what those components do explains both the simplicity and the limits of this script.

### 11.1 Why PPO has an old-policy ratio

**Gist:** PPO uses stored old-policy probabilities and a clipped surrogate to support multiple optimization passes over sampled data. Nanochat takes one accumulated update and stores no old action log probabilities.

For an action sampled under frozen $\pi_{\mathrm{old}}$, define

$$
\rho_t(\theta)=\frac{\pi_\theta(a_t\mid h_t)}{\pi_{\mathrm{old}}(a_t\mid h_t)}.
$$

A common PPO surrogate maximizes

$$
\mathbb E\left[
\min\left(\rho_tA_t,
\mathrm{clip}(\rho_t,1-\epsilon,1+\epsilon)A_t\right)
\right].
$$

The [PPO paper](https://arxiv.org/pdf/1707.06347) motivates this clipped surrogate for repeated optimization on collected trajectories. At the sampling weights, $\rho_t=1$ but its derivative is not zero: its denominator is frozen, so $\nabla\rho_t=\nabla\log\pi_\theta$ there. This explains the connection to a single policy-gradient step.

Clipping limits incentives in the surrogate; it is not a hard guarantee that every policy probability or total KL stays within a fixed trust region. Nanochat does not implement this ratio, clipping, or multiple-update reuse. Its $P$ passes merely accumulate gradients before one parameter update.

### 11.2 An old policy and a reference policy have different roles

**Gist:** an old policy identifies the distribution that generated stored data. A reference policy anchors behavior to a chosen model, often an SFT checkpoint. This trainer keeps neither as a separate scoring model.

An example reference regularizer would subtract

$$
\beta\,\mathbb E_h
\left[\sum_v\pi_\theta(v\mid h)
\log\frac{\pi_\theta(v\mid h)}{\pi_{\mathrm{ref}}(v\mid h)}\right]
$$

from an ascent objective. This penalizes distributional drift at the chosen contexts. It serves a different purpose from comparing current and old action probabilities for sampled-data updates.

Nanochat initializes from SFT but does not keep a frozen copy of those weights to score a KL term. A small LR fraction and limited training duration can reduce update size empirically; they do not mathematically impose a KL bound or guarantee preservation of general assistant behavior.

Similarly, the script adds no entropy bonus. It does not directly optimize a separate term to maintain sampling diversity.

### 11.3 The group-relative part of GRPO

**Gist:** the original GRPO formulation uses group-based advantages in a PPO-related objective and avoids a separate learned critic. Nanochat retains reward centering but omits standardization, ratios/clipping, and reference KL.

[DeepSeekMath](https://arxiv.org/html/2402.03300v3#S4.SS1) introduces GRPO and describes outcome advantages standardized using rewards from several outputs of the same question. Its formulation combines these with a clipped policy-ratio objective and a reference-policy penalty.

| Mechanism | Original GRPO formulation | This nanochat script |
| --- | --- | --- |
| Several outputs per question | Yes | Yes |
| Group-relative reward | Centered and standardized | Centered only |
| Separate critic network | Avoided | Absent |
| Old-policy ratio and clipping | Included | Absent |
| Reference-model KL | Included | Absent |
| Response reduction | Includes per-response token averaging | Per-pass valid-token averaging |

The absence of a critic saves its parameters, optimizer state, and training work. The group comparison supplies a baseline, not a learned prediction of each intermediate state's return. “Simplified group-centered REINFORCE” describes the inspected nanochat loss more precisely than treating it as a complete implementation of the paper.

## 12. The optimizer and complete update loop

**Gist:** each update samples fresh trajectories, accumulates advantage-weighted CE gradients, averages them through the existing mixed optimizer, and changes weights once. Unlike SFT, this script starts with fresh optimizer histories.

### 12.1 Fresh Muon/AdamW with smaller learning rates

**Gist:** the trainer loads SFT weights but does not restore SFT optimizer shards. It constructs the usual parameter groups and multiplies their rates by 0.05 by default.

[Optimizer setup](../nanochat/scripts/chat_rl.py#L197) passes raw defaults of embedding LR 0.2, unembedding LR 0.004, and matrix LR 0.02 to [GPT.setup_optimizer](../nanochat/nanochat/gpt.py#L419). Those are RL script defaults, not inherited SFT terminal LRs.

The width factor $c_d=(d/768)^{-1/2}$ still applies to the relevant AdamW groups. After multiplication by `init_lr_frac=0.05`, representative group rates are

$$
\eta_{\mathrm{embed},0}=0.01c_d,\qquad
\eta_{\mathrm{head},0}=0.0002c_d,\qquad
\eta_{\mathrm{matrix},0}=0.001.
$$

Other scalar/value groups retain their setup-specific scales, multiplied by the same 0.05. The matrix update retains Muon's shape-dependent scaling described in the optimizer note.

Despite [the weight-decay flag's help text](../nanochat/scripts/chat_rl.py#L54), the setup argument controls Muon groups; AdamW groups have their own hardcoded decay values in GPT. A default argument of zero therefore does not mean zero weight decay for every parameter group.

There is no separate RL Muon-momentum ramp in this script; fresh groups use the setup default. There is also no model compilation call, FP8 conversion path, FP16 GradScaler, or explicit gradient clipping added by this loop. It relies on the loaded GPT's precision behavior and the mixed optimizer implementation.

### 12.2 Horizon and LR schedule

**Gist:** the number of optimizer updates comes from question count, questions per global update, and requested epochs. Group rates decay linearly with that optimizer-step index.

[The horizon](../nanochat/scripts/chat_rl.py#L80) is

$$
K=\left\lfloor\frac{N_{\mathrm{train}}}{Q}\right\rfloor
\cdot\texttt{num\_epochs}.
$$

For 7473 GSM8K training rows, $Q=16$, and one epoch, $K=467$. That accounts for 7472 question slots, with floor rounding rather than a final partial question batch. Actual data comes from rank-strided, cyclic views of a fixed shuffled task; it is not reshuffled by this loop at every epoch boundary.

[The LR multiplier](../nanochat/scripts/chat_rl.py#L210) is

$$
\lambda(k)=1-\frac{k}{K},\qquad k=0,\ldots,K-1.
$$

The first update uses the full RL initial rate. The last executed update uses multiplier $1/K$, not zero. There is no additional zero-LR update after the loop.

The row count and requested epochs control the horizon; sampled completion lengths control the variable compute cost. Increasing the generation budget or samples per question can greatly increase work without changing $K$.

### 12.3 Trace one global update

**Gist:** sampling and backward passes can alternate across questions without changing the policy weights. The only parameter update happens after all rank-local questions are accumulated.

For the default eight-rank example, $E=2$, $S=16$, $b=8$, and $P=2$:

```text
weights theta_k
  optional test-set evaluation

  local question 1:
    generate 8 + 8 candidates with theta_k
    compute 16 rewards and group-centered advantages
    backward on candidates 1..8, then 9..16

  local question 2:
    generate 8 + 8 new candidates with the same theta_k
    compute its own 16 rewards and advantages
    backward on its two candidate slices

  set this update's LR
  optimizer: average gradients, update owned parameters, gather
  clear gradients
weights theta_(k+1)
  optional checkpoint save
```

The policy-gradient loss accumulation is [inside the question/pass loops](../nanochat/scripts/chat_rl.py#L245); the optimizer call is [outside both loops](../nanochat/scripts/chat_rl.py#L296). There is no separate DDP wrapper performing a reduction after every backward: communication remains inside `MuonAdamW`.

Generation memory is bounded by $b$ candidate caches per call; differentiable activation memory is bounded by $b$ full-sequence rows per pass. All $S$ trajectories' ID/target tensors are still held for the question, and their padding length is set by the longest candidate in that group. Device batch size therefore affects memory, computational shape, and the normalization already discussed.

## 13. Evaluation and training diagnostics

**Gist:** the trainer evaluates stochastic success on a GSM8K test subset and logs sampled training reward and sequence length. These are more informative than the signed policy-gradient loss alone, but they still measure a narrow task under particular sampling settings.

### 13.1 How pass@k is computed here

**Gist:** generate several test answers and count whether at least one of the first $k$ is correct. The trainer reports this for $k=1$ through device batch size.

[run_gsm8k_eval](../nanochat/scripts/chat_rl.py#L150) assigns test questions across ranks and returns their candidate correctness records. [The caller](../nanochat/scripts/chat_rl.py#L224) aggregates counts across ranks.

For $N$ evaluated questions with correctness indicators $c_{j,i}$,

$$
\widehat{\mathrm{pass@}k}
=\frac1N\sum_{j=1}^{N}
\mathbf1\left[\sum_{i=1}^{k}c_{j,i}>0\right].
$$

The defaults evaluate 400 test questions with eight candidates when $b=8$, and report pass@1 through pass@8. This is a direct first-$k$ empirical estimate, not the combinatorial estimator that uses all subsets of $k$ among a larger pool.

For independent samples with one-question success probability $p$, the population quantity is $1-(1-p)^k$. Across heterogeneous questions, average these question-specific quantities; substituting the overall mean $p$ into that nonlinear expression is generally different.

Pass@k measures whether a successful candidate exists among the samples. It assumes a checker identifies it for the metric; it does not show that an application without the reference answer can automatically choose the correct response.

### 13.2 Evaluation settings differ from training settings

**Gist:** evaluation uses temperature one and fixed helper defaults for top-k and completion budget. Changing training generation arguments does not automatically change these evaluation settings.

[The evaluation call](../nanochat/scripts/chat_rl.py#L228) supplies `num_samples=device_batch_size`, `max_examples=eval_examples`, and `temperature=1.0`. It does not pass `max_new_tokens` or `top_k`, so [the helper defaults](../nanochat/scripts/chat_rl.py#L150) remain a 256-token budget and top-k 50.

The helper does not pass a varying seed to the engine; the engine defaults to seed 42 per call. This gives repeatable sampling inputs in a fixed environment and can couple randomness across evaluated questions. The estimate remains a stochastic candidate-based metric, not a guarantee of independent repetitions across checkpoints or devices.

The loop evaluates at step zero and configured intervals before that step's parameter update. It does not automatically force one more evaluation after the final update. Therefore the most recently printed test score need not describe the final saved weights.

The test split is also the one used by earlier SFT validation/task evaluation. Repeatedly tuning choices using these results turns them into development feedback; it does not preserve an untouched final evaluation set.

### 13.3 What the training logs do and do not reveal

**Gist:** average reward summarizes sampled correctness on training questions; average sequence length includes the prompt and forced tool tokens. Neither establishes general assistant improvement.

[The logging block](../nanochat/scripts/chat_rl.py#L275) averages group mean rewards across the rank's questions and then averages across ranks. With compatible equal group sizes, this corresponds to average sampled training reward for that update.

`sequence_length` is the length of each returned token sequence. It includes the prompt and inserted tool-result context, and excludes sampled terminal tokens removed by the wrapper. It is not simply the number of sampled reasoning tokens or newly generated assistant tokens.

Useful additional diagnostics for a later experiment would include mixed-reward group frequency, valid action-token counts, empty/truncated response rates, parseable-answer frequency, calculator success, and broader assistant evaluations. These would help distinguish improved task solving from changes in formatting, length, or sampling behavior. They are not all logged by this script.

## 14. What the RL checkpoint contains

**Gist:** rank zero saves weights and model configuration under the RL source directory. Unlike the SFT save path, it deliberately omits optimizer state and most run metadata.

### 14.1 Files and checkpoint indexing

**Gist:** the checkpoint can be loaded for inference, but its filename index is the zero-based loop index after that iteration's update.

[The save block](../nanochat/scripts/chat_rl.py#L307) writes to

```text
<nanochat base directory>/chatrl_checkpoints/<model-tag-or-dDEPTH>/
```

It passes model weights, `None` for optimizer state, and metadata containing only `model_config` to [save_checkpoint](../nanochat/nanochat/checkpoint_manager.py#L41). The [loader's `"rl"` source mapping](../nanochat/nanochat/checkpoint_manager.py#L163) selects that directory for later evaluation or chat.

For a $K=467$ run, the last loop index is 466, and the final filename is based on 466 even though 467 parameter updates have occurred. In general, evaluation at the start of step $k$ sees weights after $k$ updates, while a checkpoint saved at the end of step $k$ contains weights after $k+1$ updates.

Thus an evaluation and save with the same printed index do not necessarily refer to the same parameter version. The index is a script convention, not a universal definition shared by every stage's checkpoints.

### 14.2 What cannot be resumed from this artifact alone

**Gist:** it is an inference artifact, not a complete interrupted-run state. Continuing exactly would require state that this save does not include.

The RL save omits optimizer moments, group schedules/counters, question iterator position, sampling RNG state, rewards/rollouts, and the parsed run configuration. The script also always starts from source `"sft"`; it has no complete resume branch restoring a previous RL loop.

The in-memory `user_config` exists for logging but is not added to this checkpoint's metadata. Reproducing a run therefore also requires recording its arguments, dataset artifacts/order, initial checkpoint, and execution environment elsewhere.

This does not prevent loading the saved GPT and sampling from it. It limits what the saved artifact establishes about exact training continuation and provenance.

## 15. Implementation boundaries in this checkout

**Gist:** several assumptions are implicit in this short script. Keep these distinct from the policy-gradient idea so that a future reproduction uses the intended sample counts, evaluation settings, and objective.

### 15.1 Sample counts require compatible batch settings

**Gist:** generation uses floor division by device batch size without processing a remainder. It can silently return fewer candidates than requested, or fail when none are generated.

[The sampling loop](../nanochat/scripts/chat_rl.py#L103) uses

```python
num_sampling_steps = args.num_samples // args.device_batch_size
```

The actual number generated is

$$
S'=b\left\lfloor S/b\right\rfloor.
$$

For $S=17,b=8$, sixteen are generated. The later divisibility assertion on the returned tensor succeeds because that tensor already has sixteen rows. The printed nominal sequence count still uses the requested $S$.

For $S<b$, zero calls are made, and taking the maximum sequence length of an empty list fails. Setting `num_samples=1` with the default device batch size is therefore not a working one-candidate ablation. Even with $b=1$, its group-centered advantages would all be zero.

Default $S=16,b=8$ is compatible. The mathematical group-size formulas should use the actual $S'$ when interpreting other accepted settings.

### 15.2 Generation limits include forced observations

**Gist:** the engine's budget counts emitted columns, including inserted calculator outputs. It does not reserve 256 model-action tokens plus an unlimited number of observation tokens.

[The engine's generation counter](../nanochat/nanochat/engine.py#L269) increments after each emitted column. A forced output marker, result digit, or output-end marker consumes that budget too.

Consequently two trajectories with different tool use can have different numbers of sampled actions even when they hit the same column limit. A cutoff can occur in the middle of a tool interaction or before the final answer marker. The script applies the ordinary final-answer checker to the returned prefix; it has no separate truncated-rollout reward shaping.

The prompt renderer also retains its own truncation behavior from the SFT note. The RL script does not install a new prompt-length policy or establish that arbitrarily long prompt-plus-response sequences are inside the context distribution used during training. Cache allocation capacity and learned context quality are separate questions.

### 15.3 Interval flags are plain modulo conditions

**Gist:** these evaluation/save intervals do not implement the nonpositive-means-disabled behavior used by some other scripts.

[Evaluation](../nanochat/scripts/chat_rl.py#L225) directly tests `step % eval_every == 0`. Zero therefore causes division by zero, while -1 makes the condition true every step. The periodic [save condition](../nanochat/scripts/chat_rl.py#L308) also uses an unchecked modulo once its positive-step branch is reached.

The final-save condition is explicit, but the final-evaluation condition is not. A run with zero calculated steps executes neither training-loop evaluation nor its final-save block. These control-flow details matter when proposing a tiny dry run; they are not part of the RL algorithm itself.

### 15.4 Which limitations affect the estimator itself?

**Gist:** token provenance is handled carefully, but the implemented update remains an approximation to the simplest expected-reward policy gradient for several independent reasons.

The main distinctions already derived are:

- **Group self-baseline:** the sample participates in its own baseline; the ideal sequence estimator has the $(S-1)/S$ scale factor.
- **Sampled-length weighting:** per-pass valid-token denominators depend on the trajectories and are not one fixed global normalization.
- **Sampler/loss mismatch:** default top-k filtering changes the behavior distribution, while CE scores the full model softmax.
- **Omitted terminal actions:** the wrapper discards sampled stop tokens before constructing the loss.

These do not imply that the script cannot learn. They limit which exact theoretical guarantees follow from calling it “on-policy REINFORCE” or “GRPO.” Measuring its behavior requires experiments with these details recorded. The small checks in section 17 establish local mechanics, not the outcome of a trained RL run.

## 16. Comparison with your implementation

**Gist:** your model already produces differentiable token probabilities and can generate candidates. RL adds a generation/reward/advantage loop around those capabilities and needs a finer loss interface than your current scalar CE return.

Your anchors are [MyGPT.forward](../transformer-from-scratch/model.py#L280), [its CE call](../transformer-from-scratch/model.py#L294), [generation](../transformer-from-scratch/model.py#L326), and [the accumulated training loop](../transformer-from-scratch/train.py#L80).

| Piece | Your current implementation | What this RL pipeline adds |
| --- | --- | --- |
| Generate text | Temperature/top-k and latent KV caching | Several candidates per question, fresh seeds, tool execution and provenance masks |
| Compute loss | Scalar next-character CE and MoE auxiliary loss | Unreduced action-token log probabilities with ignored positions |
| Targets | Shifted Shakespeare windows | Shifted self-generated trajectories |
| Supervision | Next-character labels | Reward checker plus group-relative advantages |
| Accumulation | Positive weighted training loss | Signed policy-gradient contributions over question groups |
| Evaluation | Character prediction loss | Stochastic response success and pass@k |

Conceptually, an adaptation would need to expose per-token CE or equivalent gathered log probabilities, define its ignore mask, collect sampled answers with the right action provenance, and supply a task verifier. Merely multiplying your existing scalar CE by one average batch reward would lose which response received which reward.

Your model's MoE load-balancing loss is a separate objective. If retained during RL, it would need its own explicit coefficient and interpretation; it should not accidentally be treated as a response log probability or blindly multiplied by an answer advantage. The model architecture and routing mathematics otherwise remain part of the differentiable policy.

Likewise, your latent-cache generation need not be differentiated through. One can generate without a graph and recompute probabilities on the sampled histories, just as nanochat does. The extra work lies in making the probability distribution, masks, and sampling transformation agree with the intended estimator.

These are comparisons and design implications. Neither implementation is changed here.

## 17. Small experiments and checks performed

**Gist:** the important RL ideas can be checked using small categorical distributions and synthetic trajectories. Those checks reveal gradient signs, baseline effects, and normalization changes without training a language model.

### 17.1 Compute advantages before touching the model

**Gist:** first verify which responses should be promoted, suppressed, or receive no fresh signal. This separates reward accounting from transformer computation.

```python
import torch

rewards = torch.tensor([1.0, 0.0, 1.0, 0.0])
advantages = rewards - rewards.mean()
print(advantages)  # tensor([0.5, -0.5, 0.5, -0.5])

for rewards in (torch.zeros(4), torch.ones(4)):
    assert torch.equal(rewards - rewards.mean(), torch.zeros(4))
```

Then try one success among sixteen and compare the magnitudes for the successful response and each failure. The negative weight on each failed response is small, but there are fifteen such responses.

### 17.2 A one-token policy-gradient example

**Gist:** one softmax distribution is enough to see that positive and negative advantages reverse the local CE direction.

```python
import torch

for advantage in (0.5, -0.5, 0.0):
    logits = torch.tensor([0.0, 0.0], requires_grad=True)
    logp = logits.log_softmax(dim=0)[1]
    loss = -advantage * logp
    loss.backward()
    print(advantage, logits.grad)

# +0.5 -> [ 0.25, -0.25]: descent favors sampled token 1
# -0.5 -> [-0.25,  0.25]: descent suppresses sampled token 1
#  0.0 -> [ 0.00,  0.00]
```

This isolates the supplied gradient before optimizer momentum, matrix transformations, or shared transformer parameters complicate the final parameter update.

### 17.3 Useful later ablations

**Gist:** change one estimator or sampling decision at a time, and record actual token/action counts as well as success. These are suggested experiments, not launched training runs.

| Change | Question it investigates |
| --- | --- |
| Temperature one, top-k disabled | What changes when sampling and full-softmax scoring agree? |
| Preserve terminal actions in a scratch loss path | How much gradient comes from termination decisions? |
| Group mean versus leave-one-out baseline | Is the finite-group rescaling the only change under controlled fixed-length sampling? |
| One group-wide denominator versus per-pass means | How much does device batch grouping affect gradients on identical trajectories? |
| Change candidates per question | How do mixed-group frequency, rollout cost, and success trade off? |
| Track a broader assistant evaluation before/after RL | Does GSM8K improvement coexist with retention of other behavior? |

The most informative first comparison uses **the same frozen sampled trajectories** to compare losses and gradients under different reductions. Only afterward does a new sampling/training run need to test whether that mathematical difference improves outcomes.

### 17.4 What was checked for this note

**Gist:** isolated actual source definitions and exact small-distribution enumeration verified the loss mechanics and several implementation boundaries. No real rollout model, dataset download, distributed training job, or RL quality benchmark was used.

Checks used CPU PyTorch 2.11.0 and synthetic data:

- The actual GSM8K verifier reproduced string equality, missing-marker failure, comma normalization, and first-match behavior.
- The actual engine batch wrapper retained forced-token zero masks and removed sampled terminal tokens from returned sequences.
- The actual `get_batch` function, with in-memory questions and a scripted rollout provider, reproduced shifted prompt/action/tool/padding masks, rewards, group advantages, and changing chunk seeds. It also reproduced floor-truncated sample counts and the empty-group failure.
- The actual policy-gradient loss statements matched the analytical signed CE gradient and ignored-position zeros. Splitting fixed trajectories into differently sized passes changed the accumulated gradient as predicted. Equal rewards produced zero fresh PG gradients.
- Exact enumeration over small independent categorical groups verified the $(S-1)/S$ self-baseline factor for the ideal complete-policy sequence estimator.
- A two-token top-2 behavior-policy example with exact two-sample group enumeration showed a mismatch between full-softmax-score and behavior-policy-score estimators. This was a mathematical counterexample, not a measurement of the effect in trained nanochat.
- Mixed-group probabilities, horizon rounding, LR endpoint arithmetic, Markdown structure, and local code-line targets were checked.

The scratch checks execute selected definitions without importing the RL script's top-level training code. They do not establish GPU numerical equivalence, distributed-process behavior, benchmark accuracy gains, or complete reproducibility. Both repositories remain untouched.

## 18. Putting the complete pipeline together

**Gist:** the core nanochat pipeline is now a chain of artifacts and probability objectives. Pretraining builds a general next-token model, SFT teaches demonstrated assistant behavior, and optional RL changes sampled behavior using task rewards.

### 18.1 Follow the artifacts from data to an RL assistant

**Gist:** each stage consumes a concrete artifact from earlier stages. The tokenizer and conversation protocol must remain compatible across the entire chain.

| Stage | Inputs | Learning or measurement | Main output |
| --- | --- | --- | --- |
| Tokenizer preparation | Text for BPE training | Vocabulary/compression construction | Tokenizer and token-byte metadata |
| Pretraining | Curated ClimbMix documents, tokenizer | Next-token likelihood across document targets | Base weights and optimizer/checkpoint metadata |
| Base evaluation | Base checkpoint and evaluation tasks | BPB, CORE, samples | Measurements for the base model |
| SFT | Base model, formatted demonstration mixture | Assistant/action-target masked CE | SFT assistant weights and optimizer shards |
| Chat evaluation/inference | Assistant checkpoint, prompt format, engine | Task scores and interactive generation | Responses and assistant measurements |
| Optional RL | SFT weights, GSM8K questions and answer checker | Sampled group-relative policy-gradient surrogate | RL weights and model configuration |

The [speedrun](../nanochat/runs/speedrun.sh#L62) connects pretraining, base evaluation, SFT, and chat evaluation. It does not launch this RL trainer. RL is an additional explicitly invoked stage and should not be silently included when describing the speedrun's total compute or reported results.

### 18.2 The central distinctions to retain

**Gist:** the architecture stays largely fixed while the source of training trajectories and their learning weights change. The exact data distribution, masks, and reduction are part of the algorithm.

For pretraining, the signal is: predict the actual next document token. For SFT: predict the demonstrated assistant token at selected positions. For this RL stage: increase or decrease the selected sampled action's log probability according to its response's group-relative reward, with the source's particular normalization.

The compact RL chain is

$$
\boxed{
\text{question}
\longrightarrow\text{sampled trajectories}
\longrightarrow(R_i,A_i)
\longrightarrow\text{masked log probabilities}
\longrightarrow\text{weighted gradient}.
}
$$

This completes the core stages in the approved roadmap. Larger preference-learning pipelines, DPO, full actor-critic PPO, learned reward models, large-scale serving, and upstream corpus curation remain supplementary topics beyond the code implemented here. We can unpack the RL derivation or any implementation detail before choosing one of those extensions.

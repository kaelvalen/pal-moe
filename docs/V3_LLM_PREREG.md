# V3-LLM-1: single-fact learning after deployment on a frozen 7B LM - pre-registration

Status: **proposed, not started, not approved.** Written in the v3 restructure (phase
4) together with the harness it would use (`pal_moe/core/hf_lm.py`,
`pal_moe/api/lm.py`, `pal_moe/edit/down_proj.py`, `pal_moe/eval/editing.py`). No LLM
number exists in this repository yet; nothing below is a result.

## 1. The question

> On a frozen, 4-bit, 7B causal LM, can the v3 system learn a single new fact **at any
> time after deployment by an API call** - through the FAST path (retrieval of the
> written sentence over frozen hidden-state keys, injected into the context) or the
> MEDIUM path (a closed-form, float64, order-invariant, subtractable edit of one MLP
> down-projection) - with the four guards holding on every call, and how do the two
> paths trade efficacy against locality as the number of sequential edits grows?

This is the first LM study. It asks whether the v3 contract (learning = `write`,
unlearning = `forget`, guards always on) is *implementable* at 7B scale with usable
editing quality - not whether v3 beats the knowledge-editing literature.

## 2. Design

**Base model (pinned before the run).** `Qwen/Qwen2.5-7B` (open weights, Apache-2.0),
4-bit NF4 via bitsandbytes, bf16 compute, the Hub revision hash recorded in the run
record as part of `base_hash`. One model, no alternates: if it cannot be loaded, the
study stops.

**Layers (pinned by a development smoke, see section 5).** Key layer `L_key` and edit
layer `L_edit` are chosen on 50 CounterFact cases **disjoint from the evaluation set**,
from the grid `L in {4, 8, 12, 16, 20}`, by FAST retrieval top-1 hit rate (key layer)
and MEDIUM efficacy at N = 1 (edit layer). The smoke's evaluation-set numbers are never
computed.

**Key-covariance prior.** `C0 = 15000 * E[k k^T] + 1e-4 I` over the edit layer's
down-projection inputs on 100k tokens of WikiText-103 train (the MEMIT recipe's
weighting), float64 on CPU, hashed and recorded.

**Data.**

```text
CounterFact   the first 2000 cases of the ROME release (counterfact.json)
zsRE          the first 2000 cases of the MEND eval split
MQuAKE        MQuAKE-CF-3k, the first 500 cases
canary        500 fixed prompts: 250 WikiText-103 test sentence prefixes + 250 zsRE
              `loc` questions not in the zsRE eval subset; hashed
```

**Arms.**

```text
base      no write
fast      write(prompt + " " + target_new) into the FAST memory, k = 1,
          tau = 0.95 (cosine), retrieved text prepended at predict time
medium    write(Batch([prompt], [target_new])): closed-form down-projection edit,
          target values by 20 steps of activation-space optimisation (no weight trained)
both      fast + medium for the same fact
```

**Edit regimes.** `N in {1, 100, 1000}` facts written sequentially (N = 1: each case in
isolation, then forgotten; N = 100 / 1000: consecutive cases, evaluated after the last
write). MQuAKE: all edits of a case written, then its questions asked (N = case size).

## 3. Endpoints

**Guards (every call; a failure stops the run, it is not a result).**

```text
reversibility   write(x); forget(id) restores state(), the delta and all canary
                logits bitwise - 100 % of calls, checked by the API on every write
                and on every forget
order           N = 100: the same 100 facts written in two permutations give bitwise
                identical deltas (canonical solve) and identical canary argmax
purity          router (retrieval index) trainable parameters 0, asserted
base integrity  base_hash identical before and after the whole run
locality        canary argmax flip rate and max |delta logit| logged for every write
```

**Primary (CounterFact, N = 1000, fast vs medium, paired over cases):**

```text
E   efficacy       P(new) > P(true) on the rewrite prompt
S   specificity    P(true) > P(new) on neighbourhood prompts (ROME's NS)
```

**Secondary:** paraphrase (PS), the same three at N = 1 and N = 100, zsRE efficacy /
paraphrase log-prob and neighbourhood |delta log-prob| against base, MQuAKE multi-hop
case accuracy, canary flip rate vs N, wall time per write and per forget, bytes stored
per fact on each path.

## 4. Outcome table (fixed in advance)

| result | reading |
| :-- | :-- |
| every guard passes on every call, and `E >= 0.9` for at least one path at N = 1000 | the v3 contract is implementable at 7B: facts can be learned and exactly unlearned post-deployment by API calls. Report the E/S trade-off per path |
| guards pass, `E(fast) >= 0.9`, `S(fast) >= S(base) - 0.02`, `E(medium) < 0.9` at N = 1000 | retrieval carries single facts at scale and the closed-form edit does not; the medium path is for batches, not facts - revise the v3 routing of `write(str)` accordingly |
| guards pass, `E(medium) >= 0.9` and `S(medium) < S(base) - 0.05` at N = 1000 | the edit learns but interferes; the covariance prior is not protecting the canary / neighbourhood keys at this N |
| MQuAKE `both` > max(`fast`, `medium`) by >= 5 pp | the paths are complementary for integration; motivates the SLOW path (consolidation) on the LM |
| any reversibility / order / purity guard fails | a defect: the run stops and is reported as an implementation failure, not a scientific outcome |

## 5. Vetoes and the development smoke

```text
disjoint smoke set     the 50 layer-selection cases share no case_id or subject
                       with the evaluation subsets (asserted)
pinned before run      model revision, L_key, L_edit, tau, C0 hash written into this
                       document's section 5 (a dated amendment) before evaluation
no tuning on eval      no hyperparameter is changed after the first evaluation case
                       is scored
feasibility            measured cost projects the grid under 24 h on the RTX 5060
                       (8 GB) + 62 GB RAM machine (C0 for d_ff = 18944 is 2.9 GB
                       float64 on CPU; the model ~5 GB on GPU)
```

## 6. Statistics

Case-level pairing (the same cases in every arm). Proportions with Wilson 95 %
intervals; paired arm contrasts by exact McNemar on the discordant cases; secondary
endpoints with case-level bootstrap intervals (10,000 resamples, fixed seed). No
multiplicity correction is claimed across secondary endpoints; they are read together,
not tested.

## 7. Out of scope

```text
the LM SLOW path          consolidation into frozen LoRA experts: declared, not built
parametric value injection   the FAST path uses retrieved text in context only
comparison to ROME / MEMIT   not reimplemented here; the medium path uses the MEMIT
                             closed form, which is stated, not claimed as new
other models / sizes      one model per study
```

## 8. What this licenses, and what it does not

- **Licenses:** that the v3 API contract holds on a 7B LM (with numbers), and the
  observed efficacy / specificity / locality of each path at N in {1, 100, 1000}.
- **Does not license:** a state-of-the-art editing claim, any claim about other models
  or larger N, or any claim about the SLOW path.

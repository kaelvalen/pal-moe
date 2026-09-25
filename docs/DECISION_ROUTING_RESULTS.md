# Decision Routing: comparative supervision - results

Status: **run. The comparative arm is worse than the pointwise arm in every seed
and both regimes, but the family-wise corrected test does not reject at 0.05.**
Pre-registration: `docs/DECISION_ROUTING_PREREG.md`; the frozen diagnosis it
follows: `DIAGNOSIS_SYNTHESIS.md`.

## 1. The three arms (six seeds)

| arm | `coherent` C@3 | `dispersed` C@3 |
| :-- | --: | --: |
| prototype router (training-free control) | **0.9494** | **0.8915** |
| pointwise gate (prototypes + cross-entropy) | 0.9435 | 0.8531 |
| comparative gate (prototypes + owner-vs-all hinge) | 0.9294 | 0.8279 |

Primary, `comparative - pointwise`, paired by seed:

| contrast | per-seed (six) | mean | sd | exact permutation p | WY-adjusted p |
| :-- | :-- | --: | --: | --: | --: |
| `Delta C@3`, `coherent` | -0.0145 -0.0111 -0.0168 -0.0137 -0.0127 -0.0160 | **-0.0141** | 0.0021 | 0.0312 | **0.0938** |
| `Delta C@3`, `dispersed` | -0.0229 -0.0256 -0.0267 -0.0238 -0.0266 -0.0254 | **-0.0252** | 0.0015 | 0.0312 | **0.0938** |
| `Delta C@3` excluding the first task, `coherent` | all negative | -0.0140 | 0.0027 | 0.0312 | 0.0938 |
| `Delta C@3` excluding the first task, `dispersed` | all negative | -0.0270 | 0.0015 | 0.0312 | 0.0938 |
| accuracy, `coherent` | all negative | -0.0266 | 0.0027 | 0.0312 | 0.0938 |
| accuracy, `dispersed` | all negative | -0.0479 | 0.0025 | 0.0312 | 0.0938 |
| conditional oracle@3, `coherent` | mixed signs | -0.0001 | 0.0004 | 0.7500 | 1.0000 |
| conditional oracle@3, `dispersed` | mixed signs | +0.0001 | 0.0002 | 0.2812 | 0.8125 |

**The honest reading: `Delta C@3` is negative in 6/6 seeds in both regimes, and
both regimes' raw exact tests sit at the six-seed floor of 0.0312 - but the
Westfall-Young correction over the family gives 0.0938, so the pre-registered
primary does not reject at 0.05.** The effect is small (-0.014 / -0.025 coverage)
and the direction is unambiguous. Per the pre-registration's outcome table this
is the fourth row - "the pairwise objective fits worse at this formulation and
scale" - with the added caveat that the evidence is directional rather than
family-wise significant.

## 2. What the guards establish

- **Both anchors reproduced.** The prototype router returns exactly
  `0.9494 / 0.8915` (its RRF value), so the bank, prototype and evaluation paths
  are identical to the earlier studies. The pointwise gate is deterministic under
  the pinned seeding rule: two independent processes produced identical
  `gate_param_hash` and identical `C@3` for all three arms.
- **The manipulation is loss-only.** Both trained arms share the same evidence
  (stored prototypes), the same gate construction, the same seeding order, the
  same optimizer, epochs and learning rate; only the loss differs.
- **The difference is in the ranking, not in the experts.** `conditional_oracle@3`
  moves by ~0 (-0.0001 / +0.0001, not significant). So unlike the
  Representation x Routing factorial, nothing here trades within-expert accuracy
  for cross-expert separability - the comparative loss simply ranks worse.

## 3. What this leaves

The synthesis named the invariant every attempt has kept - one independent score
per expert, a competition, a single winner - and the pre-registration split its
two components. The **pointwise** half is now tested: comparative supervision is
not better. The **winner-take-all decision** is therefore the last untested piece
of the invariant, with two independent negative results behind it (pointwise
supervision, comparative supervision) and a positive interaction signal from the
factorial that the loss and the representation are not unrelated.

## 4. What this does not say

- It does not say pairwise objectives are wrong in general: this is one scale,
  one evidence source (stored prototypes), one margin and one expert
  formulation, with the decision rule held fixed.
- It does not reopen Stage 1, the Router Ranking Study or the factorial, all of
  which stand as reported.
- The 0.0938 is not "no effect": it is a consistent negative direction that the
  family-wise correction declines to certify at six seeds. The smallest p-value
  six seeds can produce is 0.0312, and the correction multiplies it.

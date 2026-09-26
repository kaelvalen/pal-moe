# HEADROOM: is the frozen representation insufficient? - a dataset screen - pre-registration

Status: **proposed, not started.** Written after `P2_BOUND_RESULTS.md`. That study found
the bank's value path, not its boundaries, limits P2: on the frozen ViT-B/16 CIFAR-100
cache every grouping lands within 0.7 pp of a ridge readout. The next study needs a
setting where adapting the representation can matter at all. This screen chooses that
setting, or shows there is none among the candidates.

## 1. The question

> On which candidate dataset / protocol is there a large gap between the best
> **frozen** closed-form readout and what an **adapted** representation can reach,
> both read out the same way (ridge)?

The selection criterion is not difficulty. It is that frozen `z` is insufficient. With
no gap, an expert has nothing to add, however hard the data is.

## 2. Candidates and protocols

```text
DomainNet  6 domains x 345 classes, official split files (domainnet/txt/*_{train,test}.txt)
           train 409,832 / test 176,743
  - Domain-IL (PRIMARY): 6 tasks = 6 domains (clipart, infograph, painting, quickdraw,
             real, sketch - in this order), all 345 classes in every task
  - Class-IL (SECONDARY): 15 tasks x 23 classes, class order = torch.randperm(345) with
             seed 1993, every domain in every task
ImageNet-R 200 classes, 30,000 images (Class-IL)
ImageNet-A 200 classes, 7,450 images (Class-IL)
```

**Splits for ImageNet-R / -A.** Neither ships a train/test split. We use a stratified
80/20 split per class, `seed = 1993`. The file lists are written to
`data/splits/*.txt` and their sha256 is recorded in amendment 1 before any number is
computed. This is **not** PILOT's split, so literature numbers are not directly
comparable. Only this screen's arms are compared with each other.

**Backbone (pinned).** `timm` `vit_base_patch16_224.augreg_in21k` (ViT-B/16-IN21K, as in
PILOT / EASE), frozen, base weight hash `b92dc1a8cf3a...` (`AdaptedViT.base_hash`, full
value in the study JSON). Eval preprocessing: `Resize(256), CenterCrop(224)`, the model's
own normalisation.

## 3. Arms

All readouts are float64 closed-form ridge (`pal_moe.edit.LinearStats`). The ridge
`lambda` is selected per arm from `{1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4}` on a seeded 10 %
held-out slice of the **train** split, then refitted on the full train split. The test
split is never used for any choice.

| arm | what | protocols |
| :-- | :-- | :-- |
| `F1` | frozen [CLS] + ridge | all |
| `F2` | RanPAC-style: frozen [CLS] -> fixed random projection `W in R^{768 x 10000}`, `W ~ N(0,1)` (seeded), ReLU -> ridge. **Without** RanPAC's first-session adapter (that would be an adapted arm) | all |
| `C` | **joint adapter ceiling**: one adapter set (EASE form: MLP side branch, r = 16, identity at init, `pal_moe.core.vit_adapter`) trained on all train data at once with a linear head. Reported **twice**: its own linear head (`C_head`), and a ridge refitted on the adapted [CLS] (`C_ridge`) | all |
| `F1o`, `F2o` | Domain-IL only: `F1` / `F2` with one ridge **per domain** over all 345 classes, the true domain given at test (the frozen counterpart of per-domain experts with oracle routing) | DomainNet Domain-IL |
| `Co` | Domain-IL only: **per-domain adapter ceiling**: one adapter set per domain, trained on that domain only; readout one ridge per expert over all 345 classes; the true domain given at test. This is the oracle-routing ceiling of the planned MoE | DomainNet Domain-IL |
| `D` | Domain-IL only: **domain-ID accuracy of the frozen [CLS]**, by (i) the parameter-free rule (nearest domain mean, cosine) and (ii) a closed-form ridge `z -> 6`. The routing tax forecast | DomainNet Domain-IL |

**Adapter recipe (one recipe, no tuning).**

- Optimiser: AdamW, lr 1e-3, weight decay 0, batch 64, bf16 autocast, cosine schedule.
- Augmentation: `RandomResizedCrop(224, scale=(0.5, 1))` + horizontal flip.
- Epochs: DomainNet 3 (`C`), `Co` the same number of optimiser steps in total split
  over the six domains in proportion to their size; ImageNet-R / -A 10.
- The linear head is trained jointly with the adapters (cross-entropy).

## 4. Endpoints

**Primary gap (ridge vs ridge; point 1 of the review):**

```text
G_joint  = C_ridge - max(F1, F2)                      every dataset / protocol
G_DIL    = Co_ridge - max(F1o, F2o)                   DomainNet Domain-IL (primary for it)
```

`C_head - C_ridge` is reported so that a gap produced by the head rather than the
representation is visible, but the head does not enter the decision.

**Read together:** per-domain accuracies and per-domain `G_DIL` (quickdraw / infograph
are the expected carriers); `D` (both rules); `C_head`; the ceiling's train-split
accuracy (undertraining guard); F2 minus F1.

## 5. Outcome table (fixed in advance)

| result | reading |
| :-- | :-- |
| DomainNet Domain-IL: `G_DIL >= 10` and `D >= 98 %` (parameter-free rule) | **selected**, and routing is not the bottleneck. The next study tests the expert's adaptation against frozen ridge directly: per-domain frozen plastic experts + parameter-free router + fusion readout (section 8) |
| DomainNet Domain-IL: `G_DIL >= 10` and `D < 98 %` | selected, but a routing tax enters. `1 - D` is reported as its forecast and the next study must carry the fusion readout (section 8) as primary |
| any other dataset / protocol: `G_joint >= 10` | eligible. Domain-IL is preferred when it qualifies; otherwise the largest `G_joint` is selected |
| `5 <= G < 10` everywhere it is measured | no dataset selected. Report the gaps; the image line is not continued on these candidates |
| `G < 5` everywhere | frozen `z` suffices on these images: the image line closes and the programme moves to the LLM path (`V3_LLM_PREREG.md`) |
| the ceiling's `C_ridge < F1` on a dataset | the adapter recipe failed there (an identity-initialised adapter plus ridge cannot be worse than frozen ridge at the optimum). That dataset's reading is **withheld**, not read as "no headroom"; a recipe change needs an amendment |

## 6. Vetoes

```text
backbone          base weight hash identical in every arm and cell
identity at init  adapted [CLS] == frozen [CLS] bitwise before the first step
split integrity   split file sha256 as recorded in amendment 1; no test file in any
                  lambda selection, training step or normalisation statistic
ridge anchor      F1 on DomainNet Domain-IL reproduced bitwise by a second run of the
                  same seed (determinism)
ceiling fit       the ceiling's train-split accuracy is reported; if it is below its
                  test accuracy + 5 pp the fit is flagged (possible undertraining) and
                  noted next to every reading that uses it
feasibility       sustained throughput from the smoke (section 7) projects the grid
                  under the ceiling below
```

## 7. Seeds and feasibility

**Seeds.**

- Frozen arms: six seeds (42, 1, 2, 3, 4, 5). The seed changes the held-out slice for
  `lambda` and RanPAC's projection. These arms cost one feature pass.
- Adapter ceilings (`C`, `Co`): **two seeds (42, 1).** Rationale: the decision
  threshold is 10 points, and seed-to-seed spread for adapters on frozen backbones in
  this programme has been ~0.1-0.5 pp (S11, E-TID2 standard deviations). If the two
  seeds' gaps fall on opposite sides of a threshold (5 or 10), a third seed (2) is run
  for that dataset and the decision uses the mean of three.

**Feasibility - measured by the development smoke, filled in as amendment 1 before any
screen number is computed.** Procedure fixed now:

```text
experiments/headroom_smoke.py   train >= 10 min (adapter fine-tune, batch 64, bf16),
                                extract 4 min, JPEG decode 3 min (16 workers);
                                the reported value is the mean of the LAST 5 minutes
                                (train) - the laptop GPU throttles, so the first
                                minute is not used
GPU otherwise idle              the first attempt (2026-09-26) ran out of memory because
                                another application held 5.7 GB of the 8 GB; the smoke
                                is repeated with the GPU idle
ceiling                         the whole screen <= 48 h wall clock; if projected over,
                                ImageNet-A and DomainNet Class-IL `C` are dropped first
                                (in that order) and the drop is recorded
```

## 8. Draft for the next study (not tested here)

Recorded now so the next pre-registration cannot be shaped by this screen's numbers.
P2's lesson: do not throw away the router's posterior or the frozen readout.

```text
final(x) = ridge_frozen(z) + alpha * sum_{e in top-k} w_e * ridge_e(z_e)
    z   = frozen [CLS] (the address); z_e = [CLS] with expert e's adapters
    w_e = router posterior over the top-k (parameter-free scores, softmax, T = 1)
    alpha selected on a train held-out slice only
primary:   top-1 routing
secondary: top-2, reported with an "expert forwards per sample" column
baselines: F1 / F2, one shared adapter trained sequentially (sanity only: F6 already
           shows it forgets), EASE at an equal number of adapters (dense ensemble:
           T adapter forwards per sample), on the same split
```

## 9. Out of scope

```text
any continual training      this screen measures offline gaps only
the MoE itself, EASE runs   the next study
PILOT integration           needed for EASE on DomainNet (PILOT has no DomainNet loader);
                            part of the next study's harness
other backbones             one backbone per screen
```

## 10. What this licenses, and what it does not

- **Licenses:** a statement, per candidate and protocol, of how much an adapted
  representation can gain over the best frozen closed-form readout under the same
  readout; and a routing-tax forecast for DomainNet Domain-IL.
- **Does not license:** any claim about a continual or MoE system, any claim that the
  gap is reachable under the continual constraint, or any claim beyond this backbone,
  these splits and this adapter recipe.

## Amendment 1 (2026-09-26, after review, before the smoke and before any screen number)

**1. The `C_ridge < F1` row of section 5 is replaced.** Its justification was too
strong. "An identity-initialised adapter plus ridge cannot be worse than frozen ridge
at the optimum" holds for a train objective, not for test accuracy:

- the adapters are trained through a linear head with cross-entropy, so
  ridge-on-adapted-features is not what is optimised;
- adapted features can overfit the train split, and then a lower test `C_ridge` is a
  real finding (the adaptation does not generalise), not a recipe error.

`C_ridge` and `F1` (and, for Domain-IL, `Co_ridge` and `F1o`) are therefore reported
**on the train split as well** (in-sample ridge fits on both sides). The row becomes:

| train `C_ridge` vs `F1` | test `C_ridge` vs `F1` | reading |
| :-- | :-- | :-- |
| < | < | recipe failure: the dataset's reading is withheld; a recipe change needs an amendment |
| < | >= | the adapter did not fit the train split. Treated as a recipe failure too (withheld); the test ordering is not interpreted |
| >= | < | the adaptation overfits: **no headroom** for that dataset - a valid result, not a failure |
| >= | >= | normal reading (the gap thresholds of section 5) |

The same table applies to `Co_ridge` vs `F1o` for `G_DIL`.

**2. Split hashes** (the ImageNet-R / -A split of section 2; DomainNet's official lists):

```text
data/splits/imagenet-r_train.txt   24002  ab209e9534c44f5c74182e7081d39658ba5fd756663a28698f4547ef199973ef
data/splits/imagenet-r_test.txt     5998  eba499f87e6246aa62ad0fdc1b70bf72ee23a45fe3b41f9e36b577c717e2ae75
data/splits/imagenet-a_train.txt    6000  9a1ed24a6ba2d20c86235510d2121b5d4bf476f8f0f724a7a6d92f261e475b1e
data/splits/imagenet-a_test.txt     1500  f581353c379def4f96151ffffe42042228976d1316265f92ef5c0dd10b1c6e78
domainnet/clipart_train.txt               affdadf5f95a7583e6b98030a3d35007e2613ee4dd942393c5a705ec2b429312
domainnet/clipart_test.txt                62c8e36aaba1c41ad9e249099739707220a44c08f167b37ef420f34a31628a6b
domainnet/infograph_train.txt             36b44cbd41a2915e0dc73b4f4be8862a6f3c220f93c8d63fb2439beeae3aefda
domainnet/infograph_test.txt              413cfa54ac92e7e6b242f4f09fbd9c8ace85c4deaf3cb6c81ff6b68264460136
domainnet/painting_train.txt              f1da38d50a702fddf1329f85bc607d7a2abcae7e6aec75e1fd49cc1bd340d47b
domainnet/painting_test.txt               11472b13b5188e09f06918b12eebb660588c97a20e43e268d9b02e3e37eb8f97
domainnet/quickdraw_train.txt             3a5edd3bc215772010eebc87f158707703dd7e48091d13b816af81326a9d1c0f
domainnet/quickdraw_test.txt              2a00a60650a453c65c356da748fec54084a197272137c440e7d52d08947357bf
domainnet/real_train.txt                  19d483256d24aef19e581b65eda8333fd918aef3eb733fe499ff384120e3a289
domainnet/real_test.txt                   2b3f75bcde309aeb5931292084b2162e936b5f189e3bb1ad716f0f74c460864c
domainnet/sketch_train.txt                72f4bc6afa5702a1ccba800d0053c1427de87439dfaf246dbc9a2e71abfec54f
domainnet/sketch_test.txt                 3fb48fca18c507c4321f007c3f62b308ec232ff6928cbfe2af64e76ec366e840
```

**3. Split comparability.** The ImageNet-R / -A numbers use our own stratified
80/20 split, so they are **not directly comparable with published PILOT / EASE
numbers**. Every arm in this programme is run by us on the same split. Any paper
using them states this in one sentence.

**4. Feasibility moves to amendment 2.** Section 7 said it would be filled in as
amendment 1. It will be amendment 2, written from the smoke before any screen number.

## Amendment 2 (2026-09-26, from the development smoke, before any screen number)

**Feasibility, measured** (`experiments/headroom_smoke.py`, `results/headroom/smoke.json`).
GPU otherwise idle, sustained values = mean of the last 5 minutes (train) / 3 minutes
(extract) / 2 minutes (decode):

```text
adapter training (batch 64, bf16, r = 16, all 12 blocks)   204 img/s   (first minute 185)
frozen / adapted feature extraction (batch 256, bf16)       495 img/s   (first minute 491)
JPEG decode + resize (16 workers, ImageNet-R files)          979 img/s
peak GPU memory (training)                                  3.4 GB of 8 GB
GPU temperature 77-87 C, SM clock 2.5-2.7 GHz: no throttling beyond warm-up visible
```

Training and extraction are GPU-bound (decode is ~2x faster), **on ImageNet-R files**.
DomainNet's `real` / `painting` images are larger, and their decode rate was not
measured. If decode becomes the bound, wall time grows; the decision is unaffected.

**Projection:**

```text
frozen features       DomainNet 586,575 imgs / 495 = 20 min; ImageNet-R 1 min; -A 15 s
ceiling, per seed     DomainNet C: 3 epochs x 6,403 steps x 64 / 204 = 1.7 h
                      + adapted extraction 20 min; Co: the same steps 1.7 h + 20 min
                      -> ~4.0 h per seed; ImageNet-R ~21 min; ImageNet-A ~6 min
two ceiling seeds     ~8.0 h (DomainNet) + ~0.9 h (ImageNet-R/-A)
frozen arms (CPU)     not measured (float64 ridge; RanPAC at 10,001 dims dominates)
total                 ~10-12 h projected, under the 48 h ceiling: nothing is dropped
```

**Protocol clarification (no change of arms).** The screen is offline. DomainNet
Class-IL and Domain-IL use the same train data and the same 345-way test, so their
global arms `F1`, `F2` and `C` are one computation: DomainNet Class-IL's `G_joint` *is*
DomainNet's `G_joint`. The protocols differ only in the Domain-IL oracle arms (`F1o`,
`F2o`, `Co`, `D`), which are computed as registered. The class-task partition of
section 2 matters only for the next (continual) study.

**Trainable parameters of the ceiling:** adapters 304,320 (12 blocks x 2 x (768 x 16 +
biases)) + linear head (768 x C + C).

# E-TID: the offline task-ID ceiling on the frozen feature space - results

Pre-registration: **none committed.** The question, the five arms, the anchor and the
three reading thresholds were fixed in the docstring of `experiments/e_tid_ceiling.py`
before it was run, but that script was not committed before the run (it was first
committed in the Phase 0 record commit of `v3-restructure`, after the JSON existed).
Read everything below as **exploratory**: the thresholds are stated as they were
written, and no confirmatory weight is claimed for them.

Data: `results/e_tid/e_tid_ceiling.json` - 6 cells (2 constructions x seeds 42, 1, 2),
30 epochs per learned probe, `k = 20` for k-NN, device `cuda`.

Status: **run, anchor passing. The offline class-level linear probe is +7.7 pp
(`coherent`) / +10.4 pp (`dispersed`) above the prototype router's `C@1`.** Under the
script's own thresholds that is "partly continual" for `coherent` and "mainly a
continual-learning constraint" for `dispersed`. The routing tax is not the Bayes error
of task-ID given `z`.

## 1. The question

Is the routing tax a property of `z` (representation-bound), or of the continual
constraint (sequential fit, stored evidence)? Every arm sees **all tasks' training data
at once** - no sequential constraint - on the same frozen ViT-B/16 CIFAR-100 cache and
the S6b `coherent` / `dispersed` constructions at `T = 20`.

| arm | rule |
| :-- | :-- |
| `proto` | nearest class mean (cosine), max over the task's classes - the E0 rule, anchor |
| `knn` | cosine k-NN over all training features, task = similarity-weighted vote |
| `lin_task` | joint linear softmax `z -> T` |
| `lin_class` | joint linear softmax `z -> 100`, task = owner of the top class (max log-prob per task) |
| `mlp_task` | joint 2-layer MLP `z -> T` (1024 hidden, GELU, dropout 0.1) |

Readings, as written in the docstring:

```text
best_offline C@1 - proto C@1 <  2 pp  -> representation-bound; close the routing line
2 pp .. 8 pp                          -> partly continual; measure which stored evidence recovers it
> 8 pp                                -> the tax is mainly a continual-learning constraint
```

## 2. Coverage (means over three seeds)

| construction | arm | `C@1` | `C@3` | `C@1 - proto` |
| :-- | :-- | --: | --: | --: |
| `coherent` | `proto` | 0.8195 | 0.9494 | - |
| `coherent` | `knn` | 0.8556 | 0.9563 | +3.61 pp |
| `coherent` | `lin_task` | 0.8728 | 0.9708 | +5.33 pp |
| `coherent` | **`lin_class`** | **0.8967** | **0.9770** | **+7.72 pp** |
| `coherent` | `mlp_task` | 0.8874 | 0.9726 | +6.79 pp |
| `dispersed` | `proto` | 0.7130 | 0.8915 | - |
| `dispersed` | `knn` | 0.7613 | 0.9134 | +4.83 pp |
| `dispersed` | `lin_task` | 0.7569 | 0.9198 | +4.39 pp |
| `dispersed` | **`lin_class`** | **0.8170** | **0.9472** | **+10.40 pp** |
| `dispersed` | `mlp_task` | 0.7937 | 0.9310 | +8.07 pp |

Per seed, `lin_class` `C@1`: `coherent` 0.8965 / 0.8964 / 0.8972, `dispersed`
0.8170 / 0.8169 / 0.8170. `proto` and `knn` are deterministic (identical across seeds).

## 3. Guards

```text
anchor         proto C@1 against E0 (0.8195 / 0.7130): max |delta| 3.1e-08 / 4.8e-10
train fit      lin_task train C@1 0.9265 / 0.8239   (does NOT fit the training set)
               mlp_task train C@1 0.9999 / 0.9996   (fits it)
```

The train-fit guard was there so that an undertrained probe could not fake a
"representation-bound" reading. It did its job in the other direction: `lin_task`
underfits (it cannot even separate the training tasks linearly at 93 % / 82 %), which
is why a task-level linear probe is the wrong router and a class-level one is not.

## 4. Reading

- Best offline arm is `lin_class` in both constructions. `coherent` +7.72 pp falls in
  the 2-8 pp band ("partly continual"), `dispersed` +10.40 pp above 8 pp ("mainly a
  continual-learning constraint").
- The ordering `lin_class > mlp_task > lin_task` says the useful structure is at the
  **class** level: a task is a union of classes, and in `dispersed` a task is a union
  of classes from unrelated superclasses, which a task-level linear boundary cannot
  carve. Mapping the class decision to its owner task is what recovers it.
- Consequence taken forward: E-TID2 asks whether a class-level linear router that is
  reachable **under the sequential constraint** (continual ridge, whose sufficient
  statistics are additive) realises this ceiling.

## 5. What this does not say

- It is not a confirmatory result: no committed pre-registration, three seeds, and
  the thresholds were written by the same hand that ran the script.
- The arms are offline (all data at once). Nothing here says a continual router gets
  there; that is E-TID2's question.
- It says nothing about the value path (the experts or the readout), only about the
  selection's coverage.
- No claim beyond `T = 20`, these two constructions, and this ViT-B/16 cache.

## 6. Records

```text
experiments/e_tid_ceiling.py      the script (docstring = the readings)
results/e_tid/e_tid_ceiling.json  6 cells, untracked
this                              the write-up
```

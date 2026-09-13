# PAL-MoE: Prototype-Anchored Lifelong Mixture of Experts

[![CI](https://github.com/kaelvalen/pal-moe/actions/workflows/ci.yml/badge.svg)](https://github.com/kaelvalen/pal-moe/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-%3E%3D2.0-ee4c2c.svg)](https://pytorch.org/)

PyTorch implementation of PAL-MoE (Prototype-Anchored Lifelong Mixture of Experts) for continual learning without catastrophic forgetting.

---

## 1. Problem Formulation and Core Hypothesis

In standard neural networks, incorporating new knowledge updates shared weights, leading to the destruction of previously learned representations (**Catastrophic Forgetting**):
$$\text{New Information} \to \Delta \theta \to \text{Past Representations Corrupted} \to \text{Catastrophic Forgetting}$$

Mixture of Experts (MoE) provides modular localization (Expert is local, Router is global). However, standard MoE fails continually because:
1. **Router Drift**: The router's decision boundary shifts, redirecting old inputs to the wrong experts.
2. **Expert Drift**: Experts themselves undergo weight updates, corrupting old skills even if routed correctly.

### Core Hypothesis
> Dynamically localizing learning into specialized experts; anchoring both router and expert behaviors through a shared prototype stability mechanism; and expanding capacity only when quantitatively warranted via function-preserving derivation and validation gating drastically mitigates catastrophic forgetting in continual learning.

---

## 2. Mathematical Architecture

### Model Components
$$\begin{aligned}
h(x) &\in \mathbb{R}^d && \text{Shared Encoder (Frozen or EMA)} \\
g(x) &= \text{Softmax}(\text{Top-}k(W_r h(x) + b_r)) && \text{Dynamic Sparse Router} \\
E_i(h(x)) &\in \mathbb{R}^C && \text{Modular Expert Adapter / Head} \\
y &= \sum_{i \in \text{Top-}k} g_i(x) E_i(h(x)) && \text{Output Prediction}
\end{aligned}$$

### Prototype Memory $\mathcal{P}$
Stores historical behavioral anchors:
$$\mathcal{P} = \{ (v_p, r_p, o_p, x_p) \}_{p=1}^P$$
- $v_p \in \mathbb{R}^d$: Stable routing feature vector from encoder.
- $r_p \in \Delta^{N}$: Past router probability distribution $g(v_p)$.
- $o_p \in \mathbb{R}^{N \times C}$: Expert output anchors $E(v_p)$ across active experts.
- $x_p$: Bounded exemplar feature buffer for drift verification.

### Joint Stability Loss
$$\mathcal{L} = \mathcal{L}_{\mathrm{task}} + \lambda_r \mathcal{L}_{\mathrm{router}} + \lambda_e \mathcal{L}_{\mathrm{expert}}$$
where:
$$\mathcal{L}_{\mathrm{router}} = \frac{1}{|\mathcal{P}|} \sum_{p \in \mathcal{P}} \mathbb{D}_{\mathrm{KL}}\left( r_p \parallel g_{\mathrm{new}}(v_p) \right)$$
$$\mathcal{L}_{\mathrm{expert}} = \frac{1}{|\mathcal{P}|} \sum_{p \in \mathcal{P}} \sum_{i=1}^N r_p[i] \cdot \left\| E_{i, \mathrm{new}}(v_p) - o_p[i] \right\|_2^2$$

---

## 3. Quantitative Expert Creation Trigger

Rather than arbitrary heuristic expansion, new expert candidates are triggered strictly when existing experts are quantitatively insufficient:
$$S(x) = \alpha \mathcal{L}_{\mathrm{task}}(\mathrm{best}) + \beta \mathcal{H}(g(x)) + \gamma d(x, \mathcal{P}) - \delta \max_i \mathrm{conf}_i$$

- $\mathcal{L}_{\mathrm{task}}(\mathrm{best}) = \min_i \mathrm{CE}(E_i(h(x)), y)$
- $\mathcal{H}(g(x)) = -\sum_i g_i(x) \log g_i(x)$ (Router uncertainty)
- $d(x, \mathcal{P}) = \min_{p} \| h(x) - v_p \|_2$ (Novelty w.r.t. prototype memory)
- $\max_i \mathrm{conf}_i$: Penalizes overconfident familiar inputs

When batch average $\mathbb{E}[S(x)] > \tau$, an expert candidate is initialized.

---

## 4. Function-Preserving Expansion & Validation Gate

1. **Function-Preserving Clone**:
   The candidate expert $E_{\mathrm{cand}}$ inherits weights from the best parent expert $E_{\mathrm{parent}}$ with a zero-initialized residual adapter:
   $$\forall h, \quad E_{\mathrm{cand}}(h) \equiv E_{\mathrm{parent}}(h) \quad \text{at initialization}$$
2. **Anchor-Distilled Candidate Training**:
   $$\mathcal{L}_{\mathrm{cand}} = \mathrm{CE}(E_{\mathrm{cand}}(h(x)), y) + \lambda_{\mathrm{distill}} \sum_{p \in \mathcal{P}} \left\| E_{\mathrm{cand}}(v_p) - o_p[\mathrm{parent}] \right\|_2^2$$
3. **Validation Gate**:
   Admission into the active expert pool requires passing 4 quantitative checks:
   - $\mathrm{Acc}_{\mathrm{new}} \ge \tau_{\mathrm{acc}}$
   - $\Delta_{\mathrm{drift}} \le \tau_{\mathrm{drift}}$
   - Generalization gap bounded ($\mathrm{Loss}_{\mathrm{val}} - \mathrm{Loss}_{\mathrm{train}}$)
   - Expected Calibration Error $\mathrm{ECE} \le \tau_{\mathrm{ece}}$
4. **Capacity Control**:
   Maintains $N \le N_{\max}$. If exceeded, completely unused experts are pruned, or the most redundant expert pair (highest weight cosine similarity) is merged via parameter averaging.

---

## 5. Dual-Mode TTT (Test-Time Training / Adaptation)

- **Mode A: Continual Training (Labeled)**:
   Processes sequential tasks with $\mathcal{L} = \mathcal{L}_{\mathrm{CE}} + \lambda_r \mathcal{L}_{\mathrm{router}} + \lambda_e \mathcal{L}_{\mathrm{expert}}$, dynamic triggers, and validation gating.
- **Mode B: Test-Time Adaptation (Unlabeled TTT)**:
   During inference on distribution-shifted streams with frozen encoder:
   $$\mathcal{L}_{\mathrm{ttt}} = \mathcal{H}_{\min}(\hat{p}) + \lambda_{\mathrm{cons}} \left\| \hat{p}(x) - \hat{p}(x + \epsilon) \right\|_2^2$$


---

## 6. Project Structure

```text
pal-moe/
├── pal_moe/                  # Core library (aliased as dynamic_moe for compatibility)
│   ├── models/
│   │   ├── encoder.py        # SharedEncoder & EMAEncoder
│   │   ├── expert.py         # MLPExpert with Function-Preserving Net2Net clone
│   │   ├── router.py         # DynamicRouter (top-k, dynamic growth, pruning, merging)
│   │   └── moe.py            # PALMoE / DynamicMoE container
│   ├── memory/
│   │   └── prototype_memory.py # PrototypeMemory (vp, rp, op, xp, EMA, stability losses)
│   ├── trigger/
│   │   └── expert_trigger.py   # Quantitative trigger S(x)
│   ├── builder/
│   │   └── expert_builder.py   # Function-preserving builder, Validation Gate, Capacity control
│   ├── adaptation/
│   │   └── ttt.py              # ContinualTrainer (Mode A) & TestTimeAdapter (Mode B)
│   ├── baselines/
│   │   ├── naive.py          # Naive Sequential Fine-tuning
│   │   ├── ewc.py            # Elastic Weight Consolidation
│   │   ├── replay.py         # Experience Replay
│   │   └── standard_moe.py   # Standard MoE (Fixed 4 experts, no stability)
│   ├── data/
│   │   └── split_mnist.py    # 5-task Split-MNIST generator
│   └── evaluation/
│       └── metrics.py        # Accuracy, Forgetting, BWT, FWT, Router KL, Mutual Information
├── experiments/
│   ├── run_benchmark.py      # Standardized benchmark comparing 5 methods
│   ├── run_ablation.py       # Systematic ablation of 7 components
│   └── plot_results.py       # Publication-quality figure generation
├── tests/
│   └── test_pal_moe.py       # Comprehensive unit test suite
└── results/
    ├── benchmark_results.json
    └── ablation_results.json
```

---

## 7. Empirical Benchmark and Ablation Results

Experiments evaluated on 5-task Split-MNIST (2 classes per task).

### Continual Learning Benchmark (5 Tasks)

| Method | Avg Acc (↑) | Forgetting (↓) | BWT (↑) | Router KL (↓) | Spec. MI (↑) | Util. Entropy | Experts |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Naive Fine-tuning | 19.25% | 97.80% | -97.80% | - | - | - | 1 |
| EWC | 19.38% | 97.41% | -97.41% | - | - | - | 1 |
| Replay (Budgeted P=60) | 67.21% | 35.67% | -35.67% | - | - | - | 1 |
| Replay (Budgeted P=360) | 80.51% | 17.35% | -17.35% | - | - | - | 1 |
| Experience Replay (Buffer=250) | 79.98% | 18.76% | -18.76% | - | - | - | 1 |
| Standard MoE (Balanced) | 19.14% | 96.74% | -96.74% | - | 0.087 | 0.248 | 4 |
| **PAL-MoE (Ours)** | **59.43%** | **39.40%** | **-39.40%** | **0.3495** | **0.760** | **0.865** | **5** |

### Ablation Study

| Ablation Configuration | Avg Acc (↑) | Forgetting (↓) | BWT (↑) | Router KL (↓) | Experts | Rejections |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Full Proposed PAL-MoE** | **59.43%** | **39.40%** | **-39.40%** | **0.3495** | **5** | **0** |
| No Stability Loss ($\lambda_r=0, \lambda_e=0$) | 19.18% | 98.28% | -98.28% | 9.3926 | 4 | 1 |
| No Expert Anchor ($\lambda_r=0.5, \lambda_e=0$) | 19.21% | 97.96% | -97.96% | 0.0055 | 3 | 2 |
| Random Expert Init (No Net2Net) | 58.10% | 41.30% | -41.30% | 0.3487 | 5 | 0 |
| No Validation Gate | 59.43% | 39.40% | -39.40% | 0.3495 | 5 | 0 |
| Top-2 Routing | 46.73% | 63.13% | -63.13% | 0.2644 | 5 | 0 |
| Online Encoder (No EMA) | 23.13% | 95.45% | -95.45% | 1.3967 | 5 | 0 |
| EMA Encoder (Adaptive) | 24.40% | 93.88% | -93.88% | 1.3932 | 5 | 0 |

### Key Findings

1. **Router KL Stability**: Without stability loss, Router KL explodes to 9.3926 and catastrophic forgetting exceeds 98%. PAL-MoE constrains Router KL to 0.3495 via prototype anchoring.
2. **Top-1 Modular Isolation**: Top-1 routing strictly prevents inter-expert interference, outperforming Top-2 routing (59.43% vs 46.73%).
3. **Function-Preserving Net2Net Cloning**: Zero-loss initial inheritance gives +1.33% accuracy and -1.90% forgetting compared to random expert candidate initialization.
4. **Memory Efficiency**: With frozen representations, prototypes store compact feature anchors rather than full raw image buffers (only 19.5K floats / 76.2 KB for 60 prototypes).

---

## 8. How to Run

### Activate Environment
```bash
source .venv/bin/activate
export PYTHONPATH=.
```

### Run Unit Tests
```bash
pytest tests/test_pal_moe.py -v
```

### Run Continual Learning Benchmark
```bash
python experiments/run_benchmark.py --epochs 3
```

### Run Ablation Study
```bash
python experiments/run_ablation.py --epochs 3
```

### Generate Figures
```bash
python experiments/plot_results.py
```

---

## 9. Installation & Packaging

Install from local source:
```bash
pip install -e .
```

Or install with development dependencies:
```bash
pip install -e ".[dev]"
```

Build distribution packages:
```bash
python -m build
# or using uv:
uv build
```

---

## 10. Citation

If you use **PAL-MoE** in your research or benchmarks, please cite:

```bibtex
@software{pal_moe2026,
  author = {Valen, Kael},
  title = {PAL-MoE: Prototype-Anchored Lifelong Mixture of Experts},
  url = {https://github.com/kaelvalen/pal-moe},
  version = {0.1.0},
  year = {2026}
}
```

---

## 11. License

This project is licensed under the [MIT License](LICENSE).


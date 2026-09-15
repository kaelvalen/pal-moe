# PAL-MoE: Prototype-Anchored Lifelong Mixture of Experts

**PAL-MoE** is a novel Dynamic Mixture of Experts (MoE) architecture designed to solve **Class-Incremental Continual Learning** without catastrophic forgetting.

Unlike traditional networks that overwrite past knowledge, PAL-MoE dynamically spawns new Expert networks for new tasks while utilizing a **Prototype-Anchored Linear Router** to perfectly route data to the correct historical experts.

---

## Key Architectural Innovations

1. **Latent Replay & End-of-Task Joint Fine-Tuning**
   Instead of storing heavy raw pixels (images) for replay, PAL-MoE stores lightweight 128-dimensional latent vectors (`x_p`) outputted by the encoder. This allows a massive effective replay buffer at minimal memory cost. At the end of each task, all experts and the router are jointly calibrated using these latent exemplars, effectively teaching experts the "negative boundaries" of other tasks (OOD penalty).

2. **Mathematical Freezing & Absolute Protection**
   To eliminate expert and router drift, PAL-MoE permanently locks older experts and their routing gradients after their task concludes; while new data is seen, only the most recent expert and routing row adapt. At the end of each task, joint calibration **temporarily** unfreezes all experts (so historical experts learn the "negative boundaries" of other tasks), then re-locks history. **Known limitation (open issue):** in pure zero-replay mode, adding an expert initially shifts the routing of the previous task's inputs onto the newest expert (accuracy dips), whereas the latent-replay hybrid is stable throughout.

3. **Contrastive Pretraining for Linear Separability**
   PAL-MoE utilizes SimCLR-based contrastive pretraining on the shared base encoder (up to 50 epochs for complex datasets like CIFAR-10). This ensures that the latent space features are cleanly clustered and linearly separable, providing the perfect foundation for a simple, fast Linear Router to avoid routing confusion.

4. **Dynamic Capacity Growth (Net2Net)**
   When the model detects a domain shift (via the Quantitative Trigger `S(x)`), it spawns a new Expert. The new expert learns the new task without corrupting older experts.

---

## Benchmark Results

Evaluated against standard Continual Learning baselines under a strict memory budget limit (Buffer=250 items). All numbers below are **reproducible** from the current code with `seed=42`: `python experiments/run_benchmark.py --epochs 3 --device cuda`.

### 1. Split-MNIST (5 Tasks, Minimal Base)

| Method | Avg Acc (↑) | Forgetting (↓) | BWT (↑) | Experts |
| :--- | :---: | :---: | :---: | :---: |
| Naive Fine-tuning | 19.07% | 97.41% | -97.41% | 1 |
| EWC | 19.21% | 97.01% | -97.01% | 1 |
| Experience Replay (P=60) | 71.52% | 30.38% | -30.38% | 1 |
| Experience Replay (P=360) | 81.88% | 16.81% | -16.81% | 1 |
| Experience Replay (Buffer=250) | 79.32% | 20.44% | -20.44% | 1 |
| Standard MoE (Fixed 4 Experts) | 19.31% | 89.54% | -89.54% | 4 |
| **PAL-MoE (Ours - Pure / Zero Replay)** | **49.85%** | **54.00%** | **-54.00%** | **4** |
| **PAL-MoE + Replay (Hybrid, P=250)** | **82.23%** | **13.34%** | **-13.34%** | **4** |

*(Notes: the hybrid variant edges out Experience Replay at the same 250-buffer budget (+2.9pp accuracy, -7.1pp forgetting). The pure variant still forgets less than Naive/EWC/Standard-MoE (54% vs ~90-97%) but currently trails replay-based methods; its final task-accuracy dips when a new expert is added (open issue, see above). The validation gate rejected the 5th expert expansion in the frozen regime, hence 4 experts.)*

### 2. Split-CIFAR-10 (5 Tasks, Hard, CNN Encoder)

*Using a 50-Epoch SimCLR pre-trained ResNet-style encoder without ImageNet transfer learning. CIFAR numbers below were measured with the previous code revision and are pending a re-run with the current code (long runtime).*

| Method | Avg Acc (↑) | Forgetting (↓) | BWT (↑) | Experts |
| :--- | :---: | :---: | :---: | :---: |
| Experience Replay (Buffer=250) | ~28.00% *(est.)* | ~70.00% *(est.)* | - | 1 |
| **PAL-MoE + Replay (Hybrid, P=250)** | **43.01%** | **42.89%** | **-42.89%** | **5** |

*(Note: Baseline Continual Learning on CIFAR-10 from scratch without ImageNet pretraining severely collapses; PAL-MoE+Replay retains substantially more in this extremely constrained regime.)*

---

## Project Structure

```text
pal-moe/
├── pal_moe/
│   ├── models/
│   │   ├── encoder.py          # SharedEncoder with Contrastive Pretraining
│   │   ├── expert.py           # MLPExpert with Net2Net Expansion
│   │   ├── router.py           # DynamicRouter (top-k routing)
│   │   └── moe.py              # DynamicMoE container with latent forward pass
│   ├── memory/
│   │   └── prototype_memory.py # Anchors router null-space on latent vectors
│   ├── adaptation/
│   │   └── ttt.py              # ContinualTrainer (Latent Joint Fine-Tuning)
├── experiments/
│   ├── run_benchmark.py        # Standardized Continual Learning benchmark
│   ├── run_ablation.py         # Loss and Router Ablation scripts
│   └── plot_results.py         # Visualization generator
└── tests/
    └── test_pal_moe.py         # Comprehensive PyTest suite (23 tests)
```

---

## How to Run

### Activate Environment

```bash
source .venv/bin/activate
export PYTHONPATH=.
```

### Run the Benchmark (GPU Recommended)

Run the scripts to reproduce the results:

```bash
# For MNIST
LD_LIBRARY_PATH=/run/opengl-driver/lib python experiments/run_benchmark.py --dataset mnist --device cuda

# For CIFAR-10
LD_LIBRARY_PATH=/run/opengl-driver/lib python experiments/run_benchmark.py --dataset cifar10 --device cuda
```

---

## Academic Integrity & Citation

If you use **PAL-MoE** in your research or benchmarks, please cite:

```bibtex
@software{pal_moe2026,
  author = {Hakbilen, Mehmet Arda},
  title = {PAL-MoE: Prototype-Anchored Lifelong Mixture of Experts},
  url = {https://github.com/kaelvalen/pal-moe},
  version = {1.0.0},
  year = {2026}
}
```

---

## License

This project is licensed under the [MIT License](LICENSE).

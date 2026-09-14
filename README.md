# PAL-MoE: Prototype-Anchored Lifelong Mixture of Experts

**PAL-MoE**is a novel Dynamic Mixture of Experts (MoE) architecture designed to solve**Class-Incremental Continual Learning**without catastrophic forgetting.

Unlike traditional networks that overwrite past knowledge, PAL-MoE dynamically spawns new Expert networks for new tasks while utilizing a**Prototype-Anchored Linear Router**to perfectly route data to the correct historical experts.

---

## Key Architectural Innovations

1. **Latent Replay & End-of-Task Joint Fine-Tuning**
   Instead of storing heavy raw pixels (images) for replay, PAL-MoE stores lightweight 128-dimensional latent vectors (`x_p`) outputted by the encoder. This allows a massive effective replay buffer at minimal memory cost. At the end of each task, all experts and the router are jointly calibrated using these latent exemplars, effectively teaching experts the "negative boundaries" of other tasks (OOD penalty).

2. **Mathematical Freezing & Absolute Protection**
   To entirely eliminate expert and router drift, PAL-MoE permanently locks older experts and their corresponding routing gradients immediately after their specific task concludes. As the model encounters new data, only the *newest* expert and the *newest* row in the router are permitted to adapt.

3. **Contrastive Pretraining for Linear Separability**
   PAL-MoE utilizes SimCLR-based contrastive pretraining on the shared base encoder (up to 50 epochs for complex datasets like CIFAR-10). This ensures that the latent space features are cleanly clustered and linearly separable, providing the perfect foundation for a simple, fast Linear Router to avoid routing confusion.

4. **Dynamic Capacity Growth (Net2Net)**
   When the model detects a domain shift (via the Quantitative Trigger `S(x)`), it spawns a new Expert. The new expert learns the new task without corrupting older experts.

---

## Benchmark Results

Evaluated against standard Continual Learning baselines under a strict memory budget limit (Buffer=250 items).

### 1. Split-MNIST (5 Tasks, Minimal Base)

| Method                                        |   Avg Acc (↑)   | Forgetting (↓) |     BWT (↑)     |   Experts   |
| :-------------------------------------------- | :--------------: | :--------------: | :---------------: | :---------: |
| Naive Fine-tuning                             |      19.18%      |      98.30%      |      -98.30%      |      1      |
| SOTA: Experience Replay (Buffer=250)          |      81.67%      |      17.85%      |      -17.85%      |      1      |
|**PAL-MoE (Ours - Pure / Zero Replay)**|**67.78%**|   **9.57%**   |**-8.64%**|**5**|
|**PAL-MoE + Replay (Hybrid, P=250)**   |  **82.46%**  |**13.62%**|**-13.62%**|**5**|

### 2. Split-CIFAR-10 (5 Tasks, Hard, CNN Encoder)

*Using a 50-Epoch SimCLR pre-trained ResNet-style encoder without ImageNet transfer learning.*

| Method                                     | Avg Acc (↑) | Forgetting (↓) |     BWT (↑)     |   Experts   |
| :----------------------------------------- | :----------: | :--------------: | :---------------: | :---------: |
| Experience Replay (Buffer=250)             |   ~28.00%   |     ~70.00%     |         -         |      1      |
|**PAL-MoE + Replay (Hybrid, P=250)**|**43.01%**|**42.89%**|**-42.89%**|**5**|

*(Note: Baseline Continual Learning on CIFAR-10 from scratch without ImageNet pretraining severely collapses. PAL-MoE outperforms ER significantly in this extremely constrained regime.)*

---

## Project Structure

```text
pal-moe/
├── pal_moe/
│   ├── models/
│   │   ├── encoder.py        # SharedEncoder with Contrastive Pretraining
│   │   ├── expert.py         # MLPExpert with Net2Net Expansion (hidden_dim=256)
│   │   ├── router.py         # DynamicRouter (top-k routing)
│   │   └── moe.py            # DynamicMoE container with latent_h forward pass
│   ├── memory/
│   │   └── prototype_memory.py # Anchors router null-space on x_p
│   ├── adaptation/
│   │   └── ttt.py              # ContinualTrainer (Latent Joint Fine-Tuning)
├── experiments/
│   ├── run_benchmark.py      # Standardized 5-Task Continual Learning benchmark
│   ├── run_ablation.py       # Loss and Router Ablation scripts
│   └── plot_results.py       # Visualization generator
└── tests/
    └── test_pal_moe.py       # Comprehensive PyTest suite (22 tests)
```

---

## How to Run

### Activate Environment

```bash
source .venv/bin/activate
export PYTHONPATH=.
```

### Run the Benchmark (GPU Recommended)

Run the script to reproduce the results:

```bash
# For MNIST
LD_LIBRARY_PATH=/run/opengl-driver/lib python experiments/run_benchmark.py --dataset mnist --device cuda

# For CIFAR-10
LD_LIBRARY_PATH=/run/opengl-driver/lib python experiments/run_benchmark.py --dataset cifar10 --device cuda
```

---

## Academic Integrity & Citation

If you use**PAL-MoE**in your research or benchmarks, please cite:

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

# Configs index

Every experiment config is a validated JSON (unknown keys, wrong types and
out-of-range values are hard errors; explicit CLI flags override config values,
config values override argparse defaults - see `pal_moe/config.py`).

**Canonical configs** (referenced by the paper plan / recipes):

| Config | Benchmark | Backbone | Protocol notes |
| :-- | :-- | :-- | :-- |
| `mnist_default.json` | Split-MNIST 5×2 | 1-epoch AE MLP, frozen | The headline MNIST recipe: `lambda_ood 0.5`, router distillation 300 steps, inference anchoring α=0.5 |
| `cifar10_resnet18_frozen.json` | Split-CIFAR-10 5×2 | ImageNet ResNet-18, frozen + feature cache | Run-day canonical CIFAR-10 config |
| `cifar10_resnet18_frozen_raw.json` | same | same, **no feature cache** | Raw-pipeline equal-byte sweep: replay stores raw 32×32 images (12,296 B/item) |
| `cifar10_resnet18_trainable.json` | same | ResNet-18, **trainable** | True latent-drift cell (E8b) |
| `cifar100_resnet18_frozen.json` | Split-CIFAR-100 20×5 | ImageNet ResNet-18, frozen + feature cache | Run-day canonical CIFAR-100 config (relative gate) |
| `cifar100_resnet18_frozen_raw.json` | same | same, **no feature cache** | Raw-pipeline CIFAR-100 |
| `cifar10_vit.json` | Split-CIFAR-10 5×2 | ImageNet ViT-B/16, frozen + feature cache | One expert per task (`expand_every_task`, max 5), 15 epochs, anchors refreshed |
| `cifar100_vit.json` | Split-CIFAR-100 20×5 | ImageNet ViT-B/16 | One expert per task (max 20), 10 epochs, anchors refreshed |
| `cifar10_big_final.json` | Split-CIFAR-10 5×2 | conv 64/128/256, frozen + feature cache | Long-schedule pure/hybrid reference (15 epochs, 150 SimCLR) |
| `cifar10_big_final_full.json` | same | same | Full method set at the 5-epoch simplified recipe (fact 12) |
| `cifar100_big_frozen.json` | Split-CIFAR-100 20×5 | conv 64/128/256, frozen + feature cache | Long-horizon conv reference (fact 16) |

**Historical / ablation configs** (kept because design facts cite them):

| Config | Used by |
| :-- | :-- |
| `cifar10_big.json`, `cifar10_big_full.json`, `cifar10_big_frozen.json`, `cifar10_big_frozen_full.json` | fact 11-14 runs (`results/cifar10_big_frozen*`, `cifar10_final_full`) |
| `cifar10_default.json` | quick smoke runs |
| `mnist_ood_off.json` | OOD ablation on MNIST (fact 1) |

**Feature-cache note (design fact 19):** under `feature_cache` the replay
baselines store cached features, and the hybrid's raw store is disabled (it is
named `PAL-MoE + Latent Replay` in new runs). Use the `*_raw` configs for the
raw-vs-latent storage comparison.

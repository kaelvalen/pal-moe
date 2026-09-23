# Legacy / one-off scripts

Kept for history; superseded by `experiments/run_benchmark.py` and the
recipes. Nothing in `docs/`, the run queue or CI references them.

- `run_cifar10_unfrozen.py` - early Split-CIFAR-10 study of the unfrozen
  encoder with/without `L_encoder_stab`. The current runner covers the same
  regime with `--freeze_encoder` off (encoder stability is enabled
  automatically when the encoder is trainable), so use that instead.

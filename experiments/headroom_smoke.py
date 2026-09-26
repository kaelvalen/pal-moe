"""
HEADROOM screen - development smoke: COST ONLY, no dataset outcome is computed.

Measures, sustained (the laptop GPU throttles after a few minutes):

    train     adapter fine-tune through the frozen ViT-B/16 (bf16 autocast, synthetic
              224x224 input, AdamW on adapters + a linear head); >= 10 min,
              reported as the mean img/s of the LAST 5 minutes
    extract   frozen feature extraction (bf16, no grad); reported over its last 3 min
    decode    JPEG decode + resize + normalise through a 16-worker DataLoader on real
              ImageNet-R files (labels unused); last 2 min

GPU clock / temperature are sampled every 30 s. Output: results/headroom/smoke.json.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from pal_moe.core.vit_adapter import AdaptedViT  # noqa: E402


def gpu_state():
    try:
        out = (
            subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=clocks.sm,temperature.gpu,power.draw,memory.used",
                    "--format=csv,noheader,nounits",
                ]
            )
            .decode()
            .strip()
        )
        sm, temp, pw, mem = [x.strip() for x in out.split(",")]
        return {
            "sm_mhz": float(sm),
            "temp_c": float(temp),
            "power_w": float(pw),
            "mem_mib": float(mem),
        }
    except Exception:
        return {}


def sustained(step_fn, batch, minutes, tail_minutes, label):
    log, t0, n, win_n, win_t = [], time.time(), 0, 0, time.time()
    while time.time() - t0 < minutes * 60:
        step_fn()
        n += batch
        win_n += batch
        if time.time() - win_t >= 30:
            torch.cuda.synchronize()
            rate = win_n / (time.time() - win_t)
            log.append({"t": time.time() - t0, "img_s": rate, **gpu_state()})
            print(
                f"[{label}] {log[-1]['t']:6.0f}s {rate:7.1f} img/s {log[-1]}",
                flush=True,
            )
            win_n, win_t = 0, time.time()
    tail = [r["img_s"] for r in log if r["t"] >= (minutes - tail_minutes) * 60]
    first = [r["img_s"] for r in log if r["t"] <= 60]
    return {
        "minutes": minutes,
        "tail_minutes": tail_minutes,
        "sustained_img_s": sum(tail) / len(tail),
        "first_minute_img_s": sum(first) / max(1, len(first)),
        "log": log,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--train_minutes", type=float, default=10)
    p.add_argument("--extract_minutes", type=float, default=4)
    p.add_argument("--decode_minutes", type=float, default=3)
    p.add_argument("--images", default="data/imagenet_r_a/imagenet-r")
    p.add_argument("--out", default="results/headroom/smoke.json")
    args = p.parse_args()
    dev = torch.device("cuda")
    model = AdaptedViT().to(dev)
    model.add_expert("smoke")
    head = torch.nn.Linear(model.dim, 345).to(dev)
    params = list(model.adapters.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=1e-3)
    x = torch.randn(args.batch, 3, 224, 224, device=dev)
    y = torch.randint(0, 345, (args.batch,), device=dev)
    report = {
        "backbone": model.name,
        "base_hash": model.base_hash,
        "batch": args.batch,
        "trainable_params": sum(q.numel() for q in params),
        "gpu": gpu_state(),
    }

    def train_step():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = torch.nn.functional.cross_entropy(head(model(x, "smoke")), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    torch.cuda.reset_peak_memory_stats()
    report["train"] = sustained(train_step, args.batch, args.train_minutes, 5, "train")
    report["train"]["peak_mem_mib"] = torch.cuda.max_memory_allocated() / 2**20

    xb = torch.randn(256, 3, 224, 224, device=dev)

    @torch.no_grad()
    def extract_step():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model(xb, None)

    report["extract"] = sustained(extract_step, 256, args.extract_minutes, 3, "extract")

    if Path(args.images).exists():
        import torchvision.transforms as T
        from torchvision.datasets import ImageFolder

        tf = T.Compose(
            [
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize([0.5] * 3, [0.5] * 3),
            ]
        )
        dl = torch.utils.data.DataLoader(
            ImageFolder(args.images, tf),
            batch_size=256,
            shuffle=True,
            num_workers=16,
            persistent_workers=True,
        )
        it = iter(dl)

        def decode_step():
            nonlocal it
            try:
                next(it)
            except StopIteration:
                it = iter(dl)
                next(it)

        report["decode"] = sustained(decode_step, 256, args.decode_minutes, 2, "decode")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(
        json.dumps(
            {
                k: (
                    v["sustained_img_s"]
                    if isinstance(v, dict) and "sustained_img_s" in v
                    else v
                )
                for k, v in report.items()
                if k != "gpu"
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()

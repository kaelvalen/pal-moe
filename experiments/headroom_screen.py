"""
HEADROOM screen: frozen closed-form readouts vs an adapted-representation ceiling.

    docs/HEADROOM_PREREG.md (amendments 1 and 2)

Stages (each cached under results/headroom/, so a stage is never recomputed):

    features   frozen [CLS] of every train / test image (eval transform)
    frozen     F1 (ridge), F2 (RanPAC-style RP + ridge), and for DomainNet Domain-IL
               F1o / F2o (per-domain ridge, oracle domain) and D (domain-ID); 6 seeds
    ceiling    C: one adapter set trained jointly on all train data (linear head), then
               ridge on the adapted [CLS] (C_ridge) and the head itself (C_head);
               Co (DomainNet): one adapter set per domain, per-expert ridge over all
               classes, oracle domain; 2 seeds (+1 on a threshold split)

Ridge: float64, scale-free (pal_moe.eval.ridge_select, amendment 3), chosen on a seeded 10 % held-out
slice of the TRAIN split, refitted on the full train split. Train-split accuracies are
reported for every ridge (amendment 1). The test split is never used for a choice.

Usage:
    LD_LIBRARY_PATH=/run/opengl-driver/lib .venv/bin/python experiments/headroom_screen.py \
        --datasets imagenet-a,imagenet-r,domainnet --stages features,frozen,ceiling
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402

from pal_moe.core.vit_adapter import AdaptedViT  # noqa: E402
from pal_moe.eval.ridge_select import fit_select  # noqa: E402

ROOT = Path("data")
OUT = Path("results/headroom")
DOMAINS = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]
FROZEN_SEEDS = [42, 1, 2, 3, 4, 5]
RP_DIM = 10000
WORKERS = 14


def git_rev():
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


# -- data ----------------------------------------------------------------------------------


def read_list(dataset: str, split: str):
    """-> (paths, labels, domains)  (domain = -1 outside DomainNet)."""
    rows = []
    if dataset == "domainnet":
        for di, d in enumerate(DOMAINS):
            for line in (
                (ROOT / "domainnet" / f"{d}_{split}.txt").read_text().split("\n")
            ):
                if line.strip():
                    p, y = line.rsplit(" ", 1)
                    rows.append((str(ROOT / "domainnet" / p), int(y), di))
    else:
        for line in (
            (ROOT / "splits" / f"{dataset}_{split}.txt").read_text().split("\n")
        ):
            if line.strip():
                p, y = line.rsplit(" ", 1)
                rows.append((str(ROOT / "imagenet_r_a" / p), int(y), -1))
    paths, labels, domains = zip(*rows)
    return list(paths), torch.tensor(labels), torch.tensor(domains)


class Images(torch.utils.data.Dataset):
    def __init__(self, paths, labels, tf):
        self.paths, self.labels, self.tf = paths, labels, tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            return self.tf(im.convert("RGB")), self.labels[i], i


def transforms(model, train: bool):
    import torchvision.transforms as T

    cfg = model.data_config()
    norm = T.Normalize(cfg["mean"], cfg["std"])
    if train:
        return T.Compose(
            [
                T.RandomResizedCrop(224, scale=(0.5, 1.0)),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                norm,
            ]
        )
    return T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), norm])


def loader(ds, batch, shuffle, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch,
        shuffle=shuffle,
        generator=g,
        num_workers=WORKERS,
        pin_memory=True,
        persistent_workers=False,
        drop_last=shuffle,
    )


@torch.no_grad()
def extract(model, paths, labels, expert, device, head=None, expert_of=None):
    """[CLS] for every path (eval transform). `expert_of[i]` overrides `expert` per sample
    (per-domain experts). Returns float32 CPU features (+ head logits argmax)."""
    ds = Images(paths, labels, transforms(model, train=False))
    feats = torch.empty(len(paths), model.dim)
    pred = torch.empty(len(paths), dtype=torch.long) if head is not None else None
    groups = (
        [(expert, torch.arange(len(paths)))]
        if expert_of is None
        else [
            (e, torch.nonzero(torch.tensor([x == e for x in expert_of])).flatten())
            for e in sorted(set(expert_of))
        ]
    )
    for e, idx in groups:
        sub = torch.utils.data.Subset(ds, idx.tolist())
        for x, _, i in loader(sub, 256, False):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z = model(x.to(device, non_blocking=True), e).float()
            feats[i] = z.cpu()
            if head is not None:
                pred[i] = head(z).argmax(-1).cpu()
    return feats, pred


# -- ridge (amendment 3: scale-free lambda, tie tolerance, edge extension) ----------------


def ridge_select(Ztr, Ytr, C, seed, rp=None):
    return fit_select(Ztr, Ytr, C, seed, rp=rp)


def rp_matrix(seed, dim=768):
    return torch.randn(
        dim, RP_DIM, generator=torch.Generator().manual_seed(seed), dtype=torch.float64
    )


# -- stages --------------------------------------------------------------------------------


def feature_path(dataset, split, tag="frozen"):
    return OUT / "features" / f"{dataset}_{split}_{tag}.pt"


def stage_features(dataset, model, device):
    for split in ("train", "test"):
        path = feature_path(dataset, split)
        if path.exists():
            continue
        paths, y, dom = read_list(dataset, split)
        t0 = time.time()
        z, _ = extract(model, paths, y, None, device)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "z": z,
                "y": y,
                "domain": dom,
                "base_hash": model.base_hash,
                "seconds": time.time() - t0,
            },
            path,
        )
        print(
            f"[features] {dataset} {split} {len(paths)} in {time.time() - t0:.0f}s",
            flush=True,
        )


def load_feats(dataset, split, tag="frozen"):
    return torch.load(feature_path(dataset, split, tag), weights_only=False)


def _per_domain(model, z, y, dom):
    return {d: model.accuracy(z[dom == i], y[dom == i]) for i, d in enumerate(DOMAINS)}


def _oracle_readout(ztr, ytr, dtr, zte, yte, dte, C, seed, rp=None, domains=None):
    """One ridge per domain, the true domain given at test. Sample-weighted totals."""
    per, hits, n, tr_hits, tr_n, conv = {}, 0.0, 0, 0.0, 0, True
    for i, d in enumerate(DOMAINS):
        mtr, mte = dtr == i, dte == i
        m, info = ridge_select(ztr[mtr], ytr[mtr], C, seed, rp)
        a = m.accuracy(zte[mte], yte[mte])
        per[d] = {"test_acc": a, **info}
        hits += a * int(mte.sum())
        n += int(mte.sum())
        tr_hits += info["train_acc"] * int(mtr.sum())
        tr_n += int(mtr.sum())
        conv = conv and info["converged"]
    return {
        "test_acc": hits / n,
        "train_acc": tr_hits / tr_n,
        "converged": conv,
        "per_domain": per,
    }


def frozen_arms(dataset, seed):
    tr, te = load_feats(dataset, "train"), load_feats(dataset, "test")
    C = int(tr["y"].max()) + 1
    out = {}
    for name, rp in (("F1", None), ("F2", rp_matrix(seed))):
        m, info = ridge_select(tr["z"], tr["y"], C, seed, rp)
        out[name] = {"test_acc": m.accuracy(te["z"], te["y"]), **info}
        if dataset == "domainnet":
            out[name]["per_domain_test"] = _per_domain(
                m, te["z"], te["y"], te["domain"]
            )
    if dataset == "domainnet":
        for name, rp in (("F1o", None), ("F2o", rp_matrix(seed))):
            out[name] = _oracle_readout(
                tr["z"],
                tr["y"],
                tr["domain"],
                te["z"],
                te["y"],
                te["domain"],
                C,
                seed,
                rp,
            )
        means = torch.stack(
            [F.normalize(tr["z"][tr["domain"] == i].mean(0), dim=0) for i in range(6)]
        )
        nn_pred = (F.normalize(te["z"], dim=-1) @ means.t()).argmax(-1)
        md, info = ridge_select(tr["z"], tr["domain"], 6, seed)
        out["D"] = {
            "nearest_mean_acc": float((nn_pred == te["domain"]).float().mean()),
            "ridge_acc": md.accuracy(te["z"], te["domain"]),
            "ridge": info,
            "confusion_nearest_mean": torch.bincount(
                te["domain"] * 6 + nn_pred, minlength=36
            )
            .reshape(6, 6)
            .tolist(),
        }
    return out


def train_adapter(model, expert, paths, labels, C, steps, seed, device):
    torch.manual_seed(seed)
    model.add_expert(expert)
    head = torch.nn.Linear(model.dim, C).to(device)
    params = list(model.adapters[expert].parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    ds = Images(paths, labels, transforms(model, train=True))
    probe = torch.stack([ds[i][0] for i in range(4)]).to(device)
    with torch.no_grad():  # veto: identity at init, bitwise
        identity = torch.equal(model(probe, expert), model(probe, None))
    done, t0, losses = 0, time.time(), []
    if len(ds) < 64:
        raise ValueError(f"{expert}: {len(ds)} training images < one batch of 64")
    while done < steps:
        for x, y, _ in loader(ds, 64, True, seed + done):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = F.cross_entropy(
                    head(model(x.to(device, non_blocking=True), expert)), y.to(device)
                )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            done += 1
            if done % 500 == 0:
                losses.append(float(loss))
                print(
                    f"[train {expert}] {done}/{steps} loss {float(loss):.3f} "
                    f"{done * 64 / (time.time() - t0):.0f} img/s",
                    flush=True,
                )
            if done >= steps:
                break
    model.freeze_expert(expert)
    head.eval()
    return head, {
        "steps": steps,
        "seconds": time.time() - t0,
        "loss_trace": losses,
        "identity_at_init": identity,
    }


def save_expert(model, head, expert, dataset, seed, ztr, zte, extra=None):
    """Adapted features, adapter weights and head, so any readout can be re-audited
    without retraining (amendment 3)."""
    path = OUT / "ceiling" / f"{dataset}_seed{seed}_{expert}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "adapters": model.adapters[expert].state_dict(),
            "head": head.state_dict(),
            "z_train": ztr.half(),
            "z_test": zte.half(),
            "base_hash": model.base_hash,
            **(extra or {}),
        },
        path,
    )
    return str(path)


def ceiling_arms(dataset, seed, model, device, epochs):
    trp, ytr, dtr = read_list(dataset, "train")
    tep, yte, dte = read_list(dataset, "test")
    C = int(ytr.max()) + 1
    steps = epochs * (len(trp) // 64)
    out = {}
    e = f"C{seed}"
    head, info = train_adapter(model, e, trp, ytr, C, steps, seed, device)
    ztr, ptr = extract(model, trp, ytr, e, device, head)
    zte, pte = extract(model, tep, yte, e, device, head)
    saved = save_expert(model, head, e, dataset, seed, ztr, zte)
    m, rinfo = ridge_select(ztr, ytr, C, seed)
    out["C"] = {
        "C_ridge_test": m.accuracy(zte, yte),
        "C_ridge_train": rinfo["train_acc"],
        "C_head_test": float((pte == yte).float().mean()),
        "C_head_train": float((ptr == ytr).float().mean()),
        "ridge": rinfo,
        "saved": saved,
        **info,
    }
    if dataset == "domainnet":
        out["C"]["per_domain_ridge_test"] = _per_domain(m, zte, yte, dte)
        z1_all, z2_all = torch.empty_like(ztr), torch.empty_like(zte)
        head_hits, trains = {}, {}
        for i, d in enumerate(DOMAINS):
            mtr = (dtr == i).nonzero().flatten()
            mte = (dte == i).nonzero().flatten()
            e = f"Co{seed}_{d}"
            dsteps = max(1, round(steps * len(mtr) / len(trp)))
            h, dinfo = train_adapter(
                model,
                e,
                [trp[j] for j in mtr.tolist()],
                ytr[mtr],
                C,
                dsteps,
                seed,
                device,
            )
            z1, _ = extract(
                model, [trp[j] for j in mtr.tolist()], ytr[mtr], e, device, h
            )
            z2, p2 = extract(
                model, [tep[j] for j in mte.tolist()], yte[mte], e, device, h
            )
            z1_all[mtr], z2_all[mte] = z1, z2
            save_expert(model, h, e, dataset, seed, z1, z2)
            head_hits[d] = float((p2 == yte[mte]).float().mean())
            trains[d] = dinfo
        o = _oracle_readout(z1_all, ytr, dtr, z2_all, yte, dte, C, seed)
        out["Co"] = {
            "Co_ridge_test": o["test_acc"],
            "Co_ridge_train": o["train_acc"],
            "converged": o["converged"],
            "per_domain": o["per_domain"],
            "head_test_per_domain": head_hits,
            "training": trains,
        }
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", default="imagenet-a,imagenet-r,domainnet")
    p.add_argument("--stages", default="features,frozen,ceiling")
    p.add_argument("--ceiling_seeds", default="42,1")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    device = torch.device(args.device)
    epochs = {"domainnet": 3, "imagenet-r": 10, "imagenet-a": 10}
    model = AdaptedViT().to(device)
    OUT.mkdir(parents=True, exist_ok=True)
    for ds in args.datasets.split(","):
        path = OUT / f"{ds}_study.json"
        study = (
            json.loads(path.read_text())
            if path.exists()
            else {
                "dataset": ds,
                "prereg": "docs/HEADROOM_PREREG.md",
                "base_hash": model.base_hash,
                "backbone": model.name,
                "frozen": {},
                "ceiling": {},
            }
        )
        study["git_revision"] = git_rev()

        def save(path=path, study=study):
            path.write_text(json.dumps(study, indent=1))

        if "features" in args.stages:
            stage_features(ds, model, device)
        if "frozen" in args.stages:
            for seed in FROZEN_SEEDS:
                if str(seed) in study["frozen"]:
                    continue
                study["frozen"][str(seed)] = frozen_arms(ds, seed)
                print(f"[frozen] {ds} seed {seed} done", flush=True)
                save()
            if ds == "domainnet" and "determinism_F1" not in study:
                again = frozen_arms(ds, 42)["F1"]["test_acc"]
                study["determinism_F1"] = (
                    again == study["frozen"]["42"]["F1"]["test_acc"]
                )
                save()
        if "ceiling" in args.stages:
            for seed in [int(s) for s in args.ceiling_seeds.split(",")]:
                if str(seed) in study["ceiling"]:
                    continue
                study["ceiling"][str(seed)] = ceiling_arms(
                    ds, seed, model, device, epochs[ds]
                )
                print(f"[ceiling] {ds} seed {seed} done", flush=True)
                save()
    print("done")


if __name__ == "__main__":
    main()

import sys

with open('experiments/run_ablation.py', 'r') as f:
    content = f.read()

replacement = """
            if dataset == "cifar10":
                base_encoder.pretrain_contrastive(unlabeled_loader, device=device, epochs=50)
                # DO NOT freeze for CIFAR-10
            else:
                base_encoder.pretrain_unsupervised(unlabeled_loader, device=device, epochs=1)
                base_encoder.freeze()
"""

content = content.replace("""
            base_encoder.pretrain_contrastive(unlabeled_loader, device=device, epochs=50)
            base_encoder.freeze()
""", replacement)

with open('experiments/run_ablation.py', 'w') as f:
    f.write(content)

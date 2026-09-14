import sys

def fix_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    
    # We want to change the second occurrence back to pretrain_unsupervised(..., epochs=1)
    parts = content.split('base_encoder.pretrain_contrastive(unlabeled_loader, device=device, epochs=50)')
    if len(parts) == 3:
        new_content = parts[0] + 'base_encoder.pretrain_contrastive(unlabeled_loader, device=device, epochs=50)' + parts[1] + 'base_encoder.pretrain_unsupervised(unlabeled_loader, device=device, epochs=1)' + parts[2]
        with open(filepath, 'w') as f:
            f.write(new_content)

fix_file('experiments/run_benchmark.py')
fix_file('experiments/run_ablation.py')

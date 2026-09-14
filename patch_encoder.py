import sys
with open('pal_moe/models/encoder.py', 'r') as f:
    content = f.read()

replacement = """                if x.dim() == 4: # Image data (CIFAR)
                    import torchvision.transforms as T
                    # Un-normalize back to [0,1]
                    mean = torch.tensor([0.4914, 0.4822, 0.4465], device=device).view(1, 3, 1, 1)
                    std = torch.tensor([0.2470, 0.2435, 0.2616], device=device).view(1, 3, 1, 1)
                    x_unnorm = x * std + mean
                    x_unnorm = torch.clamp(x_unnorm, 0.0, 1.0)
                    
                    aug = T.Compose([
                        T.RandomResizedCrop(x.shape[-2:], scale=(0.2, 1.0), antialias=True),
                        T.RandomHorizontalFlip(p=0.5),
                        T.ColorJitter(0.4, 0.4, 0.4, 0.1),
                    ])
                    # Apply augmentations on GPU
                    x1 = aug(x_unnorm)
                    x2 = aug(x_unnorm)
                    
                    # Re-normalize
                    x1 = (x1 - mean) / std
                    x2 = (x2 - mean) / std
                else:"""

content = content.replace("""                if x.dim() == 4: # Image data (CIFAR)
                    import torchvision.transforms as T
                    aug = T.Compose([
                        T.RandomResizedCrop(x.shape[-2:], scale=(0.2, 1.0), antialias=True),
                        T.RandomHorizontalFlip(p=0.5),
                        T.ColorJitter(0.4, 0.4, 0.4, 0.1),
                    ])
                    # Apply augmentations on GPU
                    x1 = aug(x)
                    x2 = aug(x)
                else:""", replacement)

with open('pal_moe/models/encoder.py', 'w') as f:
    f.write(content)

import sys
with open('experiments/run_benchmark.py', 'r') as f:
    content = f.read()

# 1. Add dataset param
content = content.replace(
    'def run_benchmark(epochs_per_task: int = 3, device_str: str = "auto", output_dir: str = "./results"):',
    'def run_benchmark(epochs_per_task: int = 3, device_str: str = "auto", output_dir: str = "./results", dataset: str = "mnist"):'
)

# 2. Add cifar10 task loading
import_str = '''
    if dataset == "cifar10":
        from pal_moe.data.split_cifar import get_split_cifar10_tasks
        tasks = get_split_cifar10_tasks(data_dir="./data", batch_size=128, val_split=0.1, seed=42)
        num_tasks = len(tasks)
        input_dim = 3072
        mnist_train = datasets.CIFAR10("./data", train=True, download=True, transform=transforms.ToTensor())
        unlabeled_loader = torch.utils.data.DataLoader(mnist_train, batch_size=256, shuffle=True)
        base_encoder = SharedEncoder(input_dim=input_dim, hidden_dims=None, output_dim=128, arch="conv").to(device)
        base_encoder.pretrain_unsupervised(unlabeled_loader, device=device, epochs=1)
        # unfreezing happens implicitly
    else:
        tasks = get_split_mnist_tasks(data_dir="./data", batch_size=128, val_split=0.1, seed=42)
        num_tasks = len(tasks)
        input_dim = 784
        mnist_train = datasets.MNIST("./data", train=True, download=True, transform=transforms.ToTensor())
        unlabeled_loader = torch.utils.data.DataLoader(mnist_train, batch_size=256, shuffle=True)
        base_encoder = SharedEncoder(input_dim=input_dim, hidden_dims=(256, 128), output_dim=128, arch="mlp").to(device)
        base_encoder.pretrain_unsupervised(unlabeled_loader, device=device, epochs=1)
        base_encoder.freeze()
'''

content = content.replace(
    '''    # Load 5 Split-MNIST tasks
    tasks = get_split_mnist_tasks(data_dir="./data", batch_size=128, val_split=0.1, seed=42)
    num_tasks = len(tasks)

    # -------------------------------------------------------------
    # Step 0: Shared Domain Encoder Pretraining (Unsupervised, 1 epoch)
    # -------------------------------------------------------------
    print("\\n" + "=" * 60)
    print("Step 0: Pretraining Task-Agnostic Shared Encoder (Unsupervised)")
    print("=" * 60)
    set_seed(42)
    mnist_train = datasets.MNIST("./data", train=True, download=True, transform=transforms.ToTensor())
    unlabeled_loader = torch.utils.data.DataLoader(mnist_train, batch_size=256, shuffle=True)
    base_encoder = SharedEncoder(input_dim=784, hidden_dims=(256, 128), output_dim=128, arch="mlp").to(device)
    base_encoder.pretrain_unsupervised(unlabeled_loader, device=device, epochs=1)
    base_encoder.freeze()''',
    import_str
)

# 3. Replace dynamic expert builder thresholds and kwargs
content = content.replace('min_acc_threshold=0.60,', 'min_acc_threshold=0.45 if dataset == "cifar10" else 0.60,')
content = content.replace('lambda_e=2.5,\n        lr=1e-3,', 'lambda_e=2.5,\n        lambda_enc=0.5 if dataset == "cifar10" else 0.0,\n        encoder_lr=1e-4 if dataset == "cifar10" else None,\n        lr=1e-3,')

content = content.replace('DynamicMoE(input_dim=128', 'DynamicMoE(input_dim=input_dim')
content = content.replace('MLPExpert(input_dim=128', 'MLPExpert(input_dim=128') 

with open('experiments/run_benchmark.py', 'w') as f:
    f.write(content)

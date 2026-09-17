from .split_cifar import SplitCIFAR10Task, get_split_cifar10_tasks
from .split_cifar100 import SplitCIFAR100Task, get_split_cifar100_tasks
from .split_mnist import SplitMNISTTask, get_split_mnist_tasks

__all__ = [
    "get_split_mnist_tasks",
    "SplitMNISTTask",
    "get_split_cifar10_tasks",
    "SplitCIFAR10Task",
    "get_split_cifar100_tasks",
    "SplitCIFAR100Task",
]

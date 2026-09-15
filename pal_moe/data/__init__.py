from .split_mnist import get_split_mnist_tasks, SplitMNISTTask
from .split_cifar import get_split_cifar10_tasks, SplitCIFAR10Task
from .split_cifar100 import get_split_cifar100_tasks, SplitCIFAR100Task

__all__ = [
    "get_split_mnist_tasks",
    "SplitMNISTTask",
    "get_split_cifar10_tasks",
    "SplitCIFAR10Task",
    "get_split_cifar100_tasks",
    "SplitCIFAR100Task",
]

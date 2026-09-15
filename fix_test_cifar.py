import re

with open("tests/test_pal_moe.py", "r") as f:
    content = f.read()

old_test = """def test_split_cifar10_tasks():
    from pal_moe.data.split_cifar import get_split_cifar10_tasks
    tasks = get_split_cifar10_tasks(data_dir="./data", batch_size=32, val_split=0.1, max_train_samples_per_task=100)
    assert len(tasks) == 5
    assert tasks[0].classes == (0, 1)
    assert tasks[4].classes == (8, 9)
    sample_b = next(iter(tasks[0].train_loader))
    assert sample_b[0].shape[1:] == (3, 32, 32)"""

new_test = """def test_split_cifar10_tasks():
    from pal_moe.data.split_cifar import get_split_cifar10_tasks
    import torch
    from unittest.mock import patch, MagicMock

    with patch('torchvision.datasets.CIFAR10') as MockCIFAR:
        mock_dataset = MagicMock()
        # Mock targets to have 10 of each class
        mock_dataset.targets = [i for i in range(10)] * 10
        mock_dataset.__len__.return_value = 100
        mock_dataset.__getitem__.side_effect = lambda idx: (torch.zeros(3, 32, 32), mock_dataset.targets[idx])
        MockCIFAR.return_value = mock_dataset

        tasks = get_split_cifar10_tasks(data_dir="./data", batch_size=2, val_split=0.1, max_train_samples_per_task=10)
        assert len(tasks) == 5
        assert tasks[0].classes == (0, 1)
        assert tasks[4].classes == (8, 9)
        sample_b = next(iter(tasks[0].train_loader))
        assert sample_b[0].shape[1:] == (3, 32, 32)"""

content = content.replace(old_test, new_test)

with open("tests/test_pal_moe.py", "w") as f:
    f.write(content)

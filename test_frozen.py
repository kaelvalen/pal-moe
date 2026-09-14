import sys

with open('experiments/run_benchmark.py', 'r') as f:
    content = f.read()

# Make sure it freezes for cifar10 too
content = content.replace(
    '        # DO NOT freeze the encoder for CIFAR-10 so it can learn during tasks',
    '        base_encoder.freeze()'
)

with open('experiments/run_benchmark.py', 'w') as f:
    f.write(content)

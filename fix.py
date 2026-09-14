import sys
with open('experiments/run_benchmark.py', 'r') as f:
    lines = f.readlines()
with open('experiments/run_benchmark.py', 'w') as f:
    for line in lines:
        if 'parser.add_argument("--dataset"' in line:
            continue
        f.write(line)

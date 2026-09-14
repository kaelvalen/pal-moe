import sys
with open('experiments/run_benchmark.py', 'r') as f:
    lines = f.readlines()
with open('experiments/run_benchmark.py', 'w') as f:
    skip = False
    for line in lines:
        if 'Running Baseline 1' in line:
            skip = True
        if 'Running Proposed: PAL-MoE' in line:
            skip = False
        if not skip:
            f.write(line)

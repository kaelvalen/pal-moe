import sys

with open('pal_moe/adaptation/ttt.py', 'r') as f:
    content = f.read()

replacement_step8 = """
        # Step 8: End-of-task Joint Latent Fine-tuning
        # DO NOT unfreeze old experts! Only the newest expert and router learn negative boundaries.
"""

# Find the unfreeze block and replace it
import re
pattern = r"# Step 8: End-of-task Joint Latent Fine-tuning\n        # Unfreeze all experts for joint calibration\n        for exp in self\.model\.experts:\n            for p in exp\.parameters\(\):\n                p\.requires_grad = True\n        self\.optimizer = self\._build_optimizer\(\)"

content = re.sub(pattern, replacement_step8, content)

with open('pal_moe/adaptation/ttt.py', 'w') as f:
    f.write(content)

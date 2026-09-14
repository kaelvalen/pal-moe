import sys
import re

with open('pal_moe/adaptation/ttt.py', 'r') as f:
    content = f.read()

replacement = """
                loss = loss + l_replay

                loss.backward()
                
                # Zero out gradients for OLD router rows to completely prevent router forgetting
                if self.model.num_experts > 1:
                    if self.model.router.gate.weight.grad is not None:
                        self.model.router.gate.weight.grad[:self.model.num_experts-1] = 0.0
                    if self.model.router.gate.bias.grad is not None:
                        self.model.router.gate.bias.grad[:self.model.num_experts-1] = 0.0

                self.optimizer.step()
"""

# Replace loss.backward() and self.optimizer.step() inside Step 5
pattern = r"\n\s*loss\.backward\(\)\n\s*self\.optimizer\.step\(\)"
content = re.sub(pattern, replacement, content, count=1)

with open('pal_moe/adaptation/ttt.py', 'w') as f:
    f.write(content)

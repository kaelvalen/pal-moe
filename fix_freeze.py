import sys

with open('pal_moe/adaptation/ttt.py', 'r') as f:
    content = f.read()

replacement_step5 = """
        # Freeze old experts during Step 5 to prevent catastrophic forgetting
        for i, exp in enumerate(self.model.experts):
            if i < len(self.model.experts) - 1:
                for p in exp.parameters():
                    p.requires_grad = False
            else:
                for p in exp.parameters():
                    p.requires_grad = True
        # Rebuild optimizer to reflect frozen states
        self.optimizer = self._build_optimizer()

        # Step 5: Continual Training loop with joint stability loss
"""

content = content.replace("        # Step 5: Continual Training loop with joint stability loss", replacement_step5)

replacement_step8 = """
        # Step 8: End-of-task Joint Latent Fine-tuning
        # Unfreeze all experts for joint calibration
        for exp in self.model.experts:
            for p in exp.parameters():
                p.requires_grad = True
        self.optimizer = self._build_optimizer()
"""

content = content.replace("        # Step 8: End-of-task Joint Latent Fine-tuning", replacement_step8)

with open('pal_moe/adaptation/ttt.py', 'w') as f:
    f.write(content)

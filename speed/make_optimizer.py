"""A stand-in for the leaderboard checkpoint's optimizer.pt: GPT-2's Adam second moments in one parameter group, each entry naming its parameter as bergson's exports do."""
import os

import torch
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("gpt2")
named = list(model.named_parameters())
state = {
    i: {
        "step": torch.tensor(72.0),
        "exp_avg": torch.zeros_like(p),
        "exp_avg_sq": torch.full_like(p, 1e-6),
        "param_name": name,
    }
    for i, (name, p) in enumerate(named)
}
groups = [{"params": list(range(len(named))), "lr": 4e-4, "betas": (0.9, 0.999), "eps": 1e-8}]
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "optimizer.pt")
torch.save({"state": state, "param_groups": groups}, out)
print(out)

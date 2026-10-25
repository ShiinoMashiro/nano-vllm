import torch
from torch import nn


class Sampler(nn.Module):

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        greedy = temperatures <= 0
        temps = torch.where(greedy, torch.ones_like(temperatures), temperatures)
        logits = logits.float().div_(temps.unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1)
        sampled = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        greedy_tokens = logits.argmax(dim=-1)
        return torch.where(greedy, greedy_tokens, sampled)

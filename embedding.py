import torch
from torch import nn

class ConditionalEmbedding(nn.Module):
    def __init__(self, dim:int):
        super().__init__()
        self.condEmbedding = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.SiLU(),
        )

    def forward(self, t:torch.Tensor) -> torch.Tensor:
        emb = self.condEmbedding(t)
        return emb

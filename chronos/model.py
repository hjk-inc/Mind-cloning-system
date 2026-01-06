import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

class TitanBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        reduced = dim // 4
        self.block = nn.Sequential(
            nn.Linear(dim, reduced), nn.LayerNorm(reduced), nn.SiLU(),
            nn.Linear(reduced, reduced), nn.LayerNorm(reduced), nn.SiLU(),
            nn.Linear(reduced, dim), nn.LayerNorm(dim)
        )
        nn.init.zeros_(self.block[-1].weight)
        nn.init.zeros_(self.block[-1].bias)

    def forward(self, x):
        return torch.nn.functional.silu(x + self.block(x))


class SovereignTitanNet(nn.Module):
    """
    Titan Backbone:
    Input: 3-frame window [T-2, T-1, T] = 297
    Output: Predict T+1 = 99
    Active depth grows with buffer size
    """
    def __init__(self, input_dim=297, hidden_dim=512, total_blocks=450, active_blocks=50):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.backbone = nn.ModuleList([TitanBlock(hidden_dim) for _ in range(total_blocks)])
        self.output_layer = nn.Linear(hidden_dim, 99)
        self.active_blocks = active_blocks

    def forward(self, x):
        x = self.input_layer(x)
        for i in range(self.active_blocks):
            if self.training and i % 4 == 0:
                x = checkpoint(self.backbone[i], x, use_reentrant=False)
            else:
                x = self.backbone[i](x)
        return self.output_layer(x)

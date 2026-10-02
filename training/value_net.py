"""
Queue-aware value network: scores a post-placement board given the pieces to come.

Board path: 3×3 convolutions with padding keep the 20×10 grid, so the net knows
where a shape is, not just that it exists (no global pooling). A 1×1 conv shrinks
the channels before flattening to keep the dense layer small.

Context path: the piece/combo vector from value_data.encode_context.

The two are joined and passed through an MLP, so the net can learn interactions
like "this slot is worth a lot because a T is in the queue".
"""

import torch
import torch.nn as nn

from value_data import CONTEXT_SIZE


class ValueNet(nn.Module):
    def __init__(self, channels: int = 32, squeeze: int = 8, hidden: int = 128,
                 context_size: int = CONTEXT_SIZE):
        super().__init__()
        self.config = dict(channels=channels, squeeze=squeeze, hidden=hidden,
                           context_size=context_size)
        self.board = nn.Sequential(
            nn.Conv2d(1, channels, 3, padding=1), nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(),
            nn.Conv2d(channels, squeeze, 1), nn.ReLU(),
            nn.Flatten(),                                   # squeeze × 20 × 10
        )
        self.context = nn.Sequential(nn.Linear(context_size, 64), nn.ReLU())
        self.head = nn.Sequential(
            nn.Linear(squeeze * 200 + 64, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, board: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        """board (B, 20, 10) float, ctx (B, C) float → (B,) value."""
        b = self.board(board.unsqueeze(1))
        c = self.context(ctx)
        return self.head(torch.cat([b, c], dim=1)).squeeze(1)

    def macs_per_board(self) -> int:
        """Multiply-adds for one forward pass, to estimate in-browser cost."""
        ch, sq, h, cs = (self.config[k] for k in ('channels', 'squeeze', 'hidden', 'context_size'))
        cells = 200
        conv = cells * 9 * (1 * ch + ch * ch + ch * ch) + cells * ch * sq
        dense = cs * 64 + (sq * cells + 64) * h + h * h + h
        return conv + dense

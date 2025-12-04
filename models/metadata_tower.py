import torch
import torch.nn as nn
class MetaTower(nn.Module):
    def __init__(self, in_dim, hidden_dim=32, out_dim=16, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(out_dim, 1)  # metadata-only training

    def forward(self, x, return_logits=False):
        h = self.mlp(x)              # h_meta
        logits = self.classifier(h)  # (B, 1)
        if return_logits:
            return h, logits
        return h
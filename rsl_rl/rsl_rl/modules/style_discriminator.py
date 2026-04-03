import torch
import torch.nn as nn


class StyleDiscriminator(nn.Module):
    """A simple MLP-based style discriminator network.

    Place in rsl_rl so that it is versioned with the learning stack and can be
    optionally optimized/saved with policy checkpoints in the same package.
    """

    def __init__(self, input_dim: int, hidden_dims=None, dropout_p: float = 0.2, device: str = "cuda"):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if dropout_p and dropout_p > 0.0:
                layers.append(nn.Dropout(dropout_p))
            prev = h
        layers.append(nn.Linear(prev, 1))
        layers.append(nn.Sigmoid())

        self.net = nn.Sequential(*layers).to(device)
        self._init_weights()

    def _init_weights(self):
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
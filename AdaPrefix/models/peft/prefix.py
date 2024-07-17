import torch
import torch.nn as nn
from argparse import Namespace


class PrefixMLP(nn.Module):
    def __init__(
        self, prefix_length: int, dim: int, layers: int, args: Namespace
    ) -> None:
        super().__init__()
        self.low_prefix = nn.Parameter(torch.randn([prefix_length, dim]))
        bottleneck = args.bottleneck
        out_dim = dim * layers * 2
        self.mlp = nn.Sequential(
            *[
                nn.Linear(dim, bottleneck),
                nn.ReLU(),
                nn.Linear(bottleneck, out_dim),
            ]
        )

    def forward(self):
        return self.mlp(self.low_prefix)


class Prefix(nn.Module):
    def __init__(
        self, prefix_length: int, dim: int, num_head: int, args: Namespace
    ) -> None:
        super().__init__()
        self.prefix_length = prefix_length
        self.dim = dim
        self.num_head = num_head
        self.args = args

        if args.prefix_mlp:
            self.key_prefix = torch.randn(
                prefix_length * dim,
                requires_grad=True,
                device=torch.device(args.device),
            )
            self.value_prefix = torch.randn(
                prefix_length * dim,
                requires_grad=True,
                device=torch.device(args.device),
            )
        else:
            self.key_prefix = nn.Parameter(
                torch.randn(prefix_length * dim),
                requires_grad=True,
            )
            self.value_prefix = nn.Parameter(
                torch.randn(prefix_length * dim),
                requires_grad=True,
            )

    def forward(self, k_v):
        key, value = k_v[0], k_v[1]

        B, h, _, D_h = key.shape

        key_prefix = (
            self.key_prefix.unsqueeze(0)
            .expand(key.shape[0], -1)
            .view(B, h, self.prefix_length, D_h)
        )

        value_prefix = (
            self.value_prefix.unsqueeze(0)
            .expand(value.shape[0], -1)
            .view(B, h, self.prefix_length, D_h)
        )

        new_key = torch.cat([key_prefix, key], dim=2)
        new_value = torch.cat([value_prefix, value], dim=2)

        return new_key, new_value

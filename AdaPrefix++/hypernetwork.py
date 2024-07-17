import torch
import torch.nn as nn
from argparse import Namespace


class HyperPrefix(nn.Module):
    def __init__(self, args: Namespace, embed_dim: int) -> None:
        super().__init__()
        self.args = args
        hidden_dim = embed_dim // args.bottleneck_factor
        self.hnet = nn.Sequential(
            nn.Linear(
                in_features=(args.t_embeddings + args.l_embeddings),
                out_features=hidden_dim,
                bias=True,
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=hidden_dim,
                out_features=embed_dim,
                bias=True,
            ),
        )

    def forward(self, x):
        return self.hnet(x)


if __name__ == "__main__":
    args = Namespace(
        t_embeddings=768,
        l_embeddings=768,
        bottleneck_factor=2,
    )
    x = torch.randn(1, args.t_embeddings + args.l_embeddings)
    model = HyperPrefix(args, args.t_embeddings + args.l_embeddings)
    print(model(x).shape)

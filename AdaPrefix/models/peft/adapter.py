# --------------------------------------------------------
# References:
# https://github.com/jxhe/unify-parameter-efficient-tuning
# --------------------------------------------------------
# Modification:
# Added code for AdaPrefix++ implementation
# -- Sayanta Adhikari, ai22mtech12005@iith.ac.in
# ------------------------------------------

import math
import torch
import torch.nn as nn
from typing import Optional


class Adapter(nn.Module):
    def __init__(
        self,
        embed_dim: int = None,  # Transformer Embedding Dimension
        bottleneck: int = None,  # Adapter downsampling Dimension
        dropout: float = 0.0,
        init_option: str = "bert",  # Two Options: "bert" & "mam_adapter"
        adapter_layernorm_option: str = None,  # Only "in" or "out" allowed
        act_layer: nn.Module = nn.ReLU,
        args=None,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.bottleneck = bottleneck
        self.act_layer = act_layer or nn.ReLU
        self.adapter_layernorm_option = adapter_layernorm_option
        self.args = args

        # Layer Normalization either before or after Adapter
        self.adapter_layer_norm_before = None
        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.adapter_layer_norm_before = nn.LayerNorm(self.embed_dim)

        self.down_proj = nn.Linear(self.embed_dim, self.bottleneck)
        self.non_linear_func = self.act_layer()
        self.up_proj = nn.Linear(self.bottleneck, self.embed_dim)

        self.dropout = dropout

        # Initialization of Adapter Weights
        if init_option == "bert":
            with torch.no_grad():
                self.down_proj.weight.data.normal_(mean=0.0, std=0.02)
                self.up_proj.weight.data.normal_(mean=0.0, std=0.02)
                nn.init.zeros_(self.down_proj.bias)
                nn.init.zeros_(self.up_proj.bias)

        elif init_option == "mam_adapter":
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
                nn.init.zeros_(self.up_proj.weight)
                nn.init.zeros_(self.down_proj.bias)
                nn.init.zeros_(self.up_proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        add_residual: bool = True,
        residual: Optional[torch.Tensor] = None,
    ):
        # Getting residual
        residual = x if residual is None else residual

        # Layer Normalization
        if self.adapter_layernorm_option == "in":
            x = self.adapter_layer_norm_before(x)

        # Adapter
        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down)

        if self.adapter_layernorm_option == "out":
            up = self.adapter_layer_norm_before(up)

        # Adding Residual
        if add_residual:
            output = up + residual
        else:
            output = up

        return output

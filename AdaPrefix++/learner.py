import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import *
from timm import create_model
from argparse import Namespace
from hypernetwork import HyperPrefix
from typing import Dict


def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError(
            "Cannot use sin/cos positional encoding with "
            "odd dimension (got dim={:d})".format(d_model)
        )
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0.0, d_model, 2) * -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0.0, width).unsqueeze(1)
    pos_h = torch.arange(0.0, height).unsqueeze(1)
    pe[0:d_model:2, :, :] = (
        torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    )
    pe[1:d_model:2, :, :] = (
        torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    )
    pe[d_model::2, :, :] = (
        torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    )
    pe[d_model + 1 :: 2, :, :] = (
        torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    )
    return pe


class AdaPrefixPlusPlus(nn.Module):
    def __init__(self, args: Namespace) -> None:
        super().__init__()
        self.device = torch.device(args.device)
        self.args = args
        self.configs = get_vit_configs(args.model_name)
        self.leave_out_list = args.leave_out
        if type(self.configs["depth"]) is tuple or type(self.configs["depth"]) is list:
            depth = sum(self.configs["depth"])
        else:
            depth = self.configs["depth"]
        self.previous_hnet = None

        self.present_layers = [i for i in range(depth) if i not in self.leave_out_list]

        # Feature Model (Main)
        self.feature_model = self._get_feature_model(args.model_name, self.configs)

        # Making Layer Ids
        self.layer_ids = self._get_layer_embeddings()

        # Making Task Ids
        self.task_ids = nn.ParameterList()

        # Making Classifier Heads
        self.classifiers = nn.ModuleDict()

        # HyperPrefix
        self.hnet = HyperPrefix(args, embed_dim=self.configs["embed_dim"])

    def _get_layer_embeddings(self):
        args = self.args
        layer_embeddings = None
        if args.layer_embedding_type.endswith("learnable"):
            print("Using Learnable Layer Embeddings")
            layer_embeddings = nn.Parameter(
                torch.randn(
                    [
                        self.configs["depth"],
                        args.prefix_length * 2,
                        args.l_embeddings,
                    ],  # Dim: L x N x Le
                    device=self.device,
                ),
                requires_grad=True,
            )
        elif args.layer_embedding_type == "fixed_sine":
            print("Using Fixed Sine Layer Embeddings")
            layer_embeddings = nn.Parameter(
                positionalencoding2d(
                    self.configs["depth"],
                    args.prefix_length * 2,
                    args.l_embeddings,
                ),
                requires_grad=True,
            ).to(self.device)
        elif args.layer_embedding_type == "fixed_random":
            print("Using Fixed Random Layer Embeddings")
            layer_embeddings = nn.Parameter(
                torch.randn(
                    [
                        self.configs["depth"],
                        args.prefix_length * 2,
                        args.l_embeddings,
                    ],  # Dim: L x N x Le
                    device=self.device,
                ),
                requires_grad=False,
            )
        elif args.layer_embedding_type == "fixed_onehot":
            print("Using Fixed One-Hot Layer Embeddings")
            layer_embeddings = torch.zeros(
                [self.configs["depth"], args.prefix_length * 2, args.l_embeddings],
                device=self.device,
            )
            for i in range(self.configs["depth"]):
                layer_embeddings[i, :, i] = 1
            layer_embeddings = nn.Parameter(layer_embeddings, requires_grad=False)
        return layer_embeddings

    def _get_feature_model(self, model_name: str, configs: Dict):
        if model_name in [
            "vit_base_patch16_224",
            "vit_large_patch16_224",
            "deit_small_patch16_224",
            "deit_tiny_patch16_224",
        ]:
            model = VisionTransformer(
                img_size=configs["input_size"],
                embed_dim=configs["embed_dim"],
                depth=configs["depth"],
                num_heads=configs["num_heads"],
                has_output_adapter=self.args.has_output_adapter,
                has_prefix=self.args.has_prefix,
                leave_out=self.leave_out_list,
                args=self.args,
            )
        else:
            raise NotImplementedError()

        weights = create_model(model_name, pretrained=True).state_dict()
        l = [k for k in weights.keys() if "head" in k]
        for k in l:
            del weights[k]
        model.load_state_dict(weights, strict=False)
        return model

    def add_task(self, task_id: int, n_classes: int):
        self.task_ids.append(
            nn.Parameter(
                torch.randn(
                    [
                        1,
                        self.args.prefix_length * 2,
                        self.args.t_embeddings,
                    ],  # Dim: 1 x N x Te
                    device=self.device,
                ),
                requires_grad=True,
            )
        )
        self.classifiers["task_" + str(task_id)] = nn.Linear(
            in_features=self.feature_model.num_features,
            out_features=n_classes,
            bias=True,
            device=self.device,
        )

    def forward(self, x, prefix=True, adapter=True, task_id=None):
        # Getting Prefix Weights
        layer_ids = torch.index_select(
            self.layer_ids, 0, torch.tensor(self.present_layers).to(self.device)
        )
        input_token = torch.cat(
            [
                layer_ids,
                self.task_ids[task_id].expand(
                    layer_ids.shape[0], layer_ids.shape[1], -1
                ),
            ],
            dim=2,
        )
        prefix_weights = self.hnet(input_token)

        # Assigning Prefix Weights
        p_count = 0
        for m in self.feature_model.modules():
            if isinstance(m, Prefix):
                m.key_prefix = prefix_weights[
                    p_count, : self.args.prefix_length, :
                ].reshape(self.args.prefix_length * self.configs["embed_dim"])
                m.value_prefix = prefix_weights[
                    p_count, self.args.prefix_length :, :
                ].reshape(self.args.prefix_length * self.configs["embed_dim"])
                p_count += 1

        # Performing Forward Pass
        res = dict()
        res["features"] = f = self.feature_model(x, prefix, adapter)
        res["pre_logits"] = p = self.classifiers["task_" + str(task_id)](f)
        res["logits"] = F.softmax(p, dim=1)
        return res

    def save_previous_hnet(self):
        self.previous_hnet = copy.deepcopy(self.hnet.state_dict())

    def get_regularization_loss(self, task_id: int):
        loss = torch.tensor(0.0, device=self.device)
        temp_hnet = HyperPrefix(self.args, embed_dim=self.configs["embed_dim"])
        temp_hnet.load_state_dict(self.previous_hnet)
        temp_hnet.to(self.device)
        for id in range(task_id):
            input_token = torch.cat(
                [
                    self.layer_ids,
                    self.task_ids[id].expand(
                        self.layer_ids.shape[0], self.layer_ids.shape[1], -1
                    ),
                ],
                dim=2,
            )
            weights = self.hnet(input_token)
            with torch.no_grad():
                previous_weights = temp_hnet(input_token)
            n = torch.norm(weights - previous_weights, dim=(1, 2))
            loss += torch.mean(n)
        return loss / task_id

    def reset_adapters(self, init_type: str = "mam_adapter"):
        for m in self.feature_model.modules():
            if isinstance(m, Adapter):
                if init_type == "bert":
                    with torch.no_grad():
                        m.down_proj.weight.data.normal_(mean=0.0, std=0.02)
                        m.up_proj.weight.data.normal_(mean=0.0, std=0.02)
                        nn.init.zeros_(m.down_proj.bias)
                        nn.init.zeros_(m.up_proj.bias)

                elif init_type == "mam_adapter":
                    with torch.no_grad():
                        nn.init.kaiming_uniform_(m.down_proj.weight, a=math.sqrt(5))
                        nn.init.zeros_(m.up_proj.weight)
                        nn.init.zeros_(m.down_proj.bias)
                        nn.init.zeros_(m.up_proj.bias)

    def save_adapters(self, adapter_weights: Dict, task_id: int):
        adapt = {}
        counter = 0
        for m in self.feature_model.modules():
            if isinstance(m, Adapter):
                adapt[counter] = copy.deepcopy(m.state_dict())
                counter += 1
        adapter_weights[task_id] = adapt
        return adapter_weights

    def load_adapters(self, adapter_weights: Dict, task_id: int):
        counter = 0
        adapter_list = adapter_weights[task_id]
        for m in self.feature_model.modules():
            if isinstance(m, Adapter):
                m.load_state_dict(adapter_list[counter], strict=False)
                counter += 1

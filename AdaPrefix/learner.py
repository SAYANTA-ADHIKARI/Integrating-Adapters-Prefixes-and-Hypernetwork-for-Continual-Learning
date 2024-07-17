import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model
from typing import Dict
from models import *
import copy


class AdaPrefix(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.device = torch.device(args.device)
        self.args = args
        self.configs = get_vit_configs(args.model_name)
        self.leave_out_list = list(map(lambda x: int("".join(x)), args.leave_out))
        self.feature_model = self._get_feature_model(args.model_name, self.configs)

        self.classifiers = nn.ModuleDict()

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
                has_atten_adapter=self.args.has_atten_adapter,
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
        self.classifiers["task_" + str(task_id)] = nn.Linear(
            in_features=self.feature_model.num_features,
            out_features=n_classes,
            bias=True,
            device=self.device,
        )
        self.reset_adapters()
        self.reset_prefix()

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

    def reset_prefix(self):
        for m in self.feature_model.modules():
            if isinstance(m, PrefixMLP) or isinstance(m, Prefix):
                with torch.no_grad():
                    for p in m.parameters():
                        p.normal_(mean=0.0, std=0.02)

    def save_peft(self, peft_weights: Dict, task_id: int):
        adapt = {}
        adapt["adapter"] = {}
        adapt["prefix"] = {}
        adapter_counter = 0
        for m in self.feature_model.modules():
            if isinstance(m, Adapter):
                adapt["adapter"][adapter_counter] = copy.deepcopy(m.state_dict())
                adapter_counter += 1
        if self.feature_model.prefix_mlp is not None:
            adapt["prefix"] = copy.deepcopy(self.feature_model.prefix_mlp.state_dict())
        else:
            prefix_counter = 0
            for m in self.feature_model.modules():
                if isinstance(m, Prefix):
                    adapt["prefix"][prefix_counter] = copy.deepcopy(m.state_dict())
                    prefix_counter += 1
        peft_weights[task_id] = adapt
        return peft_weights

    def load_peft(self, peft_weights: Dict, task_id: int):
        adapter_list = peft_weights[task_id]["adapter"]
        prefix_list = peft_weights[task_id]["prefix"]
        counter = 0
        for m in self.feature_model.modules():
            if isinstance(m, Adapter):
                m.load_state_dict(adapter_list[counter], strict=False)
                counter += 1
        if self.feature_model.prefix_mlp is not None:
            self.feature_model.prefix_mlp.load_state_dict(prefix_list, strict=False)
        else:
            counter = 0
            for m in self.feature_model.modules():
                if isinstance(m, Prefix):
                    m.load_state_dict(prefix_list[counter], strict=False)
                    counter += 1

    def forward(self, x, prefix=False, adapter=False, task_id=None):
        # Performing Forward Pass
        res = dict()
        res["features"] = f = self.feature_model(x, prefix, adapter)
        res["pre_logits"] = p = self.classifiers["task_" + str(task_id)](f)
        res["logits"] = F.softmax(p, dim=1)
        return res

import yaml


def get_vit_configs(name: str = None):
    config = {}
    if name == "vit_base_patch16_224":
        config["input_size"] = 224
        config["patch_size"] = 16
        config["embed_dim"] = 768
        config["depth"] = 12
        config["num_heads"] = 12

        return config
    elif name == "deit_small_patch16_224":
        config["input_size"] = 224
        config["patch_size"] = 16
        config["embed_dim"] = 384
        config["depth"] = 12
        config["num_heads"] = 6

        return config
    elif name == "deit_tiny_patch16_224":
        config["input_size"] = 224
        config["patch_size"] = 16
        config["embed_dim"] = 192
        config["depth"] = 12
        config["num_heads"] = 3

        return config
    elif name == "swin_base_patch4_window7_224":
        config["input_size"] = 224
        config["patch_size"] = 4
        config["embed_dim"] = 128
        config["depth"] = (2, 2, 18, 2)
        config["num_heads"] = (4, 8, 16, 32)
        config["window_size"] = 7

        return config
    elif name == "cait_s24_224":
        config["input_size"] = 224
        config["patch_size"] = 16
        config["embed_dim"] = 384
        config["depth"] = (24, 2)
        config["num_heads"] = 8
        config["init_values"] = 1e-5

        return config
    elif name == "cvt_13_224":
        spec = yaml.safe_load(open("extra/cvt_13_224.yaml"))["SPEC"]
        config["input_size"] = 224
        config["embed_dim"] = spec["DIM_EMBED"][-1]
        config["depth"] = spec["DEPTH"]
        config["specs"] = spec

        return config
    elif name == "vit_large_patch16_224":
        config["input_size"] = 224
        config["patch_size"] = 16
        config["embed_dim"] = 1024
        config["depth"] = 24
        config["num_heads"] = 16

        return config
    else:
        raise NotImplementedError()

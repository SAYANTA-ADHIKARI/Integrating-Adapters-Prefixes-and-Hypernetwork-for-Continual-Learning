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
    elif name == "vit_large_patch16_224":
        config["input_size"] = 224
        config["patch_size"] = 16
        config["embed_dim"] = 1024
        config["depth"] = 24
        config["num_heads"] = 16

        return config
    else:
        raise NotImplementedError()

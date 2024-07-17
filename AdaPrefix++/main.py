import sys
import argparse
import datetime
import random
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn

from pathlib import Path

from data import build_continual_dataloader, get_dil_dataloader
import engine
from learner import AdaPrefixPlusPlus
import utils
import yaml

import warnings

warnings.filterwarnings(
    "ignore",
    "Argument interpolation should be of type InterpolationMode instead of int",
)


def main(args):
    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    cudnn.benchmark = True

    print(args)

    # Main Model and Dataloader
    if args.dil:
        data_loader, _ = get_dil_dataloader(args)
    else:
        data_loader, _ = build_continual_dataloader(args)
    model = AdaPrefixPlusPlus(args)
    print(model)

    model.to(device)
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )
        model_without_ddp = model.module

    if not args.distributed:
        global_batch_size = args.batch_size
    else:
        global_batch_size = args.batch_size * args.world_size
    args.lr = args.lr * global_batch_size / 256.0
    args.eta_min = args.eta_min * global_batch_size / 256.0

    criterion = torch.nn.CrossEntropyLoss().to(device)

    print(
        "*************************************** TASK INCREMENTAL TRAINING ***************************************"
    )
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    acc_matrix = engine.train_and_evaluate(
        model,
        model_without_ddp,
        criterion,
        data_loader,
        device,
        args,
    )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Total training time: {total_time_str}")
    print(
        "*************************************** TRAINING COMPLETED ***************************************"
    )
    return acc_matrix  # Acc@1 Matrix for all tasks


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Hyperfix Arguments Parser")

    config = parser.parse_known_args()[-1][0]
    subparser = parser.add_subparsers(dest="subparser_name")

    if config == "cifar100":
        from parsers.cifar100 import get_args_parser

        config_parser = subparser.add_parser("cifar100", help="Split-CIFAR100 configs")
    elif config == "five_datasets":
        from parsers.five_datasets import get_args_parser

        config_parser = subparser.add_parser("five_datasets", help="5-Datasets configs")
    elif config == "imagenet_r":
        from parsers.imagenet_r import get_args_parser

        config_parser = subparser.add_parser(
            "imagenet_r", help="Split-ImageNet-R configs"
        )
    elif config == "cddb":
        from parsers.cddb import get_args_parser

        config_parser = subparser.add_parser("cddb", help="DIL CDBB configs")
    else:
        raise NotImplementedError

    get_args_parser(config_parser)

    # Required arguments updated using yaml file
    args = parser.parse_args()
    opt = yaml.load(open(args.config), Loader=yaml.FullLoader)
    args = vars(args)
    args.update(opt)

    args = argparse.Namespace(**args)

    args.model_store_path = args.output_dir.replace("outputs", "weights")
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.model_store_path:
        Path(args.model_store_path).mkdir(parents=True, exist_ok=True)

    _ = main(args)

    sys.exit(0)

import sys
import argparse
import datetime
import random
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn

from pathlib import Path

from data import build_continual_dataloader
from learner import AdaPrefixPlusPlus
import utils
import yaml
from argparse import Namespace
from typing import Union, Iterable, Dict, List, Any
from timm.utils import accuracy

import warnings

warnings.filterwarnings(
    "ignore",
    "Argument interpolation should be of type InterpolationMode instead of int",
)


@torch.no_grad()
def load_til_trained_weights(
    model: torch.nn.Module, args: Namespace
) -> Union[torch.nn.Module, Dict]:
    if args.pretrained_weights:
        print(f"Loading pretrained weights from {args.pretrained_weights}")
        checkpoint = torch.load(args.pretrained_weights, map_location="cpu")
        for i in range(args.num_tasks):
            model.add_task(task_id=i, n_classes=args.nb_classes // args.num_tasks)
        model.classifiers.load_state_dict(checkpoint["classifiers"])
        model.hnet.load_state_dict(checkpoint["hnet"])
        model.layer_ids = torch.nn.Parameter(checkpoint["layer_ids"])
        model.task_ids.load_state_dict(checkpoint["task_ids"])
        model.feature_model.load_state_dict(checkpoint["feature_model"])
        adapter_weights = checkpoint["adapters"]
        return model, adapter_weights
    else:
        raise ValueError(
            f"Pretrained weights not found. Path in {args.pretrained_weights} is not valid"
        )


def entropy(logits):
    return -torch.sum(logits * torch.log(logits + 1e-8), dim=-1)


def get_best_logits(logits_list: List[Any], total_classes: int) -> List:
    required_logits = []
    # NOTE: Assumption that all tasks have same number of classes
    classes_per_task = logits_list[0].shape[-1]
    batch_task_logits = torch.stack(
        logits_list, dim=1
    )  # (batch_size, num_tasks, num_classes_per_task)
    batch_task_entropy = entropy(batch_task_logits)  # (batch_size, num_tasks)
    best_entropy_indexes = torch.argmin(batch_task_entropy, dim=-1)  # (batch_size,)
    for batch_index, best_entropy_index in enumerate(best_entropy_indexes):
        temp = torch.zeros(total_classes)
        temp[
            best_entropy_index
            * classes_per_task : (best_entropy_index + 1)
            * classes_per_task
        ] = logits_list[best_entropy_index][batch_index, :]
        required_logits.append(temp)
    result = torch.stack(required_logits, dim=0)
    assert result.shape == (len(logits_list[0]), total_classes)
    return result


@torch.no_grad()
def evaluate_cil(
    model: torch.nn.Module,
    model_without_ddp: torch.nn.Module,
    data_loader: Iterable,
    device: torch.device,
    task_id: int = -1,
    prefix: bool = False,
    adapter: bool = False,
    adapter_weights: Dict = None,
    args: Namespace = None,
):
    stat_matrix = np.zeros((2, args.num_tasks))  # 3 for Acc@1, Acc@5

    for i in range(task_id):
        print("-----------Testing Task {}--------------".format(i + 1))

        test_stats = evaluate(
            model=model,
            model_without_ddp=model_without_ddp,
            data_loader=data_loader[i]["test"],
            device=device,
            task_id=i,
            prefix=prefix,
            adapter=adapter,
            adapter_weights=adapter_weights,
            args=args,
        )

        stat_matrix[0, i] = test_stats["Acc@1"]
        stat_matrix[1, i] = test_stats["Acc@5"]

    avg_stat = np.divide(np.sum(stat_matrix, axis=1), task_id)

    result_str = "[Average accuracy till task{}]\tAcc@1: {:.4f}\tAcc@5: {:.4f}".format(
        task_id, avg_stat[0], avg_stat[1]
    )
    print("-------------------------------------------")
    print(result_str)
    print("-------------------------------------------")

    return test_stats, avg_stat


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    model_without_ddp: torch.nn.Module,
    data_loader: Iterable,
    device: torch.device,
    task_id: int = -1,
    prefix: bool = True,
    adapter: bool = True,
    adapter_weights: dict = None,
    args: Namespace = None,
):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test: [Task {}]".format(task_id + 1)

    # switch to evaluation mode
    model.eval()

    with torch.no_grad():
        for input, target in metric_logger.log_every(
            data_loader, args.print_freq, header
        ):
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            batch_logits = []
            for t in range(args.num_tasks):
                if args.has_atten_adapter or args.has_output_adapter:
                    model_without_ddp.load_adapters(adapter_weights, t)
                batch_logits.append(
                    model(input, prefix=prefix, adapter=adapter, task_id=task_id)[
                        "logits"
                    ]
                )
            logits = get_best_logits(
                logits_list=batch_logits, total_classes=args.nb_classes
            )
            logits = logits.to(device, non_blocking=True)
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))

            metric_logger.meters["Acc@1"].update(acc1.item(), n=input.shape[0])
            metric_logger.meters["Acc@5"].update(acc5.item(), n=input.shape[0])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print(
        "* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f}".format(
            top1=metric_logger.meters["Acc@1"],
            top5=metric_logger.meters["Acc@5"],
        )
    )

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


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
    data_loader, _ = build_continual_dataloader(args)
    model = AdaPrefixPlusPlus(args)

    # Load the model weights
    model, adapter_weights = load_til_trained_weights(model, args)

    model.to(device)
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )
        model_without_ddp = model.module

    print(
        f"************* Starting Evaluation for {args.num_tasks} tasks in CIL setup *************"
    )
    start_time = time.time()

    _, avg_stats = evaluate_cil(
        model=model,
        model_without_ddp=model_without_ddp,
        data_loader=data_loader,
        device=device,
        task_id=args.num_tasks,
        prefix=args.has_prefix,
        adapter=args.has_atten_adapter or args.has_output_adapter,
        adapter_weights=adapter_weights,
        args=args,
    )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Total training time: {total_time_str}")
    print(
        "******************************** Evaluation Completed **************************************"
    )
    return float(avg_stats[0])  # Acc@1


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
    else:
        raise NotImplementedError
    get_args_parser(config_parser)

    # Required arguments updated using yaml file
    args = parser.parse_args()
    opt = yaml.load(open(args.config), Loader=yaml.FullLoader)
    args = vars(args)
    args.update(opt)

    args = argparse.Namespace(**args)

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    _ = main(args)

    sys.exit(0)

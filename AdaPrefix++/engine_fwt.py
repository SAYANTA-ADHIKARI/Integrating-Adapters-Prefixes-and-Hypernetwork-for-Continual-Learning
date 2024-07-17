"""
Train and eval functions used in main.py
"""

import math
import sys
import os
import json
from typing import Iterable
from argparse import Namespace

import torch
import numpy as np

from timm.utils import accuracy
from optimizer import get_optimizer, get_lr_scheduler
from learner import AdaPrefixPlusPlus
from models import Adapter

import utils


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_norm: float = 0,
    set_training_mode: bool = True,
    task_id: int = -1,
    prefix: bool = True,
    adapter: bool = True,
    args: Namespace = None,
):
    model.train(set_training_mode)

    if args.distributed and utils.get_world_size() > 1:
        data_loader.sampler.set_epoch(epoch)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("Lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter(
        "Loss", utils.SmoothedValue(window_size=1, fmt="{value:.4f}")
    )
    header = f"Train Task{task_id+1}: Epoch[{epoch+1:{int(math.log10(args.epochs))+1}}/{args.epochs}]"
    for input, target in metric_logger.log_every(data_loader, args.print_freq, header):
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        target = (
            target % (args.nb_classes // args.num_tasks) if not args.dil else target
        )  # Hack Required For Training
        output = model(input, prefix=prefix, adapter=adapter, task_id=0)
        logits = output["logits"]

        loss = criterion(logits, target)

        acc1, acc5 = accuracy(logits, target, topk=(1, 5))

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        if args.distributed:
            torch.cuda.synchronize()
        metric_logger.update(Loss=loss.item())
        metric_logger.update(Lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters["Acc@1"].update(acc1.item(), n=input.shape[0])
        metric_logger.meters["Acc@5"].update(acc5.item(), n=input.shape[0])

    # gather the stats from all processes
    if args.distributed:
        metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    data_loader: Iterable,
    device: torch.device,
    task_id: int = -1,
    prefix: bool = True,
    adapter: bool = True,
    args: Namespace = None,
    val: bool = True,
):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    if val:
        header = "Validation: [Task {}]".format(task_id + 1)
    else:
        header = "Test: [Task {}]".format(task_id + 1)

    # switch to evaluation mode
    model.eval()

    with torch.no_grad():
        for input, target in metric_logger.log_every(
            data_loader, args.print_freq, header
        ):
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            target = (
                target % (args.nb_classes // args.num_tasks) if not args.dil else target
            )

            output = model(input, prefix=prefix, adapter=adapter, task_id=0)
            logits = output["logits"]

            loss = criterion(logits, target)

            acc1, acc5 = accuracy(logits, target, topk=(1, 5))

            metric_logger.meters["Loss"].update(loss.item())
            metric_logger.meters["Acc@1"].update(acc1.item(), n=input.shape[0])
            metric_logger.meters["Acc@5"].update(acc5.item(), n=input.shape[0])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print(
        "* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}".format(
            top1=metric_logger.meters["Acc@1"],
            top5=metric_logger.meters["Acc@5"],
            losses=metric_logger.meters["Loss"],
        )
    )

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_and_evaluate(
    criterion: torch.nn.Module,
    data_loader: Iterable,
    device: torch.device,
    args: Namespace = None,
):
    # create matrix to save end-of-task accuracies
    acc_matrix = np.zeros((args.num_tasks, args.num_tasks))

    for task_id in range(args.num_tasks):
        print("==================== Task {} ====================".format(task_id + 1))
        model = AdaPrefixPlusPlus(args)

        model.to(device)

        model_without_ddp = model
        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu], find_unused_parameters=True
            )
            model_without_ddp = model.module
        n_classes = (
            args.nb_classes // args.num_tasks if not args.dil else args.nb_classes
        )
        model_without_ddp.add_task(task_id=0, n_classes=n_classes)
        if args.has_output_adapter:
            model_without_ddp.reset_adapters()

        print("Resetting Optimizer and LR Scheduler")
        parameters = [
            model_without_ddp.task_ids[0],
        ]
        for p in model_without_ddp.classifiers["task_" + str(0)].parameters():
            parameters.append(p)
        for p in model_without_ddp.hnet.parameters():
            parameters.append(p)
        for m in model_without_ddp.modules():
            if isinstance(m, Adapter):
                for p in m.parameters():
                    parameters.append(p)
        if not args.layer_embedding_type.startswith("fixed"):
            parameters.append(model_without_ddp.layer_ids)

        optimizer = get_optimizer(args, parameters)

        if args.sched != "None":
            lr_scheduler = get_lr_scheduler(args, optimizer)
        elif args.sched == "None":
            lr_scheduler = None
        print("Training Task {}".format(task_id + 1))
        if args.dil:
            epochs = args.epochs[task_id]
        else:
            epochs = args.epochs
        for epoch in range(epochs):
            train_stats = train_one_epoch(
                model=model,
                criterion=criterion,
                data_loader=data_loader[task_id]["train"],
                optimizer=optimizer,
                device=device,
                epoch=epoch,
                max_norm=args.clip_grad,
                set_training_mode=True,
                task_id=task_id,
                prefix=True,
                adapter=True,
                args=args,
            )
            ## Change when performing best model training
            # Validation after each epoch
            if args.validation:
                val_stats = evaluate(
                    model=model,
                    data_loader=data_loader[task_id]["val"],
                    device=device,
                    task_id=task_id,
                    prefix=True,
                    adapter=True,
                    args=args,
                )

            if lr_scheduler:
                lr_scheduler.step(epoch)
        print(
            "=================Testing till Task {}=================".format(task_id + 1)
        )
        test_stats = evaluate(
            model=model,
            data_loader=data_loader[task_id]["test"],
            device=device,
            task_id=task_id,
            prefix=True,
            adapter=True,
            args=args,
            val=False,
        )

        acc_matrix[task_id, task_id] = test_stats["Acc@1"]

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"test_{k}": v for k, v in test_stats.items()},
            "epoch": epoch + 1,
        }
        print("=====================Logging Stats=========================")
        print(json.dumps(log_stats))
        print("===========================================================")

    if args.output_dir and utils.is_main_process():
        np.savetxt(
            os.path.join(
                args.output_dir,
                f"accuracy_matrix_fwt_{args.exp_name}.txt",
            ),
            acc_matrix,
            fmt="%.3f",
            delimiter=",",
        )
    return acc_matrix

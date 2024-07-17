"""
Train and eval functions used in main.py
"""

import math
import sys
import os
import json
from typing import Iterable, Dict, Union
from argparse import Namespace

import torch
import numpy as np
from models import Adapter, Prefix, PrefixMLP
from timm.utils import accuracy
from optimizer import get_optimizer, get_lr_scheduler

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
        target = target % (
            args.nb_classes // args.num_tasks
        )  # Hack Required For Training
        output = model(input, prefix=prefix, adapter=adapter, task_id=task_id)
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
            target = target % (args.nb_classes // args.num_tasks)

            output = model(input, prefix=prefix, adapter=adapter, task_id=task_id)
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


@torch.no_grad()
def evaluate_till_now(
    model: torch.nn.Module,
    model_without_ddp: torch.nn.Module,
    data_loader: Iterable,
    device: torch.device,
    task_id: int = -1,
    acc_matrix: np.ndarray = None,
    prefix: bool = False,
    adapter: bool = False,
    peft_weights: Dict = None,
    args: Namespace = None,
):
    stat_matrix = np.zeros((3, args.num_tasks))  # 3 for Acc@1, Acc@5, Loss

    for i in range(task_id + 1):
        print("-----------Testing Task {}--------------".format(i + 1))
        if args.has_atten_adapter or args.has_output_adapter or args.has_prefix:
            model_without_ddp.load_peft(peft_weights, i)
        # Change when performing best model training
        test_stats = evaluate(
            model=model,
            data_loader=data_loader[i]["test"],
            device=device,
            task_id=i,
            prefix=prefix,
            adapter=adapter,
            args=args,
            val=False,
        )

        stat_matrix[0, i] = test_stats["Acc@1"]
        stat_matrix[1, i] = test_stats["Acc@5"]
        stat_matrix[2, i] = test_stats["Loss"]

        acc_matrix[i, task_id] = test_stats["Acc@1"]

    avg_stat = np.divide(np.sum(stat_matrix, axis=1), task_id + 1)

    diagonal = np.diag(acc_matrix)

    result_str = "[Average accuracy till task{}]\tAcc@1: {:.4f}\tAcc@5: {:.4f}\tLoss: {:.4f}".format(
        task_id + 1, avg_stat[0], avg_stat[1], avg_stat[2]
    )
    if task_id > 0:
        forgetting = np.mean(
            (np.max(acc_matrix, axis=1) - acc_matrix[:, task_id])[:task_id]
        )
        backward = np.mean((acc_matrix[:, task_id] - diagonal)[:task_id])

        result_str += "\tForgetting: {:.4f}\tBackward: {:.4f}".format(
            forgetting, backward
        )
    print("-------------------------------------------")
    print(result_str)
    print("-------------------------------------------")

    return test_stats


def train_and_evaluate(
    model: torch.nn.Module,
    model_without_ddp: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    device: torch.device,
    args: Namespace = None,
):
    # create matrix to save end-of-task accuracies
    acc_matrix = np.zeros((args.num_tasks, args.num_tasks))
    peft_weights = {}
    for task_id in range(args.num_tasks):
        print("==================== Task {} ====================".format(task_id + 1))
        model_without_ddp.add_task(
            task_id=task_id, n_classes=args.nb_classes // args.num_tasks
        )

        print("Resetting Optimizer and LR Scheduler")
        parameters = []
        for p in model_without_ddp.classifiers["task_" + str(task_id)].parameters():
            parameters.append(p)
        for m in model_without_ddp.modules():
            if (
                isinstance(m, Adapter)
                or isinstance(m, Prefix)
                or isinstance(m, PrefixMLP)
            ):
                for p in m.parameters():
                    parameters.append(p)
        optimizer = get_optimizer(args, parameters)

        if args.sched != "None":
            lr_scheduler = get_lr_scheduler(args, optimizer)
        elif args.sched == "None":
            lr_scheduler = None
        print("Training Task {}".format(task_id + 1))
        for epoch in range(args.epochs):
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
                _ = evaluate(
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

        if args.has_atten_adapter or args.has_output_adapter or args.has_prefix:
            peft_weights = model_without_ddp.save_peft(peft_weights, task_id)
        print(
            "=================Testing till Task {}=================".format(task_id + 1)
        )

        test_stats = evaluate_till_now(
            model=model,
            model_without_ddp=model_without_ddp,
            data_loader=data_loader,
            device=device,
            task_id=task_id,
            acc_matrix=acc_matrix,
            prefix=True,
            adapter=True,
            args=args,
            peft_weights=peft_weights,
        )

        log_stats = {
            **{f"train_{k}": round(v, 4) for k, v in train_stats.items()},
            **{f"test_{k}": round(v, 4) for k, v in test_stats.items()},
            "epoch": epoch + 1,
        }
        print("=====================Logging Stats=========================")
        print(json.dumps(log_stats))
        print("===========================================================")

    if args.output_dir and utils.is_main_process():
        np.savetxt(
            os.path.join(args.output_dir, f"accuracy_matrix_{args.exp_name}.txt"),
            acc_matrix,
            fmt="%.3f",
            delimiter=",",
        )
    print("Saving Model!!!")

    if args.model_store_path and utils.is_main_process():
        os.makedirs(os.path.join(args.model_store_path), exist_ok=True)
        weights = {}
        weights["feature_model"] = model_without_ddp.feature_model.state_dict()
        weights["peft"] = peft_weights
        weights["classifiers"] = model_without_ddp.classifiers.state_dict()
        torch.save(
            weights,
            os.path.join(args.model_store_path, f"model_weights_{args.exp_name}.pth"),
        )
        print("Model Saved!!!")

    return acc_matrix

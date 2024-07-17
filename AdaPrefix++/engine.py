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
from torch.utils.data import Dataset
from timm.utils import accuracy
from optimizer import get_optimizer, get_lr_scheduler
from models import Adapter

import utils


def train_one_epoch(
    model: torch.nn.Module,
    model_without_ddp: torch.nn.Module,
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
        "TLoss", utils.SmoothedValue(window_size=1, fmt="{value:.4f}")
    )
    metric_logger.add_meter(
        "CELoss", utils.SmoothedValue(window_size=1, fmt="{value:.4f}")
    )
    metric_logger.add_meter(
        "RLoss", utils.SmoothedValue(window_size=1, fmt="{value:.4f}")
    )
    epochs = args.epochs[task_id] if args.dil else args.epochs
    header = (
        f"Train Task{task_id+1}: Epoch[{epoch+1:{int(math.log10(epochs))+1}}/{epochs}]"
    )
    for input, target in metric_logger.log_every(data_loader, args.print_freq, header):
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        target = (
            target % (args.nb_classes // args.num_tasks) if not args.dil else target
        )  # Hack Required For Training
        output = model(input, prefix=prefix, adapter=adapter, task_id=task_id)
        logits = output["logits"]

        celoss = criterion(logits, target)
        rloss = torch.tensor(0.0, device=device)
        if args.h_reg > 0.0 and task_id > 0:
            rloss += model_without_ddp.get_regularization_loss(task_id=task_id)
            loss = celoss + args.h_reg * rloss
        else:
            loss = celoss

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
        metric_logger.update(TLoss=loss.item())
        metric_logger.update(CELoss=celoss.item())
        metric_logger.update(RLoss=rloss.item())
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
    adapter_weights: Dict = None,
    args: Namespace = None,
):
    stat_matrix = np.zeros((3, args.num_tasks))  # 3 for Acc@1, Acc@5, Loss

    for i in range(task_id + 1):
        print("-----------Testing Task {}--------------".format(i + 1))
        if args.has_atten_adapter or args.has_output_adapter:
            model_without_ddp.load_adapters(adapter_weights, i)
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
    data_loader: Union[Iterable, Dataset],
    device: torch.device,
    args: Namespace = None,
):
    # create matrix to save end-of-task accuracies
    acc_matrix = np.zeros((args.num_tasks, args.num_tasks))
    adapter_weights = {}

    for task_id in range(args.num_tasks):
        print("==================== Task {} ====================".format(task_id + 1))
        n_classes = (
            args.nb_classes // args.num_tasks if not args.dil else args.nb_classes
        )
        model_without_ddp.add_task(task_id=task_id, n_classes=n_classes)
        if args.has_output_adapter:
            model_without_ddp.reset_adapters()

        print("Resetting Optimizer and LR Scheduler")
        parameters = [
            model_without_ddp.task_ids[task_id],
        ]
        for p in model_without_ddp.classifiers["task_" + str(task_id)].parameters():
            parameters.append(p)
        for p in model_without_ddp.hnet.parameters():
            parameters.append(p)
        for m in model_without_ddp.modules():
            if isinstance(m, Adapter):
                for p in m.parameters():
                    parameters.append(p)

        if not args.layer_embedding_type == "throughout_learnable":
            if task_id == 0 and not args.layer_embedding_type.startswith("fixed"):
                print("Training Layer Embeddings!!")
                parameters.append(model_without_ddp.layer_ids)
        else:
            print("Training Layer Embeddings!!")
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
                model_without_ddp=model_without_ddp,
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

        model_without_ddp.save_previous_hnet()
        if args.has_atten_adapter or args.has_output_adapter:
            adapter_weights = model_without_ddp.save_adapters(adapter_weights, task_id)
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
            adapter_weights=adapter_weights,
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
            os.path.join(
                args.output_dir,
                f"accuracy_matrix_cl_{args.exp_name}.txt",
            ),
            acc_matrix,
            fmt="%.3f",
            delimiter=",",
        )
    print("Saving Model!!!")

    if args.model_store_path and utils.is_main_process():
        os.makedirs(os.path.join(args.model_store_path), exist_ok=True)
        weights = {}
        weights["feature_model"] = model_without_ddp.feature_model.state_dict()
        weights["adapters"] = adapter_weights
        weights["task_ids"] = model_without_ddp.task_ids.state_dict()
        weights["classifiers"] = model_without_ddp.classifiers.state_dict()
        weights["hnet"] = model_without_ddp.hnet.state_dict()
        weights["layer_ids"] = model_without_ddp.layer_ids.detach().cpu()
        torch.save(
            weights,
            os.path.join(
                args.model_store_path,
                f"model_weights_{args.exp_name}.pth",
            ),
        )
        print("Model Saved!!!")

    return acc_matrix

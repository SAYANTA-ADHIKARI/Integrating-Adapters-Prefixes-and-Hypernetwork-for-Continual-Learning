import torch.optim as optim


def get_optimizer(args, parameters):
    if args.opt == "sgd":
        opt = optim.SGD(
            parameters,
            lr=args.lr,
            momentum=args.opt_momentum,
            weight_decay=args.weight_decay,
            nesterov=args.opt_nev,
        )
    elif args.opt == "adam":
        opt = optim.Adam(
            parameters,
            lr=args.lr,
            betas=args.opt_betas,
            weight_decay=args.weight_decay,
            eps=args.opt_eps,
        )
    else:
        raise NotImplementedError("This Optimizer is not Implemented!!")

    return opt


def get_lr_scheduler(args, optimizer):
    if args.sched == "constant":
        lr_sched = optim.lr_scheduler.ConstantLR(
            optimizer,
            factor=args.scale,
            total_iters=args.step_size,
            verbose=args.verbose,
        )
    elif args.sched == "step":
        lr_sched = optim.lr_scheduler.StepLR(
            optimizer, step_size=args.step_size, gamma=args.scale, verbose=args.verbose
        )

    elif args.sched == "cosine":
        lr_sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer=optimizer,
            T_0=args.step_size,
            T_mult=args.step_mult,
            eta_min=args.eta_min,
            verbose=args.verbose,
        )
    else:
        raise NotImplementedError("LR Scheduler not Implemented")

    return lr_sched

import random

import torch
from torch.utils.data.dataset import Subset
from torchvision import datasets, transforms
from torch.utils.data import random_split

from datasets_setup.continual_datasets import *

import utils


class Lambda(transforms.Lambda):
    def __init__(self, lambd, nb_classes):
        super().__init__(lambd)
        self.nb_classes = nb_classes

    def __call__(self, img):
        return self.lambd(img, self.nb_classes)


def target_transform(x, nb_classes):
    return x + nb_classes


def build_continual_dataloader(args):
    dataloader = list()
    class_mask = list() if args.task_inc or args.train_mask else None

    transform_train = build_transform(True, args)
    transform_val = build_transform(False, args)

    if args.dataset.startswith("Split-"):
        dataset_train, dataset_val = get_dataset(
            args.dataset.replace("Split-", ""), transform_train, transform_val, args
        )

        args.nb_classes = len(dataset_val.classes)

        splited_dataset, class_mask = split_single_dataset(
            dataset_train, dataset_val, args
        )
    else:
        if args.dataset == "5-datasets":
            dataset_list = ["SVHN", "MNIST", "CIFAR10", "NotMNIST", "FashionMNIST"]
        else:
            dataset_list = args.dataset.split(",")

        if args.shuffle:
            random.shuffle(dataset_list)
        print(dataset_list)

        args.nb_classes = 0

    for i in range(args.num_tasks):
        if args.dataset.startswith("Split-"):
            if args.validation:
                dataset_train, dataset_val, dataset_test = splited_dataset[i]
            else:
                dataset_train, dataset_test = splited_dataset[i]
                dataset_val = None
        else:
            dataset_train, dataset_test = get_dataset(
                dataset_list[i], transform_train, transform_val, args
            )
            # Splitting in Train and Validation set
            if args.validation:
                dataset_train, dataset_val = random_split(dataset_train, [0.8, 0.2])
            else:
                dataset_val = None

            transform_target = Lambda(target_transform, args.nb_classes)

            if class_mask is not None:
                class_mask.append(
                    [i + args.nb_classes for i in range(len(dataset_test.classes))]
                )
                args.nb_classes += len(dataset_test.classes)

            if not args.task_inc:
                dataset_train.target_transform = transform_target
                if args.validation:
                    dataset_val.target_transform = transform_target
                dataset_test.target_transform = transform_target

        if args.distributed and utils.get_world_size() > 1:
            num_tasks = utils.get_world_size()
            global_rank = utils.get_rank()

            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
            if args.validation:
                sampler_val = torch.utils.data.SequentialSampler(dataset_val)
            sampler_test = torch.utils.data.SequentialSampler(dataset_test)
        else:
            sampler_train = torch.utils.data.RandomSampler(dataset_train)
            if args.validation:
                sampler_val = torch.utils.data.SequentialSampler(dataset_val)
            sampler_test = torch.utils.data.SequentialSampler(dataset_test)

        data_loader_train = torch.utils.data.DataLoader(
            dataset_train,
            sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
        )
        if args.validation:
            data_loader_val = torch.utils.data.DataLoader(
                dataset_val,
                sampler=sampler_val,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
            )

        data_loader_test = torch.utils.data.DataLoader(
            dataset_test,
            sampler=sampler_test,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
        )
        if args.validation:
            dataloader.append(
                {
                    "train": data_loader_train,
                    "val": data_loader_val,
                    "test": data_loader_test,
                }
            )
        else:
            dataloader.append({"train": data_loader_train, "test": data_loader_test})

    return dataloader, class_mask


def get_dil_dataloader(args):
    if args.dataset.upper() == "CDDB":
        dataloaders = list()

        transform_train = build_transform(True, args)
        transform_val = build_transform(False, args)
        args.nb_classes = 2

        dataset_train, dataset_val = get_dataset(
            args.dataset, transform_train, transform_val, args
        )
        args.num_tasks = len(dataset_train)

        for train_data, val_data in zip(dataset_train, dataset_val):
            if args.distributed and utils.get_world_size() > 1:
                num_tasks = utils.get_world_size()
                global_rank = utils.get_rank()

                sampler_train = torch.utils.data.DistributedSampler(
                    train_data, num_replicas=num_tasks, rank=global_rank, shuffle=True
                )
                sampler_test = torch.utils.data.SequentialSampler(val_data)
            else:
                sampler_train = torch.utils.data.RandomSampler(train_data)
                sampler_test = torch.utils.data.SequentialSampler(val_data)
            train_loader = torch.utils.data.DataLoader(
                train_data,
                sampler=sampler_train,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
            )
            val_loader = torch.utils.data.DataLoader(
                val_data,
                sampler=sampler_test,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
            )
            dataloaders.append({"train": train_loader, "test": val_loader})

        return dataloaders, None
    elif args.dataset.upper() == "CORE50":
        dataloaders = list()

        transform_train = build_transform(True, args)
        transform_val = build_transform(False, args)

        dataset_train = CORe50(args.data_path, train=True, transform=transform_train)
        dataset_test = CORe50(args.data_path, train=False, transform=transform_val)
        args.num_tasks = len(dataset_train)
        args.nb_classes = dataset_train[0].classes

        sampler_test = torch.utils.data.SequentialSampler(dataset_test)
        val_loader = torch.utils.data.DataLoader(
            dataset_test,
            sampler=sampler_test,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
        )
        for train_data in dataset_train:
            if args.distributed and utils.get_world_size() > 1:
                num_tasks = utils.get_world_size()
                global_rank = utils.get_rank()

                sampler_train = torch.utils.data.DistributedSampler(
                    train_data, num_replicas=num_tasks, rank=global_rank, shuffle=True
                )
            else:
                sampler_train = torch.utils.data.RandomSampler(train_data)
            train_loader = torch.utils.data.DataLoader(
                train_data,
                sampler=sampler_train,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
            )
            dataloaders.append({"train": train_loader, "test": val_loader})
        return dataloaders, None
    else:
        raise ValueError("Dataset {} not found.".format(args.dataset))


def get_dataset(
    dataset,
    transform_train,
    transform_val,
    args,
):
    if dataset == "CIFAR100":
        dataset_train = datasets.CIFAR100(
            args.data_path, train=True, download=True, transform=transform_train
        )
        dataset_val = datasets.CIFAR100(
            args.data_path, train=False, download=True, transform=transform_val
        )

    elif dataset == "CIFAR10":
        dataset_train = datasets.CIFAR10(
            args.data_path, train=True, download=True, transform=transform_train
        )
        dataset_val = datasets.CIFAR10(
            args.data_path, train=False, download=True, transform=transform_val
        )

    elif dataset == "MNIST":
        dataset_train = MNIST_RGB(
            args.data_path, train=True, download=True, transform=transform_train
        )
        dataset_val = MNIST_RGB(
            args.data_path, train=False, download=True, transform=transform_val
        )

    elif dataset == "FashionMNIST":
        dataset_train = FashionMNIST(
            args.data_path, train=True, download=True, transform=transform_train
        )
        dataset_val = FashionMNIST(
            args.data_path, train=False, download=True, transform=transform_val
        )

    elif dataset == "SVHN":
        dataset_train = SVHN(
            args.data_path, split="train", download=True, transform=transform_train
        )
        dataset_val = SVHN(
            args.data_path, split="test", download=True, transform=transform_val
        )

    elif dataset == "NotMNIST":
        dataset_train = NotMNIST(
            args.data_path, train=True, download=True, transform=transform_train
        )
        dataset_val = NotMNIST(
            args.data_path, train=False, download=True, transform=transform_val
        )

    elif dataset == "Flower102":
        dataset_train = Flowers102(
            args.data_path, split="train", download=True, transform=transform_train
        )
        dataset_val = Flowers102(
            args.data_path, split="test", download=True, transform=transform_val
        )

    elif dataset == "Cars196":
        dataset_train = StanfordCars(
            args.data_path, split="train", download=True, transform=transform_train
        )
        dataset_val = StanfordCars(
            args.data_path, split="test", download=True, transform=transform_val
        )

    elif dataset == "CUB200":
        dataset_train = CUB200(
            args.data_path, train=True, download=True, transform=transform_train
        ).data
        dataset_val = CUB200(
            args.data_path, train=False, download=True, transform=transform_val
        ).data

    elif dataset == "Scene67":
        dataset_train = Scene67(
            args.data_path, train=True, download=True, transform=transform_train
        ).data
        dataset_val = Scene67(
            args.data_path, train=False, download=True, transform=transform_val
        ).data

    elif dataset == "TinyImagenet":
        dataset_train = TinyImagenet(
            args.data_path, train=True, download=True, transform=transform_train
        ).data
        dataset_val = TinyImagenet(
            args.data_path, train=False, download=True, transform=transform_val
        ).data

    elif dataset == "Imagenet-R":
        dataset_train = Imagenet_R(
            args.data_path, train=True, download=True, transform=transform_train
        )
        dataset_val = Imagenet_R(
            args.data_path, train=False, download=True, transform=transform_val
        )
    elif dataset == "DomainNet":
        dataset_train = DomainNet(
            args.data_path, train=True, download=True, transform=transform_train
        )
        dataset_val = DomainNet(
            args.data_path, train=False, download=True, transform=transform_val
        )
    elif dataset == "CDDB":
        dataset_train = CDDB_dataset(
            args.data_path, train=True, download=True, transform=transform_train
        )
        dataset_val = CDDB_dataset(
            args.data_path, train=False, download=True, transform=transform_val
        )
    else:
        raise ValueError("Dataset {} not found.".format(dataset))

    return dataset_train, dataset_val


def split_single_dataset(dataset_train, dataset_val, args):
    nb_classes = len(dataset_val.classes)
    # print(len(dataset_train.classes), len(dataset_val.classes))
    # exit()
    assert nb_classes % args.num_tasks == 0
    classes_per_task = nb_classes // args.num_tasks

    labels = [i for i in range(nb_classes)]

    split_datasets = list()
    mask = list() if args.task_inc or args.train_mask else None

    if args.shuffle:
        random.shuffle(labels)

    for _ in range(args.num_tasks):
        train_split_indices = []
        test_split_indices = []

        scope = labels[:classes_per_task]
        labels = labels[classes_per_task:]

        if mask is not None:
            mask.append(scope)

        for k in range(len(dataset_train.targets)):
            if int(dataset_train.targets[k]) in scope:
                train_split_indices.append(k)

        for h in range(len(dataset_val.targets)):
            if int(dataset_val.targets[h]) in scope:
                test_split_indices.append(h)

        subset_train, subset_test = Subset(dataset_train, train_split_indices), Subset(
            dataset_val, test_split_indices
        )

        # Generating validation set from training set
        if args.validation:
            subset_train, subset_val = random_split(subset_train, [0.9, 0.1])
            split_datasets.append([subset_train, subset_val, subset_test])
        else:
            split_datasets.append([subset_train, subset_test])

    return split_datasets, mask


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        scale = (0.05, 1.0)
        ratio = (3.0 / 4.0, 4.0 / 3.0)
        transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(args.input_size, scale=scale, ratio=ratio),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
            ]
        )
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * args.input_size)
        t.append(
            transforms.Resize(
                size, interpolation=3
            ),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))
    t.append(transforms.ToTensor())

    return transforms.Compose(t)

# ------------------------------------------
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
# ------------------------------------------

import os

import os.path
import pathlib
from pathlib import Path

from typing import Any, Tuple

import glob
from shutil import move, rmtree

import numpy as np

import torch
from torchvision import datasets
from torchvision.datasets.utils import (
    download_url,
    check_integrity,
    verify_str_arg,
    download_and_extract_archive,
)
import yaml
import gdown
import zipfile
import tarfile
import PIL
from PIL import Image
import numpy as np
import pickle as pkl
import requests
import logging
from hashlib import md5
import sys

from .dataset_utils import read_image_file, read_label_file


class MNIST_RGB(datasets.MNIST):
    def __init__(
        self, root, train=True, transform=None, target_transform=None, download=False
    ):
        super(MNIST_RGB, self).__init__(
            root,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )
        self.train = train  # training set or test set

        if self._check_legacy_exist():
            self.data, self.targets = self._load_legacy_data()
            return

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError(
                "Dataset not found. You can use download=True to download it"
            )

        self.data, self.targets = self._load_data()

    def _check_legacy_exist(self):
        processed_folder_exists = os.path.exists(self.processed_folder)
        if not processed_folder_exists:
            return False

        return all(
            check_integrity(os.path.join(self.processed_folder, file))
            for file in (self.training_file, self.test_file)
        )

    def _load_legacy_data(self):
        # This is for BC only. We no longer cache the data in a custom binary, but simply read from the raw data
        # directly.
        data_file = self.training_file if self.train else self.test_file
        return torch.load(os.path.join(self.processed_folder, data_file))

    def _load_data(self):
        image_file = f"{'train' if self.train else 't10k'}-images-idx3-ubyte"
        data = read_image_file(os.path.join(self.raw_folder, image_file))

        label_file = f"{'train' if self.train else 't10k'}-labels-idx1-ubyte"
        targets = read_label_file(os.path.join(self.raw_folder, label_file))

        return data, targets

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        try:
            img = Image.fromarray(img.numpy(), mode="L").convert("RGB")
        except:
            pass

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class FashionMNIST(MNIST_RGB):
    """`Fashion-MNIST <https://github.com/zalandoresearch/fashion-mnist>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``FashionMNIST/raw/train-images-idx3-ubyte``
            and  ``FashionMNIST/raw/t10k-images-idx3-ubyte`` exist.
        train (bool, optional): If True, creates dataset from ``train-images-idx3-ubyte``,
            otherwise from ``t10k-images-idx3-ubyte``.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    mirrors = ["http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"]

    resources = [
        ("train-images-idx3-ubyte.gz", "8d4fb7e6c68d591d4c3dfef9ec88bf0d"),
        ("train-labels-idx1-ubyte.gz", "25c81989df183df01b3e8a0aad5dffbe"),
        ("t10k-images-idx3-ubyte.gz", "bef4ecab320f06d8554ea6380940ec79"),
        ("t10k-labels-idx1-ubyte.gz", "bb300cfdad3c16e7a12a480ee83cd310"),
    ]
    classes = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]


class NotMNIST(MNIST_RGB):
    def __init__(
        self, root, train=True, transform=None, target_transform=None, download=False
    ):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        self.url = "https://github.com/facebookresearch/Adversarial-Continual-Learning/raw/main/data/notMNIST.zip"
        self.filename = "notMNIST.zip"

        fpath = os.path.join(root, self.filename)
        if not os.path.isfile(fpath):
            if not download:
                raise RuntimeError(
                    "Dataset not found. You can use download=True to download it"
                )
            else:
                print("Downloading from " + self.url)
                download_url(self.url, root, filename=self.filename)

        import zipfile

        zip_ref = zipfile.ZipFile(fpath, "r")
        zip_ref.extractall(root)
        zip_ref.close()

        if self.train:
            fpath = os.path.join(root, "notMNIST", "Train")

        else:
            fpath = os.path.join(root, "notMNIST", "Test")

        X, Y = [], []
        folders = os.listdir(fpath)

        for folder in folders:
            folder_path = os.path.join(fpath, folder)
            for ims in os.listdir(folder_path):
                try:
                    img_path = os.path.join(folder_path, ims)
                    X.append(np.array(Image.open(img_path).convert("RGB")))
                    Y.append(ord(folder) - 65)  # Folders are A-J so labels will be 0-9
                except:
                    print("File {}/{} is broken".format(folder, ims))
        self.data = np.array(X)
        self.targets = Y

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        try:
            img = Image.fromarray(img)
        except:
            pass

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class SVHN(datasets.SVHN):
    def __init__(
        self, root, split="train", transform=None, target_transform=None, download=False
    ):
        super(SVHN, self).__init__(
            root,
            split=split,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )
        self.split = verify_str_arg(split, "split", tuple(self.split_list.keys()))
        self.url = self.split_list[split][0]
        self.filename = self.split_list[split][1]
        self.file_md5 = self.split_list[split][2]

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted. You can use download=True to download it"
            )

        # import here rather than at top of file because this is
        # an optional dependency for torchvision
        import scipy.io as sio

        # reading(loading) mat file as array
        loaded_mat = sio.loadmat(os.path.join(self.root, self.filename))

        self.data = loaded_mat["X"]
        # loading from the .mat file gives an np array of type np.uint8
        # converting to np.int64, so that we have a LongTensor after
        # the conversion from the numpy array
        # the squeeze is needed to obtain a 1D tensor
        self.targets = loaded_mat["y"].astype(np.int64).squeeze()

        # the svhn dataset assigns the class label "10" to the digit 0
        # this makes it inconsistent with several loss functions
        # which expect the class labels to be in the range [0, C-1]
        np.place(self.targets, self.targets == 10, 0)
        self.data = np.transpose(self.data, (3, 2, 0, 1))
        self.classes = np.unique(self.targets)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    def _check_integrity(self) -> bool:
        root = self.root
        md5 = self.split_list[self.split][2]
        fpath = os.path.join(root, self.filename)
        return check_integrity(fpath, md5)

    def download(self) -> None:
        md5 = self.split_list[self.split][2]
        download_url(self.url, self.root, self.filename, md5)

    def extra_repr(self) -> str:
        return "Split: {split}".format(**self.__dict__)


class Flowers102(datasets.Flowers102):
    def __init__(
        self, root, split="train", transform=None, target_transform=None, download=False
    ):
        super(Flowers102, self).__init__(
            root,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )
        self._split = verify_str_arg(split, "split", ("train", "val", "test"))
        self._base_folder = Path(self.root) / "flowers-102"
        self._images_folder = self._base_folder / "jpg"

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted. You can use download=True to download it"
            )

        from scipy.io import loadmat

        set_ids = loadmat(
            self._base_folder / self._file_dict["setid"][0], squeeze_me=True
        )
        image_ids = set_ids[self._splits_map[self._split]].tolist()

        labels = loadmat(
            self._base_folder / self._file_dict["label"][0], squeeze_me=True
        )
        image_id_to_label = dict(enumerate(labels["labels"].tolist(), 1))

        self.targets = []
        self._image_files = []
        for image_id in image_ids:
            self.targets.append(
                image_id_to_label[image_id] - 1
            )  # -1 for 0-based indexing
            self._image_files.append(self._images_folder / f"image_{image_id:05d}.jpg")
        self.classes = list(set(self.targets))

    def __len__(self) -> int:
        return len(self._image_files)

    def __getitem__(self, idx) -> Tuple[Any, Any]:
        image_file, label = self._image_files[idx], self.targets[idx]
        image = Image.open(image_file).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def extra_repr(self) -> str:
        return f"split={self._split}"

    def _check_integrity(self):
        if not (self._images_folder.exists() and self._images_folder.is_dir()):
            return False

        for id in ["label", "setid"]:
            filename, md5 = self._file_dict[id]
            if not check_integrity(str(self._base_folder / filename), md5):
                return False
        return True

    def download(self):
        if self._check_integrity():
            return
        download_and_extract_archive(
            f"{self._download_url_prefix}{self._file_dict['image'][0]}",
            str(self._base_folder),
            md5=self._file_dict["image"][1],
        )
        for id in ["label", "setid"]:
            filename, md5 = self._file_dict[id]
            download_url(
                self._download_url_prefix + filename, str(self._base_folder), md5=md5
            )


class StanfordCars(datasets.StanfordCars):
    def __init__(
        self, root, split="train", transform=None, target_transform=None, download=False
    ):
        try:
            import scipy.io as sio
        except ImportError:
            raise RuntimeError(
                "Scipy is not found. This dataset needs to have scipy installed: pip install scipy"
            )

        super(StanfordCars, self).__init__(
            root,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )

        self._split = verify_str_arg(split, "split", ("train", "test"))
        self._base_folder = pathlib.Path(root) / "stanford_cars"
        devkit = self._base_folder / "devkit"

        if self._split == "train":
            self._annotations_mat_path = devkit / "cars_train_annos.mat"
            self._images_base_path = self._base_folder / "cars_train"
        else:
            self._annotations_mat_path = (
                self._base_folder / "cars_test_annos_withlabels.mat"
            )
            self._images_base_path = self._base_folder / "cars_test"

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError(
                "Dataset not found. You can use download=True to download it"
            )

        self._samples = [
            (
                str(self._images_base_path / annotation["fname"]),
                annotation["class"]
                - 1,  # Original target mapping  starts from 1, hence -1
            )
            for annotation in sio.loadmat(self._annotations_mat_path, squeeze_me=True)[
                "annotations"
            ]
        ]

        self.classes = sio.loadmat(str(devkit / "cars_meta.mat"), squeeze_me=True)[
            "class_names"
        ].tolist()
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """Returns pil_image and class_id for given index"""
        image_path, target = self._samples[idx]
        pil_image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            pil_image = self.transform(pil_image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return pil_image, target

    def download(self) -> None:
        if self._check_exists():
            return

        download_and_extract_archive(
            url="https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz",
            download_root=str(self._base_folder),
            md5="c3b158d763b6e2245038c8ad08e45376",
        )
        if self._split == "train":
            download_and_extract_archive(
                url="https://ai.stanford.edu/~jkrause/car196/cars_train.tgz",
                download_root=str(self._base_folder),
                md5="065e5b463ae28d29e77c1b4b166cfe61",
            )
        else:
            download_and_extract_archive(
                url="https://ai.stanford.edu/~jkrause/car196/cars_test.tgz",
                download_root=str(self._base_folder),
                md5="4ce7ebf6a94d07f1952d94dd34c4d501",
            )
            download_url(
                url="https://ai.stanford.edu/~jkrause/car196/cars_test_annos_withlabels.mat",
                root=str(self._base_folder),
                md5="b0a2b23655a3edd16d84508592a98d10",
            )

    def _check_exists(self) -> bool:
        if not (self._base_folder / "devkit").is_dir():
            return False

        return self._annotations_mat_path.exists() and self._images_base_path.is_dir()


class CUB200(torch.utils.data.Dataset):
    def __init__(
        self, root, train=True, transform=None, target_transform=None, download=False
    ):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        self.url = "https://data.deepai.org/CUB200(2011).zip"
        self.filename = "CUB200(2011).zip"

        fpath = os.path.join(root, self.filename)
        if not os.path.isfile(fpath):
            if not download:
                raise RuntimeError(
                    "Dataset not found. You can use download=True to download it"
                )
            else:
                print("Downloading from " + self.url)
                download_url(self.url, root, filename=self.filename)

        if not os.path.exists(os.path.join(root, "CUB_200_2011")):
            import zipfile

            zip_ref = zipfile.ZipFile(fpath, "r")
            zip_ref.extractall(root)
            zip_ref.close()

            import tarfile

            tar_ref = tarfile.open(os.path.join(root, "CUB_200_2011.tgz"), "r")
            tar_ref.extractall(root)
            tar_ref.close()

            self.split()

        if self.train:
            fpath = os.path.join(root, "CUB_200_2011", "train")

        else:
            fpath = os.path.join(root, "CUB_200_2011", "test")

        self.data = datasets.ImageFolder(fpath, transform=transform)

    def split(self):
        train_folder = self.root + "CUB_200_2011/train"
        test_folder = self.root + "CUB_200_2011/test"

        if os.path.exists(train_folder):
            rmtree(train_folder)
        if os.path.exists(test_folder):
            rmtree(test_folder)
        os.mkdir(train_folder)
        os.mkdir(test_folder)

        images = self.root + "CUB_200_2011/images.txt"
        train_test_split = self.root + "CUB_200_2011/train_test_split.txt"

        with open(images, "r") as image:
            with open(train_test_split, "r") as f:
                for line in f:
                    image_path = image.readline().split(" ")[-1]
                    image_path = image_path.replace("\n", "")
                    class_name = image_path.split("/")[0].split(" ")[-1]
                    src = self.root + "CUB_200_2011/images/" + image_path

                    if line.split(" ")[-1].replace("\n", "") == "1":
                        if not os.path.exists(train_folder + "/" + class_name):
                            os.mkdir(train_folder + "/" + class_name)
                        dst = train_folder + "/" + image_path
                    else:
                        if not os.path.exists(test_folder + "/" + class_name):
                            os.mkdir(test_folder + "/" + class_name)
                        dst = test_folder + "/" + image_path

                    move(src, dst)


class TinyImagenet(torch.utils.data.Dataset):
    def __init__(
        self, root, train=True, transform=None, target_transform=None, download=False
    ):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        self.url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
        self.filename = "tiny-imagenet-200.zip"

        fpath = os.path.join(root, self.filename)
        if not os.path.isfile(fpath):
            if not download:
                raise RuntimeError(
                    "Dataset not found. You can use download=True to download it"
                )
            else:
                print("Downloading from " + self.url)
                download_url(self.url, root, filename=self.filename)

        if not os.path.exists(os.path.join(root, "tiny-imagenet-200")):
            import zipfile

            zip_ref = zipfile.ZipFile(fpath, "r")
            zip_ref.extractall(os.path.join(root))
            zip_ref.close()

            self.split()

        if self.train:
            fpath = root + "tiny-imagenet-200/train"

        else:
            fpath = root + "tiny-imagenet-200/test"

        self.data = datasets.ImageFolder(fpath, transform=transform)

    def split(self):
        test_folder = self.root + "tiny-imagenet-200/test"

        if os.path.exists(test_folder):
            rmtree(test_folder)
        os.mkdir(test_folder)

        val_dict = {}
        with open(self.root + "tiny-imagenet-200/val/val_annotations.txt", "r") as f:
            for line in f.readlines():
                split_line = line.split("\t")
                val_dict[split_line[0]] = split_line[1]

        paths = glob.glob(self.root + "tiny-imagenet-200/val/images/*")
        for path in paths:
            if "\\" in path:
                path = path.replace("\\", "/")
            file = path.split("/")[-1]
            folder = val_dict[file]
            if not os.path.exists(test_folder + "/" + folder):
                os.mkdir(test_folder + "/" + folder)
                os.mkdir(test_folder + "/" + folder + "/images")

        for path in paths:
            if "\\" in path:
                path = path.replace("\\", "/")
            file = path.split("/")[-1]
            folder = val_dict[file]
            src = path
            dst = test_folder + "/" + folder + "/images/" + file
            move(src, dst)

        rmtree(self.root + "tiny-imagenet-200/val")


class Scene67(torch.utils.data.Dataset):
    def __init__(
        self, root, train=True, transform=None, target_transform=None, download=False
    ):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        image_url = (
            "http://groups.csail.mit.edu/vision/LabelMe/NewImages/indoorCVPR_09.tar"
        )
        train_annos_url = "http://web.mit.edu/torralba/www/TrainImages.txt"
        test_annos_url = "http://web.mit.edu/torralba/www/TestImages.txt"
        urls = [image_url, train_annos_url, test_annos_url]
        image_fname = "indoorCVPR_09.tar"
        self.train_annos_fname = "TrainImage.txt"
        self.test_annos_fname = "TestImage.txt"
        fnames = [image_fname, self.train_annos_fname, self.test_annos_fname]

        for url, fname in zip(urls, fnames):
            fpath = os.path.join(root, fname)
            if not os.path.isfile(fpath):
                if not download:
                    raise RuntimeError(
                        "Dataset not found. You can use download=True to download it"
                    )
                else:
                    print("Downloading from " + url)
                    download_url(url, root, filename=fname)
        if not os.path.exists(os.path.join(root, "Scene67")):
            import tarfile

            with tarfile.open(os.path.join(root, image_fname)) as tar:
                tar.extractall(os.path.join(root, "Scene67"))

            self.split()

        if self.train:
            fpath = os.path.join(root, "Scene67", "train")

        else:
            fpath = os.path.join(root, "Scene67", "test")

        self.data = datasets.ImageFolder(fpath, transform=transform)

    def split(self):
        if not os.path.exists(os.path.join(self.root, "Scene67", "train")):
            os.mkdir(os.path.join(self.root, "Scene67", "train"))
        if not os.path.exists(os.path.join(self.root, "Scene67", "test")):
            os.mkdir(os.path.join(self.root, "Scene67", "test"))

        train_annos_file = os.path.join(self.root, self.train_annos_fname)
        test_annos_file = os.path.join(self.root, self.test_annos_fname)

        with open(train_annos_file, "r") as f:
            for line in f.readlines():
                line = line.replace("\n", "")
                src = self.root + "Scene67/" + "Images/" + line
                dst = self.root + "Scene67/" + "train/" + line
                if not os.path.exists(
                    os.path.join(self.root, "Scene67", "train", line.split("/")[0])
                ):
                    os.mkdir(
                        os.path.join(self.root, "Scene67", "train", line.split("/")[0])
                    )
                move(src, dst)

        with open(test_annos_file, "r") as f:
            for line in f.readlines():
                line = line.replace("\n", "")
                src = self.root + "Scene67/" + "Images/" + line
                dst = self.root + "Scene67/" + "test/" + line
                if not os.path.exists(
                    os.path.join(self.root, "Scene67", "test", line.split("/")[0])
                ):
                    os.mkdir(
                        os.path.join(self.root, "Scene67", "test", line.split("/")[0])
                    )
                move(src, dst)


class DomainNet(torch.utils.data.Dataset):
    def __init__(
        self,
        root,
        download=False,
        train=True,
        transform=None,
    ):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.train = train

        self.download_shell = f"extra/domainnet_download.sh {self.root}"
        self.filename = "domainnet"
        fpath = os.path.join(root, self.filename)
        if not os.path.exists(fpath):
            if not download:
                raise RuntimeError(
                    "Dataset not found. You can use download=True to download it"
                )
            else:
                print("Downloading data " + self.download_shell)
                os.system(self.download_shell)

        self.root = os.path.join(root, self.filename)

        if self.train:
            data_config = yaml.load(
                open("extra/splits/domainnet_train.yaml", "r"), Loader=yaml.Loader
            )
        else:
            data_config = yaml.load(
                open("extra/splits/domainnet_test.yaml", "r"), Loader=yaml.Loader
            )
        self.data = np.asarray(data_config["data"])
        self.targets = np.asarray(data_config["targets"])
        self.classes = np.unique(self.targets)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class
        """
        img_path, target = self.data[index], self.targets[index]
        # print(img_path + "\n")
        img = jpg_image_to_array(os.path.join(self.root, img_path))

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        img = self.transform(img) if self.transform is not None else img

        return img, target


class Imagenet_R(torch.utils.data.Dataset):
    def __init__(
        self,
        root,
        download=False,
        train=True,
        transform=None,
    ):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.train = train

        self.url = "https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar"
        self.filename = "imagenet-r.tar"

        self.fpath = os.path.join(root, "imagenet-r")
        if not os.path.exists(self.fpath):
            if not download:
                raise RuntimeError(
                    "Dataset not found. You can use download=True to download it"
                )
            else:
                print("Downloading from " + self.url)
                download_url(self.url, root, filename=self.filename)

        if not os.path.exists(os.path.join(root, "imagenet-r")):
            import tarfile

            tar_ref = tarfile.open(os.path.join(root, self.filename), "r")
            tar_ref.extractall(root)
            tar_ref.close()

        self.root = os.path.join(root, "imagenet-r")

        if self.train:
            data_config = yaml.load(
                open("extra/splits/imagenet-r_train.yaml", "r"), Loader=yaml.Loader
            )
        else:
            data_config = yaml.load(
                open("extra/splits/imagenet-r_test.yaml", "r"), Loader=yaml.Loader
            )
        self.data = np.asarray(data_config["data"])
        self.targets = np.asarray(data_config["targets"])
        self.classes = np.unique(self.targets)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class
        """
        img_path, target = self.data[index], self.targets[index]
        img = jpg_image_to_array(os.path.join(self.root, img_path))

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        img = self.transform(img) if self.transform is not None else img
        return img, target


def jpg_image_to_array(image_path):
    """
    Loads JPEG image into 3D Numpy array of shape
    (width, height, channels)
    """
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        im_arr = np.fromstring(image.tobytes(), dtype=np.uint8)
        im_arr = im_arr.reshape((image.size[1], image.size[0], 3))
    return im_arr


def get_domain_dataset(root, train=True, transform=None, target_transform=None):
    root = os.path.expanduser(root)
    if train:
        root = os.path.join(root, "train")
    else:
        root = os.path.join(root, "val")
    return datasets.ImageFolder(
        root, transform=transform, target_transform=target_transform
    )


def CDDB_dataset(
    root, train=True, transform=None, target_transform=None, download=False
):
    root = os.path.expanduser(root)
    # NOTE: Only for CDDB-Hard
    domains = ["gaugan", "biggan", "wild", "whichfaceisreal", "san"]
    datasets = []
    if not os.path.exists(root) and download:
        os.makedirs(root)
        url = "https://drive.google.com/uc?id=1NgB8ytBMFBFwyXJQvdVT_yek1EaaEHrg"
        gdown.download(url, quiet=False)
        with zipfile.ZipFile("CDDB.tar.zip", "r") as zip_ref:
            zip_ref.extractall(root)
        os.remove("CDDB.tar.zip")
        tar_dir = os.path.join(root, "CDDB.tar")
        with tarfile.open(tar_dir, "r") as tar_ref:
            tar_ref.extractall(root)
        os.remove(tar_dir)

    for domain in domains:
        datasets.append(
            get_domain_dataset(
                os.path.join(root, domain), train, transform, target_transform
            )
        )
    return datasets


#!/usr/bin/env python
# -*- coding: utf-8 -*-

################################################################################
# Copyright (c) 2019. Vincenzo Lomonaco. All rights reserved.                  #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 23-07-2019                                                             #
# Author: Vincenzo Lomonaco                                                    #
# E-mail: vincenzo.lomonaco@unibo.it                                           #
# Website: vincenzolomonaco.com                                                #
################################################################################

""" Data Loader for the CORe50 Dataset """

# Python 2-3 compatible
# from __future__ import print_function
# from __future__ import division
# from __future__ import absolute_import

# other imports


class CORE50(object):
    """CORe50 Data Loader calss

    Args:
        root (string): Root directory of the dataset where ``core50_128x128``,
            ``paths.pkl``, ``LUP.pkl``, ``labels.pkl``, ``core50_imgs.npz``
            live. For example ``~/data/core50``.
        preload (string, optional): If True data is pre-loaded with look-up
            tables. RAM usage may be high.
        scenario (string, optional): One of the three scenarios of the CORe50
            benchmark ``ni``, ``nc``, ``nic``, `nicv2_79`,``nicv2_196`` and
             ``nicv2_391``.
        train (bool, optional): If True, creates the dataset from the training
            set, otherwise creates from test set.
        cumul (bool, optional): If True the cumulative scenario is assumed, the
            incremental scenario otherwise. Practically speaking ``cumul=True``
            means that for batch=i also batch=0,...i-1 will be added to the
            available training data.
        run (int, optional): One of the 10 runs (from 0 to 9) in which the
            training batch order is changed as in the official benchmark.
        start_batch (int, optional): One of the training incremental batches
            from 0 to max-batch - 1. Remember that for the ``ni``, ``nc`` and
            ``nic`` we have respectively 8, 9 and 79 incremental batches. If
            ``train=False`` this parameter will be ignored.
    """

    nbatch = {
        "ni": 8,
        "nc": 9,
        "nic": 79,
        "nicv2_79": 79,
        "nicv2_196": 196,
        "nicv2_391": 391,
    }

    def __init__(
        self, root="", preload=False, scenario="ni", cumul=False, run=0, start_batch=0
    ):
        """ " Initialize Object"""

        self.root = os.path.expanduser(root)
        self.preload = preload
        self.scenario = scenario
        self.cumul = cumul
        self.run = run
        self.batch = start_batch

        if preload:
            print("Loading data...")
            bin_path = os.path.join(root, "core50_imgs.bin")
            if os.path.exists(bin_path):
                with open(bin_path, "rb") as f:
                    self.x = np.fromfile(f, dtype=np.uint8).reshape(164866, 128, 128, 3)

            else:
                with open(os.path.join(root, "core50_imgs.npz"), "rb") as f:
                    npzfile = np.load(f)
                    self.x = npzfile["x"]
                    print("Writing bin for fast reloading...")
                    self.x.tofile(bin_path)

        # print("Loading paths...")
        with open(os.path.join(root, "paths.pkl"), "rb") as f:
            self.paths = pkl.load(f)

        # print("Loading LUP...")
        with open(os.path.join(root, "LUP.pkl"), "rb") as f:
            self.LUP = pkl.load(f)

        # print("Loading labels...")
        with open(os.path.join(root, "labels.pkl"), "rb") as f:
            self.labels = pkl.load(f)

    def __iter__(self):
        return self

    def __next__(self):
        """Next batch based on the object parameter which can be also changed
        from the previous iteration."""

        scen = self.scenario
        run = self.run
        batch = self.batch

        if self.batch == self.nbatch[scen]:
            raise StopIteration

        # Getting the right indexis
        if self.cumul:
            train_idx_list = []
            for i in range(self.batch + 1):
                train_idx_list += self.LUP[scen][run][i]
        else:
            train_idx_list = self.LUP[scen][run][batch]

        # loading data
        if self.preload:
            train_x = np.take(self.x, train_idx_list, axis=0).astype(np.float32)
        else:
            print("Loading data...")
            # Getting the actual paths
            train_paths = []
            for idx in train_idx_list:
                train_paths.append(os.path.join(self.root, self.paths[idx]))
            # loading imgs
            train_x = self.get_batch_from_paths(train_paths).astype(np.float32)

        # In either case we have already loaded the y
        if self.cumul:
            train_y = []
            for i in range(self.batch + 1):
                train_y += self.labels[scen][run][i]
        else:
            train_y = self.labels[scen][run][batch]

        train_y = np.asarray(train_y, dtype=np.float32)

        # Update state for next iter
        self.batch += 1

        return (train_x, train_y)

    def get_test_set(self):
        """Return the test set (the same for each inc. batch)."""

        scen = self.scenario
        run = self.run

        test_idx_list = self.LUP[scen][run][-1]

        if self.preload:
            test_x = np.take(self.x, test_idx_list, axis=0).astype(np.float32)
        else:
            # test paths
            test_paths = []
            for idx in test_idx_list:
                test_paths.append(os.path.join(self.root, self.paths[idx]))

            # test imgs
            test_x = self.get_batch_from_paths(test_paths).astype(np.float32)

        test_y = self.labels[scen][run][-1]
        test_y = np.asarray(test_y, dtype=np.float32)

        return test_x, test_y

    next = __next__  # python2.x compatibility.

    @staticmethod
    def get_batch_from_paths(
        paths, compress=False, snap_dir="", on_the_fly=True, verbose=False
    ):
        """Given a number of abs. paths it returns the numpy array
        of all the images."""

        # If we do not process data on the fly we check if the same train
        # filelist has been already processed and saved. If so, we load it
        # directly. In either case we end up returning x and y, as the full
        # training set and respective labels.
        num_imgs = len(paths)
        hexdigest = md5("".join(paths).encode("utf-8")).hexdigest()
        loaded = False
        x = None
        file_path = None

        if compress:
            file_path = snap_dir + hexdigest + ".npz"
            if os.path.exists(file_path) and not on_the_fly:
                loaded = True
                with open(file_path, "rb") as f:
                    npzfile = np.load(f)
                    x, y = npzfile["x"]
        else:
            x_file_path = snap_dir + hexdigest + "_x.bin"
            if os.path.exists(x_file_path) and not on_the_fly:
                loaded = True
                with open(x_file_path, "rb") as f:
                    x = np.fromfile(f, dtype=np.uint8).reshape(num_imgs, 128, 128, 3)

        # Here we actually load the images.
        if not loaded:
            # Pre-allocate numpy arrays
            x = np.zeros((num_imgs, 128, 128, 3), dtype=np.uint8)

            for i, path in enumerate(paths):
                if verbose:
                    print("\r" + path + " processed: " + str(i + 1), end="")
                x[i] = np.array(Image.open(path))

            if verbose:
                print()

            if not on_the_fly:
                # Then we save x
                if compress:
                    with open(file_path, "wb") as g:
                        np.savez_compressed(g, x=x)
                else:
                    x.tofile(snap_dir + hexdigest + "_x.bin")

        assert x is not None, "Problems loading data. x is None!"

        return x


class BatchDataset(torch.utils.data.Dataset):
    def __init__(self, data, targets, transform=None, target_transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform
        self.classes = np.unique(self.targets)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]

        if self.transform is not None:
            x = self.transform(x)

        if self.target_transform is not None:
            y = self.target_transform(y)

        return x, y

    def __len__(self):
        return len(self.data)


def CORe50(
    root,
    scenario="ni",
    train=True,
    transform=None,
    target_transform=None,
    download=False,
):
    root = os.path.expanduser(root)

    if not os.path.exists(root) or download:
        os.makedirs(root)
        print(" Downloading CORe50 dataset!!!!!!!!!! ")
        url = "http://bias.csr.unibo.it/maltoni/download/core50/core50_128x128.zip"
        response = requests.get(url)
        file_name = os.path.join(root, "core50_128x128.zip")
        if response.status_code == 200:
            # Open the file and write the content of the response
            with open(file_name, "wb") as f:
                f.write(response.content)
            print("File downloaded successfully.")
        else:
            print("Failed to download file.")
            sys.exit(1)
        if file_name.endswith(".zip"):
            extract_to = os.path.join(root, "core50_128x128")
            with zipfile.ZipFile(file_name, "r") as zip_ref:
                zip_ref.extractall(extract_to)

        url1 = "https://vlomonaco.github.io/core50/data/paths.pkl"
        url2 = "https://vlomonaco.github.io/core50/data/LUP.pkl"
        url3 = "https://vlomonaco.github.io/core50/data/labels.pkl"

        response = requests.get(url1)
        file_name = os.path.join(root, "paths.pkl")
        if response.status_code == 200:
            # Open the file and write the content of the response
            with open(file_name, "wb") as f:
                f.write(response.content)
            print("File downloaded successfully.")
        else:
            print("Failed to download file.")
            sys.exit(1)

        response = requests.get(url2)
        file_name = os.path.join(root, "LUP.pkl")
        if response.status_code == 200:
            # Open the file and write the content of the response
            with open(file_name, "wb") as f:
                f.write(response.content)
            print("File downloaded successfully.")
        else:
            print("Failed to download file.")
            sys.exit(1)

        response = requests.get(url3)
        file_name = os.path.join(root, "labels.pkl")
        if response.status_code == 200:
            # Open the file and write the content of the response
            with open(file_name, "wb") as f:
                f.write(response.content)
            print("File downloaded successfully.")
        else:
            print("Failed to download file.")
            sys.exit(1)
    root = os.path.join(root, "core50_128x128")
    dataset = CORE50(root, scenario=scenario)

    if not train:
        test_x, test_y = dataset.get_test_set()
        dataset = BatchDataset(test_x, test_y, transform, target_transform)
    else:
        datasets = []
        for train_x, train_y in dataset:
            datasets.append(BatchDataset(train_x, train_y, transform, target_transform))

        return datasets

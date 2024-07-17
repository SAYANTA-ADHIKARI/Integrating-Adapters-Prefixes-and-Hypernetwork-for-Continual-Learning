# AdaPrefix++: Integrating Adapters, Prefixes and Hypernetwork for Continual Learning Implementation

This folder contains PyTorch implementation of the two methods, **AdaPrefix** and **AdaPrefix++** proposed in our paper *AdaPrefix++: Integrating Adapters, Prefixes and Hypernetwork for Continual Learning*.
## Datasets

We provide experiments on four datasets. These datasets should be present in `/data/` folder, each with there respective `dataset_name`.:

1. [CIFAR100](https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz) : Downloads automatically while training
2. [ImageNet-R](https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar): Download using the link and add it to your dataset folder
3. [CDDB](https://drive.google.com/file/d/1NgB8ytBMFBFwyXJQvdVT_yek1EaaEHrg/view?usp=sharing): Download using the link and add it to your dataset folder
4. Five-Datasets: Downloads automatically while training
    * [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)
    * [SVHN](http://ufldl.stanford.edu/housenumbers/)
    * [MNIST](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)
    * [notMNIST](https://www.kaggle.com/datasets/lubaroli/notmnist)
    * [FashionMNIST](https://www.kaggle.com/datasets/zalando-research/fashionmnist)


## Models
We used four backbone models:
* ViT-Large
* ViT-Base
* DeiT-Small
* DeiT-Tiny

All the pretrained weights for these models are taken for `timm`.

## Environment

The system we used and tested in 

- Ubuntu 18.04.1 LTS
- NVIDIA Tesla V100
- Python 3.10.11

First, create an enviroment with all the particular requirements required. To perform this, you can use :
```
cd AdaPrefix_ECCV
conda env create -f environment.yml
conda activate adaprefix 
## Here the environment name is set to "adaprefix"
```
## Usage
Once the environment in created. Then, depending on whichever method you want to run:

```
cd AdaPrefix
```
OR
```
cd AdaPrefix++
```
Each of these folder, `AdaPrefix` or `AdaPrefix++` has a `/scripts/` folder.

To run training, calculate TIL inference, CIL inference and Forward Transfer you can run:

```
bash scripts/final_train.sh <logfile_name>
```
To change datasets or backbone model, change the `VAR` and `DATASET` in `scripts/final_train.sh`

To only run training along with TIL inference, you can run:
```
bash scripts/train.sh <logfile_name>
```




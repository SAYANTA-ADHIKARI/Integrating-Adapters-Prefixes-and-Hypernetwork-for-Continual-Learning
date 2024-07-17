#!/bin/bash

source /raid/srijith/miniconda3/bin/activate adaprefix
VAR="vit_base"
DATASET="cifar100"

mkdir -p ./logs/$DATASET/$VAR/
python main.py $DATASET --config ./configs/$T/$DATASET/$VAR.yml \
    > ./logs/$DATASET/$VAR/train.log
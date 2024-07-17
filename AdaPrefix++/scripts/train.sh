#!/bin/bash

source /raid/srijith/miniconda3/bin/activate adaprefix
VAR="vit_base"
DATASET="cifar100"

mkdir -p ./logs/$DATASET/$VAR/
python main.py $DATASET --config ./configs/$DATASET/$VAR.yml \
        --output_dir "./outputs/$DATASET/$VAR/pl45/" \
        > ./logs/$DATASET/$VAR/$1

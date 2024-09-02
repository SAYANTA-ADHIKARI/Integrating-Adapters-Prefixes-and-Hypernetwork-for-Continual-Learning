#!/bin/bash

source /raid/srijith/miniconda3/bin/activate adaprefix
VAR="vit_base"
DATASET="imagenet_r"
EXP="testing"

mkdir -p ./logs/$EXP/$DATASET/$VAR/
python final_main.py $DATASET \
        --config ./configs/$DATASET/$VAR.yml \
        --output_dir "./outputs/$EXP/$DATASET/$VAR/" \
        > ./logs/$EXP/$DATASET/$VAR/$1
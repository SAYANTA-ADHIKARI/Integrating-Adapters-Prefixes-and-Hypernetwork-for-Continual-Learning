#!/bin/bash

source /raid/srijith/miniconda3/bin/activate adaprefix
VAR="vit_base"
DATASET="cddb"

mkdir -p ./logs/$DATASET/$VAR/
python dil_inference.py $DATASET --config ./configs/$DATASET/$VAR.yml\
     --pretrained-weights ./weights/$DATASET/$VAR/model_weights_42.pth\
     > ./logs/$DATASET/$VAR/$1
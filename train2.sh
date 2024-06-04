#!/bin/bash

time python train2.py -g \
    --start-epoch 0 --nepoch 500 \
    --start-train 0 --ntrain 1800 \
    --start-val 1800 --nval 200 \
    --learning-rate=0.01 \
    --num-workers 4 \
    --pin-memory \
    --drop-last \
    --prefetch-factor 2 \
    --persistent-workers \
    --batch-size=1


#!/usr/bin/env bash

DEVICE="cuda:0"

# This script runs a single baseline configuration from Figure 2 for fewer epochs.
python grokking_experiments.py \
    --lr 0.0005 \
    --num_epochs 2000 \
    --log_frequency 500 \
    --device "$DEVICE" \
    --train_fraction 0.4 \
    --cross_entropy_dtype float16 \
    --adam_epsilon 1e-30
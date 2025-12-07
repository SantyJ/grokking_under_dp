#!/bin/bash

DEVICE="cuda:0"

# Paper-style stablemax reproduction run (float64, full batch).
python3 grokking_experiments.py \
    --device "$DEVICE" \
    --binary_operation="add_mod" \
    --num_epochs=20000 \
    --train_fraction=0.4 \
    --log_frequency=500 \
    --lr=0.01 \
    --optimizer="AdamW" \
    --beta2=0.999 \
    --adam_epsilon=1e-25 \
    --loss_function="stablemax" \
    --cross_entropy_dtype="float64" \
    --train_dtype="float64" \
    --full_batch

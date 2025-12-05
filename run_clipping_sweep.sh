#!/bin/bash

DEVICE="cuda:0"
MAX_GRAD_NORMS=(0.5 0.1 0.05)

for NORM in "${MAX_GRAD_NORMS[@]}"
do
    echo "Running clipping-only experiment with max_grad_norm=${NORM}"
    python3 -u clipping_only_experiments.py \
        --device "$DEVICE" \
        --binary_operation="add_mod" \
        --num_epochs=20000 \
        --train_fraction=0.4 \
        --log_frequency=500 \
        --lr=0.0005 \
        --batch_size=5107 \
        --cross_entropy_dtype=float16 \
        --adam_epsilon=1e-30 \
        --use_clipping \
        --max_grad_norm="$NORM"
done


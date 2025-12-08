#!/bin/bash

DEVICE="cuda:0"

python3 -u baseline_minibatch_experiments.py \
    --device "$DEVICE" \
    --binary_operation="add_mod" \
    --num_epochs=20000 \
    --train_fraction=0.6 \
    --log_frequency=2000 \
    --lr=0.005 \
    --batch_size=256 \
    --cross_entropy_dtype="float64" \
    --train_dtype="float32" \
    --beta2=0.999 \
    --adam_epsilon=1e-8 \
    --weight_decay=0.0005

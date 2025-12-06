#!/bin/bash

DEVICE="cuda:0"
EPSILONS=(0.5 1.0 5.0 10.0 20.0)

for EPS in "${EPSILONS[@]}"; do
    echo "Running DP experiment with epsilon=${EPS}"
    python3 dp_grokking_experiments.py \
        --device "$DEVICE" \
        --binary_operation="add_mod" \
        --num_epochs=20000 \
        --train_fraction=0.4 \
        --log_frequency=500 \
        --lr=5e-4 \
        --batch_size=256 \
        --cross_entropy_dtype="float32" \
        --adam_epsilon=1e-8 \
        --use_dp \
        --target_epsilon="$EPS" \
        --max_grad_norm=1.0
done

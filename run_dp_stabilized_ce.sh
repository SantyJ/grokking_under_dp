#!/bin/bash

DEVICE="cuda:0"

# Stabilized CE + DP: label smoothing, logit normalization, float64 loss, moderate batch.
python3 stabilized_ce_dp_experiments.py \
  --device "$DEVICE" \
  --binary_operation="add_mod" \
  --num_epochs=20000 \
  --train_fraction=0.4 \
  --log_frequency=2000 \
  --lr=0.005 \
  --batch_size=256 \
  --beta2=0.999 \
  --adam_epsilon=1e-8 \
  --alpha=0.5 \
  --label_smoothing=0.05 \
  --logit_normalize \
  --cross_entropy_dtype="float64" \
  --train_dtype="float32" \
  --weight_decay=0.0005 \
  --use_dp \
  --target_epsilon=8.0 \
  --max_grad_norm=1.0

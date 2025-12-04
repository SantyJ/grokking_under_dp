#!/bin/bash

DEVICE="cuda:0"

python3 grokking_experiments.py \
--device "$DEVICE" \
--binary_operation="add_mod" \

--num_epochs=10000 \
--train_fraction=0.4 \
--log_frequency=500 \
--lr=5e-4 \
--adam_epsilon=1e-30 \
--cross_entropy_dtype="float16"
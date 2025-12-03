#!/usr/bin/env bash

# Get the directory of the script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

DEVICE="cuda:0"

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --device)
            DEVICE="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter passed: $1"
            exit 1
            ;;
    esac
done

# This is an example of how to run the DP experiment.
# It is based on the first experiment for Figure 2 in run_main_experiments.sh.
# You can add more experiments here, following the same pattern.

python "$SCRIPT_DIR/dp_grokking_experiments.py" \
    --lr 0.0005 \
    --num_epochs 20000 \
    --log_frequency 500 \
    --device "$DEVICE" \
    --train_fraction 0.4 \
    --cross_entropy_dtype float16 \
    --adam_epsilon 1e-30 \
    --use_dp \
    --target_epsilon 1.0 \
    --max_grad_norm 1.0


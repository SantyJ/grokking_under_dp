"""Generates comparison plots for short baseline vs. DP experiments."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

def format_func(value, tick_number):
    """Formats axis ticks to be more readable."""
    return str(int(value))

def get_metric_values(df, input_type, metric_name):
    """Extracts metric values from a dataframe for a given input_type and metric_name."""
    if df.empty:
        return np.array([])
        
    if input_type not in df["input_type"].unique():
        raise ValueError(f"'{input_type}' is not a valid input_type. "
                         f"Available: {df['input_type'].unique().tolist()}")
    
    # Sorting by epoch to ensure correct order before getting values
    df_sorted = df.sort_values(by="epoch")
    metric_values = df_sorted[(df_sorted["input_type"] == input_type) &
                              (df_sorted["metric_name"] == metric_name)]["value"]
    return metric_values.values

def read_metrics(experiment_key: str) -> pd.DataFrame:
    """Reads metrics from a CSV file for a given experiment key."""
    try:
        path = f"loggs/{experiment_key}/metrics.csv"
        return pd.read_csv(path)
    except FileNotFoundError:
        print(f"Error: The directory for experiment '{experiment_key}' was not found.")
        print(f"Attempted to read from: {path}")
        print("Please make sure you have run the short experiments first.")
        return None
    except pd.errors.EmptyDataError:
        print(f"Warning: Metrics file for experiment '{experiment_key}' is empty.")
        return pd.DataFrame() # Return empty dataframe

def get_model_norm(results):
    """Calculates the L2 norm of the model weights."""
    return results[results["metric_name"] == "weights_l2"] \
        .groupby("epoch")["value"] \
        .apply(lambda x: np.sqrt((x**2).sum())).values

def generate_plots():
    """Generates and saves the comparison plot."""
    baseline_experiment_key = "add_mod|num_epochs-2000|train_fraction-0.4|log_frequency-500|lr-0.0005|batch_size-5107|cross_entropy_dtype-float16|adam_epsilon-1e-30"
    dp_experiment_key = "add_mod|num_epochs-2000|train_fraction-0.4|log_frequency-500|lr-0.0005|cross_entropy_dtype-float16|adam_epsilon-1e-30|use_dp-True|target_epsilon-1.0|target_delta-0.00019580967299784609"

    baseline_metrics = read_metrics(baseline_experiment_key)
    dp_metrics = read_metrics(dp_experiment_key)

    if baseline_metrics is None or baseline_metrics.empty:
        print("Error: Baseline metrics are missing or empty. Cannot generate plot.")
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    
    epochs = sorted(baseline_metrics["epoch"].unique())
    baseline_train_accuracy = get_metric_values(baseline_metrics, "train", "accuracy")
    baseline_test_accuracy = get_metric_values(baseline_metrics, "test", "accuracy")
    ax.plot(epochs, baseline_train_accuracy, linewidth=5, label='Baseline Training accuracy', color='blue', linestyle='--')
    ax.plot(epochs, baseline_test_accuracy, linewidth=5, label='Baseline Test accuracy', color='blue')

    if dp_metrics is not None and not dp_metrics.empty:
        dp_epochs = sorted(dp_metrics["epoch"].unique())
        if not dp_epochs:
             print("Warning: DP metrics file contains no data rows. Skipping DP plots.")
        else:
            if epochs != dp_epochs:
                print("Warning: Baseline and DP experiments have different epochs. Plot may be misleading.")

            dp_train_accuracy = get_metric_values(dp_metrics, "train", "accuracy")
            dp_test_accuracy = get_metric_values(dp_metrics, "test", "accuracy")
            ax.plot(dp_epochs, dp_train_accuracy, linewidth=5, label='DP Training accuracy', color='red', linestyle='--')
            ax.plot(dp_epochs, dp_test_accuracy, linewidth=5, label='DP Test accuracy', color='red')
    else:
        print("Warning: DP metrics file is missing or empty. The corresponding experiment may have failed. Skipping DP plots.")

    ax.set_xlabel('Epoch', fontsize=18)
    ax.set_ylabel('Accuracy (%)', fontsize=18)
    ax.grid(alpha=0.5)
    ax.legend(fontsize=14, loc="lower right")
    ax.xaxis.set_major_formatter(FuncFormatter(format_func))
    ax.set_xticks(epochs[::2])
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    output_filename = "short_comparison_plot.png"
    plt.savefig(output_filename)
    print(f"Plot saved to {output_filename}")

if __name__ == "__main__":
    generate_plots()
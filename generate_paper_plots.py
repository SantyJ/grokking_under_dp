import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

def get_metric_values(df, input_type, metric_name):
    """Extracts metric values from a dataframe for a given input_type and metric_name."""
    if df.empty:
        return np.array([])
        
    if input_type not in df["input_type"].unique():
        # return empty array if input_type is not found
        return np.array([])

    # Sorting by epoch to ensure correct order before getting values
    df_sorted = df.sort_values(by="epoch")
    metric_values = df_sorted[(df_sorted["input_type"] == input_type) &
                              (df_sorted["metric_name"] == metric_name)]["value"]
    return metric_values.values

def get_total_l2_norm(df):
    """Calculates the total L2 norm of the model weights."""
    if df.empty:
        return np.array([])
    l2_norms = df[df["metric_name"] == "weights_l2"]
    if l2_norms.empty:
        return np.array([])
    return l2_norms.groupby("epoch")["value"].apply(lambda x: np.sqrt(np.sum(x**2)))

def get_avg_cosine_similarity(df):
    """Calculates the average cosine similarity."""
    if df.empty:
        return np.array([])
    cosine_sim = df[df["metric_name"] == "grad_cosine_similarity"]
    if cosine_sim.empty:
        return np.array([])
    return cosine_sim.groupby("epoch")["value"].mean()

def get_total_zero_grad_percentage(df):
    """Calculates the total percentage of zero gradients."""
    if df.empty:
        return np.array([])
    zero_grads = df[df["metric_name"] == "zero_grad_percentage"]
    if zero_grads.empty:
        return np.array([])
    return zero_grads.groupby("epoch")["value"].mean()


def generate_plots():
    """Generates and saves the comparison plot."""
    baseline_experiment_key = "add_mod|num_epochs-10000|train_fraction-0.4|log_frequency-500|lr-0.0005|batch_size-5107|cross_entropy_dtype-float16|adam_epsilon-1e-30"
    dp_experiment_key = "add_mod|num_epochs-10000|train_fraction-0.4|log_frequency-500|lr-0.0005|cross_entropy_dtype-float16|adam_epsilon-1e-30|use_dp-True|target_epsilon-1.0|target_delta-0.00019580967299784609"

    try:
        baseline_metrics = pd.read_csv(f"loggs/{baseline_experiment_key}/metrics.csv")
        dp_metrics = pd.read_csv(f"loggs/{dp_experiment_key}/metrics.csv")
    except FileNotFoundError as e:
        print(f"Error reading metrics file: {e}")
        return

    fig = plt.figure(figsize=(20, 15))
    gs = gridspec.GridSpec(2, 2, figure=fig)

    # Plot 1: Accuracy and Softmax Collapse
    ax1 = fig.add_subplot(gs[0, 0])
    epochs = sorted(baseline_metrics["epoch"].unique())
    
    baseline_train_acc = get_metric_values(baseline_metrics, "train", "accuracy")
    baseline_test_acc = get_metric_values(baseline_metrics, "test", "accuracy")
    baseline_sc = get_metric_values(baseline_metrics, "train", "softmax_collapse")
    
    dp_train_acc = get_metric_values(dp_metrics, "train", "accuracy")
    dp_test_acc = get_metric_values(dp_metrics, "test", "accuracy")
    dp_sc = get_metric_values(dp_metrics, "train", "softmax_collapse")

    ax1.plot(epochs, baseline_train_acc, label="Baseline Train Acc", color="blue", linestyle="--")
    ax1.plot(epochs, baseline_test_acc, label="Baseline Test Acc", color="blue")
    ax1.plot(epochs, baseline_sc, label="Baseline Softmax Collapse", color="cyan", linestyle=":")
    
    ax1.plot(epochs, dp_train_acc, label="DP Train Acc", color="red", linestyle="--")
    ax1.plot(epochs, dp_test_acc, label="DP Test Acc", color="red")
    ax1.plot(epochs, dp_sc, label="DP Softmax Collapse", color="magenta", linestyle=":")
    
    ax1.set_title("Accuracy and Softmax Collapse")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Value")
    ax1.legend()

    # Plot 2: L2 Norm of Weights
    ax2 = fig.add_subplot(gs[0, 1])
    baseline_l2 = get_total_l2_norm(baseline_metrics)
    dp_l2 = get_total_l2_norm(dp_metrics)
    
    ax2.plot(epochs, baseline_l2, label="Baseline L2 Norm")
    ax2.plot(epochs, dp_l2, label="DP L2 Norm")
    ax2.set_title("L2 Norm of Weights")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("L2 Norm")
    ax2.legend()

    # Plot 3: Cosine Similarity
    ax3 = fig.add_subplot(gs[1, 0])
    baseline_cos_sim = get_avg_cosine_similarity(baseline_metrics)
    dp_cos_sim = get_avg_cosine_similarity(dp_metrics)

    ax3.plot(epochs, baseline_cos_sim, label="Baseline Cosine Similarity")
    ax3.plot(epochs, dp_cos_sim, label="DP Cosine Similarity")
    ax3.set_title("Cosine Similarity (Weights vs. Gradients)")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Cosine Similarity")
    ax3.legend()

    # Plot 4: Zero Gradient Percentage
    ax4 = fig.add_subplot(gs[1, 1])
    baseline_zero_grad = get_total_zero_grad_percentage(baseline_metrics)
    dp_zero_grad = get_total_zero_grad_percentage(dp_metrics)
    
    ax4.plot(epochs, baseline_zero_grad, label="Baseline Zero Grad %")
    ax4.plot(epochs, dp_zero_grad, label="DP Zero Grad %")
    ax4.set_title("Percentage of Zero Gradients")
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Percentage")
    ax4.legend()

    plt.tight_layout()
    plt.savefig("paper_plots.png")
    print("Plots saved to paper_plots.png")

if __name__ == "__main__":
    generate_plots()

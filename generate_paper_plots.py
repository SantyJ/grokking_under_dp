import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

def get_metric_values(df, input_type, metric_name):
    """Extracts metric values from a dataframe for a given input_type and metric_name."""
    if df.empty:
        return np.array([]), np.array([])
        
    if input_type not in df["input_type"].unique():
        # return empty array if input_type is not found
        return np.array([]), np.array([])

    # Sorting by epoch to ensure correct order before getting values
    subset_df = df[(df["input_type"] == input_type) & (df["metric_name"] == metric_name)].sort_values(by="epoch")
    return subset_df["epoch"].values, subset_df["value"].values

def get_total_l2_norm(df):
    """Calculates the total L2 norm of the model weights."""
    if df.empty:
        return np.array([]), np.array([])
    l2_norms = df[df["metric_name"] == "weights_l2"]
    if l2_norms.empty:
        return np.array([]), np.array([])
    
    grouped_norms = l2_norms.groupby("epoch")["value"].apply(lambda x: np.sqrt(np.sum(x**2)))
    return grouped_norms.index.values, grouped_norms.values

def get_avg_cosine_similarity(df):
    """Calculates the average cosine similarity."""
    if df.empty:
        return np.array([]), np.array([])
    cosine_sim = df[df["metric_name"] == "grad_cosine_similarity"]
    if cosine_sim.empty:
        return np.array([]), np.array([])
    
    grouped_sim = cosine_sim.groupby("epoch")["value"].mean()
    return grouped_sim.index.values, grouped_sim.values

def get_total_zero_grad_percentage(df):
    """Calculates the total percentage of zero gradients."""
    if df.empty:
        return np.array([]), np.array([])
    zero_grads = df[df["metric_name"] == "zero_grad_percentage"]
    if zero_grads.empty:
        return np.array([]), np.array([])
    
    grouped_grads = zero_grads.groupby("epoch")["value"].mean()
    return grouped_grads.index.values, grouped_grads.values


def generate_plots():
    """Generates and saves the comparison plot."""
    baseline_experiment_key = "add_mod|num_epochs-20000|train_fraction-0.4|log_frequency-500|lr-0.0005|batch_size-5107|cross_entropy_dtype-float16|adam_epsilon-1e-30"
    dp_experiment_key = "add_mod|num_epochs-20000|train_fraction-0.4|log_frequency-500|lr-0.0005|batch_size-5107|cross_entropy_dtype-float16|adam_epsilon-1e-30|use_clipping-True"
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
    
    baseline_train_acc_epochs, baseline_train_acc_values = get_metric_values(baseline_metrics, "train", "accuracy")
    baseline_test_acc_epochs, baseline_test_acc_values = get_metric_values(baseline_metrics, "test", "accuracy")
    baseline_sc_epochs, baseline_sc_values = get_metric_values(baseline_metrics, "train", "softmax_collapse")
    
    dp_train_acc_epochs, dp_train_acc_values = get_metric_values(dp_metrics, "train", "accuracy")
    dp_test_acc_epochs, dp_test_acc_values = get_metric_values(dp_metrics, "test", "accuracy")
    dp_sc_epochs, dp_sc_values = get_metric_values(dp_metrics, "train", "softmax_collapse")

    ax1.plot(baseline_train_acc_epochs, baseline_train_acc_values, label="Baseline Train Acc", color="blue", linestyle="--")
    ax1.plot(baseline_test_acc_epochs, baseline_test_acc_values, label="Baseline Test Acc", color="blue")
    ax1.plot(baseline_sc_epochs, baseline_sc_values, label="Baseline Softmax Collapse", color="cyan", linestyle=":")
    
    ax1.plot(dp_train_acc_epochs, dp_train_acc_values, label="DP Train Acc", color="red", linestyle="--")
    ax1.plot(dp_test_acc_epochs, dp_test_acc_values, label="DP Test Acc", color="red")
    ax1.plot(dp_sc_epochs, dp_sc_values, label="DP Softmax Collapse", color="magenta", linestyle=":")
    
    ax1.set_title("Accuracy and Softmax Collapse")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Value")
    ax1.legend()

    # Plot 2: L2 Norm of Weights
    ax2 = fig.add_subplot(gs[0, 1])
    baseline_l2_epochs, baseline_l2_values = get_total_l2_norm(baseline_metrics)
    dp_l2_epochs, dp_l2_values = get_total_l2_norm(dp_metrics)
    
    ax2.plot(baseline_l2_epochs, baseline_l2_values, label="Baseline L2 Norm")
    ax2.plot(dp_l2_epochs, dp_l2_values, label="DP L2 Norm")
    ax2.set_title("L2 Norm of Weights")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("L2 Norm")
    ax2.legend()

    # Plot 3: Cosine Similarity
    ax3 = fig.add_subplot(gs[1, 0])
    baseline_cos_sim_epochs, baseline_cos_sim_values = get_avg_cosine_similarity(baseline_metrics)
    dp_cos_sim_epochs, dp_cos_sim_values = get_avg_cosine_similarity(dp_metrics)

    ax3.plot(baseline_cos_sim_epochs, baseline_cos_sim_values, label="Baseline Cosine Similarity")
    ax3.plot(dp_cos_sim_epochs, dp_cos_sim_values, label="DP Cosine Similarity")
    ax3.set_title("Cosine Similarity (Weights vs. Gradients)")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Cosine Similarity")
    ax3.legend()

    # Plot 4: Zero Gradient Percentage
    ax4 = fig.add_subplot(gs[1, 1])
    baseline_zero_grad_epochs, baseline_zero_grad_values = get_total_zero_grad_percentage(baseline_metrics)
    dp_zero_grad_epochs, dp_zero_grad_values = get_total_zero_grad_percentage(dp_metrics)
    
    ax4.plot(baseline_zero_grad_epochs, baseline_zero_grad_values, label="Baseline Zero Grad %")
    ax4.plot(dp_zero_grad_epochs, dp_zero_grad_values, label="DP Zero Grad %")
    ax4.set_title("Percentage of Zero Gradients")
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Percentage")
    ax4.legend()

    plt.tight_layout()
    plt.savefig("paper_plots.png")
    print("Plots saved to paper_plots.png")

if __name__ == "__main__":
    generate_plots()

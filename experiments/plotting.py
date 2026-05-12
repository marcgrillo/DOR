import numpy as np
import matplotlib.pyplot as plt
import os
import common.utils as utils

def plot_metric_results(metric_name, F1, F2, NUM_STEPS_LIST, hm, dataset_fold=None, sub_fold=None, drop_index=None, save_figs=False, show_figs=True, fig_name_prefix="figure"):
    """
    Generic plotting function for ASRS, ASPS, or AIOS results.
    """
    metric_name = metric_name.lower()
    valid_metrics = ['asrs', 'asps', 'aios', 'perc_inc']
    if metric_name not in valid_metrics:
        raise ValueError(f"metric_name must be one of {valid_metrics}")

    # Configuration
    f1, f2, num_steps = F1[0], F2[0], NUM_STEPS_LIST[0]
    x = np.arange(1, num_steps + 1)

    # Load Data
    y = utils.load_test_results(metric_name, dataset_fold, sub_fold, f1, f2, num_steps)
    
    if y.size == 0:
        print(f"No data found for {metric_name} in {dataset_fold}/{sub_fold}")
        return

    if drop_index is not None:
        y = np.delete(y, drop_index, axis=1)

    # Statistics
    n_samples = y.shape[1] 
    mean = np.mean(y, axis=1)
    std = np.std(y, axis=1) * 2 / np.sqrt(n_samples) # 95% CI
    
    save_dir = os.path.join("figs", dataset_fold, metric_name) if dataset_fold else os.path.join("figs", metric_name)
    if save_figs:
        os.makedirs(save_dir, exist_ok=True)

    # --- Plot 1: Evolution ---
    plt.figure(figsize=(10, 6))
    plt.plot(x, mean, label=sub_fold, color="blue")
    plt.fill_between(x, mean - std, mean + std, color="blue", alpha=0.3, label="95% CI")

    plt.xlabel("Number of Elicitation Steps", fontsize=12)
    plt.ylabel(metric_name.upper(), fontsize=12)
    plt.title(f"{sub_fold} Evolution (N={n_samples})", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    if save_figs:
        plt.savefig(os.path.join(save_dir, f"{sub_fold}_mean_std.png"), dpi=300)
    
    if show_figs:
        plt.show()
    plt.close()
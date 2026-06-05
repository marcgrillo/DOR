import numpy as np
import matplotlib.pyplot as plt
import os
import re

# 1. Load Data
npz_file = "grid_results_full.npz"
if not os.path.exists(npz_file):
    raise FileNotFoundError(f"{npz_file} not found.")

print("Loading dataset...")
data = np.load(npz_file, allow_pickle=True)['results'].item()

# 2. Parse Keys and Build Structure
pattern = re.compile(r"F1=(\d+)_F2=(\d+)_S=(\d+)_AR=(\d+)_K=(\d+)_Inc=(\d+)")

records = []
for key, metrics_data in data.items():
    match = pattern.match(key)
    if not match:
        continue
    f1, f2, s, ar, k, inc = map(int, match.groups())
    
    ftrl_data = metrics_data.get('FTRL_DOR_PASSIVE', {})
    bayes_data = metrics_data.get('BAYES_DOR_PASSIVE', {})
    
    records.append({
        'F1': f1, 'F2': f2, 'S': s, 'AR': ar, 'K': k, 'Inc': inc,
        'FTRL': ftrl_data,
        'BAYES': bayes_data
    })

# Desired steps (1-indexed, so indices 0, 2, 4, 9)
step_indices = {1: 0, 3: 2, 5: 4, 10: 9}
metrics_to_plot = ['asrs', 'asps', 'aios']

output_dir = os.path.join("figs", "grid_summary")
os.makedirs(output_dir, exist_ok=True)

# Helper function to get mean and 95% CI across records
def aggregate_data(filter_fn, method, metric, step_idx):
    all_values = []
    for rec in records:
        if filter_fn(rec):
            arr = np.array(rec[method].get(metric, []))
            if arr.ndim == 2 and arr.shape[0] > step_idx:
                all_values.extend(arr[step_idx, :])
                
    if not all_values:
        return np.nan, np.nan
        
    all_values = np.array(all_values)
    mean = np.mean(all_values)
    ci = 1.96 * np.std(all_values) / np.sqrt(len(all_values))
    return mean, ci

inc_levels = sorted(list(set([r['Inc'] for r in records])))
T_levels = [1, 3, 5, 10]
F1_levels = sorted(list(set([r['F1'] for r in records])))
K_levels = sorted(list(set([r['K'] for r in records])))

print(f"Generating plots for metrics: {metrics_to_plot}")

methods = ['FTRL', 'BAYES']

for metric in metrics_to_plot:
    print(f"Processing {metric.upper()}...")
    
    # ---------------------------------------------------------
    # Plot 1: Evolution over Time (T) across Inconsistency levels
    # ---------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for idx_ax, method in enumerate(methods):
        ax = axes[idx_ax]
        for inc in inc_levels:
            means = []
            cis = []
            for t in T_levels:
                idx = step_indices[t]
                m, ci = aggregate_data(lambda r: r['Inc'] == inc, method, metric, idx)
                means.append(m)
                cis.append(ci)
            means = np.array(means)
            cis = np.array(cis)
            
            ax.plot(T_levels, means, marker='o', label=f'Inconsistency: {inc}%')
            ax.fill_between(T_levels, means - cis, means + cis, alpha=0.2)
            
        ax.set_title(f"{method} {metric.upper()}: Evolution over Time")
        ax.set_xlabel("Number of Interactions (T)")
        if idx_ax == 0:
            ax.set_ylabel(f"Average {metric.upper()}")
        ax.set_xticks(T_levels)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{metric}_evolution_over_time.png"), dpi=300)
    plt.close()
    
    # ---------------------------------------------------------
    # Plot 2: Performance vs F1 at fixed T across Inconsistency
    # ---------------------------------------------------------
    for t_val in [1, 3, 10]:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
        t_idx = step_indices[t_val]
        for idx_ax, method in enumerate(methods):
            ax = axes[idx_ax]
            for inc in inc_levels:
                means = []
                cis = []
                for f1 in F1_levels:
                    m, ci = aggregate_data(lambda r: r['Inc'] == inc and r['F1'] == f1, method, metric, t_idx)
                    means.append(m)
                    cis.append(ci)
                means = np.array(means)
                cis = np.array(cis)
                
                ax.plot(F1_levels, means, marker='s', label=f'Inconsistency: {inc}%')
                ax.fill_between(F1_levels, means - cis, means + cis, alpha=0.2)
                
            ax.set_title(f"{method} {metric.upper()}: Scalability (at T={t_val})")
            ax.set_xlabel("Number of Alternatives (|A|)")
            if idx_ax == 0:
                ax.set_ylabel(f"Average {metric.upper()}")
            ax.set_xticks(F1_levels)
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.6)
            
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{metric}_scalability_F1_T{t_val}.png"), dpi=300)
        # Keep legacy name for T=10 compatibility
        if t_val == 10:
            plt.savefig(os.path.join(output_dir, f"{metric}_scalability_F1.png"), dpi=300)
        plt.close()
        
    # ---------------------------------------------------------
    # Plot 3: Cognitive Load (K) at fixed T across Inconsistency
    # ---------------------------------------------------------
    for t_val in [1, 3, 10]:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
        bar_width = 0.2
        x = np.arange(len(K_levels))
        t_idx = step_indices[t_val]
        
        for idx_ax, method in enumerate(methods):
            ax = axes[idx_ax]
            for i, inc in enumerate(inc_levels):
                means = []
                cis = []
                for k in K_levels:
                    m, ci = aggregate_data(lambda r: r['Inc'] == inc and r['K'] == k, method, metric, t_idx)
                    means.append(m)
                    cis.append(ci)
                means = np.array(means)
                cis = np.array(cis)
                
                pos = x + (i - len(inc_levels)/2 + 0.5) * bar_width
                ax.bar(pos, means, width=bar_width, label=f'Inconsistency: {inc}%', yerr=cis, capsize=4)
                
            ax.set_title(f"{method} {metric.upper()}: Impact of Cognitive Load (at T={t_val})")
            ax.set_xlabel("Items Ranked per Step (k)")
            if idx_ax == 0:
                ax.set_ylabel(f"Average {metric.upper()}")
            ax.set_xticks(x)
            ax.set_xticklabels(K_levels)
            ax.legend()
            ax.grid(True, axis='y', linestyle='--', alpha=0.6)
            
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{metric}_cognitive_load_K_T{t_val}.png"), dpi=300)
        # Keep legacy name for T=10 compatibility
        if t_val == 10:
            plt.savefig(os.path.join(output_dir, f"{metric}_cognitive_load_K.png"), dpi=300)
        plt.close()

print(f"All plots saved to {output_dir}")

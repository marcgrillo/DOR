import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to sys.path to allow imports from sibling packages
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mcda.models import PiecewiseLinearTransformer
from inference.engine import PreferenceSampler

def plot_marginal_utilities(transformer, w_map, bayes_samples, step_idx, output_dir):
    """
    Plots marginal utility functions. 
    w_map: weights from FTRL.
    bayes_samples: weights from BAYES (Nested Sampling).
    """
    f2 = transformer.num_criteria
    x_plot = np.linspace(0, 1, 100)
    
    fig, axes = plt.subplots(1, f2, figsize=(4 * f2, 4))
    if f2 == 1:
        axes = [axes]
    
    # Process column ranges for weights
    col_end = np.cumsum(transformer.n_w)
    col_start = np.insert(col_end[:-1], 0, 0)
    
    for j in range(f2):
        ax = axes[j]
        start_idx, end_idx = int(col_start[j]), int(col_end[j])
        breakpoints = transformer.ch_p[j]
        
        # 1. FTRL Line (MAP)
        w_j_map = w_map[start_idx:end_idx]
        y_map = []
        for x_val in x_plot:
            val = 0
            for k in range(len(w_j_map)):
                bp_low, bp_high = breakpoints[k], breakpoints[k+1]
                feat = np.clip((x_val - bp_low) / (bp_high - bp_low), 0, 1)
                val += w_j_map[k] * feat
            y_map.append(val)
        ax.plot(x_plot, y_map, 'r-', linewidth=2, label='FTRL (MAP)')
        
        # 2. BAYES Shaded Region
        if bayes_samples is not None:
            all_y_bayes = []
            # Take a subset of samples to speed up plotting if needed
            sample_indices = np.random.choice(len(bayes_samples), min(100, len(bayes_samples)), replace=False)
            for s_idx in sample_indices:
                sample_w = bayes_samples[s_idx]
                w_j_s = sample_w[start_idx:end_idx]
                y_s = []
                for x_val in x_plot:
                    val = 0
                    for k in range(len(w_j_s)):
                        bp_low, bp_high = breakpoints[k], breakpoints[k+1]
                        feat = np.clip((x_val - bp_low) / (bp_high - bp_low), 0, 1)
                        val += w_j_s[k] * feat
                    y_s.append(val)
                all_y_bayes.append(y_s)
            
            all_y_bayes = np.array(all_y_bayes)
            y_low = np.percentile(all_y_bayes, 5, axis=0)
            y_high = np.percentile(all_y_bayes, 95, axis=0)
            ax.fill_between(x_plot, y_low, y_high, color='blue', alpha=0.2, label='BAYES (90% CI)')
        
        ax.set_title(f"Criterion {j+1}")
        ax.set_xlabel("Normalized Attribute")
        ax.set_ylabel("Marginal Utility")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.1, None)
        if j == 0:
            ax.legend()

    plt.suptitle(f"Inferred Utility Functions - Step {step_idx}", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"utilities_step_{step_idx}.png"))
    plt.close()

def run_inference_tool(data_path):
    # Load Data
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    meta = data['metadata']
    alternatives = np.array(data['alternatives'])
    steps = data['steps']
    n_max = meta.get('n_max', 5)
    num_intervals = meta.get('num_intervals', 3)
    
    # Initialize Transformer
    transformer = PiecewiseLinearTransformer.from_equal_intervals(alternatives, num_intervals)
    X_transformed = transformer.transform(alternatives)
    n_params = X_transformed.shape[1]
    
    # Setup Sampler
    sampler = PreferenceSampler(X_transformed, [], n_params, n_max)
    
    output_dir = os.path.dirname(data_path)
    print(f"Starting User Inference Tool. Results will be saved in: {output_dir}")
    
    for i, step in enumerate(steps):
        step_idx = i + 1
        print(f"\n--- Processing Step {step_idx} ---")
        
        # Add Preferences
        items = step['items']
        cards = step['cards']
        
        # Expand subrank into pairwise
        for start_idx in range(len(items)):
            for end_idx in range(start_idx + 1, len(items)):
                p_win = items[start_idx]
                p_lose = items[end_idx]
                eta = sum(cards[start_idx : end_idx])
                sampler.add_preference(int(p_win), int(p_lose), int(eta))
        
        # Run FTRL (MAP)
        print("Running FTRL (MAP estimation)...")
        w_map_full = sampler.find_map()
        w_map = w_map_full[:n_params]
        
        # Run BAYES (Nested Sampling)
        print("Running BAYES (Posterior sampling)...")
        bayes_samples_full = sampler.run_nested(nlive=200)
        bayes_samples = bayes_samples_full[:, :n_params]
        
        # Plot
        print(f"Generating plot for Step {step_idx}...")
        plot_marginal_utilities(transformer, w_map, bayes_samples, step_idx, output_dir)

    print("\nInference complete. Check the directory for PNG files.")

if __name__ == "__main__":
    data_file = os.path.join(os.path.dirname(__file__), 'data.json')
    if not os.path.exists(data_file):
        print(f"Error: {data_file} not found.")
    else:
        run_inference_tool(data_file)

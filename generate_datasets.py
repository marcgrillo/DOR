import os
import json
import numpy as np
from scipy.stats import dirichlet, norm
from scipy.optimize import brentq
import matplotlib
matplotlib.use('Agg') # Non-interactive backend
import matplotlib.pyplot as plt
from tqdm import tqdm
from common import utils
from mcda.models import PiecewiseLinearTransformer

class DatasetGenerator:
    """
    Generates synthetic MCDA datasets with controllable Inconsistency using Subranks.
    Saves data in structured JSON format with clear descriptions.
    """

    def __init__(self, output_dir="datasets", n_runs=200):
        self.output_dir = output_dir
        self.n_runs = n_runs
        utils.save_path(self.output_dir)

    def _compute_rank(self, U):
        """Computes ranks (0 = best)."""
        b = np.argsort(-U)
        ranks = np.empty_like(b)
        ranks[b] = np.arange(len(U))
        return ranks.tolist()

    def _generate_exponential_utility(self, data_matrix, f2):
        """Generates ground truth utility using Exponential function."""
        w = dirichlet.rvs(alpha=np.ones(f2))[0]
        c = np.random.uniform(-10, 10, f2)
        c[np.abs(c) < 1e-5] = 1e-5 # Avoid c=0
            
        term1 = 1 - np.exp(-c * data_matrix)
        term2 = 1 - np.exp(-c)
        u_j = w * (term1 / term2)
        return np.sum(u_j, axis=1), w, c

    def _generate_piecewise_linear_utility(self, data_matrix, f2, num_intervals=3):
        """Generates ground truth utility using Piecewise Linear functions."""
        transformer = PiecewiseLinearTransformer.from_equal_intervals(data_matrix, num_intervals)
        features = transformer.transform(data_matrix)
        
        # Features dim: (N, total_params). total_params = f2 * num_intervals
        total_params = features.shape[1]
        
        # Weights
        w = dirichlet.rvs(alpha=np.ones(total_params))[0]
        
        # Utility U = X_transformed @ w
        U = features @ w
        return U, w, transformer

    def _plot_marginal_utilities(self, utility_type, f2, weights, params, save_path):
        """Plots the marginal utility functions for each criterion."""
        x_plot = np.linspace(0, 1, 100)
        
        fig, axes = plt.subplots(1, f2, figsize=(5 * f2, 4))
        if f2 == 1: axes = [axes]
        
        for j in range(f2):
            w_j = weights[j] if utility_type == 'exponential' else 1.0 # Weights are handled differently
            
            if utility_type == 'exponential':
                # exponential: u_j(x) = w_j * (1 - exp(-c_j * x)) / (1 - exp(-c_j))
                c_j = params[j]
                y_plot = w_j * (1 - np.exp(-c_j * x_plot)) / (1 - np.exp(-c_j))
                title = f"Crit {j+1} (Exp, w={w_j:.2f}, c={c_j:.2f})"
            elif utility_type == 'piecewise_linear':
                # piecewise: params is transformer
                transformer = params
                
                col_end = np.cumsum(transformer.n_w)
                col_start = np.insert(col_end[:-1], 0, 0)
                
                start_idx = int(col_start[j])
                end_idx = int(col_end[j])
                
                w_sub = weights[start_idx:end_idx]
                breakpoints = transformer.ch_p[j] # Should be [min, ..., max]
                
                x_domain = np.linspace(breakpoints[0], breakpoints[-1], 200)
                y_vals = []
                for x_val in x_domain:
                    val_j = 0
                    for k in range(len(w_sub)):
                        bp_low = breakpoints[k]
                        bp_high = breakpoints[k+1]
                        feat = np.clip((x_val - bp_low) / (bp_high - bp_low), 0, 1)
                        val_j += w_sub[k] * feat
                    y_vals.append(val_j)
                
                x_plot = x_domain
                y_plot = np.array(y_vals)
                title = f"Crit {j+1} (PWL, imp={np.sum(w_sub):.2f})"

            ax = axes[j]
            ax.plot(x_plot, y_plot, linewidth=2)
            ax.set_title(title)
            ax.set_xlabel("Attribute Value")
            ax.set_ylabel("Marginal Utility")
            ax.grid(True, alpha=0.3)
            
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def _calculate_expected_consistency(self, sigma, utility_diffs, n_max):
        """
        Calculates expected consistency for a pair using the Thurstone model.
        The difference between two noisy utilities follows N(u_diff * n_max, sigma^2).
        """
        if sigma <= 0:
            return 1.0
        
        mu = utility_diffs * n_max
        c = np.round(mu)
        # P(c - 0.5 <= v < c + 0.5) where v ~ N(mu, sigma^2)
        probs = norm.cdf((c + 0.5 - mu) / sigma) - norm.cdf((c - 0.5 - mu) / sigma)
        return np.mean(probs)

    def _tune_sigma(self, f1, f2, target_consistency, n_max, num_intervals=3, utility_type='exponential'):
        """Finds the sigma value that results in the desired consistency rate."""
        if target_consistency >= 0.999: return 0.0

        # Calibration set
        calibration_size = 1000
        diffs = []
        for _ in range(calibration_size):
            ls = np.random.rand(f1, f2)
            if utility_type == 'piecewise_linear':
                U, _, _ = self._generate_piecewise_linear_utility(ls, f2, num_intervals)
            else:
                U, _, _ = self._generate_exponential_utility(ls, f2)
            idx = np.arange(f1)
            np.random.shuffle(idx)
            u_shuffled = U[idx]
            d = np.abs(u_shuffled[:-1] - u_shuffled[1:])
            diffs.append(d)
        
        utility_diffs = np.concatenate(diffs)

        def objective(sigma):
            return self._calculate_expected_consistency(sigma, utility_diffs, n_max) - target_consistency

        try:
            return brentq(objective, 0.001, 100.0 * n_max)
        except ValueError:
            return 1.0 * n_max # fallback

    def generate_batch(self, F1, F2, num_steps, k_subset, n_max, target_consistency=1.0, utility_type='exponential', num_intervals=3, plot_utilities=True):
        print(f"Generating datasets in '{self.output_dir}' with Consistency={target_consistency} (Utility={utility_type})...")
        
        for f1 in tqdm(F1, desc="Alternatives (F1)"):
            for f2 in F2:
                # --- Step 1: Auto-tune Sigma ---
                sigma = self._tune_sigma(f1, f2, target_consistency, n_max, num_intervals=num_intervals, utility_type=utility_type)
                
                dataset_json = {
                    "_description": {
                        "metadata": "Parameters used to generate this dataset.",
                        "runs": "List of independent runs (Human Models).",
                        "runs[].table": "Feature matrix for the alternatives (f1 x f2).",
                        "runs[].true_ranks": "Ground truth ranks of the alternatives (0 is best).",
                        "runs[].true_utilities": "Ground truth utility values for each alternative.",
                        "runs[].weights": "Weights used in the utility function.",
                        "runs[].params": "Optional parameters (e.g., exponential curvature).",
                        "runs[].subranks": "Sequence of elicited subranks. 'items' is indices ordered best to worst. 'cards' is blank cards between consecutive items."
                    },
                    "metadata": {
                        "f1": f1,
                        "f2": f2,
                        "num_steps": num_steps,
                        "k_subset": k_subset,
                        "n_max": n_max,
                        "target_consistency": target_consistency,
                        "utility_type": utility_type,
                        "sigma": sigma,
                        "n_runs": self.n_runs
                    },
                    "runs": []
                }

                for i in range(self.n_runs):
                    subranks_plot_data = []
                    ls = np.random.rand(f1, f2)
                    if utility_type == 'piecewise_linear':
                        U, w, params = self._generate_piecewise_linear_utility(ls, f2, num_intervals)
                        params_val = None # Transformer not easily JSON serializable
                    else:
                        U, w, params = self._generate_exponential_utility(ls, f2)
                        params_val = params.tolist()
                    
                    rk = self._compute_rank(U)
                    
                    subranks = []
                    
                    for j in range(1, num_steps + 1):
                        # Subset selection
                        rng_subset = np.random.default_rng(i * 1000 + j)
                        subset_indices = rng_subset.choice(f1, k_subset, replace=False)
                        
                        # Apply Thurstone Noise (Noisy Utilities)
                        # Variance of difference must be sigma^2, so variance of each item is sigma^2 / 2
                        noise_std = sigma / np.sqrt(2)
                        noisy_utilities = U[subset_indices] * n_max + np.random.normal(0, noise_std, k_subset)
                        
                        # Generate Subrank
                        sort_idx = np.argsort(-noisy_utilities)
                        ranked_items = subset_indices[sort_idx].tolist()
                        ranked_noisy_u = noisy_utilities[sort_idx]
                        
                        # Calculate Cards (eta) between consecutive elements
                        cards = [int(np.clip(np.round(ranked_noisy_u[m] - ranked_noisy_u[m+1]), 0, n_max)) for m in range(k_subset - 1)]
                        
                        subranks.append({
                            "items": ranked_items,
                            "cards": cards
                        })
                        
                        # Verification data for plotting (only for first run)
                        if i == 0 and plot_utilities:
                             step_plot_data = {
                                 "subset": subset_indices.tolist(),
                                 "true_u": (U[subset_indices] * n_max).tolist(),
                                 "noisy_u": noisy_utilities.tolist(),
                                 "true_items_sorted": subset_indices[np.argsort(-U[subset_indices])].tolist(),
                                 "noisy_items_sorted": ranked_items,
                                 "cards": cards
                             }
                             subranks_plot_data.append(step_plot_data)

                    run_data = {
                        "run_id": i,
                        "table": ls.tolist(),
                        "true_ranks": rk,
                        "true_utilities": U.tolist(),
                        "weights": w.tolist(),
                        "params": params_val,
                        "subranks": subranks
                    }
                    dataset_json["runs"].append(run_data)
                    
                    # Plots for the first run
                    if i == 0 and plot_utilities:
                        base_name = f"f1_{f1}__f2_{f2}__nsteps_{num_steps}"
                        plot_path = os.path.join(self.output_dir, f"{base_name}_utils_table0.png")
                        self._plot_marginal_utilities(utility_type, f2, w, params, plot_path)

                        # Plot: Subranks Comparison (Visual displacement)
                        self._plot_subranks_comparison(subranks_plot_data, n_max, os.path.join(self.output_dir, f"{base_name}_subranks_table0.png"))
                
                # Save config JSON
                base_name = f"f1_{f1}__f2_{f2}__nsteps_{num_steps}"
                json_path = os.path.join(self.output_dir, f"{base_name}.json")
                with open(json_path, 'w') as f:
                    json.dump(dataset_json, f, indent=2)
                print(f"Dataset saved to {json_path}")

    def _plot_subranks_comparison(self, plot_data, n_max, save_path):
        """Visualizes the difference between Ideal and Noisy subranks across all steps."""
        num_steps = len(plot_data)
        fig, axes = plt.subplots(num_steps, 1, figsize=(10, 2.5 * num_steps), sharex=False)
        if num_steps == 1: axes = [axes]

        for j, (step_data, ax) in enumerate(zip(plot_data, axes)):
            subset = step_data['subset']
            true_u = step_data['true_u']
            noisy_u = step_data['noisy_u']
            
            # Normalize X axis per step for better visibility
            all_vals = true_u + noisy_u
            x_min, x_max = min(all_vals) - 0.5, max(all_vals) + 0.5
            
            # Plot levels
            ax.axhline(1, color='gray', linestyle='--', alpha=0.3) # Ideal level
            ax.axhline(0, color='gray', linestyle='--', alpha=0.3) # Noisy level
            
            colors = plt.cm.get_cmap('tab10', len(subset))
            
            for idx_in_subset, item_idx in enumerate(subset):
                color = colors(idx_in_subset)
                t_u = true_u[idx_in_subset]
                n_u = noisy_u[idx_in_subset]
                
                # Ideal point
                ax.scatter(t_u, 1, color=color, s=100, zorder=5)
                ax.annotate(f"Alt {item_idx}", (t_u, 1.1), ha='center', fontweight='bold', color=color)
                
                # Noisy point
                ax.scatter(n_u, 0, color=color, s=100, zorder=5)
                ax.annotate(f"Alt {item_idx}", (n_u, -0.2), ha='center', color=color)
                
                # Connection
                ax.plot([t_u, n_u], [1, 0], color=color, linestyle=':', alpha=0.6)
            
            ax.set_yticks([0, 1])
            ax.set_yticklabels(['Noisy (Sample)', 'Ideal (Truth)'])
            ax.set_title(f"Step {j+1} Subrank Alignment", fontsize=12, pad=20)
            ax.set_xlim(x_min, x_max)
            ax.grid(True, axis='x', alpha=0.2)

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

if __name__ == "__main__":
    # --- Configuration ---
    DATASET_NAME = "default_dataset"
    OUTPUT_DIR = os.path.join("datasets", DATASET_NAME)
    N_RUNS = 10 # Set to 1 as per user's last edit
    
    F1_LIST = [30]
    F2_LIST = [4]
    NUM_STEPS = 7
    K_SUBSET = 3 # Set to 3 as per user's last edit
    N_MAX = 5   # Set to 5 as per user's last edit
    
    TARGET_CONSISTENCY = 1.0 # Set to 0.5 as per user's last edit
    UTILITY_TYPE = 'exponential'
    NUM_INTERVALS = 3
    PLOT_UTILITIES = True
    
    # --- Execution ---
    print(f"Starting Generation: {UTILITY_TYPE}")
    gen = DatasetGenerator(output_dir=OUTPUT_DIR, n_runs=N_RUNS)
    
    gen.generate_batch(
        F1=F1_LIST, 
        F2=F2_LIST, 
        num_steps=NUM_STEPS,
        k_subset=K_SUBSET,
        n_max=N_MAX,
        target_consistency=TARGET_CONSISTENCY,
        utility_type=UTILITY_TYPE,
        num_intervals=NUM_INTERVALS,
        plot_utilities=PLOT_UTILITIES
    )



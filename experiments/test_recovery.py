import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to sys.path to allow imports from sibling packages
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mcda.models import PiecewiseLinearTransformer
from inference.engine import PreferenceSampler, RHO
from user_inference.inference_tool import plot_marginal_utilities

def generate_synthetic_data(num_alternatives=20, num_criteria=2, num_intervals=3, n_max=5):
    np.random.seed(42)
    
    # 1. Generate Alternatives
    alternatives = np.random.rand(num_alternatives, num_criteria)
    
    # 2. Setup Transformer
    transformer = PiecewiseLinearTransformer.from_equal_intervals(alternatives, num_intervals)
    X_transformed = transformer.transform(alternatives)
    n_params = X_transformed.shape[1]
    
    # 3. Define True Parameters
    # Let's define an interesting w_true (e.g. increasing marginal utility for criterion 1, decreasing for criterion 2)
    # Criterion 1: [0.1, 0.4, 0.8] (Convex)
    # Criterion 2: [0.8, 0.4, 0.1] (Concave)
    w_true = np.array([0.1, 0.4, 0.8, 0.8, 0.4, 0.1])
    
    # True Deltas (engine convention: theta_k = sum_{m<=k} (rho + delta_raw_m))
    delta_true = np.ones(n_max) * 0.5
    theta_true = np.cumsum(delta_true + RHO)
    
    # 4. Compute True Utilities
    U_true = X_transformed @ w_true
    
    # 5. Generate Adjacent-Only Preferences (consecutive pairs of the true ranking)
    # Sort alternatives by true utility
    ranked_indices = np.argsort(-U_true)

    preferences = []
    from scipy.special import expit
    for i in range(len(ranked_indices) - 1):
        up_idx = ranked_indices[i]
        down_idx = ranked_indices[i + 1]

        u_diff = U_true[up_idx] - U_true[down_idx]

        # Most likely 'eta' under the conditional CLM with theta_0 = -inf:
        #   P(eta=0)          = sigma(theta_1 - u)
        #   P(eta=k), 0<k<max = sigma(theta_{k+1} - u) - sigma(theta_k - u)
        #   P(eta=n_max)      = 1 - sigma(theta_{n_max} - u)
        best_eta = 0
        best_prob = -1

        for k in range(n_max + 1):
            if k == 0:
                prob = expit(theta_true[0] - u_diff)
            elif k < n_max:
                prob = expit(theta_true[k] - u_diff) - expit(theta_true[k-1] - u_diff)
            else:
                prob = 1.0 - expit(theta_true[k-1] - u_diff)

            if prob > best_prob:
                best_prob = prob
                best_eta = k

        preferences.append([up_idx, down_idx, best_eta])

    return transformer, X_transformed, w_true, np.array(preferences), n_params

def run_recovery_test():
    print("Generating synthetic data...")
    n_max = 5
    transformer, X_transformed, w_true, preferences, n_params = generate_synthetic_data(n_max=n_max)
    
    print(f"Generated {len(preferences)} adjacent pairwise preferences.")
    eta_counts = np.bincount(preferences[:, 2], minlength=n_max + 1)
    print(f"Eta distribution (0..{n_max}): {eta_counts.tolist()}")
    
    # Initialize Sampler
    sampler = PreferenceSampler(X_transformed, preferences, n_params, n_max)
    
    print("Running FTRL (MAP estimation)...")
    w_map_full = sampler.find_map()
    w_map = w_map_full[:n_params]
    
    print("Running BAYES (HMC Posterior sampling)...")
    bayes_samples_full = sampler.run_hmc(num_samples=300, num_warmup=200, seed=2026)
    bayes_samples = bayes_samples_full[:, :n_params]
    
    # Normalize to compare shapes directly
    w_true_norm = w_true / np.sum(w_true)
    w_map_norm = w_map / np.sum(w_map)
    w_bayes_norm = np.mean(bayes_samples, axis=0) / np.sum(np.mean(bayes_samples, axis=0))

    # Compare with w_true
    print("\n--- Normalized Results ---")
    print("w_true_norm :", np.round(w_true_norm, 3))
    print("w_map_norm  :", np.round(w_map_norm, 3))
    print("w_bayes_norm:", np.round(w_bayes_norm, 3))
    
    # Evaluate MSE
    mse_map = np.mean((w_true_norm - w_map_norm)**2)
    mse_bayes = np.mean((w_true_norm - w_bayes_norm)**2)
    print(f"\nMSE FTRL (normalized) : {mse_map:.6f}")
    print(f"MSE BAYES (normalized): {mse_bayes:.6f}")
    
    # Plot marginal utilities including w_true (using normalized vectors)
    plot_marginal_utilities_with_true(transformer, w_true_norm, w_map_norm, bayes_samples / np.sum(bayes_samples, axis=1, keepdims=True))

def plot_marginal_utilities_with_true(transformer, w_true, w_map, bayes_samples):
    f2 = transformer.num_criteria
    x_plot = np.linspace(0, 1, 100)
    
    fig, axes = plt.subplots(1, f2, figsize=(5 * f2, 4))
    if f2 == 1:
        axes = [axes]
    
    col_end = np.cumsum(transformer.n_w)
    col_start = np.insert(col_end[:-1], 0, 0)
    
    for j in range(f2):
        ax = axes[j]
        start_idx, end_idx = int(col_start[j]), int(col_end[j])
        breakpoints = transformer.ch_p[j]
        
        # 0. TRUE Line
        w_j_true = w_true[start_idx:end_idx]
        y_true = []
        for x_val in x_plot:
            val = 0
            for k in range(len(w_j_true)):
                bp_low, bp_high = breakpoints[k], breakpoints[k+1]
                feat = np.clip((x_val - bp_low) / (bp_high - bp_low), 0, 1)
                val += w_j_true[k] * feat
            y_true.append(val)
        ax.plot(x_plot, y_true, 'k--', linewidth=2, label='TRUE')

        # 1. FTRL Line
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
        
        # 2. BAYES Region
        if bayes_samples is not None:
            all_y_bayes = []
            for s_idx in range(len(bayes_samples)):
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

    plt.suptitle("Recovery Test", fontsize=14)
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(__file__), "recovery_test.png")
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    run_recovery_test()

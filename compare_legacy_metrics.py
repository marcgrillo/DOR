import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kendalltau
matplotlib.use('Agg') # Non-interactive backend
from common import utils
from mcda.models import PiecewiseLinearTransformer

def calculate_legacy_metrics():
    print("=== Calculating Legacy DOR_2 Metrics for DOR_3 Results ===")
    
    # Configuration
    dataset_fold = 'default_dataset'
    f1, f2, f3 = 30, 4, 100
    target_methods = ['BAYES_DOR_PASSIVE', 'FTRL_DOR_PASSIVE']
    num_steps = 7
    hm = 1 # We'll just look at table 0 for a quick comparison, or HM if available
    
    # Load dataset info
    data_path = os.path.join('datasets', dataset_fold)
    tables, rankings, _, _ = utils.read_dataset(data_path, f1, f2, f3)
    
    results = {}

    for method in target_methods:
        print(f"\nProcessing {method}...")
        method_rhos = [] # ASPS in DOR_2
        method_taus = [] # ASRS in DOR_2
        
        for j in range(1, num_steps + 1):
            step_rhos = []
            step_taus = []
            
            # We loop over available tables (Human Models)
            table_idx = 0
            while True:
                sample_path = os.path.join('samples', dataset_fold, f'f1_{f1}_f2_{f2}_f3_{f3}', method, f'table_{table_idx}', f'{j}.npy')
                if not os.path.exists(sample_path):
                    break
                
                # Load samples
                samples = np.load(sample_path)
                if samples.ndim == 1:
                    samples = samples[None, :]
                
                # Setup transformer to get feature matrix X
                table = tables[table_idx]
                true_rank_indices = rankings[table_idx] # In DOR_3 rankings[i] is already rank-at-index
                # Wait, in DOR_2 R was true_rank_indices (alt indices from best to worst)
                # In DOR_3 rankings[i][alt_idx] is the rank.
                # Let's convert DOR_3 rankings to DOR_2 true_rank_indices
                R_indices = np.argsort(true_rank_indices)
                
                transformer = PiecewiseLinearTransformer.from_equal_intervals(table, 3)
                X = transformer.transform(table)
                n_params = X.shape[1]
                
                # Extract weights
                w_samples = samples[:, :n_params]
                
                # Calculate utilities
                U_samples = X @ w_samples.T # (n_alts, n_samples)
                
                # Map true_rank_indices to ranks (0 is best)
                true_ranks = true_rank_indices
                
                sample_rhos = []
                sample_taus = []
                for s in range(U_samples.shape[1]):
                    u = U_samples[:, s]
                    est_ranks = np.argsort(np.argsort(-u))
                    
                    rho, _ = spearmanr(true_ranks, est_ranks)
                    tau, _ = kendalltau(true_ranks, est_ranks)
                    sample_rhos.append(rho)
                    sample_taus.append(tau)
                
                step_rhos.append(np.mean(sample_rhos))
                step_taus.append(np.mean(sample_taus))
                table_idx += 1
            
            if not step_rhos:
                print(f"  Warning: No data found for step {j}")
                method_rhos.append(0)
                method_taus.append(0)
            else:
                method_rhos.append(np.mean(step_rhos))
                method_taus.append(np.mean(step_taus))
        
        results[method] = {'rhos': method_rhos, 'taus': method_taus}

    # Plotting
    plt.figure(figsize=(12, 5))
    
    # Plot Spearman (ASPS Legacy)
    plt.subplot(1, 2, 1)
    for method, data in results.items():
        plt.plot(range(1, num_steps + 1), data['rhos'], label=method, marker='o')
    plt.xlabel('Elicitation Step')
    plt.ylabel('Spearman Rho (Legacy ASPS)')
    plt.title('Rank Correlation (Spearman)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot Kendall (ASRS Legacy)
    plt.subplot(1, 2, 2)
    for method, data in results.items():
        plt.plot(range(1, num_steps + 1), data['taus'], label=method, marker='s')
    plt.xlabel('Elicitation Step')
    plt.ylabel('Kendall Tau (Legacy ASRS)')
    plt.title('Rank Correlation (Kendall)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('legacy_metrics_comparison.png', dpi=300)
    print("\n[Done] Plot saved as 'legacy_metrics_comparison.png'")

if __name__ == "__main__":
    calculate_legacy_metrics()

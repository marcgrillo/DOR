import time
import os
import itertools
import numpy as np

from generate_datasets import DatasetGenerator
from experiments.runner import run_batch_experiments
from experiments.metrics import BenchmarkRunner
from common.utils import parse_subfold_string
import common.utils as utils
from experiments.plotting import plot_metric_results

# Configurations
TEST_MODE = False
NUM_WORKERS = 10

if TEST_MODE:
    print(">>> RUNNING IN TINY GRID TEST MODE <<<")
    F1_list = [10]
    F2_list = [3, 4]  # 2 combinations to test parallel execution
    NUM_STEPS_list = [3]
    A_R_list = [3]
    K_list = [3]
    F_inc_list = [0]
    HM_0 = 4  # HM = 1 for BAYES
else:
    print(">>> RUNNING FULL GRID <<<")
    F1_list = [10, 20, 35, 50]
    F2_list = [3, 5, 7, 9]
    NUM_STEPS_list = [10]  # Just run the max steps, metrics will cover 1..10
    A_R_list = [3, 4, 5]
    K_list = [3, 4, 5]
    F_inc_list = [0, 15, 35, 50]
    HM_0 = 20

ETA_MAX = 5

def map_f_inc_to_consistency(f_inc):
    # 0% -> 1.0, 15% -> 0.85, 35% -> 0.65, 50% -> 0.50
    return 1.0 - (f_inc / 100.0)

def run_single_config(config, hm_0):
    f1, f2, steps, a_r, k, f_inc = config
    consistency = map_f_inc_to_consistency(f_inc)
    k_tuple = (a_r, k)
    
    config_str = f"F1={f1}_F2={f2}_S={steps}_AR={a_r}_K={k}_Inc={f_inc}"
    
    # 1. Generate Dataset
    generator = DatasetGenerator(n_runs=hm_0)
    dataset_fold = f"grid_{config_str}"
    generator.output_dir = os.path.join("datasets", dataset_fold)
    os.makedirs(generator.output_dir, exist_ok=True)
    generator.generate_batch(
        F1=[f1], F2=[f2], num_steps=steps, k_subset=k_tuple, 
        n_max=ETA_MAX, target_consistency=consistency, plot_utilities=False
    )
    
    config_results = {}
    
    # Run BAYES_DOR_PASSIVE
    sub_fold = 'BAYES_DOR_PASSIVE'
    alg_name, active_method_name = parse_subfold_string(sub_fold)
    
    algo_type, model_type = alg_name.split('-')
    HM = int(max(1, hm_0/4))
        
    # Run Batch Experiments
    run_batch_experiments(
        F1=[f1], F2=[f2], sub_fold=sub_fold, dataset_folds=[dataset_fold],
        alg=alg_name, overwrite=True, hm=HM, k_subset=k_tuple, 
        num_steps=steps, eta_max=ETA_MAX
    )
    
    # Compute Metrics
    runner = BenchmarkRunner(
        dataset_fold=dataset_fold,
        sub_fold=sub_fold,
        num_subint=3,
        hm=HM,
        F1=[f1], F2=[f2], NUM_STEPS_LIST=[steps],
        eta_max=ETA_MAX
    )
    
    runner.compute_perc_inc(force=True)
    runner.compute_metrics("poi", force=True)
    runner.compute_metrics("rai", force=True)
    runner.compute_asrs(force=True)
    runner.compute_asps(force=True)
    runner.compute_aios(force=True)
    
    # Extract Results (No Plotting)
    metrics_dict = {}
    for metric in ['perc_inc', 'asrs', 'asps', 'aios']:
        data = utils.load_test_results(metric, dataset_fold, sub_fold, f1, f2, steps)
        metrics_dict[metric] = data.tolist() if isinstance(data, np.ndarray) else data
        
    return config_str, metrics_dict

def main():
    print("Starting Grid Experiment for BAYES (Sequential)...")
    start_time = time.time()
    
    # Generate all combinations
    combinations = list(itertools.product(F1_list, F2_list, NUM_STEPS_list, A_R_list, K_list, F_inc_list))
    total_configs = len(combinations)
    print(f"Total configurations in grid: {total_configs}")
    
    save_path = "grid_results_tiny.npz" if TEST_MODE else "grid_results_full.npz"
    
    # Load existing results
    all_results = {}
    if os.path.exists(save_path):
        print(f"Loading existing results from {save_path}...")
        try:
            all_results = np.load(save_path, allow_pickle=True)['results'].item()
            print(f"Loaded {len(all_results)} configurations.")
        except Exception as e:
            print(f"Error loading {save_path}: {e}. Starting fresh.")
            
    # Find combinations that still need BAYES
    configs_to_run = []
    for config in combinations:
        f1, f2, steps, a_r, k, f_inc = config
        config_str = f"F1={f1}_F2={f2}_S={steps}_AR={a_r}_K={k}_Inc={f_inc}"
        
        # Check if BAYES is missing
        if config_str in all_results and 'BAYES_DOR_PASSIVE' in all_results[config_str]:
            continue
        configs_to_run.append(config)
        
    total_to_run = len(configs_to_run)
    if total_to_run == 0:
        print("All BAYES configurations have already been run!")
        return
        
    print(f"Configurations remaining to run for BAYES: {total_to_run}")
    
    completed = 0
    for idx, config in enumerate(configs_to_run):
        f1, f2, steps, a_r, k, f_inc = config
        config_str = f"F1={f1}_F2={f2}_S={steps}_AR={a_r}_K={k}_Inc={f_inc}"
        
        config_start = time.time()
        print(f"\n========================================================")
        print(f"[{idx+1}/{total_to_run}] Running Configuration: {config_str}")
        print(f"========================================================")
        
        try:
            # Run the single configuration sequentially
            config_str, metrics_dict = run_single_config(config, HM_0)
            
            # Ensure the configuration dictionary exists in all_results
            if config_str not in all_results:
                all_results[config_str] = {}
            
            # Store BAYES results
            all_results[config_str]['BAYES_DOR_PASSIVE'] = metrics_dict
            
            # Save progressively to avoid data loss
            np.savez_compressed(save_path, results=all_results)
            
            completed += 1
            percent = (completed / total_to_run) * 100
            print(f"[{completed}/{total_to_run}] ({percent:.1f}%) Completed in {time.time() - config_start:.2f}s")
            
        except Exception as e:
            print(f"Error running configuration {config_str}: {e}")
            
    total_time = time.time() - start_time
    print(f"\n========================================================")
    print(f"Finished! Total time: {total_time:.2f} seconds")
    print(f"========================================================")

if __name__ == "__main__":
    main()

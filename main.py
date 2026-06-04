import os
from experiments.runner import run_batch_experiments
from experiments.metrics import BenchmarkRunner
from common.utils import parse_subfold_string
from experiments.plotting import plot_metric_results

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

# ==============================================================================
# 1. Experiment Dimensions (Dataset Selection)
# ==============================================================================
F1 = [30]
F2 = [4]
NUM_STEPS_LIST = [7]

DATASET_FOLDS = ['default_dataset']

# ==============================================================================
# 2. Algorithm Configuration
# ==============================================================================
TARGET_METHODS = [
    'BAYES_DOR_PASSIVE',
    'FTRL_DOR_PASSIVE',
] 

# ==============================================================================
# 3. Simulation Parameters
# ==============================================================================
HM_0 = 10
K_SUBSET = 3
ETA_MAX = 5

# ==============================================================================
# 4. Execution Flow Flags
# ==============================================================================
OVERWRITE = True
CALCULATE_METRICS = True
FORCE_METRICS = True
GENERATE_PLOTS = True

# ----------------------------------------------------------------------
# Execution
# ----------------------------------------------------------------------

if __name__ == "__main__":
    
    # 1. Run Experiments
    for sub_fold in TARGET_METHODS:
        alg_name, active_method_name = parse_subfold_string(sub_fold)
        print(f"\n>>> Running: {alg_name} with {active_method_name}")

        algo_type, model_type = alg_name.split('-')
        if algo_type == 'BAYES':
            HM = int(max(1, HM_0/4))
        else:
            HM = HM_0
        
        run_batch_experiments(
            F1, F2, 
            sub_fold=sub_fold, 
            dataset_folds=DATASET_FOLDS, 
            alg=alg_name, 
            overwrite=OVERWRITE,
            hm=HM,
            k_subset=K_SUBSET,
            num_steps=NUM_STEPS_LIST[0],
            eta_max=ETA_MAX
        )
        
    # 2. Calculate Metrics
    force = FORCE_METRICS
    if CALCULATE_METRICS:
        f1, f2, num_steps = F1[0], F2[0], NUM_STEPS_LIST[0]
        
        for sub_fold in TARGET_METHODS:
            alg_name, active_method_name = parse_subfold_string(sub_fold)

            algo_type, model_type = alg_name.split('-')
            if algo_type == 'BAYES':
                HM = int(max(1, HM_0/4))
            else:
                HM = HM_0

            print(f"\n=== Calculating Metrics for {alg_name} ===")
            runner = BenchmarkRunner(
                dataset_fold=DATASET_FOLDS[0],
                sub_fold=sub_fold,
                num_subint=3,
                hm=HM, 
                F1=F1, F2=F2, NUM_STEPS_LIST=NUM_STEPS_LIST,
                eta_max=ETA_MAX
            )
            runner.compute_perc_inc(force=force)
            runner.compute_metrics("poi", force=force)
            runner.compute_metrics("rai", force=force)
            runner.compute_asrs(force=force)
            runner.compute_aios(force=force)
            runner.compute_asps(force=force)

    # 3. Generate Plots
    if GENERATE_PLOTS:
        f1, f2, num_steps = F1[0], F2[0], NUM_STEPS_LIST[0]
        
        metrics_to_plot = ['perc_inc', 'asrs', 'asps', 'aios']
        
        for sub_fold in TARGET_METHODS:
            alg_name, active_method_name = parse_subfold_string(sub_fold)
            
            algo_type, model_type = alg_name.split('-')
            if algo_type == 'BAYES':
                HM = int(max(1, HM_0/4))
            else:
                HM = HM_0

            print(f"\n=== Generating Plots for {alg_name} ===")
            
            for metric in metrics_to_plot:
                try:
                    plot_metric_results(
                        metric_name=metric,
                        F1=F1, F2=F2, NUM_STEPS_LIST=NUM_STEPS_LIST,
                        hm=HM,
                        dataset_fold=DATASET_FOLDS[0],
                        sub_fold=sub_fold,
                        save_figs=True,
                        show_figs=False
                    )
                except Exception as e:
                    print(f"Failed to plot {metric} for {sub_fold}: {e}")

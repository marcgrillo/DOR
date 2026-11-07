import os
from bdor import utils
from bdor.simulation import process_single_table
from bdor.paths import DATASETS_DIR, SAMPLES_DIR

def run_batch_experiments(F1, F2, sub_fold, dataset_folds, alg, overwrite, hm=None, k_subset=3, num_steps=7, eta_max=30, expansion='all', seed_base=0, log_loss=True):
    """
    Orchestrates the baseline experiments across multiple datasets and configurations.
    """
    for dataset_fold in dataset_folds:
        print(f"\n=== Processing Dataset: {dataset_fold} ===")
        
        samples_root = os.path.join(SAMPLES_DIR, dataset_fold)
        
        for f1 in F1:
            for f2 in F2:
                try:
                    # Read From datasets/dataset_name/
                    tables, rankings, dm_prefs, Us = utils.read_dataset(os.path.join(DATASETS_DIR, dataset_fold), f1, f2, num_steps)
                except FileNotFoundError as e:
                    print(f"Skipping {f1}/{f2}/{num_steps}: {e}")
                    continue
                    
                config_dir = os.path.join(samples_root, f"f1_{f1}_f2_{f2}_nsteps_{num_steps}")
                method_dir = os.path.join(config_dir, sub_fold)
                utils.save_path(method_dir)

                # --- SLICE DATASET BASED ON HM ---
                if hm is not None:
                    limit = min(hm, len(tables))
                    tables = tables[:limit]
                    rankings = rankings[:limit]
                    dm_prefs = dm_prefs[:limit]
                    Us = Us[:limit]
                
                # Process Tables
                for i, (table, ranking, prefs, u_true) in enumerate(zip(tables, rankings, dm_prefs, Us)):
                    table_dir = os.path.join(method_dir, f"table_{i}")
                    utils.save_path(table_dir)
                    
                    process_single_table(
                        table=table,
                        table_preferences=prefs,
                        num_steps=num_steps,
                        k_subset=k_subset,
                        eta_max=eta_max,
                        output_dir=table_dir,
                        alg=alg,
                        overwrite=overwrite,
                        table_index=i,
                        expansion=expansion,
                        seed_base=seed_base,
                        log_loss=log_loss
                    )
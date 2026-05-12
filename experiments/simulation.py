import os
import numpy as np
from tqdm import tqdm
from mcda.models import PiecewiseLinearTransformer
from inference.engine import PreferenceSampler

def get_sampler_state(sampler, alg):
    """
    Returns the state of the sampler (Samples for BAYES, MAP for FTRL).
    """
    algo_type, _ = alg.split('-')
    if algo_type == 'BAYES':
        return sampler.run_nested(nlive=200) # smaller nlive for speed during baseline testing
    elif algo_type == 'FTRL':
        return sampler.find_map()
    else:
        raise ValueError(f"Unknown algorithm type: {algo_type}")

def process_single_table(
    table, 
    table_preferences,
    num_steps, 
    k_subset,
    eta_max,
    output_dir, 
    alg='BAYES-DOR', 
    overwrite=False,
    table_index=0,
    disable_tqdm=False,
    progress_dict=None
):
    """
    Runs the baseline inference for a single Human Model (table) over a sequence of passive queries.
    """
    # 1. Setup
    transformer = PiecewiseLinearTransformer.from_equal_intervals(table, num_intervals=3)
    feature_matrix = transformer.transform(table)
    
    n_params = transformer.total_params
    n_max = eta_max # Use the hyperparameter
    
    sampler = PreferenceSampler(feature_matrix, [], n_params, n_max)

    # 2. Main Loop
    for j in tqdm(range(1, num_steps + 1), desc=f"  Table {table_index}", leave=False, disable=disable_tqdm):
        if progress_dict is not None:
            progress_dict[table_index] = j
            
        path_w = os.path.join(output_dir, f"{j}.npy")

        # table_preferences is now a list of subranks for this run
        subrank = table_preferences[j-1]
        items = subrank['items']
        cards = subrank['cards']

        # Skip logic
        if os.path.exists(path_w) and not overwrite:
            for start_idx in range(len(items)):
                for end_idx in range(start_idx + 1, len(items)):
                    p_win = items[start_idx]
                    p_lose = items[end_idx]
                    eta = sum(cards[start_idx : end_idx])
                    sampler.add_preference(int(p_win), int(p_lose), int(eta))
            continue

        # Add new preferences from the subrank
        # Expansion: A > B > C implies A>B, B>C, A>C
        for start_idx in range(len(items)):
            for end_idx in range(start_idx + 1, len(items)):
                p_win = items[start_idx]
                p_lose = items[end_idx]
                
                # Sum cards between them in the subrank
                eta = sum(cards[start_idx : end_idx])
                
                sampler.add_preference(int(p_win), int(p_lose), int(eta))
        
        # Run inference
        w_state = get_sampler_state(sampler, alg)
        np.save(path_w, w_state)
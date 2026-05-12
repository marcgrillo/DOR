import os
import json
import numpy as np
from itertools import combinations
from scipy.special import expit
from scipy.stats import beta

def robust_sigmoid(x):
    return expit(x)

def safe_inverse(M, jitter=1e-6):
    try:
        return np.linalg.inv(M + np.eye(M.shape[0]) * jitter)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(M)

def save_path(path: str):
    os.makedirs(path, exist_ok=True)

def get_combinations(arr):
    return np.array(list(combinations(arr, 2)))

def read_dataset(dataset_fold: str, f1: int, f2: int, num_steps: int):
    json_path = os.path.join(dataset_fold, f"f1_{f1}__f2_{f2}__nsteps_{num_steps}.json")

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Missing dataset file: {json_path}")

    with open(json_path, 'r') as f:
        dataset = json.load(f)

    runs = dataset['runs']
    tables = []
    rankings = []
    subranks_list = []
    Us = []

    for run in runs:
        tables.append(run['table'])
        rankings.append(run['true_ranks'])
        subranks_list.append(run['subranks'])
        Us.append(run['true_utilities'])

    return np.array(tables), np.array(rankings), subranks_list, np.array(Us)

def load_test_results(test_name: str, dataset_fold: str, sub_fold: str, f1: int, f2: int, num_steps: int):
    test_fold = os.path.join('tests', dataset_fold, sub_fold, test_name)
    sub_test_fold = os.path.join(test_fold, f"f1_{f1}_f2_{f2}_nsteps_{num_steps}")
    
    loaded_data = []
    
    for i in range(1, num_steps + 1):
        if test_name == "perc_inc":
            path = os.path.join(sub_test_fold, f"perc_inc_{i}.npy")
        else:
            path = os.path.join(sub_test_fold, f"{test_name}_{i}.npy")
            
        if not os.path.exists(path):
            print(path, "not found. Skipping.")
            continue

        d = np.load(path)
        loaded_data.append(d.flatten())
        
    if not loaded_data:
        return np.array([])
        
    return np.stack(loaded_data, axis=0)

def parse_subfold_string(s):
    parts = s.split('_')
    alg_str = f"{parts[0]}-{parts[1]}" 
    active_str = parts[2]
    return alg_str, active_str
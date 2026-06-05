import numpy as np
import os
import re
import csv

npz_file = "grid_results_full.npz"
csv_file = "grid_results_full.csv"

if not os.path.exists(npz_file):
    raise FileNotFoundError(f"{npz_file} not found.")

print("Loading dataset...")
data = np.load(npz_file, allow_pickle=True)['results'].item()

pattern = re.compile(r"F1=(\d+)_F2=(\d+)_S=(\d+)_AR=(\d+)_K=(\d+)_Inc=(\d+)")

headers = [
    'Method', 'F1_Alternatives', 'F2_Criteria', 'Max_Steps', 'AR_Active_Ratio', 
    'K_Subset_Size', 'Inconsistency_Pct', 'Step_T', 'Table_Index', 
    'ASRS', 'ASPS', 'AIOS', 'PERC_INC'
]

print("Writing to CSV...")
with open(csv_file, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=headers)
    writer.writeheader()
    
    for key, metrics_data in data.items():
        match = pattern.match(key)
        if not match:
            continue
        f1, f2, s_max, ar, k, inc = map(int, match.groups())
        
        for method in ['FTRL_DOR_PASSIVE', 'BAYES_DOR_PASSIVE']:
            method_data = metrics_data.get(method, {})
            if not method_data:
                continue
                
            asrs = np.array(method_data.get('asrs', []))
            asps = np.array(method_data.get('asps', []))
            aios = np.array(method_data.get('aios', []))
            perc_inc = np.array(method_data.get('perc_inc', []))
            
            if asrs.ndim == 2:
                num_steps, num_tables = asrs.shape
                for t_idx in range(num_steps):
                    t_val = t_idx + 1
                    for table_idx in range(num_tables):
                        writer.writerow({
                            'Method': 'FTRL' if 'FTRL' in method else 'BAYES',
                            'F1_Alternatives': f1,
                            'F2_Criteria': f2,
                            'Max_Steps': s_max,
                            'AR_Active_Ratio': ar,
                            'K_Subset_Size': k,
                            'Inconsistency_Pct': inc,
                            'Step_T': t_val,
                            'Table_Index': table_idx,
                            'ASRS': asrs[t_idx, table_idx],
                            'ASPS': asps[t_idx, table_idx] if t_idx < len(asps) and table_idx < asps.shape[1] else '',
                            'AIOS': aios[t_idx, table_idx] if t_idx < len(aios) and table_idx < aios.shape[1] else '',
                            'PERC_INC': perc_inc[t_idx, table_idx] if t_idx < len(perc_inc) and table_idx < perc_inc.shape[1] else '',
                        })

print(f"Exported successfully to {csv_file}!")

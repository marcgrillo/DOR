"""Pilot slice of the adjacent-v1 campaign: the 4 configs F1=10, F2=3, S=10, AR=K=3,
Inc in {0,15,35,50}, all 4 arms, merged into the same grid_results_adjacent_v1.npz
(the full campaign will skip them on resume). Prints a final-step summary table."""
import os
import time
import numpy as np
import run_grid_adjacent_v1 as camp

PILOT = [(10, 3, 10, 3, 3, inc) for inc in [0, 15, 35, 50]]

all_results = {}
if os.path.exists(camp.SAVE_PATH):
    all_results = np.load(camp.SAVE_PATH, allow_pickle=True)['results'].item()

for config in PILOT:
    f1, f2, steps, a_r, k, f_inc = config
    config_str = f"F1={f1}_F2={f2}_S={steps}_AR={a_r}_K={k}_Inc={f_inc}"
    if config_str in all_results and all(a[0] in all_results[config_str] for a in camp.ARMS):
        print(f"{config_str}: already complete, skipping")
        continue
    t0 = time.time()
    print(f"\n=== PILOT {config_str} ===")
    config_str, config_results = camp.run_single_config(config, camp.HM_0)
    all_results.setdefault(config_str, {}).update(config_results)
    np.savez_compressed(camp.SAVE_PATH, results=all_results,
                        run_tag=camp.RUN_TAG, run_seed=camp.RUN_SEED, eta_max=camp.ETA_MAX)
    print(f"done in {time.time() - t0:.1f}s")

print("\n===== FINAL-STEP (T=10) MEANS, pilot slice =====")
print(f"{'Inc':>4} {'arm':<18} {'ASRS':>7} {'ASPS':>7} {'AIOS':>7} {'cumLoss':>9} {'runtime':>8}")
for config in PILOT:
    f1, f2, steps, a_r, k, f_inc = config
    config_str = f"F1={f1}_F2={f2}_S={steps}_AR={a_r}_K={k}_Inc={f_inc}"
    for sub_fold, _, _ in camp.ARMS:
        md = all_results[config_str][sub_fold]
        vals = [np.array(md[m])[-1].mean() for m in ['asrs', 'asps', 'aios']]
        loss = np.array(md['log_loss'], dtype=float)
        cum = np.nansum(loss[:, :, 0], axis=0).mean() if loss.size else np.nan
        print(f"{f_inc:>4} {sub_fold:<18} {vals[0]:>7.3f} {vals[1]:>7.3f} {vals[2]:>7.3f} "
              f"{cum:>9.2f} {md['runtime_s']:>7.1f}s")

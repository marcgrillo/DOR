"""Regenerate the synthetic datasets of the main simulation grid (NOT the paper's own data).

Writes one dataset per grid configuration with bdor.datagen.DatasetGenerator and the settings
of the June 2026 campaign: 20 replications, each a random performance table, an exponential
true value function and the answers of all ten sessions (rankings and card counts, e_max = 5,
Thurstone noise calibrated to the configuration's Finc). Optional step 0 of the main simulation
pipeline: run_grid.py reads these datasets. Existing dataset JSONs are kept unless --overwrite.

WARNING: the paper's grid datasets were generated on 4 June 2026 WITHOUT a random seed, so they
cannot be regenerated: this script writes new, statistically equivalent data, and every result
computed from it differs from the paper's numbers. Keep a backup of data/datasets/grid_*. Only
the session subsets coincide with the paper's (they come from per-(replication, session)
seeds); the files also gain subranks[].levels (singletons) and metadata.merge_threshold = 0,
which every reader treats exactly like the older format.

Paper:   Sec. 5.1 simulation design and the Table 2 grid (inputs of every grid result)
Inputs:  none
Outputs: data/datasets/grid_<config>/f1_<|A|>__f2_<m>__nsteps_10.json
Runtime: about a minute for all 576 configurations on one desktop CPU core.

Usage:
    python scripts/simulation/generate_grid_datasets.py [--paper-only] [--seed S] [--limit N]
                                                        [--dry-run] [--overwrite]

(new; the June 2026 campaign generated these datasets inside the original run_grid.py)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, so `import bdor` works without installing
import argparse
import itertools
import os
import time

import numpy as np

from bdor.datagen import DatasetGenerator
from bdor.paths import DATASETS_DIR, METRICS_DIR, SAMPLES_DIR

# The grid of run_grid.py (keep the two in sync): 576 configurations, 192 of them with AR == K.
F1_list = [10, 20, 35, 50]     # |A|: alternatives
F2_list = [3, 5, 7, 9]         # m: criteria
NUM_STEPS_list = [10]          # S: sessions per replication
A_R_list = [3, 4, 5]           # AR: subset size k of session 1
K_list = [3, 4, 5]             # K: subset size k of sessions 2..S
F_inc_list = [0, 15, 35, 50]   # Inc: nominal inconsistency Finc (%)
HM_0 = 20                      # replications per configuration
ETA_MAX = 5                    # e_max, the largest number of blank cards in a gap

WARNING = """\
WARNING: the paper's grid datasets were generated in June 2026 WITHOUT a random seed and
         cannot be regenerated. This script writes NEW, statistically equivalent data; every
         result computed from it will differ from the paper's numbers."""


def config_string(config):
    """``(F1, F2, S, AR, K, Inc)`` -> ``'F1=.._F2=.._S=.._AR=.._K=.._Inc=..'``, as in run_grid.py."""
    f1, f2, steps, a_r, k, f_inc = config
    return f"F1={f1}_F2={f2}_S={steps}_AR={a_r}_K={k}_Inc={f_inc}"


def dataset_json(config):
    """Path of the dataset JSON of ``config``, relative to DATASETS_DIR."""
    f1, f2, steps = config[:3]
    return os.path.join(f"grid_{config_string(config)}", f"f1_{f1}__f2_{f2}__nsteps_{steps}.json")


def generate(config):
    """Write the dataset of one configuration with the June 2026 generator call."""
    f1, f2, steps, a_r, k, f_inc = config
    gen = DatasetGenerator(output_dir=os.path.join(DATASETS_DIR, f"grid_{config_string(config)}"),
                           n_runs=HM_0)
    gen.generate_batch(F1=[f1], F2=[f2], num_steps=steps, k_subset=(a_r, k), n_max=ETA_MAX,
                       target_consistency=1.0 - (f_inc / 100.0), plot_utilities=False)


def main():
    """Select the configurations, report what already exists, generate the rest."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    ap.add_argument("--paper-only", action="store_true",
                    help="only the 192 configurations with AR == K, the ones the paper reports")
    ap.add_argument("--seed", type=int, default=None, metavar="S",
                    help="seed NumPy's global generator once, before the first dataset; the "
                         "stream then runs through the configurations in grid order, so the "
                         "data of one configuration depends on which others are generated in "
                         "the same call (default: unseeded, as in June 2026)")
    ap.add_argument("--limit", type=int, default=None, metavar="N",
                    help="only the first N selected configurations, in grid order (quick tests)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print what would be generated or overwritten, write nothing")
    ap.add_argument("--overwrite", action="store_true",
                    help="also regenerate the configurations whose dataset JSON exists, "
                         "replacing it; on the real data/datasets/ this destroys the paper's "
                         "datasets")
    args = ap.parse_args()
    if args.limit is not None and args.limit < 1:
        ap.error("--limit must be at least 1")

    configs = list(itertools.product(F1_list, F2_list, NUM_STEPS_list, A_R_list, K_list,
                                     F_inc_list))
    if args.paper_only:
        configs = [c for c in configs if c[3] == c[4]]
    if args.limit is not None:
        configs = configs[:args.limit]
    existing = [c for c in configs if os.path.exists(os.path.join(DATASETS_DIR, dataset_json(c)))]
    todo = configs if args.overwrite else [c for c in configs if c not in existing]

    print(WARNING)
    print(f"\nDataset folder: {DATASETS_DIR}")
    print(f"{len(configs)} configurations selected: {len(existing)} with an existing dataset, "
          f"{len(todo)} to generate")
    if existing:
        if args.overwrite:
            print(f"\n--overwrite: these {len(existing)} existing dataset JSONs "
                  "WILL BE OVERWRITTEN:")
        else:
            print(f"\nKept, not regenerated ({len(existing)}; --overwrite would replace them):")
        for c in existing:
            print(f"    {dataset_json(c)}")

    # Inference states of an older dataset would be reused by run_grid.py with the new data.
    stale = [c for c in todo
             if any(os.path.isdir(os.path.join(root, f"grid_{config_string(c)}"))
                    for root in (SAMPLES_DIR, METRICS_DIR))]
    if stale:
        print(f"\n{len(stale)} of the configurations to generate already have inference "
              "folders, computed from other data, which run_grid.py would silently reuse with the "
              f"new datasets. Move or delete these grid_<config> folders under {SAMPLES_DIR} and "
              f"{METRICS_DIR} first:")
        for c in stale:
            print(f"    grid_{config_string(c)}")

    if args.dry_run:
        if todo:
            print(f"\nWould generate ({len(todo)}):")
            for c in todo:
                print(f"    {dataset_json(c)}")
        print("\n--dry-run: nothing written.")
        return
    if stale:
        sys.exit("Refusing to generate datasets next to stale inference folders (listed above).")
    if not todo:
        print("\nNothing to generate.")
        return

    if args.seed is not None:
        np.random.seed(args.seed)
    print(f"\nSeed: {args.seed if args.seed is not None else 'none (unseeded, as in June 2026)'}")
    t0 = time.time()
    for n, config in enumerate(todo, 1):
        print(f"\n[{n}/{len(todo)}] grid_{config_string(config)}")
        generate(config)
    print(f"\nWrote {len(todo)} dataset(s) in {time.time() - t0:.0f} s. Before running "
          "run_grid.py on them, move results/grid_results_joint_v1.npz away (run_grid.py skips "
          "every configuration already in it), together with any data/samples/grid_* and "
          "data/metrics/grid_* folders of earlier runs.")


if __name__ == "__main__":
    main()

"""Run one method over dataset folders, configurations and replications.

``run_batch_experiments`` is the simulation driver of every study in ``scripts/simulation/``
(``run_grid.py``, ``run_ties_study.py``, ``run_emax_study.py``, ``run_emax_flipmatch.py``).
It reads the datasets written by ``bdor.datagen`` and runs
``bdor.simulation.process_single_table`` on each replication, which saves the inference
state after every session under ``data/samples/``; ``bdor.metrics.BenchmarkRunner`` then
scores those states.
"""
import os
from bdor import utils
from bdor.simulation import process_single_table
from bdor.paths import DATASETS_DIR, SAMPLES_DIR


def run_batch_experiments(F1, F2, sub_fold, dataset_folds, alg, overwrite, hm=None, k_subset=3,
                          num_steps=7, eta_max=30, expansion='all', seed_base=0, log_loss=True):
    """Run one method (arm) on every replication of the given dataset folders.

    For each folder in ``dataset_folds`` and each (f1, f2) in ``F1 x F2`` it reads
    ``data/datasets/<fold>/f1_<f1>__f2_<f2>__nsteps_<num_steps>.json`` (a configuration
    without that file is skipped with a message) and calls ``process_single_table`` on
    each of the first ``hm`` replications i, with output directory
    ``data/samples/<fold>/f1_<f1>_f2_<f2>_nsteps_<num_steps>/<sub_fold>/table_<i>/``
    (created if needed; see ``process_single_table`` for the files written there).

    The defaults of ``k_subset``, ``num_steps``, ``eta_max`` and ``expansion`` are not the
    paper's settings: every study script passes explicit values (``num_steps=10``,
    ``eta_max`` = e_max = 5, or 50 in the e_max study, and ``expansion='adjacent'``
    except for the all-pairs ablation arms).

    Parameters
    ----------
    F1, F2 : list of int
        Numbers of alternatives |A| and of criteria m to run.
    sub_fold : str
        Arm tag ``<ALG>_DOR_<TAG>`` (e.g. ``'BAYES_DOR_ADJJ1'``), the output subfolder.
    dataset_folds : list of str
        Dataset folder names under ``data/datasets``.
    alg : str
        ``'<ALG>-DOR'``, the algorithm token of the arm
        (``bdor.utils.parse_subfold_string(sub_fold)[0]``).
    overwrite : bool
        Recompute existing files; otherwise an interrupted run resumes where it stopped.
    hm : int, optional
        Number of replications to run, starting from the first; all when None.
    k_subset : int or tuple
        Passed through; not used by ``process_single_table``.
    num_steps : int
        Number of sessions; also part of the dataset file name.
    eta_max : int
        e_max, the maximum number of blank cards per gap.
    expansion : {'all', 'adjacent'}
        Gap expansion of ``bdor.simulation.expand_subrank``.
    seed_base : int
        Base of the per-(replication, session) seeds, see ``process_single_table``.
    log_loss : bool
        Record the prequential log-loss files ``loss_<j>.npy``.
    """
    for dataset_fold in dataset_folds:
        print(f"\n=== Processing Dataset: {dataset_fold} ===")

        samples_root = os.path.join(SAMPLES_DIR, dataset_fold)

        for f1 in F1:
            for f2 in F2:
                try:
                    tables, rankings, dm_prefs, Us = utils.read_dataset(
                        os.path.join(DATASETS_DIR, dataset_fold), f1, f2, num_steps)
                except FileNotFoundError as e:
                    print(f"Skipping {f1}/{f2}/{num_steps}: {e}")
                    continue

                config_dir = os.path.join(samples_root, f"f1_{f1}_f2_{f2}_nsteps_{num_steps}")
                method_dir = os.path.join(config_dir, sub_fold)
                utils.save_path(method_dir)

                # Keep the first hm replications
                if hm is not None:
                    limit = min(hm, len(tables))
                    tables = tables[:limit]
                    rankings = rankings[:limit]
                    dm_prefs = dm_prefs[:limit]
                    Us = Us[:limit]

                for i, (table, ranking, prefs, u_true) in enumerate(
                        zip(tables, rankings, dm_prefs, Us)):
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

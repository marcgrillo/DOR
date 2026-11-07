"""Ranking metrics of the simulation studies (paper Sec. 5.2; eqs. (4), (5) and (23)-(25)).

``BenchmarkRunner`` scores the inference states that ``bdor.simulation`` saved after every
session, for one arm of one dataset folder, in two passes:

1. ``compute_metrics('poi')`` and ``compute_metrics('rai')`` turn every state into the
   |A| x |A| pairwise winning index matrix PWI (eq. (5); called POI in the code) and rank
   acceptability index matrix RAI (eq. (4)), cached per (replication, session);
2. ``compute_asrs``, ``compute_asps`` and ``compute_aios`` reduce the cached matrices of
   each session to one value per replication of ASR (eq. (23), from RAI), ASP (eq. (24),
   from PWI) and AIO (eq. (25), from RAI).

Each row of a state is one draw of the parameter vector, of which only the first N columns
(the weights w) are used: U(a) = v(a)^T w, eq. (2). A point estimate (a 1-D state, e.g.
FTRL-DOR or DOR) counts as a single draw, so its RAI and PWI are 0/1 indicators (Sec. 5.2).

Files under ``data/metrics/<fold>/<arm>/``, with ``<config>`` = ``f1_<|A|>_f2_<m>_nsteps_<T>``::

    pois/<config>/table_<i>/<j>.npy   PWI after session j, (|A|, |A|)
    rais/<config>/table_<i>/<j>.npy   RAI after session j, (|A|, |A|): alternative x rank
    asrs/<config>/asrs_<j>.npy        ASR after session j, one value per replication
    asps/<config>/asps_<j>.npy        ASP after session j, one value per replication
    aios/<config>/aios_<j>.npy        AIO after session j, one value per replication

``bdor.utils.load_test_results`` stacks the last three into arrays (sessions, replications).
"""
import os
import numpy as np
from tqdm import tqdm
from bdor.utils import save_path, read_dataset
from bdor.paths import DATASETS_DIR, SAMPLES_DIR, METRICS_DIR
from bdor.features import PiecewiseLinearTransformer


class BenchmarkRunner:
    """Compute and cache RAI / PWI and ASR / ASP / AIO for one arm of one dataset folder.

    Call ``compute_metrics('poi')`` and ``compute_metrics('rai')`` first: ``compute_asps``
    reads the cached PWI matrices, ``compute_asrs`` and ``compute_aios`` the RAI matrices.
    Every method loops over all combinations of ``F1``, ``F2`` and ``NUM_STEPS_LIST``, the
    first ``hm`` replications and sessions 1..T, and keeps existing files unless
    ``force=True``.

    Parameters
    ----------
    dataset_fold : str
        Dataset folder name, e.g. ``'grid_F1=10_F2=3_S=10_AR=3_K=3_Inc=0'``; the same name
        is used under ``data/datasets``, ``data/samples`` and ``data/metrics``.
    sub_fold : str
        Arm tag, e.g. ``'FTRL_DOR_ADJJ1'``.
    num_subint : int
        Equal-width intervals per criterion of the piecewise-linear features. Every study
        passes 3, the value ``bdor.simulation`` uses for inference.
    hm : int
        Number of replications to score (the first ``hm``), capped at the number of runs
        in the dataset of the first configuration.
    F1, F2, NUM_STEPS_LIST : list of int
        |A|, m and number of sessions T of the configurations to score.
    eta_max : int
        Stored only; the metrics do not depend on it.

    Notes
    -----
    The constructor creates ``data/metrics/<fold>/<arm>/``.
    """

    def __init__(self, dataset_fold, sub_fold, num_subint, hm, F1, F2, NUM_STEPS_LIST, eta_max=30):
        self.dataset_fold = os.path.join(DATASETS_DIR, dataset_fold)
        self.sub_fold = sub_fold
        self.samples_fold = os.path.join(SAMPLES_DIR, dataset_fold)
        self.tests_fold = os.path.join(METRICS_DIR, dataset_fold)
        self.sub_tests_fold = os.path.join(self.tests_fold, sub_fold)

        self.num_subint = num_subint

        # Cap hm at the number of replications in the dataset (first configuration)
        f1_0, f2_0, nsteps_0 = F1[0], F2[0], NUM_STEPS_LIST[0]
        try:
            tables, _, _, _ = read_dataset(self.dataset_fold, f1_0, f2_0, nsteps_0)
            self.hm = min(hm, len(tables))
        except Exception:
            self.hm = hm  # dataset not readable: keep the requested number

        self.F1, self.F2, self.NUM_STEPS_LIST = F1, F2, NUM_STEPS_LIST
        self.eta_max = eta_max

        save_path(self.sub_tests_fold)

    def _sample_path(self, f1, f2, num_steps, i, j):
        """Path of the state saved after session j of replication i."""
        config_dir = os.path.join(self.samples_fold, f"f1_{f1}_f2_{f2}_nsteps_{num_steps}")
        return os.path.join(config_dir, self.sub_fold, f"table_{i}", f"{j}.npy")

    # ----------------------------------------------------------------------
    # Per-state matrices: PWI (called POI here) and RAI
    # ----------------------------------------------------------------------

    def calc_poi(self, f1, t, sam):
        """Pairwise winning index matrix PWI, eq. (5) (POI in the code).

        Parameters
        ----------
        f1 : int
            Number of alternatives |A|.
        t : np.ndarray, shape (|A|, N)
            Feature vectors v(a) of the alternatives.
        sam : np.ndarray, shape (R, >= N) or (>= N,)
            State: R parameter draws, or one point estimate. Only the first N columns
            (the weights w) are used.

        Returns
        -------
        np.ndarray, shape (|A|, |A|)
            ``PWI[a, b]``: share of draws with U(a) >= U(b). Ties count for both
            alternatives, so the diagonal is 1. A state without draws gives NaN.
        """
        rank_count = np.zeros((f1, f1))
        sam = np.atleast_2d(sam)
        w = sam[:, :t.shape[1]]
        vals = t @ w.T  # (|A|, R) values U(a) of every draw

        for col in range(vals.shape[1]):
            scores = vals[:, col]
            rank_count += (scores[:, None] >= scores[None, :])

        return rank_count / vals.shape[1]

    def calc_rai(self, f1, t, sam):
        """Rank acceptability index matrix RAI, eq. (4).

        Same arguments as ``calc_poi``. Returns ``RAI[a, r]``, shape (|A|, |A|): the share
        of draws in which alternative a takes rank r (column 0 = best). Ties in U are
        broken by the order of ``np.argsort``.
        """
        rank_count = np.zeros((f1, f1))
        sam = np.atleast_2d(sam)
        w = sam[:, :t.shape[1]]
        vals = t @ w.T

        for col in range(vals.shape[1]):
            scores = vals[:, col]
            ranks = np.argsort(np.argsort(-scores))
            rank_count[np.arange(f1), ranks] += 1

        return rank_count / vals.shape[1]

    # ----------------------------------------------------------------------
    # Cached computations over configurations, replications and sessions
    # ----------------------------------------------------------------------

    def compute_metrics(self, metric_type="poi", force=False):
        """Compute and cache the PWI ('poi') or RAI ('rai') matrix of every saved state.

        Writes ``pois/<config>/table_<i>/<j>.npy`` or ``rais/<config>/table_<i>/<j>.npy``;
        states that do not exist are skipped silently.

        Parameters
        ----------
        metric_type : {'poi', 'rai'}
            'poi' for PWI (``calc_poi``), 'rai' for RAI (``calc_rai``).
        force : bool
            Recompute files that already exist.
        """
        if metric_type == "poi":
            metric_name = "pois"
            func = self.calc_poi
        elif metric_type == "rai":
            metric_name = "rais"
            func = self.calc_rai

        out_fold = os.path.join(self.sub_tests_fold, metric_name)
        save_path(out_fold)

        for f1 in self.F1:
            for f2 in self.F2:
                for num_steps in self.NUM_STEPS_LIST:
                    print(f"Calculating {metric_name} for f1 = {f1}, f2 = {f2}, "
                          f"num_steps = {num_steps} table:")
                    tables, _, _, _ = read_dataset(self.dataset_fold, f1, f2, num_steps)

                    for i in tqdm(range(self.hm)):
                        table_dir = os.path.join(out_fold, f"f1_{f1}_f2_{f2}_nsteps_{num_steps}",
                                                 f"table_{i}")
                        save_path(table_dir)

                        transformer = PiecewiseLinearTransformer.from_equal_intervals(
                            tables[i], self.num_subint)
                        t_vec = transformer.transform(tables[i])

                        for j in range(1, num_steps + 1):
                            path_std = os.path.join(table_dir, f"{j}.npy")

                            if os.path.exists(path_std) and not force:
                                continue

                            try:
                                sam = np.load(self._sample_path(f1, f2, num_steps, i, j))
                                np.save(path_std, func(f1, t_vec, sam))
                            except FileNotFoundError:
                                pass

    def compute_aios(self, force=False):
        """AIO, eq. (25): RAI(a*, 1) of the true best alternative a*, after every session.

        Reads the cached RAI matrices and writes ``aios/<config>/aios_<j>.npy`` with one
        value per replication. Replications are read in order and the vector ends at the
        first missing RAI file (nothing is written if the first one is missing).
        """
        test_name = "aios"
        out_fold = os.path.join(self.sub_tests_fold, test_name)
        rais_fold = os.path.join(self.sub_tests_fold, "rais")
        save_path(out_fold)

        for f1 in self.F1:
            for f2 in self.F2:
                for num_steps in self.NUM_STEPS_LIST:
                    print(f"Calculating {test_name} for f1 = {f1}, f2 = {f2}, "
                          f"num_steps = {num_steps} sessions:")
                    _, rankings, _, _ = read_dataset(self.dataset_fold, f1, f2, num_steps)
                    sub_out = os.path.join(out_fold, f"f1_{f1}_f2_{f2}_nsteps_{num_steps}")
                    save_path(sub_out)

                    for j in tqdm(range(1, num_steps + 1)):
                        p_std = os.path.join(sub_out, f"{test_name}_{j}.npy")
                        if os.path.exists(p_std) and not force: continue

                        res_std = []
                        for i in range(self.hm):
                            try:
                                r_path = os.path.join(
                                    rais_fold, f"f1_{f1}_f2_{f2}_nsteps_{num_steps}",
                                    f"table_{i}", f"{j}.npy")
                                rai = np.load(r_path)
                                best_idx = np.argwhere(rankings[i] == 0)[0, 0]
                                res_std.append(rai[best_idx, 0])
                            except (FileNotFoundError, IndexError):
                                break

                        if len(res_std) > 0:
                            np.save(p_std, res_std)

    def compute_asps(self, force=False):
        """ASP, eq. (24), after every session.

        The mean of PWI(a, b) over the |A| (|A| - 1) / 2 pairs in which a is truly better
        than b (true ranks of the dataset). Reads the cached PWI matrices and writes
        ``asps/<config>/asps_<j>.npy``, one value per replication, ending at the first
        missing PWI file.
        """
        test_name = "asps"
        out_fold = os.path.join(self.sub_tests_fold, test_name)
        pois_fold = os.path.join(self.sub_tests_fold, "pois")
        save_path(out_fold)

        for f1 in self.F1:
            for f2 in self.F2:
                for num_steps in self.NUM_STEPS_LIST:
                    print(f"Calculating {test_name} for f1 = {f1}, f2 = {f2}, "
                          f"num_steps = {num_steps} sessions:")
                    _, rankings, _, _ = read_dataset(self.dataset_fold, f1, f2, num_steps)
                    sub_out = os.path.join(out_fold, f"f1_{f1}_f2_{f2}_nsteps_{num_steps}")
                    save_path(sub_out)

                    for j in tqdm(range(1, num_steps + 1)):
                        p_std = os.path.join(sub_out, f"{test_name}_{j}.npy")
                        if os.path.exists(p_std) and not force: continue

                        res_std = []
                        for i in range(self.hm):
                            try:
                                poi_path = os.path.join(
                                    pois_fold, f"f1_{f1}_f2_{f2}_nsteps_{num_steps}",
                                    f"table_{i}", f"{j}.npy")
                                poi = np.load(poi_path)
                                # pref_mask[a, b] = 1 when b is truly better than a, so the
                                # sum collects PWI(better, worse) over all truly ordered pairs
                                pref_mask = rankings[i][:, None] > rankings[i][None, :]
                                coef = 2 / (f1 * (f1 - 1))
                                res_std.append(np.sum(poi.T * pref_mask) * coef)
                            except FileNotFoundError:
                                break

                        if len(res_std) > 0:
                            np.save(p_std, res_std)

    def compute_asrs(self, force=False):
        """ASR, eq. (23): mean over the alternatives of RAI(a, rank_true(a)), per session.

        Reads the cached RAI matrices and writes ``asrs/<config>/asrs_<j>.npy``, one value
        per replication, ending at the first missing RAI file.
        """
        test_name = "asrs"
        out_fold = os.path.join(self.sub_tests_fold, test_name)
        rais_fold = os.path.join(self.sub_tests_fold, "rais")
        save_path(out_fold)

        for f1 in self.F1:
            for f2 in self.F2:
                for num_steps in self.NUM_STEPS_LIST:
                    print(f"Calculating {test_name} for f1 = {f1}, f2 = {f2}, "
                          f"num_steps = {num_steps} sessions:")
                    _, rankings, _, _ = read_dataset(self.dataset_fold, f1, f2, num_steps)
                    sub_out = os.path.join(out_fold, f"f1_{f1}_f2_{f2}_nsteps_{num_steps}")
                    save_path(sub_out)

                    for j in tqdm(range(1, num_steps + 1)):
                        p_std = os.path.join(sub_out, f"{test_name}_{j}.npy")
                        if os.path.exists(p_std) and not force: continue

                        res_std = []
                        for i in range(self.hm):
                            try:
                                r_path = os.path.join(
                                    rais_fold, f"f1_{f1}_f2_{f2}_nsteps_{num_steps}",
                                    f"table_{i}", f"{j}.npy")
                                rai = np.load(r_path)
                                current_rank = rankings[i]
                                val_std = np.mean([
                                    rai[np.argwhere(current_rank == k)[0, 0], k]
                                    for k in range(f1)
                                ])
                                res_std.append(val_std)
                            except (FileNotFoundError, IndexError):
                                break

                        if len(res_std) > 0:
                            np.save(p_std, res_std)

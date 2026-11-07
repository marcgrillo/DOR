import os
import numpy as np
from tqdm import tqdm
import itertools
from bdor.utils import save_path, read_dataset, robust_sigmoid, parse_subfold_string
from bdor.paths import DATASETS_DIR, SAMPLES_DIR, METRICS_DIR
from bdor.features import PiecewiseLinearTransformer

class BenchmarkRunner:
    """
    Orchestrates batch experiments for POI, RAI, AIOS, ASPS, ASRS.
    """

    def __init__(self, dataset_fold, sub_fold, num_subint, hm, F1, F2, NUM_STEPS_LIST, eta_max=30):
        self.dataset_fold = os.path.join(DATASETS_DIR, dataset_fold)
        self.sub_fold = sub_fold
        self.samples_fold = os.path.join(SAMPLES_DIR, dataset_fold)
        self.tests_fold = os.path.join(METRICS_DIR, dataset_fold)
        self.sub_tests_fold = os.path.join(self.tests_fold, sub_fold)
        
        self.num_subint = num_subint
        
        # Check available HM in dataset
        f1_0, f2_0, nsteps_0 = F1[0], F2[0], NUM_STEPS_LIST[0]
        try:
            tables, _, _, _ = read_dataset(self.dataset_fold, f1_0, f2_0, nsteps_0)
            self.hm = min(hm, len(tables))
        except Exception:
            self.hm = hm # Fallback

        self.F1, self.F2, self.NUM_STEPS_LIST = F1, F2, NUM_STEPS_LIST
        self.eta_max = eta_max
        
        save_path(self.sub_tests_fold)

    def _sample_path(self, f1, f2, num_steps, i, j):
        """Construct path to sample files."""
        config_dir = os.path.join(self.samples_fold, f"f1_{f1}_f2_{f2}_nsteps_{num_steps}")
        
        path_specific = os.path.join(config_dir, self.sub_fold, f"table_{i}", f"{j}.npy")
        if os.path.exists(path_specific):
            return path_specific
            
        return path_specific

    # ----------------------------------------------------------------------
    # Metrics (POI / RAI)
    # ----------------------------------------------------------------------

    def calc_poi(self, f1, t, sam):
        """Preference Ordering Index: pairwise dominance probabilities."""
        rank_count = np.zeros((f1, f1))
        sam = np.atleast_2d(sam)
        # Note: In DOR, the probability is 0.5 * (1 + sigma(u_a) - sigma(u_b))
        # Wait, for POI, we need to know the probability that a dominates b.
        # we can just use the utilities directly to rank!
        w = sam[:, :t.shape[1]]
        vals = t @ w.T 
        
        for col in range(vals.shape[1]):
            scores = vals[:, col]
            rank_count += (scores[:, None] >= scores[None, :])
            
        return rank_count / vals.shape[1]

    def calc_rai(self, f1, t, sam):
        rank_count = np.zeros((f1, f1))
        sam = np.atleast_2d(sam)
        w = sam[:, :t.shape[1]]
        vals = t @ w.T
        
        for col in range(vals.shape[1]):
            scores = vals[:, col]
            ranks = np.argsort(np.argsort(-scores))
            rank_count[np.arange(f1), ranks] += 1
            
        return rank_count / vals.shape[1]
    
    def calc_perc_inc(self, f1, t, sam, true_ranking):
        """Percentage of Inconsistency: frequency of inconsistent rankings."""
        alg_name, active_method_name = parse_subfold_string(self.sub_fold)
        algo_type, model_type = alg_name.split('-')
        all_indices = np.arange(f1)
        possible_pairs = list(itertools.combinations(all_indices, 2))
        candidates_unordered = [p for p in possible_pairs]
        candidates = []
        for cand in candidates_unordered:
            rank_a = true_ranking[cand[0]]
            rank_b = true_ranking[cand[1]]
            if rank_a < rank_b:
                correct_pair = np.array([cand[0], cand[1]])
            else:
                correct_pair = np.array([cand[1], cand[0]])
            candidates.append(correct_pair)

        idx_a = [c[0] for c in candidates]
        idx_b = [c[1] for c in candidates]
        
        sam = np.atleast_2d(sam)
        w = sam[:, :t.shape[1]]
        u_a = t[idx_a] @ w.T
        u_b = t[idx_b] @ w.T

        # DOR Model Probabilities:
        probs = 0.5 * (1 + robust_sigmoid(u_a) - robust_sigmoid(u_b))
        p_mean= np.mean(probs, axis=1)

        gotem = np.zeros(len(p_mean))
        gotem[ np.random.rand(len(p_mean)) > p_mean ] = 1
            
        return np.sum(gotem)/len(gotem)

    # ----------------------------------------------------------------------
    # Main Computation Loop
    # ----------------------------------------------------------------------

    def compute_metrics(self, metric_type="poi", force=False):
        """Generic loop to compute POI or RAI and save to disk."""
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
                    print(f"Calculating {metric_name} for f1 = {f1}, f2 = {f2}, num_steps = {num_steps} table:")
                    tables, _, _, _ = read_dataset(self.dataset_fold, f1, f2, num_steps)
                    
                    for i in tqdm( range(self.hm) ):
                        table_dir = os.path.join(out_fold, f"f1_{f1}_f2_{f2}_nsteps_{num_steps}", f"table_{i}")
                        save_path(table_dir)

                        transformer = PiecewiseLinearTransformer.from_equal_intervals(tables[i], self.num_subint)
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

    def compute_perc_inc(self, force=False):
        metric_name = "perc_inc"
        func = self.calc_perc_inc
        
        out_fold = os.path.join(self.sub_tests_fold, metric_name)
        save_path(out_fold)

        for f1 in self.F1:
            for f2 in self.F2:
                for num_steps in self.NUM_STEPS_LIST:
                    print(f"Calculating {metric_name} for f1 = {f1}, f2 = {f2}, num_steps = {num_steps} dm_pref_num:")
                    out_dir = os.path.join(out_fold, f"f1_{f1}_f2_{f2}_nsteps_{num_steps}")
                    save_path(out_dir)
                    tables, rankings, _, _ = read_dataset(self.dataset_fold, f1, f2, num_steps)
                    for j in tqdm( range(1, num_steps + 1) ):
                        path_std = os.path.join(out_dir, f"perc_inc_{j}.npy")
                        if os.path.exists(path_std) and not force: continue
                        perc_inc_std = []
                        for i in range(self.hm):
                            transformer = PiecewiseLinearTransformer.from_equal_intervals(tables[i], self.num_subint)
                            t_vec = transformer.transform(tables[i])
                            try:
                                sam = np.load(self._sample_path(f1, f2, num_steps, i, j))
                                perc_inc_std.append( func(f1, t_vec, sam, rankings[i]) )
                            except FileNotFoundError:
                                pass
                        np.save(path_std, perc_inc_std )

    def compute_aios(self, force=False):
        test_name = "aios"
        out_fold = os.path.join(self.sub_tests_fold, test_name)
        rais_fold = os.path.join(self.sub_tests_fold, "rais")
        save_path(out_fold)

        for f1 in self.F1:
            for f2 in self.F2:
                for num_steps in self.NUM_STEPS_LIST:
                    print(f"Calculating {test_name} for f1 = {f1}, f2 = {f2}, num_steps = {num_steps} dm_pref_num:")
                    _, rankings, _, _ = read_dataset(self.dataset_fold, f1, f2, num_steps)
                    sub_out = os.path.join(out_fold, f"f1_{f1}_f2_{f2}_nsteps_{num_steps}")
                    save_path(sub_out)

                    for j in tqdm( range(1, num_steps + 1) ):
                        p_std = os.path.join(sub_out, f"{test_name}_{j}.npy")
                        if os.path.exists(p_std) and not force: continue

                        res_std = []
                        for i in range(self.hm):
                            try:
                                r_path = os.path.join(rais_fold, f"f1_{f1}_f2_{f2}_nsteps_{num_steps}", f"table_{i}", f"{j}.npy")
                                rai = np.load(r_path)
                                best_idx = np.argwhere(rankings[i] == 0)[0, 0]
                                res_std.append(rai[best_idx, 0])
                            except (FileNotFoundError, IndexError):
                                break
                        
                        if len(res_std) > 0:
                            np.save(p_std, res_std)

    def compute_asps(self, force=False):
        test_name = "asps"
        out_fold = os.path.join(self.sub_tests_fold, test_name)
        pois_fold = os.path.join(self.sub_tests_fold, "pois")
        save_path(out_fold)

        for f1 in self.F1:
            for f2 in self.F2:
                for num_steps in self.NUM_STEPS_LIST:
                    print(f"Calculating {test_name} for f1 = {f1}, f2 = {f2}, num_steps = {num_steps} dm_pref_num:")
                    _, rankings, _, _ = read_dataset(self.dataset_fold, f1, f2, num_steps)
                    sub_out = os.path.join(out_fold, f"f1_{f1}_f2_{f2}_nsteps_{num_steps}")
                    save_path(sub_out)

                    for j in tqdm( range(1, num_steps + 1) ):
                        p_std = os.path.join(sub_out, f"{test_name}_{j}.npy")
                        if os.path.exists(p_std) and not force: continue

                        res_std = []
                        for i in range(self.hm):
                            try:
                                poi_path = os.path.join(pois_fold, f"f1_{f1}_f2_{f2}_nsteps_{num_steps}", f"table_{i}", f"{j}.npy")
                                poi = np.load(poi_path)
                                pref_mask = rankings[i][:, None] > rankings[i][None, :]
                                coef = 2 / (f1 * (f1 - 1))
                                res_std.append(np.sum(poi.T * pref_mask) * coef)
                            except FileNotFoundError:
                                break
                        
                        if len(res_std) > 0:
                            np.save(p_std, res_std)

    def compute_asrs(self, force=False):
        test_name = "asrs"
        out_fold = os.path.join(self.sub_tests_fold, test_name)
        rais_fold = os.path.join(self.sub_tests_fold, "rais")
        save_path(out_fold)

        for f1 in self.F1:
            for f2 in self.F2:
                for num_steps in self.NUM_STEPS_LIST:
                    print(f"Calculating {test_name} for f1 = {f1}, f2 = {f2}, num_steps = {num_steps} dm_pref_num:")
                    _, rankings, _, _ = read_dataset(self.dataset_fold, f1, f2, num_steps)
                    sub_out = os.path.join(out_fold, f"f1_{f1}_f2_{f2}_nsteps_{num_steps}")
                    save_path(sub_out)

                    for j in tqdm( range(1, num_steps + 1) ):
                        p_std = os.path.join(sub_out, f"{test_name}_{j}.npy")
                        if os.path.exists(p_std) and not force: continue

                        res_std = []
                        for i in range(self.hm):
                            try:
                                r_path = os.path.join(rais_fold, f"f1_{f1}_f2_{f2}_nsteps_{num_steps}", f"table_{i}", f"{j}.npy")
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
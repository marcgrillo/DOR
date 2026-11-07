"""Synthetic decision makers and their elicitation sessions (paper Sec. 5.1; ties: Sec. 5.6).

``DatasetGenerator.generate_batch`` writes the datasets the simulation studies run on
(``bdor.runner`` reads them back through ``bdor.utils.read_dataset``). One replication
("run") of a configuration is generated as follows.

1. Performance table: g_j(a_i) ~ U[0, 1] independently, shape (|A|, m).
2. True value function (``utility_type='exponential'``, the paper's): additive, with
   marginals u_j(x) = w_j (1 - exp(-c_j x)) / (1 - exp(-c_j)), weights
   w ~ Dirichlet(1, ..., 1) and curvatures c_j ~ U[-10, 10], so U_true lies in [0, 1].
   The true ranks sort U_true in decreasing order (0 = best).
3. Session t shows a subset S_t drawn without replacement by
   ``np.random.default_rng(i * 1000 + t)`` for run i: ``k_subset[0]`` alternatives at
   t = 1 and ``k_subset[-1]`` afterwards (an int ``k_subset``: that size every session).
   The subsets therefore depend only on (run, session, |A|, subset size).
4. Thurstonian noise, eq. (21): the DM perceives e_max * U_true(a) + eps_a with
   eps_a ~ N(0, sigma^2 / 2) independently, so a perceived difference has variance sigma^2.
5. Levels: with ``merge_threshold`` tau > 0, the subset sorted by U_true is cut wherever
   two consecutive alternatives differ by at least tau in U_true, so closer neighbours
   share a level (single linkage on the true utilities, Sec. 5.6). tau = 0 gives singleton
   levels (no ties), the setting of every study except Sec. 5.6.
6. The DM orders the levels by the mean perceived value of their members and places
   min(e_max, round(difference)) blank cards between consecutive levels.

The noise scale sigma is set once per (|A|, m) by ``_tune_sigma``: brentq solves
E_mu[q(mu, sigma)] = target_consistency = 1 - Finc, eq. (22), where q(mu, sigma) is the
probability that the perceived card count equals the true one, round(mu), and mu runs over
the e_max-scaled true gaps between consecutive alternatives of a random permutation, on a
Monte Carlo calibration sample of 1000 tables. target_consistency >= 0.999 gives sigma = 0.

IMPORTANT, reproducibility: apart from the session subsets, every draw (tables, value
functions, noise and the sigma calibration) uses the GLOBAL NumPy random state
(``np.random.*`` and ``scipy.stats.dirichlet.rvs``). Call ``np.random.seed`` before
``generate_batch`` to make a dataset reproducible, as the ties and e_max study scripts do.
The grid datasets ``data/datasets/grid_*`` were generated (June 2026) without a seed and
cannot be regenerated bit for bit: never overwrite them.

Output, one file per configuration, ``<output_dir>/f1_<|A|>__f2_<m>__nsteps_<T>.json``::

    metadata   f1, f2, num_steps, k_subset, n_max (= e_max), target_consistency,
               utility_type, sigma (e_max-scaled axis), n_runs, merge_threshold
    runs[i]    run_id, table (|A| x m), true_ranks (0 = best), true_utilities (in [0, 1]),
               weights (w), params (c_j; null for 'piecewise_linear'), subranks
    runs[i].subranks[t-1]   session t, best to worst:
               levels   list of lists of alternative indices (tied alternatives share one)
               items    flat concatenation of the levels
               cards    the len(levels) - 1 card counts between consecutive levels
    _description            the same layout in words

Files written before ties were introduced (the grid datasets) have no 'levels' and no
'merge_threshold'. With ``plot_utilities=True`` two diagnostic PNGs of run 0 are written
next to the JSON file.
"""
import os
import json
import numpy as np
from scipy.stats import dirichlet, norm
from scipy.optimize import brentq
import matplotlib
matplotlib.use('Agg')  # non-interactive backend: the diagnostic plots are only saved
import matplotlib.pyplot as plt
from tqdm import tqdm
from bdor import utils
from bdor.paths import DATASETS_DIR
from bdor.features import PiecewiseLinearTransformer


class DatasetGenerator:
    """Generator of the synthetic datasets of Sec. 5.1 (process described in the module).

    Parameters
    ----------
    output_dir : str
        Dataset folder, created on construction if missing. The default
        (``data/datasets/default_dataset``) is a placeholder: the study scripts pass their
        own folder.
    n_runs : int
        Replications (table, true value function and sessions) per configuration.

    Notes
    -----
    Draws from the global NumPy random state: seed ``np.random`` before ``generate_batch``
    for a reproducible dataset.
    """

    def __init__(self, output_dir=os.path.join(DATASETS_DIR, "default_dataset"), n_runs=200):
        self.output_dir = output_dir
        self.n_runs = n_runs
        utils.save_path(self.output_dir)

    def _compute_rank(self, U):
        """Ranks by decreasing U (0 = best), as a list."""
        b = np.argsort(-U)
        ranks = np.empty_like(b)
        ranks[b] = np.arange(len(U))
        return ranks.tolist()

    def _generate_exponential_utility(self, data_matrix, f2):
        """True exponential value function of Sec. 5.1.

        Returns U (|A|,), the Dirichlet(1) weights w (m,) and the curvatures c (m,),
        c_j ~ U[-10, 10]; values within 1e-5 of 0 are set to 1e-5 to avoid 0 / 0.
        """
        w = dirichlet.rvs(alpha=np.ones(f2))[0]
        c = np.random.uniform(-10, 10, f2)
        c[np.abs(c) < 1e-5] = 1e-5  # avoid c = 0

        term1 = 1 - np.exp(-c * data_matrix)
        term2 = 1 - np.exp(-c)
        u_j = w * (term1 / term2)
        return np.sum(u_j, axis=1), w, c

    def _generate_piecewise_linear_utility(self, data_matrix, f2, num_intervals=3):
        """Alternative true model, not used in the paper.

        Dirichlet(1) weights on the equal-interval piecewise-linear features of the table.
        Returns U (|A|,), w (m * num_intervals,) and the transformer.
        """
        transformer = PiecewiseLinearTransformer.from_equal_intervals(data_matrix, num_intervals)
        features = transformer.transform(data_matrix)

        # Features dim: (N, total_params). total_params = f2 * num_intervals
        total_params = features.shape[1]

        # Weights
        w = dirichlet.rvs(alpha=np.ones(total_params))[0]

        # Utility U = X_transformed @ w
        U = features @ w
        return U, w, transformer

    def _plot_marginal_utilities(self, utility_type, f2, weights, params, save_path):
        """Plot the true marginal value functions of one run, one panel per criterion."""
        x_plot = np.linspace(0, 1, 100)

        fig, axes = plt.subplots(1, f2, figsize=(5 * f2, 4))
        if f2 == 1: axes = [axes]

        for j in range(f2):
            w_j = weights[j] if utility_type == 'exponential' else 1.0  # PWL: weights per segment

            if utility_type == 'exponential':
                # exponential: u_j(x) = w_j * (1 - exp(-c_j * x)) / (1 - exp(-c_j))
                c_j = params[j]
                y_plot = w_j * (1 - np.exp(-c_j * x_plot)) / (1 - np.exp(-c_j))
                title = f"Crit {j+1} (Exp, w={w_j:.2f}, c={c_j:.2f})"
            elif utility_type == 'piecewise_linear':
                # piecewise: params is transformer
                transformer = params

                col_end = np.cumsum(transformer.n_w)
                col_start = np.insert(col_end[:-1], 0, 0)

                start_idx = int(col_start[j])
                end_idx = int(col_end[j])

                w_sub = weights[start_idx:end_idx]
                breakpoints = transformer.ch_p[j]  # [min, ..., max]

                x_domain = np.linspace(breakpoints[0], breakpoints[-1], 200)
                y_vals = []
                for x_val in x_domain:
                    val_j = 0
                    for k in range(len(w_sub)):
                        bp_low = breakpoints[k]
                        bp_high = breakpoints[k+1]
                        feat = np.clip((x_val - bp_low) / (bp_high - bp_low), 0, 1)
                        val_j += w_sub[k] * feat
                    y_vals.append(val_j)

                x_plot = x_domain
                y_plot = np.array(y_vals)
                title = f"Crit {j+1} (PWL, imp={np.sum(w_sub):.2f})"

            ax = axes[j]
            ax.plot(x_plot, y_plot, linewidth=2)
            ax.set_title(title)
            ax.set_xlabel("Attribute Value")
            ax.set_ylabel("Marginal Utility")
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def _calculate_expected_consistency(self, sigma, utility_diffs, n_max):
        """Expected share of exact card counts, E_mu[q(mu, sigma)] of eq. (22).

        For a true gap mu = n_max * utility_diff the perceived gap is v ~ N(mu, sigma^2)
        and q(mu, sigma) = P(round(v) = round(mu)). Returns the mean of q over
        ``utility_diffs`` (1 when sigma <= 0).
        """
        if sigma <= 0:
            return 1.0

        mu = utility_diffs * n_max
        c = np.round(mu)
        # P(c - 0.5 <= v < c + 0.5) where v ~ N(mu, sigma^2)
        probs = norm.cdf((c + 0.5 - mu) / sigma) - norm.cdf((c - 0.5 - mu) / sigma)
        return np.mean(probs)

    def _tune_sigma(self, f1, f2, target_consistency, n_max, num_intervals=3,
                    utility_type='exponential'):
        """Noise scale sigma (e_max-scaled axis) solving eq. (22) for an |A| x m problem.

        The calibration sample is 1000 new tables with true value functions of the same
        family, drawn from the global NumPy random state; each contributes the |A| - 1
        gaps between consecutive alternatives of a random permutation. brentq then solves
        E_mu[q(mu, sigma)] = target_consistency on [0.001, 100 n_max], returning n_max if
        it fails. ``target_consistency >= 0.999`` returns 0 without drawing anything.
        ``scripts/simulation/run_emax_flipmatch.py`` overrides this method.
        """
        if target_consistency >= 0.999: return 0.0

        # Calibration sample: gaps between randomly paired alternatives
        calibration_size = 1000
        diffs = []
        for _ in range(calibration_size):
            ls = np.random.rand(f1, f2)
            if utility_type == 'piecewise_linear':
                U, _, _ = self._generate_piecewise_linear_utility(ls, f2, num_intervals)
            else:
                U, _, _ = self._generate_exponential_utility(ls, f2)
            idx = np.arange(f1)
            np.random.shuffle(idx)
            u_shuffled = U[idx]
            d = np.abs(u_shuffled[:-1] - u_shuffled[1:])
            diffs.append(d)

        utility_diffs = np.concatenate(diffs)

        def objective(sigma):
            return (self._calculate_expected_consistency(sigma, utility_diffs, n_max)
                    - target_consistency)

        try:
            return brentq(objective, 0.001, 100.0 * n_max)
        except ValueError:
            return 1.0 * n_max  # fallback: no sign change on the bracket

    def generate_batch(self, F1, F2, num_steps, k_subset, n_max, target_consistency=1.0,
                       utility_type='exponential', num_intervals=3, plot_utilities=True,
                       merge_threshold=0.0):
        """Write one dataset file per (|A|, m) configuration.

        For each f1 in ``F1`` and f2 in ``F2`` (in this order) it calibrates sigma, draws
        ``n_runs`` replications and writes
        ``<output_dir>/f1_<f1>__f2_<f2>__nsteps_<num_steps>.json``; the process and the
        file layout are described in the module docstring.

        Parameters
        ----------
        F1, F2 : iterable of int
            Numbers of alternatives |A| and of criteria m.
        num_steps : int
            Sessions per replication.
        k_subset : int or tuple of int
            Subset size. A pair (k_1, k) shows k_1 alternatives at session 1 and k at the
            later ones (the grid's AR and K); in general session 1 uses ``k_subset[0]``
            and the others ``k_subset[-1]``.
        n_max : int
            e_max: maximum number of blank cards per gap, and scale of the perceived
            values.
        target_consistency : float
            1 - Finc, the expected share of exact card counts of eq. (22); >= 0.999 means
            a noise-free DM.
        utility_type : {'exponential', 'piecewise_linear'}
            Family of the true value function; the paper uses 'exponential'.
        num_intervals : int
            Intervals per criterion of the 'piecewise_linear' true model.
        plot_utilities : bool
            Also write the two diagnostic PNGs of run 0 (every study script passes False).
        merge_threshold : float
            Tie threshold tau of Sec. 5.6, on the [0, 1] scale of U_true; 0 means no ties.
        """
        print(f"Generating datasets in '{self.output_dir}' with Consistency={target_consistency} "
              f"(Utility={utility_type})...")

        for f1 in tqdm(F1, desc="Alternatives (F1)"):
            for f2 in F2:
                # --- Step 1: calibrate sigma (eq. (22)) ---
                sigma = self._tune_sigma(f1, f2, target_consistency, n_max,
                                         num_intervals=num_intervals, utility_type=utility_type)

                dataset_json = {
                    "_description": {
                        "metadata": "Parameters used to generate this dataset.",
                        "runs": "List of independent runs (Human Models).",
                        "runs[].table": "Feature matrix for the alternatives (f1 x f2).",
                        "runs[].true_ranks": "Ground truth ranks of the alternatives (0 is best).",
                        "runs[].true_utilities": "Ground truth utility values for each alternative.",
                        "runs[].weights": "Weights used in the utility function.",
                        "runs[].params": "Optional parameters (e.g., exponential curvature).",
                        "runs[].subranks": (
                            "Sequence of elicited subranks. 'levels' is the declared complete "
                            "preorder, best to worst: a list of SETS of mutually indifferent "
                            "alternatives (DOR levels; alternatives whose true-utility gap is "
                            "below merge_threshold share a level). 'items' is the flat "
                            "concatenation of the levels. 'cards' is blank cards between "
                            "consecutive levels.")
                    },
                    "metadata": {
                        "f1": f1,
                        "f2": f2,
                        "num_steps": num_steps,
                        "k_subset": k_subset,
                        "n_max": n_max,
                        "target_consistency": target_consistency,
                        "utility_type": utility_type,
                        "sigma": sigma,
                        "n_runs": self.n_runs,
                        "merge_threshold": merge_threshold
                    },
                    "runs": []
                }

                for i in range(self.n_runs):
                    subranks_plot_data = []
                    ls = np.random.rand(f1, f2)
                    if utility_type == 'piecewise_linear':
                        U, w, params = self._generate_piecewise_linear_utility(ls, f2, num_intervals)
                        params_val = None  # the transformer is not JSON serializable
                    else:
                        U, w, params = self._generate_exponential_utility(ls, f2)
                        params_val = params.tolist()

                    rk = self._compute_rank(U)

                    subranks = []

                    for j in range(1, num_steps + 1):
                        # Subset shown at session j (own seeded generator, see module docstring)
                        if isinstance(k_subset, (list, tuple)):
                            current_k = k_subset[0] if j == 1 else k_subset[-1]
                        else:
                            current_k = k_subset
                        rng_subset = np.random.default_rng(i * 1000 + j)
                        subset_indices = rng_subset.choice(f1, current_k, replace=False)

                        # Thurstonian noise, eq. (21): the difference of two perceived values
                        # must have variance sigma^2, so each value gets variance sigma^2 / 2
                        noise_std = sigma / np.sqrt(2)
                        noisy_utilities = (U[subset_indices] * n_max
                                           + np.random.normal(0, noise_std, current_k))

                        # --- DOR levels (sets of indifferent alternatives, Sec. 5.6) ---
                        # Single-linkage merge of consecutive (true-utility-sorted)
                        # alternatives whose TRUE utility gap is < merge_threshold. All the
                        # random draws are made above, so merge_threshold changes only the
                        # grouping; merge_threshold = 0 gives singleton levels.
                        true_order = subset_indices[np.argsort(-U[subset_indices])]
                        levels = [[int(true_order[0])]]
                        for prev, cur in zip(true_order[:-1], true_order[1:]):
                            if (U[prev] - U[cur]) < merge_threshold:
                                levels[-1].append(int(cur))
                            else:
                                levels.append([int(cur)])

                        # The DM declares the ORDER of the levels and the cards from the
                        # perceived level values = mean of the members' noisy utilities,
                        # matching the level-mean features of eq. (8).
                        noisy_map = {int(a): noisy_utilities[pos]
                                     for pos, a in enumerate(subset_indices)}
                        level_vals = np.array([np.mean([noisy_map[a] for a in lv]) for lv in levels])
                        lv_order = np.argsort(-level_vals)
                        ranked_levels = [levels[o] for o in lv_order]
                        ranked_vals = level_vals[lv_order]
                        ranked_items = [a for lv in ranked_levels for a in lv]

                        # Cards between consecutive LEVELS: min(e_max, round(difference))
                        cards = [int(np.clip(np.round(ranked_vals[m] - ranked_vals[m+1]), 0, n_max))
                                 for m in range(len(ranked_levels) - 1)]

                        subranks.append({
                            "items": ranked_items,
                            "levels": ranked_levels,
                            "cards": cards
                        })

                        # Data for the diagnostic plot of the first run
                        if i == 0 and plot_utilities:
                            step_plot_data = {
                                "subset": subset_indices.tolist(),
                                "true_u": (U[subset_indices] * n_max).tolist(),
                                "noisy_u": noisy_utilities.tolist(),
                                "true_items_sorted":
                                    subset_indices[np.argsort(-U[subset_indices])].tolist(),
                                "noisy_items_sorted": ranked_items,
                                "cards": cards
                            }
                            subranks_plot_data.append(step_plot_data)

                    run_data = {
                        "run_id": i,
                        "table": ls.tolist(),
                        "true_ranks": rk,
                        "true_utilities": U.tolist(),
                        "weights": w.tolist(),
                        "params": params_val,
                        "subranks": subranks
                    }
                    dataset_json["runs"].append(run_data)

                    # Diagnostic plots for the first run
                    if i == 0 and plot_utilities:
                        base_name = f"f1_{f1}__f2_{f2}__nsteps_{num_steps}"
                        plot_path = os.path.join(self.output_dir, f"{base_name}_utils_table0.png")
                        self._plot_marginal_utilities(utility_type, f2, w, params, plot_path)

                        # True vs perceived values of every session
                        self._plot_subranks_comparison(
                            subranks_plot_data, n_max,
                            os.path.join(self.output_dir, f"{base_name}_subranks_table0.png"))

                # Save config JSON
                base_name = f"f1_{f1}__f2_{f2}__nsteps_{num_steps}"
                json_path = os.path.join(self.output_dir, f"{base_name}.json")
                with open(json_path, 'w', newline='\n') as f:
                    json.dump(dataset_json, f, indent=2)
                print(f"Dataset saved to {json_path}")

    def _plot_subranks_comparison(self, plot_data, n_max, save_path):
        """Plot true vs perceived values of each session's subset, one panel per session."""
        num_steps = len(plot_data)
        fig, axes = plt.subplots(num_steps, 1, figsize=(10, 2.5 * num_steps), sharex=False)
        if num_steps == 1: axes = [axes]

        for j, (step_data, ax) in enumerate(zip(plot_data, axes)):
            subset = step_data['subset']
            true_u = step_data['true_u']
            noisy_u = step_data['noisy_u']

            # Normalize X axis per step for better visibility
            all_vals = true_u + noisy_u
            x_min, x_max = min(all_vals) - 0.5, max(all_vals) + 0.5

            # Plot levels
            ax.axhline(1, color='gray', linestyle='--', alpha=0.3)  # Ideal level
            ax.axhline(0, color='gray', linestyle='--', alpha=0.3)  # Noisy level

            colors = plt.cm.get_cmap('tab10', len(subset))

            for idx_in_subset, item_idx in enumerate(subset):
                color = colors(idx_in_subset)
                t_u = true_u[idx_in_subset]
                n_u = noisy_u[idx_in_subset]

                # Ideal point
                ax.scatter(t_u, 1, color=color, s=100, zorder=5)
                ax.annotate(f"Alt {item_idx}", (t_u, 1.1), ha='center', fontweight='bold', color=color)

                # Noisy point
                ax.scatter(n_u, 0, color=color, s=100, zorder=5)
                ax.annotate(f"Alt {item_idx}", (n_u, -0.2), ha='center', color=color)

                # Connection
                ax.plot([t_u, n_u], [1, 0], color=color, linestyle=':', alpha=0.6)

            ax.set_yticks([0, 1])
            ax.set_yticklabels(['Noisy (Sample)', 'Ideal (Truth)'])
            ax.set_title(f"Step {j+1} Subrank Alignment", fontsize=12, pad=20)
            ax.set_xlim(x_min, x_max)
            ax.grid(True, axis='x', alpha=0.2)

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

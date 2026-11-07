"""Elicitation demo on your own data: FTRL-DOR and BAYES-DOR after every answered step.

Reads a small decision problem and the decision maker's DOR answers from data.json (next to
this script), adds the answers to a PreferenceSampler one step (session) at a time and, after
every step, fits FTRL-DOR and runs a short BAYES-DOR chain on all answers given so far, then
plots the inferred marginal value functions. To continue an elicitation, append a step to the
JSON file and run the script again. It is a stand-alone example, not part of the paper pipeline.

Paper:   none (runs Algorithm 1 of Sec. 4 with the estimators of Secs. 3.3-3.4 on user data)
Inputs:  examples/interactive_elicitation/data.json
Outputs: examples/interactive_elicitation/utilities_step_<k>.png, k = 1..number of steps
         (gitignored; written next to the JSON file)
Runtime: under a minute on one CPU core for the shipped data.json (5 steps)

Usage:
    python examples/interactive_elicitation/inference_tool.py

To run it on your own problem, edit data.json, or call run_inference_tool("<path>.json") from
Python to read another file (the PNG files then go next to that file). Format:

    {
      "metadata": {"num_intervals": 4, "n_max": 5},
      "alternatives": [[0.1, 0.5, 0.9], [0.4, 0.4, 0.4], [0.9, 0.1, 0.1], ...],
      "steps": [
        {"items": [0, 1, 2], "cards": [2, 1]},
        {"items": [3, 0, 4], "cards": [1, 3]},
        ...
      ]
    }

  metadata.num_intervals  linear pieces per criterion (gamma_j, default 3); the breakpoints cut
                          each criterion into equal intervals between its minimum and maximum
                          over "alternatives"
  metadata.n_max          e_max: the largest number of blank cards allowed between two
                          consecutive levels (default 5); other metadata keys, such as
                          "num_criteria", are not read
  alternatives            one row per alternative, one column per criterion. Every criterion is
                          treated as "more is better" (the inferred marginal value functions are
                          non-decreasing), so transform cost criteria first. The plots show
                          x in [0, 1]: scale every criterion to [0, 1] to make them readable (the
                          inference itself works on any scale)
  steps[i].items          alternatives ranked at step i, as 0-based row indices into
                          "alternatives", from best to worst, one alternative per level
  steps[i].cards          blank cards between consecutive levels: len(levels) - 1 integers in
                          [0, n_max]; cards[m] sits between level m and level m + 1
  steps[i].levels         optional, for ties: replaces "items" by a best-to-worst list whose
                          entries are an index or a list of indifferent alternatives, e.g.
                          [[0, 3], 1, [2, 4]] with two card counts. An alternative may appear
                          in only one level. "items" must still be present (it is read, then
                          ignored)

Each pair of consecutive levels is one gap observation (expand_subrank(...,
expansion='adjacent'); a tied level enters through the mean features of its members), and the
steps accumulate: after step k both estimators use the gaps of steps 1..k. After every step the
script fits FTRL-DOR (PreferenceSampler.find_map) and draws 200 BAYES-DOR samples after 200
NUTS warm-up iterations (PreferenceSampler.run_hmc, seed 0), then writes utilities_step_<k>.png
with one panel per criterion: the FTRL-DOR marginal value function (red) and the 5-95%
pointwise band of 100 randomly chosen posterior draws (blue), on the raw (unnormalised) weight
scale. The subset of draws is not seeded, so the band can differ slightly between runs.

(formerly user_inference/inference_tool.py)
"""
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

# Make the repository root importable (`import bdor`) without installing the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from bdor.features import PiecewiseLinearTransformer
from bdor.engine import PreferenceSampler
from bdor.simulation import expand_subrank


def plot_marginal_utilities(transformer, w_map, bayes_samples, step_idx, output_dir):
    """Plot the inferred marginal value functions after one step.

    One panel per criterion, on x in [0, 1]: the FTRL-DOR estimate (red line) and,
    unless ``bayes_samples`` is None, the 5-95% pointwise band of at most 100
    posterior draws picked at random from ``bayes_samples`` (blue). Raw weight
    scale. Saves ``utilities_step_<step_idx>.png`` in ``output_dir``.

    Parameters
    ----------
    transformer : PiecewiseLinearTransformer
        Feature map (breakpoints and number of pieces of every criterion).
    w_map : ndarray, shape (n_params,)
        FTRL-DOR piece weights, criterion after criterion.
    bayes_samples : ndarray, shape (n_draws, n_params), or None
        BAYES-DOR posterior draws of the piece weights.
    step_idx : int
        Step number (1-based), used in the title and in the file name.
    output_dir : str
        Directory of the PNG file.
    """
    f2 = transformer.num_criteria
    x_plot = np.linspace(0, 1, 100)

    fig, axes = plt.subplots(1, f2, figsize=(4 * f2, 4))
    if f2 == 1:
        axes = [axes]

    # Position of each criterion's piece weights in the parameter vector
    col_end = np.cumsum(transformer.n_w)
    col_start = np.insert(col_end[:-1], 0, 0)

    for j in range(f2):
        ax = axes[j]
        start_idx, end_idx = int(col_start[j]), int(col_end[j])
        breakpoints = transformer.ch_p[j]

        # 1. FTRL-DOR estimate
        w_j_map = w_map[start_idx:end_idx]
        y_map = []
        for x_val in x_plot:
            val = 0
            for k in range(len(w_j_map)):
                bp_low, bp_high = breakpoints[k], breakpoints[k+1]
                feat = np.clip((x_val - bp_low) / (bp_high - bp_low), 0, 1)
                val += w_j_map[k] * feat
            y_map.append(val)
        ax.plot(x_plot, y_map, 'r-', linewidth=2, label='FTRL (MAP)')

        # 2. BAYES-DOR band from a random subset of at most 100 draws (global NumPy
        #    RNG, not seeded)
        if bayes_samples is not None:
            all_y_bayes = []
            sample_indices = np.random.choice(len(bayes_samples), min(100, len(bayes_samples)),
                                              replace=False)
            for s_idx in sample_indices:
                sample_w = bayes_samples[s_idx]
                w_j_s = sample_w[start_idx:end_idx]
                y_s = []
                for x_val in x_plot:
                    val = 0
                    for k in range(len(w_j_s)):
                        bp_low, bp_high = breakpoints[k], breakpoints[k+1]
                        feat = np.clip((x_val - bp_low) / (bp_high - bp_low), 0, 1)
                        val += w_j_s[k] * feat
                    y_s.append(val)
                all_y_bayes.append(y_s)

            all_y_bayes = np.array(all_y_bayes)
            y_low = np.percentile(all_y_bayes, 5, axis=0)
            y_high = np.percentile(all_y_bayes, 95, axis=0)
            ax.fill_between(x_plot, y_low, y_high, color='blue', alpha=0.2, label='BAYES (90% CI)')

        ax.set_title(f"Criterion {j+1}")
        ax.set_xlabel("Normalized Attribute")
        ax.set_ylabel("Marginal Utility")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.1, None)
        if j == 0:
            ax.legend()

    plt.suptitle(f"Inferred Utility Functions - Step {step_idx}", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"utilities_step_{step_idx}.png"))
    plt.close()


def run_inference_tool(data_path):
    """Process the steps of a JSON elicitation file and plot the estimates after each one.

    The JSON format is described in the module docstring. The PNG files are written
    to the directory that contains ``data_path``.

    Parameters
    ----------
    data_path : str
        Path of the JSON file.
    """
    with open(data_path, 'r') as f:
        data = json.load(f)

    meta = data['metadata']
    alternatives = np.array(data['alternatives'])
    steps = data['steps']
    n_max = meta.get('n_max', 5)
    num_intervals = meta.get('num_intervals', 3)

    # Equal-width pieces between each criterion's minimum and maximum
    transformer = PiecewiseLinearTransformer.from_equal_intervals(alternatives, num_intervals)
    X_transformed = transformer.transform(alternatives)
    n_params = X_transformed.shape[1]

    # One sampler for the whole run, so the gap observations accumulate over the steps
    sampler = PreferenceSampler(X_transformed, [], n_params, n_max)

    output_dir = os.path.dirname(data_path)
    print(f"Starting User Inference Tool. Results will be saved in: {output_dir}")

    for i, step in enumerate(steps):
        step_idx = i + 1
        print(f"\n--- Processing Step {step_idx} ---")

        items = step['items']
        cards = step['cards']

        # One gap observation per pair of consecutive levels; the optional 'levels'
        # entry (tied alternatives) replaces 'items'
        for upper, lower, eta in expand_subrank(step.get('levels', items), cards,
                                                expansion='adjacent'):
            sampler.add_gap(upper, lower, eta)

        # FTRL-DOR: MAP estimate; keep the weights, drop the threshold increments
        print("Running FTRL (MAP estimation)...")
        w_map_full = sampler.find_map()
        w_map = w_map_full[:n_params]

        # BAYES-DOR: short NUTS run (200 draws after 200 warm-up iterations, seed 0)
        print("Running BAYES (Posterior sampling)...")
        bayes_samples_full = sampler.run_hmc(num_samples=200, num_warmup=200)
        bayes_samples = bayes_samples_full[:, :n_params]

        print(f"Generating plot for Step {step_idx}...")
        plot_marginal_utilities(transformer, w_map, bayes_samples, step_idx, output_dir)

    print("\nInference complete. Check the directory for PNG files.")


if __name__ == "__main__":
    data_file = os.path.join(os.path.dirname(__file__), 'data.json')
    if not os.path.exists(data_file):
        print(f"Error: {data_file} not found.")
    else:
        run_inference_tool(data_file)

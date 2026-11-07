# B-DOR: Bayesian Deck-of-cards-based Ordinal Regression

Code for the paper **"Bayesian Deck-of-cards-based Ordinal Regression with Sequential
Preference Elicitation"** by Marco Grillo and Silvano Zappalà.

The Deck-of-cards-based Ordinal Regression (DOR) infers a value function from a
ranking of reference alternatives in which the decision maker (DM) inserts blank
cards between consecutive levels to express preference intensity. B-DOR turns each
pair of adjacent levels into an ordinal observation (direction plus number of cards)
with a cumulative-link likelihood, and learns the value function over several short
elicitation sessions. The posterior after one session is the prior of the next.

* **BAYES-DOR** samples the posterior with the No-U-Turn sampler (NumPyro) and reports
  rank acceptability (RAI) and pairwise winning (PWI) indices.
* **FTRL-DOR** tracks the maximum a posteriori estimate by constrained convex
  optimisation (L-BFGS-B with exact JAX gradients).
* Baselines: **DOR** (linear programming point estimate) and **SMAA-DOR** (Hit-and-Run
  sample of the compatible value functions), and direction-only versions of both
  proposed algorithms (**FTRLdir**, **BAYESdir**).

The repository contains the library, every script behind the paper's simulation
study (Sec. 5) and illustrative example (Sec. 6), and the aggregated results, so every
table and figure except Table 1 (which needs the raw datasets, see
[data/README.md](data/README.md)) can be rebuilt in minutes without re-running the
simulations.

## Contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [Repository layout](#repository-layout)
- [Reproducing the paper](#reproducing-the-paper)
- [Running the simulations from scratch](#running-the-simulations-from-scratch)
- [Naming conventions](#naming-conventions)
- [Tests](#tests)
- [License and citation](#license-and-citation)

## Installation

Python 3.11 or 3.12 for the exact versions in `requirements.txt` (the results were
produced with Python 3.11.9 on Windows 11, CPU only). On other Python versions skip
`requirements.txt` and run `pip install -e .`, which installs current versions of the
dependencies; newer JAX/NumPyro releases can change the Monte Carlo draws.

```bash
git clone https://github.com/marcgrillo/DOR.git
cd DOR
pip install -r requirements.txt   # exact versions used for the paper
pip install -e .                  # optional: makes `bdor` importable from anywhere
```

The scripts also run without `pip install -e .`: each one puts the repository root
on `sys.path` itself. `corner` is optional; it draws the corner plot of the case study
(Figure 7), which falls back to plain matplotlib without it.

## Quick start

Rank alternatives from deck-of-cards answers collected over sessions (run it from the
repository root, e.g. in a `python` session started there, or after `pip install -e .`):

```python
import numpy as np
from bdor.features import PiecewiseLinearTransformer
from bdor.engine import PreferenceSampler
from bdor.simulation import expand_subrank

perf = np.random.default_rng(0).uniform(size=(10, 3))          # 10 alternatives x 3 criteria (higher = better)
tr = PiecewiseLinearTransformer.from_equal_intervals(perf, 3)   # 3 linear pieces per marginal value function
X = tr.transform(perf)                                          # features v(a), so that U(a) = v(a) @ w
sampler = PreferenceSampler(X, [], tr.total_params, n_max=5)    # e_max = 5 blank cards at most per gap

# One session: the DM ranks alternatives 4 > 0 > 7, with 2 blank cards between 4 and 0
# and none between 0 and 7. Repeat for every further session.
for upper, lower, cards in expand_subrank([4, 0, 7], [2, 0], 'adjacent'):
    sampler.add_gap(upper, lower, cards)

w_ftrl = sampler.find_map()[:tr.total_params]                   # FTRL-DOR: MAP estimate of w
draws = sampler.run_hmc(num_samples=200, num_warmup=200)        # BAYES-DOR: posterior draws of [w, Delta - rho]
ranking = np.argsort(-(X @ w_ftrl))                             # best alternative first
```

[`examples/interactive_elicitation/`](examples/interactive_elicitation/) runs the same
loop on a JSON file of alternatives and sessions and plots the inferred marginal
value functions after every session:

```bash
python examples/interactive_elicitation/inference_tool.py
```

## Repository layout

```
bdor/                     library
  features.py             piecewise-linear marginal value features v(a)          (Sec. 2.1)
  engine.py               PreferenceSampler: likelihood, priors, BAYES-DOR, FTRL-DOR (Secs. 3.1-3.5)
  simulation.py           one replication of the multi-session elicitation loop (Sec. 4)
  runner.py               runs a method over dataset folders and replications
  metrics.py              RAI, PWI, ASR, ASP, AIO                                (Sec. 5.2)
  datagen.py              synthetic decision makers and their answers            (Sec. 5.1)
  baselines/              DOR and SMAA-DOR (smaa_original: the paper's baselines)
  results.py              loading and pooling of the aggregated grid results
  regret.py               best-fixed-in-hindsight benchmark                      (Sec. 5.7)
  paths.py                every input/output location of the project
scripts/
  simulation/             expensive: produce data/ and results/
  paper/                  cheap: build the paper's tables, figures and text numbers from results/
  case_study/             Sec. 6, self-contained (about 1 minute)
examples/                 interactive elicitation demo on your own data
tests/                    pytest suite
results/                  aggregated results the paper is built from (see results/README.md)
outputs/                  generated figures and tables (same file names as in the manuscript)
data/                     raw simulation data, local only (~23 GB, see data/README.md)
```

`data/` is gitignored apart from its README and two small dataset folders.
Set `BDOR_DATA_DIR` to keep the raw data elsewhere; copy those two folders
(`data/datasets/grid_F1=10_F2=3_S=10_AR=5_K=5_Inc={0,50}/`) there too, since Figure 5
reads them.

## Reproducing the paper

All commands run from the repository root. Outputs go to `outputs/` with the file
names the manuscript includes.

| Paper item | Command | Output | Needs | Time |
|---|---|---|---|---|
| Table 3 | `python scripts/paper/make_summary_table.py` | `outputs/tables/summary_table_v1.tex` | `results/` | seconds |
| Supp. Tables B.1-B.9 | `python scripts/paper/make_appendix_tables.py` | `outputs/tables/appendix_tables_v1.tex` | `results/` | < 1 min |
| Figures 2, 3, 4, 5 and the HMC-diagnostic and regret numbers of Secs. 5.1 and 5.7 | `python scripts/paper/make_figures.py` | `outputs/figures/fig_factors_bdor_v1.png`, `fig_baselines_t1_v1.png`, `fig_sets_tau_v1.png`, `fig_logloss_v1.png` | `results/`, the two committed dataset folders (Fig. 5) | ~10 s |
| Sec. 5.3, "Larger sets or more sessions?" | `python scripts/paper/tk_equivalence.py` | stdout | `results/` | seconds |
| Sec. 6: Tables 5-10, Figures 7-8, Supp. Tables C.1-C.8 (Table 4 and Figure 6 are the embedded input data) | `python scripts/case_study/case_study_lea.py` | `outputs/case_study/` | nothing (`corner` for the exact Figure 7) | ~1 min |
| Table 1 and the flip shares of Sec. 5.1 (`--s10-only`: main-grid folders only) | `python scripts/paper/realized_flips.py` | stdout | full `data/datasets/` (not in the repository) | seconds |
| Sec. 5.1 flip shares by m and k; Sec. 5.3 subset-size numbers (AIO by k: last block) | `python scripts/paper/subset_density.py` | stdout | full `data/datasets/`, `results/` | seconds |
| Sec. 5.4 (e_max = 50) | `python scripts/simulation/run_emax_study.py`, then `run_emax_flipmatch.py` | `results/emax_*.csv`, stdout | runs its own simulations | ~50 min |
| Sec. 5.6 and the data of Figure 4 | `python scripts/simulation/run_ties_study.py` | `results/sets_tau_comparison.csv` | runs its own simulations | ~20 min |

`make_figures.py --only fig5` (choices `fig2 fig3 fig4 diagnostics fig5`) rebuilds a
single figure. The e_max and ties studies generate their own (seeded) datasets and are
resumable: existing datasets and per-session inference states under `data/` are reused.
They take no options (`--help` prints a summary and exits).

## Running the simulations from scratch

The main study crosses |A| in {10, 20, 35, 50}, m in {3, 5, 7, 9}, the subset size of
the first session and of the later sessions in {3, 4, 5}, and Finc in {0, 15, 35, 50}%,
with 20 replications and 10 sessions each (576 configurations). The paper reports the
192 configurations where the two subset sizes are equal, each read after T in
{1, 3, 5, 10} sessions (768 cells).

```bash
python scripts/simulation/generate_grid_datasets.py      # unless you have the paper's datasets (a clone has 2 of 576)
python scripts/simulation/run_grid.py --workers 12       # all methods on the grid      (~10 h, 12 workers)
python scripts/simulation/run_original_baselines.py      # DOR and SMAA-DOR at T = 1     (~5 min)
python scripts/simulation/inject_original_baselines.py   # adds them to the results npz (seconds)
python scripts/simulation/run_emax_study.py              # Sec. 5.4
python scripts/simulation/run_emax_flipmatch.py          # Sec. 5.4, after run_emax_study.py
python scripts/simulation/run_ties_study.py              # Sec. 5.6
```

then the commands of the previous section. `run_grid.py` costs about 110 CPU-hours
and writes about 10 GB of inference states (`data/samples/`) and 13 GB of metric
caches (`data/metrics/`). It runs one configuration per worker process and resumes
from `results/grid_results_joint_v1.npz`: arms already stored there are skipped and
existing per-session states are reused, so for a fresh campaign move that file (and
`data/samples/grid_*`) away first. `--force-arms ARM ...` re-runs selected arms.
`python scripts/simulation/run_grid.py --test` is a few-minute smoke run on its own tiny
dataset (`test_*` folders and `data/grid_results_test.npz`, recreated on every run;
campaign files are never touched).

**Datasets.** The grid datasets of the paper were generated without a fixed random
seed, so `generate_grid_datasets.py` produces statistically equivalent, not
identical, data. It skips existing dataset folders and overwrites only with an explicit
`--overwrite`; `--paper-only` generates just the 192 configurations the paper reports,
`--seed S` fixes the random stream, `--dry-run` lists what would be written. The e_max and ties datasets
are regenerated identically. Inference is seeded, so results are reproducible on the
same datasets and library versions.

Settings used to produce the paper's results: e_max = 5 cards in the simulations and 30 in the case
study; BAYES-DOR with Gamma(1, 1) priors and NUTS with 200 warm-up iterations and
200 draws per session (2000 draws after 1000 warm-up iterations in the case study);
FTRL-DOR with Gamma(2, 1) priors; threshold floor rho = 1e-6; 200 Hit-and-Run draws
for SMAA-DOR. BAYESdir (and the unreported BAYES_DOR_ALLJ1) run on the first 5 of the
20 replications of each grid cell,
so in the whole-grid comparison (Table 3 upper block, Figure 2, Tables B.1-B.5) the
BAYES-DOR and BAYESdir means are taken over those 5 shared replications; all other
arms and the T = 1 comparison use all 20. The e_max study uses 10 replications and the
ties study 20 (5 for BAYES-DOR).

## Naming conventions

Method arms are named `<ALG>_DOR_<TAG>`, which is also the key in the results npz and
the folder name under `data/samples/<fold>/f1_<|A|>_f2_<m>_nsteps_<S>/` and
`data/metrics/<fold>/`.

| Paper name | Arm |
|---|---|
| FTRL-DOR | `FTRL_DOR_ADJJ1` |
| BAYES-DOR | `BAYES_DOR_ADJJ1` |
| FTRLdir | `FTRLDIR_DOR_ADJJ1` |
| BAYESdir | `BAYESDIR_DOR_ADJJ1` |
| DOR | `DORLP_DOR_ORIG` (sample folder `DORLPORIG_T1`) |
| SMAA-DOR | `SMAA_DOR_ORIG` (sample folder `SMAAORIG_T1`) |

`ADJJ1` = adjacent-gap likelihood of the paper ("joint-v1": direction and cards
modelled jointly, theta_0 = 0); `EMXJ1` and `SETJ1` = the same model in the e_max and
ties studies. The grid campaign also runs four arms that the paper does not report:
`*_ALLJ1` (all-pairs composite likelihood, an ablation) and `SMAA_DOR_ADJJ1` /
`DORLP_DOR_ADJJ1` (an earlier per-gap adaptation of the baselines).

Configurations are written `F1=<|A|>_F2=<m>_S=<sessions>_AR=<k of session 1>_K=<k of later sessions>_Inc=<Finc %>`.
In the code, `e_max` is also called `n_max`, `eta_max` or `ETA_MAX`, the PWI is
called `poi`, and ASR / ASP / AIO are stored as `asrs` / `asps` / `aios`.

## Tests

```bash
python -m pytest -q -p no:cacheprovider     # 77 tests, about 1 minute on one CPU core
```

The suite needs no `data/` folder (it works on a fresh clone) and checks the feature map, the likelihood against its closed form,
both inference algorithms on a problem with a known answer, the session loop and its
seeds, the metrics on a hand-computed case, the DOR/SMAA-DOR baselines, the dataset
generator, a tiny end-to-end pipeline run, the interactive example, and that Table 3,
Tables B.1-B.9 and Figures 2-4 regenerate exactly from `results/` (the image
comparison is exact only with matplotlib 3.9.2 on Windows; elsewhere a mismatch is
reported as a skip). If the results are
re-run on purpose, update the reference values in `tests/test_results.py`.

## License and citation

MIT License, see [LICENSE](LICENSE). If you use this code, please cite the paper:

```
M. Grillo and S. Zappalà. Bayesian Deck-of-cards-based Ordinal Regression with
Sequential Preference Elicitation. Manuscript, 2026.
```

# data/

Raw simulation data. **Not stored in the repository** (about 23 GB and 3.6 million
files): everything in this folder is gitignored except this README and the two
dataset folders that Figure 5 needs
(`datasets/grid_F1=10_F2=3_S=10_AR=5_K=5_Inc={0,50}/`), so the regret figure can be
rebuilt from a fresh clone.

Apart from those two folders, none of it is needed to rebuild the paper's tables and
figures from `results/` (see the main README), with one exception: Table 1 and the
flip-share and distance numbers of Secs. 5.1 and 5.3 are computed directly from the
datasets. The data are also needed to re-run or extend the inference.

To keep the data elsewhere (e.g. on another drive), set the environment variable
`BDOR_DATA_DIR` to that folder before running any script; all paths are resolved in
[`bdor/paths.py`](../bdor/paths.py). Copy the two committed dataset folders there as
well, since Figure 5 reads them from `$BDOR_DATA_DIR/datasets/`.

## Layout

```
data/
  datasets/<fold>/f1_<|A|>__f2_<m>__nsteps_<S>.json        synthetic decision problems + elicitation answers
  samples/<fold>/f1_<|A|>_f2_<m>_nsteps_<S>/<arm>/table_<i>/
      <t>.npy        inference state after session t (posterior draws or point estimate)
      loss_<t>.npy   [prequential log-loss of session t's gaps, number of gaps]
      diag_<t>.npy   [divergent transitions, max split-Rhat]          (Bayesian arms)
  metrics/<fold>/<arm>/
      pois/  rais/   f1_.._nsteps_../table_<i>/<t>.npy   |A| x |A| PWI and RAI matrices
      asrs/  asps/  aios/   f1_.._nsteps_../<metric>_<t>.npy   one value per replication
```

* `<fold>`: `grid_F1=<|A|>_F2=<m>_S=<sessions>_AR=<k of session 1>_K=<k of later sessions>_Inc=<Finc %>`
  for the main grid (`run_grid.py`); `emax_..._Emax=<e_max>[_Sigma=util]` for the e_max
  study; `sets_..._Tau=<tau>` for the ties study.
* `<arm>`: method tag `<ALG>_DOR_<TAG>`, see "Naming conventions" in the main README.
* State arrays: BAYES arms store `(200, N + e_max)` float32 draws of `[w, Delta - rho]`
  (N piecewise-linear increments, e_max threshold increments); FTRL arms the MAP vector
  of the same length; SMAA arms `(200, N + 1)` draws of `[w, k]`; DOR arms the weight
  vector `(N,)`.

Each dataset JSON holds `metadata` (generation parameters, including the calibrated
noise `sigma`) and `runs`, one per replication: the performance `table`, the true
`weights`, curvature `params`, `true_utilities` and `true_ranks`, and the `subranks`
answered in each session: `levels` (the declared levels, best first, tied alternatives
together), `items` (their concatenation) and `cards` (blank cards between consecutive
levels). The paper's grid datasets predate the `levels` key (all their levels are
singletons); datasets written by the current generator always contain it.

## Provenance and reproducibility

* `datasets/grid_*_S=10_*` (576 folders) are the datasets of the paper's main grid.
  They were generated in June 2026 **without a fixed random seed**, so
  `scripts/simulation/generate_grid_datasets.py` produces statistically equivalent
  but not identical data. Keep a backup of the whole `datasets/` folder (about 120 MB;
  Table 1 also pools the legacy folders listed below); no script overwrites the grid
  datasets unless explicitly told to (`generate_grid_datasets.py --overwrite`).
* The e_max and ties datasets are regenerated identically by their study scripts
  (fixed seeds).
* `samples/` and `metrics/` are regenerable from the datasets with the simulation
  scripts. `metrics/` is a pure cache (mostly the |A| x |A| matrices).
* The local copy also contains folders from earlier development stages that no
  final script reads: grid folders with `S` in {1, 3, 5}, the metrics leftover
  `metrics/default_dataset`, and arms with the tags `PASSIVE`, `ADJACENT`, `ADJV1`,
  `ALLV1` (earlier likelihood versions). One exception:
  `scripts/paper/realized_flips.py` pools every `grid_*` folder with `AR == K`,
  including the 37 legacy ones with `S` in {1, 3, 5}, which reproduces the numbers of
  Table 1 (`--s10-only` restricts it to the main grid).

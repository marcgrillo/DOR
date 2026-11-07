# results/

Aggregated simulation results from which the simulation tables and figures of the
paper are built. They are small enough to version (about 21 MB in total), so these
tables and figures can be regenerated from a fresh clone in minutes, without
re-running the inference (see the main [README](../README.md#reproducing-the-paper)).
Table 1 is the exception: it is computed from the raw datasets (see
[data/README.md](../data/README.md)).

| File | Produced by | Read by | Content |
|---|---|---|---|
| `grid_results_joint_v1.npz` | `scripts/simulation/run_grid.py`, then `scripts/simulation/inject_original_baselines.py` | `scripts/paper/*` (via `bdor.results`) | Per-replication ASR / ASP / AIO after every session, for all 576 grid configurations and 10 method arms, plus prequential log-losses and HMC diagnostics. Layout documented in [`bdor/results.py`](../bdor/results.py). |
| `original_comparison.csv` | `scripts/simulation/run_original_baselines.py` | `scripts/simulation/inject_original_baselines.py` | T = 1 metrics of the DOR and SMAA-DOR baselines (`DORLPORIG_T1`, `SMAAORIG_T1`), plus the same metrics recomputed for the stored `FTRL_DOR_ADJJ1`, `BAYES_DOR_ADJJ1`, `SMAA_DOR_ADJJ1`, `DORLP_DOR_ADJJ1` states. 69,120 rows = 576 configurations x 20 replications x 6 methods. Columns: `fold,f1,f2,ar,k,inc,table,method,asr,asp,aio`. |
| `sets_tau_comparison.csv` | `scripts/simulation/run_ties_study.py` | `scripts/paper/make_figures.py` (Figure 4) | Ties study (Sec. 5.6): metric values per session and replication for `FTRL_DOR_SETJ1` (20 replications) and `BAYES_DOR_SETJ1` (5), for each merge threshold tau and inconsistency level. 12,000 rows. Columns: `Inc,Tau,Method,Metric,Step,Table,Value,MeanLevelsPerSubrank,MultiLevelShare`. |
| `emax_comparison.csv` | `scripts/simulation/run_emax_study.py` | none (archival: the Sec. 5.4 tests can be recomputed from it) | e_max study (Sec. 5.4): e_max = 5 vs 50 at Finc in {0, 35, 50}%, 10 paired replications, `FTRL_DOR_EMXJ1` and `BAYES_DOR_EMXJ1`. 3,600 rows. Columns: `Inc,Emax,Method,Metric,Step,Table,Value`. |
| `emax_flipmatch.csv` | `scripts/simulation/run_emax_flipmatch.py` | none (archival, as above) | Noise-scaled control of the e_max study at Finc = 50% (variants `e5_inc50`, `e50_card`, `e50_flip`, `e5_noinc`, `e50_noinc`). 3,000 rows. Columns: `Variant,Method,Metric,Step,Table,Value`. |

In the long-format CSVs (`sets_tau_comparison.csv`, `emax_*.csv`) the `Metric` column
holds `asrs` / `asps` / `aios` = ASR / ASP / AIO of the paper (Sec. 5.2, eqs.
(23)-(25)), `Step` is the session index t (from 1) and `Table` the replication index.
`original_comparison.csv` is in wide format: one row per configuration, replication
(`table`) and method, with columns `asr`, `asp`, `aio` after session 1 only.

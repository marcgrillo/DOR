"""B-DOR: Bayesian Deck-of-cards-based Ordinal Regression with sequential preference elicitation.

Code accompanying Grillo & Zappalà, "Bayesian Deck-of-cards-based Ordinal Regression
with Sequential Preference Elicitation".

Modules
-------
features     piecewise-linear marginal value features v(a) (paper Sec. 2.1)
engine       PreferenceSampler: gap-observation likelihood, BAYES-DOR (NUTS) and FTRL-DOR (MAP)
simulation   one replication of the multi-session elicitation loop
runner       loops a method over dataset folders / replications and stores its states
metrics      BenchmarkRunner: RAI, PWI and the ASR / ASP / AIO metrics (paper Sec. 5.2)
datagen      DatasetGenerator: synthetic decision makers and their elicitation data (Sec. 5.1)
baselines    DOR and SMAA-DOR (smaa_original: as in the paper; smaa_adapted: per-gap variant)
results      loading and pooling of the aggregated grid results
regret       best-fixed-in-hindsight benchmark of the regret study (Sec. 5.7)
paths        project file-system layout (data/, results/, outputs/)
utils        dataset / metric-file readers and small helpers

Submodules are imported explicitly (``from bdor.engine import PreferenceSampler``) so that
``import bdor`` stays light (JAX is only loaded by the modules that need it).
"""

__version__ = "1.0.0"

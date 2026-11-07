"""Competing methods of the DOR family, run on the same elicitation data as B-DOR.

smaa_original   DOR (LP point estimate) and SMAA-DOR (Hit-and-Run sample) in their
                original single-session formulation (paper Sec. 2.2, eq. (3)); these are
                the baselines reported in the paper (Sec. 5.5: Table 3 lower block,
                Figure 3, Tables B.6-B.9; arms DORLP_DOR_ORIG / SMAA_DOR_ORIG).
smaa_adapted    an earlier per-gap adaptation (arms DORLP_DOR_ADJJ1 / SMAA_DOR_ADJJ1), kept
                because the original formulation reuses its LP / Hit-and-Run machinery and
                the grid campaign still produces these arms; not used for reported results.

Both modules need only NumPy and SciPy (HiGHS linear programmes); they do not load JAX.
"""

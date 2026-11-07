"""Competing methods of the DOR family, run on the same elicitation data as B-DOR.

smaa_original   DOR (LP point estimate) and SMAA-DOR (Hit-and-Run sample) in their
                original single-session formulation (paper Sec. 2.2); these are the
                baselines reported in the paper (arms DORLP_DOR_ORIG / SMAA_DOR_ORIG).
smaa_adapted    an earlier per-gap adaptation (arms DORLP_DOR_ADJJ1 / SMAA_DOR_ADJJ1), kept
                because the original formulation reuses its LP / Hit-and-Run machinery and
                the grid campaign still produces these arms; not used for reported results.
"""

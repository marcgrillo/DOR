"""One replication of the multi-session elicitation loop (paper Sec. 4, Algorithm 1).

``process_single_table`` runs one method (an arm of a study) on one replication, i.e. one
performance table and the sessions of its synthetic decision maker. For every session it
turns the declared levels and blank cards into gap observations (Sec. 3.1), records the
prequential log-loss of these new gaps (eq. (13); Sec. 5.7), adds them to a
``PreferenceSampler``, refits and saves the inference state. ``bdor.runner`` loops it over
replications and configurations; ``bdor.metrics.BenchmarkRunner`` scores the saved states.

Algorithm tokens (``alg = '<ALG>-DOR'``, the first two fields of an arm tag such as
``BAYES_DOR_ADJJ1``) and the state each one saves::

    BAYES      BAYES-DOR (Sec. 3.3): NUTS posterior draws             (200, N + e_max)
    FTRL       FTRL-DOR (Sec. 3.4): constrained MAP estimate          (N + e_max,)
    BAYESDIR   BAYESdir: BAYES with the direction-only likelihood     (200, N + e_max)
    FTRLDIR    FTRLdir: FTRL with the direction-only likelihood       (N + e_max,)
    SMAA       adapted per-gap SMAA-DOR: Hit-and-Run draws of [w, k]  (200, N + 1)
    DORLP      adapted per-gap DOR linear programme: weights w        (N,)

N = 3 m is the number of value-function parameters (3 equal-width intervals per criterion)
and the B-DOR parameter vector is omega = (w, Delta - rho) as in eq. (14); the
direction-only likelihood ignores the card counts. Only the first N columns of a state
(the weights w) enter the ranking metrics. SMAA and DORLP are the adapted baselines of
``bdor.baselines.smaa_adapted`` (arms ``*_ADJJ1``), which the paper does not report; its
DOR and SMAA-DOR come from ``bdor.baselines.smaa_original`` via
``scripts/simulation/run_original_baselines.py``.
"""
import os
import numpy as np
from tqdm import tqdm
from bdor.features import PiecewiseLinearTransformer
from bdor.engine import PreferenceSampler


def get_sampler_state(sampler, alg, seed=0, return_diagnostics=False):
    """Fit the method named by ``alg`` to all gaps collected in ``sampler``.

    Parameters
    ----------
    sampler : bdor.engine.PreferenceSampler
        Holds the gap observations collected so far (``X_diff``, ``etas``) and the
        likelihood ('joint' or 'direction') used by the B-DOR methods.
    alg : str
        ``'<ALG>-DOR'``; the token before the dash selects the method:

        - 'BAYES', 'BAYESDIR': ``sampler.run_hmc`` (NUTS) with 200 warm-up iterations and
          200 draws, hard-coded here for every simulation study;
        - 'FTRL', 'FTRLDIR': ``sampler.find_map``, the constrained MAP estimate
          (L-BFGS-B; Gamma prior with alpha = 2, beta = 1 set inside ``find_map``);
        - 'SMAA': ``sample_smaa_dor`` (200 Hit-and-Run draws) and 'DORLP':
          ``solve_dor_lp``, the adapted per-gap baselines of
          ``bdor.baselines.smaa_adapted``.
    seed : int
        Seed of the NUTS chain or of the Hit-and-Run walk; FTRL and DORLP are
        deterministic and ignore it.
    return_diagnostics : bool
        BAYES* only: also return ``{'divergences': int, 'max_rhat': float}`` (number of
        divergent transitions, largest split-Rhat over the parameters). The other methods
        always return the state alone.

    Returns
    -------
    state : np.ndarray
        BAYES*: (200, N + e_max) draws; FTRL*: (N + e_max,); SMAA: (200, N + 1) draws of
        [w, k]; DORLP: (N,).
    diagnostics : dict
        Second element of a tuple, only for BAYES* with ``return_diagnostics=True``.

    Raises
    ------
    ValueError
        If the algorithm token is not one of the above.
    """
    algo_type, _ = alg.split('-')
    if algo_type in ('BAYES', 'BAYESDIR'):
        return sampler.run_hmc(num_samples=200, num_warmup=200, seed=seed,
                               return_diagnostics=return_diagnostics)
    elif algo_type in ('FTRL', 'FTRLDIR'):
        return sampler.find_map()
    elif algo_type == 'SMAA':
        from bdor.baselines.smaa_adapted import sample_smaa_dor
        return sample_smaa_dor(sampler.X_diff, sampler.etas, sampler.n_params,
                               num_samples=200, seed=seed)
    elif algo_type == 'DORLP':
        from bdor.baselines.smaa_adapted import solve_dor_lp
        return solve_dor_lp(sampler.X_diff, sampler.etas, sampler.n_params)
    else:
        raise ValueError(f"Unknown algorithm type: {algo_type}")


def expand_subrank(items, cards, expansion='all', eta_cap=None):
    """Turn one declared session (levels and blank cards) into gap observations.

    Parameters
    ----------
    items : sequence
        The declared levels, best to worst. Each entry is an alternative index (a
        singleton level) or a list / tuple / array of indices: a set of alternatives the
        DM declares indifferent (ties, Sec. 5.6).
    cards : sequence of int
        ``cards[h]`` is the number of blank cards between level h and level h + 1
        (``len(items) - 1`` values).
    expansion : {'all', 'adjacent'}
        'adjacent': one gap per pair of consecutive levels with its own card count, the
        observations of the adjacent-gap likelihood eq. (12) used throughout the paper.
        'all': one gap per ordered pair of levels with the cards between them summed, the
        all-pairs ablation (arms ``*_ALLJ1``), not used in the paper. Any value other
        than 'adjacent' gives 'all'.
    eta_cap : int, optional
        Cap every card count at this value (``process_single_table`` passes e_max). Only
        summed all-pairs counts can exceed e_max; capped, they fall in the top card band
        [theta_emax, +inf) of eq. (11) instead of being out of range.

    Returns
    -------
    list of tuple
        ``(upper_members, lower_members, eta)`` per gap, with the members as lists of
        int. ``PreferenceSampler.add_gap`` turns a gap into the difference of the two
        levels' mean feature vectors, eq. (8); for singleton levels this is the plain
        difference v(upper) - v(lower).

    Raises
    ------
    AssertionError
        If ``len(cards) != len(items) - 1`` or an alternative appears in two levels.
    """
    levels = [list(map(int, x)) if isinstance(x, (list, tuple, np.ndarray)) else [int(x)]
              for x in items]
    assert len(cards) == len(levels) - 1, \
        f"subrank malformed: {len(levels)} levels but {len(cards)} card gaps"
    flat = [i for lv in levels for i in lv]
    assert len(set(flat)) == len(flat), \
        "alternative appears in more than one level of the same subrank"

    gaps = []
    if expansion == 'adjacent':
        for m in range(len(levels) - 1):
            gaps.append((levels[m], levels[m + 1], int(cards[m])))
    else:
        for start_idx in range(len(levels)):
            for end_idx in range(start_idx + 1, len(levels)):
                eta = int(sum(cards[start_idx:end_idx]))
                gaps.append((levels[start_idx], levels[end_idx], eta))
    if eta_cap is not None:
        gaps = [(u, l, min(e, int(eta_cap))) for u, l, e in gaps]
    return gaps


def process_single_table(
    table,
    table_preferences,
    num_steps,
    k_subset,
    eta_max,
    output_dir,
    alg='BAYES-DOR',
    overwrite=False,
    table_index=0,
    disable_tqdm=False,
    progress_dict=None,
    expansion='all',
    seed_base=0,
    log_loss=True
):
    """Run one method over the sessions of one replication, saving its state after each.

    For every session j = 1, ..., ``num_steps``:

    1. expand the declared levels and cards into gaps (``expand_subrank``, cards capped
       at ``eta_max``);
    2. BAYES* and FTRL* only: record the prequential log-loss of these new gaps under
       the state after session j - 1; at j = 1 that state is 200 prior draws (BAYES*) or
       the prior mode (FTRL*, the MAP with no data). The loss is -sum_h log p_h over the
       session's gaps, where p_h is the gap's probability under the arm's likelihood,
       averaged over the draws for BAYES* (``PreferenceSampler.log_predictive_loss``);
    3. add the gaps to the sampler, refit on all gaps collected so far
       (``get_sampler_state``) and save the state.

    Files written in ``output_dir``:

    - ``<j>.npy``: state after session j (shapes in the module docstring);
    - ``loss_<j>.npy``: [loss of session j's gaps, number of gaps] (BAYES*, FTRL*);
    - ``diag_<j>.npy``: [NUTS divergences, max split-Rhat] (BAYES* only).

    Resuming: with ``overwrite=False`` existing files are kept. If ``<j>.npy`` exists,
    session j is not refitted, but its gaps are still added so that later sessions see
    all the data; a missing ``loss_<j>.npy`` is computed from the saved state of session
    j - 1.

    A session whose alternatives all share one level contributes no gap (possible in the
    ties study). If this happens at session 1, BAYES* save an empty (0, N + e_max) array
    there, so that replication's session-1 metrics and session-2 loss are NaN.

    Parameters
    ----------
    table : np.ndarray, shape (|A|, m)
        Performance table. Features use 3 equal-width intervals per criterion over the
        table's range (``PiecewiseLinearTransformer.from_equal_intervals``).
    table_preferences : list of dict
        The replication's sessions as stored by ``bdor.datagen``: entry j - 1 holds
        ``'cards'`` and the declared order, read from ``'levels'`` when present (ties)
        and otherwise from the flat ``'items'`` as singleton levels.
    num_steps : int
        Number of sessions to run (at most ``len(table_preferences)``).
    k_subset : int or tuple
        Not used; kept in the signature for the callers.
    eta_max : int
        e_max, the maximum number of blank cards per gap (number of threshold increments).
    output_dir : str
        Existing directory of this replication (``.../<arm>/table_<i>``).
    alg : str
        ``'<ALG>-DOR'``, see ``get_sampler_state``; the ``*DIR`` tokens select the
        direction-only likelihood.
    overwrite : bool
        Recompute and overwrite existing files.
    table_index : int
        Replication index i (seed and progress bar label).
    disable_tqdm : bool
        Hide the per-session progress bar.
    progress_dict : dict, optional
        If given, ``progress_dict[table_index]`` is set to the current session.
    expansion : {'all', 'adjacent'}
        Gap expansion, see ``expand_subrank``; the paper's model is 'adjacent'.
    seed_base : int
        Session j of replication i uses the seed ``seed_base + 100003 * i + j`` (NUTS
        chain, session-1 prior draws, Hit-and-Run walk); the study scripts derive
        ``seed_base`` from the configuration and the arm.
    log_loss : bool
        Record the ``loss_<j>.npy`` files.
    """
    # Value-function features: 3 equal-width intervals per criterion (gamma_j = 3)
    transformer = PiecewiseLinearTransformer.from_equal_intervals(table, num_intervals=3)
    feature_matrix = transformer.transform(table)

    n_params = transformer.total_params
    n_max = eta_max  # e_max: cap on the cards per gap = number of threshold increments

    algo_type = alg.split('-')[0]
    sampler = PreferenceSampler(
        feature_matrix, [], n_params, n_max,
        likelihood='direction' if algo_type in ('BAYESDIR', 'FTRLDIR') else 'joint'
    )
    prev_state = None  # state after the previous session, when computed in this call

    for j in tqdm(range(1, num_steps + 1), desc=f"  Table {table_index}", leave=False,
                  disable=disable_tqdm):
        if progress_dict is not None:
            progress_dict[table_index] = j

        path_w = os.path.join(output_dir, f"{j}.npy")

        # 'levels' (sets of tied alternatives) takes precedence over the flat 'items'
        # (singleton levels), which is all that the grid datasets carry.
        subrank = table_preferences[j-1]
        items = subrank.get('levels', subrank['items'])
        cards = subrank['cards']

        gaps = expand_subrank(items, cards, expansion, eta_cap=n_max)
        # Distinct, reproducible seed per (replication, session).
        step_seed = seed_base + 100003 * table_index + j

        # Prequential log-loss of this session's gaps, computed BEFORE they are added.
        loss_path = os.path.join(output_dir, f"loss_{j}.npy")
        if (log_loss and algo_type in ('BAYES', 'FTRL', 'BAYESDIR', 'FTRLDIR')
                and (overwrite or not os.path.exists(loss_path))):
            state_for_loss = prev_state
            if state_for_loss is None:
                if j == 1:
                    # No data yet: prior draws (BAYES*) or the prior mode (FTRL*)
                    state_for_loss = (sampler.sample_prior(200, seed=step_seed)
                                      if algo_type.startswith('BAYES') else sampler.find_map())
                else:
                    prev_path = os.path.join(output_dir, f"{j-1}.npy")
                    if os.path.exists(prev_path):
                        state_for_loss = np.load(prev_path)
            if state_for_loss is not None:
                # Level-mean feature differences, as in PreferenceSampler.add_gap (eq. (8))
                X_new = np.array([feature_matrix[u].mean(axis=0) - feature_matrix[l].mean(axis=0)
                                  for u, l, _ in gaps])
                etas_new = np.array([g[2] for g in gaps], dtype=int)
                batch_loss = sampler.log_predictive_loss(state_for_loss, X_new, etas_new)
                np.save(loss_path, np.array([batch_loss, len(gaps)]))

        # Resume: session j already fitted, only add its gaps for the later sessions
        if os.path.exists(path_w) and not overwrite:
            for upper, lower, eta in gaps:
                sampler.add_gap(upper, lower, eta)
            prev_state = None  # reloaded from disk if the next session's loss needs it
            continue

        # The sampler keeps the gaps of every session, so each session refits on all the
        # data collected so far: the fit after session j targets pi(phi | D_j), the same
        # posterior as the one-session-at-a-time recursion eq. (16) (for FTRL: the MAP on
        # D_j, i.e. the estimate eq. (20) uses to predict session j + 1).
        for upper, lower, eta in gaps:
            sampler.add_gap(upper, lower, eta)

        if algo_type in ('BAYES', 'BAYESDIR'):
            w_state, diag = get_sampler_state(sampler, alg, seed=step_seed,
                                              return_diagnostics=True)
            np.save(os.path.join(output_dir, f"diag_{j}.npy"),
                    np.array([diag['divergences'], diag['max_rhat']]))
        else:
            w_state = get_sampler_state(sampler, alg, seed=step_seed)
        np.save(path_w, w_state)
        prev_state = w_state

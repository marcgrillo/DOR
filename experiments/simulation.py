import os
import numpy as np
from tqdm import tqdm
from mcda.models import PiecewiseLinearTransformer
from inference.engine import PreferenceSampler

def get_sampler_state(sampler, alg, seed=0, return_diagnostics=False):
    """
    Returns the state of the sampler (Samples for BAYES/SMAA, MAP for FTRL).
    For BAYES with return_diagnostics=True, returns (samples, {'divergences', 'max_rhat'}).
    """
    algo_type, _ = alg.split('-')
    if algo_type == 'BAYES':
        return sampler.run_hmc(num_samples=200, num_warmup=200, seed=seed,
                               return_diagnostics=return_diagnostics)
    elif algo_type == 'FTRL':
        return sampler.find_map()
    elif algo_type == 'SMAA':
        from inference.smaa import sample_smaa_dor
        return sample_smaa_dor(sampler.X_diff, sampler.etas, sampler.n_params,
                               num_samples=200, seed=seed)
    else:
        raise ValueError(f"Unknown algorithm type: {algo_type}")

def expand_subrank(items, cards, expansion='all', eta_cap=None):
    """
    Expands a subrank into pairwise triples (winner, loser, eta).

    expansion='all': every ordered pair, eta = sum of cards between them
                     (A > B > C -> A>B, A>C, B>C).
    expansion='adjacent': only consecutive pairs, eta = cards separating them
                     (A > B > C -> A>B, B>C).
    eta_cap: right-censor eta into the top CLM category (eta_cap = n_max).
             Summed all-pairs scores can exceed the per-gap card cap (the legacy
             engine silently dropped those observations); with a cap they land in
             the right-censored top band P(latent >= theta_max), which is the
             CLM-consistent treatment for the ablation arm.
    """
    # Guards (plan items C4/C9): strict singleton levels, one card count per gap.
    assert len(cards) == len(items) - 1, \
        f"subrank malformed: {len(items)} items but {len(cards)} card gaps"
    assert len(set(items)) == len(items), \
        "multi-member levels/ties are not supported by the adjacent extraction (C9)"

    pairs = []
    if expansion == 'adjacent':
        for m in range(len(items) - 1):
            pairs.append((int(items[m]), int(items[m + 1]), int(cards[m])))
    else:
        for start_idx in range(len(items)):
            for end_idx in range(start_idx + 1, len(items)):
                eta = sum(cards[start_idx:end_idx])
                pairs.append((int(items[start_idx]), int(items[end_idx]), int(eta)))
    if eta_cap is not None:
        pairs = [(w, l, min(e, int(eta_cap))) for w, l, e in pairs]
    return pairs

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
    """
    Runs the baseline inference for a single Human Model (table) over a sequence of passive queries.

    Besides the per-step state ({j}.npy), for BAYES/FTRL it records:
      loss_{j}.npy = [prequential predictive log-loss of step j's batch under the state
                      after step j-1 (prior/prior-mode at j=1), batch size]      (plan C14)
      diag_{j}.npy = [HMC divergences, max split-Rhat]  (BAYES only)            (plan C10)
    """
    # 1. Setup
    transformer = PiecewiseLinearTransformer.from_equal_intervals(table, num_intervals=3)
    feature_matrix = transformer.transform(table)

    n_params = transformer.total_params
    n_max = eta_max # Use the hyperparameter

    sampler = PreferenceSampler(feature_matrix, [], n_params, n_max)
    algo_type = alg.split('-')[0]
    prev_state = None

    # 2. Main Loop
    for j in tqdm(range(1, num_steps + 1), desc=f"  Table {table_index}", leave=False, disable=disable_tqdm):
        if progress_dict is not None:
            progress_dict[table_index] = j

        path_w = os.path.join(output_dir, f"{j}.npy")

        # table_preferences is now a list of subranks for this run
        subrank = table_preferences[j-1]
        items = subrank['items']
        cards = subrank['cards']

        pairs = expand_subrank(items, cards, expansion, eta_cap=n_max)
        # Distinct, reproducible seed per (table, step) — plan C15 "fresh seeds".
        step_seed = seed_base + 100003 * table_index + j

        # Prequential predictive log-loss of the incoming batch, computed BEFORE updating (C14).
        loss_path = os.path.join(output_dir, f"loss_{j}.npy")
        if log_loss and algo_type in ('BAYES', 'FTRL') and (overwrite or not os.path.exists(loss_path)):
            state_for_loss = prev_state
            if state_for_loss is None:
                if j == 1:
                    state_for_loss = (sampler.sample_prior(200, seed=step_seed)
                                      if algo_type == 'BAYES' else sampler.find_map())
                else:
                    prev_path = os.path.join(output_dir, f"{j-1}.npy")
                    if os.path.exists(prev_path):
                        state_for_loss = np.load(prev_path)
            if state_for_loss is not None:
                wins = [p[0] for p in pairs]
                loses = [p[1] for p in pairs]
                etas_new = np.array([p[2] for p in pairs], dtype=int)
                X_new = feature_matrix[wins] - feature_matrix[loses]
                batch_loss = sampler.log_predictive_loss(state_for_loss, X_new, etas_new)
                np.save(loss_path, np.array([batch_loss, len(pairs)]))

        # Skip logic
        if os.path.exists(path_w) and not overwrite:
            for p_win, p_lose, eta in pairs:
                sampler.add_preference(p_win, p_lose, eta)
            prev_state = None  # reloaded lazily from disk if a later step needs it
            continue

        # Add new preferences from the subrank
        for p_win, p_lose, eta in pairs:
            sampler.add_preference(p_win, p_lose, eta)

        # Run inference
        if algo_type == 'BAYES':
            w_state, diag = get_sampler_state(sampler, alg, seed=step_seed, return_diagnostics=True)
            np.save(os.path.join(output_dir, f"diag_{j}.npy"),
                    np.array([diag['divergences'], diag['max_rhat']]))
        else:
            w_state = get_sampler_state(sampler, alg, seed=step_seed)
        np.save(path_w, w_state)
        prev_state = w_state

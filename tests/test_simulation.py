"""Session loop of one replication (bdor.simulation): expansion of a declared ranking into gap
observations (paper Sec. 3.1), dispatch to the estimators, the per-session state files, the
prequential log-loss sidecars (eq. (13), recorded for Sec. 5.7) and the per-session seeds."""
import numpy as np
import pytest

from bdor.engine import PreferenceSampler
from bdor.simulation import expand_subrank, get_sampler_state, process_single_table


# ------------------------------------------------------------------------ expand_subrank

def test_expand_subrank_adjacent_gaps():
    assert expand_subrank([3, 1, 2], [2, 0], expansion="adjacent") == [
        ([3], [1], 2), ([1], [2], 0)]


def test_expand_subrank_all_pairs_sums_the_cards():
    assert expand_subrank([3, 1, 2], [2, 0], expansion="all") == [
        ([3], [1], 2), ([3], [2], 2), ([1], [2], 0)]
    capped = expand_subrank([0, 1, 2, 3], [4, 3, 5], expansion="all", eta_cap=5)
    assert [g[2] for g in capped] == [4, 5, 5, 3, 5, 5]   # right-censored at the cap


def test_expand_subrank_set_valued_levels():
    gaps = expand_subrank([[0, 4], 2, (5, 6)], [1, 3], expansion="adjacent")
    assert gaps == [([0, 4], [2], 1), ([2], [5, 6], 3)]
    # a session with a single level carries no gap observation
    assert expand_subrank([[0, 1, 2]], [], expansion="adjacent") == []


def test_expand_subrank_rejects_malformed_subranks():
    with pytest.raises((AssertionError, ValueError)):
        expand_subrank([0, 1, 2], [1], expansion="adjacent")        # cards != levels - 1
    with pytest.raises((AssertionError, ValueError)):
        expand_subrank([[0, 1], [1, 2]], [1], expansion="adjacent")  # alternative repeated


# --------------------------------------------------------------------- get_sampler_state

@pytest.fixture(scope="module")
def first_session_sampler(problem):
    sampler = PreferenceSampler(problem.X, [], problem.n_params, problem.n_max)
    for upper, lower, eta in expand_subrank(problem.subranks[0]["levels"],
                                            problem.subranks[0]["cards"], "adjacent"):
        sampler.add_gap(upper, lower, eta)
    return sampler


def test_state_of_point_estimators(problem, first_session_sampler):
    p, n_max = problem.n_params, problem.n_max
    ftrl = get_sampler_state(first_session_sampler, "FTRL-DOR")
    np.testing.assert_array_equal(ftrl, first_session_sampler.find_map())
    assert ftrl.shape == (p + n_max,)

    dor = get_sampler_state(first_session_sampler, "DORLP-DOR")
    assert dor.shape == (p,)
    assert dor.min() >= 0.0 and dor.sum() == pytest.approx(1.0)


def test_state_of_sampling_estimators(problem, first_session_sampler):
    p, n_max = problem.n_params, problem.n_max
    bayes = get_sampler_state(first_session_sampler, "BAYES-DOR", seed=4)
    assert bayes.shape == (200, p + n_max)          # 200 warm-up + 200 draws in all studies
    draws, diag = get_sampler_state(first_session_sampler, "BAYES-DOR", seed=4,
                                    return_diagnostics=True)
    np.testing.assert_array_equal(draws, bayes)
    assert set(diag) == {"divergences", "max_rhat"}

    smaa = get_sampler_state(first_session_sampler, "SMAA-DOR", seed=4)
    assert smaa.shape == (200, p + 1)               # [w, k]
    assert smaa[:, :p].min() >= 0.0
    np.testing.assert_allclose(smaa[:, :p].sum(axis=1), 1.0, atol=1e-8)

    with pytest.raises(ValueError):
        get_sampler_state(first_session_sampler, "NEWTON-DOR")


# ------------------------------------------------------------------ process_single_table

def _session_batch(problem, j):
    """Feature rows and card counts of the gaps of session ``j`` (1-based)."""
    sr = problem.subranks[j - 1]
    gaps = expand_subrank(sr["levels"], sr["cards"], "adjacent", eta_cap=problem.n_max)
    X = np.array([problem.X[u].mean(axis=0) - problem.X[l].mean(axis=0) for u, l, _ in gaps])
    return gaps, X, np.array([g[2] for g in gaps])


def _run_table(problem, out_dir, alg, num_steps=2, **kwargs):
    process_single_table(
        table=problem.table, table_preferences=problem.subranks[:num_steps],
        num_steps=num_steps, k_subset=4, eta_max=problem.n_max, output_dir=str(out_dir),
        alg=alg, expansion="adjacent", disable_tqdm=True, **kwargs)


def test_ftrl_session_loop_states_and_prequential_loss(problem, tmp_path):
    _run_table(problem, tmp_path, "FTRL-DOR")
    assert sorted(f.name for f in tmp_path.iterdir()) == [
        "1.npy", "2.npy", "loss_1.npy", "loss_2.npy"]

    p, n_max = problem.n_params, problem.n_max
    reference = PreferenceSampler(problem.X, [], p, n_max)
    prev_state = reference.find_map()           # no data yet: the prior mode
    for j in (1, 2):
        gaps, X_j, etas_j = _session_batch(problem, j)
        # loss_j = [log-loss of session j's gaps under the state after session j-1, #gaps];
        # before the first session that state is the prior mode
        loss = np.load(tmp_path / f"loss_{j}.npy")
        assert loss.shape == (2,) and loss[1] == len(gaps)
        assert loss[0] == pytest.approx(reference.log_predictive_loss(prev_state, X_j, etas_j),
                                        rel=1e-6)
        for upper, lower, eta in gaps:
            reference.add_gap(upper, lower, eta)
        state = np.load(tmp_path / f"{j}.npy")
        assert state.shape == (p + n_max,)
        np.testing.assert_allclose(state, reference.find_map(), rtol=1e-6, atol=1e-9)
        prev_state = state


def test_session_loop_resumes_from_the_files_on_disk(problem, tmp_path):
    """overwrite=False (how every campaign script resumes) keeps the files already written
    and recomputes only the missing ones, from the state saved for the previous session."""
    _run_table(problem, tmp_path, "FTRL-DOR")
    complete = {name: np.load(tmp_path / name) for name in ("2.npy", "loss_2.npy")}

    sentinel = np.full(problem.n_params + problem.n_max, 7.0)
    np.save(tmp_path / "2.npy", sentinel)
    before = {f.name: f.read_bytes() for f in tmp_path.iterdir()}
    _run_table(problem, tmp_path, "FTRL-DOR", overwrite=False)
    assert {f.name: f.read_bytes() for f in tmp_path.iterdir()} == before

    for name in complete:
        (tmp_path / name).unlink()
    _run_table(problem, tmp_path, "FTRL-DOR", overwrite=False)
    for name, value in complete.items():
        np.testing.assert_allclose(np.load(tmp_path / name), value, rtol=1e-6, atol=1e-9)


def test_bayes_session_loop_seeds_and_sidecars(problem, tmp_path):
    seed_base, table_index = 11, 2
    _run_table(problem, tmp_path, "BAYES-DOR", num_steps=1, seed_base=seed_base,
               table_index=table_index)
    assert sorted(f.name for f in tmp_path.iterdir()) == ["1.npy", "diag_1.npy", "loss_1.npy"]

    # per-(table, session) seed: seed_base + 100003 * table_index + j
    step_seed = seed_base + 100003 * table_index + 1
    gaps, X_1, etas_1 = _session_batch(problem, 1)
    reference = PreferenceSampler(problem.X, [], problem.n_params, problem.n_max)
    # before the first session BAYES-DOR predicts with 200 prior draws
    prior_draws = reference.sample_prior(200, seed=step_seed)
    loss = np.load(tmp_path / "loss_1.npy")
    assert loss[0] == pytest.approx(reference.log_predictive_loss(prior_draws, X_1, etas_1),
                                    rel=1e-6)
    assert loss[1] == len(gaps)

    for upper, lower, eta in gaps:
        reference.add_gap(upper, lower, eta)
    expected, diag = reference.run_hmc(num_samples=200, num_warmup=200, seed=step_seed,
                                       return_diagnostics=True)
    np.testing.assert_array_equal(np.load(tmp_path / "1.npy"), expected)
    np.testing.assert_allclose(np.load(tmp_path / "diag_1.npy"),
                               [diag["divergences"], diag["max_rhat"]])

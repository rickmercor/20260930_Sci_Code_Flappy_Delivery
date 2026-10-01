"""
Run the inner moment-quadrature self-consistency loop of the source at the given pole order for the half-filled Hubbard model on the Bethe lattice, seeded from the given bond amplitude, and return the weight of the pole closest to zero energy at convergence. Compose the earlier steps: form the moments, select the resolvable rank, build the Jacobi matrix, extract nodes and weights, close the bond amplitude, mix the new moments linearly with the previous ones using the given mixing factor, and stop when the source's relative moment criterion falls below delta over the nonzero moments.

The fixed point makes the spectral function that closes the source's approximate moments coincide with the spectral function the quadrature reconstructs. The central pole is the finite-rank carrier of the low-energy spectral weight of that reconstruction; its weight is a property of the source's closure at the given rank, not the quasiparticle residue of the Hubbard model.

Returns
-------
float, Weight of the pole closest to zero energy at the fixed point, as a native Python float; the rule is rebuilt from the final (mixed) moment set after the stop test is met.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def self_consistent_central_weight(U: float, half_bandwidth: float, order: int, tau: float,
                                delta: float, mixing: float, seed_amplitude: float,
                                max_iter: int) -> float:
    """Run the inner moment-quadrature self-consistency loop of the source at the given pole order for the half-filled Hubbard model on the Bethe lattice, seeded from the given bond amplitude, and return the weight of the pole closest to zero energy at convergence. Compose the earlier steps: form the moments, select the resolvable rank, build the Jacobi matrix, extract nodes and weights, close the bond amplitude, mix the new moments linearly with the previous ones using the given mixing factor, and stop when the source's relative moment criterion falls below delta over the nonzero moments.

    Parameters
    ----------
    U : float
        Nonnegative on-site interaction.
    half_bandwidth : float
        Positive half-bandwidth D.
    order : int
        Positive nominal pole order N.
    tau : float
        Rank threshold in (0, 1).
    delta : float
        Positive relative convergence tolerance on the moments.
    mixing : float
        Linear mixing factor in (0, 1] applied to the new moments.
    seed_amplitude : float
        Finite bond amplitude used to seed the moments.
    max_iter : int
        Positive maximum number of iterations.

    Returns
    -------
    w_central : float
        Weight of the pole closest to zero energy at the fixed point, as a native Python float; the rule is rebuilt from the final (mixed) moment set after the stop test is met.

    Raises
    ------
    ValueError
        If any argument is outside its stated domain, or the iteration does not converge within max_iter.
    """
    return w_central

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_self_consistent_central_weight(U: float, half_bandwidth: float, order: int, tau: float,
                                delta: float, mixing: float, seed_amplitude: float,
                                max_iter: int) -> float:
    """Inner moment self-consistency loop (source Sec. IV B) and the weight of the central pole.

    Seed the moments from seed_amplitude, then iterate: rank (14) -> Jacobi -> nodes and
    weights -> bond amplitude closure -> new moments, mixed linearly with the previous
    ones, until Eq (17) max_n |mu_n' - mu_n| / |mu_n| < delta over the nonzero moments.
    Returns the weight of the node closest to zero energy at the fixed point; the rule is
    rebuilt from the final (mixed) moment set after the stop test is met.
    """
    if not (np.isfinite(U) and U >= 0.0):
        raise ValueError("U must be nonnegative and finite")
    if not (np.isfinite(half_bandwidth) and half_bandwidth > 0.0):
        raise ValueError("half_bandwidth must be positive and finite")
    if int(order) != order or order < 1:
        raise ValueError("order must be a positive integer")
    if not (np.isfinite(tau) and 0.0 < tau < 1.0):
        raise ValueError("tau must lie in (0, 1)")
    if not (np.isfinite(delta) and delta > 0.0):
        raise ValueError("delta must be positive")
    if not (np.isfinite(mixing) and 0.0 < mixing <= 1.0):
        raise ValueError("mixing must lie in (0, 1]")
    if not np.isfinite(seed_amplitude):
        raise ValueError("seed_amplitude must be finite")
    if int(max_iter) != max_iter or max_iter < 1:
        raise ValueError("max_iter must be a positive integer")
    order, max_iter = int(order), int(max_iter)

    mu = _oracle_half_filled_moments(U, half_bandwidth, seed_amplitude)
    for _ in range(max_iter):
        n_star = _oracle_resolvable_rank(mu, order, tau)
        J = _oracle_jacobi_matrix(mu, n_star)
        nw = _oracle_quadrature_nodes_weights(J, mu[0])
        amp = _oracle_bond_amplitude_closure(nw[0], nw[1], half_bandwidth)
        mu_new = _oracle_half_filled_moments(U, half_bandwidth, amp)
        mu_mixed = float(mixing) * mu_new + (1.0 - float(mixing)) * mu
        nz = np.abs(mu) > 0.0
        change = float(np.max(np.abs(mu_mixed[nz] - mu[nz]) / np.abs(mu[nz])))
        mu = mu_mixed
        if change < delta:
            break
    else:
        raise ValueError("moment iteration did not converge within max_iter")

    n_star = _oracle_resolvable_rank(mu, order, tau)
    nw = _oracle_quadrature_nodes_weights(_oracle_jacobi_matrix(mu, n_star), mu[0])
    _ = _oracle_semielliptic_dos(nw[0], half_bandwidth)     # the bare band the closure integrates against
    central = int(np.argmin(np.abs(nw[0])))
    return float(nw[1][central])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    return [
        {
            "setup": "U, half_bandwidth, order, tau, delta, mixing = 1.0, 1.0, 3, 1e-8, 1e-8, 0.5\nseed_amplitude, max_iter = 0.13509491152311703, 500\n",
            "call": "self_consistent_central_weight(U, half_bandwidth, order, tau, delta, mixing, seed_amplitude, max_iter)",
            "gold_call": "_oracle_self_consistent_central_weight(U, half_bandwidth, order, tau, delta, mixing, seed_amplitude, max_iter)",
            "tol": 1e-07,
        },
        {
            "setup": "U, half_bandwidth, order, tau, delta, mixing = 0.5, 1.0, 3, 1e-8, 1e-8, 0.5\nseed_amplitude, max_iter = 0.13509491152311703, 500\n",
            "call": "self_consistent_central_weight(U, half_bandwidth, order, tau, delta, mixing, seed_amplitude, max_iter)",
            "gold_call": "_oracle_self_consistent_central_weight(U, half_bandwidth, order, tau, delta, mixing, seed_amplitude, max_iter)",
            "tol": 1e-07,
        },
        {
            "setup": "U, half_bandwidth, order, tau, delta, mixing = 1.5, 1.0, 3, 1e-8, 1e-8, 0.5\nseed_amplitude, max_iter = 0.13509491152311703, 500\n",
            "call": "self_consistent_central_weight(U, half_bandwidth, order, tau, delta, mixing, seed_amplitude, max_iter)",
            "gold_call": "_oracle_self_consistent_central_weight(U, half_bandwidth, order, tau, delta, mixing, seed_amplitude, max_iter)",
            "tol": 1e-07,
        },
        {
            "setup": "U, half_bandwidth, order, tau, delta, mixing = 0.0, 1.0, 3, 1e-8, 1e-8, 0.5\nseed_amplitude, max_iter = 0.13509491152311703, 500\n",
            "call": "self_consistent_central_weight(U, half_bandwidth, order, tau, delta, mixing, seed_amplitude, max_iter)",
            "gold_call": "_oracle_self_consistent_central_weight(U, half_bandwidth, order, tau, delta, mixing, seed_amplitude, max_iter)",
            "tol": 1e-07,
        },
        {
            "setup": "U, half_bandwidth, order, tau, delta, mixing = 1.0, 1.0, 3, 1e-8, 1e-8, 0.5\nseed_amplitude, max_iter = 0.13509491152311703, 2\ndef run_model():\n    try:\n        self_consistent_central_weight(U, half_bandwidth, order, tau, delta, mixing, seed_amplitude, max_iter)\n        return 0\n    except ValueError:\n        return 1\ndef run_oracle():\n    try:\n        _oracle_self_consistent_central_weight(U, half_bandwidth, order, tau, delta, mixing, seed_amplitude, max_iter)\n        return 0\n    except ValueError:\n        return 1\n",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]

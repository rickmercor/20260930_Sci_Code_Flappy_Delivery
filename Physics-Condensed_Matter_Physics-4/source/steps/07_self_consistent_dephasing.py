"""
Solve for the state dephasings that are consistent with the broadened scattering rates they produce.

The dephasing of a moire exciton state is the imaginary part of its phonon self-energy, which equals hbar/2
times its total out-scattering rate, summed over every final state including itself. Because the rates depend
on the dephasings through the kernel width, the two must agree: Gamma_s = (hbar/2) * sum_f R[f, s](Gamma),
hbar = 0.6582119569 meV ps. The fixed point is found by plain iteration from a uniform starting value: each
new estimate is the right-hand side evaluated at the previous estimate, without mixing, and the iteration
stops at the first new estimate whose largest absolute change from the previous estimate is below tol (meV).

Returns
-------
gamma : np.ndarray -- (N_k * n_bands,) dephasings in meV: the first update whose largest change is below tol.
"""

import numpy as np
from scipy.linalg import expm

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def self_consistent_dephasing(energies: "np.ndarray", weights: tuple, gamma0: float, tol: float,
                              max_iter: int) -> "np.ndarray":
    """Return the converged dephasing of every state.

    Parameters
    ----------
    energies : np.ndarray
        (N_k, n_bands) mini-band energies, meV.
    weights : tuple
        (w_ac, omega, w_op, e_op) as returned by scattering_weights.
    gamma0 : float
        Uniform starting dephasing, meV, positive.
    tol : float
        Convergence threshold on the largest absolute update, meV.
    max_iter : int
        Maximum number of updates.

    Returns
    -------
    gamma : np.ndarray
        (N_k * n_bands,) dephasings in meV: the first update whose largest change is below tol.

    Raises
    ------
    ValueError
        If no update meets tol within max_iter updates.
    """
    return gamma

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_self_consistent_dephasing(energies: "np.ndarray", weights: tuple, gamma0: float, tol: float,
                                      max_iter: int) -> "np.ndarray":
    hbar = 0.6582119569
    gam = np.full(energies.size, float(gamma0))
    for _ in range(max_iter):
        new = hbar / 2.0 * _oracle_transition_rate_matrix(energies, weights, gam).sum(axis=0)
        if np.max(np.abs(new - gam)) < tol:
            return new
        gam = new
    raise ValueError("dephasing iteration did not converge")

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    base = """import numpy as np
def make(nk, nb, seed, spread, scale):
    rng = np.random.default_rng(seed)
    s = nk * nb
    e = np.sort(rng.uniform(0.0, spread, size=(nk, nb)), axis=1)
    w_ac = scale * rng.uniform(0.0, 1.0, size=(s, s, 19, 2, 2))
    om = rng.uniform(0.0, 6.0, size=(nk, nk, 19, 2))
    w_op = scale * rng.uniform(0.0, 1.0, size=(s, s, 2, 2))
    return e, (w_ac, om, w_op, np.array([36.6, 30.8]))
"""
    return [
        {"setup": base + "e, w = make(3, 2, 2, 40.0, 0.3)",
         "call": "self_consistent_dephasing(e, w, 1.0, 1e-10, 500)",
         "gold_call": "_oracle_self_consistent_dephasing(e, w, 1.0, 1e-10, 500)"},
        {"setup": base + "e, w = make(2, 3, 5, 70.0, 0.05)",
         "call": "self_consistent_dephasing(e, w, 3.0, 1e-9, 500)",
         "gold_call": "_oracle_self_consistent_dephasing(e, w, 3.0, 1e-9, 500)"},
        # loose tolerance: the returned iterate depends on the stopping rule
        {"setup": base + "e, w = make(2, 2, 8, 30.0, 0.4)",
         "call": "self_consistent_dephasing(e, w, 0.2, 0.05, 500)",
         "gold_call": "_oracle_self_consistent_dephasing(e, w, 0.2, 0.05, 500)"},
        {"setup": base + """e, w = make(2, 2, 8, 30.0, 0.4)
def run(fn):
    try:
        fn(e, w, 0.2, 1e-14, 2)
        return 0
    except ValueError:
        return 1
""",
         "call": "run(self_consistent_dephasing)",
         "gold_call": "run(_oracle_self_consistent_dephasing)"},
    ]

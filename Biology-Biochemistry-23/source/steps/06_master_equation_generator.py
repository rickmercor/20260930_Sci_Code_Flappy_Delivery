"""
Assemble the generator matrix T of the master equation dP/dt = T P for the nucleation-zipper rupture of an n_bp duplex from the rate array [k_ot, k_o(2), ..., k_o(n_bp), k_c(2), ..., k_c(n_bp)] of the previous step. States are the pairs (i, j) with i ruptured pairs counted from the left end and j from the right end, i + j <= n_bp - 1, ordered lexicographically by (i, j), followed by the fully ruptured single-strand state as the last index; n = n_bp - i - j closed pairs remain. From a state with n > 1 each end opens one pair at rate k_o(n); with n = 1 the last pair opens into the ruptured state at rate k_ot; an end with i > 0 (or j > 0) re-closes one pair at rate k_c(n + 1); the ruptured state is absorbing; bulges, internal loops and sliding are neglected. Column k of T holds the outflow of state k on the diagonal and the inflows to the other states off the diagonal.

Rupture of a short duplex proceeds by fraying single base pairs from either end, so the state of the system is the pair of frayed lengths and the kinetics is a finite continuous-time Markov chain with one absorbing state; the master equation is a linear system whose generator is sparse and banded.

Returns
-------
ndarray of float64, shape (N_states, N_states) with N_states = 1 + n_bp (n_bp + 1) / 2, the generator T.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def master_equation_generator(n_bp: int, rates: "np.ndarray") -> "np.ndarray":
    """Assemble the generator matrix T of the master equation dP/dt = T P for the nucleation-zipper rupture of an n_bp duplex from the rate array [k_ot, k_o(2), ..., k_o(n_bp), k_c(2), ..., k_c(n_bp)] of the previous step. States are the pairs (i, j) with i ruptured pairs counted from the left end and j from the right end, i + j <= n_bp - 1, ordered lexicographically by (i, j), followed by the fully ruptured single-strand state as the last index; n = n_bp - i - j closed pairs remain. From a state with n > 1 each end opens one pair at rate k_o(n); with n = 1 the last pair opens into the ruptured state at rate k_ot; an end with i > 0 (or j > 0) re-closes one pair at rate k_c(n + 1); the ruptured state is absorbing; bulges, internal loops and sliding are neglected. Column k of T holds the outflow of state k on the diagonal and the inflows to the other states off the diagonal.

    Parameters
    ----------
    n_bp : int
        Number of base pairs (>= 2).
    rates : np.ndarray
        Array [k_ot, k_o(2), ..., k_o(n_bp), k_c(2), ..., k_c(n_bp)] of length 2 n_bp - 1.

    Returns
    -------
    generator : np.ndarray
        Generator matrix T of shape (N_states, N_states).

    Raises
    ------
    ValueError
        If n_bp < 2 or rates is not a finite nonnegative array of length 2 n_bp - 1.
    """
    return generator

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq, curve_fit


def _check_int(n, name, lo):
    if isinstance(n, bool) or int(n) != n or int(n) < lo:
        raise ValueError(f"{name} must be an integer >= {lo}")
    return int(n)


def _oracle_master_equation_generator(n_bp: int, rates: "np.ndarray") -> "np.ndarray":
    """Generator T of dP/dt = T P over the states (i, j), i + j <= n_bp - 1, plus the ruptured state last.

    States are ordered lexicographically by (i, j) (i = ruptured pairs from the left, j from the right, n = n_bp
    - i - j closed pairs) with the ruptured single-strand state as the final index;
    rates = [k_ot, k_o(2..N), k_c(2..N)]. From a state with n > 1 closed pairs each end opens at k_o(n); with
    n = 1 the last pair opens at k_ot into the ruptured state; an end with i > 0 (or j > 0) closes at
    k_c(n + 1); the ruptured state is absorbing.
    """
    N = _check_int(n_bp, "n_bp", 2)
    r = np.asarray(rates, dtype=float)
    if r.ndim != 1 or r.size != 2 * N - 1 or not np.all(np.isfinite(r)) or np.any(r < 0.0):
        raise ValueError("rates must be a finite nonnegative array of length 2 n_bp - 1")
    k_ot = r[0]
    k_o = {n: r[n - 1] for n in range(2, N + 1)}
    k_c = {n: r[N + n - 2] for n in range(2, N + 1)}
    states = [(i, j) for i in range(N) for j in range(N) if i + j <= N - 1]
    idx = {s: k for k, s in enumerate(states)}
    size = len(states) + 1
    last = size - 1
    Tm = np.zeros((size, size), dtype=np.float64)
    for (i, j), k in idx.items():
        n = N - i - j
        if n == 1:
            Tm[last, k] += k_ot
            Tm[k, k] -= k_ot
        else:
            for tgt in ((i + 1, j), (i, j + 1)):
                Tm[idx[tgt], k] += k_o[n]
                Tm[k, k] -= k_o[n]
        if i > 0:
            Tm[idx[(i - 1, j)], k] += k_c[n + 1]
            Tm[k, k] -= k_c[n + 1]
        if j > 0:
            Tm[idx[(i, j - 1)], k] += k_c[n + 1]
            Tm[k, k] -= k_c[n + 1]
    return Tm

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\nn_bp = 4\nrates = np.array([5.0, 0.2, 0.2, 0.25, 0.9, 0.95, 1.1])\n",
            "call": "np.asarray(master_equation_generator(n_bp, rates))",
            "gold_call": "np.asarray(_oracle_master_equation_generator(n_bp, rates))",
        },
        {
            "setup": "import numpy as np\nn_bp = 2\nrates = np.array([2.0, 0.3, 1.2])\n",
            "call": "np.asarray(master_equation_generator(n_bp, rates))",
            "gold_call": "np.asarray(_oracle_master_equation_generator(n_bp, rates))",
        },
        {
            "setup": "import numpy as np\ndh_stack, ds_stack, dh_non, ds_non = -7.6 * 4.184, -0.0920, 4.6 * 4.184, 0.0502\nlam_ss, l_ss = 0.77, 0.7\nn_bp = 9\nrates = _oracle_transition_rates(6.0, n_bp, 303.15, dh_stack, ds_stack, dh_non, ds_non, lam_ss, l_ss)\n",
            "call": "np.asarray(master_equation_generator(n_bp, rates))",
            "gold_call": "np.asarray(_oracle_master_equation_generator(n_bp, rates))",
        },
        {
            "setup": "import numpy as np\nrates = np.array([5.0, 0.2, 0.9])\ndef run_model():\n    try:\n        master_equation_generator(4, rates)\n        return 0\n    except ValueError:\n        return 1\ndef run_oracle():\n    try:\n        _oracle_master_equation_generator(4, rates)\n        return 0\n    except ValueError:\n        return 1\n",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]

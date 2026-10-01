"""
The reduced hierarchy dy/dt = L y for y = (p_tilde^(0), p_tilde^(1), ..., p_tilde^(K)) is a linear autonomous system, so once its initial vector is fixed the whole trajectory is determined. The initial condition follows from the requirement that the expansion reproduce the prescribed reduced probabilities at t = 0: the zeroth-order reduced vector equals the initial reduced distribution p_tilde_0, and all higher-order reduced terms start at zero. The number of orders is read off the sizes: L has K + 1 blocks, each of the length of p_tilde_0, along each side.

What is wanted here is not the state at one instant. A rate measurement on a catalyst does not read the surface at a single moment; it collects product over a sampling window that opens after the run has started, and reports the mean over that window. The quantity the rest of the pipeline needs is therefore the time average of each reduced coefficient vector over the window [t_start, t_end],

    ybar = (1 / (t_end - t_start)) * integral from t_start to t_end of y(s) ds,

and it has to be exact. Nothing in this task may depend on a step size, so neither the trajectory nor the average may be produced by an explicit time-stepping scheme or by quadrature on samples of it: the average is a closed-form functional of L, of the initial vector and of the two window edges, and must be evaluated as one.

Two properties of L are worth having in mind. It is block lower triangular, with the reduced generator repeated along its diagonal, so the zeroth-order block evolves on its own and each higher block is driven by the ones below it. And its diagonal blocks are generators of a probability-conserving process, so L is singular.

The K + 1 rows of the result are the eps-independent coefficient vectors of the window average; the k-th order truncation is recombined from them afterwards as ybar^(0) + eps ybar^(1) + ... + eps^k ybar^(k).

The reduced hierarchy dy/dt = L y for y = (p_tilde^(0), p_tilde^(1), ..., p_tilde^(K)) is a linear autonomous system, so once its initial vector is fixed the whole trajectory is determined. The initial condition follows from the requirement that the expansion reproduce the prescribed reduced probabilities at t = 0: the zeroth-order reduced vector equals the initial reduced distribution p_tilde_0, and all higher-order reduced terms start at zero. The number of orders is read off the sizes: L has K + 1 blocks, each of the length of p_tilde_0, along each side.

Returns
-------
np.ndarray of float with shape (K + 1, n): row k is the average over the sampling window [t_start, t_end] of the k-th order reduced coefficient vector of the hierarchy started from (ptilde0, 0, ..., 0), evaluated in closed form with no time discretisation.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def window_averaged_state(L: "np.ndarray", ptilde0: "np.ndarray", t_start: float, t_end: float) -> "np.ndarray": """Parameters: hierarchy generator L, initial reduced distribution ptilde0, and a valid time window. Returns: exact window-averaged coefficient vectors. Raises: ValueError for invalid arrays, distribution, or time window."""; return coeffs

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_window_averaged_state(L: "np.ndarray", ptilde0: "np.ndarray",
                                  t_start: float, t_end: float) -> "np.ndarray":
    from scipy.linalg import expm
    L = np.asarray(L, dtype=float)
    ptilde0 = np.asarray(ptilde0, dtype=float).reshape(-1)
    if L.ndim != 2 or L.shape[0] != L.shape[1]:
        raise ValueError("L must be a square 2D array")
    n = ptilde0.size
    if n < 1 or L.shape[0] % n != 0 or L.shape[0] < 2 * n:
        raise ValueError("L must have shape ((K+1) n, (K+1) n) with K >= 1 and n = len(ptilde0)")
    if not (np.all(np.isfinite(L)) and np.all(np.isfinite(ptilde0))):
        raise ValueError("inputs must be finite")
    if np.any(ptilde0 < -1e-12) or abs(ptilde0.sum() - 1.0) > 1e-8:
        raise ValueError("ptilde0 must be a probability distribution")
    t_start, t_end = float(t_start), float(t_end)
    if not np.isfinite(t_start) or t_start < 0.0:
        raise ValueError("t_start must be a finite non-negative number")
    if not np.isfinite(t_end) or t_end <= t_start:
        raise ValueError("t_end must be finite and greater than t_start")
    size = L.shape[0]
    blocks = size // n
    # the last column of exp([[L, y0], [0, 0]] t) holds the integral of exp(L s) y0 over [0, t]
    aug = np.zeros((size + 1, size + 1))
    aug[:size, :size] = L
    aug[:n, size] = ptilde0
    ybar = (expm(aug * t_end)[:size, size] - expm(aug * t_start)[:size, size]) / (t_end - t_start)
    return ybar.reshape(blocks, n)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    setup_common = """import numpy as np
def _tri(n, seed, corr=1.0, blocks=4):
    g = np.random.default_rng(seed)
    W = g.uniform(0.1, 2.0, size=(n, n)); np.fill_diagonal(W, 0.0)
    W[np.diag_indices(n)] = -W.sum(axis=0)
    C = [W]
    for k in range(blocks - 1):
        M = g.normal(size=(n, n)) * corr
        M -= M.sum(axis=0, keepdims=True) / n
        C.append(M)
    L = np.zeros((blocks * n, blocks * n))
    for r in range(blocks):
        for c in range(r + 1):
            L[r * n:(r + 1) * n, c * n:(c + 1) * n] = C[r - c]
    return L
def _fp(M):
    M = np.asarray(M, dtype=float)
    W = np.cos(np.arange(M.size, dtype=float)).reshape(M.shape)
    return float(np.sum(M * W) + 1e-3 * M.shape[0])
"""
    def _raises(args):
        return """
def run_model():
    try:
        window_averaged_state(%s)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_window_averaged_state(%s)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""" % (args, args)
    return [
        # --- Normal: a four-state hierarchy to third order with the window
        # opening well after the start ---
        {"setup": setup_common + "L = _tri(4, 11)\np0 = np.array([1.0, 0.0, 0.0, 0.0])\n",
         "call": "_fp(window_averaged_state(L, p0, 0.4, 1.6))",
         "gold_call": "_fp(_oracle_window_averaged_state(L, p0, 0.4, 1.6))"},
        # --- Normal: sixteen classes at third order, starting from a single class
        # and averaged over a short window ---
        {"setup": setup_common + "L = _tri(16, 77)\np0 = np.zeros(16)\np0[0] = 1.0\n",
         "call": "_fp(window_averaged_state(L, p0, 0.03, 0.09))",
         "gold_call": "_fp(_oracle_window_averaged_state(L, p0, 0.03, 0.09))"},
        # --- Normal: 216 classes at third order, the size of the task's own
        # hierarchy, over the task's sampling window ---
        {"setup": setup_common + "L = _tri(216, 88)\np0 = np.zeros(216)\np0[0] = 1.0\n",
         "call": "_fp(window_averaged_state(L, p0, 0.02, 0.08))",
         "gold_call": "_fp(_oracle_window_averaged_state(L, p0, 0.02, 0.08))"},
        # --- Normal: a mixed initial distribution over two classes ---
        {"setup": setup_common + "L = _tri(2, 22)\np0 = np.array([0.25, 0.75])\n",
         "call": "_fp(window_averaged_state(L, p0, 0.2, 0.8))",
         "gold_call": "_fp(_oracle_window_averaged_state(L, p0, 0.2, 0.8))"},
        # --- Normal: couplings much larger than the diagonal generator ---
        {"setup": setup_common + "L = _tri(3, 33, corr=25.0)\np0 = np.array([0.2, 0.3, 0.5])\n",
         "call": "_fp(window_averaged_state(L, p0, 0.05, 0.35))",
         "gold_call": "_fp(_oracle_window_averaged_state(L, p0, 0.05, 0.35))"},
        # --- Normal: a first-order hierarchy, two blocks along each side ---
        {"setup": setup_common + "L = _tri(3, 44, blocks=2)\np0 = np.array([0.0, 1.0, 0.0])\n",
         "call": "_fp(window_averaged_state(L, p0, 0.1, 0.7))",
         "gold_call": "_fp(_oracle_window_averaged_state(L, p0, 0.1, 0.7))"},
        # --- Boundary: a window that opens at t = 0 ---
        {"setup": setup_common + "L = _tri(2, 22)\np0 = np.array([1.0, 0.0])\n",
         "call": "_fp(window_averaged_state(L, p0, 0.0, 0.5))",
         "gold_call": "_fp(_oracle_window_averaged_state(L, p0, 0.0, 0.5))"},
        # --- Boundary: a very short window ---
        {"setup": setup_common + "L = _tri(2, 22)\np0 = np.array([1.0, 0.0])\n",
         "call": "_fp(window_averaged_state(L, p0, 0.30, 0.3001))",
         "gold_call": "_fp(_oracle_window_averaged_state(L, p0, 0.30, 0.3001))"},
        # --- Boundary: a single class ---
        {"setup": setup_common + "L = _tri(1, 44)\np0 = np.array([1.0])\n",
         "call": "_fp(window_averaged_state(L, p0, 0.1, 0.9))",
         "gold_call": "_fp(_oracle_window_averaged_state(L, p0, 0.1, 0.9))"},
        # --- Edge: no couplings between the orders ---
        {"setup": setup_common + "L = _tri(4, 55, corr=0.0)\np0 = np.array([0.1, 0.2, 0.3, 0.4])\n",
         "call": "_fp(window_averaged_state(L, p0, 0.5, 2.0))",
         "gold_call": "_fp(_oracle_window_averaged_state(L, p0, 0.5, 2.0))"},
        # --- Edge: a fifth-order hierarchy over a long window ---
        {"setup": setup_common + "L = _tri(2, 66, blocks=6)\np0 = np.array([0.6, 0.4])\n",
         "call": "_fp(window_averaged_state(L, p0, 5.0, 25.0))",
         "gold_call": "_fp(_oracle_window_averaged_state(L, p0, 5.0, 25.0))"},
        # --- Invalid: a window that closes before it opens ---
        {"setup": setup_common + "L = _tri(2, 22)\np0 = np.array([1.0, 0.0])\n" + _raises("L, p0, 0.8, 0.2"),
         "call": "run_model()", "gold_call": "run_oracle()"},
        # --- Invalid: an unnormalised initial distribution ---
        {"setup": setup_common + "L = _tri(2, 22)\np0 = np.array([0.5, 0.2])\n" + _raises("L, p0, 0.1, 0.4"),
         "call": "run_model()", "gold_call": "run_oracle()"},
        # --- Invalid: a generator whose size is not a multiple of the number of
        # classes ---
        {"setup": setup_common + "L = _tri(3, 33)\np0 = np.full(5, 0.2)\n" + _raises("L, p0, 0.1, 0.4"),
         "call": "run_model()", "gold_call": "run_oracle()"},
        # --- Invalid: a generator holding a single block, so no correction order ---
        {"setup": setup_common + "L = _tri(3, 34, blocks=1)\np0 = np.array([0.2, 0.3, 0.5])\n" + _raises("L, p0, 0.1, 0.4"),
         "call": "run_model()", "gold_call": "run_oracle()"},
    ]

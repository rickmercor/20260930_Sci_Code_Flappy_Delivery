"""
Compute the regime-switching factor of the characteristic function: the conditional expectation, given the initial regime, of the exponential of the regime-dependent part of the constant term.

With a finite-state Markov chain X of generator G (row i holds the rates out of regime i and sums to zero), the regime-dependent coefficients enter the constant term through g_k(s) = -(1/2) sig2_k (i delta + delta^2) + kappa1 theta1_k D(s) + kappa2 theta2_k E(s), where D and E are the two variance coefficients at remaining time s and sig2_k is the total constant variance rate of the log price in regime k. The factor is h_j = E[exp(int g_{X_u}(tau - u) du) | X_0 = j]. Viewed as a function of remaining time, the vector h solves the linear system dh/ds = (G + diag g(s)) h with h(0) equal to a vector of ones. The system is stiff for large delta, so it is advanced over n_steps steps of equal length Delta by the exponential midpoint rule h(s + Delta) = exp(Delta (G + diag g(s + Delta/2))) h(s), where exp is the matrix exponential.

Returns
-------
np.ndarray, float, shape (2, n): real and imaginary parts of h at the initial regime, for the n transform values.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def regime_factor(delta: np.ndarray, tau: float, generator: np.ndarray, sig2: np.ndarray,
                  kappa1: float, theta1: np.ndarray, xi1: float, d_c0: float, d_c1: float,
                  kappa2: float, theta2: np.ndarray, xi2: float, e_c0: float, e_c1: float,
                  initial_regime: int, n_steps: int = 200) -> np.ndarray:
    '''Regime-switching factor of the characteristic function.

    Parameters
    ----------
    delta : np.ndarray
        One-dimensional array of n >= 1 finite transform values.
    tau : float
        Remaining time, tau >= 0.
    generator : np.ndarray
        Shape (k, k) generator of the chain, k >= 1, non-negative off-diagonal
        entries and zero row sums.
    sig2 : np.ndarray
        Shape (k,) constant variance rate of the log price in each regime.
    kappa1, kappa2 : float
        Mean-reversion rates of the two variance factors.
    theta1, theta2 : np.ndarray
        Shape (k,) long-run variances of the two factors in each regime.
    xi1, xi2 : float
        Volatilities of the two variance factors, > 0.
    d_c0, d_c1, e_c0, e_c1 : float
        Linear coefficients (c0, c1) of the Riccati equations of D and E, as
        in ``variance_riccati``.
    initial_regime : int
        Regime at time 0, numbered from 1 to k.
    n_steps : int
        Number of exponential midpoint steps, >= 1.

    Returns
    -------
    h : np.ndarray
        Shape (2, n) float array: real and imaginary parts of h_j.

    Raises
    ------
    ValueError
        If any argument is outside its domain or the shapes are inconsistent.

    Notes
    -----
    Obtain D and E from ``variance_riccati``. Include every import your
    implementation needs inside the function body.
    '''
    return np.zeros((2, np.atleast_1d(delta).size), dtype=float)  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_regime_factor(delta: np.ndarray, tau: float, generator: np.ndarray, sig2: np.ndarray,
                          kappa1: float, theta1: np.ndarray, xi1: float, d_c0: float, d_c1: float,
                          kappa2: float, theta2: np.ndarray, xi2: float, e_c0: float, e_c1: float,
                          initial_regime: int, n_steps: int = 200) -> np.ndarray:
    import numpy as np

    d = np.atleast_1d(np.asarray(delta, dtype=complex))
    G = np.asarray(generator, dtype=float)
    s2 = np.atleast_1d(np.asarray(sig2, dtype=float))
    t1 = np.atleast_1d(np.asarray(theta1, dtype=float))
    t2 = np.atleast_1d(np.asarray(theta2, dtype=float))
    if G.ndim != 2 or G.shape[0] != G.shape[1] or G.shape[0] < 1:
        raise ValueError("generator must be a square matrix")
    k = G.shape[0]
    if s2.shape != (k,) or t1.shape != (k,) or t2.shape != (k,):
        raise ValueError("sig2, theta1 and theta2 must have one entry per regime")
    if not (np.all(np.isfinite(G)) and np.all(np.isfinite(s2)) and np.all(np.isfinite(t1)) and np.all(np.isfinite(t2))):
        raise ValueError("inputs must be finite")
    off = G - np.diag(np.diag(G))
    if np.any(off < 0.0) or not np.allclose(G.sum(axis=1), 0.0, atol=1e-12):
        raise ValueError("generator must have non-negative off-diagonal entries and zero row sums")
    if isinstance(initial_regime, bool) or not isinstance(initial_regime, (int, np.integer)) or not (1 <= int(initial_regime) <= k):
        raise ValueError("initial_regime must be an integer between 1 and k")
    if isinstance(n_steps, bool) or not isinstance(n_steps, (int, np.integer)) or int(n_steps) < 1:
        raise ValueError("n_steps must be an integer >= 1")
    for name, value in (("tau", tau), ("kappa1", kappa1), ("kappa2", kappa2)):
        if isinstance(value, bool) or not np.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
    if float(tau) < 0.0:
        raise ValueError("tau must be non-negative")

    q = 1j * d + d * d
    n = int(n_steps)
    h_step = float(tau) / n

    def g_at(s):
        Dr = _oracle_variance_riccati(d, s, xi1, d_c0, d_c1)
        Er = _oracle_variance_riccati(d, s, xi2, e_c0, e_c1)
        D = Dr[0] + 1j * Dr[1]
        E = Er[0] + 1j * Er[1]
        return (-0.5 * s2[:, None] * q[None, :] + float(kappa1) * t1[:, None] * D[None, :]
                + float(kappa2) * t2[:, None] * E[None, :])          # shape (k, n)

    def expm_batch(A):
        # Scaling and squaring with a Taylor series, for a stack of (k, k)
        # complex matrices of shape (n, k, k).
        norm = np.max(np.sum(np.abs(A), axis=2), axis=1)
        sq = np.maximum(0, np.ceil(np.log2(np.maximum(norm, 1e-300) / 0.5))).astype(int)
        X = A / (2.0 ** sq)[:, None, None]
        eye = np.broadcast_to(np.eye(k, dtype=complex), A.shape)
        out = eye.copy()
        term = eye.copy()
        for j in range(1, 19):
            term = term @ X / j
            out = out + term
        for i in range(int(sq.max()) if sq.size else 0):
            mask = sq > i
            out[mask] = out[mask] @ out[mask]
        return out

    hv = np.ones((d.size, k), dtype=complex)
    for step in range(n):
        gm = g_at((step + 0.5) * h_step).T                          # shape (n, k)
        A = h_step * (G[None, :, :] + gm[:, :, None] * np.eye(k)[None, :, :])
        hv = np.einsum("nij,nj->ni", expm_batch(A), hv)
    out = hv[:, int(initial_regime) - 1]
    return np.vstack([out.real, out.imag]).astype(float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Pinned: at delta = -i the regime coefficients vanish, and with
        # zero row sums the factor is exactly one in every regime.
        {
            "setup": """import numpy as np
G = np.array([[-0.5, 0.5], [0.45, -0.45]])
EXPECTED = np.vstack([np.ones(2), np.zeros(2)])
""",
            "call": "regime_factor(np.array([-1j, -1j]), 1.0, G, np.array([0.06, 0.06]), 2.0, np.array([0.1, 0.3]), 0.1, -2.0, -0.05, 2.0, np.array([0.1, 0.3]), 0.1, -1.95, 0.05, 2, 50)",
            "gold_call": "EXPECTED",
        },
        # --- Pinned: when both regimes carry the same coefficients the chain
        # is irrelevant, so the factor equals the one with no switching.
        {
            "setup": """import numpy as np
d = np.array([0.4, 2.5, 8.0])
args = (np.array([0.06, 0.06]), 2.0, np.array([0.2, 0.2]), 0.1, -2.0, -0.05, 2.0, np.array([0.2, 0.2]), 0.1, -1.95, 0.05)
Gs = np.array([[-0.5, 0.5], [0.45, -0.45]])
EXPECTED = _oracle_regime_factor(d, 1.0, np.zeros((2, 2)), *args, 1, 200)
""",
            "call": "regime_factor(d, 1.0, Gs, *args, 1, 200)",
            "gold_call": "EXPECTED",
        },
        # --- Normal: the task's two-regime chain starting in regime 2.
        {
            "setup": """import numpy as np
d = np.linspace(0.05, 40.0, 30)
G = np.array([[-0.5, 0.5], [0.45, -0.45]])
""",
            "call": "regime_factor(d, 1.0, G, np.array([0.06, 0.06]), 2.0, np.array([0.1, 0.3]), 0.1, -2.0, -0.05, 2.0, np.array([0.1, 0.3]), 0.1, -1.95, 0.05, 2, 200)",
            "gold_call": "_oracle_regime_factor(d, 1.0, G, np.array([0.06, 0.06]), 2.0, np.array([0.1, 0.3]), 0.1, -2.0, -0.05, 2.0, np.array([0.1, 0.3]), 0.1, -1.95, 0.05, 2, 200)",
        },
        # --- Boundary: a single step and shifted transform values.
        {
            "setup": """import numpy as np
d = np.linspace(0.05, 10.0, 8) - 1j
G = np.array([[-1.0, 1.0], [2.0, -2.0]])
""",
            "call": "regime_factor(d, 0.3, G, np.array([0.04, 0.1]), 1.5, np.array([0.05, 0.2]), 0.3, -1.5, -0.15, 1.0, np.array([0.1, 0.1]), 0.2, -0.9, 0.1, 1, 1)",
            "gold_call": "_oracle_regime_factor(d, 0.3, G, np.array([0.04, 0.1]), 1.5, np.array([0.05, 0.2]), 0.3, -1.5, -0.15, 1.0, np.array([0.1, 0.1]), 0.2, -0.9, 0.1, 1, 1)",
        },
        # --- Edge: three regimes with fast switching.
        {
            "setup": """import numpy as np
d = np.array([0.2, 3.0, 15.0])
G = np.array([[-6.0, 4.0, 2.0], [1.0, -3.0, 2.0], [0.5, 0.5, -1.0]])
""",
            "call": "regime_factor(d, 2.0, G, np.array([0.02, 0.06, 0.12]), 3.0, np.array([0.05, 0.1, 0.4]), 0.5, -3.0, -0.25, 3.0, np.array([0.05, 0.1, 0.4]), 0.5, -2.75, 0.25, 3, 120)",
            "gold_call": "_oracle_regime_factor(d, 2.0, G, np.array([0.02, 0.06, 0.12]), 3.0, np.array([0.05, 0.1, 0.4]), 0.5, -3.0, -0.25, 3.0, np.array([0.05, 0.1, 0.4]), 0.5, -2.75, 0.25, 3, 120)",
        },
        # --- Invalid: rows of the generator do not sum to zero ---
        {
            "setup": """import numpy as np
G = np.array([[-0.5, 0.4], [0.45, -0.45]])
def run_model():
    try:
        regime_factor(np.array([1.0]), 1.0, G, np.array([0.06, 0.06]), 2.0, np.array([0.1, 0.3]), 0.1, -2.0, 0.0, 2.0, np.array([0.1, 0.3]), 0.1, -2.0, 0.0, 1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_regime_factor(np.array([1.0]), 1.0, G, np.array([0.06, 0.06]), 2.0, np.array([0.1, 0.3]), 0.1, -2.0, 0.0, 2.0, np.array([0.1, 0.3]), 0.1, -2.0, 0.0, 1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: initial regime out of range ---
        {
            "setup": """import numpy as np
G = np.array([[-0.5, 0.5], [0.45, -0.45]])
def run_model():
    try:
        regime_factor(np.array([1.0]), 1.0, G, np.array([0.06, 0.06]), 2.0, np.array([0.1, 0.3]), 0.1, -2.0, 0.0, 2.0, np.array([0.1, 0.3]), 0.1, -2.0, 0.0, 3)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_regime_factor(np.array([1.0]), 1.0, G, np.array([0.06, 0.06]), 2.0, np.array([0.1, 0.3]), 0.1, -2.0, 0.0, 2.0, np.array([0.1, 0.3]), 0.1, -2.0, 0.0, 3)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]

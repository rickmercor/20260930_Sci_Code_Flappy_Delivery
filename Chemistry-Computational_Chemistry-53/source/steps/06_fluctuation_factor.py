"""
Implement fluctuation_factor, which evaluates the pinned fluctuation factor of a ring-polymer
instanton: the ratio of the fluctuation determinant of the ring polymer pinned at its dividing
surface to those of the two wells, each well represented by the part of the ring assigned to it.

In instanton theory the prefactor of a tunnelling rate or splitting comes from Gaussian
fluctuations about the optimized path, measured relative to the fluctuations of the particle
localized in the wells.

Returns
-------
np.ndarray of 4 floats [logdet_J_pin, logdet_J_low, logdet_J_high, Phi_pin]: natural log-determinants of the pinned and well matrices and the pinned fluctuation factor
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def fluctuation_factor(beads: "np.ndarray", params: "np.ndarray", beta: float, n_beads: int, k: int) -> "np.ndarray":
    '''Pinned fluctuation factor Phi_pin of a ring-polymer path and its log-determinants.

    Parameters
    ----------
    beads : np.ndarray
        Half-ring bead positions y_0, ..., y_{N/2} in the folded convention of
        half_ring_action, with y_0 on the lower-well side (normally the instanton of
        optimize_instanton).
    params : np.ndarray
        Model parameters [omega_l, omega_r, x0, eps, V01] of the surface V(x) of
        two_diabat_potential.
    beta : float
        Inverse temperature in reduced units (hbar = 1).
    n_beads : int
        Number of beads N of the full ring polymer, an even integer >= 4.
    k : int
        Index of the dividing-surface bead of the half ring, 1 <= k <= N/2 - 1 (normally the
        first entry of select_dividing_surface).

    Returns
    -------
    factors : np.ndarray
        Array of 4 floats [logdet_J_pin, logdet_J_low, logdet_J_high, Phi_pin]; the first
        three are natural logarithms of determinants. With dtau = beta*hbar/N and m = 1, J
        is (dtau/m) times the Hessian of the full-ring action S_N with respect to the N ring bead
        positions x_j, evaluated at the ring built from beads. J_pin is J with the rows and
        columns of the two dividing-surface beads x_k and x_{N-k} removed. J_low (J_high) is
        the same matrix for a ring of n_low (n_high) beads with the same dtau and every bead at
        the lower (higher) minimum of locate_stationary_points, where n_low and n_high are the
        arc lengths of select_dividing_surface for this k. Finally,
        Phi_pin = sqrt(dtau/m) * [det J_pin / (det J_low * det J_high)]**(1/4).

    Raises
    ------
    ValueError
        If k is not an integer in 1..N/2 - 1, n_beads is not an even integer >= 4, beta is
        not positive, beads does not hold N/2 + 1 finite values, params is invalid, or the
        surface does not have exactly two minima.
    '''
    return factors

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _dividing_bead(k: int, n_beads: int) -> int:
    """Validated dividing-surface bead index 1 <= k <= N/2 - 1."""
    if int(k) != k or not 1 <= int(k) <= int(n_beads) // 2 - 1:
        raise ValueError("k must be an interior bead of the half ring")
    return int(k)


def _logdet_open_chain(diag: "np.ndarray") -> float:
    """log|det| of the symmetric tridiagonal matrix with diagonal diag and off-diagonals -1."""
    log_det, pivot = 0.0, None
    for a in diag:
        pivot = a if pivot is None else a - 1.0 / pivot
        log_det += np.log(abs(pivot))
    return float(log_det)


def _logdet_collapsed_ring(n: int, c: float) -> float:
    """log det of the n x n cyclic matrix with diagonal 2 + c and neighbour couplings -1."""
    theta = n * float(np.arccosh(1.0 + 0.5 * c))
    return theta + 2.0 * float(np.log1p(-np.exp(-theta)))


def _oracle_fluctuation_factor(beads: "np.ndarray", params: "np.ndarray", beta: float, n_beads: int, k: int) -> "np.ndarray":
    y, tau = _half_ring_beads(beads, beta, n_beads)
    k = _dividing_bead(k, n_beads)
    last = y.size - 1
    omega_low, omega_high = _oracle_locate_stationary_points(params)[6:]
    diag = 2.0 + tau * tau * _oracle_two_diabat_potential(y, params)[2]
    # removing x_k and x_{N-k} leaves two open chains: one through y_0, one through y_{N/2}
    low_chain = np.concatenate([diag[k - 1:0:-1], diag[:k]])
    high_chain = np.concatenate([diag[k + 1:], diag[last - 1:k:-1]])
    logdet_pin = _logdet_open_chain(low_chain) + _logdet_open_chain(high_chain)
    logdet_low = _logdet_collapsed_ring(2 * k, (tau * omega_low) ** 2)
    logdet_high = _logdet_collapsed_ring(int(n_beads) - 2 * k, (tau * omega_high) ** 2)
    phi = np.sqrt(tau) * np.exp(0.25 * (logdet_pin - logdet_low - logdet_high))
    return np.array([logdet_pin, logdet_low, logdet_high, phi])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Typical: instanton of the asymmetric model at beta = 300, N = 1024, pinned at its
        #     maximum-potential bead ---
        {
            "setup": """import numpy as np
params = np.array([1.0, 0.3, 4.6, -0.19, 1.5])
beads = _oracle_optimize_instanton(params, 300.0, 1024)
k = int(_oracle_select_dividing_surface(beads, params, 300.0, 1024)[0])
""",
            "call": "fluctuation_factor(beads.copy(), params.copy(), 300.0, 1024, k)",
            "gold_call": "_oracle_fluctuation_factor(beads.copy(), params.copy(), 300.0, 1024, k)",
            "tol": 1e-9,
        },
        # --- Typical: larger displacement and coupling, N = 512 ---
        {
            "setup": """import numpy as np
params = np.array([1.0, 0.3, 5.0, -0.46, 2.0])
beads = _oracle_optimize_instanton(params, 300.0, 512)
k = int(_oracle_select_dividing_surface(beads, params, 300.0, 512)[0])
""",
            "call": "fluctuation_factor(beads.copy(), params.copy(), 300.0, 512, k)",
            "gold_call": "_oracle_fluctuation_factor(beads.copy(), params.copy(), 300.0, 512, k)",
            "tol": 1e-9,
        },
        # --- Boundary: coarse ring (N = 256) whose lower-well arc has only four segments ---
        {
            "setup": """import numpy as np
params = np.array([1.0, 0.3, 5.0, -0.46, 2.0])
beads = _oracle_optimize_instanton(params, 300.0, 256)
k = int(_oracle_select_dividing_surface(beads, params, 300.0, 256)[0])
""",
            "call": "fluctuation_factor(beads.copy(), params.copy(), 300.0, 256, k)",
            "gold_call": "_oracle_fluctuation_factor(beads.copy(), params.copy(), 300.0, 256, k)",
            "tol": 1e-9,
        },
        # --- Edge: mirror-symmetric surface; both wells receive half of the ring ---
        {
            "setup": """import numpy as np
params = np.array([1.0, 1.0, 3.0, 0.0, 1.0])
beads = _oracle_optimize_instanton(params, 300.0, 512)
k = int(_oracle_select_dividing_surface(beads, params, 300.0, 512)[0])
""",
            "call": "fluctuation_factor(beads.copy(), params.copy(), 300.0, 512, k)",
            "gold_call": "_oracle_fluctuation_factor(beads.copy(), params.copy(), 300.0, 512, k)",
            "tol": 1e-9,
        },
        # --- Edge: the same instanton as the first case, pinned five beads further towards the
        #     higher well ---
        {
            "setup": """import numpy as np
params = np.array([1.0, 0.3, 4.6, -0.19, 1.5])
beads = _oracle_optimize_instanton(params, 300.0, 1024)
k = int(_oracle_select_dividing_surface(beads, params, 300.0, 1024)[0]) + 5
""",
            "call": "fluctuation_factor(beads.copy(), params.copy(), 300.0, 1024, k)",
            "gold_call": "_oracle_fluctuation_factor(beads.copy(), params.copy(), 300.0, 1024, k)",
            "tol": 1e-9,
        },
        # --- Invalid: the dividing surface placed on the turning-point bead y_{N/2} ---
        {
            "setup": """import numpy as np
params = np.array([1.0, 0.3, 4.6, -0.19, 1.5])
beads = np.linspace(-3.6, 4.47, 33)
def run_model():
    try:
        fluctuation_factor(beads.copy(), params.copy(), 300.0, 64, 32)
        return 0
    except ValueError:
        return 1
def run_oracle():
    try:
        _oracle_fluctuation_factor(beads.copy(), params.copy(), 300.0, 64, 32)
        return 0
    except ValueError:
        return 1
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]

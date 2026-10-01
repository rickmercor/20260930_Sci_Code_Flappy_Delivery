"""
Implement half_ring_action, which evaluates the discretized imaginary-time action of a folded
ring polymer on the two-state surface, together with its gradient and Hessian with respect to
the bead positions.

A ring polymer of N beads at inverse temperature beta represents a closed imaginary-time path of
length beta*hbar by N positions joined by harmonic springs. A tunnelling orbit runs from one
turning point to the other and returns along the same track, so only half of the ring needs to be
represented. The folded half ring runs from one turning point to the other, and the returning half
of the ring is its mirror image in imaginary time.

Returns
-------
tuple (S_half float, gradient of shape (N/2+1,), hessian_diag of shape (N/2+1,), hessian_offdiag of shape (N/2,)): half the discretized ring-polymer action and its bead derivatives
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def half_ring_action(beads: "np.ndarray", params: "np.ndarray", beta: float, n_beads: int) -> tuple:
    '''Action of the folded half ring polymer and its first and second bead derivatives.

    Parameters
    ----------
    beads : np.ndarray
        Positions y_0, ..., y_{N/2} of the N/2 + 1 beads of the folded half ring. The closed
        N-bead ring polymer is x_j = y_j for 0 <= j <= N/2 and x_j = y_{N-j} for
        N/2 < j < N, so the end beads y_0 and y_{N/2} (the turning points) appear once on the
        ring and every other bead appears twice.
    params : np.ndarray
        Model parameters [omega_l, omega_r, x0, eps, V01] of the surface V(x) of
        two_diabat_potential.
    beta : float
        Inverse temperature in reduced units (hbar = 1); neighbouring ring beads are
        separated by the imaginary-time step beta*hbar/N.
    n_beads : int
        Number of beads N of the full ring polymer, an even integer >= 4.

    Returns
    -------
    result : tuple
        (S_half, gradient, hessian_diag, hessian_offdiag):
        S_half : float, half of the standard discretized Euclidean action S_N of the closed
            N-bead ring polymer (harmonic springs between neighbouring beads, potential energy
            at every bead, particle mass m = 1, hbar = 1), as a function of the half-ring beads.
        gradient : np.ndarray, shape (N/2 + 1,), dS_half/dy_j.
        hessian_diag : np.ndarray, shape (N/2 + 1,), d2S_half/dy_j^2.
        hessian_offdiag : np.ndarray, shape (N/2,), d2S_half/(dy_j dy_{j+1}); all other
            Hessian elements vanish.

    Raises
    ------
    ValueError
        If n_beads is not an even integer >= 4, beta is not positive, beads does not hold
        N/2 + 1 finite values, or params is invalid.
    '''
    return (S_half, gradient, hessian_diag, hessian_offdiag)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _ring_step(beta: float, n_beads: int) -> float:
    """Validated imaginary-time step beta/N between neighbouring ring beads (hbar = 1)."""
    if int(n_beads) != n_beads or n_beads < 4 or int(n_beads) % 2:
        raise ValueError("n_beads must be an even integer >= 4")
    if not (np.isfinite(beta) and beta > 0.0):
        raise ValueError("beta must be positive")
    return float(beta) / int(n_beads)


def _half_ring_beads(beads: "np.ndarray", beta: float, n_beads: int) -> tuple:
    """Validated half-ring positions and the imaginary-time step beta/N."""
    tau = _ring_step(beta, n_beads)
    y = np.asarray(beads, dtype=float).ravel()
    if y.size != int(n_beads) // 2 + 1:
        raise ValueError("beads must hold n_beads/2 + 1 positions")
    return y, tau


def _oracle_half_ring_action(beads: "np.ndarray", params: "np.ndarray", beta: float, n_beads: int) -> tuple:
    y, tau = _half_ring_beads(beads, beta, n_beads)
    v, dv, d2v = _oracle_two_diabat_potential(y, params)
    # the turning-point beads occur once on the ring, all other beads twice
    weight = np.ones(y.size)
    weight[0] = weight[-1] = 0.5
    stretch = np.diff(y)
    action = 0.5 * float(np.dot(stretch, stretch)) / tau + tau * float(np.dot(weight, v))
    gradient = tau * weight * dv
    gradient[:-1] -= stretch / tau
    gradient[1:] += stretch / tau
    hessian_diag = tau * weight * d2v + 2.0 / tau
    hessian_diag[0] -= 1.0 / tau
    hessian_diag[-1] -= 1.0 / tau
    hessian_offdiag = np.full(y.size - 1, -1.0 / tau)
    return action, gradient, hessian_diag, hessian_offdiag

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    helper = """import numpy as np
def flat(result):
    action, gradient, hessian_diag, hessian_offdiag = result
    return np.concatenate([[float(action)], np.ravel(gradient), np.ravel(hessian_diag), np.ravel(hessian_offdiag)])
"""
    invalid = """
def run_model():
    try:
        half_ring_action(beads.copy(), params.copy(), beta, n_beads)
        return 0
    except ValueError:
        return 1
def run_oracle():
    try:
        _oracle_half_ring_action(beads.copy(), params.copy(), beta, n_beads)
        return 0
    except ValueError:
        return 1
"""
    return [
        # --- Typical: a smooth kink-like trial path across the barrier, N = 1024 at beta = 300 ---
        {
            "setup": helper + """
params = np.array([1.0, 0.3, 4.6, -0.19, 1.5])
beta, n_beads = 300.0, 1024
t = np.arange(n_beads // 2 + 1) * beta / n_beads
beads = -3.6 + 8.07 / (1.0 + np.exp(-(t - 4.0) / 0.9))
""",
            "call": "flat(half_ring_action(beads.copy(), params.copy(), beta, n_beads))",
            "gold_call": "flat(_oracle_half_ring_action(beads.copy(), params.copy(), beta, n_beads))",
            "tol": 1e-10,
        },
        # --- Typical: an irregular path on the mirrored surface with few, widely spaced beads ---
        {
            "setup": helper + """
params = np.array([1.0, 0.3, -4.6, -0.19, 1.5])
beta, n_beads = 40.0, 16
beads = np.array([3.9, 3.1, 2.2, 0.4, -1.3, -3.0, -4.1, -4.5, -4.4])
""",
            "call": "flat(half_ring_action(beads.copy(), params.copy(), beta, n_beads))",
            "gold_call": "flat(_oracle_half_ring_action(beads.copy(), params.copy(), beta, n_beads))",
            "tol": 1e-10,
        },
        # --- Boundary: the smallest ring (N = 4, three half-ring beads) on the symmetric surface ---
        {
            "setup": helper + """
params = np.array([1.0, 1.0, 3.0, 0.0, 1.0])
beta, n_beads = 2.5, 4
beads = np.array([-2.7, 0.3, 2.9])
""",
            "call": "flat(half_ring_action(beads.copy(), params.copy(), beta, n_beads))",
            "gold_call": "flat(_oracle_half_ring_action(beads.copy(), params.copy(), beta, n_beads))",
            "tol": 1e-10,
        },
        # --- Edge: a ring collapsed onto one point near the barrier top (springs relaxed) ---
        {
            "setup": helper + """
params = np.array([1.0, 0.3, 5.0, -0.46, 2.0])
beta, n_beads = 300.0, 64
beads = np.full(n_beads // 2 + 1, -1.7)
""",
            "call": "flat(half_ring_action(beads.copy(), params.copy(), beta, n_beads))",
            "gold_call": "flat(_oracle_half_ring_action(beads.copy(), params.copy(), beta, n_beads))",
            "tol": 1e-10,
        },
        # --- Invalid: N + 1 beads given instead of N/2 + 1 ---
        {
            "setup": """import numpy as np
params = np.array([1.0, 0.3, 4.6, -0.19, 1.5])
beta, n_beads = 300.0, 16
beads = np.linspace(-3.6, 4.4, 17)
""" + invalid,
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
        # --- Invalid: odd number of ring beads ---
        {
            "setup": """import numpy as np
params = np.array([1.0, 0.3, 4.6, -0.19, 1.5])
beta, n_beads = 300.0, 15
beads = np.linspace(-3.6, 4.4, 8)
""" + invalid,
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]

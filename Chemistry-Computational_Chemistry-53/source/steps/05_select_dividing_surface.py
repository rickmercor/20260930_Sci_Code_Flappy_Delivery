"""
Implement select_dividing_surface, which places the dividing surface on a folded ring-polymer
path and returns how the closed ring polymer, and with it the imaginary time, is shared between
the two wells, together with the imaginary-time speed of the path at the dividing surface.

Tunnelling between wells of unequal depth is described by pinning the instanton at a dividing
surface between the wells. On the discrete ring polymer the dividing surface is a bead of the
optimized path, and its two copies on the closed ring cut the ring into two arcs, one attached to
each well. The number of ring segments on each arc fixes how the total imaginary time beta*hbar is
divided between the lower and the higher well.

Returns
-------
np.ndarray of 7 floats [k, x_sigma, n_low, n_high, tau_low, tau_high, qdot_sigma]: dividing-surface bead, arc bead counts and imaginary times of the two wells, and the path speed at the dividing surface
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def select_dividing_surface(beads: "np.ndarray", params: "np.ndarray", beta: float, n_beads: int) -> "np.ndarray":
    '''Dividing-surface bead, partition of the ring between the wells and path speed there.

    Parameters
    ----------
    beads : np.ndarray
        Half-ring bead positions y_0, ..., y_{N/2} in the folded convention of
        half_ring_action, with y_0 on the lower-well side (normally the instanton of
        optimize_instanton). All quantities are evaluated for the beads as supplied.
    params : np.ndarray
        Model parameters [omega_l, omega_r, x0, eps, V01] of the surface V(x) of
        two_diabat_potential.
    beta : float
        Inverse temperature in reduced units (hbar = 1).
    n_beads : int
        Number of beads N of the full ring polymer, an even integer >= 4.

    Returns
    -------
    surface : np.ndarray
        Array of 7 floats [k, x_sigma, n_low, n_high, tau_low, tau_high, qdot_sigma].
        k is the index of the dividing-surface bead: the interior half-ring bead
        (1 <= k <= N/2 - 1) with the largest potential energy V(y_k), the smallest such
        index in case of a tie. x_sigma = y_k. On the closed ring the dividing surface is
        formed by the two copies x_k and x_{N-k} of this bead. n_low is the number of ring
        segments (springs) on the arc between them that contains y_0, and n_high = N - n_low is
        the number on the other arc. tau_low and tau_high are the imaginary times spanned by
        the two arcs (hbar = 1). qdot_sigma is the magnitude of the imaginary-time velocity of
        the path at the dividing surface, evaluated by the central difference across bead k.

    Raises
    ------
    ValueError
        If n_beads is not an even integer >= 4, beta is not positive, beads does not hold
        N/2 + 1 finite values, or params is invalid.
    '''
    return surface

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_select_dividing_surface(beads: "np.ndarray", params: "np.ndarray", beta: float, n_beads: int) -> "np.ndarray":
    y, tau = _half_ring_beads(beads, beta, n_beads)
    potential = _oracle_two_diabat_potential(y, params)[0]
    k = 1 + int(np.argmax(potential[1:-1]))
    # the arc through y_0 runs from x_{N-k} over x_0 to x_k
    n_low = 2 * k
    n_high = int(n_beads) - n_low
    qdot = abs(y[k + 1] - y[k - 1]) / (2.0 * tau)
    return np.array([k, y[k], n_low, n_high, n_low * tau, n_high * tau, qdot], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Typical: the instanton of the asymmetric model at beta = 300, N = 1024 ---
        {
            "setup": """import numpy as np
params = np.array([1.0, 0.3, 4.6, -0.19, 1.5])
beads = _oracle_optimize_instanton(params, 300.0, 1024)
""",
            "call": "select_dividing_surface(beads.copy(), params.copy(), 300.0, 1024)",
            "gold_call": "_oracle_select_dividing_surface(beads.copy(), params.copy(), 300.0, 1024)",
            "tol": 1e-10,
        },
        # --- Boundary: coarse ring (N = 256) where the dividing surface sits two beads from y_0 ---
        {
            "setup": """import numpy as np
params = np.array([1.0, 0.3, 5.0, -0.46, 2.0])
beads = _oracle_optimize_instanton(params, 300.0, 256)
""",
            "call": "select_dividing_surface(beads.copy(), params.copy(), 300.0, 256)",
            "gold_call": "_oracle_select_dividing_surface(beads.copy(), params.copy(), 300.0, 256)",
            "tol": 1e-10,
        },
        # --- Edge: mirror-symmetric surface; the dividing surface is the central bead ---
        {
            "setup": """import numpy as np
params = np.array([1.0, 1.0, 3.0, 0.0, 1.0])
beads = _oracle_optimize_instanton(params, 300.0, 512)
""",
            "call": "select_dividing_surface(beads.copy(), params.copy(), 300.0, 512)",
            "gold_call": "_oracle_select_dividing_surface(beads.copy(), params.copy(), 300.0, 512)",
            "tol": 1e-10,
        },
        # --- Typical: the wide right well is the deeper one; the path runs towards negative x ---
        {
            "setup": """import numpy as np
params = np.array([1.0, 0.3, 4.6, -1.5, 1.5])
beads = _oracle_optimize_instanton(params, 200.0, 512)
""",
            "call": "select_dividing_surface(beads.copy(), params.copy(), 200.0, 512)",
            "gold_call": "_oracle_select_dividing_surface(beads.copy(), params.copy(), 200.0, 512)",
            "tol": 1e-10,
        },
        # --- Edge: a straight trial path between the two minima (not an instanton) ---
        {
            "setup": """import numpy as np
params = np.array([1.0, 0.3, 4.6, -0.19, 1.5])
beads = np.linspace(-4.2, 4.4, 33)
""",
            "call": "select_dividing_surface(beads.copy(), params.copy(), 150.0, 64)",
            "gold_call": "_oracle_select_dividing_surface(beads.copy(), params.copy(), 150.0, 64)",
            "tol": 1e-10,
        },
        # --- Invalid: the bead array does not match n_beads ---
        {
            "setup": """import numpy as np
params = np.array([1.0, 0.3, 4.6, -0.19, 1.5])
beads = np.linspace(-4.2, 4.4, 33)
def run_model():
    try:
        select_dividing_surface(beads.copy(), params.copy(), 150.0, 128)
        return 0
    except ValueError:
        return 1
def run_oracle():
    try:
        _oracle_select_dividing_surface(beads.copy(), params.copy(), 150.0, 128)
        return 0
    except ValueError:
        return 1
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
